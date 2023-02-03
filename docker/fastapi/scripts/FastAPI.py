#!/usr/bin/env python

import logging
import os

logerror = logging.error
loginfo = logging.info
from collections import defaultdict
from json import load

import requests
import hashlib
import uuid

import numpy as np
from PIL import Image, UnidentifiedImageError
import io
from pdf2image import convert_from_bytes
import psutil

num_cpus = psutil.cpu_count()  # logical=False - for only real cpu cores

from base64 import b64encode, b64decode
import msgpack_numpy as m
from msgpack_numpy import encode
from msgpack_numpy import decode
from msgpack import unpackb
from msgpack import packb

m.patch()

from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator

# import scripts.utils as iu
from scripts.utils import read_image, image_zip, decode_msgpack_ndarray
from scripts import ArrowFlightClient as afc
from RabbitMqWorkerFastAPI import RabbitMqWorkerFastAPI


loginfo(f" | LOADING CONFIGS TO RAM...")
config_data = defaultdict()
path = r"/app/config.json"
with open(path) as json_file:
    try:
        config_data = load(json_file)
    except Exception as e:
        logerror("Cant read config from json file!", exc_info=True)


loginfo(f" | LOADING EXISTING TEMPLATES FILES TO RAM...")
templates_data = defaultdict(dict)
path = r"/app/templates/"
with os.scandir(path) as templates:
    for template in templates:
        template = str(template.name)
        templates_data[template] = defaultdict(dict)
        with os.scandir(os.path.join(path, template)) as template_files:
            for template_file in template_files:
                template_file = str(template_file.name)
                templates_data[template][template_file] = defaultdict()
                path_to_file = os.path.join(path, template, template_file)
                with open(path_to_file) as json_file:
                    try:
                        data = load(json_file)
                        templates_data[template][template_file] = data
                    except Exception as e:
                        logerror("Cant read template from json file!", exc_info=True)
                loginfo(f" | ADD: {template} <- {template_file}")


# Create afc handler to escape 10 seconds delay on clearing afc hendler object:
while True:
    try:
        afc_hendler = afc.ArrowFlightClient()
        break
    except Exception as e:
        pass


# initialization
app = FastAPI()
Instrumentator().instrument(app).expose(app)


def get_msg_from_template(template, template_file, template_name, templates_data):
    """Read msg field from specific template in RAM"""
    data = templates_data[template][template_file]
    try:
        data_template = data[template_name]
    except Exception as e:
        logerror("Cant read template from templates file!", exc_info=True)
    try:
        data_template_msg = data_template["msg"]
    except Exception as e:
        logerror("Cant read msg field from template!", exc_info=True)
    return data_template_msg


async def task_queue_handler(templates_data, task_msg):
    """
    #-USAGE------------------------------------
        | Use for send task message to QUEUE and wait result from queue of execution task.
    #-ARGUMENTS--------------------------------
        | - task_msg          - Message from POST. Task, that needed be executed.
    """
    # Reading task template, if get str in "task" field:
    if "task" in task_msg.keys():
        if type(task_msg["task"]) == str:
            loginfo(f" | CREATE MSG ...")
            template_props = task_msg["task"].split(sep="-")
            template_file_name = "_".join(template_props[0:2])
            template_name = "-".join(template_props[2:])
            msg = get_msg_from_template(
                template_props[0],
                f"{template_file_name}.json",
                template_name,
                templates_data,
            )
            task_msg["task"] = msg["task"]

    serialization = "msgpack"
    if "data" in task_msg.keys():
        if type(task_msg["data"]) == dict:
            if "images" in task_msg["data"].keys():
                if type(task_msg["data"]["images"][0]) == dict:
                    loginfo(f"| DATA PREPARATION...")

                    # Convert image from received bytes:
                    if "bytes" in task_msg["data"]["images"][0].keys():
                        images_bytes = [
                            image_bytes["bytes"]
                            for image_bytes in task_msg["data"]["images"]
                        ]
                        task_msg["data"]["images_params"] = defaultdict(dict)
                        task_msg["data"]["images"] = list()
                        append = task_msg["data"]["images"].append
                        for num, image_bytes in enumerate(images_bytes):
                            image_bytes = b64decode(image_bytes.encode("utf-8"))
                            try:
                                # Convert image type:
                                image = np.uint8(Image.open(io.BytesIO(image_bytes)))
                            except UnidentifiedImageError:
                                # Convert pdf type:
                                image = convert_from_bytes(
                                    image_bytes,
                                    fmt="jpeg",
                                    dpi=300,
                                    thread_count=num_cpus,
                                )[0]
                                image = np.uint8(image)
                            image_shape = image.shape
                            append([num, image_zip(image)])
                            task_msg["data"]["images_params"][str(num)] = defaultdict(
                                dict
                            )
                            task_msg["data"]["images_params"][str(num)]["shape"] = [
                                num,
                                image_shape,
                            ]
                        task_msg["data"] = packb(task_msg["data"], default=encode)
                        task_msg["data"] = b64encode(task_msg["data"]).decode("utf-8")
                        serialization = "bytes"

                    # Read image from path (the container needs to have access to this path):
                    elif "path" in task_msg["data"]["images"][0].keys():
                        names_images = [
                            image_path["path"]
                            for i, image_path in enumerate(task_msg["data"]["images"])
                        ]
                        task_msg["data"]["images_params"] = defaultdict(dict)
                        task_msg["data"]["images"] = list()
                        append = task_msg["data"]["images"].append
                        for num, image_path in enumerate(names_images):
                            image = read_image(image_path)
                            image_shape = image.shape
                            append([num, image_zip(image)])
                            task_msg["data"]["images_params"][str(num)] = defaultdict(
                                dict
                            )
                            task_msg["data"]["images_params"][str(num)]["shape"] = [
                                num,
                                image_shape,
                            ]
                        task_msg["data"] = packb(task_msg["data"], default=encode)
                        task_msg["data"] = b64encode(task_msg["data"]).decode("utf-8")

                    # Read image from AtcFs DB:
                    elif "atcfs_id" in task_msg["data"]["images"][0].keys():
                        HOST = config_data["host"]
                        PORT = config_data["port"]
                        AtcFs_VisId = config_data["AtcFs_VisId"]
                        AtcFs_VisUser = config_data["AtcFs_VisUser"]
                        AtcFs_RequestId = str(uuid.uuid4())
                        visSecretKey = config_data["visSecretKey"]
                        AtcFs_Sign = hashlib.md5(
                            f"{AtcFs_VisId}_{AtcFs_VisUser}_{AtcFs_RequestId}_{visSecretKey}".encode(
                                "utf-8"
                            )
                        ).hexdigest()
                        headers = {
                            "AtcFs-VisId": AtcFs_VisId,
                            "AtcFs-VisUser": AtcFs_VisUser,
                            "AtcFs-RequestId": AtcFs_RequestId,
                            "AtcFs-Sign": AtcFs_Sign,
                        }
                        names_images = [
                            image_path["atcfs_id"]
                            for image_path in task_msg["data"]["images"]
                        ]
                        task_msg["data"]["images_params"] = defaultdict(dict)
                        task_msg["data"]["images"] = list()
                        append = task_msg["data"]["images"].append
                        for num, file_id in enumerate(names_images):
                            url = f"http://{HOST}:{PORT}/atcfs/fileinfo/{file_id}"
                            try:
                                loginfo(f" | CONNECTING TO {url}")
                                request = requests.get(url, headers=headers)
                            except Exception as e:
                                logerror("Error occured while processing POST request!")
                                logerror(e)
                            else:
                                loginfo(f" | GET REQUEST")
                            loginfo(f" | REQUEST:")
                            file_info = request.json()
                            loginfo(file_info)
                            file_name = file_info["fileName"]
                            loginfo(f" | LOADING FILE:")
                            url = f"http://{HOST}:{PORT}/atcfs/files/{file_id}"
                            try:
                                loginfo(f" | CONNECTING TO {url}")
                                request = requests.get(url, headers=headers)
                            except Exception as e:
                                logerror("Error occured while processing POST request!")
                                logerror(e)
                            else:
                                loginfo(f" | GET REQUEST")
                            # Get image as bytes:
                            image_bytes = request.content
                            fc = file_name[-5:]
                            if ".pdf" in fc:
                                # Convert pdf type:
                                image = convert_from_bytes(
                                    image_bytes,
                                    fmt="jpeg",
                                    dpi=300,
                                    thread_count=num_cpus,
                                )[0]
                                image = np.uint8(image)
                            else:
                                try:
                                    # Convert image type:
                                    image = np.uint8(
                                        Image.open(io.BytesIO(image_bytes))
                                    )
                                except UnidentifiedImageError:
                                    # Convert pdf type:
                                    image = convert_from_bytes(
                                        image_bytes,
                                        fmt="jpeg",
                                        dpi=300,
                                        thread_count=num_cpus,
                                    )[0]
                                    image = np.uint8(image)
                            image_shape = image.shape
                            append([num, image_zip(image)])
                            task_msg["data"]["images_params"][str(num)] = defaultdict(
                                dict
                            )
                            task_msg["data"]["images_params"][str(num)]["shape"] = [
                                num,
                                image_shape,
                            ]
                        task_msg["data"] = packb(task_msg["data"], default=encode)
                        task_msg["data"] = b64encode(task_msg["data"]).decode("utf-8")
                        serialization = "atcfs"

    loginfo(f" | TASK SENDING...")

    rmw_fastapi_hendler = RabbitMqWorkerFastAPI(
        "RUNNING",
        "end_pipeline",
        task_msg,
        afc_hendler=afc_hendler,
        serialization=serialization,
    )
    result_msg = {"error": "Error occured while execute task!"}
    loginfo(f" | TASK STARTED")
    rmw_fastapi_hendler.listen_queue()

    result_msg = rmw_fastapi_hendler.result_msg
    if "error" not in result_msg.keys():
        try:
            if (
                result_msg == {}
                or result_msg == None
                or result_msg == 0
                or result_msg["data"] == {}
                or result_msg["data"] == None
                or result_msg["data"] == 0
            ):
                result_msg = {"error": "Cant find processed data!"}
        except Exception as e:
            logerror(
                f"| Error occured while execute task! Maybe cant find processed data!"
            )
            logerror(e)
            result_msg = {
                "error": "Error occured while execute task! Maybe cant find processed data!"
            }
    return result_msg


@app.post("/task/")
async def create_task(request: Request):
    """
    #-USAGE------------------------------------
        | Used to receive messages on the set port (in main.py).
    #-ARGUMENTS--------------------------------
        | - request          - Message from POST. Task, that needed be executed.
    """
    try:
        task_msg = await request.json()
    except Exception as e:
        logerror("Cant read json data!", exc_info=True)
        return {"error": "Cant read json data!"}

    try:
        result_msg = await task_queue_handler(templates_data, task_msg)
    except Exception as e:
        logerror(
            "Error occured while doing task in QUEUE! Task aborted!", exc_info=True
        )
        return {"error": "Error occured while doing task in QUEUE! Task aborted!"}

    return result_msg
