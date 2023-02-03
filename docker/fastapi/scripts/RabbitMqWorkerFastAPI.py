#!/usr/bin/env python
import logging

logerror = logging.error
loginfo = logging.info
import uuid
import io

from PIL import Image
from base64 import b64decode, b64encode
import msgpack_numpy as m
from msgpack import unpackb
from msgpack import packb
from msgpack_numpy import encode
from msgpack_numpy import decode

m.patch()

from scripts.RabbitMqWorker import RabbitMqWorker

# import scripts.utils as iu


class RabbitMqWorkerFastAPI(RabbitMqWorker):
    def __init__(
        self, status, name_container, task_msg, afc_hendler=None, serialization=None
    ):
        super().__init__(status, name_container, afc_hendler)

        loginfo(f" | INIT | TAKE TASK | PIPELINE:")
        loginfo(f' | {str(task_msg["task"]["pipeline"]):<10}')
        task_msg["name"] = str(uuid.uuid4())

        self.send_data(task_msg)
        loginfo(f" | INIT | TASK SENDED TO QUEUE")

        self.task_msg = task_msg
        self.result_msg = {}

        if serialization:
            self.serialization = serialization if serialization else "msgpack"

    def do_on_message(self, ch, method, properties, body):
        try:
            result_msg = RabbitMqWorkerFastAPI.read_data(ch, method, properties, body)
            if "error" not in result_msg.keys():
                result_msg["task"]["step"] -= 1  # bc +1 in read data

                loginfo(f" | PACK MSG ...")
                if "data" in result_msg.keys():
                    if "msgpack" in self.serialization:
                        if "images" in result_msg["data"].keys():
                            result_msg["data"][
                                "images"
                            ] = RabbitMqWorkerFastAPI.encode_ndarray_msgpack(
                                result_msg["data"]["images"]
                            )
                        result_msg["data"] = packb(result_msg["data"], default=encode)
                        loginfo(f" | DECODE MSG ...")
                        result_msg["data"] = b64encode(result_msg["data"]).decode(
                            "utf-8"
                        )
                    elif "bytes" in self.serialization or "atcfs" in self.serialization:
                        if "images" in result_msg["data"].keys():
                            images_bytes = []
                            for num, image in result_msg["data"]["images"]:
                                img_bytes = io.BytesIO()
                                Image.fromarray(image).save(img_bytes, format="JPEG")
                                img_bytes_arr = img_bytes.getvalue()
                                images_bytes.append(
                                    {
                                        self.serialization: b64encode(
                                            img_bytes_arr
                                        ).decode("utf-8")
                                    }
                                )
                            result_msg["data"]["images"] = images_bytes

            self.result_msg = result_msg
            ch.stop_consuming()
            self._connection.close()
        except Exception as e:
            self.exception_handler(e)
