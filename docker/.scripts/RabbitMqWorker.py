#!/usr/bin/env python

import os
import json
import orjson
import logging

logerror = logging.error
loginfo = logging.info
import time

import base64
import pika

import numpy as np
import abc

import msgpack
import msgpack_numpy as m

m.patch()

# from . import utils
from . import ArrowFlightClient as afc


class NumpyEncoder(json.JSONEncoder):
    """
    #-USAGE------------------------------------
        | Used for serialization np.array and float to JSON
    #-ARGUMENTS--------------------------------
        | - filename    - Path to file.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


class RabbitMqWorker:
    """
    #-INIT-----------------------------------------
        |   status             - curently status (now dont using anywhere)
        |   name_container     - name of current container, used for QUEUE
    #-USAGE----------------------------------------
        | Used for read and send messages to QUEUE
    #-AVAILABLE-FUNCTIONS--------------------------
        |
        | + do_on_message(ch, method, properties, body) - Put code here, that need been executed, when corrensponding MESSAGE in QUEUE occured
        | - read_data(ch, method, properties, body)     - Use for read data from message from QUEUE.
        | - send_data(ch, method, properties, body)                                 - Use for send data as message to QUEUE.
        | - get_connection()                            - Set connection to QUEUE server.
        | - listen_queue()                              - Connect to QUEUE server and listen in loop messages from RabbitMq QUEUE.
        | - exception_handler(self, e, id_request=0)    - Use for handle exceptions in pipeline queue run process.
    """

    def __init__(self, status, name_container, afc_hendler=None):
        self._status = status
        self._name_container = name_container
        self._connection = self.get_connection()
        self._channel = self._connection.channel()
        self._logs = []
        self._afc_hendler = afc_hendler if afc_hendler else afc.ArrowFlightClient()

    @staticmethod
    def file_get_contents(filename):
        """
        #-USAGE------------------------------------
            | Read file into bytes.
        #-ARGUMENTS--------------------------------
            | - filename    - Path to file.
        """
        with open(filename, "rb") as f:
            return f.read()

    @staticmethod
    def encode_files_base64(filePaths):
        """
        #-USAGE------------------------------------
            | Read files from path and encode to base64, and decode to utf-8.
        #-ARGUMENTS--------------------------------
            | - filePaths    - Path to files.
        """
        resultEncoded = []
        for filePath in filePaths:
            resultEncoded.append(
                base64.b64encode(RabbitMqWorker.file_get_contents(filePath)).decode(
                    "utf-8"
                )
            )
        return resultEncoded

    @staticmethod
    def image_zip(image):
        """
        #-USAGE------------------------------------
            | Compress data (use for compress images) using zlib.
        #-ARGUMENTS--------------------------------
            | - image    - Image (ndarray).
        """
        image = msgpack.packb(image, default=m.encode)

        return image

    @staticmethod
    def image_unzip(image):
        """
        #-USAGE------------------------------------
            | Decompress data (use for decompress images) using zlib.
        #-ARGUMENTS--------------------------------
            | - image    - Image (ndarray).
        """
        try:
            image = np.uint8(msgpack.unpackb(image, object_hook=m.decode))
        except:
            raise RuntimeError("Could not decode the contents")

        return image

    @staticmethod
    def encode_ndarray_msgpack(images):
        """
        #-USAGE------------------------------------
            | Encode list of ndarray(or list with shape as image nd.array) to base64, and decode to utf-8.
        #-ARGUMENTS--------------------------------
            | - images    - List of images (ndarrays or lists). Format: [[N, IMAGE, IMAGE, ... ], [N, IMAGE, IMAGE, ... ], ... ]
        """
        result_encoded = []

        for images_ in images:
            numbr = images_[0]
            encoded_images = [numbr]

            for image in images_[1:]:
                image = np.uint8(image)
                encoded_images.append(RabbitMqWorker.image_zip(image))

            result_encoded.append(encoded_images)

        # Format: [[N, [BS64IMAGE, SHAPE], [BS64IMAGE, SHAPE], ... ], ... ]
        return result_encoded

    @staticmethod
    def decode_msgpack_ndarray(data):
        """
        #-USAGE------------------------------------
            | Decode list of base64.utf-8 to ndarray.uint8.
        #-ARGUMENTS--------------------------------
            | - data    - List of base64.utf-8. Format: [[N, [BS64IMAGE, SHAPE], [BS64IMAGE, SHAPE], ... ], ... ]
        """
        result_decoded = []

        for data_ in data:
            numbr = data_[0]
            decoded_images = [numbr]

            for image in data_[1:]:
                image = RabbitMqWorker.image_unzip(image)
                decoded_images.append(image)

            result_decoded.append(decoded_images)

        # [[N, np.uint8_IMAGE, np.uint8_IMAGE, ... ], ... ]
        return result_decoded

    def exception_handler(self, e, id_request=0):
        """
        #-USAGE------------------------------------
            | Use for handle exceptions in pipeline queue run process.
            | Abort running task and send error message to end_pipeline
        #-ARGUMENTS--------------------------------
            | - e           Exception.
            | - id_request  ID of request, then exception is occured.
        """
        logerror(
            "Error occured while doing task in QUEUE! Task aborted!", exc_info=True
        )
        msg = {"error": f"Error occured in {self._name_container}"}

        # connection = RabbitMqWorker.get_connection()
        # channel = connection.channel()
        self._channel.queue_declare(queue="end_pipeline")

        properties = msg

        self._channel.basic_publish(
            exchange="",
            routing_key="end_pipeline",
            body=json.dumps(msg),
            properties=pika.BasicProperties(headers=properties),
        )
        self._channel.stop_consuming()
        self._connection.close()

    @staticmethod
    def get_connection():
        """
        #-USAGE------------------------------------
            | Set connection to QUEUE server.
        """
        loginfo(f" | SETUP CONNECTION... -> 1")

        host = os.environ["QUEUE_SERVER_HOST"]
        port = os.environ["QUEUE_SERVER_PORT"]
        username = os.environ["QUEUE_SERVER_USERNAME"]
        password = os.environ["QUEUE_SERVER_PASSWORD"]

        credentials = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=host, port=port, credentials=credentials)
        )

        loginfo(f" | SETUP CONNECTION... -> 2")
        return connection

    def listen_queue(self):
        """
        #-USAGE------------------------------------
            | Connect to QUEUE server and listen in loop messages from RabbitMq QUEUE.
        """
        loginfo(f" | SETUP CONNECTION... ")

        try:
            self._channel.queue_declare(queue=self._name_container)
            self._channel.basic_consume(
                queue=self._name_container,
                on_message_callback=self.do_on_message,
                auto_ack=True,
            )

            loginfo(f" | WAITING FOR MESSAGES... ")
            self._channel.start_consuming()
        except Exception as e:
            self.exception_handler(e)

    @abc.abstractmethod
    def do_on_message(ch, method, properties, body):
        """
        #-USAGE------------------------------------
            | Put code here, that need been executed, when corrensponding MESSAGE in QUEUE occured
        #-ARGUMENTS--------------------------------
            | - ch         Relate to channel obj
            | - method     ?
            | - properties Properties of message. Can set in channel.basic_publish(properties=...)
            | - body       Body of message. Can set in channel.basic_publish(body=json.dump(...))
        #-EXAMPLE----------------------------------
            |
                @staticmethod
                def do_on_message(ch, method, properties, body):
                    try:
                        taken_msg = RabbitMqWorkerOpencv.read_data(ch, method, properties, body)

                        print('| LOG | START |  IMAGE PREPROCESSING | ')
                        images = ip.image_preprocess(taken_msg['body']['images'],
                                                    taken_msg['body']['do_pipeline'][taken_msg['properties']['step'] - 1],
                                                    taken_msg['body']['stacked_data'][taken_msg['properties']['step'] - 1])
                        print('| DONE | ')

                        msg = {
                                'error': '',
                                'stacked_data': taken_msg['body']['stacked_data'],
                                'do_pipeline': taken_msg['body']['do_pipeline'],
                                'images': images,
                                'data': taken_msg['body']['data'],
                            }

                        RabbitMqWorkerOpencv.send_data(taken_msg, msg)
                    except Exception as e:
                        RabbitMqWorkerOpencv.exception_handler(e)
        #-MSG-------------------------------------
            |
                taken_msg = {
                # Properties of task:
                'properties': {
                    # Pipeline of all task. Available:
                        - opencv        - Preprocessing of image.
                        - paddle-ocr    - For detect text and cut image on boxes with text, Digit and ENG recognize.
                        - tesseract     - For detect text and recognize text (RUS, ENG, RUS+ENG and more).
                        - super-res     - For upscaling image. 2x or 4x support.
                    'pipeline': ['opencv', 'paddle-ocr'],
                    # Current step in pipekine:
                    'step': 0,
                },
                # Data of task and some propertyes for each image in each microservice.
                'body': {
                    # Copy or not previos images (False - not, True - copy).
                    'stacked_data': [False, False],
                    # Pipeline for each image in each microservice.
                    # [['GT']], [['Er', 'Rd']] - GT do one time for opencv microservice for all images.
                                            - Er do only for first image (paddle-ocr or tesseract).
                                            - Rd fo only for second image (paddle-ocr or tesseract).
                                            + Availaible keys can be found into containers.
                    'do_pipeline': [['GT'], ['Erdc']],
                    # List of images before running self task on this container. [[N, image], [N, image], ... ]
                    'images': images,
                    # List of all previos data
                    'data': data,
                },
            }

            msg = {
                    # Copy or not previos images (False - not, True - copy).
                    'stacked_data': [False, False],
                    # Pipeline for each image in each microservice.
                    # [['GT']], [['Er', 'Rd']] - GT do one time for opencv microservice for all images.
                                            - Er do only for first image (paddle-ocr or tesseract).
                                            - Rd fo only for second image (paddle-ocr or tesseract).
                                            + Availaible keys can be found into containers.
                    'do_pipeline': [['GT'], ['Erdc']],
                    # List of images + images (if stacked_data = true) after running self task on this container. [[N, image], [N, image], ... ]
                    'images': images,
                    # List of all previos data + data from executed task for this contaner (dont need set in first send).
                    'data': [],
                }
        """

    @staticmethod
    def read_data(ch, method, properties, body):
        """
        #-USAGE------------------------------------
            | Use for read data from message from QUEUE.
            | And for converting to a common format.
            | For sample of message watch MSG.
        #-ARGUMENTS--------------------------------TODO: same
            | - ch         Relate to channel obj
            | - method     ?
            | - properties Properties of message. Can set in channel.basic_publish(properties=...)
            | - body       Body of message. Can set in channel.basic_publish(body=json.dump(...))
        #-MSG--------------------------------------
            # Dict message to dump JSON and send to FastAPI:
            msg = {
                # Properties of task:
                'properties': {
                    # Pipeline of all task. Available:
                        - opencv        - Preprocessing of image.
                        - paddle-ocr    - For detect text and cut image on boxes with text, Digit and ENG recognize.
                        - tesseract     - For detect text and recognize text (RUS, ENG, RUS+ENG and more).
                        - super-res     - For upscaling image. 2x or 4x support.
                    'pipeline': ['opencv', 'paddle-ocr'],
                    # Current step in pipekine:
                    'step': 0,
                },
                # Data of task and some propertyes for each image in each microservice.
                'body': {
                    # Copy or not previos images (False - not, True - copy).
                    'stacked_data': [False, False],
                    # Pipeline for each image in each microservice.
                    # [['GT']], [['Er', 'Rd']] - GT do one time for opencv microservice for all images.
                                            - Er do only for first image (paddle-ocr or tesseract).
                                            - Rd fo only for second image (paddle-ocr or tesseract).
                                            + Availaible keys can be found into containers.
                    'do_pipeline': [['GT'], ['Erdc']],
                    # List of images. [[N, image], [N, image], ... ]
                    'images': images,
                    # List of all data (dont need set in first send).
                    'data': [],
                },
            }
        """
        loginfo(f" | TAKE MESSAGE")
        if not "error" in properties.headers.keys():
            # Sample of base properties in each json message.
            name = properties.headers["name"]

            data_from_arrow = []
            try:
                # Get data from Arrow Server:
                afc_hendler = afc.ArrowFlightClient()
                loginfo(
                    f" | READING DATA... <- ARROW SERVER: {afc_hendler.location} ..."
                )
                data_from_arrow = afc_hendler.read_data(command=name)
                loginfo(f" | DONE")

                loginfo(f" | DECODING  MSG ...")
                data_from_arrow = base64.b64decode(data_from_arrow.encode("utf-8"))
                loginfo(f" | MSG UNPACK ...")
                data_from_arrow = msgpack.unpackb(data_from_arrow, object_hook=m.decode)
                loginfo(f" | DECODING  IMAGES ...")
                data_from_arrow["images"] = RabbitMqWorker.decode_msgpack_ndarray(
                    data_from_arrow["images"]
                )
            except Exception as e:
                loginfo(f" | CANT READ DATA.")
                loginfo(e)
                loginfo(f" | CONTINUE WITHOUT DATA.")

            # Main data for pipeline:
            loginfo(f" | LOADS TASK ...")
            decoded_data = orjson.loads(body)

            # All pipeline of request:
            pipeline = decoded_data["pipeline"]
            # Current step in pipeline:
            # Checking that we can be the last stage of the pipeline:
            current_step = decoded_data["step"]
            if current_step < len(pipeline):
                decoded_data["step"] = decoded_data["step"] + 1
            # else:
            #     if data_from_arrow:
            #         if 'images' in data_from_arrow.keys():
            #             data_from_arrow['images'] = []

            loginfo(f" | CHECK STEP DONE ")

            if data_from_arrow:
                msg = {"name": name, "task": decoded_data, "data": data_from_arrow}
            else:
                msg = {"name": name, "task": decoded_data}

            loginfo(f" | TAKEN MESSAGE | TASK:")
            loginfo(f"{str(pipeline):<10}")
            return msg
        else:
            return {"error": properties.headers["error"]}

    def prepare_data(self, taken_msg):
        loginfo(f" | SENDING DATA... ->")

        # Sample of base properties in each json message.
        # All pipeline of request:
        pipeline = taken_msg["task"]["pipeline"]
        # Current step in pipeline:
        current_step = taken_msg["task"]["step"]

        # Checking that we can be the last stage of the pipeline:
        if current_step == len(pipeline):
            next_queue = "end_pipeline"
            # if 'data' in taken_msg.keys():
            #     if 'images' in taken_msg['data'].keys():
            #         taken_msg['data']['images'] = []
        else:
            next_queue = pipeline[current_step]

        loginfo(f" | SENDING DATA... -> {next_queue}")
        properties = {"name": taken_msg["name"]}

        # Get JSON Object end encode it to bytes:
        json_dumps_zip = orjson.dumps(
            taken_msg["task"], option=orjson.OPT_SERIALIZE_NUMPY
        )
        return json_dumps_zip, properties, next_queue

    def send_to_afc(self, taken_msg):
        loginfo(f" | SENDING DATA... PACK MSG ...")
        if "data" in taken_msg.keys():
            if type(taken_msg["data"]) != str:
                if "images" in taken_msg["data"].keys():
                    taken_msg["data"]["images"] = RabbitMqWorker.encode_ndarray_msgpack(
                        taken_msg["data"]["images"]
                    )
                data_msg = msgpack.packb(taken_msg["data"], default=m.encode)
                loginfo(f" | SENDING DATA... DECODE MSG ...")
                data_msg = base64.b64encode(data_msg).decode("utf-8")
            else:
                data_msg = taken_msg["data"]

            # Send data to Arrow Server:
            loginfo(
                f" | SENDING DATA... -> ARROW SERVER: {self._afc_hendler.location} ..."
            )
            self._afc_hendler.send_data(data_msg, command=taken_msg["name"])

    def send_to_mq(self, json_data, properties, next_queue):
        # Publish TASK in QUEUE:
        loginfo(f" | SENDING DATA... -> QUEUE...")
        start_time = time.time()
        self._channel.queue_declare(queue=next_queue)
        start_time = time.time()
        self._channel.basic_publish(
            exchange="",
            routing_key=next_queue,
            body=json_data,
            properties=pika.BasicProperties(headers=properties),
        )

    def send_data(self, taken_msg):
        json_data, properties, next_queue = self.prepare_data(taken_msg)
        self.send_to_afc(taken_msg)
        self.send_to_mq(json_data, properties, next_queue)
        self.start_time = time.time()
