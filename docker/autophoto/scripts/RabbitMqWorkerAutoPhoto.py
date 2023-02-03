#!/usr/bin/env python

import scripts.RabbitMqWorker as rmw
import sys

sys.path.append("/app/Detection/scripts")
from AutocropNN import AutocropNN


class RabbitMqWorkerAutoPhoto(rmw.RabbitMqWorker):
    def __init__(self, status, name_container):
        super().__init__(status, name_container)
        self.autocrop_nn = AutocropNN()

    def do_on_message(self, ch, method, properties, body):
        print(
            "------------------------------------------------------------------------"
        )
        try:
            taken_msg = RabbitMqWorkerAutoPhoto.read_data(ch, method, properties, body)
            images_cropped, errors = self.autocrop_nn.cropping_NN_from_msg(
                taken_msg["data"]["images"],
                taken_msg["task"]["do_pipeline"][taken_msg["task"]["step"] - 1],
            )

            taken_msg["data"]["data"] = errors
            taken_msg["data"]["images"] = images_cropped

            self.send_data(taken_msg)
        except Exception as e:
            self.exception_handler(e)
