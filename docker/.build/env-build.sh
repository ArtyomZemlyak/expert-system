#!/usr/bin/env bash

# Base libraryes for QUEUE:
apt-get update
apt install python3-pip -y
# Need install to all containers:
# For working with RabbitMq:
pip3 install pika
# For abstract method in RabbitMqWorker:
pip3 install decorators
# For pretty CLI:
pip3 install rich
# For some evaluations:
pip3 install numpy
