#!/usr/bin/env python

import os
import logging

logerror = logging.error
loginfo = logging.info

import scripts.utils as iu
import RabbitMqWorkerAutoPhoto as rmw_ap


rmw_ap_hendler = rmw_ap.RabbitMqWorkerAutoPhoto("RUNNING", "autophoto")

# Start listen document for processing
rmw_ap_hendler.listen_queue()
