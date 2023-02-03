#!/usr/bin/env python3

import os
from pprint import pprint
import json
from operator import itemgetter

import sys

sys.path.insert(0, "scripts")
from dbes.DBESTelegramAdapter import DBESTelegramAdapter
from es.ESNLP import ESNLP


##############################################
#### CONFIG ##################################
##############################################
import pathlib

path_cfg = os.path.join(
    pathlib.Path(__file__).parent.parent.parent.parent.resolve(), "config.json"
)
CONFIG = json.loads(open(path_cfg, "r").read())
API_ID = CONFIG["telegram_app_api_auth"]["api_id"]
API_HASH = CONFIG["telegram_app_api_auth"]["api_hash"]
PHONE = CONFIG["telegram_app_api_auth"]["phone"]
PASS = CONFIG["telegram_app_api_auth"]["pass"]
USERNAME = CONFIG["telegram_app_api_auth"]["username"]
CHANNEL = CONFIG["telegram_app_api_auth"]["channel"]

COUNT_MSGS_LIMIT = 0
IGNORE_COUNTERS = set([1])
IGNORE_IDS = set([])
##############################################
##############################################


####################################################################################################
## Подсчёт простой статистики для чата Телеграма
####################################################################################################


dbes_ta = DBESTelegramAdapter()
dbes_ta.start()


text_channel = dbes_ta.get_text_from_channel(
    COUNT_MSGS_LIMIT=COUNT_MSGS_LIMIT,
    IGNORE_COUNTERS=IGNORE_COUNTERS,
    IGNORE_IDS=IGNORE_IDS,
)

es_nlp = ESNLP()


tokens = es_nlp.count_tokens(text=" ".join([msg["message"] for msg in text_channel]))

tokens = sorted(
    [(token, count) for token, count in tokens.items()], key=itemgetter(1), reverse=True
)

pprint(tokens[:20])
