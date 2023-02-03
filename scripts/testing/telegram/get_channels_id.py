#!/usr/bin/env python3

import os
import json
from pprint import pprint

from telethon import TelegramClient
from telethon.utils import get_display_name

import sys

sys.path.insert(0, "scripts")
from dbes.DBESTelegramAdapter import DBESTelegramAdapter


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
# CHANNEL = CONFIG['telegram_app_api_auth']['channel']

DIALOGS_COUNT = 10
##############################################
##############################################


####################################################################################################
## Разработка адаптера Телеграма
####################################################################################################


# print("Creating TelegramClient...")
# client = TelegramClient(USERNAME, API_ID, API_HASH)

# print("Starting TelegramClient...")
# client.start(phone = PHONE if PHONE else None,
#              password = PASS if PASS else None)

# print("Client Created")


# async def main():
#     dialogs = await client.get_dialogs(DIALOGS_COUNT)
#     for dialog in dialogs:
#         print(get_display_name(dialog.entity), dialog.entity.id)

# with client:
#     client.loop.run_until_complete(main())


####################################################################################################
## DBESTelegramAdapter
####################################################################################################


dbes_ta = DBESTelegramAdapter()
dbes_ta.start()

dialogs_ids = dbes_ta.get_dialogs_ids(DIALOGS_COUNT=DIALOGS_COUNT)

pprint(dialogs_ids)
