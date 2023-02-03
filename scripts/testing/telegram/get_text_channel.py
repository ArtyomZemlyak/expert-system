#!/usr/bin/env python3

import os
from pprint import pprint
import json

from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import PeerChannel

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
CHANNEL = CONFIG["telegram_app_api_auth"]["channel"]

COUNT_MSGS_LIMIT = 0
IGNORE_COUNTERS = set([1])
IGNORE_IDS = set([])
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
#     if type(CHANNEL) == int:
#         channel_peer = PeerChannel(int(CHANNEL))
#     else:
#         channel_peer = CHANNEL

#     channel_entity = await client.get_entity(channel_peer)
#     print("Get channel entity")


#     offset_id = 0
#     all_messages = []


#     while True:
#         print("Current Offset ID is:", offset_id, end='')

#         history = await client(GetHistoryRequest(
#             peer=channel_entity,
#             offset_id=offset_id,
#             offset_date=None,
#             add_offset=0,
#             limit=10 if COUNT_MSGS_LIMIT else 100,
#             max_id=0,
#             min_id=0,
#             hash=0
#         ))

#         if not history.messages:
#             break

#         messages = history.messages

#         for message in messages:
#             message = message.to_dict()

#             if 'message' in message:
#                 if message['id'] not in IGNORE_IDS:
#                     all_messages.append(message)

#         offset_id = messages[len(messages) - 1].id

#         total_messages = len(all_messages)
#         print("; Total Messages:", total_messages)

#         if COUNT_MSGS_LIMIT != 0 and total_messages >= COUNT_MSGS_LIMIT:
#             break

#     # Bc first msg - is a last msg in channel:
#     all_messages.reverse()

#     all_messages = [msg for i, msg in enumerate(all_messages) if i+1 not in IGNORE_COUNTERS]

#     print(all_messages[0])
#     print(all_messages[-1])


# with client:
#     client.loop.run_until_complete(main())


####################################################################################################
## DBESTelegramAdapter
####################################################################################################

dbes_ta = DBESTelegramAdapter()
dbes_ta.start()

# # Test of working 2 async func in a row:
# dialogs_ids = dbes_ta.get_dialogs_ids(DIALOGS_COUNT=10)
# pprint(dialogs_ids)

text_channel = dbes_ta.get_text_from_channel(
    COUNT_MSGS_LIMIT=COUNT_MSGS_LIMIT,
    IGNORE_COUNTERS=IGNORE_COUNTERS,
    IGNORE_IDS=IGNORE_IDS,
)

print(text_channel[0])
print(text_channel[-1])
