#!/usr/bin/env python3

import os
from pprint import pprint
import json
from typing import Union
from functools import wraps

from telethon import TelegramClient
from telethon.utils import get_display_name
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import PeerChannel


##############################################
#### CONFIG ##################################
##############################################
import pathlib

path_cfg = os.path.join(
    pathlib.Path(__file__).parent.parent.parent.resolve(), "config.json"
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


def async_worker(function):
    """
    For functions, that need work in async mode.
    """

    @wraps(function)
    def wrapper(self, *args, **kwargs):
        if self.client:
            returned = None
            with self.client:
                returned = self.client.loop.run_until_complete(
                    function(self, *args, **kwargs)
                )
            return returned
        else:
            raise RuntimeError("Client is not started! Use .start()")

    return wrapper


class DBESTelegramAdapter:
    def __init__(self):
        self.client = None

    def start(
        self,
        session: str = USERNAME,
        api_id: int = API_ID,
        api_hash: str = API_HASH,
        phone: str = PHONE,
        password: str = PASS,
    ):
        self.client = TelegramClient(session=session, api_id=api_id, api_hash=api_hash)

        self.client.start(
            phone=phone if phone else None, password=password if password else None
        )

    @async_worker
    async def get_dialogs_ids(self, DIALOGS_COUNT: int = 10) -> list:
        dialogs_ids = []

        dialogs = await self.client.get_dialogs(DIALOGS_COUNT)

        for dialog in dialogs:
            dialogs_ids.append((get_display_name(dialog.entity), dialog.entity.id))

        return dialogs_ids

    @async_worker
    async def get_text_from_channel(
        self,
        channel_id: Union[int, str] = None,
        COUNT_MSGS_LIMIT: int = 0,
        IGNORE_COUNTERS: set = None,
        IGNORE_IDS: set = None,
    ) -> list:
        if not channel_id:
            if not CHANNEL:
                raise ValueError("Need set channel id!")
            else:
                channel_id = CHANNEL

        if type(channel_id) == int:
            channel_peer = PeerChannel(int(channel_id))
        else:
            channel_peer = channel_id

        channel_entity = await self.client.get_entity(channel_peer)

        offset_id = 0
        all_messages = []

        while True:
            history = await self.client(
                GetHistoryRequest(
                    peer=channel_entity,
                    offset_id=offset_id,
                    offset_date=None,
                    add_offset=0,
                    limit=10 if COUNT_MSGS_LIMIT else 100,
                    max_id=0,
                    min_id=0,
                    hash=0,
                )
            )

            if not history.messages:
                break

            messages = history.messages

            for message in messages:
                message = message.to_dict()

                if "message" in message:
                    if not IGNORE_IDS or message["id"] not in IGNORE_IDS:
                        all_messages.append(message)

            offset_id = messages[len(messages) - 1].id

            total_messages = len(all_messages)

            if COUNT_MSGS_LIMIT != 0 and total_messages >= COUNT_MSGS_LIMIT:
                break

        # Bc first msg - is a last msg in channel:
        all_messages.reverse()

        if IGNORE_COUNTERS:
            all_messages = [
                msg
                for i, msg in enumerate(all_messages)
                if i + 1 not in IGNORE_COUNTERS
            ]

        return all_messages
