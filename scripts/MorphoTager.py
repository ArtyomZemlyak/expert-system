"""
Morphological Tagging

pip install deeppavlov
python -m deeppavlov install DeepPavlov/deeppavlov/configs/morpho_tagger/BERT/morpho_ru_syntagrus_bert.json
python -m deeppavlov download DeepPavlov/deeppavlov/configs/morpho_tagger/BERT/morpho_ru_syntagrus_bert.json
python -m deeppavlov interact DeepPavlov/deeppavlov/configs/morpho_tagger/BERT/morpho_ru_syntagrus_bert.json
pip install rich
"""

import os
from os import listdir
from os.path import isfile, join
from collections import Counter
from typing import Any, Generator, List, Union
import time
import json
import re

import requests
import numpy as np
from deeppavlov import build_model
from rich.progress import Progress

from HtmlParser import HtmlParser
from TrackerLoader import TrackerLoader, TrackerLoaderHtmlParser


CONFIG = {
    **json.loads(open("config.json", "r").read()),
    **json.loads(open("scripts/config.json", "r").read()),
}

BLACKLIST = set(CONFIG["save"].values())


class MorphoTager:
    def __init__(self, config: str = None) -> None:

        if config:
            self.CONFIG = config
        else:
            self.CONFIG = CONFIG

        if not os.path.exists(self.CONFIG["save"]["path_save"]):
            os.mkdir(self.CONFIG["save"]["path_save"])

        self.pages = {}
        self.pages_versions = {}

        files_names = [
            f
            for f in listdir(self.CONFIG["save"]["path_save"])
            if isfile(join(self.CONFIG["save"]["path_save"], f)) and f not in BLACKLIST
        ]

        for file_name in files_names:
            name = re.sub("[\\.]{1}[A-Za-z\\d]+", "", file_name)
            json_file_data = json.loads(
                open(
                    os.path.join(self.CONFIG["save"]["path_save"], file_name),
                    "r",
                    encoding="utf8",
                ).read()
            )
            self.pages[name] = json_file_data

        if os.path.exists(
            os.path.join(
                self.CONFIG["save"]["path_save"], self.CONFIG["save"]["morph_tags"]
            )
        ):

            self.tags = json.loads(
                open(
                    os.path.join(
                        self.CONFIG["save"]["path_save"],
                        self.CONFIG["save"]["morph_tags"],
                    ),
                    "r",
                    encoding="utf8",
                ).read()
            )
            self.first_time = False
        else:
            self.tags = {}
            self.first_time = True

        self.file_name_all_href = None
        self.auth = None

        if "all_href" in self.CONFIG["save"].keys():
            self.file_name_all_href = self.CONFIG["save"]["all_href"]

        if (
            "confluence_auth" in self.CONFIG.keys()
            and "user" in self.CONFIG["confluence_auth"].keys()
            and "pass" in self.CONFIG["confluence_auth"].keys()
        ):
            self.auth = (
                self.CONFIG["confluence_auth"]["user"],
                self.CONFIG["confluence_auth"]["pass"],
            )

        if "confl" in self.CONFIG["mode"] and self.file_name_all_href and self.auth:
            print("Mode: confluence. Auth finded.")

            try:
                all_href_dict = json.loads(
                    open(
                        os.path.join(
                            self.CONFIG["save"]["path_save"], self.file_name_all_href
                        ),
                        "r",
                        encoding="utf8",
                    ).read()
                )

                self.files_names = all_href_dict["true"]
                self.url_href = all_href_dict["url"]
                print(
                    f"Use saved URL: {self.url_href} in {self.CONFIG['save']['path_save']}"
                )

                self.files = Generator
                print(
                    f"Use saved URLs: {self.file_name_all_href} in {self.CONFIG['save']['path_save']}"
                )

            except Exception as e:
                print(
                    f"Cant find {self.file_name_all_href} in {self.CONFIG['save']['path_save']}"
                )

                self.files_names = [
                    f
                    for f in listdir(self.CONFIG["save"]["path_save"])
                    if isfile(join(self.CONFIG["save"]["path_save"], f))
                    and f not in BLACKLIST
                ]

                self.files = [
                    open(
                        os.path.join(self.CONFIG["save"]["path_save"], file_path)
                    ).read()
                    for file_path in self.files_names
                ]

        elif "post" in self.CONFIG["mode"] and self.CONFIG["postgres_auth"]:
            print("Mode: postgres. Auth finded.")
            tracker_loader = TrackerLoader(self.CONFIG["postgres_auth"])
            self.files_names = tracker_loader.get_files_names()
            self.files = tracker_loader.get_files()

        else:
            print(f"Use files in {self.CONFIG['save']['path_save']}")
            self.files_names = [
                f
                for f in listdir(self.CONFIG["save"]["path_save"])
                if isfile(join(self.CONFIG["save"]["path_save"], f))
                and f not in BLACKLIST
            ]

            self.files = [
                open(os.path.join(self.CONFIG["save"]["path_save"], file_path)).read()
                for file_path in self.files_names
            ]

    def __format_name__(self, file_name: str) -> str:

        if self.file_name_all_href and self.auth:
            name = re.sub(r"/", "__", file_name)
            name = re.sub("[\\.]", "___", name)
        else:
            name = re.sub("[\\.]{1}[A-Za-z\\d]+", "", file_name)

        return name

    def __morph_model_pred__(self, line_batch: Union[str, List[str]]) -> List[str]:
        parse = []

        try:
            # start_time = time.time()

            _parse = self.model(line_batch)

            # print(f'Model pred time: {time.time() - start_time}')

            for parse_batch in _parse:
                parse.extend(parse_batch.split("\n"))

        except Exception as e:  # (RuntimeError, IndexError)
            print("1 Exception in __morph_model_pred__:", e)

            for line in line_batch:
                line_split = line.split(" ")

                try:
                    # start_time = time.time()

                    _parse = self.model(line_split)

                    # print(f'Model pred time: {time.time() - start_time}')

                    for parse_batch in _parse:
                        parse.extend(parse_batch.split("\n"))

                except Exception as e:
                    print("2:", e)
                    return parse

        return parse

    def __get_files_from_urls__(self):

        for file_name in self.files_names:

            try:

                yield requests.get(self.url_href + file_name, auth=self.auth).text

            except Exception as e:

                yield False

    def __get_dict_of_tags__(self, parse: List[str]) -> dict:
        tags = {}

        for line in parse:

            try:
                res = line.split("\t")

                token_init = self.__check_tag__(res[1])
                token_common = self.__check_tag__(res[2])

                if token_init != "" and token_common != "":
                    tags[token_init] = token_common

            except IndexError:
                continue

        return tags

    def __check_text__(self, text: str) -> bool:
        return re.sub(f"[^{self.CONFIG['analyzer']['word_symbols']}]", "", text) != ""

    def __check_tag__(self, tag: str) -> str:
        if self.__check_text__(tag):
            tag = re.sub(
                f"[^{self.CONFIG['analyzer']['word_symbols'] + self.CONFIG['analyzer']['filter_symbols']}]",
                "",
                tag,
            )
            return tag
        else:
            return ""

    def __check_text_line__(self, line) -> bool:
        if (
            line != []
            and line != ""
            and line != 0
            and line != " "
            and line != "  "
            and line != "   "
            and line != "    "
        ):
            if self.__check_text__(line):
                return True
        return False

    def __check_page_version__(self) -> str:

        if self.first_time:
            return "NEW"

        if self.__current_name not in self.pages.keys():

            self.pages[self.__current_name] = {}
            self.pages_versions[self.__current_name] = self.parser.get_page_version()

            return "NEW"

        else:
            if "page_version" in self.pages[self.__current_name].keys():

                current_remote_page_version = self.parser.get_page_version()

                if (
                    self.pages[self.__current_name]["page_version"]
                    < current_remote_page_version
                ):
                    self.pages_versions[
                        self.__current_name
                    ] = current_remote_page_version

                    return "UPDATE"

            else:
                return "NEW"

        return "ACTUAL"

    def __add_tags__(self, tags: dict) -> None:
        self.tags = {**self.tags, **tags}

    def __add_all_tags__(self) -> None:

        for field_name, field_property in self.parser.FIELDS_FIND.items():

            field_text = self.parser.get_field_text(field_name)

            if "conf" in CONFIG["mode"]:
                context = field_text.split("\n")
            if "post" in CONFIG["mode"]:
                context = [field_text]

            tags = []

            for line in context:

                if self.__check_text_line__(line):

                    if self.__temp_batch:
                        self.__line_batch.append(self.__temp_batch[0])
                        self.__border_val += self.__temp_batch[1]
                        self.__temp_batch = []

                    len_tokens = len(line.split(" "))

                    if (
                        self.__border_val + len_tokens
                        < self.CONFIG["analyzer"]["max_len_of_tokens"]
                        or len_tokens >= self.CONFIG["analyzer"]["enough_len_of_tokens"]
                    ):

                        self.__line_batch.append(line)
                        self.__border_val += len_tokens
                    else:
                        self.__temp_batch = (line, len_tokens)

                    if (
                        self.__border_val
                        >= self.CONFIG["analyzer"]["enough_len_of_tokens"]
                        or self.__last_page
                    ):
                        print(f"Len of batch: {len(self.__line_batch)}")

                        parse = self.__morph_model_pred__(self.__line_batch)

                        if parse == []:
                            self.__line_batch = []
                            self.__border_val = 0

                            continue

                        tags = self.__get_dict_of_tags__(parse)

                        if tags == {}:
                            self.__line_batch = []
                            self.__border_val = 0

                            continue

                        self.__add_tags__(tags)
                        self.__line_batch = []
                        self.__border_val = 0

    def recognition(self, model: str, Parser: Any) -> None:

        if not self.pages:
            self.pages = {}

        if "confl" in self.CONFIG["mode"] and self.file_name_all_href and self.auth:
            self.files = self.__get_files_from_urls__()

        self.model = build_model(model)
        print("MODEL LOADED")

        with Progress() as progress:
            task = progress.add_task(
                "[green]Processing...", total=len(self.files_names)
            )
            start_time = time.time()

            self.__line_batch = []
            self.__border_val = 0
            self.__temp_batch = []
            self.__last_page = False

            for i, (file, file_name) in enumerate(zip(self.files, self.files_names)):

                if (i + 1) == len(self.files_names):
                    self.__last_page = True

                name = self.__format_name__(file_name)
                self.__current_name = name
                page_status = "DELETE"

                if file:
                    self.parser = Parser(file)
                    page_status = self.__check_page_version__()

                    if page_status == "NEW" or page_status == "UPDATE":
                        self.__add_all_tags__()

                print(
                    f"TIME: {(time.time() - start_time):.4f} STATUS: {page_status}  FILE: {name}"
                )
                start_time = time.time()
                progress.update(task, advance=1)

        self.model.destroy()
        print("MODEL DESTROYED. TIME: ", (time.time() - start_time))

    def save(self) -> None:

        if self.tags != {}:
            print("SAVING...")

            if not os.path.exists(self.CONFIG["save"]["path_save"]):
                os.mkdir(self.CONFIG["save"]["path_save"])

            with open(
                os.path.join(
                    self.CONFIG["save"]["path_save"], self.CONFIG["save"]["morph_tags"]
                ),
                "w",
                encoding="utf8",
            ) as json_file:

                json_file.write(
                    json.dumps(self.tags, indent=4, sort_keys=True, ensure_ascii=False)
                )
                json_file.close()

            if "morph_tags_swap" in self.CONFIG["save"].keys():
                tags_swap = {}

                for i in self.tags:
                    tags_swap.setdefault(self.tags[i], []).append(i)

                with open(
                    os.path.join(
                        self.CONFIG["save"]["path_save"],
                        self.CONFIG["save"]["morph_tags_swap"],
                    ),
                    "w",
                    encoding="utf8",
                ) as json_file:

                    json_file.write(
                        json.dumps(
                            tags_swap, indent=4, sort_keys=True, ensure_ascii=False
                        )
                    )
                    json_file.close()

    def create_morph_files(self) -> None:

        if "morph" in self.CONFIG.keys() and self.CONFIG["morph"]:

            if "post" in self.CONFIG["mode"]:
                parser = TrackerLoaderHtmlParser
            if "conf" in self.CONFIG["mode"]:
                parser = HtmlParser

            self.recognition(
                os.path.join(
                    self.CONFIG["path_configs"], self.CONFIG["path_to_models"]["morph"]
                ),
                Parser=parser,
            )

        else:
            raise ValueError("Morphological recognition not set in CONFIG!")

        self.save()


if __name__ == "__main__":
    start_time = time.time()

    morph = MorphoTager()
    morph.create_morph_files()

    print("TOTAL TIME: ", (time.time() - start_time))
