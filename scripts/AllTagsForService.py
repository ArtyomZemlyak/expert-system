import os
from os import listdir
from os.path import isfile, join
from operator import itemgetter

from typing import List, Tuple
from pprint import pprint
import string
import json
import re

import networkx as nx
import numpy as np
from numpy import sqrt
from rich.progress import Progress

from EntityTag import EntityTag


CONFIG = json.loads(open("scripts/config.json", "r").read())

VALID_CHARS = CONFIG["finder"]["valid_chars"]
VALID_CHARS += string.ascii_lowercase
VALID_CHARS += string.ascii_uppercase

STOP_WORDS_FINDER = set(CONFIG["finder"]["stop_words"])
STOP_WORDS_FINDER.update("йцукенгшщзхъфывапролджэячсмитьбюё")

COUNT_OF_TAGS_FOR_DISPLAY = CONFIG["finder"]["count_of_tags_for_display"]
COUNT_OF_RES_FOR_DISPLAY = CONFIG["finder"]["count_of_res_for_display"]

BLACKLIST = set(CONFIG["save"].values())

MAX_FILTER_OF_COUNT = CONFIG["visualizer"]["max_filter_of_count"]


class AllTagsForService:
    def __init__(self, config: str = None) -> None:

        if config:
            self.CONFIG = config
        else:
            self.CONFIG = CONFIG

        self.pages = {}
        self.freq_all_tags = {}

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

        self.tags_index = json.loads(
            open(
                os.path.join(
                    self.CONFIG["save"]["path_save"],
                    self.CONFIG["save"]["all_tags_idx"],
                ),
                "r",
                encoding="utf8",
            ).read()
        )

    def __check_STOP_WORDS_FINDER__(self, list_of_words: List[str]) -> bool:
        stopped_words = 0

        for word in list_of_words:
            if word in STOP_WORDS_FINDER:
                stopped_words += 1

        if stopped_words == len(list_of_words):
            return False

        return True

    def __get_freq_all_tags__(self) -> dict:

        freq_all_tags = {}

        with Progress() as progress:
            task = progress.add_task("[green]Processing...", total=len(self.tags_index))

            for idx, tag in self.tags_index.items():
                counter = 0

                for name, page_data in self.pages.items():
                    if idx in page_data["tags"]["values"].keys():
                        counter += 1

                freq = counter / len(self.pages)
                freq_all_tags[idx] = freq
                progress.update(task, advance=1)

        return freq_all_tags

    def __get_all_sorted_tags__(
        self, page_name: str, find: str = "", freq_coeff: float = -1
    ) -> List[Tuple[str, int]]:

        all_tags = self.pages[page_name]["tags"]["values"]

        if freq_coeff != -1 and self.freq_all_tags != {}:
            _all_tags = {}
            for tag, counter in all_tags.items():
                if (
                    tag in self.freq_all_tags.keys()
                    and self.freq_all_tags[tag] < freq_coeff
                ):
                    _all_tags[tag] = counter
            all_tags = _all_tags

        sorted_all_tags = sorted(
            [
                [tag, counter]
                for tag, counter in all_tags.items()
                if tag in self.tags_index.keys()
            ],
            key=itemgetter(1),
            reverse=True,
        )
        sorted_all_tags = [
            [EntityTag.pretty_tag(self.tags_index[tag]), counter]
            for tag, counter in sorted_all_tags
            if self.__check_STOP_WORDS_FINDER__(self.tags_index[tag]["tag"])
        ]
        if find != "":
            sorted_all_tags = [
                [tag, counter] for tag, counter in sorted_all_tags if find in tag
            ]

        return sorted_all_tags

    def save(self):

        self.freq_all_tags = self.__get_freq_all_tags__()

        for name, page_data in self.pages.items():

            ajson = {"list": self.__get_all_sorted_tags__(name)}
            with open(
                f"TEST/services_tags/Отсортированный_список_тегов_{name}.json",
                "w",
                encoding="utf8",
            ) as json_file:
                json_file.write(
                    json.dumps(ajson, indent=4, sort_keys=True, ensure_ascii=False)
                )
                json_file.close()

            ajson = {"list": self.__get_all_sorted_tags__(name, "(text)")}
            with open(
                f"TEST/services_tags/Отсортированный_список_тегов__{name}_только_сущности.json",
                "w",
                encoding="utf8",
            ) as json_file:
                json_file.write(
                    json.dumps(ajson, indent=4, sort_keys=True, ensure_ascii=False)
                )
                json_file.close()

            freq_coeff = 0.8

            ajson = {"list": self.__get_all_sorted_tags__(name, freq_coeff=freq_coeff)}
            with open(
                f"TEST/services_tags/Отсортированный_список_тегов_{name}_fq{int(freq_coeff*100)}.json",
                "w",
                encoding="utf8",
            ) as json_file:
                json_file.write(
                    json.dumps(ajson, indent=4, sort_keys=True, ensure_ascii=False)
                )
                json_file.close()

            ajson = {
                "list": self.__get_all_sorted_tags__(
                    name, "(text)", freq_coeff=freq_coeff
                ),
            }
            with open(
                f"TEST/services_tags/Отсортированный_список_тегов__{name}_только_сущности_fq{int(freq_coeff*100)}.json",
                "w",
                encoding="utf8",
            ) as json_file:
                json_file.write(
                    json.dumps(ajson, indent=4, sort_keys=True, ensure_ascii=False)
                )
                json_file.close()


if __name__ == "__main__":
    atfs = AllTagsForService()
    atfs.save()
