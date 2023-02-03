# Morphological Tagging
#
# pip install deeppavlov
# python -m deeppavlov install deeppavlov/configs/classifiers/rel_ranking_bert_rus.json
# python -m deeppavlov download deeppavlov/configs/classifiers/rel_ranking_bert_rus.json
# python -m deeppavlov interact deeppavlov/configs/classifiers/rel_ranking_bert_rus.json
# pip install rich


import os
from os import listdir
from os.path import isfile, join
from collections import Counter
from typing import Any, Generator, List
import time
import json
import re

import requests
import numpy as np
from deeppavlov import build_model
from rich.progress import Progress

from NER import SAVE_ALL_TAGS, BLACKLIST
from ConfluencePageLoader import PATH_SAVE, SAVE_ALL_HREF, AUTH_CONFL, CONFIG
from HtmlParser import HtmlParser


PATH_CONFIGS = r"DeepPavlov/deeppavlov/configs"
PATH_TO_PARA_MODEL = os.path.join(
    PATH_CONFIGS, r"classifiers/rel_ranking_bert_rus.json"
)

PATH_CONTEXT = PATH_SAVE
SAVE_PARAPHRASE_TAGS = "_paraphrase_tags.json"

FILTER_SYMBOLS = "A-Za-z\\dА-Яа-я\\.\\-_"
BORDER_PEREPHRASE = 90
BATCH_SIZE = 64


class Paraphraser:
    def __init__(self) -> None:
        files_names = [
            f
            for f in listdir(PATH_SAVE)
            if isfile(join(PATH_SAVE, f)) and f not in BLACKLIST
        ]
        self.pages = {}
        for file_name in files_names:
            name = re.sub("[\\.]{1}[A-Za-z\\d]+", "", file_name)
            json_file_data = json.loads(
                open(os.path.join(PATH_SAVE, file_name), "r", encoding="utf8").read()
            )
            self.pages[name] = json_file_data
        if os.path.exists(os.path.join(PATH_SAVE, SAVE_ALL_TAGS)):
            self.all_tags = [
                key
                for key, value in json.loads(
                    open(
                        os.path.join(PATH_SAVE, SAVE_ALL_TAGS), "r", encoding="utf8"
                    ).read()
                ).items()
            ]
        else:
            raise FileExistsError(f"File {SAVE_ALL_TAGS} dont exist!")
        if os.path.exists(os.path.join(PATH_SAVE, SAVE_PARAPHRASE_TAGS)):
            self.tags = json.loads(
                open(
                    os.path.join(PATH_SAVE, SAVE_PARAPHRASE_TAGS), "r", encoding="utf8"
                ).read()
            )
            self.first_time = False
        else:
            self.tags = {}
            # self.diff_tags = {}
            self.first_time = True

    def __paraphrase_model_pred__(
        self, model, tag_one: str, tags_two: List[str]
    ) -> List[int]:
        result = []
        try:
            result = model([tag_one for i in range(len(tags_two))], tags_two).tolist()
        except RuntimeError:
            return result
        return result

    def __add_tags__(
        self, result: List[str], tag_one: str, tags_two: List[str]
    ) -> None:
        for pred, tag_two in zip(result, tags_two):
            if float(pred[1]) > BORDER_PEREPHRASE / 100:
                if tag_one not in self.tags.keys():
                    self.tags[tag_one] = []
                self.tags[tag_one].append(tag_two)
                if tag_two not in self.tags.keys():
                    self.tags[tag_two] = []
                self.tags[tag_two].append(tag_one)

    def __add_all_tags__(self, model, tag_one: str, tags_two: List[str]) -> None:
        result = self.__paraphrase_model_pred__(model, tag_one, tags_two)
        self.__add_tags__(result, tag_one, tags_two)

    def recognition(self, model: str) -> None:
        model = build_model(model)
        print("MODEL LOADED")

        with Progress() as progress:
            task = progress.add_task("[green]Processing...", total=len(self.all_tags))
            start_time = time.time()
            for j, tag_one in enumerate(self.all_tags):
                # if tag_one not in
                last_i = 0
                tags_two = []
                tags_counter = 0
                for i, tag_two in enumerate(self.all_tags[j:]):
                    tags_two.append(tag_two)
                    tags_counter += 1
                    if ((i - last_i) > BATCH_SIZE) or (
                        (len(self.all_tags[j:]) - i) == 1
                    ):
                        self.__add_all_tags__(model, tag_one, tags_two)
                        last_i = i
                        tags_two = []
                print(
                    f"TIME: {(time.time() - start_time):.4f} TAGS PROCESSED: {tags_counter}"
                )
                start_time = time.time()
                progress.update(task, advance=1)

        model.destroy()
        print("MODEL DESTROYED. TIME: ", (time.time() - start_time))

    def save(
        self,
        path_save: str = PATH_SAVE,
        file_name_para_tags: str = SAVE_PARAPHRASE_TAGS,
    ) -> None:
        if self.tags != {}:
            print("SAVING...")
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            json_file = open(
                os.path.join(path_save, file_name_para_tags), "w", encoding="utf8"
            )
            json_file.write(
                json.dumps(self.tags, indent=4, sort_keys=True, ensure_ascii=False)
            )
            json_file.close()

    def create_paraphrase_files(self) -> None:
        if "paraphrase" in CONFIG.keys() and CONFIG["morph"]:
            self.recognition(PATH_TO_PARA_MODEL)
        else:
            raise ValueError("Paraphrase recognition not set in CONFIG!")
        self.save()


if __name__ == "__main__":
    total_start_time = time.time()

    morph = Paraphraser()
    morph.create_paraphrase_files()

    print("TOTAL TIME: ", (time.time() - total_start_time))
