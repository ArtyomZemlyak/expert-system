# Многоязычная растановка тегов:
#     PERSON      People including fictional
#     NORP        Nationalities or religious or political groups
#     FACILITY    Buildings, airports, highways, bridges, etc.
#     ORGANIZATION    Companies, agencies, institutions, etc.
#     GPE         Countries, cities, states
#     LOCATION    Non-GPE locations, mountain ranges, bodies of water
#     PRODUCT     Vehicles, weapons, foods, etc. (Not services)
#     EVENT       Named hurricanes, battles, wars, sports events, etc.
#     WORK OF ART     Titles of books, songs, etc.
#     LAW         Named documents made into laws
#     LANGUAGE    Any named language
#     DATE        Absolute or relative dates or periods
#     TIME        Times smaller than a day
#     PERCENT     Percentage (including “%”)
#     MONEY       Monetary values, including unit
#     QUANTITY    Measurements, as of weight or distance
#     ORDINAL     “first”, “second”
#     CARDINAL    Numerals that do not fall under another type

# ner_bert_ent_and_type_rus:
#       O-TAG - for other tokens
#       E-TAG - for entities
#       T-TAG - corresponds to tokens of entity types

# cd DeepPavlov
# pip install -e .
# cd ..
# # For NER taggig:
# python -m deeppavlov install DeepPavlov/deeppavlov/configs/ner/ner_ontonotes_bert_mult.json
# python -m deeppavlov download DeepPavlov/deeppavlov/configs/ner/ner_ontonotes_bert_mult.json
# # For Entity tagging:
# python -m deeppavlov install DeepPavlov/deeppavlov/configs/ner/ner_bert_ent_and_type_rus.json
# python -m deeppavlov download DeepPavlov/deeppavlov/configs/ner/ner_bert_ent_and_type_rus.json
# # For Syntactic parsing:
# python -m deeppavlov install DeepPavlov/deeppavlov/configs/syntax/syntax_ru_syntagrus_bert.json
# python -m deeppavlov download DeepPavlov/deeppavlov/configs/syntax/syntax_ru_syntagrus_bert.json
# # For Morphological Tagging:
# python -m deeppavlov install DeepPavlov/deeppavlov/configs/morpho_tagger/BERT/morpho_ru_syntagrus_bert.json
# python -m deeppavlov download DeepPavlov/deeppavlov/configs/morpho_tagger/BERT/morpho_ru_syntagrus_bert.json

# pip install rich

# pip install fast-mosestokenizer
# pip install nltk
# nltk.download("stopwords")
# pip install pymystem3

# # Dynamic pages loading:
# pip install playwright
# playwright install chromium
# playwright install-deps chromium

import os
from os import listdir
from os.path import isfile, join
from collections import Counter
from typing import Any, Generator, List, Tuple, Union
import copy
import time
import json
import re

import requests
import numpy as np
from rich.progress import Progress
from playwright.sync_api import sync_playwright

from deeppavlov import build_model
from mosestokenizer import MosesTokenizer
from pymystem3 import Mystem
from nltk.corpus import stopwords

try:
    from .HtmlParser import HtmlParser, HtmlGosuParser, HtmlPguParser
    from .EntityTag import EntityTag, EntityTagJsonEncoder
    from .TrackerLoader import TrackerLoader, TrackerLoaderHtmlParser
except ImportError:
    from HtmlParser import HtmlParser, HtmlGosuParser, HtmlPguParser
    from EntityTag import EntityTag, EntityTagJsonEncoder
    from TrackerLoader import TrackerLoader, TrackerLoaderHtmlParser


CONFIG = {
    **json.loads(open("config.json", "r").read()),
    **json.loads(open("scripts/config.json", "r").read()),
}

BLACKLIST = set(CONFIG["save"].values())


class Analyzer:
    def __init__(self, config: str = None) -> None:

        if config:
            self.CONFIG = config
        else:
            self.CONFIG = CONFIG

        self.pages = {}
        self.pages_versions = {}
        self.tags_idx = {}

        self.model = None
        self.parser = None

        self.morph_tags = None
        self.last_iter = False

        self.TAGS_FIND = {
            **self.CONFIG["analyzer"]["tags_kinds"],
            **self.CONFIG["analyzer"]["tags_find"],
        }

        if not os.path.exists(self.CONFIG["save"]["path_save"]):
            os.mkdir(self.CONFIG["save"]["path_save"])

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

        self.file_name_all_href = None
        self.auth = None
        self.browser = None

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

        if (
            "confl" in self.CONFIG["mode"] and self.file_name_all_href and self.auth
        ) or "gosu" in self.CONFIG["mode"]:
            if "confl" in self.CONFIG["mode"]:
                print("Mode: confluence. Auth finded.")
            else:
                print("Mode: gosuslugi.")

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

        if "morph" in self.CONFIG.keys() and self.CONFIG["morph"]:
            if os.path.exists(
                os.path.join(
                    self.CONFIG["save"]["path_save"], self.CONFIG["save"]["morph_tags"]
                )
            ):

                self.morph_tags = json.loads(
                    open(
                        os.path.join(
                            self.CONFIG["save"]["path_save"],
                            self.CONFIG["save"]["morph_tags"],
                        ),
                        "r",
                        encoding="utf8",
                    ).read()
                )

                if self.morph_tags == {}:
                    raise ValueError(
                        f"File {self.CONFIG['save']['morph_tags']} is empty!"
                    )

            else:
                raise FileExistsError(
                    f"File {self.CONFIG['save']['morph_tags']} dont exist!"
                )

    def __format_name__(self, file_name: str) -> str:

        if self.file_name_all_href and self.auth:
            name = re.sub(r"/", "__", file_name)
            name = re.sub("[\\.]", "___", name)

        else:
            name = re.sub("[\\.]{1}[A-Za-z\\d]+", "", file_name)

        return name

    def __model_ner_pred__(
        self, line_batch: List[str]
    ) -> Tuple[List[str], Union[List[str], List[float]]]:

        tokens_batch = []
        predictions_batch = []

        try:
            # start_time = time.time()

            tokens_batch, predictions_batch = self.model(line_batch)

            # print(f'Model pred time: {time.time() - start_time}')

        except Exception as e:  # (RuntimeError, IndexError)
            print("1 Exception in __model_ner_pred__:", e)

            for line in line_batch:
                tokens = []
                predictions = []
                line_split = line.split("\n")

                try:
                    # start_time = time.time()

                    _tokens, _predictions = self.model(line_split)

                    # print(f'Model pred time: {time.time() - start_time}')

                    for token_batch, prediction_batch in zip(_tokens, _predictions):
                        tokens.extend(token_batch)
                        predictions.extend(prediction_batch)

                except Exception as e:
                    print("2:", e)

                    for line_ in line_split:
                        line_split_ = line_.split(";")

                        try:
                            # start_time = time.time()

                            _tokens, _predictions = self.model(line_split_)

                            # print(f'Model pred time: {time.time() - start_time}')

                            for token_batch, prediction_batch in zip(
                                _tokens, _predictions
                            ):
                                tokens.extend(token_batch)
                                predictions.extend(prediction_batch)

                        except Exception as e:
                            print("3:", e)

                            for line__ in line_split_:
                                line_split__ = line__.split(",")

                                try:
                                    # start_time = time.time()

                                    _tokens, _predictions = self.model(line_split__)

                                    # print(f'Model pred time: {time.time() - start_time}')

                                    for token_batch, prediction_batch in zip(
                                        _tokens, _predictions
                                    ):
                                        tokens.extend(token_batch)
                                        predictions.extend(prediction_batch)

                                except Exception as e:
                                    print("4:", e)
                                    # return tokens_batch, predictions_batch

                tokens_batch.append(tokens)
                predictions_batch.append(predictions)

        return (tokens_batch, predictions_batch)

    def __model_nlp_pred__(self, line_batch: List[str]) -> Tuple[List[str], List[str]]:

        tokens_batch = []
        predictions_batch = []

        for line in line_batch:
            tokens = self.nlp.tokenize(line)

            if self.mystem and self.__stopwords:
                tokens = [token.lower() for token in tokens]
                tokens = [
                    self.mystem.lemmatize(token)[0]
                    for token in tokens
                    if token not in self.__stopwords
                ]

            tokens_batch.append(tokens)

        predictions_batch = tokens_batch

        return (tokens_batch, predictions_batch)

    def __model_syn_pred__(
        self, line_batch: List[str]
    ) -> Tuple[List[str], Union[List[str], List[float]]]:

        tokens_batch = []
        predictions_batch = []

        try:
            _predictions_batch = self.model(line_batch)

            for prediction in _predictions_batch:
                _words = prediction.split("\n")
                words = []

                for word in _words:
                    preds = [i for i in word.split("\t") if i != "_"]

                    if preds != [""] and preds != [] and preds != "":
                        words.append(preds)

                predictions_batch.append(words)

        except Exception as e:  # (RuntimeError, IndexError)
            print("1 Exception in __model_syn_pred__:", e)

            for line in line_batch:
                predictions = []
                line_split = line.split("\n")

                try:
                    _predictions_batch = self.model(line_split)

                    for prediction in _predictions_batch:
                        _words = prediction.split("\n")
                        words = []

                        for word in _words:
                            preds = [i for i in word.split("\t") if i != "_"]

                            if preds != [""] and preds != [] and preds != "":
                                words.append(preds)

                        predictions.extend(words)

                except Exception as e:
                    print("2:", e)

                    for line_ in line_split:
                        line_split_ = line_.split(";")

                        try:
                            _predictions_batch = self.model(line_split_)

                            for prediction in _predictions_batch:
                                _words = prediction.split("\n")
                                words = []

                                for word in _words:
                                    preds = [i for i in word.split("\t") if i != "_"]

                                    if preds != [""] and preds != [] and preds != "":
                                        words.append(preds)

                                predictions.extend(words)

                        except Exception as e:
                            print("3:", e)

                            for line__ in line_split_:
                                line_split__ = line__.split(",")

                                try:
                                    _predictions_batch = self.model(line_split__)

                                    for prediction in _predictions_batch:
                                        _words = prediction.split("\n")
                                        words = []

                                        for word in _words:
                                            preds = [
                                                i for i in word.split("\t") if i != "_"
                                            ]

                                            if (
                                                preds != [""]
                                                and preds != []
                                                and preds != ""
                                            ):
                                                words.append(preds)

                                        predictions.extend(words)

                                except Exception as e:
                                    print("4:", e)
                                    # tokens_batch = predictions_batch
                                    # return tokens_batch, predictions_batch

                predictions_batch.append(predictions)

        tokens_batch = predictions_batch
        return (tokens_batch, predictions_batch)

    def __model_mor_pred__(
        self, line_batch: List[str]
    ) -> Tuple[List[str], Union[List[str], List[float]]]:

        tokens_batch = []
        predictions_batch = []

        tokens_batch = predictions_batch
        return (tokens_batch, predictions_batch)

    def __get_files_from_urls__(self):

        for file_name in self.files_names:

            if self.browser:
                try:
                    page = self.browser.new_page()
                    if "#" not in file_name:
                        page.goto(self.url_href + file_name)
                    else:
                        page.goto("https://gosuslugi.bashkortostan.ru/" + file_name)
                    # page.click('[ng-click="frgu.consulting.admReglament.blockReglament = !frgu.consulting.admReglament.blockReglament"]')
                    # page.click('[data-ng-click="getModalData(frgu.consulting.admReglament.items[0].reglamentPopup)"]')
                    # page.wait_for_timeout(500)
                    page_content = page.content()
                    yield page_content

                except Exception as e:
                    yield False

            else:
                try:
                    yield requests.get(self.url_href + file_name, auth=self.auth).text
                except Exception as e:
                    yield False

    def __get_field_priority__(
        self, field_property: Union[int, dict]
    ) -> Tuple[int, bool]:

        if type(field_property) == int:
            field_priority = field_property
            field_tft = False

        elif type(field_property) == dict:
            field_priority = field_property["priority"]
            field_tft = field_property["tags_from_text"]

        return (field_priority, field_tft)

    def __get_step_tag_pos__(self, current_tag_pos, step):
        i, j = current_tag_pos.split(":")
        return f"{i}:{int(j) + step}"

    def __get_tag_from_idx__(self, idx: int) -> EntityTag:
        return copy.deepcopy(self.tags_idx[idx])

    def __del_tag_idx__(self, idx: int) -> None:
        try:
            del self.tags_idx[idx]
        except KeyError:
            pass

    def __add_tag_idx__(self, tag: EntityTag) -> bool:

        idx = tag.idx
        new_tag = False

        if idx in self.tags_idx.keys():
            self.tags_idx[idx].update(tag)
        else:
            new_tag = True
            self.tags_idx[idx] = tag

        return new_tag

    def __add_tag__(
        self, token: str, tags_type: str, tag_pos: str, field_priority: int
    ) -> None:

        if self.__check_text__(token):
            tag_val = tags_type

            if tag_pos not in self.pages[self._current_name]["tags"]["entries"].keys():

                if tags_type not in self.CONFIG["analyzer"]["tags_kinds"].keys():
                    tags_type = "text"
                text_tag = self.__check_tag__(token)

                if self.morph_tags and text_tag in self.morph_tags.keys():
                    text_tag = self.morph_tags[text_tag]

                text_tag = text_tag.lower()
                tag = None

                if (
                    "I-" in tags_type and self.__b_tag
                ):  # and len(self.__list_of_tags_pos) > 0:
                    tags_type = tags_type[2:]
                    prev_pos = self.__list_of_tags_pos[-1]
                    prev_idx = self.pages[self._current_name]["tags"]["entries"][
                        prev_pos
                    ]["idx"]
                    tag = self.__get_tag_from_idx__(prev_idx)

                    if self.__new_tag:
                        self.__del_tag_idx__(prev_idx)
                    tag.update_tag(text_tag)

                    for prev_pos in reversed(self.__list_of_tags_pos):
                        if (
                            self.pages[self._current_name]["tags"]["entries"][prev_pos][
                                "idx"
                            ]
                            == prev_idx
                        ):
                            self.pages[self._current_name]["tags"]["entries"][prev_pos][
                                "idx"
                            ] = tag.idx

                        else:
                            break

                else:
                    if "B-" in tags_type:
                        self.__b_tag = True
                    else:
                        self.__b_tag = False

                    if "B-" in tags_type or "I-" in tags_type:
                        tags_type = tags_type[2:]
                    tag = EntityTag(tags_type, text_tag)

                self.pages[self._current_name]["tags"]["entries"][tag_pos] = {
                    "idx": tag.idx,
                    "val": set([tag_val]),
                    "text": token,
                    "priority": field_priority,
                }
                self.__new_tag = self.__add_tag_idx__(tag)
                self.__list_of_tags_pos.append(tag_pos)

            else:
                self.__b_tag = False
                self.__new_tag = False
                self.pages[self._current_name]["tags"]["entries"][tag_pos]["val"].add(
                    tag_val
                )

    def __add_relation__(
        self, pred: str, tags_type: str, tag_pos: str, other_pos: str
    ) -> None:

        if (
            tag_pos in self.pages[self._current_name]["tags"]["entries"].keys()
            and other_pos in self.pages[self._current_name]["tags"]["entries"].keys()
        ):
            number, token, other, type_rel = pred

            tag_idx = self.pages[self._current_name]["tags"]["entries"][tag_pos]["idx"]
            other_idx = self.pages[self._current_name]["tags"]["entries"][other_pos][
                "idx"
            ]

            if tag_idx != other_idx:
                if (
                    "relation"
                    not in self.pages[self._current_name]["tags"]["entries"][
                        tag_pos
                    ].keys()
                ):
                    self.pages[self._current_name]["tags"]["entries"][tag_pos][
                        "relation"
                    ] = {other_idx: type_rel}
                else:
                    self.pages[self._current_name]["tags"]["entries"][tag_pos][
                        "relation"
                    ][other_idx] = type_rel

                rel_tag = EntityTag("rel", type_rel)
                self.tags_idx[rel_tag.idx] = rel_tag
                self.tags_idx[
                    self.pages[self._current_name]["tags"]["entries"][tag_pos]["idx"]
                ].add_relation(other_idx, rel_tag.idx)

            else:
                text_tag = self.pages[self._current_name]["tags"]["entries"][tag_pos][
                    "text"
                ]
                text_tag = self.__check_tag__(text_tag)
                if self.morph_tags and text_tag in self.morph_tags.keys():
                    text_tag = self.morph_tags[text_tag]
                text_tag = text_tag.lower()

                other_tag = self.pages[self._current_name]["tags"]["entries"][
                    other_pos
                ]["text"]
                other_tag = self.__check_tag__(other_tag)
                if self.morph_tags and other_tag in self.morph_tags.keys():
                    other_tag = self.morph_tags[other_tag]
                other_tag = other_tag.lower()

                rel_tag = EntityTag("rel", type_rel)
                self.tags_idx[rel_tag.idx] = rel_tag
                self.tags_idx[
                    self.pages[self._current_name]["tags"]["entries"][tag_pos]["idx"]
                ].add_self_relation(text_tag, other_tag, rel_tag.idx)

    def __add_tags__(self, task: str) -> None:

        tags_pos = {}
        self.__b_tag = False
        self.__new_tag = False
        self.__list_of_tags_pos = []

        if "gosu" in CONFIG["mode"]:
            for field_idx, (field_name, field_property) in enumerate(
                self.parser.FIELDS_FIND.items()
            ):
                field_priority = 1
                tag_pos = f"{field_idx}:0:0"
                tags_pos[field_name] = tag_pos
                cur_tag = self.parser.tags_counter_idx[field_name]
                self.__add_tag__(cur_tag, "text_tree", tag_pos, field_priority)

            title_of_page = self.parser.get_field_text("title")
            self.pages[self._current_name]["fields"]["title"] = title_of_page

        for field_idx, (field_name, field_property) in enumerate(
            self.parser.FIELDS_FIND.items()
        ):
            if "gosu" not in CONFIG["mode"]:
                field_priority, field_tft = self.__get_field_priority__(field_property)
            else:
                field_priority = 1  # ? add checking h3 and h2 and h4 ...
                field_tft = False

                if "@prev_key" in field_property.keys():
                    prev_tag = field_property["@prev_key"]
                    prediction = (
                        0,
                        0,
                        0,
                        "text_tree",
                    )  # checking level of relation ???
                    tag_pos = f"{field_idx}:0:0"
                    other_pos = tags_pos[prev_tag]
                    self.__add_relation__(prediction, "text_tree", tag_pos, other_pos)

            field_text = self.parser.get_field_text(field_name)
            if field_name == "title":
                self.pages[self._current_name]["fields"]["title"] = field_text

            if "conf" in CONFIG["mode"]:
                context = field_text.split("\n")
            if "post" in CONFIG["mode"]:
                context = [field_text]
            if "gosu" in CONFIG["mode"]:
                context = field_text

            if type(context) == list:
                for i, line in enumerate(context):

                    if self.__check_text_line__(line):
                        if self.__temp_batch:
                            self.__f_batch.append(self.__temp_batch[0])
                            self.__i_batch.append(self.__temp_batch[1])
                            self.__line_batch.append(self.__temp_batch[2])
                            self.__name_batch.append(self.__temp_batch[3])
                            self.__border_val += self.__temp_batch[4]
                            self.__temp_batch = []

                        len_tokens = len(line.split(" "))

                        if (
                            self.__border_val + len_tokens
                            < self.CONFIG["analyzer"]["max_len_of_tokens"]
                            or len_tokens
                            >= self.CONFIG["analyzer"]["enough_len_of_tokens"]
                        ):

                            self.__f_batch.append((field_idx, field_priority))
                            self.__i_batch.append(i)
                            self.__line_batch.append(line)
                            self.__name_batch.append(self._current_name)
                            self.__border_val += len_tokens
                        else:
                            self.__temp_batch = (
                                (field_idx, field_priority),
                                i,
                                line,
                                self._current_name,
                                len_tokens,
                            )

                    if (
                        self.__border_val
                        >= self.CONFIG["analyzer"]["enough_len_of_tokens"]
                        or self.__last_page
                    ):
                        # print(f'Len of batch: {len(self.__line_batch)}')

                        if task == "ner" or task == "entity":
                            tokens_batch, predictions_batch = self.__model_ner_pred__(
                                self.__line_batch
                            )
                        elif task == "syntactic":
                            tokens_batch, predictions_batch = self.__model_syn_pred__(
                                self.__line_batch
                            )
                        elif task == "morphological":
                            tokens_batch, predictions_batch = self.__model_mor_pred__(
                                self.__line_batch
                            )
                        elif self.nlp:
                            tokens_batch, predictions_batch = self.__model_nlp_pred__(
                                self.__line_batch
                            )
                        else:
                            raise ValueError("Model for processing not set!")

                        if tokens_batch == [] or predictions_batch == []:
                            self.__f_batch = []
                            self.__i_batch = []
                            self.__line_batch = []
                            self.__name_batch = []
                            self.__border_val = 0
                            continue

                        for tokens, predictions, name, (f_idx, f_priority), i in zip(
                            tokens_batch,
                            predictions_batch,
                            self.__name_batch,
                            self.__f_batch,
                            self.__i_batch,
                        ):

                            self._current_name = name
                            self.__b_tag = False
                            self.__new_tag = False
                            self.__list_of_tags_pos = []

                            for j, (token, prediction) in enumerate(
                                zip(tokens, predictions)
                            ):
                                tag_pos = f"{f_idx}:{i+1}:{j}"

                                if task == "ner":
                                    if prediction in self.TAGS_FIND.keys():
                                        self.__add_tag__(
                                            token, prediction, tag_pos, f_priority
                                        )
                                elif task == "entity":
                                    if np.argmax(prediction) == 1:
                                        self.__add_tag__(
                                            token, task, tag_pos, f_priority
                                        )
                                elif task == "text":
                                    self.__add_tag__(token, task, tag_pos, f_priority)
                                elif task == "syntactic" or task == "morphological":
                                    other_pos = (
                                        f"{f_idx}:{i+1}:{int(prediction[2]) - 1}"
                                    )
                                    self.__add_relation__(
                                        prediction, task, tag_pos, other_pos
                                    )
                                    if "gosu" in CONFIG["mode"]:
                                        other_pos_ = (
                                            f"{f_idx}:{0}:{0}"  # tags_pos[field_name]
                                        )
                                        prediction_ = (
                                            0,
                                            0,
                                            0,
                                            "text_tree",
                                        )  # checking level of relation ???
                                        self.__add_relation__(
                                            prediction_,
                                            "text_tree",
                                            tag_pos,
                                            other_pos_,
                                        )

                                if (
                                    self.last_iter
                                    and field_tft
                                    and "text_tags" in CONFIG.keys()
                                    and CONFIG["text_tags"]
                                ):
                                    self.__add_tag__(token, "text", tag_pos, f_priority)

                            self.__add_tags_val__()

                        self.__f_batch = []
                        self.__i_batch = []
                        self.__line_batch = []
                        self.__name_batch = []
                        self.__border_val = 0

    def __add_tags_val__(self, normalize: bool = None) -> None:
        coeff = 1

        if not normalize and "normalize" in self.CONFIG["analyzer"].keys():
            normalize = self.CONFIG["analyzer"]["normalize"]
        else:
            normalize = False

        if normalize:
            tags_summ = 1
            for val in self.pages[self._current_name]["tags"]["entries"].values():
                entry_val = 1

                for v in val["val"]:
                    entry_val *= self.TAGS_FIND[v]
                entity_val = Counter({val["idx"]: val["priority"] * entry_val})

                for val_ in entity_val.values():
                    tags_summ += val_

            coeff = tags_summ

        for val in self.pages[self._current_name]["tags"]["entries"].values():
            entry_val = 1

            for v in val["val"]:
                entry_val *= self.TAGS_FIND[v]
            if coeff != 1:
                entry_val /= coeff

            entity_val = Counter({val["idx"]: val["priority"] * entry_val})
            self.pages[self._current_name]["tags"]["values"] += entity_val

    def __check_text__(self, text: str) -> bool:

        if re.sub(f"[{self.CONFIG['analyzer']['stop_symbols']}]", "", text) != text:

            return False

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
            and re.sub(f"[^{self.CONFIG['analyzer']['word_symbols']}]", "", line) != ""
        ):
            return True
        return False

    def __check_page_version__(self) -> str:

        name = self._current_name

        if name not in self.pages.keys():
            self.pages[name] = {
                "tags": {"values": Counter([]), "entries": {}},
                "fields": {},
            }
            self.pages_versions[name] = self.parser.get_page_version()

            return "NEW"

        else:
            if "page_version" in self.pages[name].keys():
                current_remote_page_version = self.parser.get_page_version()

                if self.pages[name]["page_version"] < current_remote_page_version:
                    self.pages_versions[name] = current_remote_page_version
                    self.pages[name] = {
                        "tags": {"values": Counter([]), "entries": {}},
                        "fields": {},
                    }

                    return "UPDATE"

            else:
                return "NEW"

        return "ACTUAL"

    def __save_file__(self, data: Any, file_name: str, path_save: str = None) -> None:

        if not path_save:
            path_save = self.CONFIG["save"]["path_save"]

        with open(
            os.path.join(path_save, file_name), "w", encoding="utf8"
        ) as json_file:
            json_file.write(
                json.dumps(
                    data,
                    indent=4,
                    sort_keys=True,
                    ensure_ascii=False,
                    cls=EntityTagJsonEncoder,
                )
            )
            json_file.close()

    def recognition(self, task: str, Parser: Any) -> None:

        if not self.pages:
            self.pages = {}

        self.model = None

        if task in self.CONFIG["path_to_models"].keys():
            self.model = build_model(
                os.path.join(
                    self.CONFIG["path_configs"], self.CONFIG["path_to_models"][task]
                )
            )
            print(f"MODEL LOADED: {task}")

        if "ner" in task:
            tags_type = "ner"
        elif "ent" in task:
            tags_type = "entity"
        elif "syn" in task:
            tags_type = "syntactic"
        elif "morph" in task:
            tags_type = "morphological"
        elif "text" in task:
            tags_type = "text"
            self.nlp = MosesTokenizer("ru")
            self.mystem = Mystem()
            self.__stopwords = {
                *stopwords.words("russian"),
                *stopwords.words("english"),
                *self.CONFIG["analyzer"]["stop_words"],
            }
        else:
            raise TypeError("Cant recognize type of task")

        if (
            "confl" in self.CONFIG["mode"] and self.file_name_all_href and self.auth
        ) or "gosu" in self.CONFIG["mode"]:
            self.files = self.__get_files_from_urls__()

        with Progress() as progress:
            task = progress.add_task(
                "[green]Processing...", total=len(self.files_names)
            )
            start_time = time.time()

            self.__f_batch = []
            self.__i_batch = []
            self.__line_batch = []
            self.__name_batch = []
            self.__temp_batch = []
            self.__border_val = 0
            self.__last_page = False

            for i, (file, file_name) in enumerate(zip(self.files, self.files_names)):
                if (i + 1) == len(self.files_names):
                    self.__last_page = True

                name = self.__format_name__(file_name)
                self._current_name = name
                page_status = "NEW"

                if file:
                    self.parser = Parser(file)
                    page_status = self.__check_page_version__()
                    if page_status != "ACTUAL":
                        self.__add_tags__(tags_type)
                else:
                    if name in self.pages:
                        del self.pages[name]
                    if name in self.pages_versions:
                        del self.pages_versions[name]

                # print(f"TIME: {(time.time() - start_time):.4f} STATUS: {page_status}  FILE: {name}")
                start_time = time.time()
                progress.update(task, advance=1)

        if self.model:
            self.model.destroy()
        self.first_time = False
        print("MODEL DESTROYED. TIME: ", (time.time() - start_time))

    def update_pages_versions(self) -> None:

        if self.pages_versions != {}:
            for page_name, page_version in self.pages_versions.items():
                self.pages[page_name]["page_version"] = page_version
                print(f"Page: {page_name} updated!")

    def save(self) -> None:

        if self.pages_versions != {}:
            if not os.path.exists(self.CONFIG["save"]["path_save"]):
                os.mkdir(self.CONFIG["save"]["path_save"])

            with Progress() as progress:
                task = progress.add_task(
                    "[green]Saving...", total=len(self.files_names) + 60
                )

                for file_name in self.files_names:
                    name = self.__format_name__(file_name)
                    self.__save_file__(self.pages[name], name + ".json")

                    progress.update(task, advance=1)

                if "all_tags" in self.CONFIG["save"].keys():
                    all_tags = Counter([])
                    for file_name in self.files_names:
                        name = self.__format_name__(file_name)
                        all_tags = all_tags + Counter(
                            self.pages[name]["tags"]["values"]
                        )

                    all_tags_dict = {**all_tags}
                    self.__save_file__(all_tags_dict, self.CONFIG["save"]["all_tags"])

                    progress.update(task, advance=10)

                    if "all_tags_swap" in self.CONFIG["save"].keys():
                        all_tags_dict_swap = {}
                        for i in all_tags_dict:
                            all_tags_dict_swap.setdefault(all_tags_dict[i], []).append(
                                i
                            )
                        self.__save_file__(
                            all_tags_dict_swap, self.CONFIG["save"]["all_tags_swap"]
                        )

                    progress.update(task, advance=10)

                    if "all_tags_tags" in self.CONFIG["save"].keys():
                        all_tags_dict_tags = {}
                        for idx, tag in self.tags_idx.items():
                            tagl = list(tag.tag)
                            for tag_ in tagl:
                                if tag_ not in all_tags_dict_tags.keys():
                                    all_tags_dict_tags[tag_] = set([idx])
                                else:
                                    all_tags_dict_tags[tag_].add(idx)
                        self.__save_file__(
                            all_tags_dict_tags, self.CONFIG["save"]["all_tags_tags"]
                        )

                    progress.update(task, advance=10)

                    if "all_tags_formatted" in self.CONFIG["save"].keys():
                        all_tags_dict_formatted = {}
                        for key, value in all_tags_dict.items():
                            if key in self.tags_idx:
                                tag = self.tags_idx[key]
                                tagl = list(tag.tag)
                                kind = "_".join(sorted(list(tag.kind)))
                                tag_ = " ".join(sorted(tagl))
                                all_tags_dict_formatted[tag_] = [
                                    {kind: tag_},
                                    tag_,
                                    value,
                                ]
                                for tag_ in tagl:
                                    all_tags_dict_formatted[tag_] = [
                                        {kind: tag_},
                                        tag_,
                                        int(value / len(tagl)),
                                    ]
                        self.__save_file__(
                            all_tags_dict_formatted,
                            self.CONFIG["save"]["all_tags_formatted"],
                        )

                    progress.update(task, advance=10)
                else:
                    progress.update(task, advance=40)

                if "all_tags_with_paths" in self.CONFIG["save"].keys():
                    all_tags_dict_with_paths = {}
                    for name, page_data in self.pages.items():
                        for tag, val in page_data["tags"]["values"].items():
                            if tag not in all_tags_dict_with_paths.keys():
                                all_tags_dict_with_paths[tag] = {}
                            if val not in all_tags_dict_with_paths[tag].keys():
                                all_tags_dict_with_paths[tag][val] = [name]
                            else:
                                all_tags_dict_with_paths[tag][val].append(name)
                                all_tags_dict_with_paths[tag][val].sort()
                    self.__save_file__(
                        all_tags_dict_with_paths,
                        self.CONFIG["save"]["all_tags_with_paths"],
                    )

                progress.update(task, advance=10)

                if "all_tags_idx" in self.CONFIG["save"].keys():
                    self.__save_file__(
                        self.tags_idx, self.CONFIG["save"]["all_tags_idx"]
                    )

                progress.update(task, advance=10)

    def delete(self) -> None:

        files_names = [
            f
            for f in listdir(self.CONFIG["save"]["path_save"])
            if isfile(join(self.CONFIG["save"]["path_save"], f))
        ]

        files_names_set = set(
            [self.__format_name__(file_name) for file_name in self.files_names]
        )

        with Progress() as progress:
            task = progress.add_task(
                "[green]Check and deleting...", total=len(files_names)
            )

            for file_name in files_names:
                if file_name not in BLACKLIST:
                    name = re.sub("[\\.]{1}[A-Za-z\\d]+", "", file_name)

                    if name not in files_names_set:
                        file_path = os.path.join(
                            self.CONFIG["save"]["path_save"], name + ".json"
                        )

                        if os.path.exists(file_path):
                            os.remove(file_path)
                            print(
                                f"The file {file_path} deleted!\nBecause the corresponding page in origin not found."
                            )
                        else:
                            print(f"The file {file_path} does not exist!")

                progress.update(task, advance=1)

    def create_db(self) -> None:

        if "pipeline" in self.CONFIG.keys():
            for task in self.CONFIG["pipeline"]:

                if "conf" in self.CONFIG["mode"]:
                    parser = HtmlParser
                if "gosu" in self.CONFIG["mode"]:
                    parser = HtmlGosuParser  # HtmlGosuParser HtmlPguParser
                if "post" in self.CONFIG["mode"]:
                    parser = TrackerLoaderHtmlParser
                if task == self.CONFIG["pipeline"][-1]:
                    self.last_iter = True

                self.recognition(task, Parser=parser)

        else:
            raise ValueError("No type of NER recognition set")

        self.update_pages_versions()
        self.save()
        if "delete" in self.CONFIG.keys() and self.CONFIG["delete"]:
            self.delete()

    def analyze(self) -> None:

        if "gosu" in self.CONFIG["mode"]:

            with sync_playwright() as p:
                self.browser = p.chromium.launch()
                self.create_db()
                self.browser.close()

        else:
            self.create_db()


if __name__ == "__main__":
    total_start_time = time.time()

    ner = Analyzer()
    ner.analyze()

    print("TOTAL TIME: ", (time.time() - total_start_time))
