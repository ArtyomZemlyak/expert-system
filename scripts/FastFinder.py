# pip install fast-autocomplete[levenshtein]

from argparse import ArgumentError
import os
from os import listdir
from os.path import isfile, join
from operator import itemgetter
from collections import Counter, defaultdict

from typing import List, Tuple, Union
from pprint import pprint
import string
import json
import time
import re

from rich.progress import Progress
from deeppavlov import build_model
from fast_autocomplete import autocomplete_factory

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


class FastFinder:
    def __init__(
        self,
        config: str = None,
        model_qa: str = None,
        model_ner: str = None,
        valid_chars: str = VALID_CHARS,
    ) -> None:

        if config:
            self.CONFIG = config
        else:
            self.CONFIG = CONFIG

        with open(
            os.path.join(
                self.CONFIG["save"]["path_save"], self.CONFIG["save"]["valid_chars"]
            ),
            "w",
            encoding="utf8",
        ) as json_file:

            json_file.write(
                json.dumps(valid_chars, indent=4, sort_keys=True, ensure_ascii=False)
            )
            json_file.close()

        content_files = {
            "words": {
                "filepath": os.path.join(
                    self.CONFIG["save"]["path_save"],
                    self.CONFIG["save"]["all_tags_formatted"],
                ),
                "compress": True,  # means compress the graph data in memory
            },
            "valid_chars_for_string": {
                "filepath": os.path.join(
                    self.CONFIG["save"]["path_save"], self.CONFIG["save"]["valid_chars"]
                ),
                "compress": False,  # need
            },
        }

        self.pages = {}

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

        self.all_tags_tags = json.loads(
            open(
                os.path.join(
                    self.CONFIG["save"]["path_save"],
                    self.CONFIG["save"]["all_tags_tags"],
                ),
                "r",
                encoding="utf8",
            ).read()
        )

        self.all_tags_dict_with_paths = json.loads(
            open(
                os.path.join(
                    self.CONFIG["save"]["path_save"],
                    self.CONFIG["save"]["all_tags_with_paths"],
                ),
                "r",
                encoding="utf8",
            ).read()
        )

        self.sorted_all_tags = self.__get_all_sorted_tags__()
        self.current_tags = self.sorted_all_tags
        self.used_tags = []

        self.autocomplete = autocomplete_factory(content_files=content_files)

        if type(model_qa) == str:
            self.model_qa = build_model(model_qa)
        elif not model_qa:
            self.model_qa = None
        else:
            self.model_qa = model_qa

        if type(model_ner) == str:
            self.model_ner = build_model(model_ner)
        elif not model_ner:
            self.model_ner = None
        else:
            self.model_ner = model_ner

        self.find_titles = True
        self.find_average = True
        self.CONFIG = CONFIG

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

    def __check_STOP_WORDS_FINDER__(self, list_of_words: List[str]) -> bool:
        stopped_words = 0

        for word in list_of_words:
            if word in STOP_WORDS_FINDER:
                stopped_words += 1

        if stopped_words == len(list_of_words):
            return False

        return True

    def __get_formatted_word__(self, word: str) -> str:

        text_tag = self.__check_tag__(word)

        if self.morph_tags and text_tag in self.morph_tags.keys():
            text_tag = self.morph_tags[text_tag]

        return text_tag

    def __get_all_sorted_tags__(self):

        all_tags = json.loads(
            open(
                os.path.join(
                    self.CONFIG["save"]["path_save"], self.CONFIG["save"]["all_tags"]
                ),
                "r",
                encoding="utf8",
            ).read()
        )

        sorted_all_tags = sorted(
            [[tag, counter] for tag, counter in all_tags.items()],
            key=itemgetter(1),
            reverse=True,
        )

        return [
            EntityTag.pretty_tag(self.tags_index[tag])
            for tag, counter in sorted_all_tags
            if self.__check_STOP_WORDS_FINDER__(self.tags_index[tag]["tag"])
        ]

    def __get_all_input_words__(
        self, input_words: str, enumeration: bool = False
    ) -> List[str]:

        input_words = input_words.lower().split(" ")
        input_words = [
            self.__get_formatted_word__(word)
            for word in input_words
            if self.__check_text__(word)
        ]

        input_words_ = []

        if enumeration:

            for i in range(len(input_words)):
                k = len(input_words) - i

                for j in range(len(input_words) - k + 1):
                    words = " ".join(input_words[j : j + k])

                    if words != "":
                        input_words_.append(words)

        input_words_ = input_words

        return input_words_

    def __get_used_tags__(self):
        return [
            EntityTag.pretty_tag(self.tags_index[str(tag)]) for tag in self.used_tags
        ]

    def __get_sorted_tags__(self):
        return [tag for tag in self.sorted_all_tags if tag not in self.used_tags]

    def __get_rank_sorted_result__(self, dict_of_paths: dict) -> List[str]:

        max_rank = 0
        rank_sorted_res = []

        for path, val in dict_of_paths.items():
            if val["rank"] > max_rank:
                max_rank = val["rank"]

        if max_rank > 0:

            for i in range(max_rank):
                rank = max_rank - i
                rank_sorted_batch = []

                for path, val in dict_of_paths.items():
                    if val["rank"] == rank:
                        rank_sorted_batch.append([val["val"], path])

                rank_sorted_batch.sort(key=itemgetter(0), reverse=True)
                rank_sorted_res.extend(rank_sorted_batch)

        return rank_sorted_res

    def __autocomplete__(
        self,
        input_words: List[str],
        sorted_list_of_paths: List[list] = None,
        res_set: List[set] = None,
    ) -> Tuple[List[list], List[set]]:

        self.used_tags = []

        if not sorted_list_of_paths:
            sorted_list_of_paths = []

        if not res_set:
            res_set = []

        dict_of_paths = {}

        with Progress() as progress:
            task = progress.add_task("[green]Processing...", total=len(input_words))

            for input_word in input_words:
                rank = len(input_word.split(" "))
                results_autocomplete = self.autocomplete.search(
                    word=input_word, max_cost=3, size=200
                )

                # print(f'    | автоподстановка: {results_autocomplete} для слова: {input_word}')

                for results in results_autocomplete:
                    res = results[0]

                    if res in self.all_tags_tags.keys():
                        res_idxs = self.all_tags_tags[res]

                        for idx in res_idxs:
                            if idx not in self.used_tags:
                                self.used_tags.append(idx)

                            if str(idx) in self.all_tags_dict_with_paths.keys():
                                for counter, paths in self.all_tags_dict_with_paths[
                                    str(idx)
                                ].items():
                                    counter = float(counter)

                                    for path in paths:

                                        if path not in dict_of_paths.keys():
                                            dict_of_paths[path] = {
                                                "rank": rank,
                                                "val": counter,
                                            }
                                        else:
                                            if dict_of_paths[path]["rank"] < rank:
                                                dict_of_paths[path]["rank"] = rank
                                            dict_of_paths[path]["val"] += counter

                    else:
                        continue

                progress.update(task, advance=1)

        self.used_tags = self.__get_used_tags__()
        self.current_tags = self.__get_sorted_tags__()

        return dict_of_paths

    def __idx_to_titles__(self, res_end: Tuple[float, str]) -> Tuple[float, str]:

        if self.find_titles:
            res_end_ = []

            for score, idx in res_end:
                res_end_.append([score, self.pages[idx]["fields"]["title"]])

            res_end = res_end_

        return res_end

    def __average__(self, res_end: Tuple[float, str]) -> Tuple[float, str]:

        if self.find_average:

            dict_of_average = {}
            dict_of_count_idx = {}
            res_end_ = []

            for score, idx in res_end:

                if idx not in dict_of_average.keys():
                    dict_of_average[idx] = score
                else:
                    dict_of_average[idx] += score

                if idx not in dict_of_count_idx.keys():
                    dict_of_count_idx[idx] = 1
                else:
                    dict_of_count_idx[idx] += 1

            for key, val in dict_of_average.items():
                res_end_.append([val / dict_of_count_idx[key], key])

            res_end = sorted(res_end_, key=itemgetter(0), reverse=True)

        return res_end

    def __qa_pred__(
        self, res_end: List[Tuple[float, str]], input_words: List[str]
    ) -> List[str]:

        if self.model_qa:
            predictions = []

            for res in res_end:

                context = " ".join(self.pages[res[1]]["fields"].values())

                if (
                    context != []
                    and context != ""
                    and context != 0
                    and context != " "
                    and context != "  "
                    and context != "   "
                    and context != "    "
                ):

                    if self.__check_text__(context) != "":
                        predictions.append(
                            [res, self.model_qa([context], [" ".join(input_words)])]
                        )

            res_end = predictions

        return res_end

    def __ner_model_pred__(
        self, line: str
    ) -> Tuple[List[str], Union[List[str], List[float]]]:

        tokens = []
        predictions = []

        try:
            tokens, predictions = self.model([line])

            tokens = tokens[0]
            predictions = predictions[0]

        except Exception as e:  # (RuntimeError, IndexError)
            line_split = line.split(".")

            try:
                _tokens, _predictions = self.model(line_split)

                for token_batch, prediction_batch in zip(_tokens, _predictions):
                    tokens.extend(token_batch)
                    predictions.extend(prediction_batch)

            except Exception as e:
                line_split = line.split(" ")

                try:
                    _tokens, _predictions = self.model(line_split)

                    for token_batch, prediction_batch in zip(_tokens, _predictions):
                        tokens.extend(token_batch)
                        predictions.extend(prediction_batch)

                except Exception as e:
                    return tokens, predictions

        return (tokens, predictions)

    def find(self, input_text: str) -> List[str]:

        input_words = self.__get_all_input_words__(input_text)
        dict_of_paths = self.__autocomplete__(input_words)
        res_end = self.__get_rank_sorted_result__(dict_of_paths)
        res_end = self.__idx_to_titles__(res_end)
        res_end = self.__average__(res_end)
        res_end = self.__qa_pred__(res_end, input_words)

        return res_end


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-qa",
        action="store_true",
        help="ON or OFF QA mode. Def == OFF. If flag exist - ON, else - OFF.",
    )
    parser.add_argument(
        "-ner",
        action="store_true",
        help="ON or OFF QA mode. Def == OFF. If flag exist - ON, else - OFF.",
    )

    args = parser.parse_args()

    # if args.qa and args.ner: raise ArgumentError('Cant use in one time NER and QA model!')
    print(f" Use QA mode: {args.qa}")
    print(f" Use NER mode: {args.ner}")

    fast_finder = FastFinder(
        model_qa=os.path.join(CONFIG["path_configs"], CONFIG["path_to_models"]["qa"])
        if args.qa == True
        else None,
        model_ner=os.path.join(CONFIG["path_configs"], CONFIG["path_to_models"]["ner"])
        if args.ner == True
        else None,
    )

    while True:
        print(f"------------------------------------")
        if COUNT_OF_TAGS_FOR_DISPLAY < len(fast_finder.current_tags):
            print(
                f" --доступные теги: {fast_finder.current_tags[:COUNT_OF_TAGS_FOR_DISPLAY]}"
            )
        else:
            print(f" --доступные теги: {fast_finder.current_tags}")
        # if COUNT_OF_TAGS_FOR_DISPLAY < len(fast_finder.current_tags):
        #     print(f' --использованные теги: {fast_finder.used_tags[:COUNT_OF_TAGS_FOR_DISPLAY]}')
        # else: print(f' --использованные теги: {fast_finder.used_tags}')

        input_text = input(" --запрос:")
        if input_text == "q":
            break

        start_time = time.time()

        print(" --результат:")

        result = fast_finder.find(input_text)

        if COUNT_OF_RES_FOR_DISPLAY < len(result):
            result = result[:COUNT_OF_RES_FOR_DISPLAY]

        pprint(result, width=300, indent=4)
        print(" | время выполнения: ", (time.time() - start_time))
