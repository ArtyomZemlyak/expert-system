import json
from pprint import pprint
import re
import string
import time
import os
from os import listdir, times
from os.path import isfile, join, exists
from typing import Dict, List, Union, Counter as CounterT
from operator import itemgetter
from collections import Counter
import pathlib

from mosestokenizer import MosesTokenizer
from pymystem3 import Mystem
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
from fast_autocomplete import autocomplete_factory


##############################################
#### CONFIG ##################################
##############################################
path_cfg = join(pathlib.Path(__file__).parent.parent.resolve(), "config.json")
CONFIG = json.loads(open(path_cfg, "r").read())
##############################################
##############################################

VALID_CHARS = CONFIG["finder"]["valid_chars"]
VALID_CHARS += string.ascii_lowercase
VALID_CHARS += string.ascii_uppercase

STOP_WORDS_FINDER = set(CONFIG["finder"]["stop_words"])
STOP_WORDS_FINDER.update("йцукенгшщзхъфывапролджэячсмитьбюё")


# from nltk.tokenize import sent_tokenize, word_tokenize
# nltk.download('punkt')
# class MyTokenizer:
#     def __init__(self) -> None:
#         pass
#     def tokenize(self, text: str, lang='russian'):
#         return word_tokenize(text, language=lang)


class ESTargetFinder:
    def __init__(
        self,
        config: str = None,
        tags: dict = None,
        targets: dict = None,
        word_to_target_idxs: Dict[str, CounterT] = None,
        _profile: str = None,
    ) -> None:  # ? NOne or not None???
        if config:
            self._CONFIG = config
        else:
            self._CONFIG = CONFIG

        self._profile = _profile

        self.nlp = MosesTokenizer("ru")
        self.mystem = Mystem()
        self.__stopwords = {
            *stopwords.words("russian"),
            *stopwords.words("english"),
            *self._CONFIG["query"]["stop_words"],
        }

        self._tags = tags if tags else {}
        self._targets = targets if targets else {}
        self._word_to_target_idxs = word_to_target_idxs if word_to_target_idxs else {}

        self._fmt = {}
        self._init_fmt()

    def _init_fmt(self) -> None:
        if self._tags:
            self._save_json(
                join("common", self._CONFIG["save"]["common"]["valid_chars"]),
                VALID_CHARS,
            )

            all_tags_formatted = {
                tag_text.lower(): [{"text": tag_text.lower()}, tag_text, tag_value]
                for tag_text, tag_value in self._tags.items()
            }

            self._save_json(
                join("common", self._CONFIG["save"]["common"]["all_tags_formatted"]),
                all_tags_formatted,
            )  # ? conflicts with over instances?

            content_files = {
                "words": {
                    "filepath": join(
                        self._CONFIG["save"]["path_save"],
                        "common",
                        self._CONFIG["save"]["common"]["all_tags_formatted"],
                    ),
                    "compress": True,  # means compress the graph data in memory
                },
                "valid_chars_for_string": {
                    "filepath": join(
                        self._CONFIG["save"]["path_save"],
                        "common",
                        self._CONFIG["save"]["common"]["valid_chars"],
                    ),
                    "compress": False,  # need
                },
            }

            self._fmt = autocomplete_factory(content_files=content_files)

    def _open_json(
        self, file_name: str, path_save: str = None, create_if_not_exist: bool = False
    ) -> Union[dict, None]:
        if not path_save:
            path_save = self._CONFIG["save"]["path_save"]
        if exists(join(path_save, file_name)):
            return json.loads(
                open(join(path_save, file_name), "r", encoding="utf8").read()
            )
        elif create_if_not_exist:
            open(join(path_save, file_name), "w").write(json.dumps({}))
        else:
            return None

    def _save_json(
        self, file_name: str, file_data: dict, path_save: str = None
    ) -> None:
        if not path_save:
            path_save = self._CONFIG["save"]["path_save"]
        with open(join(path_save, file_name), "w", encoding="utf8") as json_file:
            json_file.write(
                json.dumps(file_data, indent=4, sort_keys=True, ensure_ascii=False)
            )
            json_file.close()

    def _check_text(self, text: str, strict: bool = False) -> bool:
        compared = "" if not strict else text
        return (
            re.sub(f"[^{self._CONFIG['analyzer']['word_symbols']}]", "", text)
            != compared
        )

    def _clear_text(self, tag: str) -> str:
        return re.sub(
            f"[^{self._CONFIG['analyzer']['word_symbols'] + self._CONFIG['analyzer']['filter_symbols']}]",
            "",
            tag,
        )

    @staticmethod
    def _equal(one_set: set, two_set: set) -> int:
        idx_equal = 0

        for item in one_set:
            if item in two_set:
                idx_equal += 1

        return idx_equal

    def lemmatize(self, text: str) -> Union[str, None]:
        if text not in self.__stopwords:
            return (
                self.mystem.lemmatize(text)[0]
                if text not in self._CONFIG["query"]["white_words"]
                else text
            )
        return None

    def tokenize(
        self, text: str, lemmatize: bool = True, clear_symbols: bool = True
    ) -> List[str]:
        tokens = self.nlp.tokenize(text)

        if self.mystem and self.__stopwords:
            tokens = [token.lower() for token in tokens]

            if clear_symbols:
                tokens = [
                    self._clear_text(token)
                    for token in tokens
                    if self._check_text(token)
                ]
            if lemmatize:
                tokens = [
                    self.mystem.lemmatize(token)[0]
                    if token not in self._CONFIG["query"]["white_words"]
                    else token
                    for token in tokens
                    if token not in self.__stopwords
                ]

        return tokens

    def count_tokens(
        self, text: str, lemmatize: bool = True, clear_symbols: bool = True
    ) -> CounterT:
        return Counter(
            self.tokenize(text, lemmatize=lemmatize, clear_symbols=clear_symbols)
        )

    def count_tokens_to_target(
        self,
        token: str,
        target_idx: str,
        coeff: Union[int, float],
        tags: dict = None,
        word_to_target_idxs: dict = None,
    ) -> None:
        tags = self._tags if tags == None else tags
        word_to_target_idxs = (
            self._word_to_target_idxs
            if word_to_target_idxs == None
            else word_to_target_idxs
        )

        if token not in tags:
            tags[token] = coeff
            word_to_target_idxs[token] = Counter({target_idx: coeff})
        else:
            tags[token] += coeff
            word_to_target_idxs[token].update({target_idx: coeff})

    def autocomplete(
        self, input_words: List[str], type_output: str = "short"
    ) -> Union[dict, set]:
        """
        type_output = "short" | "full" | "only_autocomplete"
        """
        input_word_to_res = {} if type_output != "only_autocomplete" else set([])

        for input_word in input_words:
            if type_output != "only_autocomplete":
                input_word_to_res[input_word] = (
                    {"autocomplete": [], "results": []} if type_output == "full" else []
                )

            results_autocomplete = self._fmt.search(
                word=input_word, max_cost=3, size=200
            )
            # print(results_autocomplete)
            # pprint(self._fmt.words)
            # print(self._fmt.words[results_autocomplete[0][0]])

            for results in results_autocomplete:
                if type_output != "only_autocomplete":
                    for res in results:
                        res = self._fmt.words[res].display

                        if type_output == "full":
                            input_word_to_res[input_word]["autocomplete"].append(res)

                            if self._word_to_target_idxs:
                                input_word_to_res[input_word]["results"].extend(
                                    [(res, self._word_to_target_idxs[res])]
                                )

                        else:
                            input_word_to_res[input_word].append(res)

                else:
                    input_word_to_res.update(
                        [self._fmt.words[res].display for res in results]
                    )  # ? need counter of same autocomplete tokens???

        return input_word_to_res

    def find_targets(
        self,
        res_words: List[str],
        targets: dict = None,
        word_to_target_idxs: Dict[str, CounterT] = None,
        sort_output: bool = False,
    ) -> Union[dict, list]:
        finded_targets = {}

        if targets == None:
            targets = self._targets
        if word_to_target_idxs == None:
            word_to_target_idxs = self._word_to_target_idxs

        if word_to_target_idxs:
            for word in res_words:
                for target, coeff in word_to_target_idxs[word].items():
                    if targets:
                        coeff = coeff * targets[target]

                    if target not in finded_targets:
                        finded_targets[target] = coeff
                    else:
                        finded_targets[target] += coeff

        else:
            raise ValueError(
                "Property word_to_target_idxs needs specified in fuinction or __init__ of ESTargetFinder!"
            )

        if sort_output:
            finded_targets = sorted(
                [(idx, coeff) for idx, coeff in finded_targets.items()],
                key=itemgetter(1),
                reverse=True,
            )
        return finded_targets


"""
Time MosesTokenizer tokenize:  0.00048232078552246094
Time NLTK tokenize:  0.009296417236328125
Time Rust tantivy tokenize (eng) : 0.0000005 - 0.000458
"""
