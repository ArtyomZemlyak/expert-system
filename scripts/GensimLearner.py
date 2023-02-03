from operator import itemgetter
import os
from os import listdir
from os.path import isfile, join

import logging

from collections import Counter
from typing import List, Tuple, Union
from pprint import pprint
import string
import json
import time
import re
import copy
import random
import math

import numpy as np
import pandas as pd
from rich.progress import Progress

from deslib.des import KNORAU
from deslib.static.oracle import Oracle

from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
from imblearn.metrics import classification_report_imbalanced
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
from matplotlib import pyplot as plt

import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.models.fasttext import FastText
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.models.nmf import Nmf
from gensim.models import LogEntropyModel
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from gensim.test.utils import common_corpus, common_dictionary
from gensim import corpora


CONFIG = json.loads(open("scripts/config.json", "r").read())

BLACKLIST = set(CONFIG["save"].values())

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


class СallbackGensim(CallbackAny2Vec):
    """Callback to print loss after each epoch."""

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print("Loss after epoch {}: {}".format(self.epoch, loss_now))
        self.epoch += 1


class GensimLearner:
    def __init__(self, config: str = None) -> None:

        if config:
            self.CONFIG = config
        else:
            self.CONFIG = CONFIG

        if (
            "dont_know_class" in self.CONFIG["ml_learner"].keys()
            and self.CONFIG["ml_learner"]["dont_know_class"]
        ):

            self.titles_idx = {"dont_know": 0}
            self.titles_idx_swap = {0: "dont_know"}
        else:
            self.titles_idx = {}
            self.titles_idx_swap = {}

        self.titles_tags_exist = {}
        self.files_names = [
            f
            for f in listdir(self.CONFIG["save"]["path_save"])
            if isfile(join(self.CONFIG["save"]["path_save"], f)) and f not in BLACKLIST
        ]

    def __load_data__(
        self,
        num_classes: int = None,
        shuffle: bool = None,
        seed: int = 12,
        tags_max_norm: bool = None,
        threshold: float = None,
        rare_tag: bool = None,
    ):

        if not num_classes == None:
            if "num_classes" in self.CONFIG["ml_learner"].keys():
                num_classes = self.CONFIG["ml_learner"]["num_classes"]

        if shuffle == None:
            if "shuffle" in self.CONFIG["ml_learner"].keys():
                shuffle = self.CONFIG["ml_learner"]["shuffle"]
            else:
                shuffle = True

        if tags_max_norm == None:
            if "tags_max_norm" in self.CONFIG["ml_learner"].keys():
                tags_max_norm = self.CONFIG["ml_learner"]["tags_max_norm"]
            else:
                tags_max_norm = False

        if threshold == None:
            if "tag_threshold" in self.CONFIG["ml_learner"].keys():
                threshold = self.CONFIG["ml_learner"]["tag_threshold"]
            else:
                threshold = 0

        if rare_tag == None:
            if "rare_tag" in self.CONFIG["ml_learner"].keys():
                rare_tag = self.CONFIG["ml_learner"]["rare_tag"]
            else:
                rare_tag = False

        if (
            "dont_know_class" in self.CONFIG["ml_learner"].keys()
            and self.CONFIG["ml_learner"]["dont_know_class"]
        ):
            self.titles_counter = {"dont_know": 0}
            counter_of_titles = 1
        else:
            self.titles_counter = {}
            counter_of_titles = 0

        self.pages = {}
        classes_counter = 0

        with Progress() as progress:
            task = progress.add_task(
                "[green]Loading data ...", total=len(self.files_names)
            )

            for i, file_name in enumerate(self.files_names):

                name = re.sub("[\\.]{1}[A-Za-z\\d]+", "", file_name)

                json_file_data = json.loads(
                    open(
                        os.path.join(self.CONFIG["save"]["path_save"], file_name),
                        "r",
                        encoding="utf8",
                    ).read()
                )

                if json_file_data["fields"]["title"] not in self.titles_idx.keys():

                    self.titles_idx[
                        json_file_data["fields"]["title"]
                    ] = counter_of_titles
                    self.titles_idx_swap[counter_of_titles] = json_file_data["fields"][
                        "title"
                    ]

                    self.titles_tags_exist[json_file_data["fields"]["title"]] = Counter(
                        []
                    )
                    counter_of_titles += 1

                self.titles_tags_exist[json_file_data["fields"]["title"]] += Counter(
                    list(json_file_data["tags"]["values"].keys())
                )

                if json_file_data["fields"]["title"] not in self.titles_counter.keys():
                    if not num_classes or classes_counter < num_classes:
                        self.titles_counter[json_file_data["fields"]["title"]] = 1
                        self.pages[name] = json_file_data
                    classes_counter += 1
                else:
                    self.titles_counter[json_file_data["fields"]["title"]] += 1
                    self.pages[name] = json_file_data

                progress.update(task, advance=1)

        tags_index = json.loads(
            open(
                os.path.join(
                    self.CONFIG["save"]["path_save"],
                    self.CONFIG["save"]["all_tags_idx"],
                ),
                "r",
                encoding="utf8",
            ).read()
        )

        tags_with_paths = json.loads(
            open(
                os.path.join(
                    self.CONFIG["save"]["path_save"],
                    self.CONFIG["save"]["all_tags_with_paths"],
                ),
                "r",
                encoding="utf8",
            ).read()
        )

        print(f"All tags: {len(tags_index)}")

        tags_exist_percent = {}
        tags_max_val = {}
        self.tags_index = {}
        self.tags_sum_in_pages = {}

        if rare_tag:
            with Progress() as progress:
                task = progress.add_task(
                    "[green]Calc tags exist percent ...", total=len(tags_with_paths)
                )

                for tag in tags_with_paths.keys():
                    tags_exist_percent[tag] = {}

                    for title, counts in self.titles_tags_exist.items():
                        if tag in counts.keys():
                            tags_exist_percent[tag][title] = (
                                counts[tag] / self.titles_counter[title]
                            )

                    progress.update(task, advance=1)

        with Progress() as progress:
            task = progress.add_task(
                "[green]Calc sum tags in pages ...", total=len(self.pages)
            )

            for page, page_data in self.pages.items():
                self.tags_sum_in_pages[page] = np.sum(
                    list(page_data["tags"]["values"].values())
                )

                progress.update(task, advance=1)

        with Progress() as progress:
            task = progress.add_task(
                "[green]Calc tags max val ...", total=len(tags_with_paths)
            )

            for tag, paths_values in tags_with_paths.items():
                if tags_max_norm:
                    tag_relative_vals = []

                    for val, paths in paths_values.items():
                        for path in paths:
                            tag_relative_vals.append(
                                float(val) / self.tags_sum_in_pages[path]
                            )
                    paths_values = tag_relative_vals

                else:
                    paths_values = [float(val) for val in paths_values.keys()]

                tags_max_val[tag] = np.max(paths_values)

                progress.update(task, advance=1)

        with Progress() as progress:
            task = progress.add_task(
                "[green]Create tags index ...", total=len(tags_with_paths)
            )

            for tag in tags_index.keys():
                if tag in tags_max_val.keys():

                    if tags_max_val[tag] > threshold or (
                        rare_tag
                        and self.__check_tags_exist_percent__(tag, tags_exist_percent)
                    ):
                        self.tags_index[tag] = 0

                progress.update(task, advance=1)

        self.pages = [page for page in self.pages.items()]

        if shuffle:
            random.seed(seed)
            random.shuffle(self.pages)

    def __check_tags_exist_percent__(
        self,
        tag: str,
        tags_exist_percent: dict,
        threshold_up: float = None,
        threshold_bottom: float = None,
    ) -> bool:

        if threshold_up == None:
            if "threshold_up" in self.CONFIG["ml_learner"].keys():
                threshold_up = self.CONFIG["ml_learner"]["threshold_up"]
            else:
                threshold_up = 0.9

        if threshold_bottom == None:
            if "threshold_bottom" in self.CONFIG["ml_learner"].keys():
                threshold_bottom = self.CONFIG["ml_learner"]["threshold_bottom"]
            else:
                threshold_bottom = 0.1

        list_tep = list(tags_exist_percent[tag].values())
        list_tep_k = list(tags_exist_percent[tag].keys())

        if len(list_tep) == 1:
            if tags_exist_percent[tag][list_tep_k[0]] < threshold_up:
                return False
            else:
                return True

        title = list_tep_k[np.argmax(list_tep)]
        if tags_exist_percent[tag][title] < threshold_up:
            return False

        list_tep = [
            val for title_, val in tags_exist_percent[tag].items() if title_ != title
        ]
        list_tep_k = [
            title_ for title_ in tags_exist_percent[tag].keys() if title_ != title
        ]
        title = list_tep_k[np.argmax(list_tep)]
        if tags_exist_percent[tag][title] > threshold_bottom:
            return False

        return True

    def __get_sample__(self, idx: int, title: str, page_data: dict):

        page_tokens = []

        for tag in page_data["tags"]["entries"].values():
            if tag["priority"] != 0 and tag["idx"] in self.tags_index.keys():
                page_tokens.append(tag["text"])

        x_sample = (
            page_tokens  # gensim.models.doc2vec.TaggedDocument(page_tokens, [idx])
        )
        y_title = self.titles_idx[title]

        return x_sample, y_title

    def __get_data__(
        self,
        train: float = None,
        valid: float = None,
        test: float = None,
        balanced: bool = None,
        min_counter: int = None,
        dont_know_class: bool = None,
    ):

        if train == None:
            if (
                "data" in self.CONFIG["ml_learner"].keys()
                and "train" in self.CONFIG["ml_learner"]["data"].keys()
            ):
                train = self.CONFIG["ml_learner"]["data"]["train"]
            else:
                train = 0.8

        if valid == None:
            if (
                "data" in self.CONFIG["ml_learner"].keys()
                and "val" in self.CONFIG["ml_learner"]["data"].keys()
            ):
                valid = self.CONFIG["ml_learner"]["data"]["val"]
            else:
                valid = 0.1

        if test == None:
            if (
                "data" in self.CONFIG["ml_learner"].keys()
                and "test" in self.CONFIG["ml_learner"]["data"].keys()
            ):
                test = self.CONFIG["ml_learner"]["data"]["test"]
            else:
                test = 0.1

        if balanced == None:
            if (
                "data" in self.CONFIG["ml_learner"].keys()
                and "balanced" in self.CONFIG["ml_learner"]["data"].keys()
            ):
                balanced = self.CONFIG["ml_learner"]["data"]["balanced"]
            else:
                balanced = False

        if min_counter == None:
            if "min_counter" in self.CONFIG["ml_learner"].keys():
                min_counter = self.CONFIG["ml_learner"]["min_counter"]
            else:
                min_counter = 0

        if dont_know_class == None:
            if "dont_know_class" in self.CONFIG["ml_learner"].keys():
                dont_know_class = self.CONFIG["ml_learner"]["dont_know_class"]
            else:
                dont_know_class = False

        train_idx = 0
        val_idx = 0
        test_idx = 0
        x_train = []
        y_train = []
        x_val = []
        y_val = []
        x_test = []
        y_test = []
        titles_counter = {}
        titles_counter_temp = {}

        if dont_know_class:
            with Progress() as progress:
                task = progress.add_task(
                    "[green]Checking dont know class ...", total=len(self.pages)
                )

                self.titles_counter["dont_know"] = 0

                for i, (page_name, page_data) in enumerate(self.pages):
                    if self.titles_counter[page_data["fields"]["title"]] < min_counter:
                        self.titles_counter["dont_know"] += 1
                        self.pages[i][1]["fields"]["title"] = "dont_know"

                    progress.update(task, advance=1)

        if balanced:
            min_count_of_classes = np.min(list(self.titles_counter.values()))
            for key in self.titles_counter.keys():
                titles_counter[key] = min_count_of_classes
        else:
            titles_counter = self.titles_counter

        with Progress() as progress:
            task = progress.add_task(
                "[green]Creating train data ...", total=len(self.pages)
            )

            for page_name, page_data in self.pages:

                if (
                    titles_counter[page_data["fields"]["title"]] >= min_counter
                    or dont_know_class
                ):

                    title = page_data["fields"]["title"]

                    if title not in titles_counter_temp.keys():
                        titles_counter_temp[title] = 1
                    else:
                        titles_counter_temp[title] += 1

                    if titles_counter_temp[title] < titles_counter[title] * train:
                        x_sample, y_title = self.__get_sample__(
                            train_idx, title, page_data
                        )
                        x_train.append(x_sample)
                        y_train.append(y_title)
                        train_idx += 1

                    elif titles_counter_temp[title] < titles_counter[title] * (
                        train + valid
                    ):
                        x_sample, y_title = self.__get_sample__(
                            val_idx, title, page_data
                        )
                        x_val.append(x_sample)
                        y_val.append(y_title)
                        val_idx += 1

                    elif titles_counter_temp[title] < titles_counter[title] * (
                        train + valid + test
                    ):
                        x_sample, y_title = self.__get_sample__(
                            test_idx, title, page_data
                        )
                        x_test.append(x_sample)
                        y_test.append(y_title)
                        test_idx += 1

                progress.update(task, advance=1)

        if min_counter != 0:
            print(f"Classes with count >= {min_counter}: {len(titles_counter_temp)}")
            self.titles_counter = titles_counter_temp

        return x_train, y_train, x_val, y_val, x_test, y_test

    def __get_gen_pages_tokens__(self):

        with Progress() as progress:
            task = progress.add_task("[green]Loading tokens ...", total=len(self.pages))

            for i, (page_name, page_data) in enumerate(self.pages.items()):
                page_tokens = []

                for tag in page_data["tags"]["entries"].values():
                    page_tokens.append(tag["text"])

                progress.update(task, advance=1)

                yield gensim.models.doc2vec.TaggedDocument(page_tokens, [i])

    def __dv_model_create__(self, x_train):
        self.dv_model = Doc2Vec(
            vector_size=700, min_count=0, dm=0, epochs=30, workers=4, seed=42
        )  # dbow_words=1
        self.dv_model.build_vocab(corpus_iterable=x_train)
        self.dv_model.train(
            corpus_iterable=x_train,
            total_examples=self.dv_model.corpus_count,
            epochs=self.dv_model.epochs,
        )

        print(f"Documents in Doc2Vec Gensim model: {len(self.dv_model.dv)}")

    def __dv_model_get_data__(self, data):

        data_vec = []

        with Progress() as progress:
            task = progress.add_task("[green]Convert data to Vec ...", total=len(data))

            for sample in data:
                tokens = sample.words
                vector = self.dv_model.infer_vector(tokens)

                data_vec.append(vector)

                progress.update(task, advance=1)

        return np.float32(data_vec)

    def __lda_model_create__(self, x_train):

        self.dct = corpora.Dictionary(x_train)
        self.corpus = [self.dct.doc2bow(line) for line in x_train]

        self.lda_model = LdaMulticore(
            corpus=self.corpus,
            id2word=self.dct,
            random_state=100,
            num_topics=4,
            passes=30,
            iterations=200,
        )

        # print('\n', self.lda_model.print_topics(-1))
        # print(self.lda_model[corpus_test[5]])
        # for c in lda_model[corpus_test[5:8]]:
        #     print("Document Topics      : ", c[0])      # [(Topics, Perc Contrib)]
        #     print("Word id, Topics      : ", c[1][:3])  # [(Word id, [Topics])]
        #     print("Phi Values (word id) : ", c[2][:2])  # [(Word id, [(Topic, Phi Value)])]
        #     print("Word, Topics         : ", [(dct[wd], topic) for wd, topic in c[1][:2]])   # [(Word, [Topics])]
        #     print("Phi Values (word)    : ", [(dct[wd], topic) for wd, topic in c[2][:2]])  # [(Word, [(Topic, Phi Value)])]
        #     print("------------------------------------------------------\n")

    def __lda_model_get_data__(self, data):

        data_vec = []
        corpus_test = [self.dct.doc2bow(line) for line in data]

        with Progress() as progress:
            task = progress.add_task("[green]Convert data to Vec ...", total=len(data))

            for c in self.lda_model[corpus_test]:
                sorted_c = sorted(c, key=itemgetter(1), reverse=True)

                vector = np.zeros(len(self.titles_counter))
                vector[sorted_c[0][0]] = sorted_c[0][1]

                data_vec.append(vector)

                progress.update(task, advance=1)

        return np.float32(data_vec)

    def __lsi_model_create__(self, x_train):

        self.dct = corpora.Dictionary(x_train)
        self.corpus = [self.dct.doc2bow(line) for line in x_train]

        self.lsi_model = LsiModel(
            corpus=self.corpus,
            id2word=self.dct,
            num_topics=4,
            power_iters=100,
            extra_samples=1000,
        )

    def __lsi_model_get_data__(self, data):

        corpus = [self.dct.doc2bow(line) for line in data]
        data_vec = []

        with Progress() as progress:
            task = progress.add_task("[green]Convert data to Vec ...", total=len(data))

            for c in self.lsi_model[corpus]:
                sorted_c = sorted(c, key=itemgetter(1), reverse=True)
                vector = np.zeros(len(self.titles_counter))

                if c != []:
                    vector[sorted_c[0][0]] = sorted_c[0][1]
                data_vec.append(vector)

                progress.update(task, advance=1)

        return np.float32(data_vec)

    def __nmf_model_create__(self, x_train):

        self.dct = corpora.Dictionary(x_train)
        self.corpus = [self.dct.doc2bow(line) for line in x_train]

        self.nmf_model = Nmf(
            corpus=self.corpus, id2word=self.dct, num_topics=4, passes=10000
        )

    def __nmf_model_get_data__(self, data):

        corpus = [self.dct.doc2bow(line) for line in data]
        data_vec = []

        with Progress() as progress:
            task = progress.add_task("[green]Convert data to Vec ...", total=len(data))

            for c in self.nmf_model[corpus]:
                sorted_c = sorted(c, key=itemgetter(1), reverse=True)
                vector = np.zeros(len(self.titles_counter))

                if c != []:
                    vector[sorted_c[0][0]] = sorted_c[0][1]
                data_vec.append(vector)

                progress.update(task, advance=1)

        return np.float32(data_vec)

    def __tf_idf_model_create__(self, x_train):

        self.dct = corpora.Dictionary(x_train)
        self.corpus = [self.dct.doc2bow(line) for line in x_train]

        self.tf_idf = TfidfModel(corpus=self.corpus, id2word=self.dct, smartirs="nfu")

    def __tf_idf_model_get_data__(self, data):

        corpus = [self.dct.doc2bow(line) for line in data]
        data_vec = []

        with Progress() as progress:
            task = progress.add_task("[green]Convert data to Vec ...", total=len(data))

            for c in self.tf_idf[corpus]:
                vector = np.zeros(len(self.dct))

                for idx, val in c:
                    vector[idx] = val

                data_vec.append(vector)

                progress.update(task, advance=1)

        return np.float32(data_vec)

    def __lem_model_create__(self, x_train):

        self.dct = corpora.Dictionary(x_train)
        self.corpus = [self.dct.doc2bow(line) for line in x_train]

        self.lem_model = LogEntropyModel(corpus=self.corpus)

    def __lem_model_get_data__(self, data):

        data_vec = []
        corpus_test = [self.dct.doc2bow(line) for line in data]

        with Progress() as progress:
            task = progress.add_task("[green]Convert data to Vec ...", total=len(data))

            for c in self.lem_model[corpus_test]:
                vector = np.zeros(len(self.dct))

                for idx, val in c:
                    vector[idx] = val

                data_vec.append(vector)

                progress.update(task, advance=1)

        return np.float32(data_vec)

    def __phrase_model_convert_data__(self, data):

        phrase_model = Phrases(
            data,
            min_count=1,
            threshold=1,
            max_vocab_size=10000,
            connector_words=ENGLISH_CONNECTOR_WORDS,
        )

        return [phrase_model[new_sentence] for new_sentence in data]

    def learn(self):
        self.__load_data__()
        x_train, y_train, x_val, y_val, x_test, y_test = self.__get_data__()

        print(f"X Train: {len(x_train)}")
        print(f"Y Train: {len(y_train)}")
        print(f"X Val:   {len(x_val)}")
        print(f"Y Val:   {len(y_val)}")
        print(f"X Test:  {len(x_test)}")
        print(f"Y Test:  {len(y_test)}")

        pool_classifiers = [
            RandomForestClassifier(
                n_estimators=100,
                n_jobs=-1,
                max_features="sqrt",
                random_state=42,
                bootstrap=False,
                min_samples_split=10,
            ),
            OneVsRestClassifier(RandomForestClassifier(n_estimators=40), n_jobs=-1),
            OneVsOneClassifier(RandomForestClassifier(n_estimators=40), n_jobs=-1),
        ]

        self.__tf_idf_model_create__(x_train)

        x_train_vec = self.__tf_idf_model_get_data__(x_train)
        print(f"X Train Vec: {x_train_vec.shape}")

        # input_shape = x_train_vec.shape[1]

        # dlnn = keras.Sequential(
        #     [
        #         keras.Input(shape=input_shape),
        #         keras.layers.Dropout(0.6),
        #         keras.layers.Dense(6000, activation="relu"),
        #         keras.layers.Dropout(0.8),
        #         keras.layers.Dense(5000, activation="relu"),
        #         keras.layers.Dropout(0.8),
        #         keras.layers.Dense(3000, activation="relu"),
        #         keras.layers.Dropout(0.8),
        #         keras.layers.Dense(2000, activation="relu"),
        #         keras.layers.Dropout(0.7),
        #         keras.layers.Dense(500, activation="relu"),
        #         keras.layers.Dropout(0.1),
        #         keras.layers.Dense(len(self.titles_counter), activation="softmax"),
        #     ]
        # )

        # print(dlnn.summary())

        # y_train_ = keras.utils.to_categorical(y_train)
        # dlnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        # history = dlnn.fit(x_train_vec, y_train_, epochs=100, batch_size=512, validation_split=0.2)

        # x_test_vec = self.__tf_idf_model_get_data__(x_test)
        # print(f'X Test Vec: {x_test_vec.shape}')

        # y_test_ = keras.utils.to_categorical(y_test)
        # score_dlnn = dlnn.evaluate(x_test_vec, y_test_, verbose=0)
        # print("dlnn Test loss:", score_dlnn[0])
        # print("dlnn Test accuracy:", score_dlnn[1])

        # plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'val'], loc='upper left')
        # plt.show()

        print("Training pool classifiers models ...")
        start_time = time.time()

        for model in pool_classifiers:
            model.fit(x_train_vec, y_train)

        print(f"Time training: {time.time() - start_time}")

        x_train_nb, y_train_nb, x_val_, y_val_, x_test, y_test = self.__get_data__(
            balanced=True
        )

        pool_classifiers_non_balanced = [
            RandomForestClassifier(
                n_estimators=100,
                n_jobs=-1,
                max_features="sqrt",
                random_state=42,
                bootstrap=False,
                min_samples_split=10,
            ),
            OneVsRestClassifier(RandomForestClassifier(n_estimators=40), n_jobs=-1),
            OneVsOneClassifier(RandomForestClassifier(n_estimators=40), n_jobs=-1),
        ]

        x_train_nb_vec = self.__tf_idf_model_get_data__(x_train_nb)
        print(f"X Train nb Vec: {x_train_nb_vec.shape}")

        print("Training pool classifiers models on non-balanced data...")
        start_time = time.time()

        for model in pool_classifiers_non_balanced:
            model.fit(x_train_nb_vec, y_train_nb)

        print(f"Time training: {time.time() - start_time}")

        pool_classifiers.extend(pool_classifiers_non_balanced)

        knorau = KNORAU(pool_classifiers, k=2)  # k=2  # with_IH # DFP # Oracle

        print("Training KNORAU ...")
        start_time = time.time()

        knorau.fit(x_train_vec, y_train)

        print(f"Time training: {time.time() - start_time}")

        x_test_vec = self.__tf_idf_model_get_data__(x_test)
        print(f"X Test Vec: {x_test_vec.shape}")

        print("Testing model ...")
        start_time = time.time()
        # y_pred = model.predict(x_test_vec)

        y_pred = knorau.predict(x_test_vec)

        print(f"Time testing: {time.time() - start_time}")

        report = classification_report_imbalanced(
            y_test,
            y_pred,
            target_names=list(self.titles_counter.keys()),
            output_dict=True,
        )

        report_with_class_names = {}
        last_line = {"name": "avg"}

        for key, val in report.items():
            if key in self.titles_idx_swap.keys():
                name = self.titles_idx_swap[key]
                val["name"] = name
                report_with_class_names[key] = val

            else:
                if key == "avg_pre":
                    name = "pre"
                elif key == "avg_rec":
                    name = "rec"
                elif key == "avg_spe":
                    name = "spe"
                elif key == "avg_f1":
                    name = "f1"
                elif key == "avg_geo":
                    name = "geo"
                elif key == "avg_iba":
                    name = "iba"
                elif key == "total_support":
                    name = "sup"
                last_line[name] = val

        report_with_class_names[len(self.titles_idx_swap) + 1] = last_line

        pd.set_option("display.precision", 3)
        df_report = pd.DataFrame.from_dict(report_with_class_names).T
        print(df_report)

        for col in ["pre", "rec", "spe", "f1", "geo", "iba"]:
            df_report[col] = (
                df_report[col].apply(pd.to_numeric, downcast="float").fillna(0)
            )

        df_report.to_csv("out.csv", index=False, float_format="%.3f")

        # print(f'Test Document: {tokens}')
        # print(f'SIMILAR/DISSIMILAR DOCS PER MODEL {model}')
        # print(sims)
        # for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        #     print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(x_train[sims[index][0]].words)))


if __name__ == "__main__":
    mll = GensimLearner()
    mll.learn()


# balanced
# # One vs One
#      pre    rec    spe     f1    geo    iba   sup                 name
# 0  0.602  0.602  0.867  0.602  0.723  0.509   337      SERVICE_REQUEST
# 1  0.626  0.754   0.85  0.684    0.8  0.634   337                  SLA
# 2  0.506  0.481  0.844  0.493  0.637  0.391   337         CONSULTATION
# 3  0.782  0.662  0.939  0.717  0.788  0.604   337  IMPROVEMENT_REQUEST
# 5  0.629  0.625  0.875  0.624  0.737  0.534  1348                  avg

# # KNORAU
#      pre    rec    spe     f1    geo    iba   sup                 name
# 0  0.561  0.659  0.828  0.606  0.738  0.536   337      SERVICE_REQUEST
# 1  0.621  0.786   0.84  0.694  0.813  0.657   337                  SLA
# 2  0.522  0.448  0.864  0.482  0.622  0.371   337         CONSULTATION
# 3  0.839  0.588  0.962  0.691  0.752  0.544   337  IMPROVEMENT_REQUEST
# 5  0.636   0.62  0.873  0.618  0.731  0.527  1348                  avg
