import os
from os import listdir
from os.path import isfile, join

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

# from sklearnex import patch_sklearn
# patch_sklearn()

from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
from imblearn.metrics import classification_report_imbalanced
from sklearn.preprocessing import StandardScaler

from tensorflow import keras

import tensorflow as tf
import autokeras as ak

# import autosklearn.classification
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
import mlflow


CONFIG = json.loads(open("scripts/config_ml.json", "r").read())

BLACKLIST = set(CONFIG["save"]["common"].values())


class MLLearner:
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
        lower_border: int = None,
        upper_border: int = None,
        num_classes: int = None,
        shuffle: bool = None,
        seed: int = 12,
        tags_max_norm: bool = None,
        threshold: float = None,
        rare_tag: bool = None,
        pos_tags: bool = None,
    ):

        if lower_border == None:
            if "lower_border" in self.CONFIG["ml_learner"].keys():
                lower_border = self.CONFIG["ml_learner"]["lower_border"]
            else:
                lower_border = 0

        if upper_border == None:
            if "upper_border" in self.CONFIG["ml_learner"].keys():
                upper_border = self.CONFIG["ml_learner"]["upper_border"]
            else:
                upper_border = len(self.files_names)

        if num_classes == None:
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

        if pos_tags == None:
            if "pos_tags" in self.CONFIG["ml_learner"].keys():
                pos_tags = self.CONFIG["ml_learner"]["pos_tags"]
            else:
                pos_tags = False

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
                "[green]Loading data ...", total=(upper_border - lower_border)
            )

            for i, file_name in enumerate(self.files_names):

                if lower_border <= i <= upper_border:
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
                        self.titles_idx_swap[counter_of_titles] = json_file_data[
                            "fields"
                        ]["title"]

                        self.titles_tags_exist[
                            json_file_data["fields"]["title"]
                        ] = Counter([])
                        counter_of_titles += 1

                    self.titles_tags_exist[
                        json_file_data["fields"]["title"]
                    ] += Counter(list(json_file_data["tags"]["values"].keys()))

                    if (
                        json_file_data["fields"]["title"]
                        not in self.titles_counter.keys()
                    ):
                        if not num_classes or classes_counter < num_classes:
                            self.titles_counter[json_file_data["fields"]["title"]] = 1
                            self.pages[name] = json_file_data
                        classes_counter += 1
                    else:
                        self.titles_counter[json_file_data["fields"]["title"]] += 1
                        self.pages[name] = json_file_data

                elif upper_border < i:
                    break

                progress.update(task, advance=1)

        self.pages = [page for page in self.pages.items()]
        if shuffle:
            random.seed(seed)
            random.shuffle(self.pages)

        tags_index = json.loads(
            open(
                os.path.join(
                    self.CONFIG["save"]["path_save"],
                    self.CONFIG["save"]["common"]["all_tags_idx"],
                ),
                "r",
                encoding="utf8",
            ).read()
        )

        tags_with_paths = json.loads(
            open(
                os.path.join(
                    self.CONFIG["save"]["path_save"],
                    self.CONFIG["save"]["common"]["all_tags_with_paths"],
                ),
                "r",
                encoding="utf8",
            ).read()
        )

        print(f"All tags: {len(tags_index)}")

        tags_exist_percent = {}
        self.tags_mean_val = {}
        self.tags_max_val = {}
        self.tags_index = {}
        self.tags_to_int = {}
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

            for page, page_data in self.pages:
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

                self.tags_max_val[tag] = np.max(paths_values)
                self.tags_mean_val[tag] = np.mean(paths_values)

                progress.update(task, advance=1)

        self.tags_all_mean = np.mean(list(self.tags_mean_val.values()))

        if pos_tags:
            counter_of_tags = 1

        with Progress() as progress:
            task = progress.add_task(
                "[green]Create tags index ...", total=len(tags_with_paths)
            )

            for tag in tags_index.keys():
                if tag in self.tags_max_val.keys():

                    if self.tags_max_val[tag] > threshold or (
                        rare_tag
                        and self.__check_tags_exist_percent__(tag, tags_exist_percent)
                    ):
                        self.tags_index[tag] = 0  # self.tags_mean_val[tag]

                        if pos_tags:
                            self.tags_to_int[tag] = counter_of_tags
                            counter_of_tags += 1

                progress.update(task, advance=1)

        if pos_tags:
            self.__check_tags_entries__()

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

    def __check_tags_entries__(self):

        max_entry_shape = np.zeros(3)

        with Progress() as progress:
            task = progress.add_task(
                "[green]Checking tags entries ...", total=len(self.pages)
            )

            for i, (page_name, page_data) in enumerate(self.pages):
                for entry_pos in page_data["tags"]["entries"].keys():

                    cur_entry_shape = (
                        int(
                            re.sub("[\\:]{1}[\\d]{1,}[\\:]{1}[\\d]{1,}", "", entry_pos)
                        ),
                        int(
                            re.sub(
                                "((^)[\\d]{1,}[\\:]{1})|([\\:]{1}[\\d]{1,}($))",
                                "",
                                entry_pos,
                            )
                        ),
                        int(
                            re.sub("[\\d]{1,}[\\:]{1}[\\d]{1,}[\\:]{1}", "", entry_pos)
                        ),
                    )

                    self.pages[i][1]["tags"]["entries"][entry_pos][
                        "entry_shape"
                    ] = cur_entry_shape

                    max_entry_shape = np.maximum(max_entry_shape, cur_entry_shape)

                progress.update(task, advance=1)

        self.max_entry_shape = np.uint64(max_entry_shape) + 1
        print(f"Founded max entry shape: {self.max_entry_shape}")

    def __get_num_of_classes__(self):
        titles_idx = {}
        counter_of_titles = 0

        with Progress() as progress:
            task = progress.add_task(
                "[green]Checking num of classes ...", total=len(self.pages)
            )

            for file_name in self.files_names:
                name = re.sub("[\\.]{1}[A-Za-z\\d]+", "", file_name)
                json_file_data = json.loads(
                    open(
                        os.path.join(self.CONFIG["save"]["path_save"], file_name),
                        "r",
                        encoding="utf8",
                    ).read()
                )

                if json_file_data["fields"]["title"] not in titles_idx.keys():
                    counter_of_titles += 1

                progress.update(task, advance=1)

        return counter_of_titles

    def __get_sample__(self, title: str, page_data: dict, pos_tags: bool = False):

        x_sample = []

        if pos_tags:
            x_sample_entry = np.zeros((*self.max_entry_shape, len(self.tags_index)))

            for tag in page_data["tags"]["entries"].values():

                cur_entry_shape = tag["entry_shape"]

                if (
                    tag["idx"] in self.tags_to_int.keys()
                    and tag["idx"] in page_data["tags"]["values"].keys()
                ):
                    x_sample_entry[cur_entry_shape[0]][cur_entry_shape[1]][
                        cur_entry_shape[2]
                    ] = [
                        self.tags_to_int[tag["idx"]],
                        page_data["tags"]["values"][tag["idx"]],
                    ]

            x_sample = x_sample_entry

        else:
            x_sample = {**self.tags_index}
            for tag_idx, val in page_data["tags"]["values"].items():
                if tag_idx in x_sample.keys():
                    x_sample[
                        tag_idx
                    ] = val  # / self.tags_all_mean + 7 - self.tags_mean_val[tag_idx] + self.tags_max_val[tag_idx]
            x_sample = [val for val in x_sample.values()]

        y_title = self.titles_idx[title]

        return x_sample, y_title

    def __get_data__(
        self,
        train: float = None,
        valid: float = None,
        test: float = None,
        balanced: bool = None,
        pos_tags: bool = None,
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

        if pos_tags == None:
            if "pos_tags" in self.CONFIG["ml_learner"].keys():
                pos_tags = self.CONFIG["ml_learner"]["pos_tags"]
            else:
                pos_tags = False

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
                            title, page_data, pos_tags
                        )
                        x_train.append(x_sample)
                        y_train.append(y_title)
                    elif titles_counter_temp[title] < titles_counter[title] * (
                        train + valid
                    ):
                        x_sample, y_title = self.__get_sample__(
                            title, page_data, pos_tags
                        )
                        x_val.append(x_sample)
                        y_val.append(y_title)
                    elif titles_counter_temp[title] < titles_counter[title] * (
                        train + valid + test
                    ):
                        x_sample, y_title = self.__get_sample__(
                            title, page_data, pos_tags
                        )
                        x_test.append(x_sample)
                        y_test.append(y_title)

                progress.update(task, advance=1)

        if min_counter != 0:
            print(f"Classes with count >= {min_counter}: {len(titles_counter_temp)}")
            self.titles_counter = titles_counter_temp

        return (
            np.float32(x_train),
            np.array(y_train),
            np.float32(x_val),
            np.array(y_val),
            np.float32(x_test),
            np.array(y_test),
        )

    def __get_gen_data__(self, batch: int = 4):

        BATCH = batch
        STEPS = int(len(self.pages) / BATCH)

        cur_batch_start = 0
        cur_batch_end = 0

        for i in range(STEPS):
            cur_batch_start += (BATCH * i) / len(self.pages)
            cur_batch_end += (BATCH * (i + 1)) / len(self.pages)

            x_train, y_train, x_val, y_val, x_test, y_test = self.__get_data__(
                train=cur_batch_start,
                valid=cur_batch_end,
                test=0,
                balanced=False,
                dont_know_class=False,
            )

            x_val = np.reshape(
                x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[3], x_val.shape[4])
            )
            y_val = keras.utils.to_categorical(y_val)

            yield [x_val, y_val]

    def __save_model_onnx__(
        self, models: list, models_names: List[str], input_shape: int
    ):
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [("float_input", FloatTensorType([None, input_shape]))]

        for model, model_name in zip(models, models_names):
            print(f"Saving model {model_name} ...")

            onx = convert_sklearn(model, initial_types=initial_type)

            with open(
                os.path.join(self.CONFIG["save"]["ml_models"], f"{model_name}.onnx"),
                "wb",
            ) as f:
                f.write(onx.SerializeToString())

    def learn(self):

        from numpy.distutils.system_info import get_info

        print(get_info("blas_opt"))
        print(get_info("lapack_opt"))

        num_pages = len(self.files_names)
        print(f"Pages: {num_pages}")

        if "batch" in self.CONFIG["ml_learner"].keys():
            BATCH = self.CONFIG["ml_learner"]["batch"]
        else:
            BATCH = num_pages
        print(f"Batch: {BATCH}")

        num_batches = math.ceil(num_pages / BATCH)
        print(f"Batches: {num_batches}")

        models = [
            #    RandomForestClassifier(n_estimators      = 100,
            #                           n_jobs            = -1,
            #                           random_state      = 42,
            #                           bootstrap         = False,
            #                           min_samples_split = 10),
            #    BalancedRandomForestClassifier(n_estimators  = 100,
            #                           n_jobs            = -1,
            #                           max_features      = 'sqrt',
            #                           random_state      = 42,
            #                           bootstrap         = False,
            #                           min_samples_split = 10)
            # OneVsOneClassifier(RandomForestClassifier(n_estimators = 100), n_jobs = -1)
            #    OneVsOneClassifier(SVC(), n_jobs = -1),
            #    autosklearn.classification.AutoSklearnClassifier(n_jobs=-1),
            AutoSklearn2Classifier(time_left_for_this_task=600, n_jobs=-1)
        ]
        models_names = [
            #  'RandomForestClassifier',
            # 'BalancedRandomForestClassifier',
            # 'SVC',
            # 'AutoSklearnClassifier',
            "AutoSklearnClassifier2",
        ]

        input_shape = ()

        for i in range(num_batches):
            print(f"------------------------------------------------")
            print(f"Batch: {i + 1} / {num_batches}")

            upper_border = i * BATCH + BATCH
            self.__load_data__(
                lower_border=i * BATCH,
                upper_border=upper_border if upper_border < num_pages else num_pages,
            )

            print(f"Classes num: {len(self.titles_idx)}")

            x_train, y_train, x_val, y_val, x_test, y_test = self.__get_data__()

            # print(f'Classes counters: {self.titles_counter}')
            print(f"X Train: {x_train.shape}")
            print(f"Y Train: {y_train.shape}")
            print(f"X Val:   {x_val.shape}")
            print(f"Y Val:   {y_val.shape}")
            print(f"X Test:  {x_test.shape}")
            print(f"Y Test:  {y_test.shape}")
            input_shape = x_train.shape[1:]

            # scaler = StandardScaler()
            # x_train = scaler.fit_transform(x_train)
            # x_test = scaler.transform(x_test)

            # norm = np.linalg.norm(x_train)
            # x_train = x_train / norm
            # x_test = x_test / norm

            with Progress() as progress:
                task = progress.add_task("[green]Training ...", total=len(models_names))

                for model, model_name in zip(models, models_names):
                    start_time = time.time()
                    print(f"Fit: {model_name}")

                    model.fit(x_train, y_train)

                    # print(model.leaderboard())
                    # pprint(model.show_models(), indent=4)

                    print(f"Time fitting: {time.time() - start_time}")
                    progress.update(task, advance=1)

            with Progress() as progress:
                task = progress.add_task("[green]Testing ...", total=len(models_names))

                import matplotlib.pyplot as plt

                for model, model_name in zip(models, models_names):
                    start_time = time.time()
                    print(f"Predict: {model_name}")

                    y_pred = model.predict(x_test)
                    score = accuracy_score(y_test, y_pred)
                    balanced_score = balanced_accuracy_score(y_test, y_pred)

                    print(f"Time predicting: {time.time() - start_time}")
                    print(f"Accuracy of {model_name}: {score:0.3f}")
                    print(f"Balanced Accuracy of {model_name}: {balanced_score:0.3f}")

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
                            df_report[col]
                            .apply(pd.to_numeric, downcast="float")
                            .fillna(0)
                        )

                    df_report.to_csv("out.csv", index=False, float_format="%.3f")

                    progress.update(task, advance=1)

        if (
            "save_model" in self.CONFIG["ml_learner"].keys()
            and self.CONFIG["ml_learner"]["save_model"]
        ):
            self.__save_model_onnx__(models, models_names, input_shape)

    def automl_learn(self):
        CONFIG_AUTH = json.loads(open("config.json", "r").read())
        # Set MinIO credintials:
        os.environ["AWS_ACCESS_KEY_ID"] = CONFIG_AUTH["minio_auth"]["AWS_ACCESS_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = CONFIG_AUTH["minio_auth"][
            "AWS_SECRET_ACCESS_KEY"
        ]
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = CONFIG_AUTH["minio_auth"][
            "MLFLOW_S3_ENDPOINT_URL"
        ]
        # Setting MLFlow Tracking Server URI:
        mlflow.set_tracking_uri(CONFIG_AUTH["mlflow_auth"]["tracking_uri"])
        tracking_uri = mlflow.get_tracking_uri()
        print("Current MLFlow Tracking Server URI: {}".format(tracking_uri))

        # Train
        # Create an experiment name, which must be unique and case sensitive
        EXPERIMENT_NAME = "4cls_Tracker_auto_sklearn_Train"
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            experiment_id = mlflow.create_experiment(
                EXPERIMENT_NAME,
                artifact_location=f"s3://mlflow-artifact/{EXPERIMENT_NAME}",
            )
            experiment = mlflow.get_experiment(experiment_id)
        else:
            experiment_id = experiment.experiment_id
        print("Name: {}".format(experiment.name))
        print("Experiment_id: {}".format(experiment.experiment_id))
        print("Artifact Location: {}".format(experiment.artifact_location))
        print("Tags: {}".format(experiment.tags))
        print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

        mlflow.sklearn.autolog()

        from numpy.distutils.system_info import get_info

        print(get_info("blas_opt"))
        print(get_info("lapack_opt"))

        num_pages = len(self.files_names)
        print(f"Pages: {num_pages}")

        if "batch" in self.CONFIG["ml_learner"].keys():
            BATCH = self.CONFIG["ml_learner"]["batch"]
        else:
            BATCH = num_pages
        print(f"Batch: {BATCH}")

        num_batches = math.ceil(num_pages / BATCH)
        print(f"Batches: {num_batches}")

        TOTAL_TIME = 60
        # model = autosklearn.classification.AutoSklearnClassifier(
        #     time_left_for_this_task=TOTAL_TIME,
        #     n_jobs=-1,
        #     # tmp_folder='/tmp/autosklearn_classification_example_tmp',
        #     # include={
        #     #     'classifier': ['random_forest']
        #     # },
        #     # ensemble_size=1,
        # )
        model = AutoSklearn2Classifier(time_left_for_this_task=TOTAL_TIME, n_jobs=-1)

        # model_name = 'AutoSklearnClassifier'
        model_name = "AutoSklearnClassifier2"

        input_shape = ()

        for i in range(num_batches):
            print(f"------------------------------------------------")
            print(f"Batch: {i + 1} / {num_batches}")

            upper_border = i * BATCH + BATCH
            self.__load_data__(
                lower_border=i * BATCH,
                upper_border=upper_border if upper_border < num_pages else num_pages,
            )

            print(f"Classes num: {len(self.titles_idx)}")

            x_train, y_train, x_val, y_val, x_test, y_test = self.__get_data__()

            # print(f'Classes counters: {self.titles_counter}')
            print(f"X Train: {x_train.shape}")
            print(f"Y Train: {y_train.shape}")
            print(f"X Val:   {x_val.shape}")
            print(f"Y Val:   {y_val.shape}")
            print(f"X Test:  {x_test.shape}")
            print(f"Y Test:  {y_test.shape}")
            input_shape = x_train.shape[1:]

            start_time = time.time()
            print(f"Fit: {model_name}")

            with mlflow.start_run(
                experiment_id=experiment_id, run_name=f"training_run_{TOTAL_TIME}"
            ) as run:
                model.fit(x_train, y_train)

                mlflow.log_text(model.leaderboard().to_csv(), "leaderboard.csv")
                mlflow.log_text(str(model.show_models()), "modells.txt")

                mlflow.sklearn.log_model(model, "4clsTrackerModel")
                model_uri = mlflow.get_artifact_uri("4clsTrackerModel")
                result = mlflow.evaluate(
                    model_uri,
                    data=x_test,
                    targets=y_test,
                    model_type="classifier",
                    dataset_name="4clsTracker",
                    evaluators=["default"],
                    evaluator_config={
                        "default": {"log_model_explainability": False},
                    },
                )

            print(f"Time fitting: {time.time() - start_time}")

    def autokeras_learn(self):
        CONFIG_AUTH = json.loads(open("config.json", "r").read())
        # Set MinIO credintials:
        os.environ["AWS_ACCESS_KEY_ID"] = CONFIG_AUTH["minio_auth"]["AWS_ACCESS_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = CONFIG_AUTH["minio_auth"][
            "AWS_SECRET_ACCESS_KEY"
        ]
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = CONFIG_AUTH["minio_auth"][
            "MLFLOW_S3_ENDPOINT_URL"
        ]
        # Setting MLFlow Tracking Server URI:
        mlflow.set_tracking_uri(CONFIG_AUTH["mlflow_auth"]["tracking_uri"])
        tracking_uri = mlflow.get_tracking_uri()
        print("Current MLFlow Tracking Server URI: {}".format(tracking_uri))

        # Train
        # Create an experiment name, which must be unique and case sensitive
        EXPERIMENT_NAME = "4cls_Tracker_auto_keras_Train"
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            experiment_id = mlflow.create_experiment(
                EXPERIMENT_NAME,
                artifact_location=f"s3://mlflow-artifact/{EXPERIMENT_NAME}",
            )
            experiment = mlflow.get_experiment(experiment_id)
        else:
            experiment_id = experiment.experiment_id
        print("Name: {}".format(experiment.name))
        print("Experiment_id: {}".format(experiment.experiment_id))
        print("Artifact Location: {}".format(experiment.artifact_location))
        print("Tags: {}".format(experiment.tags))
        print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

        mlflow.keras.autolog()

        from numpy.distutils.system_info import get_info

        print(get_info("blas_opt"))
        print(get_info("lapack_opt"))

        num_pages = len(self.files_names)
        print(f"Pages: {num_pages}")

        if "batch" in self.CONFIG["ml_learner"].keys():
            BATCH = self.CONFIG["ml_learner"]["batch"]
        else:
            BATCH = num_pages
        print(f"Batch: {BATCH}")

        num_batches = math.ceil(num_pages / BATCH)
        print(f"Batches: {num_batches}")

        MAX_TRIALS = 20
        model = ak.StructuredDataClassifier(overwrite=True, max_trials=MAX_TRIALS)
        model_name = "AutoKeras"

        input_shape = ()

        for i in range(num_batches):
            print(f"------------------------------------------------")
            print(f"Batch: {i + 1} / {num_batches}")

            upper_border = i * BATCH + BATCH
            self.__load_data__(
                lower_border=i * BATCH,
                upper_border=upper_border if upper_border < num_pages else num_pages,
            )

            print(f"Classes num: {len(self.titles_idx)}")

            x_train, y_train, x_val, y_val, x_test, y_test = self.__get_data__()

            # print(f'Classes counters: {self.titles_counter}')
            print(f"X Train: {x_train.shape}")
            print(f"Y Train: {y_train.shape}")
            print(f"X Val:   {x_val.shape}")
            print(f"Y Val:   {y_val.shape}")
            print(f"X Test:  {x_test.shape}")
            print(f"Y Test:  {y_test.shape}")
            input_shape = x_train.shape[1:]

            start_time = time.time()
            print(f"Fit: {model_name}")

            EPOCHS = 10
            with mlflow.start_run(
                experiment_id=experiment_id, run_name=f"{MAX_TRIALS}_{EPOCHS}"
            ) as run:
                model.fit(x_train, y_train, epochs=EPOCHS)
                model_keras = model.export_model()
                mlflow.keras.log_model(
                    model_keras, "4clsTrackerModel", keras_module=tf.keras
                )
                results = model.evaluate(x_test, y_test)
                mlflow.log_metrics(
                    {
                        "test_loss": results[0],
                        "test_acc": results[1],
                    }
                )
                print("test loss, acc:", results)
                # model_uri = mlflow.get_artifact_uri("4clsTrackerModel")
                # result = mlflow.evaluate(
                #     model_uri,
                #     data=x_test,
                #     targets=y_test,
                #     model_type="classifier",
                #     dataset_name="4clsTracker",
                #     evaluators=["default"],
                #     evaluator_config={
                #         'default': {'log_model_explainability': False},
                #     },
                # )

            print(f"Time fitting: {time.time() - start_time}")


if __name__ == "__main__":
    mll = MLLearner()
    mll.autokeras_learn()
