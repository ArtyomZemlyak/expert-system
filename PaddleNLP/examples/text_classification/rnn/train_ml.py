# pip install fast-mosestokenizer
# pip install nltk
# nltk.download("stopwords")
# pip install pymystem3

import os
import re
import json
from collections import Counter, defaultdict
import random

import numpy as np
import pandas as pd
from rich.progress import Progress

from sklearn.ensemble import RandomForestClassifier
from imblearn.metrics import classification_report_imbalanced

from gensim.models import TfidfModel
from gensim import corpora

from mosestokenizer import MosesTokenizer
from pymystem3 import Mystem
from nltk.corpus import stopwords

CONFIG = {**json.loads(open('config.json', "r").read()), **json.loads(open('scripts/config.json', "r").read())}

PATH_TRAIN = 'TEST/train_data_4c/train.csv'
PATH_VALID = 'TEST/train_data_4c/val.csv'

df_train = pd.read_csv(PATH_TRAIN, header=0)
df_test = pd.read_csv(PATH_VALID, header=0)

nlp = MosesTokenizer('ru')
mystem = Mystem()
extend_stopwords = {*stopwords.words("russian"), *stopwords.words("english"), *CONFIG['analyzer']['stop_words']}

x_train = []
y_train = []
x_test  = []
y_test  = []

with Progress() as progress:
    task = progress.add_task("[green]Prepare train data ...", total=len(df_train))

    for str_page_data, title in zip(df_train['text'], df_train['classes']):
        tokens = nlp.tokenize(str_page_data)
        tokens = [token.lower() for token in tokens]
        tokens = [mystem.lemmatize(token)[0] for token in tokens
                  if token not in extend_stopwords]
        tokens = [token for token in tokens
                  if token not in extend_stopwords and token != '' and token != ' ']

        x_train.append(tokens)
        y_train.append(title)

        progress.update(task, advance=1)

with Progress() as progress:
    task = progress.add_task("[green]Prepare test data ...", total=len(df_test))

    for str_page_data, title in zip(df_test['text'], df_test['classes']):
        tokens = nlp.tokenize(str_page_data)
        tokens = [token.lower() for token in tokens]
        tokens = [mystem.lemmatize(token)[0] for token in tokens
                  if token not in extend_stopwords]
        tokens = [token for token in tokens
                  if token not in extend_stopwords and token != '' and token != ' ']

        x_test.append(tokens)
        y_test.append(title)

        progress.update(task, advance=1)

print(f'X Train: {len(x_train)}')
print(f'Y Train: {len(y_train)}')
print(f'X Test:  {len(x_test)}')
print(f'Y Test:  {len(y_test)}')

dct = corpora.Dictionary(x_train)
corpus = [dct.doc2bow(line) for line in x_train]

tf_idf = TfidfModel(corpus=corpus,
                    id2word=dct,
                    smartirs='nfu')


data_vec = []

with Progress() as progress:
    task = progress.add_task("[green]Convert train data to Vec ...", total=len(x_train))

    for c in tf_idf[corpus]:
        vector = np.zeros(len(dct))

        for idx, val in c:
            vector[idx] = val

        data_vec.append(vector)

        progress.update(task, advance=1)

x_train_vec = np.float32(data_vec)
print(f'X Train Vec: {x_train_vec.shape}')

model = RandomForestClassifier(n_estimators       = 100,
                                n_jobs            = -1,
                                max_features      = 'sqrt',
                                random_state      = 42,
                                bootstrap         = False,
                                min_samples_split = 10)

model.fit(x_train_vec, y_train)

data_vec = []
corpus = [dct.doc2bow(line) for line in x_test]

with Progress() as progress:
    task = progress.add_task("[green]Convert test data to Vec ...", total=len(x_test))

    for c in tf_idf[corpus]:
        vector = np.zeros(len(dct))

        for idx, val in c:
            vector[idx] = val

        data_vec.append(vector)

        progress.update(task, advance=1)

x_test_vec = np.float32(data_vec)
print(f'X Test Vec: {x_test_vec.shape}')

y_pred = model.predict(x_test_vec)

print(classification_report_imbalanced(y_test, y_pred))