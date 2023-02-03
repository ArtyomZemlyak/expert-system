import collections
import json
import os

import pandas as pd

from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['Tracker4']


class Tracker4(DatasetBuilder):

    SPLITS = {
        'train': 'train.csv',
        'val': 'val.csv',
        'test': 'test.csv',
    }

    def _get_data(self, mode, **kwargs):
        """Downloads dataset."""
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        filename = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)

        return fullname

    def _read(self, filename, split):
        """Reads data."""

        qid = 0
        df = pd.read_csv(filename, header=0)

        for i in range(len(df)):
            text = df['text'][i]
            label = df['classes'][i]

            if label =='CONSULTATION': label = '0'
            elif label == 'IMPROVEMENT_REQUEST': label = '1'
            elif label == 'SERVICE_REQUEST': label = '2'
            elif label == 'SLA': label = '3'

            qid += 1
            yield {"text": text, "label": label, "qid": qid}

    def get_labels(self):
        """
        Return labels of the Tracker4 object.
        """
        return ["0", "1", "2", "3"]
