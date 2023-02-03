import collections
import json
import os

import pandas as pd

from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['Tracker350']


class Tracker350(DatasetBuilder):

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

        from pathlib import Path
        path = Path(filename).parent
        self.labels = json.loads(open(os.path.join(path, 'labels.json'), "r").read())

        for i in range(len(df)):
            text = df['text'][i]
            label = df['classes'][i]

            label = str(self.labels[label])

            qid += 1
            yield {"text": text, "label": label, "qid": qid}

    def get_labels(self):
        """
        Return labels of the Tracker4 object.
        """
        try:
            return [str(val) for val in self.labels.values()]

        except AttributeError:
            default_root = os.path.join(DATA_HOME, self.__class__.__name__)
            self.labels = json.loads(open(os.path.join(default_root, 'labels.json'), "r").read())

            return [str(val) for val in self.labels.values()]