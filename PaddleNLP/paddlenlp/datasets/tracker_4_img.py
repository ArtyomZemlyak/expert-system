import collections
import json
import os

import numpy as np
import pandas as pd
from rich.progress import Progress

import paddle.vision.transforms as T
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['Tracker4']


TRANSFORM = T.Compose([
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], data_format='HWC'),
    T.Transpose()
    #T.ToTensor()
])


class Tracker4(DatasetBuilder):

    SPLITS = {
        'train': 'train.csv',
        'val': 'val.csv',
        'test': 'test.csv',
    }
    DIM = 60

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

        with Progress() as progress:
            task = progress.add_task("[green]Loading dataset ...", total=len(df))

            for i in range(len(df)):
                text = df['text'][i]
                label = df['classes'][i]

                if label =='CONSULTATION': label = '0'
                elif label == 'IMPROVEMENT_REQUEST': label = '1'
                elif label == 'SERVICE_REQUEST': label = '2'
                elif label == 'SLA': label = '3'

                qid += 1
                text = self.txt_to_img(text)
                progress.update(task, advance=1)

                yield {"text": text, "label": label, "qid": qid}

    def txt_to_img(self, text):

        text_image = np.zeros(shape=(self.DIM, self.DIM, 3), dtype=np.uint8)

        try:
            for i, char in enumerate(text):
                char_idx = ord(char)
                rgb = (char_idx // 1000, char_idx // 100, char_idx % 100)
                #print(f'{char} -> {rgb}')
                row_idx = i // self.DIM
                text_image[i - row_idx * self.DIM][row_idx] = rgb

        except IndexError: pass

        return TRANSFORM(text_image)

    def get_labels(self):
        """
        Return labels of the Tracker4 object.
        """
        return ["0", "1", "2", "3"]
