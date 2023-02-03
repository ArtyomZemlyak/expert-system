# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
import argparse
import os
import random
import json
from typing import List
from examples.text_classification.rnn.model import TCNModel

import numpy as np
import pandas as pd

import paddle
from paddle.fluid.contrib.model_stat import summary

import paddlenlp as ppnlp
import paddle.nn.functional as F
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab
from paddlenlp.datasets import load_dataset

from imblearn.metrics import classification_report_imbalanced

from model import BoWModel, BiLSTMAttentionModel, CNNModel, TextCNNModel, TCNModel, LSTMGRUModel, LSTMModel, GRUModel, RNNModel, SelfInteractiveAttention
from utils import convert_example

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--epochs", type=int, default=10, help="Number of epoches for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate used to train.")
parser.add_argument("--save_dir", type=str, default='checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number of a batch for training.")
parser.add_argument("--vocab_path", type=str, default="./senta_word_dict.txt", help="The directory to dataset.")
parser.add_argument('--network', choices=['bow', 'lstmgru', 'lstm', 'bilstm', 'gru', 'bigru', 'rnn', 'birnn', 'bilstm_attn', 'cnn', 'tcnn', 'tcn'],
    default="bilstm", help="Select which network to train, defaults to bilstm.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
args = parser.parse_args()
# yapf: enable


def set_seed(seed=1000):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def create_dataloader(dataset,
                      trans_fn=None,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None):
    """
    Creats dataloader.

    Args:
        dataset(obj:`paddle.io.Dataset`): Dataset instance.
        trans_fn(obj:`callable`, optional, defaults to `None`): function to convert a data sample to input ids, etc.
        mode(obj:`str`, optional, defaults to obj:`train`): If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): The sample number of a mini-batch.
        batchify_fn(obj:`callable`, optional, defaults to `None`): function to generate mini-batch data by merging
            the sample list, None for only stack each fields of sample in axis
            0(same as :attr::`np.stack(..., axis=0)`).

    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == "train":
        sampler = paddle.io.DistributedBatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        sampler = paddle.io.BatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader = paddle.io.DataLoader(
        dataset, batch_sampler=sampler, collate_fn=batchify_fn)
    return dataloader


class SaveBestModel(paddle.callbacks.Callback):
    def __init__(self, target=0.3, path='work/best_model', verbose=0):
        self.target = target
        self.epoch = None
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch

    def on_eval_end(self, logs=None):
        if logs.get('acc') > self.target:
            self.target = logs.get('acc')
            self.model.save(self.path)
            print('best acc is {} at epoch {}'.format(self.target, self.epoch))


if __name__ == "__main__":
    paddle.set_device(args.device)
    set_seed()

    # Loads vocab.
    if not os.path.exists(args.vocab_path):
        raise RuntimeError('The vocab_path  can not be found in the path %s' %
                           args.vocab_path)

    # vocab = Vocab.load_vocabulary(args.vocab_path, unk_token='[UNK]', pad_token='[PAD]')

    # vocab_dict = json.loads(open(args.vocab_path, "r").read())['model']['vocab']
    # vocab = Vocab.from_dict(vocab_dict, unk_token='[UNK]', pad_token='[PAD]')

    # tokenizer = JiebaTokenizer(vocab)
    tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    train_ds, dev_ds, test_ds  =  load_dataset("tracker_4",  splits=["train", "val", "test"])  # load_dataset(read_dataset, lazy=False)
    #train_ds, dev_ds, test_ds  =  load_dataset("tracker_350",  splits=["train", "val", "test"])  # load_dataset(read_dataset, lazy=False)

    print('----------------------------------------')
    network = args.network.lower()
    print(f'Network: {network.upper()}')

    vocab_size = len(tokenizer.vocab)
    print(f'Vocab size: {vocab_size}')

    num_classes = len(train_ds.label_list)
    print(f'Num classes: {num_classes}')

    pad_token_id = tokenizer.vocab.to_indices('[PAD]')
    print(f'Pad token ID: {pad_token_id}')

    if network == 'bow':
        model = BoWModel(vocab_size,
                         num_classes,
                         padding_idx=pad_token_id)

    elif network == 'bigru':
        model = GRUModel(vocab_size,
                         num_classes,
                         direction='bidirect',
                         padding_idx=pad_token_id)

    elif network == 'bilstm':
        model = LSTMModel(vocab_size,
                          num_classes,
                          direction='bidirect',
                          padding_idx=pad_token_id)

    elif network == 'bilstm_attn':
        lstm_hidden_size = 512
        attention = SelfInteractiveAttention(hidden_size=lstm_hidden_size)
        model = BiLSTMAttentionModel(attention_layer=attention,
                                     vocab_size=vocab_size,
                                     lstm_hidden_size=lstm_hidden_size,
                                     num_classes=num_classes,
                                     padding_idx=pad_token_id)

    elif network == 'birnn':
        model = RNNModel(vocab_size,
                         num_classes,
                         direction='bidirect',
                         padding_idx=pad_token_id)

    elif network == 'cnn':
        model = CNNModel(vocab_size,
                         num_classes,
                         padding_idx=pad_token_id)

    elif network == 'tcnn':
        model = TextCNNModel(vocab_size,
                             num_classes,
                             padding_idx=pad_token_id)

    elif network == 'tcn':
        model = TCNModel(vocab_size,
                         num_classes,
                         padding_idx=pad_token_id)

    elif network == 'gru':
        model = GRUModel(vocab_size,
                         num_classes,
                         direction='forward',
                         padding_idx=pad_token_id,
                         pooling_type='max')

    elif network == 'lstm':
        model = LSTMModel(vocab_size,
                          num_classes,
                          direction='forward',
                          padding_idx=pad_token_id,
                          pooling_type='max')

    elif network == 'lstmgru':
        model = LSTMGRUModel(vocab_size,
                             num_classes,
                             padding_idx=pad_token_id,
                             pooling_type='mean')

    elif network == 'rnn':
        model = RNNModel(vocab_size,
                         num_classes,
                         direction='forward',
                         padding_idx=pad_token_id,
                         pooling_type='max')
    else:
        raise ValueError(f"Unknown network: {network}, it must be one of bow, lstm, bilstm, cnn, tcnn, gru, bigru, rnn, birnn and bilstm_attn.")

    model = paddle.Model(model)

    trans_fn = partial(convert_example, tokenizer=tokenizer, is_test=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.vocab.token_to_idx.get('[PAD]', 0)),  # input_ids
        Stack(dtype="int64"),  # seq len
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]

    train_loader = create_dataloader(
        train_ds,
        trans_fn=trans_fn,
        batch_size=args.batch_size,
        mode='train',
        batchify_fn=batchify_fn)

    dev_loader = create_dataloader(
        dev_ds,
        trans_fn=trans_fn,
        batch_size=args.batch_size,
        mode='validation',
        batchify_fn=batchify_fn)

    test_loader = create_dataloader(
        test_ds,
        trans_fn=trans_fn,
        batch_size=args.batch_size,
        mode='test',
        batchify_fn=batchify_fn)

    optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                      learning_rate=args.lr)

    # Defines loss and metric.
    criterion = paddle.nn.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    model.prepare(optimizer, criterion, metric)

    # Loads pre-trained parameters.
    if args.init_from_ckpt:
        model.load(args.init_from_ckpt)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    # Starts training and evaluating.
    callback = paddle.callbacks.ProgBarLogger(log_freq=10, verbose=3)
    callback_vdl = paddle.callbacks.VisualDL(log_dir=args.save_dir)
    callback_savebestmodel = SaveBestModel(target=0.3, path=os.path.join(args.save_dir, 'best'))

    try:
        model.fit(train_loader,
                dev_loader,
                epochs=args.epochs,
                save_dir=args.save_dir,
                callbacks=[callback, callback_vdl, callback_savebestmodel])

    except KeyboardInterrupt: pass

    label_map = {0: 'CONSULTATION',
                 1: 'IMPROVEMENT_REQUEST',
                 2: 'SERVICE_REQUEST',
                 3: 'SLA'}

    # label_map = json.loads(open('TEST/train_data_350c/labels_map.json', "r").read())
    # label_map = {int(idx): label for idx, label in label_map.items()}

    results = []

    model.load(os.path.join(args.save_dir, 'best'))

    logits = model.predict(test_loader)

    for step in logits:
        for batch in step:
            for pos in batch:
                results.append(label_map[np.argmax(pos)])

    texts = [''.join(tokenizer.vocab.to_tokens(sample[0])) for sample in test_ds]
    labels = [label_map[int(sample[2])] for sample in test_ds]

    csv_data = pd.DataFrame(data={'text': texts, 'classes': labels, 'predictions': results})
    csv_data.to_csv('TEST/res.csv', index=False)

    print(classification_report_imbalanced(labels, results))




"""
pip install -e .
pip install imbalanced-learn

DLNNs:
'bow', 'lstm', 'bilstm', 'gru', 'bigru', 'rnn', 'birnn', 'bilstm_attn', 'cnn', 'tcnn', 'tcn'
| 模型  | dev acc | test acc |
| ---- | ------- | -------- |
| BoW  |  0.8970 | 0.8908   |
| Bi-LSTM  | 0.9098  | 0.8983  |
| Bi-GRU  | 0.9014  | 0.8785  |
| Bi-RNN  | 0.8649  |  0.8504 |
| Bi-LSTM Attention |  0.8992 |  0.8856 |
| TextCNN  | 0.9102  | 0.9107 |

Vocabs:
1 - $HOME'/.deeppavlov/downloads/bert_models/multi_cased_L-12_H-768_A-12/vocab.txt
2 - $HOME'/.deeppavlov/downloads/bert_models/rubert_cased_L-12_H-768_A-12_v1/vocab.txt
3 - $HOME'/.paddlenlp/models/ernie-tiny/vocab.txt
4 - $HOME'/BI/expert-system/TEST/train_data_4c/vocab.json' (+ 0.015)

    --init_from_ckpt

-- 350 c
test f1: 0.38 (train - non bal) (test - non bal) (emb_dim=128, padding_idx=0, num_filter=256, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh dropout1=0.1 max avg pool bert_tokenizer
test f1: 0.46 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) (emb_dim=128, padding_idx=0, num_filter=256, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh dropout1=0.1 max avg pool bert_tokenizer
-- 4 c
val acc: 0.7 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) (emb_dim=300, padding_idx=0, num_filter=256, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh dropout1=0.1 max avg pool bert_tokenizer deeppavlov_fasttext_emb
val acc: 0.714 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) (emb_dim=128, padding_idx=0, num_filter=256, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh dropout1=0.1 max avg pool bert_tokenizer
val acc: 0.713 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) (emb_dim=1024, padding_idx=0, num_filter=256, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh dropout1=0.1 max avg pool
val acc: 0.7 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) (emb_dim=256, padding_idx=0, num_filter=256, ngram_filter_sizes=(3, ), fc_hidden_size=128) Tanh dropout1=0.1 max avg pool
val acc: 0.7 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) (emb_dim=128, padding_idx=0, num_filter=256, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh dropout1=0.1 max avg pool
val acc: 0.682 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) (emb_dim=512, padding_idx=0, num_filter=128, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh dropout1=0.1 max avg pool
val acc: 0.684 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) (emb_dim=128, padding_idx=0, num_filter=128, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh dropout1=0.1 max avg pool
val acc: 0.679 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) (emb_dim=128, padding_idx=0, num_filter=128, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh dropout1=0.1 dropout2=0.1
val acc: 0.696 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) (emb_dim=128, padding_idx=0, num_filter=128, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh dropout1=0.1
val acc: 0.683 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) (emb_dim=128, padding_idx=0, num_filter=128, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh dropout1=0.2
val acc: 0.695 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) (emb_dim=128, padding_idx=0, num_filter=128, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh
val acc: 0.718 (train - non bal + (non bal nltk -> +aug bal)) (test - non bal) (emb_dim=128, padding_idx=0, num_filter=128, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh
val acc: 0.64 (train - non bal -> +copy bal) (test - non bal) (emb_dim=128, padding_idx=0, num_filter=128, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh
val acc: 0.678 (train - non bal -> +aug bal) (test - non bal) (emb_dim=128, padding_idx=0, num_filter=128, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh
val acc: 0.644 (train - bal) (test - bal) (emb_dim=128, padding_idx=0, num_filter=128, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh
val acc: 0.6 (train - bal) (test - non bal) (emb_dim=128, padding_idx=0, num_filter=128, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh
val acc: 0.69 (train - non bal) (test - non bal) (emb_dim=128, padding_idx=0, num_filter=128, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh
val acc: 0.853 (train - non bal) (test - bad bal) (emb_dim=128, padding_idx=0, num_filter=128, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh
val acc: 0.571 (train - non bal) (test - good bal) (emb_dim=128, padding_idx=0, num_filter=128, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh
val acc: 0.574 (train - non bal) (test - good bal) (emb_dim=128, padding_idx=0, num_filter=512, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh
val acc: 0.564 (train - non bal) (test - good bal) (emb_dim=128, padding_idx=0, num_filter=128, ngram_filter_sizes=(3, ), fc_hidden_size=96) ReLU
val acc: 0.62 (train - non bal + nltk non bal) (test - good bal) (emb_dim=128, padding_idx=0, num_filter=128, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh
val acc: 0.67 (train - nltk non bal) (test - nltk non bal) (emb_dim=128, padding_idx=0, num_filter=128, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh
val acc: 0.58 (train - nltk non bal) (test - nltk good bal) (emb_dim=128, padding_idx=0, num_filter=128, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh
-- 4 c fl
test f1: 0.70 (train - non bal) (test - non bal) (emb_dim=128, padding_idx=0, num_filter=256, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh dropout1=0.1 max avg pool bert_tokenizer
test f1: 0.69 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) (emb_dim=128, padding_idx=0, num_filter=256, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh dropout1=0.1 max avg pool bert_tokenizer
test f1: 0.61 (train - non bal -> +aug bal) (test - bal) (emb_dim=128, padding_idx=0, num_filter=256, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh dropout1=0.1 max avg pool bert_tokenizer
test f1: 0.71 (train - non bal -> + nltk) (test - non bal) (emb_dim=128, padding_idx=0, num_filter=256, ngram_filter_sizes=(3, ), fc_hidden_size=96) Tanh dropout1=0.1 max avg pool bert_tokenizer
python PaddleNLP/examples/text_classification/rnn/train.py \
    --vocab_path=$HOME'/BI/expert-system/TEST/train_data_4c/vocab.json' \
    --device=gpu \
    --network=cnn \
    --lr=5e-4 \
    --batch_size=256 \
    --epochs=60 \
    --save_dir='./TEST/train_data_4c_fl/checkpoints/cnn'

val acc: 0.627 (train - bal) (test - bal) (emb_dim=128, padding_idx=0, num_filter=128, ngram_filter_sizes=(1, 2, 3), fc_hidden_size=96)
val acc: 0.633 (train - bal) (test - bal) (emb_dim=512, padding_idx=0, num_filter=128, ngram_filter_sizes=(1, 2, 3), fc_hidden_size=96)
val acc: 0.648 (train - non bal) (test - non bal) (emb_dim=512, padding_idx=0, num_filter=128, ngram_filter_sizes=(1, 2, 3), fc_hidden_size=96) dropout=0.1
val acc: 0.876 (train - non bal) (test - bad bal) (emb_dim=512, padding_idx=0, num_filter=128, ngram_filter_sizes=(1, 2, 3), fc_hidden_size=96)
val acc: 0.54 (train - non bal) (test - good bal) (emb_dim=512, padding_idx=0, num_filter=128, ngram_filter_sizes=(1, 2, 3), fc_hidden_size=96)
python PaddleNLP/examples/text_classification/rnn/train.py \
    --vocab_path=$HOME'/BI/expert-system/TEST/train_data_4c/vocab.json' \
    --device=gpu \
    --network=tcnn \
    --lr=5e-4 \
    --batch_size=128 \
    --epochs=60 \
    --save_dir='./TEST/train_data_4c/checkpoints/tcnn'

val acc: 0. (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) (emb_dim=128, padding_idx=0, num_channels=[128], kernel_size=3, fc_hidden_size=96)
python PaddleNLP/examples/text_classification/rnn/train.py \
    --vocab_path=$HOME'/BI/expert-system/TEST/train_data_4c/vocab.json' \
    --device=gpu \
    --network=tcn \
    --lr=5e-4 \
    --batch_size=64 \
    --epochs=60 \
    --save_dir='./TEST/train_data_4c/checkpoints/tcn'

val acc: 0.891 (train - non bal) (test - bad bal) (emb_dim=512, padding_idx=0, gru_hidden_size=198, direction='forward', gru_layers=1, dropout_rate=0.0, pooling_type='max', fc_hidden_size=96)
val acc: 0.546 (train - non bal) (test - good bal) (emb_dim=128, padding_idx=0, gru_hidden_size=198, direction='forward', gru_layers=1, dropout_rate=0.2, pooling_type='max', fc_hidden_size=96)
val acc: 0.575 (train - nltk non bal) (test - nltk good bal) (emb_dim=128, padding_idx=0, gru_hidden_size=198, direction='forward', gru_layers=1, dropout_rate=0.2, pooling_type='max', fc_hidden_size=96)
python PaddleNLP/examples/text_classification/rnn/train.py \
    --vocab_path=$HOME'/BI/expert-system/TEST/train_data_4c/vocab.json' \
    --device=gpu \
    --network=gru \
    --lr=5e-4 \
    --batch_size=128 \
    --epochs=60 \
    --save_dir='./TEST/train_data_4c/checkpoints/gru'

val acc: 0.76 (train - non bal) (test - bad bal) (emb_dim=512, padding_idx=0, gru_hidden_size=198, direction='bidirect', gru_layers=1, dropout_rate=0.0, pooling_type=None, fc_hidden_size=96)
python PaddleNLP/examples/text_classification/rnn/train.py \
    --vocab_path=$HOME'/BI/expert-system/TEST/train_data_4c/vocab.json' \
    --device=gpu \
    --network=bigru \
    --lr=5e-4 \
    --batch_size=32 \
    --epochs=100 \
    --save_dir='./TEST/train_data_4c/checkpoints/bigru'

val acc: 0.69 (train - non bal) (test - bad bal)
python PaddleNLP/examples/text_classification/rnn/train.py \
    --vocab_path=$HOME'/BI/expert-system/TEST/train_data_4c/vocab.json' \
    --device=gpu \
    --network=bow \
    --lr=5e-4 \
    --batch_size=512 \
    --epochs=100 \
    --save_dir='./TEST/train_data_4c/checkpoints/bow'

-- 350 c
test f1: 0.4 (train - non bal) (test - non bal) (emb_dim=768, padding_idx=0, lstm_hidden_size=256, direction='forward', lstm_layers=1, dropout_rate=0.05, pooling_type='max', fc_hidden_size=96) dropout1=0.1 bert_tokenizer
test f1: 0.5 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) (emb_dim=768, padding_idx=0, lstm_hidden_size=256, direction='forward', lstm_layers=1, dropout_rate=0.05, pooling_type='max', fc_hidden_size=96) dropout1=0.1 bert_tokenizer
-- 4 c
val acc: 0.723 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) (emb_dim=768, padding_idx=0, lstm_hidden_size=256, direction='forward', lstm_layers=1, dropout_rate=0.05, pooling_type='max', fc_hidden_size=96) dropout1=0.1 bert_tokenizer
val acc: 0.73 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) (emb_dim=768, padding_idx=0, lstm_hidden_size=256, direction='forward', lstm_layers=1, dropout_rate=0.05, pooling_type='max', fc_hidden_size=96) dropout1=0.1
val acc: 0.71 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) (emb_dim=512, padding_idx=0, lstm_hidden_size=198, direction='forward', lstm_layers=1, dropout_rate=0.0, pooling_type='max', fc_hidden_size=96)
val acc: 0.66 (train - non bal -> +aug bal) (test - non bal) (emb_dim=512, padding_idx=0, lstm_hidden_size=198, direction='forward', lstm_layers=1, dropout_rate=0.0, pooling_type='max', fc_hidden_size=96)
val acc: 0.71 (train - non bal) (test - non bal) (emb_dim=512, padding_idx=0, lstm_hidden_size=198, direction='forward', lstm_layers=1, dropout_rate=0.0, pooling_type='max', fc_hidden_size=96)
val acc: 0.7  (train - non bal + nltk non bal) (test - non bal) (emb_dim=512, padding_idx=0, lstm_hidden_size=256, direction='forward', lstm_layers=1, dropout_rate=0.2, pooling_type='mean', fc_hidden_size=96)
val acc: 0.91 (train - non bal) (test - bad bal) (emb_dim=512, padding_idx=0, lstm_hidden_size=198, direction='forward', lstm_layers=1, dropout_rate=0.0, pooling_type='max', fc_hidden_size=96)
val acc: 0.92 (train - non bal) (test - bad bal) (emb_dim=1024, padding_idx=0, lstm_hidden_size=512, direction='forward', lstm_layers=3, dropout_rate=0.1, pooling_type='max', fc_hidden_size=128)
python PaddleNLP/examples/text_classification/rnn/train.py \
    --vocab_path=$HOME'/BI/expert-system/TEST/train_data_4c/vocab.json' \
    --device=gpu \
    --network=lstm \
    --lr=5e-4 \
    --batch_size=32 \
    --epochs=100 \
    --save_dir='./TEST/train_data_4c/checkpoints/lstm'

python PaddleNLP/examples/text_classification/rnn/train.py \
    --vocab_path=$HOME'/BI/expert-system/TEST/train_data_4c/vocab.json' \
    --device=gpu \
    --network=lstmgru \
    --lr=5e-4 \
    --batch_size=32 \
    --epochs=100 \
    --save_dir='./TEST/train_data_4c/checkpoints/lstmgru'

val acc: 0.7 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) (emb_dim=512, padding_idx=0, lstm_hidden_size=256, direction='bidirect', lstm_layers=2, dropout_rate=0.05, pooling_type=None, fc_hidden_size=96) dropout1=0.1
python PaddleNLP/examples/text_classification/rnn/train.py \
    --vocab_path=$HOME'/BI/expert-system/TEST/train_data_4c/vocab.json' \
    --device=gpu \
    --network=bilstm \
    --lr=5e-4 \
    --batch_size=16 \
    --epochs=100 \
    --save_dir='./TEST/train_data_4c/checkpoints/bilstm'

val acc: 0.69 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) (emb_dim=1024, padding_idx=0, lstm_hidden_size=512, direction='forward', lstm_layers=2, dropout_rate=0.05, pooling_type=None, fc_hidden_size=96) dropout1=0.1
val acc: 0.67 (non balanced) (emb_dim=1024, padding_idx=0, lstm_hidden_size=512, lstm_layers=2, dropout_rate=0.1, fc_hidden_size=256)
python PaddleNLP/examples/text_classification/rnn/train.py \
    --vocab_path=$HOME'/BI/expert-system/TEST/train_data_4c/vocab.json' \
    --device=gpu \
    --network=bilstm_attn \
    --lr=5e-4 \
    --batch_size=16 \
    --epochs=100 \
    --save_dir='./TEST/train_data_4c/checkpoints/lstm_attn'

visualdl --logdir='./TEST/train_data_4c/checkpoints/lstm' \
    --port=8040 \
    --cache-timeout=5 \
    --language=en
"""