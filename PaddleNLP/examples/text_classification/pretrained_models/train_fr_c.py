# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import time
import json

import pandas as pd
import numpy as np
import paddle
import paddle.nn.functional as F

import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad, tokenizer, Vocab
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup

from imblearn.metrics import classification_report_imbalanced

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=3, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proption over the training process.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))
    model.train()
    metric.reset()


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens. And creates a mask from the two sequences passed 
    to be used in a sequence-pair classification task.
        
    A BERT sequence has the following format:

    - single sequence: ``[CLS] X [SEP]``
    - pair of sequences: ``[CLS] A [SEP] B [SEP]``

    A BERT sequence pair mask has the following format:
    ::
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |

    If only one sequence, only returns the first portion of the mask (0's).


    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): List of sequence pair mask.
        label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
    """
    encoded_inputs = tokenizer(text=example["text"], max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)


def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    train_ds, dev_ds, test_ds = load_dataset("tracker_4", splits=["train", "val", "test"])

    # tokenizer = ppnlp.transformers.ErnieTokenizer(vocab_file='TEST/train_data_4c/vocab.txt')
    # tokenizer = ppnlp.transformers.BertTokenizer(vocab_file='TEST/train_data_4c/vocab.txt')
    # tokenizer = ppnlp.transformers.ErnieTinyTokenizer.from_pretrained('ernie-tiny')

    tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    # tokenizer = ppnlp.transformers.GPTTokenizer.from_pretrained('gpt2-medium-en')

    # tokenizer = ppnlp.transformers.AlbertTokenizer.from_pretrained('albert-chinese-tiny')
    # tokenizer = ppnlp.transformers.RobertaTokenizer.from_pretrained('roberta-wwm-ext')
    # tokenizer = ppnlp.transformers.ElectraTokenizer.from_pretrained('chinese-electra-small', num_classes=2)
    #tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained('ernie-2.0-en')

    # model = ppnlp.transformers.ErnieForSequenceClassification(
    #     ppnlp.transformers.ErnieModel(vocab_size=tokenizer.vocab_size), num_classes=len(train_ds.label_list))
    # model = ppnlp.transformers.BertForSequenceClassification(
    #     ppnlp.transformers.BertModel(vocab_size=tokenizer.vocab_size), num_classes=len(train_ds.label_list))

    model = ppnlp.transformers.BertForSequenceClassification.from_pretrained(
        'bert-base-multilingual-cased', num_classes=len(train_ds.label_list))
    # model = ppnlp.transformers.GPTForSequenceClassification.from_pretrained(
    #     'gpt2-medium-en', num_classes=len(train_ds.label_list))

    # model = ppnlp.transformers.AlbertForSequenceClassification.from_pretrained(
    #     'albert-chinese-tiny', num_classes=len(train_ds.label_list))
    # model = ppnlp.transformers.RobertaForSequenceClassification.from_pretrained('roberta-wwm-ext', num_classes=len(train_ds.label_list))
    # model = ppnlp.transformers.ElectraForSequenceClassification.from_pretrained('chinese-electra-small', num_classes=2)
    # model = ppnlp.transformers.ErnieForSequenceClassification.from_pretrained(
    #     'ernie-2.0-en', num_classes=len(train_ds.label_list))

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]

    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    dev_data_loader = create_dataloader(
        dev_ds,
        mode='validation',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    test_loader = create_dataloader(
        test_ds,
        mode='validation',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)

    model = paddle.DataParallel(model)

    num_training_steps = len(train_data_loader) * args.epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    global_step = 0
    tic_train = time.time()

    try:
        for epoch in range(1, args.epochs + 1):
            for step, batch in enumerate(train_data_loader, start=1):
                input_ids, token_type_ids, labels = batch
                logits = model(input_ids, token_type_ids)
                loss = criterion(logits, labels)
                probs = F.softmax(logits, axis=1)
                correct = metric.compute(probs, labels)
                metric.update(correct)
                acc = metric.accumulate()

                global_step += 1
                if global_step % 10 == 0 and rank == 0:
                    print("global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                            % (global_step, epoch, step, loss, acc, 10 / (time.time() - tic_train)))
                    tic_train = time.time()

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()

                if global_step % 100 == 0 and rank == 0:
                    evaluate(model, criterion, metric, dev_data_loader)

                if global_step % 10000 == 0 and rank == 0:
                    save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    model._layers.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)

    except KeyboardInterrupt:
        save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model._layers.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    label_map = {0: 'CONSULTATION',
                 1: 'IMPROVEMENT_REQUEST',
                 2: 'SERVICE_REQUEST',
                 3: 'SLA'}
    results = []

    model.eval()

    for batch in test_loader:
        input_ids, seq_lens, labels = batch
        input_ids = paddle.to_tensor(input_ids)
        seq_lens = paddle.to_tensor(seq_lens)
        logits = model(input_ids, seq_lens)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)

    texts = [''.join(tokenizer.vocab.to_tokens(sample[0])) for sample in test_ds]
    labels = [label_map[int(sample[2])] for sample in test_ds]

    csv_data = pd.DataFrame(data={'text': texts, 'classes': labels, 'predictions': results})
    csv_data.to_csv('TEST/res.csv', index=False)

    print(classification_report_imbalanced(labels, results))


if __name__ == "__main__":
    do_train()


"""
| 模型  | dev acc | test acc |
| ---- | ------- | -------- |
| bert-base-chinese  | 0.93833 | 0.94750 |
| bert-wwm-chinese | 0.94583 | 0.94917 |
| bert-wwm-ext-chinese | 0.94667 | 0.95500 |
| ernie-1.0  | 0.94667  | 0.95333  |
| ernie-tiny  | 0.93917  | 0.94833 |
| roberta-wwm-ext  | 0.94750  | 0.95250 |
| roberta-wwm-ext-large | 0.95250 | 0.95333 |
| rbt3 | 0.92583 | 0.93250 |
| rbtl3 | 0.9341 | 0.93583 |

    --max_seq_length=256 \
    --save_dir='./TEST/train_data_4c/checkpoints/ernie_2.0'

    bert_base_multilingual_cased
    albert-chinese-tiny
    ernie-2.0-en
    gpt2-medium-en

71 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal)
python PaddleNLP/examples/text_classification/pretrained_models/train.py \
    --device=gpu \
    --learning_rate=1e-5 \
    --batch_size=1 \
    --epochs=100 \
    --warmup_proportion=0.01 \
    --save_dir='./TEST/train_data_4c/checkpoints/gpt2-medium-en'

71 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal) vocab.txt
73.9 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal)
71.6 (train - non bal) (test - non bal)
python PaddleNLP/examples/text_classification/pretrained_models/train.py \
    --device=gpu \
    --learning_rate=1e-5 \
    --batch_size=6 \
    --epochs=100 \
    --warmup_proportion=0.01 \
    --save_dir='./TEST/train_data_4c/checkpoints/bert_base_multilingual_cased'

70 (train - non bal) (test - non bal)
python PaddleNLP/examples/text_classification/pretrained_models/train.py \
    --device=gpu \
    --learning_rate=1e-5 \
    --batch_size=16 \
    --epochs=100 \
    --warmup_proportion=0.01 \
    --save_dir='./TEST/train_data_4c/checkpoints/bert_vocab'

63.8 (train - non bal) (test - non bal)
python PaddleNLP/examples/text_classification/pretrained_models/train.py \
    --device=gpu \
    --learning_rate=1e-5 \
    --batch_size=16 \
    --epochs=100 \
    --warmup_proportion=0.01 \
    --save_dir='./TEST/train_data_4c/checkpoints/albert-chinese-tiny'

66.6 (train - non bal) (test - non bal)
70.6 (train - non bal -> +aug bal + nltk +aug bal) (test - non bal)
python PaddleNLP/examples/text_classification/pretrained_models/train.py \
    --device=gpu \
    --learning_rate=1e-5 \
    --batch_size=8 \
    --epochs=100 \
    --warmup_proportion=0.01 \
    --save_dir='./TEST/train_data_4c/checkpoints/ernie-2.0-en'

"""