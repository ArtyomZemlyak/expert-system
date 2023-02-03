from functools import partial
import numpy as np

from functools import partial
import argparse
import os
import random
import json

import pandas as pd
from rich.progress import Progress
from imblearn.metrics import classification_report_imbalanced

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as T

from paddle.io import Dataset

from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab
from paddlenlp.datasets import load_dataset
import paddlenlp as ppnlp

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


zeros_ = nn.initializer.Constant(value=0.)
ones_ = nn.initializer.Constant(value=1.)
trunc_normal_ = nn.initializer.TruncatedNormal(std=.02)


def to_2tuple(x):
    return tuple([x] * 2)


def drop_path(x, drop_prob = 0., training = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = paddle.to_tensor(keep_prob) + paddle.rand(shape)
    random_tensor = paddle.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output

class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbed(nn.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose((0, 2, 1))  # BCHW -> BNC
        x = self.norm(x)
        return x

class HybridEmbed(nn.Layer):

    def __init__(self, backbone, img_size=224, patch_size=1, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        if feature_size is None:
            with paddle.no_grad():

                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(paddle.zeros([1, in_chans, img_size[0], img_size[1]]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        self.num_patches = feature_size[0] // patch_size[0] * feature_size[1] // patch_size[1]
        self.proj = nn.Conv2D(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  
        x = self.proj(x).flatten(2).transpose([0, 2, 1])
        return x


def repeat(x, rep):
    return paddle.to_tensor(np.tile(x.numpy(), rep))


def repeat_interleave(x, rep, axis):
    return paddle.to_tensor(np.repeat(x.numpy(), rep, axis=axis))


def einsum(str, distances, attn_map):
    d = distances.numpy()
    a = attn_map.numpy()
    out = np.einsum(str, (d, a))

    return paddle.to_tensor(out)

class GPSA(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 locality_strength=1., use_local_init=True):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = nn.Linear(dim, dim * 2, bias_attr=qkv_bias)
        self.v = nn.Linear(dim, dim, bias_attr=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(3, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.locality_strength = locality_strength

        self.gating_param = self.create_parameter(shape=[self.num_heads], default_initializer=ones_)
        self.add_parameter("gating_param", self.gating_param)


    def forward(self, x):
        B, N, C = x.shape
        if not hasattr(self, 'rel_indices') or self.rel_indices.shape[1]!=N:
            self.get_rel_indices(N)

        attn = self.get_attention(x)
        v = self.v(x).reshape([B, N, self.num_heads, C // self.num_heads]).transpose([0, 2, 1, 3])
        x = (attn @ v).transpose([0, 2, 1, 3])
        x = x.reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attention(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape([B, N, 2, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        q, k = qk[0], qk[1]
        pos_score = self.rel_indices.expand([B, -1, -1,-1])
        pos_score = self.pos_proj(pos_score).transpose([0,3,1,2]) 
        patch_score = (q @ k.transpose([0, 1, 3, 2])) * self.scale
        patch_score = F.softmax(patch_score, axis=-1)
        pos_score = F.softmax(pos_score, axis=-1)

        gating = self.gating_param.reshape([1, -1, 1, 1])
        attn = (1. - F.sigmoid(gating)) * patch_score + F.sigmoid(gating) * pos_score
        attn /= attn.sum(axis=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, x, return_map = False):

        attn_map = self.get_attention(x).mean(0) 
        distances = self.rel_indices.squeeze()[:,:,-1]**.5
        dist = einsum('nm,hnm->h', distances, attn_map)      # einsum
        dist /= distances.shape[0]
        if return_map:
            return dist, attn_map
        else:
            return dist

    def get_rel_indices(self, num_patches):
        img_size = int(num_patches**.5)
        rel_indices = paddle.zeros([1, num_patches, num_patches, 3])
        ind = paddle.arange(img_size).reshape([1,-1]) - paddle.arange(img_size).reshape([-1, 1])
        indx = repeat(ind, [img_size, img_size])
        indy = repeat_interleave(ind, img_size, axis=0)
        indy = repeat_interleave(indy, img_size, axis=1)
        indd = indx**2 + indy**2
        indd = indd.astype('float32')
        indy = indy.astype('float32')
        indx = indx.astype('float32')
        rel_indices[:,:,:,2] = indd.unsqueeze(0)
        rel_indices[:,:,:,1] = indy.unsqueeze(0)
        rel_indices[:,:,:,0] = indx.unsqueeze(0)
        self.rel_indices = rel_indices

    def local_init(self):
        self.v.weight.set_value(paddle.eye(self.dim))
        locality_distance = 1  # max(1,1/locality_strength**.5)

        kernel_size = int(self.num_heads ** .5)
        center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1 + kernel_size * h2
                self.pos_proj.weight[2, position] = -1
                self.pos_proj.weight[1, position] = 2 * (h1 - center) * locality_distance
                self.pos_proj.weight[0, position] = 2 * (h2 - center) * locality_distance

        self.pos_proj.weight.set_value(self.pos_proj.weight * self.locality_strength)

class MHSA(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def get_attention_map(self, x, return_map = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_map = (q @ k.transpose([0, 1, 3, 2])) * self.scale
        attn_map = F.softmax(attn_map, axis=-1).mean(0)

        img_size = int(N**.5)
        ind = paddle.arange(img_size).reshape([1,-1]) - paddle.arange(img_size).reshape([-1, 1])
        indx = repeat(ind, [img_size, img_size])
        indy = repeat_interleave(ind, img_size, axis=0)
        indy = repeat_interleave(indy, img_size, axis=1)
        indd = indx**2 + indy**2
        distances = indd**.5

        dist = einsum('nm,hnm->h', distances, attn_map)   # einsum
        dist /= N

        if return_map:
            return dist, attn_map
        else:
            return dist

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose([0, 1, 3, 2])) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose([0,2,1,3]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Layer):

    def __init__(self, dim, num_heads,  mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_gpsa=True, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_gpsa = use_gpsa
        if self.use_gpsa:
            self.attn = GPSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, **kwargs)
        else:
            self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Layer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, vocab_size=30000, emb_dim=256, padding_idx=0, img_size=224, patch_size=16, in_chans=438, num_classes=1000, embed_dim=48, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, global_pool=None,
                 local_up_to_layer=10, locality_strength=1., use_pos_embed=True):
        super().__init__()
        embed_dim *= num_heads
        self.num_classes = num_classes
        self.local_up_to_layer = local_up_to_layer
        self.num_features = self.embed_dim = embed_dim
        self.use_pos_embed = use_pos_embed

        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = self.create_parameter(shape=[1, 1, embed_dim], default_initializer=nn.initializer.TruncatedNormal(mean=0.0, std=.02))
        self.add_parameter("cls_token", self.cls_token)

        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.use_pos_embed:

            self.pos_embed = self.create_parameter(shape=[1, num_patches, embed_dim], default_initializer=nn.initializer.TruncatedNormal(mean=0.0, std=.02))
            self.add_parameter("pos_embed", self.pos_embed)


        dpr = [x for x in paddle.linspace(0, drop_path_rate, depth)]  
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_gpsa=True,
                locality_strength=locality_strength)
            if i<local_up_to_layer else
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_gpsa=False)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else Identity()

        self.apply(self._init_weights)
        for n, m in self.named_sublayers():
            if hasattr(m, 'local_init'):
                m.local_init()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        B = x.shape[0]

        encoder_out = self.embedder(x)
        x = paddle.reshape(encoder_out, (B,  x.shape[1], 16, 16))
        print(x.shape)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand([B, -1, -1])

        if self.use_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for u,blk in enumerate(self.blocks):
            if u == self.local_up_to_layer :
                x = paddle.concat((cls_tokens, x), axis=1)
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


DIM = 16


def convit_tiny(**kwargs):
    model = VisionTransformer(
        img_size=DIM,
        patch_size=8,
        # drop_rate=0.1,
        # drop_path_rate=0.1,
        # attn_drop_rate=0.1,
        num_heads=4,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def convit_small(**kwargs):
    model = VisionTransformer(
        img_size=DIM,
        patch_size=8,
        # drop_rate=0.1,
        # drop_path_rate=0.1,
        # attn_drop_rate=0.1,
        num_heads=9,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


def convit_base(**kwargs):
    model = VisionTransformer(
        img_size=DIM,
        patch_size=8,
        # drop_rate=0.1,
        # drop_path_rate=0.1,
        # attn_drop_rate=0.1,
        num_heads=16,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    return model


TRANSFORM = T.Compose([
    #T.Resize(size=(60,60)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], data_format='HWC'),
    T.Transpose()
    #T.ToTensor()
])

class Tracker4(Dataset):

    def __init__(self,
                 path: str,
                 mode: str):

        self.tokenizer = None
        if mode == 'word':
            self.tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        self.data = self._load_dataset(path)

    def _load_dataset(self, path):

        df = pd.read_csv(path, header=0)

        data = []

        with Progress() as progress:
            task = progress.add_task("[green]Loading dataset ...", total=len(df))

            for i in range(len(df)):
                text = df['text'][i]
                label = df['classes'][i]

                if label =='CONSULTATION': label = '0'
                elif label == 'IMPROVEMENT_REQUEST': label = '1'
                elif label == 'SERVICE_REQUEST': label = '2'
                elif label == 'SLA': label = '3'

                text_image = self._txt_to_img(text)

                progress.update(task, advance=1)

                data.append([text_image, np.array(int(label))])

        return data

    def _txt_to_img(self, text):

        text_image = np.zeros(shape=(DIM, DIM, 3), dtype=np.float32)

        if self.tokenizer: tokens = self.tokenizer.tokenize(text)
        else: tokens = text

        try:
            for i, token in enumerate(tokens):
                if self.tokenizer: token_idx = self.tokenizer.vocab.to_indices(token)
                else: token_idx = ord(token)

                rgb = (token_idx // 1000, token_idx // 100, token_idx % 100)
                #print(f'{char} -> {rgb}')

                row_idx = i // DIM
                text_image[i - row_idx * DIM][row_idx] = rgb

        except IndexError: pass

        return TRANSFORM(text_image)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class SaveBestModel(paddle.callbacks.Callback):

    def __init__(self, target=0.5, path='work/best_model', verbose=0):

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


if __name__ == "__main__":

    paddle.set_device(args.device)
    set_seed()

    # train_ds = Tracker4('TEST/train_data_4c/train_aug_nltk.csv', mode='word')
    # dev_ds = Tracker4('TEST/train_data_4c/val.csv', mode='word')
    # test_ds = Tracker4('TEST/train_data_4c/test.csv', mode='word')

    tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    train_ds, dev_ds, test_ds  =  load_dataset("tracker_4",  splits=["train", "val", "test"])  # load_dataset(read_dataset, lazy=False)

    print('----------------------------------------')
    network = args.network.lower()
    print(f'Network: {network.upper()}')

    vocab_size = len(tokenizer.vocab)
    print(f'Vocab size: {vocab_size}')

    num_classes = len(train_ds.label_list)
    print(f'Num classes: {num_classes}')

    pad_token_id = tokenizer.vocab.to_indices('[PAD]')
    print(f'Pad token ID: {pad_token_id}')

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

    model = paddle.Model(convit_tiny(num_classes=4))

    print(model.summary((1, 438, DIM, DIM)))

    model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=args.lr, parameters=model.parameters()),
                  loss=paddle.nn.CrossEntropyLoss(),
                  metrics=paddle.metric.Accuracy())

    callback = paddle.callbacks.ProgBarLogger(log_freq=10, verbose=3)
    callback_vdl = paddle.callbacks.VisualDL(log_dir=args.save_dir)
    callback_savebestmodel = SaveBestModel(target=0.5, path=os.path.join(args.save_dir, 'best'))

    model.fit(
        train_data=train_loader,#train_ds,
        eval_data=dev_loader,#dev_ds,
        batch_size=args.batch_size,
        epochs=args.epochs,
        #save_dir=args.save_dir,
        callbacks=[callback, callback_vdl, callback_savebestmodel])

    label_map = {0: 'CONSULTATION',
                 1: 'IMPROVEMENT_REQUEST',
                 2: 'SERVICE_REQUEST',
                 3: 'SLA'}
    results = []

    model.load(os.path.join(args.save_dir, 'best'))

    logits = model.predict(test_loader)#test_ds)

    for step in logits:
        for batch in step:
            for pos in batch:
                results.append(label_map[np.argmax(pos)])

    df = pd.read_csv('TEST/train_data_4c/test.csv', header=0)
    texts = df.text
    labels = df.classes

    csv_data = pd.DataFrame(data={'text': texts, 'classes': labels, 'predictions': results})
    csv_data.to_csv('TEST/res.csv', index=False)

    print(classification_report_imbalanced(labels, results))


"""
val acc: 0. (train - non bal -> +aug bal + nltk +aug bal) (test - non bal)
python PaddleNLP/examples/text_classification/rnn/train_convit.py \
    --device=gpu \
    --lr=1e-3 \
    --batch_size=128 \
    --epochs=60 \
    --save_dir='./TEST/train_data_4c/checkpoints/convit'

visualdl --logdir='./TEST/train_data_4c/checkpoints/convit' \
    --port=8040 \
    --cache-timeout=5 \
    --language=en
"""
