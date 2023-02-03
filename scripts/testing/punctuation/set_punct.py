# https://github.com/benob/recasepunc

# REC:
# git+https://github.com/benob/mosestokenizer.git
# numpy==1.19.5
# regex==2021.8.28
# torch==1.9.0+cu111
# tqdm==4.62.2
# transformers==4.10.0

#  python scripts/testing/punctuation/set_punct.py scripts/testing/punctuation/ru-test.txt

import os
import sys
import time
import pathlib
import pprint

from transformers import logging

import sys

sys.path.insert(0, "scripts")
from Punct import CasePuncPredictor
from Punct import WordpieceTokenizer
from Punct import Config


logging.set_verbosity_error()

PATH_CHECKPONT = os.path.join(pathlib.Path(__file__).parent.resolve(), "checkpoint")
LANG = "ru"


predictor = CasePuncPredictor(PATH_CHECKPONT, lang=LANG)  # , device='cpu'

text = " ".join(open(sys.argv[1]).readlines())

st_t = time.time()
tokens = list(enumerate(predictor.tokenize(text)))
et_t = time.time() - st_t

results = ""

st_p = time.time()
for token, case_label, punc_label in predictor.predict(tokens, lambda x: x[1]):
    prediction = predictor.map_punc_label(
        predictor.map_case_label(token[1], case_label), punc_label
    )
    if token[1][0] != "#":
        results = results + " " + prediction
    else:
        results = results + prediction
et_p = time.time() - st_p

print(results.strip())
print("TIME (tokenize): ", et_t)
print("TIME (predict): ", et_p)


"""

Performance test:

-- GPU RTX 3080 Laptop 8GB
VRAM: 3-4 GB used

ru-test.txt
TIME (tokenize):  0.0031158924102783203
TIME (tokenize):  0.00475764274597168
TIME (predict):  0.44614410400390625
TIME (predict):  0.5393309593200684

ru-test-talk.txt
TIME (tokenize):  0.0006060600280761719
TIME (tokenize):  0.001958608627319336
TIME (predict):  0.3897697925567627
TIME (predict):  0.49454522132873535

ru-test-small.txt
TIME (tokenize):  0.00026297569274902344
TIME (tokenize):  0.0003178119659423828
TIME (predict):  0.4022097587585449
TIME (predict):  0.44533419609069824

ru-test-large.txt
TIME (tokenize):  0.017473459243774414
TIME (tokenize):  0.017130374908447266
TIME (predict):  0.503040075302124
TIME (predict):  0.5920202732086182

-- CPU i7-11800H 16 T
RAM: 5-6 GB

ru-test.txt
TIME (tokenize):  0.0022864341735839844
TIME (predict):  0.46126437187194824


after warmup:

-- GPU RTX 3080 Laptop 8GB
VRAM: 3-4 GB used

ru-test.txt
TIME (tokenize):  0.00260162353515625
TIME (predict):  0.010879278182983398

ru-test-talk.txt
TIME (tokenize):  0.0006690025329589844
TIME (predict):  0.009139537811279297

ru-test-small.txt
TIME (tokenize):  0.0002522468566894531
TIME (predict):  0.009634971618652344

ru-test-large.txt
TIME (tokenize):  0.017720460891723633
TIME (tokenize):  0.018797874450683594
TIME (predict):  0.0955495834350586
TIME (predict):  0.147047758102417


-- CPU i7-11800H  16 T
RAM: 5-6 GB

ru-test.txt
TIME (tokenize):  0.0022115707397460938
TIME (predict):  0.016228199005126953

ru-test-talk.txt
TIME (tokenize):  0.0011470317840576172
TIME (predict):  0.011316776275634766

ru-test-small.txt
TIME (tokenize):  0.0002493858337402344
TIME (predict):  0.009507894515991211

ru-test-large.txt
TIME (tokenize):  0.018456220626831055
TIME (tokenize):  0.10389113426208496
TIME (predict):  0.023000240325927734
TIME (predict):  0.11957955360412598
"""
