import time
from pprint import pprint

import sys

sys.path.insert(0, "scripts")

from dbes.DBESNet import DBESNet
from dbes.DBESNode import DBESNode


st = time.time()
d = DBESNet()
print("Time up: ", time.time() - st)


################################################################################################################################
# Генеративная огромная база
################################################################################################################################
# 10_000
# init    py - 284 ms    rust - 74 ms
# val     py - 6 ms;     rust - 3.5 ms
# 100_000
# init    py - 3.04 s    rust - 0.64 s
# val     py - 63 ms     rust - 46 ms
# rec rel_NORec   py - 278 ms   rust - 143 ms
# rust with FxHashMap (not sj::Value)
# init            rust - 0.508 s
# rec rel_NORec   rust - 29 ms
len_db = 100000
val_dict = {}
for i in range(len_db):
    val_dict[str(i)] = str(i)
rel_dict = {}
for i in range(len_db - 1):
    if str(i) in rel_dict.keys():
        rel_dict[str(i)] = {**rel_dict[str(i)], **{str(i + 1): {}}}
    else:
        rel_dict[str(i)] = {str(i + 1): {}}
for i in range(int(len_db / 2) - 1):
    if str(i) in rel_dict.keys():
        rel_dict[str(i)] = {**rel_dict[str(i)], **{str(i * 2): {}}}
    else:
        rel_dict[str(i)] = {str(i * 2): {}}
for i in range(int(len_db / 7) - 7):
    if str(i) in rel_dict.keys():
        rel_dict[str(i)] = {**rel_dict[str(i)], **{str(i * 7): {}}}
    else:
        rel_dict[str(i)] = {str(i * 7): {}}
d.import_from_sep_json(val_dict, rel_dict)
