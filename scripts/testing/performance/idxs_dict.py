from operator import itemgetter
import time

a = {"a": {"aaa": 1}, "b": {"bbb": 2}, "c": {"ccc": 3}, "d": {"ddd": 4}}
b = {0: {"aaa": 1}, 1: {"bbb": 2}, 2: {"ccc": 3}, 3: {"ddd": 4}}


# st = time.time()
# for i in range(1000000):
#     aa = a['a']
#     ab = a['b']
#     ac = a['c']
#     ad = a['d']
# print("Time dict[str]: ", (time.time() - st))

# st = time.time()
# for i in range(1000000):
#     aa = a.get('a')
#     ab = a.get('b')
#     ac = a.get('c')
#     ad = a.get('d')
# print("Time dict.get(str): ", (time.time() - st))

# st = time.time()
# for i in range(1000000):
#     ba = b[0]
#     bb = b[1]
#     bc = b[2]
#     bd = b[3]
# print("Time dict[int]: ", (time.time() - st))

# st = time.time()
# for i in range(1000000):
#     ba = b.get(0)
#     bb = b.get(1)
#     bc = b.get(2)
#     bd = b.get(3)
# print("Time dict.get(int): ", (time.time() - st))


# from sortedcontainers import SortedDict
# sa = SortedDict(a)

# st = time.time()
# for i in range(1000000):
#     saa = sa['a']
#     sab = sa['b']
#     sac = sa['c']
#     sad = sa['d']
# print("Time SortedDict[str]: ", (time.time() - st))

# st = time.time()
# for i in range(1000000):
#     saa = sa.get('a')
#     sab = sa.get('b')
#     sac = sa.get('c')
#     sad = sa.get('d')
# print("Time SortedDict.get(str): ", (time.time() - st))

# import random
# random.seed(12)

# ilr = random.sample([j for j in range(1000000)], 1000000)

# st = time.time()
# lr = []
# for i in ilr:
#     lr.append((i, i))
# lrs = sorted(lr, key=itemgetter(1), reverse=True)
# print("Time sorting list: ", (time.time() - st))

# random.seed(12)
# from sortedcontainers import SortedList

# st = time.time()
# slr = SortedList()
# for i in ilr:
#     slr.add(i)
# print("Time SortedList: ", (time.time() - st))

# st = time.time()
# lrs = sorted(ilr, reverse=True)
# print("Time sorting list init: ", (time.time() - st))

# st = time.time()
# slr = SortedList(ilr)
# print("Time SortedList init: ", (time.time() - st))

"""
Time dict[str]:  0.11409163475036621
Time dict.get(str):  0.16921734809875488
Time dict[int]:  0.11444973945617676
Time dict.get(int):  0.1674649715423584
Time SortedDict[str]:  0.17408084869384766
Time SortedDict.get(str):  0.2012622356414795
Time sorting list:  0.5627508163452148
Time SortedList:  1.7157132625579834
Time sorting list init:  0.22346735000610352
Time SortedList init:  0.23517751693725586
"""

# from bidict import bidict

# bda = bidict({'a': 'aa', 'b': 'bb', 'c': 'cc', 'd': 'dd'})
# st = time.time()
# for i in range(1000000):
#     bdaa = bda['a']
#     bdab = bda['b']
#     bdac = bda['c']
#     bdad = bda['d']
# print("Time bidict[str]: ", (time.time() - st))
# st = time.time()
# for i in range(1000000):
#     bdaa = bda.inverse['aa']
#     bdab = bda.inverse['bb']
#     bdac = bda.inverse['cc']
#     bdad = bda.inverse['dd']
# print("Time bidict[str] back: ", (time.time() - st))

"""
Time bidict[str]:  0.46440982818603516
Time bidict[str] back:  1.0031468868255615
"""

# from cytoolz.itertoolz import get

# st = time.time()
# for i in range(1000000):
#     aa = get('a', a)
#     ab = get('b', a)
#     ac = get('c', a)
#     ad = get('d', a)
# print("Time dict. cytoolz.itertoolz.get(str): ", (time.time() - st))

"""
Time dict. cytoolz.itertoolz.get(str):  0.24231672286987305
"""

# from collections import namedtuple

# Point = namedtuple('Point', ['a', 'b', 'c', 'd'])

# ant = Point({'aaa': 1}, {'bbb': 2}, {'ccc': 3}, {'ddd': 4})

# st = time.time()
# for i in range(1000000):
#     aa = ant.a
#     ab = ant.b
#     ac = ant.c
#     ad = ant.d
# print("Time namedtuple.attr: ", (time.time() - st))

# st = time.time()
# for i in range(1000000):
#     aa = ant[0]
#     ab = ant[1]
#     ac = ant[2]
#     ad = ant[3]
# print("Time namedtuple[int]: ", (time.time() - st))

# at = ({'aaa': 1}, {'bbb': 2}, {'ccc': 3}, {'ddd': 4})
# st = time.time()
# for i in range(1000000):
#     aa = at[0]
#     ab = at[1]
#     ac = at[2]
#     ad = at[3]
# print("Time tuple[int]: ", (time.time() - st))

"""
Time dict[str]:  0.11524081230163574
Time namedtuple.attr:  0.16199827194213867
Time namedtuple[int]:  0.10444426536560059
Time tuple[int]:  0.10629844665527344
"""

# st = time.time()
# for i in range(1000000):
#     if 'a' in a.keys():
#         aa = a['a']
# print("Time if 'a' in a.keys(): ", (time.time() - st))

# st = time.time()
# for i in range(1000000):
#     if 'a' in a:
#         aa = a['a']
# print("Time if 'a' in a: ", (time.time() - st))

# st = time.time()
# for i in range(1000000):
#     aa = a.get('a')
# print("Time a.get('a'): ", (time.time() - st))


# st = time.time()
# for i in range(1000000):
#     for j in ['in', 'out', 'bi', 'none']:
#         aa = j
# print("Time for in list: ", (time.time() - st))

# st = time.time()
# for i in range(1000000):
#     for j in ('in', 'out', 'bi', 'none'):
#         aa = j
# print("Time for in tuple: ", (time.time() - st))

"""
Time for in list:  0.1329951286315918
Time for in tuple:  0.13979578018188477
"""


# import polars as pl
# df = pl.DataFrame(
#     {
#         # "idx": [0, 1, 2, 3],
#         "fruits": [{'aaa': 1}, {'bbb': 2}, {'ccc': 3}, {'ddd': 4}],
#     }
# )

# # print(df[1])
# st = time.time()
# for i in range(1000000):
#     aa = df[1]
#     ab = df[2]
#     ac = df[3]
#     ad = df[4]
# print("Time polars[int]: ", (time.time() - st))
# """
# Time polars[int]:  4.507963418960571
# """


# from pymystem3 import Mystem
# import pickle
# a = Mystem()

# st = time.time()
# for i in range(1000):
#     with open("TESTINGGGG", 'wb') as pickle_file:
#         pickle.dump(a, pickle_file)
#         pickle_file.close()
# print("Time pickle.dump(Mystem): ", (time.time() - st))

# import joblib

# st = time.time()
# for i in range(1000):
#     with open("TESTINGGGG", 'wb') as pickle_file:
#         joblib.dump(a, pickle_file)
#         pickle_file.close()
# print("Time joblib.dump(Mystem): ", (time.time() - st))

# a = {f"{i}": {f"{i*i}": i*i} for i in range(1000000)}

# st = time.time()
# # for i in range(1000):
# with open("TESTINGGGG", 'wb') as pickle_file:
#     pickle.dump(a, pickle_file)
#     pickle_file.close()
# print("Time pickle.dump(Mystem): ", (time.time() - st))

# import joblib

# st = time.time()
# # for i in range(1000):
# with open("TESTINGGGG", 'wb') as pickle_file:
#     joblib.dump(a, pickle_file)
#     pickle_file.close()
# print("Time joblib.dump(Mystem): ", (time.time() - st))

"""
Time pickle.dump(Mystem):  0.12641143798828125
Time joblib.dump(Mystem):  0.18958139419555664
Time pickle.dump(dict):  0.6744322776794434
Time joblib.dump(dict):  11.051838636398315
"""

# import joblib
# from joblib import Parallel, delayed

# def op_save():
#     with open("TESTINGGGG", 'wb') as pickle_file:
#         joblib.dump(a, pickle_file)
#         pickle_file.close()


# st = time.time()
# Parallel(n_jobs=16, prefer="threads")(delayed(op_save)() for i in range(1000))
# print("Time joblib.Parallel.dump(Mystem): ", (time.time() - st))

"""
Time joblib.Parallel.dump(Mystem):  0.3055100440979004
Time joblib.Parallel pickle.dump(Mystem):  0.20440673828125
"""


# def op_save(aa, ii):
#     with open("TESTINGGGG" + str(ii), 'wb') as pickle_file:
#         pickle.dump(aa, pickle_file)
#         pickle_file.close()

# alist = [{f"{i}": {f"{i*i}": i*i} for i in range(100000)},
#          {f"{i}": {f"{i*i}": i*i} for i in range(100000, 200000)},
#          {f"{i}": {f"{i*i}": i*i} for i in range(200000, 300000)},
#          {f"{i}": {f"{i*i}": i*i} for i in range(300000, 400000)},
#          {f"{i}": {f"{i*i}": i*i} for i in range(400000, 500000)},
#          {f"{i}": {f"{i*i}": i*i} for i in range(500000, 600000)},
#          {f"{i}": {f"{i*i}": i*i} for i in range(600000, 700000)},
#          {f"{i}": {f"{i*i}": i*i} for i in range(700000, 800000)},
#          {f"{i}": {f"{i*i}": i*i} for i in range(800000, 900000)},
#          {f"{i}": {f"{i*i}": i*i} for i in range(900000, 1000000)}]

# st = time.time()
# Parallel(n_jobs=10, prefer="threads")(delayed(op_save)(data, i) for i, data in enumerate(alist))
# print("Time joblib.Parallel pickle.dump(dict): ", (time.time() - st))
"""
Time joblib.Parallel.dump(dict) threads:  11.430967807769775
Time joblib.Parallel.dump(dict) processes:  9.812779664993286
Time joblib.Parallel pickle.dump(dict) threads: 0.42201852798461914
Time joblib.Parallel pickle.dump(dict) processes: 7.8042449951171875
"""
