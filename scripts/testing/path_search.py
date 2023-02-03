from operator import itemgetter
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
# Поиск путей между узлами и вар опросник
################################################################################################################################
# questions_idxs = d.find_mult_rel_idxs({"in": [DBESNode.to_idx('Опросник стройки'), DBESNode.to_idx('Статистический показатель')]}, and_on=False)
# last_dist_nodes = d.find_shortest_dist(filter_idxs=questions_idxs)
# for i in range(100):
#     dist_nodes = d.find_shortest_dist_z(filter_idxs=questions_idxs)
#     print(last_dist_nodes == dist_nodes)

# pprint({d._net[idx].value: {d._net[idx_].value: val for idx_, val in val_dict.items()} for idx, val_dict in dist_nodes.items()})
# pprint({d._net[idx].value: val for idx, val in  d.find_all_dir_rel_idxs_lvls(DBESNode.to_idx("во9"), dir_rel="in", filter_idxs=questions_idxs).items()})

# pattern = {
#     "relation": {
#         "settings": {
#             "and_on": False
#         },
#         "task": {
#             "in": [DBESNode.to_idx('Опросник стройки'), DBESNode.to_idx('Статистический показатель')]
#         }
#     },
#     "shortest_dist": { "task": {} }
# }
# pprint(d.find(pattern))

# val_dict = {
#     DBESNode.to_idx('Тест кратчайшая дистанция'): 'Тест кратчайшая дистанция',
#     DBESNode.to_idx('sd1'): 'sd1',
#     DBESNode.to_idx('sd2'): 'sd2',
#     DBESNode.to_idx('sd3'): 'sd3',
#     DBESNode.to_idx('sd4'): 'sd4',
#     DBESNode.to_idx('sd5'): 'sd5',
#     DBESNode.to_idx('sd6'): 'sd6',
#     DBESNode.to_idx('sd7'): 'sd7',
#     DBESNode.to_idx('sd8'): 'sd8',
#     DBESNode.to_idx('sd9'): 'sd9',
#     DBESNode.to_idx('sd10'): 'sd10',
#     DBESNode.to_idx('sd11'): 'sd11',
#     DBESNode.to_idx('sd12'): 'sd12',
#     DBESNode.to_idx('sd13'): 'sd13',
#     DBESNode.to_idx('sd14'): 'sd14'
# }
# rel_dict = {
#     DBESNode.to_idx('Тест кратчайшая дистанция'): {
#         DBESNode.to_idx('sd1'): {},
#         DBESNode.to_idx('sd2'): {},
#         DBESNode.to_idx('sd3'): {},
#         DBESNode.to_idx('sd4'): {},
#         DBESNode.to_idx('sd5'): {},
#         DBESNode.to_idx('sd6'): {},
#         DBESNode.to_idx('sd7'): {},
#         DBESNode.to_idx('sd8'): {},
#         DBESNode.to_idx('sd9'): {},
#         DBESNode.to_idx('sd10'): {},
#         DBESNode.to_idx('sd11'): {},
#         DBESNode.to_idx('sd12'): {},
#         DBESNode.to_idx('sd13'): {},
#         DBESNode.to_idx('sd14'): {}
#     },
#     DBESNode.to_idx('sd7'): {
#         DBESNode.to_idx('sd1'): {
#             DBESNode.to_idx('sd2'): {
#                 DBESNode.to_idx('sd3'): {
#                     DBESNode.to_idx('sd4'): {},
#                 }
#             },
#             DBESNode.to_idx('sd4'): {DBESNode.to_idx('sd6'): {}},
#             DBESNode.to_idx('sd5'): {DBESNode.to_idx('sd2'): {}}
#         },
#         DBESNode.to_idx('sd8'): {DBESNode.to_idx('sd9'): {DBESNode.to_idx('sd10'): {DBESNode.to_idx('sd11'): {DBESNode.to_idx('sd12'): {DBESNode.to_idx('sd13'): {DBESNode.to_idx('sd14'): {}, DBESNode.to_idx('sd6'): {}}}}}}}
#     },
# }
# d.import_from_sep_json(val_dict, rel_dict)

# questions_idxs = d.find_mult_rel_idxs({"in": DBESNode.to_idx('Тест кратчайшая дистанция')})
# dist_nodes = d.find_shortest_dist_z(filter_idxs=questions_idxs)
# pprint({d._net[idx].value: {d._net[idx_].value: val for idx_, val in val_dict.items()} for idx, val_dict in dist_nodes.items()})


# fi = d.find_mult_rel_idxs({"in": [
#     "adc51d28318a7219b48182d6703e0053",     # Индекс узла "Опросник стройка". Мы ищем все вопросы, которые к нему относятся
#     "a9234fdfc01fd9ee2d11b54b8c641033",     # Добавляем новые узлы с входящей зависимостью "услуга"
# ]}, and_on=False)
# fis = d.sort(fi, {"recursive_relation": {
#     "in": [["8967ef8b318c2014d0214e2f045202e5", "prob"]]
# }})
# fis = {idx: val for idx, val in fis}
# fis2 = d.sort(fi, {"relation": {
#     "in": [["8967ef8b318c2014d0214e2f045202e5", "prob"]]
# }})
# fis = {**fis, **{idx: val for idx, val in fis2}}
# fis_sort = sorted([[d._net[idx].value, val] for idx, val in fis.items()], key=itemgetter(1), reverse=True)
# pprint(fis_sort)

# dist_nodes = d.find_shortest_dist_z(filter_idxs=fi)

# node_idx = DBESNode.to_idx("во6")
# sign = -1
# checked_nodes = set([node_idx])

# fsi = d.find_all_dir_rel_idxs_lvls(node_idx, dir_rel="out", filter_idxs=fi)

# for idx, lvl in fsi.items():
#     if idx not in checked_nodes:
#         fis[idx] += (1 - fis[node_idx]) / lvl * sign
#         checked_nodes.add(idx)

# bsi = d.find_all_dir_rel_idxs_lvls(node_idx, dir_rel="in", filter_idxs=fi)

# for idx, lvl in bsi.items():
#     if idx not in checked_nodes:
#         fis[idx] += (1 - fis[node_idx]) / lvl / 2 * sign
#         checked_nodes.add(idx)

# for idx, lvl in fsi.items():
#     bsi = d.find_all_dir_rel_idxs_lvls(idx, dir_rel="in", filter_idxs=fi)

#     for idx_back, lvl_back in bsi.items():
#         if idx_back not in checked_nodes:
#             fis[idx_back] += (1 - fis[node_idx]) / dist_nodes[node_idx][idx_back] / 3 * sign
#             checked_nodes.add(idx_back)

# for idx in fis.keys():
#     if idx not in checked_nodes:
#         if idx in dist_nodes[node_idx].keys():
#             fis[idx] += (1 - fis[node_idx]) / dist_nodes[node_idx][idx] / 5 * sign
#             checked_nodes.add(idx)
#         else:
#             fis[idx] -= fis[node_idx] / 5 * sign

# fis_sort = sorted([[d._net[idx].value, val] for idx, val in fis.items()], key=itemgetter(1), reverse=True)
# pprint(fis_sort)


# pgu = d.find_val_idxs("Портал госуслуг").pop()
# fi = d.find_rel_idxs_NORec(pgu)
# fi.add(pgu)

# start_time = time.time()
# dist_nodes = d.find_shortest_dist_z(filter_idxs=fi)
# print("Time find_shortest_dist_z: ", (time.time() - start_time))  # 0.00147

# start_time = time.time()
# dist_nodes2 = d.find_shortest_dist_dijkstra_amx(filter_idxs=fi)
# print("Time find_shortest_dist_dijkstra: ", (time.time() - start_time))     # 0.00156

# start_time = time.time()
# dist_nodes3 = d.find_shortest_dist_dijkstra_als(filter_idxs=fi)
# print("Time find_shortest_dist_dijkstra_als: ", (time.time() - start_time))       # 0.0029

# print(dist_nodes == dist_nodes2)        # True
# print(dist_nodes == dist_nodes3)        # True


# tu = d.find_val_idxs("Тег услуги").pop()
# fi = d.find_rel_idxs_NORec(tu)
# fi.add(tu)

# start_time = time.time()
# dist_nodes = d.find_shortest_dist_z(filter_idxs=fi)
# print("Time find_shortest_dist_z: ", (time.time() - start_time))  # 0.099

# start_time = time.time()
# dist_nodes2 = d.find_shortest_dist_dijkstra_amx(filter_idxs=fi)
# print("Time find_shortest_dist_dijkstra: ", (time.time() - start_time))     # 0.148

# start_time = time.time()
# dist_nodes3 = d.find_shortest_dist_dijkstra_als(filter_idxs=fi)
# print("Time find_shortest_dist_dijkstra_als: ", (time.time() - start_time))       # 0.245

# start_time = time.time()
# dist_nodes4 = d.find_shortest_dist_DAG(filter_idxs=fi)
# print("Time find_shortest_dist_DAG: ", (time.time() - start_time))       # 0.006 - best

# start_time = time.time()
# dist_nodes5 = d.find_shortest_dist_fw(filter_idxs=fi)
# print("Time find_shortest_dist_fw: ", (time.time() - start_time))   # 0.129

# print(dist_nodes == dist_nodes2)          # True
# print(dist_nodes == dist_nodes3)          # True    or False if not fi.add(tu)
# print(dist_nodes == dist_nodes4)          # True
# print(dist_nodes == dist_nodes5)          # True


# gsppo = d.find_val_idxs("Государственная, социальная поддержка и пенсионное обеспечение").pop()

# a = sorted([[key, val] for key, val in dist_nodes[gsppo].items() if key in dist_nodes5[gsppo].keys() ], key=itemgetter(0) )
# b = sorted([[key, val] for key, val in dist_nodes5[gsppo].items()], key=itemgetter(0))

# print(a == b)
# pprint(a)
# pprint(b)

# for key, val in dist_nodes[gsppo].items():
#     if key not in dist_nodes5[gsppo].keys():
#         print(d._net[key].value)


pgu = d.find_val_idxs("Словарь").pop()
fi = d.find_rel_idxs_NORec(pgu)
print(fi)
start_time = time.time()
amx, _ = d.get_adjacency_matrix(filter_idxs=fi, type_amx="float", symetric=False)
print("Time get_adjacency_matrix: ", (time.time() - start_time))
# pprint(amx)

ldfw = d.find_longest_mult_dist_fw(
    filter_idxs=fi, type_amx="float", symetric=False, to_val=True
)

# pprint(ldfw)
