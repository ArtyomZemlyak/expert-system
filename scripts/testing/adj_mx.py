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
# Adjacency matrix
################################################################################################################################

# st = time.time()
# amx, _ = d.get_adjacency_matrix()
# print("Time get_adjacency_matrix: ", time.time() - st)

# pgu = d.find_val_idxs("Портал госуслуг").pop()
# fi = d.find_rel_idxs_NORec(pgu)

# st = time.time()
# amx, _ = d.get_adjacency_matrix(filter_idxs=fi)
# print(f"Time get_adjacency_matrix(len={len(fi)}): ", time.time() - st)


pgu = d.find_val_idxs("Услуга").pop()
fi = d.find_rel_idxs_NORec(pgu)
pgu = d.find_val_idxs("Тег услуги").pop()
fi.update(d.find_rel_idxs_NORec(pgu))

start_time = time.time()
dist_nodes = d.find_shortest_dist_z(
    filter_idxs=fi,
)
print("Time find_shortest_dist_z: ", (time.time() - start_time))

# start_time = time.time()
# dist_nodes = d.find_shortest_dist_dijkstra_amx(
#     filter_idxs=fi,
# )
# print("Time find_shortest_dist_dijkstra_amx: ", (time.time() - start_time))


# pgu = d.find_val_idxs("Словарь").pop()
# fi = d.find_rel_idxs_NORec(pgu)
# print(fi)
start_time = time.time()
amx, _ = d.get_adjacency_matrix(filter_idxs=fi, type_amx="float", symetric=False)
print("Time get_adjacency_matrix: ", (time.time() - start_time))
pprint(amx)

################################################################################################################################
# Adjacency list
################################################################################################################################

st = time.time()
als, _ = d.get_adjacency_list()
print("Time get_adjacency_list: ", time.time() - st)

# pgu = d.find_val_idxs("Портал госуслуг").pop()
# fi = d.find_rel_idxs_NORec(pgu)
# fi.add(pgu)

st = time.time()
aaa = d.get_adjacency_list(filter_idxs=fi)
print(f"Time get_adjacency_list(len={len(fi)}): ", time.time() - st)
# pprint(aaa)
# print(len(als))
"""
Time get_adjacency_list:  0.07283163070678711
Time get_adjacency_list(len=8):  5.7220458984375e-05
(defaultdict(<class 'list'>,
             {0: [[2, 1]],
              1: [[2, 1]],
              2: [[3, 1], [6, 1], [5, 1], [1, 1], [4, 1], [7, 1], [0, 1]],
              3: [[2, 1]],
              4: [[2, 1]],
              5: [[2, 1]],
              6: [[2, 1]],
              7: [[2, 1]]}),
 {'130c7c31318f86eba2b1f9c2902e1682': 0,
  '4250d6e9348d5d5c7e4ff44efd6c8f97': 2,
  '47bc897795b659f44465488de1852625': 7,
  '6620d223a250de770b5a5017b3acb96d': 4,
  '74c72bc697e7a9ec5634cb6d43a2c6e6': 1,
  'ac4b6234a271aab60d134dda46a5bed5': 5,
  'c0141337cb5cb6d534bc7938b1f13a77': 6,
  'd78497174dc17113b66aa6290441d624': 3})
"""


start_time = time.time()
dist_nodes1 = d.find_shortest_dist_dijkstra_als(filter_idxs=fi)
print("Time find_shortest_dist_dijkstra_als: ", (time.time() - start_time))

start_time = time.time()
dist_nodes2 = d.find_shortest_dist_DAG(filter_idxs=fi)
print("Time find_shortest_dist_DAG: ", (time.time() - start_time))

start_time = time.time()
dist_nodes2 = d.find_shortest_dist_fw(filter_idxs=fi)
print("Time find_shortest_dist_fw: ", (time.time() - start_time))


# dc = {'mystr{}'.format(i): i for i in range(30)}

# start_time = time.time()
# for i in range(1000):
#     dc['mystr25']
# print("t: ", (time.time() - start_time))

# dc = {sys.intern('mystr{}'.format(i)): i for i in range(30)}

# start_time = time.time()
# for i in range(1000):
#     dc['mystr25']
# print("t: ", (time.time() - start_time))


"""
Услуги и Теги:
Time find_shortest_dist_z:  13.37240195274353
Time find_shortest_dist_dijkstra_amx: around 30 min or 1 hour
Time get_adjacency_matrix:  0.014950275421142578
Time get_adjacency_list(len=1870):  0.007077455520629883
Time find_shortest_dist_dijkstra_als:  253.19896006584167
Time find_shortest_dist_DAG:  1.5844831466674805
Time find_shortest_dist_fw:  1300.819284439087
"""
