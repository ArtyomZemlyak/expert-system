import time
from pprint import pprint


import sys

sys.path.insert(0, "scripts")
from dbes.DBESNet import DBESNet
from dbes.DBESNode import DBESNode
from dbes.DBESGraphVisualizer import DBESGraphVisualizer

st = time.time()
d = DBESNet()
print("Time up: ", time.time() - st)


################################################################################################################################
# Визуализация узлов графа и их связей.
################################################################################################################################

important_nodes = {}
important_nodes["eid"] = d.find_val_idxs("eid").pop()
important_nodes["Услуга"] = d.find_val_idxs("Услуга").pop()
important_nodes["Тег услуги"] = d.find_val_idxs("Тег услуги").pop()
important_nodes["Параметр тега"] = d.find_val_idxs("Параметр тега").pop()
important_nodes["Параметр услуги"] = d.find_val_idxs("Параметр услуги").pop()
important_nodes["SERVICE"] = d.find_val_idxs("SERVICE").pop()
important_nodes["CONSULTATION"] = d.find_val_idxs("CONSULTATION").pop()
important_nodes["DEPERSONALIZED_CONSULTATION"] = d.find_val_idxs(
    "DEPERSONALIZED_CONSULTATION"
).pop()

fi = d.find_rel_idxs_NORec(important_nodes["Параметр тега"])
tag_params = {d._net[idx].value: idx for idx in fi}

fi = d.find_rel_idxs_NORec(important_nodes["Тег услуги"])

dbes_gv = DBESGraphVisualizer({idx: d._net[idx] for idx in fi})

html_str = dbes_gv.visualize()

print(html_str[:100])
