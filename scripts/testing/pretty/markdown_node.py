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
# markdown печать данных узла
################################################################################################################################

pgu = d.find_val_idxs("Портал госуслуг").pop()

node_data = d._net[pgu].toJSON()

pprint(node_data)


def pretty(d, indent=0):
    return_str = ""
    for key, value in d.items():
        return_str += "  " * indent + str(key) + "\n"
        if isinstance(value, dict):
            return_str += pretty(value, indent + 1) + "\n"
        else:
            return_str += "  " * (indent + 1) + str(value) + "\n\n"
    return return_str


r_str = ""
r_str = pretty(node_data)

print(r_str)
