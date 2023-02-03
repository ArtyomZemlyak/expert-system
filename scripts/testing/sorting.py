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
# Сортировка узлов по различным значениям или значениям отношений
################################################################################################################################
important_nodes = {}
important_nodes["id"] = d.find_val_idxs("id").pop()
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

sum_stats = sum(
    [
        d._net[i].value if d._net[i].value else 0
        for i in d.find_mult_rel_idxs(
            {
                "in": [
                    important_nodes["SERVICE"],
                    important_nodes["CONSULTATION"],
                    important_nodes["DEPERSONALIZED_CONSULTATION"],
                ]
            },
            and_on=False,
        )
    ]
)

# tu = d.find_val_idxs("Портал госуслуг").pop()
# fi = d.find_rel_idxs_NORec(tu)
# fi.add(tu)

# start_time = time.time()
# fi_sorted = d.sort(fi, { "value_recursive_relation": { "in": [important_nodes["SERVICE"]] } }, lambda x: x / sum_stats)
# print("Time sort: ", (time.time() - start_time))

# pprint([[d._net[idx].value, val] for idx, val in fi_sorted])


# tu = d.find_val_idxs("Имущественные и земельные отношения, строительство").pop()
# fi = d.find_rel_idxs_NORec(tu)
# fi.add(tu)

# start_time = time.time()
# fi_sorted = d.sort(fi, { "value_recursive_relation": { "in": [important_nodes["SERVICE"]] } }, lambda x: x / sum_stats)
# print("Time sort: ", (time.time() - start_time))

# pprint([[d._net[idx].value, val] for idx, val in fi_sorted])
# """
# [['Имущественные и земельные отношения, строительство', 0.1895234740602828],
#  ['Приобретение недвижимого имущества', 0.09234639963769939],
#  ['Регистрация недвижимого имущества', 0.0886967241986615],
#  ['Выдача разрешений, согласований, ордеров и пр.', 0.00847934383334172]]
#  """


tu = d.find_val_idxs("Тег услуги").pop()
fi = d.find_rel_idxs_NORec(important_nodes["Тег услуги"])
# fi.add(tu)

start_time = time.time()
fi_sorted = d.sort(
    fi,
    {"value_recursive_relation": {"in": [important_nodes["SERVICE"]]}},
    lambda x: x / sum_stats,
)
print("Time sort: ", (time.time() - start_time))

# pprint([[d._net[idx].value, val] for idx, val in fi_sorted])
