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
# Тест вариативного опросника
################################################################################################################################
# val_dict = {
#     DBESNode.to_idx('у1'): 'у1',
#     DBESNode.to_idx('у2'): 'у2',
#     DBESNode.to_idx('у3'): 'у3',
#     DBESNode.to_idx('во1'): 'во1',
#     DBESNode.to_idx('во2'): 'во2',
#     DBESNode.to_idx('во3'): 'во3',
#     DBESNode.to_idx('во4'): 'во4',
#     DBESNode.to_idx('во5'): 'во5',
#     DBESNode.to_idx('во6'): 'во6',
#     DBESNode.to_idx('во7'): 'во7',
#     DBESNode.to_idx('во8'): 'во8',
#     DBESNode.to_idx('во9'): 'во9',
#     DBESNode.to_idx('во10'): 'во10',
#     DBESNode.to_idx('во11'): 'во11',
#     DBESNode.to_idx('Тест вариативный опросник'): 'Тест вариативный опросник',
# }
# rel_dict = {
#     DBESNode.to_idx('Тест вариативный опросник'): {
#         DBESNode.to_idx('во1'): {},
#         DBESNode.to_idx('во2'): {},
#         DBESNode.to_idx('во3'): {},
#         DBESNode.to_idx('во4'): {},
#         DBESNode.to_idx('во5'): {},
#         DBESNode.to_idx('во6'): {},
#         DBESNode.to_idx('во7'): {},
#         DBESNode.to_idx('во8'): {},
#         DBESNode.to_idx('во9'): {},
#         DBESNode.to_idx('во10'): {},
#         DBESNode.to_idx('во11'): {},
#     },
#     DBESNode.to_idx('у1'): {
#         "@inlink": {DBESNode.to_idx('во8'): "depends_and"}
#     },
#     DBESNode.to_idx('во8'): {
#         "@inlink": {DBESNode.to_idx('во4'): "depends_or", DBESNode.to_idx('во5'): "depends_or"}
#     },
#     DBESNode.to_idx('во4'): {
#         "@inlink": {DBESNode.to_idx('во1'): "depends_and"}
#     },
#     DBESNode.to_idx('во5'): {
#         "@inlink": {DBESNode.to_idx('во2'): "depends_and"}
#     },
#     DBESNode.to_idx('у2'): {
#         "@inlink": {DBESNode.to_idx('во9'): "depends_and", DBESNode.to_idx('во10'): "depends_and"}
#     },
#     DBESNode.to_idx('во10'): {
#         "@inlink": {DBESNode.to_idx('во6'): "depends_and", DBESNode.to_idx('во7'): "depends_and"}
#     },
#     DBESNode.to_idx('во6'): {
#         "@inlink": {DBESNode.to_idx('во3'): "depends_and"}
#     },
#     DBESNode.to_idx('у3'): {
#         "@inlink": {DBESNode.to_idx('во11'): "depends_and"}
#     },
# }
# d.import_from_sep_json(val_dict, rel_dict)
# val_dict = {
#     DBESNode.to_idx('Статистический показатель'): 'Статистический показатель',
# }
# rel_dict = {
#     DBESNode.to_idx('у1'): {
#         "@inlink": {DBESNode.to_idx('Статистический показатель'): {"prob": 0.6}}
#     },
#     DBESNode.to_idx('у2'): {
#         "@inlink": {DBESNode.to_idx('Статистический показатель'): {"prob": 0.1}}
#     },
#     DBESNode.to_idx('у3'): {
#         "@inlink": {DBESNode.to_idx('Статистический показатель'): {"prob": 0.3}}
#     },
# }
# d.import_from_sep_json(val_dict, rel_dict)

# rel_dict = {
#     DBESNode.to_idx('во9'): {
#         "@inlink": {DBESNode.to_idx('во5'): "depends_and"}
#     },
# }
# d.import_from_sep_json(path_rel=rel_dict)
