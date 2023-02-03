# distutils: language=c++
# cython: language_level=3

from collections import defaultdict
from operator import itemgetter
import sys
import os
from os import listdir, times
from os.path import isfile, join, exists
import uuid

import zipfile
import pickle

from typing import Any, Iterable, List, Tuple, Union
from pprint import pprint
import json
import re
import csv
import pathlib
from functools import wraps

import numpy as np


from dbes.DBESNode import DBESNode
from dbes.DBESTemplate import DBESTemplate, DBESNodeJsonEncoder
from dbes.DBESPorstgresAdapter import DBESPostgresAdapter


##############################################
#### CONFIG ##################################
##############################################
path_cfg = join(pathlib.Path(__file__).parent.parent.resolve(), 'config.json')
CONFIG = json.loads(open(path_cfg, "r").read())
##############################################
##############################################


def caching_profile(function):
    """
    For cache user profiles (args and return values).
    """
    function.caching_profile = True

    @wraps(function)
    def wrapper(self, *args, **kwargs):
        if self._cache:
            profile_name = kwargs.get('_profile')

            if profile_name:
                if profile_name in self._cache.keys():
                    func_name = function.__name__

                    if func_name in self._cache[profile_name].keys():
                        runtype = kwargs.get('_runtype')

                        if not runtype and "return" in self._cache[profile_name][func_name] or runtype == "return":
                            return self._cache[profile_name][func_name]["return"]
                        elif "args" in self._cache[profile_name][func_name] or runtype == "args":
                            return function(self, *args, **self._cache[profile_name][func_name]["args"], **kwargs)

                        else: raise ValueError("Cant find appropriate runtype for function!")
                else:
                    runtype = kwargs.get('_runtype')

                    if runtype:
                        func_name = function.__name__

                        if runtype == "args":
                            if args: raise ValueError("Cant convert *args to **kwargs - please use attribute=value in functions!")
                            self.add_profile(profile_name, func_name, func_args=kwargs)

                        elif runtype == "return":
                            returned = function(self, *args, **kwargs)
                            self.add_profile(profile_name, func_name, func_return=returned)
                            return returned

                    else: raise ValueError("Setted profile_name non exist! For adding profile for DBES use add_profile func or specify _runtype parameter!")

            else: return function(self, *args, **kwargs)
        else: return function(self, *args, **kwargs)
    return wrapper


def caching_cleaner(function):
    """
    For functions, that in any way modify DBESNet._net data.
    """
    function.caching_cleaner = True

    @wraps(function)
    def wrapper(self, *args, **kwargs):
        if self._cache:
            parent = sys._getframe().f_back.f_code.co_name

            if parent not in self._cc_methods:
                # parent not in self._cc_methods - clear cache with appropriate type for each profile
                returned = function(self, *args, **kwargs)

                for profile_name in list(self._cache.keys()):
                    for func_name in list(self._cache[profile_name].keys()):
                        type_caching = self._cache[profile_name][func_name]['type_caching']

                        if type_caching == 'delete':
                            del self._cache[profile_name][func_name]

                        elif type_caching == 'recompute':
                            try:
                                func_call = getattr(self, func_name)
                                self._cache[profile_name][func_name]['return'] = func_call(**self._cache[profile_name][func_name]['args'])
                            except Exception as e:
                                del self._cache[profile_name][func_name]

                    if self._cache[profile_name] == {}:
                        del self._cache[profile_name]
                        self._del_file(profile_name, join(self._CONFIG['save']['path_save'], 'caching'))

                for profile_name, profile_data in self._cache.items():
                    self._save_pickle(profile_name, profile_data, join(self._CONFIG['save']['path_save'], 'caching'))

                return returned
            else:
                # parent in self._cc_methods - do nothing
                return function(self, *args, **kwargs)
        else: return function(self, *args, **kwargs)
    return wrapper


class Heap:
    """
    For find_shortest_dist_dijkstra_als func.
    """

    def __init__(self):
        self.array = []
        self.size = 0
        self.pos = []

    def newMinHeapNode(self, v, dist):
        minHeapNode = [v, dist]
        return minHeapNode

    def swapMinHeapNode(self,a, b):
        t = self.array[a]
        self.array[a] = self.array[b]
        self.array[b] = t

    def minHeapify(self, idx):
        smallest = idx
        left = 2*idx + 1
        right = 2*idx + 2

        if left < self.size and self.array[left][1] < self.array[smallest][1]:
            smallest = left

        if right < self.size and self.array[right][1] < self.array[smallest][1]:
            smallest = right

        if smallest != idx:
            self.pos[ self.array[smallest][0]] = idx
            self.pos[ self.array[idx][0]] = smallest

            self.swapMinHeapNode(smallest, idx)
            self.minHeapify(smallest)

    def extractMin(self):
        if self.isEmpty() == True:
            return None

        root = self.array[0]

        lastNode = self.array[self.size - 1]
        self.array[0] = lastNode

        self.pos[lastNode[0]] = 0
        self.pos[root[0]] = self.size - 1

        self.size -= 1
        self.minHeapify(0)

        return root

    def isEmpty(self):
        return True if self.size == 0 else False

    def decreaseKey(self, v, dist):
        i = self.pos[v]

        self.array[i][1] = dist

        while i > 0 and self.array[i][1] < self.array[i - 1][1]:
            self.pos[ self.array[i][0] ] = i - 1
            self.pos[ self.array[i - 1][0] ] = i
            self.swapMinHeapNode(i, i - 1)

            i = i - 1

    def isInMinHeap(self, v):

        if self.pos[v] < self.size:
            return True
        return False


class DBESNet:

    def __init__(self, config: str = None) -> None:
        if config: self._CONFIG = config
        else: self._CONFIG = CONFIG

        self._net           = {}
        self._common        = {}
        self._templates     = {}
        self._cache         = {}

        net_path        = join(self._CONFIG['save']['path_save'], 'net')
        common_path     = join(self._CONFIG['save']['path_save'], 'common')
        templates_path  = join(self._CONFIG['save']['path_save'], 'templates')
        caching_path    = join(self._CONFIG['save']['path_save'], 'caching')

        pathlib.Path(net_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(common_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(templates_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(caching_path).mkdir(parents=True, exist_ok=True)

        ##### Net
        files_names = [f for f in listdir(net_path) if isfile(join(net_path, f))]

        for file_name in files_names:
            name = re.sub('[\\.]{1}[A-Za-z\\d]+', '', file_name)
            json_file_data = self._open_json(file_name, net_path)
            self._net[name] = DBESNode.from_dict(json_file_data)

        #### Common files
        for file, file_name in self._CONFIG['save']['common'].items():
            if '.json' in file_name:
                try:
                    self._common[file] = self._open_json(file_name, common_path, create_if_not_exist=True)
                except FileNotFoundError: pass

        ##### Templates
        files_names = [f for f in listdir(templates_path) if isfile(join(templates_path, f))]

        for file_name in files_names:
            name = re.sub('[\\.]{1}[A-Za-z\\d]+', '', file_name)
            json_file_data = self._open_json(file_name, templates_path)
            self._templates[name] = DBESTemplate.from_dict(json_file_data)

        #### Files of cached profiles
        files_names = [f for f in listdir(caching_path) if isfile(join(caching_path, f))]

        for file_name in files_names:
            self._cache[file_name] = self._open_pickle(file_name, caching_path)

        #### Postgres connection (for import from postgres tables)
        self.postgres = None
        try:
            self.postgres = DBESPostgresAdapter()
        except Exception as e:
            print("Cant init postgres adapter!")
            print(e)

        self._methods = self._get_methods()
        self._cp_methods = self._get_decorated_methods("caching_profile")
        self._cc_methods = self._get_decorated_methods("caching_cleaner")

    def _open_text(self, file_name: str, path_save: str=None, create_if_not_exist: bool=False) -> Union[str, None]:
        if not path_save: path_save = self.CONFIG['save']['path_save']
        if exists(join(path_save, file_name)):
            return open(join(path_save, file_name), "r", encoding='utf8').read()
        elif create_if_not_exist: open(join(path_save, file_name), "w").write("")
        else: return None

    def _open_csv(self, file_name: str, path_save: str=None, create_if_not_exist: bool=False, delimiter: str=',', quotechar: str='"') -> Union[csv.reader, None]:
        if not path_save: path_save = self.CONFIG['save']['path_save']
        if exists(join(path_save, file_name)):
            with open(join(path_save, file_name), newline='') as csv_file:
                return csv.reader(csv_file, delimiter=delimiter, quotechar=quotechar)
        elif create_if_not_exist: open(join(path_save, file_name), "w").write("")
        else: return None

    def _open_json(self, file_name: str, path_save: str=None, create_if_not_exist: bool=False) -> Union[dict, None]:
        if not path_save: path_save = self.CONFIG['save']['path_save']
        if exists(join(path_save, file_name)):
            return json.loads(open(join(path_save, file_name), "r", encoding='utf8').read())
        elif create_if_not_exist: open(join(path_save, file_name), "w").write(json.dumps({}))
        else: return None

    def _open_pickle(self, file_name: str, path_save: str=None) -> Union[Any, None]:
        if not path_save: path_save = self.CONFIG['save']['path_save']
        if exists(join(path_save, file_name)):
            with open(join(path_save, file_name), 'rb') as pickle_file:
                return pickle.load(pickle_file)
        else: return None

    def _save_json(self, file_name: str, file_data: dict, path_save: str=None) -> None:
        if not path_save: path_save = self._CONFIG['save']['path_save']
        with open(join(path_save, file_name), "w", encoding='utf8') as json_file:
            json_file.write(json.dumps(file_data, indent=4, sort_keys=True, ensure_ascii=False, cls=DBESNodeJsonEncoder))
            json_file.close()

    def _save_pickle(self, file_name: str, file_data: Any, path_save: str=None) -> None:
        if not path_save: path_save = self._CONFIG['save']['path_save']
        with open(join(path_save, file_name), 'wb') as pickle_file:
            pickle.dump(file_data, pickle_file)
            pickle_file.close()

    def _del_file(self, file_name: str, path_save: str=None) -> None:
        if not path_save: path_save = self._CONFIG['save']['path_save']
        if exists(join(path_save, file_name)):
            os.remove(join(path_save, file_name))

    @caching_cleaner
    def _del_node(self, idx: str) -> None:
        del self._net[idx]
        self._del_file(self._get_file_name(idx),join(self._CONFIG['save']['path_save'], 'net'))

    @staticmethod
    def _get_file_name(file: str) -> str:
        return file + '.json'

    def _get_methods(self, class_instance: Any=None) -> set:
        non_magic_class = set([])
        class_methods = dir(self if not class_instance else class_instance)

        for m in class_methods:
            if m.startswith('__'): continue
            else: non_magic_class.add(m)

        return non_magic_class

    def _get_decorated_methods(self, decorator_tag: str="caching_profile", class_instance: Any=None) -> set:
        class_instance = self if not class_instance else class_instance

        return {name
            # get all attributes, including methods, properties, and builtins
            for name in dir(class_instance)
                # but we only want methods
                if callable(getattr(class_instance, name))
                # and we don't need builtins
                and not name.startswith("__")
                # and we only want the cool methods
                and hasattr(getattr(class_instance, name), decorator_tag)}

    def _update_config(self) -> None:
        self._save_json('config.json', self._CONFIG, 'scripts')

    def _update_pattern_xor(self, pattern: dict, filter_idxs: set=None) -> dict:
        depends_xor_indxs = self.find_type_rel_idxs('depends_xor', filter_idxs=filter_idxs)

        for node_idx_ in depends_xor_indxs:
            flag_break_ = False
            can_add_off_ = True
            add_off_ = False

            for node_rel_idx, rel_idxs in self._net[node_idx_].relation['in'].items():
                if flag_break_: break

                for rel_idx in rel_idxs.keys():
                    if 'depends_xor' in rel_idx:
                        if node_rel_idx in pattern.keys():
                            if pattern[node_rel_idx] == 'on':
                                pattern[node_idx_] = 'on'
                                flag_break_ = True
                                can_add_off_ = False
                                add_off_ = False
                                break

                            elif can_add_off_ and pattern[node_rel_idx] == 'off':
                                add_off_ = True
                                break

                        else:
                            can_add_off_ = False
                            add_off_ = False

            if add_off_: pattern[node_idx_] = 'off'

        return pattern

    @staticmethod
    def _zip_db(folder_path, path_save) -> bool:
        with zipfile.ZipFile(path_save, mode='w') as zipf:
            for dirpath, subdirs, files in os.walk(folder_path):
                dirname = dirpath.split('/')[-1]
                if dirname != 'dbes':
                    zipf.write(dirpath, arcname=dirname)

                    for filename in files:
                        zipf.write(os.path.join(dirpath, filename), arcname=os.path.join(dirname, filename))

        return True

    @staticmethod
    def _import_from_zip(path_zip_file: str, path_save: str) -> bool:
        with zipfile.ZipFile(path_zip_file, 'r') as zip_ref:
            zip_ref.extractall(path_save)
        return True

    @staticmethod
    def _replace_idx_from(some_replace_idxs, some_idx, some_column_indx, raise_err=False):  #################################################### что если нету индекса? - фильровать индексы изначально?
        if some_column_indx in some_replace_idxs.keys() or "all" in some_replace_idxs.keys():
            for replace_col_idx, replace_table in some_replace_idxs.items():
                if some_column_indx == replace_col_idx or replace_col_idx == "all":
                    for table_name, prev_idx_to_idx in replace_table.items():
                        if some_idx in prev_idx_to_idx.keys():
                            return prev_idx_to_idx[some_idx]
            if raise_err: raise KeyError("Cant find idx to replace!")
        else: return some_idx

    def _replace_idx_from_val_rel(self, some_replace_idxs, some_idx, some_column_indx, raise_err=False):  #################################################### что если нету индекса? - фильровать индексы изначально?
        if some_column_indx in some_replace_idxs or "all" in some_replace_idxs:
            val_rel_idxs = self.find_val_idxs(some_idx)
            if len(val_rel_idxs) == 1:
                val_rel_idx = val_rel_idxs.pop()
                return list(self._net[val_rel_idx].relation['out'].keys())[0]
            if raise_err: raise KeyError("Cant find idx to replace!")
        else: return some_idx

    @staticmethod
    def _add_some_relation(some_rels, some_relation, some_idx):
        dir_rel = f"@{some_relation['dir']}link"
        if some_relation["idx"] not in some_rels.keys():
            some_rels[some_relation["idx"]] = {dir_rel: {some_idx: some_relation["type"]}}
        elif dir_rel not in some_rels[some_relation["idx"]].keys():
            some_rels[some_relation["idx"]][dir_rel] = {some_idx: some_relation["type"]}
        else: some_rels[some_relation["idx"]][dir_rel][some_idx] = some_relation["type"]

    @staticmethod
    def _add_and_combine(some_idx, some_coeff, some_combined_idxs_rel_coef, some_combine="+"):
        if some_idx not in some_combined_idxs_rel_coef.keys():
            some_combined_idxs_rel_coef[some_idx] = some_coeff
        else:
            if      some_combine == '+': some_combined_idxs_rel_coef[some_idx] += some_coeff
            elif    some_combine == '-': some_combined_idxs_rel_coef[some_idx] -= some_coeff
            elif    some_combine == '*': some_combined_idxs_rel_coef[some_idx] *= some_coeff
            elif    some_combine == '/': some_combined_idxs_rel_coef[some_idx] /= some_coeff

    @staticmethod
    def _struct_idxs_to_val(self, struct_dict):
        return {
            lvl: {
                self._net[lvl_idx].value: {
                    self._net[lvl_rel_idx].value for lvl_rel_idx in lvl_rel_idxs.keys()
                } for lvl_idx, lvl_rel_idxs in lvl_idxs.items()
            } for lvl, lvl_idxs in struct_dict.items()
        }

    @caching_profile
    def get_adjacency_matrix(self, data_dict: dict=None, find_idxs: set=None, filter_idxs: set=None, replace_zero: int=None, symetric: bool=True, **kwargs) -> Tuple[np.uint64, dict]:
        if not data_dict: data_dict = self._net

        net_idx_to_amx_idx = {}
        for i, node_idx in enumerate(data_dict.keys() if not filter_idxs else filter_idxs):
            net_idx_to_amx_idx[node_idx] = i

        len_data = len(data_dict.keys() if not filter_idxs else filter_idxs)
        shape_amx = (len_data, len_data)

        if replace_zero == None:
            amx = np.zeros(shape_amx, dtype=np.uint64)
        else:
            amx = np.asarray([[replace_zero for i in range(len_data)] for j in range(len_data)], dtype=np.uint64)
            for i in range(len_data):
                amx[i][i] = 0

        for node_idx in data_dict.keys() if not find_idxs and not filter_idxs else find_idxs if find_idxs else filter_idxs:
            for dir_rel in ['in', 'out'] if symetric else ['out']:
                for other_idx, relations in data_dict[node_idx].relation[dir_rel].items():
                    if not filter_idxs and other_idx in data_dict.keys() or filter_idxs and other_idx in filter_idxs:
                        for idx_rel, count_rel in relations.items():
                            amx[net_idx_to_amx_idx[node_idx]][net_idx_to_amx_idx[other_idx]] = count_rel

        return (amx, net_idx_to_amx_idx)

    @caching_profile
    def get_adjacency_list(self, data_dict: dict=None, find_idxs: set=None, filter_idxs: set=None, symetric: bool=True, **kwargs) ->  Tuple[dict, dict]:
        als = defaultdict(list)

        if not data_dict: data_dict = self._net

        net_idx_to_als_idx = {}
        for i, node_idx in enumerate(data_dict.keys() if not filter_idxs else filter_idxs):
            net_idx_to_als_idx[node_idx] = i

        for node_idx in data_dict.keys() if not find_idxs and not filter_idxs else find_idxs if find_idxs else filter_idxs:
            als[net_idx_to_als_idx[node_idx]] = []
            for dir_rel in ['in', 'out'] if symetric else ['out']:
                for other_idx, relations in data_dict[node_idx].relation[dir_rel].items():
                    if not filter_idxs and other_idx in data_dict.keys() or filter_idxs and other_idx in filter_idxs:
                        for idx_rel, count_rel in relations.items():
                            als[net_idx_to_als_idx[node_idx]].insert(0, [net_idx_to_als_idx[other_idx], count_rel])

        return (als, net_idx_to_als_idx)

    def get_net(self, all_data: bool=False) -> Union[dict, set]:
        return {node_idx: node_data.value if not all_data else node_data.toJSON() for node_idx, node_data in self._net.items()}

    def get_net_file(self, idx: str, to_idx: bool=False) -> Union[dict, None]:
        if to_idx: idx = DBESNode.to_idx(idx)
        if idx in self._net.keys():
            return self._net[idx].toJSON()
        else: return None

    def get_common_file(self, file: str) -> Union[dict, None]:
        if file in self._common.keys():
            return self._common[file]
        else: return None

    def get_template_file(self, idx: str, to_idx: bool=False) -> Union[dict, None]:
        if to_idx: idx = DBESNode.to_idx(idx)
        if idx in self._templates.keys():
            return self._templates[idx].toJSON()
        else: return None

    def get_value_rel(self, node_idx: str, value_idx: str) -> Union[str, int, float, None]:
        value_idx = self.find_rel_idxs_NORec(value_idx, dir_rel="in", filter_idxs=set(self._net[node_idx].relation['in'].keys()))
        if len(value_idx) == 1:
            return self._net[value_idx.pop()].value
        else: return None

    def get_sum_of_values(self, find_idxs: set) -> Union[str, int, float, None]:
        return sum([self._net[i].value if self._net[i].value else 0 for i in find_idxs])

    @caching_cleaner
    def remove_net_file(self, idx: str, del_on_disk: bool=True) -> bool:
        file_name = self._get_file_name(idx)

        if idx in self._net.keys():
            del self._net[idx]

            if del_on_disk:
                self._del_file(file_name, join(self._CONFIG['save']['path_save'], 'net'))

        else: return False
        return True

    def remove_common_file(self, file: str, del_on_disk: bool=True) -> bool:
        file_name = self._get_file_name(file)

        if file in self._common.keys():
            del self._common[file]

            if del_on_disk:
                del self._CONFIG['save']['common'][file]
                self._update_config()
                self._del_file(file_name, join(self._CONFIG['save']['path_save'], 'common'))

        else: return False
        return True

    def remove_template_file(self, idx: str, del_on_disk: bool=True) -> bool:
        file_name = self._get_file_name(idx)

        if idx in self._templates.keys():
            del self._templates[idx]

            for node_rel_idx in [i for i in self._common['templates_rel'].keys()]:
                if idx in self._common['templates_rel'][node_rel_idx].keys():
                    del self._common['templates_rel'][node_rel_idx][idx]
                if self._common['templates_rel'][node_rel_idx] == {}:
                    del self._common['templates_rel'][node_rel_idx]

            if del_on_disk:
                self._del_file(file_name, join(self._CONFIG['save']['path_save'], 'templates'))
                self._save_json(self._CONFIG['save']['common']['templates_rel'], self._common['templates_rel'], join(self._CONFIG['save']['path_save'], 'common'))

        else: return False
        return True

    @caching_cleaner
    def remove_node_idx(self, idx: str, clear: bool=False, clear_type_rel: dict=None, recursive: bool=False) -> bool:
        clear_type_rel = clear_type_rel if clear_type_rel else {}

        for dir_rel, rev_dir_rel in zip(['in', 'out'], ['out', 'in']):
            for node_rel_idx in self._net[idx].relation[dir_rel].keys():
                del self._net[node_rel_idx].relation[rev_dir_rel][idx]
                self._save_json(self._get_file_name(node_rel_idx), self._net[node_rel_idx], join(self._CONFIG['save']['path_save'], 'net'))
                if recursive: self.remove_node_idx(node_rel_idx, recursive=True)

                elif clear:
                    if self._net[node_rel_idx].relation[rev_dir_rel] == {} and \
                       self._net[node_rel_idx].relation[dir_rel] == {}:
                        self._del_node(node_rel_idx)

                    elif clear_type_rel != {}:
                        for dir_rel_ in ['in', 'out', 'any']:
                            if dir_rel_ in clear_type_rel.keys():
                                clear_type_rel[dir_rel_] = set(clear_type_rel[dir_rel_])

                        for type_rel in self._net[idx].relation[dir_rel][node_rel_idx].keys():
                            if (dir_rel in clear_type_rel.keys() and type_rel in clear_type_rel[dir_rel]) or \
                                ('any' in clear_type_rel.keys() and type_rel in clear_type_rel['any']):
                                self.remove_node_idx(node_rel_idx)
                                break

        if idx in self._common['templates_rel'].keys():
            for temlate_idx, node_rel_idxs in [(i,b) for (i,b) in self._common['templates_rel'][idx].items()]:
                for node_rel_idx in node_rel_idxs.keys():
                    for dir_rel in ['in', 'out']:
                        if idx in self._templates[temlate_idx].template[node_rel_idx]['relation'][dir_rel].keys():
                            del self._templates[temlate_idx].template[node_rel_idx]['relation'][dir_rel][idx]

                self.add_template_file(temlate_idx, self._templates[temlate_idx])

        self._del_node(idx)

        return True

    @caching_cleaner
    def remove_rel_between(self, in_node_idx: str, out_node_idx: str, rel_idx: Union[set, list, str]=None, save: bool=False) -> bool:
        if type(rel_idx) == list or type(rel_idx) == set:
            for rel_idx_ in rel_idx:
                self.remove_rel_between(in_node_idx, out_node_idx, rel_idx_)

        elif in_node_idx in self._net.keys() and out_node_idx in self._net.keys():
            self._net[in_node_idx].remove_relation(out_node_idx, rel_idx, type_rel='in')
            self._net[out_node_idx].remove_relation(in_node_idx, rel_idx, type_rel='out')

            if save:
                self.add_net_file(in_node_idx, self._net[in_node_idx])
                self.add_net_file(out_node_idx, self._net[out_node_idx])

            return True

        else: return False

    @caching_cleaner
    def add_net_file(self, idx: str, file_data: Union[DBESNode, dict], rewrite: bool=True, write_on_disk: bool=True) -> bool:
        file_name = self._get_file_name(idx)

        if idx not in self._net.keys() or rewrite:
            if type(file_data) == dict:
                self._net[idx] = DBESNode.from_dict(file_data)
            elif type(file_data) == DBESNode:
                self._net[idx] = file_data
            else: raise TypeError('DBESNet.add_net_file can get only dict or DBESNode for file_data argument')

            if write_on_disk:
                self._save_json(file_name, file_data, join(self._CONFIG['save']['path_save'], 'net'))

            return True
        else: return False

    def add_common_file(self, file: str, file_data: dict, rewrite: bool=True, write_on_disk: bool=True) -> bool:
        file_name = self._get_file_name(file)

        if file not in self._common.keys() or rewrite:
            self._common[file] = file_data

            if write_on_disk:
                self._CONFIG['save']['common'][file] = file_name
                self._update_config()
                self._save_json(file_name, file_data, join(self._CONFIG['save']['path_save'], 'common'))

        return True

    def add_template_file(self, idx: str, file_data: Union[DBESTemplate, dict], rewrite: bool=True, write_on_disk: bool=True) -> bool:
        file_name = self._get_file_name(idx)

        if idx not in self._templates.keys() or rewrite:
            if idx in self._templates.keys():
                self.remove_template_file(idx)

            if type(file_data) == dict:
                self._templates[idx] = DBESTemplate.from_dict(file_data)
            elif type(file_data) == DBESTemplate:
                self._templates[idx] = file_data
            else: raise TypeError('DBESNet.add_template_file can get only dict or DBESTemplate for file_data argument')

            for node_idx, node_dict in file_data.items():
                for dir_rel in ['in', 'out']:
                    for node_rel_idx in node_dict['relation'][dir_rel].keys():
                        if "@gen" not in node_rel_idx:
                            if node_rel_idx not in self._common['templates_rel'].keys():
                                self._common['templates_rel'][node_rel_idx] = {idx: {node_idx: {}}}
                            elif idx not in self._common['templates_rel'][node_rel_idx].keys():
                                self._common['templates_rel'][node_rel_idx][idx] = {node_idx: {}}
                            else:
                                self._common['templates_rel'][node_rel_idx][idx][node_idx] = {}

            if write_on_disk:
                self._save_json(file_name, file_data, join(self._CONFIG['save']['path_save'], 'templates'))
                self._save_json(self._CONFIG['save']['common']['templates_rel'], self._common['templates_rel'], join(self._CONFIG['save']['path_save'], 'common'))

            return True
        else: return False

    def add_profile(self, profile_name: str, func_name: str, func_args: dict=None, func_return: Any=None, type_caching: str="delete", rewrite: bool=True, write_on_disk: bool=True) -> bool:
        """
        Add profile for function allows for pre-defined arguments and pre-computed return results, when function called with `_profile`="Name_Profile" argument.
        Caching for this functions have different types of behavior - check `type_caching` parameter.

        Args
        ----------
            `profile_name` : Unique profile name.

            `func_name` : Function name of DBESNet with `@caching_profile` support.

            `func_args` : Places this arguments into the function, when she called with `_profile`="Name_Profile" argument. Run this even if there is `func_return` in the profile if `_runtype`=="args"

            `func_return` : Return this, when function called with `_profile`="Name_Profile" argument. Will launch this option even if there is `func_args` in the profile (but if `_runtype`!="args").

            `type_caching` : Specifies what happens to the profile cache when data changes.
                "delete" : Delete permanently this profile, when any of data change occures.
                "recompute" : Recompute return values (using `func_args` parameter). Delete if cant recompute (if some important data deleted).
                "none" : Nothing happend.

        Return
        ----------
            `bool` : True - if adding successful. Else - error.
        """
        if not func_args and not func_return:
            raise ValueError("Need specify func_args or func_return parameter!")

        if profile_name in self._cache.keys() and func_name in self._cache[profile_name].keys() and not rewrite:
            raise ValueError("Setted profile_name already exist for this func_name! Choose another name for profile or set rewrite=True.")

        if func_name not in self._methods:
            raise ValueError("Setted func_name non exist! Choose from existing functions!")

        if func_name not in self._cp_methods:
            raise ValueError("Setted func_name exist, but cant support caching_profile! Choose from another existing functions, which supported caching_profile feature!")

        if type_caching == "recompute" and not func_args:
            raise ValueError("Setted type_caching = recompute -> needed func_args setting too!")

        new_profile = {func_name: {"type_caching": type_caching}}

        if func_args: new_profile[func_name]['args'] = func_args
        if func_return: new_profile[func_name]['return'] = func_return

        self._cache[profile_name] = new_profile

        if write_on_disk:
            self._save_pickle(profile_name, new_profile, join(self._CONFIG['save']['path_save'], 'caching'))

        return True

    @caching_cleaner
    def add_node(self, init_node: Union[DBESNode, dict, str]=None, idx: str=None, save: bool=True) -> None:
        if init_node:
            if type(init_node) == dict:
                node = DBESNode.from_dict(init_node)
            if type(init_node) == str:
                node = DBESNode(init_node)
            if type(init_node) == DBESNode:
                node = init_node
        if idx: node = DBESNode(idx=idx)

        if init_node or idx:
            self._net[node.idx] = node
            if save: self.add_net_file(node.idx, self._net[node.idx])

    def add_relation(self, in_node_idx: str, out_node_idx: str, rel_idx: Union[dict, str], coeff: Union[str, int, float]=None) -> bool:
        if type(rel_idx) == dict:
            for rel_idx_, coeff_ in rel_idx.items():
                return self.add_relation(in_node_idx, out_node_idx, rel_idx_, coeff_)

        elif in_node_idx in self._net.keys() and out_node_idx in self._net.keys():
            self._net[in_node_idx].add_relation(out_node_idx, rel_idx, coeff, type_rel='in')
            self._net[out_node_idx].add_relation(in_node_idx, rel_idx, coeff, type_rel='out')
            return True

        else: return False

    def add_relation_dict(self, dict_data: dict, in_node_idx: str, out_node_idx: str, rel_idx: Union[dict, str], coeff: Union[str, int, float]=None) -> dict:
        if type(rel_idx) == dict:
            for rel_idx_, coeff_ in rel_idx.items():
                dict_data = self.add_relation_dict(dict_data, in_node_idx, out_node_idx, rel_idx_, coeff_)

        elif in_node_idx in dict_data.keys() and out_node_idx in dict_data.keys():
            dict_data[in_node_idx].add_relation(out_node_idx, rel_idx, coeff, type_rel='in')
            dict_data[out_node_idx].add_relation(in_node_idx, rel_idx, coeff, type_rel='out')

        return dict_data

    def add_nodes_from_dict(self, nodes_dict: dict) -> set:
        save_nodes = set([])

        for node_idx, node_value in nodes_dict.items():
            if "@@to_idx" in node_idx:
                node_value_ = node_idx[8:]
                node_idx = DBESNode.to_idx(node_value_)

                if node_value == "" or node_value == {} or not node_value:
                    node_value = node_value_

            if "@@gen" in node_idx:
                if node_value == "" or node_value == {} or not node_value:
                    if len(node_idx) > 5:
                        node_value = node_idx[5:]
                    else: raise ValueError("Need specify value for node! Or write @@ganSOME_VALUE_HERE")

                node_idx = DBESNode.to_idx(DBESNode.generate_idx())

            if node_idx not in self._net.keys():
                self.add_node({'idx': node_idx, 'value': node_value}, save=False)
                save_nodes.add(node_idx)

            else:
                self._net[node_idx].update_value(node_value)
                save_nodes.add(node_idx)

        return save_nodes

    def add_relations_from_dict(self, nodes_dict: dict) -> set:
        save_nodes = set([])

        for node_one_idx, nodes_two in nodes_dict.items():
            if "@@to_idx" in node_one_idx:
                node_one_idx = DBESNode.to_idx(node_one_idx[8:])

            for node_two_idx, nodes_three in nodes_two.items():
                if "@@to_idx" in node_two_idx:
                    node_two_idx = DBESNode.to_idx(node_two_idx[8:])

                if node_two_idx == '@inlink':
                    for out_node_idx, type_rel in nodes_three.items():
                        if "@@to_idx" in out_node_idx:
                            out_node_idx = DBESNode.to_idx(out_node_idx[8:])
                        if self.add_relation(node_one_idx, out_node_idx, type_rel):
                            save_nodes.update([node_one_idx, out_node_idx])

                elif node_two_idx == '@outlink':
                    for in_node_idx, type_rel in nodes_three.items():
                        if "@@to_idx" in in_node_idx:
                            in_node_idx = DBESNode.to_idx(in_node_idx[8:])
                        if self.add_relation(in_node_idx, node_one_idx, type_rel):
                            save_nodes.update([node_one_idx, in_node_idx])

                else:
                    if self.add_relation(node_two_idx, node_one_idx, 'struct'): # self.struct_rel_idx
                        save_nodes.update([node_one_idx, node_two_idx])

                    if type(nodes_three) == dict and nodes_three != {}:
                        save_nodes.update(self.add_relations_from_dict({node_two_idx: nodes_three}))

        return save_nodes

    def export_to_zip(self, path_save: str=None) -> str:
        if not path_save: path_save = self._CONFIG['save']['path_save'][:-4] + 'export.zip'
        self._zip_db(self._CONFIG['save']['path_save'], path_save)

        return path_save

    def import_from_zip(self, path_zip_file: str,  path_save: str=None) -> str:
        if not path_save: path_save = self._CONFIG['save']['path_save']
        self._import_from_zip(path_zip_file, path_save)
        self.__init__()
        return path_save

    @caching_cleaner
    def import_from_json(self, path: Union[dict, str]=None) -> bool:
        if type(path) == str:
            json_data = json.loads(open(path, "r").read())
        elif type(path) == dict: json_data = path
        else: raise TypeError('ERROR: import_from_json get str or dict!')

        for node_idx, node_dict in json_data.items():
            if node_idx in self._net.keys():
                self.remove_node_idx(node_idx)

        json_data_gen_idxs  = {}

        for node_idx in [i for i in json_data.keys()]:
            if '@gen' in node_idx:
                json_data_gen_idxs[node_idx] = DBESNode.to_idx(DBESNode.generate_idx())
                json_data[node_idx]['idx'] = json_data_gen_idxs[node_idx]

                for dir_rel, rev_dir_rel in zip(['in', 'out'], ['out', 'in']):
                    for node_rel_idx in json_data[node_idx]['relation'][dir_rel].keys():
                        if node_rel_idx in json_data.keys():
                            json_data[node_rel_idx]['relation'][rev_dir_rel][json_data_gen_idxs[node_idx]] = json_data[node_rel_idx]['relation'][rev_dir_rel][node_idx]
                            del json_data[node_rel_idx]['relation'][rev_dir_rel][node_idx]

                json_data[json_data_gen_idxs[node_idx]] = json_data[node_idx]
                del json_data[node_idx]

        for node_idx in [i for i in json_data.keys()]:
            json_data[node_idx] = DBESNode.from_dict(json_data[node_idx])

        for node_idx in [i for i in json_data.keys()]:
            for dir_rel in ['in', 'out']:
                for node_rel_idx in [i for i in json_data[node_idx].relation[dir_rel].keys()] :
                    if node_rel_idx in json_data_gen_idxs.keys():
                        json_data[node_idx].relation[dir_rel][json_data_gen_idxs[node_rel_idx]] = json_data[node_idx].relation[dir_rel][node_rel_idx]
                        del json_data[node_idx].relation[dir_rel][node_rel_idx]

        for node_idx, node in json_data.items():
            self.add_node(node)

        for node_idx, node_dict in json_data.items():
            for dir_rel in ['in', 'out']:
                for node_rel_idx, rel_idxs in node_dict.relation[dir_rel].items():
                    if dir_rel == 'in':  self.add_relation(node_idx, node_rel_idx, rel_idxs)
                    if dir_rel == 'out': self.add_relation(node_rel_idx, node_idx, rel_idxs)
                    self.add_net_file(node_rel_idx, self._net[node_rel_idx])

        for node_idx, node in json_data.items():
            self.add_net_file(node_idx, node)

        return True

    @caching_cleaner
    def import_from_sep_json(self, path_val: Union[dict, str]=None, path_rel: Union[dict, str]=None) -> bool:
        if type(path_val) == str:
            json_data_val = json.loads(open(path_val, "r").read())
        elif type(path_val) == dict: json_data_val = path_val
        elif not path_val: json_data_val = None
        else: raise TypeError('ERROR: import_from_sep_json get str or dict!')

        if type(path_rel) == str:
            json_data_rel = json.loads(open(path_rel, "r").read())
        elif type(path_rel) == dict: json_data_rel = path_rel
        elif not path_rel: json_data_rel = None
        else: raise TypeError('ERROR: import_from_sep_json get str or dict!')

        save_nodes = set([])

        if json_data_val:
            save_nodes = self.add_nodes_from_dict(json_data_val)

        if json_data_rel: save_nodes.update(self.add_relations_from_dict(json_data_rel))

        for idx in save_nodes:
            self.add_net_file(idx, self._net[idx])

        return True

    @caching_cleaner
    def import_from_bad_json(self, path: Union[dict, str]) -> bool:
        if type(path) == str:
            json_data = json.loads(open(path, "r").read())
        elif type(path) == dict: json_data = path
        else: raise TypeError('ERROR: import_from_json get str or dict!')

        def add_nodes_from_dict(nodes_dict: dict, node_init: str = None) -> dict:
            new_nodes_dict = {}
            _root = False

            for node_one_val, nodes_two in nodes_dict.items():
                if not node_init:
                    _root = True
                    node_init = node_one_val

                if '@' not in node_one_val:
                    for link in ['@inlink', '@outlink']:
                        if link in nodes_two.keys():
                            nodes_two_ = {}
                            for node_name, type_rel in nodes_two[link].items():
                                if node_name == '@root':
                                    idx_ = DBESNode.to_idx(node_init)
                                else: idx_ = DBESNode.to_idx(node_name)
                                nodes_two_[idx_] = type_rel
                            nodes_two[link] = nodes_two_

                    if '@type' not in nodes_two.keys():
                        idx = DBESNode.to_idx(node_one_val)
                    elif nodes_two['@type'] == 'value':
                        idx = DBESNode.to_idx(str(uuid.uuid4()))
                    else: idx = DBESNode.to_idx(node_one_val)

                    self.add_node({'idx': idx, 'value': node_one_val}, save=False)

                    if type(nodes_two) == dict and nodes_two != {}:
                        nodes_two_ = add_nodes_from_dict(nodes_two, node_init)
                        if nodes_two_ != {}:
                            if '@type' in nodes_two.keys():
                                nodes_two_['@type'] = nodes_two['@type']
                            nodes_two = nodes_two_

                    new_nodes_dict[idx] = nodes_two

                if _root: node_init = None

            return new_nodes_dict

        def add_relations_from_dict(nodes_dict: dict) -> None:
            for node_one_idx, nodes_two in nodes_dict.items():

                if type(nodes_two) == dict and nodes_two != {}:
                    for node_two_idx, nodes_three in nodes_two.items():

                        if node_two_idx == '@inlink':
                            for out_node_idx, type_rel in nodes_three.items():
                                self.add_relation(node_one_idx, out_node_idx, type_rel)

                        elif node_two_idx == '@outlink':
                            for in_node_idx, type_rel in nodes_three.items():
                                self.add_relation(in_node_idx, node_one_idx, type_rel)

                        else:
                            if '@type' not in nodes_two.keys() or \
                                '@type' in nodes_two.keys() and nodes_two['@type'] != '@root':
                                self.add_relation(node_two_idx, node_one_idx, 'struct') # self.struct_rel_idx

                            if type(nodes_three) == dict and nodes_three != {}:
                                add_relations_from_dict({node_two_idx: nodes_three})

        new_nodes_dict = add_nodes_from_dict(json_data)
        self._save_json('bla.json', new_nodes_dict)
        add_relations_from_dict(new_nodes_dict)

        for idx, data in self._net.items():   ##################### save
            self.add_net_file(idx, data)

        return True

    @caching_cleaner
    def import_from_sep_table(self, val_table: Iterable[Iterable]=None, rel_table: Iterable[Iterable]=None, idx_idx: int=None, val_idx: int=None, rel_idxs: set=None, vert_header_idx: int=None,
    type_rel_idx: int=None, type_rel: str=None, idx_relation: dict=None, header_relation: dict=None, header: Iterable[str]=None, header_to_idx: bool=False, first_column_to_idx: bool=False,
    header_to_val: bool=False, first_column_to_val: bool=False, header_to_gen_idx: bool=False, first_column_to_gen_idx: bool=False, idx_val_relation: dict=None, idx_replace_idxs: dict=None, idx_replace_from_val_rel: set=None) -> dict:
        if header_to_val and not header_to_gen_idx and not idx_replace_from_val_rel: header_to_idx = True
        if first_column_to_val and not first_column_to_gen_idx and not idx_replace_from_val_rel: first_column_to_idx= True

        nodes_vals = {}
        nodes_rels = {}
        nodes_prev_idx_to_idxs = {}

        ######################################
        # Checking attributes on incompatibles
        if idx_idx == None:
            if val_idx == 0 or vert_header_idx == 0 or type_rel_idx == 0 or rel_idxs and 0 in rel_idxs:
                raise ValueError("Need specify idx_idx attribute!")
            idx_idx = 0

        vert_probs = [vert_header_idx]
        check_props = [val_idx, idx_idx, vert_header_idx, type_rel_idx]
        check_props_str = ["val_idx", "idx_idx", "vert_header_idx", "type_rel_idx"]
        checked_props = set([])

        for i, prop in enumerate(check_props):
            if prop != None:
                if rel_idxs and prop in rel_idxs:
                    raise ValueError(f"Idx attribute {check_props_str[i]} must not contains in rel_idxs attribute!")
                for j, other_prop in enumerate(checked_props):
                    if other_prop != None and j not in checked_props and j != i and prop == other_prop:
                        raise ValueError(f"Attributes for function need dont be equal: {check_props_str[i]} == {check_props_str[j]} - WRONG!")
            checked_props.add(i)

        check_rel_probs = [rel_idxs, vert_header_idx, type_rel_idx, header, header_to_idx, header_to_val, header_to_gen_idx, header_relation, idx_val_relation]
        check_rel_probs_str = ["rel_idxs", "vert_header_idx", "type_rel_idx", "header", "header_to_idx", "header_to_val", "header_to_gen_idx", "header_relation"]
        if not rel_table:
            for i, ch_rel_prob in enumerate(check_rel_probs):
                if ch_rel_prob != None and type(ch_rel_prob) != bool or type(ch_rel_prob) == bool and ch_rel_prob == True:
                    raise AttributeError(f"Used some of attributes for relation table: {check_rel_probs_str[i]}! Need place relation table in rel_table attribute of function!")

        if header:
            if vert_header_idx != None:
                raise ValueError("Attributes header and vert_header_idx cant use together!")

            if rel_idxs:
                if len(header) != len(rel_idxs):
                    raise ValueError(f"If header and rel_idxs specifyed, need: len(header) == len(rel_idxs), but they not equal: {len(header)} != {len(rel_idxs)}!")

        if idx_replace_idxs and ((first_column_to_idx or first_column_to_gen_idx) and (idx_idx in idx_replace_idxs.keys() or "any" in idx_replace_idxs.keys()) or \
           (header_to_idx or header_to_gen_idx) and "header" in idx_replace_idxs.keys()):
            raise ValueError("Need choose between idx_replace_idxs and (first_column_to_idx, first_column_to_gen_idx, header_to_idx, header_to_gen_idx)!")

        if idx_replace_from_val_rel and (((first_column_to_idx or first_column_to_gen_idx) and (idx_idx in idx_replace_from_val_rel.keys() or "any" in idx_replace_from_val_rel) or \
           (header_to_idx or header_to_gen_idx) and "header" in idx_replace_from_val_rel) or \
           idx_replace_idxs and len(set(idx_replace_idxs.keys()).intersection(set(idx_replace_from_val_rel))) > 0):
            raise ValueError("Need choose between idx_replace_idxs and idx_replace_from_val_rel and (first_column_to_idx, first_column_to_gen_idx, header_to_idx, header_to_gen_idx)!")

        if first_column_to_idx and first_column_to_gen_idx:
            raise ValueError("Cant use first_column_to_idx and first_column_to_gen_idx in one time!")

        if header_to_idx and header_to_gen_idx:
            raise ValueError("Cant use header_to_idx and header_to_gen_idx in one time!")
        ######################################

        for some_relation in [idx_relation, header_relation, idx_val_relation]:
            if some_relation:
                prev_idx = some_relation["idx"]

                if "@@to_idx" in some_relation["idx"]:
                    prev_idx = some_relation["idx"][8:]
                    some_relation["idx"] = DBESNode.to_idx(some_relation["idx"][8:])

                if "@@gen" in some_relation["idx"]:
                    prev_idx = some_relation["idx"][5:]
                    some_relation["idx"] = DBESNode.to_idx(DBESNode.generate_idx())

                nodes_vals[some_relation["idx"]] = some_relation["value"]
                # nodes_prev_idx_to_idxs[some_relation["value"]] = some_relation["idx"]
                nodes_prev_idx_to_idxs[prev_idx] = some_relation["idx"]

        if val_table:
            for line in val_table:
                if line != []:
                    node_val = None
                    node_idx = None
                    if not first_column_to_val:
                        node_idx = line[idx_idx]

                        if val_idx == None:
                            if idx_idx != 0:
                                node_val = line[0]
                            else: node_val = line[1]

                        else:
                            node_val = line[val_idx]

                    else:
                        node_idx = line[idx_idx]
                        node_val = node_idx

                    if node_val and node_idx:
                        node_idx_prev = node_idx

                        if idx_replace_idxs:            node_idx = self._replace_idx_from(idx_replace_idxs, node_idx, idx_idx, raise_err=True)
                        if idx_replace_from_val_rel:    node_idx = self._replace_idx_from_val_rel(idx_replace_from_val_rel, node_idx, idx_idx, raise_err=True)
                        if first_column_to_idx:         node_idx = DBESNode.to_idx(node_idx)
                        elif first_column_to_gen_idx:   node_idx = DBESNode.to_idx(DBESNode.generate_idx())

                        nodes_vals[node_idx] = node_val

                        ####
                        if idx_relation: self._add_some_relation(nodes_rels, idx_relation, node_idx)
                        # nodes_prev_idx_to_idxs[node_val] = node_idx
                        nodes_prev_idx_to_idxs[node_idx_prev] = node_idx

                        if idx_val_relation:
                            gen_idx = DBESNode.to_idx(DBESNode.generate_idx())

                            nodes_vals[gen_idx] = node_idx_prev
                            self._add_some_relation(nodes_rels, idx_val_relation, gen_idx)

                            val_type_rel = "value"
                            if node_idx not in nodes_rels.keys():
                                nodes_rels[node_idx] = {"@inlink": {gen_idx: val_type_rel}}
                            elif "@inlink" not in nodes_rels[node_idx].keys():
                                nodes_rels[node_idx]["@inlink"] = {gen_idx: val_type_rel}
                            else: nodes_rels[node_idx]["@inlink"][gen_idx] = val_type_rel

        if rel_table:
            header_idxs = []
            header_ = True

            for line in rel_table:
                if line != []:
                    if header_ and vert_header_idx == None:
                        header_ = False

                        for i, node_idx in enumerate(line) if not header else enumerate(header):
                            if header or i not in check_props:
                                node_idx_ = node_idx

                                if idx_replace_idxs:            node_idx_ = self._replace_idx_from(idx_replace_idxs, node_idx, "header", raise_err=True)
                                if idx_replace_from_val_rel:    node_idx_ = self._replace_idx_from_val_rel(idx_replace_from_val_rel, node_idx, "header", raise_err=True)
                                if header_to_idx:               node_idx_ = DBESNode.to_idx(node_idx)
                                elif header_to_gen_idx:         node_idx_ = DBESNode.to_idx(DBESNode.generate_idx())

                                if header_to_val: nodes_vals[node_idx_] = node_idx
                                header_idxs.append(node_idx_)

                                ####
                                if header_relation: self._add_some_relation(nodes_rels, header_relation, node_idx_)
                                nodes_prev_idx_to_idxs[node_idx] = node_idx_

                        if not header: continue

                    node_idx = line[idx_idx]
                    node_idx_ = node_idx

                    if idx_replace_idxs:            node_idx_ = self._replace_idx_from(idx_replace_idxs, node_idx, idx_idx)
                    if idx_replace_from_val_rel:    node_idx_ = self._replace_idx_from_val_rel(idx_replace_from_val_rel, node_idx, idx_idx)
                    if first_column_to_idx:         node_idx_ = DBESNode.to_idx(node_idx)
                    elif first_column_to_gen_idx:   node_idx_ = DBESNode.to_idx(DBESNode.generate_idx())

                    if node_idx_ == None: continue

                    if first_column_to_val: nodes_vals[node_idx_] = node_idx

                    ####
                    if idx_relation: self._add_some_relation(nodes_rels, idx_relation, node_idx_)
                    nodes_prev_idx_to_idxs[node_idx] = node_idx_

                    if idx_val_relation:
                        gen_idx = DBESNode.to_idx(DBESNode.generate_idx())

                        nodes_vals[gen_idx] = node_idx
                        self._add_some_relation(nodes_rels, idx_val_relation, gen_idx)

                        val_type_rel = "value"
                        if node_idx_ not in nodes_rels.keys():
                            nodes_rels[node_idx_] = {"@inlink": {gen_idx: val_type_rel}}
                        elif "@inlink" not in nodes_rels[node_idx_].keys():
                            nodes_rels[node_idx_]["@inlink"] = {gen_idx: val_type_rel}
                        else: nodes_rels[node_idx_]["@inlink"][gen_idx] = val_type_rel

                    node_idx = node_idx_

                    counter_i = 0
                    for i, node_val in enumerate(line):
                        if i == val_idx:
                            nodes_vals[node_idx] = node_val
                            nodes_prev_idx_to_idxs[node_val] = node_idx

                        if (not rel_idxs or i in rel_idxs) and (i not in check_props and vert_header_idx == None or vert_header_idx != None and i == vert_header_idx):
                            cor_i = counter_i

                            curr_header_idx = header_idxs[cor_i] if vert_header_idx == None else node_val
                            if vert_header_idx != None:
                                node_val = ""
                                curr_header_idx_ = curr_header_idx

                                if idx_replace_idxs:            curr_header_idx_ = self._replace_idx_from(idx_replace_idxs, curr_header_idx, "header")
                                if idx_replace_from_val_rel:    curr_header_idx_ = self._replace_idx_from_val_rel(idx_replace_from_val_rel, curr_header_idx, "header")
                                if header_to_idx:               curr_header_idx_ = DBESNode.to_idx(curr_header_idx)
                                elif header_to_gen_idx:         curr_header_idx_ = DBESNode.to_idx(DBESNode.generate_idx())

                                if curr_header_idx_ == None: continue

                                if header_to_val: nodes_vals[curr_header_idx_] = curr_header_idx

                                ####
                                if header_relation: self._add_some_relation(nodes_rels, header_relation, curr_header_idx_)
                                nodes_prev_idx_to_idxs[curr_header_idx] = curr_header_idx_
                                curr_header_idx = curr_header_idx_

                            if node_val == "" or type(node_val) == str and node_val[0:2] == "@@":
                                if node_val == "": node_val = "@@struct" if not type_rel else type_rel if type_rel_idx == None else line[type_rel_idx]
                                cur_type_rel = node_val[2:]

                                if node_idx not in nodes_rels.keys():
                                    nodes_rels[node_idx] = {"@inlink": {curr_header_idx: cur_type_rel}}
                                elif "@inlink" not in nodes_rels[node_idx].keys():
                                    nodes_rels[node_idx]["@inlink"] = {curr_header_idx: cur_type_rel}
                                else: nodes_rels[node_idx]["@inlink"][curr_header_idx] = cur_type_rel

                            else:
                                gen_idx = DBESNode.to_idx(DBESNode.generate_idx())
                                if (curr_header_idx in nodes_vals.keys() or curr_header_idx in self._net.keys() or header_to_val) and \
                                    (node_idx in nodes_vals.keys() or node_idx in self._net.keys() or first_column_to_val):
                                    nodes_vals[gen_idx] = node_val

                                if curr_header_idx not in nodes_rels.keys():
                                    nodes_rels[curr_header_idx] = {gen_idx: {}}
                                else: nodes_rels[curr_header_idx][gen_idx] = {}

                                cur_type_rel = "@@value" if not type_rel else type_rel if type_rel_idx == None else line[type_rel_idx]
                                cur_type_rel = cur_type_rel[2:]

                                if node_idx not in nodes_rels.keys():
                                    nodes_rels[node_idx] = {"@inlink": {gen_idx: cur_type_rel}}
                                elif "@inlink" not in nodes_rels[node_idx].keys():
                                    nodes_rels[node_idx]["@inlink"] = {gen_idx: cur_type_rel}
                                else: nodes_rels[node_idx]["@inlink"][gen_idx] = cur_type_rel

                            counter_i += 1

        if nodes_vals or nodes_rels:
            self.import_from_sep_json(nodes_vals, nodes_rels)

        return nodes_prev_idx_to_idxs

    @caching_cleaner
    def import_from_sep_csv(self, file_csv: str=None, path: str=None,  delimiter: str=',', quotechar: str='"', properties: dict=None) -> bool:
        if path: file_csv = self._open_csv(file_name=path, path_save="./")
        elif file_csv: file_csv = csv.reader(file_csv, delimiter=delimiter, quotechar=quotechar)

        if "type_table" in properties.keys():
            if "val" == properties["type_table"]:
                del properties["type_table"]
                self.import_from_sep_table(val_table=file_csv, **properties)
            elif "rel" == properties["type_table"]:
                del properties["type_table"]
                self.import_from_sep_table(rel_table=file_csv, **properties)

        return True

    @caching_cleaner
    def import_from_postgres(self, commands: Iterable[dict]):
        if self.postgres:
            self.postgres.connect()

            prev_idxs = {}

            for command_dict in commands:
                table_name = command_dict["table_name"]
                command = command_dict["command"]

                if "type_table" in command.keys():
                    table_data = self.postgres.get(command['sql'])
                    del command['sql']

                    if "idx_replace_idxs" in command.keys():
                        if command["idx_replace_idxs"] != {}:
                            for idx_replace, replace_table in command["idx_replace_idxs"].items():
                                for t_name, _ in replace_table.items():
                                    if t_name in prev_idxs.keys():
                                        command["idx_replace_idxs"][idx_replace][t_name] = prev_idxs[t_name]
                                    else: raise KeyError(f"Cant find {t_name} table in previos replace idxs!")
                        else: command["idx_replace_idxs"] = {"all": prev_idxs}

                    if "val" in command["type_table"]:
                        del command["type_table"]

                        prev_idxs_ = self.import_from_sep_table(val_table=table_data, **command)
                        prev_idxs[table_name] = prev_idxs_

                    elif "rel" in command["type_table"]:
                        del command["type_table"]
                        prev_idxs_ = self.import_from_sep_table(rel_table=table_data, **command)
                        prev_idxs[table_name] = prev_idxs_

        else: print("Postgres Adapter have not initialized!")

    @caching_profile
    def find_template_val_idxs(self, value: str, contains: bool=False, data_dict: dict=None, filter_idxs: set=None, **kwargs) -> set:
        return self.find_val_idxs(value         = value,
                                  contains      = contains,
                                  data_dict     = data_dict if data_dict else self._templates,
                                  filter_idxs   = filter_idxs)

    @caching_profile
    def find_template_rel_idxs(self, node_rel_idx: str, dir_rel: str='in', data_dict: dict=None, find_idxs: set=None, filter_idxs: set=None, recursive: bool=False, use_filter_idxs_for_recursive: bool=True, checked_nodes: dict=None, root_: bool=True, **kwargs) -> set:
        return self.find_rel_idxs(node_rel_idx  = node_rel_idx,
                                  dir_rel       = dir_rel,
                                  data_dict     = data_dict if data_dict else self._templates,
                                  find_idxs     = find_idxs,
                                  filter_idxs   = filter_idxs,
                                  recursive     = recursive,
                                  checked_nodes = checked_nodes,
                                  root_         = root_,
                                  use_filter_idxs_for_recursive = use_filter_idxs_for_recursive)

    @caching_profile
    def find_template_mult_rel_idxs(self, node_rel_idxs: dict, data_dict: dict=None, find_idxs: set=None, filter_idxs: set=None, and_on: bool=True, recursive: bool=False, use_filter_idxs_for_recursive: bool=True, **kwargs) -> set:
        return self.find_mult_rel_idxs(node_rel_idxs    = node_rel_idxs,
                                       data_dict        = data_dict if data_dict else self._templates,
                                       find_idxs        = find_idxs,
                                       filter_idxs      = filter_idxs,
                                       and_on           = and_on,
                                       recursive        = recursive,
                                       use_filter_idxs_for_recursive = use_filter_idxs_for_recursive)

    @caching_profile
    def find_all_out_idxs(self, node_idx: str, filter_idxs: set=None, **kwargs) -> set:
        finded_idxs = set([node_idx])

        for node_rel_idx in self._net[node_idx].relation['out'].keys():
            if not filter_idxs or node_rel_idx in filter_idxs:
                finded_idxs.add(node_rel_idx)
                finded_idxs.update(self.find_all_out_idxs(node_rel_idx, filter_idxs=filter_idxs))

        return finded_idxs

    @caching_profile
    def find_all_in_idxs(self, node_idx: str, filter_idxs: set=None, **kwargs) -> set:
        finded_idxs = set([node_idx])

        for node_rel_idx in self._net[node_idx].relation['in'].keys():
            if not filter_idxs or node_rel_idx in filter_idxs:
                finded_idxs.add(node_rel_idx)
                finded_idxs.update(self.find_all_in_idxs(node_rel_idx, filter_idxs=filter_idxs))

        return finded_idxs

    @caching_profile
    def find_all_dir_rel_idxs_lvls(self, node_idx: str, dir_rel: str="in", filter_idxs: set=None, lvl: int=1, use_filter_idxs_for_recursive: bool=True, use_filter_idxs_after: bool=False, **kwargs) -> dict:
        finded_idxs_lvls = {node_idx: lvl}

        lvl += 1
        for node_rel_idx in self._net[node_idx].relation[dir_rel].keys():
            if not filter_idxs or not use_filter_idxs_for_recursive or node_rel_idx in filter_idxs:

                for node_idx, lvl_idx in self.find_all_dir_rel_idxs_lvls(node_rel_idx, dir_rel=dir_rel, filter_idxs=filter_idxs, lvl=lvl).items():
                    if node_idx not in finded_idxs_lvls.keys():
                        finded_idxs_lvls[node_idx] = lvl_idx
                    elif finded_idxs_lvls[node_idx] > lvl_idx:
                        finded_idxs_lvls[node_idx] = lvl_idx

        if use_filter_idxs_after:
            finded_idxs_lvls = {idx: lvl for idx, lvl in finded_idxs_lvls.items() if idx in filter_idxs}

        return finded_idxs_lvls

    @caching_profile
    def find_type_rel_idxs(self, rel_idx: str, dir_rel: str='in', data_dict: dict=None, find_idxs: set=None, filter_idxs: set=None, recursive: bool=False, use_filter_idxs_for_recursive: bool=True, checked_nodes: dict=None, **kwargs) -> set:
        finded_idxs = set([])
        if not checked_nodes: checked_nodes = {'true': {'in': set([]), 'out': set([])}, 'false': {'in': set([]), 'out': set([])}}

        if not data_dict: data_dict = self._net

        if not filter_idxs: use_filter_idxs_for_recursive = False

        if dir_rel == 'any':    dir_rels = ['in', 'out'] #### for recursive ??
        else:                   dir_rels = [dir_rel]

        for dir_rel in dir_rels:
            for node_idx in data_dict.keys() if not find_idxs and not filter_idxs else find_idxs if find_idxs else filter_idxs:
                finded_ = False

                for node_rel_idx in data_dict[node_idx].relation[dir_rel].keys():
                    if rel_idx in data_dict[node_idx].relation[dir_rel][node_rel_idx].keys():
                        finded_ = True
                        finded_idxs.add(node_idx)
                        checked_nodes['true'][dir_rel].add(node_idx)
                        break

                if not finded_ and recursive and data_dict[node_idx].relation[dir_rel] != {}:
                    filter_idxs_ = set([])

                    for rel_node_idx in data_dict[node_idx].relation[dir_rel].keys():
                        if rel_node_idx in checked_nodes['true'][dir_rel]:
                            finded_ = True
                            finded_idxs.add(node_idx)
                            checked_nodes['true'][dir_rel].add(node_idx)
                            filter_idxs_ = set([])
                            break

                        elif rel_node_idx not in checked_nodes['false'][dir_rel]:
                            if not use_filter_idxs_for_recursive or rel_node_idx in filter_idxs:
                                filter_idxs_.add(rel_node_idx)

                    if filter_idxs_:
                        finded_idxs_ = self.find_type_rel_idxs(rel_idx, dir_rel, find_idxs=filter_idxs_, filter_idxs=filter_idxs, recursive=True, checked_nodes=checked_nodes)

                        if finded_idxs_:
                            finded_ = True
                            finded_idxs.add(node_idx)
                            checked_nodes['true'][dir_rel].add(node_idx)

                if not finded_: checked_nodes['false'][dir_rel].add(node_idx)

        return finded_idxs

    @caching_profile
    def find_rel_idxs(self, node_rel_idx: str, dir_rel: str='in', data_dict: dict=None, find_idxs: set=None, filter_idxs: set=None, recursive: bool=False, use_filter_idxs_for_recursive: bool=True, checked_nodes: dict=None, root_: bool=True, **kwargs) -> set:
        finded_idxs = set([])
        if not checked_nodes: checked_nodes = {}

        if not data_dict: data_dict = self._net

        if not filter_idxs: use_filter_idxs_for_recursive = False

        if dir_rel == 'any':    dir_rels = ['in', 'out'] #### for recursive ??
        else:                   dir_rels = [dir_rel]

        for dir_rel in dir_rels:
            for node_idx in data_dict.keys() if not find_idxs and not filter_idxs else find_idxs if find_idxs else filter_idxs:
                checked_nodes[node_idx] = False

                if node_rel_idx in data_dict[node_idx].relation[dir_rel].keys():
                    finded_idxs.add(node_idx)
                    checked_nodes[node_idx] = True

                elif recursive and data_dict[node_idx].relation[dir_rel] != {}:
                    filter_idxs_ = set([])

                    for rel_node_idx in data_dict[node_idx].relation[dir_rel].keys():
                        if rel_node_idx not in checked_nodes.keys():
                            if not use_filter_idxs_for_recursive or rel_node_idx in filter_idxs:
                                filter_idxs_.add(rel_node_idx)

                        elif checked_nodes[rel_node_idx]:
                            finded_idxs.add(node_idx)
                            checked_nodes[node_idx] = True
                            filter_idxs_ = set([])
                            break

                    if filter_idxs_:
                        finded_idxs_ = self.find_rel_idxs(node_rel_idx, dir_rel, find_idxs=filter_idxs_, filter_idxs=filter_idxs, recursive=True, checked_nodes=checked_nodes, root_=False)

                        if finded_idxs_:
                            finded_idxs.add(node_idx)
                            checked_nodes[node_idx] = True

        return finded_idxs

    @caching_profile
    def find_rel_idxs_NORec(self, node_rel_idx: str, dir_rel: str='in', data_dict: dict=None, find_idxs: set=None, filter_idxs: set=None, recursive: bool=False, use_filter_idxs_for_recursive: bool=True, **kwargs) -> set:
        finded_idxs = set([])

        if not data_dict: data_dict = self._net

        if not filter_idxs: use_filter_idxs_for_recursive = False

        if dir_rel == 'any':
            dir_rels = ['in', 'out'] #### for recursive ??
        else:
            dir_rels = [dir_rel]

        for dir_rel in dir_rels:
            opposite_dir_rel = "out" if dir_rel == "in" else "in"

            for node_idx in data_dict.keys() if not find_idxs and not filter_idxs else find_idxs if find_idxs else filter_idxs:
                if node_rel_idx in data_dict[node_idx].relation[dir_rel].keys():
                    finded_idxs.add(node_idx)

                    if recursive:
                        current_nodes = set([node_idx])

                        while True:
                            next_nodes = set([])

                            for current_node in current_nodes:
                                finded_idxs.add(current_node)

                                for rel_node_idx in data_dict[current_node].relation[opposite_dir_rel].keys():
                                    if rel_node_idx not in finded_idxs and (not use_filter_idxs_for_recursive or rel_node_idx in filter_idxs):
                                        next_nodes.add(rel_node_idx)

                            if next_nodes:
                                current_nodes = next_nodes
                            else: break

        return finded_idxs

    @caching_profile
    def find_mult_rel_idxs(self, node_rel_idxs: dict, data_dict: dict=None, find_idxs: set=None, filter_idxs: set=None, and_on: bool=True, recursive: bool=False, use_filter_idxs_for_recursive: bool=True, **kwargs) -> set:
        finded_idxs = set([])

        if not data_dict: data_dict = self._net

        if not filter_idxs: use_filter_idxs_for_recursive = False

        for dir_rel, set_node_rel_idxs in node_rel_idxs.items():
            if type(set_node_rel_idxs) == set:
                node_rel_idxs[dir_rel] = set_node_rel_idxs
            elif type(set_node_rel_idxs) == list:
                node_rel_idxs[dir_rel] = set(set_node_rel_idxs)
            elif type(set_node_rel_idxs) == str:
                node_rel_idxs[dir_rel] = set([set_node_rel_idxs])
            else: raise TypeError(f'ERROR! find_mult_rel_idxs can get only list of str idxs or str idx! Get {type(set_node_rel_idxs)}')

        for node_idx in data_dict.keys() if not find_idxs and not filter_idxs else find_idxs if find_idxs else filter_idxs:
            flag_add = and_on

            for dir_rel, set_node_rel_idxs in node_rel_idxs.items():
                if flag_add == (not and_on): break

                node_relations = data_dict[node_idx].relation[dir_rel]
                if node_relations != {}:
                    for node_rel_idx in set_node_rel_idxs:
                        if node_rel_idx not in node_relations.keys():
                            if recursive:
                                if use_filter_idxs_for_recursive:
                                    filter_idxs_ = set([idx for idx in node_relations.keys() if idx in filter_idxs])
                                else: filter_idxs_ = set(node_relations.keys())

                                finded_idxs_ = self.find_rel_idxs_NORec(node_rel_idx, dir_rel,
                                                                        use_filter_idxs_for_recursive=use_filter_idxs_for_recursive,
                                                                        find_idxs=filter_idxs_,
                                                                        filter_idxs=filter_idxs,
                                                                        recursive=True)

                                if (not finded_idxs_ and and_on) or (finded_idxs_ and not and_on):
                                    flag_add = not and_on; break

                            elif and_on: flag_add = False; break
                        elif not and_on: flag_add = True; break
                elif and_on: flag_add = False; break

            if flag_add == True: finded_idxs.add(node_idx)

        return finded_idxs

    @caching_profile
    def find_val_idxs(self, value: Union[str, int, float], contains: bool=False, data_dict: dict=None, filter_idxs: set=None, **kwargs) -> set:
        finded_idxs = set([])

        if not data_dict: data_dict = self._net

        for node_idx in data_dict.keys() if not filter_idxs else filter_idxs:
            if value == data_dict[node_idx].value:
                finded_idxs.add(node_idx)
            elif contains and type(value) == str and type(data_dict[node_idx].value) == str and value in data_dict[node_idx].value:
                finded_idxs.add(node_idx)

        return finded_idxs

    @caching_profile
    def find_idxs_type_rel_coeff(self, node_rel_idxs: dict, data_dict: dict=None, filter_idxs: set=None, combine: str='+', **kwargs) -> dict:
        idxs_rel_coef = {}

        if not data_dict: data_dict = self._net

        for node_idx in data_dict.keys() if not filter_idxs else filter_idxs:
            for dir_rel, list_node_rel_idxs in node_rel_idxs.items():
                if data_dict[node_idx].relation[dir_rel] != {}:
                    for node_rel_idx, rel_idx in list_node_rel_idxs:
                        if node_rel_idx in data_dict[node_idx].relation[dir_rel].keys() and \
                           rel_idx in data_dict[node_idx].relation[dir_rel][node_rel_idx].keys() :
                            type_rel_coeff = data_dict[node_idx].relation[dir_rel][node_rel_idx][rel_idx]
                            type_rel_coeff = type_rel_coeff if type_rel_coeff else 0

                            if node_idx not in idxs_rel_coef.keys():
                                idxs_rel_coef[node_idx] = type_rel_coeff
                            else:
                                if      combine == '+': idxs_rel_coef[node_idx] += type_rel_coeff        # несколько уровней сортировки а не комбинация?
                                elif    combine == '-': idxs_rel_coef[node_idx] -= type_rel_coeff
                                elif    combine == '*': idxs_rel_coef[node_idx] *= type_rel_coeff
                                elif    combine == '/': idxs_rel_coef[node_idx] /= type_rel_coeff

        return idxs_rel_coef

    @caching_profile
    def find_and_combine_idxs_type_rel_coeff(self, finded_idxs: set, node_rel_idxs: dict, combine: str='+', **kwargs) -> dict:
        combined_idxs_rel_coef = {}
        idxs_rel_coef = self.find_idxs_type_rel_coeff(node_rel_idxs)

        for node_rel_idx, coeff in idxs_rel_coef.items():
            finded_idxs_ = self.find_rel_idxs_NORec(node_rel_idx, dir_rel='out', recursive=True, find_idxs=finded_idxs)

            for idx in finded_idxs_:
                if idx not in combined_idxs_rel_coef.keys():
                    combined_idxs_rel_coef[idx] = coeff
                else:
                    if      combine == '+': combined_idxs_rel_coef[idx] += coeff
                    elif    combine == '-': combined_idxs_rel_coef[idx] -= coeff
                    elif    combine == '*': combined_idxs_rel_coef[idx] *= coeff
                    elif    combine == '/': combined_idxs_rel_coef[idx] /= coeff

        return combined_idxs_rel_coef

    @caching_profile
    def find_idxs_val_rel_coeff(self, node_rel_idxs: dict, data_dict: dict=None, filter_idxs: set=None, combine: str='+', **kwargs) -> dict:
        idxs_rel_coef = {}

        if not data_dict: data_dict = self._net

        for dir_rel, list_node_rel_idxs in node_rel_idxs.items():
            opposite_dir_rel = 'out' if dir_rel == 'in' else 'in'
            for node_rel_idx in list_node_rel_idxs:
                for node_val_idx in data_dict[node_rel_idx].relation[opposite_dir_rel].keys():
                    node_idx = list(data_dict[node_val_idx].relation[opposite_dir_rel].keys())[0]

                    if not filter_idxs or node_idx in filter_idxs:
                        val_rel = data_dict[node_val_idx].value
                        val_rel = val_rel if val_rel else 0

                        self._add_and_combine(node_idx, val_rel, idxs_rel_coef, combine)

        return idxs_rel_coef

    @caching_profile
    def find_and_combine_idxs_val_rel(self, finded_idxs: set, node_rel_idxs: dict, combine: str='+', divide_U: bool=True, return_divided_map: bool=False, filter_idxs: set=None, common_filter_idxs: set=None, **kwargs) -> Union[dict, Tuple[dict, dict]]:
        combined_idxs_rel_coef = {}
        idxs_rel_coef = self.find_idxs_val_rel_coeff(node_rel_idxs, filter_idxs=filter_idxs)

        dist_nodes = {}
        idx_to_divided_idx_val = defaultdict(dict)

        back_struct_nodes = {}
        if divide_U:
            dist_nodes = self.find_shortest_dist_DAG(filter_idxs=finded_idxs if not common_filter_idxs else common_filter_idxs)

        elif return_divided_map: raise ValueError("For return_divided_map need divide_U=True")

        for node_rel_idx, coeff in idxs_rel_coef.items():
            finded_idxs_ = self.find_all_in_idxs(node_rel_idx, filter_idxs=finded_idxs if not common_filter_idxs else common_filter_idxs)
            finded_idxs_.remove(node_rel_idx)

            if finded_idxs_:
                if dist_nodes:
                    divided_idxs: List[list] = []

                    for finded_idx in finded_idxs_:
                        if finded_idx in finded_idxs if not common_filter_idxs else common_filter_idxs:
                            add_idx = False

                            for i, divide_batch in enumerate(divided_idxs):
                                if finded_idx in dist_nodes[divide_batch[0]].keys():
                                    divided_idxs[i].append(finded_idx)
                                    add_idx = True
                                    break

                            if not add_idx:
                                divided_idxs.append([finded_idx])

                    coeff = coeff / len(divided_idxs)

                    for batch_divided_idxs in divided_idxs:
                        back_pattern_nodes = self.find_all_in_idxs(node_rel_idx, filter_idxs=set(batch_divided_idxs))
                        back_struct_nodes = self.find_structure_idxs(back_pattern_nodes, dir_rel='out', root_idx=node_rel_idx)

                        divided_coeff = coeff
                        _temp_combined_idxs_rel_coef = {}
                        for lvl, lvl_idxs in back_struct_nodes.items():
                            if lvl == 1: continue

                            elif lvl == 2:
                                divided_coeff = coeff / len(lvl_idxs)

                                for init_idx in lvl_idxs.keys():
                                    self._add_and_combine(init_idx, divided_coeff, _temp_combined_idxs_rel_coef, some_combine=combine)
                                    if return_divided_map: idx_to_divided_idx_val[init_idx].update({node_rel_idx: divided_coeff})

                            else:
                                for lvl_idx, lvl_rel_idxs in lvl_idxs.items():
                                    comb_coeff = 0

                                    for lvl_rel_idx in lvl_rel_idxs:
                                        if lvl_rel_idx != node_rel_idx:
                                            comb_coeff += _temp_combined_idxs_rel_coef[lvl_rel_idx]

                                    self._add_and_combine(lvl_idx, comb_coeff, _temp_combined_idxs_rel_coef, some_combine=combine)

                        for idx, temp_coeff in _temp_combined_idxs_rel_coef.items():
                            self._add_and_combine(idx, temp_coeff, combined_idxs_rel_coef, some_combine=combine)
                            if return_divided_map: idx_to_divided_idx_val[idx].update({node_rel_idx: temp_coeff})

                else:
                    for idx in finded_idxs_:
                        if node_rel_idx != idx and idx in finded_idxs if not common_filter_idxs else common_filter_idxs:
                            self._add_and_combine(idx, coeff, combined_idxs_rel_coef, some_combine=combine)

        if common_filter_idxs:
            combined_idxs_rel_coef = {idx: val for idx, val in combined_idxs_rel_coef.items() if idx in finded_idxs}

        if not return_divided_map:
            return combined_idxs_rel_coef
        else:
            return combined_idxs_rel_coef, idx_to_divided_idx_val

    @caching_profile
    def find_pattern_idxs(self, pattern: dict, data_dict: dict=None, find_idxs: set=None, filter_idxs: set=None, recursive: bool=True, filter_recursive_idxs: set=None, use_filter_idxs_for_recursive: bool=True, use_filter_idxs_after: bool=False,
    checked_nodes: set=None, tree_root: bool=False, dir_rel: str='in', root_: bool=True, **kwargs) -> set:
        finded_idxs = set([])

        if not filter_recursive_idxs: filter_recursive_idxs = set([])

        if not checked_nodes: checked_nodes = set([])

        if not data_dict: data_dict = self._net

        if not filter_idxs: use_filter_idxs_for_recursive = False

        if root_:
            pattern = self._update_pattern_xor(pattern, filter_idxs=filter_idxs)

            if recursive and filter_idxs:
                if not use_filter_idxs_for_recursive:
                    for filter_node_idx in filter_idxs:
                        filter_recursive_idxs.update(self.find_all_out_idxs(filter_node_idx))
                        filter_recursive_idxs.update(self.find_all_in_idxs(filter_node_idx))

                else: filter_recursive_idxs = set(filter_idxs)

        back_dir_rel = 'out' if dir_rel == 'in' else 'in'

        for node_idx in data_dict.keys() if not find_idxs and not filter_idxs else find_idxs if find_idxs else data_dict.keys() if not use_filter_idxs_for_recursive and root_ else filter_idxs:
            if node_idx not in pattern.keys() and node_idx not in finded_idxs:
                checked_nodes.add(node_idx)
                flag_ = False  # bc we need only existance nodes
                flag_break_ = False  # if we have True depends_or
                root_tree_ = True

                for node_rel_idx, rel_idxs in data_dict[node_idx].relation[dir_rel].items():
                    if flag_break_: break

                    if not filter_recursive_idxs or node_rel_idx in filter_recursive_idxs:
                        for rel_idx in rel_idxs.keys():
                            if 'depends' in rel_idx:
                                if not filter_idxs or node_rel_idx in filter_idxs: root_tree_ = False
                                if node_rel_idx in pattern.keys():
                                    if pattern[node_rel_idx] == 'on':
                                        flag_ = True
                                        if '_or' in rel_idx or '_xor' in rel_idx:
                                            flag_break_ = True
                                            break
                                    elif pattern[node_rel_idx] == 'open':
                                        flag_ = True
                                    elif pattern[node_rel_idx] == 'off':
                                        flag_ = False
                                        if '_and' in rel_idx:
                                            flag_break_ = True
                                            break

                                elif recursive:
                                    flag_ = False

                                    if node_rel_idx not in checked_nodes and (not use_filter_idxs_for_recursive or node_rel_idx in filter_idxs):
                                        finded_idxs_ = self.find_pattern_idxs({**pattern, **{finded_idx: 'open' for finded_idx in finded_idxs}},
                                                                                use_filter_idxs_for_recursive = use_filter_idxs_for_recursive,
                                                                                filter_recursive_idxs = filter_recursive_idxs,
                                                                                find_idxs     = set([node_rel_idx]),
                                                                                filter_idxs   = filter_idxs,
                                                                                recursive     = True,
                                                                                tree_root     = tree_root,
                                                                                checked_nodes = checked_nodes,
                                                                                dir_rel       = dir_rel,
                                                                                root_         = False)

                                        if finded_idxs_:
                                            finded_idxs = finded_idxs.union(finded_idxs_)

                                    if node_rel_idx in finded_idxs:
                                        flag_ = True
                                        if '_or' in rel_idx or '_xor' in rel_idx:
                                            flag_break_ = True
                                            break

                                    if not flag_ and '_and' in rel_idx:
                                        flag_break_ = True
                                        break

                                else:
                                    flag_ = False
                                    if '_and' in rel_idx:
                                        flag_break_ = True
                                        break

                if flag_ or (root_tree_ and tree_root):
                    flag_add_ = True

                    if root_tree_ and tree_root:
                        flag_break__ = False
                        flag_add_ = False

                        if not filter_idxs or node_idx in filter_idxs:
                            for node_rel_idx, rel_idxs in data_dict[node_idx].relation[back_dir_rel].items():
                                if flag_break__: break

                                for rel_idx in rel_idxs.keys():
                                    if 'depends' in rel_idx:
                                        flag_break__ = True
                                        flag_add_ = True
                                        break

                    if flag_add_:
                        finded_idxs.add(node_idx)

        if root_:
            finded_idxs_on_off = set([])
            finded_idxs_open = set([])

            for node_idx_, state in pattern.items():
                if state == 'on' or state == 'off':
                    finded_idxs_on_off.update(self.find_back_pattern_idxs(node_idx_, {}))
                elif state == 'open':
                    finded_idxs_open.update(self.find_back_pattern_idxs(node_idx_, {}))

            finded_idxs = set([idx for idx in finded_idxs if (idx not in finded_idxs_on_off or idx in finded_idxs_open) and (not use_filter_idxs_after or idx in filter_idxs)])
        return finded_idxs

    @caching_profile
    def find_back_pattern_idxs(self, node_idx: str, pattern: dict, data_dict: dict=None, filter_idxs: set=None, recursive: bool=True, use_filter_idxs_for_recursive: bool=True, use_filter_idxs_after: bool=False,
    checked_nodes: set=None, finded_opened_idxs: set=None, clear_opened: bool=False, root_=True, **kwargs) -> set:
        finded_idxs = set([])
        flag_break_ = False
        can_add_or_ = False

        if not data_dict: data_dict = self._net

        if not filter_idxs: use_filter_idxs_for_recursive = False

        if root_:
            pattern = self._update_pattern_xor(pattern, filter_idxs=filter_idxs)

            if clear_opened:
                finded_opened_idxs = self.find_pattern_idxs(pattern, filter_idxs=filter_idxs)

        if not checked_nodes: checked_nodes = set([])
        checked_nodes.add(node_idx)

        flag_break_ = False
        for node_rel_idx, rel_idxs in data_dict[node_idx].relation['in'].items():
            if flag_break_: break

            if not use_filter_idxs_for_recursive or not filter_idxs or node_rel_idx in filter_idxs:
                for rel_idx in rel_idxs.keys():
                    if 'depends' in rel_idx:
                        if '_and' in rel_idx and (node_rel_idx not in pattern.keys() or pattern[node_rel_idx] == 'off'):
                            if not finded_opened_idxs or node_rel_idx not in finded_opened_idxs:
                                finded_idxs.add(node_rel_idx)

                        elif '_or' in rel_idx or '_xor' in rel_idx:
                            can_add_or_ = True
                            if node_rel_idx in pattern.keys() and pattern[node_rel_idx] == 'on' or \
                                finded_opened_idxs and node_rel_idx in finded_opened_idxs:
                                can_add_or_ = False
                                flag_break_ = True
                                break

        if can_add_or_:
            finded_idxs.update(set([idx for idx in data_dict[node_idx].relation['in'].keys() \
                                    if 'depends_or'  in data_dict[node_idx].relation['in'][idx] or \
                                       'depends_xor' in data_dict[node_idx].relation['in'][idx]]))

        if recursive:
            for node_idx_ in [idx for idx in finded_idxs]:
                if node_idx_ not in checked_nodes:
                    recursive_on = False

                    flag_break_ = False
                    for node_rel_idx, rel_idxs in data_dict[node_idx_].relation['in'].items():
                        if flag_break_: break

                        for rel_idx in rel_idxs.keys():
                            if 'depends' in rel_idx:
                                recursive_on = True
                                flag_break_ = True
                                break

                    if recursive_on:
                        finded_idxs.update(self.find_back_pattern_idxs(node_idx_, pattern, filter_idxs=filter_idxs, recursive=recursive,
                                                                       use_filter_idxs_for_recursive=use_filter_idxs_for_recursive,
                                                                       finded_opened_idxs=finded_opened_idxs, checked_nodes=checked_nodes, root_=False))

        if use_filter_idxs_after and root_:
            finded_idxs = set([idx for idx in finded_idxs if idx in filter_idxs])

        return finded_idxs

    @caching_profile
    def find_structure_idxs(self, finded_idxs: set, dir_rel: str='in', root_idx: str=None, **kwargs) -> dict:
        finded_idxs_ = set(finded_idxs)
        finded_idxs_dict = {}
        counter = 0

        if root_idx:
            finded_idxs_dict[counter + 1] = {root_idx: {}}

        while True:
            counter += 1
            finded_idxs_dict[counter] = {}
            checked_false_nodes = {}
            finded_idxs_new = set(finded_idxs_)

            for node_idx in finded_idxs_:
                tree_root = True
                node_relations = self._net[node_idx].relation[dir_rel].items()

                for node_rel_idx, rel_idxs in node_relations:
                    if node_rel_idx in finded_idxs_ and ('depends_or' in rel_idxs.keys() or  'depends_and' in rel_idxs.keys()) and \
                        (node_rel_idx not in checked_false_nodes.keys() or checked_false_nodes[node_rel_idx] != {node_idx: {}}): ## need better cycle checker

                        if node_idx not in checked_false_nodes.keys():
                            checked_false_nodes[node_idx] = {node_rel_idx: {}}
                        else: checked_false_nodes[node_idx][node_rel_idx] = {}

                        tree_root = False
                        break

                if tree_root:
                    finded_idxs_dict[counter][node_idx] = {node_rel_idx: {} for node_rel_idx, rel_idxs in node_relations \
                                                           if node_rel_idx in finded_idxs and ('depends_or' in rel_idxs.keys() or 'depends_and' in rel_idxs.keys())}
                    finded_idxs_new.remove(node_idx)

            if len(finded_idxs_new) == 0: break
            else: finded_idxs_ = finded_idxs_new

        return finded_idxs_dict

    @caching_profile
    def find_shortest_dist_z(self, data_dict: dict=None, find_idxs: set=None, filter_idxs: set=None, **kwargs) -> dict:
        dist_nodes = {}

        if not data_dict: data_dict = self._net

        for node_idx_init in data_dict.keys() if not find_idxs and not filter_idxs else find_idxs if find_idxs else filter_idxs:
            if node_idx_init not in dist_nodes.keys():
                dist_nodes[node_idx_init] = {}

            node_idxs_with_lvls = {0: [node_idx_init]}
            checked_nodes_forward = set([])
            checked_nodes_backward = set([])
            flag_begin_ = True
            last_chance_ = True

            while True:
                if not node_idxs_with_lvls:
                    if last_chance_:
                        last_chance_ = False
                        flag_begin_ = True
                        node_idxs_with_lvls = {0: [node_idx_init]}
                    else: break

                for dir_rel in ['out', 'in']:
                    si_corr = {}

                    prev_idxs = set([])
                    for val in node_idxs_with_lvls.values():
                        prev_idxs.update(val)

                    for lvl_init, node_idxs in node_idxs_with_lvls.items():
                        for node_idx in node_idxs:
                            if node_idx != node_idx_init or flag_begin_:
                                struct_idxs = self.find_all_dir_rel_idxs_lvls(node_idx, dir_rel=dir_rel, filter_idxs=filter_idxs, lvl=lvl_init) #??? .difference(prev_idxs)

                                for idx, lvl in struct_idxs.items():
                                    if idx not in checked_nodes_forward and dir_rel=='out' or idx not in checked_nodes_backward and dir_rel=='in':
                                        if idx != node_idx_init or flag_begin_:
                                            if dir_rel=='out': checked_nodes_forward.add(idx)
                                            elif dir_rel=='in': checked_nodes_backward.add(idx)

                                            lvl_send = lvl

                                            if idx not in dist_nodes[node_idx_init].keys():
                                                dist_nodes[node_idx_init][idx] = lvl

                                            elif dist_nodes[node_idx_init][idx] > lvl:
                                                dist_nodes[node_idx_init][idx] = lvl
                                                checked_nodes_forward = set([])
                                                checked_nodes_backward = set([])
                                            else: lvl_send = dist_nodes[node_idx_init][idx]

                                            if idx not in dist_nodes.keys():
                                                dist_nodes[idx] = {node_idx_init: lvl}
                                            elif node_idx_init not in dist_nodes[idx].keys():
                                                dist_nodes[idx][node_idx_init] = lvl
                                            elif dist_nodes[idx][node_idx_init] > lvl:
                                                dist_nodes[idx][node_idx_init] = lvl
                                                checked_nodes_forward = set([])
                                                checked_nodes_backward = set([])

                                            if lvl_send not in si_corr.keys():
                                                si_corr[lvl_send] = set([idx])
                                            else:
                                                si_corr[lvl_send].add(idx) #???

                    node_idxs_with_lvls = si_corr
                flag_begin_ = False

        return dist_nodes

    @caching_profile
    def find_shortest_dist_dijkstra_amx(self, data_dict: dict=None, find_idxs: set=None, filter_idxs: set=None, **kwargs):
        dist_nodes = {}

        if not data_dict: data_dict = self._net

        amx, net_idx_to_amx_idx = self.get_adjacency_matrix(data_dict=data_dict, find_idxs=find_idxs, filter_idxs=filter_idxs)
        amx_idx_to_net_idx = {val: idx for idx, val in net_idx_to_amx_idx.items()}

        for node_idx in data_dict.keys() if not find_idxs and not filter_idxs else find_idxs if find_idxs else filter_idxs:
            row = len(amx)
            col = len(amx[0])

            dist    = [float("Inf") for i in range(row)]
            parent  = [-1 for i in range(row)]
            queue   = [i for i in range(row)]

            if node_idx not in dist_nodes.keys():
                dist_nodes[node_idx] = {node_idx: 0}
            else: dist_nodes[node_idx][node_idx] = 0

            dist[net_idx_to_amx_idx[node_idx]] = 0

            while queue:
                minimum = float("Inf")
                min_index = -1

                for i in range(len(dist)):
                    if dist[i] < minimum and i in queue:
                        minimum = dist[i]
                        min_index = i

                if min_index == -1:
                    break

                queue.remove(min_index)

                for i in range(col):
                    if amx[min_index][i] and i in queue:
                        if dist[min_index] + amx[min_index][i] < dist[i]:
                            dist[i] = dist[min_index] + amx[min_index][i]
                            parent[i] = min_index

            for i, dist_val in enumerate(dist):
                if dist_val != 0 and dist_val != float("Inf"):
                    if node_idx not in dist_nodes.keys():
                        dist_nodes[node_idx] = {amx_idx_to_net_idx[i]: dist_val}
                    else: dist_nodes[node_idx][amx_idx_to_net_idx[i]] = dist_val

        return dist_nodes

    @caching_profile
    def find_shortest_dist_dijkstra_als(self, data_dict: dict=None, find_idxs: set=None, filter_idxs: set=None, **kwargs):
        dist_nodes = {}

        if not data_dict: data_dict = self._net

        als, net_idx_to_als_idx = self.get_adjacency_list(data_dict=data_dict, find_idxs=find_idxs, filter_idxs=filter_idxs)
        als_idx_to_net_idx = {val: idx for idx, val in net_idx_to_als_idx.items()}
        V = len(als)
        max_val = np.uint32(-1)

        for node_idx in data_dict.keys() if not find_idxs and not filter_idxs else find_idxs if find_idxs else filter_idxs:
            if node_idx not in dist_nodes.keys():
                dist_nodes[node_idx] = {node_idx: 0}
            else: dist_nodes[node_idx][node_idx] = 0

            node_idx_als = net_idx_to_als_idx[node_idx]
            dist = []

            minHeap = Heap()

            for v in range(V):
                dist.append(max_val)
                minHeap.array.append(minHeap.newMinHeapNode(v, dist[v]))
                minHeap.pos.append(v)

            minHeap.pos[node_idx_als] = node_idx_als
            dist[node_idx_als] = 0
            minHeap.decreaseKey(node_idx_als, dist[node_idx_als])

            minHeap.size = V

            while minHeap.isEmpty() == False:
                newHeapNode = minHeap.extractMin()
                u = newHeapNode[0]

                for pCrawl in als[u]:
                    v = pCrawl[0]

                    if minHeap.isInMinHeap(v) and  dist[u] != max_val and pCrawl[1] + dist[u] < dist[v]:
                        dist[v] = pCrawl[1] + dist[u]
                        minHeap.decreaseKey(v, dist[v])

            for i, dist_val in enumerate(dist):
                if dist_val != 0 and dist_val != max_val:
                    if node_idx not in dist_nodes.keys():
                        dist_nodes[node_idx] = {als_idx_to_net_idx[i]: dist_val}
                    else: dist_nodes[node_idx][als_idx_to_net_idx[i]] = dist_val

        return dist_nodes

    @caching_profile
    def find_shortest_dist_DAG(self, data_dict: dict=None, find_idxs: set=None, filter_idxs: set=None, **kwargs):
        def topologicalSortUtil(graph, v, visited, stack):
            visited[v] = True

            if v in graph.keys():
                for node, weight in graph[v]:
                    if visited[node] == False:
                        topologicalSortUtil(graph, node, visited, stack)

            stack.append(v)
        dist_nodes = {}

        if not data_dict: data_dict = self._net

        als, net_idx_to_als_idx = self.get_adjacency_list(data_dict=data_dict, find_idxs=find_idxs, filter_idxs=filter_idxs)
        als_idx_to_net_idx = {val: idx for idx, val in net_idx_to_als_idx.items()}
        V = len(als)

        for node_idx in data_dict.keys() if not find_idxs and not filter_idxs else find_idxs if find_idxs else filter_idxs:
            if node_idx not in dist_nodes.keys():
                dist_nodes[node_idx] = {node_idx: 0}
            else: dist_nodes[node_idx][node_idx] = 0

            node_idx_als = net_idx_to_als_idx[node_idx]
            visited = [False]*V
            stack = []

            for i in range(V):
                if visited[i] == False:
                    topologicalSortUtil(als, node_idx_als, visited, stack)

            dist = [float("Inf")] * (V)
            dist[node_idx_als] = 0

            while stack:
                i = stack.pop()

                for node,weight in als[i]:
                    if dist[node] > dist[i] + weight:
                        dist[node] = dist[i] + weight

            for i, dist_val in enumerate(dist):
                if dist_val != 0 and dist_val != float("Inf"):
                    if node_idx not in dist_nodes.keys():
                        dist_nodes[node_idx] = {als_idx_to_net_idx[i]: dist_val}
                    else: dist_nodes[node_idx][als_idx_to_net_idx[i]] = dist_val

        return dist_nodes

    @caching_profile
    def find_shortest_dist_fw(self, data_dict: dict=None, find_idxs: set=None, filter_idxs: set=None, **kwargs):
        dist_nodes = {}

        if not data_dict: data_dict = self._net

        max_val = np.uint32(-1)

        amx, net_idx_to_amx_idx = self.get_adjacency_matrix(data_dict=data_dict, find_idxs=find_idxs, filter_idxs=filter_idxs, replace_zero=max_val)
        amx_idx_to_net_idx = {val: idx for idx, val in net_idx_to_amx_idx.items()}

        dist = list(map(lambda i: list(map(lambda j: j, i)), amx))

        V = len(amx)

        for k in range(V):
            for i in range(V):
                for j in range(V):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

        for i, line in enumerate(dist):
            node_idx = amx_idx_to_net_idx[i]
            for j, dist_val in enumerate(line):
                if dist_val == 0 and i == j or dist_val != 0 and dist_val != max_val:
                    if node_idx not in dist_nodes.keys():
                        dist_nodes[amx_idx_to_net_idx[i]] = {amx_idx_to_net_idx[j]: dist_val}
                    else: dist_nodes[amx_idx_to_net_idx[i]][amx_idx_to_net_idx[j]] = dist_val

        return dist_nodes

    @caching_profile
    def find(self, pattern: dict, **kwargs) -> dict:
        finded_idxs = set([])
        finded_idxs_dict = {}
        finded_shortest_dist = {}
        search_finded_idxs = set([])

        value_in_task           = False
        relation_in_task        = False
        pattern_in_task         = False
        back_pattern_in_task    = False

        if 'value' in pattern.keys() and 'task' in pattern['value'].keys():
            value_in_task       = True
            contains            = False

            if 'settings' in pattern['value'].keys():
                if 'contains' in pattern['value']['settings'].keys():
                    contains = pattern['value']['settings']['contains']

            finded_idxs = self.find_val_idxs(pattern['value']['task'], contains=contains)
            search_finded_idxs = set(finded_idxs)

        if 'relation' in pattern.keys() and 'task' in pattern['relation'].keys():
            relation_in_task    = True
            and_on              = True
            recursive           = False

            if 'settings' in pattern['relation'].keys():
                if 'and_on' in pattern['relation']['settings'].keys():
                    and_on = pattern['relation']['settings']['and_on']
                if 'recursive' in pattern['relation']['settings'].keys():
                    recursive = pattern['relation']['settings']['recursive']

            finded_idxs = self.find_mult_rel_idxs(pattern['relation']['task'], and_on=and_on, recursive=recursive, filter_idxs=finded_idxs)
            search_finded_idxs = set(finded_idxs)

        if 'pattern' in pattern.keys() and 'task' in pattern['pattern'].keys():
            pattern_in_task         = True
            tree_root               = True
            recursive               = False
            use_filter_idxs_after   = False
            use_filter_idxs_for_recursive = True

            if 'settings' in pattern['pattern'].keys():
                if 'use_filter_idxs_for_recursive' in pattern['pattern']['settings'].keys():
                    use_filter_idxs_for_recursive = pattern['pattern']['settings']['use_filter_idxs_for_recursive']
                if 'use_filter_idxs_after' in pattern['pattern']['settings'].keys():
                    use_filter_idxs_after = pattern['pattern']['settings']['use_filter_idxs_after']
                if 'tree_root' in pattern['pattern']['settings'].keys():
                    tree_root = pattern['pattern']['settings']['tree_root']
                if 'recursive' in pattern['pattern']['settings'].keys():
                    recursive = pattern['pattern']['settings']['recursive']

            if finded_idxs:
                finded_idxs = self.find_pattern_idxs(pattern['pattern']['task'], filter_idxs=finded_idxs,
                                                     use_filter_idxs_for_recursive=use_filter_idxs_for_recursive,
                                                     use_filter_idxs_after=use_filter_idxs_after,
                                                     recursive=recursive, tree_root=tree_root)
            else: finded_idxs = self.find_pattern_idxs(pattern['pattern']['task'], recursive=recursive, tree_root=tree_root)

        elif 'back_pattern' in pattern.keys() and 'task' in pattern['back_pattern'].keys():
            back_pattern_in_task    = True
            clear_opened            = True
            recursive               = False
            use_filter_idxs_after   = False
            use_filter_idxs_for_recursive = True

            if 'settings' in pattern['back_pattern'].keys():
                if 'use_filter_idxs_for_recursive' in pattern['back_pattern']['settings'].keys():
                    use_filter_idxs_for_recursive = pattern['back_pattern']['settings']['use_filter_idxs_for_recursive']
                if 'use_filter_idxs_after' in pattern['back_pattern']['settings'].keys():
                    use_filter_idxs_after = pattern['back_pattern']['settings']['use_filter_idxs_after']
                if 'clear_opened' in pattern['back_pattern']['settings'].keys():
                    clear_opened = pattern['back_pattern']['settings']['clear_opened']
                if 'recursive' in pattern['back_pattern']['settings'].keys():
                    recursive = pattern['back_pattern']['settings']['recursive']

            if finded_idxs:
                finded_idxs = self.find_back_pattern_idxs(pattern['back_pattern']['task']['root_idx'],
                                                          pattern['back_pattern']['task']['pattern'],
                                                          filter_idxs=finded_idxs,
                                                          use_filter_idxs_for_recursive=use_filter_idxs_for_recursive,
                                                          use_filter_idxs_after=use_filter_idxs_after,
                                                          recursive=recursive, clear_opened=clear_opened)

            else: finded_idxs = self.find_back_pattern_idxs(pattern['back_pattern']['task']['root_idx'],
                                                            pattern['back_pattern']['task']['pattern'],
                                                            recursive=recursive, clear_opened=clear_opened)

        if 'end_filter' in pattern.keys() and 'task' in pattern['end_filter'].keys():
            exclude = False

            if 'settings' in pattern['end_filter'].keys():
                if 'exclude' in pattern['end_filter']['settings'].keys():
                    exclude = pattern['end_filter']['settings']['exclude']

            if 'value' in pattern['end_filter']['task'].keys() and 'task' in pattern['end_filter']['task']['value'].keys():
                contains            = False

                if 'settings' in pattern['end_filter']['task']['value'].keys():
                    if 'contains' in pattern['end_filter']['task']['value']['settings'].keys():
                        contains = pattern['end_filter']['task']['value']['settings']['contains']

                finded_idxs_ = self.find_val_idxs(pattern['end_filter']['task']['value']['task'], contains=contains)
                finded_idxs = set([idx for idx in finded_idxs if (not exclude and idx in finded_idxs_) or (exclude and idx not in finded_idxs_)])

            elif 'relation' in pattern['end_filter']['task'].keys() and 'task' in pattern['end_filter']['task']['relation'].keys():
                and_on              = True
                recursive           = False

                if 'settings' in pattern['end_filter']['task']['relation'].keys():
                    if 'and_on' in pattern['end_filter']['task']['relation']['settings'].keys():
                        and_on = pattern['end_filter']['task']['relation']['settings']['and_on']
                    if 'recursive' in pattern['end_filter']['task']['relation']['settings'].keys():
                        recursive = pattern['end_filter']['task']['relation']['settings']['recursive']

                finded_idxs_ = self.find_mult_rel_idxs(pattern['end_filter']['task']['relation']['task'], and_on=and_on, recursive=recursive)
                finded_idxs = set([idx for idx in finded_idxs if (not exclude and idx in finded_idxs_) or (exclude and idx not in finded_idxs_)])

        if 'shortest_dist' in pattern.keys() and 'task' in pattern['shortest_dist'].keys():
            if pattern['shortest_dist'] == "z":
                finded_shortest_dist = self.find_shortest_dist_z(filter_idxs=finded_idxs)
            elif pattern['shortest_dist'] == "dijkstra_amx":
                finded_shortest_dist = self.find_shortest_dist_dijkstra_amx(filter_idxs=finded_idxs)
            elif pattern['shortest_dist'] == "dijkstra_als":
                finded_shortest_dist = self.find_shortest_dist_dijkstra_als(filter_idxs=finded_idxs)
            elif pattern['shortest_dist'] == "fw":
                finded_shortest_dist = self.find_shortest_dist_fw(filter_idxs=finded_idxs)
            elif pattern['shortest_dist'] == "DAG":
                finded_shortest_dist = self.find_shortest_dist_DAG(filter_idxs=finded_idxs)
            else: finded_shortest_dist = self.find_shortest_dist_DAG(filter_idxs=finded_idxs)

        sum_of_values = None

        if 'sum_of_values' in pattern.keys() and 'task' in pattern['sum_of_values'].keys():
            if finded_idxs:
                sum_of_values = self.get_sum_of_values(finded_idxs)
            else: sum_of_values = 0

        if  'settings' in pattern.keys() and 'return_structure_dict' in pattern['settings'].keys() and pattern['settings']['return_structure_dict']:
            finded_idxs_dict = self.find_structure_idxs(finded_idxs)

        add_in_type_relation_idxs_dict = {}

        if 'add_in_type_relation' in pattern.keys() and 'task' in pattern['add_in_type_relation'].keys():
            if 'settings' in pattern.keys():
                pattern['settings']['type_output'] = 'full'
            else: pattern['settings'] = {'type_output': 'full'}

            add_in_type_relation_task = pattern['add_in_type_relation']['task']
            if type(pattern['add_in_type_relation']['task']) == str:
                add_in_type_relation_task = [pattern['add_in_type_relation']['task']]

            for rel_idx in add_in_type_relation_task:
                finded_idxs_ = self.find_type_rel_idxs(rel_idx)

                for node_idx in finded_idxs_:
                    if (not back_pattern_in_task and node_idx in finded_idxs and (not pattern_in_task or pattern_in_task and node_idx not in pattern['pattern']['task'].keys())) or \
                        set([node_rel_idx for node_rel_idx, rel_idxs in self._net[node_idx].relation['in'].items() if rel_idx in rel_idxs.keys()]).intersection(finded_idxs) or \
                        (back_pattern_in_task and node_idx in finded_idxs):

                        add_in_type_relation_idxs_dict[node_idx] = {
                            'value': self._net[node_idx].value,
                            'items': {rel_idx: {}}
                        }

                        for node_rel_idx, rel_idxs in self._net[node_idx].relation['in'].items():
                            if rel_idx in rel_idxs:
                                add_in_type_relation_idxs_dict[node_idx]['items'][rel_idx][node_rel_idx] = {"value": self._net[node_rel_idx].value}

                                if node_rel_idx in finded_idxs_dict.keys():
                                    del finded_idxs_dict[node_rel_idx]
                                if node_rel_idx in finded_idxs:
                                    finded_idxs.discard(node_rel_idx)

        sorted_idxs = []
        if 'sort' in pattern.keys() and 'task' in pattern['sort'].keys() and (len(finded_idxs) != 0 or add_in_type_relation_idxs_dict != {}):
            fx = None
            divide_U = False
            common_filter_idxs = search_finded_idxs if search_finded_idxs else finded_idxs

            if 'settings' in pattern['sort'].keys():
                if 'fx' in pattern['sort']['settings'].keys():
                    fx = pattern['sort']['settings']['fx']
                if 'divide_U' in pattern['sort']['settings'].keys():
                    divide_U = pattern['sort']['settings']['divide_U']
                if 'common_filter_idxs' in pattern['sort']['settings'].keys():
                    common_filter_idxs =  pattern['sort']['settings']['common_filter_idxs']

            sorted_idxs = self.sort({*finded_idxs, *add_in_type_relation_idxs_dict}, pattern['sort']['task'], fx=fx, divide_U=divide_U, common_filter_idxs=common_filter_idxs)

        res_msg = {}

        if 'settings' in pattern.keys():
            add_props_idxs = {}

            if 'add_props_output' in pattern['settings'].keys():
                add_props_idxs = pattern['settings']['add_props_output']

            if 'type_output' in pattern['settings'].keys() or add_props_idxs:
                if not add_props_idxs and pattern['settings']['type_output'] == 'short':
                    res_msg = {'nodes': {idx: self._net[idx].value for idx in finded_idxs}}

                elif add_props_idxs or pattern['settings']['type_output'] == 'full':
                    res_msg = {
                        'nodes': {
                            **{
                                idx: {
                                    "idx": idx,
                                    "value": self._net[idx].value,
                                    **{prop: self.get_value_rel(idx, idx_prop) for prop, idx_prop in add_props_idxs.items()}
                                } for idx in finded_idxs
                            },
                            **add_in_type_relation_idxs_dict
                        }
                    }

        if res_msg == {}:
            res_msg = {'nodes': {idx: self._net[idx].value for idx in finded_idxs}}

        if finded_idxs_dict != {}:
            res_msg['structure'] = finded_idxs_dict

        if finded_shortest_dist != {}:
            res_msg['shortest_dist'] = finded_shortest_dist

        if sum_of_values != None:
            res_msg['sum_of_values'] = sum_of_values

        if len(sorted_idxs) > 0:
            res_msg['sort'] = sorted_idxs

        return res_msg

    @caching_profile
    def sort(self, finded_idxs: set, rule_sort: dict, fx=None, filter_idxs: set=None, return_divided_map: bool=False, divide_U: bool=True, common_filter_idxs: set=None, **kwargs) -> Union[list, Tuple[list, dict]]:
        sorted_idxs = []
        divided_map = []

        if 'relation' in rule_sort.keys():
            for dir_rel in ['in', 'out']:   # поддержка и того и другого одновременно?
                if dir_rel in rule_sort['relation'].keys():
                    for node_rel_idx, rel_type in rule_sort['relation'][dir_rel]:  # будет ли поддерживать на самом деле несколько индексов?
                        sorted_idxs.extend([[idx, self._net[idx].relation[dir_rel][node_rel_idx][rel_type] if fx == None else fx(self._net[idx].relation[dir_rel][node_rel_idx][rel_type])] for idx in finded_idxs \
                                             if node_rel_idx in self._net[idx].relation[dir_rel].keys() and \
                                                rel_type in self._net[idx].relation[dir_rel][node_rel_idx].keys()])  # какая политика если всё-таки нет соотв узал? ведь тогда элемент пропадёт

        elif 'recursive_relation' in rule_sort.keys():
            sorted_idxs = [[idx, val if fx == None else fx(val)] for idx, val in self.find_and_combine_idxs_type_rel_coeff(finded_idxs, rule_sort['recursive_relation']).items()]

        elif 'value_recursive_relation' in rule_sort.keys():
            if not return_divided_map:
                sorted_idxs = [[idx, val if fx == None else fx(val)] for idx, val in self.find_and_combine_idxs_val_rel(finded_idxs, rule_sort['value_recursive_relation'], filter_idxs=filter_idxs, divide_U=divide_U, common_filter_idxs=common_filter_idxs).items()]
            else:
                dict_idxs, divided_map = self.find_and_combine_idxs_val_rel(finded_idxs, rule_sort['value_recursive_relation'], filter_idxs=filter_idxs, return_divided_map=return_divided_map, divide_U=divide_U, common_filter_idxs=common_filter_idxs)
                sorted_idxs = [[idx, val if fx == None else fx(val)] for idx, val in dict_idxs.items()]

        elif 'value_relation' in rule_sort.keys():
            sorted_idxs = [[idx, val if fx == None else fx(val)] for idx, val in self.find_idxs_val_rel_coeff(rule_sort['value_relation'], filter_idxs=finded_idxs).items()]

        sorted_idxs.sort(key=itemgetter(1), reverse=True)

        if not return_divided_map:
            return sorted_idxs
        else: return sorted_idxs, divided_map
