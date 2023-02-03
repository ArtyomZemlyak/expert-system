# distutils: language=c++
# cython: language_level=3

import json
import hashlib
from typing import Union
import uuid


class DBESNodeJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, DBESNode):
            return obj.toJSON()
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class DBESNode:

    def __init__(self, value: Union[dict, set, list, str, int, float, bool]=None, relation: dict=None, idx: str=None):
        self.__hash_idx         = None
        self.__idx              = None
        self.__value            = None
        self.__relation         = None
        self.value          = value if value else ""
        self.relation       = relation if relation else {'in': {}, 'out':{}}

        if idx: self.idx = idx
        else: self._generate_idx()

    @property
    def idx(self)       -> str:     return str(self.__idx)
    @property
    def value(self)     -> Union[dict, set, list, str, int, float, bool]:    return self.__value
    @property
    def relation(self)  -> dict:    return self.__relation

    @idx.setter
    def idx(self, idx: str):
        if type(idx) != str: raise TypeError('DBESNode.idx: str type!')
        self.__idx = idx
        self.__hash_idx = int(self.to_idx(idx), base=16)
    @value.setter
    def value(self, value: Union[dict, set, list, str, int, float, bool]):
        if type(value) not in [dict, set, list, str, int, float, bool]:
            if value: raise TypeError('DBESNode.value: Union[dict, set, list, str, int, float, bool] type!')
        self.__value = value
    @relation.setter
    def relation(self, relation: dict):
        if type(relation) != dict:
            raise TypeError('DBESNode.relation: dict type!')
        elif 'in' not in relation.keys() and 'out' not in relation.keys():
            raise ValueError('DBESNode.relation: -in- and -out- fields must specified in input dict!')
        self.__relation = relation

    def __repr__(self)      -> str:     return self.idx
    def __str__(self)       -> str:     return self.pretty_str()
    def __hash__(self)      -> int:     return self.__hash_idx
    def __eq__(self, other) -> bool:    return self.idx == other.idx
    def __lt__(self, other) -> bool:    return self.idx <  other.idx
    def __le__(self, other) -> bool:    return self.idx <= other.idx
    def __ne__(self, other) -> bool:    return self.idx != other.idx
    def __gt__(self, other) -> bool:    return self.idx >  other.idx
    def __ge__(self, other) -> bool:    return self.idx >= other.idx

    def _generate_idx(self):
        self.idx = str(uuid.uuid4())

    @staticmethod
    def generate_idx() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def to_idx(string: str) -> str:
        return hashlib.md5(string.encode('utf8')).hexdigest()

    @classmethod
    def from_dict(cls, data_dict):
        if type(data_dict) != dict:
            raise TypeError('DBESNode.from_dict can get only dict type!')

        dbes_tag = cls(idx      = data_dict['idx']      if 'idx'        in data_dict.keys() else None,
                       value    = data_dict['value']    if 'value'      in data_dict.keys() else None,
                       relation = data_dict['relation'] if 'relation'   in data_dict.keys() else {})

        return dbes_tag

    def update_value(self, value: Union[dict, set, list, str, int, float, bool]) -> None:
        if type(self.value) == dict and type(value) == dict:
            self.value = {**self.value, **value}

        elif type(self.value) == set:
            if   type(value) == set:  self.value.update(value)
            elif type(value) == list: self.value.update(set(value))
            elif type(value) == str:  self.value.add(value)
            elif type(value) == int:  self.value.add(value)
            else: raise TypeError('DBESNode.update_value - wrong type!')

        elif type(self.value) == list:
            if   type(value) == set:  self.value.extend(value)
            elif type(value) == list: self.value.extend(value)
            elif type(value) == str:  self.value.append(value)
            elif type(value) == int:  self.value.append(value)
            else: raise TypeError('DBESNode.update_value - wrong type!')

        elif type(self.value) == str:
            if   type(value) == str:  self.value = value
            else: raise TypeError('DBESNode.update_value - wrong type!')

        elif type(self.value) == int:
            if   type(value) == int:  self.value = value
            else: raise TypeError('DBESNode.update_value - wrong type!')

        elif type(self.value) == float:
            if   type(value) == float:  self.value = value
            else: raise TypeError('DBESNode.update_value - wrong type!')

        elif type(self.value) == bool:
            if   type(value) == bool:  self.value = value
            else: raise TypeError('DBESNode.update_value - wrong type!')

        elif not self.value and type(value) in [dict, set, list, str, int, float, bool]:
            self.value = value

        else: raise TypeError('DBESNode.update_value - wrong type!')

    def add_relation(self, other_idx: str, rel_idx: str, coeff: Union[str, int, float]=None, type_rel: str='out') -> None:
        if type_rel == 'in' or type_rel == 'out':
            if other_idx not in self.relation[type_rel].keys():
                if coeff: self.relation[type_rel][other_idx] = {rel_idx: coeff}
                else: self.relation[type_rel][other_idx] = {rel_idx: 1}

            else:
                if rel_idx not in self.relation[type_rel][other_idx].keys():
                    if coeff: self.relation[type_rel][other_idx][rel_idx] = coeff
                    else: self.relation[type_rel][other_idx][rel_idx] = 1

                else:
                    if coeff: self.relation[type_rel][other_idx][rel_idx] = coeff
                    else: self.relation[type_rel][other_idx][rel_idx] += 1

    def remove_relation(self, other_idx: str, rel_idx: str=None, type_rel: str='out') -> None:
        if type_rel == 'in' or type_rel == 'out':
            if other_idx in self.relation[type_rel].keys():
                if rel_idx and rel_idx in self.relation[type_rel][other_idx].keys():
                    del self.relation[type_rel][other_idx][rel_idx]

                    if self.relation[type_rel][other_idx] == {}:
                        del self.relation[type_rel][other_idx]

                else:
                    del self.relation[type_rel][other_idx]

    def toJSON(self) -> dict:
        self_dict = {}

        for key, value in self.__dict__.items():
            if type(value) == int:
                value = str(value)
            if type(value) == set:
                value = list(value)
            self_dict[key.split('__')[-1]] = value

        return self_dict
