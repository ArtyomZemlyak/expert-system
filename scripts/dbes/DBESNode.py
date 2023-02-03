import json
import hashlib
from typing import Union
import uuid
import copy


DIR_RELATIONS = ["in", "out", "bi", "none"]
DIR_RELATIONS_REV = ["out", "in", "bi", "none"]
DIR_RELATIONS_NON_SYM = ["out", "bi", "none"]
DIR_RELATIONS_OD = ["in", "out"]
DIR_RELATIONS_BD = ["bi", "none"]
DIR_RELATIONS_DICT = {"in": {}, "out": {}, "bi": {}, "none": {}}
DIR_RELATIONS_REV_DICT = {"in": "out", "out": "in", "bi": "bi", "none": "none"}


class DBESNodeJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, DBESNode):
            return obj.toJSON()
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class DBESNode:
    def __init__(
        self,
        value: Union[dict, set, list, str, int, float, bool] = None,
        relation: dict = None,
        idx: str = None,
    ):
        self.__hash_idx = None
        self.__idx = None
        self.__value = None
        self.__relation = None
        self.value = value if value else ""
        self.relation = (
            relation if relation else {"in": {}, "out": {}, "bi": {}, "none": {}}
        )

        if idx:
            self.idx = idx
        else:
            self._generate_idx()

    @property
    def idx(self) -> str:
        return str(self.__idx)

    @property
    def value(self) -> Union[dict, set, list, str, int, float, bool]:
        return self.__value

    @property
    def relation(self) -> dict:
        return self.__relation

    @idx.setter
    def idx(self, idx: str):
        if type(idx) != str:
            raise TypeError("DBESNode.idx: str type!")
        self.__idx = idx
        self.__hash_idx = int(self.to_idx(idx), base=16)

    @value.setter
    def value(self, value: Union[dict, set, list, str, int, float, bool]):
        if type(value) not in [dict, set, list, str, int, float, bool]:
            if value:
                raise TypeError(
                    "DBESNode.value: Union[dict, set, list, str, int, float, bool] type!"
                )
        self.__value = value

    @relation.setter
    def relation(self, relation: dict):
        relation = copy.deepcopy({**relation})
        if type(relation) != dict:
            raise TypeError("DBESNode.relation: dict type!")
        if "in" not in relation:
            relation["in"] = {}
        if "out" not in relation:
            relation["out"] = {}
        if "bi" not in relation:
            relation["bi"] = {}
        if "none" not in relation:
            relation["none"] = {}
        self.__relation = relation

    def __repr__(self) -> str:
        return self.idx

    def __str__(self) -> str:
        return self.pretty_str()

    def __hash__(self) -> int:
        return self.__hash_idx

    def __eq__(self, other) -> bool:
        return self.idx == other.idx

    def __lt__(self, other) -> bool:
        return self.idx < other.idx

    def __le__(self, other) -> bool:
        return self.idx <= other.idx

    def __ne__(self, other) -> bool:
        return self.idx != other.idx

    def __gt__(self, other) -> bool:
        return self.idx > other.idx

    def __ge__(self, other) -> bool:
        return self.idx >= other.idx

    def _generate_idx(self):
        self.idx = self.to_idx(str(uuid.uuid4()))

    @staticmethod
    def generate_idx() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def to_idx(string: str) -> str:
        return hashlib.md5(string.encode("utf8")).hexdigest()

    @classmethod
    def from_dict(cls, data_dict):
        if type(data_dict) != dict:
            raise TypeError("DBESNode.from_dict can get only dict type!")

        dbes_tag = cls(
            idx=data_dict["idx"] if "idx" in data_dict.keys() else None,
            value=data_dict["value"] if "value" in data_dict.keys() else None,
            relation=data_dict["relation"] if "relation" in data_dict.keys() else {},
        )

        return dbes_tag

    def update_value(
        self, value: Union[dict, set, list, str, int, float, bool]
    ) -> None:
        if type(self.value) == dict and type(value) == dict:
            self.value = {**self.value, **value}

        elif type(self.value) == set:
            if type(value) == set:
                self.value.update(value)
            elif type(value) == list:
                self.value.update(set(value))
            elif type(value) == str:
                self.value.add(value)
            elif type(value) == int:
                self.value.add(value)
            else:
                raise TypeError("DBESNode.update_value - wrong type!")

        elif type(self.value) == list:
            if type(value) == set:
                self.value.extend(value)
            elif type(value) == list:
                self.value.extend(value)
            elif type(value) == str:
                self.value.append(value)
            elif type(value) == int:
                self.value.append(value)
            else:
                raise TypeError("DBESNode.update_value - wrong type!")

        elif type(self.value) == str:
            if type(value) == str:
                self.value = value
            else:
                raise TypeError("DBESNode.update_value - wrong type!")

        elif type(self.value) == int:
            if type(value) == int:
                self.value = value
            else:
                raise TypeError("DBESNode.update_value - wrong type!")

        elif type(self.value) == float:
            if type(value) == float:
                self.value = value
            else:
                raise TypeError("DBESNode.update_value - wrong type!")

        elif type(self.value) == bool:
            if type(value) == bool:
                self.value = value
            else:
                raise TypeError("DBESNode.update_value - wrong type!")

        elif not self.value and type(value) in [dict, set, list, str, int, float, bool]:
            self.value = value

        else:
            raise TypeError("DBESNode.update_value - wrong type!")

    def add_relation(
        self,
        other_idx: str,
        rel_idx: str,
        coeff: Union[str, int, float] = None,
        dir_rel: str = "out",
    ) -> None:
        if dir_rel == "in" or dir_rel == "out" or dir_rel == "bi" or dir_rel == "none":
            if other_idx not in self.relation[dir_rel]:
                if coeff:
                    self.relation[dir_rel][other_idx] = {rel_idx: coeff}
                else:
                    self.relation[dir_rel][other_idx] = {rel_idx: 1}

            else:
                if rel_idx not in self.relation[dir_rel][other_idx]:
                    if coeff:
                        self.relation[dir_rel][other_idx][rel_idx] = coeff
                    else:
                        self.relation[dir_rel][other_idx][rel_idx] = 1

                else:
                    if coeff:
                        self.relation[dir_rel][other_idx][rel_idx] = coeff
                    else:
                        self.relation[dir_rel][other_idx][rel_idx] += 1
        else:
            raise ValueError("Supported relations: in, out, bi, none!")

    def remove_relation(
        self, other_idx: str, rel_idx: str = None, dir_rel: str = "out"
    ) -> None:
        if dir_rel == "in" or dir_rel == "out" or dir_rel == "bi" or dir_rel == "none":
            if other_idx in self.relation[dir_rel]:
                if rel_idx and rel_idx in self.relation[dir_rel][other_idx]:
                    del self.relation[dir_rel][other_idx][rel_idx]

                    if self.relation[dir_rel][other_idx] == {}:
                        del self.relation[dir_rel][other_idx]

                else:
                    del self.relation[dir_rel][other_idx]
        else:
            raise ValueError("Supported relations: in, out, bi, none!")

    def toJSON(self) -> dict:
        self_dict = {}

        for key, value in self.__dict__.items():
            if type(value) == int:
                value = str(value)
            if type(value) == set:
                value = list(value)
            self_dict[key.split("__")[-1]] = value

        return self_dict

    def pretty_tag(self, tag=None) -> str:
        if type(tag) == DBESNode:
            return f"{tag.idx} ({tag.value})"
        elif type(tag) == dict:
            return f'{tag["idx"]} ({tag["value"]})'
        elif not tag:
            return f"{self.idx} ({self.value})"

    def pretty_str(self, new_line="\n", bold=""):
        if bold == "":
            bold = ["", ""]

        str_dd = f"{bold[0]}DBESNode{bold[1]}({self.idx}){new_line}"
        str_dd += f"{bold[0]}value{bold[1]}:        {self.value}{new_line}"
        str_dd += f"{bold[0]}relation{bold[1]}:     {self.relation}{new_line}"

        return str_dd


############################################################################################################################
# test:
############################################################################################################################
def test_main():
    from collections import Counter
    import time

    print("============================================")
    print("--TESTING")
    print("--------------------------------------------")
    print("--Creation----------------------------------")
    start_time = time.time()
    a = DBESNode()
    a = DBESNode(123)
    a = DBESNode("aaaaa")
    a = DBESNode([1, 2, 3])
    a = DBESNode({1, 2, "3"})
    a = DBESNode({1: 1, 2: "2", "3": 3})
    a = DBESNode(relation={"out": {"1": "22"}})
    a = DBESNode({1: 1, 2: "2", "3": 3}, {"in": {"1": "22"}})
    a = DBESNode({1: 1, 2: "2", "3": 3}, {"in": {"1": "22"}}, "435623456577")
    print(f"Create time: {time.time() - start_time}")
    print(f"Create time /9: {(time.time() - start_time) / 8}")
    print("\nstr(DBESNode)")
    print(str(a))

    print("--Update------------------------------------")
    start_time = time.time()
    a = DBESNode()
    a.update_value(123)
    assert a.value == 123
    a.update_value(234)
    assert a.value == 234
    try:
        a.update_value("aaaa")
        assert False
    except TypeError:
        assert True

    a = DBESNode()
    a.update_value("aaaa")
    assert a.value == "aaaa"
    a.update_value("bbbb")
    assert a.value == "bbbb"
    try:
        a.update_value(123)
        assert False
    except TypeError:
        assert True

    a = DBESNode()
    a.update_value([1, 2, 3])
    assert a.value == [1, 2, 3]
    a.update_value(4)
    assert a.value == [1, 2, 3, 4]
    a.update_value("5")
    assert a.value == [1, 2, 3, 4, "5"]
    a.update_value([7, 7])
    assert a.value == [1, 2, 3, 4, "5", 7, 7]
    a.update_value({8, 9})
    assert a.value == [1, 2, 3, 4, "5", 7, 7, 8, 9]
    try:
        a.update_value({1: 2})
        assert False
    except TypeError:
        assert True

    a = DBESNode()
    a.update_value({1, 2, 3})
    assert a.value == {1, 2, 3}
    a.update_value(4)
    assert a.value == {1, 2, 3, 4}
    a.update_value("5")
    assert a.value == {1, 2, 3, 4, "5"}
    a.update_value([7, 7])
    assert a.value == {1, 2, 3, 4, "5", 7}
    a.update_value({8, 9})
    assert a.value == {1, 2, 3, 4, "5", 7, 8, 9}
    try:
        a.update_value({1: 2})
        assert False
    except TypeError:
        assert True

    a = DBESNode()
    a.update_value({1: 2})
    a.update_value({3: 4})
    assert a.value == {1: 2, 3: 4}
    try:
        a.update_value(1)
        assert False
    except TypeError:
        assert True
    try:
        a.update_value("1")
        assert False
    except TypeError:
        assert True
    try:
        a.update_value([1, 2])
        assert False
    except TypeError:
        assert True
    try:
        a.update_value({1, 2})
        assert False
    except TypeError:
        assert True
    print(f"Update time: {time.time() - start_time}")
    print(a)

    print("--Relation----------------------------------")
    e = DBESNode(
        relation={
            "in": {"2141241": {"12412321": 1}},
            "out": {"2141241": {"41124123": 2}},
        }
    )
    e.add_relation("123", "3")
    e.add_relation("532", "5", "idx_134124")
    e.add_relation("623", "1", "idx_234234", "in")
    try:
        a.relation = {1: 2}
        assert False
    except ValueError:
        assert True
    print(e)


if __name__ == "__main__":
    test_main()
