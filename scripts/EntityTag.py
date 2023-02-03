import json
import hashlib
from typing import List, Set, Union

import networkx as nx


class EntityTagJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, EntityTag):
            return obj.toJSON()
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class EntityTag:
    def __init__(
        self,
        kind: set,
        tag: set,
        relation: dict = None,
        self_relation: dict = None,
        meaning: List[str] = None,
        context: List[str] = None,
        properties: dict = None,
    ):
        self.__kind = set([])
        self.__tag = set([])
        self.__idx = None
        self.__relation = None
        self.__self_relation = None
        self.__meaning = None
        self.__context = None
        self.__history = nx.Graph()
        self.__properties = None
        self.kind = kind
        self.tag = tag
        self.relation = {} if not relation else relation
        self.self_relation = {} if not self_relation else self_relation
        self.meaning = [] if not meaning else meaning
        self.context = [] if not context else context
        self.properties = {} if not properties else properties

    @property
    def kind(self) -> set:
        return self.__kind

    @property
    def tag(self) -> set:
        return self.__tag

    @property
    def idx(self) -> str:
        return str(self.__idx)

    @property
    def relation(self) -> dict:
        return self.__relation

    @property
    def self_relation(self) -> dict:
        return self.__self_relation

    @property
    def meaning(self) -> List[str]:
        return self.__meaning

    @property
    def context(self) -> List[str]:
        return self.__context

    @property
    def history(self) -> nx.Graph:
        return self.__history

    @property
    def properties(self) -> dict:
        return self.__properties

    @kind.setter
    def kind(self, kind: Union[str, set]):
        if type(kind) == set:
            self.__kind = kind
            self.__update_idx__()
        elif type(kind) == str:
            self.update_kind(kind)
        else:
            raise TypeError("EntityTag.kind: set type! Or can add str")

    @tag.setter
    def tag(self, tag: Union[str, set]):
        if type(tag) == set:
            self.__tag = tag
            self.__update_idx__()
        elif type(tag) == str:
            self.update_tag(tag)
        else:
            raise TypeError("EntityTag.tag: set type! Or can add str")

    @idx.setter
    def idx(self, idx: int):
        if type(idx) != int:
            raise TypeError("EntityTag.idx: int type!")
        self.__add_tag_idx_to_history__(self.__idx, idx)
        self.__idx = idx

    @relation.setter
    def relation(self, relation: dict):
        if type(relation) != dict:
            raise TypeError("EntityTag.relation: dict type!")
        self.__relation = relation

    @self_relation.setter
    def self_relation(self, self_relation: dict):
        if type(self_relation) != dict:
            raise TypeError("EntityTag.self_relation: dict type!")
        self.__self_relation = self_relation

    @meaning.setter
    def meaning(self, meaning: List[str]):
        if (type(meaning) != list) or (len(meaning) > 0 and type(meaning[0]) != str):
            raise TypeError("EntityTag.meaning: List[str] type!")
        self.__meaning = meaning

    @context.setter
    def context(self, context: List[str]):
        if (type(context) != list) or (len(context) > 0 and type(context[0]) != str):
            raise TypeError("EntityTag.context: List[str] type!")
        self.__context = context

    @history.setter
    def history(self, history: nx.Graph):
        if type(history) != nx.Graph and type(history) != dict:
            raise TypeError(
                "EntityTag.history: nx.Graph type! Or can convert from dict."
            )
        self.__history = history if type(history) == nx.Graph else nx.Graph(history)

    @properties.setter
    def properties(self, properties: dict):
        if type(properties) != dict:
            raise TypeError("EntityTag.properties: dict type!")
        self.__properties = properties

    def __repr__(self) -> str:
        return "_".join(sorted(list(self.tag)))

    def __str__(self) -> str:
        return self.pretty_str()

    def __hash__(self) -> int:
        return self.__idx

    def __eq__(self, other) -> bool:
        return self.idx == other.idx

    def __lt__(self, other) -> bool:
        return self.tag < other.tag

    def __le__(self, other) -> bool:
        return self.tag <= other.tag

    def __ne__(self, other) -> bool:
        return self.idx != other.idx

    def __gt__(self, other) -> bool:
        return self.tag > other.tag

    def __ge__(self, other) -> bool:
        return self.tag >= other.tag

    def __update_idx__(self):

        if self.tag and self.kind:
            self.idx = int(
                hashlib.md5(
                    (
                        "_".join(sorted(list(self.tag)))
                        + "+"
                        + "_".join(sorted(list(self.kind)))
                    ).encode("utf8")
                ).hexdigest(),
                base=16,
            )

    def __add_tag_idx_to_history__(self, last_tag_idx, new_tag_idx):

        if last_tag_idx:
            self.__history.add_edge(
                last_tag_idx, new_tag_idx, weight=3
            )  # can use weight = ? ?
        else:
            self.__history.add_node(new_tag_idx)

    @classmethod
    def from_dict(cls, data_dict):

        if type(data_dict) != dict:
            raise TypeError("EntityTag.from_dict can get only dict type!")
        ent_tag = cls(
            kind=set(data_dict["kind"]),
            tag=set(data_dict["tag"]),
            relation=data_dict["relation"] if "relation" in data_dict.keys() else None,
            self_relation=data_dict["self_relation"]
            if "self_relation" in data_dict.keys()
            else None,
            meaning=data_dict["meaning"] if "meaning" in data_dict.keys() else None,
            context=data_dict["context"] if "context" in data_dict.keys() else None,
            properties=data_dict["properties"]
            if "properties" in data_dict.keys()
            else None,
        )
        ent_tag.history = (
            data_dict["history"] if "history" in data_dict.keys() else None
        )
        return ent_tag

    def update_kind(self, str_kind: str) -> None:

        self.kind.add(str_kind)
        self.__update_idx__()

    def update_tag(self, str_tag: str) -> None:

        self.tag.add(str_tag)
        self.__update_idx__()

    def update(self, other) -> None:

        if self == other:
            self.meaning.extend(other.meaning)
            self.context.extend(other.context)
            self.history = nx.compose(self.history, other.history)
            self.properties = {**self.properties, **other.properties}
            self.relation = {**self.relation, **other.relation}
            self.self_relation = {**self.self_relation, **other.self_relation}
            return self
        else:
            print(
                f"Cant update from {other.pretty_str()} bc its != {self.pretty_str()}"
            )
            raise ValueError(f"Cant update from {type(other)} bc its != {type(self)}")

    def add_relation(self, other_idx, rel_idx):

        if other_idx not in self.relation.keys():
            self.relation[other_idx] = {rel_idx: 1}
        else:
            if rel_idx not in self.relation[other_idx].keys():
                self.relation[other_idx][rel_idx] = 1
            else:
                self.relation[other_idx][rel_idx] += 1

    def add_self_relation(self, text_tag, other_tag, rel_idx):

        if text_tag not in self.tag and other_tag not in self.tag:
            raise ValueError(
                "Cant add a relationship between nonexistent internal tags!"
            )

        if other_tag not in self.self_relation.keys():
            self.self_relation[other_tag] = {text_tag: {rel_idx: 1}}
        else:
            if text_tag not in self.self_relation[other_tag].keys():
                self.self_relation[other_tag][text_tag] = {rel_idx: 1}
            else:
                if rel_idx not in self.self_relation[other_tag][text_tag].keys():
                    self.self_relation[other_tag][text_tag][rel_idx] = 1
                else:
                    self.self_relation[other_tag][text_tag][rel_idx] += 1

    def toJSON(self) -> dict:

        self_dict = {}
        for key, value in self.__dict__.items():
            if type(value) == int:
                value = str(value)
            if type(value) == set:
                value = list(value)
            if type(value) == nx.Graph:
                value = nx.to_dict_of_dicts(self.history)
            self_dict[key[len(f"_{self.__class__.__name__}__") :]] = value
        return self_dict

    @staticmethod
    def pretty_tag(tag=None) -> str:

        if type(tag) == EntityTag:
            return f'{" ".join(sorted(list(tag.tag)))} ({" ".join(sorted(list(tag.kind)))})'
        elif type(tag) == dict:
            return f'{" ".join(tag["tag"])} ({" ".join(tag["kind"])})'

    def pretty_str(self, new_line="\n", bold=""):

        if bold == "":
            bold = ["", ""]
        prop_str = ""
        for key, value in self.properties.items():
            prop_str += f"{str(key)}: {str(value)}, "
        str_dd = f"{bold[0]}EntityTag{bold[1]}({self.idx}){new_line}"
        str_dd += f'{bold[0]}kind{bold[1]}:       {", ".join(sorted(list(self.kind)))}{new_line}'
        str_dd += f'{bold[0]}tag{bold[1]}:        {", ".join(sorted(list(self.tag)))}{new_line}'
        str_dd += f"{bold[0]}relation{bold[1]}:   {self.relation}{new_line}"
        str_dd += f"{bold[0]}self_relation{bold[1]}:   {self.self_relation}{new_line}"
        # str_dd += f'{bold[0]}meaning{bold[1]}:    {", ".join(self.meaning)}{new_line}'
        # str_dd += f'{bold[0]}context{bold[1]}:    {", ".join(self.context)}{new_line}'
        # str_dd += f'{bold[0]}history{bold[1]}:    {nx.to_dict_of_dicts(self.history)}{new_line}'
        # str_dd += f'{bold[0]}properties{bold[1]}: {prop_str}{new_line}'
        return str_dd


############################################################################################################################
# test:
############################################################################################################################
def test_main():
    from collections import Counter
    import time

    start_time = time.time()
    a = EntityTag({"tags_ent"}, {"Docker"})
    print(f"Create time: {time.time() - start_time}")
    print(a)
    print(str(a))

    b = EntityTag({"tags_ent"}, {"Docker"})
    assert a == b
    print(a == b)

    c = EntityTag({"tags_ent"}, {"Python 3"})
    d = Counter([a, b, c])
    j = Counter([c, b, a])
    h = Counter([c, b, b])
    k = Counter([c, a, a])
    assert d == j == h == k
    print(d)

    a.tag = {"docker"}
    assert a != b
    print(a != b)

    e = EntityTag(
        {"PROD"},
        {"nvidia"},
        meaning=["company"],
        context=["shop"],
        properties={"info": "blabla", "info2": "blabla"},
    )
    print(e)
    dc = {"et": e}
    print(dc)
    print(dc["et"])

    import json

    ej = e.toJSON()
    assert type(ej) == dict
    ed = json.dumps(ej)
    assert type(ed) == str
    print(ed)

    ed = json.dumps({"a": e, "b": {"c": e}}, cls=EntityTagJsonEncoder)
    assert type(ed) == str
    print(ed)

    ed = json.dumps({e.idx: e}, cls=EntityTagJsonEncoder)
    assert type(ed) == str
    print(ed)

    start_time = time.time()
    es = EntityTag(
        {"PROD"},
        {"GTX"},
        meaning=["compsdaasdany"],
        context=["shasdasdop"],
        properties={"info": "123123", "2d12d": "1231231"},
    )
    es.tag = {"RTX"}
    es.tag = {"nvidia"}
    eu = EntityTag(
        {"PROD"},
        {"GPU"},
        meaning=["company"],
        context=["shop"],
        properties={"info": "blabla", "info2": "blabla"},
    )
    eu.tag = {"Nvidia RTX"}
    eu.tag = {"Nvidia", "AMD"}
    eu.update_tag("Nvidia")
    eu.tag = {"nvidia"}
    print(f"2 Create time: {time.time() - start_time}")
    start_time = time.time()
    e.update(es)
    print(f"Update time: {time.time() - start_time}")
    start_time = time.time()
    assert e == eu == es
    print(e == eu == es)
    print(e)
    print(f"2 Eq time: {time.time() - start_time}")

    ez = EntityTag(
        {"PROD"},
        {"Hardware"},
        meaning=["company"],
        context=["shop"],
        properties={"info": "blabla", "info2": "blabla"},
    )
    ez.tag = {"nvidia"}
    e.update(eu)
    e.update(ez)
    # import matplotlib.pyplot as plt
    # options = {
    #     "node_size": [100 for i in range(len(e.history))],
    #     "font_size": [12 for i in range(len(e.history))],
    #     #"node_color": ,
    #     "edge_color": "#A0CBE2",
    #     "alpha": 0.7,
    #     "width": 4,
    #     "with_labels": True,
    # }
    # start_time = time.time()
    # print(type(e.history))
    # nx.draw(e.history,  **options)
    # plt.show()
    # print(f'Draw time: {time.time() - start_time}')

    ed = e.toJSON()
    start_time = time.time()
    ess = EntityTag.from_dict(ed)
    print(f"Create from dict Time: {time.time() - start_time}")

    start_time = time.time()
    assert e == ess
    print(e == ess)
    print(f"1 Eq time: {time.time() - start_time}")

    ez = EntityTag(
        "PROD",
        "Hardware",
        meaning=["company"],
        context=["shop"],
        properties={"info": "blabla", "info2": "blabla"},
    )
    ev = EntityTag(
        {"PROD"},
        {"Hardware"},
        meaning=["company"],
        context=["shop"],
        properties={"info": "blabla", "info2": "blabla"},
    )
    assert ez == ev
    print(ez == ev)
    ev.update_tag("OU")
    print(ev)
    print(ev.idx)
    print(ez.idx)
    assert ez != ev
    print(ez != ev)

    print("--Relation----------------------------------")
    e = EntityTag(
        "PROD",
        "Hardware",
        relation={"123": {"3": 1}},
        meaning=["company"],
        context=["shop"],
        properties={"info": "blabla", "info2": "blabla"},
    )
    ev = EntityTag(
        {"PROD"},
        {"Hardware"},
        meaning=["company"],
        context=["shop"],
        properties={"info": "blabla", "info2": "blabla"},
    )
    ev.add_relation("123", "3")
    assert e == ev
    print(e == ev)
    print(e)
    ev.add_relation("123", "2")
    assert e == ev
    print(e == ev)
    print(ev)
    ev.add_relation("123", "3")
    assert e == ev
    print(e == ev)
    print(ev)

    print("--Self-Relation-----------------------------")
    e = EntityTag(
        "PROD",
        {"Hardware", "nvidia", "gpu"},
        self_relation={"Hardware": {"nvidia": {"1": 1}}},
        meaning=["company"],
        context=["shop"],
        properties={"info": "blabla", "info2": "blabla"},
    )
    ev = EntityTag(
        {"PROD"},
        {"Hardware", "nvidia", "gpu"},
        meaning=["company"],
        context=["shop"],
        properties={"info": "blabla", "info2": "blabla"},
    )
    ev.add_self_relation("nvidia", "Hardware", "1")
    assert e == ev
    print(e == ev)
    print(e)
    ev.add_self_relation("nvidia", "gpu", "1")
    assert e == ev
    print(e == ev)
    print(ev)
    ev.add_self_relation("nvidia", "Hardware", "2")
    assert e == ev
    print(e == ev)
    print(ev)
    ev.add_self_relation("nvidia", "Hardware", "1")
    assert e == ev
    print(e == ev)
    print(ev)


if __name__ == "__main__":
    test_main()

