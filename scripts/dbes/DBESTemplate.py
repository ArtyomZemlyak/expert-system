import json
from typing import Union
import copy

from dbes.DBESNode import DBESNode


class DBESNodeJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, DBESNode):
            return obj.toJSON()
        if isinstance(obj, DBESTemplate):
            return obj.toJSON()
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class DBESTemplate(DBESNode):
    def __init__(
        self,
        value: Union[dict, set, list, str, int] = None,
        relation: dict = None,
        idx: str = None,
        template: dict = None,
    ):
        super().__init__(value=value, relation=relation, idx=idx)
        self.__template = None
        self.template = template if template else {}

    @property
    def template(self) -> dict:
        return copy.deepcopy({**self.__template})  # ? how avoid this??

    @template.setter
    def template(self, template: dict):
        if type(template) != dict:
            raise TypeError("DBESTemplate.template: dict type!")
        self.__template = copy.deepcopy({**template})

    @classmethod
    def from_dict(cls, data_dict):
        if type(data_dict) != dict:
            raise TypeError("DBESTemplate.from_dict can get only dict type!")

        dbes_tag = cls(
            idx=data_dict["idx"]
            if "idx" in data_dict.keys()
            else data_dict["node"]["idx"]
            if "node" in data_dict.keys() and "idx" in data_dict["node"].keys()
            else None,
            value=data_dict["value"]
            if "value" in data_dict.keys()
            else data_dict["node"]["value"]
            if "node" in data_dict.keys() and "value" in data_dict["node"].keys()
            else None,
            relation=data_dict["relation"]
            if "relation" in data_dict.keys()
            else data_dict["node"]["relation"]
            if "node" in data_dict.keys() and "relation" in data_dict["node"].keys()
            else {},
            template=data_dict["template"] if "template" in data_dict.keys() else {},
        )

        return dbes_tag

    def update_value(self, template: dict) -> None:
        if type(template) == dict:
            self.template = {**self.template, **template}
        else:
            raise TypeError("DBESNode.update_value can get only dict type!")

    def pretty_str(self, new_line="\n", bold=""):
        if bold == "":
            bold = ["", ""]

        str_dd = f"{bold[0]}DBESNode{bold[1]}({self.idx}){new_line}"
        str_dd += f"{bold[0]}value{bold[1]}:        {self.value}{new_line}"
        str_dd += f"{bold[0]}relation{bold[1]}:     {self.relation}{new_line}"
        str_dd += f"{bold[0]}template{bold[1]}:     {self.template}{new_line}"

        return str_dd

    def toJSON(self) -> dict:
        self_dict = {"node": {}}
        node_keys = set(["hash_idx", "idx", "value", "relation"])

        for key, value in self.__dict__.items():
            if type(value) == int:
                value = str(value)
            if type(value) == set:
                value = list(value)

            key = key.split("__")[-1]

            if key in node_keys:
                self_dict["node"][key] = value
            else:
                self_dict[key] = value

        return self_dict
