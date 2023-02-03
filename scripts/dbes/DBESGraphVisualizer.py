# pip install plotly
# pip install pyvis
# pip install python-igraph
# pip install numpy==1.19

from os.path import join, exists
from typing import Union
import json
import pathlib

import networkx as nx

import sys

sys.path.insert(0, "scripts")

from GraphVisualizer import GraphVisualizer
from dbes.DBESNode import DBESNode, DIR_RELATIONS
from dbes.DBESNet import DBESNet


##############################################
#### CONFIG ##################################
##############################################
path_cfg = join(pathlib.Path(__file__).parent.parent.resolve(), "config.json")
CONFIG = json.loads(open(path_cfg, "r").read())
MAX_FILTER_OF_COUNT = CONFIG["visualizer"]["max_filter_of_count"]
STOP_WORDS_FINDER = set(CONFIG["finder"]["stop_words"])
STOP_WORDS_FINDER.update("йцукенгшщзхъфывапролджэячсмитьбюё")
##############################################
##############################################


class DBESGraphVisualizer(GraphVisualizer):
    def __init__(self, nodes_data: dict, config: str = None) -> None:

        if config:
            self._CONFIG = config
        else:
            self._CONFIG = CONFIG

        self.path_data_folder = self._CONFIG["save"]["path_save"]

        self.max_tag_val = 110  # np.max(np.uint64(tags_values))

        self.spec_nodes = {}

        self._net = nodes_data

        self.G = nx.Graph()

        self.node_sizes = []
        self.font_sizes = []
        self.node_colors = []
        self.node_text = []
        self.node_shapes = []
        self.info_text = []
        self.line_width = []
        self.tags = []
        self.alpha = []
        self.marker_symbols = []
        self.page_data = {}

        self.flag_add_page = True

    def _open_json(self, file_name: str, path_save: str = None) -> Union[dict, None]:
        if not path_save:
            path_save = self.CONFIG["save"]["path_save"]
        if exists(join(path_save, file_name)):
            return json.loads(
                open(join(path_save, file_name), "r", encoding="utf8").read()
            )
        else:
            return None

    def __add_tag__(self, tag: DBESNode) -> None:
        if tag.idx not in self.tags:
            alpha_node = (
                0.5  # float(sqrt(self.all_tags[str(tag.idx)] / self.max_tag_val))
            )
            if alpha_node < 0.3:
                alpha_node = 0.3
            self.alpha.append(alpha_node)
            count_all = (
                100  # int(self.all_tags[str(tag.idx)])# / (self.max_tag_val / 500))
            )

            coeff = 1
            if (
                tag.idx in self.spec_nodes.keys()
                and "node_size_coeff" in self.spec_nodes[tag.idx].keys()
            ):
                coeff = self.spec_nodes[tag.idx]["node_size_coeff"]
                count_all = count_all * self.spec_nodes[tag.idx]["node_size_coeff"]

            self.node_sizes.append(count_all / 5 * coeff)

            self.node_text.append(tag.value)
            self.G.add_node(str(tag.idx))
            self.info_text.append(
                tag.pretty_str("<br>", ("<b>", "</b>")) + f"<b>count:</b> {count_all}"
            )
            self.line_width.append(0)
            self.node_shapes.append("dot")
            count_all = int(count_all / 2)
            if count_all > 12:
                font_size = count_all
            else:
                font_size = 12
            self.font_sizes.append(font_size)

            color = "#373f87"
            # if tag.idx in self.spec_nodes.keys() and "node_color" in self.spec_nodes.keys():
            #     color = self.spec_nodes["node_color"]
            self.node_colors.append(color)

            self.marker_symbols.append("circle")
            self.tags.append(tag.idx)

    def __create_graph__(self) -> None:
        for node_idx, node in self._net.items():
            self.flag_add_page = True

            current_tags_idx = set([])

            self.__add_tag__(node)

            current_tags_idx.add(str(node.idx))

            for dir_rel in DIR_RELATIONS:
                for other_idx, relations in node.relation[dir_rel].items():
                    for idx_rel, count_rel in relations.items():
                        if node_idx in self.G.nodes() and other_idx in self.G.nodes():
                            self.G.add_edge(node_idx, other_idx, weight=count_rel * 10)


if __name__ == "__main__":
    graph = DBESGraphVisualizer()
    graph.visualize("2DGraph.html")
    # graph.visualize3D('3DGraph.html')
