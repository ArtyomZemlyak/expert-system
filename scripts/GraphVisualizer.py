# pip install plotly
# pip install pyvis
# pip install python-igraph
# pip install numpy==1.19


from operator import itemgetter
import os
from os import listdir
from os.path import isfile, join
from typing import List, Union
import json
import re
import pathlib

import numpy as np
from numpy import sqrt
import networkx as nx
from numpy.linalg.linalg import norm
import plotly

from EntityTag import EntityTag


path_cfg = join(pathlib.Path(__file__).parent.resolve(), "config.json")
CONFIG = json.loads(open(path_cfg, "r").read())

# BLACKLIST = set(CONFIG['save'].values())
MAX_FILTER_OF_COUNT = CONFIG["visualizer"]["max_filter_of_count"]

STOP_WORDS_FINDER = set(CONFIG["finder"]["stop_words"])
STOP_WORDS_FINDER.update("йцукенгшщзхъфывапролджэячсмитьбюё")


class GraphVisualizer:
    def __init__(self, config: str = None) -> None:

        if config:
            self.CONFIG = config
        else:
            self.CONFIG = CONFIG

        self.path_data_folder = self.CONFIG["save"]["path_save"]

        tags_values = [
            i
            for i in json.loads(
                open(
                    os.path.join(
                        self.path_data_folder, self.CONFIG["save"]["all_tags_swap"]
                    ),
                    "r",
                    encoding="utf8",
                ).read()
            ).keys()
        ]

        self.max_tag_val = np.max(np.uint64(tags_values))
        self.files_names = [
            f
            for f in listdir(self.path_data_folder)
            if isfile(join(self.path_data_folder, f))
            if f not in BLACKLIST
        ]
        self.pages = self.__load_pages_data__()

        self.all_tags = json.loads(
            open(
                os.path.join(self.path_data_folder, self.CONFIG["save"]["all_tags"]),
                "r",
                encoding="utf8",
            ).read()
        )

        self.all_tags_idx = json.loads(
            open(
                os.path.join(
                    self.path_data_folder, self.CONFIG["save"]["all_tags_idx"]
                ),
                "r",
                encoding="utf8",
            ).read()
        )

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

    def __load_pages_data__(self) -> dict:

        pages = {}

        for file_name in self.files_names:
            name = re.sub("[\\.]{1}[A-Za-z\\d]+", "", file_name)
            json_file_data = json.loads(
                open(
                    os.path.join(self.path_data_folder, file_name), "r", encoding="utf8"
                ).read()
            )
            pages[name] = json_file_data

        return pages

    def __get_name__(self, name: str) -> str:

        try:
            if self.page_data["fields"]["title"] != "":
                name_cutted = self.page_data["fields"]["title"]
                name_cutted = re.sub("Confluence", "", name_cutted)
                name_cutted = re.sub("Digit", "", name_cutted)
                name_cutted = re.sub("  ", " ", name_cutted)

            else:
                name_cutted = re.sub("__confluence__display__", "", name)
                name_cutted = re.sub("__confluence__", "", name_cutted)
                name_cutted = re.sub(
                    "pages__viewpage___action\?pageId\=", "", name_cutted
                )

        except IndexError:
            name_cutted = re.sub("__confluence__display__", "", name)
            name_cutted = re.sub("__confluence__", "", name_cutted)
            name_cutted = re.sub("pages__viewpage___action\?pageId\=", "", name_cutted)

        return name_cutted

    def __get_count_all_tags_for_page__(self):

        count_all_tags_for_page = 0

        for tag, count in self.page_data["tags"]["values"].items():
            count_all_tags_for_page += count

        return count_all_tags_for_page

    def __get_str_all_tags_for_page__(self):

        all_tags_for_page = ""
        sorted_all_tags_for_page = sorted(
            self.page_data["tags"]["values"].items(), key=itemgetter(1), reverse=True
        )

        for tag, count in sorted_all_tags_for_page:
            if tag in self.all_tags_idx.keys():
                all_tags_for_page += (
                    f'<b>{" ".join(self.all_tags_idx[tag]["tag"])}:</b> {count}<br>'
                )

        return all_tags_for_page

    def __add_page__(self, name: str) -> None:
        if self.flag_add_page:
            count_all_tags_for_page = int(
                self.__get_count_all_tags_for_page__() / 10
            )  # / (self.max_tag_val / 1000))
            alpha_node = float(sqrt(count_all_tags_for_page / self.max_tag_val))
            if alpha_node < 0.3:
                alpha_node = 0.3
            self.alpha.append(alpha_node)
            self.node_sizes.append(count_all_tags_for_page)
            name_cutted = self.__get_name__(name)
            self.node_text.append(name_cutted)
            self.G.add_node(name)
            self.marker_symbols.append("diamond")
            self.info_text.append(
                f"<b>full_name:</b> {name}<br>" + self.__get_str_all_tags_for_page__()
            )
            self.node_colors.append("#d4ff5e")
            self.node_shapes.append("square")
            self.line_width.append(3)
            count_all_tags_for_page = int(count_all_tags_for_page / 2)
            if count_all_tags_for_page > 12:
                font_size = count_all_tags_for_page
            else:
                font_size = 12
            self.font_sizes.append(font_size)
            self.flag_add_page = False

    def __add_tag__(self, tag: EntityTag) -> None:
        if tag.idx not in self.tags:
            alpha_node = float(sqrt(self.all_tags[str(tag.idx)] / self.max_tag_val))
            if alpha_node < 0.3:
                alpha_node = 0.3
            self.alpha.append(alpha_node)
            count_all = int(self.all_tags[str(tag.idx)])  # / (self.max_tag_val / 500))
            self.node_sizes.append(count_all)
            self.node_text.append("-".join(list(tag.tag)))
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
            self.node_colors.append("#373f87")
            self.marker_symbols.append("circle")
            self.tags.append(tag.idx)

    def __check_stop_words__(self, list_of_words: List[str]) -> bool:
        stopped_words = 0
        for word in list_of_words:
            if word in STOP_WORDS_FINDER:
                stopped_words += 1
        if stopped_words == len(list_of_words):
            return False
        return True

    def __create_graph__(self) -> None:
        for name, page_data in self.pages.items():
            if "__25" in name:
                # if page_data['fields']['title'] == 'Взаимодействие с прочими системами':
                self.page_data = page_data
                self.flag_add_page = True

                current_tags_idx = set([])

                for tag, count in self.page_data["tags"]["values"].items():
                    if tag in self.all_tags_idx.keys():
                        tag = EntityTag.from_dict(self.all_tags_idx[tag])
                        if self.__check_stop_words__(list(tag.tag)):
                            if count > self.max_tag_val / MAX_FILTER_OF_COUNT:
                                self.__add_page__(name)
                                self.__add_tag__(tag)
                                self.G.add_edge(name, str(tag.idx), weight=count)
                                current_tags_idx.add(str(tag.idx))

                for tag_idx, tag_data in self.all_tags_idx.items():
                    if tag_idx in current_tags_idx:
                        if "relation" in tag_data.keys():
                            for other_idx, relations in tag_data["relation"].items():
                                for idx_rel, count_rel in relations.items():
                                    if (
                                        tag_idx in self.G.nodes()
                                        and other_idx in self.G.nodes()
                                    ):
                                        self.G.add_edge(
                                            tag_idx, other_idx, weight=count_rel * 10
                                        )

                # Как строить граф при разных связях между одними и теми же двумя тегами?
                # Стоит ли добавлять отдельные узлы в виде связей (по узлу для вида связи, или же прям уникальным связью-узлом между двумя тегами)?
                # Глубина связи тегов и обычного текста - на сколько учитывать и учитывать ли?
                # В таком случае придётся указывать теги вручную - и соответсвующие им данные можно пытаться находить с помощью нейронки поиска в тексте  ответа на вопрос.
                # Есть нюанс с Entity нейронкой - очень по разному распознаёт в зависимости от объёма текста поданного в неё. Местами сохраняет...
                # Так же стал вопрос со связями через структуру. Грубо говоря заголовок "Стоимость" и абзац в виде текста "300" получаются на отдельных строках и друг с другом не связываются (? точно ли) - как этот момент решить и нужно ли?

    def visualize(self, save: str = None) -> Union[None, str]:
        import plotly.graph_objects as go
        import igraph

        self.__create_graph__()
        # S = nx.spring_layout(self.G, iterations=100, seed=12)
        # S = nx.kamada_kawai_layout(self.G)

        iG = igraph.Graph.from_networkx(self.G)

        # S = iG.layout(layout='auto')
        # S = iG.layout_kamada_kawai_3d(seed=12)     # No good in this implementation. Kamada-Kawai force-directed algorithm in three dimensions
        # S = iG.layout_drl(dim=2)                   # realy good. The Distributed Recursive Layout algorithm for large graphs
        import random

        random.seed(1234)
        S = iG.layout_fruchterman_reingold()  # really good
        # S = iG.layout_reingold_tilford()           # intersting. Reingold-Tilford tree layout, useful for (almost) tree-like graphs
        # S = iG.layout_reingold_tilford_circular()  # interesting. Reingold-Tilford tree layout with a polar coordinate post-transformation, useful for (almost) tree-like graphs
        # S = iG.layout_davidson_harel()             # Good. Places the vertices on a 2D plane according to the Davidson-Harel layout algorithm.
        # S = iG.layout_graphopt()                   # normal. This is a port of the graphopt layout algorithm by Michael Schmuhl. graphopt version 0.4.1 was rewritten in C and the support for layers was removed.
        # S = iG.layout_mds()                        # high compressing
        # igraph.plot(iG, layout=S, target='myfile.pdf')

        node_x = []
        node_y = []
        nodes_xy_dict = {}
        for node, s in zip(self.G.nodes(), S):
            x, y = s  # s # S[node]
            node_x.append(x)
            node_y.append(y)
            nodes_xy_dict[node] = (x, y)
        S = nodes_xy_dict

        edge_x = []
        edge_y = []
        for edge in self.G.edges():
            x0, y0 = S[edge[0]]
            x1, y1 = S[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_adjacencies = []
        for node, adjacencies in enumerate(self.G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))

        def normalization(list_data: List[Union[float, int]]) -> List[float]:
            list_data_mean = np.mean(list_data)
            list_data_std = np.std(list_data)
            return [i * list_data_mean / list_data_std for i in list_data]

        # node_sizes =  normalization(self.node_sizes)#np.array(self.node_sizes)
        # node_sizes = np.array(node_sizes)/500
        node_sizes = self.node_sizes
        font_sizes = np.array(self.font_sizes) / 100

        for i in range(3):
            font_sizes = normalization(font_sizes)
        # node_sizes = np.uint64(np.multiply((node_sizes / np.linalg.norm(node_sizes)), 1000))
        font_sizes = [i if i > 12 else 12 for i in font_sizes]
        node_sizes_max = np.max(node_sizes)
        alpha = self.alpha  # [1 for i in node_sizes]
        # for i in node_sizes:
        #     alpha_i = (1- i/node_sizes_max)
        #     if alpha_i < 0.2:                           alpha.append(0.2)
        #     elif alpha_i >= 0.2 and alpha_i <= 0.5 :    alpha.append(alpha_i)
        #     else:                                       alpha.append(0.5)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            customdata=self.info_text,
            mode="markers+text",
            hoverinfo="text",
            hovertext=self.info_text,
            textfont=dict(
                size=font_sizes,
                color=[f"rgba(20,20,20,{i})" for i in alpha],
            ),
            marker_symbol=self.marker_symbols,
            marker=dict(
                showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale="YlGnBu",
                reversescale=True,
                color=[],
                opacity=alpha,
                size=node_sizes,
                sizemin=2,
                colorbar=dict(
                    thickness=15,
                    title="Number of tags",
                    xanchor="left",
                    titleside="right",
                ),
                line_width=self.line_width,
                line_color="rgba(255,255,0,0.9)",
            ),
        )

        node_trace.marker.color = node_sizes
        node_trace.text = self.node_text

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="<br>Graph of NER Analysis",
                # plot_bgcolor='rgba(100,200,230,0.7)',
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Test",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        if save:
            return fig.write_html(save)
        else:
            return fig.to_dict()
            # return fig.to_html()
            # fig.show()

    def visualize3D(self, save: str = None) -> None:
        import plotly.graph_objects as go
        import igraph

        self.__create_graph__()

        # S = nx.spring_layout(self.G, iterations=100, seed=12)
        # S = nx.kamada_kawai_layout(self.G)

        iG = igraph.Graph.from_networkx(self.G)
        # S = iG.layout_kamada_kawai_3d(seed=12)     # No good in this implementation. Kamada-Kawai force-directed algorithm in three dimensions
        # S = iG.layout_drl(dim=3)                    # realy good. The Distributed Recursive Layout algorithm for large graphs
        S = (
            iG.layout_fruchterman_reingold_3d()
        )  # good too. Fruchterman-Reingold force-directed algorithm in three dimensions
        # S = iG.layout_reingold_tilford()           # intersting. Reingold-Tilford tree layout, useful for (almost) tree-like graphs
        # S = iG.layout_reingold_tilford_circular()  # interesting. Reingold-Tilford tree layout with a polar coordinate post-transformation, useful for (almost) tree-like graphs
        # S = iG.layout_davidson_harel()             # normal. Places the vertices on a 2D plane according to the Davidson-Harel layout algorithm.
        # S = iG.layout_graphopt()                   # normal. This is a port of the graphopt layout algorithm by Michael Schmuhl. graphopt version 0.4.1 was rewritten in C and the support for layers was removed.
        # S = iG.layout_mds()                        # interesting. not working for 3D. Places the vertices in an Euclidean space with the given number of dimensions using multidimensional scaling.
        node_sizes = self.node_sizes

        node_x = []
        node_y = []
        node_z = []
        nodes_xy_dict = {}
        node_sizes_dict = {}
        for node, node_size, s in zip(self.G.nodes(), node_sizes, S):
            x, y, z = s  # S[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            nodes_xy_dict[node] = (x, y, z)
            node_sizes_dict[node] = node_size
        S = nodes_xy_dict

        edge_x = []
        edge_y = []
        edge_z = []
        for edge in self.G.edges():
            x0, y0, z0 = S[edge[0]]
            x1, y1, z1 = S[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            edge_z.append(z0)  # node_sizes_dict[edge[0]])
            edge_z.append(z1)  # node_sizes_dict[edge[1]])
            edge_z.append(None)

        node_adjacencies = []
        for node, adjacencies in enumerate(self.G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))

        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        def normalization(list_data: List[Union[float, int]]) -> List[float]:
            list_data_mean = np.mean(list_data)
            list_data_std = np.std(list_data)
            return [i * list_data_mean / list_data_std for i in list_data]

        # node_sizes =  normalization(self.node_sizes)#np.array(self.node_sizes)
        # node_sizes = np.array(node_sizes)/5000
        # font_sizes = np.array(self.font_sizes)/1000
        # for i in range(3):
        #     font_sizes = normalization(font_sizes)
        # #node_sizes = np.uint64(np.multiply((node_sizes / np.linalg.norm(node_sizes)), 1000))
        # font_sizes = [i if i > 12 else 12 for i in font_sizes]
        # node_sizes_max = np.max(node_sizes)
        # node_sizes =  normalization(self.node_sizes)#np.array(self.node_sizes)
        # node_sizes = np.array(node_sizes)/500
        node_sizes = self.node_sizes
        font_sizes = np.array(self.font_sizes) / 100

        for i in range(3):
            font_sizes = normalization(font_sizes)
        # node_sizes = np.uint64(np.multiply((node_sizes / np.linalg.norm(node_sizes)), 1000))
        font_sizes = [i if i > 12 else 12 for i in font_sizes]
        node_sizes_max = np.max(node_sizes)
        alpha = self.alpha  # [1 for i in node_sizes]
        # for i in node_sizes:
        #     alpha_i = (1- i/node_sizes_max)
        #     if alpha_i < 0.2:                           alpha.append(0.2)
        #     elif alpha_i >= 0.2 and alpha_i <= 0.5 :    alpha.append(alpha_i)
        #     else:                                       alpha.append(0.5)

        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            customdata=self.info_text,
            mode="markers+text",
            marker_symbol=self.marker_symbols,
            hoverinfo="text",
            hovertext=self.info_text,
            textfont=dict(
                size=font_sizes,
                color=f"rgba(20,20,20,0.5)",
            ),
            marker=dict(
                showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale="YlGnBu",
                reversescale=True,
                color=[],
                opacity=0.5,
                size=node_sizes,
                sizemin=2,
                colorbar=dict(
                    thickness=15,
                    title="Number of tags",
                    xanchor="left",
                    titleside="right",
                ),
                line_width=3,  # self.line_width,
                line_color="rgba(255,30,50,0.9)",
            ),
        )

        node_trace.marker.color = self.node_sizes
        node_trace.text = self.node_text

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="<br>Graph of NER Analysis",
                # plot_bgcolor='rgba(100,200,230,0.7)',
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Test",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )
        if save:
            fig.write_html(save)
        fig.show()

    def visualize_pyvis(self):
        from pyvis.network import Network

        net = Network(height="100%", width="60%")
        self.__create_graph__()
        # S = nx.spring_layout(self.G, iterations=50, seed=12)
        # node_x = []
        # node_y = []
        # for node in self.G.nodes():
        #     x, y = S[node]
        #     node_x.append(x)
        #     node_y.append(y)
        net.from_nx(self.G)
        for node, name, size, font_size, shape in zip(
            net.nodes,
            self.node_text,
            self.node_sizes,
            self.font_sizes,
            self.node_shapes,
        ):
            node["label"] = name
            node["size"] = size
            node["shape"] = shape
            node["font"] = {"size": font_size}
            # node['x']           = x
            # node['y']           = y
            node["color"] = "rgba(100,200,230,0.7)"
        net.show_buttons()
        # net.toggle_physics(False)
        # net.toggle_stabilization(False)
        net.show("example.html")


if __name__ == "__main__":
    graph = GraphVisualizer()
    # graph.visualize('2DGraph.html')
    graph.visualize3D("3DGraph.html")
