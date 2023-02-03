# pip install fast-autocomplete[levenshtein]

from argparse import ArgumentError
import os
from os import listdir
from os.path import isfile, join
from operator import itemgetter
from collections import Counter, defaultdict

from typing import List, Tuple, Union
from pprint import pprint
import string
import json
import time
import re

import networkx as nx
import numpy as np
from numpy import sqrt
from rich.progress import Progress
from deeppavlov import build_model
from fast_autocomplete import autocomplete_factory

from EntityTag import EntityTag


CONFIG = json.loads(open("scripts/config.json", "r").read())

VALID_CHARS = CONFIG["finder"]["valid_chars"]
VALID_CHARS += string.ascii_lowercase
VALID_CHARS += string.ascii_uppercase

STOP_WORDS_FINDER = set(CONFIG["finder"]["stop_words"])
STOP_WORDS_FINDER.update("йцукенгшщзхъфывапролджэячсмитьбюё")

COUNT_OF_TAGS_FOR_DISPLAY = CONFIG["finder"]["count_of_tags_for_display"]
COUNT_OF_RES_FOR_DISPLAY = CONFIG["finder"]["count_of_res_for_display"]

BLACKLIST = set(CONFIG["save"].values())

MAX_FILTER_OF_COUNT = CONFIG["visualizer"]["max_filter_of_count"]


class TagRelFinder:
    def __init__(self, config: str = None, valid_chars: str = VALID_CHARS) -> None:

        if config:
            self.CONFIG = config
        else:
            self.CONFIG = CONFIG

        with open(
            os.path.join(
                self.CONFIG["save"]["path_save"], self.CONFIG["save"]["valid_chars"]
            ),
            "w",
            encoding="utf8",
        ) as json_file:

            json_file.write(
                json.dumps(valid_chars, indent=4, sort_keys=True, ensure_ascii=False)
            )
            json_file.close()

        content_files = {
            "words": {
                "filepath": os.path.join(
                    self.CONFIG["save"]["path_save"],
                    self.CONFIG["save"]["all_tags_formatted"],
                ),
                "compress": True,  # means compress the graph data in memory
            },
            "valid_chars_for_string": {
                "filepath": os.path.join(
                    self.CONFIG["save"]["path_save"], self.CONFIG["save"]["valid_chars"]
                ),
                "compress": False,  # need
            },
        }

        self.pages = {}

        files_names = [
            f
            for f in listdir(self.CONFIG["save"]["path_save"])
            if isfile(join(self.CONFIG["save"]["path_save"], f)) and f not in BLACKLIST
        ]

        for file_name in files_names:
            name = re.sub("[\\.]{1}[A-Za-z\\d]+", "", file_name)
            json_file_data = json.loads(
                open(
                    os.path.join(self.CONFIG["save"]["path_save"], file_name),
                    "r",
                    encoding="utf8",
                ).read()
            )
            self.pages[name] = json_file_data

        self.tags_index = json.loads(
            open(
                os.path.join(
                    self.CONFIG["save"]["path_save"],
                    self.CONFIG["save"]["all_tags_idx"],
                ),
                "r",
                encoding="utf8",
            ).read()
        )

        self.all_tags_tags = json.loads(
            open(
                os.path.join(
                    self.CONFIG["save"]["path_save"],
                    self.CONFIG["save"]["all_tags_tags"],
                ),
                "r",
                encoding="utf8",
            ).read()
        )

        self.all_tags_dict_with_paths = json.loads(
            open(
                os.path.join(
                    self.CONFIG["save"]["path_save"],
                    self.CONFIG["save"]["all_tags_with_paths"],
                ),
                "r",
                encoding="utf8",
            ).read()
        )

        self.sorted_all_tags = self.__get_all_sorted_tags__()
        self.current_tags = self.sorted_all_tags
        self.used_tags = []

        self.autocomplete = autocomplete_factory(content_files=content_files)

        self.find_titles = True
        self.find_average = True
        self.CONFIG = CONFIG

        if "morph" in self.CONFIG.keys() and self.CONFIG["morph"]:
            if os.path.exists(
                os.path.join(
                    self.CONFIG["save"]["path_save"], self.CONFIG["save"]["morph_tags"]
                )
            ):

                self.morph_tags = json.loads(
                    open(
                        os.path.join(
                            self.CONFIG["save"]["path_save"],
                            self.CONFIG["save"]["morph_tags"],
                        ),
                        "r",
                        encoding="utf8",
                    ).read()
                )

                if self.morph_tags == {}:
                    raise ValueError(
                        f"File {self.CONFIG['save']['morph_tags']} is empty!"
                    )

            else:
                raise FileExistsError(
                    f"File {self.CONFIG['save']['morph_tags']} dont exist!"
                )

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

        self.all_tags = json.loads(
            open(
                os.path.join(self.path_data_folder, self.CONFIG["save"]["all_tags"]),
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

    def __check_text__(self, text: str) -> bool:
        return re.sub(f"[^{self.CONFIG['analyzer']['word_symbols']}]", "", text) != ""

    def __check_tag__(self, tag: str) -> str:

        if self.__check_text__(tag):
            tag = re.sub(
                f"[^{self.CONFIG['analyzer']['word_symbols'] + self.CONFIG['analyzer']['filter_symbols']}]",
                "",
                tag,
            )

            return tag

        else:
            return ""

    def __check_STOP_WORDS_FINDER__(self, list_of_words: List[str]) -> bool:
        stopped_words = 0

        for word in list_of_words:
            if word in STOP_WORDS_FINDER:
                stopped_words += 1

        if stopped_words == len(list_of_words):
            return False

        return True

    def __get_formatted_word__(self, word: str) -> str:

        text_tag = self.__check_tag__(word)

        if self.morph_tags and text_tag in self.morph_tags.keys():
            text_tag = self.morph_tags[text_tag]

        return text_tag

    def __get_all_sorted_tags__(self):

        all_tags = json.loads(
            open(
                os.path.join(
                    self.CONFIG["save"]["path_save"], self.CONFIG["save"]["all_tags"]
                ),
                "r",
                encoding="utf8",
            ).read()
        )

        sorted_all_tags = sorted(
            [
                [tag, counter]
                for tag, counter in all_tags.items()
                if tag in self.tags_index.keys()
            ],
            key=itemgetter(1),
            reverse=True,
        )

        return [
            EntityTag.pretty_tag(self.tags_index[tag])
            for tag, counter in sorted_all_tags
            if self.__check_STOP_WORDS_FINDER__(self.tags_index[tag]["tag"])
        ]

    def __get_all_input_words__(
        self, input_words: str, enumeration: bool = False
    ) -> List[str]:

        input_words = input_words.lower().split(" ")
        input_words = [
            self.__get_formatted_word__(word)
            for word in input_words
            if self.__check_text__(word)
        ]

        input_words_ = []

        if enumeration:

            for i in range(len(input_words)):
                k = len(input_words) - i

                for j in range(len(input_words) - k + 1):
                    words = " ".join(input_words[j : j + k])

                    if words != "":
                        input_words_.append(words)

        input_words_ = input_words

        return input_words_

    def __get_used_tags__(self):
        return [
            EntityTag.pretty_tag(self.tags_index[str(tag)]) for tag in self.used_tags
        ]

    def __get_sorted_tags__(self):
        return [tag for tag in self.sorted_all_tags if tag not in self.used_tags]

    def __autocomplete__(
        self,
        input_words: List[str],
        sorted_list_of_paths: List[list] = None,
        res_set: List[set] = None,
    ) -> Tuple[List[list], List[set]]:

        self.used_tags = []

        if not sorted_list_of_paths:
            sorted_list_of_paths = []

        if not res_set:
            res_set = []

        idxs = []

        with Progress() as progress:
            task = progress.add_task("[green]Processing...", total=len(input_words))

            for input_word in input_words:
                results_autocomplete = self.autocomplete.search(
                    word=input_word, max_cost=3, size=200
                )

                for results in results_autocomplete:
                    res = results[0]

                    if res in self.all_tags_tags.keys():
                        res_idxs = self.all_tags_tags[res]
                        idxs.extend(res_idxs)

                    else:
                        continue

                progress.update(task, advance=1)

        return idxs

    def __print_all_relations__(self, idxs: List[str]) -> None:

        for idx in idxs:
            print("------------------------------")
            print(self.tags_index[idx]["tag"], f'({self.tags_index[idx]["kind"]}) :')

            print("to")
            for key, val in self.tags_index[idx]["relation"].items():
                print(self.tags_index[key]["tag"], " ", val.values())

            print("from")
            for key, val in self.tags_index.items():
                if idx in val["relation"].keys():
                    print(val["tag"], " ", val["relation"][idx].values())

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
            if tag in self.tags_index.keys():
                all_tags_for_page += (
                    f'<b>{" ".join(self.tags_index[tag]["tag"])}:</b> {count}<br>'
                )

        return all_tags_for_page

    def __add_tag__(self, tag: EntityTag) -> None:

        if tag.idx not in self.tags:
            self.G.add_node(str(tag.idx))
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

        with Progress() as progress:
            task = progress.add_task(
                "[green]Data loading...", total=len(self.pages.keys())
            )

            for name, page_data in self.pages.items():
                if "17201" in name:
                    print(name)
                    # if page_data['fields']['title'] == 'Взаимодействие с прочими системами':
                    self.page_data = page_data
                    self.flag_add_page = True

                    current_tags_idx = set([])

                    for tag, count in self.page_data["tags"]["values"].items():
                        if tag in self.tags_index.keys():
                            tag = EntityTag.from_dict(self.tags_index[tag])
                            if True:  #'CARDINAL' not in tag.kind:
                                # if self.__check_stop_words__(list(tag.tag)):
                                #     if count > self.max_tag_val / MAX_FILTER_OF_COUNT:
                                self.__add_tag__(tag)
                                current_tags_idx.add(str(tag.idx))

                    for tag_idx, tag_data in self.tags_index.items():
                        if tag_idx in current_tags_idx:
                            if "relation" in tag_data.keys():
                                for other_idx, relations in tag_data[
                                    "relation"
                                ].items():
                                    for idx_rel, count_rel in relations.items():
                                        if (
                                            tag_idx in self.G.nodes()
                                            and other_idx in self.G.nodes()
                                        ):
                                            self.G.add_edge(
                                                tag_idx,
                                                other_idx,
                                                weight=count_rel * 10,
                                            )

                progress.update(task, advance=1)

    def __visualize__(self, idxs):

        self.__create_graph__()

        for idx in idxs:
            if idx in self.G:
                print("--------------------------------------")
                print(f"idx: {idx}")
                print(f'tag: {self.tags_index[idx]["tag"]}')

                G = nx.bfs_tree(self.G, idx, reverse=False)
                print(G[idx])
                import plotly.graph_objects as go
                import igraph

                # S = nx.spring_layout(self.G, iterations=100, seed=12)
                # S = nx.kamada_kawai_layout(G)

                iG = igraph.Graph.from_networkx(G)

                # S = iG.layout(layout='auto')
                # S = iG.layout_kamada_kawai_3d(seed=12)     # No good in this implementation. Kamada-Kawai force-directed algorithm in three dimensions
                # S = iG.layout_drl(dim=2)                   # realy good. The Distributed Recursive Layout algorithm for large graphs
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
                node_sizes = []
                for node, s in zip(G.nodes(), S):
                    x, y = s  # s # S[node]
                    node_x.append(x)
                    node_y.append(y)
                    nodes_xy_dict[node] = (x, y)
                    node_sizes.append(20)

                    tag = EntityTag.from_dict(self.tags_index[node])
                    alpha_node = float(
                        sqrt(self.all_tags[str(node)] / self.max_tag_val)
                    )
                    if alpha_node < 0.3:
                        alpha_node = 0.3
                    self.alpha.append(alpha_node)
                    count_all = int(
                        self.all_tags[str(node)]
                    )  # / (self.max_tag_val / 500))
                    self.node_sizes.append(count_all)
                    self.node_text.append("-".join(list(tag.tag)))
                    self.info_text.append(
                        tag.pretty_str("<br>", ("<b>", "</b>"))
                        + f"<b>count:</b> {count_all}"
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
                S = nodes_xy_dict

                edge_x = []
                edge_y = []
                for edge in G.edges():
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
                for node, adjacencies in enumerate(G.adjacency()):
                    node_adjacencies.append(len(adjacencies[1]))

                def normalization(list_data: List[Union[float, int]]) -> List[float]:
                    list_data_mean = np.mean(list_data)
                    list_data_std = np.std(list_data)
                    return [i * list_data_mean / list_data_std for i in list_data]

                node_sizes = normalization(node_sizes)  # np.array(self.node_sizes)
                node_sizes = np.array(node_sizes) / 500
                font_sizes = np.array(node_sizes) / 100

                for i in range(3):
                    font_sizes = normalization(font_sizes)
                # node_sizes = np.uint64(np.multiply((node_sizes / np.linalg.norm(node_sizes)), 1000))
                font_sizes = [i if i > 12 else 12 for i in font_sizes]
                node_sizes_max = np.max(node_sizes)
                alpha = []
                for i in node_sizes:
                    alpha_i = 1 - i / node_sizes_max
                    if alpha_i < 0.2:
                        alpha.append(0.2)
                    elif alpha_i >= 0.2 and alpha_i <= 0.5:
                        alpha.append(alpha_i)
                    else:
                        alpha.append(0.5)

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
                        xaxis=dict(
                            showgrid=False, zeroline=False, showticklabels=False
                        ),
                        yaxis=dict(
                            showgrid=False, zeroline=False, showticklabels=False
                        ),
                    ),
                )
                fig.show()

    def find(self, input_text: str) -> None:

        input_words = self.__get_all_input_words__(input_text)
        idxs = self.__autocomplete__(input_words)
        self.__print_all_relations__(idxs)
        self.__visualize__(idxs)


if __name__ == "__main__":

    tag_rel_finder = TagRelFinder()

    while True:
        print(f"------------------------------------")
        if COUNT_OF_TAGS_FOR_DISPLAY < len(tag_rel_finder.current_tags):
            print(
                f" --доступные теги: {tag_rel_finder.current_tags[:COUNT_OF_TAGS_FOR_DISPLAY]}"
            )
        else:
            print(f" --доступные теги: {tag_rel_finder.current_tags}")
        # if COUNT_OF_TAGS_FOR_DISPLAY < len(tag_rel_finder.current_tags):
        #     print(f' --использованные теги: {tag_rel_finder.used_tags[:COUNT_OF_TAGS_FOR_DISPLAY]}')
        # else: print(f' --использованные теги: {tag_rel_finder.used_tags}')

        input_text = input(" --запрос:")
        if input_text == "q":

            ajson = {"list": tag_rel_finder.current_t3ags}
            # ajson = {'list' : [tag for tag in tag_rel_finder.current_tags if '(text)' in tag]}
            with open(
                "TEST/tag_rel_finder_current_tags_17201_1.json", "w", encoding="utf8"
            ) as json_file:
                json_file.write(
                    json.dumps(ajson, indent=4, sort_keys=True, ensure_ascii=False)
                )
                json_file.close()

            break

        start_time = time.time()

        print(" --результат:")

        tag_rel_finder.find(input_text)

        print(" | время выполнения: ", (time.time() - start_time))
