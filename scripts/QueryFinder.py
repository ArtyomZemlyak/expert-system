import json
import time
from typing import List
from operator import itemgetter

from mosestokenizer import MosesTokenizer
from pymystem3 import Mystem
from nltk.corpus import stopwords


CONFIG = {
    **json.loads(open("config.json", "r").read()),
    **json.loads(open("scripts/config.json", "r").read()),
}


class QueryFinder:
    def __init__(self, config: str = None) -> None:

        if config:
            self.CONFIG = config
        else:
            self.CONFIG = CONFIG

        self.nlp = MosesTokenizer("ru")
        self.mystem = Mystem()
        self.__stopwords = {
            *stopwords.words("russian"),
            *stopwords.words("english"),
            *self.CONFIG["query"]["stop_words"],
        }

    @staticmethod
    def _equal(one_set: set, two_set: set) -> int:

        idx_equal = 0

        for item in one_set:
            if item in two_set:
                idx_equal += 1

        return idx_equal

    def tokenize(self, text: str) -> List[str]:

        tokens = self.nlp.tokenize(text)

        if self.mystem and self.__stopwords:
            tokens = [token.lower() for token in tokens]
            tokens = [
                self.mystem.lemmatize(token)[0]
                if token not in self.CONFIG["query"]["white_words"]
                else token
                for token in tokens
                if token not in self.__stopwords
            ]

        return tokens

    def find_task(self, tokens: List[str], tags: dict = None) -> str:

        task = ""
        finded_tags = set([])
        if not tags:
            tags = self.CONFIG["query"]["tags"]
        tasks = self.CONFIG["query"]["tasks"]

        for token in tokens:
            if token in tags.keys():
                finded_tags.add(tags[token]["@idx"])

        for _task, task_prop in tasks.items():
            if finded_tags == set(task_prop["@tags"]):
                task = _task

        if task == "":
            tasks_idx_equal = [
                [
                    _task,
                    self._equal(finded_tags, set(task_prop["@tags"])),
                    task_prop["@stat"],
                ]
                for _task, task_prop in tasks.items()
            ]
            tasks_idx_equal.sort(key=itemgetter(1, 2), reverse=True)

            for _task, idx_equal, stat_task in tasks_idx_equal:

                task_tags = set(tasks[_task]["@tags"])
                for finded_tag in finded_tags:
                    if finded_tag not in task_tags:
                        print(
                            " --Обработка не удалась. Попробуйте переформулировать запрос!"
                        )
                        return ""
                    task_tags.remove(finded_tag)

                task_tags_words = []
                for tag in task_tags:
                    for _tag, tag_prop in tags.items():
                        if tag_prop["@idx"] == tag:
                            task_tags_words.append(_tag)
                            break

                sorted_task_tags = [
                    [tag, tags[tag]["@stat"]] for tag in task_tags_words
                ]
                sorted_task_tags.sort(key=itemgetter(1), reverse=True)

                for tag, stat in sorted_task_tags:

                    answer = input(f" --Является ли тег -{tag}- верным:")

                    if answer == "+":
                        tokens.append(tag)
                        task = self.find_task(tokens)
                        if task != "":
                            return task

        # for token in tokens:
        #     if token in tasks.keys():
        #         task.append(tasks[token]['@id'])
        #         _task = self.find_task([tok for tok in tokens if tok != token], tasks[token])
        #         task.extend(_task)

        # if task == []:
        #     for _task, subtasks in tasks.items():
        #         if _task != '@id':
        #             task.extend(self.find_task(tokens, subtasks))

        return task


if __name__ == "__main__":

    query_finder = QueryFinder()

    while True:

        input_text = input(" --запрос:")

        if input_text == "q":
            break

        start_time = time.time()

        tokens = query_finder.tokenize(input_text)
        task = query_finder.find_task(tokens)

        print(" --токены:\n", tokens)
        print(" --задача:\n", task)

        print(" | время выполнения: ", (time.time() - start_time))
