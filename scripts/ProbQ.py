from operator import itemgetter
import json
import time
from pprint import pprint
from typing import Union

import numpy as np


class ProbQ:
    def __init__(self, path: Union[dict, str] = None):

        if type(path) == str:
            self.services_prop = json.loads(open(path, "r").read())
        elif type(path) == dict:
            self.services_prop = path
        else:
            raise TypeError("Need str path or dict in spec format")

        self.questions = {}
        self.used_questions = {}
        self.last_services = set([])
        self.k = {service: 1 for service in self.services_prop.keys()}
        self.last_true_services = [set([]), 0]

        self.services_res = {}
        for service, params in self.services_prop.items():
            service_relevance = 0
            for val in params["conditions"].values():
                service_relevance += val
            self.services_res[service] = service_relevance

        self._update()

    def _update(self):

        self.questions = {}
        self.k = {service: 1 for service in self.services_prop.keys()}

        if self.last_true_services[1] >= 3:
            for service in self.last_true_services[0]:
                for question, prob in self.services_prop[service]["questions"].items():
                    if (
                        question in self.used_questions.keys()
                        and self.used_questions[question] == "-"
                    ):
                        del self.used_questions[question]

        for service, params in self.services_prop.items():
            service_relevance = self.services_res[service]
            service_relevance = np.arctan(service_relevance) / np.pi + 1 / 2

            for question, prob in params["questions"].items():
                if question not in self.used_questions.keys():

                    if service in self.last_services:
                        self.k[service] = 5

                    if question not in self.questions.keys():
                        self.questions[question] = (
                            prob * service_relevance * self.k[service]
                        )
                    else:
                        self.questions[question] += (
                            prob * service_relevance * self.k[service]
                        )  # ? - mb need / on quantaty of questions in diff services

        self.current_questions = sorted(
            [[question, prob] for question, prob in self.questions.items()],
            key=itemgetter(1),
            reverse=True,
        )
        self.last_services = set([])

    def find(self, question, answer):

        self.used_questions[question] = answer
        last_true_services = self.last_true_services[0]
        if self.last_true_services[0] != set([]):
            self.last_true_services = [set([]), self.last_true_services[1] + 1]

        for service, params in self.services_prop.items():

            if question in params["questions"].keys():
                if answer == "+":
                    self.last_services.add(service)
                    if service in last_true_services:
                        self.last_true_services[0].add(service)
                    self.services_res[service] += (
                        params["questions"][question] * self.k[service] * 3
                    )
                else:
                    self.services_res[service] -= params["questions"][question] * 10
            else:
                if answer == "+":
                    self.services_res[service] -= self.questions[question] * 10
                else:
                    self.services_res[service] += self.questions[question]

        sr_array = [i for i in self.services_res.values()]
        normalized = sr_array  # np.arctan(sr_array) / np.pi + 1/2

        for service, val in zip(self.services_res.keys(), normalized):
            self.services_res[service] = val

        if self.last_true_services[0] == set([]):
            if self.last_services != set([]):
                self.last_true_services[0] = self.last_services
            else:
                self.last_true_services = [set([]), 0]

        self._update()

        return sorted(
            [
                [service, prob, np.arctan(prob) / np.pi + 1 / 2]
                for service, prob in self.services_res.items()
            ],
            key=itemgetter(1),
            reverse=True,
        )


if __name__ == "__main__":

    esprob = ProbQ("PATH")
    print(" --Список услуг:")
    pprint(
        sorted(
            [[service, prob] for service, prob in esprob.services_res.items()],
            key=itemgetter(1),
            reverse=True,
        ),
        width=400,
    )

    while True:
        print(f"------------------------------------")
        print(f" --Список вопросов:")
        pprint(esprob.current_questions[:20])

        current_question = esprob.current_questions[0][0]

        input_text = input(f" --Вопрос: {current_question}\n --Ответ: ")

        start_time = time.time()

        if input_text == "q":
            break
        elif input_text == "+" or input_text == "-":
            print(" --Список услуг:")
            pprint(esprob.find(current_question, input_text), width=400)
        else:
            print(" Неккоректный ввод! Используйте + или - для ответа на вопрос!")

        print(" | время выполнения: ", (time.time() - start_time))

