from collections import Counter, defaultdict
from functools import wraps
from operator import itemgetter, le
import os
from os import listdir, times
from os.path import isfile, join, exists


from typing import Any, Dict, Iterable, List, Tuple, Union, Counter as CounterT
from pprint import pprint
import json
import pathlib

from nltk.util import pr

from dbes.DBESNet import DBESNet
from es.ESHierarchicProb import ESHierarchicProb
from es.ESAcyclicProb import ESAcyclicProb
from es.ESTargetFinder import ESTargetFinder
from ProbQ import ProbQ


##############################################
#### CONFIG ##################################
##############################################
path_cfg = join(pathlib.Path(__file__).parent.parent.resolve(), "config.json")
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
        dbes = kwargs.get("dbes")
        dbes = self._dbes if not dbes else dbes

        if dbes and dbes._cache:
            if "_profile" in kwargs and kwargs["_profile"]:
                profile_name = kwargs["_profile"]

                if profile_name in dbes._cache:
                    func_name = function.__name__

                    if func_name in dbes._cache[profile_name]:
                        runtype = kwargs.get("_runtype")
                        # print('get: ', func_name)

                        if (
                            not runtype
                            and "return" in dbes._cache[profile_name][func_name]
                            or runtype == "return"
                        ):
                            return dbes._cache[profile_name][func_name]["return"]
                        elif (
                            "args" in dbes._cache[profile_name][func_name]
                            or runtype == "args"
                        ):
                            return function(
                                self,
                                *args,
                                **{
                                    **dbes._cache[profile_name][func_name]["args"],
                                    **kwargs,
                                },
                            )

                        else:
                            raise ValueError(
                                "Cant find appropriate runtype for function!"
                            )
                else:
                    if "_runtype" in kwargs and kwargs["_runtype"]:
                        runtype = kwargs["_runtype"]
                        func_name = function.__name__
                        # print('add: ', func_name)

                        if "_no_save" in kwargs and kwargs["_no_save"]:
                            write_on_disk = False
                        else:
                            write_on_disk = True

                        if runtype == "args":
                            if args:
                                raise ValueError(
                                    "Cant convert *args to **kwargs - please use attribute=value in functions!"
                                )
                            returned = function(self, *args, **kwargs)

                            if "_ignore_attr" in kwargs:
                                _ignore_attr = kwargs["_ignore_attr"]
                                kwargs = {
                                    attr_name: attr_data
                                    for attr_name, attr_data in kwargs.items()
                                    if attr_name not in _ignore_attr
                                }

                            self.add_profile(
                                profile_name,
                                func_name,
                                func_args=kwargs,
                                write_on_disk=write_on_disk,
                                dbes=dbes,
                            )
                            return returned

                        elif runtype == "return":
                            returned = function(self, *args, **kwargs)
                            self.add_profile(
                                profile_name,
                                func_name,
                                func_return=returned,
                                write_on_disk=write_on_disk,
                                dbes=dbes,
                            )
                            return returned

                    else:
                        raise ValueError(
                            "Setted profile_name non exist! For adding profile for DBES use add_profile func or specify _runtype parameter!"
                        )
            else:
                return function(self, *args, **kwargs)
        else:
            return function(self, *args, **kwargs)

    return wrapper


class ES:
    def __init__(self, dbes: DBESNet = None, config: str = None) -> None:
        if config:
            self._CONFIG = config
        else:
            self._CONFIG = CONFIG

        self._dbes = dbes
        self.hprob = ESHierarchicProb()
        self.aprob = ESAcyclicProb()
        self.tfind = ESTargetFinder()

        self._methods = self._get_methods()
        self._cp_methods = self._get_decorated_methods("caching_profile")

    def _get_methods(self, class_instance: Any = None) -> set:
        non_magic_class = set([])
        class_methods = dir(self if not class_instance else class_instance)

        for m in class_methods:
            if m.startswith("__"):
                continue
            else:
                non_magic_class.add(m)

        return non_magic_class

    def _get_decorated_methods(
        self, decorator_tag: str = "caching_profile", class_instance: Any = None
    ) -> set:
        class_instance = self if not class_instance else class_instance

        return {
            name
            # get all attributes, including methods, properties, and builtins
            for name in dir(class_instance)
            # but we only want methods
            if callable(getattr(class_instance, name))
            # and we don't need builtins
            and not name.startswith("__")
            # and we only want the cool methods
            and hasattr(getattr(class_instance, name), decorator_tag)
        }

    def add_profile(
        self,
        profile_name: str,
        func_name: str,
        func_args: dict = None,
        func_return: Any = None,
        type_caching: str = "delete",
        rewrite: bool = True,
        write_on_disk: bool = True,
        dbes: DBESNet = None,
    ) -> bool:
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
                "delete" : ! NOT SUPPORTED FOR NON DBESNET! Delete permanently this profile, when any of data change occures.
                "none" : Nothing happend.

        Return
        ----------
            `bool` : True - if adding successful. Else - error.
        """
        dbes = self._dbes if not dbes else dbes

        if not func_args and not func_return:
            raise ValueError("Need specify func_args or func_return parameter!")

        if (
            profile_name in dbes._cache
            and func_name in dbes._cache[profile_name]
            and not rewrite
        ):
            raise ValueError(
                "Setted profile_name already exist for this func_name! Choose another name for profile or set rewrite=True."
            )

        if func_name not in self._methods:
            raise ValueError(
                "Setted func_name non exist! Choose from existing functions!"
            )

        if func_name not in self._cp_methods:
            raise ValueError(
                "Setted func_name exist, but cant support caching_profile! Choose from another existing functions, which supported caching_profile feature!"
            )

        if type_caching == "recompute":
            raise ValueError(
                "Setted type_caching==recompute -> NOT SUPPORTED FOR NON DBESNET!"
            )

        if profile_name in dbes._cache and func_name in dbes._cache[profile_name]:
            dbes.remove_profile(profile_name, func_name, del_on_disk=write_on_disk)

        if profile_name not in dbes._cache:
            dbes._cache[profile_name] = {}

        dbes._cache[profile_name][func_name] = {"type_caching": type_caching}

        if func_args:
            dbes._cache[profile_name][func_name]["args"] = func_args
        if func_return:
            dbes._cache[profile_name][func_name]["return"] = func_return

        if write_on_disk:
            dbes._save_pickle(
                profile_name,
                dbes._cache[profile_name],
                join(dbes._CONFIG["save"]["path_save"], "caching"),
            )

        return True

    def hierarchic_prob_AIB(
        self,
        idx_B: Union[str, dict],
        dict_probs: dict,
        dbes: DBESNet = None,
        filter_idxs: set = None,
        STAT_ERR: float = 0,
    ) -> set:
        """
        Recalculate probabilitiies P(A|B) for dict of probs and True idx B. B - root, and A - all childs.
        The set of positive cases A is contained in the set of positive cases B.
        Hierarchical structure. Adjacent sets do not intersect.
        """
        dbes = self._dbes if not dbes else dbes

        if dbes:
            fsi = dbes.find_all_dir_rel_idxs_lvls(
                idx_B if type(idx_B) != dict else idx_B["idx"],
                dir_rel="out",
                filter_idxs=filter_idxs,
            )

            return self.hprob.hierarchic_prob_AIB(
                idx_B=idx_B,
                dict_probs=dict_probs,
                forward_struct_idxs=fsi,
                filter_idxs=filter_idxs,
                STAT_ERR=STAT_ERR,
            )

        else:
            raise ValueError(
                "DBES instance not found! Init ES with DBES instance, or put in dbes attribute of a function"
            )

    def hierarchic_prob_BIA(
        self,
        idx_A: Union[str, dict],
        dict_probs: dict,
        dbes: DBESNet = None,
        filter_idxs: set = None,
        STAT_ERR: float = 0,
    ) -> set:
        """
        Recalculate probabilities P(B|A) for dict of probs and True idx A. B - all roots, and A - child.
        The set of positive cases A is contained in the set of positive cases B.
        Hierarchical structure. Adjacent sets do not intersect.
        """
        dbes = self._dbes if not dbes else dbes

        if dbes:
            bsi = dbes.find_all_dir_rel_idxs_lvls(
                idx_A if type(idx_A) != dict else idx_A["idx"],
                dir_rel="in",
                filter_idxs=filter_idxs,
            )

            return self.hprob.hierarchic_prob_BIA(
                idx_A=idx_A,
                dict_probs=dict_probs,
                back_struct_idxs=bsi,
                filter_idxs=filter_idxs,
                STAT_ERR=STAT_ERR,
            )

        else:
            raise ValueError(
                "DBES instance not found! Init ES with DBES instance, or put in dbes attribute of a function"
            )

    def hierarchic_all_probs_A(
        self,
        idx_A: Union[str, dict],
        dict_probs: dict,
        filter_idxs: set = None,
        STAT_ERR: float = 0,
    ) -> set:
        """
        Recalculate all probabilities for dict of probs and True idx A. B - all nodes in filter_idxs, and A - some node.
        Hierarchical structure. Adjacent sets do not intersect.
        """
        return self.hprob.hierarchic_all_probs_A(
            idx_A=idx_A,
            dict_probs=dict_probs,
            filter_idxs=filter_idxs,
            STAT_ERR=STAT_ERR,
        )

    def hierarchic_prob_AInotB(
        self,
        idx_B: Union[str, dict],
        dict_probs: dict,
        dbes: DBESNet = None,
        filter_idxs: set = None,
        STAT_ERR: float = 0,
    ) -> set:
        """
        Recalculate probabilities P(A|not B) for dict of probs and False idx B. B - root, and A - all childs.
        The set of positive cases A is contained in the set of positive cases B.
        Hierarchical structure. Adjacent sets do not intersect.
        """
        dbes = self._dbes if not dbes else dbes

        if dbes:
            fsi = dbes.find_all_dir_rel_idxs_lvls(
                idx_B if type(idx_B) != dict else idx_B["idx"],
                dir_rel="out",
                filter_idxs=filter_idxs,
            )

            return self.hprob.hierarchic_prob_AInotB(
                idx_B=idx_B,
                dict_probs=dict_probs,
                forward_struct_idxs=fsi,
                filter_idxs=filter_idxs,
                STAT_ERR=STAT_ERR,
            )

        else:
            raise ValueError(
                "DBES instance not found! Init ES with DBES instance, or put in dbes attribute of a function"
            )

    def hierarchic_prob_BInotA(
        self,
        idx_A: Union[str, dict],
        dict_probs: dict,
        dbes: DBESNet = None,
        filter_idxs: set = None,
        STAT_ERR: float = 0,
    ) -> set:
        """
        Recalculate probabilities P(B|not A) for dict of probs and False idx A. B - all roots, and A - child.
        The set of positive cases A is contained in the set of positive cases B.
        Hierarchical structure. Adjacent sets do not intersect.
        """
        dbes = self._dbes if not dbes else dbes

        if dbes:
            bsi = dbes.find_all_dir_rel_idxs_lvls(
                idx_A if type(idx_A) != dict else idx_A["idx"],
                dir_rel="in",
                filter_idxs=filter_idxs,
            )

            return self.hprob.hierarchic_prob_BInotA(
                idx_A=idx_A,
                dict_probs=dict_probs,
                back_struct_idxs=bsi,
                filter_idxs=filter_idxs,
                STAT_ERR=STAT_ERR,
            )

        else:
            raise ValueError(
                "DBES instance not found! Init ES with DBES instance, or put in dbes attribute of a function"
            )

    def hierarchic_all_probs_notA(
        self,
        idx_A: Union[str, dict],
        dict_probs: dict,
        filter_idxs: set = None,
        STAT_ERR: float = 0,
    ) -> set:
        """
        Recalculate all probabilities for dict of probs and False idx A. B - all nodes in filter_idxs, and A - some node.
        Hierarchical structure. Adjacent sets do not intersect.
        """
        return self.hprob.hierarchic_all_probs_notA(
            idx_A=idx_A,
            dict_probs=dict_probs,
            filter_idxs=filter_idxs,
            STAT_ERR=STAT_ERR,
        )

    def acyclic_prob_AIB(
        self,
        idx_B: Union[str, dict],
        dict_probs: dict,
        idx_to_divided_idx_prob: dict,
        dbes: DBESNet = None,
        STAT_ERR: float = 0,
    ) -> set:
        """
        Recalculate probabilitiies P(A|B) for dict of probs and True idx B. B - root, and A - all childs.
        The set of positive cases A can be contained in the set of positive cases B, C, D and another.
        Acyclic structure. Adjacent sets can intersect.
        """
        dbes = self._dbes if not dbes else dbes

        if dbes:
            return self.aprob.acyclic_prob_AIB(
                idx_B=idx_B,
                dict_probs=dict_probs,
                idx_to_divided_idx_prob=idx_to_divided_idx_prob,
                STAT_ERR=STAT_ERR,
            )

        else:
            raise ValueError(
                "DBES instance not found! Init ES with DBES instance, or put in dbes attribute of a function"
            )

    # def acyclic_prob_BIA(self, idx_A: Union[str, dict], dict_probs: dict, idx_to_divided_idx_prob: dict, dbes: DBESNet=None, filter_idxs: set=None, STAT_ERR: float = 0) -> set:
    #     """
    #     Recalculate probabilities P(B|A) for dict of probs and True idx A. B - all roots, and A - child.
    #     The set of positive cases A can be contained in the set of positive cases B, C, D and another.
    #     Acyclic structure. Adjacent sets can intersect.
    #     """
    #     dbes = self._dbes if not dbes else dbes

    #     if dbes:
    #         bsi = dbes.find_all_dir_rel_idxs_lvls(idx_A if type(idx_A) != dict else idx_A['idx'], dir_rel="in", filter_idxs=filter_idxs, use_filter_idxs_for_recursive=False, use_filter_idxs_after=True)

    #         return self.aprob.acyclic_prob_BIA(idx_A                     = idx_A,
    #                                            dict_probs                = dict_probs,
    #                                            idx_to_divided_idx_prob   = idx_to_divided_idx_prob,
    #                                            back_struct_idxs          = bsi,
    #                                            filter_idxs               = filter_idxs,
    #                                            STAT_ERR                  = STAT_ERR)

    #     else: raise ValueError("DBES instance not found! Init ES with DBES instance, or put in dbes attribute of a function")

    def acyclic_prob_AInotB(
        self,
        idx_B: Union[str, dict],
        dict_probs: dict,
        idx_to_divided_idx_prob: dict,
        dbes: DBESNet = None,
        STAT_ERR: float = 0,
    ) -> set:
        """
        Recalculate probabilities P(A|not B) for dict of probs and False idx B. B - root, and A - all childs.
        The set of positive cases A can be contained in the set of positive cases B, C, D and another.
        Acyclic structure. Adjacent sets can intersect.
        """
        dbes = self._dbes if not dbes else dbes

        if dbes:
            return self.aprob.acyclic_prob_AInotB(
                idx_B=idx_B,
                dict_probs=dict_probs,
                idx_to_divided_idx_prob=idx_to_divided_idx_prob,
                STAT_ERR=STAT_ERR,
            )

        else:
            raise ValueError(
                "DBES instance not found! Init ES with DBES instance, or put in dbes attribute of a function"
            )

    # def acyclic_prob_BInotA(self, idx_A: Union[str, dict], dict_probs: dict, idx_to_divided_idx_prob: dict, dbes: DBESNet=None, filter_idxs: set=None, STAT_ERR: float = 0) -> set:
    #     """
    #     Recalculate probabilities P(B|not A) for dict of probs and False idx A. B - all roots, and A - child.
    #     The set of positive cases A can be contained in the set of positive cases B, C, D and another.
    #     Acyclic structure. Adjacent sets can intersect.
    #     """
    #     dbes = self._dbes if not dbes else dbes

    #     if dbes:
    #         bsi = dbes.find_all_dir_rel_idxs_lvls(idx_A if type(idx_A) != dict else idx_A['idx'], dir_rel="in", filter_idxs=filter_idxs, use_filter_idxs_for_recursive=False, use_filter_idxs_after=True)

    #         return self.aprob.acyclic_prob_BInotA(idx_A                      = idx_A,
    #                                               dict_probs                 = dict_probs,
    #                                               idx_to_divided_idx_prob    = idx_to_divided_idx_prob,
    #                                               back_struct_idxs           = bsi,
    #                                               filter_idxs                = filter_idxs,
    #                                               STAT_ERR                   = STAT_ERR)

    #     else: raise ValueError("DBES instance not found! Init ES with DBES instance, or put in dbes attribute of a function")

    @caching_profile
    def probq_one_lvl_one_target(
        self,
        target_rel_idxs: dict,
        question_rel_idxs: dict,
        answers: dict,
        dbes: DBESNet = None,
        type_output: str = "full",
        top_targets_output: int = None,
        top_questions_output: int = None,
        **kwargs,
    ) -> dict:
        """
        One level graph structure.
        One target (choose from multiple end targets).
        """
        dbes = self._dbes if not dbes else dbes

        if dbes:
            targets = dbes.find_mult_rel_idxs(target_rel_idxs, and_on=False)
            questions = dbes.find_mult_rel_idxs(question_rel_idxs, and_on=False)

            prob_q = {target: {"questions": {}, "conditions": {}} for target in targets}

            for target in targets:
                for rel in dbes._net[target].relation["in"].keys():
                    if rel in questions:
                        prob_q[target]["questions"][rel] = dbes._net[target].relation[
                            "in"
                        ][rel]["prob"]

            pq = ProbQ(prob_q)

            targets = {}

            for question, answer in answers.items():
                targets = pq.find(question, answer)

            if type_output == "full":
                if top_questions_output:
                    pq.current_questions = pq.current_questions[:top_questions_output]
                if top_targets_output:
                    targets = targets[:top_targets_output]

                return {
                    "remaining_questions": [
                        [{idx: {"value": dbes._net[idx].value}}, val_1]
                        for (idx, val_1) in pq.current_questions
                    ],
                    "targets": [
                        [{idx: {"value": dbes._net[idx].value}}, val_1, val_2]
                        for (idx, val_1, val_2) in targets
                    ],
                }

            elif type_output == "short":
                return {
                    "remaining_questions": pq.current_questions
                    if not top_questions_output
                    else pq.current_questions[:top_questions_output],
                    "targets": targets
                    if not top_targets_output
                    else targets[:top_targets_output],
                }

            else:
                raise TypeError("ERROR! type_output can be 'full' or 'short'")

        else:
            raise ValueError(
                "DBES instance not found! Init ES with DBES instance, or put in dbes attribute of a function"
            )

    @caching_profile
    def _prepare_probq_MHA_lvl_one_target(
        self,
        question_rel_idxs: dict,
        target_rel_idxs: dict,
        true_stats_idxs: dict,
        all_stats_idxs: dict,
        dbes: DBESNet = None,
        **kwargs,
    ) -> Tuple[set, set, dict, dict, dict, float, float]:
        dbes = self._dbes if not dbes else dbes

        if dbes:
            _profile = kwargs.get("_profile")
            _runtype = kwargs.get("_runtype")

            # Sum stats of only true events:
            sum_true_stats = dbes.get_sum_of_values(
                dbes.find_mult_rel_idxs(
                    true_stats_idxs,
                    and_on=False,
                    _profile=f"{_profile}@@_prepare_probq_MHA_lvl_one_target_sum_true_stats"
                    if _profile
                    else None,
                    _runtype="return",
                )
            )

            # All events:
            sum_all_stats = dbes.get_sum_of_values(
                dbes.find_mult_rel_idxs(
                    all_stats_idxs,
                    and_on=False,
                    _profile=f"{_profile}@@_prepare_probq_MHA_lvl_one_target_sum_all_stats"
                    if _profile
                    else None,
                    _runtype="return",
                )
            )

            # Function for converting quantitative statistics to global probability:
            fx = lambda x: (x + 0.0000001) / sum_all_stats

            # Getting an sorted list of all targets (sorting by target probability stat):
            fi_s = dbes.find_mult_rel_idxs(
                target_rel_idxs,
                and_on=False,
                _profile=f"{_profile}@@_prepare_probq_MHA_lvl_one_target_fis"
                if _profile
                else None,
                _runtype="return",
            )

            all_targets_sort = dbes.sort(
                fi_s,
                {"value_relation": true_stats_idxs},
                fx=fx,
                _profile=f"{_profile}@@_prepare_probq_MHA_lvl_one_target_targets"
                if _profile
                else None,
                _runtype="return",
            )

            all_targets_dict = {idx: val for idx, val in all_targets_sort}

            # Getting an sorted list of all questions (Sorting by question's relation to popular targets):
            fi = dbes.find_mult_rel_idxs(
                question_rel_idxs,
                and_on=False,
                _profile=f"{_profile}@@_prepare_probq_MHA_lvl_one_target_fi"
                if _profile
                else None,
                _runtype="return",
            )

            all_questions_sort, tag_to_service_div_prob = dbes.sort(
                fi,
                {"value_recursive_relation": true_stats_idxs},
                fx=fx,
                filter_idxs=fi_s,
                return_divided_map=True,
                _profile=f"{_profile}@@_prepare_probq_MHA_lvl_one_target_questions"
                if _profile
                else None,
                _runtype="return",
            )

            all_questions_dict = {idx: val for idx, val in all_questions_sort}

            # A dictionary showing how much of a certain goal is "contained" in a certain question:
            tag_to_service_div_prob = {
                idx: {
                    service_idx: fx(val) for service_idx, val in services_idxs.items()
                }
                for idx, services_idxs in tag_to_service_div_prob.items()
            }

            tag_to_service_div_prob = {**tag_to_service_div_prob, **all_targets_dict}

        return (
            fi,
            fi_s,
            all_questions_dict,
            tag_to_service_div_prob,
            all_targets_dict,
            sum_true_stats,
            sum_all_stats,
        )

    @caching_profile
    def probq_MHA_lvl_one_target(
        self,
        question_rel_idxs: dict,
        target_rel_idxs: dict,
        answers: dict,
        true_stats_idxs: dict,
        all_stats_idxs: dict,
        question_add_props_idxs: dict = None,
        target_add_props_idxs: dict = None,
        stat_error_answers: dict = None,
        dbes: DBESNet = None,
        type_output: str = "full",
        top_questions_output: int = None,
        top_targets_output: int = None,
        sort_output: bool = True,
        return_questions: bool = True,
        **kwargs,
    ) -> dict:
        """
        It is intended for variable passage through the tree of questions/tags with finding suitable target.
        M: Multi level graph structure.
        H: Hierarchical structure of questions.
        A: Acyclic structure of questions - targets.

        Structure:
        ----------
        - It is assumed that the questionnaire has a hierarchical structure (the parent node contains all child nodes, there are no intersections between adjacent nodes).
        - But tags can link to several services, and services can receive links from different tags (at the service level, the hierarchy collapses).

        Args:
        ----------
           ` _profile` : Use specific caching profile for some of inner functions.
        """
        dbes = self._dbes if not dbes else dbes

        if dbes:
            _profile = kwargs.get("_profile")
            _runtype = kwargs.get("_runtype")

            if not question_add_props_idxs:
                question_add_props_idxs = {}
            if not target_add_props_idxs:
                target_add_props_idxs = {}

            (
                fi,
                fi_s,
                all_questions_dict,
                tag_to_service_div_prob,
                all_targets_dict,
                sum_true_stats,
                sum_all_stats,
            ) = self._prepare_probq_MHA_lvl_one_target(
                question_rel_idxs=question_rel_idxs,
                target_rel_idxs=target_rel_idxs,
                true_stats_idxs=true_stats_idxs,
                all_stats_idxs=all_stats_idxs,
                _profile=f"{_profile}@@probq_MHA_lvl_one_target_all_questions_dict"
                if _profile
                else None,
                _runtype="return",
                dbes=dbes,
            )

            all_questions_dict = {**all_questions_dict}
            all_targets_dict = {**all_targets_dict}

            # Stat error based on the relationship between true events and all number of events:
            STAT_ERR = (sum_all_stats - sum_true_stats) / sum_all_stats
            STAT_ERR_INIT = STAT_ERR

            # ? Пока единственная ошибка (или недочёт) есть с тем, что после ответа на конечный тег\вопрос вероятности вопросов начинают все сильно убывать, а вероятности услуг сильно возрастать за единицу
            for current_question, input_text in answers.items():
                if stat_error_answers:
                    STAT_ERR = STAT_ERR_INIT + stat_error_answers[current_question]

                curr_q = {
                    "idx": current_question,
                    "val": all_questions_dict[current_question],
                }
                recalc_nodes = set([])
                if input_text == "on":
                    recalc_nodes.update(
                        self.hierarchic_prob_AIB(
                            current_question,
                            all_questions_dict,
                            dbes=dbes,
                            filter_idxs=fi,
                            STAT_ERR=STAT_ERR,
                        )
                    )
                    recalc_nodes.update(
                        self.hierarchic_prob_BIA(
                            current_question,
                            all_questions_dict,
                            dbes=dbes,
                            filter_idxs=fi,
                            STAT_ERR=STAT_ERR,
                        )
                    )
                    recalc_nodes.update(
                        self.hierarchic_all_probs_A(
                            current_question,
                            all_questions_dict,
                            filter_idxs=fi.difference(recalc_nodes),
                            STAT_ERR=STAT_ERR,
                        )
                    )

                    recalc_nodes.update(
                        self.acyclic_prob_AIB(
                            curr_q,
                            all_targets_dict,
                            tag_to_service_div_prob,
                            dbes=dbes,
                            STAT_ERR=STAT_ERR,
                        )
                    )
                    recalc_nodes.update(
                        self.hierarchic_all_probs_A(
                            curr_q,
                            all_targets_dict,
                            filter_idxs=fi_s.difference(recalc_nodes),
                            STAT_ERR=STAT_ERR,
                        )
                    )

                elif input_text == "off":
                    recalc_nodes.update(
                        self.hierarchic_prob_AInotB(
                            current_question,
                            all_questions_dict,
                            dbes=dbes,
                            filter_idxs=fi,
                            STAT_ERR=STAT_ERR,
                        )
                    )
                    recalc_nodes.update(
                        self.hierarchic_prob_BInotA(
                            current_question,
                            all_questions_dict,
                            dbes=dbes,
                            filter_idxs=fi,
                            STAT_ERR=STAT_ERR,
                        )
                    )
                    recalc_nodes.update(
                        self.hierarchic_all_probs_notA(
                            current_question,
                            all_questions_dict,
                            filter_idxs=fi.difference(recalc_nodes),
                            STAT_ERR=STAT_ERR,
                        )
                    )

                    recalc_nodes.update(
                        self.acyclic_prob_AInotB(
                            curr_q,
                            all_targets_dict,
                            tag_to_service_div_prob,
                            dbes=dbes,
                            STAT_ERR=STAT_ERR,
                        )
                    )
                    recalc_nodes.update(
                        self.hierarchic_all_probs_notA(
                            curr_q,
                            all_targets_dict,
                            filter_idxs=fi_s.difference(recalc_nodes),
                            STAT_ERR=STAT_ERR,
                        )
                    )

                del all_questions_dict[current_question]

            all_questions_sort = []
            if return_questions:
                all_questions_sort = [
                    [idx, val] for idx, val in all_questions_dict.items()
                ]
            all_targets_sort = [[idx, val] for idx, val in all_targets_dict.items()]

            if sort_output:
                if return_questions:
                    all_questions_sort = sorted(
                        all_questions_sort, key=itemgetter(1), reverse=True
                    )
                all_targets_sort = sorted(
                    all_targets_sort, key=itemgetter(1), reverse=True
                )

            if type_output == "full":
                if top_questions_output:
                    all_questions_sort = all_questions_sort[:top_questions_output]
                if top_targets_output:
                    all_targets_sort = all_targets_sort[:top_targets_output]

                return {
                    "remaining_questions": [
                        {
                            "idx": idx,
                            "value": dbes._net[idx].value,
                            "prob": coeff,
                            **{
                                prop: dbes.get_value_rel(idx, idx_prop)
                                for prop, idx_prop in question_add_props_idxs.items()
                            },
                        }
                        for (idx, coeff) in all_questions_sort
                    ],
                    "targets": [
                        {
                            "idx": idx,
                            "value": dbes._net[idx].value,
                            "prob": coeff,
                            **{
                                prop: dbes.get_value_rel(idx, idx_prop)
                                for prop, idx_prop in target_add_props_idxs.items()
                            },
                        }
                        for (idx, coeff) in all_targets_sort
                    ],
                }

            elif type_output == "short":
                return {
                    "remaining_questions": all_questions_sort
                    if not top_questions_output
                    else all_questions_sort[:top_questions_output],
                    "targets": all_targets_sort
                    if not top_targets_output
                    else all_targets_sort[:top_targets_output],
                }

            else:
                raise TypeError("ERROR! type_output can be 'full' or 'short'")

        else:
            raise ValueError(
                "DBES instance not found! Init ES with DBES instance, or put in dbes attribute of a function"
            )

    @caching_profile
    def _prepare_probq_MnHA_lvl_one_target(
        self,
        question_rel_idxs: dict,
        target_rel_idxs: dict,
        true_stats_idxs: dict,
        all_stats_idxs: dict,
        dbes: DBESNet = None,
        **kwargs,
    ) -> Tuple[dict, dict, dict, float, float]:
        dbes = self._dbes if not dbes else dbes

        if dbes:
            _profile = kwargs.get("_profile")
            _runtype = kwargs.get("_runtype")

            # Sum stats of only true events:
            sum_true_stats = dbes.get_sum_of_values(
                dbes.find_mult_rel_idxs(
                    true_stats_idxs,
                    and_on=False,
                    _profile=f"{_profile}@@_prepare_probq_MnHA_lvl_one_target_sum_true_stats"
                    if _profile
                    else None,
                    _runtype="return",
                )
            )

            # All events:
            sum_all_stats = dbes.get_sum_of_values(
                dbes.find_mult_rel_idxs(
                    all_stats_idxs,
                    and_on=False,
                    _profile=f"{_profile}@@_prepare_probq_MnHA_lvl_one_target_sum_all_stats"
                    if _profile
                    else None,
                    _runtype="return",
                )
            )

            # Function for converting quantitative statistics to global probability:
            fx = lambda x: (x + 0.0000001) / sum_all_stats

            # Getting an sorted list of all targets (sorting by target probability stat):
            fi_s = dbes.find_mult_rel_idxs(
                target_rel_idxs,
                and_on=False,
                _profile=f"{_profile}@@_prepare_probq_MnHA_lvl_one_target_fis"
                if _profile
                else None,
                _runtype="return",
            )

            all_targets_sort = dbes.sort(
                fi_s,
                {"value_relation": true_stats_idxs},
                fx=fx,
                _profile=f"{_profile}@@_prepare_probq_MnHA_lvl_one_target_targets"
                if _profile
                else None,
                _runtype="return",
            )

            all_targets_dict = {idx: 0.001 for idx, val in all_targets_sort}

            # Getting an sorted list of all questions (Sorting by sum of question's relation to targets):
            fi = dbes.find_mult_rel_idxs(
                question_rel_idxs,
                and_on=False,
                _profile=f"{_profile}@@_prepare_probq_MnHA_lvl_one_target_fi"
                if _profile
                else None,
                _runtype="return",
            )

            all_questions_sort, ques_to_target_coeff = dbes.sort(
                fi,
                {"value_union_recursive_relation": true_stats_idxs},
                fx=fx,
                filter_idxs=fi_s,
                common_filter_idxs={*fi_s, *fi},
                return_rel_map=True,
                _profile=f"{_profile}@@_prepare_probq_MnHA_lvl_one_target_questions"
                if _profile
                else None,
                _runtype="return",
            )

            all_questions_dict = {idx: val for idx, val in all_questions_sort}

        return (
            all_questions_dict,
            ques_to_target_coeff,
            all_targets_dict,
            sum_true_stats,
            sum_all_stats,
        )

    @caching_profile
    def probq_MnHA_lvl_one_target(
        self,
        question_rel_idxs: dict,
        target_rel_idxs: dict,
        answers: dict,
        true_stats_idxs: dict,
        all_stats_idxs: dict,
        question_add_props_idxs: dict = None,
        target_add_props_idxs: dict = None,
        stat_error_answers: dict = None,
        dbes: DBESNet = None,
        type_output: str = "full",
        top_questions_output: int = None,
        top_targets_output: int = None,
        sort_output: bool = True,
        return_questions: bool = True,
        **kwargs,
    ) -> dict:
        """
        It is intended for variable passage through the tree of questions/tags with finding suitable target.
        M: Multi level graph structure.
        nH: Non hierarchical structure of questions.
        A: Acyclic structure of questions - targets.

        Structure:
        ----------
        - It is assumed that the questionnaire can have or not hierarchical structure (the parent node not contains all child nodes as summary (but contains its union), there are can be intersections between adjacent nodes).
        - But tags can link to several services, and services can receive links from different tags.

        Args:
        ----------
           ` _profile` : Use specific caching profile for some of inner functions.
        """
        dbes = self._dbes if not dbes else dbes

        if dbes:
            _profile = kwargs.get("_profile")
            _runtype = kwargs.get("_runtype")

            if not question_add_props_idxs:
                question_add_props_idxs = {}
            if not target_add_props_idxs:
                target_add_props_idxs = {}

            (
                all_questions_dict,
                ques_to_target_coeff,
                all_targets_dict,
                sum_true_stats,
                sum_all_stats,
            ) = self._prepare_probq_MnHA_lvl_one_target(
                question_rel_idxs=question_rel_idxs,
                target_rel_idxs=target_rel_idxs,
                true_stats_idxs=true_stats_idxs,
                all_stats_idxs=all_stats_idxs,
                _profile=f"{_profile}@@probq_MnHA_lvl_one_target_all_questions_dict"
                if _profile
                else None,
                _runtype="return",
                dbes=dbes,
            )

            all_questions_dict = {**all_questions_dict}
            all_targets_dict = {**all_targets_dict}

            # Stat error based on the relationship between true events and all number of events:
            STAT_ERR = (sum_all_stats - sum_true_stats) / sum_all_stats
            STAT_ERR_REV = 1 - STAT_ERR
            STAT_ERR_INIT = STAT_ERR

            # Function for converting quantitative statistics to global probability:
            fx = lambda x: (x + 0.0000001) / sum_all_stats

            answered_nodes = set([])
            all_questions_dict_answ = {}

            for current_question, input_text in answers.items():
                if stat_error_answers:
                    STAT_ERR = STAT_ERR_INIT + stat_error_answers[current_question]
                    STAT_ERR_REV = 1 - STAT_ERR

                if input_text == "on":
                    for node_idx, coeff in ques_to_target_coeff[
                        current_question
                    ].items():
                        if node_idx in all_targets_dict:
                            all_targets_dict[node_idx] += (
                                fx(coeff)
                                * (1 - all_questions_dict[current_question])
                                * STAT_ERR_REV
                            )
                        elif return_questions and node_idx in all_questions_dict:
                            if node_idx not in all_questions_dict_answ:
                                all_questions_dict_answ[node_idx] = (
                                    fx(coeff)
                                    * (1 - all_questions_dict[current_question])
                                    * STAT_ERR_REV
                                )
                            else:
                                all_questions_dict_answ[node_idx] += (
                                    fx(coeff)
                                    * (1 - all_questions_dict[current_question])
                                    * STAT_ERR_REV
                                )

                # elif input_text == 'off':             #? how it shoud works?
                if return_questions:
                    answered_nodes.add(current_question)

            for idx, val in all_questions_dict_answ.items():
                all_questions_dict[idx] = val

            all_questions_sort = []
            if return_questions:
                all_questions_sort = [
                    [idx, val]
                    for idx, val in all_questions_dict.items()
                    if idx not in answered_nodes
                ]
            all_targets_sort = [[idx, val] for idx, val in all_targets_dict.items()]

            if sort_output:
                if return_questions:
                    all_questions_sort = sorted(
                        all_questions_sort, key=itemgetter(1), reverse=True
                    )
                all_targets_sort = sorted(
                    all_targets_sort, key=itemgetter(1), reverse=True
                )

            if type_output == "full":
                if top_questions_output:
                    all_questions_sort = all_questions_sort[:top_questions_output]
                if top_targets_output:
                    all_targets_sort = all_targets_sort[:top_targets_output]

                return {
                    "remaining_questions": [
                        {
                            "idx": idx,
                            "value": dbes._net[idx].value,
                            "prob": coeff,
                            **{
                                prop: dbes.get_value_rel(idx, idx_prop)
                                for prop, idx_prop in question_add_props_idxs.items()
                            },
                        }
                        for (idx, coeff) in all_questions_sort
                    ],
                    "targets": [
                        {
                            "idx": idx,
                            "value": dbes._net[idx].value,
                            "prob": coeff,
                            **{
                                prop: dbes.get_value_rel(idx, idx_prop)
                                for prop, idx_prop in target_add_props_idxs.items()
                            },
                        }
                        for (idx, coeff) in all_targets_sort
                    ],
                }

            elif type_output == "short":
                return {
                    "remaining_questions": all_questions_sort
                    if not top_questions_output
                    else all_questions_sort[:top_questions_output],
                    "targets": all_targets_sort
                    if not top_targets_output
                    else all_targets_sort[:top_targets_output],
                }

            else:
                raise TypeError("ERROR! type_output can be 'full' or 'short'")

        else:
            raise ValueError(
                "DBES instance not found! Init ES with DBES instance, or put in dbes attribute of a function"
            )

    @caching_profile
    def _prepare_text_find_one_lvl_one_target(
        self,
        target_rel_idxs: dict,
        target_stats_idxs: dict = None,
        target_rule_sort: str = None,
        vocab_rel_idxs: dict = None,
        target_add_text_idxs: dict = None,
        dbes: DBESNet = None,
        **kwargs,
    ) -> Tuple[dict, dict, dict]:
        dbes = self._dbes if not dbes else dbes

        all_targets_dict = {}
        all_targets_words = {}
        target_words_to_target_idxs = {}

        if dbes:
            _profile = kwargs.get("_profile")
            _runtype = kwargs.get("_runtype")

            # Vocabulary of words (using for synonyms search):
            vocab_idxs = set([])
            vocab_syn_dist = {}
            if vocab_rel_idxs:
                vocab_idxs = dbes.find_mult_rel_idxs(
                    vocab_rel_idxs,
                    and_on=False,
                    _profile=f"{_profile}@@_prepare_text_find_one_lvl_one_target_vocab_idxs"
                    if _profile
                    else None,
                    _runtype="return",
                )

                vocab_syn_dist = dbes.find_longest_mult_dist_fw(
                    filter_idxs=vocab_idxs,
                    type_amx="float",
                    symetric=False,
                    to_val=True,
                    _profile=f"{_profile}@@_prepare_text_find_one_lvl_one_target_vocab_syn_dist"
                    if _profile
                    else None,
                    _runtype="return",
                )

            fx = None
            if target_stats_idxs:
                # Sum stats of events:
                sum_stats = dbes.get_sum_of_values(
                    dbes.find_mult_rel_idxs(
                        target_stats_idxs,
                        and_on=False,
                        _profile=f"{_profile}@@_prepare_text_find_one_lvl_one_target_sum_true_stats"
                        if _profile
                        else None,
                        _runtype="return",
                    )
                )

                # Function for converting quantitative statistics to global probability:
                fx = lambda x: (x + 0.0000001) / sum_stats

            # Getting an sorted list of all targets (sorting by target probability stat):
            fi_s = dbes.find_mult_rel_idxs(
                target_rel_idxs,
                and_on=False,
                _profile=f"{_profile}@@_prepare_text_find_one_lvl_one_target_fis"
                if _profile
                else None,
                _runtype="return",
            )

            all_targets_sort = (
                dbes.sort(
                    finded_idxs=fi_s,
                    rule_sort={target_rule_sort: target_stats_idxs}
                    if target_rule_sort
                    else {"value_recursive_relation": target_stats_idxs},
                    fx=fx,
                    _profile=f"{_profile}@@_prepare_text_find_one_lvl_one_target_targets"
                    if _profile
                    else None,
                    _runtype="return",
                )
                if target_stats_idxs
                else sorted([(idx, 1) for idx in fi_s], key=itemgetter(1), reverse=True)
            )

            all_targets_dict = {idx: coeff for idx, coeff in all_targets_sort}

            for target_idx, coeff in all_targets_sort:
                tokenized_words = self.tfind.tokenize(
                    dbes._net[target_idx].value.lower(), lemmatize=False
                )

                coeff_init = coeff
                if target_add_text_idxs:
                    coeff = coeff_init * 1  # ? how estimate this coeff
                coeff = coeff / len(tokenized_words)

                for word in tokenized_words:
                    lemm_word = ""

                    if vocab_syn_dist:
                        lemm_word = self.tfind.lemmatize(word)

                        if lemm_word not in vocab_syn_dist:
                            self.tfind.count_tokens_to_target(
                                word,
                                target_idx=target_idx,
                                coeff=coeff,
                                tags=all_targets_words,
                                word_to_target_idxs=target_words_to_target_idxs,
                            )

                        else:
                            for synonyms in vocab_syn_dist[
                                lemm_word
                            ]:  # ? maybe need divide coeff for synonyms?
                                for word, syn_coeff in synonyms.items():
                                    self.tfind.count_tokens_to_target(
                                        word,
                                        target_idx=target_idx,
                                        coeff=coeff * syn_coeff,
                                        tags=all_targets_words,
                                        word_to_target_idxs=target_words_to_target_idxs,
                                    )

                    else:
                        self.tfind.count_tokens_to_target(
                            word,
                            target_idx=target_idx,
                            coeff=coeff,
                            tags=all_targets_words,
                            word_to_target_idxs=target_words_to_target_idxs,
                        )

                if target_add_text_idxs:
                    coeff = coeff_init * 1  # ? how estimate this coeff

                    for prop, idx_prop in target_add_text_idxs.items():
                        add_prop = dbes.get_value_rel(target_idx, idx_prop)

                        if add_prop and add_prop != "":
                            tokenized_words = self.tfind.tokenize(
                                add_prop, lemmatize=False
                            )
                            coeff = coeff / len(tokenized_words)

                            for word in tokenized_words:
                                lemm_word = ""

                                if vocab_syn_dist:
                                    lemm_word = self.tfind.lemmatize(word)

                                    if lemm_word not in vocab_syn_dist:
                                        self.tfind.count_tokens_to_target(
                                            word,
                                            target_idx=target_idx,
                                            coeff=coeff,
                                            tags=all_targets_words,
                                            word_to_target_idxs=target_words_to_target_idxs,
                                        )

                                    else:
                                        for synonyms in vocab_syn_dist[lemm_word]:
                                            for word, syn_coeff in synonyms.items():
                                                self.tfind.count_tokens_to_target(
                                                    word,
                                                    target_idx=target_idx,
                                                    coeff=coeff * syn_coeff,
                                                    tags=all_targets_words,
                                                    word_to_target_idxs=target_words_to_target_idxs,
                                                )

                                else:
                                    self.tfind.count_tokens_to_target(
                                        word,
                                        target_idx=target_idx,
                                        coeff=coeff,
                                        tags=all_targets_words,
                                        word_to_target_idxs=target_words_to_target_idxs,
                                    )

        return (all_targets_dict, all_targets_words, target_words_to_target_idxs)

    @caching_profile
    def _init_tfind(
        self,
        tags: dict = None,
        targets: dict = None,
        word_to_target_idxs: Dict[str, CounterT] = None,
        dbes: DBESNet = None,
        **kwargs,
    ) -> ESTargetFinder:
        return ESTargetFinder(
            tags=tags, targets=targets, word_to_target_idxs=word_to_target_idxs
        )

    @caching_profile
    def text_find_one_lvl_one_target(
        self,
        input_text: str,
        target_rel_idxs: dict,
        target_stats_idxs: dict = None,
        target_rule_sort: str = None,
        vocab_rel_idxs: dict = None,
        target_add_text_idxs: dict = None,
        dbes: DBESNet = None,
        target_add_props_idxs: dict = None,
        type_output: str = "full",
        top_targets_output: int = None,
        sort_output: bool = True,
        **kwargs,
    ) -> Union[dict, list]:
        """
        Find targets (represented as text values) from input text.

        Args:
        ----------
           ` _profile` : Use specific caching profile for some of inner functions.
        """
        dbes = self._dbes if not dbes else dbes

        if dbes:
            _profile = kwargs.get("_profile")
            _runtype = kwargs.get("_runtype")

            if not target_add_props_idxs:
                target_add_props_idxs = {}

            (
                all_targets_dict,
                all_targets_words,
                target_words_to_target_idxs,
            ) = self._prepare_text_find_one_lvl_one_target(
                target_rel_idxs=target_rel_idxs,
                target_stats_idxs=target_stats_idxs,
                target_rule_sort=target_rule_sort,
                vocab_rel_idxs=vocab_rel_idxs,
                target_add_text_idxs=target_add_text_idxs,
                _profile=f"{_profile}@@text_find_one_lvl_one_target_all_questions_words"
                if _profile
                else None,
                _runtype="return",
                dbes=dbes,
            )

            self.tfind = self._init_tfind(
                tags=all_targets_words,
                word_to_target_idxs=target_words_to_target_idxs,
                dbes=dbes,
                _profile=f"{_profile}@@text_find_one_lvl_one_target_tfind"
                if _profile
                else None,
                _runtype="return",
                _no_save=True,
            )

            tokens = self.tfind.tokenize(input_text, lemmatize=False)
            actokens = self.tfind.autocomplete(tokens, type_output="only_autocomplete")
            finded_targets_sort = self.tfind.find_targets(
                actokens, sort_output=sort_output
            )

            if type_output == "full":
                if top_targets_output:
                    finded_targets_sort = finded_targets_sort[:top_targets_output]

                return [
                    {
                        "idx": idx,
                        "value": dbes._net[idx].value,
                        "prob": coeff,
                        "rel_prob": coeff / all_targets_dict[idx],
                        **{
                            prop: dbes.get_value_rel(idx, idx_prop)
                            for prop, idx_prop in target_add_props_idxs.items()
                        },
                    }
                    for (idx, coeff) in (
                        finded_targets_sort
                        if type(finded_targets_sort) == list
                        else finded_targets_sort.items()
                    )
                ]

            elif type_output == "short":
                return (
                    finded_targets_sort
                    if not top_targets_output
                    else finded_targets_sort[:top_targets_output]
                )

            else:
                raise TypeError("ERROR! type_output can be 'full' or 'short'")

    @caching_profile
    def text_find_MHA_lvl_one_target(
        self,
        input_text: str,
        question_params: dict,
        target_params: dict,
        true_stats_idxs: dict,
        all_stats_idxs: dict,
        dbes: DBESNet = None,
        target_add_props_idxs: dict = None,
        type_output: str = "full",
        top_targets_output: int = None,
        **kwargs,
    ) -> list:
        """
        Find targets (represented as text values) from input text.
        M: Multi level graph structure.
        H: Hierarchical structure of questions.
        A: Acyclic structure of questions - targets.

        Structure:
        ----------
        - It is assumed that the questionnaire has a hierarchical structure (the parent node contains all child nodes, there are no intersections between adjacent nodes).
        - But tags can link to several services, and services can receive links from different tags (at the service level, the hierarchy collapses).

        Args:
        ----------
           ` _profile` : Use specific caching profile for some of inner functions.
        """
        dbes = self._dbes if not dbes else dbes

        if dbes:
            _profile = kwargs.get("_profile")
            _runtype = kwargs.get("_runtype")

            if not target_add_props_idxs:
                target_add_props_idxs = {}

            finded_questions_sort = self.text_find_one_lvl_one_target(
                input_text=input_text,
                **question_params,
                type_output="full",
                sort_output=False,
                dbes=dbes,
                _profile=f"{_profile}@@text_find_MHA_lvl_one_target_finded_questions_sort"
                if _profile
                else None,
                _runtype="args",
                _ignore_attr={
                    "input_text",
                    "type_output",
                    "dbes",
                    "_profile",
                    "_runtype",
                    "_ignore_attr",
                },
            )

            finded_targets_sort = self.text_find_one_lvl_one_target(
                input_text=input_text,
                **target_params,
                type_output="short",
                sort_output=False,
                dbes=dbes,
                _profile=f"{_profile}@@text_find_MHA_lvl_one_target_finded_targets_sort"
                if _profile
                else None,
                _runtype="args",
                _ignore_attr={
                    "input_text",
                    "type_output",
                    "dbes",
                    "_profile",
                    "_runtype",
                    "_ignore_attr",
                },
            )

            if finded_questions_sort:
                REL_PROB_LOWER_BORDER = 0.5  # ? what a value in this?
                MIN_VAL_ERR_REL_PROB = 0.001
                PROBQ_COEFF = 0.1

                answers = {
                    finded_question["idx"]: "on"
                    for finded_question in finded_questions_sort
                    if finded_question["rel_prob"] > REL_PROB_LOWER_BORDER
                }
                stat_error_answers = {
                    finded_question["idx"]: 1 - finded_question["rel_prob"]
                    if finded_question["rel_prob"] < 1
                    else MIN_VAL_ERR_REL_PROB
                    for finded_question in finded_questions_sort
                }

                all_targets_sort = self.probq_MHA_lvl_one_target(
                    question_rel_idxs=question_params["target_rel_idxs"],
                    target_rel_idxs=target_params["target_rel_idxs"],
                    answers=answers,  # ? maybe split answers on line-roads ? (from root to leaves).
                    stat_error_answers=stat_error_answers,
                    true_stats_idxs=true_stats_idxs,
                    all_stats_idxs=all_stats_idxs,
                    type_output="short",
                    sort_output=False,
                    return_questions=False,
                    dbes=dbes,
                    _profile=f"{_profile}@@text_find_MHA_lvl_one_target_all_targets_sort"
                    if _profile
                    else None,
                    _runtype="args",
                    _ignore_attr={
                        "answers",
                        "stat_error_answers",
                        "type_output",
                        "dbes",
                        "_profile",
                        "_runtype",
                        "_ignore_attr",
                    },
                )

                if finded_targets_sort:
                    all_targets_sort = [
                        (idx, coeff * PROBQ_COEFF + finded_targets_sort.get(idx, 0))
                        for idx, coeff in all_targets_sort["targets"]
                    ]  # ? what operation need? + * or ?   and coeffs?
                else:
                    all_targets_sort = all_targets_sort["targets"]

                all_targets_sort = sorted(
                    all_targets_sort, key=itemgetter(1), reverse=True
                )

                if type_output == "full":
                    if top_targets_output:
                        all_targets_sort = all_targets_sort[:top_targets_output]

                    return [
                        {
                            "idx": idx,
                            "value": dbes._net[idx].value,
                            "prob": coeff,
                            "coeff": finded_targets_sort.get(idx, 0)
                            if finded_targets_sort
                            else 0,
                            **{
                                prop: dbes.get_value_rel(idx, idx_prop)
                                for prop, idx_prop in target_add_props_idxs.items()
                            },
                        }
                        for idx, coeff in all_targets_sort
                    ]

                elif type_output == "short":
                    return (
                        all_targets_sort
                        if not top_targets_output
                        else all_targets_sort[:top_targets_output]
                    )

                else:
                    raise TypeError("ERROR! type_output can be 'full' or 'short'")

            else:
                return []

    @caching_profile
    def text_find_MnHA_lvl_one_target(
        self,
        input_text: str,
        question_params: dict,
        target_params: dict,
        true_stats_idxs: dict,
        all_stats_idxs: dict,
        dbes: DBESNet = None,
        target_add_props_idxs: dict = None,
        type_output: str = "full",
        top_targets_output: int = None,
        **kwargs,
    ) -> list:
        """
        Find targets (represented as text values) from input text.
        M: Multi level graph structure.
        nH: Non hierarchical structure of questions.
        A: Acyclic structure of questions - targets.

        Structure:
        ----------
        - It is assumed that the questionnaire can have or not hierarchical structure (the parent node not contains all child nodes as summary (but contains its union), there are can be intersections between adjacent nodes).
        - But tags can link to several services, and services can receive links from different tags.

        Args:
        ----------
           ` _profile` : Use specific caching profile for some of inner functions.
        """
        dbes = self._dbes if not dbes else dbes

        if dbes:
            _profile = kwargs.get("_profile")
            _runtype = kwargs.get("_runtype")

            if not target_add_props_idxs:
                target_add_props_idxs = {}

            finded_questions_sort = self.text_find_one_lvl_one_target(
                input_text=input_text,
                **question_params,
                type_output="full",
                sort_output=False,
                dbes=dbes,
                _profile=f"{_profile}@@text_find_MnHA_lvl_one_target_finded_questions_sort"
                if _profile
                else None,
                _runtype="args",
                _ignore_attr={
                    "input_text",
                    "type_output",
                    "dbes",
                    "_profile",
                    "_runtype",
                    "_ignore_attr",
                },
            )

            finded_targets_sort = self.text_find_one_lvl_one_target(
                input_text=input_text,
                **target_params,
                type_output="short",
                sort_output=False,
                dbes=dbes,
                _profile=f"{_profile}@@text_find_MnHA_lvl_one_target_finded_targets_sort"
                if _profile
                else None,
                _runtype="args",
                _ignore_attr={
                    "input_text",
                    "type_output",
                    "dbes",
                    "_profile",
                    "_runtype",
                    "_ignore_attr",
                },
            )

            if finded_questions_sort:
                REL_PROB_LOWER_BORDER = 0.01  # ? what a value in this?
                MIN_VAL_ERR_REL_PROB = 0.001
                PROBQ_COEFF = 0.9

                answers = {
                    finded_question["idx"]: "on"
                    for finded_question in finded_questions_sort
                    if finded_question["rel_prob"] > REL_PROB_LOWER_BORDER
                }
                stat_error_answers = {
                    finded_question["idx"]: 1 - finded_question["rel_prob"]
                    if finded_question["rel_prob"] < 1
                    else MIN_VAL_ERR_REL_PROB
                    for finded_question in finded_questions_sort
                }

                all_targets_sort = self.probq_MnHA_lvl_one_target(
                    question_rel_idxs=question_params["target_rel_idxs"],
                    target_rel_idxs=target_params["target_rel_idxs"],
                    answers=answers,
                    stat_error_answers=stat_error_answers,
                    true_stats_idxs=true_stats_idxs,
                    all_stats_idxs=all_stats_idxs,
                    type_output="short",
                    sort_output=False,
                    return_questions=False,
                    dbes=dbes,
                    _profile=f"{_profile}@@text_find_MnHA_lvl_one_target_all_targets_sort"
                    if _profile
                    else None,
                    _runtype="args",
                    _ignore_attr={
                        "answers",
                        "stat_error_answers",
                        "type_output",
                        "dbes",
                        "_profile",
                        "_runtype",
                        "_ignore_attr",
                    },
                )

                if finded_targets_sort:
                    all_targets_sort = [
                        (idx, coeff * PROBQ_COEFF + finded_targets_sort.get(idx, 0))
                        for idx, coeff in all_targets_sort["targets"]
                    ]  # ? what operation need? + * or ?   and coeffs?
                else:
                    all_targets_sort = all_targets_sort["targets"]

                all_targets_sort = sorted(
                    all_targets_sort, key=itemgetter(1), reverse=True
                )

                if type_output == "full":
                    if top_targets_output:
                        all_targets_sort = all_targets_sort[:top_targets_output]

                    return [
                        {
                            "idx": idx,
                            "value": dbes._net[idx].value,
                            "prob": coeff,
                            "coeff": finded_targets_sort.get(idx, 0),
                            **{
                                prop: dbes.get_value_rel(idx, idx_prop)
                                for prop, idx_prop in target_add_props_idxs.items()
                            },
                        }
                        for idx, coeff in all_targets_sort
                    ]

                elif type_output == "short":
                    return (
                        all_targets_sort
                        if not top_targets_output
                        else all_targets_sort[:top_targets_output]
                    )

                else:
                    raise TypeError("ERROR! type_output can be 'full' or 'short'")

            else:
                return []
