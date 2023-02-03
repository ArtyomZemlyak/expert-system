class ESAcyclicProb:
    """
    Class for compute probs on DBES graph (data with relations).
    Acyclic structure. Adjacent sets can intersect.
    """

    @staticmethod
    def acyclic_prob_AIB(
        idx_B: str, dict_probs: dict, idx_to_divided_idx_prob: dict, STAT_ERR: float = 0
    ) -> set:
        """
        Recalculate probabilitiies P(A|B) for dict of probs and True idx B. B - root, and A - all childs.
        The set of positive cases A can be contained in the set of positive cases B, C, D and another.
        Acyclic structure. Adjacent sets can intersect.
        P(A|YB) = P(A|Y) * P(A|B) / P(A)
        P(A|Y) = current Node prob
        P(A|B) = P(A and B) / P(B)
        P(A and B) = P(A)   - bc A contained in B
        P(A) = P(A) * (1 - ERR)
        """
        recalculated_nodes = set([])

        prob_B = 0
        if type(idx_B) == str:
            prob_B = dict_probs[idx_B]
        elif type(idx_B) == dict:
            prob_B = idx_B["val"]
            idx_B = idx_B["idx"]

        for node_idx, prob_A_div in idx_to_divided_idx_prob[idx_B].items():
            if (
                node_idx != idx_B and node_idx in dict_probs.keys()
            ):  # ? maybe better change all nodes, but in pprint print only non answered
                prob_AIY = dict_probs[node_idx]
                prob_A = idx_to_divided_idx_prob[node_idx]
                prob_AxB = prob_A_div * (
                    1 - STAT_ERR
                )  # ? need check for all formulas for acyclic
                prob_AIB = prob_AxB / prob_B
                prob_AIYB = prob_AIY * prob_AIB / prob_A
                dict_probs[node_idx] = prob_AIYB
                recalculated_nodes.add(node_idx)

        return recalculated_nodes

    # @staticmethod
    # def acyclic_prob_BIA(idx_A: str, dict_probs: dict, idx_to_divided_idx_prob: dict, STAT_ERR: float=0) -> set:
    #     """
    #     Recalculate probabilities P(B|A) for dict of probs and True idx A. B - all roots, and A - child.
    #     The set of positive cases A can be contained in the set of positive cases B, C, D and another.
    #     Acyclic structure. Adjacent sets can intersect.
    #     """
    #     recalculated_nodes = set([])

    #     prob_A = 0
    #     if type(idx_A) == str:
    #         prob_A = dict_probs[idx_A]
    #     elif type(idx_A) == dict:
    #         prob_A = idx_A['val']       #? no need in this func
    #         idx_A = idx_A['idx']

    #     for node_idx, prob_A_div in idx_to_divided_idx_prob[idx_A].items(): #?????
    #         if node_idx != idx_A and node_idx in dict_probs.keys():
    #             prob_A = idx_to_divided_idx_prob[node_idx][idx_A]   # divided prob of child for specific root  #? how work for this func
    #             prob_B = dict_probs[node_idx]                       # one of roots
    #             prob_AxB = prob_A * (1 - STAT_ERR)
    #             prob_AIB = prob_AxB / prob_B
    #             prob_AxnotB = prob_A - prob_AxB
    #             prob_notB = (1 - prob_B) * (1 - STAT_ERR)
    #             prob_AInotB = prob_AxnotB / prob_notB
    #             prob_BIA = prob_AIB * prob_B / (prob_AIB * prob_B + prob_AInotB * (1 - prob_B))

    #             dict_probs[node_idx] = prob_BIA
    #             recalculated_nodes.add(node_idx)

    #     return recalculated_nodes

    # @staticmethod
    # def acyclic_all_probs_A(idx_A: str, dict_probs: dict, filter_idxs: set=None, STAT_ERR: float=0) -> set:
    #     """
    #     Recalculate all probabilities for dict of probs and True idx A. B - all nodes in filter_idxs, and A - some node.
    #     Acyclic structure. Adjacent sets can intersect.
    #     """
    #     recalculated_nodes = set([])

    #     prob_A = 0
    #     if type(idx_A) == str:
    #         prob_A = dict_probs[idx_A]
    #     elif type(idx_A) == dict:
    #         prob_A = idx_A['val']           #? no need in this func
    #         idx_A = idx_A['idx']

    #     for node_idx in filter_idxs:
    #         if node_idx != idx_A and node_idx in dict_probs.keys():
    #             prob_B = dict_probs[node_idx]
    #             dict_probs[node_idx] = prob_B * STAT_ERR # * prob_A
    #             recalculated_nodes.add(node_idx)

    #     return recalculated_nodes

    @staticmethod
    def acyclic_prob_AInotB(
        idx_B: str, dict_probs: dict, idx_to_divided_idx_prob: dict, STAT_ERR: float = 0
    ) -> set:
        """
        Recalculate probabilities P(A|not B) for dict of probs and False idx B. B - root, and A - all childs.
        The set of positive cases A can be contained in the set of positive cases B, C, D and another.
        Acyclic structure. Adjacent sets can intersect.
        P(A|YnotB) = P(A|Y) P(A|notB) / P(A)
        P(A|Y) = current Node prob
        P(A|notB) = P(A and notB) / P(notB)
        P(A and notB) = P(A) - P(A and B)
        P(A and B) = P(A)   - bc A contained in B
        P(notB) = (1 - P(B)) * (1 - ERR)
        P(A) = P(A) * (1 - ERR)
        """
        recalculated_nodes = set([])

        prob_B = 0
        if type(idx_B) == str:
            prob_B = dict_probs[idx_B]
        elif type(idx_B) == dict:
            prob_B = idx_B["val"]
            idx_B = idx_B["idx"]

        for node_idx, prob_A_div in idx_to_divided_idx_prob[idx_B].items():
            if (
                node_idx != idx_B and node_idx in dict_probs.keys()
            ):  # ? maybe better change all nodes, but in pprint print only non answered
                prob_AIY = dict_probs[node_idx]
                prob_A = idx_to_divided_idx_prob[node_idx]
                prob_AxB = prob_A_div * (1 - STAT_ERR)
                prob_AxnotB = prob_A_div - prob_AxB
                prob_notB = (1 - prob_B) * (1 - STAT_ERR)
                prob_AInotB = prob_AxnotB / prob_notB

                dict_probs[node_idx] = prob_AIY * prob_AInotB / prob_A
                recalculated_nodes.add(node_idx)

        return recalculated_nodes

    # @staticmethod
    # def acyclic_prob_BInotA(idx_A: str, dict_probs: dict, idx_to_divided_idx_prob: dict, back_struct_idxs: dict, filter_idxs: set=None, STAT_ERR: float=0) -> set:
    #     """
    #     Recalculate probabilities P(B|not A) for dict of probs and False idx A. B - all roots, and A - child.
    #     The set of positive cases A can be contained in the set of positive cases B, C, D and another.
    #     Acyclic structure. Adjacent sets can intersect.
    #     """
    #     recalculated_nodes = set([])

    #     prob_A = 0
    #     if type(idx_A) == str:
    #         prob_A = dict_probs[idx_A]
    #     elif type(idx_A) == dict:
    #         prob_A = idx_A['val']
    #         idx_A = idx_A['idx']

    #     for node_idx, lvl in back_struct_idxs.items():
    #         if node_idx != idx_A and node_idx in dict_probs.keys():     #? need ckeck for this func
    #             prob_B = dict_probs[node_idx]         # one of roots

    #             dict_probs[node_idx] = prob_B * (1 - prob_A * (1 - STAT_ERR) / prob_B)
    #             recalculated_nodes.add(node_idx)

    #     return recalculated_nodes

    # @staticmethod
    # def acyclic_all_probs_notA(idx_A: str, dict_probs: dict, dict, filter_idxs: set=None, STAT_ERR: float=0) -> set:
    #     """
    #     Recalculate all probabilities for dict of probs and False idx A. B - all nodes in filter_idxs, and A - some node.
    #     Acyclic structure. Adjacent sets can intersect.
    #     """
    #     recalculated_nodes = set([])

    #     prob_A = 0
    #     if type(idx_A) == str:
    #         prob_A = dict_probs[idx_A]
    #     elif type(idx_A) == dict:
    #         prob_A = idx_A['val']
    #         idx_A = idx_A['idx']

    #     for node_idx in filter_idxs:
    #         if node_idx != idx_A and node_idx in dict_probs.keys():
    #             prob_B = dict_probs[node_idx]
    #             dict_probs[node_idx] = prob_B / (1 - prob_A * (1 - STAT_ERR))
    #             recalculated_nodes.add(node_idx)

    #     return recalculated_nodes
