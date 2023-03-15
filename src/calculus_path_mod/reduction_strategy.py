from abc import ABC, abstractmethod
from calculus_path_mod.term_engine import Term
import random
import numpy as np


class OneStepStrategy(ABC):
    @abstractmethod
    def redex_index(self, term: Term, init_index=0) -> int:
        """
        :return: index of the vertex of a subterm that has an outer redex.
                The index of a vertex is the index of this vertex in the topological sort of the tree vertices.
                Indexing starts at 1.
        """


class LOStrategy(OneStepStrategy):
    def redex_index(self, term: Term, init_index=0) -> int:
        # The main idea check current vertex
        # if it's a redex return its coordinate in the tree

        if (term.kind == "atom") or (len(term.redexes) == 0):
            raise ValueError("The term doesn't contain a redex")
        if term.kind == "application":
            if term.is_beta_redex:
                return init_index + 1
            if len(term._data[0].redexes) != 0:
                return self.redex_index(term._data[0], init_index + 1)
            return self.redex_index(term._data[1],
                                    init_index + term._data[0].vertices_number + 1)
        # self is Abstraction:
        return self.redex_index(term._data[1], init_index + 1)


class RIStrategy(OneStepStrategy):
    def redex_index(self, term: Term, init_index=0) -> int:
        # The main idea go in the right vertex (object) of the application
        # until there is no other way to go right, go the left vertex,
        # until there is no way and define current app as a redex.

        if (term.kind == "atom") or (len(term.redexes) == 0):
            raise ValueError("The term doesn't contain a redex")
        if term.kind == "application":
            if len(term._data[1].redexes) != 0:
                return self.redex_index(term._data[1],
                                        init_index + term._data[0].vertices_number + 1)
            if len(term._data[0].redexes) != 0:
                return self.redex_index(term._data[0], init_index + 1)
            return init_index + 1
        # self is Abstraction:
        return self.redex_index(term._data[1], init_index + 1)


class RandomLeftOuterStrategy(OneStepStrategy):
    def redex_index(self, term: Term, init_index=0) -> int:
        # The main idea check current vertex
        # if it's a redex return its coordinate in the tree with some probability
        # and with some probability go to the left vertex of right.

        count_redexes = len(term.redexes)
        if term.kind == "atom" or count_redexes == 0:
            raise ValueError("The term doesn't contain a redex")
        elif term.kind == "application":
            index = random.randint(0, count_redexes - 1)
            if term.is_beta_redex and index <= int(count_redexes / 2):
                return init_index + 1
            elif len(term._data[0].redexes) >= index and len(term._data[0].redexes) != 0:
                return self.redex_index(term._data[0], init_index + 1)
            else:
                return self.redex_index(term._data[1], init_index + term._data[0].vertices_number + 1)
        else:
            return self.redex_index(term._data[1], init_index + 1)


class RandomRightInnerStrategy(OneStepStrategy):
    def redex_index(self, term: Term, init_index=0) -> int:
        # The main idea go in the right vertex (object) of the application
        # until there is no other way to go right, go the left vertex,
        # until there is no way and define current app as a redex.

        count_redexes = len(term.redexes)
        if term.kind == "atom" or count_redexes == 0:
            raise ValueError("The term doesn't contain a redex")
        elif term.kind == "application":
            index = random.randint(0, count_redexes - 1)
            if len(term._data[1].redexes) != 0 and index <= int(count_redexes / 2):
                return self.redex_index(term._data[1],
                                        init_index + term._data[0].vertices_number + 1)
            if len(term._data[0].redexes) > 0:
                return self.redex_index(term._data[0], init_index + 1)
            return init_index + 1
        else:
            return self.redex_index(term._data[1], init_index + 1)


class RandomOuterStrategy(OneStepStrategy):
    def __init__(self, prob_norm="softmax"):
        """
        :param prob_norm: str of 'softmax', or 'sum', a 'pow_val'
                          (where val - num value of power, example 'pow_2'),
                          which define calculation of probabilities
        """
        self.prob_norm = prob_norm

    def _get_redexes_indexes(self, term: Term, init_index: int = 0, path_depth: int = 0) -> dict:
        """
        Using Left Outer strategy find all redexes, it's coordinates in the term tree,
        and the depth value for each redex.

        :param term: some Term is a current vertex
        :param init_index: some index is a coordinate of the current vertex in the term tree
        :param path_depth: depth in the term tree (simply recursion layer).
        :return: dict where a key is a index (coordinate of redex in the term tree),
                 and var is a depth from the tree root
        """

        if (term.kind == "atom") or (len(term.redexes) == 0):
            raise ValueError("The term doesn't contain a redex")
        if term.kind == "application":
            result = dict()
            if term.is_beta_redex:
                result = {**result,
                          **{init_index + 1: path_depth + 1}}
            if len(term._data[0].redexes) != 0:
                result = {**result,
                          **self._get_redexes_indexes(term._data[0], init_index + 1, path_depth + 1)}
            if len(term._data[1].redexes) != 0:
                result = {**result,
                          **self._get_redexes_indexes(term._data[1],
                                                      init_index + term._data[0].vertices_number + 1,
                                                      path_depth + 1)}
            return result
            # self is Abstraction:
        return self._get_redexes_indexes(term._data[1], init_index + 1, path_depth + 1)

    def redex_index(self, term: Term, init_index=0) -> int:
        # Using the redex coordinates and the depth values generate probabilities
        # of choosing for each redex, and in the end choose and return redex coordinate.

        dict_redexes_indexes = self._get_redexes_indexes(term)
        list_of_indexes = list(dict_redexes_indexes.keys())
        list_of_prob = np.array(list(dict_redexes_indexes.values()), dtype="float64")

        max_depth = 1. + np.amax(list_of_prob)
        list_of_prob *= -1.
        list_of_prob += max_depth

        if self.prob_norm == "softmax":
            list_of_prob = np.exp(list_of_prob) / np.sum(np.exp(list_of_prob))
        elif self.prob_norm == "sum":
            list_of_prob /= np.sum(list_of_prob)
        elif "pow" in self.prob_norm:
            pow_val = float(self.prob_norm.split("_")[1])
            list_of_prob = np.power(list_of_prob, pow_val)
            list_of_prob /= np.sum(list_of_prob)
        else:
            raise ValueError("Inappropriate value of prob_norm")

        return np.random.choice(list_of_indexes, p=list_of_prob)


class RandomInnerStrategy(RandomOuterStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def redex_index(self, term: Term, init_index=0) -> int:
        # Using the redex coordinates and the depth values generate probabilities
        # of choosing for each redex, and in the end choose and return redex coordinate.

        dict_redexes_indexes = self._get_redexes_indexes(term)
        list_of_indexes = list(dict_redexes_indexes.keys())
        list_of_prob = np.array(list(dict_redexes_indexes.values()), dtype="float64")

        if self.prob_norm == "softmax":
            list_of_prob = np.exp(list_of_prob) / np.sum(np.exp(list_of_prob))
        elif self.prob_norm == "sum":
            list_of_prob /= np.sum(list_of_prob)
        elif "pow" in self.prob_norm:
            pow_val = float(self.prob_norm.split("_")[1])
            list_of_prob = np.power(list_of_prob, pow_val)
            list_of_prob /= np.sum(list_of_prob)
        else:
            raise ValueError("Inappropriate value of prob_norm")

        return np.random.choice(list_of_indexes, p=list_of_prob)
