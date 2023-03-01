from abc import ABC, abstractmethod
from calculus_path_mod.term_engine import Term
import random


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


class RandomOuterStrategy(OneStepStrategy):
    def redex_index(self, term: Term, init_index=0) -> int:
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


class RandomInnerStrategy(OneStepStrategy):
    def redex_index(self, term: Term, init_index=0) -> int:
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
