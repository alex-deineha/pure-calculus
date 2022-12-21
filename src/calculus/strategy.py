import random
from abc import ABC, abstractmethod
from typing import List

from calculus.term import Term


class OneStepStrategy(ABC):

    @abstractmethod
    def redexIndex(self, term: Term, initIndex=0) -> int:
        """
      :return: index of the vertex of a subterm that has an outer redex.
              The index of a vertex is the index of this vertex in the topological sort of the tree vertices.
              Indexing starts at 1.
      """


class LeftmostOutermostStrategy(OneStepStrategy):

    def redexIndex(self, term: Term, initIndex=0) -> int:
        if term.isAtom or len(term.redexes) == 0:
            ValueError('the term does not contain a redex')
        elif term.isApplication:
            if term.isBetaRedex:
                return initIndex + 1
            elif len(term._sub.redexes) != 0:
                return self.redexIndex(term._sub, initIndex + 1)
            else:
                return self.redexIndex(term._obj, initIndex + term._sub.verticesNumber + 1)
        else:  # self is Abstraction
            return self.redexIndex(term._body, initIndex + 1)


class LeftmostInnermostStrategy(OneStepStrategy):

    def redexIndex(self, term: Term, initIndex=0) -> int:
        if term.isAtom or len(term.redexes) == 0:
            ValueError('the term does not contain a redex')
        elif term.isApplication:
            if len(term._sub.redexes) != 0:
                return self.redexIndex(term._sub, initIndex + 1)
            elif len(term._obj.redexes) != 0:
                return self.redexIndex(term._obj, initIndex + term._sub.verticesNumber + 1)
            else:
                return initIndex + 1
        else:  # self is Abstraction
            return self.redexIndex(term._body, initIndex + 1)


class RightmostInnermostStrategy(OneStepStrategy):

    def redexIndex(self, term: Term, initIndex=0) -> int:
        if term.isAtom or len(term.redexes) == 0:
            ValueError('the term does not contain a redex')
        elif term.isApplication:
            if len(term._obj.redexes) != 0:
                return self.redexIndex(term._obj, initIndex + term._sub.verticesNumber + 1)
            elif len(term._sub.redexes) != 0:
                return self.redexIndex(term._sub, initIndex + 1)
            else:
                return initIndex + 1
        else:  # self is Abstraction
            return self.redexIndex(term._body, initIndex + 1)


class RightmostOutermostStrategy(OneStepStrategy):

    def redexIndex(self, term: Term, initIndex=0) -> int:
        if term.isAtom or len(term.redexes) == 0:
            ValueError('the term does not contain a redex')
        elif term.isApplication:
            if term.isBetaRedex:
                return initIndex + 1
            elif len(term._obj.redexes) != 0:
                return self.redexIndex(term._obj, initIndex + term._sub.verticesNumber + 1)
            else:
                return self.redexIndex(term._sub, initIndex + 1)
        else:  # self is Abstraction
            return self.redexIndex(term._body, initIndex + 1)


class RandomStrategy(OneStepStrategy):

    def redexIndex(self, term: Term, initIndex=0) -> int:
        redexes = term.redexes
        if term.isAtom or len(redexes) == 0:
            ValueError('the term does not contain a redex')
        elif term.isApplication:
            index = random.randint(0, len(redexes) - 1)
            if term.isBetaRedex and index == 0:
                return initIndex + 1
            elif len(term._sub.redexes) >= index and len(term._sub.redexes) != 0:
                return self.redexIndex(term._sub, initIndex + 1)
            else:
                return self.redexIndex(term._obj, initIndex + term._sub.verticesNumber + 1)
        else:  # self is Abstraction
            return self.redexIndex(term._body, initIndex + 1)


class MixedStrategy(OneStepStrategy):

    def __init__(self, strategies: List[OneStepStrategy], probability_vector: list):
        self.strategies = strategies
        self.probability_vector = probability_vector

    def redexIndex(self, term: Term) -> int:
        p = random.random()
        index = 0
        index_prob = self.probability_vector[0]
        while (p > index_prob):
            index += 1
            index_prob += self.probability_vector[index]

        return self.strategies[index].redexIndex(term)
