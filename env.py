# -*- coding: utf-8 -*-
"""PureCalculus.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RM87rnBWlmBdw4x5HZZqLTo3Ua_3IVIw

# **Pure $\lambda$-Calculus**

[The deatailed script](https://www.mathcha.io/editor/Pvvz5UZ1t7ktL6sZJYp19sZnX9vVserJMEKhJvvMx7)

## **Variables**

The code below models variables.

Using the `natgen()` generator in this code ensures that a fresh variable is returned in response to each constructor call.
"""


def natgen():
    n = 0
    while True:
        yield n
        n += 1


class Var:
    __nats = natgen()

    def __init__(self):
        self._idx = next(Var.__nats)

    def __hash__(self):
        return self._idx.__hash__()

    def __str__(self):
        return "v[" + str(self._idx) + "]"

    def __eq__(self, other):
        return self._idx == other._idx


"""## **Terms**

"""


class Term:

    @property
    def isAtom(self):
        """checks whether the term is an atom"""
        return isinstance(self, Atom)

    @property
    def isApplication(self):
        """checks whether the term is an application"""
        return isinstance(self, Application)

    @property
    def isAbstraction(self):
        """checks whether the term is an abstraction"""
        return isinstance(self, Abstraction)

    def __str__(self):
        if self.isAtom:
            return str(self._var)
        if self.isApplication:
            return "(" + str(self._sub) + " " + str(self._obj) + ")"
        # self is Abbstraction
        return "(fun " + str(self._head) + " => " + str(self._body) + ")"

    def __eq__(self, other):
        if self.isAtom and other.isAtom:
            return self._var == other._var
        if isinstance(self, Application) and isinstance(other, Application):
            return self._sub == other._sub and self._obj == other._obj
        if isinstance(self, Abstraction) and isinstance(other, Abstraction):
            return self._head == other._head and self._body == other._body

    @property
    def isBetaRedex(self):
        """checks whether the term is a beta-redex"""
        return self.isApplication and self._sub.isAbstraction

    @property
    def redexes(self):
        """determiness all beta-redexes in the term"""
        if self.isAtom:
            return []
        if self.isAbstraction:
            return self._body.redexes
        # self is Application
        temp = [self] if self.isBetaRedex else []
        temp += (self._sub.redexes + self._obj.redexes)
        return temp

    @property
    def _vars(self):
        """
        returns
        -------
            the dictionary stuctured as follows
                dict[Var, dict[['free' | 'bound'], int]]
            Here, keys of the external dictionary are the variables that
            are occurred in 'self', and values of the internal dictionaries
            relate respectively to the numbers of free and bound occurrences
            of the variables.
        """
        if self.isAtom:
            return {self._var: {'free': 1, 'bound': 0}}
        if self.isApplication:
            vars, auxvars = dict(self._sub._vars), self._obj._vars
            for var in auxvars:
                try:
                    for key in {'free', 'bound'}:
                        vars[var][key] += self._obj._vars[var][key]
                except KeyError:
                    vars[var] = dict(self._obj._vars[var])
            return vars
        # self is Abstraction
        vars = dict(self._body._vars)
        try:
            vars[self._head]['bound'] += vars[self._head]['free']
            vars[self._head]['free'] = 0
        except KeyError:
            pass
        return vars

    @property
    def verticesNumber(self):
        """return the number of nodes in the tree representing the lambda term"""
        if self.isAtom:
            return 1
        elif self.isApplication:
            return 1 + self._sub.verticesNumber + self._obj.verticesNumber
        else:  # self is Abstraction
            return 1 + self._body.verticesNumber

    def normalize(self, strategy):
        """
        :param strategy: OneStepStrategy
        :return tuple of the normal form of the term and number of steps of betta reduction
        """
        term = self._updateBoundVariables()
        count = 0
        while term.redexes != []:
            term = term._betaConversion(strategy)
            count += 1
            if term.verticesNumber > 7000 or count > 400:
                return (self, float('inf'))
        return (term, count)

    def _betaConversion(self, strategy):
        """
        :param strategy: OneStepStrategy
        :return term with redex eliminated using the given strategy
        """
        index = strategy.redexIndex(self)
        subterm = self.subterm(index)
        reducedTerm = subterm._removeOuterRedex()
        return self.setSubterm(index, reducedTerm)

    def _betaConversion_index(self, index):
        """
        :param strategy: OneStepStrategy
        :return term with redex eliminated using the given strategy
        """
        subterm = self.subterm(index)
        reducedTerm = subterm._removeOuterRedex()
        return self.setSubterm(index, reducedTerm)

    def subterm(self, index: int):
        """
        By representing the term as a tree, a subtree is returned, which is also a lambda term.
        The vertex of this subtree has a given index in the topological sorting of the vertices of the original term.
        :param index - subterm index
        :return: subterm: Term
        """
        if index == 1:
            return self

        if self.isAtom:
            ValueError('index value is incorrect')
        elif self.isApplication:
            if self._sub.verticesNumber + 1 >= index:
                return self._sub.subterm(index - 1)
            else:
                return self._obj.subterm(index - self._sub.verticesNumber - 1)
        else:  # self is Abstraction
            return self._body.subterm(index - 1)

    def setSubterm(self, index: int, term):
        """
        By representing the term as a tree, a subtree is set, which is also a lambda term.
        The vertex of this subtree has a given index in the topological sorting of the vertices of the original term.
        :param index - subterm index
        :param term - λ-term to which the subterm will be replaced
        :return: updated λ-term
        """
        if index == 1:
            return term

        if self.isAtom:
            ValueError('index value is incorrect')
        elif self.isApplication:
            if self._sub.verticesNumber + 1 >= index:
                return Application(self._sub.setSubterm(index - 1, term), self._obj)
            else:
                return Application(self._sub, self._obj.setSubterm(index - self._sub.verticesNumber - 1, term))
        else:  # self is Abstraction
            return Abstraction(self._head, self._body.setSubterm(index - 1, term))

    def _updateBoundVariables(self):
        """return λ-term with updated bound variables"""
        if self.isAtom:
            return self
        elif self.isApplication:
            return Application(self._sub._updateBoundVariables(), self._obj._updateBoundVariables())
        else:  # self is Abstraction
            newVar = Var()
            return Abstraction(newVar, self._body._replaceVariable(self._head, Atom(newVar))._updateBoundVariables())

    def _removeOuterRedex(self):
        """apply the betta conversion to the lambda term, removing the outer betta redex"""
        if self.isBetaRedex:
            head = self._sub._head
            body = self._sub._body
            return body._replaceVariable(head, self._obj)
        else:
            return self

    def _replaceVariable(self, var: Var, term):
        """return λ-term with replaced variable"""
        if self.isAtom:
            return term if self._var == var else self
        elif self.isApplication:
            return Application(self._sub._replaceVariable(var, term), self._obj._replaceVariable(var, term))
        else:  # self is Abstraction
            return Abstraction(self._head, self._body._replaceVariable(var, term))


class Atom(Term):
    def __init__(self, x: Var):
        if isinstance(x, Var):
            self._var = x
        else:
            raise TypeError("a variable is waiting")


class Application(Term):
    def __init__(self, X: Term, Y: Term):
        if isinstance(X, Term) and isinstance(Y, Term):
            self._sub = X
            self._obj = Y
        else:
            raise TypeError("a term is waiting")


class Abstraction(Term):
    def __init__(self, x: Var, X: Term):
        if isinstance(x, Var):
            if isinstance(X, Term):
                self._head = x
                self._body = X
            else:
                raise TypeError("a term is waiting")
        else:
            raise TypeError("a variable is waiting")


"""## Strategy

"""

import random
from abc import ABC, abstractmethod


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


"""## Generating lambda terms"""

import random
from typing import List
import sys

sys.setrecursionlimit(40000)


def genTerm(p: float, uplimit: int, vars: List[Var] = []):
    if uplimit < 1:
        return None

    pVar = (1 - p * p) / 2
    pAbs = pVar + p * p

    rand = random.random()

    if rand < pVar and len(vars) > 0:
        index = random.randint(0, len(vars) - 1)
        return Atom(vars[index])
    elif rand < pAbs:
        head = Var()
        body = genTerm(p, uplimit - 1, vars + [head])
        return Abstraction(head, body) if body else None
    else:
        sub = genTerm(p, uplimit - 1, vars)
        obj = genTerm(p, uplimit - 1, vars)
        if sub and obj and sub.verticesNumber + obj.verticesNumber <= uplimit:
            return Application(sub, obj)
        else:
            return None


import numpy as np

UPLIMIT = 80
DOWNLIMIT = 60

RANDOM_COUNT = 100


def filterTerms(term):
    return term and DOWNLIMIT < term.verticesNumber < UPLIMIT


def flatten(t):
    return [item for sublist in t for item in sublist]


import gym
from gym import spaces
import numpy as np
import random
from typing import List
import sys


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.terms = self.get_new_terms()
        self.term = None
        self.number_of_term = None
        self.action_space = None
        # Example for using image as input:
        self.observation_space = None
        self.number_of_term = -1
        self.reset()

    def step(self, index):
        self.term = self.term._betaConversion_index(index)
        obs = self.term
        reward = -1
        done = self.term.redexes == []
        return obs, reward, done, {}

    def reset(self):
        self.number_of_term += 1
        self.term = self.terms[self.number_of_term]._updateBoundVariables()
        return self.term

    def render(self, mode='human', close=False):
        print(self.term)

    def get_new_terms(self):
        return flatten(
            [list(filter(filterTerms, [genTerm(p, UPLIMIT) for i in range(800)])) for p in np.arange(0.37, 0.44, 0.01)])


if __name__ == '__main__':
    env = CustomEnv()

    # strategy = LeftmostOutermostStrategy()
    strategy = RightmostInnermostStrategy()
    # strategy =  RandomStrategy()

    obs = env.reset()
    done = False
    for i in range(2000):

        action = strategy.redexIndex(obs)
        print(action)
        if done or not action:
            env.reset()
        obs, rewards, done, info = env.step(action)

        env.render()