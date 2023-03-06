from random import random
from typing import List
import numpy as np

from calculus_path_mod.reduction_strategy import *
from calculus_path_mod.term_engine import Atom, Var, Abstraction, Application


def genTerm(p: float, uplimit: int, vars: List[Var] = [], trigger_by_application=False):
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
        new_vars = vars + [head]
        body = genTerm(p, uplimit - 1, new_vars)
        return Abstraction(head, body) if body else None
    else:
        sub = genTerm(p, uplimit - 1, vars, trigger_by_application=True)
        obj = genTerm(p, uplimit - 1, vars)
        if sub and obj and sub.vertices_number + obj.vertices_number <= uplimit:
            return Application(sub, obj)
        else:
            return None


def gen_lambda_terms(
        count=100,
        down_vertices_limit=50,
        up_vertices_limit=60,
        gen_const=7_000,
        return_exact=True):
    def filter_terms(term):
        return term and down_vertices_limit < term.vertices_number < up_vertices_limit

    def flatten(t):
        return [item for sublist in t for item in sublist]

    terms = []
    while True:
        terms += flatten(
            [
                list(
                    filter(
                        filter_terms,
                        [genTerm(p, up_vertices_limit) for i in range(gen_const)],
                    )
                )
                for p in np.arange(0.49, 0.51, 0.02)
            ]
        )
        if len(terms) > count:
            break

    if return_exact:
        return terms[:count]
    return terms


def gen_filtered_lambda_terms(
        count_terms=100,
        down_vertices_limit=50,
        up_vertices_limit=60,
        filtering_strategy=LOStrategy()):
    terms = []
    steps_by_strategy = []
    while len(terms) < count_terms:
        unfiltered_terms = gen_lambda_terms(
            count=1,
            up_vertices_limit=up_vertices_limit,
            down_vertices_limit=down_vertices_limit,
            gen_const=40,
            return_exact=False,
        )
        for term in unfiltered_terms:
            _, steps = term.normalize(filtering_strategy)
            if steps != float("inf"):
                terms.append(term)
                steps_by_strategy.append(steps)

    return terms, steps_by_strategy
