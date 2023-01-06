from random import random
from tqdm import tqdm
import numpy as np

from calculus.strategy import *
from calculus.term import Atom, Var, Abstraction, Application


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
        if sub and obj and sub.verticesNumber + obj.verticesNumber <= uplimit:
            return Application(sub, obj)
        else:
            return None


def gen_lambda_terms(count=100, down_vertices_limit=50, up_vertices_limit=60):
    def filter_terms(term):
        return term and down_vertices_limit < term.verticesNumber < up_vertices_limit

    def flatten(t):
        return [item for sublist in t for item in sublist]

    terms = []
    while True:
        terms += flatten(
            [
                list(
                    filter(
                        filter_terms,
                        [genTerm(p, up_vertices_limit) for i in range(7000)],
                    )
                )
                for p in np.arange(0.49, 0.51, 0.02)
            ]
        )
        if len(terms) > count:
            break

    terms = terms[:count]
    return terms


def gen_filtered_lambda_terms(
    count_terms=100, down_vertices_limit=50, up_vertices_limit=60
):
    def filter_terms(term):
        return term and down_vertices_limit < term.verticesNumber < up_vertices_limit

    def flatten(t):
        return [item for sublist in t for item in sublist]

    terms = []
    while True:
        terms += flatten(
            [
                list(
                    filter(
                        filter_terms,
                        [genTerm(p, up_vertices_limit) for i in range(7000)],
                    )
                )
                for p in np.arange(0.49, 0.51, 0.02)
            ]
        )
        print("Generated terms:", len(terms))
        if len(terms) > count_terms:
            break
    print("LO strategy applying")
    stepsLO = list(
        map(lambda term: term.normalize(LeftmostOutermostStrategy())[1], terms)
    )
    print("Remove unormalized terms")
    terms_with_normal_form = []
    stepsLO_temp = []
    for i, term in enumerate(terms):
        if stepsLO[i] != float("inf"):
            terms_with_normal_form.append(term)
            stepsLO_temp.append(stepsLO[i])
    print("Left", count_terms, "normalizeble terms")
    terms = terms_with_normal_form[:count_terms]
    stepsLO = stepsLO_temp[:count_terms]

    return terms, stepsLO


if __name__ == "__main__":
    terms = gen_lambda_terms(count=100)
    print(len(terms))
    print(terms[0])

    from strategy import LeftmostOutermostStrategy

    strat = LeftmostOutermostStrategy()
    i = 0

    terms[0].restart_normalizetion()
    while terms[0].normalize_step(strat):
        print("step", terms[0].normalization_term)
