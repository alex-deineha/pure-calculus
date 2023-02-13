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


def gen_lambda_terms(
    count=100,
    down_vertices_limit=50,
    up_vertices_limit=60,
    gen_const=7_000,
    return_exact=True,
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
    count_terms=100, down_vertices_limit=50, up_vertices_limit=60, terms=None
):
    if terms is None:
        # because at least 30% terms will filter out so increase gen count
        terms = gen_lambda_terms(
            count=int(count_terms * 1.3),
            down_vertices_limit=down_vertices_limit,
            up_vertices_limit=up_vertices_limit,
        )
        print("Generated terms:", len(terms))

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
    print(f"Gen terms with normal form: {len(terms_with_normal_form)}")
    terms = terms_with_normal_form[:count_terms]
    stepsLO = stepsLO_temp[:count_terms]

    return terms, stepsLO


def gen_filtered_lambda_terms_v2(
    count_terms=100,
    down_vertices_limit=50,
    up_vertices_limit=60,
    filtering_strategy=LeftmostOutermostStrategy(),
    update_bound_vars=True,
):
    terms = []
    stepsLO = []
    while len(terms) < count_terms:
        unfiltered_terms = gen_lambda_terms(
            count=1,
            up_vertices_limit=up_vertices_limit,
            down_vertices_limit=down_vertices_limit,
            gen_const=40,
            return_exact=False,
        )
        for term in unfiltered_terms:
            _, steps = term.normalize(filtering_strategy, update_bound_vars)
            if steps != float("inf"):
                terms.append(term)
                stepsLO.append(steps)

    return terms, stepsLO


def test_run():
    terms = gen_lambda_terms(count=100)
    print(len(terms))
    print(terms[0])

    from strategy import LeftmostOutermostStrategy

    strat = LeftmostOutermostStrategy()
    i = 0

    terms[0].restart_normalizetion()
    while terms[0].normalize_step(strat):
        print("step", terms[0].normalization_term)


if __name__ == "__main__":
    test_run()
