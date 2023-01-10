from random import random
from tqdm import tqdm
import numpy as np
from threading import Thread

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


def gen_lambda_terms(count=100, down_vertices_limit=50, up_vertices_limit=60,
                     gen_const=7_000, return_exact=True):
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
        count_terms=100, down_vertices_limit=50, up_vertices_limit=60,
        terms=None
):
    if terms is None:
        # because at least 30% terms will filter out so increase gen count
        terms = gen_lambda_terms(count=int(count_terms * 1.3), down_vertices_limit=down_vertices_limit,
                                 up_vertices_limit=up_vertices_limit)
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


class GenTermsThread(Thread):
    def __init__(self, count_terms=100, down_vertices_limit=50, up_vertices_limit=60,
                 random_average_count=20, thread_name=""):
        super().__init__()
        self.unfiltered_terms = gen_lambda_terms(count=count_terms, down_vertices_limit=down_vertices_limit,
                                                 up_vertices_limit=up_vertices_limit)
        self.gen_terms = []
        self.gen_stepsLO = []
        self.gen_stepsRI = []
        self.gen_stepsRand = []
        self.count_terms = count_terms
        self.down_vertices_limit = down_vertices_limit
        self.up_vertices_limit = up_vertices_limit
        self.random_average_count = random_average_count
        self.thread_name = thread_name
        self.mode = "all"

    def set_mode(self, mode):
        self.mode = mode

    def run(self):
        print(f"Running thread: {self.thread_name}")
        if self.mode in ["all", "gen"]:
            self.gen_terms, self.gen_stepsLO = \
                gen_filtered_lambda_terms(count_terms=self.count_terms,
                                          down_vertices_limit=self.down_vertices_limit,
                                          up_vertices_limit=self.up_vertices_limit,
                                          terms=self.unfiltered_terms)
            print(f"Thread {self.thread_name} is generated and filtered terms")

        if self.mode in ["all", "RI"]:
            print(f"Thread {self.thread_name} is doing RI norm")
            self.gen_stepsRI = [term.normalize(RightmostInnermostStrategy())[1] for term in tqdm(self.gen_terms)]
            print(f"Thread {self.thread_name} is DONE RI norm")

        if self.mode in ["all", "Rand"]:
            print(f"Thread {self.thread_name} is doing Random norm")
            self.gen_stepsRand = [
                sum([term.normalize(RandomStrategy())[1] for i in range(self.random_average_count)])
                / self.random_average_count
                for term in tqdm(self.gen_terms)
            ]
            print(f"Thread {self.thread_name} is DONE Random norm")
        print(f"Thread {self.thread_name} is DONE")

    def get_terms(self):
        return self.gen_terms

    def get_stepsLO(self):
        return self.gen_stepsLO

    def get_stepsRI(self):
        return self.gen_stepsRI

    def get_stepsRand(self):
        return self.gen_stepsRand


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
