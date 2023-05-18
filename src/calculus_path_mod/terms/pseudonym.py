from calculus_path_mod.term_engine import Term
from calculus_path_mod.term_engine import Application as App
from calculus_path_mod.term_engine import Abstraction as Lambda
from calculus_path_mod.term_engine import Var, Atom


def multi_app_term(term_0: Term, term_1: Term, *terms: Term):
    """
    :param: term_0 - any type of Term
    :param: term_1 - any type of Term
    :param: terms  - any count of any type of Term

    :return: App(App(...(App(term_0, term_1), terms[0]), ...), terms[i])
            or simply application terms from left to right: (( ...(term_0 term_1) terms[0]) ... terms[i])
    """

    res_app_term = App(term_0, term_1)
    for term in terms:
        res_app_term = App(res_app_term, term)
    return res_app_term
