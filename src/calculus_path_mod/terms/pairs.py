from calculus_path_mod.terms.pseudonym import *
from calculus_path_mod.terms import combinators


def pair_term():
    """:return: PAIR ≡ 𝜆 x.𝜆 y.𝜆 p.pxy"""

    x, y, p = Var(), Var(), Var()
    x_, y_, p_ = Atom(x), Atom(y), Atom(p)
    return Lambda(x, Lambda(y, Lambda(p, multi_app_term(p_, x_, y_))))


def first_term():
    """:return: FIRST ≡ 𝜆 p. p K"""

    p = Var()
    p_ = Atom(p)
    return Lambda(p, App(p_, combinators.k_term()))


def second_term():
    """:return: SECOND ≡ 𝜆 p. p K\index{*}"""

    p = Var()
    p_ = Atom(p)
    return Lambda(p, App(p_, combinators.k_star_term()))
