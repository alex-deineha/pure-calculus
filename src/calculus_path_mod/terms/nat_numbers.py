from calculus_path_mod.terms.pseudonym import *


def num_zero_term():
    """:return: 0 := λsz. z"""

    s, z = Var(), Var()
    z_ = Atom(z)
    return Lambda(s, Lambda(z, z_))


def num_term(n: int):
    """
    Represent any natural number as lambda-term
    For 0 returns num_zero_term()
    For n < 0 returns num_zero_term()
    :return: 1 := λsz. s z
             2 := λsz. s (s z)
             3 := λsz. s (s (s z))
    """

    if n <= 0:
        return num_zero_term()
    s, z = Var(), Var()
    s_, z_ = Atom(s), Atom(z)
    core_term = App(s_, z_)
    for _ in range(n - 1):
        core_term = App(s_, core_term)
    return Lambda(s, Lambda(z, core_term))