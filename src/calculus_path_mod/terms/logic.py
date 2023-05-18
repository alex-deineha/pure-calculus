from calculus_path_mod.terms.pseudonym import *
from calculus_path_mod.terms import combinators


def true_term():
    """:return: TRUE := λxy. x ≡ K"""
    return combinators.k_term()


def false_term():
    """:return: FALSE := λxy. y ≡ 0 ≡ K*"""
    return combinators.k_star_term()


def ite_term():
    """
    ITE = IF_THEN_ELSE
    :return: ITE≡𝜆 c.𝜆 x.𝜆 y.cxy
    """
    x, y, c = Var(), Var(), Var()
    x_, y_, c_ = Atom(x), Atom(y), Atom(c)
    return Lambda(c, Lambda(x, Lambda(y, multi_app_term(c_, x_, y_))))


def not_term():
    """:return: NOT ≡ 𝜆 a. ITE a FALSE TRUE"""
    a = Var()
    a_ = Atom(a)
    return Lambda(a, multi_app_term(ite_term(), a_, false_term(), true_term()))


def and_term():
    """:return: AND ≡ 𝜆 a.𝜆 b.ITE a b a"""
    a, b = Var(), Var()
    a_, b_ = Atom(a), Atom(b)
    return Lambda(a, Lambda(b, multi_app_term(ite_term(), a_, b_, a_)))


def or_term():
    """:return: OR ≡ 𝜆 a.𝜆 b.ITE a a b"""
    a, b = Var(), Var()
    a_, b_ = Atom(a), Atom(b)
    return Lambda(a, Lambda(b, multi_app_term(ite_term(), a_, a_, b_)))
