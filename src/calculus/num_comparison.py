from calculus.pseudonym import *
from calculus import logic
from calculus import arithm_ops


def iszero_term():
    """Test whether a number is zero
    :return: ISZERO ≡ 𝜆 n.n(𝜆 x.FALSE) TRUE)"""

    x, n = Var(), Var()
    n_ = Atom(n)
    return Lambda(n, App(App(n_, Lambda(x, logic.false_term())), logic.true_term()))


def leq_term():
    """Less than or equal to
    :return: LEQ ≡ 𝜆 n.𝜆 m.ISZERO (SUBTRACT n m)"""

    n, m = Var(), Var()
    n_, m_ = Atom(n), Atom(m)
    return Lambda(
        n, Lambda(m, App(iszero_term(), multi_app_term(arithm_ops.subtract_term(), n_, m_)))
    )


def eq_term():
    """ Equal to
    :return: EQ≡𝜆 n.𝜆 m.AND (LE [n] [m]) (LE [m] [n])"""

    n, m = Var(), Var()
    n_, m_ = Atom(n), Atom(m)
    return Lambda(
        n,
        Lambda(
            m,
            multi_app_term(
                logic.and_term(),
                multi_app_term(leq_term(), n_, m_),
                multi_app_term(leq_term(), m_, n_),
            ),
        ),
    )


def lt_term():
    """Less than
    :return: LT := λab. NOT (LEQ b a)"""

    a, b = Var(), Var()
    a_, b_ = Atom(a), Atom(b)
    return Lambda(a, Lambda(b, App(
        logic.not_term(),
        multi_app_term(leq_term(), b_, a_)
    )))


def neq_term():
    """Not equal to
    :return: NEQ := λab. OR (NOT (LEQ a b)) (NOT (LEQ b a))"""

    a, b = Var(), Var()
    a_, b_ = Atom(a), Atom(b)
    return Lambda(a, Lambda(b, multi_app_term(
        logic.or_term(),
        App(logic.not_term(), multi_app_term(leq_term(), a_, b_)),
        App(logic.not_term(), multi_app_term(leq_term(), b_, a_)),
    )))


def geq_term():
    """Greater than or equal to
    :return: GEQ := λab. LEQ b a"""

    a, b = Var(), Var()
    a_, b_ = Atom(a), Atom(b)
    return Lambda(a, Lambda(b, multi_app_term(
        leq_term(), b_, a_
    )))


def gt_term():
    """Greater than:
    :return: GT := λab. NOT (LEQ a b)"""
    a, b = Var(), Var()
    a_, b_ = Atom(a), Atom(b)
    return Lambda(a, Lambda(b, App(
        logic.not_term(),
        multi_app_term(leq_term(), a_, b_)
    )))
