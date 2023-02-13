from calculus.pseudonym import *
from calculus import pairs
from calculus import nat_numbers
from calculus import combinators
from calculus import num_comparison


def succ_term():
    """:return: SUCC ≡ 𝜆 n.𝜆 x.𝜆 y.x(nxy)"""

    x, y, n = Var(), Var(), Var()
    x_, y_, n_ = Atom(x), Atom(y), Atom(n)
    return Lambda(n, Lambda(x, Lambda(y, App(x_, multi_app_term(n_, x_, y_)))))


def sinc_term():
    """:return: SINC ≡ 𝜆 p. PAIR (SECOND p) (SUCC (SECOND p))"""

    p = Var()
    p_ = Atom(p)
    return Lambda(
        p,
        multi_app_term(
            pairs.pair_term(),
            App(pairs.second_term(), p_),
            App(succ_term(), App(pairs.second_term(), p_)),
        ),
    )


def pred_term():
    """:return: PRED≡𝜆 n. FIRST ([n] SINC (PAIR [0] [0]))"""

    n = Var()
    n_ = Atom(n)
    return Lambda(
        n,
        App(
            pairs.first_term(),
            multi_app_term(
                n_, sinc_term(),
                multi_app_term(pairs.pair_term(), nat_numbers.num_term(0), nat_numbers.num_term(0))
            ),
        ),
    )


def subtract_term():
    """:return: SUBTRACT ≡ 𝜆 n.𝜆 m.m PRED n"""

    n, m = Var(), Var()
    n_, m_ = Atom(n), Atom(m)
    return Lambda(n, Lambda(m, multi_app_term(m_, pred_term(), n_)))


def div_term():
    """Division — DIV a b evaluates to a pair of two numbers, a idiv b and a mod b
    :return: DIV := Y (λgqab. LT a b (PAIR q a)
                        (g (SUCC q) (SUB a b) b) ) 0"""

    g, q, a, b = Var(), Var(), Var(), Var()
    g_, q_, a_, b_ = Atom(g), Atom(q), Atom(a), Atom(b)

    body_term = multi_app_term(num_comparison.lt_term(), a_, b_,
                               multi_app_term(pairs.pair_term(), q_, a_),
                               multi_app_term(g_, App(succ_term(), q_), multi_app_term(subtract_term(), a_, b_), b_))
    inner_lambda_term = Lambda(g, Lambda(q, Lambda(a, Lambda(b, body_term))))

    return multi_app_term(combinators.y_term(), inner_lambda_term, nat_numbers.num_term(0))


def mod_term():
    """Modulus
    :return: MOD := λab. CDR (DIV a b)"""

    a, b = Var(), Var()
    a_, b_ = Atom(a), Atom(b)
    return Lambda(a, Lambda(b, App(pairs.second_term(), multi_app_term(div_term(), a_, b_))))


def idiv_term():
    """Integer division
    :return: IDIV := λab. CAR (DIV a b)"""

    a, b = Var(), Var()
    a_, b_ = Atom(a), Atom(b)
    return Lambda(a, Lambda(b, App(pairs.first_term(), multi_app_term(div_term(), a_, b_))))


def plus_term():
    """Addition
    :return: PLUS := λmnfx. n f (m f x) ≡ λmn. n SUCC m"""

    n, m = Var(), Var()
    n_, m_ = Atom(n), Atom(m)
    return Lambda(m, Lambda(n, multi_app_term(n_, succ_term(), m_)))


def mult_term():
    """Multiplication:
    :return: MULT := λmnf. m (n f) ≡ λmn. m (PLUS n) 0 ≡ B"""

    n, m = Var(), Var()
    n_, m_ = Atom(n), Atom(m)
    return Lambda(m, Lambda(n, multi_app_term(m_, App(plus_term(), n_), nat_numbers.num_term(0))))
