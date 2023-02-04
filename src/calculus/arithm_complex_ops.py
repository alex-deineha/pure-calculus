from calculus.pseudonym import *
from calculus import combinators
from calculus import logic
from calculus import num_comparison
from calculus import arithm_ops
from calculus import nat_numbers


def gcd_term_v0():
    """
    :return: GCD ≡ Y (λf.λn.λm.  ITE (OR (ISZERO n) (ISZERO m)) [1]
             (ITE (EQ n m) n  (ITE (LE n m) (f n (SUBSTR m n)) (f m (SUBSTR n m))))))"""

    n, m, f = Var(), Var(), Var()
    n_, m_, f_ = Atom(n), Atom(m), Atom(f)

    swap_line = multi_app_term(
        logic.ite_term(),
        multi_app_term(num_comparison.leq_term(), n_, m_),
        multi_app_term(f_, n_, multi_app_term(arithm_ops.subtract_term(), m_, n_)),
        multi_app_term(f_, m_, multi_app_term(arithm_ops.subtract_term(), n_, m_)),
    )

    eq_nm_line = multi_app_term(logic.ite_term(),
                                multi_app_term(num_comparison.eq_term(), n_, m_),
                                n_, swap_line)
    zero_check = multi_app_term(logic.or_term(),
                                App(num_comparison.iszero_term(), n_),
                                App(num_comparison.iszero_term(), m_))

    inner_app = multi_app_term(logic.ite_term(), zero_check, nat_numbers.num_term(1), eq_nm_line)
    inner_lambda = Lambda(f, Lambda(n, Lambda(m, inner_app)))
    return App(combinators.y_term(), inner_lambda)


def gcd_term_v1():
    """:return: gcd ≡ (Y (λg.λa.λb. ITE (OR (ISZERO n) (ISZERO m)) [1]
                      (ite (equal a b) a (ite (greater_than a b) (g (minus a b) b) (g (minus b a) a) ))) )"""

    g, a, b = Var(), Var(), Var()
    g_, a_, b_ = Atom(g), Atom(a), Atom(b)
    equal_ab = multi_app_term(num_comparison.eq_term(), a_, b_)

    minus_ab = multi_app_term(g_, multi_app_term(arithm_ops.subtract_term(), a_, b_), b_)
    minus_ba = multi_app_term(g_, multi_app_term(arithm_ops.subtract_term(), b_, a_), a_)
    le_ba = multi_app_term(num_comparison.leq_term(), b_, a_)
    inner_ite_term = multi_app_term(logic.ite_term(), le_ba, minus_ab, minus_ba)

    ite_left_term = multi_app_term(logic.ite_term(), equal_ab, a_, inner_ite_term)
    main_or_term = multi_app_term(logic.or_term(),
                                  App(num_comparison.iszero_term(), a_),
                                  App(num_comparison.iszero_term(), b_))
    ite_main_term = multi_app_term(logic.ite_term(), main_or_term, nat_numbers.num_term(1), ite_left_term)
    gcd_lambda = Lambda(g, Lambda(a, Lambda(b, ite_main_term)))
    return App(combinators.y_term(), gcd_lambda)


def gcd_term_v3():  # TODO test it in more complex way, it doesn't work so fast as expected
    """Greatest common divisor/highest common factor:
       :return: GCD := (λgmn. LEQ m n (g n m) (g m n)) (Y (λgxy. ISZERO y x (g y (MOD x y))))"""
    g, m, n, x, y = Var(), Var(), Var(), Var(), Var()
    g_, m_, n_, x_, y_ = Atom(g), Atom(m), Atom(n), Atom(x), Atom(y)

    body_left_inner_term = multi_app_term(
        num_comparison.leq_term(), m_, n_,
        multi_app_term(g_, n_, m_),
        multi_app_term(g_, m_, n_)
    )
    left_inner_term = Lambda(g, Lambda(m, Lambda(n, body_left_inner_term)))

    body_right_inner_term = multi_app_term(
        num_comparison.iszero_term(), y_, x_,
        multi_app_term(g_, y_, multi_app_term(arithm_ops.mod_term(), x_, y_))
    )
    right_inner_term = Lambda(g, Lambda(x, Lambda(y, body_right_inner_term)))
    right_inner_term = App(combinators.y_term(), right_inner_term)

    return App(left_inner_term, right_inner_term)


def pythagorean_term():
    """:return: PYTHAGOREAN = λa.λb.λc (EQ (PLUS mul_aa, mul_bb) mul_cc)"""

    a, b, c = Var(), Var(), Var()
    a_, b_, c_ = Atom(a), Atom(b), Atom(c)

    mul_aa = multi_app_term(arithm_ops.mult_term(), a_, a_)
    mul_bb = multi_app_term(arithm_ops.mult_term(), b_, b_)
    mul_cc = multi_app_term(arithm_ops.mult_term(), c_, c_)

    plus_aa_bb = multi_app_term(arithm_ops.plus_term(), mul_aa, mul_bb)

    return Lambda(a, Lambda(b, Lambda(c, multi_app_term(num_comparison.eq_term(), plus_aa_bb, mul_cc))))
