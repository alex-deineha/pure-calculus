from calculus.term import Application as App
from calculus.term import Abstraction as Lambda
from calculus.term import Var, Atom


def multi_app_term(term_0, term_1, *terms):
    res_app_term = App(term_0, term_1)
    for term in terms:
        res_app_term = App(res_app_term, term)
    return res_app_term


# BLOCK OF USEFUL COMBINATORS
def i_term():
    x = Var()
    x_ = Atom(x)
    return Lambda(x, x_)


def k_term():
    x, y = Var(), Var()
    x_ = Atom(x)
    return Lambda(x, Lambda(y, x_))


def k_star_term():
    x, y = Var(), Var()
    y_ = Atom(y)
    return Lambda(x, Lambda(y, y_))


def s_term():
    x, y, z = Var(), Var(), Var()
    x_, y_, z_ = Atom(x), Atom(y), Atom(z)
    # return Lambda(x, Lambda(y, Lambda(z, App(x_, App(z_, App(y_, z_))))))
    return Lambda(x, Lambda(y, Lambda(z, multi_app_term(x_, z_, App(y_, z_)))))


# BLOCK OF FIXED-POINT COMBINATORS
def y_term():
    x, f = Var(), Var()
    x_, f_ = Atom(x), Atom(f)
    return Lambda(
        f,
        App(
            Lambda(x, multi_app_term(f_, x_, x_)), Lambda(x, multi_app_term(f_, x_, x_))
        ),
    )


# BLOCK OF LOGIC
def true_term():
    return k_term()


def false_term():
    return k_star_term()


def ite_term():
    x, y, c = Var(), Var(), Var()
    x_, y_, c_ = Atom(x), Atom(y), Atom(c)
    # return Lambda(c, Lambda(x, Lambda(y, App(App(c_, x_), y_))))
    return Lambda(c, Lambda(x, Lambda(y, multi_app_term(c_, x_, y_))))


def not_term():
    a = Var()
    a_ = Atom(a)
    return Lambda(a, multi_app_term(ite_term(), a_, false_term(), true_term()))


def and_term():
    a, b = Var(), Var()
    a_, b_ = Atom(a), Atom(b)
    # return Lambda(a, Lambda(b, App(ite_term(), App(a_, App(b_, a_)))))
    return Lambda(a, Lambda(b, multi_app_term(ite_term(), a_, b_, a_)))


def or_term():
    a, b = Var(), Var()
    a_, b_ = Atom(a), Atom(b)
    return Lambda(a, Lambda(b, multi_app_term(ite_term(), a_, a_, b_)))


# BLOC OF NUMBERS
def n0_term():
    s, z = Var(), Var()
    z_ = Atom(z)
    return Lambda(s, Lambda(z, z_))


def n_term(n: int):
    if n < 0:
        raise ValueError("in lambda calculus number can't be less than 0")
    if n == 0:
        return n0_term()
    s, z = Var(), Var()
    s_, z_ = Atom(s), Atom(z)
    core_term = App(s_, z_)
    for _ in range(n - 1):
        core_term = App(s_, core_term)
    return Lambda(s, Lambda(z, core_term))


# BLOCK OF ARITHMETICS
def succ_term():
    x, y, n = Var(), Var(), Var()
    x_, y_, n_ = Atom(x), Atom(y), Atom(n)
    return Lambda(n, Lambda(x, Lambda(y, App(x_, multi_app_term(n_, x_, y_)))))


def iszero_term():
    x, n = Var(), Var()
    n_ = Atom(n)
    return Lambda(n, App(App(n_, Lambda(x, false_term())), true_term()))


def pair_term():
    x, y, p = Var(), Var(), Var()
    x_, y_, p_ = Atom(x), Atom(y), Atom(p)
    return Lambda(x, Lambda(y, Lambda(p, multi_app_term(p_, x_, y_))))


def first_term():
    p = Var()
    p_ = Atom(p)
    return Lambda(p, App(p_, k_term()))


def second_term():
    p = Var()
    p_ = Atom(p)
    return Lambda(p, App(p_, k_star_term()))


def sinc_term():
    p = Var()
    p_ = Atom(p)
    return Lambda(
        p,
        multi_app_term(
            pair_term(),
            App(second_term(), p_),
            App(succ_term(), App(second_term(), p_)),
        ),
    )


def fsinc_term():
    p, z = Var(), Var()
    p_, z_ = Atom(p), Atom(z)
    return Lambda(
        p,
        Lambda(
            z,
            multi_app_term(
                z_, App(succ_term(), App(p_, true_term())), App(p_, true_term())
            ),
        ),
    )


def fpred_term():
    n, z = Var(), Var()
    n_, z_ = Atom(n), Atom(z)
    return Lambda(
        n,
        multi_app_term(
            n_,
            fsinc_term(),
            Lambda(z, multi_app_term(z_, n0_term(), n0_term())),
            false_term(),
        ),
    )


def fsubstr_term():
    n, m = Var(), Var()
    n_, m_ = Atom(n), Atom(m)
    return Lambda(n, Lambda(m, multi_app_term(m_, fpred_term(), n_)))


def pred_term():
    n = Var()
    n_ = Atom(n)
    return Lambda(
        n,
        App(
            first_term(),
            multi_app_term(
                n_, sinc_term(), multi_app_term(pair_term(), n0_term(), n0_term())
            ),
        ),
    )


def substr_term():
    n, m = Var(), Var()
    n_, m_ = Atom(n), Atom(m)
    return Lambda(n, Lambda(m, multi_app_term(m_, pred_term(), n_)))


def le_term():
    n, m = Var(), Var()
    n_, m_ = Atom(n), Atom(m)
    return Lambda(
        n, Lambda(m, App(iszero_term(), multi_app_term(substr_term(), n_, m_)))
    )


def eq_term():
    n, m = Var(), Var()
    n_, m_ = Atom(n), Atom(m)
    return Lambda(
        n,
        Lambda(
            m,
            multi_app_term(
                and_term(),
                multi_app_term(le_term(), n_, m_),
                multi_app_term(le_term(), n_, m_),
            ),
        ),
    )


def gcd_term():
    n, m, f = Var(), Var(), Var()
    n_, m_, f_ = Atom(n), Atom(m), Atom(f)
    sub_m_n = App(App(substr_term(), m_), n_)
    sub_n_m = App(App(substr_term(), n_), m_)
    iszero_n = App(iszero_term(), n_)
    iszero_m = App(iszero_term(), m_)
    f_n_sub = App(App(f_, n_), sub_m_n)
    f_m_sub = App(App(f_, m_), sub_n_m)
    le_n_m = App(App(le_term(), n_), m_)
    eq_n_m = App(App(eq_term(), n_), m_)
    or_is_zero = App(or_term(), App(iszero_n, iszero_m))
    ite_le_sub_sub = App(App(App(ite_term(), le_n_m), f_n_sub), f_m_sub)
    ite_eq_ite = App(App(App(ite_term(), eq_n_m), n_), ite_le_sub_sub)
    or_is_ite = App(App(or_is_zero, n0_term()), ite_eq_ite)
    inner_app = App(ite_term(), or_is_ite)
    inner_lambda = Lambda(f, Lambda(n, Lambda(m, inner_app)))
    return App(y_term(), inner_lambda)


def main():
    from strategy import LeftmostOutermostStrategy

    print("term ITE TRUE x y == x")
    term = App(ite_term(), App(true_term(), App(Atom(Var()), Atom(Var()))))
    norm_res = term.normalize(strategy=LeftmostOutermostStrategy())
    print(norm_res[0])
    print(norm_res[1])

    print("\n\nSUCC 0 == 1")
    term = App(succ_term(), n_term(0))
    norm_res = term.normalize(strategy=LeftmostOutermostStrategy())
    print(norm_res[0])
    print(norm_res[1])

    print("\n\nSUCC 1 == 1")
    term = App(succ_term(), n_term(1))
    norm_res = term.normalize(strategy=LeftmostOutermostStrategy())
    print(norm_res[0])
    print(norm_res[1])

    print("\n\nISZERO 0 == TRUE")
    term = App(iszero_term(), n_term(0))
    norm_res = term.normalize(strategy=LeftmostOutermostStrategy())
    print(norm_res[0])
    print(norm_res[1])

    print("\n\nISZERO 1 == TRUE")
    term = App(iszero_term(), n_term(1))
    norm_res = term.normalize(strategy=LeftmostOutermostStrategy())
    print(norm_res[0])
    print(norm_res[1])


if __name__ == "__main__":
    main()
