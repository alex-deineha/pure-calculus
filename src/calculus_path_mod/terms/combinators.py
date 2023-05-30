from calculus_path_mod.terms.pseudonym import *


def k_term():
    """:return: K := λxy. x"""

    x, y = Var(), Var()
    x_ = Atom(x)
    return Lambda(x, Lambda(y, x_))


def k_star_term():
    """:return: K* := λxy. y"""

    x, y = Var(), Var()
    y_ = Atom(y)
    return Lambda(x, Lambda(y, y_))


def s_term():
    """:return: S := λxyz. (x z) (y z)"""

    x, y, z = Var(), Var(), Var()
    x_, y_, z_ = Atom(x), Atom(y), Atom(z)
    return Lambda(x, Lambda(y, Lambda(z, multi_app_term(x_, z_, App(y_, z_)))))


def i_term():
    """:return: I := λx. x"""

    x = Var()
    x_ = Atom(x)
    return Lambda(x, x_)


def y_term():
    """:return: Y := λg. (λx. g (x x)) (λx. g (x x))"""

    g, x = Var(), Var()
    g_, x_ = Atom(g), Atom(x)
    return Lambda(g, App(
        Lambda(x, App(g_, App(x_, x_))),
        Lambda(x, App(g_, App(x_, x_)))
    ))


def z_term():
    """:return: Z := λf. (λx.f(λv.xxv))(λx.f(λv.xxv))"""

    f, x, v = Var(), Var(), Var()
    f_, x_, v_ = Atom(f), Atom(x), Atom(v)
    return Lambda(f, App(
        Lambda(x, App(f_, Lambda(v, multi_app_term(x_, x_, v_)))),
        Lambda(x, App(f_, Lambda(v, multi_app_term(x_, x_, v_))))
    ))


def etta_v_term():
    """:return: Etta_v := (λxy.y(λx.xxyz))(λxy.y(λx.xxyz))"""

    x, y, z = Var(), Var(), Var()
    x_, y_, z_ = Atom(x), Atom(y), Atom(y)

    return App(
        Lambda(x, Lambda(y, App(y_, Lambda(z, multi_app_term(x_, x_, y_, z_))))),
        Lambda(x, Lambda(y, App(y_, Lambda(z, multi_app_term(x_, x_, y_, z_)))))
    )

