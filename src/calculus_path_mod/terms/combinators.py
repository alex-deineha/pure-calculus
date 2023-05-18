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
