from strategy import *
from term import Atom, Var, Abstraction, Application


def genTerm(p: float, uplimit: int, vars: List[Var] = [], trigger_by_application=False):
    if uplimit < 1:
        return None

    pVar = (1 - p * p) / 2
    pAbs = pVar + p * p

    rand = random.random()

    if rand < pVar and len(vars) > 0:
        index = random.randint(0, len(vars) - 1)
        return Atom(vars[index])
    elif rand < pAbs:
        head = Var()
        new_vars = vars + [head]
        body = genTerm(p, uplimit - 1, new_vars)
        return Abstraction(head, body) if body else None
    else:
        sub = genTerm(p, uplimit - 1, vars, trigger_by_application=True)
        obj = genTerm(p, uplimit - 1, vars)
        if sub and obj and sub.verticesNumber + obj.verticesNumber <= uplimit:
            return Application(sub, obj)
        else:
            return None




