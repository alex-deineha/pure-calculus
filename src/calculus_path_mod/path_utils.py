from calculus_path_mod.term_engine import Term, LambdaError


def is_path(s):
    return isinstance(s, str) and len(s) == len([c for c in s if c in "ldr"])


def subref(t, p):
    if isinstance(t, Term) and is_path(p):
        if p == "":
            return t
        if p[0] == 'l' and t.kind == "application":
            return subref(t._data[0], p[1:])
        if p[0] == 'r' and t.kind == "application":
            return subref(t._data[1], p[1:])
        if p[0] == 'd' and t.kind == "abstraction":
            return subref(t._data[1], p[1:])
        # all other cases
        return None
    raise LambdaError("'subref' waits for an instance of Term and a path"
                      f", but it received '{t}' and '{p}'")


def paths(t):
    """collects all paths that refer to some correct subterm of 't'
    Result is a dictionary whose keys are paths determining
        the corresponding subterm
    """
    if isinstance(t, Term):
        result = {"": t}
        if t.kind == "atom":
            return result
        if t.kind == "application":
            return {**result,
                    **{("l" + key): val for (key, val) in
                       paths(subref(t, "l")).items()},
                    **{("r" + key): val for (key, val) in
                       paths(subref(t, "r")).items()}}
        # t.kind == "abstraction"
        return {**result,
                **{("d" + key): val for (key, val) in
                   paths(subref(t, "d")).items()}}
    raise LambdaError("'paths' waits for an instance of Term"
                      f", but it received '{t}'")


def similar(t1, t2):
    if isinstance(t1, Term) and isinstance(t2, Term):
        return paths(t1).keys() == paths(t2).keys()
    raise LambdaError("'similar' waits for two instances of Term"
                      f", but it received '{t1}' and '{t2}'")


def vars(t):
    """builds a dictionary, in which keys are refs to term variables,
    values are pairs constructed from the corresponding variable and
    the ref to the abstraction-superterm that bound the variable if
    it is bound or None elsewhen.
    """
    varoccs = {key: st._data
               for (key, st) in paths(t).items() if st.kind == "atom"}
    result = {}
    for key in varoccs:
        free = True
        for ie in range(1, len(key) + 1):
            subkey = key[: - ie]
            term = subref(t, subkey)
            if (term.kind == "abstraction" and
                    term._data[0] == varoccs[key]):
                result[key] = (varoccs[key], subkey)
                free = False
                break
        if free:
            result[key] = (varoccs[key], None)
    return result
