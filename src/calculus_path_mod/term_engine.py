DEF_VAR_NAMES = "xyabcdejinmtrqwuopsfghklzv"


class LambdaError(Exception):
    __errmsg = [
        "unrecognised error",
    ]

    def __init__(self, errDescription):
        if isinstance(errDescription, int):
            try:
                self._msg = LambdaError.__errmsg[errDescription]
            except:
                self._msg = LambdaError.__errmsg[0]
        elif isinstance(errDescription, str):
            self._msg = errDescription
        else:
            self._msg = LambdaError.__errmsg[0]
        super().__init__(self._msg)


class Var:
    __cvar = 0

    def __init__(self):
        self._data = Var.__cvar
        Var.__cvar += 1

    def __str__(self):
        return f"#{self._data}"

    def __eq__(self, another):
        if isinstance(another, Var):
            return self._data == another._data
        raise LambdaError("Var.__eq__ waits for an instance of Var"
                          f", but it received '{another}'")


class Term:  # the basic abstract class for representing a term

    @property
    def kind(self):  # returns the kind of the term
        if isinstance(self, Atom):
            return "atom"
        if isinstance(self, Application):
            return "application"
        if isinstance(self, Abstraction):
            return "abstraction"

    def __str__(self):
        if self.kind == "atom":
            return f"{self._data}"
        if self.kind == "application":
            return f"({self._data[0]} {self._data[1]})"
        else:  # self.kind == "absraction"
            return f"(λ{self._data[0]}. {self._data[1]})"

    def _get_list_variables(self):
        if self.kind == "atom":
            return [self._data._data]
        if self.kind == "application":
            return self._data[0]._get_list_variables() + self._data[1]._get_list_variables()
        return [self._data[0]._data] + self._data[1]._get_list_variables()

    def _get_var_pseudonyms(self):
        unique_vars_inx = set(self._get_list_variables())
        pseudonyms = dict()
        for inx, uvi in enumerate(unique_vars_inx):
            pseudonyms[uvi] = (
                DEF_VAR_NAMES[inx]
                if inx < len(DEF_VAR_NAMES)
                else DEF_VAR_NAMES[inx % len(DEF_VAR_NAMES)]
                     + "_"
                     + str(int(inx / len(DEF_VAR_NAMES)))
            )

        return pseudonyms

    def funky_str(self, pseudonyms: dict = None):
        if pseudonyms is None:
            pseudonyms = self._get_var_pseudonyms()
        if self.kind == "atom":
            return pseudonyms[self._data._data]
        if self.kind == "application":
            return (
                f"({self._data[0].funky_str(pseudonyms)} {self._data[1].funky_str(pseudonyms)})"
            )
        return f"(λ{pseudonyms[self._data[0]._data]}.{self._data[1].funky_str(pseudonyms)})"

    # def __eq__(self, another):
    #     if isinstance(another, Term):
    #         if self.kind != another.kind:
    #             return False
    #         return self._data == another._data
    #     else:
    #         raise LambdaError(3)
    def __eq__(self, other):
        if not isinstance(other, Term):
            return False
        if self.kind == "atom" and other.kind == "atom":
            return self._data == other._data
        if self.kind == "application" and other.kind == "application":
            return self._data[0] == other._data[0] and self._data[1] == other._data[1]
        if self.kind == "abstraction" and other.kind == "abstraction":
            return self._data[0] == other._data[0] and self._data[1] == other._data[1]
        return False

    def call_as_method(self, fun, *args):
        return fun(self, *args)

    @property
    def is_beta_redex(self):
        """:return: bool is the term is a beta-redex"""
        return (self.kind == "application") and (self._data[0].kind == "abstraction")

    @property
    def redexes(self):
        """:return: list of all beta-redexes in the term"""
        if self.kind == "atom":
            return []
        if self.kind == "abstraction":
            return self._data[1].redexes
        # self is App:
        redexes_list = [self] if self.is_beta_redex else []
        redexes_list += self._data[0].redexes + self._data[1].redexes
        return redexes_list

    @property
    def _vars(self):
        """
        Here, keys of the external dictionary are the variables that
        are occurred in 'self', and values of the internal dictionaries
        relate respectively to the numbers of free and bound occurrences
        of the variables.
        :return: dict[Var, dict[('free'/'bound'), int]]
        """
        if self.kind == "atom":
            return {self._data: {"free": 1, "bound": 0}}
        if self.kind == "application":
            vars_, auxvars_ = dict(self._data[0]._vars), self._data[1]._vars
            for var_ in auxvars_:
                try:
                    for key_ in ("free", "bound"):
                        vars_[var_][key_] += self._data[1]._vars[var_][key_]
                except KeyError:
                    vars_[var_] = dict(self._data[1]._vars[var_])
            return vars_
        # self is Abstraction:
        vars_ = dict(self._data[1]._vars)
        try:
            vars_[self._data[0]]["bound"] += vars_[self._data[0]]["free"]
            vars_[self._data[0]]["free"] = 0
        except KeyError:
            pass
        return vars_

    @property
    def vertices_number(self):
        """:return: the number of nodes in the tree representation the lambda term"""
        if self.kind == "atom":
            return 1
        if self.kind == "application":
            return 1 + self._data[0].vertices_number + self._data[1].vertices_number
        # self is Abstraction
        return 1 + self._data[1].vertices_number

    def normalize(self, strategy):
        """
        :param strategy: OneStepStrategy
        :return: tuple of the normal form of the term and number of steps of betta reduction
        """
        term = self._update_bound_vars()
        count_steps = 0
        while len(term.redexes) > 0:
            term = term._beta_conversion(strategy)._update_bound_vars()
            count_steps += 1
            # computation limitation
            if (term.vertices_number > 7_000) or (count_steps > 400):
                return term, float("inf")
        return term, count_steps

    def _beta_conversion(self, strategy):
        """
        :param strategy: OneStepStrategy
        :return: term with redex eliminated using the given strategy
        """
        index = strategy.redex_index(self)
        subterm_ = self.subterm(index)
        reduced_term = subterm_._remove_outer_redex()
        return self.set_subterm(index, reduced_term)

    def subterm(self, index: int):
        """
        By representing the term as a tree, a subtree is returned,
        which is also a lambda term.
        The vertex of this subtree has a given index in the topological
        sorting of the vertices of the original term.
        :param index: int subterm index
        :return: subterm: Term
        """
        if index == 1:
            return self
        if self.kind == "atom":
            raise ValueError("index value is incorrect")
        elif self.kind == "application":
            if self._data[0].vertices_number + 1 >= index:
                return self._data[0].subterm(index - 1)
            else:
                return self._data[1].subterm(index - self._data[0].vertices_number - 1)
        else:
            return self._data[1].subterm(index - 1)

    def set_subterm(self, index: int, term):
        """
        By representing the term as a tree, a subtree is set, which is also a lambda term.
        The vertex of this subtree has a given index in the topological sorting of the vertices of the original term.
        :param index: subterm index
        :param term: λ-term to which the subterm will be replaced
        :return: updated λ-term
        """
        if index == 1:
            return term

        if self.kind == "atom":
            raise ValueError("index value is incorrect")
        elif self.kind == "application":
            if self._data[0].vertices_number + 1 >= index:
                return Application(self._data[0].set_subterm(index - 1, term), self._data[1])
            else:
                return Application(self._data[0],
                                   self._data[1].set_subterm(index - self._data[0].vertices_number - 1, term))
        else:
            return Abstraction(self._data[0], self._data[1].set_subterm(index - 1, term))

    def _update_bound_vars(self):
        """:return: λ-term with updated bound variables"""
        if self.kind == "atom":
            return self
        if self.kind == "application":
            return Application(
                self._data[0]._update_bound_vars(),
                self._data[1]._update_bound_vars()
            )
        # self is abstraction
        new_var = Var()
        return Abstraction(
            new_var,
            self._data[1]._replace_variable(self._data[0], Atom(new_var))._update_bound_vars()
        )

    def _remove_outer_redex(self):
        """Apply the betta conversion to the lambda term, removing the outer betta redex"""
        if self.is_beta_redex:
            head = self._data[0]._data[0]
            body = self._data[0]._data[1]
            return body._replace_variable(head, self._data[1])
        else:
            return self

    def _replace_variable(self, var: Var, term):
        """Return λ-term with replaced variable"""
        if self.kind == "atom":
            return term if self._data == var else self
        if self.kind == "application":
            return Application(self._data[0]._replace_variable(var, term),
                               self._data[1]._replace_variable(var, term))
        # self is abstraction
        return Abstraction(self._data[0], self._data[1]._replace_variable(var, term))


class Atom(Term):  # the class of terms created with the first rule

    def __init__(self, v):
        if isinstance(v, Var):
            self._data = v
        else:
            raise LambdaError("Atom.__init__ waits for an instance of Var"
                              f", but it received '{v}'")


class Application(Term):  # the class of terms created with the second rule

    def __init__(self, t1, t2):
        if isinstance(t1, Term) and isinstance(t2, Term):
            self._data = (t1, t2)
        else:
            raise LambdaError("Application.__init__ waits for two instances"
                              f" of Term, but it received '{t1}', '{t2}'")


class Abstraction(Term):  # the class of terms created with the third rule

    def __init__(self, v, t):
        if isinstance(v, Var) and isinstance(t, Term):
            self._data = (v, t)
        else:
            raise LambdaError("Abstraction.__init__ waits for an instance of"
                              " Var and an instance of Term"
                              f", but it receive '{v}' and '{t}'")
