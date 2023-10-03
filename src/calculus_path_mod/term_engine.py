import time

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

    def _get_list_variables(self) -> list:
        """
        :return: list of all variable indexes, which appear in
                 the term tree, using left-recursion through the tree
        """

        if self.kind == "atom":
            return [self._data._data]
        if self.kind == "application":
            return self._data[0]._get_list_variables() + self._data[1]._get_list_variables()
        return [self._data[0]._data] + self._data[1]._get_list_variables()

    def _get_var_pseudonyms(self) -> dict:
        """
        Using a list variables of the terms generate pseudonyms of the term variables.
        Can be used for printing nice view of term or for colorizing the tree.
        :return: dict, where key is a var index and var is a variable name
        """

        list_vars = self._get_list_variables()
        set_vars = set()
        list_ordered_vars = []

        for var_ in list_vars:
            if var_ not in set_vars:
                set_vars.add(var_)
                list_ordered_vars.append(var_)

        pseudonyms = dict()
        for inx, uvi in enumerate(list_ordered_vars):
            pseudonyms[uvi] = (
                DEF_VAR_NAMES[inx]
                if inx < len(DEF_VAR_NAMES)
                else DEF_VAR_NAMES[inx % len(DEF_VAR_NAMES)]
                     + "_"
                     + str(int(inx / len(DEF_VAR_NAMES)))
            )

        return pseudonyms

    def funky_str(self, pseudonyms: dict = None, redex_index=-1) -> str:
        """
        ! Warning when call the method you don't need to set any parameters.
        :param redex_index: redex index, set actual redex index instead of '-1' to show redex
        :param pseudonyms: dict, where key is a var index and var is a variable name
        :return: str representation of the
        """

        redex_index -= 1
        if pseudonyms is None:
            pseudonyms = self._get_var_pseudonyms()
        if self.kind == "atom":
            return pseudonyms[self._data._data]
        if self.kind == "application":
            if redex_index == 0:
                return f"[>> {self._data[0].funky_str(pseudonyms, redex_index)} *** " \
                       f"{self._data[1].funky_str(pseudonyms, redex_index - self._data[0].vertices_number)} <<]"
            else:
                return f"({self._data[0].funky_str(pseudonyms, redex_index)} " \
                       f"{self._data[1].funky_str(pseudonyms, redex_index - self._data[0].vertices_number)})"
        return f"(λ{pseudonyms[self._data[0]._data]}.{self._data[1].funky_str(pseudonyms, redex_index)})"

    def funky_v2_str(self, pseudonyms: dict = None, redex_index=-1) -> str:
        """
        ! Use this method for preparing preprocessing your
        :param redex_index: redex index, set actual redex index instead of '-1' to show redex
        :param pseudonyms: dict, where key is a var index and var is a variable name
        :return: str representation of the term
        """

        redex_index -= 1
        if pseudonyms is None:
            pseudonyms = self._get_var_pseudonyms()
        if self.kind == "atom":
            return pseudonyms[self._data._data]
        if self.kind == "application":
            if redex_index == 0:
                return ""
            else:
                return f"({self._data[0].funky_v2_str(pseudonyms, redex_index)} " \
                       f"{self._data[1].funky_v2_str(pseudonyms, redex_index - self._data[0].vertices_number)})"
        return f"[{pseudonyms[self._data[0]._data]} {self._data[1].funky_v2_str(pseudonyms, redex_index)}]"

    def simple_str(self):
        if self.kind == "atom":
            return "x"
        if self.kind == "application":
            return f"({self._data[0].simple_str()} {self._data[1].simple_str()})"
        else:  # self.kind == "absraction"
            return f"(@x. {self._data[1].simple_str()})"

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
            return {self._data._data: {"free": 1, "bound": 0}}
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
            vars_[self._data[0]._data]["bound"] += vars_[self._data[0]._data]["free"]
            vars_[self._data[0]._data]["free"] = 0
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

    @property
    def term_width(self):
        """:return: count of applications in the term"""

        if self.kind == "atom":
            return 0
        if self.kind == "application":
            return 1 + self._data[0].term_width + self._data[1].term_width
        # self is Abstraction
        return self._data[1].term_width

    @property
    def term_height(self):
        """:return: longest recursion length in the term tree"""

        if self.kind == "atom":
            return 0
        if self.kind == "application":
            return 1 + max(self._data[0].term_height, self._data[1].term_height)
        # self is Abstraction
        return 1 + self._data[1].term_height

    def redex_depth(self, redex_index):
        if redex_index == -1:
            return -1
        elif redex_index == 1:
            return 1
        if self.kind == "atom":
            raise ValueError("Redex index value is incorrect")
        elif self.kind == "application":
            if self._data[0].vertices_number + 1 >= redex_index:
                return 1 + self._data[0].redex_depth(redex_index - 1)
            else:
                return 1 + self._data[1].redex_depth(redex_index - self._data[0].vertices_number - 1)
        else:
            return 1 + self._data[1].redex_depth(redex_index - 1)

    def normalize_with_params(self, strategy, is_limited=True, steps_lim=400, vertices_lim=7_000):
        norm_params = {
            "vertices": [self.vertices_number],
            "redexes": [len(self.redexes)],
            "redex_depths": [],
            "redex_indexes": [],
            "heights": [self.term_height],
            "widths": [self.term_width],
            "steps_time": [],
        }

        (step_term, redex_index, reduction_time), norm_term = self.one_step_normalize_visual(strategy)
        norm_params["redex_depths"].append(self.redex_depth(redex_index))
        norm_params["redex_indexes"].append(redex_index)
        norm_params["steps_time"].append(reduction_time)

        while norm_term:
            (step_term, redex_index, reduction_time), norm_term = norm_term.one_step_normalize_visual(strategy)
            norm_params["vertices"].append(step_term.vertices_number)
            norm_params["redexes"].append(len(step_term.redexes))
            norm_params["heights"].append(step_term.term_height)
            norm_params["widths"].append(step_term.term_width)
            norm_params["redex_depths"].append(step_term.redex_depth(redex_index))
            norm_params["redex_indexes"].append(redex_index)
            norm_params["steps_time"].append(reduction_time)

            # computation limitation
            if is_limited and ((step_term.vertices_number > vertices_lim) or (len(norm_params) > steps_lim)):
                raise Exception("Too many vertices, or too many steps to visualize")

        return norm_params

    def count_same_vars(self, var_):
        if self.kind == "atom":
            return 1 if self._data == var_ else 0
        if self.kind == "application":
            return self._data[0].count_same_vars(var_) + self._data[1].count_same_vars(var_)
        return self._data[1].count_same_vars(var_)

    def normalize_with_params_v2(self, strategy, is_limited=True, steps_lim=400, vertices_lim=7_000):
        norm_params = {
            "vertices": [self.vertices_number],
            "redexes": [len(self.redexes)],
            "redex_depths": [],
            "redex_indexes": [],
            "heights": [self.term_height],
            "widths": [self.term_width],

            "redex_subj_vars": [],
            "redex_obj_vertices": [],
            "redex_obj_heights": [],
            "redex_obj_widths": [],

            "steps_time": [],
        }

        (step_term, redex_index, reduction_time), norm_term = self.one_step_normalize_visual(strategy)

        redex_term = self.subterm(redex_index)
        norm_params["redex_subj_vars"].append(redex_term._data[0].count_same_vars(redex_term._data[0]._data[0]))
        norm_params["redex_obj_vertices"].append(redex_term._data[1].vertices_number)
        norm_params["redex_obj_heights"].append(redex_term._data[1].term_height)
        norm_params["redex_obj_widths"].append(redex_term._data[1].term_width)

        norm_params["redex_depths"].append(self.redex_depth(redex_index))
        norm_params["redex_indexes"].append(redex_index)
        norm_params["steps_time"].append(reduction_time)

        while norm_term:
            try:
                redex_index = strategy.redex_index(norm_term)
            except Exception:
                redex_index = -1
            if redex_index > 0:
                redex_term = norm_term.subterm(redex_index)
                norm_params["redex_subj_vars"].append(redex_term._data[0].count_same_vars(redex_term._data[0]._data[0]))
                norm_params["redex_obj_vertices"].append(redex_term._data[1].vertices_number)
                norm_params["redex_obj_heights"].append(redex_term._data[1].term_height)
                norm_params["redex_obj_widths"].append(redex_term._data[1].term_width)
            else:
                norm_params["redex_subj_vars"].append(-1)
                norm_params["redex_obj_vertices"].append(-1)
                norm_params["redex_obj_heights"].append(-1)
                norm_params["redex_obj_widths"].append(-1)

            (step_term, redex_index, reduction_time), norm_term = norm_term.one_step_normalize_visual(strategy)
            norm_params["vertices"].append(step_term.vertices_number)
            norm_params["redexes"].append(len(step_term.redexes))
            norm_params["heights"].append(step_term.term_height)
            norm_params["widths"].append(step_term.term_width)
            norm_params["redex_depths"].append(step_term.redex_depth(redex_index))
            norm_params["redex_indexes"].append(redex_index)
            norm_params["steps_time"].append(reduction_time)

            # computation limitation
            if is_limited and ((step_term.vertices_number > vertices_lim) or (len(norm_params) > steps_lim)):
                raise Exception("Too many vertices, or too many steps to visualize")

        return norm_params

    def one_step_normalize_visual(self, strategy):
        """
        :param strategy:OneStepStrategy
        :return: (term, redex_index, process_time), reduced_term
        """
        term = self._update_bound_vars()

        if len(term.redexes) > 0:
            start_time = time.process_time_ns()
            redex_index = strategy.redex_index(term)
            reduced_term = term._beta_conversion_visual(redex_index)
            end_time = time.process_time_ns()
            return (term, redex_index, end_time - start_time), reduced_term
        else:
            return (term, -1, 0), None

    def _beta_conversion_visual(self, redex_index):
        """
        :param redex_index: a redex position in the tree term.
        :return: term with redex eliminated using the given strategy
        """
        subterm_ = self.subterm(redex_index)
        reduced_term = subterm_._remove_outer_redex()
        return self.set_subterm(redex_index, reduced_term)

    def normalize_visual(self, strategy):
        """
        :param strategy: OneStepStrategy
        :return: list of (step_term, redex_index, reduction_time)
        """
        list_steps = list()
        (step_term, redex_index, reduction_time), norm_term = self.one_step_normalize_visual(strategy)
        list_steps.append((step_term, redex_index, reduction_time))

        while norm_term:
            (step_term, redex_index, reduction_time), norm_term = norm_term.one_step_normalize_visual(strategy)
            list_steps.append((step_term, redex_index, reduction_time))
            # computation limitation
            if (step_term.vertices_number > 7_000) or (len(list_steps) > 400):
                raise Exception("Too many vertices, or too many steps to visualize")
        return list_steps

    def normalize(self, strategy, is_limited=True, steps_lim=400, vertices_lim=7_000):
        """
        :param strategy: OneStepStrategy
        :param is_limited: if it's True normalization process stops till
                        steps_lim and vertices_lim reached.
        :param steps_lim: maximum steps to normalize, applied only if is_limited=True
        :param vertices_lim: maximum count of vertices in the term,
                        applied only if is_limited=True
        :return: tuple of the normal form of the term and number of steps of betta reduction
        """
        term = self._update_bound_vars()
        count_steps = 0
        while len(term.redexes) > 0:
            term = term._beta_conversion(strategy)._update_bound_vars()
            count_steps += 1
            # computation limitation
            if is_limited and ((term.vertices_number > vertices_lim) or (count_steps > steps_lim)):
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
            self._data[1]._replace_bound_variable(self._data[0], Atom(new_var))._update_bound_vars()
        )

    def _replace_bound_variable(self, var: Var, term):
        """Return λ-term with replaced variable"""
        if self.kind == "atom":
            return term if self._data._data == var._data else self
        if self.kind == "application":
            return Application(self._data[0]._replace_bound_variable(var, term),
                               self._data[1]._replace_bound_variable(var, term))
        # self is abstraction
        if self._data[0]._data == var._data:
            new_var = Var()
            return Abstraction(new_var, self._data[1]._replace_bound_variable(self._data[0], Atom(new_var)))
        return Abstraction(self._data[0], self._data[1]._replace_bound_variable(var, term))

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
