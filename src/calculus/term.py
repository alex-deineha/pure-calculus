def natgen():
    n = 0
    while True:
        yield n
        n += 1


class Var:
    __nats = natgen()

    def __init__(self):
        self._idx = next(Var.__nats)

    def __hash__(self):
        return self._idx.__hash__()

    def __str__(self):
        return "v[" + str(self._idx) + "]"

    def __eq__(self, other):
        return self._idx == other._idx


class Term:

    @property
    def isAtom(self):
        """checks whether the term is an atom"""
        return isinstance(self, Atom)

    @property
    def isApplication(self):
        """checks whether the term is an application"""
        return isinstance(self, Application)

    @property
    def isAbstraction(self):
        """checks whether the term is an abstraction"""
        return isinstance(self, Abstraction)

    def __str__(self):
        if self.isAtom:
            return str(self._var)
        if self.isApplication:
            return "(" + str(self._sub) + " " + str(self._obj) + ")"
        # self is Abbstraction
        return "(fun " + str(self._head) + " => " + str(self._body) + ")"

    def __eq__(self, other):
        if self.isAtom and other.isAtom:
            return self._var == other._var
        if isinstance(self, Application) and isinstance(other, Application):
            return self._sub == other._sub and self._obj == other._obj
        if isinstance(self, Abstraction) and isinstance(other, Abstraction):
            return self._head == other._head and self._body == other._body

    @property
    def isBetaRedex(self):
        """checks whether the term is a beta-redex"""
        return self.isApplication and self._sub.isAbstraction

    @property
    def redexes(self):
        """determiness all beta-redexes in the term"""
        if self.isAtom:
            return []
        if self.isAbstraction:
            return self._body.redexes
        # self is Application
        temp = [self] if self.isBetaRedex else []
        temp += (self._sub.redexes + self._obj.redexes)
        return temp

    @property
    def _vars(self):
        """
        returns
        -------
            the dictionary stuctured as follows
                dict[Var, dict[['free' | 'bound'], int]]
            Here, keys of the external dictionary are the variables that
            are occurred in 'self', and values of the internal dictionaries
            relate respectively to the numbers of free and bound occurrences
            of the variables.
        """
        if self.isAtom:
            return {self._var: {'free': 1, 'bound': 0}}
        if self.isApplication:
            vars, auxvars = dict(self._sub._vars), self._obj._vars
            for var in auxvars:
                try:
                    for key in {'free', 'bound'}:
                        vars[var][key] += self._obj._vars[var][key]
                except KeyError:
                    vars[var] = dict(self._obj._vars[var])
            return vars
        # self is Abstraction
        vars = dict(self._body._vars)
        try:
            vars[self._head]['bound'] += vars[self._head]['free']
            vars[self._head]['free'] = 0
        except KeyError:
            pass
        return vars

    @property
    def verticesNumber(self):
        """return the number of nodes in the tree representing the lambda term"""
        if self.isAtom:
            return 1
        elif self.isApplication:
            return 1 + self._sub.verticesNumber + self._obj.verticesNumber
        else:  # self is Abstraction
            return 1 + self._body.verticesNumber

    def normalize(self, strategy):
        """
      :param strategy: OneStepStrategy
      :return tuple of the normal form of the term and number of steps of betta reduction
      """
        term = self._updateBoundVariables()
        count = 0
        while term.redexes != []:
            term = term._betaConversion(strategy)
            count += 1
            if term.verticesNumber > 7000 or count > 400:
                return (self, float('inf'))
        return (term, count)

    def normalize_step(self, strategy):
        """
        :param strategy: OneStepStrategy
        :return True -- if it done Betta conversion step
                False -- otherwise
        """
        if self.normalization_term is None:
            self.normalization_term = self._updateBoundVariables()

        if self.normalization_term.redexes != []:
            if self.normalization_term.verticesNumber > 7000:
                return True
            self.normalization_term = self.normalization_term._betaConversion(strategy)
            return True
        return False

    def restart_normalization(self):
        """
        Restart, for possibility to do normalization
        """
        self.normalization_term = None

    def _betaConversion(self, strategy):
        """
      :param strategy: OneStepStrategy
      :return term with redex eliminated using the given strategy
      """
        index = strategy.redexIndex(self)
        subterm = self.subterm(index)
        reducedTerm = subterm._removeOuterRedex()
        return self.setSubterm(index, reducedTerm)

    def subterm(self, index: int):
        """
      By representing the term as a tree, a subtree is returned, which is also a lambda term.
      The vertex of this subtree has a given index in the topological sorting of the vertices of the original term.
      :param index - subterm index
      :return: subterm: Term
      """
        if index == 1:
            return self

        if self.isAtom:
            ValueError('index value is incorrect')
        elif self.isApplication:
            if self._sub.verticesNumber + 1 >= index:
                return self._sub.subterm(index - 1)
            else:
                return self._obj.subterm(index - self._sub.verticesNumber - 1)
        else:  # self is Abstraction
            return self._body.subterm(index - 1)

    def setSubterm(self, index: int, term):
        """
      By representing the term as a tree, a subtree is set, which is also a lambda term.
      The vertex of this subtree has a given index in the topological sorting of the vertices of the original term.
      :param index - subterm index
      :param term - λ-term to which the subterm will be replaced
      :return: updated λ-term
      """
        if index == 1:
            return term

        if self.isAtom:
            ValueError('index value is incorrect')
        elif self.isApplication:
            if self._sub.verticesNumber + 1 >= index:
                return Application(self._sub.setSubterm(index - 1, term), self._obj)
            else:
                return Application(self._sub, self._obj.setSubterm(index - self._sub.verticesNumber - 1, term))
        else:  # self is Abstraction
            return Abstraction(self._head, self._body.setSubterm(index - 1, term))

    def _updateBoundVariables(self):
        """return λ-term with updated bound variables"""
        if self.isAtom:
            return self
        elif self.isApplication:
            return Application(self._sub._updateBoundVariables(), self._obj._updateBoundVariables())
        else:  # self is Abstraction
            newVar = Var()
            return Abstraction(newVar, self._body._replaceVariable(self._head, Atom(newVar))._updateBoundVariables())

    def _removeOuterRedex(self):
        """apply the betta conversion to the lambda term, removing the outer betta redex"""
        if self.isBetaRedex:
            head = self._sub._head
            body = self._sub._body
            return body._replaceVariable(head, self._obj)
        else:
            return self

    def _replaceVariable(self, var: Var, term):
        """return λ-term with replaced variable"""
        if self.isAtom:
            return term if self._var == var else self
        elif self.isApplication:
            return Application(self._sub._replaceVariable(var, term), self._obj._replaceVariable(var, term))
        else:  # self is Abstraction
            return Abstraction(self._head, self._body._replaceVariable(var, term))


class Atom(Term):
    def __init__(self, x: Var):
        if isinstance(x, Var):
            self._var = x
        else:
            raise TypeError("a variable is waiting")


class Application(Term):
    def __init__(self, X: Term, Y: Term):
        if isinstance(X, Term) and isinstance(Y, Term):
            self._sub = X
            self._obj = Y
        else:
            raise TypeError("a term is waiting")


class Abstraction(Term):
    def __init__(self, x: Var, X: Term):
        if isinstance(x, Var):
            if isinstance(X, Term):
                self._head = x
                self._body = X
            else:
                raise TypeError("a term is waiting")
        else:
            raise TypeError("a variable is waiting")
