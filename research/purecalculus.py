# import random
#
#
# def natgen():
#     n = 0
#     while True:
#         yield n
#         n += 1
#
#
# class Var:
#     __nats = natgen()
#
#     def __init__(self):
#         self._idx = next(Var.__nats)
#
#     def __hash__(self):
#         return self._idx.__hash__()
#
#     def __str__(self):
#         return "v[" + str(self._idx) + "]"
#
#     def __eq__(self, other):
#         return self._idx == other._idx
#
#
# class Term:
#     @property
#     def isAtom(self):
#         """checks whether the term is an atom"""
#         return isinstance(self, Atom)
#
#     @property
#     def isApplication(self):
#         """checks whether the term is an application"""
#         return isinstance(self, Application)
#
#     @property
#     def isAbstraction(self):
#         """checks whether the term is an abstraction"""
#         return isinstance(self, Abstraction)
#
#     def __str__(self):
#         if self.isAtom:
#             return str(self._var)
#         if self.isApplication:
#             return "(" + str(self._sub) + " " + str(self._obj) + ")"
#         # self is Abbstraction
#         return "(fun " + str(self._head) + " => " + str(self._body) + ")"
#
#     def __eq__(self, other):
#         if self.isAtom and other.isAtom:
#             return self._var == other._var
#         if isinstance(self, Application) and isinstance(other, Application):
#             return self._sub == other._sub and self._obj == other._obj
#         if isinstance(self, Abstraction) and isinstance(other, Abstraction):
#             return self._head == other._head and self._body == other._body
#
#     @property
#     def isBetaRedex(self):
#         """checks whether the term is a beta-redex"""
#         return self.isApplication and self._sub.isAbstraction
#
#     @property
#     def redexes(self):
#         """determiness all beta-redexes in the term"""
#         if self.isAtom:
#             return []
#         if self.isAbstraction:
#             return self._body.redexes
#         # self is Application
#         temp = [self] if self.isBetaRedex else []
#         temp += self._sub.redexes + self._obj.redexes
#         return temp
#
#     @property
#     def _vars(self):
#         """
#         returns
#         -------
#             the dictionary stuctured as follows
#                 dict[Var, dict[['free' | 'bound'], int]]
#             Here, keys of the external dictionary are the variables that
#             are occurred in 'self', and values of the internal dictionaries
#             relate respectively to the numbers of free and bound occurrences
#             of the variables.
#         """
#         if self.isAtom:
#             return {self._var: {"free": 1, "bound": 0}}
#         if self.isApplication:
#             vars, auxvars = dict(self._sub._vars), self._obj._vars
#             for var in auxvars:
#                 try:
#                     for key in {"free", "bound"}:
#                         vars[var][key] += self._obj._vars[var][key]
#                 except KeyError:
#                     vars[var] = dict(self._obj._vars[var])
#             return vars
#         # self is Abstraction
#         vars = dict(self._body._vars)
#         try:
#             vars[self._head]["bound"] += vars[self._head]["free"]
#             vars[self._head]["free"] = 0
#         except KeyError:
#             pass
#         return vars
#
#     @property
#     def verticesNumber(self):
#         """return the number of nodes in the tree representing the lambda term"""
#         if self.isAtom:
#             return 1
#         elif self.isApplication:
#             return 1 + self._sub.verticesNumber + self._obj.verticesNumber
#         else:  # self is Abstraction
#             return 1 + self._body.verticesNumber
#
#     def normalize(self, strategy):
#         """
#         :param strategy: OneStepStrategy
#         :return tuple of the normal form of the term and number of steps of betta reduction
#         """
#         term = self._updateBoundVariables()
#         count = 0
#         while term.redexes != []:
#             term = term._betaConversion(strategy)
#             count += 1
#             if term.verticesNumber > 7000 or count > 400:
#                 return (self, float("inf"))
#         return (term, count)
#
#     def _betaConversion(self, strategy):
#         """
#         :param strategy: OneStepStrategy
#         :return term with redex eliminated using the given strategy
#         """
#         index = strategy.redexIndex(self)
#         subterm = self.subterm(index)
#         reducedTerm = subterm._removeOuterRedex()
#         return self.setSubterm(index, reducedTerm)
#
#     def subterm(self, index: int):
#         """
#         By representing the term as a tree, a subtree is returned, which is also a lambda term.
#         The vertex of this subtree has a given index in the topological sorting of the vertices of the original term.
#         :param index - subterm index
#         :return: subterm: Term
#         """
#         if index == 1:
#             return self
#
#         if self.isAtom:
#             ValueError("index value is incorrect")
#         elif self.isApplication:
#             if self._sub.verticesNumber + 1 >= index:
#                 return self._sub.subterm(index - 1)
#             else:
#                 return self._obj.subterm(index - self._sub.verticesNumber - 1)
#         else:  # self is Abstraction
#             return self._body.subterm(index - 1)
#
#     def setSubterm(self, index: int, term):
#         """
#         By representing the term as a tree, a subtree is set, which is also a lambda term.
#         The vertex of this subtree has a given index in the topological sorting of the vertices of the original term.
#         :param index - subterm index
#         :param term - λ-term to which the subterm will be replaced
#         :return: updated λ-term
#         """
#         if index == 1:
#             return term
#
#         if self.isAtom:
#             ValueError("index value is incorrect")
#         elif self.isApplication:
#             if self._sub.verticesNumber + 1 >= index:
#                 return Application(self._sub.setSubterm(index - 1, term), self._obj)
#             else:
#                 return Application(
#                     self._sub,
#                     self._obj.setSubterm(index - self._sub.verticesNumber - 1, term),
#                 )
#         else:  # self is Abstraction
#             return Abstraction(self._head, self._body.setSubterm(index - 1, term))
#
#     def _updateBoundVariables(self):
#         """return λ-term with updated bound variables"""
#         if self.isAtom:
#             return self
#         elif self.isApplication:
#             return Application(
#                 self._sub._updateBoundVariables(), self._obj._updateBoundVariables()
#             )
#         else:  # self is Abstraction
#             newVar = Var()
#             return Abstraction(
#                 newVar,
#                 self._body._replaceVariable(
#                     self._head, Atom(newVar)
#                 )._updateBoundVariables(),
#             )
#
#     def _removeOuterRedex(self):
#         """apply the betta conversion to the lambda term, removing the outer betta redex"""
#         if self.isBetaRedex:
#             head = self._sub._head
#             body = self._sub._body
#             return body._replaceVariable(head, self._obj)
#         else:
#             return self
#
#     def _replaceVariable(self, var: Var, term):
#         """return λ-term with replaced variable"""
#         if self.isAtom:
#             return term if self._var == var else self
#         elif self.isApplication:
#             return Application(
#                 self._sub._replaceVariable(var, term),
#                 self._obj._replaceVariable(var, term),
#             )
#         else:  # self is Abstraction
#             return Abstraction(self._head, self._body._replaceVariable(var, term))
#
#
# class Atom(Term):
#     def __init__(self, x: Var):
#         if isinstance(x, Var):
#             self._var = x
#         else:
#             raise TypeError("a variable is waiting")
#
#
# class Application(Term):
#     def __init__(self, X: Term, Y: Term):
#         if isinstance(X, Term) and isinstance(Y, Term):
#             self._sub = X
#             self._obj = Y
#         else:
#             raise TypeError("a term is waiting")
#
#
# class Abstraction(Term):
#     def __init__(self, x: Var, X: Term):
#         if isinstance(x, Var):
#             if isinstance(X, Term):
#                 self._head = x
#                 self._body = X
#             else:
#                 raise TypeError("a term is waiting")
#         else:
#             raise TypeError("a variable is waiting")
#
#
# """## Strategy
#
# """
#
# from abc import ABC, abstractmethod
# from typing import List
#
#
# class OneStepStrategy(ABC):
#     @abstractmethod
#     def redexIndex(self, term: Term, initIndex=0) -> int:
#         """
#         :return: index of the vertex of a subterm that has an outer redex.
#                 The index of a vertex is the index of this vertex in the topological sort of the tree vertices.
#                 Indexing starts at 1.
#         """
#
#
# class LeftmostOutermostStrategy(OneStepStrategy):
#     def redexIndex(self, term: Term, initIndex=0) -> int:
#         if term.isAtom or len(term.redexes) == 0:
#             ValueError("the term does not contain a redex")
#         elif term.isApplication:
#             if term.isBetaRedex:
#                 return initIndex + 1
#             elif len(term._sub.redexes) != 0:
#                 return self.redexIndex(term._sub, initIndex + 1)
#             else:
#                 return self.redexIndex(
#                     term._obj, initIndex + term._sub.verticesNumber + 1
#                 )
#         else:  # self is Abstraction
#             return self.redexIndex(term._body, initIndex + 1)
#
#
# class LeftmostInnermostStrategy(OneStepStrategy):
#     def redexIndex(self, term: Term, initIndex=0) -> int:
#         if term.isAtom or len(term.redexes) == 0:
#             ValueError("the term does not contain a redex")
#         elif term.isApplication:
#             if len(term._sub.redexes) != 0:
#                 return self.redexIndex(term._sub, initIndex + 1)
#             elif len(term._obj.redexes) != 0:
#                 return self.redexIndex(
#                     term._obj, initIndex + term._sub.verticesNumber + 1
#                 )
#             else:
#                 return initIndex + 1
#         else:  # self is Abstraction
#             return self.redexIndex(term._body, initIndex + 1)
#
#
# class RightmostInnermostStrategy(OneStepStrategy):
#     def redexIndex(self, term: Term, initIndex=0) -> int:
#         if term.isAtom or len(term.redexes) == 0:
#             ValueError("the term does not contain a redex")
#         elif term.isApplication:
#             if len(term._obj.redexes) != 0:
#                 return self.redexIndex(
#                     term._obj, initIndex + term._sub.verticesNumber + 1
#                 )
#             elif len(term._sub.redexes) != 0:
#                 return self.redexIndex(term._sub, initIndex + 1)
#             else:
#                 return initIndex + 1
#         else:  # self is Abstraction
#             return self.redexIndex(term._body, initIndex + 1)
#
#
# class RightmostOutermostStrategy(OneStepStrategy):
#     def redexIndex(self, term: Term, initIndex=0) -> int:
#         if term.isAtom or len(term.redexes) == 0:
#             ValueError("the term does not contain a redex")
#         elif term.isApplication:
#             if term.isBetaRedex:
#                 return initIndex + 1
#             elif len(term._obj.redexes) != 0:
#                 return self.redexIndex(
#                     term._obj, initIndex + term._sub.verticesNumber + 1
#                 )
#             else:
#                 return self.redexIndex(term._sub, initIndex + 1)
#         else:  # self is Abstraction
#             return self.redexIndex(term._body, initIndex + 1)
#
#
# class RandomStrategy(OneStepStrategy):
#     def redexIndex(self, term: Term, initIndex=0) -> int:
#         redexes = term.redexes
#         if term.isAtom or len(redexes) == 0:
#             ValueError("the term does not contain a redex")
#         elif term.isApplication:
#             index = random.randint(0, len(redexes) - 1)
#             if term.isBetaRedex and index == 0:
#                 return initIndex + 1
#             elif len(term._sub.redexes) >= index and len(term._sub.redexes) != 0:
#                 return self.redexIndex(term._sub, initIndex + 1)
#             else:
#                 return self.redexIndex(
#                     term._obj, initIndex + term._sub.verticesNumber + 1
#                 )
#         else:  # self is Abstraction
#             return self.redexIndex(term._body, initIndex + 1)
#
#
# class MixedStrategy(OneStepStrategy):
#     def __init__(self, strategies: List[OneStepStrategy], probability_vector: list):
#         self.strategies = strategies
#         self.probability_vector = probability_vector
#
#     def redexIndex(self, term: Term) -> int:
#         p = random.random()
#         index = 0
#         index_prob = self.probability_vector[0]
#         while p > index_prob:
#             index += 1
#             index_prob += self.probability_vector[index]
#
#         return self.strategies[index].redexIndex(term)
#
#
# """## Generating lambda terms"""
#
# import seaborn as sns
# import numpy as np
# from typing import List
# import sys
#
# sys.setrecursionlimit(40000)
#
#
# def genTerm(p: float, uplimit: int, vars: List[Var] = [], trigger_by_application=False):
#     if uplimit < 1:
#         return None
#
#     pVar = (1 - p * p) / 2
#     pAbs = pVar + p * p
#
#     rand = random.random()
#
#     if rand < pVar and len(vars) > 0:
#         index = random.randint(0, len(vars) - 1)
#         return Atom(vars[index])
#     elif rand < pAbs:
#         head = Var()
#         new_vars = vars + [head]
#         body = genTerm(p, uplimit - 1, new_vars)
#         return Abstraction(head, body) if body else None
#     else:
#         sub = genTerm(p, uplimit - 1, vars, trigger_by_application=True)
#         obj = genTerm(p, uplimit - 1, vars)
#         if sub and obj and sub.verticesNumber + obj.verticesNumber <= uplimit:
#             return Application(sub, obj)
#         else:
#             return None
#
#
# UPLIMIT = 60
# DOWNLIMIT = 50
#
# LAMBDA_TERM_COUNT = 100
#
# RANDOM_AVERAGE_COUNT = 20
#
#
# def filterTerms(term):
#     return term and DOWNLIMIT < term.verticesNumber < UPLIMIT
#
#
# def flatten(t):
#     return [item for sublist in t for item in sublist]
#
#
# terms = flatten(
#     [
#         list(filter(filterTerms, [genTerm(p, UPLIMIT) for i in range(7000)]))
#         for p in np.arange(0.49, 0.51, 0.02)
#     ]
# )
#
# countVertices = list(map(lambda term: term.verticesNumber, terms))
# countRedexes = list(map(lambda term: len(term.redexes), terms))
#
# print(f"number of lambda terms {len(terms)}")
# print("number of vertices= {}".format(countVertices))
# print("number of redexes= {}".format(countRedexes))
#
# stepsLO = list(map(lambda term: term.normalize(LeftmostOutermostStrategy())[1], terms))
# print("number of steps to normalize using LO strategy= {}".format(stepsLO))
#
# terms_with_normal_form = []
# stepsLO_temp = []
# for i, term in enumerate(terms):
#     if stepsLO[i] != float("inf"):
#         terms_with_normal_form.append(term)
#         stepsLO_temp.append(stepsLO[i])
# terms = terms_with_normal_form[:LAMBDA_TERM_COUNT]
# stepsLO = stepsLO_temp[:LAMBDA_TERM_COUNT]
#
# print(f"number of terms with normal form {len(terms)}")
# assert len(terms) == LAMBDA_TERM_COUNT
#
# print("number of steps to normalize using LO strategy= {}".format(stepsLO))
#
# stepsRI = [term.normalize(RightmostInnermostStrategy())[1] for term in terms]
# print("number of steps to normalize using RI strategy= {}".format(stepsRI))
#
# stepsRand = [
#     sum([term.normalize(RandomStrategy())[1] for i in range(RANDOM_AVERAGE_COUNT)])
#     / RANDOM_AVERAGE_COUNT
#     for term in terms
# ]
# print("number of steps to normalize using Random strategy= {}".format(stepsRand))
#
# import matplotlib.pyplot as plt
# from fitter import Fitter, get_common_distributions
#
#
# def draw_hist(data, file_name: str):
#     steps = [x for x in data if x != float("inf")]
#
#     distributions = get_common_distributions()
#     distributions.remove("expon")
#     distributions.remove("cauchy")
#     f = Fitter(steps, distributions=distributions)
#     f.fit()
#     summary = f.summary()
#     distribution = f.get_best(method="sumsquare_error")
#
#     print("==============")
#     print(f"number of not normalized terms: {len(data) - len(steps)}")
#     print(summary)
#     print(distribution)
#     print(f'Norm distribution: {f.fitted_param["norm"]}')
#     print("==============")
#     plt.xlabel("Number of reduction steps")
#     plt.savefig(file_name, dpi=300)
#     plt.show()
#
#     f_ln = Fitter([np.log(step) for step in steps], distributions=distributions)
#     f_ln.fit()
#     mu, sigma = f_ln.fitted_param["norm"]
#     print(
#         f"Log Norm distribution params: ({mu}, {sigma}); expected value = {np.e ** (mu + (sigma ** 2) / 2)}"
#     )
#
#
# draw_hist(stepsLO, "./hist-LO.png")
# draw_hist(stepsRI, "./hist-RI.png")
# draw_hist(stepsRand, "./hist-Rand.png")
#
# results = []
# values = list(range(0, 101, 5))
# for p_lo in values:
#     p = (p_lo / 100, 1 - p_lo / 100)
#     steps = [
#         sum(
#             [
#                 term.normalize(
#                     MixedStrategy(
#                         [LeftmostOutermostStrategy(), RightmostInnermostStrategy()], p
#                     )
#                 )[1]
#                 for i in range(RANDOM_AVERAGE_COUNT)
#             ]
#         )
#         / RANDOM_AVERAGE_COUNT
#         for term in terms
#     ]
#     steps = list(filter(lambda x: x != float("inf"), steps))
#
#     distributions = get_common_distributions()
#     distributions.remove("expon")
#     f_ln = Fitter([np.log(step) for step in steps], distributions=distributions)
#     f_ln.fit()
#
#     mu, sigma = f_ln.fitted_param["norm"]
#     result = np.e ** (mu + (sigma**2) / 2)
#     results.append(result)
#
# plt.plot(values, results)
# plt.xlabel("p_LO")
# plt.ylabel("Expected number of steps")
# plt.savefig("LOvsRI.png", dpi=300)
#
# from deap import tools, algorithms, base, creator
#
#
# def eaSimpleWithElitism(
#     population,
#     toolbox,
#     cxpb,
#     mutpb,
#     ngen,
#     stats=None,
#     halloffame=None,
#     verbose=__debug__,
# ):
#     """This algorithm is similar to DEAP eaSimple() algorithm, with the modification that
#     halloffame is used to implement an elitism mechanism. The individuals contained in the
#     halloffame are directly injected into the next generation and are not subject to the
#     genetic operators of selection, crossover and mutation.
#     """
#     logbook = tools.Logbook()
#     logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
#
#     # Evaluate the individuals with an invalid fitness
#     invalid_ind = [ind for ind in population if not ind.fitness.valid]
#     fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
#     for ind, fit in zip(invalid_ind, fitnesses):
#         ind.fitness.values = fit
#
#     if halloffame is None:
#         raise ValueError("halloffame parameter must not be empty!")
#
#     halloffame.update(population)
#     hof_size = len(halloffame.items) if halloffame.items else 0
#
#     record = stats.compile(population) if stats else {}
#     logbook.record(gen=0, nevals=len(invalid_ind), **record)
#     if verbose:
#         print(logbook.stream)
#
#     # Begin the generational process
#     for gen in range(1, ngen + 1):
#
#         # Select the next generation individuals
#         offspring = toolbox.select(population, len(population) - hof_size)
#
#         # Vary the pool of individuals
#         offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
#
#         # Evaluate the individuals with an invalid fitness
#         invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
#         fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
#         for ind, fit in zip(invalid_ind, fitnesses):
#             ind.fitness.values = fit
#
#         # add the best back to population:
#         offspring.extend(halloffame.items)
#
#         # Update the hall of fame with the generated individuals
#         halloffame.update(offspring)
#
#         # Replace the current population by the offspring
#         population[:] = offspring
#
#         # Append the current generation statistics to the logbook
#         record = stats.compile(population) if stats else {}
#         logbook.record(gen=gen, nevals=len(invalid_ind), **record)
#         if verbose:
#             print(logbook.stream)
#
#     return population, logbook
#
#
# STRATEGIES = [
#     LeftmostOutermostStrategy(),
#     RightmostInnermostStrategy(),
#     LeftmostInnermostStrategy(),
#     RightmostOutermostStrategy(),
#     RandomStrategy(),
# ]
#
# BOUNDS_LOW = [0.1, 0.0, 0.0, 0.0]
# BOUNDS_HIGH = [
#     1.0,
#     1.0,
#     1.0,
#     1.0,
# ]
#
# NUM_OF_PARAMS = len(BOUNDS_HIGH)
#
# # Genetic Algorithm constants:
# POPULATION_SIZE = 20
# P_CROSSOVER = 0.9  # probability for crossover
# P_MUTATION = 0.3  # probability for mutating an individual
# MAX_GENERATIONS = 7
# HALL_OF_FAME_SIZE = 4
# CROWDING_FACTOR = 10  # crowding factor for crossover and mutation
#
# toolbox = base.Toolbox()
#
# # define a single objective, maximizing fitness strategy:
# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# # create the Individual class based on list:
# creator.create("Individual", list, fitness=creator.FitnessMin)
#
# # define the hyperparameter attributes individually:
# for i in range(NUM_OF_PARAMS):
#     # "hyperparameter_0", "hyperparameter_1", ...
#     toolbox.register(
#         "hyperparameter_" + str(i), random.uniform, BOUNDS_LOW[i], BOUNDS_HIGH[i]
#     )
#
# # create a tuple containing an attribute generator for each param searched:
# hyperparameters = ()
# for i in range(NUM_OF_PARAMS):
#     hyperparameters = hyperparameters + (
#         toolbox.__getattribute__("hyperparameter_" + str(i)),
#     )
#
#
# # create the individual operator to fill up an Individual instance:
# def individual_creator() -> creator.Individual:
#     indv = [0 for _ in range(NUM_OF_PARAMS)]
#     for i in range(NUM_OF_PARAMS):
#         indv[i] = random.uniform(BOUNDS_LOW[i], 1 - sum(indv))
#     return creator.Individual(indv)
#
#
# # create the population operator to generate a list of individuals:
# toolbox.register("populationCreator", tools.initRepeat, list, individual_creator)
#
#
# # fitness calculation
# def fitness(individual):
#     p = individual.copy()
#     p.append(max(0, 1 - sum(individual)))
#     steps = [
#         sum(
#             [
#                 term.normalize(MixedStrategy(STRATEGIES, p))[1]
#                 for i in range(RANDOM_AVERAGE_COUNT)
#             ]
#         )
#         / RANDOM_AVERAGE_COUNT
#         for term in terms
#     ]
#     steps = list(filter(lambda x: x != float("inf"), steps))
#
#     distributions = get_common_distributions()
#     distributions.remove("expon")
#     f_ln = Fitter([np.log(step) for step in steps], distributions=distributions)
#     f_ln.fit()
#
#     mu, sigma = f_ln.fitted_param["norm"]
#     result = np.e ** (mu + (sigma**2) / 2)
#     print(
#         "expected number of steps to normalize using Mixed strategy= {}".format(result)
#     )
#     if 1 - sum(individual) < 0:
#         result += 100 * (sum(individual) - 1)
#     return (result,)
#
#
# toolbox.register("evaluate", fitness)
#
# # genetic operators:
# toolbox.register("select", tools.selTournament, tournsize=2)
# toolbox.register(
#     "mate",
#     tools.cxSimulatedBinaryBounded,
#     low=BOUNDS_LOW,
#     up=BOUNDS_HIGH,
#     eta=CROWDING_FACTOR,
# )
# toolbox.register(
#     "mutate",
#     tools.mutPolynomialBounded,
#     low=BOUNDS_LOW,
#     up=BOUNDS_HIGH,
#     eta=CROWDING_FACTOR,
#     indpb=1.0 / NUM_OF_PARAMS,
# )
#
# # create initial population (generation 0):
# population = toolbox.populationCreator(n=POPULATION_SIZE)
#
# # prepare the statistics object:
# stats = tools.Statistics(lambda ind: ind.fitness.values)
# stats.register("min", np.min)
# stats.register("avg", np.mean)
#
# # define the hall-of-fame object:
# hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
#
# # perform the Genetic Algorithm flow with hof feature added:
# population, logbook = eaSimpleWithElitism(
#     population,
#     toolbox,
#     cxpb=P_CROSSOVER,
#     mutpb=P_MUTATION,
#     ngen=MAX_GENERATIONS,
#     stats=stats,
#     halloffame=hof,
#     verbose=True,
# )
#
# # print best solution found:
# print("- Best solution is: ")
# print("p = ", hof.items[0])
#
# # extract statistics:
# maxFitnessValues, meanFitnessValues = logbook.select("min", "avg")
# # plot statistics:
# sns.set_style("whitegrid")
# plt.plot(maxFitnessValues, color="red")
# plt.plot(meanFitnessValues, color="green")
# plt.xlabel("Generation")
# plt.ylabel("Max / Average Fitness")
# plt.title("Max and Average fitness over Generations")
# plt.show()
#
# p = hof.items[0]
# p.append(max(0, 1 - sum(p)))
# MixedStrategySteps = [
#     sum(
#         [
#             term.normalize(MixedStrategy(STRATEGIES, p))[1]
#             for i in range(RANDOM_AVERAGE_COUNT)
#         ]
#     )
#     / RANDOM_AVERAGE_COUNT
#     for term in terms
# ]
# draw_hist(MixedStrategySteps, "./hist-Mixed.png")
#
# p = [0.98, 0.005, 0.005, 0.005, 0.005]
# steps = [
#     sum(
#         [
#             term.normalize(MixedStrategy(STRATEGIES, p))[1]
#             for i in range(RANDOM_AVERAGE_COUNT)
#         ]
#     )
#     / RANDOM_AVERAGE_COUNT
#     for term in terms
# ]
# draw_hist(steps, "./hist-Mixed-custom.png")
#
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
#
#
# def draw_2d_distribution(ax, x, y, xlabel, ylabel):
#     ax.scatter(x, y, color="blue")
#     ax.set(
#         title="Distribution of " + xlabel.lower() + "-" + ylabel.lower(),
#         xlabel=xlabel,
#         ylabel=ylabel,
#     )
#     ax.legend(prop={"size": 10})
#
#
# def draw_2d_strategy_distribution(ax, x, y, xlabel, ylabel):
#     x = list(map(lambda v: -1 if v == float("inf") else v, x))
#     y = list(map(lambda v: -1 if v == float("inf") else v, y))
#     greater = list(
#         zip(
#             *list(
#                 filter(lambda z: z[0] >= z[1] and z[0] != -1 and z[1] != -1, zip(x, y))
#             )
#         )
#     )
#     less = list(
#         zip(
#             *list(
#                 filter(lambda z: z[0] < z[1] and z[0] != -1 and z[1] != -1, zip(x, y))
#             )
#         )
#     )
#     inf_x = list(zip(*list(filter(lambda z: z[0] == -1 and z[1] != -1, zip(x, y)))))
#     inf_y = list(zip(*list(filter(lambda z: z[0] != -1 and z[1] == -1, zip(x, y)))))
#     inf_xy = list(zip(*list(filter(lambda z: z[0] == -1 and z[1] == -1, zip(x, y)))))
#     ax.scatter(
#         greater[0], greater[1], color="blue", label="{} <= {}".format(ylabel, xlabel)
#     )
#     ax.scatter(less[0], less[1], color="red", label="{} < {}".format(xlabel, ylabel))
#     if inf_x:
#         ax.scatter(
#             inf_x[0],
#             inf_x[1],
#             color="lime",
#             label="{} doesn't normalize".format(xlabel),
#         )
#     if inf_y:
#         ax.scatter(
#             inf_y[0],
#             inf_y[1],
#             color="yellow",
#             label="{} dooesn't normalize".format(ylabel),
#         )
#     if inf_xy:
#         ax.scatter(
#             inf_xy[0],
#             inf_xy[1],
#             color="orange",
#             label="Both strategies don't normalize",
#         )
#     ax.set(
#         title="Distribution of " + xlabel.lower() + "-" + ylabel.lower(),
#         xlabel=xlabel,
#         ylabel=ylabel,
#     )
#     ax.legend(prop={"size": 10})
#
#
# figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
# ax0, ax1, ax2, ax3 = axes.flatten()
# draw_2d_distribution(
#     ax0, countVertices, countRedexes, "Vertices number", "Redexes number"
# )
# draw_2d_strategy_distribution(
#     ax1, stepsLO, stepsRI, "Leftmost outermost strategy", "Rightmost inermost strategy"
# )
# draw_2d_strategy_distribution(
#     ax2, stepsLO, stepsRand, "Leftmost outermost strategy", "Random strategy"
# )
# draw_2d_strategy_distribution(
#     ax3, stepsRI, stepsRand, "Rightmost inermost strategy", "Random strategy"
# )
# plt.show()
#
# import pandas as pd
#
#
# def draw_plot(x, y, z, q, labels, colors, file_name=""):
#     x = list(map(lambda v: -1 if v == float("inf") else v, x))
#     y = list(map(lambda v: -1 if v == float("inf") else v, y))
#     z = list(map(lambda v: -1 if v == float("inf") else v, z))
#     q = list(map(lambda v: -1 if v == float("inf") else v, q))
#
#     data = pd.DataFrame(zip(x, y, z, q), columns=labels)
#     ax0 = data.plot(figsize=(20, 10), kind="bar", color=colors)
#     ax0.set(
#         title="Distribution of number of reduction steps for each term",
#         xlabel="Term index",
#         ylabel="Number of reduction steps",
#     )
#     ax0.legend(prop={"size": 10})
#     plt.savefig(file_name, dpi=300)
#     return ax0
#
#
# colors = ["lime", "blue", "red", "orange"]
# labels = ["Leftmost outermost", "Rightmost inermost", "Uniformly random", "Mixed"]
# n = 20
# draw_plot(
#     stepsLO[10:n],
#     stepsRI[10:n],
#     stepsRand[10:n],
#     MixedStrategySteps[10:n],
#     labels,
#     colors,
#     "reduction-strategies.png",
# )
#
# terms_dict = {i: [] for i in range(DOWNLIMIT, UPLIMIT)}
#
# for i, term in enumerate(terms):
#     terms_dict[term.verticesNumber].append(
#         (term, {"LO": stepsLO[i], "RI": stepsRI[i], "Rand": stepsRand[i]})
#     )
#
# average_term_data = dict()
# for verticesNumber, data in terms_dict.items():
#     data_without_inf = [
#         d[1]
#         for d in data
#         if d[1]["LO"] != float("inf")
#         and d[1]["RI"] != float("inf")
#         and d[1]["Rand"] != float("inf")
#     ]
#     avgLO, avgRI, avgRand = 0, 0, 0
#     for d in data_without_inf:
#         avgLO += d["LO"]
#         avgRI += d["RI"]
#         avgRand += d["Rand"]
#
#     count = len(data_without_inf)
#     if count != 0:
#         average_term_data[verticesNumber] = {
#             "LO": avgLO / count,
#             "RI": avgRI / count,
#             "Rand": avgRand / count,
#         }
#
# plt.figure(figsize=(20, 15))
# ax = plt.gca()
# ax.plot(
#     list(average_term_data.keys()),
#     [data["LO"] for i, data in average_term_data.items()],
#     color="blue",
#     label="LO",
# )
# ax.plot(
#     list(average_term_data.keys()),
#     [data["RI"] for i, data in average_term_data.items()],
#     color="lime",
#     label="RI",
# )
# ax.plot(
#     list(average_term_data.keys()),
#     [data["Rand"] for i, data in average_term_data.items()],
#     color="red",
#     label="Rand",
# )
# ax.set(title="Distribution", xlabel="Vertices number", ylabel="Strategy steps number")
# ax.legend(prop={"size": 10})
#
# plt.show()
#
# """## Tests"""
#
# x, y, z = Var(), Var(), Var()
# X, Z = Atom(x), Atom(z)
# XXX = Application(Application(X, X), X)
# XZ = Application(X, Z)
# T = Application(Abstraction(x, XXX), Abstraction(x, Application(Abstraction(y, Z), XZ)))
#
# print(T)
# for var, item in T._vars.items():
#     print("\t{}".format(var), end=": ")
#     print(item)
#
# x, y, z, w, v = Var(), Var(), Var(), Var(), Var()
# # (λx.(λy.( ((λz.(y z)) ((λw.w) x)) v )))
# lambdaTerm = Abstraction(
#     x,
#     Abstraction(
#         y,
#         Application(
#             Application(
#                 Abstraction(z, Application(Atom(y), Atom(z))),
#                 Application(Abstraction(w, Atom(w)), Atom(w)),
#             ),
#             Atom(v),
#         ),
#     ),
# )
#
#
# def testTerm():
#     assert len(lambdaTerm.redexes) == 2
#     assert lambdaTerm.verticesNumber == 13
#
#     subterm = Application(Atom(y), Atom(z))
#     assert lambdaTerm.subterm(1) == lambdaTerm
#     assert lambdaTerm.subterm(6) == subterm
#     assert lambdaTerm.setSubterm(1, subterm) == subterm
#
#     assert (
#         lambdaTerm._updateBoundVariables().verticesNumber == lambdaTerm.verticesNumber
#     )
#     assert len(lambdaTerm._updateBoundVariables().redexes) == len(lambdaTerm.redexes)
#
#     strategy = LeftmostOutermostStrategy()
#     assert len(lambdaTerm._betaConversion(strategy).redexes) == 1
#     assert lambdaTerm._betaConversion(strategy).verticesNumber == 10
#
#     assert len(lambdaTerm.normalize(strategy)[0].redexes) == 0
#     assert lambdaTerm.normalize(strategy)[1] == 2
#
#
# def testStrategy():
#     strategy = LeftmostOutermostStrategy()
#     assert strategy.redexIndex(lambdaTerm) == 4
#
#
# testTerm()
# testStrategy()
#
# stepsLOWithoutTail = [x for x in stepsLO if x < 150]
# draw_hist(stepsLOWithoutTail, "./hist-LO-without-tail.png")
