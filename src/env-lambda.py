import gym
import numpy as np

from src.calculus.generation import genTerm
from src.calculus.strategy import LeftmostOutermostStrategy, RightmostInnermostStrategy, LeftmostInnermostStrategy, \
    RightmostOutermostStrategy, RandomStrategy
from src.experiment import flatten, filterTerms, UPLIMIT

STRATEGIES = [LeftmostOutermostStrategy(), RightmostInnermostStrategy(), LeftmostInnermostStrategy(),
              RightmostOutermostStrategy(), RandomStrategy()]


def is_reduced(term):
    return False


def have_seen_before(history, term):
    return False


def reward_function(history, term):
    if is_reduced(term):
        return 1.0
    if have_seen_before(history, term):
        return -0.25

    return -0.04


def f(term):
    return len(term)


def is_done(term, cumulative_reward):
    if is_reduced(term):
        return True
    if cumulative_reward <= -0.5 * f(term):
        return True
    return False


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.terms = self.get_new_terms()

        self.strategies = STRATEGIES
        self.term = None
        self.number_of_term = None
        self.action_space = None
        # Example for using image as input:
        self.observation_space = None
        self.number_of_term = -1
        self.reset()

    def step(self, strategy):
        index = self.strategies[strategy].redexIndex(self.term)

        self.term = self.term._betaConversion_index(index)
        obs = self.term
        reward = -1
        done = self.term.redexes == []
        return obs, reward, done, {}

    def reset(self, **kwargs):
        self.number_of_term += 1
        self.term = self.terms[self.number_of_term]._updateBoundVariables()
        return self.term

    def render(self, mode='human', close=False):
        print(self.term)

    def get_new_terms(self):
        return flatten(
            [list(filter(filterTerms, [genTerm(p, UPLIMIT) for i in range(800)])) for p in np.arange(0.37, 0.44, 0.01)])


if __name__ == '__main__':
    env = CustomEnv()

    # strategy = LeftmostOutermostStrategy()
    # strategy = RightmostInnermostStrategy()
    # strategy =  RandomStrategy()

    obs = env.reset()
    done = False
    for i in range(2000):

        action = strategy.redexIndex(obs)
        print(action)
        if done or not action:
            env.reset()
        obs, rewards, done, info = env.step(action)

        env.render()
