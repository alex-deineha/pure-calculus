import gym
import sys
import numpy as np

from tqdm import tqdm

sys.path.append('../')
from calculus.generation import gen_lambda_terms
from calculus.strategy import LeftmostOutermostStrategy, RightmostInnermostStrategy


class LambdaEnv(gym.Env):
    def __init__(self, strategies, lambda_terms=None, max_step_term=500, count_terms=100):
        self.lambda_terms = None
        self.term_idx = 0
        self.state = None
        self._max_step_term = max_step_term
        self._count_terms = count_terms
        self.strategies = strategies
        if lambda_terms is None:
            self.reset()
        else:
            self.reset_(lambda_terms)

    def step(self, action):
        if self.term_idx >= self._count_terms:
            return self.state, 0, True, None

        selected_strategy = self.strategies[action]
        is_done_norm = self.lambda_terms[self.term_idx].normalize_step(selected_strategy)
        self.state[self.term_idx].append([action, -1 if is_done_norm else 0])

        # check is it possible to normalize the term more
        # by calculating total reward
        total_term_reward = self._max_step_term + sum([row[1] for row in self.state[self.term_idx]])
        # print(self.term_idx, total_term_reward)

        # check is it done with current term normalization:
        done = (not is_done_norm) or total_term_reward == 0

        debug = None
        reward = -1 if is_done_norm else 0
        return self.state, reward, done, debug

    def next_term(self):
        """
        Select the next term idx if possible
        """
        # self.term_idx = self.term_idx + 1 if (self._count_terms - 1) > self.term_idx else self.term_idx
        self.term_idx += 1

    def is_has_next_term(self):
        """
        return: bool True if it's possible to select the next term
        """
        # return (self._count_terms - 1) > self.term_idx
        return self._count_terms > self.term_idx

    def get_current_term_idx(self):
        """
        return: int index of selected term
        """
        return self.term_idx

    def reset_(self, lambda_terms):
        self.term_idx = 0
        if lambda_terms is None:
            self.lambda_terms = gen_lambda_terms(count=self._count_terms)
        else:
            self.lambda_terms = lambda_terms
        self.state = {}
        for i in range(self._count_terms):
            self.state[i] = []
            self.lambda_terms[i].restart_normalization()
        return self.state

    def reset(self):
        return self.reset_(None)

    def reset_soft(self):
        self.term_idx = 0
        self.state = {}
        for i in range(self._count_terms):
            self.state[i] = []
            self.lambda_terms[i].restart_normalization()
        return self.state

    def render(self, mode="ascii"):
        for i in range(self.term_idx):
            total_term_reward = self._max_step_term + sum([row[1] for row in self.state[i]])
            print('Term No_{}: total reward:{}'.format(i, total_term_reward))


def get_simple_env(max_step_term=500, count_terms=100):
    strategies = [LeftmostOutermostStrategy(), RightmostInnermostStrategy()]
    return LambdaEnv(strategies=strategies,
                     max_step_term=max_step_term,
                     count_terms=count_terms)


if __name__ == '__main__':
    lambda_env = get_simple_env(count_terms=300, max_step_term=80)
    is_not_done = True
    for _ in tqdm(range(300)):
        if lambda_env.is_has_next_term():
            while True:
                _, _, done, _ = lambda_env.step(0)
                if done:
                    break
        lambda_env.next_term()

    lambda_env.render()
