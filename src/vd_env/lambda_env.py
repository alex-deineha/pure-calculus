import gym
import sys
import numpy as np

sys.path.append('../')
from calculus.generation import gen_lambda_terms
from calculus.strategy import LeftmostOutermostStrategy, RightmostOutermostStrategy


class LambdaEnv(gym.Env):
    def __init__(self, strategies, max_step_term=500, count_terms=100):
        self.lambda_terms = None
        self.term_idx = 0
        self.state = None
        self._max_step_term = max_step_term
        self._count_terms = count_terms
        self.strategies = strategies
        self.reset()

    def step(self, action):
        selected_strategy = self.strategies[action]
        is_done_norm = self.lambda_terms[self.term_idx].normalize_step(selected_strategy)
        self.state[self.term_idx].append([action, -1 if is_done_norm else 0])

        # check is it possible to normalize the term more
        # by calculating total reward
        total_term_reward = self._max_step_term + sum([row[1] for row in self.state[self.term_idx]])
        print(self.term_idx, total_term_reward)

        # check is it done with all terms normalization:
        done = ((not is_done_norm) or total_term_reward == 0) \
               and ((self._count_terms - 1) == self.term_idx)

        if (not is_done_norm) or (total_term_reward == 0):
            self.lambda_terms[self.term_idx].restart_normalization()

        if total_term_reward == 0 or (not is_done_norm):
            self.term_idx = self.term_idx if ((self._count_terms - 1) == self.term_idx) \
                else self.term_idx + 1

        debug = None
        reward = -1 if is_done_norm else 0
        return self.state, reward, done, debug

    def reset(self):
        self.term_idx = 0
        self.lambda_terms = gen_lambda_terms(count=self._count_terms)
        self.state = {}
        for i in range(self._count_terms):
            self.state[i] = []
            self.lambda_terms[i].restart_normalization()
        return self.state

    def render(self, mode="ascii"):
        for i in range(self.term_idx + 1):
            total_term_reward = self._max_step_term + sum([row[1] for row in self.state[i]])
            print('Term No_{}: total reward:{}'.format(i, total_term_reward))


def get_simple_env(max_step_term=500, count_terms=100):
    strategies = [LeftmostOutermostStrategy(), RightmostOutermostStrategy()]
    return LambdaEnv(strategies=strategies,
                     max_step_term=max_step_term,
                     count_terms=count_terms)


if __name__ == '__main__':
    lambda_env = get_simple_env(count_terms=100, max_step_term=30)
    is_not_done = True
    while is_not_done:
        _, _, done, _ = lambda_env.step(0)
        is_not_done = not done

    lambda_env.render()
