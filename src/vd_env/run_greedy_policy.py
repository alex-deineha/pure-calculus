import random
import sys

sys.path.append("../")

from vd_env.lambda_env import get_simple_env
from vd_env.greedy_policy import greedy_policy
import matplotlib.pyplot as plt
import numpy as np


def run_greedy_policy(environment, exploration=10, max_term_reward=30):
    state = environment.reset_soft()
    rewards = []

    while environment.is_has_next_term():
        while True:
            action = greedy_policy(environment, state, explore=exploration)
            state, _, done, _ = environment.step(action)
            if done:
                break
        environment.next_term()

    for term_history in state.values():
        rewards.append(max_term_reward - (len(term_history) - 1))

    # environment.close()
    return environment, rewards


if __name__ == "__main__":
    # seed = 0
    # random.seed(seed)

    count_terms = 50
    max_reward = 30
    env = get_simple_env(count_terms=count_terms, max_step_term=max_reward)
    env, rewards_ = run_greedy_policy(env, exploration=10, max_term_reward=max_reward)
    env.render()

    plt.plot(rewards_)
    plt.title("Greedy Policy")
    plt.xlabel("Trials")
    plt.ylabel("Reward")
    plt.show()

    cum_rewards = np.cumsum(rewards_)
    plt.plot(cum_rewards)
    plt.title("Greedy Policy")
    plt.xlabel("Trials")
    plt.ylabel("Cum Reward")
    plt.show()

    for i in range(len(cum_rewards)):
        cum_rewards[i] /= i + 1
    plt.plot(cum_rewards)
    plt.title("Greedy Policy")
    plt.xlabel("Trials")
    plt.ylabel("Avg Reward")
    plt.show()
