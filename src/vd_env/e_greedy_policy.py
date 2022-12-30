import random
import numpy as np


def e_greedy_policy(env, state, explore=10, count_strategies=2, epsilon=.1, deterministic=True):
    term_trials = env.get_current_term_idx()
    total_explore_trials = count_strategies * explore

    # exploration
    if term_trials < total_explore_trials:
        return term_trials % count_strategies

    # random strategy with some epsilon probability
    if random.random() < epsilon:
        return random.randint(0, count_strategies - 1)

    # exploitation
    avg_rewards = np.zeros(count_strategies)
    for i in range(total_explore_trials):
        avg_rewards[state[i][0][0]] -= len(state[i])

    if deterministic:
        best_strategy = np.argmax(avg_rewards)
    else:
        probab = avg_rewards / np.sum(avg_rewards)
        probab = 1. - probab
        probab = probab / np.sum(probab)
        best_strategy = np.random.choice(np.arange(count_strategies), p=probab)

    return best_strategy
