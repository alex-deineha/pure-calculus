import random
import numpy as np


def e_greedy_policy(env, state, explore=10, count_strategies=2, epsilon=.1):
    term_trials = env.get_current_term_idx()
    total_explore_trials = count_strategies * explore

    # exploration
    if term_trials < total_explore_trials:
        return term_trials % count_strategies

    # random strategy with some epsilon probability
    if random.random() < epsilon:
        return random.randint(0, count_strategies - 1)

    # exploitation
    avg_rewards = [0 for _ in range(count_strategies)]
    for i in range(total_explore_trials):
        avg_rewards[state[i][0][0]] -= len(state[i])

    best_strategy = np.argmax(avg_rewards)
    return best_strategy
