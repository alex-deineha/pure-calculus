import random
import numpy as np


def e_greedy_policy(
    env, state, explore=10, count_strategies=2, epsilon=0.1, deterministic=True
):
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
        probab = 1.0 - probab
        probab = probab / np.sum(probab)
        best_strategy = np.random.choice(np.arange(count_strategies), p=probab)

    return best_strategy


def e_greedy_action_based_policy(
    env, state, explore_steps=100, count_strategies=2, epsilon=0.1, deterministic=True
):
    # exploration
    trial_steps = 0
    for term_story in state.values():
        trial_steps += len(term_story)
        if trial_steps > explore_steps:
            break

    if trial_steps <= explore_steps:
        return trial_steps % count_strategies
        # return random.randint(0, count_strategies - 1)    # which one?

    # random strategy with some epsilon probability
    if random.random() < epsilon:
        return random.randint(0, count_strategies - 1)

    # exploitation
    avg_rewards = np.zeros(count_strategies)
    steps_per_strategy = np.zeros(count_strategies)
    for term_story in state.values():
        for action, reward in term_story:
            avg_rewards[action] += reward
            steps_per_strategy[action] += 1
    avg_rewards = avg_rewards / steps_per_strategy

    if deterministic:
        best_strategy = np.argmax(avg_rewards)
    else:
        probab = avg_rewards / np.sum(avg_rewards)
        probab = 1.0 - probab
        probab = probab / np.sum(probab)
        best_strategy = np.random.choice(np.arange(count_strategies), p=probab)

    return best_strategy
