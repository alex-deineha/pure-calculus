import sys
from tqdm import tqdm

sys.path.append("../")
from vd_env.lambda_env import get_simple_env
from vd_env.e_greedy_policy import e_greedy_policy, e_greedy_action_based_policy
import matplotlib.pyplot as plt
import numpy as np


def run_e_greedy_policy(
    environment,
    exploration=10,
    max_term_reward=30,
    epsilon=0.1,
    deterministic=True,
    is_action_based=False,
    return_steps=False,
):
    state = environment.reset_soft()

    while environment.is_has_next_term():
        while True:
            if is_action_based:
                action = e_greedy_action_based_policy(
                    environment,
                    state,
                    explore_steps=exploration,
                    count_strategies=len(environment.strategies),
                    epsilon=epsilon,
                    deterministic=deterministic,
                )
            else:
                action = e_greedy_policy(
                    environment,
                    state,
                    explore=exploration,
                    count_strategies=len(environment.strategies),
                    epsilon=epsilon,
                    deterministic=deterministic,
                )
            state, _, done, _ = environment.step(action)
            if done:
                break
        environment.next_term()

    steps, rewards = [], []
    if return_steps:
        steps = [(len(term_history) - 1) for term_history in state.values()]
    else:
        rewards = [
            (max_term_reward - (len(term_history) - 1))
            for term_history in state.values()
        ]

    # environment.close()
    return environment, steps if return_steps else rewards


def __test_run__():
    # seed = 0
    # random.seed(seed)

    count_terms = 100
    max_reward = 30
    epsilon_ = 0.1
    exploration = 100

    env = get_simple_env(count_terms=count_terms, max_step_term=max_reward)

    env, rewards_det = run_e_greedy_policy(
        env,
        exploration=15,
        max_term_reward=max_reward,
        deterministic=True,
        is_action_based=True,
    )
    print("Deterministic: ---------------------->")
    env.render()
    print("\n\n")

    env.reset_soft()
    env, rewards_non_det = run_e_greedy_policy(
        env,
        exploration=exploration,
        max_term_reward=max_reward,
        deterministic=False,
        is_action_based=True,
    )
    print("Non-Deterministic: ---------------------->")
    env.render()
    print("\n\n")

    rewards_det = [max_reward - rw for rw in rewards_det]
    rewards_non_det = [max_reward - rw for rw in rewards_non_det]

    cum_rewards_det = np.cumsum(rewards_det)
    for i in range(len(cum_rewards_det)):
        cum_rewards_det[i] /= i + 1

    cum_rewards_non_det = np.cumsum(rewards_non_det)
    for i in range(len(cum_rewards_non_det)):
        cum_rewards_non_det[i] /= i + 1

    plt.plot(range(1, len(cum_rewards_det) + 1), cum_rewards_det, label="Deterministic")
    plt.plot(
        range(1, len(cum_rewards_non_det) + 1),
        cum_rewards_non_det,
        label="not_Deterministic",
    )
    plt.legend(loc="upper right")
    plt.title("E Greedy Policy Steps Count")
    plt.xlabel("Trials")
    plt.ylabel("Avg Steps")
    plt.show()


if __name__ == "__main__":
    __test_run__()
