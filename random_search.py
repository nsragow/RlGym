import numpy as np
import gym
from policy_evaluator import evaluate_policy


def get_random_policy(env):
    return np.random.choice(env.action_space.n, env.observation_space.n)


def random_policy_list(env,random_policy_count = 1000):
    random_policies = list()
    for _ in range(random_policy_count):
        random_policies.append(get_random_policy)
    return random_policies


if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    policies = random_policy_list(env)
    best_performer = None
    best_performance = -1
    for policy in policies:
        score = evaluate_policy(env,policy)
        if score > best_performance:
            best_performance = score
            best_performer = policy
    print(score)


