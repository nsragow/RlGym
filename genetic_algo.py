import numpy as np
import gym
from policy_evaluator import evaluate_policy
from random_search import random_policy_list
from random import uniform


def genetic(population_size=1000, cutoff=100, iterations=100):
    env = gym.make("FrozenLake-v0")
    population = random_policy_list(env, population_size)
    population_with_score = []
    for i in range(iterations):
        for policy in population:
            score = evaluate_policy(env, policy)
            population_with_score.append((policy, score))

        sorted_pop = sorted(population_with_score, key=lambda tup: tup[1], reverse=True)
        parents = sorted_pop[:cutoff]
        children = []
        for _ in range(population_size):
            children.append(baby(parents[np.random.choice(cutoff)][0], parents[np.random.choice(cutoff)][0],env.action_space.n))
        population = children
    return sorted_pop[0]


def baby(policy1, policy2, n):
    random_var = uniform(0, 1)
    new_policy = []
    for i in range(len(policy1)):

        if random_var < .2:
            new_policy.append(np.random.choice(n))
        elif random_var < .6:
            new_policy.append(policy1[i])
        else:
            new_policy.append(policy2[i])
    return new_policy


if __name__ == "__main__":
    print(genetic())









