def evaluate_policy(env, policy, max_steps=1000, evaluations=100, render=False):
    score = 0
    for _ in range(evaluations):
        obs = env.reset()
        step_count = 0
        reward = 0
        done = False
        if render:
            env.render()
        while (step_count < max_steps) and not done:
            action = policy[obs]
            obs, additional_reward, is_done, info = env.step(action)
            done = is_done
            reward += additional_reward
            step_count += 1
            if render:
                input(f"action is {action}")
                env.render()
        score += reward
    return score


if __name__ == "__main__":
    import gym

    policy_genetic = [0, 3, 2, 2, 0, 0, 0, 1, 3, 1, 1, 1, 3, 3, 1, 1]

    policy_value_iteration = [0, 3, 0, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]

    env = gym.make("FrozenLake-v0")

    print(evaluate_policy(env, policy_value_iteration, evaluations=100, render=False))
