def evaluate_policy(env, policy, max_steps=1000, evaluations = 100):
    score = 0
    for _ in range(evaluations):
        obs = env.reset()
        step_count = 0
        reward = 0
        done = False
        while (step_count < max_steps) and not done:
            action = policy[obs]
            obs, additional_reward, is_done, info = env.step(action)
            done = is_done
            reward += additional_reward
            step_count += 1

        score += reward
    return score
