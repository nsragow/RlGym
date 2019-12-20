def evaluate_actor(actor, max_steps=1000, evaluations=100, render=False):
    score = 0
    for _ in range(evaluations):
        actor.reset()
        step_count = 0
        reward = 0
        done = False
        if render:
            actor.env.render()
        while (step_count < max_steps) and not done:
            step_reward, step_done = actor.step(learning=False)
            done = step_done
            reward += step_reward
            step_count += 1
            if render:
                input("next step")
                actor.env.render()

        score += reward
    return score


if __name__ == "__main__":
    import pickle
    from q_learning import QLearner
    file = open("./pickles/q_learner.pkl", 'rb')
    actor = pickle.load(file)
    print(f'score: {evaluate_actor(actor, render=True)}')
