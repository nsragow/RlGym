from q_learning import QLearner
from dqn import DQNLearner
import numpy as np
from keras.utils import to_categorical

def evaluate_actor(actor, max_steps=1000, evaluations=100, render=False, wait_on_enter=False):
    score = 0

    for _ in range(evaluations):
        actor.reset()
        step_count = 0
        if max_steps == -1:
            step_count = -2
        reward = 0
        done = False
        if render:
            actor.env.render()
        while (step_count < max_steps) and not done:
            step_reward, step_done = actor.step(learning=False)
            done = step_done
            reward += step_reward
            if step_count >= 0:
                step_count += 1
            if render:
                if wait_on_enter:
                    input("next step")
                actor.env.render()

        score += reward
    return score


def optimal_actions(actor):
    """
    Only intended for Discrete observation space
    """
    optims = []
    for i in range(actor.env.observation_space.n):
        actions = actor.Qnn.predict(np.expand_dims(to_categorical(i,16), axis=0))[0]

        optims.append((np.argmax(actions), max(actions)))

    return optims
if __name__ == "__main__":
    import pickle

    file = open("./pickles/selective_dqn.pkl", 'rb')
    actor = pickle.load(file)
    print(f'score: {evaluate_actor(actor, render=True, max_steps=-1)}')
