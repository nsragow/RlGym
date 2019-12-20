"""
Noah Sragow
This is a q learning implementation of the TD(0) algorithm.

1) Create Q matrix
2) Take action (explore vs exploit)
3) Update Q values with TD(0) formula
4) Go back to step #2 while further learning is desired
5)
"""

import numpy as np
import numpy.random as random


class QLearner:
    def __init__(self, env, lr=.3, gamma=.98, epsilon=.3):
        # Current implementation assumes Frozen-lake env
        self.env = env
        self.state = None
        self.reset()
        self.state_range = len(env.env.P.keys())
        self.action_range = len(env.env.P[0].keys())
        # Make Q matrix of shape states,actions
        self.Q = np.zeros((self.state_range, self.action_range))
        self.lr = lr
        self.gamma = gamma  # also know as discount rate
        self.epsilon = epsilon

    def step(self, learning=True):
        """
        Take step in simulation and update Q values

        if learning == False the QLearner will always exploit and will not update
        Q values

        Return:
            reward - reward received from step
            done - True if simulation ended
        """
        exploring = random.uniform() < self.epsilon
        if learning and exploring:
            action = random.randint(0, self.action_range)
        else:  # exploiting
            action = self.optimal_action()
        obs, reward, done, info = self.env.step(action)
        prev_state = self.state
        self.state = obs
        if learning:
            self.update_q(prev_state, action, reward, self.lr)
        if done:
            self.reset()
        return reward, done

    def update_q(self, previous_state, action_taken, reward, lr):
        """
        Q_last_s_a = Q_last_s_a + learning_rate * (reward + gamma(max(Q_cur_s)) - Q_last_s_a)
        """
        prev_q = self.Q[previous_state][action_taken]
        self.Q[previous_state][action_taken] += lr * (reward + self.gamma * max(self.Q[self.state]) - prev_q)

    def optimal_action(self, state=None):
        if state is None:
            state = self.state
        action = -1
        highest_quality = -1

        for pos_action in range(self.action_range):
            ret_quality = self.Q[state][pos_action]
            if ret_quality > highest_quality:
                action = pos_action
                highest_quality = ret_quality

        return action

    @property
    def optimal_actions(self):
        actions = list()
        for i in range(self.state_range):
            actions.append(self.optimal_action(i))
        return actions

    def learn(self, steps=1000, lr=None, epsilon=None):
        if lr is not None:
            self.lr = lr
        if epsilon is not None:
            self.epsilon = epsilon
        while steps > 0:
            steps -= 1
            self.step(learning=True)
        self.reset()

    def reset(self):
        """
        Reset to the beginning of env simulation
        """
        self.state = self.env.reset()


if __name__ == '__main__':
    import pickle
    import gym

    env = gym.make("FrozenLake-v0", is_slippery=True, map_name='4x4')
    actor = QLearner(env)
    actor.learn()
    file = open("./pickles/q_learner.pkl", 'wb+')
    pickle.dump(actor, file)