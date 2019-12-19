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
import random


class QLearner:
    def __init__(self, env, lr=.3, gamma=.8):
        # Current implementation assumes Frozen-lake env
        self.env = env
        self.state = env.reset()
        self.state_range = len(env.env.P.keys())
        self.action_range = len(env.env.P[0].keys())
        # Make Q matrix of shape states,actions
        self.Q = np.zeroes((self.state_range,self.action_range))
        self.lr = lr
        self.gamma = gamma  # also know as discount rate

    def step(self, epsilon=.2, learning=True):
        """
        Take step in simulation and update Q values

        if learning == False the QLearner will always exploit and will not update
        Q values

        Return:
            reward - reward received from step
            done - True if simulation ended
        """
        exploring = random.uniform() > epsilon
        if learning and exploring:
            action = random.randint(0, self.action_range - 1)
        else:  # exploiting
            action = self.optimal_action()
        obs, reward, done, info = self.env.step(action)
        prev_state = self.state
        self.state = obs
        if learning:
            self.update_q(prev_state, action, reward, self.lr)
        if done:
            self.state = self.env.reset()
        return reward, done

    def update_q(self, previous_state, action_taken, reward, lr):
        """
        Q_last_s_a = Q_last_s_a + learning_rate * (reward + gamma(max(Q_cur_s)) - Q_last_s_a)
        """
        prev_q = self.Q[previous_state][action_taken]
        self.Q[previous_state][action_taken] += lr * (reward + self.gamma * max(self.Q[self.state]) - prev_q)

    def optimal_action(self):
        action = -1
        highest_quality = -1

        for pos_action in range(self.action_range):
            ret_quality = self.Q[self.state][pos_action]
            if ret_quality > highest_quality:
                action = pos_action
                highest_quality = ret_quality

        return action
