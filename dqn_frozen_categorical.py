import gym
from random import sample

from keras import optimizers, Sequential
from keras.layers import Dense
from numpy import random, argmax
import numpy as np
from gym.spaces import Box, Discrete
import logging
from keras.utils import to_categorical

import sys
import pickle
import logging
from actor_evaluator import evaluate_actor, optimal_actions
import math

from actor_evaluator import evaluate_actor

class DQNLearnerFrozenLake:
    def __init__(self, env, epsilon=.3, N=100, sample_size=32, lr=.2, discount=.96, obs_preprocess=lambda obs: obs):
        self.preprocess = obs_preprocess
        self.env = env
        self._state = None
        self.reset()
        self.epsilon = epsilon
        self.Qnn = make_dnn_frozen(env)
        self.bad_memory = []
        self.good_memory = []
        self.temp_memory = []
        self.memory_size = N
        self.sample_size = sample_size
        self.lr = lr
        self.discount = discount
        self.silent = True

    def reset(self, good=False):
        self.state = self.preprocess(self.env.reset())
        if good:
            for trans in self.temp_memory:
                self._remember(trans, bad=False, dumping=True)
        self.temp_memory = []

    def step(self, learning):
        exploring = random.uniform() < self.epsilon
        if learning and exploring:
            action = self.env.action_space.sample()
        else:  # exploiting
            action = self.optimal_action()
        obs, reward, done, info = self.env.step(action)
        obs = self.preprocess(obs)
        prev_state = self.state
        self.state = obs
        if learning:
            transition = (prev_state, action, reward, self.state, done)
            self._remember(transition)
            self._update_q()
        if done:
            self.reset(reward > 0)
        return reward, done

    def _update_q(self):
        X, y = self._get_x_y(self.sample_size * 2, self.sample_size)
        # ®self.Qnn.optimizer.lr.assign(self.lr)
        error = self.Qnn.train_on_batch(X, y)
        outprint = ''
        for i in range(len(self.Qnn.metrics_names)):
            outprint += f'{self.Qnn.metrics_names[i]}: {error[i]} '
        logging.debug(outprint)

    def _get_x_y(self, good, bad):
        good_replay = sample(self.good_memory, min(good, len(self.good_memory)))
        bad_replay = sample(self.bad_memory, min(bad, len(self.bad_memory)))
        replay = np.stack(good_replay + bad_replay, axis=0)
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])
        Q = self.Qnn.predict(states)
        Q_new = self.Qnn.predict(new_states)

        replay_size = len(replay)
        X = np.empty((replay_size, len(states[0])))
        y = np.empty((replay_size, len(Q[0])))
        for i in range(replay_size):
            state_r, action_r, reward_r, new_state_r, done_r = replay[i]

            target = Q[i]
            target[action_r] = reward_r
            # If we're done the utility is simply the reward of executing action a in
            # state s, otherwise we add the expected maximum future reward as well
            if not done_r:
                target[action_r] += self.discount * np.amax(Q_new[i])

            X[i] = state_r
            y[i] = target
        return X, y

    def silent_level(self):
        if self.silent:
            return 0
        else:
            return 2

    def optimal_action(self, state=None):
        if state is None:
            state = self.state
        actions = self.Qnn.predict(np.expand_dims(state, axis=0))[0]
        assert max(actions) != float('nan') and max(actions) != float('inf')
        return argmax(actions)

    def _remember(self, transition, bad=True, dumping=False):
        if not dumping:
            self.temp_memory.append(transition)
        if not bad:
            self.good_memory.append(transition)
        else:
            self.bad_memory.append(transition)
        while len(self.good_memory) > self.memory_size * 2:
            self.bad_memory.append(self.good_memory.pop(0))
        while len(self.bad_memory) > self.memory_size:
            self.bad_memory.pop(0)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        if type(value) is np.ndarray:
            self._state = value
        elif type(value) is int or type(value) is np.int64:
            arr = np.ndarray((1,))
            assert len(arr) == 1
            arr[0] = value
            self._state = arr

        else:
            raise ValueError(f'got value {value} of type {type(value)}')

    def learn(self, steps, static_epsilon=None):
        i = 0
        self.reset()
        while i < steps:
            epsilon = (steps - i) / steps
            self.epsilon = epsilon
            if static_epsilon is not None:
                self.epsilon = static_epsilon
            i += 1
            self.step(True)


def preproc(obs):
    return to_categorical(obs, num_classes=16)

if __name__ == '__main__':
    import pickle

    env_name = 'FrozenLake-v0'
    env = gym.make(env_name)

    actor = DQNLearnerFrozenLake(env, obs_preprocess=preproc)
    file = open("pickles/dqn_f.pkl", 'wb+')
    pickle.dump(actor, file)



def make_dnn_frozen(env):
    adam = optimizers.Adam(learning_rate=0.0003)

    input_dim = get_dimension(env.observation_space, False)
    output_dim = get_dimension(env.action_space, True)
    input_dim = 16
    model = Sequential()
    # model.add(Embedding(1000, 64, input_length=10))
    model.add(Dense(units=10, activation='relu', input_dim=input_dim))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=output_dim, activation='linear'))

    model.compile(loss='mean_squared_error',
                  optimizer=adam,
                  metrics=['mse', 'accuracy'])
    return model


def get_dimension(space, action):
    if type(space) is Box:
        return space.shape[0]
    elif type(space) is Discrete:
        if len(space.shape) > 0:
            raise ValueError(f'unexpected val {len(space.shape)}')
        elif action:
            return space.n
        else:
            return 1
    else:
        raise ValueError(f'Unexpected type {type(space)}')