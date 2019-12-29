import gym
from random import sample
from numpy import random, argmax
import numpy as np
from nn_maker import make_dnn


class SelectiveDQNLearner:
    def __init__(self, env, good_episode, epsilon=.3, N=1000, sample_size=32, lr=.2, discount=.96):
        """
        Used with Acrobot-v1

        good episode is fed the reward upon termination
        """
        self.good_episode = good_episode
        self.env = env
        self._state = None
        self.epsilon = epsilon
        self.bad_memory = []
        self.good_memory = []
        self.temp_memory = []
        self.memory_size = N
        self.sample_size = sample_size
        self.lr = lr
        self.discount = discount
        self.silent = True

        self.Qnn = make_dnn(env)
        self.reset()

    def reset(self):
        if self.good_episode(self):  # see if last memory is good
            for trans in self.temp_memory:
                self._remember(trans, dumping=True)
        self.temp_memory = []

        self.state = self.env.reset()

    def _get_action(self, learning):
        exploring = random.uniform() < self.epsilon
        if learning and exploring:
            return self.env.action_space.sample()
        else:  # exploiting
            return self.optimal_action(self.state)

    def step(self, learning):
        action = self._get_action(learning)
        obs, reward, done, info = self.env.step(action)

        prev_state = self.state
        self.state = obs
        if learning:
            transition = (prev_state, action, reward, obs, done)
            self._remember(transition)
            self._update_q()
        if done:
            self.reset()
        return reward, done

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

    def _update_q(self):
        X, y = self._get_x_y(self.sample_size * 2, self.sample_size)
        # ®self.Qnn.optimizer.lr.assign(self.lr)
        self.Qnn.train_on_batch(X, y)

    def silent_level(self):
        if self.silent:
            return 0
        else:
            return 2

    def optimal_action(self, state):
        actions = self.Qnn.predict(np.expand_dims(state, axis=0))[0]
        assert max(actions) != float('nan') and max(actions) != float('inf')
        return argmax(actions)

    def _remember(self, transition, dumping=False):
        if not dumping:
            self.temp_memory.append(transition)
            self.bad_memory.append(transition)
        else:
            self.good_memory.append(transition)
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


def good_episode(learner):
    max_steps = learner.env._max_episode_steps
    steps = learner.env._elapsed_steps

    return steps is not None and steps != 0 and steps < max_steps


if __name__ == '__main__':
    import pickle
    from gym.envs.toy_text.frozen_lake import generate_random_map

    env_name = 'Acrobot-v1'

    env = gym.make(env_name)

    actor = SelectiveDQNLearner(env, good_episode)
    file = open("pickles/selective_dqn.pkl", 'wb+')
    pickle.dump(actor, file)

