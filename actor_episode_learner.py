import sys
import pickle
import logging
from actor_evaluator import evaluate_actor, optimal_actions
import math

from actor_evaluator import evaluate_actor
from dqn import *
from selective_memory_dqn import *

filename = './pickles/selective_dqn.pkl'
file = open(filename, 'rb')
actor = pickle.load(file)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


episodes = 10000
try:
    for i in range(episodes):
        actor.reset()
        epsilon = ((episodes - i) / episodes) * .3
        # epsilon = .1
        # lr = ((episodes - i) / episodes) * .001

        done = False
        while not done:
            actor.epsilon = epsilon
            # actor.lr = lr
            reward, done = actor.step(learning=True)
            # logging.debug(f'reward {reward}')
        if i % 10 == 0:
            pass
            logging.info(f'evaluation[{int(i/10)}/{int(episodes/10)}] {evaluate_actor(actor)}')

except KeyboardInterrupt as e:
    pass
logging.info('Saving...')
file = open(filename, 'wb')
pickle.dump(actor, file)
