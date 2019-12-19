lr = 0

if __name__ == '__main__':
    import sys
    import pickle
    import logging
    from q_learning import QLearner
    import math
    from actor_evaluator import evaluate_actor
    file = open(sys.argv[1], 'rb')
    actor = pickle.load(file)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    if len(sys.argv) > 2:
        try:
            sprint_len = 1000
            steps = int(sys.argv[2])
            sprints = steps/sprint_len
            frac, whole_sprints = math.modf(sprints)
            for _ in range(int(whole_sprints)):
                actor.learn(sprint_len, lr=lr)
                logging.info(f'Score: {evaluate_actor(actor)}')
            actor.learn(int(frac*sprint_len), lr=lr)
            logging.info(f'Final: {evaluate_actor(actor)}')
        except KeyboardInterrupt:
            logging.info('Saving...')
    else:
        actor.learn()
    file = open(sys.argv[1], 'wb')
    pickle.dump(actor, file)
