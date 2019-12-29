lr = .05

if __name__ == '__main__':
    import sys
    import pickle
    import logging

    import math

    from actor_evaluator import evaluate_actor
    from dqn import DQNLearner

    file = open(sys.argv[1], 'rb')
    actor = pickle.load(file)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    if len(sys.argv) > 2:
        try:
            score = 0.0
            sprint_len = 500
            steps = int(sys.argv[2])
            sprints = steps/sprint_len
            frac, whole_sprints = math.modf(sprints)

            for i in range(int(whole_sprints)):

                epsilon = ((whole_sprints - i) / whole_sprints) * .3
                epsilon = .5  # could use either
                lr = (1 - math.sqrt(i)/math.sqrt(whole_sprints)) * .2
                actor.learn(sprint_len)
                score = evaluate_actor(actor, max_steps=-1, evaluations=10)
                logging.info(f'Score: {score}               lr: {lr} epsilon: {epsilon}  i: {i}/{whole_sprints}')

        except KeyboardInterrupt:
            logging.info('Saving...')
    else:
        actor.learn()
    file = open(sys.argv[1], 'wb')
    pickle.dump(actor, file)