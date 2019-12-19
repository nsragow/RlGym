if __name__ == '__main__':
    import sys
    import pickle
    file = open(sys.argv[1], 'rb')
    actor = pickle.load(file)
    if len(sys.argv) > 2:
        actor.learn(int(sys.argv[2]))
    else:
        actor.learn()
    file = open(sys.argv[1], 'wb')
    pickle.dump(actor, file)
