from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras import optimizers
from gym.spaces import Box, Discrete


def make_dnn(env):
    adam = optimizers.Adam(learning_rate=0.0003)

    input_dim = get_dimension(env.observation_space, False)
    output_dim = get_dimension(env.action_space, True)

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