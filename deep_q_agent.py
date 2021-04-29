import warnings
warnings.filterwarnings("ignore")

import random
from time import sleep
from time import time
from engine import TetrisEngine
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib import pyplot
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

## use pip install --upgrade --force-reinstall  git+https://github.com/Bosmansc/tetris_openai.git
## not pip install  pip install keras-rl2, this is not compatible with the custom tetris environment

from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import SequentialMemory


def timer(func):
    def f(*args, **kwargs):
        before = time()
        rv = func(*args, **kwargs)
        after = time()
        print('function ' + func.__name__ + ' took ' + str(round(after - before, 2)) + ' seconds')
        return rv
    return f


class Agent:
    def __init__(self):
        # Initializes a Tetris playing field of width 10 and height 20.
        self.env = TetrisEngine()
        self.agent = None

    @timer
    def train(self, nb_steps=1000, visualise=True):
        # Resets the environment
        self.env.reset_environment()

        # init Neural network
        actions = 6  # there are 6 discrete actions
        model = self.build_model_conv(actions)
        model.summary()

        # init and fit the agent
        dqn = self.build_agent(model, actions)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        history_training = dqn.fit(self.env, nb_steps=nb_steps, visualize=visualise)

        # plot the results
        self.env.plot_results(history_training, 'training')

        # save trained agent
        self.agent = dqn

        return dqn

    @timer
    def test(self, nb_episodes=10, visualize=True):
        self.env.reset_environment()
        history_test = self.agent.test(self.env, nb_episodes=nb_episodes, visualize=visualize,
                                       nb_max_episode_steps=300)

        print(np.mean(history_test.history['episode_reward']))

        # plot the results
        self.env.plot_results(history_test, 'test')

    def save(self, name):
        self.agent.save_weights(f'models/{name}.model', overwrite=False)

    def build_model_conv(self, actions):
        # Network defined by the Deepmind paper
        model = tf.keras.models.Sequential()

        model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform',
                         kernel_constraint=max_norm(4), input_shape=(1, self.env.height, self.env.width)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # model.add(MaxPooling2D(pool_size=(2,2)))

        # end of convolutional layers, start of 'hidden' dense layers
        model.add(Flatten())
        model.add(Dense(128, kernel_initializer='he_uniform', kernel_constraint=max_norm(3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Final dense layer
        model.add(Dense(actions, activation='linear'))

        return model

    @staticmethod
    def build_agent(model, actions):
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=0,
                                      value_test=0, nb_steps=8000)
        memory = SequentialMemory(limit=50000, window_length=1)
        build_agent = DQNAgent(model=model, memory=memory, policy=policy, gamma=.8, batch_size=32,
                               nb_actions=actions, nb_steps_warmup=100, target_model_update=250)
        return build_agent


if __name__ == '__main__':
    agent = Agent()

    # train the agent
    agent.train(nb_steps=10000, visualise=False)

    # test the agent
    agent.test(nb_episodes=5)

    # save the agent
    agent.save('two_20000')
