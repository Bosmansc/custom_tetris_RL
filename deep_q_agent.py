import warnings

warnings.filterwarnings("ignore")

import random
from time import sleep
import time
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
import json
import pandas as pd

deprecation._PRINT_DEPRECATION_WARNINGS = False

## use pip install --upgrade --force-reinstall  git+https://github.com/Bosmansc/tetris_openai.git
## not pip install  pip install keras-rl2, this is not compatible with the custom tetris environment

from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint, FileLogger

from engine import TetrisEngine
from custom_logging import plot_custom_results
from custom_logging import plot_metrics


def timer(func):
    def f(*args, **kwargs):
        before = time.time()
        rv = func(*args, **kwargs)
        after = time.time()
        print('function ' + func.__name__ + ' took ' + str(round(after - before, 2)) + ' seconds')
        return rv

    return f


def build_callbacks():
    checkpoint_weights_filename = 'model_checkpoints/dqn_weights_.h5f'
    log_filename = 'dqn_log.json'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=500)]
    callbacks += [FileLogger(log_filename, interval=100)]
    return callbacks


class Agent:
    def __init__(self):
        # Initializes a Tetris playing field of width 10 and height 20.
        self.env = TetrisEngine()
        self.agent = None

        # hyperparameters:
        self.LEARNING_RATE = 0.1    # default = 0.001
        self.GAMMA = 0.9  # gamma defines penalty for future reward
        self.BATCH_SIZE = 100  # default = 32 -> too small for tetris?
        self.EPSILON_START = 1
        self.EPSILON_END = 0.1
        self.TARGET_MODEL_UPDATE = 500    # default is 10000
        self.EPSILON_TEST = 0
        self.SEQUENTIAL_MEMORY_LIMIT = 50000

    @timer
    def train(self, nb_steps=1000, visualise=True):
        # Resets the environment
        self.env.reset_environment()

        # init Neural network
        actions = 6  # there are 6 discrete actions
        model = self.build_model_conv(actions)
        model.summary()

        # define callbacks
        callbacks = build_callbacks()

        # init and fit the agent
        dqn = self.build_agent(model, actions, nb_steps)
        dqn.compile(Adam(lr=self.LEARNING_RATE), metrics=['mae', 'mse'])
        history_training = dqn.fit(self.env,
                                   nb_steps=nb_steps,
                                   callbacks=callbacks,
                                   visualize=visualise)

        # plot the results
        plot_custom_results(self.env.df_info, history_training, mode='training')

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
        plot_custom_results(self.env.df_info, history_test, mode='test')

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

    def build_agent(self, model, actions, nb_steps):
        """
        GAMMA:
        REWARD = r1 + gamma*r2 + gamma^2*r3 + gamma^3*r4 ...
        -> gamma defines penalty for future reward
        In general, most algorithms learn faster when they don't have to look too far into the future.
        So, it sometimes helps the performance to set gamma relatively low.
        for many problems a gamma of 0.9 or 0.95 is fine

        LAMBDA:
        The lambda parameter determines how much you bootstrap on earlier learned value versus using
        the current Monte Carlo roll-out. This implies a trade-off between more bias (low lambda)
        and more variance (high lambda).
        A general rule of thumb is to use a lambda equal to 0.9.
        However, it might be good just to try a few settings (e.g., 0, 0.5, 0.8, 0.9, 0.95 and 1.0)
        """
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),  # takes current best action with prob (1 - epsilon)
                                      attr='eps',  # decay epsilon (=exploration) per agent step
                                      value_max=self.EPSILON_START,  # start value of epsilon (default =1)
                                      value_min=self.EPSILON_END,  # last value of epsilon (default =0
                                      value_test=self.EPSILON_TEST,
                                      nb_steps=nb_steps)
        memory = SequentialMemory(limit=self.SEQUENTIAL_MEMORY_LIMIT, window_length=1)
        build_agent = DQNAgent(model=model, memory=memory, policy=policy, gamma=self.GAMMA, batch_size=self.BATCH_SIZE,
                               nb_actions=actions, nb_steps_warmup=100, target_model_update=self.TARGET_MODEL_UPDATE)
        return build_agent


if __name__ == '__main__':
    agent = Agent()

    # train the agent
    agent.train(nb_steps=100, visualise=False)

    # test the agent
    agent.test(nb_episodes=1)

    # save the agent
    # agent.save('only_square_10000.model')

    # plot the logs
    plot_metrics(save_fig=False)
