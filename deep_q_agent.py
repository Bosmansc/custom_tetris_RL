import random

from time import sleep
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

## use pip install --upgrade --force-reinstall  git+https://github.com/Bosmansc/tetris_openai.git
## not pip install  pip install keras-rl2, this is not compatible with the custom tetris environment

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory


class Agent():
    def __init__(self):
        # Initializes a Tetris playing field of width 10 and height 20.
        self.env = TetrisEngine()
        self.agent = None

    def train(self, nb_steps=1000, visualise=True):
        # Resets the environment
        self.env.reset_environment()

        # init Neural network
        actions = 6  # there are 6 discrete actions
        model = self.build_model_conv(actions)
        # model.summary()

        # init and fit the agent
        dqn = self.build_agent(model, actions)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        history_training = dqn.fit(self.env, nb_steps=nb_steps, visualize=visualise)

        # plot the results
        self.env.plot_results(history_training, 'training')

        # save trained agent
        self.agent = dqn

        return dqn

    def test(self, nb_episodes=10, visualize=True):
        self.env.reset_environment()
        history_test = self.agent.test(self.env, nb_episodes=nb_episodes, visualize=visualize
                                       , nb_max_episode_steps=300
                                       )

        print(np.mean(history_test.history['episode_reward']))

        # plot the results
        self.env.plot_results(history_test, 'test')

    def save(self):
        self.agent.save_weights('models/dqn_model.model', overwrite=False)

    @staticmethod
    def build_model_conv(actions):
        # Network defined by the Deepmind paper
        model = tf.keras.models.Sequential()

        model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_uniform',
                         kernel_constraint=max_norm(4), input_shape=(1, 16, 6)))
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
        # policy = GreedyQPolicy() ## hyperparm, GreedyQPolicy is used in paper: https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2008-118.pdf
        policy = BoltzmannQPolicy()
        memory = SequentialMemory(limit=50000, window_length=1)
        build_agent = DQNAgent(model=model, memory=memory, policy=policy,
                               nb_actions=actions, nb_steps_warmup=100, target_model_update=1e-2)
        return build_agent


if __name__ == '__main__':
    agent = Agent()

    # train the agent
    agent.train(nb_steps=1000, visualise=True)

    # test the agent
    agent.test(nb_episodes=5)

    # save the agent
    #agent.save()
