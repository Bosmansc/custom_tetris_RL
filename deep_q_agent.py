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


def timer(func):
    """
    method used as wrapper to time functions
    """
    def f(*args, **kwargs):
        before = time.time()
        rv = func(*args, **kwargs)
        after = time.time()
        print('function ' + func.__name__ + ' took ' + str(round(after - before, 2)) + ' seconds')
        return rv

    return f


def build_callbacks():
    """
    callbacks for the deep q agent
    """
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
        self.LEARNING_RATE = 0.001  # default = 0.001 -> higher LR is faster learning but can become unstable and local minimum
        self.GAMMA = 0.9  # gamma defines penalty for future reward
        self.BATCH_SIZE = 32  # default = 32 -> too small for tetris?
        self.EPSILON_START = 1
        self.EPSILON_END = 0.2
        self.TARGET_MODEL_UPDATE = 500  # default is 10000
        self.EPSILON_TEST = 0.1
        self.SEQUENTIAL_MEMORY_LIMIT = 50000

        # target model update in source code:
        # if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
        # -> I think that the total steps have to be multiple of target_model_update to work

    @timer
    def train(self, nb_steps=1000, visualise=True):
        """
        the training process of the deep Q agent
        """
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
                                   visualize=visualise,
                                   log_interval=self.TARGET_MODEL_UPDATE)

        # plot the results
        self._plot_custom_results(self.env.df_info, history_training, mode='training')

        # save trained agent
        self.agent = dqn

        return dqn

    @timer
    def test(self, nb_episodes=10, visualize=True):
        """
        The testing process of the deep q agent
        """
        self.env.reset_environment()
        history_test = self.agent.test(self.env, nb_episodes=nb_episodes, visualize=visualize,
                                       nb_max_episode_steps=300)

        print(np.mean(history_test.history['episode_reward']))

        # plot the results
        self._plot_custom_results(self.env.df_info, history_test, mode='test')

    def save(self, name):
        """
        saving the model weights for future use
        """
        self.agent.save_weights(f'models/{name}.model', overwrite=False)

    def build_model_conv(self, actions):
        """
        define the neural network model architecture for the deep q agent
        """
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
        building the deep q agent

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
                               nb_actions=actions, nb_steps_warmup=1000, target_model_update=self.TARGET_MODEL_UPDATE)
        return build_agent

    def _plot_custom_results(self, df, history, mode='training'):
        """
        plot custom results
        """
        # input data
        if 'new_episode' not in df:
            raise KeyError('the dataframe has to have the new_episode column to plot the results')
        df["nr_episode"] = df["new_episode"].cumsum()

        df_results = df.groupby('nr_episode', as_index=False) \
            .agg(heigt_diff_sum=('height_difference', 'sum'),
                 new_block_sum=('new_block', 'sum'),
                 nr_lines_sum=('number_of_lines', 'max'),
                 score_sum=('score', 'sum'),
                 score_avg=('score', 'mean'),
                 count_steps=('nr_episode', 'count'))

        df_results['moving_average_score'] = df_results.score_sum.expanding().mean()
        df_results['moving_average_lines'] = df_results.nr_lines_sum.expanding().mean()

        # init plot
        figure = pyplot.figure(figsize=(20, 10), dpi=80)
        figure.canvas.set_window_title(mode)

        # PLOT 1: EPISODE REWARD
        pyplot.subplot(221)

        # data (the dict keys are different for training and test)
        if mode == 'training':
            episode_key = 'nb_episode_steps'
        else:
            episode_key = 'nb_steps'

        y_1 = history.history[episode_key]
        y_2 = history.history['episode_reward']
        ind = np.arange(len(y_1))

        # bars
        width = 0.35  # the width of the bars
        pyplot.bar(ind, y_1, width, color='g', label='nb_episode_steps')
        pyplot.ylabel('nr steps per episode')
        pyplot.xlabel('episode')
        pyplot.legend(loc="upper left")

        # line
        axes2 = pyplot.twinx()
        axes2.plot(ind, y_2, color='k', label='episode_reward')
        axes2.set_ylabel('episode reward')
        pyplot.legend(loc="upper right")

        # title
        pyplot.title(mode + ': episode reward and steps per episode')

        # PLOT 2: NR OF LINES CLEARED PER EPISODE
        pyplot.subplot(222)
        x = df_results['nr_episode']
        y = df_results['nr_lines_sum']

        # plotting the points
        pyplot.plot(x, y)

        # naming the x axis
        pyplot.xlabel('episodes')
        # naming the y axis
        pyplot.ylabel('nr_of_lines')

        # title
        pyplot.title(mode + ': number of lines per episode')

        # save the plots
        timestr = time.strftime("%m%d_%H%M%S")
        pyplot.savefig("logs/img_info_" + timestr)

        # PLOT 3: MOVING AVERAGE TOTAL SCORE
        pyplot.subplot(223)
        x = df_results['nr_episode']
        y = df_results['moving_average_score']

        # plotting the points
        pyplot.plot(x, y)

        # naming the x axis
        pyplot.xlabel('episodes')
        # naming the y axis
        pyplot.ylabel('moving average total score')

        # title
        pyplot.title(mode + ': moving average total score')

        # PLOT 4: MOVING AVERAGE LINES CLEARED
        pyplot.subplot(224)
        x = df_results['nr_episode']
        y = df_results['moving_average_lines']

        # plotting the points
        pyplot.plot(x, y)

        # naming the x axis
        pyplot.xlabel('episodes')
        # naming the y axis
        pyplot.ylabel('moving average total score')

        self.LEARNING_RATE = 0.01  # default = 0.001 -> higher LR is faster learning but can become unstable and local minimum
        self.GAMMA = 0.9  # gamma defines penalty for future reward
        self.BATCH_SIZE = 100  # default = 32 -> too small for tetris?
        self.EPSILON_START = 1
        self.EPSILON_END = 0.2
        self.TARGET_MODEL_UPDATE = 1000  # default is 10000
        self.EPSILON_TEST = 0.1
        self.SEQUENTIAL_MEMORY_LIMIT = 50000

        # add subtitle with hyperparams
        subtitile = f"Epsilon start: {self.EPSILON_START}, Epsilon end: {self.EPSILON_END}, Gamma: {self.GAMMA}, LR: {self.LEARNING_RATE}, " \
                    f"target model update: {self.TARGET_MODEL_UPDATE}, Batch size: {self.BATCH_SIZE}"
        pyplot.figtext(0.01, 0.01, subtitile, fontsize=15)

        # title
        pyplot.title(mode + ': moving average nr of lines')

        # save the plots
        timestr = time.strftime("%m%d_%H%M%S")
        pyplot.savefig("logs/img_info_" + timestr)

        # show the plots
        pyplot.show()
        pyplot.close()

    def plot_metrics(self, save_fig=False):
        """
        plot the callback metrics
        """
        # plot the logs
        with open('dqn_log.json') as json_file:
            data = json.load(json_file)
        df_log = pd.DataFrame.from_dict(data)
        figure = pyplot.figure(figsize=(20, 10), dpi=80)
        for idx, col in enumerate(df_log.columns):
            self._combine_metrics(df_log, col, idx)

        # add subtitle
        subtitle = f"Epsilon start: {self.EPSILON_START}, Epsilon end: {self.EPSILON_END}, Gamma: {self.GAMMA}, LR: {self.LEARNING_RATE}, " \
                    f"target model update: {self.TARGET_MODEL_UPDATE}, Batch size: {self.BATCH_SIZE}"
        pyplot.figtext(0.01, 0.01, subtitle, fontsize=15)

        # save fig
        timestr = time.strftime("%m%d_%H%M%S")
        if save_fig:
            pyplot.savefig("logs/img_logs_" + timestr)
        pyplot.show()

    @staticmethod
    def _combine_metrics(df, key, index):
        """
        helper method for the plot_metrics function
        """
        pyplot.subplot(4, 3, index + 1)
        pyplot.subplots_adjust(hspace=0.5)

        y = df[key]
        x = df['episode']

        # plotting the points
        pyplot.plot(x, y)

        # naming the x axis
        pyplot.xlabel('episode nr')
        # naming the y axis
        pyplot.ylabel(key.replace('_', ' '))

        # title
        pyplot.title(key.replace('_', ' '))


if __name__ == '__main__':
    agent = Agent()

    # train the agent
    agent.train(nb_steps=1000, visualise=False)

    # test the agent
    agent.test(nb_episodes=10, visualize=True)

    # save the agent
    # agent.save('square_and_rect_1000000_0205.model')

    # plot the logs
    agent.plot_metrics(save_fig=False)
