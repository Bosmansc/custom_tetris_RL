import time
import numpy as np
import random
from PIL import Image
import cv2
import pandas as pd
from matplotlib import pyplot
from time import sleep
import matplotlib.pyplot as plt
import json


def plot_custom_results(df, history, mode='training'):
    # input data
    if 'new_episode' not in df:
        raise KeyError('the dataframe has to have the new_episode column to plot the results')
    df["nr_episode"] = df["new_episode"].cumsum()

    df_results = df.groupby('nr_episode', as_index=False) \
        .agg(heigt_diff_sum=('height_difference', 'sum'),
             new_block_sum=('new_block', 'sum'),
             nr_lines_sum=('number_of_lines', 'sum'),
             score_sum=('score', 'sum'),
             score_avg=('score', 'mean'),
             count_steps=('nr_episode', 'count'))

    # init plot
    figure = pyplot.figure(figsize=(20, 10), dpi=80)
    figure.canvas.set_window_title(mode)

    # PLOT EPISODE REWARD
    pyplot.subplot(131)

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

    # PLOT NR OF LINES CLEARED PER EPISODE
    pyplot.subplot(133)
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

    # show the plots
    pyplot.show()
    pyplot.close()


def plot_metrics(save_fig=False):
    # plot the logs
    with open('dqn_log.json') as json_file:
        data = json.load(json_file)
    df_log = pd.DataFrame.from_dict(data)
    for idx, col in enumerate(df_log.columns):
        _combine_metrics(df_log, col, idx)
    timestr = time.strftime("%m%d_%H%M%S")
    if save_fig:
        pyplot.savefig("logs/img_logs_" + timestr)
    pyplot.show()


def _combine_metrics(df, key, index):
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