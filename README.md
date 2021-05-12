# Tetis Reinforcement Learning

## Description

In this project we try to train an agent to play Tetris. 
The agent's only input is the image the environment creates after every action the agent takes.


## Framework

In the engine.py file there is the TetrisEngine class which defines the tetris environment. 

In the deep_q_agent.py file there is the Agent class which defines our agent playing the tetris game.

For the Reinforcement Learning framework we are using the Keras-RL2 library. This code provides us methods as 'compile', 'train' and 'test'.

## Packages
The packages needed to run this code are defined in the requirements.txt file. One important note is that our modified Keras RL 2 framework has to be downloaded since the general framework was not compatible with the used tetris engine.

So Keras RL 2 has to be installed this way:
pip install --upgrade --force-reinstall  git+https://github.com/Bosmansc/tetris_openai.git
