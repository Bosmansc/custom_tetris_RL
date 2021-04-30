# custom_tetris_RL

## Parameters

Q:

The ‘q’ in q-learning stands for quality. Quality in this case represents how useful a given action is in gaining some future reward.

## Learning Process


1- Initialize replay memory capacity.

2- Initialize the policy network with random weights.

3- Clone the policy network, and call it the target network (target_model).

4- For each episode:

    1. Initialize the starting state.
    2. For each time step:
        1- Select an action.
        - Via exploration or exploitation, which depends on epsilon.
        2- Execute selected action in an emulator (the environment).
        3- Observe reward and next state.
        4- Store experience in replay memory.
        5- Sample random batch from replay memory.
        6- Preprocess states from batch (normalization).
        7- Pass batch of preprocessed states to policy network.
        8- Calculate loss between output Q-values and target Q-values.
        - Requires a pass to the target network for the next state
        9- Gradient descent updates weights in the policy network to minimize loss.
        - After x time steps, weights in the target network are updated to the weights in the policy network.