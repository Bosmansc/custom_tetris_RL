#from __future__ import print_function

import numpy as np
import random
from PIL import Image
import cv2

shapes = {
  #  'T': [(0, 0), (-1, 0), (1, 0), (0, -1)], ## triangle
  #  'J': [(0, 0), (-1, 0), (0, -1), (0, -2)]
  #  'L': [(0, 0), (1, 0), (0, -1), (0, -2)],
  #  'Z': [(0, 0), (-1, 0), (0, -1), (1, -1)],
  #  'S': [(0, 0), (-1, -1), (0, -1), (1, 0)],
  #  'I': [(0, 0), (0, -1), (0, -2), (0, -3)],
    'O': [(0, 0), (0, -1), (-1, 0), (-1, -1)],
}
shape_names = [#'J', 'L', 'Z', 'S', 'I',
                'O']

colors = {
        0: (255, 255, 255),
        1: (247, 64, 99),
        2: (0, 167, 247),
    }


def rotated(shape, cclk=False):
    if cclk:
        return [(-j, i) for i, j in shape]
    else:
        return [(j, -i) for i, j in shape]


def is_occupied(shape, anchor, board):
    for i, j in shape:
        x, y = anchor[0] + i, anchor[1] + j
        if x < 0 or x >= board.shape[0] or y >= board.shape[1] or (y >=0 and board[x, y]):
            return True
    return False


def left(shape, anchor, board):
    new_anchor = (anchor[0] - 1, anchor[1])
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def right(shape, anchor, board):
    new_anchor = (anchor[0] + 1, anchor[1])
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def soft_drop(shape, anchor, board):
    new_anchor = (anchor[0], anchor[1] + 1)
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def rotate_left(shape, anchor, board):
    new_shape = rotated(shape, cclk=False)
    return (shape, anchor) if is_occupied(new_shape, anchor, board) else (new_shape, anchor)


def rotate_right(shape, anchor, board):
    new_shape = rotated(shape, cclk=True)
    return (shape, anchor) if is_occupied(new_shape, anchor, board) else (new_shape, anchor)


def hard_drop(shape, anchor, board):
    while True:
        _, anchor_new = soft_drop(shape, anchor, board)
        if anchor_new == anchor:
            return shape, anchor_new
        anchor = anchor_new


def height(state):
    """
    Height of the highest column on the board
    """
    return max(get_column_heights(state))

def get_column_heights(state):
    """
    Helper function to calculate the height of each column
    """
    column_heights = []

    for i in range(state.shape[1]):
        column_height = 0
        for j in range(state.shape[0]):
            if state[j][i] == 1:
                column_height = state.shape[0] - j
                break

        column_heights.append(column_height)

    return column_heights


class TetrisEngine:
    def __init__(self, max_actions=5):
        # self.width = 10 # = initial
        # self.height = 20 # = initial

        self.width = 6
        self.height = 8

        self.board = np.zeros(shape=(self.width, self.height), dtype=np.float)
        self.action_count = 0
        self.max_actions = max_actions

        # actions are triggered by letters
        self.value_action_map = {
            0: rotate_left,
            1: rotate_right,
            2: right,
            3: left,
            4: soft_drop,
            5: hard_drop
        }
        self.action_value_map = dict([(j, i) for i, j in self.value_action_map.items()])
        self.nb_actions = len(self.value_action_map)

        # for running the engine
        self.score = -1
        self.anchor = None
        self.shape = None
        self.n_deaths = 0
        self.number_of_lines = 0

        # used for generating shapes
        self._shape_counts = [0] * len(shapes)

        # clear after initializing
        self.clear()

    def _choose_shape(self):
        maxm = max(self._shape_counts)
        m = [5 + maxm - x for x in self._shape_counts]
        r = random.randint(1, sum(m))
        for i, n in enumerate(m):
            r -= n
            if r <= 0:
                self._shape_counts[i] += 1
                return shapes[shape_names[i]]

    def _new_piece(self):
        self.action_count = 0
        self.anchor = (self.width / 2, 0)
        self.shape = self._choose_shape()

    def _has_dropped(self):
        return is_occupied(self.shape, (self.anchor[0], self.anchor[1] + 1), self.board)

    def _clear_lines(self):
        can_clear = [np.all(self.board[:, i]) for i in range(self.height)]
        new_board = np.zeros_like(self.board)
        j = self.height - 1
        for i in range(self.height - 1, -1, -1):
            if not can_clear[i]:
                new_board[:, j] = self.board[:, i]
                j -= 1

        if sum(can_clear) == 1:
            self.score += 40
        elif sum(can_clear) == 2:
            self.score += 100
        elif sum(can_clear) == 3:
            self.score += 300
        elif sum(can_clear) == 4:
            self.score += 1200
        self.board = new_board

        return sum(can_clear)

    def step(self, action):
        # Save previous score and height to calculate difference
        old_score = self.score
        old_height = height(np.transpose(np.copy(self.board)))

        self.anchor = (int(self.anchor[0]), int(self.anchor[1]))
        self.shape, self.anchor = self.value_action_map[action](self.shape, self.anchor, self.board)

        # Drop each step (unless action was already a soft drop)
        self.action_count += 1
        if action == 5:
            self.action_count = 0
        elif self.action_count == self.max_actions:
            self.shape, self.anchor = soft_drop(self.shape, self.anchor, self.board)
            self.action_count = 0

        done = False
        new_block = False

        if self._has_dropped():
            self._set_piece(True)
            lines_cleared = self._clear_lines()
            self.number_of_lines += lines_cleared
            if np.any(self.board[:, 0]):
                self.clear()
                self.n_deaths += 1
                done = True
            else:
                self._new_piece()
                new_block = True

        self._set_piece(2)
        state = np.transpose(np.copy(self.board))
        self._set_piece(False)

        height_difference = old_height - height(state)
        self._calculate_reward(height_difference, new_block)

        reward = self.score
        info = dict(score=reward, number_of_lines=self.number_of_lines, new_block=new_block,
                    height_difference=height_difference)

        return state, reward, done, info

    def clear(self):
        self.score = 0
        self.number_of_lines = 0
        self._new_piece()
        self.board = np.zeros_like(self.board)
        self._shape_counts = [0] * len(shapes)

        return np.transpose(self.board)

    def reset(self):
        return self.clear()

    def _calculate_reward(self, height_difference, new_block):
        if new_block and height_difference == 0:
            self.score = 10   # reward for keeping height low
        else:
            self.score = -0.2   # small penalty for each 'useless' step

    def _set_piece(self, on=False):
        for i, j in self.shape:
            x, y = i + self.anchor[0], j + self.anchor[1]
            if x < self.width and x >= 0 and y < self.height and y >= 0:
                self.board[int(self.anchor[0] + i), int(self.anchor[1] + j)] = on

    def __repr__(self):
        self._set_piece(True)
        s = 'o' + '-' * self.width + 'o\n'
        s += '\n'.join(['|' + ''.join(['X' if j else ' ' for j in i]) + '|' for i in self.board.T])
        s += '\no' + '-' * self.width + 'o'
        self._set_piece(False)
        return s

    def render(self, mode='human'):
        '''Renders the current board'''
        self._set_piece(2)
        state = np.copy(self.board)
        self._set_piece(False)
        img = [colors[p] for row in np.transpose(state) for p in row]
        img = np.array(img).reshape(self.height, self.width, 3).astype(np.uint8)
        img = img[..., ::-1] # Convert RRG to BGR (used by cv2)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((self.width * 25, self.height * 25), resample=Image.BOX)
        img = np.array(img)
        cv2.putText(img, str(self.score), (22, 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow('image', np.array(img))
        cv2.waitKey(1)

