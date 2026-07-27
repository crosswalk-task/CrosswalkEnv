from enum import Enum

import numpy as np

class Action(Enum):
    NOTHING = 0
    UP = 1
    DOWN = 2
    RIGHT = 3
    LEFT = 4

    def __str__(self):
        string_map = {
            Action.NOTHING: "Nothing",
            Action.UP: "UP",
            Action.DOWN: "DOWN",
            Action.RIGHT: "RIGHT",
            Action.LEFT: "LEFT",
        }
        return string_map[self]

ACTION_TO_DELTA = {
    Action.NOTHING.value: np.array([0, 0]),
    Action.UP.value: np.array([0, -1]),
    Action.DOWN.value: np.array([0, 1]),
    Action.RIGHT.value: np.array([1, 0]),
    Action.LEFT.value: np.array([-1, 0]),
}

ACTION_DURATION = {
    Action.NOTHING.value: 1,
    Action.UP.value: 4,
    Action.DOWN.value: 4,
    Action.RIGHT.value: 4,
    Action.LEFT.value: 4,
}
