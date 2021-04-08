import numpy as np

OUTLINER = np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]
])

VERTICAL_EDGE_DETECTOR = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

HORIZONTAL_EDGE_DETECTOR = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

BLUR = lambda n: np.ones((n, n)) / (n * n)