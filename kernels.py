import numpy as np

OUTLINER = np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]
])

TEST = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

BLUR = lambda n: np.ones((n, n)) / (n * n)