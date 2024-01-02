import numpy as np

def euclidean_distances(x: np.array, target: np.array):
    return np.sqrt(np.sum((x - target) ** 2, axis=1))

def manhattan_distances(x: np.array, target: np.array):
    return np.sum(np.absolute(x - target), axis=1)
