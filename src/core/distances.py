import numpy as np

def euclidean_distances(x_train: np.array, x_test_point: np.array):
    return np.sqrt(np.sum((x_train - x_test_point) ** 2, axis=1))
