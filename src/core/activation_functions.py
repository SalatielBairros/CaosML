import numpy as np

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def softmax(x: np.array) -> np.array:
    exps = np.exp(x)
    return exps / np.sum(exps, axis=0)

def relu(x: np.array) -> np.array:
    return np.maximum(0, x)
