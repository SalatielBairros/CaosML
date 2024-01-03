import numpy as np

class DecisionTreeClassifier:
    def __init__(self):
        pass

    def get_entropy(self, y: np.array):
        y_size = len(y)

        if y_size <= 1:
            return 0
        
        count_per_index = np.bincount(y)
        total_count = count_per_index[np.nonzero(count_per_index)]
        probabilities = total_count / y_size

        if len(probabilities) <= 1:
            return 0

        return - np.sum(probabilities * np.log2(probabilities))
    