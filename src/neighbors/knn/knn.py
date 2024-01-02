from collections import Counter
import numpy as np
from core.distances import euclidean_distances, manhattan_distances
from core.exceptions.not_implemented import NotImplementedException

class KNN:

    def __init__(self, k: int, distance: str = 'euclidean', task: str = 'classification'):
        self.k = k
        self.train_data = []
        self.labels = []
        self.distance = distance
        self.task = task

    def fit(self, x: np.array, y: np.array):
        self.train_data = x
        self.labels = y

    def predict(self, x: np.array):
        predictions = []

        for item in x:
            distances = self.__calculate_distances__(item)
            distances_with_labels = np.c_[distances, self.labels]
            nearest = self.__nearest_neighborns__(distances_with_labels)
            
            if self.task == 'classification':
                prediction = self.__voting__(nearest[:,1])
            elif self.task == 'regression':
                prediction = self.__regression__(nearest[:,1])
            else:
                raise NotImplementedException(f"task {self.task}")
            predictions.append(prediction)
        
        return np.asarray(predictions)

    def fit_predict(self, x: np.array, y: np.array):
        self.fit(x, y)
        return self.predict(x)
    
    def accuracy(self, x: np.array, y: np.array):
        predictions = self.predict(x)
        return np.mean(predictions == y)
    
    def __calculate_distances__(self, target: np.array) -> np.array:
        if self.distance == 'euclidean':
            return euclidean_distances(self.train_data, target)
        if self.distance == 'manhattan':
            return manhattan_distances(self.train_data, target)
        
        raise NotImplementedException(f'{self.distance} distance')

    def __nearest_neighborns__(self, distances: np.array):
        return distances[distances[:, 0].argsort()][:self.k]
    
    def __voting__(self, nearest_neighborns_labels: np.array):
        return Counter(nearest_neighborns_labels).most_common()[0][0]
    
    def __regression__(self, nearest_neighborns_target: np.array):
        return np.mean(nearest_neighborns_target)
