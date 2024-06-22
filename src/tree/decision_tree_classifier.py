import numpy as np
from scipy import stats
from base_functions import is_numerical_attribute

class DecisionTreeClassifier:
    def __init__(self, entropy_type: str = 'normalized', min_size: int = 2, max_depth: int = 5):
        self.entropy_type = entropy_type
        self.min_size = min_size
        self.max_depth = max_depth

    def __get_entropy__(self, y: np.array):
        y_size = len(y)

        if y_size <= 1:
            return 0
        
        count_per_index = np.bincount(y)
        total_count = count_per_index[np.nonzero(count_per_index)]
        probabilities = total_count / y_size

        if len(probabilities) <= 1:
            return 0

        if self.entropy_type == 'normalized':
            return - np.sum(probabilities * np.log(probabilities)) / np.log(len(probabilities))
        elif self.entropy_type == 'shannon':
            return - np.sum(probabilities * np.log2(probabilities))
        elif self.entropy_type == 'gini':
            return 1 - np.sum(probabilities ** 2)
    
    def __information_gain__(self, previous_y: np.array, current_y_left: np.array, current_y_right: np.array):
        initial_entropy = self.__get_entropy__(previous_y)
        size_previous_y = len(previous_y)

        current_y_left_entropy = self.__get_entropy__(current_y_left)
        current_y_right_entropy = self.__get_entropy__(current_y_right)

        left_weight = len(current_y_left) / size_previous_y
        right_weight = len(current_y_right) / size_previous_y

        weight_entropy = (current_y_left_entropy * left_weight) + (current_y_right_entropy * right_weight)

        return initial_entropy - weight_entropy
    
    def __split_classes__(self, X: np.array, y: np.array, attribute_index: int, split_value):
        column_to_split = X[:, attribute_index]

        node = DecisionTreeClassifierSplit(split_value, attribute_index)
        counter = 0

        split_function = (lambda x: x <= split_value) if is_numerical_attribute(split_value) else (lambda x: x == split_value)
        
        for column_value in column_to_split:
            if split_function(column_value):
                node.left_x.append(X[counter])
                node.left_y.append(y[counter])
            else:
                node.right_x.append(X[counter])
                node.right_y.append(y[counter])
            counter += 1        

        return node

    def __find_best_split__(self, X: np.array, y: np.array, attribute_index: int):
        column_to_split = X[:, attribute_index]
        unique_values = np.unique(column_to_split)

        best_split_node = DecisionTreeClassifierSplit.empty()
        for value in unique_values:
            node = self.__split_classes__(X, y, attribute_index, value)
            information_gain = self.__information_gain__(y, node.left_y, node.right_y)
            node.set_information_gain(information_gain)

            if best_split_node.information_gain < node.information_gain:
                best_split_node = node

        return best_split_node

    def __find_best_feature__(self, X: np.array, y: np.array):
        n_features = len(X[0])
        best_feature_node = DecisionTreeClassifierSplit.empty()
        for feature_index in range(n_features):
            best_value_split = self.__find_best_split__(X, y, feature_index)
            if best_value_split.information_gain > best_feature_node.information_gain:
                best_feature_node = best_value_split
        return best_feature_node
    
    def __validate__(self, X: np.array, y: np.array):
        X_size = len(X)
        
        if X_size <= self.min_size:
            raise Exception("Input size must be bigger than min_size")

        if X_size != len(y):
            raise Exception("Both X and y must have the same size")
        
    def __fit_node__(self, node, depth: int = 1):
        best_split = self.__find_best_feature__(node.X, node.y)
        if best_split.information_gain > 0:
            node.add_children(best_split, self.min_size)
            if (depth + 1) <= self.max_depth:
                if node.left:
                    self.__fit_node__(node.left, (depth + 1))
                if node.right:
                    self.__fit_node__(node.right, (depth + 1))

    def fit(self, X: np.array, y: np.array):
        self.__validate__(X, y)
        root = DecisionTreeClassifierNode(X, y)
        self.__fit_node__(root)
        self.root = root
        return self
    
    def __predict_single(self, x: np.array, node):
        if node.left is None and node.right is None:
            return node.vote()

        if node.split_function(x[node.splited_feature_index]):
            if node.left is None:
                return node.vote()
            return self.__predict_single(x, node.left)
        else:
            if node.right is None:
                return node.vote()
            return self.__predict_single(x, node.right)

    def predict(self, X: np.array):
        predictions = [self.__predict_single(x, self.root) for x in X]
        return np.array(predictions)


class DecisionTreeClassifierSplit:
    def __init__(self, value, feature_index: int) -> None:
        self.left_x = []
        self.left_y = []
        self.right_x = []
        self.right_y = []
        self.value = value
        self.feature_index = feature_index
        self.information_gain = 0
    
    def set_information_gain(self, information_gain: int):
        if information_gain > 0:
            self.information_gain = information_gain

    @staticmethod
    def empty():
        return DecisionTreeClassifierSplit(0, -1)
    
class DecisionTreeClassifierNode:
    def __init__(self, X: np.array, y: np.array, parent_node = None):
        self.X = X
        self.y = y
        self.splited_value = None
        self.splited_feature_index = None
        self.left: DecisionTreeClassifierNode = None
        self.right: DecisionTreeClassifierNode = None
        self.parent: DecisionTreeClassifierNode = parent_node
        self.split_function = (lambda x: x == self.splited_value) 

    def add_children(self, split: DecisionTreeClassifierSplit, min_size: int):
        self.splited_value = split.value
        if is_numerical_attribute(split.value):
            self.split_function = (lambda x: x <= self.splited_value)

        self.splited_feature_index = split.feature_index

        if len(split.left_x) >= min_size:
            self.left = DecisionTreeClassifierNode(split.left_x, split.left_y, self)

        if len(split.right_x) >= min_size:
            self.right = DecisionTreeClassifierNode(split.right_x, split.right_y, self)

    def vote(self):
        return stats.mode(self.y)[0]