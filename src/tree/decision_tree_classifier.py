import numpy as np
from core.attribute_validation import is_numerical_attribute

class DecisionTreeClassifier:
    def __init__(self, entropy_type: str = 'normalized'):
        self.entropy_type = entropy_type

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

        node = DecisionTreeClassifierNode(split_value, attribute_index)
        counter = 0
        if is_numerical_attribute(split_value):
            for column_value in column_to_split:
                if column_value <= split_value:
                    node.left_x.append(X[counter])
                    node.left_y.append(y[counter])
                else:
                    node.right_x.append(X[counter])
                    node.right_y.append(y[counter])
                counter += 1
        else:
            for column_value in column_to_split:
                if column_value == split_value:
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

        best_split_node = DecisionTreeClassifierNode.empty()
        for value in unique_values:
            node = self.__split_classes__(X, y, attribute_index, value)
            information_gain = self.__information_gain__(y, node.left_y, node.right_y)
            node.set_information_gain(information_gain)

            if best_split_node.information_gain < node.information_gain:
                best_split_node = node

        return best_split_node

    def __find_best_feature__(self, X: np.array, y: np.array):
        n_features = len(X[0])
        best_feature_node = DecisionTreeClassifierNode.empty()
        for feature_index in range(n_features):
            best_value_split = self.__find_best_split__(X, y, feature_index)
            if best_value_split.information_gain > best_feature_node.information_gain:
                best_feature_node = best_value_split
            
            


class DecisionTreeClassifierNode:
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
        return DecisionTreeClassifierNode(0, -1)