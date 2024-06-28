from src.tree.decision_tree_classifier import DecisionTreeClassifier
import numpy as np
from collections import defaultdict, Counter

class RandomForestClasifier:
    def __init__(self, 
                 n_estimators=50, 
                 criterion='entropy', 
                 max_depth=5, 
                 max_features: float=0.7,
                 min_estimator_size: int = 3,
                 random_state: int = None) -> None:
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_estimator_size = min_estimator_size
        self.random_state = random_state

        if self.random_state is not None:
            np.random.seed(self.random_state)        

        trees = []
        for _ in range(n_estimators):
            trees.append(TreeEstimator(criterion, min_estimator_size, max_depth))

        self.estimators:list[TreeEstimator] = trees

    def __bootstrap__(self, training_data_size: int, training_data_n_features: int):
        sample_indexes = list(range(training_data_size))
        selected_rows_indexes = np.random.choice(sample_indexes, training_data_size)
        n_features_to_select = int(training_data_n_features * self.max_features)
        selected_columns_indexes = np.random.permutation(training_data_n_features)[:n_features_to_select]

        return selected_rows_indexes, selected_columns_indexes
    
    def __bootstrap_all_estimators__(self, training_data_size: int, training_data_n_features: int):
        for estimator in self.estimators:
            idx_rows, idx_columns = self.__bootstrap__(training_data_size, training_data_n_features)
            # Get the row indexes that were not taken
            all_indexes = set(list(range(training_data_size)))
            total = all_indexes - set(idx_rows)

            estimator.set_bootstrap_result(
                idx_rows,
                idx_columns,
                total
            )

    def fit(self, X: np.array, y:np.array):
        training_data_size, training_data_n_features = X.shape
        self.__bootstrap_all_estimators__(training_data_size, training_data_n_features)
        for estimator in self.estimators:
            estimator.fit(X, y)

    def out_of_bag_score(self, X: np.array, y: np.array):
        all_predictions = []
        for estimator in self.estimators:
            predictions = estimator.out_of_bag_predictions(X)
            all_predictions.append(predictions)

        idx_group = defaultdict(list)

        for predictions_list in all_predictions:
            for idx, prediction in predictions_list:
                idx_group [idx].append(prediction)

        result = []
        for idx, predictions in idx_group.items():
            most_frequent_value = Counter(predictions).most_common(1)[0][0]
            correct = 1 if y[idx] == most_frequent_value else 0
            result.append((idx, most_frequent_value, correct))

        return np.mean(np.array(result)[:,2])

    def predict(self, X):
        pass

class TreeEstimator:
    def __init__(self, criterion: str, min_estimator_size: int, max_depth: int) -> None:
        self.estimator_model = DecisionTreeClassifier(criterion, min_estimator_size, max_depth)
        self.bootstrap_row_indexes = []
        self.feature_indexes = []
        self.out_of_bag_indexes = []

    def set_bootstrap_result(self, row_indexes: list, feature_indexes: list, out_of_bag: list):
        self.bootstrap_row_indexes = []
        self.feature_indexes = []
        self.out_of_bag_indexes = []
        
        self.bootstrap_row_indexes.extend(row_indexes)
        self.feature_indexes.extend(feature_indexes)
        self.out_of_bag_indexes.extend(out_of_bag)

    def fit(self, X: np.array, y: np.array):
        current_X = X[self.bootstrap_row_indexes[:, np.newaxis], self.feature_indexes]
        current_y = y[self.bootstrap_row_indexes]
        self.estimator_model.fit(current_X, current_y)

    def out_of_bag_predictions(self, X):
        response = []
        for oob_idx in self.out_of_bag_indexes:
            row = X[oob_idx, self.feature_indexes]
            prediction = self.estimator_model.predict(row)
            response.append((oob_idx, prediction))
        return response
    
    def predict(self, X):
        return self.estimator_model.predict(X)
