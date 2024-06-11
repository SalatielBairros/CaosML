from unittest import TestCase
import numpy as np
from scipy.stats import entropy
from src.tree.decision_tree_classifier import DecisionTreeClassifier

class TestDecisionTreeClassifier(TestCase):
    def test_should_calculate_entropy(self):
        y_labels = np.asarray([
            1, 2, 1, 3, 4, 5, 1, 1, 4, 4, 5, 0
        ])
        total_count = np.bincount(y_labels)

        expected = entropy(total_count, base=2)

        classifier = DecisionTreeClassifier()
        actual = classifier.__get_entropy__(y_labels)

        assert round(expected, 4) == round(actual, 4)
        