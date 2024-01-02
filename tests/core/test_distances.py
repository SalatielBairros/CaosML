from unittest import TestCase
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as sk_ed
from sklearn.metrics.pairwise import manhattan_distances as sk_mh
from src.core.distances import euclidean_distances, manhattan_distances

class TestDistances(TestCase):
    def test_should_calculate_euclidean_distance(self):
        x = np.array([
            [0.5, 0.3, 0.23],
            [0.4, 0.4, 0.43],
            [0.3, 0.5, 0.33]
        ])

        target = np.array([0.3, 0.5, 0.33])

        distances = euclidean_distances(x, target)
        expected_distances = sk_ed(x, [target])

        for distance_actual, distance_expected in zip(distances, expected_distances):
            assert round(distance_actual, 4) == round(distance_expected[0], 4)

    def test_should_calculate_manhattan_distance(self):
        x = np.array([
            [0.5, 0.3, 0.23],
            [0.4, 0.4, 0.43],
            [0.3, 0.5, 0.33]
        ])

        target = np.array([0.3, 0.5, 0.33])

        distances = manhattan_distances(x, target)
        expected_distances = sk_mh(x, [target])

        for distance_actual, distance_expected in zip(distances, expected_distances):
            assert round(distance_actual, 4) == round(distance_expected[0], 4)
