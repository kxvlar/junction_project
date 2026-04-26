from __future__ import annotations

import unittest

import numpy as np

from junction_portfolio.stats import diff_in_means, permutation_test


class StatsTests(unittest.TestCase):
    def test_diff_in_means(self) -> None:
        outcome = np.array([1.0, 2.0, 4.0, 5.0])
        treatment = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(diff_in_means(outcome, treatment), 3.0)

    def test_permutation_test_is_deterministic(self) -> None:
        outcome = np.array([1.0, 2.0, 4.0, 5.0, 6.0, 7.0])
        treatment = np.array([0, 0, 0, 1, 1, 1])
        first = permutation_test(outcome, treatment, n_permutations=100, seed=7)
        second = permutation_test(outcome, treatment, n_permutations=100, seed=7)
        self.assertAlmostEqual(first["tau_obs"], second["tau_obs"])
        self.assertAlmostEqual(first["p_value"], second["p_value"])
        self.assertTrue(np.array_equal(first["null_dist"], second["null_dist"]))


if __name__ == "__main__":
    unittest.main()
