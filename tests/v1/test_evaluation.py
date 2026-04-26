from __future__ import annotations

import unittest

from junction_portfolio.v1.evaluation import (
    assign_recommendation_flag,
    compute_provider_agreement_score,
)


class EvaluationTests(unittest.TestCase):
    def test_provider_agreement_defaults_for_single_provider(self) -> None:
        score = compute_provider_agreement_score(80, [80], tolerance=15.0)
        self.assertAlmostEqual(score, 0.7)

    def test_provider_agreement_penalizes_large_gap(self) -> None:
        score = compute_provider_agreement_score(50, [50, 80], tolerance=10.0)
        self.assertEqual(score, 0.0)

    def test_recommendation_assigns_show(self) -> None:
        flag = assign_recommendation_flag(0.82, True, 0.90)
        self.assertEqual(flag, "show")

    def test_recommendation_assigns_warning(self) -> None:
        flag = assign_recommendation_flag(0.64, True, 0.75)
        self.assertEqual(flag, "show_with_warning")

    def test_recommendation_assigns_suppress_without_signal(self) -> None:
        flag = assign_recommendation_flag(0.90, False, 0.90)
        self.assertEqual(flag, "suppress")


if __name__ == "__main__":
    unittest.main()
