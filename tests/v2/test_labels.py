from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from junction_portfolio.v2.labels import add_reliability_features, build_label_table


class LabelTests(unittest.TestCase):
    def test_leave_one_out_provider_agreement(self) -> None:
        frame = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2"],
                "provider": ["oura", "fitbit", "oura"],
                "date": ["2026-03-01", "2026-03-01", "2026-03-01"],
                "average_hrv": [60.0, 70.0, 55.0],
                "recovery_readiness_score": [None, None, None],
            }
        )
        enriched = add_reliability_features(frame)
        u1 = enriched[enriched["user_id"] == "u1"].sort_values("provider").reset_index(drop=True)
        self.assertEqual(int(u1.loc[0, "provider_agreement_known"]), 1)
        self.assertEqual(int(u1.loc[1, "provider_agreement_known"]), 1)
        self.assertAlmostEqual(float(u1.loc[0, "provider_agreement_score"]), 0.0)
        self.assertAlmostEqual(float(u1.loc[1, "provider_agreement_score"]), 0.0)

        u2 = enriched[enriched["user_id"] == "u2"].iloc[0]
        self.assertEqual(int(u2["provider_agreement_known"]), 0)
        self.assertTrue(pd.isna(u2["provider_agreement_score"]))

    def test_build_label_table_assigns_labels(self) -> None:
        early = pd.DataFrame(
            {
                "user_id": ["u1", "u2", "u3"],
                "provider": ["oura", "fitbit", "oura"],
                "date": pd.to_datetime(["2026-03-01", "2026-03-01", "2026-03-01"]),
                "source_type": ["ring", "watch", "ring"],
                "average_hrv": [60.0, 50.0, 58.0],
                "recovery_readiness_score": [80.0, None, 75.0],
                "sleep_efficiency_pct": [90.0, 85.0, 88.0],
                "total_sleep_hours": [7.5, 7.0, 7.2],
                "deep_sleep_hours": [1.4, 1.2, 1.3],
                "rem_sleep_hours": [1.5, 1.4, 1.2],
                "awake_hours": [0.7, 0.8, 0.6],
                "hr_average": [55, 57, 56],
                "hr_lowest": [47, 49, 48],
                "window_start_date": ["2026-03-01"] * 3,
                "window_end_date": ["2026-03-14"] * 3,
                "days_observed_total": [10, 10, 10],
                "backfill_completeness": [0.7, 0.7, 0.7],
                "core_signal_available": [True, True, True],
                "completeness_7d": [0.8, 0.8, 0.8],
                "baseline_nights_28d": [10, 10, 10],
                "stability_score": [0.7, 0.7, 0.7],
            }
        )
        mature = pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "provider": ["oura", "fitbit"],
                "date": pd.to_datetime(["2026-03-01", "2026-03-01"]),
                "source_type": ["ring", "watch"],
                "average_hrv": [61.0, 65.0],
                "recovery_readiness_score": [82.0, None],
                "sleep_efficiency_pct": [89.0, 84.0],
                "total_sleep_hours": [7.6, 7.1],
                "deep_sleep_hours": [1.4, 1.2],
                "rem_sleep_hours": [1.5, 1.4],
                "awake_hours": [0.7, 0.8],
                "hr_average": [55, 57],
                "hr_lowest": [47, 49],
                "window_start_date": ["2026-03-01"] * 2,
                "window_end_date": ["2026-03-21"] * 2,
                "days_observed_total": [18, 18],
                "backfill_completeness": [0.9, 0.9],
                "core_signal_available": [True, True],
                "completeness_7d": [1.0, 1.0],
                "baseline_nights_28d": [18, 18],
                "stability_score": [0.8, 0.8],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            early_path = root / "early.csv"
            mature_path = root / "mature.csv"
            output_path = root / "labels.csv"
            early.to_csv(early_path, index=False)
            mature.to_csv(mature_path, index=False)

            labels = build_label_table(early_path, mature_path, output_path)
            self.assertEqual(len(labels), 3)
            self.assertEqual(int(labels.loc[labels["user_id"] == "u1", "safe_to_show"].iloc[0]), 1)
            self.assertEqual(int(labels.loc[labels["user_id"] == "u2", "safe_to_show"].iloc[0]), 0)
            self.assertEqual(int(labels.loc[labels["user_id"] == "u3", "label_observed"].iloc[0]), 0)


if __name__ == "__main__":
    unittest.main()
