from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import pandas as pd

from junction_portfolio.v2.reliability import (
    evaluate_reliability_model,
    score_reliability,
    train_reliability_model,
)


def _synthetic_label_table() -> pd.DataFrame:
    rows = []
    for user_id, provider, base_hrv, target in [
        ("u1", "oura", 60.0, 1),
        ("u1", "fitbit", 45.0, 0),
        ("u2", "oura", 62.0, 1),
        ("u2", "fitbit", 48.0, 0),
        ("u3", "oura", 64.0, 1),
        ("u3", "fitbit", 50.0, 0),
        ("u4", "oura", 66.0, 1),
        ("u4", "fitbit", 52.0, 0),
    ]:
        rows.append(
            {
                "user_id": user_id,
                "provider": provider,
                "date": "2026-03-01",
                "source_type": "ring" if provider == "oura" else "watch",
                "average_hrv": base_hrv,
                "recovery_readiness_score": 80.0 if provider == "oura" else None,
                "sleep_efficiency_pct": 90.0 if target == 1 else 82.0,
                "total_sleep_hours": 7.5,
                "deep_sleep_hours": 1.5,
                "rem_sleep_hours": 1.6,
                "awake_hours": 0.7,
                "hr_average": 55,
                "hr_lowest": 47,
                "window_start_date": "2026-03-01",
                "window_end_date": "2026-03-14",
                "days_observed_total": 10,
                "backfill_completeness": 0.8 if target == 1 else 0.45,
                "core_signal_available": True,
                "completeness_7d": 0.85 if target == 1 else 0.4,
                "baseline_nights_28d": 10 if target == 1 else 4,
                "stability_score": 0.8 if target == 1 else 0.3,
                "provider_agreement_score": 0.7,
                "provider_agreement_known": 1,
                "uses_recovery_readiness": 1 if provider == "oura" else 0,
                "signal_value": 80.0 if provider == "oura" else base_hrv,
                "label_observed": 1 if user_id != "u4" else 0,
                "safe_to_show": target if user_id != "u4" else None,
            }
        )
    return pd.DataFrame(rows)


class ReliabilityTests(unittest.TestCase):
    def test_train_score_evaluate_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            label_path = root / "label_table.csv"
            model_path = root / "models" / "reliability_model.pkl"
            scores_path = root / "reliability_scores.csv"
            metrics_path = root / "metrics.json"
            threshold_path = root / "thresholds.csv"
            comparison_path = root / "comparison.json"

            label_frame = _synthetic_label_table()
            label_frame.to_csv(label_path, index=False)

            bundle = train_reliability_model(label_path, model_path)
            self.assertIn("base_model", bundle)

            scored = score_reliability(label_path, model_path, scores_path)
            self.assertIn("p_safe_to_show_calibrated", scored.columns)
            self.assertIn("policy_action_v2", scored.columns)

            metrics = evaluate_reliability_model(
                label_table_path=label_path,
                model_path=model_path,
                scored_path=scores_path,
                metrics_output_path=metrics_path,
                threshold_sweep_output_path=threshold_path,
                comparison_output_path=comparison_path,
                v1_evaluation_path=None,
            )
            self.assertIn("unweighted", metrics)
            self.assertIn("ipw", metrics)
            self.assertIn("aipw", metrics)
            self.assertTrue(metrics_path.exists())
            self.assertTrue(threshold_path.exists())
            self.assertTrue(comparison_path.exists())


if __name__ == "__main__":
    unittest.main()
