from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import pandas as pd

from junction_portfolio.v2.report import render_report_v2


class ReportV2Tests(unittest.TestCase):
    def test_render_report_v2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            label_path = root / "labels.csv"
            score_path = root / "scores.csv"
            metrics_path = root / "metrics.json"
            threshold_path = root / "thresholds.csv"
            comparison_path = root / "comparison.json"
            output_path = root / "reports" / "reliability_model_report.md"

            pd.DataFrame(
                {
                    "user_id": ["u1", "u2"],
                    "provider": ["oura", "fitbit"],
                    "date": pd.to_datetime(["2026-03-01", "2026-03-01"]),
                    "label_observed": [1, 1],
                    "safe_to_show": [1, 0],
                }
            ).to_csv(label_path, index=False)
            pd.DataFrame(
                {
                    "user_id": ["u1", "u2"],
                    "provider": ["oura", "fitbit"],
                    "date": pd.to_datetime(["2026-03-01", "2026-03-01"]),
                    "p_safe_to_show_calibrated": [0.8, 0.2],
                    "policy_action_v2": ["show", "suppress"],
                }
            ).to_csv(score_path, index=False)
            metrics_path.write_text(
                json.dumps(
                    {
                        "unweighted": {
                            "positive_rate": 0.5,
                            "brier": 0.05,
                            "log_loss": 0.2,
                            "show_precision": 1.0,
                            "coverage_show": 0.5,
                        },
                        "ipw": {"brier": 0.05, "show_precision": 1.0},
                        "aipw": {"estimated_safe_to_show_rate": 0.5},
                        "leave_one_user_out": {"brier": 0.1},
                    }
                )
            )
            pd.DataFrame(
                {
                    "warning_threshold": [0.5],
                    "show_threshold": [0.75],
                    "coverage_show": [0.5],
                    "show_precision": [1.0],
                    "warning_rate": [0.0],
                }
            ).to_csv(threshold_path, index=False)
            comparison_path.write_text(json.dumps({"available": False}))

            report = render_report_v2(
                label_table_path=label_path,
                scored_path=score_path,
                metrics_path=metrics_path,
                threshold_sweep_path=threshold_path,
                comparison_path=comparison_path,
                output_path=output_path,
            )
            self.assertIn("# Reliability Model Report (V2)", report)
            self.assertTrue(output_path.exists())
            self.assertTrue((output_path.parent / "v2_calibration.png").exists())
            self.assertTrue((output_path.parent / "v2_coverage_risk.png").exists())


if __name__ == "__main__":
    unittest.main()
