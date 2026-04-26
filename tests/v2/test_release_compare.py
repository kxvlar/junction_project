from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import pandas as pd

from junction_portfolio.v2.compare import compare_runs
from junction_portfolio.v2.releases import archive_run


class ReleaseAndCompareTests(unittest.TestCase):
    def test_archive_run_writes_manifest_and_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir) / "repo"
            (project_root / "artifacts" / "raw").mkdir(parents=True)
            (project_root / "artifacts" / "processed").mkdir(parents=True)
            (project_root / "artifacts" / "reports").mkdir(parents=True)
            (project_root / "artifacts" / "raw" / "junction_sleep_payload.json").write_text(
                json.dumps({"source": "sample", "start_date": "2026-03-01", "end_date": "2026-03-14", "records": []})
            )
            pd.DataFrame({"recommendation_flag": ["show"]}).to_csv(
                project_root / "artifacts" / "processed" / "evaluation_table.csv", index=False
            )
            (project_root / "artifacts" / "reports" / "recovery_signal_report.md").write_text("# report\n")

            snapshot = archive_run(project_root, version="v1.0.0", command_lines=["python -m test"], notes="test snapshot")
            self.assertTrue((snapshot / "manifest.json").exists())
            self.assertTrue((snapshot / "checksums.sha256").exists())
            manifest = json.loads((snapshot / "manifest.json").read_text())
            self.assertEqual(manifest["version"], "v1.0.0")

    def test_compare_runs_reports_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            left = Path(tmp_dir) / "left"
            right = Path(tmp_dir) / "right"
            for root, count in [(left, 1), (right, 3)]:
                (root / "artifacts" / "raw").mkdir(parents=True)
                (root / "artifacts" / "processed").mkdir(parents=True)
                (root / "artifacts" / "reports").mkdir(parents=True)
                (root / "artifacts" / "raw" / "junction_sleep_payload.json").write_text(
                    json.dumps(
                        {
                            "source": "sample",
                            "start_date": "2026-03-01",
                            "end_date": "2026-03-14",
                            "records": [{}] * count,
                        }
                    )
                )
                pd.DataFrame({"provider": ["oura"] * count}).to_csv(
                    root / "artifacts" / "processed" / "feature_table.csv", index=False
                )

            comparison = compare_runs(left, right)
            self.assertEqual(comparison["deltas"]["raw_record_count"], 2)
            self.assertEqual(comparison["deltas"]["feature_row_count"], 2)


if __name__ == "__main__":
    unittest.main()
