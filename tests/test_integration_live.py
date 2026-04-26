from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from junction_portfolio.api import JunctionClient, pull_sleep_payload, seed_demo_users
from junction_portfolio.config import load_config


@unittest.skipUnless(os.getenv("JUNCTION_API_KEY"), "Live Junction credentials not configured.")
class LiveIntegrationTests(unittest.TestCase):
    def test_seed_and_pull_from_sandbox(self) -> None:
        client = JunctionClient(load_config())
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            cohort = base / "cohort.json"
            raw = base / "raw.json"
            seeded = seed_demo_users(
                client=client,
                output_path=cohort,
                providers=["oura"],
                users_per_provider=1,
                prefix="junction_portfolio_test",
            )
            self.assertEqual(len(seeded), 1)
            payload = pull_sleep_payload(
                client=client,
                cohort_path=cohort,
                output_path=raw,
                start_date="2026-03-01",
                end_date="2026-03-14",
            )
            self.assertIn("records", payload)


if __name__ == "__main__":
    unittest.main()
