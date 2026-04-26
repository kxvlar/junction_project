from __future__ import annotations

import unittest

from junction_portfolio.normalization import parse_sleep_response


class SchemaTests(unittest.TestCase):
    def test_parse_sleep_response_flattens_provider_fields(self) -> None:
        payload = {
            "sleep": [
                {
                    "calendar_date": "2026-03-03",
                    "average_hrv": 67,
                    "recovery_readiness_score": 81,
                    "source": {
                        "device_id": "device-1",
                        "provider": "oura",
                        "type": "ring",
                    },
                }
            ]
        }
        rows = parse_sleep_response(user_id="user-1", payload=payload)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["provider"], "oura")
        self.assertEqual(rows[0]["source_type"], "ring")
        self.assertEqual(rows[0]["source_device_id"], "device-1")
        self.assertEqual(rows[0]["user_id"], "user-1")


if __name__ == "__main__":
    unittest.main()
