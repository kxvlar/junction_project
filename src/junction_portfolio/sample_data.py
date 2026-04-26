from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import json


def generate_sample_payload(
    output_path: Path,
    start_date: str = "2026-03-01",
    end_date: str = "2026-03-14",
    scenario: str = "baseline",
) -> dict:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    total_days = (end - start).days + 1
    scenario = scenario.lower()

    providers = [
        ("portfolio_oura_primary", "user_demo_oura", "oura", "ring"),
        ("portfolio_fitbit_primary", "user_demo_fitbit", "fitbit", "watch"),
        ("portfolio_multi_oura", "user_demo_multi", "oura", "ring"),
        ("portfolio_multi_fitbit", "user_demo_multi", "fitbit", "watch"),
        ("portfolio_apple_primary", "user_demo_apple", "apple_health_kit", "phone"),
    ]

    rows: list[dict] = []
    for offset in range(total_days):
        current_date = start + timedelta(days=offset)
        day_factor = offset % 5
        for client_user_id, user_id, provider, source_type in providers:
            if provider == "apple_health_kit" and offset in {3, 8, 11} and scenario not in {"dense", "mature"}:
                continue
            if provider == "fitbit" and offset == 5 and user_id != "user_demo_multi" and scenario != "dense":
                continue
            if scenario == "mature" and user_id == "user_demo_fitbit" and offset in {2, 8}:
                continue

            recovery = {
                "oura": 79 + day_factor,
                "fitbit": 73 + day_factor,
                "apple_health_kit": None,
            }[provider]

            average_hrv = {
                "oura": 62 + offset * 0.6,
                "fitbit": 54 + offset * 0.5,
                "apple_health_kit": 58 + offset * 0.3,
            }[provider]

            if user_id == "user_demo_multi" and provider == "fitbit" and offset == 9:
                average_hrv -= 14
                if recovery is not None:
                    recovery -= 12

            if scenario == "mature":
                average_hrv += 0.4
                if recovery is not None:
                    recovery += 1
                if user_id == "user_demo_apple" and offset in {4, 10}:
                    average_hrv += 12
                if user_id == "user_demo_multi" and provider == "fitbit" and offset in {6, 9}:
                    average_hrv += 16
                    if recovery is not None:
                        recovery += 8
            elif scenario == "disagreement" and user_id == "user_demo_multi":
                average_hrv += 9 if provider == "oura" else -9
                if recovery is not None:
                    recovery += 7 if provider == "oura" else -7
            elif scenario == "sparse" and offset % 3 == 1:
                continue

            rows.append(
                {
                    "id": f"{provider}-{user_id}-{current_date.isoformat()}",
                    "user_id": user_id,
                    "client_user_id": client_user_id,
                    "calendar_date": current_date.isoformat(),
                    "date": f"{current_date.isoformat()}T22:00:00+00:00",
                    "average_hrv": round(average_hrv, 1),
                    "awake": 2400 + offset * 30,
                    "deep": 5200 + offset * 60,
                    "duration": 28600 + offset * 90,
                    "efficiency": round(0.88 + (day_factor * 0.01), 3),
                    "hr_average": 56 - (offset % 3),
                    "hr_lowest": 47 - (offset % 2),
                    "light": 11400 + offset * 55,
                    "rem": 6100 + offset * 45,
                    "total": 27100 + offset * 80,
                    "latency": 850 + (offset % 4) * 25,
                    "recovery_readiness_score": recovery,
                    "source": {
                        "device_id": f"{provider}-device-{user_id}",
                        "provider": provider,
                        "type": source_type,
                    },
                    "source_provider": provider,
                    "source_type": source_type,
                }
            )

    payload = {
        "source": "sample",
        "scenario": scenario,
        "start_date": start_date,
        "end_date": end_date,
        "records": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return payload
