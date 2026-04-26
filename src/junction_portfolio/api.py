from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import json
from typing import Iterable

import requests

from junction_portfolio.config import JunctionConfig
from junction_portfolio.normalization import parse_sleep_response


@dataclass
class SeededUser:
    user_id: str
    client_user_id: str
    provider: str


class JunctionClient:
    def __init__(self, config: JunctionConfig, timeout: int = 30) -> None:
        self.config = config
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        response = requests.request(
            method=method,
            url=f"{self.config.base_url}{path}",
            headers=self.config.headers,
            timeout=self.timeout,
            **kwargs,
        )
        response.raise_for_status()
        return response

    def create_user(self, client_user_id: str) -> str:
        response = requests.post(
            f"{self.config.base_url}/v2/user/",
            headers=self.config.headers,
            json={"client_user_id": client_user_id},
            timeout=self.timeout,
        )

        if response.status_code == 200:
            return response.json()["user_id"]

        payload = response.json()
        detail = payload.get("detail", {})
        existing_id = detail.get("user_id") if isinstance(detail, dict) else None
        if existing_id:
            return existing_id

        response.raise_for_status()
        raise RuntimeError("Unreachable user creation branch.")

    def connect_demo_provider(self, user_id: str, provider: str) -> dict:
        response = self._request(
            "POST",
            "/v2/link/connect/demo",
            json={"user_id": user_id, "provider": provider},
        )
        return response.json()

    def get_sleep_summary(
        self,
        user_id: str,
        start_date: str,
        end_date: str,
        provider: str | None = None,
    ) -> list[dict]:
        params = {"start_date": start_date, "end_date": end_date}
        if provider:
            params["provider"] = provider

        response = self._request(
            "GET",
            f"/v2/summary/sleep/{user_id}",
            params=params,
        )
        return parse_sleep_response(user_id=user_id, payload=response.json())


def seed_demo_users(
    client: JunctionClient,
    output_path: Path,
    providers: Iterable[str],
    users_per_provider: int,
    prefix: str,
) -> list[SeededUser]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seeded: list[SeededUser] = []
    for provider in providers:
        provider_slug = provider.strip()
        for index in range(users_per_provider):
            client_user_id = f"{prefix}_{provider_slug}_{index:02d}"
            user_id = client.create_user(client_user_id=client_user_id)
            client.connect_demo_provider(user_id=user_id, provider=provider_slug)
            seeded.append(
                SeededUser(
                    user_id=user_id,
                    client_user_id=client_user_id,
                    provider=provider_slug,
                )
            )

    output_path.write_text(
        json.dumps(
            {
                "seeded_at": date.today().isoformat(),
                "providers": list(providers),
                "users_per_provider": users_per_provider,
                "users": [seed.__dict__ for seed in seeded],
            },
            indent=2,
        )
    )
    return seeded


def pull_sleep_payload(
    client: JunctionClient,
    cohort_path: Path,
    output_path: Path,
    start_date: str,
    end_date: str,
) -> dict:
    payload = json.loads(cohort_path.read_text())
    users = payload.get("users", [])

    rows: list[dict] = []
    for user in users:
        user_rows = client.get_sleep_summary(
            user_id=user["user_id"],
            start_date=start_date,
            end_date=end_date,
            provider=user.get("provider"),
        )
        rows.extend(user_rows)

    combined = {
        "source": "junction_api",
        "start_date": start_date,
        "end_date": end_date,
        "records": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(combined, indent=2, default=str))
    return combined
