from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


class MissingConfigError(RuntimeError):
    """Raised when required Junction credentials are missing."""


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class JunctionConfig:
    api_key: str
    region: str = "us"
    environment: str = "sandbox"

    @property
    def base_url(self) -> str:
        region = self.region.lower()
        environment = self.environment.lower()

        if environment == "sandbox":
            return f"https://api.sandbox.{region}.junction.com"
        if environment == "production":
            return f"https://api.{region}.junction.com"
        raise MissingConfigError(
            "JUNCTION_ENV must be either 'sandbox' or 'production'."
        )

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-vital-api-key": self.api_key,
        }


def load_config(env_path: str | Path = ".env") -> JunctionConfig:
    load_env_file(env_path)

    api_key = os.getenv("JUNCTION_API_KEY", "").strip()
    if not api_key:
        raise MissingConfigError(
            "Missing JUNCTION_API_KEY. Copy .env.example to .env and populate it."
        )

    region = os.getenv("JUNCTION_REGION", "us").strip().lower()
    environment = os.getenv("JUNCTION_ENV", "sandbox").strip().lower()
    return JunctionConfig(api_key=api_key, region=region, environment=environment)
