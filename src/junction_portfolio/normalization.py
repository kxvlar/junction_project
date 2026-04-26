from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


def parse_sleep_response(user_id: str, payload: dict) -> list[dict]:
    rows: list[dict] = []
    for sleep in payload.get("sleep", []):
        source = sleep.get("source") or {}
        normalized = dict(sleep)
        normalized["user_id"] = sleep.get("user_id", user_id)
        normalized["provider"] = (
            sleep.get("source_provider")
            or source.get("provider")
            or normalized.get("provider")
        )
        normalized["source_type"] = sleep.get("source_type") or source.get("type")
        normalized["source_device_id"] = source.get("device_id")
        normalized["calendar_date"] = sleep.get("calendar_date") or str(
            pd.to_datetime(sleep["date"]).date()
        )
        rows.append(normalized)
    return rows


def load_raw_payload(raw_path: Path) -> tuple[pd.DataFrame, str, str]:
    payload = json.loads(raw_path.read_text())
    frame = pd.DataFrame(payload.get("records", []))
    if frame.empty:
        return frame, payload["start_date"], payload["end_date"]

    source_frame = pd.json_normalize(frame["source"]).add_prefix("source.")
    frame = pd.concat([frame.drop(columns=["source"], errors="ignore"), source_frame], axis=1)
    frame["date"] = pd.to_datetime(frame["calendar_date"], errors="coerce")
    if "provider" not in frame.columns:
        frame["provider"] = frame.get("source_provider")
    else:
        frame["provider"] = frame["provider"].fillna(frame.get("source_provider"))
    if "source.provider" in frame.columns:
        frame["provider"] = frame["provider"].fillna(frame["source.provider"])
    if "source_type" not in frame.columns:
        frame["source_type"] = frame.get("source.type", "unknown")
    else:
        frame["source_type"] = frame["source_type"].fillna("unknown")
    if "source.type" in frame.columns:
        frame["source_type"] = frame["source_type"].fillna(frame["source.type"])
    if "source_device_id" not in frame.columns and "source.device_id" in frame.columns:
        frame["source_device_id"] = frame["source.device_id"]
    return frame, payload["start_date"], payload["end_date"]


def _to_percent(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.where(values <= 1.0, values * 100.0, values)


def _to_hours(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values / 3600.0


def build_feature_table(raw_path: Path, output_path: Path) -> pd.DataFrame:
    df, start_date, end_date = load_raw_payload(raw_path)
    if df.empty:
        raise ValueError("No records found in raw Junction payload.")

    feature_table = pd.DataFrame(
        {
            "user_id": df["user_id"],
            "provider": df["provider"],
            "date": pd.to_datetime(df["calendar_date"]).dt.date,
            "source_type": df["source_type"],
            "source_device_id": df.get("source_device_id"),
            "average_hrv": pd.to_numeric(df.get("average_hrv"), errors="coerce"),
            "recovery_readiness_score": pd.to_numeric(
                df.get("recovery_readiness_score"), errors="coerce"
            ),
            "sleep_efficiency_pct": _to_percent(df.get("efficiency")),
            "total_sleep_hours": _to_hours(df.get("total")),
            "deep_sleep_hours": _to_hours(df.get("deep")),
            "rem_sleep_hours": _to_hours(df.get("rem")),
            "awake_hours": _to_hours(df.get("awake")),
            "hr_average": pd.to_numeric(df.get("hr_average"), errors="coerce"),
            "hr_lowest": pd.to_numeric(df.get("hr_lowest"), errors="coerce"),
            "window_start_date": start_date,
            "window_end_date": end_date,
        }
    )

    feature_table = feature_table.sort_values(["user_id", "provider", "date"]).reset_index(
        drop=True
    )

    grouped = feature_table.groupby(["user_id", "provider"], group_keys=False)
    expected_days = (
        pd.to_datetime(end_date).date() - pd.to_datetime(start_date).date()
    ).days + 1

    feature_table["days_observed_total"] = grouped["date"].transform("count")
    feature_table["backfill_completeness"] = (
        feature_table["days_observed_total"] / expected_days
    ).clip(upper=1.0)

    feature_table["core_signal_available"] = (
        feature_table["recovery_readiness_score"].notna()
        | feature_table["average_hrv"].notna()
    )

    def add_group_features(group: pd.DataFrame) -> pd.DataFrame:
        ordered = group.sort_values("date").copy()
        ordered["user_id"] = group.name[0]
        ordered["provider"] = group.name[1]
        complete_dates = pd.date_range(start=start_date, end=end_date, freq="D")
        observed = (
            ordered.set_index(pd.to_datetime(ordered["date"]))
            .reindex(complete_dates)
            .assign(
                metric=lambda frame: frame["recovery_readiness_score"].fillna(
                    frame["average_hrv"]
                ),
                observed=lambda frame: frame["metric"].notna().astype(int),
            )
        )

        observed["completeness_7d"] = (
            observed["observed"].rolling(window=7, min_periods=1).mean()
        )
        observed["baseline_nights_28d"] = (
            observed["observed"].rolling(window=28, min_periods=1).sum()
        )
        metric = observed["metric"]
        rolling_median = metric.rolling(window=7, min_periods=3).median()
        rolling_mad = (
            (metric - rolling_median)
            .abs()
            .rolling(window=7, min_periods=3)
            .median()
            .replace(0, np.nan)
        )
        deviation = (metric - rolling_median).abs()
        observed["stability_score"] = (
            1 - (deviation / (rolling_mad * 3)).clip(lower=0, upper=1)
        ).fillna(0.6)

        feature_days = pd.to_datetime(ordered["date"])
        ordered["completeness_7d"] = observed.loc[feature_days, "completeness_7d"].to_numpy()
        ordered["baseline_nights_28d"] = observed.loc[
            feature_days, "baseline_nights_28d"
        ].to_numpy()
        ordered["stability_score"] = observed.loc[feature_days, "stability_score"].to_numpy()
        return ordered

    feature_table = grouped.apply(add_group_features).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(output_path, index=False)
    return feature_table
