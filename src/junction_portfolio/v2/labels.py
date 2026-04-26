from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["user_id", "provider", "date"]


def _signal_value(frame: pd.DataFrame) -> pd.Series:
    return frame["recovery_readiness_score"].fillna(frame["average_hrv"])


def add_reliability_features(frame: pd.DataFrame) -> pd.DataFrame:
    augmented = frame.copy()
    augmented["date"] = pd.to_datetime(augmented["date"])
    augmented["uses_recovery_readiness"] = augmented["recovery_readiness_score"].notna().astype(int)
    augmented["signal_value"] = _signal_value(augmented)
    augmented["provider_agreement_score"] = np.nan
    augmented["provider_agreement_known"] = 0
    augmented["provider_agreement_gap"] = np.nan

    for (_, date_value), group in augmented.groupby(["user_id", "date"], sort=False):
        metric = group["signal_value"]
        for index, row in group.iterrows():
            peers = metric.drop(index=index).dropna()
            if peers.empty:
                augmented.loc[index, "provider_agreement_known"] = 0
                continue
            peer_median = float(np.nanmedian(peers.to_numpy(dtype=float)))
            tolerance = 15.0 if pd.notna(row["recovery_readiness_score"]) else 10.0
            gap = abs(float(row["signal_value"]) - peer_median)
            score = max(0.0, 1.0 - (gap / tolerance))
            augmented.loc[index, "provider_agreement_known"] = 1
            augmented.loc[index, "provider_agreement_score"] = score
            augmented.loc[index, "provider_agreement_gap"] = gap

    return augmented


def _safe_to_show(row: pd.Series) -> float | pd.NA:
    if int(row["label_observed"]) != 1:
        return pd.NA

    efficiency_ok = True
    if pd.notna(row.get("sleep_efficiency_pct")) and pd.notna(row.get("mature_sleep_efficiency_pct")):
        efficiency_ok = abs(float(row["sleep_efficiency_pct"]) - float(row["mature_sleep_efficiency_pct"])) <= 5.0

    if pd.notna(row.get("recovery_readiness_score")) and pd.notna(row.get("mature_recovery_readiness_score")):
        readiness_ok = abs(
            float(row["recovery_readiness_score"]) - float(row["mature_recovery_readiness_score"])
        ) <= 5.0
        return int(readiness_ok and efficiency_ok)

    if pd.notna(row.get("average_hrv")) and pd.notna(row.get("mature_average_hrv")):
        mature_value = abs(float(row["mature_average_hrv"]))
        threshold = max(5.0, mature_value * 0.15)
        hrv_ok = abs(float(row["average_hrv"]) - float(row["mature_average_hrv"])) <= threshold
        return int(hrv_ok and efficiency_ok)

    return 0


def build_label_table(
    early_feature_path: Path,
    mature_feature_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    early = pd.read_csv(early_feature_path, parse_dates=["date"])
    mature = pd.read_csv(mature_feature_path, parse_dates=["date"])

    early = add_reliability_features(early)
    mature = add_reliability_features(mature)
    mature["mature_core_signal_available"] = mature["core_signal_available"].astype(int)

    mature_columns = [
        "average_hrv",
        "recovery_readiness_score",
        "sleep_efficiency_pct",
        "core_signal_available",
        "window_end_date",
    ]
    mature_subset = mature[KEY_COLUMNS + mature_columns].rename(
        columns={
            "average_hrv": "mature_average_hrv",
            "recovery_readiness_score": "mature_recovery_readiness_score",
            "sleep_efficiency_pct": "mature_sleep_efficiency_pct",
            "core_signal_available": "mature_core_signal_available",
            "window_end_date": "mature_window_end_date",
        }
    )

    label_table = early.merge(mature_subset, on=KEY_COLUMNS, how="left")
    label_table["label_observed"] = label_table["mature_core_signal_available"].fillna(0).astype(int)
    label_table["safe_to_show"] = label_table.apply(_safe_to_show, axis=1)
    label_table["observation_gap_days"] = (
        pd.to_datetime(label_table["mature_window_end_date"], errors="coerce")
        - pd.to_datetime(label_table["window_end_date"], errors="coerce")
    ).dt.days
    label_table["safe_to_show"] = label_table["safe_to_show"].astype("Int64")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_table.to_csv(output_path, index=False)
    return label_table
