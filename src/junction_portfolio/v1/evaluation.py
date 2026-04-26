from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def compute_provider_agreement_score(
    metric_value: float | int | None,
    peer_values: list[float],
    tolerance: float,
    single_provider_default: float = 0.7,
) -> float:
    if metric_value is None or pd.isna(metric_value):
        return 0.0
    if len(peer_values) <= 1:
        return single_provider_default

    peer_array = np.asarray(peer_values, dtype=float)
    peer_median = float(np.nanmedian(peer_array))
    gap = abs(float(metric_value) - peer_median)
    return float(max(0.0, 1.0 - (gap / tolerance)))


def assign_recommendation_flag(
    confidence_score: float,
    core_signal_available: bool,
    backfill_completeness: float,
) -> str:
    if not core_signal_available:
        return "suppress"
    if backfill_completeness < 0.35 or confidence_score < 0.5:
        return "suppress"
    if confidence_score < 0.75:
        return "show_with_warning"
    return "show"


def _build_group_agreement(group: pd.DataFrame, user_id: str, date_value: pd.Timestamp) -> pd.DataFrame:
    group = group.copy()
    group["user_id"] = user_id
    group["date"] = date_value
    metric = group["recovery_readiness_score"].fillna(group["average_hrv"])
    tolerance = 15.0 if group["recovery_readiness_score"].notna().any() else 10.0
    peer_values = [value for value in metric.dropna().tolist()]

    group["provider_agreement_score"] = [
        compute_provider_agreement_score(value, peer_values, tolerance=tolerance)
        for value in metric
    ]
    if len(peer_values) <= 1:
        group["provider_agreement_gap"] = np.nan
    else:
        peer_median = float(np.nanmedian(np.asarray(peer_values, dtype=float)))
        group["provider_agreement_gap"] = (metric - peer_median).abs()
    return group


def run_evaluation(feature_table_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(feature_table_path, parse_dates=["date"])
    if df.empty:
        raise ValueError("Feature table is empty.")

    evaluated_groups = [
        _build_group_agreement(group, user_id=user_id, date_value=date_value)
        for (user_id, date_value), group in df.groupby(["user_id", "date"], sort=False)
    ]
    evaluated = pd.concat(evaluated_groups, ignore_index=True)

    evaluated["completeness_7d"] = pd.to_numeric(
        evaluated["completeness_7d"], errors="coerce"
    ).fillna(0.0)
    evaluated["backfill_completeness"] = pd.to_numeric(
        evaluated["backfill_completeness"], errors="coerce"
    ).fillna(0.0)
    evaluated["stability_score"] = pd.to_numeric(
        evaluated["stability_score"], errors="coerce"
    ).fillna(0.5)
    evaluated["provider_agreement_score"] = pd.to_numeric(
        evaluated["provider_agreement_score"], errors="coerce"
    ).fillna(0.7)

    evaluated["confidence_score"] = (
        evaluated["completeness_7d"] * 0.35
        + evaluated["backfill_completeness"] * 0.25
        + (evaluated["baseline_nights_28d"].clip(upper=21) / 21.0) * 0.20
        + evaluated["stability_score"] * 0.10
        + evaluated["provider_agreement_score"] * 0.10
    ).clip(upper=1.0)

    evaluated["recommendation_flag"] = [
        assign_recommendation_flag(
            confidence_score=score,
            core_signal_available=bool(signal_available),
            backfill_completeness=backfill,
        )
        for score, signal_available, backfill in zip(
            evaluated["confidence_score"],
            evaluated["core_signal_available"],
            evaluated["backfill_completeness"],
        )
    ]

    evaluated["warning_reason"] = np.select(
        [
            ~evaluated["core_signal_available"],
            evaluated["backfill_completeness"] < 0.35,
            evaluated["provider_agreement_score"] < 0.55,
            evaluated["stability_score"] < 0.45,
        ],
        [
            "missing_core_signal",
            "low_backfill_coverage",
            "provider_disagreement",
            "unstable_relative_to_recent_baseline",
        ],
        default="clear_to_show",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(output_path, index=False)
    return evaluated
