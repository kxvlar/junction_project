from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


def _artifact_root(run_root: Path) -> Path:
    if (run_root / "artifacts").exists():
        return run_root / "artifacts"
    return run_root


def summarize_run(run_root: Path) -> dict:
    artifact_root = _artifact_root(run_root)
    summary: dict[str, object] = {"run_root": str(run_root)}

    raw_path = artifact_root / "raw" / "junction_sleep_payload.json"
    if raw_path.exists():
        payload = json.loads(raw_path.read_text())
        summary["raw_record_count"] = len(payload.get("records", []))
        summary["raw_source"] = payload.get("source")
        summary["raw_window"] = {
            "start_date": payload.get("start_date"),
            "end_date": payload.get("end_date"),
        }

    feature_path = artifact_root / "processed" / "feature_table.csv"
    if feature_path.exists():
        feature_df = pd.read_csv(feature_path)
        summary["feature_row_count"] = int(len(feature_df))
        if {"provider"}.issubset(feature_df.columns):
            summary["feature_provider_mix"] = (
                feature_df["provider"].value_counts(dropna=False).to_dict()
            )

    evaluation_path = artifact_root / "processed" / "evaluation_table.csv"
    if evaluation_path.exists():
        eval_df = pd.read_csv(evaluation_path)
        summary["evaluation_row_count"] = int(len(eval_df))
        if {"recommendation_flag"}.issubset(eval_df.columns):
            summary["v1_recommendation_mix"] = (
                eval_df["recommendation_flag"].value_counts(dropna=False).to_dict()
            )

    scores_path = artifact_root / "processed" / "reliability_scores.csv"
    if scores_path.exists():
        score_df = pd.read_csv(scores_path)
        summary["reliability_row_count"] = int(len(score_df))
        if {"policy_action_v2"}.issubset(score_df.columns):
            summary["v2_policy_mix"] = (
                score_df["policy_action_v2"].value_counts(dropna=False).to_dict()
            )

    metrics_path = artifact_root / "metrics" / "model_metrics.json"
    if metrics_path.exists():
        summary["model_metrics"] = json.loads(metrics_path.read_text())

    report_v1_path = artifact_root / "reports" / "recovery_signal_report.md"
    report_v2_path = artifact_root / "reports" / "reliability_model_report.md"
    summary["report_files"] = [
        path.name for path in [report_v1_path, report_v2_path] if path.exists()
    ]
    return summary


def compare_runs(left_root: Path, right_root: Path, output_path: Path | None = None) -> dict:
    left = summarize_run(left_root)
    right = summarize_run(right_root)

    comparison = {
        "left": left,
        "right": right,
        "deltas": {
            "raw_record_count": int(right.get("raw_record_count", 0)) - int(left.get("raw_record_count", 0)),
            "feature_row_count": int(right.get("feature_row_count", 0)) - int(left.get("feature_row_count", 0)),
            "evaluation_row_count": int(right.get("evaluation_row_count", 0))
            - int(left.get("evaluation_row_count", 0)),
            "reliability_row_count": int(right.get("reliability_row_count", 0))
            - int(left.get("reliability_row_count", 0)),
        },
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(comparison, indent=2))
    return comparison
