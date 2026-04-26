from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _as_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    headers = list(df.columns)
    divider = ["---"] * len(headers)
    rows = [headers, divider] + df.astype(str).values.tolist()
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def _plot_calibration(scored: pd.DataFrame, output_path: Path) -> None:
    labeled = scored[(scored["label_observed"] == 1) & scored["safe_to_show"].notna()].copy()
    if labeled.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No labeled rows for calibration plot", ha="center", va="center")
        ax.axis("off")
    else:
        labeled["bin"] = pd.qcut(
            labeled["p_safe_to_show_calibrated"].rank(method="first"),
            q=min(5, len(labeled)),
            labels=False,
            duplicates="drop",
        )
        grouped = (
            labeled.groupby("bin")
            .agg(
                predicted=("p_safe_to_show_calibrated", "mean"),
                observed=("safe_to_show", lambda col: pd.Series(col).astype(float).mean()),
            )
            .reset_index(drop=True)
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
        ax.scatter(grouped["predicted"], grouped["observed"], s=70, color="#1f77b4")
        ax.set_title("Calibration plot")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed safe-to-show rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_coverage_risk(threshold_sweep: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if threshold_sweep.empty:
        ax.text(0.5, 0.5, "No threshold sweep available", ha="center", va="center")
        ax.axis("off")
    else:
        for warning_threshold, group in threshold_sweep.groupby("warning_threshold"):
            ax.plot(
                group["coverage_show"],
                group["show_precision"],
                marker="o",
                label=f"warning={warning_threshold:.2f}",
            )
        ax.set_title("Coverage-risk tradeoff")
        ax.set_xlabel("Coverage of show decisions")
        ax.set_ylabel("Show precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_report_v2(
    label_table_path: Path,
    scored_path: Path,
    metrics_path: Path,
    threshold_sweep_path: Path,
    comparison_path: Path,
    output_path: Path,
) -> str:
    labeled = pd.read_csv(label_table_path, parse_dates=["date"])
    scored = pd.read_csv(scored_path, parse_dates=["date"])
    merged = labeled.merge(
        scored[["user_id", "provider", "date", "p_safe_to_show_calibrated", "policy_action_v2"]],
        on=["user_id", "provider", "date"],
        how="left",
    )
    metrics = json.loads(metrics_path.read_text())
    threshold_sweep = pd.read_csv(threshold_sweep_path) if threshold_sweep_path.exists() else pd.DataFrame()
    comparison = json.loads(comparison_path.read_text()) if comparison_path.exists() else {"available": False}

    report_dir = output_path.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    calibration_plot = report_dir / "v2_calibration.png"
    coverage_plot = report_dir / "v2_coverage_risk.png"
    _plot_calibration(merged, calibration_plot)
    _plot_coverage_risk(threshold_sweep, coverage_plot)

    labeled_rows = merged[(merged["label_observed"] == 1) & merged["safe_to_show"].notna()].copy()
    failure_cases = pd.DataFrame()
    if not labeled_rows.empty:
        labeled_rows["error"] = (
            labeled_rows["safe_to_show"].astype(float) - labeled_rows["p_safe_to_show_calibrated"].astype(float)
        ).abs()
        failure_cases = labeled_rows.sort_values("error", ascending=False).head(5)[
            [
                "user_id",
                "provider",
                "date",
                "safe_to_show",
                "p_safe_to_show_calibrated",
                "policy_action_v2",
                "error",
            ]
        ].copy()
        failure_cases["date"] = failure_cases["date"].dt.date.astype(str)

    threshold_preview = threshold_sweep.sort_values(
        ["show_precision", "coverage_show"], ascending=[False, False]
    ).head(5)

    report = f"""# Reliability Model Report (V2)

## Iteration Summary

V1 was a heuristic proof of concept built around hand-tuned confidence weights. V2 extends that work into a labeled reliability-model pipeline that predicts whether an early Junction recovery-style signal is safe to show before the backfill matures.

This version keeps the causal framing narrow: IPW and AIPW are used only to correct evaluation when label observability is selective. The model itself remains an interpretable regularized logistic regression.

## Data and Labels

- Total early rows: {len(merged)}
- Labeled rows with mature verification: {int((merged['label_observed'] == 1).sum())}
- Positive `safe_to_show` rate among labeled rows: {metrics['unweighted']['positive_rate']:.3f}

## Model Performance

- Unweighted Brier score: {metrics['unweighted']['brier']:.4f}
- Unweighted log loss: {metrics['unweighted']['log_loss']:.4f}
- Show precision at default threshold: {metrics['unweighted']['show_precision']}
- Coverage of show decisions: {metrics['unweighted']['coverage_show']:.4f}
- IPW-corrected Brier score: {metrics['ipw']['brier']}
- IPW-corrected show precision: {metrics['ipw']['show_precision']}
- AIPW estimated safe-to-show rate: {metrics['aipw']['estimated_safe_to_show_rate']}
- Leave-one-user-out Brier score: {metrics['leave_one_user_out']['brier']}

## V1 vs V2

{json.dumps(comparison, indent=2)}

## Threshold Sweep

{_as_markdown_table(threshold_preview.round(4))}

## Failure Cases

{_as_markdown_table(failure_cases.round(4) if not failure_cases.empty else failure_cases)}

## Narrative

This iteration strengthens the original project in three ways:

1. It replaces arbitrary weighted scoring with a learned reliability probability.
2. It separates label construction from scoring, which makes the target explicit.
3. It evaluates the model under selective observability instead of pretending all rows are equally verifiable.

## Figures

- ![Calibration]({calibration_plot.name})
- ![Coverage risk]({coverage_plot.name})
"""
    output_path.write_text(report)
    return report
