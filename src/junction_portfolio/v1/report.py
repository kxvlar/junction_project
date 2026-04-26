from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from junction_portfolio.v1.mmash import run_mmash_validation


REPORT_QUESTION = (
    "When is a recovery-style signal trustworthy enough to expose to users "
    "or downstream product logic?"
)


def _as_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    divider = ["---"] * len(headers)
    rows = [headers, divider] + df.astype(str).values.tolist()
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def _plot_provider_confidence(df: pd.DataFrame, output_path: Path) -> None:
    provider_summary = (
        df.groupby("provider")["confidence_score"].mean().sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    provider_summary.plot(kind="bar", ax=ax, color="#1f77b4", alpha=0.85)
    ax.set_title("Average confidence score by provider")
    ax.set_xlabel("Provider")
    ax.set_ylabel("Confidence score")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_recommendation_mix(df: pd.DataFrame, output_path: Path) -> None:
    mix = df["recommendation_flag"].value_counts().reindex(
        ["show", "show_with_warning", "suppress"], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    mix.plot(kind="bar", ax=ax, color=["#2ca02c", "#ff7f0e", "#d62728"], alpha=0.85)
    ax.set_title("Recommendation mix")
    ax.set_xlabel("Recommendation")
    ax.set_ylabel("Rows")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_report(
    evaluation_path: Path,
    report_path: Path,
    confidence_plot_path: Path,
    mix_plot_path: Path,
    mmash_plot_path: Path,
    mmash_data_dir: Path,
) -> str:
    df = pd.read_csv(evaluation_path, parse_dates=["date"])
    report_path.parent.mkdir(parents=True, exist_ok=True)

    _plot_provider_confidence(df, confidence_plot_path)
    _plot_recommendation_mix(df, mix_plot_path)
    mmash_summary = run_mmash_validation(data_dir=mmash_data_dir, figure_path=mmash_plot_path)

    provider_table = (
        df.groupby("provider")
        .agg(
            rows=("user_id", "count"),
            confidence=("confidence_score", "mean"),
            show_rate=("recommendation_flag", lambda col: (col == "show").mean()),
            warning_rate=(
                "recommendation_flag",
                lambda col: (col == "show_with_warning").mean(),
            ),
            suppress_rate=("recommendation_flag", lambda col: (col == "suppress").mean()),
        )
        .round(3)
        .reset_index()
    )

    lowest_confidence = (
        df.sort_values("confidence_score")
        .head(5)[
            [
                "user_id",
                "provider",
                "date",
                "confidence_score",
                "recommendation_flag",
                "warning_reason",
            ]
        ]
        .copy()
    )
    lowest_confidence["date"] = lowest_confidence["date"].dt.date.astype(str)

    report = f"""# Decision-Grade Recovery Signal Report

## Product Question

{REPORT_QUESTION}

## Summary

This case study reframes Junction wearable data work around a product decision rather than a notebook output. The pipeline ingests standardized Junction sleep summaries, normalizes them at the `user_id` / `provider` / `date` grain, and scores each row for whether the recovery-style signal is reliable enough to show.

The confidence policy combines five ingredients:

- recent completeness over the last 7 days
- full-window backfill coverage
- baseline sufficiency over the last 28 days
- stability relative to recent history
- provider agreement when multiple providers report on the same date

## Recommended Product Policy

- `show`: confidence score >= 0.75 and the row has a usable recovery proxy
- `show_with_warning`: confidence score between 0.50 and 0.75
- `suppress`: confidence score < 0.50, low backfill coverage, or no usable recovery proxy

This makes the product conservative by default: missing or contradictory wearables data should degrade the recommendation before it reaches the user.

## Results

### Provider Summary

{_as_markdown_table(provider_table)}

### Lowest-Confidence Rows

{_as_markdown_table(lowest_confidence)}

## Interpretation

The dataset shows that confidence is not just a property of one score field. It is a property of the data pipeline around the score: whether the user has enough recent observations, whether the backfill is complete enough to set a baseline, and whether multiple providers agree when they overlap.

This is the kind of policy layer that turns standardized wearable ingestion into a decision-grade product surface.

## Secondary Validation: MMASH

The MMASH dataset remains a supporting artifact to stress-test the reliability claim with an external source of higher-quality heart-rate variability data. It is not used as the main product dataset, but it is useful for showing why confidence and measurement error need to be first-class concerns.

- Users with complete MMASH rows: {int(mmash_summary['n_users'])}
- True beta of RMSSD -> sleep efficiency: {mmash_summary['beta_true']:.4f}
- Naive beta after simulated wearable-style error: {mmash_summary['beta_naive']:.4f}
- Absolute slope distortion under simulated wearable error: {mmash_summary['slope_distortion_pct']:.1f}%
- Permutation-test p-value: {mmash_summary['frt_p_value']:.4f}

## Limitations

- The default run uses deterministic sample Junction data so the repo is reproducible without credentials.
- The confidence thresholds are designed as a defendable product policy, not a medically validated standard.
- Provider agreement is strongest when the same user has overlapping providers on the same date; single-provider rows fall back to a neutral prior rather than true agreement evidence.

## Artifacts

- ![Confidence by provider]({confidence_plot_path.name})
- ![Recommendation mix]({mix_plot_path.name})
- ![MMASH validation]({mmash_plot_path.name})
"""
    report_path.write_text(report)
    return report
