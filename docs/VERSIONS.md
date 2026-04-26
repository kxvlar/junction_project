# Version Guide

This repo has two project layers and one archive layer.

## V1

V1 is the heuristic confidence-policy version of the project.

Question:
- When is a recovery-style signal trustworthy enough to show?

Method:
- engineer completeness, baseline, stability, and provider-agreement features
- combine them into a hand-tuned confidence score
- map that score to `show`, `show_with_warning`, or `suppress`

Best artifact:
- `artifacts/reports/recovery_signal_report.md`

Canonical files:
- `artifacts/raw/junction_sleep_payload.json`
- `artifacts/processed/feature_table.csv`
- `artifacts/processed/evaluation_table.csv`

## V2

V2 is the hybrid causal + ML extension.

Question:
- Can we predict whether an early signal is safe to show before the backfill matures?

Method:
- align an early and mature snapshot
- create `label_observed` and `safe_to_show`
- train a regularized logistic regression reliability model
- calibrate probabilities
- use IPW/AIPW to correct evaluation under selective label observability

Best artifact:
- `artifacts/reports/reliability_model_report.md`

Canonical files:
- `artifacts/processed/label_table.csv`
- `artifacts/processed/reliability_scores.csv`
- `artifacts/metrics/model_metrics.json`
- `artifacts/metrics/threshold_sweep.csv`
- `artifacts/metrics/comparison_v1_v2.json`
- `artifacts/models/reliability_model.pkl`

## Staging

The repo keeps only a small amount of regenerable staging data for the v2 sample early-vs-mature workflow:

- `artifacts/staging/v2_training/raw/`
- `artifacts/staging/v2_training/processed/`

These are not the main portfolio outputs. They exist so the v2 label-construction example is reproducible.

## Archives

Immutable run snapshots live outside the active repo:

- `/Users/kevalamin/Documents/Playground/junction_portfolio_work_archives/`

Use those for historical versions, not the active workspace.
