# Junction Wearables Reliability Case Study

This repo answers one product question for a Junction-style wearables system:

**When is a recovery-style signal trustworthy enough to show to users or downstream product logic?**

Instead of presenting health-data analysis as a loose notebook, this project turns Junction sleep summaries into a decision-grade evaluation dataset with:

- standardized daily features by `user_id`, `provider`, and `date`
- missingness and backfill coverage checks
- stability and provider-agreement scoring
- a confidence score and product recommendation flag
- a polished report that explains the policy and its tradeoffs

The MMASH dataset remains in the repo as a secondary validation source for measurement-error and causal-risk discussion. It is not the headline artifact.

## Repo Shape

The repo is intentionally organized as:

- `v1`: heuristic confidence-policy pipeline
- `v2`: learned reliability-model extension with debiased evaluation
- `archives`: immutable snapshots stored outside the active repo

The code stays in one package under `src/junction_portfolio/`, but the outputs are split by role:

- canonical v1 artifacts live in `artifacts/raw`, `artifacts/processed`, and `artifacts/reports`
- canonical v2 artifacts live in `artifacts/processed`, `artifacts/metrics`, `artifacts/models`, and `artifacts/reports`
- one-off early/mature sample construction files live in `artifacts/staging/v2_training/`

If you are reviewing the project, the shortest path is:

- v1 report: `artifacts/reports/recovery_signal_report.md`
- v2 report: `artifacts/reports/reliability_model_report.md`
- methodology note: `docs/reliability_model_note.tex`

## Quick Start

Use the local virtualenv that already exists in this repo:

```bash
source .venv/bin/activate
export PYTHONPATH=src
```

Generate the offline sample pipeline and render the report:

```bash
python -m junction_portfolio.cli pull-junction-data --use-sample-data
python -m junction_portfolio.cli build-feature-table
python -m junction_portfolio.cli run-evaluation
python -m junction_portfolio.cli render-report
```

That produces:

- `artifacts/raw/junction_sleep_payload.json`
- `artifacts/processed/feature_table.csv`
- `artifacts/processed/evaluation_table.csv`
- `artifacts/reports/recovery_signal_report.md`
- `artifacts/reports/confidence_by_provider.png`
- `artifacts/reports/recommendation_mix.png`
- `artifacts/reports/mmash_validation.png`

## Live Junction Sandbox Flow

The live flow uses current Junction terminology and base URLs. Configure `.env` from `.env.example`, then:

```bash
python -m junction_portfolio.cli seed-demo-users --providers oura,fitbit,apple_health_kit
python -m junction_portfolio.cli pull-junction-data --start-date 2026-03-01 --end-date 2026-03-28
python -m junction_portfolio.cli build-feature-table
python -m junction_portfolio.cli run-evaluation
python -m junction_portfolio.cli render-report
```

The live commands use:

- user creation
- demo-provider connections in sandbox
- standardized sleep summaries
- provider attribution and recovery/readiness-oriented fields

## Command Surface

- `seed-demo-users`: create Junction sandbox users and attach demo providers
- `pull-junction-data`: pull Junction sleep summaries or generate deterministic sample data
- `build-feature-table`: normalize raw payloads into daily provider-level features
- `run-evaluation`: calculate completeness, stability, provider agreement, confidence, and recommendation flags
- `render-report`: build the main portfolio artifact and MMASH secondary validation outputs

## Project Layout

```text
src/junction_portfolio/
  api.py
  cli.py
  config.py
  normalization.py
  sample_data.py
  stats.py
  v1/
    evaluation.py
    mmash.py
    report.py
  v2/
    compare.py
    labels.py
    releases.py
    reliability.py
    report.py
tests/
  v1/
  v2/
docs/
DataPaper/
```

## Main Artifact

The report in `artifacts/reports/recovery_signal_report.md` is the artifact meant for a hiring review. It states:

- the product question
- the scoring methodology
- the validation results
- the limitations
- the recommended product policy

## Canonical Outputs

### V1

- `artifacts/raw/junction_sleep_payload.json`
- `artifacts/processed/feature_table.csv`
- `artifacts/processed/evaluation_table.csv`
- `artifacts/reports/recovery_signal_report.md`

### V2

- `artifacts/processed/label_table.csv`
- `artifacts/processed/reliability_scores.csv`
- `artifacts/metrics/model_metrics.json`
- `artifacts/metrics/threshold_sweep.csv`
- `artifacts/metrics/comparison_v1_v2.json`
- `artifacts/reports/reliability_model_report.md`

### Staging

These are regenerable helper files used to construct the v2 sample early-vs-mature training setup:

- `artifacts/staging/v2_training/raw/v2_early_payload.json`
- `artifacts/staging/v2_training/raw/v2_mature_payload.json`
- `artifacts/staging/v2_training/processed/v2_early_feature_table.csv`
- `artifacts/staging/v2_training/processed/v2_mature_feature_table.csv`
- `artifacts/staging/v2_training/processed/v2_early_evaluation_table.csv`

## V2 Extension

V2 preserves the v1 heuristic policy and adds a hybrid causal + ML reliability layer.

New commands:

```bash
python -m junction_portfolio.cli archive-run --version v2.0.0-dev --notes "Snapshot before or after a major iteration"
python -m junction_portfolio.cli build-label-table --early-feature-table artifacts/staging/v2_training/processed/v2_early_feature_table.csv --mature-feature-table artifacts/staging/v2_training/processed/v2_mature_feature_table.csv
python -m junction_portfolio.cli train-reliability-model
python -m junction_portfolio.cli score-reliability
python -m junction_portfolio.cli evaluate-reliability-model --v1-evaluation artifacts/staging/v2_training/processed/v2_early_evaluation_table.csv
python -m junction_portfolio.cli render-report-v2
```

V2 outputs:

- `artifacts/processed/label_table.csv`
- `artifacts/processed/reliability_scores.csv`
- `artifacts/metrics/model_metrics.json`
- `artifacts/metrics/threshold_sweep.csv`
- `artifacts/metrics/comparison_v1_v2.json`
- `artifacts/reports/reliability_model_report.md`

Snapshots are archived outside the active repo in:

- `/Users/kevalamin/Documents/Playground/junction_portfolio_work_archives/`
