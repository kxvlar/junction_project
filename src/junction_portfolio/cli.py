from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from junction_portfolio.api import JunctionClient, pull_sleep_payload, seed_demo_users
from junction_portfolio.v2.compare import compare_runs
from junction_portfolio.config import MissingConfigError, load_config
from junction_portfolio.v1.evaluation import run_evaluation
from junction_portfolio.v2.labels import build_label_table
from junction_portfolio.normalization import build_feature_table
from junction_portfolio.v2.releases import archive_run, get_archive_root
from junction_portfolio.v2.reliability import (
    MODEL_BUNDLE_NAME,
    evaluate_reliability_model,
    score_reliability,
    train_reliability_model,
)
from junction_portfolio.sample_data import generate_sample_payload


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_COHORT_PATH = ARTIFACT_ROOT / "raw" / "junction_cohort.json"
DEFAULT_RAW_PATH = ARTIFACT_ROOT / "raw" / "junction_sleep_payload.json"
DEFAULT_FEATURE_TABLE = ARTIFACT_ROOT / "processed" / "feature_table.csv"
DEFAULT_EVALUATION_TABLE = ARTIFACT_ROOT / "processed" / "evaluation_table.csv"
DEFAULT_LABEL_TABLE = ARTIFACT_ROOT / "processed" / "label_table.csv"
DEFAULT_RELIABILITY_SCORES = ARTIFACT_ROOT / "processed" / "reliability_scores.csv"
DEFAULT_MODEL_PATH = ARTIFACT_ROOT / "models" / MODEL_BUNDLE_NAME
DEFAULT_METRICS_PATH = ARTIFACT_ROOT / "metrics" / "model_metrics.json"
DEFAULT_THRESHOLD_SWEEP = ARTIFACT_ROOT / "metrics" / "threshold_sweep.csv"
DEFAULT_COMPARISON_PATH = ARTIFACT_ROOT / "metrics" / "comparison_v1_v2.json"
DEFAULT_V2_REPORT = ARTIFACT_ROOT / "reports" / "reliability_model_report.md"
ARCHIVE_ROOT = get_archive_root(PROJECT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Junction portfolio CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser("seed-demo-users")
    seed.add_argument("--prefix", default="junction_portfolio")
    seed.add_argument("--providers", default="oura,fitbit,apple_health_kit")
    seed.add_argument("--users-per-provider", type=int, default=2)
    seed.add_argument("--output", type=Path, default=DEFAULT_COHORT_PATH)

    pull = subparsers.add_parser("pull-junction-data")
    pull.add_argument("--cohort", type=Path, default=DEFAULT_COHORT_PATH)
    pull.add_argument("--output", type=Path, default=DEFAULT_RAW_PATH)
    pull.add_argument("--start-date", default="2026-03-01")
    pull.add_argument("--end-date", default="2026-03-14")
    pull.add_argument("--use-sample-data", action="store_true")
    pull.add_argument("--scenario", default="baseline")

    features = subparsers.add_parser("build-feature-table")
    features.add_argument("--input", type=Path, default=DEFAULT_RAW_PATH)
    features.add_argument("--output", type=Path, default=DEFAULT_FEATURE_TABLE)

    evaluation = subparsers.add_parser("run-evaluation")
    evaluation.add_argument("--input", type=Path, default=DEFAULT_FEATURE_TABLE)
    evaluation.add_argument("--output", type=Path, default=DEFAULT_EVALUATION_TABLE)

    report = subparsers.add_parser("render-report")
    report.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION_TABLE)
    report.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_ROOT / "reports" / "recovery_signal_report.md",
    )
    report.add_argument(
        "--mmash-data-dir",
        type=Path,
        default=PROJECT_ROOT / "DataPaper",
    )

    archive_cmd = subparsers.add_parser("archive-run")
    archive_cmd.add_argument("--version", default="v2.0.0-dev")
    archive_cmd.add_argument("--notes", default="Archived run snapshot")

    compare_cmd = subparsers.add_parser("compare-runs")
    compare_cmd.add_argument("--left", type=Path, required=True)
    compare_cmd.add_argument("--right", type=Path, required=True)
    compare_cmd.add_argument("--output", type=Path, default=DEFAULT_COMPARISON_PATH)

    labels = subparsers.add_parser("build-label-table")
    labels.add_argument("--early-feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    labels.add_argument("--mature-feature-table", type=Path, required=True)
    labels.add_argument("--output", type=Path, default=DEFAULT_LABEL_TABLE)

    train = subparsers.add_parser("train-reliability-model")
    train.add_argument("--input", type=Path, default=DEFAULT_LABEL_TABLE)
    train.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)

    score = subparsers.add_parser("score-reliability")
    score.add_argument("--input", type=Path, default=DEFAULT_LABEL_TABLE)
    score.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    score.add_argument("--output", type=Path, default=DEFAULT_RELIABILITY_SCORES)
    score.add_argument("--show-threshold", type=float, default=0.75)
    score.add_argument("--warning-threshold", type=float, default=0.50)

    evaluate = subparsers.add_parser("evaluate-reliability-model")
    evaluate.add_argument("--label-table", type=Path, default=DEFAULT_LABEL_TABLE)
    evaluate.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    evaluate.add_argument("--scores", type=Path, default=DEFAULT_RELIABILITY_SCORES)
    evaluate.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    evaluate.add_argument("--threshold-output", type=Path, default=DEFAULT_THRESHOLD_SWEEP)
    evaluate.add_argument("--comparison-output", type=Path, default=DEFAULT_COMPARISON_PATH)
    evaluate.add_argument("--v1-evaluation", type=Path, default=DEFAULT_EVALUATION_TABLE)

    report_v2 = subparsers.add_parser("render-report-v2")
    report_v2.add_argument("--label-table", type=Path, default=DEFAULT_LABEL_TABLE)
    report_v2.add_argument("--scores", type=Path, default=DEFAULT_RELIABILITY_SCORES)
    report_v2.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    report_v2.add_argument("--threshold-sweep", type=Path, default=DEFAULT_THRESHOLD_SWEEP)
    report_v2.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON_PATH)
    report_v2.add_argument("--output", type=Path, default=DEFAULT_V2_REPORT)
    return parser


def _load_live_client() -> JunctionClient:
    config = load_config(PROJECT_ROOT / ".env")
    return JunctionClient(config=config)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "seed-demo-users":
        client = _load_live_client()
        providers = [item.strip() for item in args.providers.split(",") if item.strip()]
        seeded = seed_demo_users(
            client=client,
            output_path=args.output,
            providers=providers,
            users_per_provider=args.users_per_provider,
            prefix=args.prefix,
        )
        print(f"Seeded {len(seeded)} Junction demo users -> {args.output}")
        return

    if args.command == "pull-junction-data":
        if args.use_sample_data:
            generate_sample_payload(
                output_path=args.output,
                start_date=args.start_date,
                end_date=args.end_date,
                scenario=args.scenario,
            )
            print(f"Wrote deterministic sample payload -> {args.output}")
            return

        client = _load_live_client()
        pull_sleep_payload(
            client=client,
            cohort_path=args.cohort,
            output_path=args.output,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        print(f"Pulled Junction sleep payload -> {args.output}")
        return

    if args.command == "build-feature-table":
        frame = build_feature_table(raw_path=args.input, output_path=args.output)
        print(f"Wrote {len(frame)} feature rows -> {args.output}")
        return

    if args.command == "run-evaluation":
        frame = run_evaluation(feature_table_path=args.input, output_path=args.output)
        print(f"Wrote {len(frame)} evaluated rows -> {args.output}")
        return

    if args.command == "render-report":
        from junction_portfolio.v1.report import render_report

        report_dir = args.output.parent
        report = render_report(
            evaluation_path=args.evaluation,
            report_path=args.output,
            confidence_plot_path=report_dir / "confidence_by_provider.png",
            mix_plot_path=report_dir / "recommendation_mix.png",
            mmash_plot_path=report_dir / "mmash_validation.png",
            mmash_data_dir=args.mmash_data_dir,
        )
        print(f"Wrote report -> {args.output}")
        print(report.splitlines()[0])
        return

    if args.command == "archive-run":
        snapshot = archive_run(
            project_root=PROJECT_ROOT,
            version=args.version,
            command_lines=[" ".join(sys.argv)],
            notes=args.notes,
        )
        print(f"Archived run -> {snapshot}")
        return

    if args.command == "compare-runs":
        comparison = compare_runs(args.left, args.right, output_path=args.output)
        print(f"Wrote comparison -> {args.output}")
        print(json.dumps(comparison["deltas"], indent=2))
        return

    if args.command == "build-label-table":
        frame = build_label_table(
            early_feature_path=args.early_feature_table,
            mature_feature_path=args.mature_feature_table,
            output_path=args.output,
        )
        print(f"Wrote {len(frame)} label rows -> {args.output}")
        return

    if args.command == "train-reliability-model":
        bundle = train_reliability_model(
            label_table_path=args.input,
            model_path=args.model_output,
        )
        print(f"Trained reliability model ({bundle['calibration_method']}) -> {args.model_output}")
        return

    if args.command == "score-reliability":
        frame = score_reliability(
            input_path=args.input,
            model_path=args.model,
            output_path=args.output,
            show_threshold=args.show_threshold,
            warning_threshold=args.warning_threshold,
        )
        print(f"Wrote {len(frame)} reliability scores -> {args.output}")
        return

    if args.command == "evaluate-reliability-model":
        metrics = evaluate_reliability_model(
            label_table_path=args.label_table,
            model_path=args.model,
            scored_path=args.scores,
            metrics_output_path=args.metrics_output,
            threshold_sweep_output_path=args.threshold_output,
            comparison_output_path=args.comparison_output,
            v1_evaluation_path=args.v1_evaluation,
        )
        print(f"Wrote model metrics -> {args.metrics_output}")
        print(f"Unweighted Brier: {metrics['unweighted']['brier']:.4f}")
        return

    if args.command == "render-report-v2":
        from junction_portfolio.v2.report import render_report_v2

        report = render_report_v2(
            label_table_path=args.label_table,
            scored_path=args.scores,
            metrics_path=args.metrics,
            threshold_sweep_path=args.threshold_sweep,
            comparison_path=args.comparison,
            output_path=args.output,
        )
        print(f"Wrote v2 report -> {args.output}")
        print(report.splitlines()[0])
        return

    raise MissingConfigError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
