from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from junction_portfolio.v2.labels import KEY_COLUMNS, add_reliability_features


SHOW_THRESHOLD = 0.75
WARNING_THRESHOLD = 0.50
MODEL_BUNDLE_NAME = "reliability_model.pkl"


NUMERIC_FEATURES = [
    "completeness_7d",
    "backfill_completeness",
    "baseline_nights_28d",
    "stability_score",
    "provider_agreement_score",
    "provider_agreement_known",
    "core_signal_available",
    "uses_recovery_readiness",
    "average_hrv",
    "recovery_readiness_score",
    "sleep_efficiency_pct",
    "total_sleep_hours",
    "deep_sleep_hours",
    "rem_sleep_hours",
    "awake_hours",
    "hr_average",
    "hr_lowest",
    "days_observed_total",
    "signal_value",
]

CATEGORICAL_FEATURES = ["provider", "source_type"]


def _ensure_reliability_features(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"])
    if "provider_agreement_known" not in working.columns or "signal_value" not in working.columns:
        working = add_reliability_features(working)
    return working


def _make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def _make_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _make_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def _prepare_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    working = _ensure_reliability_features(frame)
    for column in NUMERIC_FEATURES:
        if column not in working.columns:
            working[column] = np.nan
    for column in CATEGORICAL_FEATURES:
        if column not in working.columns:
            working[column] = "unknown"
    if "provider_agreement_score" in working.columns:
        working["provider_agreement_score"] = working["provider_agreement_score"].fillna(0.5)
    return working


def _brier_score(y_true: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None) -> float:
    if weights is None:
        return float(np.mean((y_true - p) ** 2))
    return float(np.average((y_true - p) ** 2, weights=weights))


def _precision_at_show(y_true: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None, threshold: float = SHOW_THRESHOLD) -> float | None:
    mask = p >= threshold
    if mask.sum() == 0:
        return None
    if weights is None:
        return float(y_true[mask].mean())
    return float(np.average(y_true[mask], weights=weights[mask]))


def _coverage_at_show(p: np.ndarray, threshold: float = SHOW_THRESHOLD) -> float:
    return float(np.mean(p >= threshold))


def _bootstrap_metric(
    y_true: np.ndarray,
    p: np.ndarray,
    metric_name: str,
    n_bootstrap: int = 200,
) -> dict[str, float | None]:
    if len(y_true) < 5:
        return {"point": None, "ci_low": None, "ci_high": None}

    rng = np.random.default_rng(42)
    values: list[float] = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(y_true), len(y_true))
        y_sample = y_true[index]
        p_sample = p[index]
        if metric_name == "brier":
            values.append(_brier_score(y_sample, p_sample))
        else:
            precision = _precision_at_show(y_sample, p_sample)
            if precision is not None:
                values.append(precision)
    if not values:
        return {"point": None, "ci_low": None, "ci_high": None}
    point = _brier_score(y_true, p) if metric_name == "brier" else _precision_at_show(y_true, p)
    return {
        "point": point,
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def _fit_propensity_model(frame: pd.DataFrame) -> Pipeline | None:
    target = frame["label_observed"].astype(int)
    if target.nunique() < 2:
        return None
    model = _make_classifier()
    model.fit(_prepare_feature_matrix(frame), target)
    return model


def _choose_calibration_cv(y: pd.Series) -> int | None:
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        return None
    min_count = int(class_counts.min())
    if min_count < 2 or len(y) < 8:
        return None
    return min(3, min_count)


def train_reliability_model(
    label_table_path: Path,
    model_path: Path,
) -> dict:
    frame = pd.read_csv(label_table_path, parse_dates=["date"])
    frame = _prepare_feature_matrix(frame)
    labeled = frame[(frame["label_observed"] == 1) & frame["safe_to_show"].notna()].copy()
    if labeled.empty or labeled["safe_to_show"].nunique() < 2:
        raise ValueError("Need at least two reliability classes in the labeled subset.")

    X = labeled[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = labeled["safe_to_show"].astype(int)

    base_model = _make_classifier()
    base_model.fit(X, y)

    calibration_cv = _choose_calibration_cv(y)
    calibrated_model = None
    calibration_method = "identity"
    if calibration_cv is not None:
        calibrated_model = CalibratedClassifierCV(
            estimator=clone(base_model),
            method="sigmoid",
            cv=calibration_cv,
        )
        calibrated_model.fit(X, y)
        calibration_method = f"sigmoid_cv_{calibration_cv}"

    bundle = {
        "base_model": base_model,
        "calibrated_model": calibrated_model,
        "calibration_method": calibration_method,
        "show_threshold": SHOW_THRESHOLD,
        "warning_threshold": WARNING_THRESHOLD,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "labeled_row_count": int(len(labeled)),
        "positive_rate": float(y.mean()),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return bundle


def _load_bundle(model_path: Path) -> dict:
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def _apply_policy(probability: float, core_signal_available: bool, show_threshold: float, warning_threshold: float) -> str:
    if not core_signal_available:
        return "suppress"
    if probability >= show_threshold:
        return "show"
    if probability >= warning_threshold:
        return "show_with_warning"
    return "suppress"


def score_reliability(
    input_path: Path,
    model_path: Path,
    output_path: Path,
    show_threshold: float = SHOW_THRESHOLD,
    warning_threshold: float = WARNING_THRESHOLD,
) -> pd.DataFrame:
    frame = pd.read_csv(input_path, parse_dates=["date"])
    working = _prepare_feature_matrix(frame)
    bundle = _load_bundle(model_path)

    X = working[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    raw_prob = bundle["base_model"].predict_proba(X)[:, 1]
    if bundle["calibrated_model"] is not None:
        calibrated_prob = bundle["calibrated_model"].predict_proba(X)[:, 1]
    else:
        calibrated_prob = raw_prob

    scored = working.copy()
    scored["p_safe_to_show_raw"] = raw_prob
    scored["p_safe_to_show_calibrated"] = calibrated_prob
    scored["policy_action_v2"] = [
        _apply_policy(prob, bool(core), show_threshold=show_threshold, warning_threshold=warning_threshold)
        for prob, core in zip(scored["p_safe_to_show_calibrated"], scored["core_signal_available"])
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    return scored


def _threshold_sweep(scored: pd.DataFrame) -> pd.DataFrame:
    labeled = scored[(scored["label_observed"] == 1) & scored["safe_to_show"].notna()].copy()
    if labeled.empty:
        return pd.DataFrame(
            columns=[
                "warning_threshold",
                "show_threshold",
                "coverage_show",
                "show_precision",
                "warning_rate",
            ]
        )

    rows: list[dict[str, float]] = []
    y = labeled["safe_to_show"].astype(int).to_numpy()
    p = labeled["p_safe_to_show_calibrated"].to_numpy()

    for warning_threshold in [0.40, 0.50, 0.60]:
        for show_threshold in [0.65, 0.75, 0.85]:
            if show_threshold <= warning_threshold:
                continue
            show_mask = p >= show_threshold
            warning_mask = (p >= warning_threshold) & (p < show_threshold)
            precision = float(y[show_mask].mean()) if show_mask.any() else np.nan
            rows.append(
                {
                    "warning_threshold": warning_threshold,
                    "show_threshold": show_threshold,
                    "coverage_show": float(show_mask.mean()),
                    "show_precision": precision,
                    "warning_rate": float(warning_mask.mean()),
                }
            )
    return pd.DataFrame(rows)


def _leave_one_user_out_summary(frame: pd.DataFrame) -> dict:
    labeled = frame[(frame["label_observed"] == 1) & frame["safe_to_show"].notna()].copy()
    unique_users = labeled["user_id"].astype(str).unique().tolist()
    if len(unique_users) < 2:
        return {"n_folds": 0, "brier": None, "coverage_show": None}

    probabilities: list[float] = []
    outcomes: list[int] = []
    for user_id in unique_users:
        train = labeled[labeled["user_id"].astype(str) != user_id]
        test = labeled[labeled["user_id"].astype(str) == user_id]
        if train.empty or test.empty or train["safe_to_show"].nunique() < 2:
            continue
        model = _make_classifier()
        train_features = _prepare_feature_matrix(train)
        test_features = _prepare_feature_matrix(test)
        model.fit(
            train_features[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
            train["safe_to_show"].astype(int),
        )
        fold_prob = model.predict_proba(test_features[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
        probabilities.extend(fold_prob.tolist())
        outcomes.extend(test["safe_to_show"].astype(int).tolist())

    if not probabilities:
        return {"n_folds": 0, "brier": None, "coverage_show": None}

    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    return {
        "n_folds": len(unique_users),
        "brier": _brier_score(y, p),
        "coverage_show": _coverage_at_show(p),
    }


def _merge_v1_v2(v1_path: Path | None, scored: pd.DataFrame, output_path: Path) -> dict:
    if v1_path is None or not v1_path.exists():
        comparison = {"available": False}
        output_path.write_text(json.dumps(comparison, indent=2))
        return comparison

    v1 = pd.read_csv(v1_path, parse_dates=["date"])
    scored = scored.copy()
    scored["date"] = pd.to_datetime(scored["date"])
    merged = v1.merge(
        scored[KEY_COLUMNS + ["policy_action_v2", "p_safe_to_show_calibrated"]],
        on=KEY_COLUMNS,
        how="inner",
    )
    if merged.empty:
        comparison = {"available": False, "reason": "no_shared_rows"}
    else:
        comparison = {
            "available": True,
            "shared_rows": int(len(merged)),
            "v1_recommendation_mix": merged["recommendation_flag"].value_counts(dropna=False).to_dict(),
            "v2_policy_mix": merged["policy_action_v2"].value_counts(dropna=False).to_dict(),
            "changed_action_rate": float(
                (merged["recommendation_flag"] != merged["policy_action_v2"]).mean()
            ),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2))
    return comparison


def evaluate_reliability_model(
    label_table_path: Path,
    model_path: Path,
    scored_path: Path,
    metrics_output_path: Path,
    threshold_sweep_output_path: Path,
    comparison_output_path: Path,
    v1_evaluation_path: Path | None = None,
) -> dict:
    label_frame = pd.read_csv(label_table_path, parse_dates=["date"])
    scored = pd.read_csv(scored_path, parse_dates=["date"])
    merged = label_frame.merge(
        scored[KEY_COLUMNS + ["p_safe_to_show_raw", "p_safe_to_show_calibrated", "policy_action_v2"]],
        on=KEY_COLUMNS,
        how="left",
    )

    labeled = merged[(merged["label_observed"] == 1) & merged["safe_to_show"].notna()].copy()
    y = labeled["safe_to_show"].astype(int).to_numpy() if not labeled.empty else np.asarray([], dtype=int)
    p = (
        labeled["p_safe_to_show_calibrated"].astype(float).to_numpy()
        if not labeled.empty
        else np.asarray([], dtype=float)
    )

    propensity_model = _fit_propensity_model(_prepare_feature_matrix(merged))
    ipw_weights = None
    if propensity_model is not None and not labeled.empty:
        propensity = propensity_model.predict_proba(_prepare_feature_matrix(merged))[:, 1]
        merged["label_propensity"] = np.clip(propensity, 0.05, 0.95)
        labeled = merged[(merged["label_observed"] == 1) & merged["safe_to_show"].notna()].copy()
        ipw_weights = 1.0 / labeled["label_propensity"].to_numpy(dtype=float)
        p = labeled["p_safe_to_show_calibrated"].astype(float).to_numpy()
        y = labeled["safe_to_show"].astype(int).to_numpy()

    if labeled.empty:
        raise ValueError("No labeled rows available for evaluation.")

    unweighted = {
        "n_labeled": int(len(labeled)),
        "positive_rate": float(y.mean()),
        "brier": _brier_score(y, p),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "show_precision": _precision_at_show(y, p),
        "coverage_show": _coverage_at_show(p),
        "brier_bootstrap": _bootstrap_metric(y, p, "brier"),
        "show_precision_bootstrap": _bootstrap_metric(y, p, "precision"),
    }

    if ipw_weights is not None:
        ipw = {
            "brier": _brier_score(y, p, ipw_weights),
            "show_precision": _precision_at_show(y, p, ipw_weights),
            "coverage_show": float(np.average((p >= SHOW_THRESHOLD).astype(float), weights=ipw_weights)),
        }
    else:
        ipw = {"brier": None, "show_precision": None, "coverage_show": None}

    if ipw_weights is not None:
        q = p
        r = labeled["label_observed"].astype(int).to_numpy()
        y_obs = y
        pi = np.clip(labeled["label_propensity"].to_numpy(dtype=float), 0.05, 0.95)
        aipw_terms = q + r / pi * (y_obs - q)
        show_mask = (p >= SHOW_THRESHOLD).astype(float)
        aipw = {
            "estimated_safe_to_show_rate": float(np.mean(aipw_terms)),
            "estimated_show_policy_value": float(
                np.sum(show_mask * aipw_terms) / np.sum(show_mask)
            )
            if show_mask.sum() > 0
            else None,
        }
    else:
        aipw = {"estimated_safe_to_show_rate": None, "estimated_show_policy_value": None}

    threshold_sweep = _threshold_sweep(merged)
    threshold_sweep_output_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_sweep.to_csv(threshold_sweep_output_path, index=False)

    comparison = _merge_v1_v2(v1_evaluation_path, scored, comparison_output_path)

    metrics = {
        "training_bundle": _load_bundle(model_path)["calibration_method"],
        "unweighted": unweighted,
        "ipw": ipw,
        "aipw": aipw,
        "leave_one_user_out": _leave_one_user_out_summary(merged),
        "comparison_v1_v2": comparison,
    }
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.write_text(json.dumps(metrics, indent=2))
    return metrics
