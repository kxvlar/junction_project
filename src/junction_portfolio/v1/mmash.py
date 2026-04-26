from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from junction_portfolio.stats import permutation_test


def compute_rmssd(ibi_seconds: pd.Series | np.ndarray) -> float:
    rr = np.asarray(ibi_seconds, dtype=float) * 1000.0
    rr = rr[(rr >= 375) & (rr <= 2000)]
    if len(rr) < 20:
        return float("nan")
    diffs = np.abs(np.diff(rr))
    valid = np.concatenate([[True], diffs < 200])
    rr = rr[valid]
    if len(rr) < 20:
        return float("nan")
    return float(np.sqrt(np.mean(np.diff(rr) ** 2)))


def load_mmash_feature_table(data_dir: Path) -> pd.DataFrame:
    records: list[dict] = []

    for user_dir in sorted(data_dir.glob("user_*")):
        rr_path = user_dir / "RR.csv"
        sleep_path = user_dir / "sleep.csv"
        questionnaire_path = user_dir / "questionnaire.csv"
        if not rr_path.exists() or not sleep_path.exists() or not questionnaire_path.exists():
            continue

        rr_df = pd.read_csv(rr_path)
        sleep_df = pd.read_csv(sleep_path)
        questionnaire_df = pd.read_csv(questionnaire_path)
        if rr_df.empty or sleep_df.empty or questionnaire_df.empty:
            continue

        sleep = sleep_df.iloc[0]
        questionnaire = questionnaire_df.iloc[0]

        bed_dt = pd.to_datetime(sleep["In Bed Time"], format="%H:%M", errors="coerce")
        wake_dt = pd.to_datetime(sleep["Out Bed Time"], format="%H:%M", errors="coerce")
        night = rr_df[rr_df["day"] == 2].copy()
        night["time_dt"] = pd.to_datetime(night["time"], format="%H:%M:%S", errors="coerce")

        if bed_dt > wake_dt:
            mask = (night["time_dt"] >= bed_dt) | (night["time_dt"] <= wake_dt)
        else:
            mask = (night["time_dt"] >= bed_dt) & (night["time_dt"] <= wake_dt)

        night_rr = night.loc[mask, "ibi_s"].dropna()
        if len(night_rr) < 50:
            night_rr = night["ibi_s"].dropna()

        records.append(
            {
                "user_id": user_dir.name,
                "rmssd_ms": compute_rmssd(night_rr),
                "sleep_efficiency": float(sleep["Efficiency"]),
                "total_sleep_mins": float(sleep["Total Sleep Time (TST)"]),
                "latency_mins": float(sleep["Latency"]),
                "daily_stress": float(questionnaire["Daily_stress"]),
                "psqi": float(questionnaire["Pittsburgh"]),
            }
        )

    frame = pd.DataFrame(records).dropna(subset=["rmssd_ms", "sleep_efficiency"])
    return frame


def run_mmash_validation(
    data_dir: Path,
    figure_path: Path,
) -> dict[str, float | pd.DataFrame]:
    frame = load_mmash_feature_table(data_dir)
    frame["rmssd_true"] = frame["rmssd_ms"]

    model = smf.ols("sleep_efficiency ~ rmssd_true", data=frame).fit()
    beta_true = float(model.params["rmssd_true"])

    reliability = 0.962**2
    mean_bias = -14.97
    rng = np.random.default_rng(42)
    var_true = float(np.var(frame["rmssd_true"]))
    var_error = var_true * (1 / reliability - 1)
    noisy_rmssd = frame["rmssd_true"] + rng.normal(
        loc=mean_bias, scale=np.sqrt(var_error), size=len(frame)
    )
    frame["rmssd_observed"] = noisy_rmssd

    naive_model = smf.ols("sleep_efficiency ~ rmssd_observed", data=frame).fit()
    beta_naive = float(naive_model.params["rmssd_observed"])

    treatment = (frame["rmssd_true"] >= frame["rmssd_true"].median()).astype(int).to_numpy()
    outcome = frame["sleep_efficiency"].to_numpy()
    frt = permutation_test(outcome=outcome, treatment=treatment, n_permutations=5_000)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(
        frame["rmssd_true"],
        frame["sleep_efficiency"],
        s=55,
        color="#1f77b4",
        alpha=0.85,
        edgecolor="white",
    )
    x_values = np.linspace(frame["rmssd_true"].min(), frame["rmssd_true"].max(), 100)
    axes[0].plot(
        x_values,
        model.params["Intercept"] + beta_true * x_values,
        color="#d62728",
        linewidth=2,
        label=f"beta={beta_true:.3f}",
    )
    axes[0].set_title("MMASH secondary validation")
    axes[0].set_xlabel("Polar H7 RMSSD (ms)")
    axes[0].set_ylabel("Sleep efficiency (%)")
    axes[0].legend(fontsize=8)

    axes[1].hist(frt["null_dist"], bins=35, color="#4c72b0", alpha=0.8, edgecolor="white")
    axes[1].axvline(frt["tau_obs"], color="#d62728", linewidth=2)
    axes[1].axvline(-frt["tau_obs"], color="#d62728", linestyle="--", linewidth=2)
    axes[1].set_title(f"Permutation test on MMASH\np={frt['p_value']:.4f}")
    axes[1].set_xlabel("Permuted tau")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_users": float(len(frame)),
        "beta_true": beta_true,
        "beta_naive": beta_naive,
        "slope_distortion_pct": float(
            (abs(beta_naive) - abs(beta_true)) / abs(beta_true) * 100
        ),
        "frt_p_value": float(frt["p_value"]),
        "table": frame,
    }
