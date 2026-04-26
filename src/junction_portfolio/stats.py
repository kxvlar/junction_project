from __future__ import annotations

import numpy as np


def diff_in_means(outcome: np.ndarray, treatment: np.ndarray) -> float:
    treated = outcome[treatment == 1]
    control = outcome[treatment == 0]
    if len(treated) == 0 or len(control) == 0:
        raise ValueError("Both treatment groups must have at least one observation.")
    return float(treated.mean() - control.mean())


def permutation_test(
    outcome: np.ndarray,
    treatment: np.ndarray,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> dict[str, np.ndarray | float]:
    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    rng = np.random.default_rng(seed)

    tau_obs = diff_in_means(outcome, treatment)
    null_dist = np.empty(n_permutations, dtype=float)

    for index in range(n_permutations):
        shuffled = rng.permutation(treatment)
        null_dist[index] = diff_in_means(outcome, shuffled)

    p_value = float(np.mean(np.abs(null_dist) >= abs(tau_obs)))
    return {
        "tau_obs": tau_obs,
        "null_dist": null_dist,
        "p_value": p_value,
    }
