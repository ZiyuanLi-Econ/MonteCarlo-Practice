#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCT (2014) DGP2 non-normal error Monte Carlo extension.

Purpose
-------
Keep the DGP2 running variable, regression function, treatment effect, sample size,
and bandwidth rule fixed. Change only the standardized error distribution U_i in

    Y_i = mu_2(X_i) + 0.1295 * U_i,
    E[U_i] = 0, Var(U_i) = 1.

Design
------
The error distribution is not standardized within each simulated sample. Each
scenario uses its theoretical centering and scale. The Monte Carlo is paired in
the running variable: within replication r, N0-N3 use the same X_i sample. N0
uses the baseline normal-error random stream; N1-N3 use their own error streams,
so changing non-normality does not change the running-variable draw.

Scenarios
---------
N0: U ~ N(0, 1)
N1: U = (chi-square_1 - 1) / sqrt(2), severe skewness
N2: U = t_3 / sqrt(3), extreme tails
N3: U = (Z + 10 B - 0.2) / sqrt(2.96), one-sided contamination

Methods
-------
1. Conventional MSE: rdrobust conventional CI with bwselect='mserd'
2. RBC MSE:          rdrobust robust bias-corrected CI with normal critical value

Run examples
------------
Quick check:
    python cct_2_2_nonnormal_paired.py --reps 20

Main run:
    python cct_2_2_nonnormal_paired.py

Outputs
-------
    If this file is stored inside a code/ folder, output is written to the
    sibling results/ folder. If the file is run as a standalone script,
    output is written to cct_2_2_nonnormal_paired_results/ next to the script.

    run_config_nonnormal_paired.txt
    results or cct_2_2_nonnormal_paired_results/
        nonnormal_raw.csv
        nonnormal_summary.csv
        nonnormal_error_diagnostics.csv
        nonnormal_error_summary.csv
        nonnormal_pairwise_checks.csv
        figures/
            dgp2_curve.png
            error_distribution_by_scenario.png
            coverage_by_scenario_method.png
            interval_length_by_scenario_method.png
            rmse_by_scenario_method.png
            coverage_length_tradeoff.png
            rbc_tstat_N0.png ... rbc_tstat_N3.png
            rbc_tstat_by_scenario.png
            rbc_noncoverage_asymmetry.png
            h_left_distribution_by_scenario.png
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# Global constants: CCT DGP2
# -----------------------------
TRUE_TAU = 0.26 - 3.71
SIGMA_EPS = 0.1295
CUTOFF = 0.0
DEFAULT_REPS = 5000
DEFAULT_N = 500

SCENARIO_ORDER = [
    "N0",
    "N1",
    "N2",
    "N3",
]

SCENARIO_DISPLAY = {
    "N0": "N0 Normal",
    "N1": "N1 Severe skewness",
    "N2": "N2 Extreme t3 tails",
    "N3": "N3 One-sided contamination",
}

METHOD_ORDER = [
    "conventional_mserd",
    "rbc_mserd",
]

METHOD_DISPLAY = {
    "conventional_mserd": "Conventional MSE",
    "rbc_mserd": "RBC MSE",
}

SCENARIO_TICK_LABELS = {
    "N0": "N0\nNormal",
    "N1": "N1\nSevere skewness",
    "N2": "N2\nExtreme t3 tails",
    "N3": "N3\nOne-sided contamination",
}

METHOD_COLORS = {
    "conventional_mserd": "#4C78A8",
    "rbc_mserd": "#F58518",
}

METHOD_MARKERS = {
    "conventional_mserd": "o",
    "rbc_mserd": "s",
}

RDROBUST_ROWS = {
    "conventional_mserd": "Conventional",
    "rbc_mserd": "Robust",
}


def default_output_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    if script_dir.name.lower() == "code":
        return script_dir.parent / "results"
    return script_dir / f"{Path(__file__).resolve().stem}_results"


DEFAULT_OUTDIR = default_output_dir()


RAW_COLUMNS = [
    "scenario",
    "scenario_label",
    "rep",
    "method",
    "method_label",
    "success",
    "error",
    "tau_true",
    "estimate",
    "se",
    "t_stat",
    "ci_lower",
    "ci_upper",
    "cover",
    "ci_length",
    "miss_below_true",
    "miss_above_true",
    "h_left",
    "h_right",
    "b_left",
    "b_right",
    "N_left",
    "N_right",
    "N_h_left",
    "N_h_right",
    "N_b_left",
    "N_b_right",
    "M_left",
    "M_right",
    "bwselect",
    "vce",
    "nnmatch",
    "kernel",
    "p",
    "q",
    "ci_method",
]

ERROR_DIAG_COLUMNS = [
    "scenario",
    "scenario_label",
    "rep",
    "u_mean",
    "u_sd",
    "u_skew",
    "u_excess_kurtosis",
    "u_q01",
    "u_q05",
    "u_q50",
    "u_q95",
    "u_q99",
    "contam_share",
]

SUMMARY_COLUMNS = [
    "scenario",
    "method",
    "reps_success",
    "fail_rate",
    "EC_percent",
    "coverage_error_pp",
    "coverage_mcse",
    "IL",
    "avg_estimate",
    "bias",
    "abs_bias",
    "median_bias",
    "trimmed_bias_1pct",
    "RMSE",
    "avg_se",
    "sd_estimate",
    "se_to_sd_ratio",
    "miss_below_true",
    "miss_above_true",
    "avg_h_left",
    "avg_h_right",
    "avg_b_left",
    "avg_b_right",
    "t_mean",
    "t_sd",
    "t_skew",
    "t_excess_kurtosis",
    "t_q025",
    "t_q975",
]

PAIRWISE_COLUMNS = [
    "scenario",
    "comparison",
    "paired_reps",
    "EC_diff_pp",
    "IL_diff",
    "bias_diff",
    "abs_bias_diff",
    "RMSE_diff",
    "se_to_sd_ratio_diff",
]


@dataclass(frozen=True)
class SimConfig:
    reps: int = DEFAULT_REPS
    n: int = DEFAULT_N
    seed: int = 12345
    outdir: Path = DEFAULT_OUTDIR
    scenarios: Tuple[str, ...] = tuple(SCENARIO_ORDER)
    bwselect: str = "mserd"
    vce: str = "nn"
    nnmatch: int = 3
    kernel: str = "tri"
    p: int = 1
    q: int = 2
    level: float = 95.0
    masspoints: str = "off"
    progress_every: int = 100


# -----------------------------
# DGP2 and error scenarios
# -----------------------------
def mu_dgp2(x: np.ndarray) -> np.ndarray:
    """CCT supplementary material Model 2 regression function."""
    x = np.asarray(x)
    left = (
        3.71
        + 2.30 * x
        + 3.28 * x**2
        + 1.45 * x**3
        + 0.23 * x**4
        + 0.03 * x**5
    )
    right = (
        0.26
        + 18.49 * x
        - 54.81 * x**2
        + 74.30 * x**3
        - 45.02 * x**4
        + 9.83 * x**5
    )
    return np.where(x < 0, left, right)


def draw_standardized_u(
    scenario: str,
    n: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Draw U with theoretical mean 0 and variance 1 for each scenario."""
    contam_share = np.nan

    if scenario == "N0":
        u = rng.normal(loc=0.0, scale=1.0, size=n)

    elif scenario == "N1":
        # E[chi2_1] = 1, Var[chi2_1] = 2.
        u = (rng.chisquare(df=1, size=n) - 1.0) / math.sqrt(2.0)

    elif scenario == "N2":
        # Var(t_3) = 3/(3-2) = 3.
        u = rng.standard_t(df=3, size=n) / math.sqrt(3.0)

    elif scenario == "N3":
        # B ~ Bernoulli(.02), Z ~ N(0,1), V=Z+10B; E[V]=.2, Var(V)=2.96.
        b = rng.random(n) < 0.02
        z = rng.normal(loc=0.0, scale=1.0, size=n)
        v = z + 10.0 * b.astype(float)
        u = (v - 0.2) / math.sqrt(2.96)
        contam_share = float(b.mean())

    else:
        raise ValueError(f"Unknown non-normal scenario: {scenario}")

    return u, {"contam_share": contam_share}


def moment_skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    sd = float(np.std(x, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    z = (x - float(np.mean(x))) / sd
    return float(np.mean(z**3))


def moment_excess_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    sd = float(np.std(x, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    z = (x - float(np.mean(x))) / sd
    return float(np.mean(z**4) - 3.0)


def error_diagnostics(
    scenario: str,
    rep: int,
    u: np.ndarray,
    extra: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "scenario": scenario,
        "scenario_label": SCENARIO_DISPLAY[scenario],
        "rep": rep,
        "u_mean": float(np.mean(u)),
        "u_sd": float(np.std(u, ddof=1)),
        "u_skew": moment_skew(u),
        "u_excess_kurtosis": moment_excess_kurtosis(u),
        "u_q01": float(np.quantile(u, 0.01)),
        "u_q05": float(np.quantile(u, 0.05)),
        "u_q50": float(np.quantile(u, 0.50)),
        "u_q95": float(np.quantile(u, 0.95)),
        "u_q99": float(np.quantile(u, 0.99)),
        "contam_share": extra.get("contam_share", np.nan),
    }


def draw_dataset(
    n: int,
    scenario: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Draw one simulated DGP2 dataset under a non-normal error scenario."""
    x = 2.0 * rng.beta(2.0, 4.0, size=n) - 1.0
    u, extra = draw_standardized_u(scenario, n, rng)
    return make_dataset_from_x(x=x, scenario=scenario, u=u, extra=extra)


def make_dataset_from_x(
    x: np.ndarray,
    scenario: str,
    u: np.ndarray,
    extra: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Construct one non-normal scenario from a fixed running-variable sample."""
    if extra is None:
        extra = {"contam_share": np.nan}
    eps = SIGMA_EPS * u
    y = mu_dgp2(x) + eps
    return y, x, u, extra


# -----------------------------
# rdrobust wrapper utilities
# -----------------------------
def import_rdrobust():
    """Import rdrobust with a helpful error message."""
    try:
        from rdrobust import rdrobust  # type: ignore
        return rdrobust
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The Python package 'rdrobust' is required. Install it manually, e.g.\n"
            "    pip install rdrobust\n"
            "This script does not auto-install packages."
        ) from exc


def first_value(obj: Any) -> float:
    """Safely convert a pandas Series/DataFrame/scalar cell to float."""
    if isinstance(obj, pd.DataFrame):
        return float(obj.iloc[0, 0])
    if isinstance(obj, pd.Series):
        return float(obj.iloc[0])
    arr = np.asarray(obj)
    return float(arr.reshape(-1)[0])


def safe_attr(fit: Any, name: str, default: Any = np.nan) -> Any:
    return getattr(fit, name, default)


def extract_bandwidth(fit: Any, row: str, col: str) -> float:
    bws = safe_attr(fit, "bws", None)
    if bws is None:
        return np.nan
    try:
        return float(bws.loc[row, col])
    except Exception:
        return np.nan


def extract_count_pair(fit: Any, name: str) -> Tuple[float, float]:
    value = safe_attr(fit, name, None)
    if value is None:
        return np.nan, np.nan
    try:
        return float(value[0]), float(value[1])
    except Exception:
        return np.nan, np.nan


def extract_estimate_and_se(fit: Any, rd_row: str) -> Tuple[float, float]:
    coef = safe_attr(fit, "coef")
    se = safe_attr(fit, "se")
    estimate = first_value(coef.loc[rd_row])
    std_error = first_value(se.loc[rd_row])
    return estimate, std_error


def extract_ci(fit: Any, rd_row: str) -> Tuple[float, float]:
    ci = safe_attr(fit, "ci")
    return float(ci.loc[rd_row].iloc[0]), float(ci.loc[rd_row].iloc[1])


def rdrobust_call(
    *,
    y: np.ndarray,
    x: np.ndarray,
    cfg: SimConfig,
    bwselect: Optional[str] = None,
    h: Optional[List[float]] = None,
    b: Optional[List[float]] = None,
) -> Any:
    rdrobust = import_rdrobust()
    kwargs: Dict[str, Any] = dict(
        y=y,
        x=x,
        c=CUTOFF,
        p=cfg.p,
        q=cfg.q,
        kernel=cfg.kernel,
        vce=cfg.vce,
        nnmatch=cfg.nnmatch,
        level=cfg.level,
        masspoints=cfg.masspoints,
        all=True,
    )
    if h is not None:
        kwargs["h"] = h
    if b is not None:
        kwargs["b"] = b
    if h is None:
        kwargs["bwselect"] = bwselect or cfg.bwselect
    return rdrobust(**kwargs)


def extract_fit_row(
    *,
    fit: Any,
    scenario: str,
    rep: int,
    method: str,
    cfg: SimConfig,
    ci_lower_override: Optional[float] = None,
    ci_upper_override: Optional[float] = None,
    ci_method: str = "normal_approx",
    true_tau: float = TRUE_TAU,
    error: str = "",
) -> Dict[str, Any]:
    """Extract one method row from an rdrobust result object."""
    rd_row = RDROBUST_ROWS[method]
    estimate, std_error = extract_estimate_and_se(fit, rd_row)

    if ci_lower_override is None or ci_upper_override is None:
        ci_lower, ci_upper = extract_ci(fit, rd_row)
    else:
        ci_lower, ci_upper = float(ci_lower_override), float(ci_upper_override)

    N_left, N_right = extract_count_pair(fit, "N")
    N_h_left, N_h_right = extract_count_pair(fit, "N_h")
    N_b_left, N_b_right = extract_count_pair(fit, "N_b")
    M_left, M_right = extract_count_pair(fit, "M")

    cover = ci_lower <= true_tau <= ci_upper
    miss_below_true = ci_upper < true_tau
    miss_above_true = ci_lower > true_tau
    t_stat = (estimate - true_tau) / std_error if std_error and np.isfinite(std_error) else np.nan

    return {
        "scenario": scenario,
        "scenario_label": SCENARIO_DISPLAY[scenario],
        "rep": rep,
        "method": method,
        "method_label": METHOD_DISPLAY[method],
        "success": True,
        "error": error,
        "tau_true": true_tau,
        "estimate": estimate,
        "se": std_error,
        "t_stat": t_stat,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cover": int(cover),
        "ci_length": ci_upper - ci_lower,
        "miss_below_true": int(miss_below_true),
        "miss_above_true": int(miss_above_true),
        "h_left": extract_bandwidth(fit, "h", "left"),
        "h_right": extract_bandwidth(fit, "h", "right"),
        "b_left": extract_bandwidth(fit, "b", "left"),
        "b_right": extract_bandwidth(fit, "b", "right"),
        "N_left": N_left,
        "N_right": N_right,
        "N_h_left": N_h_left,
        "N_h_right": N_h_right,
        "N_b_left": N_b_left,
        "N_b_right": N_b_right,
        "M_left": M_left,
        "M_right": M_right,
        "bwselect": safe_attr(fit, "bwselect", cfg.bwselect),
        "vce": safe_attr(fit, "vce", cfg.vce),
        "nnmatch": cfg.nnmatch,
        "kernel": safe_attr(fit, "kernel", cfg.kernel),
        "p": safe_attr(fit, "p", cfg.p),
        "q": safe_attr(fit, "q", cfg.q),
        "ci_method": ci_method,
    }


def failed_row(
    scenario: str,
    rep: int,
    method: str,
    exc: BaseException,
    cfg: SimConfig,
) -> Dict[str, Any]:
    row = {col: np.nan for col in RAW_COLUMNS}
    row.update(
        {
            "scenario": scenario,
            "scenario_label": SCENARIO_DISPLAY[scenario],
            "rep": rep,
            "method": method,
            "method_label": METHOD_DISPLAY[method],
            "success": False,
            "error": repr(exc),
            "tau_true": TRUE_TAU,
            "cover": np.nan,
            "miss_below_true": np.nan,
            "miss_above_true": np.nan,
            "bwselect": cfg.bwselect,
            "vce": cfg.vce,
            "nnmatch": cfg.nnmatch,
            "kernel": cfg.kernel,
            "p": cfg.p,
            "q": cfg.q,
        }
    )
    return row


# -----------------------------
# Monte Carlo driver
# -----------------------------
def run_simulation(cfg: SimConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline_rng = np.random.default_rng(cfg.seed)
    scenario_rngs = {
        "N1": np.random.default_rng(cfg.seed + 100_003),
        "N2": np.random.default_rng(cfg.seed + 200_003),
        "N3": np.random.default_rng(cfg.seed + 300_003),
    }
    rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []

    total_jobs = len(cfg.scenarios) * cfg.reps
    done = 0
    start = time.time()

    for rep in range(1, cfg.reps + 1):
        x = 2.0 * baseline_rng.beta(2.0, 4.0, size=cfg.n) - 1.0
        u_n0 = baseline_rng.normal(loc=0.0, scale=1.0, size=cfg.n)

        for scenario in cfg.scenarios:
            done += 1
            if scenario == "N0":
                u = u_n0
                extra = {"contam_share": np.nan}
            else:
                u, extra = draw_standardized_u(scenario, cfg.n, scenario_rngs[scenario])
            y, _, u, extra = make_dataset_from_x(x=x, scenario=scenario, u=u, extra=extra)
            diag_rows.append(error_diagnostics(scenario, rep, u, extra))

            try:
                base_fit = rdrobust_call(y=y, x=x, cfg=cfg, bwselect=cfg.bwselect)
                rows.append(
                    extract_fit_row(
                        fit=base_fit,
                        scenario=scenario,
                        rep=rep,
                        method="conventional_mserd",
                        cfg=cfg,
                        ci_method="normal_approx",
                    )
                )
                rows.append(
                    extract_fit_row(
                        fit=base_fit,
                        scenario=scenario,
                        rep=rep,
                        method="rbc_mserd",
                        cfg=cfg,
                        ci_method="normal_approx",
                    )
                )
            except BaseException as exc:
                base_fit = None
                rows.append(failed_row(scenario, rep, "conventional_mserd", exc, cfg))
                rows.append(failed_row(scenario, rep, "rbc_mserd", exc, cfg))

            if cfg.progress_every and done % cfg.progress_every == 0:
                elapsed = time.time() - start
                print(
                    f"Completed {done}/{total_jobs} scenario-rep jobs "
                    f"({scenario}, rep {rep}); elapsed {elapsed/60:.1f} min",
                    flush=True,
                )

    raw = pd.DataFrame(rows).reindex(columns=RAW_COLUMNS)
    diag = pd.DataFrame(diag_rows).reindex(columns=ERROR_DIAG_COLUMNS)
    return raw, diag


# -----------------------------
# Summary, pairwise checks, plots
# -----------------------------
def monte_carlo_se_for_coverage(coverage_values: pd.Series) -> float:
    s = coverage_values.dropna().astype(float)
    if len(s) == 0:
        return np.nan
    p_hat = float(s.mean())
    return math.sqrt(p_hat * (1.0 - p_hat) / len(s))


def trimmed_mean(x: Iterable[float], prop: float = 0.01) -> float:
    arr = np.sort(np.asarray(list(x), dtype=float))
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    k = int(math.floor(prop * len(arr)))
    if 2 * k >= len(arr):
        return float(np.mean(arr))
    return float(np.mean(arr[k: len(arr) - k]))


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scenario in SCENARIO_ORDER:
        if scenario not in set(raw["scenario"].dropna().unique()):
            continue
        for method in METHOD_ORDER:
            df_m = raw[(raw["scenario"] == scenario) & (raw["method"] == method)].copy()
            ok = df_m[df_m["success"] == True].copy()
            reps_total = len(df_m)
            reps_success = len(ok)
            fail_rate = 1.0 - reps_success / reps_total if reps_total else np.nan

            base_row = {
                "scenario": scenario,
                "scenario_label": SCENARIO_DISPLAY[scenario],
                "method": method,
                "method_label": METHOD_DISPLAY[method],
                "reps_total": reps_total,
                "reps_success": reps_success,
                "fail_rate": fail_rate,
            }

            if reps_success == 0:
                rows.append(base_row)
                continue

            err = ok["estimate"].astype(float) - TRUE_TAU
            ec = float(ok["cover"].mean())
            sd_est = float(ok["estimate"].std(ddof=1))
            avg_se = float(ok["se"].mean())
            t = ok["t_stat"].dropna().astype(float)

            base_row.update(
                {
                    "EC": ec,
                    "EC_percent": 100.0 * ec,
                    "coverage_error": ec - 0.95,
                    "coverage_error_pp": 100.0 * (ec - 0.95),
                    "coverage_mcse": monte_carlo_se_for_coverage(ok["cover"]),
                    "IL": float(ok["ci_length"].mean()),
                    "avg_estimate": float(ok["estimate"].mean()),
                    "bias": float(err.mean()),
                    "abs_bias": abs(float(err.mean())),
                    "median_bias": float(np.median(err)),
                    "trimmed_bias_1pct": trimmed_mean(err, 0.01),
                    "RMSE": float(np.sqrt(np.mean(err**2))),
                    "sd_estimate": sd_est,
                    "avg_se": avg_se,
                    "se_to_sd_ratio": avg_se / sd_est if sd_est > 0 else np.nan,
                    "miss_below_true": float(ok["miss_below_true"].mean()),
                    "miss_above_true": float(ok["miss_above_true"].mean()),
                    "avg_h_left": float(ok["h_left"].mean()),
                    "avg_h_right": float(ok["h_right"].mean()),
                    "avg_b_left": float(ok["b_left"].mean()),
                    "avg_b_right": float(ok["b_right"].mean()),
                    "avg_N_h_left": float(ok["N_h_left"].mean()),
                    "avg_N_h_right": float(ok["N_h_right"].mean()),
                    "avg_N_b_left": float(ok["N_b_left"].mean()),
                    "avg_N_b_right": float(ok["N_b_right"].mean()),
                    "t_mean": float(t.mean()) if len(t) else np.nan,
                    "t_sd": float(t.std(ddof=1)) if len(t) > 1 else np.nan,
                    "t_skew": moment_skew(t.to_numpy()) if len(t) > 2 else np.nan,
                    "t_excess_kurtosis": moment_excess_kurtosis(t.to_numpy()) if len(t) > 3 else np.nan,
                    "t_q025": float(np.quantile(t, 0.025)) if len(t) else np.nan,
                    "t_q975": float(np.quantile(t, 0.975)) if len(t) else np.nan,
                }
            )
            rows.append(base_row)

    summary = pd.DataFrame(rows)
    summary["scenario"] = pd.Categorical(summary["scenario"], categories=SCENARIO_ORDER, ordered=True)
    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
    summary = summary.sort_values(["scenario", "method"]).reset_index(drop=True)
    preferred = [col for col in SUMMARY_COLUMNS if col in summary.columns]
    extra = [col for col in summary.columns if col not in preferred]
    return summary[preferred + extra]


def summarize_error_diagnostics(diag: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scenario in SCENARIO_ORDER:
        df = diag[diag["scenario"] == scenario].copy()
        if df.empty:
            continue
        row = {"scenario": scenario, "scenario_label": SCENARIO_DISPLAY[scenario], "reps": len(df)}
        for col in [
            "u_mean",
            "u_sd",
            "u_skew",
            "u_excess_kurtosis",
            "u_q01",
            "u_q05",
            "u_q50",
            "u_q95",
            "u_q99",
            "contam_share",
        ]:
            row[f"avg_{col}"] = float(df[col].mean(skipna=True))
        rows.append(row)
    out = pd.DataFrame(rows)
    preferred = [col for col in PAIRWISE_COLUMNS if col in out.columns]
    extra = [col for col in out.columns if col not in preferred]
    return out[preferred + extra] if not out.empty else out


def pairwise_checks(raw: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("rbc_mserd", "conventional_mserd"),
    ]
    rows: List[Dict[str, Any]] = []
    for scenario in SCENARIO_ORDER:
        df = raw[(raw["scenario"] == scenario) & (raw["success"] == True)].copy()
        if df.empty:
            continue
        for a, b in pairs:
            wide_cover = df.pivot_table(index="rep", columns="method", values="cover", aggfunc="first")
            wide_len = df.pivot_table(index="rep", columns="method", values="ci_length", aggfunc="first")
            wide_est = df.pivot_table(index="rep", columns="method", values="estimate", aggfunc="first")
            wide_se = df.pivot_table(index="rep", columns="method", values="se", aggfunc="first")
            if a not in wide_cover.columns or b not in wide_cover.columns:
                continue
            d_cover = (wide_cover[a] - wide_cover[b]).dropna().astype(float)
            d_len = (wide_len[a] - wide_len[b]).dropna().astype(float)
            est_pair = wide_est[[a, b]].dropna().astype(float)
            se_pair = wide_se[[a, b]].dropna().astype(float)
            err_a = est_pair[a] - TRUE_TAU
            err_b = est_pair[b] - TRUE_TAU
            bias_a = float(err_a.mean()) if len(err_a) else np.nan
            bias_b = float(err_b.mean()) if len(err_b) else np.nan
            rmse_a = float(np.sqrt(np.mean(np.square(err_a)))) if len(err_a) else np.nan
            rmse_b = float(np.sqrt(np.mean(np.square(err_b)))) if len(err_b) else np.nan
            sd_a = float(est_pair[a].std(ddof=1)) if len(est_pair) > 1 else np.nan
            sd_b = float(est_pair[b].std(ddof=1)) if len(est_pair) > 1 else np.nan
            avg_se_a = float(se_pair[a].mean()) if len(se_pair) else np.nan
            avg_se_b = float(se_pair[b].mean()) if len(se_pair) else np.nan
            se_ratio_a = avg_se_a / sd_a if np.isfinite(sd_a) and sd_a > 0 else np.nan
            se_ratio_b = avg_se_b / sd_b if np.isfinite(sd_b) and sd_b > 0 else np.nan
            se_cover = float(d_cover.std(ddof=1) / math.sqrt(len(d_cover))) if len(d_cover) > 1 else np.nan
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_DISPLAY[scenario],
                    "comparison": f"{METHOD_DISPLAY[a]} minus {METHOD_DISPLAY[b]}",
                    "paired_reps": len(d_cover),
                    "EC_diff_pp": 100.0 * float(d_cover.mean()) if len(d_cover) else np.nan,
                    "EC_diff_mcse_pp": 100.0 * se_cover if np.isfinite(se_cover) else np.nan,
                    "IL_diff": float(d_len.mean()) if len(d_len) else np.nan,
                    "bias_diff": bias_a - bias_b if np.isfinite(bias_a) and np.isfinite(bias_b) else np.nan,
                    "abs_bias_diff": abs(bias_a) - abs(bias_b) if np.isfinite(bias_a) and np.isfinite(bias_b) else np.nan,
                    "RMSE_diff": rmse_a - rmse_b if np.isfinite(rmse_a) and np.isfinite(rmse_b) else np.nan,
                    "se_to_sd_ratio_diff": se_ratio_a - se_ratio_b if np.isfinite(se_ratio_a) and np.isfinite(se_ratio_b) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def ensure_dirs(outdir: Path) -> Path:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    return figdir


def ordered_methods_present(df: pd.DataFrame) -> List[str]:
    return [method for method in METHOD_ORDER if method in set(df["method"].dropna())]


def ordered_scenarios_present(df: pd.DataFrame) -> List[str]:
    return [scenario for scenario in SCENARIO_ORDER if scenario in set(df["scenario"].dropna())]


def scenario_tick_labels(scenarios: List[str]) -> List[str]:
    return [SCENARIO_TICK_LABELS.get(s, SCENARIO_DISPLAY.get(s, s)) for s in scenarios]


def style_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, color="#8c8c8c", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_dgp2_curve(figdir: Path) -> None:
    x_grid = np.linspace(-1.0, 1.0, 1000)
    y_grid = mu_dgp2(x_grid)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    ax.plot(x_grid, y_grid, linewidth=2, color="#4C78A8")
    ax.axvline(CUTOFF, linestyle="--", linewidth=1, color="#444444")
    ax.set_xlabel("Running variable X")
    ax.set_ylabel("mu_2(X)")
    ax.set_title("CCT (2014) DGP2 regression function", pad=10)
    style_axes(ax, grid_axis="both")
    fig.tight_layout()
    fig.savefig(figdir / "dgp2_curve.png", dpi=300)
    plt.close(fig)


def plot_error_distribution_by_scenario(figdir: Path) -> None:
    """Plot the original standardized error distributions U for each scenario."""
    rng = np.random.default_rng(20260615)
    n_draws = 250_000
    scenarios = list(SCENARIO_ORDER)
    ncols = 2
    nrows = int(math.ceil(len(scenarios) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.2, 4.0 * nrows), dpi=120)
    axes_flat = np.asarray(axes).reshape(-1)

    for ax, scenario in zip(axes_flat, scenarios):
        u, _ = draw_standardized_u(scenario, n_draws, rng)
        finite = u[np.isfinite(u)]
        x_min = max(-8.0, float(np.quantile(finite, 0.001)))
        x_max = min(8.0, float(np.quantile(finite, 0.999)))
        x_grid = np.linspace(x_min, x_max, 500)
        normal_pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x_grid**2)

        ax.hist(finite, bins=90, range=(x_min, x_max), density=True, alpha=0.70, color="#4C78A8")
        ax.plot(x_grid, normal_pdf, color="#F58518", linewidth=1.5, label="N(0,1)")
        ax.axvline(0.0, color="#444444", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_xlim(x_min, x_max)
        ax.set_title(SCENARIO_DISPLAY[scenario], pad=8)
        ax.set_xlabel("Standardized error U")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, loc="upper right")
        style_axes(ax, grid_axis="both")

    for ax in axes_flat[len(scenarios):]:
        ax.axis("off")

    fig.suptitle("Original error distributions by scenario", y=0.995, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(figdir / "error_distribution_by_scenario.png", dpi=300)
    plt.close(fig)


def plot_grouped_bar(summary: pd.DataFrame, value_col: str, ylabel: str, title: str, filename: str, figdir: Path) -> None:
    df = summary.dropna(subset=[value_col]).copy()
    if df.empty:
        return

    scenarios = ordered_scenarios_present(df)
    methods = ordered_methods_present(df)
    if not scenarios or not methods:
        return

    pivot = df.pivot(index="scenario", columns="method", values=value_col).reindex(scenarios)
    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=120)
    x = np.arange(len(scenarios))
    width = min(0.22, 0.72 / max(1, len(methods)))

    bars_by_method = []
    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            pivot[method].to_numpy(dtype=float),
            width=width,
            label=METHOD_DISPLAY[method],
            color=METHOD_COLORS.get(method),
        )
        bars_by_method.append(bars)

    if value_col in {"EC", "EC_percent"}:
        ax.axhline(0.95 if value_col == "EC" else 95.0, linestyle="--", linewidth=1, color="#222222")
        if value_col == "EC_percent":
            finite = pivot[methods].to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            low = max(0.0, math.floor((float(finite.min()) - 6.0) / 5.0) * 5.0) if len(finite) else 75.0
            ax.set_ylim(low, 99.0)
            ax.set_yticks(np.arange(low, 100.0, 5.0))
            ax.set_yticklabels([f"{v / 100.0:.2f}" for v in np.arange(low, 100.0, 5.0)])
            for bars in bars_by_method:
                labels = [f"{bar.get_height():.1f}%" for bar in bars]
                ax.bar_label(bars, labels=labels, padding=4, fontsize=9)
            ax.text(
                0.015,
                95.8,
                "Nominal 95%",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="bottom",
                fontsize=9,
                color="#222222",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            )
    else:
        vals = pivot[methods].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if len(finite):
            ax.set_ylim(0.0, float(finite.max()) * 1.18)

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_tick_labels(scenarios), rotation=0, ha="center")
    ax.set_xlabel("Scenario")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=28)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=len(methods), frameon=False)
    style_axes(ax)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.82)
    fig.savefig(figdir / filename, dpi=300)
    plt.close(fig)


def plot_coverage_length_tradeoff(summary: pd.DataFrame, figdir: Path) -> None:
    df = summary.dropna(subset=["EC", "IL"]).copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9.2, 6.2), dpi=120)
    methods = ordered_methods_present(df)
    label_offsets = {
        ("conventional_mserd", "N0"): (16, -10),
        ("conventional_mserd", "N1"): (-12, -18),
        ("conventional_mserd", "N2"): (-18, -16),
        ("conventional_mserd", "N3"): (-14, -18),
        ("rbc_mserd", "N0"): (34, -4),
        ("rbc_mserd", "N1"): (12, 12),
        ("rbc_mserd", "N2"): (-32, -18),
        ("rbc_mserd", "N3"): (14, 14),
    }

    for method in methods:
        sub = df[df["method"] == method]
        ax.scatter(
            sub["IL"],
            sub["EC"],
            s=70,
            marker=METHOD_MARKERS.get(method, "o"),
            color=METHOD_COLORS.get(method),
            edgecolor="white",
            linewidth=0.7,
            label=METHOD_DISPLAY[method],
            zorder=3,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                str(row["scenario"]),
                (row["IL"], row["EC"]),
                textcoords="offset points",
                xytext=label_offsets.get((method, row["scenario"]), (6, 6)),
                fontsize=8,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#666666",
                    "alpha": 0.65,
                    "linewidth": 0.6,
                    "shrinkA": 2,
                    "shrinkB": 4,
                },
            )

    ax.axhline(0.95, linestyle="--", linewidth=1, color="#222222")
    il = df["IL"].to_numpy(dtype=float)
    ec = df["EC"].to_numpy(dtype=float)
    ax.set_xlim(float(np.nanmin(il)) - 0.007, float(np.nanmax(il)) + 0.008)
    ax.set_ylim(float(np.nanmin(ec)) - 0.006, min(1.0, float(np.nanmax(ec)) + 0.006))
    ax.set_xlabel("Average interval length")
    ax.set_ylabel("Empirical coverage")
    fig.suptitle("Non-normal extension: coverage-length tradeoff", y=0.975, fontsize=15)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=len(methods), frameon=False)
    style_axes(ax, grid_axis="both")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.72)
    fig.savefig(figdir / "coverage_length_tradeoff.png", dpi=300)
    plt.close(fig)


def plot_rbc_tstat_by_scenario(raw: pd.DataFrame, figdir: Path) -> None:
    ok = raw[(raw["success"] == True) & (raw["method"] == "rbc_mserd")].copy()
    if ok.empty:
        return
    for scenario in SCENARIO_ORDER:
        vals = ok.loc[ok["scenario"] == scenario, "t_stat"].dropna().astype(float).to_numpy()
        if len(vals) == 0:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
        ax.hist(vals, bins=40, density=True, alpha=0.65, color="#4C78A8")
        grid = np.linspace(-5, 5, 500)
        normal_pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * grid**2)
        ax.plot(grid, normal_pdf, linewidth=1.5, color="#F58518")
        ax.axvline(-1.96, linestyle="--", linewidth=1, color="#444444")
        ax.axvline(1.96, linestyle="--", linewidth=1, color="#444444")
        ax.set_xlabel("T_RBC = (tau_hat_bc - tau_true) / SE_RBC")
        ax.set_ylabel("Density")
        ax.set_title(f"RBC t-statistic distribution: {SCENARIO_DISPLAY[scenario]}", pad=10)
        style_axes(ax)
        fig.tight_layout()
        safe_name = scenario.replace("/", "_")
        fig.savefig(figdir / f"rbc_tstat_{safe_name}.png", dpi=300)
        plt.close(fig)


def plot_rbc_tstat_grid(raw: pd.DataFrame, figdir: Path) -> None:
    ok = raw[(raw["success"] == True) & (raw["method"] == "rbc_mserd")].copy()
    if ok.empty:
        return

    scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in set(ok["scenario"].dropna())]
    if not scenarios:
        return

    ncols = 2
    nrows = int(math.ceil(len(scenarios) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.2, 4.0 * nrows), dpi=120)
    axes_flat = np.asarray(axes).reshape(-1)
    grid = np.linspace(-5, 5, 500)
    normal_pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * grid**2)

    for ax, scenario in zip(axes_flat, scenarios):
        vals = ok.loc[ok["scenario"] == scenario, "t_stat"].dropna().astype(float).to_numpy()
        if len(vals) == 0:
            ax.axis("off")
            continue
        ax.hist(vals, bins=40, range=(-5, 5), density=True, alpha=0.65, color="#4C78A8")
        ax.plot(grid, normal_pdf, linewidth=1.5, color="#F58518", label="N(0,1)")
        ax.axvline(-1.96, linestyle="--", linewidth=1, color="#444444")
        ax.axvline(1.96, linestyle="--", linewidth=1, color="#444444")
        ax.set_xlim(-5, 5)
        ax.set_xlabel("T_RBC")
        ax.set_ylabel("Density")
        ax.set_title(SCENARIO_DISPLAY[scenario], pad=8)
        ax.legend(frameon=False, loc="upper right")
        style_axes(ax)

    for ax in axes_flat[len(scenarios):]:
        ax.axis("off")

    fig.suptitle("RBC t-statistic distributions by scenario", y=0.995, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(figdir / "rbc_tstat_by_scenario.png", dpi=300)
    plt.close(fig)


def plot_rbc_noncoverage_asymmetry(summary: pd.DataFrame, figdir: Path) -> None:
    df = summary[(summary["method"] == "rbc_mserd")].dropna(subset=["miss_below_true", "miss_above_true"]).copy()
    if df.empty:
        return
    scenarios = ordered_scenarios_present(df)
    df = df.set_index("scenario").reindex(scenarios)
    x = np.arange(len(scenarios))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=120)
    ax.bar(x - width / 2, df["miss_below_true"].to_numpy(dtype=float), width, label="CI below true tau", color="#4C78A8")
    ax.bar(x + width / 2, df["miss_above_true"].to_numpy(dtype=float), width, label="CI above true tau", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_tick_labels(scenarios), rotation=0, ha="center")
    ax.set_ylabel("Noncoverage probability")
    ax.set_title("RBC noncoverage asymmetry", pad=28)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    style_axes(ax)
    vals = df[["miss_below_true", "miss_above_true"]].to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if len(finite):
        ax.set_ylim(0.0, max(0.01, float(finite.max()) * 1.22))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.80)
    fig.savefig(figdir / "rbc_noncoverage_asymmetry.png", dpi=300)
    plt.close(fig)


def plot_bandwidth_box(raw: pd.DataFrame, figdir: Path) -> None:
    ok = raw[(raw["success"] == True) & (raw["method"].isin(["conventional_mserd", "rbc_mserd"]))].copy()
    if ok.empty:
        return
    data: List[np.ndarray] = []
    labels: List[str] = []
    for scenario in SCENARIO_ORDER:
        vals = ok.loc[(ok["scenario"] == scenario) & (ok["method"] == "rbc_mserd"), "h_left"].dropna().astype(float).to_numpy()
        if len(vals):
            data.append(vals)
            labels.append(SCENARIO_DISPLAY[scenario])
    if not data:
        return
    fig, ax = plt.subplots(figsize=(11.2, 5.2), dpi=120)
    try:
        ax.boxplot(data, tick_labels=scenario_tick_labels([s for s in SCENARIO_ORDER if SCENARIO_DISPLAY[s] in labels]), showfliers=False)
    except TypeError:  # Matplotlib < 3.9
        ax.boxplot(data, labels=scenario_tick_labels([s for s in SCENARIO_ORDER if SCENARIO_DISPLAY[s] in labels]), showfliers=False)
    ax.set_ylabel("Left bandwidth h")
    ax.set_title("RBC/MSE bandwidth distribution by scenario", pad=10)
    ax.tick_params(axis="x", labelrotation=0)
    style_axes(ax)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.90)
    fig.savefig(figdir / "h_left_distribution_by_scenario.png", dpi=300)
    plt.close(fig)


def make_plots(raw: pd.DataFrame, summary: pd.DataFrame, outdir: Path) -> None:
    figdir = ensure_dirs(outdir)
    plot_dgp2_curve(figdir)
    plot_error_distribution_by_scenario(figdir)
    plot_grouped_bar(summary, "EC_percent", "Empirical coverage (%)", "Coverage by non-normal scenario", "coverage_by_scenario_method.png", figdir)
    plot_grouped_bar(summary, "IL", "Average interval length", "Interval length by non-normal scenario", "interval_length_by_scenario_method.png", figdir)
    plot_grouped_bar(summary, "RMSE", "RMSE", "RMSE by non-normal scenario", "rmse_by_scenario_method.png", figdir)
    plot_coverage_length_tradeoff(summary, figdir)
    plot_rbc_tstat_by_scenario(raw, figdir)
    plot_rbc_tstat_grid(raw, figdir)
    plot_rbc_noncoverage_asymmetry(summary, figdir)
    plot_bandwidth_box(raw, figdir)


# -----------------------------
# CLI
# -----------------------------
def parse_scenarios(text: str) -> Tuple[str, ...]:
    if text.strip().lower() in {"all", ""}:
        return tuple(SCENARIO_ORDER)
    aliases = {
        "n0": "N0",
        "normal": "N0",
        "n0_normal": "N0",
        "n1": "N1",
        "severe_skew": "N1",
        "severe_skewness": "N1",
        "n1_severe_skew": "N1",
        "n4": "N1",
        "n4_severe_skew": "N1",
        "n2": "N2",
        "t3": "N2",
        "extreme_tail": "N2",
        "extreme_tails": "N2",
        "n2_t3_extreme_tail": "N2",
        "n5": "N2",
        "n5_t3_extreme_tail": "N2",
        "n3": "N3",
        "one_sided": "N3",
        "one_sided_contamination": "N3",
        "n3_one_sided_contamination": "N3",
        "n6": "N3",
        "n6_one_sided_contamination": "N3",
    }
    out: List[str] = []
    for raw in text.split(","):
        key = raw.strip()
        if not key:
            continue
        key_norm = aliases.get(key.lower(), key)
        if key_norm not in SCENARIO_ORDER:
            raise ValueError(f"Unknown scenario '{raw}'. Valid: {SCENARIO_ORDER} or aliases n0,n1,n2,n3")
        out.append(key_norm)
    if not out:
        raise ValueError("No scenarios selected.")
    return tuple(dict.fromkeys(out))


def parse_args(argv: Optional[List[str]] = None) -> SimConfig:
    parser = argparse.ArgumentParser(description="CCT DGP2 non-normal Monte Carlo extension without bootstrap.")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help="Monte Carlo replications per scenario.")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Sample size per replication.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory.")
    parser.add_argument("--scenarios", type=str, default="all", help="Comma-separated scenarios: all or n0,n1,n2,n3.")
    parser.add_argument("--bwselect", type=str, default="mserd", help="rdrobust bandwidth selector for main methods.")
    parser.add_argument("--vce", type=str, default="nn", choices=["nn", "hc0", "hc1", "hc2", "hc3"], help="rdrobust variance estimator.")
    parser.add_argument("--nnmatch", type=int, default=3, help="Nearest-neighbor matches when vce='nn'.")
    parser.add_argument("--kernel", type=str, default="tri", help="Kernel: tri, triangular, uniform, epa, epanechnikov.")
    parser.add_argument("--p", type=int, default=1, help="Local polynomial order for point estimator.")
    parser.add_argument("--q", type=int, default=2, help="Local polynomial order for bias correction.")
    parser.add_argument("--level", type=float, default=95.0, help="Confidence level.")
    parser.add_argument("--masspoints", type=str, default="off", choices=["off", "check", "adjust"], help="Mass-points option.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every k scenario-rep jobs; 0 disables.")

    args = parser.parse_args(argv)
    scenarios = parse_scenarios(args.scenarios)

    if args.reps <= 0:
        raise ValueError("--reps must be positive.")
    if args.n <= 20:
        raise ValueError("--n should be larger than 20.")
    if args.nnmatch <= 0:
        raise ValueError("--nnmatch must be positive.")

    return SimConfig(
        reps=args.reps,
        n=args.n,
        seed=args.seed,
        outdir=args.outdir,
        scenarios=scenarios,
        bwselect=args.bwselect,
        vce=args.vce,
        nnmatch=args.nnmatch,
        kernel=args.kernel,
        p=args.p,
        q=args.q,
        level=args.level,
        masspoints=args.masspoints,
        progress_every=args.progress_every,
    )


def run_config_path(outdir: Path) -> Path:
    if outdir.name.lower() == "results":
        return outdir.parent / "run_config_nonnormal_paired.txt"
    return outdir / "run_config_nonnormal_paired.txt"


def write_config(cfg: SimConfig, path: Path) -> None:
    lines = [
        "CCT DGP2 severe non-normal comparison config (paired X design)",
        "design=paired_running_variable_by_replication",
        "error_streams=N0_baseline_stream,N1_N2_N3_independent_streams",
        f"reps={cfg.reps}",
        f"n={cfg.n}",
        f"seed={cfg.seed}",
        f"scenarios={','.join(cfg.scenarios)}",
        f"methods={','.join(METHOD_ORDER)}",
        "bootstrap=removed",
        f"true_tau={TRUE_TAU}",
        f"sigma_eps={SIGMA_EPS}",
        f"bwselect={cfg.bwselect}",
        f"vce={cfg.vce}",
        f"nnmatch={cfg.nnmatch}",
        f"kernel={cfg.kernel}",
        f"p={cfg.p}",
        f"q={cfg.q}",
        f"level={cfg.level}",
        f"masspoints={cfg.masspoints}",
        f"outdir={'results' if cfg.outdir.name.lower() == 'results' else str(cfg.outdir)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    config_path = run_config_path(cfg.outdir)

    # Fail early instead of spending all replications recording the same import error.
    import_rdrobust()

    print("CCT DGP2 non-normal comparison")
    print(f"Output directory: {cfg.outdir}")
    print(f"reps={cfg.reps}, n={cfg.n}, seed={cfg.seed}")
    print(f"scenarios={','.join(cfg.scenarios)}")
    print(f"methods={', '.join(METHOD_DISPLAY[m] for m in METHOD_ORDER)}")
    print(f"true_tau={TRUE_TAU:.6f}, sigma_eps={SIGMA_EPS:.4f}")
    print(f"bwselect={cfg.bwselect}, vce={cfg.vce}, nnmatch={cfg.nnmatch}, kernel={cfg.kernel}, p={cfg.p}, q={cfg.q}")

    write_config(cfg, config_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw, diag = run_simulation(cfg)

    summary = summarize(raw)
    error_summary = summarize_error_diagnostics(diag)
    pairwise = pairwise_checks(raw)

    raw_path = cfg.outdir / "nonnormal_raw.csv"
    summary_path = cfg.outdir / "nonnormal_summary.csv"
    diag_path = cfg.outdir / "nonnormal_error_diagnostics.csv"
    error_summary_path = cfg.outdir / "nonnormal_error_summary.csv"
    pairwise_path = cfg.outdir / "nonnormal_pairwise_checks.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    diag.to_csv(diag_path, index=False)
    error_summary.to_csv(error_summary_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    make_plots(raw, summary, cfg.outdir)

    failures = raw.loc[raw["success"] != True, ["scenario", "method", "error"]]
    if not failures.empty:
        print("\nFailures detected, first unique examples:")
        print(failures.drop_duplicates().head(12).to_string(index=False))

    print("\nConcise summary:")
    printable_cols = [
        "scenario",
        "method",
        "reps_success",
        "fail_rate",
        "EC_percent",
        "coverage_error_pp",
        "coverage_mcse",
        "IL",
        "avg_estimate",
        "bias",
        "abs_bias",
        "median_bias",
        "trimmed_bias_1pct",
        "RMSE",
        "avg_se",
        "sd_estimate",
        "se_to_sd_ratio",
        "miss_below_true",
        "miss_above_true",
        "avg_h_left",
        "avg_h_right",
        "avg_b_left",
        "avg_b_right",
        "t_mean",
        "t_sd",
        "t_skew",
        "t_excess_kurtosis",
        "t_q025",
        "t_q975",
    ]
    existing_cols = [c for c in printable_cols if c in summary.columns]
    print(summary[existing_cols].to_string(index=False))

    print(f"\nSaved raw results to:       {raw_path}")
    print(f"Saved summary results to:   {summary_path}")
    print(f"Saved diagnostics to:       {diag_path}")
    print(f"Saved error summary to:     {error_summary_path}")
    print(f"Saved pairwise checks to:   {pairwise_path}")
    print(f"Saved figures to:           {cfg.outdir / 'figures'}")
    print(f"Saved run config to:        {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
