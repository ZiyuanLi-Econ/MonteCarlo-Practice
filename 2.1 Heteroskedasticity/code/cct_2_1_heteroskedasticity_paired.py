#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCT (2014) baseline heteroskedasticity extension.

Purpose
-------
Starting from the CCT baseline design, change only the conditional variance of
the error term and check whether RBC remains useful when inference already uses
heteroskedasticity-robust standard errors.

Design
------
H0-H3 keep the same running variable, regression function, true treatment
effect, and sample size. The Monte Carlo is strictly paired across scenarios:
within replication r, all scenarios use the same X_i sample and the same base
standard normal shock u_i. Only the conditional variance multiplier v_s(X_i)
changes.

Methods
-------
Main comparison:
    1. Conventional CI + heteroskedasticity-robust SE, vce='nn'
    2. RBC CI + heteroskedasticity-robust RBC SE, vce='nn'

Optional appendix check:
    3. Coverage-error-rate-optimal RBC, bwselect='cerrd'

Run examples
------------
Main run:
    python cct_2_1_heteroskedasticity_paired.py --reps 5000 --n 500

Quick check:
    python cct_2_1_heteroskedasticity_paired.py --reps 50 --n 500

Outputs
-------
    If this file is stored inside a code/ folder, output is written to the
    sibling results/ folder. If the file is run as a standalone script,
    output is written to cct_2_1_heteroskedasticity_paired_results/ next to the
    script.

    run_config_hetero_paired.txt
    results or cct_2_1_heteroskedasticity_paired_results/
        hetero_raw.csv
        hetero_summary_long.csv
        hetero_interpretation_table.csv
        figures/
            fig_hetero_variance_functions.png
            fig_hetero_coverage_main.png
            fig_hetero_interval_length_main.png
            fig_hetero_rbc_coverage_gain.png
            fig_hetero_coverage_length_tradeoff.png
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Baseline constants
# ---------------------------------------------------------------------------

TRUE_TAU_BASELINE = -3.45
SIGMA_EPS = 0.1295
DEFAULT_SCENARIOS = ("H0", "H1", "H2", "H3")
SCENARIO_LABELS = {
    "H0": "Homoskedastic baseline",
    "H1": "Smooth heteroskedasticity",
    "H2": "Side-specific heteroskedasticity",
    "H3": "Near-cutoff volatility",
}
SCENARIO_FORMULAS = {
    "H0": "v(x)=1",
    "H1": "v(x)=1+gamma*x^2",
    "H2": "v(x)=1+gamma*1{x>=0}",
    "H3": "v(x)=1+gamma*exp(-|x|/0.10)",
}


def default_output_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    if script_dir.name.lower() == "code":
        return script_dir.parent / "results"
    return script_dir / f"{Path(__file__).resolve().stem}_results"


DEFAULT_OUTDIR = default_output_dir()

# Row positions in rdrobust(..., all=True) output.
# 0 = Conventional, 1 = Bias-corrected, 2 = Robust/RBC.
RDROBUST_ROWS = {
    "conventional": 0,
    "bias_corrected": 1,
    "rbc": 2,
}


@dataclass(frozen=True)
class EstimationSpec:
    """A single rdrobust estimation specification."""

    spec_id: str
    paper_label: str
    rd_row: str
    bwselect: str = "mserd"
    vce: str = "nn"
    include_in_main: bool = False


MAIN_SPECS: Tuple[EstimationSpec, ...] = (
    EstimationSpec(
        spec_id="conv_nn_mserd",
        paper_label="Conventional + robust SE (nn)",
        rd_row="conventional",
        bwselect="mserd",
        vce="nn",
        include_in_main=True,
    ),
    EstimationSpec(
        spec_id="rbc_nn_mserd",
        paper_label="RBC + robust SE (nn)",
        rd_row="rbc",
        bwselect="mserd",
        vce="nn",
        include_in_main=True,
    ),
)

CER_SPEC = EstimationSpec(
    spec_id="rbc_nn_cerrd",
    paper_label="CER-optimal RBC (cerrd, nn)",
    rd_row="rbc",
    bwselect="cerrd",
    vce="nn",
    include_in_main=False,
)


# ---------------------------------------------------------------------------
# 2. DGP: baseline conditional mean with heteroskedastic errors
# ---------------------------------------------------------------------------


def draw_running_variable(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw X = 2*Beta(2,4)-1, so X lies in [-1,1]."""
    return 2.0 * rng.beta(2.0, 4.0, size=n) - 1.0


def mu_baseline(x: np.ndarray) -> np.ndarray:
    """Baseline conditional mean function."""
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


def raw_variance_multiplier(
    x: np.ndarray,
    scenario: str,
    gamma: float,
    cutoff_scale: float = 0.10,
) -> np.ndarray:
    """
    Raw conditional variance multiplier v(x).

    The actual error uses sqrt(v(x) / mean(v(x))) so that the unconditional
    error variance is kept comparable to the baseline sigma_eps^2.
    """
    scenario = scenario.upper()
    x = np.asarray(x)
    if scenario == "H0":
        v = np.ones_like(x, dtype=float)
    elif scenario == "H1":
        v = 1.0 + gamma * x**2
    elif scenario == "H2":
        v = 1.0 + gamma * (x >= 0).astype(float)
    elif scenario == "H3":
        v = 1.0 + gamma * np.exp(-np.abs(x) / cutoff_scale)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    if np.any(v <= 0) or not np.all(np.isfinite(v)):
        raise ValueError(f"Invalid variance multiplier in scenario {scenario}.")
    return v


def normalized_variance_multiplier(
    x: np.ndarray,
    scenario: str,
    gamma: float,
    normalize: str = "sample",
    cutoff_scale: float = 0.10,
) -> np.ndarray:
    """
    Return v(x) normalized to have mean one.

    normalize='sample' normalizes within each Monte Carlo replication.
    This keeps the average conditional error variance in each sample equal to
    sigma_eps^2, isolating variance-shape changes from global-noise changes.
    """
    v = raw_variance_multiplier(x, scenario=scenario, gamma=gamma, cutoff_scale=cutoff_scale)
    if normalize == "none" or scenario.upper() == "H0":
        return v
    if normalize == "sample":
        denom = float(np.mean(v))
    else:
        raise ValueError("normalize must be either 'sample' or 'none'.")
    if denom <= 0 or not math.isfinite(denom):
        raise ValueError("Invalid normalization denominator.")
    return v / denom


def make_dataset(
    n: int,
    scenario: str,
    rng: np.random.Generator,
    gamma: float,
    sigma_eps: float = SIGMA_EPS,
    normalize: str = "sample",
    cutoff_scale: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a single Monte Carlo dataset.

    Returns
    -------
    y : outcome
    x : running variable
    v_norm : normalized variance multiplier used in the error term
    """
    x = draw_running_variable(n, rng)
    z = rng.normal(loc=0.0, scale=1.0, size=n)
    return make_dataset_from_draws(
        x=x,
        z=z,
        scenario=scenario,
        gamma=gamma,
        sigma_eps=sigma_eps,
        normalize=normalize,
        cutoff_scale=cutoff_scale,
    )


def make_dataset_from_draws(
    x: np.ndarray,
    z: np.ndarray,
    scenario: str,
    gamma: float,
    sigma_eps: float = SIGMA_EPS,
    normalize: str = "sample",
    cutoff_scale: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct one scenario from a fixed X sample and fixed base shocks."""
    v_norm = normalized_variance_multiplier(
        x, scenario=scenario, gamma=gamma, normalize=normalize, cutoff_scale=cutoff_scale
    )
    eps = sigma_eps * np.sqrt(v_norm) * z
    y = mu_baseline(x) + eps
    return y, x, v_norm


# ---------------------------------------------------------------------------
# 3. rdrobust wrappers and safe extraction
# ---------------------------------------------------------------------------


def import_rdrobust():
    """Import rdrobust with a clean error message if missing."""
    try:
        from rdrobust import rdrobust  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Could not import rdrobust. Install it first with: pip install rdrobust"
        ) from exc
    return rdrobust


def _scalar_from_table(obj: Any, row: int, col: int = 0) -> float:
    """Extract a scalar robustly from pandas/numpy rdrobust outputs."""
    if obj is None:
        return float("nan")
    if isinstance(obj, pd.DataFrame):
        return float(obj.iloc[row, col])
    if isinstance(obj, pd.Series):
        return float(obj.iloc[row])
    arr = np.asarray(obj)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return float(arr[row])
    return float(arr[row, col])


def _pair_from_attr(obj: Any, attr_name: str) -> Tuple[float, float]:
    """Extract a left/right pair from an rdrobust attribute if available."""
    if not hasattr(obj, attr_name):
        return float("nan"), float("nan")
    val = getattr(obj, attr_name)
    if val is None:
        return float("nan"), float("nan")
    arr = np.asarray(val, dtype=float).reshape(-1)
    if len(arr) >= 2:
        return float(arr[0]), float(arr[1])
    return float("nan"), float("nan")


def _bandwidth_pair(result: Any, row_name: str) -> Tuple[float, float]:
    """Extract h or b left/right bandwidths from result.bws."""
    if not hasattr(result, "bws") or result.bws is None:
        return float("nan"), float("nan")
    bws = result.bws
    try:
        vals = np.asarray(bws.loc[row_name, :], dtype=float).reshape(-1)
    except Exception:
        # Fallback: assume row 0 is h and row 1 is b.
        row_idx = 0 if row_name == "h" else 1
        vals = np.asarray(bws, dtype=float)[row_idx, :].reshape(-1)
    if len(vals) >= 2:
        return float(vals[0]), float(vals[1])
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    return float("nan"), float("nan")


def extract_rdrobust_result(
    result: Any,
    spec: EstimationSpec,
    scenario: str,
    rep: int,
    true_tau: float,
    n: int,
    gamma: float,
) -> Dict[str, Any]:
    """Extract one row from rdrobust(..., all=True)."""
    row = RDROBUST_ROWS[spec.rd_row]
    ci_l = _scalar_from_table(result.ci, row=row, col=0)
    ci_r = _scalar_from_table(result.ci, row=row, col=1)
    estimate = _scalar_from_table(result.coef, row=row, col=0)
    se = _scalar_from_table(result.se, row=row, col=0)
    h_l, h_r = _bandwidth_pair(result, "h")
    b_l, b_r = _bandwidth_pair(result, "b")
    n_h_l, n_h_r = _pair_from_attr(result, "N_h")
    n_b_l, n_b_r = _pair_from_attr(result, "N_b")
    covered = bool(ci_l <= true_tau <= ci_r)
    return {
        "rep": rep,
        "scenario": scenario,
        "scenario_label": SCENARIO_LABELS[scenario],
        "variance_structure": SCENARIO_FORMULAS[scenario],
        "gamma": gamma,
        "n": n,
        "spec_id": spec.spec_id,
        "paper_label": spec.paper_label,
        "rd_row": spec.rd_row,
        "bwselect": spec.bwselect,
        "vce": spec.vce,
        "success": True,
        "error_message": "",
        "estimate": estimate,
        "se": se,
        "ci_l": ci_l,
        "ci_r": ci_r,
        "covered": int(covered),
        "interval_length": ci_r - ci_l,
        "center_bias": estimate - true_tau,
        "h_l": h_l,
        "h_r": h_r,
        "b_l": b_l,
        "b_r": b_r,
        "N_h_l": n_h_l,
        "N_h_r": n_h_r,
        "N_b_l": n_b_l,
        "N_b_r": n_b_r,
    }


def failed_result_row(
    spec: EstimationSpec,
    scenario: str,
    rep: int,
    n: int,
    gamma: float,
    exc: Exception,
) -> Dict[str, Any]:
    """Return a placeholder row when rdrobust fails."""
    return {
        "rep": rep,
        "scenario": scenario,
        "scenario_label": SCENARIO_LABELS[scenario],
        "variance_structure": SCENARIO_FORMULAS[scenario],
        "gamma": gamma,
        "n": n,
        "spec_id": spec.spec_id,
        "paper_label": spec.paper_label,
        "rd_row": spec.rd_row,
        "bwselect": spec.bwselect,
        "vce": spec.vce,
        "success": False,
        "error_message": repr(exc),
        "estimate": np.nan,
        "se": np.nan,
        "ci_l": np.nan,
        "ci_r": np.nan,
        "covered": np.nan,
        "interval_length": np.nan,
        "center_bias": np.nan,
        "h_l": np.nan,
        "h_r": np.nan,
        "b_l": np.nan,
        "b_r": np.nan,
        "N_h_l": np.nan,
        "N_h_r": np.nan,
        "N_b_l": np.nan,
        "N_b_r": np.nan,
    }


def run_estimation_specs(
    y: np.ndarray,
    x: np.ndarray,
    specs: Sequence[EstimationSpec],
    scenario: str,
    rep: int,
    n: int,
    gamma: float,
    true_tau: float,
    kernel: str,
    p: int,
    q: int,
    level: float,
    masspoints: str,
) -> List[Dict[str, Any]]:
    """Run all rdrobust specifications for one dataset."""
    rdrobust = import_rdrobust()
    rows: List[Dict[str, Any]] = []

    # Avoid repeated calls for specs that share bwselect and vce.
    cache: Dict[Tuple[str, str], Any] = {}

    for spec in specs:
        key = (spec.bwselect, spec.vce)
        try:
            if key not in cache:
                cache[key] = rdrobust(
                    y=y,
                    x=x,
                    c=0,
                    p=p,
                    q=q,
                    kernel=kernel,
                    bwselect=spec.bwselect,
                    vce=spec.vce,
                    level=level,
                    masspoints=masspoints,
                    all=True,
                )
            rows.append(
                extract_rdrobust_result(
                    cache[key],
                    spec=spec,
                    scenario=scenario,
                    rep=rep,
                    true_tau=true_tau,
                    n=n,
                    gamma=gamma,
                )
            )
        except Exception as exc:
            rows.append(failed_result_row(spec, scenario, rep, n, gamma, exc))
    return rows


# ---------------------------------------------------------------------------
# 4. Monte Carlo, summaries, interpretation
# ---------------------------------------------------------------------------


def run_monte_carlo(
    reps: int,
    n: int,
    seed: int,
    scenarios: Sequence[str],
    gamma: float,
    sigma_eps: float,
    normalize: str,
    cutoff_scale: float,
    kernel: str,
    p: int,
    q: int,
    level: float,
    masspoints: str,
    include_cerrd: bool,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Run the heteroskedasticity Monte Carlo design."""
    specs = list(MAIN_SPECS)
    if include_cerrd:
        specs.append(CER_SPEC)

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    total = reps * len(scenarios)
    counter = 0

    checked_scenarios = []
    for scenario in scenarios:
        scenario = scenario.upper()
        if scenario not in SCENARIO_LABELS:
            raise ValueError(f"Unknown scenario: {scenario}")
        checked_scenarios.append(scenario)

    for rep in range(1, reps + 1):
        x = draw_running_variable(n, rng)
        z = rng.normal(loc=0.0, scale=1.0, size=n)

        for scenario in checked_scenarios:
            y, _, _ = make_dataset_from_draws(
                x=x,
                z=z,
                scenario=scenario,
                gamma=gamma,
                sigma_eps=sigma_eps,
                normalize=normalize,
                cutoff_scale=cutoff_scale,
            )
            counter += 1
            rows.extend(
                run_estimation_specs(
                    y=y,
                    x=x,
                    specs=specs,
                    scenario=scenario,
                    rep=rep,
                    n=n,
                    gamma=gamma,
                    true_tau=TRUE_TAU_BASELINE,
                    kernel=kernel,
                    p=p,
                    q=q,
                    level=level,
                    masspoints=masspoints,
                )
            )
            if progress_every > 0 and counter % progress_every == 0:
                print(f"Progress: {counter}/{total} scenario-rep datasets completed", flush=True)

    return pd.DataFrame(rows)


def summarize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Summarize Monte Carlo results by scenario and method spec."""
    summary_rows: List[Dict[str, Any]] = []
    group_cols = [
        "scenario",
        "scenario_label",
        "variance_structure",
        "gamma",
        "spec_id",
        "paper_label",
        "rd_row",
        "bwselect",
        "vce",
    ]
    for keys, g in raw.groupby(group_cols, dropna=False):
        g_ok = g[g["success"] == True].copy()
        total = len(g)
        successful = len(g_ok)
        failure_rate = 1.0 - successful / total if total else np.nan
        if successful == 0:
            summary_rows.append(dict(zip(group_cols, keys)) | {
                "reps_total": total,
                "reps_success": successful,
                "failure_rate": failure_rate,
                "empirical_coverage": np.nan,
                "coverage_error_vs_0.95": np.nan,
                "mcse_coverage": np.nan,
                "avg_interval_length": np.nan,
                "avg_center_bias": np.nan,
                "avg_estimate": np.nan,
                "sd_estimate": np.nan,
                "avg_se": np.nan,
                "se_to_empirical_sd_ratio": np.nan,
                "avg_h_l": np.nan,
                "avg_h_r": np.nan,
                "avg_b_l": np.nan,
                "avg_b_r": np.nan,
                "avg_N_h_l": np.nan,
                "avg_N_h_r": np.nan,
                "avg_N_b_l": np.nan,
                "avg_N_b_r": np.nan,
            })
            continue

        ec = float(g_ok["covered"].mean())
        sd_estimate = float(g_ok["estimate"].std(ddof=1)) if successful > 1 else np.nan
        avg_se = float(g_ok["se"].mean())
        summary_rows.append(dict(zip(group_cols, keys)) | {
            "reps_total": total,
            "reps_success": successful,
            "failure_rate": failure_rate,
            "empirical_coverage": ec,
            "coverage_error_vs_0.95": ec - 0.95,
            "mcse_coverage": math.sqrt(ec * (1.0 - ec) / successful) if successful > 0 else np.nan,
            "avg_interval_length": float(g_ok["interval_length"].mean()),
            "avg_center_bias": float(g_ok["center_bias"].mean()),
            "avg_estimate": float(g_ok["estimate"].mean()),
            "sd_estimate": sd_estimate,
            "avg_se": avg_se,
            "se_to_empirical_sd_ratio": avg_se / sd_estimate if sd_estimate and sd_estimate > 0 else np.nan,
            "avg_h_l": float(g_ok["h_l"].mean()),
            "avg_h_r": float(g_ok["h_r"].mean()),
            "avg_b_l": float(g_ok["b_l"].mean()),
            "avg_b_r": float(g_ok["b_r"].mean()),
            "avg_N_h_l": float(g_ok["N_h_l"].mean()),
            "avg_N_h_r": float(g_ok["N_h_r"].mean()),
            "avg_N_b_l": float(g_ok["N_b_l"].mean()),
            "avg_N_b_r": float(g_ok["N_b_r"].mean()),
        })
    out = pd.DataFrame(summary_rows)
    order_map = {s: i for i, s in enumerate(DEFAULT_SCENARIOS)}
    out["scenario_order"] = out["scenario"].map(order_map).fillna(99)
    out = out.sort_values(["scenario_order", "spec_id"]).drop(columns=["scenario_order"])
    return out


def classify_interpretation(row: pd.Series) -> str:
    """
    Mechanical interpretation for the final table.

    This is intentionally simple and should be edited after you inspect results.
    """
    scenario = row.get("scenario", "")
    conv_ec = row.get("conv_nn_ec", np.nan)
    rbc_ec = row.get("rbc_nn_ec", np.nan)

    if scenario == "H0":
        return "Baseline comparison"
    if not np.isfinite(conv_ec) or not np.isfinite(rbc_ec):
        return "Check failures"

    gain = rbc_ec - conv_ec

    rbc_close = 0.935 <= rbc_ec <= 0.965
    conv_close = 0.935 <= conv_ec <= 0.965

    if conv_close and rbc_close and abs(gain) < 0.01:
        verdict = "Robust SE alone broadly sufficient"
    elif rbc_close and gain >= 0.015:
        verdict = "RBC adds value beyond robust SE"
    elif rbc_ec >= 0.925 and gain >= 0.015:
        verdict = "RBC improves coverage, not fully restored"
    elif rbc_ec < 0.925 and gain >= 0.015:
        verdict = "RBC helps, but heteroskedasticity remains severe"
    elif rbc_ec < 0.925 and abs(gain) < 0.015:
        verdict = "Not solved by RBC under this design"
    else:
        verdict = "Mixed; inspect SE and interval length"

    return verdict


def make_interpretation_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Create a wide table for the paper's heteroskedasticity section."""
    cols = [
        "scenario",
        "scenario_label",
        "variance_structure",
        "spec_id",
        "empirical_coverage",
        "coverage_error_vs_0.95",
        "avg_interval_length",
        "avg_center_bias",
        "avg_se",
        "sd_estimate",
        "se_to_empirical_sd_ratio",
        "failure_rate",
    ]
    s = summary[cols].copy()
    wide = s.pivot_table(
        index=["scenario", "scenario_label", "variance_structure"],
        columns="spec_id",
        values=[
            "empirical_coverage",
            "coverage_error_vs_0.95",
            "avg_interval_length",
            "avg_center_bias",
            "avg_se",
            "sd_estimate",
            "se_to_empirical_sd_ratio",
            "failure_rate",
        ],
        aggfunc="first",
    )
    wide.columns = [f"{spec}_{metric}" for metric, spec in wide.columns]
    wide = wide.reset_index()

    rename = {
        "conv_nn_mserd_empirical_coverage": "conv_nn_ec",
        "rbc_nn_mserd_empirical_coverage": "rbc_nn_ec",
        "rbc_nn_cerrd_empirical_coverage": "rbc_cerrd_ec",
        "conv_nn_mserd_avg_interval_length": "conv_nn_il",
        "rbc_nn_mserd_avg_interval_length": "rbc_nn_il",
        "rbc_nn_cerrd_avg_interval_length": "rbc_cerrd_il",
        "conv_nn_mserd_avg_center_bias": "conv_nn_bias",
        "rbc_nn_mserd_avg_center_bias": "rbc_nn_bias",
        "rbc_nn_mserd_se_to_empirical_sd_ratio": "rbc_nn_se_ratio",
        "conv_nn_mserd_se_to_empirical_sd_ratio": "conv_nn_se_ratio",
    }
    wide = wide.rename(columns=rename)
    if "conv_nn_ec" in wide.columns and "rbc_nn_ec" in wide.columns:
        wide["rbc_minus_conv_ec"] = wide["rbc_nn_ec"] - wide["conv_nn_ec"]
    else:
        wide["rbc_minus_conv_ec"] = np.nan
    wide["interpretation"] = wide.apply(classify_interpretation, axis=1)

    order_map = {s: i for i, s in enumerate(DEFAULT_SCENARIOS)}
    wide["scenario_order"] = wide["scenario"].map(order_map).fillna(99)
    wide = wide.sort_values("scenario_order").drop(columns=["scenario_order"])
    return wide


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------


def save_variance_function_plot(outdir: Path, gamma: float, cutoff_scale: float) -> None:
    """Plot normalized variance multiplier functions for H0-H3."""
    x_grid = np.linspace(-1, 1, 401)
    fig, ax = plt.subplots(figsize=(9, 5))
    # Approximate expected normalization denominator under X = 2*Beta(2,4)-1.
    rng = np.random.default_rng(20260613)
    x_ref = draw_running_variable(200_000, rng)
    for scenario in DEFAULT_SCENARIOS:
        v_grid_raw = raw_variance_multiplier(x_grid, scenario, gamma, cutoff_scale)
        v_ref_raw = raw_variance_multiplier(x_ref, scenario, gamma, cutoff_scale)
        if scenario == "H0":
            v_plot = v_grid_raw
        else:
            v_plot = v_grid_raw / float(np.mean(v_ref_raw))
        ax.plot(x_grid, v_plot, label=f"{scenario}: {SCENARIO_LABELS[scenario]}")
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.axhline(1, linestyle=":", linewidth=1)
    ax.set_title("Baseline heteroskedasticity extension: normalized variance multipliers")
    ax.set_xlabel("Running variable X")
    ax.set_ylabel("Normalized variance multiplier")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig_hetero_variance_functions.png", dpi=200)
    plt.close(fig)


def save_grouped_bar(
    data: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    filename: Path,
    nominal_line: Optional[float] = None,
) -> None:
    """Grouped bar plot for main methods across scenarios."""
    main = data[data["spec_id"].isin(["conv_nn_mserd", "rbc_nn_mserd"])].copy()
    scenarios = list(DEFAULT_SCENARIOS)
    labels = [SCENARIO_LABELS[s].replace(" ", "\n") for s in scenarios]
    methods = ["conv_nn_mserd", "rbc_nn_mserd"]
    method_labels = {
        "conv_nn_mserd": "Conv. + robust SE",
        "rbc_nn_mserd": "RBC + robust SE",
    }
    method_colors = {
        "conv_nn_mserd": "#4c78a8",
        "rbc_nn_mserd": "#f58518",
    }
    is_coverage = "coverage" in value_col

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    x = np.arange(len(scenarios))
    width = 0.36
    offsets = [-width / 2, width / 2]
    plotted_vals: List[float] = []
    for j, method in enumerate(methods):
        vals = []
        for s in scenarios:
            sub = main[(main["scenario"] == s) & (main["spec_id"] == method)]
            vals.append(float(sub[value_col].iloc[0]) if len(sub) else np.nan)
        plotted_vals.extend([v for v in vals if np.isfinite(v)])
        bars = ax.bar(
            x + offsets[j],
            vals,
            width,
            label=method_labels[method],
            color=method_colors[method],
        )
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                text = f"{val:.1%}" if is_coverage else f"{val:.3f}"
                label_pad = 0.004 if is_coverage else max(abs(val) * 0.015, 0.003)
                ax.text(bar.get_x() + bar.get_width() / 2, val + label_pad, text,
                        ha="center", va="bottom", fontsize=8)
    if nominal_line is not None:
        ax.axhline(nominal_line, linestyle="--", linewidth=1.2, color="#222222")
        ax.annotate(
            "Nominal 95%",
            xy=(0.985, nominal_line),
            xycoords=("axes fraction", "data"),
            xytext=(-4, 6),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9,
            color="#222222",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
    if plotted_vals:
        if is_coverage:
            low = max(0.0, min(0.75, min(plotted_vals) - 0.06))
            high = min(1.02, max(0.99, max(plotted_vals) + 0.045, (nominal_line or 0.95) + 0.04))
            ax.set_ylim(low, high)
        else:
            high = max(plotted_vals)
            ax.set_ylim(0.0, high * 1.14 if high > 0 else 1.0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.26, top=0.86)
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def save_rbc_gain_plot(summary: pd.DataFrame, outdir: Path) -> None:
    interp = make_interpretation_table(summary)
    if "rbc_minus_conv_ec" not in interp.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(interp))
    vals = interp["rbc_minus_conv_ec"].to_numpy(dtype=float)
    bars = ax.bar(x, vals)
    ax.axhline(0, linewidth=1)
    for bar, val in zip(bars, vals):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1%}",
                    ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(interp["scenario"], fontsize=9)
    ax.set_ylabel("RBC EC - Conventional EC")
    ax.set_title("Baseline heteroskedasticity extension: RBC coverage gain")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig_hetero_rbc_coverage_gain.png", dpi=200)
    plt.close(fig)


def save_coverage_length_tradeoff(summary: pd.DataFrame, outdir: Path) -> None:
    main = summary[summary["spec_id"].isin(["conv_nn_mserd", "rbc_nn_mserd"])].copy()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for _, row in main.iterrows():
        ax.scatter(row["avg_interval_length"], row["empirical_coverage"])
        ax.text(row["avg_interval_length"], row["empirical_coverage"],
                f" {row['scenario']} {row['rd_row']}", fontsize=8)
    ax.axhline(0.95, linestyle="--", linewidth=1)
    ax.set_xlabel("Average interval length")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Baseline heteroskedasticity extension: coverage-length tradeoff")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig_hetero_coverage_length_tradeoff.png", dpi=200)
    plt.close(fig)


def make_plots(summary: pd.DataFrame, outdir: Path, gamma: float, cutoff_scale: float) -> None:
    ensure_output_dir(outdir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    save_variance_function_plot(figdir, gamma=gamma, cutoff_scale=cutoff_scale)
    save_grouped_bar(
        summary,
        value_col="empirical_coverage",
        title="Baseline heteroskedasticity extension: empirical coverage of 95% CIs",
        ylabel="Empirical coverage",
        filename=figdir / "fig_hetero_coverage_main.png",
        nominal_line=0.95,
    )
    save_grouped_bar(
        summary,
        value_col="avg_interval_length",
        title="Baseline heteroskedasticity extension: average interval length",
        ylabel="Average interval length",
        filename=figdir / "fig_hetero_interval_length_main.png",
        nominal_line=None,
    )
    save_rbc_gain_plot(summary, figdir)
    save_coverage_length_tradeoff(summary, figdir)


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------


def ensure_output_dir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def run_config_path(outdir: Path) -> Path:
    if outdir.name.lower() == "results":
        return outdir.parent / "run_config_hetero_paired.txt"
    return outdir / "run_config_hetero_paired.txt"


def write_run_config(args: argparse.Namespace, outdir: Path) -> None:
    ensure_output_dir(outdir)
    lines = [
        "Baseline heteroskedasticity extension config (paired design)",
        "design=paired_X_and_base_normal_shock_by_replication",
        f"reps={args.reps}",
        f"n={args.n}",
        f"seed={args.seed}",
        f"true_tau={TRUE_TAU_BASELINE}",
        f"sigma_eps={args.sigma_eps}",
        f"gamma={args.gamma}",
        f"normalize={args.normalize}",
        f"cutoff_scale={args.cutoff_scale}",
        f"scenarios={','.join(args.scenarios)}",
        f"main_bwselect=mserd",
        f"main_vce=nn",
        f"include_cerrd={args.include_cerrd}",
        f"kernel={args.kernel}",
        f"p={args.p}",
        f"q={args.q}",
        f"level={args.level}",
        f"masspoints={args.masspoints}",
    ]
    run_config_path(outdir).write_text("\n".join(lines), encoding="utf-8")


def print_terminal_summary(summary: pd.DataFrame, interpretation: pd.DataFrame) -> None:
    main = summary[summary["spec_id"].isin(["conv_nn_mserd", "rbc_nn_mserd"])].copy()
    if not main.empty:
        main["method"] = main["spec_id"].map(
            {
                "conv_nn_mserd": "Conventional",
                "rbc_nn_mserd": "RBC",
            }
        )
        cols = [
            "scenario",
            "scenario_label",
            "method",
            "reps_success",
            "failure_rate",
            "empirical_coverage",
            "avg_interval_length",
            "avg_center_bias",
            "avg_h_l",
            "avg_h_r",
            "avg_b_l",
            "avg_b_r",
        ]
        print("\nConcise summary:")
        print(
            main[cols].to_string(
                index=False,
                formatters={
                    "failure_rate": lambda v: "" if pd.isna(v) else f"{100 * v:.1f}%",
                    "empirical_coverage": lambda v: "" if pd.isna(v) else f"{100 * v:.1f}%",
                    "avg_interval_length": lambda v: "" if pd.isna(v) else f"{v:.3f}",
                    "avg_center_bias": lambda v: "" if pd.isna(v) else f"{v:.3f}",
                },
            )
        )

    interp_cols = [
        "scenario",
        "conv_nn_ec",
        "rbc_nn_ec",
        "rbc_minus_conv_ec",
        "interpretation",
    ]
    existing_cols = [c for c in interp_cols if c in interpretation.columns]
    if existing_cols:
        print("\nInterpretation summary:")
        print(
            interpretation[existing_cols].to_string(
                index=False,
                formatters={
                    "conv_nn_ec": lambda v: "" if pd.isna(v) else f"{100 * v:.1f}%",
                    "rbc_nn_ec": lambda v: "" if pd.isna(v) else f"{100 * v:.1f}%",
                    "rbc_minus_conv_ec": lambda v: "" if pd.isna(v) else f"{100 * v:.1f} pp",
                },
            )
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline heteroskedasticity extension Monte Carlo"
    )
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--sigma-eps", type=float, default=SIGMA_EPS)
    parser.add_argument("--normalize", choices=["sample", "none"], default="sample")
    parser.add_argument("--cutoff-scale", type=float, default=0.10)
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--kernel", type=str, default="tri")
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--level", type=float, default=95.0)
    parser.add_argument("--masspoints", type=str, default="off")
    parser.add_argument("--include-cerrd", action="store_true")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    ensure_output_dir(outdir)
    config_path = run_config_path(outdir)

    # Fail early instead of spending all replications recording the same import error.
    import_rdrobust()

    print("CCT baseline heteroskedasticity extension")
    print(f"Output directory: {outdir}")
    print(f"reps={args.reps}, n={args.n}, seed={args.seed}")
    print(f"true_tau={TRUE_TAU_BASELINE:.6f}, sigma_eps={args.sigma_eps:.4f}")
    print(f"gamma={args.gamma:.3f}, normalize={args.normalize}, cutoff_scale={args.cutoff_scale:.3f}")
    print(f"scenarios={','.join(args.scenarios)}, bwselect=mserd, vce=nn, kernel={args.kernel}, p={args.p}, q={args.q}")
    print(f"include_cerrd={args.include_cerrd}")
    write_run_config(args, outdir)

    raw = run_monte_carlo(
        reps=args.reps,
        n=args.n,
        seed=args.seed,
        scenarios=args.scenarios,
        gamma=args.gamma,
        sigma_eps=args.sigma_eps,
        normalize=args.normalize,
        cutoff_scale=args.cutoff_scale,
        kernel=args.kernel,
        p=args.p,
        q=args.q,
        level=args.level,
        masspoints=args.masspoints,
        include_cerrd=args.include_cerrd,
        progress_every=args.progress_every,
    )
    ensure_output_dir(outdir)
    raw_path = outdir / "hetero_raw.csv"
    raw.to_csv(raw_path, index=False)

    summary = summarize_raw(raw)
    ensure_output_dir(outdir)
    summary_path = outdir / "hetero_summary_long.csv"
    summary.to_csv(summary_path, index=False)

    interpretation = make_interpretation_table(summary)
    ensure_output_dir(outdir)
    interp_path = outdir / "hetero_interpretation_table.csv"
    interpretation.to_csv(interp_path, index=False)

    make_plots(summary, outdir=outdir, gamma=args.gamma, cutoff_scale=args.cutoff_scale)

    print_terminal_summary(summary, interpretation)

    print(f"\nSaved raw results to:       {raw_path}")
    print(f"Saved summary results to:   {summary_path}")
    print(f"Saved interpretation to:    {interp_path}")
    print(f"Saved figures to:           {outdir / 'figures'}")
    print(f"Saved run config to:        {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
