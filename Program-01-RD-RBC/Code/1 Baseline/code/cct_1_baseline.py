#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCT (2014) baseline design Monte Carlo comparison.

Purpose
-------
Replicate the CCT baseline sharp RD design and compare alternative confidence
interval constructions under the same DGP, sample size, and rdrobust settings.

Design
------
Running variable: X = 2 * Beta(2, 4) - 1.
Outcome: CCT baseline regression function with homoskedastic error.
True treatment effect: tau = mu_+(0) - mu_-(0).

Methods
-------
1. conventional_mserd
   Conventional local-linear CI using replication-specific data-driven mserd bandwidth.

2. undersmoothing_mserd
   Conventional local-linear CI using h_US = us_scale * h_mserd.
   Default us_scale is 0.75: a mild and still defensible undersmoothing benchmark.

3. bias_corrected_mserd
   Bias-corrected point estimator using the non-robust / conventional SE row from rdrobust.

4. rbc_mserd
   Robust bias-corrected CI from rdrobust: bias-corrected estimator + robust SE.

5. rbc_cerrd
   Robust bias-corrected CI using rdrobust's coverage-error-rate-optimal bandwidth.

Run examples
------------
Quick check:
    python cct_1_baseline.py --reps 100 --n 500

Main run:
    python cct_1_baseline.py --reps 5000 --n 500 --us-scale 0.75

More aggressive / older undersmoothing benchmark:
    python cct_1_baseline.py --reps 5000 --n 500 --us-scale 0.50 \
        --outdir outputs/cct_baseline_us050

Outputs
-------
    If this file is stored inside a code/ folder, output is written to the
    sibling results/ folder. If the file is run as a standalone script, output
    is written to cct_1_baseline_results/ next to the script.

    run_config.txt
    results or cct_1_baseline_results/
        baseline_raw.csv
        baseline_summary.csv
        figures/baseline_regression_function.png
        figures/coverage_bar.png
        figures/interval_length_bar.png
        figures/bias_bar.png
        figures/coverage_length_tradeoff.png
        figures/bandwidth_distribution.png
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Baseline design constants from the CCT simulation setup
# -----------------------------------------------------------------------------
CUTOFF = 0.0
SIGMA_EPS = 0.1295

# Baseline true RD effect: mu_+(0) - mu_-(0)
TRUE_TAU = 0.26 - 3.71


def default_output_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    if script_dir.name.lower() == "code":
        return script_dir.parent / "results"
    return script_dir / f"{Path(__file__).resolve().stem}_results"


DEFAULT_OUTDIR = default_output_dir()


METHOD_DISPLAY = {
    "conventional_mserd": "Conventional MSE",
    "undersmoothing_mserd": "Undersmoothing",
    "bias_corrected_mserd": "Bias-corrected MSE",
    "rbc_mserd": "RBC-MSE",
    "rbc_cerrd": "RBC-CER",
}

METHOD_ORDER = [
    "conventional_mserd",
    "undersmoothing_mserd",
    "bias_corrected_mserd",
    "rbc_mserd",
    "rbc_cerrd",
]

METHOD_COLORS = {
    "conventional_mserd": "#4c78a8",
    "undersmoothing_mserd": "#f58518",
    "bias_corrected_mserd": "#54a24b",
    "rbc_mserd": "#b279a2",
    "rbc_cerrd": "#e45756",
}

# Rows in rdrobust output.
RDROBUST_ROWS = {
    "conventional_mserd": "Conventional",
    "undersmoothing_mserd": "Conventional",
    "bias_corrected_mserd": "Bias-Corrected",
    "rbc_mserd": "Robust",
    "rbc_cerrd": "Robust",
}

RAW_COLUMNS = [
    "rep",
    "method",
    "method_label",
    "bandwidth_type",
    "success",
    "error",
    "tau_true",
    "estimate",
    "se",
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
    "kernel",
    "p",
    "q",
    "us_scale",
]


@dataclass(frozen=True)
class SimConfig:
    reps: int = 5000
    n: int = 500
    seed: int = 12345
    outdir: Path = DEFAULT_OUTDIR
    bwselect: str = "mserd"
    vce: str = "nn"
    kernel: str = "tri"
    p: int = 1
    q: int = 2
    level: float = 95.0
    masspoints: str = "off"
    us_scale: float = 0.75
    progress_every: int = 100


# -----------------------------------------------------------------------------
# Baseline regression function
# -----------------------------------------------------------------------------
def mu_baseline_left(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x)
    out = (
        3.71
        + 2.30 * x_arr
        + 3.28 * x_arr**2
        + 1.45 * x_arr**3
        + 0.23 * x_arr**4
        + 0.03 * x_arr**5
    )
    if np.isscalar(x):
        return float(out)
    return out


def mu_baseline_right(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x)
    out = (
        0.26
        + 18.49 * x_arr
        - 54.81 * x_arr**2
        + 74.30 * x_arr**3
        - 45.02 * x_arr**4
        + 9.83 * x_arr**5
    )
    if np.isscalar(x):
        return float(out)
    return out


def mu_baseline(x: np.ndarray | float) -> np.ndarray | float:
    """Baseline regression function, based on Ludwig and Miller (2007)."""
    x_arr = np.asarray(x)
    out = np.where(x_arr < 0, mu_baseline_left(x_arr), mu_baseline_right(x_arr))
    if np.isscalar(x):
        return float(out)
    return out


def draw_dataset(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw one simulated dataset from the baseline design."""
    x = 2.0 * rng.beta(2.0, 4.0, size=n) - 1.0
    eps = rng.normal(loc=0.0, scale=SIGMA_EPS, size=n)
    y = mu_baseline(x) + eps
    return y, x


# -----------------------------------------------------------------------------
# rdrobust wrapper utilities
# -----------------------------------------------------------------------------
def import_rdrobust():
    """Import rdrobust function with a helpful error message."""
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


def extract_count_pair(fit: Any, name: str) -> tuple[float, float]:
    value = safe_attr(fit, name, None)
    if value is None:
        return np.nan, np.nan
    try:
        return float(value[0]), float(value[1])
    except Exception:
        return np.nan, np.nan


def fit_bandwidth_type(method: str) -> str:
    if method == "undersmoothing_mserd":
        return "ad hoc h = us_scale * h_mserd"
    if method == "rbc_cerrd":
        return "data-driven cerrd"
    return "data-driven mserd"


def extract_fit_row(
    *,
    fit: Any,
    rep: int,
    method: str,
    cfg: SimConfig,
    true_tau: float = TRUE_TAU,
    error: str = "",
) -> Dict[str, Any]:
    """Extract one method row from an rdrobust result object."""
    rd_row = RDROBUST_ROWS[method]

    coef = safe_attr(fit, "coef")
    se = safe_attr(fit, "se")
    ci = safe_attr(fit, "ci")

    estimate = first_value(coef.loc[rd_row])
    std_error = first_value(se.loc[rd_row])
    ci_lower = float(ci.loc[rd_row].iloc[0])
    ci_upper = float(ci.loc[rd_row].iloc[1])

    N_left, N_right = extract_count_pair(fit, "N")
    N_h_left, N_h_right = extract_count_pair(fit, "N_h")
    N_b_left, N_b_right = extract_count_pair(fit, "N_b")
    M_left, M_right = extract_count_pair(fit, "M")

    cover = ci_lower <= true_tau <= ci_upper
    miss_below_true = ci_upper < true_tau
    miss_above_true = ci_lower > true_tau

    return {
        "rep": rep,
        "method": method,
        "method_label": METHOD_DISPLAY[method],
        "bandwidth_type": fit_bandwidth_type(method),
        "success": True,
        "error": error,
        "tau_true": true_tau,
        "estimate": estimate,
        "se": std_error,
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
        "bwselect": safe_attr(fit, "bwselect", ""),
        "vce": safe_attr(fit, "vce", ""),
        "kernel": safe_attr(fit, "kernel", ""),
        "p": safe_attr(fit, "p", np.nan),
        "q": safe_attr(fit, "q", np.nan),
        "us_scale": cfg.us_scale if method == "undersmoothing_mserd" else np.nan,
    }


def failed_row(rep: int, method: str, cfg: SimConfig, exc: BaseException) -> Dict[str, Any]:
    row = {col: np.nan for col in RAW_COLUMNS}
    row.update(
        {
            "rep": rep,
            "method": method,
            "method_label": METHOD_DISPLAY[method],
            "bandwidth_type": fit_bandwidth_type(method),
            "success": False,
            "error": repr(exc),
            "tau_true": TRUE_TAU,
            "cover": np.nan,
            "miss_below_true": np.nan,
            "miss_above_true": np.nan,
            "us_scale": cfg.us_scale if method == "undersmoothing_mserd" else np.nan,
        }
    )
    return row


def run_base_mserd(y: np.ndarray, x: np.ndarray, cfg: SimConfig) -> Any:
    rdrobust = import_rdrobust()
    return rdrobust(
        y=y,
        x=x,
        c=CUTOFF,
        p=cfg.p,
        q=cfg.q,
        kernel=cfg.kernel,
        bwselect=cfg.bwselect,
        vce=cfg.vce,
        level=cfg.level,
        masspoints=cfg.masspoints,
        all=True,
    )


def run_rbc_cerrd(y: np.ndarray, x: np.ndarray, cfg: SimConfig) -> Any:
    """Run RBC inference with rdrobust's CER-optimal RD bandwidth selector."""
    rdrobust = import_rdrobust()
    return rdrobust(
        y=y,
        x=x,
        c=CUTOFF,
        p=cfg.p,
        q=cfg.q,
        kernel=cfg.kernel,
        bwselect="cerrd",
        vce=cfg.vce,
        level=cfg.level,
        masspoints=cfg.masspoints,
        all=True,
    )


def run_undersmoothing(y: np.ndarray, x: np.ndarray, base_fit: Any, cfg: SimConfig) -> Any:
    """Run conventional local-linear CI with h_US = us_scale * h_mserd."""
    rdrobust = import_rdrobust()

    h_left = extract_bandwidth(base_fit, "h", "left")
    h_right = extract_bandwidth(base_fit, "h", "right")
    b_left = extract_bandwidth(base_fit, "b", "left")
    b_right = extract_bandwidth(base_fit, "b", "right")

    if not np.all(np.isfinite([h_left, h_right, b_left, b_right])):
        raise RuntimeError("Cannot compute undersmoothing because h/b are not finite.")

    h_us = [cfg.us_scale * h_left, cfg.us_scale * h_right]
    # The conventional row does not use b, but rdrobust with all=True expects a
    # coherent b for the full object. Scaling b consistently avoids hidden changes.
    b_us = [cfg.us_scale * b_left, cfg.us_scale * b_right]

    return rdrobust(
        y=y,
        x=x,
        c=CUTOFF,
        p=cfg.p,
        q=cfg.q,
        h=h_us,
        b=b_us,
        kernel=cfg.kernel,
        vce=cfg.vce,
        level=cfg.level,
        masspoints=cfg.masspoints,
        all=True,
    )


# -----------------------------------------------------------------------------
# Monte Carlo driver
# -----------------------------------------------------------------------------
def run_simulation(cfg: SimConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows: List[Dict[str, Any]] = []

    for rep in range(1, cfg.reps + 1):
        y, x = draw_dataset(cfg.n, rng)

        # Base data-driven mserd object: Conventional, Bias-Corrected, Robust.
        try:
            base_fit = run_base_mserd(y, x, cfg)
            rows.append(extract_fit_row(fit=base_fit, rep=rep, method="conventional_mserd", cfg=cfg))
        except BaseException as exc:
            base_fit = None
            rows.append(failed_row(rep, "conventional_mserd", cfg, exc))

        # Mild undersmoothing benchmark based on the same replication-specific h_mserd.
        try:
            if base_fit is None:
                raise RuntimeError("Base mserd rdrobust failed; undersmoothing skipped.")
            us_fit = run_undersmoothing(y, x, base_fit, cfg)
            rows.append(extract_fit_row(fit=us_fit, rep=rep, method="undersmoothing_mserd", cfg=cfg))
        except BaseException as exc:
            rows.append(failed_row(rep, "undersmoothing_mserd", cfg, exc))

        if base_fit is not None:
            for method in ["bias_corrected_mserd", "rbc_mserd"]:
                try:
                    rows.append(extract_fit_row(fit=base_fit, rep=rep, method=method, cfg=cfg))
                except BaseException as exc:
                    rows.append(failed_row(rep, method, cfg, exc))
        else:
            for method in ["bias_corrected_mserd", "rbc_mserd"]:
                rows.append(failed_row(rep, method, cfg, RuntimeError("Base mserd rdrobust failed.")))

        # Coverage-error-rate-optimal bandwidth target for RBC inference.
        try:
            cerrd_fit = run_rbc_cerrd(y, x, cfg)
            rows.append(extract_fit_row(fit=cerrd_fit, rep=rep, method="rbc_cerrd", cfg=cfg))
        except BaseException as exc:
            rows.append(failed_row(rep, "rbc_cerrd", cfg, exc))

        if cfg.progress_every and rep % cfg.progress_every == 0:
            print(f"Completed {rep}/{cfg.reps} replications", flush=True)

    raw = pd.DataFrame(rows)
    raw = raw.reindex(columns=RAW_COLUMNS)
    return raw


# -----------------------------------------------------------------------------
# Summary and plots
# -----------------------------------------------------------------------------
def monte_carlo_se_for_coverage(coverage_values: pd.Series) -> float:
    s = coverage_values.dropna().astype(float)
    if len(s) == 0:
        return np.nan
    p_hat = float(s.mean())
    return math.sqrt(p_hat * (1.0 - p_hat) / len(s))


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    numeric_summary_cols = [
        "EC",
        "EC_percent",
        "coverage_error",
        "coverage_error_pp",
        "coverage_mcse",
        "IL",
        "avg_estimate",
        "bias",
        "abs_bias",
        "rmse",
        "sd_estimate",
        "avg_se",
        "se_to_sd_ratio",
        "miss_below_true",
        "miss_above_true",
        "avg_h_left",
        "avg_h_right",
        "avg_b_left",
        "avg_b_right",
        "avg_N_h_left",
        "avg_N_h_right",
        "avg_N_b_left",
        "avg_N_b_right",
    ]
    summary_rows = []
    for method in METHOD_ORDER:
        df_m = raw[raw["method"] == method].copy()
        ok = df_m[df_m["success"] == True].copy()
        reps_total = len(df_m)
        reps_success = len(ok)
        fail_rate = 1.0 - reps_success / reps_total if reps_total else np.nan

        if reps_success == 0:
            summary_rows.append(
                {
                    "method": method,
                    "method_label": METHOD_DISPLAY[method],
                    "bandwidth_type": fit_bandwidth_type(method),
                    "reps_total": reps_total,
                    "reps_success": reps_success,
                    "fail_rate": fail_rate,
                    **{col: np.nan for col in numeric_summary_cols},
                }
            )
            continue

        estimates = ok["estimate"].astype(float)
        errors = estimates - TRUE_TAU
        ec = float(ok["cover"].mean())
        avg_est = float(estimates.mean())
        sd_est = float(estimates.std(ddof=1))
        avg_se = float(ok["se"].mean())
        row = {
            "method": method,
            "method_label": METHOD_DISPLAY[method],
            "bandwidth_type": fit_bandwidth_type(method),
            "reps_total": reps_total,
            "reps_success": reps_success,
            "fail_rate": fail_rate,
            "EC": ec,
            "EC_percent": 100.0 * ec,
            "coverage_error": ec - 0.95,
            "coverage_error_pp": 100.0 * (ec - 0.95),
            "coverage_mcse": monte_carlo_se_for_coverage(ok["cover"]),
            "IL": float(ok["ci_length"].mean()),
            "avg_estimate": avg_est,
            "bias": avg_est - TRUE_TAU,
            "abs_bias": float(abs(avg_est - TRUE_TAU)),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "sd_estimate": sd_est,
            "avg_se": avg_se,
            "se_to_sd_ratio": float(avg_se / sd_est) if sd_est > 0 else np.nan,
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
        }
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary["method"] = pd.Categorical(summary["method"], categories=METHOD_ORDER, ordered=True)
    summary = summary.sort_values("method").reset_index(drop=True)
    return summary


def ensure_dirs(outdir: Path) -> Path:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    return figdir


def run_config_path(outdir: Path) -> Path:
    if outdir.name.lower() == "results":
        return outdir.parent / "run_config.txt"
    return outdir / "run_config.txt"


def style_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, color="#8c8c8c", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.9)


def bar_colors(df: pd.DataFrame) -> list[str]:
    return [METHOD_COLORS.get(str(method), "#777777") for method in df["method"]]


def wrap_method_labels(labels: list[str]) -> list[str]:
    return [label.replace(" MSE", "\nMSE") for label in labels]


def set_centered_xticklabels(ax: plt.Axes, labels: list[str]) -> None:
    ax.set_xticks(np.arange(len(labels)), wrap_method_labels(labels))
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")
        label.set_va("top")


def label_nominal_coverage_line(ax: plt.Axes, *, above_plot: bool = False) -> None:
    if above_plot:
        ax.text(
            0.985,
            1.015,
            "Dashed line = nominal 95%",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#222222",
            clip_on=False,
        )
        return

    ax.annotate(
        "Nominal 95%",
        xy=(0.985, 0.95),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#222222",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )


def plot_baseline_design(figdir: Path) -> None:
    x_left = np.linspace(-1.0, CUTOFF, 500)
    x_right = np.linspace(CUTOFF, 1.0, 500)
    y_left = mu_baseline_left(x_left)
    y_right = mu_baseline_right(x_right)
    y0_left = float(mu_baseline_left(CUTOFF))
    y0_right = float(mu_baseline_right(CUTOFF))
    jump_mid = 0.5 * (y0_left + y0_right)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.plot(x_left, y_left, color="#2f6f9f", linewidth=2.8, label="Left-side mean")
    ax.plot(x_right, y_right, color="#b33a4a", linewidth=2.8, label="Right-side mean")
    ax.axvline(CUTOFF, color="#222222", linestyle="--", linewidth=1.3, label="Cutoff c = 0")
    ax.scatter([CUTOFF], [y0_left], s=82, color="#2f6f9f", edgecolor="white", linewidth=0.9, zorder=5)
    ax.scatter([CUTOFF], [y0_right], s=82, color="#b33a4a", edgecolor="white", linewidth=0.9, zorder=5)
    ax.text(
        -0.46,
        jump_mid,
        "jump = -3.45",
        fontsize=12,
        va="center",
        ha="left",
        color="#222222",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )
    ax.set_xlim(-1.1, 1.1)
    y_min = min(float(np.min(y_left)), float(np.min(y_right)), y0_right)
    y_max = max(float(np.max(y_left)), float(np.max(y_right)), y0_left)
    y_pad = 0.08 * (y_max - y_min)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel("Running variable X", fontsize=13)
    ax.set_ylabel("Conditional mean", fontsize=13)
    ax.set_title("CCT (2014) Baseline Sharp RD Design", fontsize=16, pad=10)
    ax.legend(frameon=False, loc="best", fontsize=12)
    style_axes(ax, grid_axis="both")
    fig.tight_layout()
    fig.savefig(figdir / "baseline_regression_function.png", dpi=300)
    plt.close(fig)


def plot_coverage_bar(summary: pd.DataFrame, figdir: Path) -> None:
    df = summary.dropna(subset=["EC"]).copy()
    if df.empty:
        return
    labels = df["method_label"].tolist()
    values = df["EC"].to_numpy()
    x_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=120)
    bars = ax.bar(x_pos, values, color=bar_colors(df), width=0.62)
    ax.axhline(0.95, color="#222222", linestyle="--", linewidth=1.2)
    label_nominal_coverage_line(ax, above_plot=True)
    low = max(0.0, min(0.75, float(np.nanmin(values)) - 0.05))
    ax.set_ylabel("Empirical coverage")
    set_centered_xticklabels(ax, labels)
    ax.bar_label(bars, labels=[f"{v * 100:.1f}%" for v in values], padding=3, fontsize=9)
    top = min(1.02, max(0.965, float(np.nanmax(values)) + 0.02))
    ax.set_ylim(low, top)
    ax.set_title("Baseline design: empirical coverage of 95% CIs", pad=12)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(figdir / "coverage_bar.png", dpi=300)
    plt.close(fig)


def plot_interval_length_bar(summary: pd.DataFrame, figdir: Path) -> None:
    df = summary.dropna(subset=["IL"]).copy()
    if df.empty:
        return
    labels = df["method_label"].tolist()
    values = df["IL"].to_numpy()
    x_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=120)
    bars = ax.bar(x_pos, values, color=bar_colors(df), width=0.62)
    ax.set_ylabel("Average interval length")
    ax.set_title("Baseline design: average interval length")
    set_centered_xticklabels(ax, labels)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=3, fontsize=9)
    ax.set_ylim(0.0, float(np.nanmax(values)) * 1.12)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(figdir / "interval_length_bar.png", dpi=300)
    plt.close(fig)


def plot_bias_bar(summary: pd.DataFrame, figdir: Path) -> None:
    df = summary.dropna(subset=["bias"]).copy()
    if df.empty:
        return
    labels = df["method_label"].tolist()
    values = df["bias"].to_numpy()
    x_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=120)
    bars = ax.bar(x_pos, values, color=bar_colors(df), width=0.62)
    ax.set_ylabel("Average estimate - true tau")
    ax.set_title("Baseline design: finite-sample bias of CI center")
    set_centered_xticklabels(ax, labels)
    label_padding = 3 if np.nanmax(values) >= 0 else -12
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=label_padding, fontsize=9)
    ymin = float(np.nanmin(values))
    ymax = float(np.nanmax(values))
    if ymin >= 0:
        ax.set_ylim(0.0, ymax * 1.12 if ymax > 0 else 1.0)
    else:
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.1)
        plot_min = min(0.0, ymin)
        plot_max = max(0.0, ymax)
        pad = 0.12 * (plot_max - plot_min if plot_max > plot_min else 1.0)
        ax.set_ylim(plot_min - pad, plot_max + pad)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(figdir / "bias_bar.png", dpi=300)
    plt.close(fig)


def plot_coverage_length_tradeoff(summary: pd.DataFrame, figdir: Path) -> None:
    df = summary.dropna(subset=["EC", "IL"]).copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=120)
    ax.scatter(df["IL"], df["EC"], s=95, c=bar_colors(df), edgecolor="white", linewidth=0.8, zorder=3)
    ax.axhline(0.95, color="#222222", linestyle="--", linewidth=1.1)
    label_nominal_coverage_line(ax)
    min_il = float(df["IL"].min())
    max_il = float(df["IL"].max())
    x_pad = 0.08 * (max_il - min_il if max_il > min_il else 1.0)
    ax.set_xlim(min_il - x_pad, max_il + x_pad)
    min_ec = min(0.95, float(df["EC"].min()))
    max_ec = max(0.95, float(df["EC"].max()))
    y_pad = 0.10 * (max_ec - min_ec if max_ec > min_ec else 0.05)
    ax.set_ylim(min_ec - y_pad, max_ec + y_pad)
    label_offsets = {
        "conventional_mserd": (6, -16, "left"),
        "bias_corrected_mserd": (6, 8, "left"),
        "rbc_mserd": (6, 8, "left"),
        "rbc_cerrd": (6, 8, "left"),
        "undersmoothing_mserd": (-6, 8, "right"),
    }
    for _, row in df.iterrows():
        dx, dy, ha = label_offsets.get(str(row["method"]), (6, 6, "left"))
        ax.annotate(
            str(row["method_label"]),
            (row["IL"], row["EC"]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            fontsize=9,
        )
    ax.set_xlabel("Average interval length")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Baseline design: coverage-length tradeoff")
    style_axes(ax, grid_axis="both")
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.15, top=0.90)
    fig.savefig(figdir / "coverage_length_tradeoff.png", dpi=300)
    plt.close(fig)


def plot_bandwidth_distribution(raw: pd.DataFrame, figdir: Path) -> None:
    bandwidth_methods = ["conventional_mserd", "undersmoothing_mserd", "rbc_mserd", "rbc_cerrd"]
    ok = raw[(raw["success"] == True) & raw["method"].isin(bandwidth_methods)].copy()
    if ok.empty:
        return
    data = []
    labels = []
    plotted_methods = []
    for method in bandwidth_methods:
        vals = ok.loc[ok["method"] == method, "h_left"].dropna().astype(float).to_numpy()
        if len(vals):
            data.append(vals)
            labels.append(METHOD_DISPLAY[method])
            plotted_methods.append(method)
    if not data:
        return

    fig, ax = plt.subplots(figsize=(8, 5.1), dpi=120)
    box = ax.boxplot(data, tick_labels=wrap_method_labels(labels), showfliers=False, patch_artist=True)
    for patch, method in zip(box["boxes"], plotted_methods):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.72)
        patch.set_edgecolor("#333333")
    for median in box["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.3)
    ax.set_ylabel("Left bandwidth h")
    ax.set_title("Baseline design: bandwidth distribution")
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")
        label.set_va("top")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(figdir / "bandwidth_distribution.png", dpi=300)
    plt.close(fig)


def make_plots(raw: pd.DataFrame, summary: pd.DataFrame, outdir: Path) -> None:
    figdir = ensure_dirs(outdir)
    plot_baseline_design(figdir)
    plot_coverage_bar(summary, figdir)
    plot_interval_length_bar(summary, figdir)
    plot_bias_bar(summary, figdir)
    plot_coverage_length_tradeoff(summary, figdir)
    plot_bandwidth_distribution(raw, figdir)


def write_config(cfg: SimConfig, path: Path) -> None:
    text = [
        "CCT baseline design replication config",
        f"reps={cfg.reps}",
        f"n={cfg.n}",
        f"seed={cfg.seed}",
        f"true_tau={TRUE_TAU}",
        f"sigma_eps={SIGMA_EPS}",
        f"bwselect={cfg.bwselect}",
        "rbc_cerrd_bwselect=cerrd",
        f"vce={cfg.vce}",
        f"kernel={cfg.kernel}",
        f"p={cfg.p}",
        f"q={cfg.q}",
        f"level={cfg.level}",
        f"masspoints={cfg.masspoints}",
        f"us_scale={cfg.us_scale}",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> SimConfig:
    parser = argparse.ArgumentParser(description="CCT baseline design Monte Carlo replication.")
    parser.add_argument("--reps", type=int, default=5000, help="Number of Monte Carlo replications.")
    parser.add_argument("--n", type=int, default=500, help="Sample size per replication.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory.")
    parser.add_argument("--bwselect", type=str, default="mserd", help="rdrobust bandwidth selector for data-driven rows.")
    parser.add_argument("--vce", type=str, default="nn", help="rdrobust variance estimator: nn, hc0, hc1, hc2, hc3, ...")
    parser.add_argument("--kernel", type=str, default="tri", help="Kernel: tri, uniform, epa.")
    parser.add_argument("--p", type=int, default=1, help="Local polynomial order for point estimator.")
    parser.add_argument("--q", type=int, default=2, help="Local polynomial order for bias correction.")
    parser.add_argument("--level", type=float, default=95.0, help="Confidence level.")
    parser.add_argument("--masspoints", type=str, default="off", choices=["off", "check", "adjust"], help="Mass-points option.")
    parser.add_argument("--us-scale", type=float, default=0.5, help="Undersmoothing multiplier for h and b. Default 0.75 is mild undersmoothing.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every k reps; 0 disables.")

    args = parser.parse_args(argv)

    if args.reps <= 0:
        raise ValueError("--reps must be positive.")
    if args.n <= 20:
        raise ValueError("--n should be larger than 20.")
    if args.us_scale <= 0 or args.us_scale >= 1:
        raise ValueError("--us-scale should be in (0, 1) for undersmoothing.")

    return SimConfig(
        reps=args.reps,
        n=args.n,
        seed=args.seed,
        outdir=args.outdir,
        bwselect=args.bwselect,
        vce=args.vce,
        kernel=args.kernel,
        p=args.p,
        q=args.q,
        level=args.level,
        masspoints=args.masspoints,
        us_scale=args.us_scale,
        progress_every=args.progress_every,
    )


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    config_path = run_config_path(cfg.outdir)

    # Fail early instead of spending all replications recording the same import error.
    import_rdrobust()

    print("CCT baseline design replication")
    print(f"Output directory: {cfg.outdir}")
    print(f"reps={cfg.reps}, n={cfg.n}, seed={cfg.seed}")
    print(f"true_tau={TRUE_TAU:.6f}, sigma_eps={SIGMA_EPS:.4f}")
    print(f"bwselect={cfg.bwselect}, vce={cfg.vce}, kernel={cfg.kernel}, p={cfg.p}, q={cfg.q}")
    print("additional method: rbc_cerrd uses bwselect=cerrd")
    print(f"undersmoothing scale={cfg.us_scale:.3f}")

    write_config(cfg, config_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = run_simulation(cfg)

    summary = summarize(raw)

    raw_path = cfg.outdir / "baseline_raw.csv"
    summary_path = cfg.outdir / "baseline_summary.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    make_plots(raw, summary, cfg.outdir)

    failures = raw.loc[raw["success"] != True, ["method", "error"]]
    if not failures.empty:
        print("\nFailures detected:")
        print(failures.drop_duplicates().head(10).to_string(index=False))

    print("\nConcise summary:")
    printable_cols = [
        "method_label",
        "reps_success",
        "fail_rate",
        "EC_percent",
        "IL",
        "bias",
        "avg_h_left",
        "avg_h_right",
        "avg_b_left",
        "avg_b_right",
    ]
    existing_cols = [c for c in printable_cols if c in summary.columns]
    print(summary[existing_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nSaved raw results to:       {raw_path}")
    print(f"Saved summary results to:   {summary_path}")
    print(f"Saved figures to:           {cfg.outdir / 'figures'}")
    print(f"Saved run config to:        {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
