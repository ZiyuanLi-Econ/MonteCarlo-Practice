#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCT (2014) DGP2 discrete running-variable Monte Carlo extension.

This single script integrates the two discrete-running-variable experiments:

1. General experiment: continuous, fixed-support, and shrinking-support paths
   over several sample sizes.
2. Failure-frontier experiment: a refined fixed-delta grid at n=500 that
   identifies where local support becomes insufficient and RBC output loses its
   ordinary Monte Carlo interpretation.

The experiments share one DGP, one estimator implementation, one set of support
rules, and one paired common-random-number stream. Cells shared by the two
experiments are simulated only once. By default, the script generates two
paper-facing PNG figures, one bias-aware auxiliary check, and three compact
appendix diagnostics.

Examples
--------
Full unified run:
    python cct_2_3_discrete_paired.py

Quick check:
    python cct_2_3_discrete_paired.py --reps 20 --general-n-grid 500 1000

General experiment only:
    python cct_2_3_discrete_paired.py --experiment general

Failure-frontier only:
    python cct_2_3_discrete_paired.py --experiment frontier

Rebuild tables and figures from an existing raw file:
    python cct_2_3_discrete_paired.py --raw-input results/discrete_mechanism_raw.csv
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from rdrobust import rdrobust
except Exception as exc:  # pragma: no cover
    rdrobust = None
    _RDROBUST_IMPORT_ERROR = exc
else:
    _RDROBUST_IMPORT_ERROR = None


# =============================================================================
# 0. Global design parameters
# =============================================================================

TRUE_TAU = -3.45
ERROR_SD = 0.1295
CUTOFF = 0.0
ALPHA = 0.05
LEVEL = 95.0

# CCT/RD settings
P = 1
Q = 2
KERNEL = "triangular"
BWSELECT = "mserd"
VCE = "nn"
MASSPOINTS = "adjust"
BWCHECK = 10

# Unified design grids. The general grid stops at delta=0.05, while the
# n=500 frontier extends to delta=0.12. Shared cells are evaluated once.
DEFAULT_REPS = 5000
DEFAULT_EXPERIMENT = "all"
DEFAULT_GENERAL_N_GRID = (500, 750, 1000, 1500, 2000, 3000, 4000, 5000)
DEFAULT_GENERAL_FIXED_DELTAS = (
    0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05,
)
DEFAULT_FRONTIER_N = 500
DEFAULT_FRONTIER_FIXED_DELTAS = (
    0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05,
    0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.09, 0.10, 0.12,
)
DEFAULT_STRESS_DELTAS = (0.15, 0.20)

# Backward-compatible aliases used by a few helper functions.
DEFAULT_N_GRID = DEFAULT_GENERAL_N_GRID
DEFAULT_FIXED_DELTAS_MAIN = DEFAULT_GENERAL_FIXED_DELTAS
DEFAULT_FIXED_DELTAS_STRESS = DEFAULT_STRESS_DELTAS


def default_output_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    if script_dir.name.lower() == "code":
        return script_dir.parent / "results"
    return script_dir / f"{Path(__file__).resolve().stem}_results"


DEFAULT_OUTDIR = default_output_dir()

# Shrinking support path: delta_n = base_delta * (base_n / n)^power.
SHRINK_BASE_DELTA = 0.05
SHRINK_BASE_N = 500
SHRINK_POWER = 0.5

# Usability/pathology thresholds. These are intentionally conservative relative
# to normal CI lengths in this design, which are usually around 0.1--0.4.
LENGTH_PATHOLOGY_CUTOFF = 10.0
HONEST_LENGTH_PATHOLOGY_CUTOFF = 20.0
USABILITY_PLOT_THRESHOLD = 0.80
MIN_DISPLAY_USABLE = 30

# Auxiliary bias-aware benchmark. This uses the same MSE bandwidth selected by
# rdrobust and a curvature bound from the known DGP; it is not presented as the
# canonical optimal honest RD procedure.
HONEST_CURVATURE_MULTIPLIER = 1.10
HONEST_CURVATURE_GRID = 20001

# Failure-frontier display rule. This does not delete rows; it only marks
# coverage numbers that have enough local support to be interpreted normally.
FRONTIER_MIN_OBS_H = 20
FRONTIER_DISPLAY_MIN_UNIQUE_H = 5
FRONTIER_DISPLAY_MIN_UNIQUE_B = 6

BASE_METHODS = ("Conventional MSE", "RBC MSE", "Bias-aware/Honest")
ALL_METHODS = BASE_METHODS

# Minimal support rules.
CONVENTIONAL_MIN_UNIQUE_H = 2
RBC_MIN_UNIQUE_H = 2
RBC_MIN_UNIQUE_B = 3
HONEST_MIN_UNIQUE_H = 2

# Compact raw columns. Full raw can be saved with --save-full-raw.
COMPACT_RAW_COLUMNS = [
    "rep", "n", "regime", "scenario", "delta",
    "in_general", "in_frontier", "design_scope", "method",
    "raw_success", "success", "minimal_usable", "support_adequate",
    "rbc_support_adequate", "formal_support_ok", "frontier_display_ok",
    "pathological_length", "nonusable_reason", "frontier_failure_reason", "error_msg",
    "h_left", "h_right", "b_left", "b_right",
    "unique_x_left", "unique_x_right", "unique_x_total", "n_left", "n_right",
    "unique_h_left", "unique_h_right", "unique_b_left", "unique_b_right",
    "obs_h_left", "obs_h_right", "obs_b_left", "obs_b_right",
    "min_unique_h", "min_unique_b", "local_support_score",
    "mass_share_left", "mass_share_right", "mass_share_h_left", "mass_share_h_right",
    "mass_share_b_left", "mass_share_b_right",
    "h_over_delta_left", "h_over_delta_right",
    "b_over_delta_left", "b_over_delta_right", "nearest_support_distance",
    "N_left_rdrobust", "N_right_rdrobust", "N_h_left_rdrobust", "N_h_right_rdrobust",
    "N_b_left_rdrobust", "N_b_right_rdrobust", "M_left_rdrobust", "M_right_rdrobust",
    "estimate", "se", "ci_l", "ci_u", "bias_bound", "bias_bound_left", "bias_bound_right",
    "covered", "ci_length", "est_error", "abs_est_error", "miss_below", "miss_above",
]


# =============================================================================
# 1. DGP2 functions
# =============================================================================

def mu2(x: np.ndarray | float) -> np.ndarray | float:
    """CCT DGP2 conditional mean."""
    x_arr = np.asarray(x)
    left = 3.71 + 2.30 * x_arr + 3.28 * x_arr**2 + 1.45 * x_arr**3 + 0.23 * x_arr**4 + 0.03 * x_arr**5
    right = 0.26 + 18.49 * x_arr - 54.81 * x_arr**2 + 74.30 * x_arr**3 - 45.02 * x_arr**4 + 9.83 * x_arr**5
    out = np.where(x_arr < CUTOFF, left, right)
    if np.isscalar(x):
        return float(out)
    return out


def mu2_second_derivative(x: np.ndarray | float) -> np.ndarray | float:
    """Second derivative of the CCT DGP2 conditional mean."""
    x_arr = np.asarray(x)
    left = 2 * 3.28 + 6 * 1.45 * x_arr + 12 * 0.23 * x_arr**2 + 20 * 0.03 * x_arr**3
    right = 2 * (-54.81) + 6 * 74.30 * x_arr + 12 * (-45.02) * x_arr**2 + 20 * 9.83 * x_arr**3
    out = np.where(x_arr < CUTOFF, left, right)
    if np.isscalar(x):
        return float(out)
    return out


def compute_curvature_bounds(multiplier: float = HONEST_CURVATURE_MULTIPLIER,
                             grid_size: int = HONEST_CURVATURE_GRID) -> Tuple[float, float]:
    """Compute side-specific sup |mu''(x)| bounds on [-1,0) and [0,1]."""
    x_left = np.linspace(-1.0, -1e-12, grid_size)
    x_right = np.linspace(0.0, 1.0, grid_size)
    m_left = float(np.max(np.abs(mu2_second_derivative(x_left))) * multiplier)
    m_right = float(np.max(np.abs(mu2_second_derivative(x_right))) * multiplier)
    return m_left, m_right


HONEST_M_LEFT, HONEST_M_RIGHT = compute_curvature_bounds()


def discretize_midpoint_grid(x: np.ndarray, delta: Optional[float]) -> np.ndarray:
    """
    Discretize X into bins with mass points at midpoints.

    For delta=0.10, the closest support points to c=0 are -0.05 and +0.05.
    """
    if delta is None or (isinstance(delta, float) and np.isnan(delta)):
        return x.copy()
    if delta <= 0:
        raise ValueError("delta must be positive or None")
    x_disc = np.floor(x / delta) * delta + 0.5 * delta
    lower = -1.0 + 0.5 * delta
    upper = 1.0 - 0.5 * delta
    return np.clip(x_disc, lower, upper)


def shrinking_delta(n: int) -> float:
    return float(SHRINK_BASE_DELTA * (SHRINK_BASE_N / n) ** SHRINK_POWER)


def delta_label(delta: float, digits: int = 3) -> str:
    """Stable label that does not collapse 0.015 into 0.01."""
    return f"{float(delta):.{digits}f}".rstrip("0").rstrip(".")


def percent_label(rate: float, digits: int = 1) -> str:
    """Display percentages without rounding values below one up to 100%."""
    if not math.isfinite(float(rate)):
        return ""
    value = float(rate) * 100.0
    if value < 100.0 and round(value, digits) >= 100.0:
        return f"{value:.2f}%"
    return f"{value:.{digits}f}%"


# =============================================================================
# 2. Small utilities
# =============================================================================

@contextlib.contextmanager
def suppress_stdout_stderr(enabled: bool = True):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = devnull, devnull
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def finite_or_nan(x: Any) -> float:
    try:
        val = float(x)
    except Exception:
        return float("nan")
    return val if math.isfinite(val) else float("nan")


def first_two_numeric(obj: Any) -> Tuple[float, float]:
    vals = pd.to_numeric(pd.Series(np.asarray(obj).ravel()), errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) < 2:
        return float("nan"), float("nan")
    return float(vals[0]), float(vals[1])


def as_dataframe(obj: Any) -> Optional[pd.DataFrame]:
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, pd.Series):
        return obj.to_frame().T
    try:
        return pd.DataFrame(obj)
    except Exception:
        return None


def get_attr_any(obj: Any, names: Sequence[str]) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def row_by_keywords(df: pd.DataFrame, keywords: Sequence[str], default_pos: int) -> pd.Series:
    """Find a DataFrame row by index/first-column keywords, with positional fallback."""
    if df is None or df.empty:
        raise ValueError("empty DataFrame")

    kw = tuple(k.lower() for k in keywords)

    for idx in df.index:
        idx_str = str(idx).lower()
        if any(k in idx_str for k in kw):
            return df.loc[idx]

    first_col = df.columns[0]
    for i in range(len(df)):
        cell_str = str(df.iloc[i][first_col]).lower()
        if any(k in cell_str for k in kw):
            return df.iloc[i]

    pos = min(max(default_pos, 0), len(df) - 1)
    return df.iloc[pos]


def numeric_values_from_row(row: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(row, errors="coerce").dropna().to_numpy(dtype=float)
    return vals


def extract_table_row_values(obj: Any,
                             keywords: Sequence[str],
                             default_pos: int,
                             min_values: int = 1) -> np.ndarray:
    df = as_dataframe(obj)
    if df is None or df.empty:
        raise ValueError("cannot convert object to DataFrame")
    row = row_by_keywords(df, keywords, default_pos)
    vals = numeric_values_from_row(row)
    if len(vals) < min_values:
        raise ValueError(f"not enough numeric values in row: need {min_values}, got {len(vals)}")
    return vals


def extract_ci_from_rdrobust(res: Any, method: str) -> Tuple[float, float]:
    ci_obj = get_attr_any(res, ("ci", "CI"))
    if ci_obj is None:
        raise ValueError("rdrobust result has no ci/CI attribute")
    if method == "Conventional MSE":
        vals = extract_table_row_values(ci_obj, ("conventional",), default_pos=0, min_values=2)
    elif method == "RBC MSE":
        vals = extract_table_row_values(ci_obj, ("robust",), default_pos=2, min_values=2)
    else:
        raise ValueError(f"unknown rdrobust method {method}")
    return float(vals[-2]), float(vals[-1])


def extract_se_from_rdrobust(res: Any, method: str) -> float:
    se_obj = get_attr_any(res, ("se", "SE"))
    if se_obj is None:
        raise ValueError("rdrobust result has no se/SE attribute")
    if method == "Conventional MSE":
        vals = extract_table_row_values(se_obj, ("conventional",), default_pos=0, min_values=1)
    elif method == "RBC MSE":
        vals = extract_table_row_values(se_obj, ("robust",), default_pos=2, min_values=1)
    else:
        raise ValueError(f"unknown rdrobust method {method}")
    return float(vals[-1])


def extract_coef_from_rdrobust(res: Any, method: str, ci_l: float, ci_u: float) -> float:
    """Extract coefficient. Fallback to CI midpoint if rdrobust object layout differs."""
    coef_obj = get_attr_any(res, ("coef", "Coef", "Estimate", "estimate"))
    if coef_obj is not None:
        try:
            if method == "Conventional MSE":
                vals = extract_table_row_values(coef_obj, ("conventional",), default_pos=0, min_values=1)
            elif method == "RBC MSE":
                try:
                    vals = extract_table_row_values(coef_obj, ("robust",), default_pos=2, min_values=1)
                except Exception:
                    vals = extract_table_row_values(coef_obj, ("bias", "corrected"), default_pos=1, min_values=1)
            else:
                raise ValueError(f"unknown rdrobust method {method}")
            return float(vals[-1])
        except Exception:
            pass

    if method == "Conventional MSE":
        tau = get_attr_any(res, ("tau_cl", "tau_us", "tau_cl_l"))
    else:
        tau = get_attr_any(res, ("tau_bc", "tau_bc_l"))
    if tau is not None:
        vals = pd.to_numeric(pd.Series(np.asarray(tau).ravel()), errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) > 0:
            return float(vals[0])

    return float((ci_l + ci_u) / 2.0)


def extract_bandwidths_from_rdrobust(res: Any) -> Dict[str, float]:
    bws_obj = get_attr_any(res, ("bws", "Bws", "BW", "bw"))
    df = as_dataframe(bws_obj)
    h_left = h_right = b_left = b_right = float("nan")

    if df is not None and not df.empty:
        row_h = None
        row_b = None
        for idx in df.index:
            idx_str = str(idx).lower().strip()
            if idx_str == "h" or idx_str.startswith("h"):
                row_h = df.loc[idx]
            if idx_str == "b" or idx_str.startswith("b"):
                row_b = df.loc[idx]
        if row_h is None and len(df) >= 1:
            row_h = df.iloc[0]
        if row_b is None and len(df) >= 2:
            row_b = df.iloc[1]
        if row_h is not None:
            vals = numeric_values_from_row(row_h)
            if len(vals) >= 2:
                h_left, h_right = float(vals[0]), float(vals[1])
        if row_b is not None:
            vals = numeric_values_from_row(row_b)
            if len(vals) >= 2:
                b_left, b_right = float(vals[0]), float(vals[1])
    else:
        vals = pd.to_numeric(pd.Series(np.asarray(bws_obj).ravel()), errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) >= 4:
            h_left, h_right, b_left, b_right = map(float, vals[:4])

    return {"h_left": h_left, "h_right": h_right, "b_left": b_left, "b_right": b_right}


def extract_pair_attr(res: Any, names: Sequence[str]) -> Tuple[float, float]:
    obj = get_attr_any(res, names)
    if obj is None:
        return float("nan"), float("nan")
    return first_two_numeric(obj)


def slug(s: str) -> str:
    return (
        s.lower()
        .replace("/", "_")
        .replace(">=", "ge")
        .replace(" ", "_")
        .replace(",", "")
        .replace(".", "")
        .replace("-", "_")
    )


# =============================================================================
# 3. Support diagnostics
# =============================================================================

def unique_count(x: np.ndarray, mask: np.ndarray) -> int:
    vals = x[mask]
    if vals.size == 0:
        return 0
    return int(np.unique(np.round(vals, 12)).size)


def mass_share(n_unique: Any, n_obs: Any) -> float:
    """Mass-point concentration proxy: 1 - unique support points / observations."""
    try:
        u = float(n_unique)
        n = float(n_obs)
    except Exception:
        return float("nan")
    if not math.isfinite(u) or not math.isfinite(n) or n <= 0:
        return float("nan")
    return float(1.0 - u / n)


def support_diagnostics(x: np.ndarray,
                        delta: Optional[float],
                        h_left: float,
                        h_right: float,
                        b_left: float,
                        b_right: float) -> Dict[str, float]:
    left = x < CUTOFF
    right = x >= CUTOFF
    left_h = (x < CUTOFF) & (x >= CUTOFF - h_left) if math.isfinite(h_left) and h_left > 0 else np.zeros_like(x, dtype=bool)
    right_h = (x >= CUTOFF) & (x <= CUTOFF + h_right) if math.isfinite(h_right) and h_right > 0 else np.zeros_like(x, dtype=bool)
    left_b = (x < CUTOFF) & (x >= CUTOFF - b_left) if math.isfinite(b_left) and b_left > 0 else np.zeros_like(x, dtype=bool)
    right_b = (x >= CUTOFF) & (x <= CUTOFF + b_right) if math.isfinite(b_right) and b_right > 0 else np.zeros_like(x, dtype=bool)

    ux_l = unique_count(x, left)
    ux_r = unique_count(x, right)
    ux_total = unique_count(x, np.ones_like(x, dtype=bool))
    n_l = int(np.sum(left))
    n_r = int(np.sum(right))

    uh_l = unique_count(x, left_h)
    uh_r = unique_count(x, right_h)
    ub_l = unique_count(x, left_b)
    ub_r = unique_count(x, right_b)
    oh_l = int(np.sum(left_h))
    oh_r = int(np.sum(right_h))
    ob_l = int(np.sum(left_b))
    ob_r = int(np.sum(right_b))

    nearest = float(np.min(np.abs(np.unique(np.round(x - CUTOFF, 12))))) if x.size else float("nan")
    dval = float(delta) if delta is not None and not (isinstance(delta, float) and np.isnan(delta)) else float("nan")
    min_h = min(uh_l, uh_r)
    min_b = min(ub_l, ub_r)

    return {
        "unique_x_left": ux_l,
        "unique_x_right": ux_r,
        "unique_x_total": ux_total,
        "n_left": n_l,
        "n_right": n_r,
        "unique_h_left": uh_l,
        "unique_h_right": uh_r,
        "unique_b_left": ub_l,
        "unique_b_right": ub_r,
        "obs_h_left": oh_l,
        "obs_h_right": oh_r,
        "obs_b_left": ob_l,
        "obs_b_right": ob_r,
        "min_unique_h": min_h,
        "min_unique_b": min_b,
        "local_support_score": min(min_h, min_b),
        "mass_share_left": mass_share(ux_l, n_l),
        "mass_share_right": mass_share(ux_r, n_r),
        "mass_share_h_left": mass_share(uh_l, oh_l),
        "mass_share_h_right": mass_share(uh_r, oh_r),
        "mass_share_b_left": mass_share(ub_l, ob_l),
        "mass_share_b_right": mass_share(ub_r, ob_r),
        "h_over_delta_left": h_left / dval if math.isfinite(dval) and dval > 0 and math.isfinite(h_left) else float("nan"),
        "h_over_delta_right": h_right / dval if math.isfinite(dval) and dval > 0 and math.isfinite(h_right) else float("nan"),
        "b_over_delta_left": b_left / dval if math.isfinite(dval) and dval > 0 and math.isfinite(b_left) else float("nan"),
        "b_over_delta_right": b_right / dval if math.isfinite(dval) and dval > 0 and math.isfinite(b_right) else float("nan"),
        "nearest_support_distance": nearest if math.isfinite(dval) else 0.0,
    }


def bool_support_ge(value: Any, threshold: int) -> bool:
    try:
        v = float(value)
    except Exception:
        return False
    return math.isfinite(v) and v >= threshold


# =============================================================================
# 4. Auxiliary bias-aware / honest benchmark
# =============================================================================

@dataclass
class SideFit:
    intercept: float
    weights: np.ndarray
    residuals: np.ndarray
    dx: np.ndarray
    n_obs: int
    n_unique: int


def triangular_kernel(u: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, 1.0 - np.abs(u))


def local_linear_side_fit(x: np.ndarray,
                          y: np.ndarray,
                          h: float,
                          side: str) -> SideFit:
    if not math.isfinite(h) or h <= 0:
        raise ValueError(f"invalid bandwidth h={h}")

    if side == "left":
        mask = (x < CUTOFF) & (x >= CUTOFF - h)
    elif side == "right":
        mask = (x >= CUTOFF) & (x <= CUTOFF + h)
    else:
        raise ValueError("side must be 'left' or 'right'")

    idx = np.flatnonzero(mask)
    if idx.size < 3:
        raise ValueError("too few observations for local linear bias-aware CI weights")

    xs = x[idx] - CUTOFF
    ys = y[idx]
    n_unique = int(np.unique(np.round(xs, 12)).size)
    if n_unique < 2:
        raise ValueError("too few distinct points for local linear bias-aware CI weights")

    u = xs / h
    k = triangular_kernel(u)
    keep = k > 0
    xs = xs[keep]
    ys = ys[keep]
    u = u[keep]
    k = k[keep]
    idx = idx[keep]

    if idx.size < 3:
        raise ValueError("too few positive-kernel observations for bias-aware CI")

    xmat = np.column_stack([np.ones_like(u), u])
    xtw = xmat.T * k
    xtwx = xtw @ xmat
    if not np.all(np.isfinite(xtwx)):
        raise ValueError("non-finite local linear moment matrix")

    cond = np.linalg.cond(xtwx)
    if not math.isfinite(cond) or cond > 1e12:
        raise ValueError(f"ill-conditioned local linear moment matrix, cond={cond}")

    inv = np.linalg.inv(xtwx)
    beta = inv @ (xtw @ ys)
    fitted = xmat @ beta
    residuals = ys - fitted
    weights = np.array([1.0, 0.0]) @ inv @ xtw

    return SideFit(
        intercept=float(beta[0]),
        weights=weights.astype(float),
        residuals=residuals.astype(float),
        dx=xs.astype(float),
        n_obs=int(idx.size),
        n_unique=n_unique,
    )


def honest_ci(x: np.ndarray,
              y: np.ndarray,
              h_left: float,
              h_right: float,
              alpha: float = ALPHA,
              m_left: float = HONEST_M_LEFT,
              m_right: float = HONEST_M_RIGHT) -> Dict[str, float]:
    """Curvature-bound bias-aware benchmark using the same MSE bandwidth h."""
    left = local_linear_side_fit(x, y, h_left, side="left")
    right = local_linear_side_fit(x, y, h_right, side="right")

    tau_hat = right.intercept - left.intercept
    var_left = float(np.sum((left.weights ** 2) * (left.residuals ** 2)))
    var_right = float(np.sum((right.weights ** 2) * (right.residuals ** 2)))
    se = math.sqrt(max(var_left + var_right, 0.0))

    bias_bound_left = 0.5 * m_left * float(np.sum(np.abs(left.weights) * (left.dx ** 2)))
    bias_bound_right = 0.5 * m_right * float(np.sum(np.abs(right.weights) * (right.dx ** 2)))
    bias_bound = bias_bound_left + bias_bound_right

    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    half_length = z * se + bias_bound
    ci_l = tau_hat - half_length
    ci_u = tau_hat + half_length

    if not all(math.isfinite(v) for v in (tau_hat, se, ci_l, ci_u, bias_bound)):
        raise ValueError("non-finite bias-aware CI output")

    return {
        "estimate": tau_hat,
        "se": se,
        "ci_l": ci_l,
        "ci_u": ci_u,
        "bias_bound": bias_bound,
        "bias_bound_left": bias_bound_left,
        "bias_bound_right": bias_bound_right,
    }


# =============================================================================
# 5. rdrobust wrapper and method-level records
# =============================================================================

def call_rdrobust(y: np.ndarray, x: np.ndarray, silence: bool = True) -> Any:
    if rdrobust is None:
        raise RuntimeError(f"rdrobust import failed: {_RDROBUST_IMPORT_ERROR}")
    with suppress_stdout_stderr(silence):
        return rdrobust(
            y=y,
            x=x,
            c=CUTOFF,
            p=P,
            q=Q,
            kernel=KERNEL,
            bwselect=BWSELECT,
            vce=VCE,
            masspoints=MASSPOINTS,
            bwcheck=BWCHECK,
            level=LEVEL,
        )


def empty_method_record(rep: int,
                        n: int,
                        regime: str,
                        scenario: str,
                        delta: Optional[float],
                        method: str,
                        error_msg: str) -> Dict[str, Any]:
    return {
        "rep": rep,
        "n": n,
        "regime": regime,
        "scenario": scenario,
        "delta": np.nan if delta is None else float(delta),
        "method": method,
        "raw_success": False,
        "success": False,  # backward-compatible alias for raw_success
        "minimal_usable": False,
        "support_adequate": False,
        "rbc_support_adequate": False,
        "formal_support_ok": False,
        "frontier_display_ok": False,
        "pathological_length": False,
        "nonusable_reason": "raw_failure",
        "frontier_failure_reason": "raw_failure",
        "error_msg": error_msg,
        "h_left": np.nan,
        "h_right": np.nan,
        "b_left": np.nan,
        "b_right": np.nan,
        "unique_x_left": np.nan,
        "unique_x_right": np.nan,
        "unique_x_total": np.nan,
        "n_left": np.nan,
        "n_right": np.nan,
        "unique_h_left": np.nan,
        "unique_h_right": np.nan,
        "unique_b_left": np.nan,
        "unique_b_right": np.nan,
        "obs_h_left": np.nan,
        "obs_h_right": np.nan,
        "obs_b_left": np.nan,
        "obs_b_right": np.nan,
        "min_unique_h": np.nan,
        "min_unique_b": np.nan,
        "local_support_score": np.nan,
        "mass_share_left": np.nan,
        "mass_share_right": np.nan,
        "mass_share_h_left": np.nan,
        "mass_share_h_right": np.nan,
        "mass_share_b_left": np.nan,
        "mass_share_b_right": np.nan,
        "h_over_delta_left": np.nan,
        "h_over_delta_right": np.nan,
        "b_over_delta_left": np.nan,
        "b_over_delta_right": np.nan,
        "nearest_support_distance": np.nan,
        "N_left_rdrobust": np.nan,
        "N_right_rdrobust": np.nan,
        "N_h_left_rdrobust": np.nan,
        "N_h_right_rdrobust": np.nan,
        "N_b_left_rdrobust": np.nan,
        "N_b_right_rdrobust": np.nan,
        "M_left_rdrobust": np.nan,
        "M_right_rdrobust": np.nan,
        "estimate": np.nan,
        "se": np.nan,
        "ci_l": np.nan,
        "ci_u": np.nan,
        "bias_bound": np.nan,
        "bias_bound_left": np.nan,
        "bias_bound_right": np.nan,
        "covered": np.nan,
        "ci_length": np.nan,
        "est_error": np.nan,
        "abs_est_error": np.nan,
        "miss_below": np.nan,
        "miss_above": np.nan,
    }


def raw_output_is_finite(rec: Dict[str, Any]) -> bool:
    vals = [rec.get("estimate"), rec.get("se"), rec.get("ci_l"), rec.get("ci_u")]
    try:
        vals = [float(v) for v in vals]
    except Exception:
        return False
    if not all(math.isfinite(v) for v in vals):
        return False
    if vals[1] < 0:
        return False
    if vals[3] <= vals[2]:
        return False
    return True


def frontier_failure_reason(rec: Dict[str, Any]) -> str:
    """Standardized diagnostic reason for failure-frontier summaries."""
    error_msg = str(rec.get("error_msg", "") or "").lower()
    if "rdrobust call failed" in error_msg:
        return "rdrobust_exception"

    ci_l = finite_or_nan(rec.get("ci_l"))
    ci_u = finite_or_nan(rec.get("ci_u"))
    se = finite_or_nan(rec.get("se"))
    estimate = finite_or_nan(rec.get("estimate"))
    if not all(math.isfinite(v) for v in (ci_l, ci_u, se, estimate)):
        return "nan_ci"
    if ci_u <= ci_l:
        return "invalid_ci_order"

    h_left = finite_or_nan(rec.get("h_left"))
    h_right = finite_or_nan(rec.get("h_right"))
    b_left = finite_or_nan(rec.get("b_left"))
    b_right = finite_or_nan(rec.get("b_right"))
    if not all(math.isfinite(v) for v in (h_left, h_right, b_left, b_right)):
        return "missing_bandwidth"
    if h_left <= 0 or h_right <= 0 or b_left <= 0 or b_right <= 0:
        return "nonpositive_bandwidth"

    min_h = finite_or_nan(rec.get("min_unique_h"))
    min_b = finite_or_nan(rec.get("min_unique_b"))
    if not math.isfinite(min_h) or min_h < P + 1:
        return "insufficient_unique_h"
    if not math.isfinite(min_b) or min_b < Q + 1:
        return "insufficient_unique_b"

    obs_h_left = finite_or_nan(rec.get("obs_h_left"))
    obs_h_right = finite_or_nan(rec.get("obs_h_right"))
    obs_b_left = finite_or_nan(rec.get("obs_b_left"))
    obs_b_right = finite_or_nan(rec.get("obs_b_right"))
    if math.isfinite(obs_h_left) and math.isfinite(obs_h_right) and min(obs_h_left, obs_h_right) < 10:
        return "too_few_obs_h"
    if math.isfinite(obs_b_left) and math.isfinite(obs_b_right) and min(obs_b_left, obs_b_right) < 10:
        return "too_few_obs_b"
    if bool(rec.get("pathological_length", False)):
        return "length_pathology"
    return "ok"


def classify_record(rec: Dict[str, Any],
                    length_cutoff: float = LENGTH_PATHOLOGY_CUTOFF,
                    honest_length_cutoff: float = HONEST_LENGTH_PATHOLOGY_CUTOFF) -> Dict[str, Any]:
    """Add raw_success, minimal_usable, support diagnostics, and coverage fields."""
    method = str(rec.get("method", ""))
    raw_success = raw_output_is_finite(rec)
    rec["raw_success"] = bool(raw_success)
    rec["success"] = bool(raw_success)

    if raw_success:
        ci_l = float(rec["ci_l"])
        ci_u = float(rec["ci_u"])
        est = float(rec["estimate"])
        rec["covered"] = bool(ci_l <= TRUE_TAU <= ci_u)
        rec["ci_length"] = float(ci_u - ci_l)
        rec["est_error"] = float(est - TRUE_TAU)
        rec["abs_est_error"] = float(abs(est - TRUE_TAU))
        rec["miss_below"] = bool(TRUE_TAU < ci_l)
        rec["miss_above"] = bool(TRUE_TAU > ci_u)
    else:
        rec["minimal_usable"] = False
        rec["support_adequate"] = False
        rec["rbc_support_adequate"] = False
        rec["formal_support_ok"] = False
        rec["frontier_display_ok"] = False
        rec["pathological_length"] = False
        if not rec.get("nonusable_reason") or str(rec.get("nonusable_reason")) == "":
            rec["nonusable_reason"] = "raw_failure"
        rec["frontier_failure_reason"] = frontier_failure_reason(rec)
        return rec

    min_h = rec.get("min_unique_h")
    min_b = rec.get("min_unique_b")

    conventional_support_ok = bool_support_ge(min_h, CONVENTIONAL_MIN_UNIQUE_H)
    rbc_minimal_support_ok = bool_support_ge(min_h, RBC_MIN_UNIQUE_H) and bool_support_ge(min_b, RBC_MIN_UNIQUE_B)
    rbc_diagnostic_ok = bool_support_ge(min_h, 3) and bool_support_ge(min_b, 5)
    honest_support_ok = bool_support_ge(min_h, HONEST_MIN_UNIQUE_H)
    formal_support_ok = bool_support_ge(min_h, P + 1) and bool_support_ge(min_b, Q + 1)

    rec["rbc_support_adequate"] = bool(rbc_diagnostic_ok)
    rec["formal_support_ok"] = bool(formal_support_ok)

    try:
        length = float(rec.get("ci_length"))
    except Exception:
        length = float("nan")
    cutoff = honest_length_cutoff if method == "Bias-aware/Honest" else length_cutoff
    pathological_length = (not math.isfinite(length)) or length <= 0 or length > cutoff
    rec["pathological_length"] = bool(pathological_length)

    if method == "Conventional MSE":
        support_ok = conventional_support_ok
    elif method == "RBC MSE":
        support_ok = rbc_minimal_support_ok
    elif method == "Bias-aware/Honest":
        support_ok = honest_support_ok
    else:
        support_ok = False

    rec["support_adequate"] = bool(support_ok)
    rec["minimal_usable"] = bool(raw_success and support_ok and not pathological_length)
    rec["frontier_display_ok"] = bool(
        raw_success
        and formal_support_ok
        and bool_support_ge(rec.get("obs_h_left"), FRONTIER_MIN_OBS_H)
        and bool_support_ge(rec.get("obs_h_right"), FRONTIER_MIN_OBS_H)
        and bool_support_ge(min_h, FRONTIER_DISPLAY_MIN_UNIQUE_H)
        and bool_support_ge(min_b, FRONTIER_DISPLAY_MIN_UNIQUE_B)
    )

    reasons = []
    if not support_ok:
        reasons.append("support_infeasible")
    if pathological_length:
        reasons.append("length_pathology")
    rec["nonusable_reason"] = ";".join(reasons) if reasons else ""
    rec["frontier_failure_reason"] = frontier_failure_reason(rec)
    return rec


def run_methods_for_sample(rep: int,
                           n: int,
                           regime: str,
                           scenario: str,
                           delta: Optional[float],
                           x: np.ndarray,
                           y: np.ndarray,
                           silence_rdrobust: bool = True,
                           length_cutoff: float = LENGTH_PATHOLOGY_CUTOFF,
                           honest_length_cutoff: float = HONEST_LENGTH_PATHOLOGY_CUTOFF) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    try:
        res = call_rdrobust(y, x, silence=silence_rdrobust)
        bws = extract_bandwidths_from_rdrobust(res)
        h_left = finite_or_nan(bws["h_left"])
        h_right = finite_or_nan(bws["h_right"])
        b_left = finite_or_nan(bws["b_left"])
        b_right = finite_or_nan(bws["b_right"])

        if not all(math.isfinite(v) and v > 0 for v in (h_left, h_right)):
            raise ValueError(f"invalid rdrobust h bandwidths: {bws}")

        # b can be non-finite in a pathological call. This should not by itself
        # contaminate Conventional MSE, but it will make RBC support inadequate.
        diag = support_diagnostics(x, delta, h_left, h_right, b_left, b_right)
        N_left, N_right = extract_pair_attr(res, ("N", "n"))
        N_h_left, N_h_right = extract_pair_attr(res, ("N_h", "Nh"))
        N_b_left, N_b_right = extract_pair_attr(res, ("N_b", "Nb"))
        M_left, M_right = extract_pair_attr(res, ("M", "masspoints"))

        common = {
            "h_left": h_left,
            "h_right": h_right,
            "b_left": b_left,
            "b_right": b_right,
            **diag,
            "N_left_rdrobust": N_left,
            "N_right_rdrobust": N_right,
            "N_h_left_rdrobust": N_h_left,
            "N_h_right_rdrobust": N_h_right,
            "N_b_left_rdrobust": N_b_left,
            "N_b_right_rdrobust": N_b_right,
            "M_left_rdrobust": M_left,
            "M_right_rdrobust": M_right,
        }

        for method in ("Conventional MSE", "RBC MSE"):
            rec = empty_method_record(rep, n, regime, scenario, delta, method, "")
            rec.update(common)
            try:
                ci_l, ci_u = extract_ci_from_rdrobust(res, method)
                se = extract_se_from_rdrobust(res, method)
                estimate = extract_coef_from_rdrobust(res, method, ci_l, ci_u)
                rec.update({
                    "estimate": estimate,
                    "se": se,
                    "ci_l": ci_l,
                    "ci_u": ci_u,
                    "error_msg": "",
                })
            except Exception as exc:
                rec["error_msg"] = f"rdrobust extraction failed: {exc}"
                rec["nonusable_reason"] = "raw_failure"
            rec = classify_record(rec, length_cutoff, honest_length_cutoff)
            records.append(rec)

        rec = empty_method_record(rep, n, regime, scenario, delta, "Bias-aware/Honest", "")
        rec.update(common)
        try:
            hres = honest_ci(x, y, h_left=h_left, h_right=h_right)
            rec.update({
                "estimate": hres["estimate"],
                "se": hres["se"],
                "ci_l": hres["ci_l"],
                "ci_u": hres["ci_u"],
                "bias_bound": hres["bias_bound"],
                "bias_bound_left": hres["bias_bound_left"],
                "bias_bound_right": hres["bias_bound_right"],
                "error_msg": "",
            })
        except Exception as exc:
            rec["error_msg"] = f"bias-aware CI failed: {exc}"
            rec["nonusable_reason"] = "raw_failure"
        rec = classify_record(rec, length_cutoff, honest_length_cutoff)
        records.append(rec)

    except Exception as exc:
        msg = f"rdrobust call failed: {exc}"
        for method in ALL_METHODS:
            records.append(empty_method_record(rep, n, regime, scenario, delta, method, msg))

    return records


# =============================================================================
# 6. Simulation design
# =============================================================================

def normalise_deltas(values: Sequence[float]) -> Tuple[float, ...]:
    """Return finite, positive, sorted, de-duplicated grid spacings."""
    cleaned = []
    for value in values:
        val = float(value)
        if not math.isfinite(val) or val <= 0:
            raise ValueError(f"all delta values must be finite and positive; got {value!r}")
        cleaned.append(round(val, 12))
    return tuple(sorted(set(cleaned)))


def scope_label(in_general: bool, in_frontier: bool) -> str:
    if in_general and in_frontier:
        return "both"
    if in_general:
        return "general"
    if in_frontier:
        return "frontier"
    return "unassigned"


def build_scenario_plan(experiment: str,
                        general_n_grid: Sequence[int],
                        general_fixed_deltas: Sequence[float],
                        frontier_n: int,
                        frontier_fixed_deltas: Sequence[float],
                        stress_deltas: Sequence[float] = (),
                        include_shrinking: bool = True) -> Dict[int, List[Dict[str, Any]]]:
    """Build a de-duplicated per-n plan for the two linked experiments.

    A continuous or fixed-delta cell that belongs to both experiments is stored
    once with both membership flags. The shrinking path remains a separate
    scenario even when its numerical delta coincides with a fixed-grid cell.
    """
    experiment = str(experiment).lower().strip()
    if experiment not in {"all", "general", "frontier"}:
        raise ValueError("experiment must be one of: all, general, frontier")

    general_n_grid = tuple(sorted(set(int(n) for n in general_n_grid)))
    if any(n <= 0 for n in general_n_grid):
        raise ValueError("sample sizes must be positive")
    frontier_n = int(frontier_n)
    if frontier_n <= 0:
        raise ValueError("frontier_n must be positive")

    general_fixed_deltas = normalise_deltas(general_fixed_deltas)
    frontier_fixed_deltas = normalise_deltas(tuple(frontier_fixed_deltas) + tuple(stress_deltas))

    plan_map: Dict[int, Dict[Tuple[str, Optional[float]], Dict[str, Any]]] = {}

    def upsert(n: int, regime: str, scenario: str, delta: Optional[float],
               in_general: bool, in_frontier: bool, is_stress: bool = False) -> None:
        bucket = plan_map.setdefault(int(n), {})
        dkey = None if delta is None else round(float(delta), 12)
        key = (str(regime), dkey)
        if key not in bucket:
            bucket[key] = {
                "regime": str(regime),
                "scenario": str(scenario),
                "delta": None if delta is None else float(delta),
                "in_general": bool(in_general),
                "in_frontier": bool(in_frontier),
                "is_stress": bool(is_stress),
            }
        else:
            bucket[key]["in_general"] = bool(bucket[key]["in_general"] or in_general)
            bucket[key]["in_frontier"] = bool(bucket[key]["in_frontier"] or in_frontier)
            bucket[key]["is_stress"] = bool(bucket[key]["is_stress"] or is_stress)

    if experiment in {"all", "general"}:
        for n in general_n_grid:
            upsert(n, "continuous", "C0_continuous", None, True, False)
            for d in general_fixed_deltas:
                lab = delta_label(d)
                upsert(n, f"fixed_delta_{lab}", f"F_delta_{lab}", d, True, False)
            if include_shrinking:
                d_shrink = shrinking_delta(n)
                upsert(
                    n, "shrinking_delta", f"S_shrink_delta_{d_shrink:.4f}",
                    d_shrink, True, False,
                )

    if experiment in {"all", "frontier"}:
        upsert(frontier_n, "continuous", "C0_continuous", None, False, True)
        main_frontier = set(normalise_deltas(frontier_fixed_deltas))
        stress_set = set(normalise_deltas(stress_deltas)) if stress_deltas else set()
        for d in sorted(main_frontier):
            lab = delta_label(d)
            upsert(
                frontier_n, f"fixed_delta_{lab}", f"F_delta_{lab}", d,
                False, True, is_stress=(d in stress_set),
            )

    plan: Dict[int, List[Dict[str, Any]]] = {}
    for n, bucket in plan_map.items():
        specs = list(bucket.values())
        for spec in specs:
            spec["design_scope"] = scope_label(spec["in_general"], spec["in_frontier"])

        def sort_key(spec: Dict[str, Any]) -> Tuple[int, float, str]:
            regime = str(spec["regime"])
            if regime == "continuous":
                return (0, 0.0, regime)
            if regime.startswith("fixed_delta_"):
                return (1, finite_or_nan(spec["delta"]), regime)
            if regime == "shrinking_delta":
                return (2, finite_or_nan(spec["delta"]), regime)
            return (3, finite_or_nan(spec["delta"]), regime)

        plan[n] = sorted(specs, key=sort_key)

    return dict(sorted(plan.items()))


def scenario_specs_for_n(n: int,
                         scenario_plan: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return list(scenario_plan.get(int(n), []))


def flush_records(records: List[Dict[str, Any]],
                  raw_path: Path,
                  save_full_raw: bool,
                  header_written: bool) -> bool:
    if not records:
        return header_written
    df = pd.DataFrame.from_records(records)
    if not save_full_raw:
        for col in COMPACT_RAW_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[COMPACT_RAW_COLUMNS]
    mode = "a" if header_written else "w"
    df.to_csv(raw_path, index=False, mode=mode, header=not header_written)
    records.clear()
    return True


def infer_next_rep_from_raw(raw_path: Path) -> Tuple[int, int, int]:
    """Return (next_rep, max_rep, rows_per_rep) for a compact raw CSV."""
    if not raw_path.exists():
        return 1, 0, 0
    reps = pd.read_csv(raw_path, usecols=["rep"])["rep"]
    if reps.empty:
        return 1, 0, 0
    counts = reps.value_counts().sort_index()
    max_rep = int(counts.index.max())
    missing = sorted(set(range(1, max_rep + 1)).difference(int(v) for v in counts.index))
    if missing:
        raise ValueError(f"raw file has missing rep ids before {max_rep}: {missing[:10]}")
    if counts.nunique() != 1:
        raise ValueError(
            "raw file has uneven rows per rep; refusing to resume because the last chunk may be incomplete"
        )
    return max_rep + 1, max_rep, int(counts.iloc[0])


def advance_rng_to_rep(rng: np.random.Generator, start_rep: int, n_max: int) -> None:
    """Advance the common-random-number stream to the requested replication."""
    for _ in range(1, int(start_rep)):
        rng.beta(2.0, 4.0, size=n_max)
        rng.normal(0.0, 1.0, size=n_max)


def run_simulation(reps: int,
                   scenario_plan: Dict[int, List[Dict[str, Any]]],
                   seed: int,
                   output_dir: Path,
                   print_every: int = 100,
                   silence_rdrobust: bool = True,
                   length_cutoff: float = LENGTH_PATHOLOGY_CUTOFF,
                   honest_length_cutoff: float = HONEST_LENGTH_PATHOLOGY_CUTOFF,
                   save_full_raw: bool = False,
                   chunk_reps: int = 25,
                   overwrite_raw: bool = False,
                   resume_raw: bool = False,
                   start_rep: Optional[int] = None) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not scenario_plan:
        raise ValueError("scenario plan is empty")

    rng = np.random.default_rng(seed)
    n_grid = tuple(sorted(int(n) for n in scenario_plan))
    n_max = max(n_grid)
    expected_rows_per_rep = sum(len(scenario_plan[n]) * len(ALL_METHODS) for n in n_grid)

    raw_path = output_dir / "discrete_mechanism_raw.csv"
    header_written = False
    if resume_raw:
        if raw_path.exists():
            inferred_start, max_rep, rows_per_rep = infer_next_rep_from_raw(raw_path)
            if rows_per_rep != expected_rows_per_rep:
                raise ValueError(
                    "existing raw file was generated by a different scenario plan: "
                    f"rows per rep={rows_per_rep}, expected={expected_rows_per_rep}"
                )
            if start_rep is None:
                start_rep = inferred_start
            elif int(start_rep) <= max_rep:
                raise ValueError(
                    f"--start-rep {start_rep} would duplicate existing raw reps 1..{max_rep}; "
                    f"use --start-rep {inferred_start} or omit --start-rep"
                )
            header_written = True
            print(
                f"resuming raw at rep {start_rep}; existing max rep={max_rep}, "
                f"rows per rep={rows_per_rep}",
                flush=True,
            )
        else:
            start_rep = int(start_rep or 1)
    else:
        start_rep = int(start_rep or 1)
        if start_rep != 1:
            raise ValueError("--start-rep requires --resume-raw")
        if raw_path.exists():
            if overwrite_raw:
                raw_path.unlink()
            else:
                raise FileExistsError(
                    f"raw output already exists: {raw_path}. "
                    "Use --raw-input to post-process it, --resume-raw to append missing reps, "
                    "or --overwrite-raw to intentionally replace it."
                )

    buffer: List[Dict[str, Any]] = []
    start = time.time()

    if start_rep > reps:
        print(f"raw already contains reps through {start_rep - 1}; no new simulation needed")
        return pd.read_csv(raw_path)

    if start_rep > 1:
        advance_rng_to_rep(rng, start_rep, n_max)

    for rep in range(int(start_rep), reps + 1):
        # Nested common random numbers: all n use prefixes of the same latent X
        # and error vectors; all scenarios at a given n use those same draws.
        x_latent_full = 2.0 * rng.beta(2.0, 4.0, size=n_max) - 1.0
        eps_full = ERROR_SD * rng.normal(0.0, 1.0, size=n_max)

        for n in n_grid:
            x_latent = x_latent_full[:n]
            eps = eps_full[:n]
            fit_cache: Dict[Tuple[str, Optional[float]], List[Dict[str, Any]]] = {}

            for spec in scenario_specs_for_n(n, scenario_plan):
                delta = spec["delta"]
                # Fixed and shrinking scenarios can share the same numerical
                # delta (e.g. n=500, delta=0.05). Compute the fit once and clone
                # the method records under the two conceptual labels.
                cache_key = (
                    "continuous" if delta is None else "discrete",
                    None if delta is None else round(float(delta), 12),
                )
                if cache_key not in fit_cache:
                    x_obs = discretize_midpoint_grid(x_latent, delta)
                    y = mu2(x_obs) + eps
                    fit_cache[cache_key] = run_methods_for_sample(
                        rep=rep,
                        n=n,
                        regime=spec["regime"],
                        scenario=spec["scenario"],
                        delta=delta,
                        x=x_obs,
                        y=y,
                        silence_rdrobust=silence_rdrobust,
                        length_cutoff=length_cutoff,
                        honest_length_cutoff=honest_length_cutoff,
                    )

                for template in fit_cache[cache_key]:
                    rec = template.copy()
                    rec.update({
                        "rep": rep,
                        "n": n,
                        "regime": spec["regime"],
                        "scenario": spec["scenario"],
                        "delta": np.nan if delta is None else float(delta),
                        "in_general": bool(spec["in_general"]),
                        "in_frontier": bool(spec["in_frontier"]),
                        "design_scope": str(spec["design_scope"]),
                    })
                    buffer.append(rec)

        if chunk_reps and rep % chunk_reps == 0:
            header_written = flush_records(buffer, raw_path, save_full_raw, header_written)

        if print_every and rep % print_every == 0:
            elapsed = time.time() - start
            print(f"rep {rep:>6}/{reps} finished, elapsed={elapsed/60:.1f} min", flush=True)

    flush_records(buffer, raw_path, save_full_raw, header_written)
    print(f"saved raw results to {raw_path}")
    return pd.read_csv(raw_path)


# =============================================================================
# 7. Summaries and paired comparisons
# =============================================================================

def mc_se_coverage(p: float, n_valid: int) -> float:
    if n_valid <= 0 or not math.isfinite(p):
        return float("nan")
    return math.sqrt(max(p * (1.0 - p), 0.0) / n_valid)


def safe_bool_mean(s: pd.Series) -> float:
    if s.empty:
        return np.nan
    return float(s.astype(float).mean())


def trimmed_mean(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> float:
    vals = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return np.nan
    lo = np.quantile(vals, lower)
    hi = np.quantile(vals, upper)
    vals = vals[(vals >= lo) & (vals <= hi)]
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals))


def add_delta_plot_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["delta_plot"] = np.where(out["regime"].eq("continuous"), 0.0, out["delta"])
    out["delta_label_plot"] = out["delta_plot"].apply(lambda z: "0" if pd.isna(z) or abs(float(z)) < 1e-15 else delta_label(float(z)))
    return out


def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    # Normalise boolean-ish columns after CSV read.
    for col in ["raw_success", "success", "minimal_usable", "support_adequate", "rbc_support_adequate",
                "pathological_length", "covered", "miss_below", "miss_above",
                "formal_support_ok", "frontier_display_ok", "in_general", "in_frontier"]:
        if col in raw.columns:
            raw[col] = raw[col].map(lambda v: np.nan if pd.isna(v) else bool(v) if isinstance(v, bool) else str(v).lower() in ("true", "1", "1.0"))

    group_cols = ["n", "regime", "scenario", "delta", "method"]
    rows: List[Dict[str, Any]] = []

    for keys, g in raw.groupby(group_cols, dropna=False):
        n, regime, scenario, delta, method = keys
        in_general = bool((g["in_general"] == True).any()) if "in_general" in g else False
        in_frontier = bool((g["in_frontier"] == True).any()) if "in_frontier" in g else False
        design_scope = scope_label(in_general, in_frontier)
        n_total = len(g)
        raw_valid = g[g["raw_success"] == True].copy()
        usable = g[g["minimal_usable"] == True].copy()
        n_raw_success = len(raw_valid)
        n_minimal_usable = len(usable)
        n_support_adequate = int((g["support_adequate"] == True).sum()) if "support_adequate" in g else 0
        n_rbc_support_adequate = int((g["rbc_support_adequate"] == True).sum()) if "rbc_support_adequate" in g else 0
        n_formal_support_ok = int((g["formal_support_ok"] == True).sum()) if "formal_support_ok" in g else 0
        n_frontier_display_ok = int((g["frontier_display_ok"] == True).sum()) if "frontier_display_ok" in g else 0

        raw_success_rate = n_raw_success / n_total if n_total else np.nan
        minimal_usability_rate = n_minimal_usable / n_total if n_total else np.nan
        support_adequacy_rate = n_support_adequate / n_total if n_total else np.nan
        rbc_support_adequacy_rate = n_rbc_support_adequate / n_total if n_total else np.nan
        formal_support_ok_rate = n_formal_support_ok / n_total if n_total else np.nan
        frontier_display_ok_rate = n_frontier_display_ok / n_total if n_total else np.nan
        raw_failure_rate = 1.0 - raw_success_rate if math.isfinite(raw_success_rate) else np.nan
        nonusable_rate = 1.0 - minimal_usability_rate if math.isfinite(minimal_usability_rate) else np.nan

        if n_raw_success > 0:
            coverage_raw = safe_bool_mean(raw_valid["covered"])
        else:
            coverage_raw = np.nan

        if n_minimal_usable > 0:
            coverage_usable = safe_bool_mean(usable["covered"])
            bias = float(usable["est_error"].mean())
            sd = float(usable["estimate"].std(ddof=1)) if n_minimal_usable > 1 else np.nan
            rmse = float(np.sqrt(np.mean(usable["est_error"] ** 2)))
            avg_length = float(usable["ci_length"].mean())
            median_length = float(usable["ci_length"].median())
            p90_length = float(usable["ci_length"].quantile(0.90))
            p95_length = float(usable["ci_length"].quantile(0.95))
            p99_length = float(usable["ci_length"].quantile(0.99))
            trimmed_mean_length = trimmed_mean(usable["ci_length"])
            avg_estimate = float(usable["estimate"].mean())
            avg_se = float(usable["se"].mean())
            miss_below_rate = safe_bool_mean(usable["miss_below"])
            miss_above_rate = safe_bool_mean(usable["miss_above"])
        else:
            coverage_usable = bias = sd = rmse = avg_length = median_length = np.nan
            p90_length = p95_length = p99_length = trimmed_mean_length = np.nan
            avg_estimate = avg_se = miss_below_rate = miss_above_rate = np.nan

        coverage_unconditional = float(((g["covered"] == True) & (g["minimal_usable"] == True)).sum() / n_total) if n_total else np.nan
        pathological_length_rate = float((g["pathological_length"] == True).sum() / n_total) if n_total else np.nan

        def mean_col(col: str, source: pd.DataFrame = raw_valid) -> float:
            return float(pd.to_numeric(source[col], errors="coerce").mean()) if len(source) > 0 and col in source else np.nan

        row = {
            "n": n,
            "regime": regime,
            "scenario": scenario,
            "delta": delta,
            "in_general": in_general,
            "in_frontier": in_frontier,
            "design_scope": design_scope,
            "method": method,
            "n_total": n_total,
            "n_raw_success": n_raw_success,
            "raw_success_rate": raw_success_rate,
            "raw_failure_rate": raw_failure_rate,
            "n_minimal_usable": n_minimal_usable,
            "minimal_usability_rate": minimal_usability_rate,
            "n_nonusable": n_total - n_minimal_usable,
            "nonusable_rate": nonusable_rate,
            "n_support_adequate": n_support_adequate,
            "support_adequacy_rate": support_adequacy_rate,
            "n_rbc_support_adequate": n_rbc_support_adequate,
            "rbc_support_adequacy_rate": rbc_support_adequacy_rate,
            "n_formal_support_ok": n_formal_support_ok,
            "formal_support_ok_rate": formal_support_ok_rate,
            "n_frontier_display_ok": n_frontier_display_ok,
            "frontier_display_ok_rate": frontier_display_ok_rate,
            "coverage_conditional_on_raw_success": coverage_raw,
            "coverage_conditional_on_minimal_usable": coverage_usable,
            "coverage_unconditional": coverage_unconditional,
            # Backward-compatible names; from now on, coverage means usable coverage.
            "coverage": coverage_usable,
            "coverage_mc_se": mc_se_coverage(coverage_usable, n_minimal_usable),
            "bias": bias,
            "abs_bias_over_avg_se": abs(bias) / avg_se if math.isfinite(bias) and math.isfinite(avg_se) and avg_se > 0 else np.nan,
            "sd": sd,
            "rmse": rmse,
            "avg_length": avg_length,
            "median_length": median_length,
            "p90_length": p90_length,
            "p95_length": p95_length,
            "p99_length": p99_length,
            "trimmed_mean_length": trimmed_mean_length,
            "pathological_length_rate": pathological_length_rate,
            "avg_estimate": avg_estimate,
            "avg_se": avg_se,
            "miss_below_rate": miss_below_rate,
            "miss_above_rate": miss_above_rate,
            "avg_h_left": mean_col("h_left"),
            "avg_h_right": mean_col("h_right"),
            "avg_b_left": mean_col("b_left"),
            "avg_b_right": mean_col("b_right"),
            "avg_local_support_score": mean_col("local_support_score"),
            "avg_min_unique_h": mean_col("min_unique_h"),
            "avg_min_unique_b": mean_col("min_unique_b"),
            "avg_unique_h_left": mean_col("unique_h_left"),
            "avg_unique_h_right": mean_col("unique_h_right"),
            "avg_unique_b_left": mean_col("unique_b_left"),
            "avg_unique_b_right": mean_col("unique_b_right"),
            "avg_h_over_delta_left": mean_col("h_over_delta_left"),
            "avg_h_over_delta_right": mean_col("h_over_delta_right"),
            "avg_b_over_delta_left": mean_col("b_over_delta_left"),
            "avg_b_over_delta_right": mean_col("b_over_delta_right"),
            "avg_nearest_support_distance": mean_col("nearest_support_distance"),
            "avg_mass_share_h_left": mean_col("mass_share_h_left"),
            "avg_mass_share_h_right": mean_col("mass_share_h_right"),
            "avg_mass_share_b_left": mean_col("mass_share_b_left"),
            "avg_mass_share_b_right": mean_col("mass_share_b_right"),
            "avg_bias_bound": mean_col("bias_bound", usable),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = add_delta_plot_cols(summary)
    summary = summary.sort_values(["method", "regime", "n", "delta_plot"]).reset_index(drop=True)
    return summary


def diagnostic_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """One diagnostic row per design cell, based on rdrobust-produced diagnostics."""
    group_cols = ["n", "regime", "scenario", "delta"]
    diag_cols = [
        "h_left", "h_right", "b_left", "b_right",
        "unique_h_left", "unique_h_right", "unique_b_left", "unique_b_right",
        "obs_h_left", "obs_h_right", "obs_b_left", "obs_b_right",
        "min_unique_h", "min_unique_b",
        "h_over_delta_left", "h_over_delta_right", "b_over_delta_left", "b_over_delta_right",
        "nearest_support_distance", "M_left_rdrobust", "M_right_rdrobust",
    ]
    out_cols = group_cols + ["n_with_diagnostics", "rbc_min_support_rate", "rbc_support_adequate_rate"]
    for col in diag_cols:
        out_cols += [f"{col}_mean", f"{col}_median", f"{col}_min", f"{col}_max"]

    required = {"method", "h_left", *group_cols}
    if raw.empty or not required.issubset(raw.columns):
        return pd.DataFrame(columns=out_cols)

    base = raw[(raw["method"] == "RBC MSE") & (raw["h_left"].notna())].copy()
    if base.empty:
        return pd.DataFrame(columns=out_cols)

    rows: List[Dict[str, Any]] = []
    for keys, g in base.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_with_diagnostics"] = len(g)
        row["rbc_min_support_rate"] = float(((g["min_unique_h"] >= RBC_MIN_UNIQUE_H) & (g["min_unique_b"] >= RBC_MIN_UNIQUE_B)).mean())
        row["rbc_support_adequate_rate"] = float(((g["min_unique_h"] >= 3) & (g["min_unique_b"] >= 5)).mean())
        for col in diag_cols:
            if col in g.columns:
                vals = pd.to_numeric(g[col], errors="coerce")
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_median"] = float(vals.median())
                row[f"{col}_min"] = float(vals.min())
                row[f"{col}_max"] = float(vals.max())
            else:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_median"] = np.nan
                row[f"{col}_min"] = np.nan
                row[f"{col}_max"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=out_cols).sort_values(["regime", "delta", "n"]).reset_index(drop=True)


def boolish_to_float(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return 1.0 if bool(value) else 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        val = float(value)
        if not math.isfinite(val):
            return np.nan
        return 1.0 if val != 0.0 else 0.0
    text = str(value).strip().lower()
    if text in ("true", "t", "yes", "y", "1", "1.0"):
        return 1.0
    if text in ("false", "f", "no", "n", "0", "0.0"):
        return 0.0
    return np.nan


def comparison_values(series: pd.Series, variable: str) -> pd.Series:
    vals = series.copy()
    if variable == "covered":
        return vals.map(boolish_to_float)
    return pd.to_numeric(vals, errors="coerce")


def paired_method_comparisons(raw: pd.DataFrame) -> pd.DataFrame:
    """Paired method differences within same rep/n/regime/scenario, using minimal usable fits."""
    variables = ["covered", "estimate", "ci_length"]
    pairs = [
        ("RBC MSE", "Conventional MSE"),
        ("Bias-aware/Honest", "RBC MSE"),
    ]
    rows = []
    id_cols = ["rep", "n", "regime", "scenario", "delta"]
    usable = raw[raw["minimal_usable"] == True].copy()
    for var in variables:
        usable_var = usable.copy()
        usable_var[var] = comparison_values(usable_var[var], var)
        wide = usable_var.pivot_table(index=id_cols, columns="method", values=var, aggfunc="first")
        for m1, m0 in pairs:
            if m1 not in wide.columns or m0 not in wide.columns:
                continue
            diff = wide[m1] - wide[m0]
            diff = diff.dropna()
            if len(diff) == 0:
                continue
            tmp = diff.reset_index(name="diff")
            for keys, g in tmp.groupby(["n", "regime", "scenario", "delta"], dropna=False):
                n, regime, scenario, delta = keys
                rows.append({
                    "n": n,
                    "regime": regime,
                    "scenario": scenario,
                    "delta": delta,
                    "variable": var,
                    "comparison": f"{m1} - {m0}",
                    "n_pairs": len(g),
                    "mean_diff": float(g["diff"].mean()),
                    "se_diff": float(g["diff"].std(ddof=1) / math.sqrt(len(g))) if len(g) > 1 else np.nan,
                })
    return pd.DataFrame(rows)


def paired_scenario_comparisons(raw: pd.DataFrame) -> pd.DataFrame:
    """Paired scenario differences versus continuous within rep/n/method."""
    variables = ["covered", "estimate", "ci_length"]
    id_cols = ["rep", "n", "method"]
    scenario_cols = ["regime", "scenario", "delta"]
    usable = raw[raw["minimal_usable"] == True].copy()
    if usable.empty or "continuous" not in set(usable.get("regime", pd.Series(dtype=str)).astype(str)):
        return pd.DataFrame(columns=[
            "n", "method", "regime", "scenario", "delta", "variable", "comparison",
            "n_pairs", "mean_diff", "se_diff",
        ])

    rows = []
    continuous = usable[usable["regime"] == "continuous"].copy()
    scenarios = usable[usable["regime"] != "continuous"].copy()
    for var in variables:
        cont = continuous[id_cols + [var]].copy()
        scen = scenarios[id_cols + scenario_cols + [var]].copy()
        cont[var] = comparison_values(cont[var], var)
        scen[var] = comparison_values(scen[var], var)
        merged = scen.merge(cont, on=id_cols, how="inner", suffixes=("", "_continuous"))
        if merged.empty:
            continue
        merged["diff"] = merged[var] - merged[f"{var}_continuous"]
        merged = merged.dropna(subset=["diff"])
        for keys, g in merged.groupby(["n", "method", "regime", "scenario", "delta"], dropna=False):
            n, method, regime, scenario, delta = keys
            rows.append({
                "n": n,
                "method": method,
                "regime": regime,
                "scenario": scenario,
                "delta": delta,
                "variable": var,
                "comparison": f"{scenario} - continuous",
                "n_pairs": len(g),
                "mean_diff": float(g["diff"].mean()),
                "se_diff": float(g["diff"].std(ddof=1) / math.sqrt(len(g))) if len(g) > 1 else np.nan,
            })
    return pd.DataFrame(rows)


def failure_reason_counts(raw: pd.DataFrame) -> pd.DataFrame:
    tmp = raw[raw["minimal_usable"] != True].copy()
    if tmp.empty:
        return pd.DataFrame()
    out = (
        tmp.groupby(["method", "nonusable_reason"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["method", "count"], ascending=[True, False])
    )
    return out


def boolish_series(s: pd.Series) -> pd.Series:
    return s.map(lambda v: np.nan if pd.isna(v) else bool(v) if isinstance(v, bool) else str(v).lower() in ("true", "1", "1.0"))


def ensure_frontier_diagnostics(raw: pd.DataFrame) -> pd.DataFrame:
    """Add failure-frontier diagnostics to old or new raw files."""
    out = raw.copy()

    bool_cols = [
        "raw_success", "success", "minimal_usable", "support_adequate",
        "rbc_support_adequate", "pathological_length", "covered", "miss_below",
        "miss_above", "formal_support_ok", "frontier_display_ok",
    ]
    for col in bool_cols:
        if col in out.columns:
            out[col] = boolish_series(out[col])

    numeric_cols = [
        "delta", "h_left", "h_right", "b_left", "b_right",
        "unique_x_left", "unique_x_right", "unique_x_total", "n_left", "n_right",
        "unique_h_left", "unique_h_right", "unique_b_left", "unique_b_right",
        "obs_h_left", "obs_h_right", "obs_b_left", "obs_b_right",
        "min_unique_h", "min_unique_b", "local_support_score",
        "mass_share_left", "mass_share_right", "mass_share_h_left", "mass_share_h_right",
        "mass_share_b_left", "mass_share_b_right", "estimate", "se", "ci_l", "ci_u", "ci_length",
        "bias_bound", "bias_bound_left", "bias_bound_right",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "local_support_score" not in out.columns and {"min_unique_h", "min_unique_b"}.issubset(out.columns):
        out["local_support_score"] = out[["min_unique_h", "min_unique_b"]].min(axis=1)

    for side in ("left", "right"):
        u_col = f"unique_h_{side}"
        n_col = f"obs_h_{side}"
        m_col = f"mass_share_h_{side}"
        if m_col not in out.columns and {u_col, n_col}.issubset(out.columns):
            out[m_col] = 1.0 - out[u_col] / out[n_col].replace(0, np.nan)

        u_col = f"unique_b_{side}"
        n_col = f"obs_b_{side}"
        m_col = f"mass_share_b_{side}"
        if m_col not in out.columns and {u_col, n_col}.issubset(out.columns):
            out[m_col] = 1.0 - out[u_col] / out[n_col].replace(0, np.nan)

        u_col = f"unique_x_{side}"
        n_col = f"n_{side}"
        m_col = f"mass_share_{side}"
        if m_col not in out.columns and {u_col, n_col}.issubset(out.columns):
            out[m_col] = 1.0 - out[u_col] / out[n_col].replace(0, np.nan)

    if {"min_unique_h", "min_unique_b"}.issubset(out.columns):
        out["formal_support_ok"] = (out["min_unique_h"] >= P + 1) & (out["min_unique_b"] >= Q + 1)
    elif "formal_support_ok" not in out.columns:
        out["formal_support_ok"] = False

    required_display = {"raw_success", "formal_support_ok", "obs_h_left", "obs_h_right", "min_unique_h", "min_unique_b"}
    if required_display.issubset(out.columns):
        out["frontier_display_ok"] = (
            (out["raw_success"] == True)
            & (out["formal_support_ok"] == True)
            & (out["obs_h_left"] >= FRONTIER_MIN_OBS_H)
            & (out["obs_h_right"] >= FRONTIER_MIN_OBS_H)
            & (out["min_unique_h"] >= FRONTIER_DISPLAY_MIN_UNIQUE_H)
            & (out["min_unique_b"] >= FRONTIER_DISPLAY_MIN_UNIQUE_B)
        )
    elif "frontier_display_ok" not in out.columns:
        out["frontier_display_ok"] = False

    if "frontier_failure_reason" not in out.columns:
        out["frontier_failure_reason"] = out.apply(frontier_failure_reason, axis=1)
    else:
        missing = out["frontier_failure_reason"].isna() | (out["frontier_failure_reason"].astype(str).str.len() == 0)
        if missing.any():
            out.loc[missing, "frontier_failure_reason"] = out.loc[missing].apply(frontier_failure_reason, axis=1)

    return out


def support_bin(score: Any) -> str:
    val = finite_or_nan(score)
    if not math.isfinite(val):
        return "missing"
    if val <= 2:
        return "0-2"
    if val <= 4:
        return "3-4"
    if val <= 7:
        return "5-7"
    if val <= 12:
        return "8-12"
    return "13+"


SUPPORT_BIN_ORDER = ("0-2", "3-4", "5-7", "8-12", "13+", "missing")


def mc_se_bool(series: pd.Series) -> float:
    vals = series.dropna().astype(float)
    if vals.empty:
        return np.nan
    p = float(vals.mean())
    return mc_se_coverage(p, len(vals))


def summarize_by_support_frontier(raw: pd.DataFrame) -> pd.DataFrame:
    """Coverage/usability frontier by local-support score bins."""
    if raw.empty or "local_support_score" not in raw.columns:
        return pd.DataFrame()

    df = raw.copy()
    if "regime" in df.columns:
        df = df[
            df["regime"].astype(str).eq("continuous")
            | df["regime"].astype(str).str.startswith("fixed_delta_")
        ].copy()
    if df.empty:
        return pd.DataFrame()

    df["support_bin"] = df["local_support_score"].map(support_bin)
    rows: List[Dict[str, Any]] = []
    group_cols = ["n", "support_bin", "method"] if "n" in df.columns else ["support_bin", "method"]

    for keys, g in df.groupby(group_cols, dropna=False):
        if len(group_cols) == 3:
            n, bin_label, method = keys
        else:
            n, bin_label, method = np.nan, keys[0], keys[1]
        success = g[g["raw_success"] == True].copy() if "raw_success" in g else g.iloc[0:0].copy()
        usable = g[g["minimal_usable"] == True].copy() if "minimal_usable" in g else g.iloc[0:0].copy()
        display = g[g["frontier_display_ok"] == True].copy() if "frontier_display_ok" in g else g.iloc[0:0].copy()
        rows.append({
            "n": n,
            "support_bin": bin_label,
            "method": method,
            "n_rows": len(g),
            "n_success": len(success),
            "n_minimal_usable": len(usable),
            "n_frontier_display_ok": len(display),
            "success_rate": safe_bool_mean(g["raw_success"]) if "raw_success" in g else np.nan,
            "minimal_usability_rate": safe_bool_mean(g["minimal_usable"]) if "minimal_usable" in g else np.nan,
            "coverage": safe_bool_mean(success["covered"]) if len(success) and "covered" in success else np.nan,
            "coverage_conditional_on_raw_success": safe_bool_mean(success["covered"]) if len(success) and "covered" in success else np.nan,
            "coverage_conditional_on_minimal_usable": safe_bool_mean(usable["covered"]) if len(usable) and "covered" in usable else np.nan,
            "coverage_conditional_on_frontier_display_ok": safe_bool_mean(display["covered"]) if len(display) and "covered" in display else np.nan,
            "mc_se_coverage": mc_se_bool(success["covered"]) if len(success) and "covered" in success else np.nan,
            "coverage_unconditional": float(((g["covered"] == True) & (g["raw_success"] == True)).sum() / len(g)) if len(g) and {"covered", "raw_success"}.issubset(g.columns) else np.nan,
            "avg_ci_length": float(pd.to_numeric(success["ci_length"], errors="coerce").mean()) if len(success) and "ci_length" in success else np.nan,
            "median_ci_length": float(pd.to_numeric(success["ci_length"], errors="coerce").median()) if len(success) and "ci_length" in success else np.nan,
            "p95_ci_length": float(pd.to_numeric(success["ci_length"], errors="coerce").quantile(0.95)) if len(success) and "ci_length" in success else np.nan,
            "p99_ci_length": float(pd.to_numeric(success["ci_length"], errors="coerce").quantile(0.99)) if len(success) and "ci_length" in success else np.nan,
            "pathological_length_rate": safe_bool_mean(g["pathological_length"]) if "pathological_length" in g else np.nan,
            "formal_support_ok_rate": safe_bool_mean(g["formal_support_ok"]) if "formal_support_ok" in g else np.nan,
            "coverage_display_ok_rate": safe_bool_mean(g["frontier_display_ok"]) if "frontier_display_ok" in g else np.nan,
            "avg_local_support_score": float(pd.to_numeric(g["local_support_score"], errors="coerce").mean()),
            "avg_min_unique_h": float(pd.to_numeric(g["min_unique_h"], errors="coerce").mean()) if "min_unique_h" in g else np.nan,
            "avg_min_unique_b": float(pd.to_numeric(g["min_unique_b"], errors="coerce").mean()) if "min_unique_b" in g else np.nan,
            "avg_obs_h_left": float(pd.to_numeric(g["obs_h_left"], errors="coerce").mean()) if "obs_h_left" in g else np.nan,
            "avg_obs_h_right": float(pd.to_numeric(g["obs_h_right"], errors="coerce").mean()) if "obs_h_right" in g else np.nan,
            "avg_obs_b_left": float(pd.to_numeric(g["obs_b_left"], errors="coerce").mean()) if "obs_b_left" in g else np.nan,
            "avg_obs_b_right": float(pd.to_numeric(g["obs_b_right"], errors="coerce").mean()) if "obs_b_right" in g else np.nan,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["support_bin"] = pd.Categorical(out["support_bin"], categories=SUPPORT_BIN_ORDER, ordered=True)
    sort_cols = ["n", "method", "support_bin"] if "n" in out.columns else ["method", "support_bin"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def frontier_failure_reason_counts(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or "frontier_failure_reason" not in raw.columns:
        return pd.DataFrame()
    group_cols = [c for c in ["n", "regime", "method", "frontier_failure_reason"] if c in raw.columns]
    out = raw.groupby(group_cols, dropna=False).size().reset_index(name="count")
    denom_cols = [c for c in ["n", "regime", "method"] if c in raw.columns]
    denom = out.groupby(denom_cols)["count"].transform("sum") if denom_cols else out["count"].sum()
    out["share_within_scenario_method"] = out["count"] / denom
    return out.sort_values(denom_cols + ["count"], ascending=[True] * len(denom_cols) + [False]).reset_index(drop=True)


def scalar_from_summary(summary: pd.DataFrame,
                        n: int,
                        method: str,
                        regime: str,
                        column: str) -> float:
    if summary.empty or column not in summary.columns:
        return np.nan
    g = summary[(summary["n"].astype(int) == int(n)) & (summary["method"] == method) & (summary["regime"] == regime)]
    if g.empty:
        return np.nan
    return finite_or_nan(g.iloc[0][column])


def make_interpretation_table(summary: pd.DataFrame,
                              support_frontier: pd.DataFrame) -> pd.DataFrame:
    """Generate result-dependent answers for the failure-frontier extension."""
    rows: List[Dict[str, Any]] = []
    if summary.empty or "n" not in summary.columns:
        return pd.DataFrame(columns=["n", "question", "answer_from_results", "supporting_metric"])

    for n in sorted(int(v) for v in pd.to_numeric(summary["n"], errors="coerce").dropna().unique()):
        conv_cont = scalar_from_summary(summary, n, "Conventional MSE", "continuous", "coverage")
        rbc_cont = scalar_from_summary(summary, n, "RBC MSE", "continuous", "coverage")
        if math.isfinite(conv_cont) and math.isfinite(rbc_cont):
            diff = rbc_cont - conv_cont
            if diff > 0.01:
                answer = "RBC improves coverage relative to conventional inference in the continuous baseline."
            elif diff < -0.01:
                answer = "RBC has lower coverage than conventional inference in the continuous baseline."
            else:
                answer = "RBC and conventional coverage are similar in the continuous baseline."
            rows.append({
                "n": n,
                "question": "Does RBC improve coverage relative to conventional inference in the continuous baseline?",
                "answer_from_results": answer,
                "supporting_metric": f"RBC={rbc_cont:.4f}, Conventional={conv_cont:.4f}, diff={diff:.4f}",
            })

        fine_regime = "fixed_delta_0.01"
        rbc_fine = scalar_from_summary(summary, n, "RBC MSE", fine_regime, "coverage")
        if math.isfinite(rbc_cont) and math.isfinite(rbc_fine):
            gap = rbc_fine - rbc_cont
            answer = (
                "Fine discretization preserves RBC coverage close to the continuous baseline."
                if abs(gap) <= 0.015 else
                "Fine discretization moves RBC coverage noticeably away from the continuous baseline."
            )
            rows.append({
                "n": n,
                "question": "Does fine discretization preserve RBC coverage relative to the continuous baseline?",
                "answer_from_results": answer,
                "supporting_metric": f"RBC fine delta=0.01={rbc_fine:.4f}, continuous={rbc_cont:.4f}, gap={gap:.4f}",
            })

        fixed = summary[(summary["n"].astype(int) == n) & (summary["method"] == "RBC MSE") & (summary["regime"].astype(str).str.startswith("fixed_delta_"))].copy()
        if not fixed.empty and "avg_local_support_score" in fixed:
            fixed["delta_num"] = pd.to_numeric(fixed["delta"], errors="coerce")
            fine = fixed.loc[fixed["delta_num"].idxmin()]
            coarse = fixed.loc[fixed["delta_num"].idxmax()]
            fine_support = finite_or_nan(fine.get("avg_local_support_score"))
            coarse_support = finite_or_nan(coarse.get("avg_local_support_score"))
            if math.isfinite(fine_support) and math.isfinite(coarse_support):
                answer = (
                    "Coarser discretization reduces distinct local support points."
                    if coarse_support < fine_support else
                    "Coarser discretization does not reduce average local support in this run."
                )
                rows.append({
                    "n": n,
                    "question": "Does coarse discretization reduce local support points?",
                    "answer_from_results": answer,
                    "supporting_metric": f"finest delta={fine['delta_num']:.4g}: {fine_support:.3f}; coarsest delta={coarse['delta_num']:.4g}: {coarse_support:.3f}",
                })

        sf = support_frontier[(support_frontier["n"].astype(int) == n) & (support_frontier["method"] == "RBC MSE")].copy() if not support_frontier.empty and "n" in support_frontier else pd.DataFrame()
        if not sf.empty:
            low = sf[sf["support_bin"].astype(str).isin(["0-2", "3-4"])]
            high = sf[sf["support_bin"].astype(str).isin(["8-12", "13+"])]
            low_cov = float(np.average(low["coverage"], weights=low["n_success"])) if not low.empty and low["coverage"].notna().any() else np.nan
            high_cov = float(np.average(high["coverage"], weights=high["n_success"])) if not high.empty and high["coverage"].notna().any() else np.nan
            low_display = float(np.average(low["coverage_display_ok_rate"], weights=low["n_rows"])) if not low.empty and low["coverage_display_ok_rate"].notna().any() else np.nan
            if math.isfinite(low_cov) and math.isfinite(high_cov):
                if low_cov < high_cov - 0.02:
                    answer = "RBC coverage is lower in low-support bins."
                elif math.isfinite(low_display) and low_display < 0.20:
                    answer = "RBC coverage does not fall sharply, but low-support bins fail the display/usability diagnostic."
                else:
                    answer = "RBC coverage does not show a clear low-support decline in the available bins."
                rows.append({
                    "n": n,
                    "question": "Does RBC coverage decline when local support score is low?",
                    "answer_from_results": answer,
                    "supporting_metric": f"low-support coverage={low_cov:.4f}, high-support coverage={high_cov:.4f}, low-support display-ok={low_display:.4f}",
                })

        if not fixed.empty and "frontier_display_ok_rate" in fixed:
            worst = fixed.loc[pd.to_numeric(fixed["coverage"], errors="coerce").idxmin()]
            worst_cov = finite_or_nan(worst.get("coverage"))
            worst_display = finite_or_nan(worst.get("frontier_display_ok_rate"))
            worst_support = finite_or_nan(worst.get("avg_local_support_score"))
            answer = (
                "The weakest RBC coverage cell also has poor frontier usability/local support."
                if math.isfinite(worst_display) and worst_display < USABILITY_PLOT_THRESHOLD else
                "The weakest RBC coverage cell still passes the aggregate usability threshold."
            )
            rows.append({
                "n": n,
                "question": "Are low-coverage scenarios also low-usability scenarios?",
                "answer_from_results": answer,
                "supporting_metric": f"worst fixed delta={finite_or_nan(worst.get('delta')):.4g}, coverage={worst_cov:.4f}, display-ok={worst_display:.4f}, support={worst_support:.3f}",
            })

        if not fixed.empty and "frontier_display_ok_rate" in fixed:
            coarsest = fixed.loc[pd.to_numeric(fixed["delta"], errors="coerce").idxmax()]
            display = finite_or_nan(coarsest.get("frontier_display_ok_rate"))
            path_rate = finite_or_nan(coarsest.get("pathological_length_rate"))
            support = finite_or_nan(coarsest.get("avg_local_support_score"))
            if math.isfinite(display) and display < USABILITY_PLOT_THRESHOLD:
                answer = "The evidence points to a local-support estimation/identification boundary, not only an inference problem."
            else:
                answer = "The available grid has not yet pushed far enough to show a decisive local-support boundary."
            rows.append({
                "n": n,
                "question": "Is the discrete failure primarily an inference problem or a local support problem?",
                "answer_from_results": answer,
                "supporting_metric": f"coarsest fixed delta={finite_or_nan(coarsest.get('delta')):.4g}, display-ok={display:.4f}, pathological-length={path_rate:.4f}, support={support:.3f}",
            })

    return pd.DataFrame(rows, columns=["n", "question", "answer_from_results", "supporting_metric"])


def coverage_display_mask(summary: pd.DataFrame,
                          usability_threshold: float,
                          min_display_usable: int) -> pd.Series:
    """True when a coverage estimate has enough usable Monte Carlo evidence to display."""
    if summary.empty:
        return pd.Series(dtype=bool)
    usability_source = summary["minimal_usability_rate"] if "minimal_usability_rate" in summary else pd.Series(np.nan, index=summary.index)
    usable_source = summary["n_minimal_usable"] if "n_minimal_usable" in summary else pd.Series(np.nan, index=summary.index)
    usability = pd.to_numeric(usability_source, errors="coerce")
    n_usable = pd.to_numeric(usable_source, errors="coerce")
    return (usability >= usability_threshold) & (n_usable >= int(min_display_usable))


def add_coverage_display_ok(summary: pd.DataFrame,
                            usability_threshold: float,
                            min_display_usable: int) -> pd.DataFrame:
    out = summary.copy()
    out["coverage_display_ok"] = coverage_display_mask(out, usability_threshold, min_display_usable)
    out["min_display_usable"] = int(min_display_usable)
    return out


def breakdown_frontier(summary: pd.DataFrame,
                       fixed_deltas_main: Sequence[float],
                       usability_threshold: float = USABILITY_PLOT_THRESHOLD,
                       min_display_usable: int = MIN_DISPLAY_USABLE) -> pd.DataFrame:
    df = summary[(summary["method"] == "RBC MSE") & (summary["regime"].str.startswith("fixed_delta_"))].copy()
    df = df[df["delta"].isin([float(d) for d in fixed_deltas_main])]
    df = add_coverage_display_ok(df, usability_threshold, min_display_usable)
    rows = []
    for n, g in df.groupby("n"):
        ok = g[g["coverage_display_ok"] == True]
        max_delta = float(ok["delta"].max()) if not ok.empty else np.nan
        rows.append({
            "n": int(n),
            "max_operational_delta": max_delta,
            "criterion": "coverage_display_ok",
            "usability_threshold": float(usability_threshold),
            "min_display_usable": int(min_display_usable),
        })
    return pd.DataFrame(rows).sort_values("n")



# =============================================================================
# 8. Unified experiment flags, paper-facing figures, and outputs
# =============================================================================

plt = None


def ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    return _plt


def ensure_experiment_flags(raw: pd.DataFrame,
                            general_n_grid: Sequence[int],
                            general_fixed_deltas: Sequence[float],
                            frontier_n: int,
                            frontier_fixed_deltas: Sequence[float]) -> pd.DataFrame:
    """Add or repair experiment-membership flags, including for legacy raw files."""
    out = raw.copy()
    general_n_set = {int(n) for n in general_n_grid}
    general_delta_set = {round(float(d), 12) for d in general_fixed_deltas}
    frontier_delta_set = {round(float(d), 12) for d in frontier_fixed_deltas}

    n_series = pd.to_numeric(out.get("n", pd.Series(np.nan, index=out.index)), errors="coerce")
    regime = out.get("regime", pd.Series("", index=out.index)).astype(str)
    delta = pd.to_numeric(out.get("delta", pd.Series(np.nan, index=out.index)), errors="coerce")
    delta_key = delta.round(12)

    inferred_general = (
        n_series.isin(general_n_set)
        & (
            regime.eq("continuous")
            | regime.eq("shrinking_delta")
            | (regime.str.startswith("fixed_delta_") & delta_key.isin(general_delta_set))
        )
    )
    inferred_frontier = (
        n_series.eq(int(frontier_n))
        & (
            regime.eq("continuous")
            | (regime.str.startswith("fixed_delta_") & delta_key.isin(frontier_delta_set))
        )
    )

    if "in_general" in out.columns and boolish_series(out["in_general"]).notna().any():
        existing = boolish_series(out["in_general"])
        out["in_general"] = existing.where(existing.notna(), inferred_general).astype(bool)
    else:
        out["in_general"] = inferred_general

    if "in_frontier" in out.columns and boolish_series(out["in_frontier"]).notna().any():
        existing = boolish_series(out["in_frontier"])
        out["in_frontier"] = existing.where(existing.notna(), inferred_frontier).astype(bool)
    else:
        out["in_frontier"] = inferred_frontier

    out["design_scope"] = [scope_label(bool(g), bool(f)) for g, f in zip(out["in_general"], out["in_frontier"])]
    return out


def coverage_heatmap_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "rbc_coverage",
        ["#0B2B5A", "#566074", "#9B967C", "#CDBD67", "#FFE54F"],
    )


def _find_summary_row(df: pd.DataFrame, n: int, regime: str,
                      delta: Optional[float] = None) -> Optional[pd.Series]:
    g = df[(pd.to_numeric(df["n"], errors="coerce") == int(n)) & (df["regime"].astype(str) == regime)]
    if delta is not None:
        g = g[np.isclose(pd.to_numeric(g["delta"], errors="coerce"), float(delta), atol=1e-12, rtol=0.0)]
    return None if g.empty else g.iloc[0]


def plot_figure_4_10(summary: pd.DataFrame,
                     output_dir: Path,
                     general_n_grid: Sequence[int],
                     general_fixed_deltas: Sequence[float],
                     usability_threshold: float = USABILITY_PLOT_THRESHOLD,
                     min_display_usable: int = MIN_DISPLAY_USABLE) -> Optional[Path]:
    """General experiment: fixed-delta coverage heatmap with aggregate usability mask."""
    global plt
    plt = ensure_matplotlib()

    rbc = summary[(summary["method"] == "RBC MSE") & (summary["in_general"] == True)].copy()
    if rbc.empty:
        return None

    n_order = [int(n) for n in general_n_grid if int(n) in set(pd.to_numeric(rbc["n"], errors="coerce").dropna().astype(int))]
    delta_order = [0.0] + [float(d) for d in general_fixed_deltas]
    if not n_order:
        return None

    coverage = np.full((len(n_order), len(delta_order)), np.nan)
    usability = np.full_like(coverage, np.nan)
    n_usable = np.full_like(coverage, np.nan)

    for i, n in enumerate(n_order):
        row = _find_summary_row(rbc, n, "continuous")
        if row is not None:
            coverage[i, 0] = finite_or_nan(row.get("coverage_conditional_on_minimal_usable", row.get("coverage")))
            usability[i, 0] = finite_or_nan(row.get("minimal_usability_rate"))
            n_usable[i, 0] = finite_or_nan(row.get("n_minimal_usable"))
        for j, d in enumerate(delta_order[1:], start=1):
            row = _find_summary_row(rbc, n, f"fixed_delta_{delta_label(d)}", d)
            if row is not None:
                coverage[i, j] = finite_or_nan(row.get("coverage_conditional_on_minimal_usable", row.get("coverage")))
                usability[i, j] = finite_or_nan(row.get("minimal_usability_rate"))
                n_usable[i, j] = finite_or_nan(row.get("n_minimal_usable"))

    display_ok = (usability >= float(usability_threshold)) & (n_usable >= int(min_display_usable))
    shown = np.where(display_ok, coverage, np.nan)

    x_grid = np.arange(len(delta_order), dtype=float)
    path_x, path_y = [], []
    for i, n in enumerate(n_order):
        d = shrinking_delta(n)
        if d <= max(delta_order) + 1e-12:
            path_x.append(float(np.interp(d, delta_order, x_grid)))
            path_y.append(float(i))

    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.12, top=0.79)

    cmap = coverage_heatmap_cmap()
    cmap.set_bad("#D9D9D9")
    im = ax.imshow(np.ma.masked_invalid(shown), aspect="auto", cmap=cmap, vmin=0.90, vmax=0.95)

    fig.suptitle("RBC coverage over the fixed-delta grid", fontsize=19, y=0.965)
    ax.set_xlabel("Fixed delta; 0 = continuous", fontsize=12)
    ax.set_ylabel("Sample size n", fontsize=12)
    ax.set_xticks(np.arange(len(delta_order)))
    ax.set_xticklabels(["0"] + [delta_label(d) for d in delta_order[1:]], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(n_order)))
    ax.set_yticklabels([f"{n:,}" for n in n_order])

    if path_x:
        ax.plot(path_x, path_y, color="#0096FF", marker="o", markersize=11,
                markerfacecolor="white", markeredgecolor="#0096FF",
                markeredgewidth=2.4, linewidth=3.0, linestyle=(0, (5, 3)),
                zorder=4,
                label=r"shrinking-grid location, $\delta_n=0.05\sqrt{500/n}$")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.885),
                   frameon=False, fontsize=11, handlelength=3.0)

    for i in range(len(n_order)):
        for j in range(len(delta_order)):
            if np.isfinite(shown[i, j]):
                ax.text(j, i, f"{shown[i, j]:.3f}", ha="center", va="center",
                        fontsize=9, color="#000000", fontweight="bold", zorder=6)
            elif np.isfinite(coverage[i, j]) and np.isfinite(usability[i, j]):
                ax.text(j, i, f"u={percent_label(usability[i, j])}", ha="center",
                        va="center", fontsize=9, color="#000000", zorder=6)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.028)
    cb.set_label("Coverage conditional on usable fits (fixed scale)", fontsize=11)
    cb.set_ticks([0.90, 0.925, 0.95])

    png = plot_dir / "Figure 4.10 revised - RBC Coverage and Aggregate Usability.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png


def plot_figure_4_10b_shrinking_path(summary: pd.DataFrame,
                                     output_dir: Path,
                                     general_n_grid: Sequence[int],
                                     usability_threshold: float = USABILITY_PLOT_THRESHOLD) -> Optional[Path]:
    """General experiment: explicit shrinking-grid coverage and usability path."""
    global plt
    plt = ensure_matplotlib()

    rbc = summary[(summary["method"] == "RBC MSE") & (summary["in_general"] == True)].copy()
    if rbc.empty:
        return None

    rows: List[Dict[str, Any]] = []
    for n in general_n_grid:
        row = _find_summary_row(rbc, int(n), "shrinking_delta")
        if row is None:
            continue
        d = finite_or_nan(row.get("delta"))
        if not math.isfinite(d):
            d = shrinking_delta(int(n))
        rows.append({
            "n": int(n),
            "delta": float(d),
            "coverage": finite_or_nan(row.get("coverage_conditional_on_minimal_usable", row.get("coverage"))),
            "minimal_usability": finite_or_nan(row.get("minimal_usability_rate")),
        })
    data = pd.DataFrame(rows)
    if data.empty:
        return None

    x_pos = np.arange(len(data), dtype=float)
    n_values = data["n"].to_numpy(dtype=int)
    deltas = data["delta"].to_numpy(dtype=float)
    coverage = data["coverage"].to_numpy(dtype=float)
    usability = data["minimal_usability"].to_numpy(dtype=float)
    x_labels = [f"{n:,}\n{delta_label(d)}" for n, d in zip(n_values, deltas)]

    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.20, top=0.80, wspace=0.18)
    fig.suptitle(r"RBC shrinking-grid path, $\delta_n=0.05\sqrt{500/n}$",
                 fontsize=17, y=0.96)

    ax = axes[0]
    ax.plot(x_pos, coverage, color="#0878B7", marker="o", linewidth=2.6, markersize=6.5)
    ax.axhline(0.95, color="#374151", linewidth=1.0, linestyle=(0, (5, 4)))
    ax.text(x_pos[-1] + 0.18, 0.95, "nominal 95%", va="center",
            fontsize=9, color="#374151", clip_on=False)
    finite_cov = coverage[np.isfinite(coverage)]
    if finite_cov.size:
        ymin = min(0.90, float(np.nanmin(finite_cov)) - 0.01)
        ymax = max(0.96, float(np.nanmax(finite_cov)) + 0.01)
        ax.set_ylim(ymin, ymax)
    ax.set_title("A. RBC coverage", loc="left", fontsize=12.5, fontweight="bold")
    ax.set_ylabel("coverage")
    for xp, y in zip(x_pos, coverage):
        if np.isfinite(y):
            ax.text(xp, y + 0.0025, f"{y:.3f}", ha="center", va="bottom",
                    fontsize=8.8, color="#000000", fontweight="bold")

    ax = axes[1]
    ax.plot(x_pos, usability, color="#6B4BA8", marker="o", linewidth=2.6, markersize=6.5)
    ax.axhline(usability_threshold, color="#374151", linewidth=1.0, linestyle=(0, (5, 4)))
    ax.text(x_pos[-1] + 0.18, usability_threshold, f"{usability_threshold:.0%} cutoff",
            va="center", fontsize=9, color="#374151", clip_on=False)
    ax.set_ylim(0.78, 1.02)
    ax.set_title("B. Minimal usability", loc="left", fontsize=12.5, fontweight="bold")
    ax.set_ylabel("rate")
    for xp, y in zip(x_pos, usability):
        if np.isfinite(y):
            ax.text(xp, min(y + 0.006, 1.012), percent_label(y),
                    ha="center", va="bottom", fontsize=8.8,
                    color="#000000", fontweight="bold")

    for ax in axes:
        ax.grid(axis="y", color="#D9DEE7", linewidth=0.75, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=8.7)
        ax.set_xlabel("Sample size n and grid spacing")
        ax.set_xlim(x_pos[0] - 0.35, x_pos[-1] + 0.35)

    png = plot_dir / "Figure 4.10b - RBC Shrinking-Grid Path.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png


def _annotate_selected(ax, x: np.ndarray, y: np.ndarray,
                       selected: Sequence[float], fmt: str,
                       color: str, dy: float = 0.025) -> None:
    for target in selected:
        if len(x) == 0:
            continue
        idx = int(np.argmin(np.abs(x - float(target))))
        if abs(float(x[idx]) - float(target)) > 1e-8 or not math.isfinite(float(y[idx])):
            continue
        ax.text(float(x[idx]), float(y[idx]) + dy, fmt.format(float(y[idx])),
                ha="center", va="bottom", fontsize=8.3, color="#000000", fontweight="bold",
                zorder=8)


def plot_figure_4_11(summary: pd.DataFrame,
                     output_dir: Path,
                     frontier_n: int,
                     frontier_fixed_deltas: Sequence[float],
                     usability_threshold: float = USABILITY_PLOT_THRESHOLD,
                     min_display_usable: int = MIN_DISPLAY_USABLE) -> Optional[Path]:
    """Failure-frontier experiment: coverage, usability, and support at n=500."""
    global plt
    plt = ensure_matplotlib()

    rbc = summary[
        (summary["method"] == "RBC MSE")
        & (pd.to_numeric(summary["n"], errors="coerce") == int(frontier_n))
        & (summary["in_frontier"] == True)
    ].copy()
    if rbc.empty:
        return None
    conventional = summary[
        (summary["method"] == "Conventional MSE")
        & (pd.to_numeric(summary["n"], errors="coerce") == int(frontier_n))
        & (summary["in_frontier"] == True)
    ].copy()

    delta_order = [0.0] + list(normalise_deltas(frontier_fixed_deltas))
    rows = []
    conventional_coverage: List[float] = []
    for d in delta_order:
        if d == 0.0:
            row = _find_summary_row(rbc, frontier_n, "continuous")
            conv_row = _find_summary_row(conventional, frontier_n, "continuous") if not conventional.empty else None
        else:
            row = _find_summary_row(rbc, frontier_n, f"fixed_delta_{delta_label(d)}", d)
            conv_row = _find_summary_row(conventional, frontier_n, f"fixed_delta_{delta_label(d)}", d) if not conventional.empty else None
        if row is None:
            continue
        conventional_coverage.append(
            finite_or_nan(conv_row.get("coverage_conditional_on_minimal_usable", conv_row.get("coverage")))
            if conv_row is not None else np.nan
        )
        min_usable = finite_or_nan(row.get("minimal_usability_rate"))
        n_min_usable = finite_or_nan(row.get("n_minimal_usable"))
        display_obj = row.get("coverage_display_ok", np.nan)
        if pd.isna(display_obj):
            display_ok = bool(
                math.isfinite(min_usable)
                and min_usable >= float(usability_threshold)
                and math.isfinite(n_min_usable)
                and n_min_usable >= int(min_display_usable)
            )
        elif isinstance(display_obj, str):
            display_ok = display_obj.strip().lower() in ("true", "1", "1.0")
        else:
            display_ok = bool(display_obj)
        rows.append({
            "delta": d,
            "coverage_usable": finite_or_nan(row.get("coverage_conditional_on_minimal_usable", row.get("coverage"))),
            "raw_success_rate": finite_or_nan(row.get("raw_success_rate")),
            "minimal_usability_rate": min_usable,
            "display_ok": display_ok,
            "strict_support_rate": finite_or_nan(row.get("frontier_display_ok_rate")),
            "support_score": finite_or_nan(row.get("avg_local_support_score")),
        })
    data = pd.DataFrame(rows).sort_values("delta")
    if data.empty:
        return None

    x = data["delta"].to_numpy(dtype=float)
    cov_usable = data["coverage_usable"].to_numpy(dtype=float)
    conv_cov = np.asarray(conventional_coverage, dtype=float)
    if len(conv_cov) != len(x):
        conv_cov = np.full_like(cov_usable, np.nan)
    raw_success = data["raw_success_rate"].to_numpy(dtype=float)
    minimal_usable = data["minimal_usability_rate"].to_numpy(dtype=float)
    display_ok = data["display_ok"].to_numpy(dtype=bool)
    strict_support = data["strict_support_rate"].to_numpy(dtype=float)
    score = data["support_score"].to_numpy(dtype=float)

    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 10.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.08, 1.0, 0.82], "hspace": 0.37})
    fig.subplots_adjust(left=0.085, right=0.94, bottom=0.115, top=0.88)

    max_delta = max(float(np.nanmax(x)), 0.12)
    x_right = max_delta + 0.003
    bands = [
        (0.02, 0.025, "#FCE6CA", "#F0A24A", 0.025, 0.68),
        (0.06, 0.075, "#DCEFF8", "#4DA3D9", 0.06, 0.58),
        (0.08, x_right, "#E5E7EB", "#9CA3AF", 0.08, 0.42),
    ]
    for ax in axes:
        for lo, hi, fill_color, edge_color, boundary, alpha in bands:
            if hi > 0 and lo < max_delta + 1e-12:
                ax.axvspan(lo, min(hi, x_right), color=fill_color, alpha=alpha, zorder=0)
                ax.axvline(boundary, color=edge_color, linewidth=1.1, alpha=0.9, zorder=1)
        ax.grid(axis="y", color="#D9DEE7", linewidth=0.75, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    # A. Coverage
    ax = axes[0]
    finite_cov = np.isfinite(cov_usable)
    diagnostic = finite_cov & (~display_ok)
    display = finite_cov & display_ok
    ax.plot(x, cov_usable, color="#7CBCE4", linewidth=2.4, alpha=0.72, zorder=2)
    if np.isfinite(conv_cov).any():
        ax.plot(x, conv_cov, color="#6B7280", marker="s", markersize=3.8,
                linewidth=1.8, linestyle=(0, (4, 3)), alpha=0.85, zorder=3,
                label="conventional MSE")
    ax.plot(x, np.where(display, cov_usable, np.nan), color="#0878B7",
            marker="o", linewidth=2.6, markersize=6.2, zorder=4,
            label="RBC conditional on minimal usable")
    if diagnostic.any():
        ax.scatter(x[diagnostic], cov_usable[diagnostic], facecolors="white",
                   edgecolors="#0878B7", linewidths=1.6, s=42, zorder=5,
                   label="diagnostic coverage point")
    ax.axhline(0.95, color="#374151", linewidth=1.0, linestyle=(0, (5, 4)))
    ax.text(max_delta + 0.004, 0.95, "nominal 95%", va="center", fontsize=9, color="#374151", clip_on=False)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("coverage")
    ax.set_title("A. Coverage", loc="left", fontsize=12.5, fontweight="bold")
    ax.text(0.0225, 0.07, "support\nwarning", ha="center", va="bottom",
            fontsize=7.8, color="#7C4A12")
    ax.text(0.0675, 0.07, "coverage\ndeterioration", ha="center", va="bottom",
            fontsize=7.8, color="#1F5F85")
    ax.text(0.10, 0.07, "operational\nfailure", ha="center", va="bottom",
            fontsize=7.8, color="#4B5563")
    _annotate_selected(ax, x, cov_usable, [0.02, 0.05, 0.06, 0.075, 0.08, 0.12],
                       "{:.3f}", "#0878B7", dy=0.035)
    if diagnostic.any():
        first_diag = int(np.flatnonzero(diagnostic)[0])
        ax.text(min(float(x[first_diag]) + 0.004, max_delta - 0.012),
                max(0.14, float(cov_usable[first_diag]) - 0.20),
                "diagnostic only\nu < 80%", fontsize=8.7, color="#4B5563",
                ha="left", va="center")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.10), ncol=3, fontsize=9)

    # B. Success and display rates
    ax = axes[1]
    ax.plot(x, raw_success, color="#07966B", marker="o", linewidth=2.2, markersize=5.8, label="raw success")
    ax.plot(x, minimal_usable, color="#6B4BA8", marker="o", linewidth=2.2, markersize=5.8, label="minimal usable")
    ax.plot(x, strict_support, color="#D04A4A", marker="o", linewidth=2.2, markersize=5.8,
            label="strict diagnostic pass rate")
    ax.axhline(usability_threshold, color="#374151", linewidth=1.0, linestyle=(0, (5, 4)))
    ax.text(max_delta + 0.004, usability_threshold, f"{usability_threshold:.0%} display/usability cutoff",
            va="center", fontsize=9, color="#374151", clip_on=False)
    ax.set_ylim(0.0, 1.04)
    ax.set_ylabel("Share of simulations")
    ax.set_title("B. Success and display rates", loc="left", fontsize=12.5, fontweight="bold")
    _annotate_selected(ax, x, strict_support, [0, 0.02, 0.025, 0.03], "{:.2f}", "#D04A4A", dy=0.045)
    _annotate_selected(ax, x, minimal_usable, [0.075, 0.08, 0.12], "{:.2f}", "#6B4BA8", dy=0.045)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.11), ncol=3, fontsize=9)

    # C. Average local support score
    ax = axes[2]
    score_cap = 11.0
    plotted_score = np.minimum(score, score_cap)
    ax.plot(x, plotted_score, color="#D95F02", marker="o", linewidth=2.5, markersize=6.0)
    ax.axhline(5.0, color="#374151", linewidth=1.0, linestyle=(0, (5, 4)))
    ax.axhline(2.0, color="#D04A4A", linewidth=1.0, linestyle=(0, (5, 4)))
    ax.text(max_delta - 0.001, 5.0 + 0.36, "preferred support >= 5", ha="right", fontsize=9, color="#374151")
    ax.text(max_delta - 0.001, 2.0 + 0.36, "local-linear boundary: 2 points", ha="right", fontsize=9, color="#D04A4A")
    ax.set_ylim(0.0, score_cap)
    ax.set_yticks([0, 2, 5, 8, 11])
    ax.set_ylabel("Average local support score")
    ax.set_title("C. Average local support score", loc="left", fontsize=12.5, fontweight="bold")
    if len(score) and math.isfinite(score[0]) and score[0] > score_cap:
        ax.scatter([x[0]], [score_cap], marker="^", s=80, color="#D95F02",
                   zorder=5, clip_on=False)
        ax.text(
            float(x[0]) + 0.004,
            score_cap - 1.35,
            f"continuous S={score[0]:.1f} (off scale)",
            color="#000000", fontsize=8.8, fontweight="bold",
            ha="left", va="top",
        )
    _annotate_selected(ax, x, plotted_score, [0.02, 0.05, 0.075],
                       "{:.3f}", "#D95F02", dy=0.35)

    preferred_ticks = [0.0, 0.02, 0.025, 0.05, 0.06, 0.075, 0.08, 0.10, 0.12]
    ticks = [t for t in preferred_ticks if t <= float(np.nanmax(x)) + 1e-12 and np.any(np.isclose(x, t, atol=1e-12))]
    if not ticks:
        ticks = list(x)
    axes[2].set_xticks(ticks)
    axes[2].set_xticklabels(["0" if t == 0 else delta_label(t) for t in ticks],
                            rotation=35, ha="right")
    axes[2].tick_params(axis="x", labelsize=8.8)
    axes[2].set_xlim(min(-0.001, float(np.nanmin(x)) - 0.002), x_right)
    axes[2].set_xlabel(r"Grid spacing $\delta$")

    fig.suptitle(f"RBC fixed-delta grid at n = {int(frontier_n):,}", fontsize=20, fontweight="bold", y=0.965)

    png = plot_dir / "Figure 4.11 revised - Discrete RBC Failure Frontier.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png


def build_honest_auxiliary_summary(summary: pd.DataFrame,
                                   frontier_n: int,
                                   frontier_fixed_deltas: Sequence[float],
                                   usability_threshold: float = USABILITY_PLOT_THRESHOLD,
                                   min_display_usable: int = MIN_DISPLAY_USABLE) -> pd.DataFrame:
    """Extract the n=frontier_n RBC-vs-bias-aware benchmark used in Figure 4.12."""
    methods = ("RBC MSE", "Bias-aware/Honest")
    delta_order = [0.0] + list(normalise_deltas(frontier_fixed_deltas))
    rows: List[Dict[str, Any]] = []

    for method in methods:
        method_summary = summary[
            (summary["method"] == method)
            & (pd.to_numeric(summary["n"], errors="coerce") == int(frontier_n))
        ].copy()
        if method_summary.empty:
            continue

        for d in delta_order:
            if d == 0.0:
                row = _find_summary_row(method_summary, frontier_n, "continuous")
            else:
                row = _find_summary_row(method_summary, frontier_n, f"fixed_delta_{delta_label(d)}", d)
            if row is None:
                continue

            usability = finite_or_nan(row.get("minimal_usability_rate"))
            n_usable = finite_or_nan(row.get("n_minimal_usable"))
            display_ok = bool(
                math.isfinite(usability)
                and usability >= float(usability_threshold)
                and math.isfinite(n_usable)
                and n_usable >= int(min_display_usable)
            )
            rows.append({
                "n": int(frontier_n),
                "delta": float(d),
                "regime": "continuous" if d == 0.0 else f"fixed_delta_{delta_label(d)}",
                "method": method,
                "coverage": finite_or_nan(row.get("coverage_conditional_on_minimal_usable", row.get("coverage"))),
                "coverage_conditional_on_raw_success": finite_or_nan(row.get("coverage_conditional_on_raw_success")),
                "median_ci_length": finite_or_nan(row.get("median_length")),
                "avg_ci_length": finite_or_nan(row.get("avg_length")),
                "minimal_usability_rate": usability,
                "n_minimal_usable": n_usable,
                "display_ok": display_ok,
                "avg_bias_bound": finite_or_nan(row.get("avg_bias_bound")),
                "raw_success_rate": finite_or_nan(row.get("raw_success_rate")),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["method", "delta"]).reset_index(drop=True)


def plot_figure_4_12(honest_aux: pd.DataFrame,
                     output_dir: Path,
                     usability_threshold: float = USABILITY_PLOT_THRESHOLD) -> Optional[Path]:
    """Auxiliary n=500 check: RBC versus the curvature-bound bias-aware interval."""
    global plt
    plt = ensure_matplotlib()

    if honest_aux.empty:
        return None
    needed = {"method", "delta", "coverage", "median_ci_length", "display_ok"}
    if not needed.issubset(honest_aux.columns):
        return None

    methods = ["RBC MSE", "Bias-aware/Honest"]
    colors = {"RBC MSE": "#0878B7", "Bias-aware/Honest": "#D95F02"}
    labels = {"RBC MSE": "RBC MSE", "Bias-aware/Honest": "bias-aware / Honest benchmark"}

    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.24, top=0.78, wspace=0.18)

    for ax, metric, title, reference in (
        (axes[0], "coverage", "A. Coverage", 0.95),
        (axes[1], "median_ci_length", "B. Median CI length", np.nan),
    ):
        for method in methods:
            g = honest_aux[honest_aux["method"] == method].sort_values("delta").copy()
            if g.empty:
                continue
            x = pd.to_numeric(g["delta"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
            display_ok = g["display_ok"].map(lambda v: bool(v) if isinstance(v, bool) else str(v).lower() in ("true", "1", "1.0")).to_numpy(dtype=bool)
            finite = np.isfinite(x) & np.isfinite(y)
            if not finite.any():
                continue
            ax.plot(x[finite], y[finite], color=colors[method], linewidth=2.2, alpha=0.78, zorder=2)
            shown = finite & display_ok
            diagnostic = finite & (~display_ok)
            if shown.any():
                ax.scatter(x[shown], y[shown], color=colors[method], s=42, zorder=4, label=labels[method])
            if diagnostic.any():
                ax.scatter(x[diagnostic], y[diagnostic], facecolors="white",
                           edgecolors=colors[method], linewidths=1.5, s=42, zorder=5)

        if math.isfinite(reference):
            ax.axhline(reference, color="#6B7280", linewidth=1.0, linestyle=(0, (5, 4)))
            ax.text(0.121, reference, "nominal 95%", va="center", ha="right", fontsize=8.8, color="#4B5563")
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Grid spacing delta")
        ax.grid(axis="y", color="#D9DEE7", linewidth=0.75, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("coverage")
    axes[1].set_ylabel("median CI length")
    axes[0].set_ylim(0.0, 1.02)
    ymin, ymax = axes[1].get_ylim()
    axes[1].set_ylim(max(0.0, ymin), ymax)

    ticks = [0.0, 0.02, 0.05, 0.075, 0.08, 0.10, 0.12]
    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xticklabels(["0" if t == 0 else delta_label(t) for t in ticks], rotation=35, ha="right")
        ax.set_xlim(-0.002, 0.123)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center",
               bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False, fontsize=10.2)
    fig.suptitle("Figure 4.12. Bias-aware / Honest auxiliary check at n = 500",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.085,
        "The bias-aware interval uses the same MSE bandwidth as rdrobust and a DGP curvature bound; "
        "it is an auxiliary benchmark, not a canonical optimal honest RD procedure.",
        ha="center", fontsize=9.4, color="#4B5563",
    )
    fig.text(
        0.5, 0.055,
        f"Hollow points have minimal usability below {usability_threshold:.0%} and are shown only as diagnostics.",
        ha="center", fontsize=9.4, color="#4B5563",
    )

    png = plot_dir / "Figure_4_12_Bias_Aware_Honest_Check_n500.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png


def _rbc_grid_matrix(summary: pd.DataFrame,
                     metric: str,
                     general_n_grid: Sequence[int],
                     general_fixed_deltas: Sequence[float]) -> Tuple[List[int], List[float], np.ndarray]:
    """Return an n-by-delta matrix for RBC general-experiment diagnostics."""
    rbc = summary[(summary["method"] == "RBC MSE") & (summary["in_general"] == True)].copy()
    n_order = [int(n) for n in general_n_grid if int(n) in set(pd.to_numeric(rbc["n"], errors="coerce").dropna().astype(int))]
    delta_order = [0.0] + [float(d) for d in general_fixed_deltas]
    values = np.full((len(n_order), len(delta_order)), np.nan)

    for i, n in enumerate(n_order):
        row = _find_summary_row(rbc, n, "continuous")
        if row is not None:
            values[i, 0] = finite_or_nan(row.get(metric))
        for j, d in enumerate(delta_order[1:], start=1):
            row = _find_summary_row(rbc, n, f"fixed_delta_{delta_label(d)}", d)
            if row is not None:
                values[i, j] = finite_or_nan(row.get(metric))
    return n_order, delta_order, values


def plot_appendix_rbc_usability_heatmap(summary: pd.DataFrame,
                                        output_dir: Path,
                                        general_n_grid: Sequence[int],
                                        general_fixed_deltas: Sequence[float],
                                        usability_threshold: float = USABILITY_PLOT_THRESHOLD) -> Optional[Path]:
    """Old-code diagnostic: where the 80% minimal-usability rule starts to fail."""
    global plt
    plt = ensure_matplotlib()

    n_order, delta_order, values = _rbc_grid_matrix(
        summary, "minimal_usability_rate", general_n_grid, general_fixed_deltas,
    )
    if not n_order or values.size == 0 or not np.isfinite(values).any():
        return None

    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.4, 6.6))
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.14, top=0.88)

    im = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("RBC minimal-usability rate", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Fixed delta; 0 = continuous")
    ax.set_ylabel("Sample size n")
    ax.set_xticks(np.arange(len(delta_order)))
    ax.set_xticklabels(["0"] + [delta_label(d) for d in delta_order[1:]], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(n_order)))
    ax.set_yticklabels([f"{n:,}" for n in n_order])

    for i in range(len(n_order)):
        for j in range(len(delta_order)):
            if np.isfinite(values[i, j]):
                color = "white" if values[i, j] < 0.55 else "#111827"
                ax.text(j, i, percent_label(values[i, j]), ha="center", va="center",
                        fontsize=8.6, color=color, fontweight="bold")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.028)
    cb.set_label("minimal-usability rate")

    png = plot_dir / "Appendix A - RBC Minimal Usability Heatmap.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png


def plot_appendix_local_support_heatmaps(summary: pd.DataFrame,
                                         output_dir: Path,
                                         general_n_grid: Sequence[int],
                                         general_fixed_deltas: Sequence[float]) -> Optional[Path]:
    """Old-code diagnostic: local support inside h and b selected by rdrobust."""
    global plt
    plt = ensure_matplotlib()

    n_order, delta_order, h_vals = _rbc_grid_matrix(
        summary, "avg_min_unique_h", general_n_grid, general_fixed_deltas,
    )
    _, _, b_vals = _rbc_grid_matrix(
        summary, "avg_min_unique_b", general_n_grid, general_fixed_deltas,
    )
    if not n_order or not (np.isfinite(h_vals).any() or np.isfinite(b_vals).any()):
        return None

    cap = 12.0
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.14, top=0.91, wspace=0.08)

    for ax, vals, title in zip(
        axes,
        (h_vals, b_vals),
        ("Distinct support points inside h", "Distinct support points inside b"),
    ):
        shown = np.minimum(vals, cap)
        im = ax.imshow(np.ma.masked_invalid(shown), aspect="auto", cmap="cividis", vmin=0.0, vmax=cap)
        ax.set_title(title, fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Fixed delta; 0 = continuous")
        ax.set_xticks(np.arange(len(delta_order)))
        ax.set_xticklabels(["0"] + [delta_label(d) for d in delta_order[1:]], rotation=35, ha="right")
        for i in range(len(n_order)):
            for j in range(len(delta_order)):
                if np.isfinite(vals[i, j]):
                    label = f">{cap:.0f}" if vals[i, j] > cap else f"{vals[i, j]:.1f}"
                    color = "white" if shown[i, j] < cap * 0.35 else "#111827"
                    ax.text(j, i, label, ha="center", va="center", fontsize=8.0,
                            color=color, fontweight="bold")
    axes[0].set_ylabel("Sample size n")
    axes[0].set_yticks(np.arange(len(n_order)))
    axes[0].set_yticklabels([f"{n:,}" for n in n_order])
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.035, pad=0.025)
    cb.set_label(f"average distinct support points (capped at {cap:.0f})")

    png = plot_dir / "Appendix B - Local Support Heatmaps.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png


def plot_appendix_rbc_path_comparison(summary: pd.DataFrame,
                                      output_dir: Path,
                                      general_n_grid: Sequence[int],
                                      general_fixed_deltas: Sequence[float],
                                      usability_threshold: float = USABILITY_PLOT_THRESHOLD) -> Optional[Path]:
    """Old-code diagnostic: continuous, fixed-grid, and shrinking-grid paths by n."""
    global plt
    plt = ensure_matplotlib()

    rbc = summary[(summary["method"] == "RBC MSE") & (summary["in_general"] == True)].copy()
    if rbc.empty:
        return None

    fixed_delta = 0.05 if any(abs(float(d) - 0.05) < 1e-12 for d in general_fixed_deltas) else max(float(d) for d in general_fixed_deltas)
    n_order = [int(n) for n in general_n_grid if int(n) in set(pd.to_numeric(rbc["n"], errors="coerce").dropna().astype(int))]
    if not n_order:
        return None

    def series_for(regime: str, metric: str) -> List[float]:
        vals: List[float] = []
        for n in n_order:
            if regime == "continuous":
                row = _find_summary_row(rbc, n, "continuous")
            elif regime == "shrinking_delta":
                row = _find_summary_row(rbc, n, "shrinking_delta")
            else:
                row = _find_summary_row(rbc, n, f"fixed_delta_{delta_label(fixed_delta)}", fixed_delta)
            vals.append(finite_or_nan(row.get(metric)) if row is not None else np.nan)
        return vals

    metrics = [
        ("coverage_conditional_on_minimal_usable", "RBC coverage", 0.95),
        ("minimal_usability_rate", "Minimal-usability rate", usability_threshold),
    ]
    colors = {"continuous": "#111827", "shrinking_delta": "#0096FF", "fixed": "#D95F02"}
    labels = {
        "continuous": "continuous benchmark",
        "shrinking_delta": r"shrinking grid, $\delta_n=0.05\sqrt{500/n}$",
        "fixed": fr"fixed grid, $\delta={fixed_delta:g}$",
    }

    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.16, top=0.84, wspace=0.18)

    x = np.arange(len(n_order))
    for ax, (metric, title, reference) in zip(axes, metrics):
        ax.plot(x, series_for("continuous", metric), color=colors["continuous"], marker="o",
                linewidth=2.1, label=labels["continuous"])
        ax.plot(x, series_for("shrinking_delta", metric), color=colors["shrinking_delta"], marker="o",
                linewidth=2.4, linestyle=(0, (5, 3)), label=labels["shrinking_delta"])
        ax.plot(x, series_for("fixed", metric), color=colors["fixed"], marker="o",
                linewidth=2.1, label=labels["fixed"])
        ax.axhline(reference, color="#6B7280", linewidth=1.0, linestyle=(0, (5, 4)))
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold")
        ax.set_ylim(0.0 if "usability" in title else 0.80, 1.02)
        ax.grid(axis="y", color="#D9DEE7", linewidth=0.75, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n:,}" for n in n_order], rotation=35, ha="right")
        ax.set_xlabel("Sample size n")
    axes[0].set_ylabel("rate")

    handles, labels_text = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_text, loc="upper center", bbox_to_anchor=(0.5, 0.985),
               ncol=3, frameon=False, fontsize=10.0)

    png = plot_dir / "Appendix C - RBC Fixed vs Shrinking Path.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png


def build_general_figure_data(summary: pd.DataFrame,
                              general_n_grid: Sequence[int],
                              general_fixed_deltas: Sequence[float]) -> pd.DataFrame:
    allowed_n = {int(n) for n in general_n_grid}
    allowed_d = {round(float(d), 12) for d in general_fixed_deltas}
    out = summary[(summary["method"] == "RBC MSE") & (summary["in_general"] == True)].copy()
    out = out[pd.to_numeric(out["n"], errors="coerce").isin(allowed_n)]
    fixed = out["regime"].astype(str).str.startswith("fixed_delta_")
    keep = out["regime"].astype(str).isin(["continuous", "shrinking_delta"]) | (fixed & pd.to_numeric(out["delta"], errors="coerce").round(12).isin(allowed_d))
    return out[keep].sort_values(["n", "delta_plot", "regime"]).reset_index(drop=True)


def build_frontier_figure_data(summary: pd.DataFrame,
                               frontier_n: int,
                               frontier_fixed_deltas: Sequence[float]) -> pd.DataFrame:
    allowed_d = {round(float(d), 12) for d in frontier_fixed_deltas}
    out = summary[
        (summary["method"] == "RBC MSE")
        & (pd.to_numeric(summary["n"], errors="coerce") == int(frontier_n))
        & (summary["in_frontier"] == True)
    ].copy()
    fixed = out["regime"].astype(str).str.startswith("fixed_delta_")
    keep = out["regime"].astype(str).eq("continuous") | (fixed & pd.to_numeric(out["delta"], errors="coerce").round(12).isin(allowed_d))
    return out[keep].sort_values(["delta_plot", "regime"]).reset_index(drop=True)


# =============================================================================
# 9. Configuration, post-processing, and command-line entry point
# =============================================================================


def run_config_path(outdir: Path) -> Path:
    if outdir.name.lower() == "results":
        return outdir.parent / "run_config_discrete_paired.txt"
    return outdir / "run_config_discrete_paired.txt"


def readme_path(outdir: Path) -> Path:
    if outdir.name.lower() == "results":
        return outdir.parent / "README_discrete_paired.md"
    return outdir / "README_discrete_paired.md"


def write_config(output_dir: Path,
                 experiment: str,
                 reps: int,
                 scenario_plan: Dict[int, List[Dict[str, Any]]],
                 general_n_grid: Sequence[int],
                 general_fixed_deltas: Sequence[float],
                 frontier_n: int,
                 frontier_fixed_deltas: Sequence[float],
                 stress_deltas: Sequence[float],
                 seed: int,
                 length_cutoff: float,
                 honest_length_cutoff: float,
                 usability_threshold: float,
                 min_display_usable: int) -> Path:
    output_dir_label = "results" if output_dir.name.lower() == "results" else str(output_dir)
    figure_output_dir_label = "results/figures" if output_dir_label == "results" else str(output_dir / "figures")

    lines = [
        "CCT DGP2 discrete running-variable unified config",
        f"script_file={Path(__file__).name}",
        f"experiment={experiment}",
        f"reps={int(reps)}",
        f"seed={int(seed)}",
        f"general_n_grid={list(map(int, general_n_grid))}",
        f"general_fixed_deltas={list(map(float, general_fixed_deltas))}",
        f"frontier_n={int(frontier_n)}",
        f"frontier_fixed_deltas={list(map(float, frontier_fixed_deltas))}",
        f"stress_deltas={list(map(float, stress_deltas))}",
        f"scenario_counts_by_n={{{', '.join(f'{n}: {len(v)}' for n, v in scenario_plan.items())}}}",
        "paired_design=nested common random numbers across n and paired latent X/epsilon across scenarios",
        "shared_cells=simulated once and tagged as design_scope=both",
        f"shrinking_delta={SHRINK_BASE_DELTA} * ({SHRINK_BASE_N}/n)^{SHRINK_POWER}",
        f"true_tau={TRUE_TAU}",
        f"error_sd={ERROR_SD}",
        f"p={P}", f"q={Q}", f"kernel={KERNEL}", f"bwselect={BWSELECT}",
        f"vce={VCE}", f"masspoints={MASSPOINTS}", f"bwcheck={BWCHECK}",
        f"methods={list(ALL_METHODS)}",
        f"length_pathology_cutoff={float(length_cutoff)}",
        f"honest_length_pathology_cutoff={float(honest_length_cutoff)}",
        f"usability_threshold={float(usability_threshold)}",
        f"min_display_usable={int(min_display_usable)}",
        f"output_dir={output_dir_label}",
        "honest_auxiliary_method=same MSE bandwidth, DGP curvature-bound bias-aware benchmark; not canonical optimal honest RD",
        f"frontier_strict_rule=raw success; obs_h per side >= {FRONTIER_MIN_OBS_H}; min_unique_h >= {FRONTIER_DISPLAY_MIN_UNIQUE_H}; min_unique_b >= {FRONTIER_DISPLAY_MIN_UNIQUE_B}",
        "figures_png=Figure 4.10 revised - RBC Coverage and Aggregate Usability; "
        "Figure 4.10b - RBC Shrinking-Grid Path; "
        "Figure 4.11 revised - Discrete RBC Failure Frontier; "
        "Figure_4_12_Bias_Aware_Honest_Check_n500; "
        "Appendix A - RBC Minimal Usability Heatmap; "
        "Appendix B - Local Support Heatmaps; "
        "Appendix C - RBC Fixed vs Shrinking Path",
        f"figure_output_dir={figure_output_dir_label}",
    ]
    path = run_config_path(output_dir)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_readme(output_dir: Path,
                 experiment: str,
                 reps: int,
                 general_n_grid: Sequence[int],
                 general_fixed_deltas: Sequence[float],
                 frontier_n: int,
                 frontier_fixed_deltas: Sequence[float]) -> Path:
    script_name = Path(__file__).name
    output_dir_label = "results" if output_dir.name.lower() == "results" else str(output_dir)
    text = f"""# Unified Discrete Running-Variable Extension

This script combines the general discrete experiment and the n={frontier_n} failure-frontier experiment under one simulation engine. Shared n={frontier_n} cells use exactly the same Monte Carlo draws and are evaluated only once.

## Main run

```bash
python {script_name}
```

The default run uses `{reps}` replications, general sample sizes `{list(general_n_grid)}`, general fixed deltas `{list(general_fixed_deltas)}`, and frontier deltas `{list(frontier_fixed_deltas)}`.

Current output directory: `{output_dir_label}`.

Run one component only:

```bash
python {script_name} --experiment general
python {script_name} --experiment frontier
```

Rebuild outputs without rerunning Monte Carlo:

```bash
python {script_name} --raw-input results/discrete_mechanism_raw.csv
```

## Figure outputs

1. `figures/Figure 4.10 revised - RBC Coverage and Aggregate Usability.png`
2. `figures/Figure 4.10b - RBC Shrinking-Grid Path.png`
3. `figures/Figure 4.11 revised - Discrete RBC Failure Frontier.png`
4. `figures/Figure_4_12_Bias_Aware_Honest_Check_n500.png`
5. `figures/Appendix A - RBC Minimal Usability Heatmap.png`
6. `figures/Appendix B - Local Support Heatmaps.png`
7. `figures/Appendix C - RBC Fixed vs Shrinking Path.png`

Only PNG figures are generated by default; PDF export is intentionally disabled. Figure 4.10, Figure 4.10b, and Figure 4.11 form the main discrete-support display. Figure 4.12 is an auxiliary bias-aware/Honest check, and the appendix figures are compact diagnostics adapted from the older scripts.

The bias-aware/Honest interval uses the same MSE bandwidth as `rdrobust` and a DGP curvature-bound bias adjustment. It is included as a benchmark only, not as a canonical optimal honest RD procedure. Its dedicated CSV is `discrete_honest_auxiliary_summary.csv`.

CSV outputs are written in the selected output directory, while all figures are written under its `figures/` subfolder.

Current experiment selector: `{experiment}`.
"""
    path = readme_path(output_dir)
    path.write_text(text, encoding="utf-8")
    return path


def cli_option_present(option: str) -> bool:
    prefix = option + "="
    return any(arg == option or arg.startswith(prefix) for arg in sys.argv[1:])


def infer_design_from_raw(raw: pd.DataFrame) -> Tuple[Tuple[int, ...], Tuple[float, ...], int, Tuple[float, ...]]:
    n_vals = tuple(sorted(int(v) for v in pd.to_numeric(raw.get("n"), errors="coerce").dropna().unique())) if "n" in raw else DEFAULT_GENERAL_N_GRID
    fixed = raw[raw.get("regime", pd.Series("", index=raw.index)).astype(str).str.startswith("fixed_delta_")].copy()
    fixed_vals = tuple(sorted(float(v) for v in pd.to_numeric(fixed.get("delta"), errors="coerce").dropna().unique())) if not fixed.empty else tuple()
    general_deltas = tuple(v for v in fixed_vals if v <= 0.05 + 1e-12) or DEFAULT_GENERAL_FIXED_DELTAS
    frontier_n = DEFAULT_FRONTIER_N if DEFAULT_FRONTIER_N in n_vals else (min(n_vals) if n_vals else DEFAULT_FRONTIER_N)
    frontier_fixed = fixed[pd.to_numeric(fixed.get("n"), errors="coerce") == int(frontier_n)] if not fixed.empty else fixed
    frontier_deltas = tuple(sorted(float(v) for v in pd.to_numeric(frontier_fixed.get("delta"), errors="coerce").dropna().unique())) if not frontier_fixed.empty else DEFAULT_FRONTIER_FIXED_DELTAS
    # Multiple n values indicate the general experiment. For a frontier-only raw
    # file, keep the observed n instead of inventing missing general rows.
    general_n = n_vals if len(n_vals) > 1 or ("regime" in raw and raw["regime"].astype(str).eq("shrinking_delta").any()) else tuple(n_vals)
    return general_n or DEFAULT_GENERAL_N_GRID, general_deltas, frontier_n, frontier_deltas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified CCT DGP2 discrete and failure-frontier Monte Carlo extension.")
    parser.add_argument("--experiment", choices=["all", "general", "frontier"], default=DEFAULT_EXPERIMENT)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--general-n-grid", "--n-grid", dest="general_n_grid", type=int, nargs="+", default=list(DEFAULT_GENERAL_N_GRID))
    parser.add_argument("--general-fixed-deltas", "--fixed-deltas", dest="general_fixed_deltas", type=float, nargs="+", default=list(DEFAULT_GENERAL_FIXED_DELTAS))
    parser.add_argument("--frontier-n", type=int, default=DEFAULT_FRONTIER_N)
    parser.add_argument("--frontier-deltas", type=float, nargs="+", default=list(DEFAULT_FRONTIER_FIXED_DELTAS))
    parser.add_argument("--stress-deltas", type=float, nargs="*", default=list(DEFAULT_STRESS_DELTAS))
    parser.add_argument("--include-stress", action="store_true", help="Add stress deltas at frontier_n only.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--outdir", "--output-dir", dest="outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--chunk-reps", type=int, default=25)
    parser.add_argument("--length-pathology-cutoff", type=float, default=LENGTH_PATHOLOGY_CUTOFF)
    parser.add_argument("--honest-length-pathology-cutoff", type=float, default=HONEST_LENGTH_PATHOLOGY_CUTOFF)
    parser.add_argument("--usability-threshold", type=float, default=USABILITY_PLOT_THRESHOLD)
    parser.add_argument("--min-display-usable", type=int, default=MIN_DISPLAY_USABLE)
    parser.add_argument("--raw-input", type=str, default="", help="Existing raw CSV to post-process without rerunning Monte Carlo.")
    parser.add_argument("--save-full-raw", action="store_true")
    parser.add_argument("--overwrite-raw", action="store_true")
    parser.add_argument("--resume-raw", action="store_true")
    parser.add_argument("--start-rep", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-shrinking", action="store_true")
    parser.add_argument("--show-rdrobust-output", action="store_true")
    return parser.parse_args()


def postprocess_outputs(raw: pd.DataFrame,
                        output_dir: Path,
                        general_n_grid: Sequence[int],
                        general_fixed_deltas: Sequence[float],
                        frontier_n: int,
                        frontier_fixed_deltas: Sequence[float],
                        usability_threshold: float,
                        min_display_usable: int,
                        no_plots: bool) -> None:
    raw = ensure_experiment_flags(raw, general_n_grid, general_fixed_deltas, frontier_n, frontier_fixed_deltas)
    raw = ensure_frontier_diagnostics(raw)
    summary = summarize_results(raw)
    summary = add_coverage_display_ok(summary, usability_threshold, min_display_usable)
    diag = diagnostic_summary(raw)
    pair_methods = paired_method_comparisons(raw)
    pair_scenarios = paired_scenario_comparisons(raw)
    fail_counts = failure_reason_counts(raw)
    support_frontier = summarize_by_support_frontier(raw)
    frontier_fail_counts = frontier_failure_reason_counts(raw)
    interpretation = make_interpretation_table(summary, support_frontier)
    general_data = build_general_figure_data(summary, general_n_grid, general_fixed_deltas)
    frontier_data = build_frontier_figure_data(summary, frontier_n, frontier_fixed_deltas)
    honest_aux = build_honest_auxiliary_summary(
        summary, frontier_n, frontier_fixed_deltas, usability_threshold, min_display_usable,
    )
    breakdown = breakdown_frontier(summary, general_fixed_deltas, usability_threshold, min_display_usable)

    outputs = {
        "discrete_mechanism_summary.csv": summary,
        "discrete_mechanism_diagnostics.csv": diag,
        "discrete_mechanism_paired_method_comparisons.csv": pair_methods,
        "discrete_mechanism_paired_scenario_comparisons.csv": pair_scenarios,
        "discrete_mechanism_failure_reason_counts.csv": fail_counts,
        "discrete_mechanism_breakdown_frontier.csv": breakdown,
        "discrete_failure_frontier_by_support.csv": support_frontier,
        "discrete_frontier_failure_reason_counts.csv": frontier_fail_counts,
        "discrete_frontier_interpretation_table.csv": interpretation,
        "discrete_general_figure_data.csv": general_data,
        "discrete_frontier_figure_data.csv": frontier_data,
        "discrete_honest_auxiliary_summary.csv": honest_aux,
    }
    for name, df in outputs.items():
        path = output_dir / name
        df.to_csv(path, index=False)
        print(f"saved {path}")

    if not no_plots:
        fig10 = plot_figure_4_10(
            summary, output_dir, general_n_grid, general_fixed_deltas,
            usability_threshold, min_display_usable,
        )
        fig10b = plot_figure_4_10b_shrinking_path(
            summary, output_dir, general_n_grid, usability_threshold,
        )
        fig11 = plot_figure_4_11(
            summary, output_dir, frontier_n, frontier_fixed_deltas,
            usability_threshold, min_display_usable,
        )
        fig12 = plot_figure_4_12(honest_aux, output_dir, usability_threshold)
        appendix_figs = [
            plot_appendix_rbc_usability_heatmap(
                summary, output_dir, general_n_grid, general_fixed_deltas,
                usability_threshold,
            ),
            plot_appendix_local_support_heatmaps(
                summary, output_dir, general_n_grid, general_fixed_deltas,
            ),
            plot_appendix_rbc_path_comparison(
                summary, output_dir, general_n_grid, general_fixed_deltas,
                usability_threshold,
            ),
        ]
        if fig10 is not None:
            print(f"saved {fig10}")
        if fig10b is not None:
            print(f"saved {fig10b}")
        if fig11 is not None:
            print(f"saved {fig11}")
        if fig12 is not None:
            print(f"saved {fig12}")
        for fig in appendix_figs:
            if fig is not None:
                print(f"saved {fig}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.raw_input and args.resume_raw:
        raise ValueError("--raw-input and --resume-raw cannot be used together")

    general_n_grid = tuple(int(n) for n in args.general_n_grid)
    general_fixed_deltas = normalise_deltas(args.general_fixed_deltas)
    frontier_n = int(args.frontier_n)
    frontier_fixed_deltas = normalise_deltas(args.frontier_deltas)
    stress_deltas = normalise_deltas(args.stress_deltas) if args.include_stress and args.stress_deltas else tuple()

    raw: Optional[pd.DataFrame] = None
    if args.raw_input:
        raw_path = Path(args.raw_input)
        if not raw_path.exists():
            raise FileNotFoundError(f"raw input not found: {raw_path}")
        raw = pd.read_csv(raw_path, compression="infer")
        inferred_general_n, inferred_general_d, inferred_frontier_n, inferred_frontier_d = infer_design_from_raw(raw)
        if not cli_option_present("--general-n-grid") and not cli_option_present("--n-grid"):
            general_n_grid = inferred_general_n
        if not cli_option_present("--general-fixed-deltas") and not cli_option_present("--fixed-deltas"):
            general_fixed_deltas = inferred_general_d
        if not cli_option_present("--frontier-n"):
            frontier_n = inferred_frontier_n
        if not cli_option_present("--frontier-deltas"):
            frontier_fixed_deltas = inferred_frontier_d
        reps_for_config = int(raw["rep"].nunique()) if "rep" in raw.columns else args.reps
        print(f"loaded raw input from {raw_path}")
    else:
        if rdrobust is None:
            raise RuntimeError(
                f"rdrobust import failed: {_RDROBUST_IMPORT_ERROR}. "
                "Install rdrobust to run the simulation, or use --raw-input to post-process existing raw results."
            )
        reps_for_config = args.reps

    scenario_plan = build_scenario_plan(
        experiment=args.experiment,
        general_n_grid=general_n_grid,
        general_fixed_deltas=general_fixed_deltas,
        frontier_n=frontier_n,
        frontier_fixed_deltas=frontier_fixed_deltas,
        stress_deltas=stress_deltas,
        include_shrinking=not args.no_shrinking,
    )
    frontier_all = normalise_deltas(tuple(frontier_fixed_deltas) + tuple(stress_deltas))

    config_path = write_config(
        output_dir, args.experiment, reps_for_config, scenario_plan,
        general_n_grid, general_fixed_deltas, frontier_n, frontier_fixed_deltas,
        stress_deltas, args.seed, args.length_pathology_cutoff,
        args.honest_length_pathology_cutoff, args.usability_threshold,
        args.min_display_usable,
    )
    module_readme = write_readme(
        output_dir, args.experiment, reps_for_config, general_n_grid,
        general_fixed_deltas, frontier_n, frontier_fixed_deltas,
    )

    if raw is None:
        raw = run_simulation(
            reps=args.reps,
            scenario_plan=scenario_plan,
            seed=args.seed,
            output_dir=output_dir,
            print_every=args.print_every,
            silence_rdrobust=not args.show_rdrobust_output,
            length_cutoff=args.length_pathology_cutoff,
            honest_length_cutoff=args.honest_length_pathology_cutoff,
            save_full_raw=args.save_full_raw,
            chunk_reps=args.chunk_reps,
            overwrite_raw=args.overwrite_raw,
            resume_raw=args.resume_raw,
            start_rep=args.start_rep,
        )

    postprocess_outputs(
        raw=raw,
        output_dir=output_dir,
        general_n_grid=general_n_grid,
        general_fixed_deltas=general_fixed_deltas,
        frontier_n=frontier_n,
        frontier_fixed_deltas=frontier_all,
        usability_threshold=args.usability_threshold,
        min_display_usable=args.min_display_usable,
        no_plots=args.no_plots,
    )
    print(f"saved run config to {config_path}")
    print(f"saved README to     {module_readme}")


if __name__ == "__main__":
    main()
