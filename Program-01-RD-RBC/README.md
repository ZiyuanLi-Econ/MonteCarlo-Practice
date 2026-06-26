# Program 1: RD Robust Bias-Correction

Portfolio-style Monte Carlo program in applied econometrics. Program 1 studies robust bias-corrected inference in sharp regression discontinuity designs, based on the seminar project on Calonico, Cattaneo, and Titiunik style RD inference.

**TL;DR.** This program asks when robust bias correction (RBC) improves regression discontinuity confidence intervals, and when finite-sample complications such as heteroskedasticity, non-normal errors, and discrete running variables make inference fragile.

**Paper.** [PDF](Paper/Robust%20Bias-Corrected%20Inference%20in%20RDDs.pdf)

## Structure

```text
Code/
  1 Baseline/
  2.1 Heteroskedasticity/
  2.2 Nonnormal/
  2.3 Discrete/
  scripts/
  requirements.txt
  DATA_NOTES.md

Folie/

Paper/
```

## Key findings

- **Baseline design:** conventional MSE confidence intervals undercover in the baseline RD design, with empirical coverage of about 85.2%. Robust bias correction raises coverage to about 93.0% with only a moderate increase in average interval length.
- **Heteroskedasticity:** RBC improves coverage by roughly 7-8 percentage points relative to conventional robust-SE intervals across the heteroskedastic designs. The improvement is meaningful, but the side-specific and near-cutoff volatility designs still fall short of nominal 95% coverage.
- **Non-normal errors:** RBC remains comparatively stable under severe skewness, heavy tails, and one-sided contamination. Across the four non-normal scenarios, RBC coverage is about 93.0%-95.9%, compared with about 85.2%-87.4% for conventional intervals.
- **Discrete running variables:** fine discretization can preserve much of the continuous-design RBC performance, but coarser support quickly becomes a local-support problem. At `n = 500`, fixed grids around `delta = 0.08` and above show sharp drops in RBC usability and coverage as the number of support points inside the local window becomes too small.

The detailed evidence is in the saved summary tables and figures under each experiment's `results/` folder.

## Experiment map

| Experiment | Question | Main script | Main outputs |
| --- | --- | --- | --- |
| Baseline | Does RBC repair undercoverage in the homoskedastic CCT RD design? | [`cct_1_baseline.py`](Code/1%20Baseline/code/cct_1_baseline.py) | [`baseline_summary.csv`](Code/1%20Baseline/results/baseline_summary.csv), [`coverage_bar.png`](Code/1%20Baseline/results/figures/coverage_bar.png) |
| Heteroskedasticity | Does RBC still help when errors are heteroskedastic and robust SEs are already used? | [`cct_2_1_heteroskedasticity_paired.py`](Code/2.1%20Heteroskedasticity/code/cct_2_1_heteroskedasticity_paired.py) | [`hetero_interpretation_table.csv`](Code/2.1%20Heteroskedasticity/results/hetero_interpretation_table.csv), [`fig_hetero_coverage_main.png`](Code/2.1%20Heteroskedasticity/results/figures/fig_hetero_coverage_main.png) |
| Non-normal errors | How stable is RBC under skewness, heavy tails, and one-sided contamination? | [`cct_2_2_nonnormal_paired.py`](Code/2.2%20Nonnormal/code/cct_2_2_nonnormal_paired.py) | [`nonnormal_summary.csv`](Code/2.2%20Nonnormal/results/nonnormal_summary.csv), [`coverage_by_scenario_method.png`](Code/2.2%20Nonnormal/results/figures/coverage_by_scenario_method.png) |
| Discrete running variable | When does discrete support turn RBC into a local-support problem? | [`cct_2_3_discrete_paired.py`](Code/2.3%20Discrete/code/cct_2_3_discrete_paired.py) | [`discrete_frontier_interpretation_table.csv`](Code/2.3%20Discrete/results/discrete_frontier_interpretation_table.csv), [`Figure 4.11`](Code/2.3%20Discrete/results/figures/Figure%204.11%20revised%20-%20Discrete%20RBC%20Failure%20Frontier.png) |

## Key figures

Baseline coverage improvement:

![Baseline coverage](Code/1%20Baseline/results/figures/coverage_bar.png)

Heteroskedasticity coverage comparison:

![Heteroskedasticity coverage](Code/2.1%20Heteroskedasticity/results/figures/fig_hetero_coverage_main.png)

Discrete-running-variable failure frontier:

![Discrete RBC failure frontier](Code/2.3%20Discrete/results/figures/Figure%204.11%20revised%20-%20Discrete%20RBC%20Failure%20Frontier.png)

## Setup

Use Python 3.10+ if possible.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r Code/requirements.txt
```

The scripts depend on the Python `rdrobust` package. If `rdrobust` is not available in your Python environment, install it before running the experiments.

## Run quick checks

```powershell
powershell -ExecutionPolicy Bypass -File "Code/scripts/run_quick_checks.ps1"
```

Or run the scripts directly:

```bash
python "Code/1 Baseline/code/cct_1_baseline.py" --reps 100 --n 500
python "Code/2.1 Heteroskedasticity/code/cct_2_1_heteroskedasticity_paired.py" --reps 100
python "Code/2.2 Nonnormal/code/cct_2_2_nonnormal_paired.py" --reps 100
python "Code/2.3 Discrete/code/cct_2_3_discrete_paired.py" --reps 20 --general-n-grid 500 1000
```

Full Program 1 seminar runs use larger replication counts and can take substantially longer.

## Data and output policy

Saved summary tables and figure data are suitable for version control. Very large raw simulation outputs are intentionally excluded from GitHub when they exceed normal repository limits. See [`Code/DATA_NOTES.md`](Code/DATA_NOTES.md).

## Notes

Program 1 was initialized from the local seminar folder on June 23, 2026.
