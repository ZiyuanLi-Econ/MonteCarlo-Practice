$ErrorActionPreference = 'Stop'

$CodeRoot = Split-Path -Parent $PSScriptRoot

& python (Join-Path $CodeRoot "1 Baseline/code/cct_1_baseline.py") --reps 100 --n 500
& python (Join-Path $CodeRoot "2.1 Heteroskedasticity/code/cct_2_1_heteroskedasticity_paired.py") --reps 100
& python (Join-Path $CodeRoot "2.2 Nonnormal/code/cct_2_2_nonnormal_paired.py") --reps 100
& python (Join-Path $CodeRoot "2.3 Discrete/code/cct_2_3_discrete_paired.py") --reps 20 --general-n-grid 500 1000
