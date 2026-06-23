# Project Manifest

Source folder used for this repository:

```text
C:\Users\17964\Desktop\Seminar
```

## Core scripts

```text
1 Baseline/code/cct_1_baseline.py
2.1 Heteroskedasticity/code/cct_2_1_heteroskedasticity_paired.py
2.2 Nonnormal/code/cct_2_2_nonnormal_paired.py
2.3 Discrete/code/cct_2_3_discrete_paired.py
```

## Configuration files

```text
1 Baseline/run_config.txt
2.1 Heteroskedasticity/run_config_hetero_paired.txt
2.2 Nonnormal/run_config_nonnormal_paired.txt
2.3 Discrete/run_config_discrete_paired.txt
```

## Recommended version-control policy

Commit these directly to GitHub:

```text
code/*.py
run_config*.txt
README*.md
summary CSV files
interpretation tables
figure-data CSV files
selected PNG figures
final paper PDF, if desired
```

Do not commit these through ordinary GitHub upload:

```text
Seminar code.zip
2.3 Discrete/results/discrete_mechanism_raw.csv
```

The discrete raw file is around 967 MB locally and needs Git LFS, external storage, or regeneration.
