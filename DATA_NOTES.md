# Data Notes

This repository is intended to store source code, configuration files, compact summary tables, figure data, and selected rendered figures for the Monte Carlo seminar project.

Large raw simulation files are not always appropriate for normal GitHub storage. In particular, the local file below is intentionally excluded:

```text
2.3 Discrete/results/discrete_mechanism_raw.csv
```

The local copy is approximately 967 MB, which exceeds GitHub's ordinary file-size workflow and should be handled with Git LFS, an external archive, or regeneration from `2.3 Discrete/code/cct_2_3_discrete_paired.py`.

The smaller raw CSVs for the baseline, heteroskedasticity, and nonnormal experiments are also candidates for Git LFS if the repository should stay lightweight. Summary and interpretation tables should remain in Git.
