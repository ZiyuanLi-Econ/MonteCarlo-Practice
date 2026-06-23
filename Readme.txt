Seminar Folder
==============

This folder contains the code and saved outputs for four Monte Carlo experiments.


Folder Structure
----------------

Seminar/
  Readme.txt

  1 Baseline/
    code/
    results/
    run_config.txt

  2.1 Heteroskedasticity/
    code/
    results/
    run_config_hetero_paired.txt

  2.2 Nonnormal/
    code/
    results/
    run_config_nonnormal_paired.txt

  2.3 Discrete/
    code/
    results/
    run_config_discrete_paired.txt
    README_discrete_paired.md




Experiments
-----------

1 Baseline
Script: code/cct_1_baseline.py
This is the baseline CCT sharp RD Monte Carlo design with homoskedastic errors. It compares conventional, undersmoothing, bias-corrected, and RBC confidence intervals.


2.1 Heteroskedasticity
Script: code/cct_2_1_heteroskedasticity_paired.py
This experiment keeps the baseline RD design but changes the conditional error variance. It compares conventional and RBC inference with robust standard errors.


2.2 Nonnormal
Script: code/cct_2_2_nonnormal_paired.py
This experiment keeps the DGP fixed but changes the error distribution. It studies normal errors, skewed errors, heavy tails, and one-sided contamination.


2.3 Discrete
Script: code/cct_2_3_discrete_paired.py
This experiment studies discrete running variables. It includes a general discrete-support experiment and an n=500 failure-frontier experiment.
