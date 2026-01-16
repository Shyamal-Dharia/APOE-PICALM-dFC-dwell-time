# Genetic Interaction of APOE-ε4 and PICALM rs3851179 is Associated With Dynamic Functional Connectivity in Healthy Middle-Aged Adults at Risk for Alzheimer’s Disease

This repository contains scripts and notebooks for preprocessing and analyzing fMRI data, including temporal analysis of functional connectivity and a machine learning model based on dwell-time features.

## Files

* **`fmriprep_ALL_SUB.sh`**
  A Bash script to preprocess all subjects using fMRIPrep, to be run after topup correction is complete.

* **`topup_script.sh`**
  A Bash script for performing topup correction on fMRI data using FSL’s `topup` tool.

* **`temporal_analysis_final.ipynb`**
  A Jupyter Notebook that performs:

  * Functional connectivity extraction
  * Dwell-time computation
  * Hubness (network hub) analysis

* **`machine_learning_final.ipynb`**
  A Jupyter Notebook implementing a logistic regression model that uses dwell-time features for classification or prediction tasks.

* **`README.md`**
  This file, providing an overview of the repository contents.
