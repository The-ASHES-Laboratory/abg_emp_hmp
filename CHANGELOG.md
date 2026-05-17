# Changelog

All notable changes to this project will be documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to semantic versioning.

## [Unreleased]

### Planned

- Companion notebooks rendering the manuscript figures from the script outputs.
- Snakemake wrapper for end-to-end pipeline execution.
- Extension to additional reference cohorts beyond EMP and HMP.

## [1.0] - 2026-02

### Added

- **Core analyses (scripts 00-13):**
  - Taxonomy harmonization across SILVA and Greengenes at genus level (`00_harmonize_taxonomy.py`)
  - Cohort metadata construction with body-site and environment labels (`01_build_metadata.py`)
  - Alpha diversity (Shannon, Simpson, Chao1, observed features) with statistical tests (`02_alpha_diversity.py`)
  - Bray-Curtis PCoA, PERMANOVA, PERMDISP, ANOSIM (`03_beta_ordination.py`)
  - CLR-Euclidean (Aitchison) PCoA and PERMANOVA (`04_aitchison_ordination.py`)
  - IndVal indicator species analysis (`05_indicator_species.R`)
  - IndVal-based source-taxon network construction (`06_source_network.py`)
  - FEAST microbial source tracking, overall and stratified, with stability analysis (`07_feast_source_tracking.R`)
  - ANCOM-BC2 differential abundance at genus and family levels (`08_ancombc2.R`)
  - ALDEx2 differential abundance with CLR effect sizes (`09_aldex2.R`)
  - Cross-method differential abundance consensus (`10_da_consensus.py`)
  - Random Forest and L1-LR body-site classification (`11_classify_samples.py`)
  - Prevalence-based decontamination via `decontam` (`12_decontamination.R`)
  - Decontamination verification (`13_decontam_check.py`)
- **Sensitivity and robustness (scripts 14-25):**
  - Rarefaction depth sensitivity (`14_rarefaction_sweep.py`)
  - Prevalence filter sweep (`15_prevalence_sweep.py`)
  - Body-region confound test via partial PERMANOVA (`16_confound_permanova.py`)
  - Burial-level pseudoreplication correction (`17_pseudoreplication.py`, `18_pseudorep_permanova.R`)
  - Cranial proximity gradient analysis (`19_cranium_analysis.py`)
  - Soil matrix versus body-associated partitioning (`20_soilmatrix_analysis.py`)
  - Staphylococcus removal sensitivity (`21_staphylococcus_check.py`)
  - EMP holdout null distribution (`22_classifier_null.py`)
  - Post-hoc power analysis (`23_power_analysis.py`)
  - Sequencing depth distribution (`24_sequencing_depth.py`)
  - Within-burial variance via ICC and pairwise distances (`25_within_burial_variance.py`)
- **Comparative analyses (scripts 26-28):**
  - Burial vs control network comparison (`26_compare_networks.py`)
  - FEAST and network integration (`27_integrate_feast_network.py`)
  - Comprehensive burial vs control comparison (`28_burial_vs_control.py`)
- Reproducible conda environments for Python and R (`envs/python.yml`, `envs/rstats.yml`).
- Shared utility module (`scripts/utils.py`).
