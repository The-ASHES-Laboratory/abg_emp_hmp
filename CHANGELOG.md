# Changelog

All notable changes to this project will be documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to semantic versioning.

## [Unreleased]

### Planned

- Companion notebooks rendering the manuscript figures from the script outputs.
- Snakemake wrapper for end-to-end pipeline execution.
- Extension to additional reference cohorts beyond EMP and HMP.

## [1.1.0] - 2026-11-13 (revision of the Microbiome submission)

### Added
- Full-depth workflow on Slurm (`slurm/`): DADA2 re-denoising of all libraries at full read depth (the archived 1.0.x tables used a 30% random read subsample), harmonization, ordination, FEAST, indicator species and networks, differential abundance, classifiers, gradient tests, contamination sensitivity and classifier cross-validation.
- Depth-matched FEAST: reference sources rarefied to 500 reads (`34_make_depth_matched_sources.py`, `07_feast_source_tracking_v2.R`). With no coverage supplied, FEAST rarefies the sink and all sources to the smallest library in the draw (11 reads in the 1.0.x configuration).
- EMP reference-pool sensitivity: human-host studies (550, 2192, 1481, 958) and sterile blanks removed (`35_filter_metadata.py`).
- `10b_da_consensus_oriented.py`: ALDEx2 effects re-oriented to the first-named group before comparison with ANCOM-BC2.
- `11c_classifier_cv.py`: stratified five-fold cross-validation of the reference classifiers.
- `30_permdisp_balanced.py`, `31_coverage_richness.py`, `32_gradient_tests.py`, `33_gradient_glmm.R`.
- `36` to `44`: body-region PERMANOVA, beta and zero-inflated beta GLMMs, burial-level aggregation, blank-based decontamination, handling-associated genera, per-sample human-associated genera, FEAST summaries and gradient tests for each FEAST configuration. These reproduce analyses first run interactively during the revision.
- `figures/`: scripts for Figs. 1 to 8 and Additional files 1 to 6.
- `data_release/`: NYABG genus-level read counts (882 genera x 78 samples; SILVA 138, harmonized to the merged reference table) and per-sample metadata (Additional file 7). Raw reads are not included.

### Fixed
- `10b_da_consensus_oriented.py` selects the ALDEx2 Wilcoxon q value (`wi.eBH`) first, matching `10_da_consensus.py`. Hazel job 1813266 ran an earlier version that selected the Welch q value (`we.eBH`), giving 144 consensus taxa (153 with EMP human-host studies excluded); the released script gives 157 (168), as reported in the manuscript.
- `04_aitchison_ordination.py`: the F column held the statistic name ("pseudo-F"); it now holds the statistic.
- `16_confound_permanova.py`: the R2 column held the pseudo-F statistic; R2 is now computed from F. The script read a malformed metadata file and matched no samples; the default is now `abg_16s_meta_clean.tsv`. Superseded by `36_bodyregion_permanova.R`.
- `31_coverage_richness.py`: the "common" coverage target (minimum observed coverage, 1.0 without singletons) is no longer computed.
- `11c` and the Slurm jobs reference `scripts/` (previously `scripts_new/`).

### Changed
- Documentation: taxonomies are SILVA 138 (NYABG, EMP) and RDP classifier (HMP DACC tables), not Greengenes; classifier and parameter descriptions corrected (`docs/parameters.md`); the study aim is stated as detection of human-associated signatures, not persistence of a human microbiome.

### Deprecated
- `12_decontamination.R` and `13_decontam_check.py` (control soils as the negative set cannot flag taxa that are rare in controls), `16_confound_permanova.py`, `18_pseudorep_permanova.R` (one stratum per reference sample; no permutation possible), `21_staphylococcus_check.py` (genus-level; see 41).

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
