# ABG x EMP x HMP Microbial Source Tracking

## Overview

Analysis scripts for microbial source tracking of centuries-old burial soils from the New York African Burial Ground (NYABG). This pipeline compares NYABG burial microbiomes with Earth Microbiome Project (EMP) and Human Microbiome Project (HMP) reference datasets to assess whether detectable human-associated microbial signatures persist in historic interment contexts.

The dataset comprises 70 burial samples, 6 control (non-burial) soil samples, 17,482 EMP environmental samples, and 4,743 HMP human body-site samples (22,301 total) analyzed at genus level.

## Citation

> Clinton, C.K. et al. (2025). *Manuscript in preparation.*

## Getting Started

### Prerequisites

- [Conda](https://docs.conda.io/) or [Mamba](https://mamba.readthedocs.io/)
- Python 3.11+
- R 4.3+

### Environment Setup

```bash
# Python environment
conda env create -f envs/python.yml
conda activate abg-emp-hmp-py

# R environment
conda env create -f envs/rstats.yml
conda activate abg-emp-hmp-r

# FEAST must be installed manually (not available on conda)
Rscript -e 'devtools::install_github("cozygene/FEAST")'
```

### Input Data

Scripts expect a QIIME2-exported genus-level feature table (`merged_L6genus.tsv`) and associated metadata files. See individual script `--help` flags for required inputs and default paths.

## Pipeline

### Core Analyses (Scripts 00-13)

| Script | Description | Language |
|--------|-------------|----------|
| `00_harmonize_taxonomy.py` | Merge SILVA and Greengenes taxonomy at genus level | Python |
| `01_build_metadata.py` | Build cohort metadata with body-site and environment labels | Python |
| `02_alpha_diversity.py` | Shannon, Simpson, Chao1, observed features with statistical tests | Python |
| `03_beta_ordination.py` | Bray-Curtis PCoA, PERMANOVA, PERMDISP, ANOSIM | Python |
| `04_aitchison_ordination.py` | CLR-Euclidean (Aitchison) PCoA and PERMANOVA | Python |
| `05_indicator_species.R` | IndVal indicator species analysis | R |
| `06_source_network.py` | IndVal-based source-taxon network construction | Python |
| `07_feast_source_tracking.R` | FEAST microbial source tracking (overall, stratified, stability) | R |
| `08_ancombc2.R` | ANCOM-BC2 differential abundance at genus and family levels | R |
| `09_aldex2.R` | ALDEx2 differential abundance | R |
| `10_da_consensus.py` | Cross-method differential abundance consensus | Python |
| `11_classify_samples.py` | Random Forest and L1-LR body-site classification | Python |
| `12_decontamination.R` | Prevalence-based decontamination (decontam) | R |
| `13_decontam_check.py` | Verify human signals are not driven by contaminants | Python |

### Sensitivity and Robustness (Scripts 14-25)

| Script | Description | Language |
|--------|-------------|----------|
| `14_rarefaction_sweep.py` | Rarefaction depth sensitivity analysis | Python |
| `15_prevalence_sweep.py` | Prevalence filter threshold sensitivity | Python |
| `16_confound_permanova.py` | Body-region confound test (partial PERMANOVA) | Python |
| `17_pseudoreplication.py` | Burial-level aggregation for pseudoreplication correction | Python |
| `18_pseudorep_permanova.R` | PERMANOVA with strata (vegan::adonis2) | R |
| `19_cranium_analysis.py` | Cranial proximity gradient analysis | Python |
| `20_soilmatrix_analysis.py` | Soil matrix vs body-associated partitioning | Python |
| `21_staphylococcus_check.py` | Impact of Staphylococcus removal on results | Python |
| `22_classifier_null.py` | Null distribution baseline (EMP holdout) | Python |
| `23_power_analysis.py` | Post-hoc statistical power analysis | Python |
| `24_sequencing_depth.py` | Sequencing depth distribution analysis | Python |
| `25_within_burial_variance.py` | ICC and pairwise distances within burials | Python |

### Comparative Analyses (Scripts 26-28)

| Script | Description | Language |
|--------|-------------|----------|
| `26_compare_networks.py` | Burial vs control network comparison | Python |
| `27_integrate_feast_network.py` | FEAST and network integration analysis | Python |
| `28_burial_vs_control.py` | Comprehensive burial vs control comparison | Python |

## Notes

- All Python scripts accept `--help` for usage details.
- R scripts accept `--help` via the `argparse` package.
- Figures were generated using matplotlib and seaborn (Python) and ggplot2 (R).
- EMP and HMP reference datasets are subsampled with stratification to address sample-size imbalance (default: 500 per cohort for ordination, 100 for FEAST).

## Data Availability

- **NYABG burial and control samples**: Available upon request from the corresponding author.
- **Earth Microbiome Project (EMP)**: [https://earthmicrobiome.org](https://earthmicrobiome.org)
- **Human Microbiome Project (HMP)**: [https://hmpdacc.org](https://hmpdacc.org)

## Repository Structure

```
abg_emp_hmp/
├── README.md
├── LICENSE
├── .gitignore
├── envs/
│   ├── python.yml
│   └── rstats.yml
└── scripts/
    ├── utils.py
    ├── 00_harmonize_taxonomy.py
    ├── 01_build_metadata.py
    ├── 02_alpha_diversity.py
    ├── 03_beta_ordination.py
    ├── 04_aitchison_ordination.py
    ├── 05_indicator_species.R
    ├── 06_source_network.py
    ├── 07_feast_source_tracking.R
    ├── 08_ancombc2.R
    ├── 09_aldex2.R
    ├── 10_da_consensus.py
    ├── 11_classify_samples.py
    ├── 12_decontamination.R
    ├── 13_decontam_check.py
    ├── 14_rarefaction_sweep.py
    ├── 15_prevalence_sweep.py
    ├── 16_confound_permanova.py
    ├── 17_pseudoreplication.py
    ├── 18_pseudorep_permanova.R
    ├── 19_cranium_analysis.py
    ├── 20_soilmatrix_analysis.py
    ├── 21_staphylococcus_check.py
    ├── 22_classifier_null.py
    ├── 23_power_analysis.py
    ├── 24_sequencing_depth.py
    ├── 25_within_burial_variance.py
    ├── 26_compare_networks.py
    ├── 27_integrate_feast_network.py
    └── 28_burial_vs_control.py
```

## License

MIT
