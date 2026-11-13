# ABG x EMP x HMP Microbial Source Tracking

## Overview

Analysis scripts for a reference-database comparison of 16S rRNA gene amplicon profiles from burial soils of the New York African Burial Ground (NYABG, Lower Manhattan; in use from approximately the 1640s to 1797) against the Earth Microbiome Project (EMP) and the Human Microbiome Project (HMP). The pipeline tests whether human-associated compositional signatures can be detected in these soils, how they differ from non-burial control soils, and how they vary with anatomical proximity of the sampled soil to the skeleton.

Dataset (full-depth DADA2 denoising): 72 burial soil samples from 47 burials (cranium-associated 15, other body region 31, soil matrix 26), 6 non-burial control soils from public parks within a 2-mile radius of the excavation site, 17,482 EMP samples from 46 Qiita studies and 4,743 HMP samples; 22,303 samples and 3,602 genus-level taxa after taxonomy harmonization.

Version 1.1.0 accompanies the revised manuscript. It adds the revision analyses (scripts 07_v2, 10b, 11c and 30 to 44), the Slurm workflow used on the NC State Hazel cluster (`slurm/`) and the figure code (`figures/`), and corrects several scripts; see `CHANGELOG.md`.

## Citation

> Clinton CK. Human-associated microbial signatures in 300-year-old burial soils from the New York African Burial Ground: a systematic comparison against the Earth Microbiome Project and Human Microbiome Project. Manuscript under revision (Microbiome).

See `CITATION.cff` for machine-readable citation metadata. The Zenodo concept DOI 10.5281/zenodo.20563368 resolves to the latest archived version.

## Getting started

### Prerequisites

- [Conda](https://docs.conda.io/) or [Mamba](https://mamba.readthedocs.io/)
- Python 3.11 or later, R 4.3 or later
- QIIME 2 (amplicon distribution 2024.10) for denoising and classification (`slurm/05_fulldepth_denoise.sbatch`)

### Environments

```bash
conda env create -f envs/python.yml    # abg-emp-hmp-py
conda env create -f envs/rstats.yml    # abg-emp-hmp-r
# FEAST is not on conda:
Rscript -e 'devtools::install_github("cozygene/FEAST")'
```

### Input data

- NYABG raw reads are held under managed access (see Data availability). The pipeline starts from the full-depth genus-level table merged with the archived EMP and HMP genus-level tables (`results/compositional/merged_L6genus_original.tsv`, produced by `slurm/05_fulldepth_denoise.sbatch`).
- Sample metadata: `data/nyabg/abg_16s_meta.tsv` (NYABG), `data/emp/emp_metadata.tsv`, `data/hmp/hmp_metadata.tsv`.
- All scripts accept `--help` (Python) or `key=value` / `--flag` arguments (R) and default to paths relative to the project root.

## Pipeline

### Core analyses (scripts 00 to 13)

| Script | Description |
|---|---|
| `00_harmonize_taxonomy.py` | Genus-level merge of SILVA 138 (NYABG, EMP) and RDP classifier (HMP) taxonomies |
| `01_build_metadata.py` | Cohort metadata with body-site and environment labels |
| `02_alpha_diversity.py` | Shannon, Simpson (unrarefied genus counts); Kruskal-Wallis and Mann-Whitney U with BH correction |
| `03_beta_ordination.py` | Bray-Curtis PCoA, PERMANOVA, PERMDISP on stratified subsamples (500 EMP, 500 HMP, all NYABG; 5 replicates) |
| `04_aitchison_ordination.py` | CLR-Euclidean (Aitchison) PCoA and PERMANOVA |
| `05_indicator_species.R` | IndVal.g indicator species (500 EMP, 500 HMP, all NYABG; 5% prevalence filter) |
| `06_source_network.py` | Bipartite source networks (presence in at least one sample) |
| `07_feast_source_tracking.R` | FEAST source tracking (overall, stratified, stability) |
| `08_ancombc2.R`, `09_aldex2.R` | ANCOM-BC2 and ALDEx2 differential abundance at genus and family level |
| `10_da_consensus.py` | Cross-method consensus (see 10b for effect orientation) |
| `11_classify_samples.py` | Random forest (200 trees) and one-vs-rest L1-LR (C = 1) trained on EMP (EMPO-3) or HMP (18 subsites) |
| `12_decontamination.R`, `13_decontam_check.py` | Control-soil decontamination of the submitted version; superseded by 39 and 40 |

### Sensitivity and comparative analyses (scripts 14 to 28)

| Script | Description |
|---|---|
| `14_rarefaction_sweep.py`, `15_prevalence_sweep.py` | Depth (500 to 10,000 reads) and prevalence (1 to 25%) sweeps of the network assignment |
| `16_confound_permanova.py` | Body-region PERMANOVA of the submitted version; superseded by 36 |
| `17_pseudoreplication.py`, `18_pseudorep_permanova.R` | Pseudoreplication analyses of the submitted version; 18 superseded by 38 |
| `19` to `25` | Cranium, soil matrix, Staphylococcus, classifier null (`22`), power, depth and within-burial analyses |
| `26` to `28` | Network comparison, FEAST and network integration, burial vs control summaries |

### Revision analyses (v1.1.0)

| Script | Description |
|---|---|
| `07_feast_source_tracking_v2.R` | FEAST with `--overall-only` and `--stop-after-bodysite`; logs the FEAST rarefaction depth |
| `34_make_depth_matched_sources.py` | Reference sources rarefied to 500 reads (primary FEAST) or to depths drawn from the burial distribution |
| `35_filter_metadata.py` | Metadata with EMP human-host studies (550, 2192, 1481, 958) and sterile blanks removed |
| `10b_da_consensus_oriented.py` | ALDEx2 effects re-oriented to the first-named group; consensus with ANCOM-BC2 (q < 0.05, wi.eBH < 0.05) |
| `11c_classifier_cv.py` | Stratified five-fold cross-validation of the reference classifiers |
| `30_permdisp_balanced.py` | PERMDISP and balanced-design PERMANOVA (100 draws of 70 EMP soil and 70 HMP samples) |
| `31_coverage_richness.py` | Coverage-based rarefaction of genus richness to 90% and 95% sample coverage |
| `32_gradient_tests.py` | Jonckheere-Terpstra proximity trend among burial samples (two-sided, 9,999 permutations), pairwise tests, depth tests |
| `33_gradient_glmm.R` | lme4 linear mixed models of HMP attribution (burial random intercept, log10 depth) |
| `36_bodyregion_permanova.R` | Body-region and proximity PERMANOVA among burial samples, permutations restricted within burials |
| `37_gradient_beta_glmm.R` | Beta and zero-inflated beta GLMMs (glmmTMB) of HMP attribution |
| `38_burial_level_aggregation.R` | Burial-level composite profiles: PERMANOVA, PERMDISP and Fisher tests vs controls |
| `39_blank_genus_table.py`, `40_decontam_blank.R` | decontam with the touchdown-PCR water control as the negative set |
| `41_handling_genera.py` | Proximity trend of Staphylococcus (with and without the mock-identical ASV), Corynebacterium, Cutibacterium, Streptococcus |
| `42_human_genera_per_sample.py` | Per-sample detection of 87 reference-defined human-associated genera |
| `43_feast_summaries.py` | Summaries of all FEAST runs, stability replicates and body-site components |
| `44_gradient_variants.sh` | Proximity trend tests for each FEAST configuration |

### Running on Slurm

`slurm/` contains the jobs used for the revision on the NC State Hazel cluster, in run order: `05_fulldepth_denoise` (QIIME 2 import, DADA2 at full depth, SILVA 138 classification, merge with EMP and HMP), `00_prep` (harmonization, metadata), `10_ordination`, `20` to `23` (FEAST), `30_isa_networks_decontam`, `40_da`, `50_classifier`, `60_gradient`, `70` to `72` (contamination sensitivity) and `73_classifier_cv`. `slurm/common.sh` sets the project root and environment paths; edit it for another cluster. Scripts 36 to 44 run locally in a few minutes from the pulled `results/` tree and write to `results/revision/`. Submit from the project root with `bash slurm/submit_all.sh` (or set `ABG_P` to the project root); `ABG_ENVROOT` or `ABG_QENV` locate the conda environments, and `ABG_INPUT` the directory holding the archived merged EMP and HMP genus table used by `05` and `70`.

### Figures

`figures/make_*.py` regenerate Figs. 1 to 8 and Additional files 1 to 6 from `results/` (style module `figures/figstyle_nyabg.py`; Arial). Run from the project root after the analyses; output goes to `figures/out/` (PNG, TIFF, PDF and a JSON file of the numbers shown in each figure). Inputs are listed in `figures/inputs_manifest.md`.

## Data availability

- **NYABG burial and control samples:** raw reads are under managed access, in recognition of the ethical obligations to the descendant community and the stewardship of the National Park Service and the U.S. General Services Administration; requests may be directed to the corresponding author or via https://theasheslab.com/resources.
- **NYABG genus-level data:** `data_release/` holds the genus-level read counts (882 genera x 78 samples, SILVA 138, harmonized to the merged reference table) and per-sample metadata (Additional file 7 of the manuscript).
- **EMP:** Qiita (https://qiita.ucsd.edu/); the 46 studies used are listed in Additional file 11 of the manuscript.
- **HMP:** HMP Data Analysis and Coordinating Center (https://hmpdacc.org/), QIIME Community Profiling tables.

## Repository structure

```
abg_emp_hmp/
├── README.md  CHANGELOG.md  CITATION.cff  CONTRIBUTING.md  LICENSE.md
├── docs/        architecture.md, extending.md, parameters.md
├── envs/        python.yml, rstats.yml
├── scripts/     00 to 44 (see tables above), utils.py, rev_args.R
├── slurm/       Slurm jobs and helpers for the Hazel cluster
├── figures/     figure scripts, style module, input manifest
└── data_release/ NYABG genus-level counts and sample metadata
```

## License

MIT. See `LICENSE.md`.
