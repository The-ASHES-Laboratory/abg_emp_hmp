# Architecture

This document describes the analytical structure of the `abg_emp_hmp` pipeline.

## Goal

Test whether human-associated microbial signatures can be detected in centuries-old burial soils from the New York African Burial Ground (NYABG), by comparing 72 burial samples (47 burials) and 6 non-burial control soils against two well-characterized reference cohorts: the Earth Microbiome Project (EMP, 17,482 environmental samples) and the Human Microbiome Project (HMP, 4,743 body-site samples).

## Data model

All analyses operate on a single genus-level (L6) feature table merged across the three cohorts (22,303 samples, 3,602 genus-level taxa), exported from QIIME 2 and harmonized across the SILVA 138 (NYABG, EMP) and RDP classifier (HMP) taxonomies by `00_harmonize_taxonomy.py`.

Cohort metadata is constructed by `01_build_metadata.py` and encodes:

- Cohort (NYABG burial / NYABG control / EMP / HMP)
- For HMP: body site (oral, nasal, skin, gut, urogenital)
- For EMP: environment (soil, plant, water, sediment, etc.)
- For NYABG: burial ID and proximity level (cranium-associated, other body region, soil matrix)

## Analytical layers

The pipeline is intentionally layered so that each downstream analysis can be re-run independently of the others as long as the upstream feature table and metadata are stable.

```
┌──────────────────────────────────────────────────────────────┐
│ Layer 1: Harmonization and metadata (scripts 00-01)           │
│   QIIME 2 L6 table  +  SILVA/GG mapping  →  unified table    │
└──────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────────┐
│ Layer 2:       │  │ Layer 3:       │  │ Layer 4:           │
│ Diversity      │  │ Source         │  │ Differential       │
│ and ordination │  │ tracking       │  │ abundance          │
│ (02-04)        │  │ (05-07)        │  │ (08-10)            │
└────────────────┘  └────────────────┘  └────────────────────┘
                                                 │
                                                 ▼
                                  ┌────────────────────────────┐
                                  │ Layer 5: Classification    │
                                  │ and decontamination (11-13)│
                                  └────────────────────────────┘
                                                 │
                                                 ▼
                                  ┌────────────────────────────┐
                                  │ Layer 6: Sensitivity and   │
                                  │ robustness (14-25)         │
                                  └────────────────────────────┘
                                                 │
                                                 ▼
                                  ┌────────────────────────────┐
                                  │ Layer 7: Comparative       │
                                  │ network and integration    │
                                  │ (26-28)                    │
                                  └────────────────────────────┘
```

## Statistical strategy

Three lines of evidence are combined; none authenticates age, and 16S amplicon data cannot separate viable, dormant and relic DNA:

1. **Source tracking (FEAST, scripts 07, 07_v2, 34):** Fractional contribution of EMP and HMP sources to each NYABG sample, with reference sources rarefied to 500 reads, stratified analyses by body site and environment type, five stability replicates, and sensitivity runs (EMP human-host studies excluded, blank- and mock-derived ASVs removed, burial-matched source depths, FEAST default coverage).

2. **Differential abundance consensus (scripts 08-10):** ANCOM-BC2 and ALDEx2 are run independently and their results are consensus-filtered (`10_da_consensus.py`) to reduce method-specific false positives.

3. **Classification (scripts 11, 22, 11c):** Random forest and L1-regularized logistic regression are trained on all EMP (EMPO-3) or HMP (18 subsites) samples and applied to NYABG samples; a five-site HMP model applied to 500 EMP samples gives a null distribution of classifier confidence, and stratified five-fold cross-validation (11c) gives held-out accuracy on the reference data.

## Robustness battery (scripts 14-25)

Sensitivity analyses bound the inferential reach of the headline results (scripts 14 to 25 and 30 to 44):

- Rarefaction depth, prevalence filter, and Staphylococcus removal probe the analytical pipeline.
- Body-region PERMANOVA (36), burial-level aggregation (38), mixed models with a burial random intercept (33, 37), and the proximity trend among burial samples (32, 44) probe the study design.
- EMP holdout null distribution, post-hoc power, sequencing depth, and within-burial variance probe statistical inference.

Each sensitivity script writes summary statistics to a results directory so its output can be referenced from the manuscript Supplementary Information without re-running the upstream pipeline.

## Decontamination

Prevalence-based `decontam` (thresholds 0.1 and 0.5) is run with the touchdown-PCR water control as the negative set (scripts 39 and 40). The contamination sensitivity analysis (`slurm/70` to `72`) removes every ASV detected in the water control or the mock communities and recomputes PERMANOVA, FEAST and the gradient tests. Scripts 12 and 13 (control soils as the negative set) are retained for reference only: taxa that are rare in control soils cannot be flagged by that design.

## Outputs

Every script writes:

- A CSV or TSV of its primary numerical result, named consistently with the script number.
- A log file capturing the parameter values used (seed, alpha, subsampling depth, etc.).
- Optionally, plots in PNG and PDF format under `figures/`.

These outputs are not tracked in the repository.
