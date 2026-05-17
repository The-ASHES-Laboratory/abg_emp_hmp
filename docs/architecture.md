# Architecture

This document describes the analytical structure of the `abg_emp_hmp` pipeline.

## Goal

Quantify whether detectable human-associated microbial signatures persist in centuries-old burial soils from the New York African Burial Ground (NYABG), by comparing 70 burial and 6 non-burial control soil samples against two well-characterized reference cohorts: the Earth Microbiome Project (EMP, 17,482 environmental samples) and the Human Microbiome Project (HMP, 4,743 body-site samples).

## Data model

All analyses operate on a single genus-level (L6) feature table merged across the three cohorts (22,301 samples total), exported from QIIME 2 and harmonized across SILVA and Greengenes taxonomy by `00_harmonize_taxonomy.py`.

Cohort metadata is constructed by `01_build_metadata.py` and encodes:

- Cohort (NYABG burial / NYABG control / EMP / HMP)
- For HMP: body site (oral, nasal, skin, gut, urogenital)
- For EMP: environment (soil, plant, water, sediment, etc.)
- For NYABG: burial ID, cranial-proximity stratum, soil matrix indicator

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

Three independent lines of evidence are combined to support the central claim of persistent human-associated signatures in burial soils:

1. **Source tracking (FEAST, script 07):** Quantifies the fractional contribution of HMP body sites and EMP environments to each NYABG sample, with stratified analyses by body site and stability assessment under subsampling.

2. **Differential abundance consensus (scripts 08-10):** ANCOM-BC2 and ALDEx2 are run independently and their results are consensus-filtered (`10_da_consensus.py`) to reduce method-specific false positives.

3. **Classification (script 11):** Random Forest and L1-regularized logistic regression are trained to predict body-site of origin from genus-level composition. Performance on NYABG samples (held out from training) is interpreted as a sanity check on the source-tracking results.

## Robustness battery (scripts 14-25)

Eleven sensitivity analyses are designed to address specific reviewer concerns and to bound the inferential reach of the headline results:

- Rarefaction depth, prevalence filter, and Staphylococcus removal probe the analytical pipeline.
- Body-region confound, pseudoreplication, cranial proximity, and soil matrix probe the study design.
- EMP holdout null distribution, post-hoc power, sequencing depth, and within-burial variance probe statistical inference.

Each sensitivity script writes summary statistics to a results directory so its output can be referenced from the manuscript Supplementary Information without re-running the upstream pipeline.

## Decontamination

Prevalence-based decontamination via `decontam` (script 12) is run on the burial subset only; the `13_decontam_check.py` verifier confirms that the human-associated signals identified upstream are not driven by contaminant taxa flagged at the prevalence threshold.

## Outputs

Every script writes:

- A CSV or TSV of its primary numerical result, named consistently with the script number.
- A log file capturing the parameter values used (seed, alpha, subsampling depth, etc.).
- Optionally, plots in PNG and PDF format under `figures/`.

These outputs are not tracked in the repository.
