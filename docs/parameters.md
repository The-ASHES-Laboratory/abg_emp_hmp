# Key parameters

This document collects the analytical parameters that most influence pipeline output. Defaults reproduce the manuscript results; the sensitivity sweeps in scripts 14-25 establish the ranges over which the headline conclusions are stable.

## Cohort subsampling

| Parameter | Default | Where set | Rationale |
|---|---|---|---|
| Ordination subsample (per cohort) | 500 | `03_beta_ordination.py`, `04_aitchison_ordination.py` | Balances cohort representation while keeping ordination computation tractable. Cohort sample sizes range from 76 (NYABG) to 17,482 (EMP). |
| FEAST source pool size (per source) | 100 | `07_feast_source_tracking.R` | FEAST runtime scales with source pool size; 100 per source preserves statistical power without exhausting memory on a workstation. |
| FEAST stability replicates | 50 | `07_feast_source_tracking.R` | Yields per-source confidence intervals tight enough for inferential use. |

## Rarefaction and prevalence

| Parameter | Default | Where set | Rationale |
|---|---|---|---|
| Rarefaction depth | 5,000 reads/sample | `02_alpha_diversity.py` | Excludes 3 of 76 NYABG samples; sensitivity sweep in `14_rarefaction_sweep.py` covers 1,000-20,000. |
| Prevalence filter | Present in >= 5% of samples in any cohort | `15_prevalence_sweep.py` defines the sweep | Removes low-prevalence genera that contribute noise to source tracking. |

## Statistical thresholds

| Parameter | Default | Where set | Rationale |
|---|---|---|---|
| PERMANOVA permutations | 999 | `03_beta_ordination.py`, `18_pseudorep_permanova.R` | Standard convention; tighter than 99 (which is the QIIME 2 default) without becoming computationally expensive. |
| Differential abundance significance | q < 0.05 (BH-adjusted) | `08_ancombc2.R`, `09_aldex2.R` | Method-specific defaults; consensus filter requires agreement across at least two methods. |
| Consensus threshold | k = 2 of 2 methods | `10_da_consensus.py` | All methods must agree; reduces method-specific false positives. |

## Decontamination

| Parameter | Default | Where set | Rationale |
|---|---|---|---|
| `decontam` prevalence threshold | 0.5 | `12_decontamination.R` | `decontam` author recommendation for prevalence-based decontamination. |
| Decontam verification | Compare human-associated taxa list before and after | `13_decontam_check.py` | Ensures that the source-tracking and differential abundance results are not driven by contaminants flagged at the prevalence threshold. |

## Classification

| Parameter | Default | Where set | Rationale |
|---|---|---|---|
| Random Forest trees | 500 | `11_classify_samples.py` | Standard convention; OOB estimates stabilize by 200-500 trees on this feature space. |
| L1-LR regularization sweep | C in [0.01, 0.1, 1, 10] | `11_classify_samples.py` | Inner-CV-selected; reported metric is held-out NYABG accuracy. |

## Reproducibility flags common to every script

- `--seed` — RNG seed (default 42); guarantees deterministic outputs.
- `--log-file` — destination for the parameter-and-result log file.
- `--threads` — parallelism for FEAST, classification, and ordination.
