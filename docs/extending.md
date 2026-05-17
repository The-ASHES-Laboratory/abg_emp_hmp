# Extending the framework

This document gives entry points for the most common extensions.

## Adding a new reference cohort

The pipeline assumes two reference cohorts (EMP, HMP). To add a third (for example, the Microbiome Quality Control project, an unpublished site-specific reference, or a new public release):

1. Add a cohort label to `01_build_metadata.py` and update the metadata schema.
2. Update the cohort filter logic in `03_beta_ordination.py`, `04_aitchison_ordination.py`, and `07_feast_source_tracking.R`.
3. Add the new cohort to the FEAST source matrix and re-run `07_feast_source_tracking.R`. Confirm that the source fractions still sum to 1.0 within FEAST tolerance.
4. Re-run `10_da_consensus.py`; the consensus filter is cohort-aware and will pick up the new cohort automatically.
5. Document the new cohort and its sample-size implications in `docs/parameters.md`.

## Adding a new differential abundance method

The differential abundance consensus (`10_da_consensus.py`) currently combines ANCOM-BC2 (script 08) and ALDEx2 (script 09). To add a third method (for example, MaAsLin2 or Corncob):

1. Add the new method as a numbered script (`09a_maaslin2.R` or similar). Write its output to a CSV with the same column schema as the existing methods.
2. Register the new CSV in the file list at the top of `10_da_consensus.py`.
3. Update the manuscript Supplementary Methods to describe the new method and the rationale for inclusion.

The consensus rule (presently: a taxon is flagged if it is significant in at least k of n methods) is parameterized via the `--min-methods` flag.

## Changing the rarefaction depth

The default rarefaction depth is set in `02_alpha_diversity.py`. To change it:

1. Update the `--depth` default in the relevant script.
2. Re-run `14_rarefaction_sweep.py` over your new depth range to confirm that the headline results are stable.
3. Document the choice and the stability evidence in the manuscript.

## Adding a new body site stratification

HMP samples are currently stratified by body site (oral, nasal, skin, gut, urogenital). To add a finer-grained stratum (for example, splitting oral into supragingival/subgingival):

1. Update `01_build_metadata.py` to encode the finer stratum.
2. Update `07_feast_source_tracking.R` to use the new stratum as the source label.
3. Re-run the burial-vs-source FEAST analysis and confirm that the new stratum is well-represented (sample size > 30 by default).

## Reproducibility expectations

Every extension should preserve the existing reproducibility guarantees:

- Seeded RNG at the top of each script.
- CLI parameters with documented defaults that reproduce the manuscript results.
- Conda environments updated if new dependencies are introduced.
- New analyses added to the appropriate sensitivity sweep where applicable.

If a proposed extension would break reproducibility of the v1.0 manuscript figures, surface this in the pull request description so reviewers can decide whether to gate it behind a version bump.
