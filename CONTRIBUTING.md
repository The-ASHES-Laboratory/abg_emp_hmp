# Contributing

This repository is part of an active research program in the ASHES Laboratory, NC State University. Contributions are welcome via pull request after the v1.0 release.

## Issues

Use GitHub Issues for:

- Bug reports (with a minimal reproducible example)
- Questions about analysis parameters or reference cohort selection
- Requests to extend the framework to additional reference cohorts beyond EMP and HMP
- Documentation gaps

Tag issues with one of `bug`, `enhancement`, `documentation`, `parameters`, `reference-cohort`.

## Pull requests

1. Fork and create a topic branch named `feature/short-description`.
2. Add or update tests where applicable so analyses remain reproducible.
3. Confirm that the cross-method differential abundance consensus (`10_da_consensus.py`) still produces the manuscript taxa list after your changes.
4. Update `CHANGELOG.md` with a brief entry under `[Unreleased]`.
5. Open a PR against `main`.

## Coding style

- Python 3.11+, type hints where helpful (not enforced).
- Black formatting for Python; default settings.
- Each script is a CLI with `argparse`; defaults reproduce the manuscript figures and tables.
- R scripts accept `--help` via the `argparse` package; follow tidyverse style.
- Random seeds are set at module top; never call `np.random.seed` or `set.seed` inside functions.

## Reproducibility

- All randomized analyses (FEAST stability, classifier null, subsampling for ordination) must be seeded.
- New scripts that depend on conda packages should update the appropriate environment file in `envs/`.
- Do not commit large intermediate files; reference data should be staged via the cohort metadata file.

## Community-engaged research note

This software supports research conducted in collaboration with descendant-community partners at the New York African Burial Ground. Contributions that propose extending the framework to additional descendant lineages should include a brief note on the community-engagement framework that will support the proposed extension.
