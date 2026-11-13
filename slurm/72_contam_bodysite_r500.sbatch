#!/bin/bash
#SBATCH --job-name=abg72_contambs
#SBATCH --account=clintonlab
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Body-site-stratified FEAST (depth-matched, 500-read sources) on the blank/mock-ASV-filtered table, then the
# within-burial gradient tests on each body-site component (tests the nasal component against mock-identical
# Staphylococcus). Overall + body-site sections only.
source "${ABG_P:-$SLURM_SUBMIT_DIR}/slurm/common.sh" || { echo "FATAL: cannot source common.sh"; exit 1; }
say "=== abg72_contambs job ${SLURM_JOB_ID:-local} on $(hostname) ==="
C=results/contam_sensitivity
[ -s $C/feast_depth_matched/merged_L6genus_r500.tsv ] || die "missing $C/feast_depth_matched/merged_L6genus_r500.tsv (job 71)"
use_r
run_stage $C/feast_r500_bodysite/burial_source_props_by_bodysite.tsv "07v2 FEAST r500 overall + body-site (filtered)" Rscript scripts/07_feast_source_tracking_v2.R --table $C/feast_depth_matched/merged_L6genus_r500.tsv --metadata $C/feast_depth_matched/metadata_cohort_split_r500.tsv --out-dir $C/feast_r500_bodysite/ --stop-after-bodysite
use_py
run_stage $C/gradient_r500_bodysite/gradient_tests.tsv "32 gradient tests incl. body-site components (filtered)" python scripts/32_gradient_tests.py --feast-burial $C/feast_r500_bodysite/burial_source_props.tsv --feast-control $C/feast_r500_bodysite/control_source_props.tsv --feast-bodysite $C/feast_r500_bodysite/burial_source_props_by_bodysite.tsv --table $C/merged_L6genus.tsv --permutations 9999 --seed 42 --out-dir $C/gradient_r500_bodysite/
say "=== DONE abg72_contambs ==="
