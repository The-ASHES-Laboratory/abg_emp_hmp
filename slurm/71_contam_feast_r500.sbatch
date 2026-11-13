#!/bin/bash
#SBATCH --job-name=abg71_contamr500
#SBATCH --account=clintonlab
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# After 70_contam_sensitivity: depth-matched FEAST and gradient tests on the blank/mock-ASV-filtered table.
source "${ABG_P:-$SLURM_SUBMIT_DIR}/slurm/common.sh" || { echo "FATAL: cannot source common.sh"; exit 1; }
say "=== abg71_contamr500 job ${SLURM_JOB_ID:-local} on $(hostname) ==="
C=results/contam_sensitivity
[ -s $C/merged_L6genus.tsv ] || die "missing $C/merged_L6genus.tsv (job 70)"
use_py
run_stage $C/feast_depth_matched/merged_L6genus_r500.tsv "34 depth-matched sources (filtered table)" python scripts/34_make_depth_matched_sources.py --table $C/merged_L6genus.tsv --metadata results/metadata_cohort_split.tsv --out-dir $C/feast_depth_matched/
use_r
run_stage $C/feast_r500/burial_source_props.tsv "07v2 FEAST r500 (filtered table)" Rscript scripts/07_feast_source_tracking_v2.R --table $C/feast_depth_matched/merged_L6genus_r500.tsv --metadata $C/feast_depth_matched/metadata_cohort_split_r500.tsv --out-dir $C/feast_r500/ --overall-only
use_py
run_stage $C/gradient_r500/gradient_tests.tsv "32 gradient tests (filtered, r500)" python scripts/32_gradient_tests.py --feast-burial $C/feast_r500/burial_source_props.tsv --feast-control $C/feast_r500/control_source_props.tsv --feast-bodysite $C/feast_r500/burial_source_props_by_bodysite.tsv --table $C/merged_L6genus.tsv --permutations 9999 --seed 42 --out-dir $C/gradient_r500/
use_r
run_stage $C/gradient_r500/gradient_glmm.tsv "33 gradient mixed models (filtered, r500)" Rscript scripts/33_gradient_glmm.R --sample-table $C/gradient_r500/gradient_sample_table.tsv --out-dir $C/gradient_r500/
say "=== DONE abg71_contamr500 ==="
