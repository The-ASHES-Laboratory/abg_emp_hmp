#!/bin/bash
#SBATCH --job-name=abg23_r500sens
#SBATCH --account=clintonlab
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Depth-matched (sources rarefied to 500 reads) FEAST sensitivity runs and gradient tests on the r500 FEAST.
# Uses 07_feast_source_tracking_v2.R (adds --overall-only and logs the FEAST rarefaction depth); the original
# 07 script is left untouched while jobs 20/21 are still running.
source "${ABG_P:-$SLURM_SUBMIT_DIR}/slurm/common.sh" || { echo "FATAL: cannot source common.sh"; exit 1; }
say "=== abg23_r500sens job ${SLURM_JOB_ID:-local} on $(hostname) ==="
D=results/feast_depth_matched
[ -s $D/merged_L6genus_r500.tsv ] && [ -s $D/r500/burial_source_props.tsv ] || die "r500 inputs missing (job 21 overall stage)"
use_py
run_stage $D/metadata_cohort_split_r500_noHumanEMP.tsv "35 EMP exclusion on r500 metadata" python scripts/35_filter_metadata.py --metadata $D/metadata_cohort_split_r500.tsv --exclude data/emp/emp_exclusion_human_and_blanks.tsv --out $D/metadata_cohort_split_r500_noHumanEMP.tsv
use_r
run_stage $D/r500_noHumanEMP/burial_source_props.tsv "07v2 FEAST r500, EMP human-host + blanks excluded" Rscript scripts/07_feast_source_tracking_v2.R --table $D/merged_L6genus_r500.tsv --metadata $D/metadata_cohort_split_r500_noHumanEMP.tsv --out-dir $D/r500_noHumanEMP/ --overall-only
use_py
run_stage results/gradient_r500/gradient_tests.tsv "32 gradient tests on r500 FEAST" python scripts/32_gradient_tests.py --feast-burial $D/r500/burial_source_props.tsv --feast-control $D/r500/control_source_props.tsv --feast-bodysite $D/r500/burial_source_props_by_bodysite.tsv --permutations 9999 --seed 42 --out-dir results/gradient_r500/
use_r
run_stage results/gradient_r500/gradient_glmm.tsv "33 gradient mixed models on r500 FEAST" Rscript scripts/33_gradient_glmm.R --sample-table results/gradient_r500/gradient_sample_table.tsv --out-dir results/gradient_r500/
soft_stage $D/r500_overall_check/burial_source_props.tsv "07v2 reproduction check of job 21 r500 overall" Rscript scripts/07_feast_source_tracking_v2.R --table $D/merged_L6genus_r500.tsv --metadata $D/metadata_cohort_split_r500.tsv --out-dir $D/r500_overall_check/ --overall-only
say "=== DONE abg23_r500sens ==="
