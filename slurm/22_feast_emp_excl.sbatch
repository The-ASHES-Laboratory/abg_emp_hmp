#!/bin/bash
#SBATCH --job-name=abg22_feastex
#SBATCH --account=clintonlab
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=YOUR_ADDRESS@ncsu.edu
# after 00_prep
source "${ABG_P:-$SLURM_SUBMIT_DIR}/slurm/common.sh" || { echo "FATAL: cannot source common.sh (not on Hazel or package not extracted)"; exit 1; }
say "=== abg22_feastex job ${SLURM_JOB_ID:-local} on $(hostname) ==="

use_r
run_stage results/emp_reference/feast/burial_source_props.tsv "07 FEAST noHumanEMP" Rscript scripts/07_feast_source_tracking.R --metadata results/emp_reference/metadata_cohort_split_noHumanEMP.tsv --out-dir results/emp_reference/feast/

say "=== DONE abg22_feastex ==="
