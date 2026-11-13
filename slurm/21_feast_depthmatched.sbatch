#!/bin/bash
#SBATCH --job-name=abg21_feastdm
#SBATCH --account=clintonlab
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=YOUR_ADDRESS@ncsu.edu
# after 00_prep; two FEAST runs
source "${ABG_P:-$SLURM_SUBMIT_DIR}/slurm/common.sh" || { echo "FATAL: cannot source common.sh (not on Hazel or package not extracted)"; exit 1; }
say "=== abg21_feastdm job ${SLURM_JOB_ID:-local} on $(hostname) ==="

use_py
run_stage results/feast_depth_matched/merged_L6genus_r500.tsv "34 build depth-matched source tables" python scripts/34_make_depth_matched_sources.py --out-dir results/feast_depth_matched/
use_r
for L in r500 burialmatched; do
  run_stage results/feast_depth_matched/$L/burial_source_props.tsv "07 FEAST sources $L" Rscript scripts/07_feast_source_tracking.R --table results/feast_depth_matched/merged_L6genus_$L.tsv --metadata results/feast_depth_matched/metadata_cohort_split_$L.tsv --out-dir results/feast_depth_matched/$L/
done

say "=== DONE abg21_feastdm ==="
