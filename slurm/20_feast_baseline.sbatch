#!/bin/bash
#SBATCH --job-name=abg20_feast
#SBATCH --account=clintonlab
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=YOUR_ADDRESS@ncsu.edu
# after 00_prep; ~1 to 12 h
source "${ABG_P:-$SLURM_SUBMIT_DIR}/slurm/common.sh" || { echo "FATAL: cannot source common.sh (not on Hazel or package not extracted)"; exit 1; }
say "=== abg20_feast job ${SLURM_JOB_ID:-local} on $(hostname) ==="

use_r
run_stage results/feast_split/burial_source_props.tsv "07 FEAST baseline (overall + stratified, published defaults)" Rscript scripts/07_feast_source_tracking.R --out-dir results/feast_split/
ls -la results/feast_split/ | tee -a "$LOG"
[ -s results/feast_split/control_source_props.tsv ] || die "control_source_props.tsv missing after FEAST"

say "=== DONE abg20_feast ==="
