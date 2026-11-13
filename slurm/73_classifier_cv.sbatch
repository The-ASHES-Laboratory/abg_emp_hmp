#!/bin/bash
#SBATCH --job-name=abg73_cv
#SBATCH --account=clintonlab
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# held-out (5-fold CV) evaluation of the reference classifiers; after 00_prep. Results are written per model as each finishes.
source "${ABG_P:-$SLURM_SUBMIT_DIR}/slurm/common.sh" || { echo "FATAL: cannot source common.sh"; exit 1; }
say "=== abg73_cv job ${SLURM_JOB_ID:-local} on $(hostname) ==="
use_py
export JOBLIB_MULTIPROCESSING=0
mkdir -p results/classifier_cv
python scripts/11c_classifier_cv.py --out results/classifier_cv/ || die "11c failed"
say "=== DONE abg73_cv ==="
