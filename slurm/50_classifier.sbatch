#!/bin/bash
#SBATCH --job-name=abg50_clf
#SBATCH --account=clintonlab
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=YOUR_ADDRESS@ncsu.edu
# after 00_prep
source "${ABG_P:-$SLURM_SUBMIT_DIR}/slurm/common.sh" || { echo "FATAL: cannot source common.sh (not on Hazel or package not extracted)"; exit 1; }
say "=== abg50_clf job ${SLURM_JOB_ID:-local} on $(hostname) ==="

use_py
export JOBLIB_MULTIPROCESSING=0   # avoid read-only memmap error in joblib worker processes; results unchanged (fixed seeds)
mkdir -p results/classifier results/figures
awk -F'\t' '$2 == "burial"' results/metadata_cohort_split.tsv | cut -f1 > results/burial_ids.txt
awk -F'\t' '$2 == "control"' results/metadata_cohort_split.tsv | cut -f1 > results/control_ids.txt
for T in burial control; do
  run_stage results/classifier/${T}_predictions.csv "11 classifier $T (RF + L1-LR, EMP + HMP)" python scripts/11_classify_samples.py --table results/compositional/merged_L6genus.tsv --nyabg_meta data/nyabg/abg_16s_meta.tsv --emp_meta data/emp/emp_metadata.tsv --hmp_meta data/hmp/hmp_metadata.tsv --hmp_label hmp_body_subsite --emp_label empo_3 --target-ids-file results/${T}_ids.txt --sample-type $T --out results/classifier/${T}_predictions.csv --fig_cal results/figures/calibration_${T}.png --fig_imp results/figures/importance_${T}.png
done
soft_stage "" "22 classifier null distribution" python scripts/22_classifier_null.py --n-emp-holdout 500 --seed 42 --out-dir results/classifier_null/

say "=== DONE abg50_clf ==="
