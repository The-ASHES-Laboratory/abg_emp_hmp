#!/bin/bash
#SBATCH --job-name=abg40_da
#SBATCH --account=clintonlab
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=YOUR_ADDRESS@ncsu.edu
# after 00_prep
source "${ABG_P:-$SLURM_SUBMIT_DIR}/slurm/common.sh" || { echo "FATAL: cannot source common.sh (not on Hazel or package not extracted)"; exit 1; }
say "=== abg40_da job ${SLURM_JOB_ID:-local} on $(hostname) ==="

use_r
run_stage results/diff_abun/ancombc2_split/ancombc2_burial_vs_emp_L5.tsv "08 ANCOM-BC2 (5 comparisons, L6+L5)" Rscript scripts/08_ancombc2.R --n-subsample 500 --seed 42 --out-dir results/diff_abun/ancombc2_split/
run_stage results/diff_abun/aldex2/aldex2_burial_vs_emp_L5.tsv "09 ALDEx2 (5 comparisons, L6+L5)" Rscript scripts/09_aldex2.R --n-subsample 500 --mc-samples 128 --seed 42 --out-dir results/diff_abun/aldex2/
grep -l "NO_RESULTS" results/diff_abun/ancombc2_split/*.tsv 2>/dev/null | tee -a "$LOG" | grep -q . && say "WARNING: some ANCOM-BC2 comparisons failed (NO_RESULTS files above)"
use_py
run_stage results/diff_abun/da_consensus.tsv "10 consensus (published script, raw signs)" python scripts/10_da_consensus.py --out-dir results/diff_abun/
run_stage results/diff_abun/da_consensus_oriented.tsv "10b consensus with ALDEx2 orientation" python scripts/10b_da_consensus_oriented.py --out-dir results/diff_abun/
# sensitivity: burial vs emp with human-host EMP excluded
use_r
run_stage results/emp_reference/diff_abun/ancombc2_split/ancombc2_burial_vs_emp_L6.tsv "08 ANCOM-BC2 noHumanEMP" Rscript scripts/08_ancombc2.R --metadata results/emp_reference/metadata_cohort_split_noHumanEMP.tsv --n-subsample 500 --seed 42 --out-dir results/emp_reference/diff_abun/ancombc2_split/
run_stage results/emp_reference/diff_abun/aldex2/aldex2_burial_vs_emp_L6.tsv "09 ALDEx2 noHumanEMP" Rscript scripts/09_aldex2.R --metadata results/emp_reference/metadata_cohort_split_noHumanEMP.tsv --n-subsample 500 --mc-samples 128 --seed 42 --out-dir results/emp_reference/diff_abun/aldex2/
use_py
run_stage results/emp_reference/diff_abun/da_consensus_oriented.tsv "10b consensus noHumanEMP" python scripts/10b_da_consensus_oriented.py --ancombc2-dir results/emp_reference/diff_abun/ancombc2_split/ --aldex2-dir results/emp_reference/diff_abun/aldex2/ --out-dir results/emp_reference/diff_abun/

say "=== DONE abg40_da ==="
