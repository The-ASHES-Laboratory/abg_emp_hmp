#!/bin/bash
#SBATCH --job-name=abg10_ord
#SBATCH --account=clintonlab
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=YOUR_ADDRESS@ncsu.edu
# after 00_prep
source "${ABG_P:-$SLURM_SUBMIT_DIR}/slurm/common.sh" || { echo "FATAL: cannot source common.sh (not on Hazel or package not extracted)"; exit 1; }
say "=== abg10_ord job ${SLURM_JOB_ID:-local} on $(hostname) ==="

use_py
run_stage results/ordination/permanova_global.tsv "03 Bray-Curtis ordination/PERMANOVA" python scripts/03_beta_ordination.py --n-subsample 500 --n-replicates 5 --seed 42 --out-dir results/ordination/
run_stage results/ordination/aitchison_permanova_global.tsv "04 Aitchison ordination/PERMANOVA" python scripts/04_aitchison_ordination.py --n-subsample 500 --n-replicates 5 --seed 42 --out-dir results/ordination/
run_stage results/ordination/table2_permanova_permdisp.tsv "30 PERMDISP + balanced PERMANOVA" python scripts/30_permdisp_balanced.py --out-dir results/ordination/
run_stage results/alpha_diversity/alpha_values.tsv "02 alpha diversity" python scripts/02_alpha_diversity.py --table results/compositional/merged_L6genus.tsv --metadata results/metadata_cohort_split.tsv --out-dir results/alpha_diversity/
run_stage results/alpha_diversity/coverage_richness_summary.tsv "31 coverage-based richness" python scripts/31_coverage_richness.py --out-dir results/alpha_diversity/
run_stage results/sequencing_depth/depth_distribution.tsv "24 sequencing depth" python scripts/24_sequencing_depth.py
run_stage results/sensitivity/confound_permanova.tsv "16 body-region confound PERMANOVA" python scripts/16_confound_permanova.py
soft_stage "" "14 rarefaction sweep" python scripts/14_rarefaction_sweep.py --table results/compositional/merged_L6genus.tsv --metadata results/metadata_cohort_split.tsv --out results/sensitivity/rarefaction_sweep.tsv --seed 42
soft_stage "" "15 prevalence sweep" python scripts/15_prevalence_sweep.py --table results/compositional/merged_L6genus.tsv --metadata results/metadata_cohort_split.tsv --out results/sensitivity/prevalence_sweep.tsv
# EMP human-host-excluded sensitivity (ordination)
run_stage results/emp_reference/ordination/permanova_global.tsv "03 noHumanEMP" python scripts/03_beta_ordination.py --metadata results/emp_reference/metadata_cohort_split_noHumanEMP.tsv --n-subsample 500 --n-replicates 5 --seed 42 --out-dir results/emp_reference/ordination/
run_stage results/emp_reference/ordination/aitchison_permanova_global.tsv "04 noHumanEMP" python scripts/04_aitchison_ordination.py --metadata results/emp_reference/metadata_cohort_split_noHumanEMP.tsv --n-subsample 500 --n-replicates 5 --seed 42 --out-dir results/emp_reference/ordination/

say "=== DONE abg10_ord ==="
