#!/bin/bash
#SBATCH --job-name=abg30_net
#SBATCH --account=clintonlab
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=YOUR_ADDRESS@ncsu.edu
# after 00_prep
source "${ABG_P:-$SLURM_SUBMIT_DIR}/slurm/common.sh" || { echo "FATAL: cannot source common.sh (not on Hazel or package not extracted)"; exit 1; }
say "=== abg30_net job ${SLURM_JOB_ID:-local} on $(hostname) ==="

use_r
run_stage results/sensitivity/isa_results.tsv "05 indicator species" Rscript scripts/05_indicator_species.R --n-subsample 500 --seed 42 --out-dir results/sensitivity/
run_stage results/emp_reference/isa/isa_results.tsv "05 ISA noHumanEMP" Rscript scripts/05_indicator_species.R --metadata results/emp_reference/metadata_cohort_split_noHumanEMP.tsv --n-subsample 500 --seed 42 --out-dir results/emp_reference/isa/
use_py
TABLE=results/compositional/merged_L6genus.tsv; META=results/metadata_cohort_split.tsv
run_stage results/networks/burial/overall/source_network_taxa.tsv "06 network burial overall" python scripts/06_source_network.py --table $TABLE --metadata $META --cohorts emp,hmp,burial --source-cohorts emp,hmp --target-cohort burial --out-dir results/networks/burial/overall/
run_stage results/networks/control/overall/source_network_taxa.tsv "06 network control overall" python scripts/06_source_network.py --table $TABLE --metadata $META --cohorts emp,hmp,control --source-cohorts emp,hmp --target-cohort control --out-dir results/networks/control/overall/
for SITE in oral skin stool vaginal nasal; do
  run_stage results/networks/burial/vs_hmp_${SITE}/source_network_taxa.tsv "06 burial vs hmp_$SITE" python scripts/06_source_network.py --table $TABLE --metadata $META --cohorts emp,hmp_${SITE},burial --source-cohorts emp,hmp_${SITE} --target-cohort burial --out-dir results/networks/burial/vs_hmp_${SITE}/
  run_stage results/networks/control/vs_hmp_${SITE}/source_network_taxa.tsv "06 control vs hmp_$SITE" python scripts/06_source_network.py --table $TABLE --metadata $META --cohorts emp,hmp_${SITE},control --source-cohorts emp,hmp_${SITE} --target-cohort control --out-dir results/networks/control/vs_hmp_${SITE}/
done
for ENVTYPE in soil water sediment animal plant; do
  run_stage results/networks/burial/vs_emp_${ENVTYPE}/source_network_taxa.tsv "06 burial vs emp_$ENVTYPE" python scripts/06_source_network.py --table $TABLE --metadata $META --cohorts emp_${ENVTYPE},hmp,burial --source-cohorts emp_${ENVTYPE},hmp --target-cohort burial --out-dir results/networks/burial/vs_emp_${ENVTYPE}/
  run_stage results/networks/control/vs_emp_${ENVTYPE}/source_network_taxa.tsv "06 control vs emp_$ENVTYPE" python scripts/06_source_network.py --table $TABLE --metadata $META --cohorts emp_${ENVTYPE},hmp,control --source-cohorts emp_${ENVTYPE},hmp --target-cohort control --out-dir results/networks/control/vs_emp_${ENVTYPE}/
done
METAX=results/emp_reference/metadata_cohort_split_noHumanEMP.tsv
run_stage results/emp_reference/networks/burial/source_network_taxa.tsv "06 burial noHumanEMP" python scripts/06_source_network.py --table $TABLE --metadata $METAX --cohorts emp,hmp,burial --source-cohorts emp,hmp --target-cohort burial --out-dir results/emp_reference/networks/burial/
run_stage results/emp_reference/networks/control/source_network_taxa.tsv "06 control noHumanEMP" python scripts/06_source_network.py --table $TABLE --metadata $METAX --cohorts emp,hmp,control --source-cohorts emp,hmp --target-cohort control --out-dir results/emp_reference/networks/control/
soft_stage results/comparison/network_shift_summary.tsv "26 compare networks" python scripts/26_compare_networks.py
use_r
run_stage results/qc/decontam_results.tsv "12 decontam (control soils as negatives, published)" Rscript scripts/12_decontamination.R --threshold 0.1 --out-dir results/qc/
use_py
soft_stage "" "13 decontam check vs HMP indicators" python scripts/13_decontam_check.py --out-dir results/qc/

say "=== DONE abg30_net ==="
