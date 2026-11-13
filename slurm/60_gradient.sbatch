#!/bin/bash
#SBATCH --job-name=abg60_grad
#SBATCH --account=clintonlab
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=YOUR_ADDRESS@ncsu.edu
# after 20_feast_baseline and 30_isa_networks_decontam (classifier outputs used only by soft stage 28)
source "${ABG_P:-$SLURM_SUBMIT_DIR}/slurm/common.sh" || { echo "FATAL: cannot source common.sh (not on Hazel or package not extracted)"; exit 1; }
say "=== abg60_grad job ${SLURM_JOB_ID:-local} on $(hostname) ==="

use_py
[ -s results/feast_split/burial_source_props.tsv ] || die "FEAST baseline output missing; run 20_feast_baseline first"
run_stage results/cranium/cranium_gradient.tsv "19 cranium analysis (published)" python scripts/19_cranium_analysis.py --nyabg-meta data/nyabg/abg_16s_meta_clean.tsv
run_stage results/soilmatrix/soilmatrix_vs_bodyassoc.tsv "20 soil matrix analysis (published)" python scripts/20_soilmatrix_analysis.py --nyabg-meta data/nyabg/abg_16s_meta_clean.tsv
run_stage results/within_burial/within_burial_variance.tsv "25 within-burial variance" python scripts/25_within_burial_variance.py --nyabg-meta data/nyabg/abg_16s_meta_clean.tsv
run_stage results/pseudoreplication/sample_burial_id_map.tsv "17 pseudoreplication" python scripts/17_pseudoreplication.py --nyabg-meta data/nyabg/abg_16s_meta_clean.tsv
run_stage results/gradient/gradient_tests.tsv "32 gradient re-tests (within-burial JT, two-sided, depth)" python scripts/32_gradient_tests.py --permutations 9999 --seed 42 --out-dir results/gradient/
soft_stage "" "21 Staphylococcus check" python scripts/21_staphylococcus_check.py
soft_stage "" "23 power analysis" python scripts/23_power_analysis.py
soft_stage "" "27 integrate FEAST + network (burial)" python scripts/27_integrate_feast_network.py --table results/compositional/merged_L6genus.tsv --metadata results/metadata_cohort_split.tsv --feast results/feast_split/burial_source_props.tsv --ambiguous results/networks/burial/overall/ambiguous_taxa_assignments.tsv --target-cohort burial --out-dir results/networks/burial/feast_integration/ --cyto-out results/networks/burial/overall/cytoscape/source_network_nodes_feast.tsv
soft_stage "" "27 integrate FEAST + network (control)" python scripts/27_integrate_feast_network.py --table results/compositional/merged_L6genus.tsv --metadata results/metadata_cohort_split.tsv --feast results/feast_split/control_source_props.tsv --ambiguous results/networks/control/overall/ambiguous_taxa_assignments.tsv --target-cohort control --out-dir results/networks/control/feast_integration/ --cyto-out results/networks/control/overall/cytoscape/source_network_nodes_feast.tsv
soft_stage "" "28 burial vs control integrated comparison" python scripts/28_burial_vs_control.py
use_r
run_stage results/gradient/gradient_glmm.tsv "33 gradient mixed models" Rscript scripts/33_gradient_glmm.R --sample-table results/gradient/gradient_sample_table.tsv --out-dir results/gradient/
run_stage results/pseudoreplication/pseudorep_permanova.tsv "18 pseudoreplication strata PERMANOVA" Rscript scripts/18_pseudorep_permanova.R --n-subsample 500 --seed 42 --out-dir results/pseudoreplication/

say "=== DONE abg60_grad ==="
