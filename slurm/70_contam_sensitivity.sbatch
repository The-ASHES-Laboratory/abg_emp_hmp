#!/bin/bash
#SBATCH --job-name=abg70_contam
#SBATCH --account=clintonlab
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Contamination sensitivity (Reviewer 1 major 3; Reviewer 2 major 3). From the FULL-DEPTH ASV table:
#  (1) identify ASVs present in the touchdown-PCR water control (Water-neg-TD-plateD) and ASVs shared with the
#      two ZymoBIOMICS mocks (mock-identical sequences), (2) remove them from all samples, (3) drop any soil sample
#      whose remaining depth falls below 100 reads or that lost >50% of its reads (310A-type cases), (4) rebuild the
#      genus table, re-harmonize, and rerun PERMANOVA (03), Aitchison (04) and FEAST overall (07). Outputs under
#      results/contam_sensitivity/. Depends on 05_fulldepth_denoise and 00_prep.
source "${ABG_P:-$SLURM_SUBMIT_DIR}/slurm/common.sh" || { echo "FATAL: cannot source common.sh"; exit 1; }
say "=== abg70_contam job ${SLURM_JOB_ID:-local} ==="
QENV=${ABG_QENV:-${ABG_ENVROOT:-$HOME/conda_envs}/qiime2-amplicon-2024.10}; ADMIX=${ABG_INPUT:?set ABG_INPUT to the directory with the archived merged EMP/HMP genus table}
U=$P/results/upstream; C=$P/results/contam_sensitivity; mkdir -p "$C"
for f in "$U/table.qza" "$U/taxonomy_silva.qza" "$U/rep_seqs_export/dna-sequences.fasta" "$U/table_export/feature-table.tsv" "$P/results/metadata_cohort_split.tsv"; do [ -s "$f" ] || die "missing input $f"; done
export PATH="$QENV/bin:$PATH"; export CONDA_PREFIX="$QENV"; unset PYTHONPATH PYTHONHOME
# ---- 1. decide which ASVs to remove (python in the QIIME env: pandas available) ----
run_stage "$C/asvs_to_remove.tsv" "identify blank-derived and mock-identical ASVs" python - <<'EOF'
import os, pandas as pd
P=os.environ["P"]; U=f"{P}/results/upstream"; C=f"{P}/results/contam_sensitivity"
t=pd.read_csv(f"{U}/table_export/feature-table.tsv", sep="\t", skiprows=1, index_col=0); t.index.name="asv"
blank=[c for c in t.columns if c.startswith("Water-neg-TD")]; mocks=[c for c in t.columns if c.startswith("Zymo-mock")]
meta=pd.read_csv(f"{P}/results/metadata_cohort_split.tsv", sep="\t", dtype=str); soil=[s for s in meta["sample-id"] if s in t.columns]
b=t[blank].sum(axis=1) if blank else pd.Series(0,index=t.index); m=t[mocks].sum(axis=1) if mocks else pd.Series(0,index=t.index)
rec=pd.DataFrame({"blank_reads":b, "mock_reads":m, "soil_reads":t[soil].sum(axis=1), "soil_samples_present":(t[soil]>0).sum(axis=1)})
rec["reason"]=""; rec.loc[rec.blank_reads>0,"reason"]+="blank;"; rec.loc[rec.mock_reads>0,"reason"]+="mock;"
rm=rec[rec.reason!=""].sort_values("soil_reads", ascending=False)
rm.to_csv(f"{C}/asvs_to_remove.tsv", sep="\t"); print("ASVs flagged:", len(rm), " blank-derived:", int((rec.blank_reads>0).sum()), " mock-identical:", int((rec.mock_reads>0).sum()))
print("soil reads removed:", int(rm.soil_reads.sum()), "of", int(rec.soil_reads.sum()), f"({100*rm.soil_reads.sum()/rec.soil_reads.sum():.2f}%)")
# per-sample impact
keep=t.drop(index=rm.index); before=t[soil].sum(); after=keep[soil].sum(); imp=pd.DataFrame({"depth_before":before,"depth_after":after,"frac_lost":1-after/before})
imp["drop_sample"]=(imp.depth_after<100)|(imp.frac_lost>0.5); imp.sort_values("frac_lost",ascending=False).to_csv(f"{C}/sample_impact.tsv", sep="\t")
print("samples dropped (depth<100 or >50% reads lost):", imp.drop_sample.sum(), list(imp[imp.drop_sample].index))
keep=keep.drop(columns=[s for s in imp[imp.drop_sample].index])
keep.index.name="#OTU ID"; keep.to_csv(f"{C}/asv_table_filtered.tsv", sep="\t")
open(f"{C}/asv_table_filtered.biom.hdr","w").write("# Constructed from biom file\n")
EOF
# ---- 2. rebuild genus table: biom -> qza -> filter euk/chloro/mito -> collapse L6 -> merge with EMP/HMP ----
run_stage "$C/asv_table_filtered.biom" "ASV TSV to biom" biom convert -i "$C/asv_table_filtered.tsv" -o "$C/asv_table_filtered.biom" --table-type="OTU table" --to-hdf5
run_stage "$C/asv_table_filtered.qza" "import filtered ASV table" qiime tools import --type 'FeatureTable[Frequency]' --input-path "$C/asv_table_filtered.biom" --output-path "$C/asv_table_filtered.qza"
run_stage "$C/nyabg_filt.qza" "remove Eukaryota/chloroplast/mitochondria" qiime taxa filter-table --i-table "$C/asv_table_filtered.qza" --i-taxonomy "$U/taxonomy_silva.qza" --p-exclude "Eukaryota,chloroplast,mitochondria" --o-filtered-table "$C/nyabg_filt.qza"
run_stage "$C/nyabg_L6.qza" "collapse L6" qiime taxa collapse --i-table "$C/nyabg_filt.qza" --i-taxonomy "$U/taxonomy_silva.qza" --p-level 6 --o-collapsed-table "$C/nyabg_L6.qza"
run_stage "$C/merged_L6.qza" "merge with EMP/HMP L6" qiime feature-table merge --i-tables "$C/nyabg_L6.qza" --i-tables "$ADMIX/results/tables/emp_filtered_L6.qza" --i-tables "$ADMIX/results/tables/hmp_filtered_L6.qza" --o-merged-table "$C/merged_L6.qza"
run_stage "$C/merged_export/feature-table.tsv" "export merged" bash -c "qiime tools export --input-path '$C/merged_L6.qza' --output-path '$C/merged_export' && biom convert -i '$C/merged_export/feature-table.biom' -o '$C/merged_export/feature-table.tsv' --to-tsv"
[ -s "$C/merged_L6genus_original.tsv" ] || { printf 'taxon\t'; sed -n '2p' "$C/merged_export/feature-table.tsv" | cut -f2-; tail -n +3 "$C/merged_export/feature-table.tsv"; } > "$C/merged_L6genus_original.tsv"
# ---- 3. harmonize and rerun 03, 04, 07 with the filtered table ----
use_py
run_stage "$C/merged_L6genus.tsv" "00 harmonize (filtered)" python scripts/00_harmonize_taxonomy.py --input "$C/merged_L6genus_original.tsv" --output "$C/merged_L6genus.tsv" --mapping "$C/taxonomy_mapping.tsv"
run_stage "$C/ordination/permanova_global.tsv" "03 PERMANOVA (filtered)" python scripts/03_beta_ordination.py --table "$C/merged_L6genus.tsv" --n-subsample 500 --n-replicates 5 --seed 42 --out-dir "$C/ordination/"
run_stage "$C/ordination/aitchison_permanova_global.tsv" "04 Aitchison (filtered)" python scripts/04_aitchison_ordination.py --table "$C/merged_L6genus.tsv" --n-subsample 500 --n-replicates 5 --seed 42 --out-dir "$C/ordination/"
use_r
run_stage "$C/feast/burial_source_props.tsv" "07 FEAST overall (filtered)" Rscript scripts/07_feast_source_tracking.R --table "$C/merged_L6genus.tsv" --out-dir "$C/feast/"
use_py
run_stage "$C/gradient/gradient_tests.tsv" "32 gradient tests (filtered)" python scripts/32_gradient_tests.py --feast-burial "$C/feast/burial_source_props.tsv" --feast-control "$C/feast/control_source_props.tsv" --feast-bodysite "$C/feast/burial_source_props_by_bodysite.tsv" --table "$C/merged_L6genus.tsv" --permutations 9999 --seed 42 --out-dir "$C/gradient/"
say "=== DONE abg70_contam ==="
