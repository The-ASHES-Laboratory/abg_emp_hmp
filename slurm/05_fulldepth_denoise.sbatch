#!/bin/bash
#SBATCH --job-name=abg05_fulldepth
#SBATCH --account=clintonlab
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Full-depth re-denoising of the 92 NYABG libraries (no demux subsample), SILVA 138 classification,
# genus (L6) collapse and merge with the archived EMP/HMP L6 tables. Replaces the 30%-subsampled NYABG table.
source "${ABG_P:-$SLURM_SUBMIT_DIR}/slurm/common.sh" || { echo "FATAL: cannot source common.sh"; exit 1; }
say "=== abg05_fulldepth job ${SLURM_JOB_ID:-local} on $(hostname) ==="
QENV=${ABG_QENV:-${ABG_ENVROOT:-$HOME/conda_envs}/qiime2-amplicon-2024.10}
[ -x "$QENV/bin/qiime" ] || die "QIIME 2 env not found at $QENV"
export PATH="$QENV/bin:$PATH"; export CONDA_PREFIX="$QENV"; unset PYTHONPATH PYTHONHOME
export MPLBACKEND=Agg TMPDIR="$P/tmp"; mkdir -p "$P/tmp" "$P/raw" "$P/results/upstream" "$P/results/compositional"
ADMIX=${ABG_INPUT:?set ABG_INPUT to the directory with the archived merged EMP/HMP genus table}
CLASSIFIER=$ADMIX/data/ref/silva-138-99-nb-classifier.qza
EMP_L6=$ADMIX/results/tables/emp_filtered_L6.qza; HMP_L6=$ADMIX/results/tables/hmp_filtered_L6.qza
for f in "$CLASSIFIER" "$EMP_L6" "$HMP_L6" "$P/abg_raw_reads_fastqgz.tar"; do [ -s "$f" ] || die "required input missing: $f"; done
qiime --version | tee -a "$LOG"
U=$P/results/upstream

# --- 1. raw reads and manifest -------------------------------------------------
if [ ! -s "$U/manifest.tsv" ]; then
  say "--- extracting raw reads ---"; tar xf "$P/abg_raw_reads_fastqgz.tar" -C "$P/raw" || die "tar extract failed"
  find "$P/raw" -name '._*' -delete; find "$P/raw" -name '.DS_Store' -delete   # macOS AppleDouble metadata files
  N=$(ls "$P/raw"/*_R1_001.fastq.gz | grep -v '/\._' | wc -l); say "R1 files: $N"; [ "$N" -eq 92 ] || die "expected 92 R1 files, found $N"
  python - <<'EOF' || die "manifest build failed"
import os, re, sys, csv
P=os.environ["P"]; raw=f"{P}/raw"; U=f"{P}/results/upstream"
# fastq prefix -> metadata id (two libraries were named differently in the sample sheet; 'NA' is renamed because QIIME/pandas read it as missing)
RENAME={"310E-1":"310-00E","310E-2":"310-01E","NA":"s_NA"}
with open(f"{P}/data/nyabg/abg_16s_meta.tsv") as fh: meta=[l.split("\t")[0].strip() for l in fh.read().splitlines()[2:] if l.strip()]
rows=[]; idmap=[]
for f in sorted(os.listdir(raw)):
    if f.startswith("._"): continue
    m=re.match(r"(.+)_S(\d+)_L001_R1_001\.fastq\.gz$", f)
    if not m: continue
    pre,idx=m.group(1),m.group(2); sid=RENAME.get(pre,pre); r2=f.replace("_R1_","_R2_")
    if not os.path.exists(f"{raw}/{r2}"): sys.exit(f"missing R2 for {f}")
    rows.append((sid,f"{raw}/{f}",f"{raw}/{r2}")); idmap.append((sid,pre,idx,f,"renamed" if pre in RENAME else ""))
ids=[r[0] for r in rows]
dups=[i for i in set(ids) if ids.count(i)>1]
if dups: sys.exit(f"duplicate sample ids in manifest: {dups}")
not_in_meta=[i for i in ids if i not in meta and i!="s_NA"]
if not_in_meta: sys.exit(f"manifest ids absent from metadata (fix RENAME): {not_in_meta}")
with open(f"{U}/manifest.tsv","w") as fh:
    fh.write("sample-id\tforward-absolute-filepath\treverse-absolute-filepath\n")
    for r in rows: fh.write("\t".join(r)+"\n")
with open(f"{U}/manifest_id_map.tsv","w") as fh:
    fh.write("manifest_id\traw_prefix\tsample_sheet_index\tfile_R1\tnote\n")
    for r in idmap: fh.write("\t".join(r)+"\n")
print(f"manifest: {len(rows)} libraries; renamed: {[r for r in idmap if r[4]]}")
EOF
fi
# --- 2. import, summarize, filter ------------------------------------------------
run_stage "$U/demux.qza" "import paired-end reads" qiime tools import --type 'SampleData[PairedEndSequencesWithQuality]' --input-path "$U/manifest.tsv" --output-path "$U/demux.qza" --input-format PairedEndFastqManifestPhred33V2
run_stage "$U/demux.qzv" "demux summarize" qiime demux summarize --i-data "$U/demux.qza" --o-visualization "$U/demux.qzv"
run_stage "$U/demux_summary/per-sample-fastq-counts.tsv" "export demux counts" qiime tools export --input-path "$U/demux.qzv" --output-path "$U/demux_summary"
run_stage "$U/demux_filt.qza" "filter libraries with <=100 forward reads (as in original run)" qiime demux filter-samples --i-demux "$U/demux.qza" --m-metadata-file "$U/demux_summary/per-sample-fastq-counts.tsv" --p-where 'CAST([forward sequence count] AS INT) > 100' --o-filtered-demux "$U/demux_filt.qza"
# --- 3. DADA2 with the original parameters, full depth ---------------------------
run_stage "$U/table.qza" "DADA2 denoise-paired trunc 240/220 maxEE 2/2 truncQ 2 (no subsample)" qiime dada2 denoise-paired --i-demultiplexed-seqs "$U/demux_filt.qza" --p-trunc-len-f 240 --p-trunc-len-r 220 --p-trim-left-f 0 --p-trim-left-r 0 --p-max-ee-f 2 --p-max-ee-r 2 --p-trunc-q 2 --p-n-threads ${SLURM_CPUS_PER_TASK:-8} --o-table "$U/table.qza" --o-representative-sequences "$U/rep_seqs.qza" --o-denoising-stats "$U/denoising_stats.qza"
run_stage "$U/denoising_stats/stats.tsv" "export denoising stats" qiime tools export --input-path "$U/denoising_stats.qza" --output-path "$U/denoising_stats"
run_stage "$U/table_export/feature-table.biom" "export ASV table" qiime tools export --input-path "$U/table.qza" --output-path "$U/table_export"
run_stage "$U/table_export/feature-table.tsv" "ASV table to TSV" biom convert -i "$U/table_export/feature-table.biom" -o "$U/table_export/feature-table.tsv" --to-tsv
run_stage "$U/rep_seqs_export/dna-sequences.fasta" "export rep seqs" qiime tools export --input-path "$U/rep_seqs.qza" --output-path "$U/rep_seqs_export"
# --- 4. SILVA 138 taxonomy, filter, collapse ---------------------------------------
run_stage "$U/taxonomy_silva.qza" "classify-sklearn SILVA 138 99% full-length NB classifier" qiime feature-classifier classify-sklearn --i-classifier "$CLASSIFIER" --i-reads "$U/rep_seqs.qza" --p-n-jobs 4 --p-reads-per-batch 2000 --o-classification "$U/taxonomy_silva.qza"
run_stage "$U/taxonomy_export/taxonomy.tsv" "export taxonomy" qiime tools export --input-path "$U/taxonomy_silva.qza" --output-path "$U/taxonomy_export"
run_stage "$U/nyabg_filt.qza" "remove Eukaryota/chloroplast/mitochondria" qiime taxa filter-table --i-table "$U/table.qza" --i-taxonomy "$U/taxonomy_silva.qza" --p-exclude "Eukaryota,chloroplast,mitochondria" --o-filtered-table "$U/nyabg_filt.qza"
run_stage "$U/nyabg_filtered_L6.qza" "collapse to genus (L6)" qiime taxa collapse --i-table "$U/nyabg_filt.qza" --i-taxonomy "$U/taxonomy_silva.qza" --p-level 6 --o-collapsed-table "$U/nyabg_filtered_L6.qza"
# --- 5. merge with archived EMP + HMP L6 tables and export -------------------------
run_stage "$U/merged_L6.qza" "merge NYABG(full depth) + EMP + HMP L6" qiime feature-table merge --i-tables "$U/nyabg_filtered_L6.qza" --i-tables "$EMP_L6" --i-tables "$HMP_L6" --o-merged-table "$U/merged_L6.qza"
run_stage "$U/merged_export/feature-table.biom" "export merged table" qiime tools export --input-path "$U/merged_L6.qza" --output-path "$U/merged_export"
run_stage "$U/merged_export/feature-table.tsv" "merged table to TSV" biom convert -i "$U/merged_export/feature-table.biom" -o "$U/merged_export/feature-table.tsv" --to-tsv
# header: '# Constructed from biom file' + '#OTU ID' -> rename first column to 'taxon' (00_harmonize_taxonomy handles both, but keep the archived format)
if [ ! -s "$P/results/compositional/merged_L6genus_original.tsv" ]; then
  { printf 'taxon\t'; sed -n '2p' "$U/merged_export/feature-table.tsv" | cut -f2-; tail -n +3 "$U/merged_export/feature-table.tsv"; } > "$P/results/compositional/merged_L6genus_original.tsv" || die "header rewrite failed"
fi
sha256sum "$P/results/compositional/merged_L6genus_original.tsv" > "$P/results/provenance_input_sha256.txt"
NT=$(( $(wc -l < "$P/results/compositional/merged_L6genus_original.tsv") - 1 )); NS=$(( $(head -1 "$P/results/compositional/merged_L6genus_original.tsv" | tr '\t' '\n' | wc -l) - 1 ))
say "merged table: $NT taxa x $NS samples (archived 30% table: 4293 x 22306)"
[ "$NS" -ge 22290 ] || die "merged table has too few samples ($NS); EMP/HMP merge failed"
python - <<'EOF' | tee -a "$LOG"
import os, pandas as pd
P=os.environ["P"]; st=pd.read_csv(f"{P}/results/upstream/denoising_stats/stats.tsv", sep="\t", comment="#"); st=st[st.iloc[:,0]!="#q2:types"]
st.columns=[c.strip() for c in st.columns]; num=[c for c in st.columns if c!=st.columns[0]]
for c in num: st[c]=pd.to_numeric(st[c], errors="coerce")
print("libraries denoised:", len(st)); print("non-chimeric reads: total", int(st["non-chimeric"].sum()), "median", st["non-chimeric"].median(), "min", st["non-chimeric"].min(), "max", st["non-chimeric"].max())
EOF
say "=== DONE abg05_fulldepth ==="
