#!/bin/bash
#SBATCH --job-name=abg00_prep
#SBATCH --account=clintonlab
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=YOUR_ADDRESS@ncsu.edu
# no dependencies; writes envs.sh, harmonized table, metadata
source "${ABG_P:-$SLURM_SUBMIT_DIR}/slurm/common.sh" || { echo "FATAL: cannot source common.sh (not on Hazel or package not extracted)"; exit 1; }
say "=== abg00_prep job ${SLURM_JOB_ID:-local} on $(hostname) ==="

# --- probe environments and write envs.sh ---
ENVROOT=${ABG_ENVROOT:-$HOME/conda_envs}
ENV_R="$ENVROOT/rstats-nyabg"; [ -x "$ENV_R/bin/Rscript" ] || die "R env not found: $ENV_R"
ENV_PY=""; [ -x "$P/envs/py/bin/python" ] && "$P/envs/py/bin/python" -c "import skbio, sklearn" 2>/dev/null && ENV_PY="$P/envs/py"
for c in qiime2-amplicon-2024.10 nyabg-pytools nyabg-tools nyabg-workflow; do
  [ -n "$ENV_PY" ] && break
  if [ -x "$ENVROOT/$c/bin/python" ] && "$ENVROOT/$c/bin/python" -c "import skbio, sklearn, pandas, numpy, scipy" 2>/dev/null; then ENV_PY="$ENVROOT/$c"; break; fi
done
if [ -z "$ENV_PY" ]; then
  say "no existing env with scikit-bio + scikit-learn; attempting to create $P/envs/py from envs/python.yml"
  CONDA_BIN=$(command -v mamba || command -v conda || ls /usr/local/apps/*/bin/conda /usr/local/usrapps/*/conda/bin/conda 2>/dev/null | head -1)
  [ -n "$CONDA_BIN" ] || die "no conda/mamba binary found to create a Python env; create one manually: conda env create -p $P/envs/py -f envs/python.yml"
  "$CONDA_BIN" env create -p "$P/envs/py" -f envs/python.yml >>"$LOG" 2>&1 || die "conda env create failed (see $LOG); create manually: conda env create -p $P/envs/py -f envs/python.yml"
  "$P/envs/py/bin/python" -c "import skbio, sklearn" || die "new env still lacks scikit-bio/scikit-learn"
  ENV_PY="$P/envs/py"
fi
printf 'export ENV_R=%s\nexport ENV_PY=%s\n' "$ENV_R" "$ENV_PY" > envs.sh; source envs.sh
say "ENV_PY=$ENV_PY  ENV_R=$ENV_R"
use_py; python - <<'EOF' | tee -a "$LOG"
import sys, skbio, sklearn, pandas, numpy, scipy
print("python", sys.version.split()[0], "skbio", skbio.__version__, "sklearn", sklearn.__version__, "pandas", pandas.__version__, "numpy", numpy.__version__, "scipy", scipy.__version__)
try:
    import statsmodels; print("statsmodels", statsmodels.__version__)
except Exception as e: print("statsmodels MISSING (32_gradient_tests logistic/OLS rows will be skipped)")
EOF
use_r; Rscript - <<'EOF' 2>&1 | tee -a "$LOG"
pk <- c("FEAST","vegan","indicspecies","ALDEx2","ANCOMBC","decontam","data.table","argparse","lme4","glmmTMB","TreeSummarizedExperiment","phyloseq")
v <- sapply(pk, function(p) tryCatch(as.character(packageVersion(p)), error=function(e) "MISSING")); cat(R.version.string, "\n"); print(v)
req <- c("FEAST","vegan","indicspecies","ALDEx2","ANCOMBC","decontam","data.table","argparse","lme4")
if (any(v[req]=="MISSING")) { cat("FATAL: missing required R packages:", names(v[req])[v[req]=="MISSING"], "\n"); quit(status=1) }
if (v["glmmTMB"]=="MISSING") cat("NOTE: glmmTMB missing; 33_gradient_glmm.R will use lme4 models only (M2/M3)\n")
EOF
[ ${PIPESTATUS[0]} -eq 0 ] || die "R package probe failed"
# --- input table: produced by 05_fulldepth_denoise (results/compositional/merged_L6genus_original.tsv) ---
[ -s results/compositional/merged_L6genus_original.tsv ] || die "full-depth merged table missing; 05_fulldepth_denoise must finish first"
[ -s results/provenance_input_sha256.txt ] || sha256sum results/compositional/merged_L6genus_original.tsv > results/provenance_input_sha256.txt
use_py
run_stage results/compositional/merged_L6genus_harmonized.tsv "00 harmonize taxonomy" python scripts/00_harmonize_taxonomy.py
[ -s results/compositional/merged_L6genus.tsv ] || cp results/compositional/merged_L6genus_harmonized.tsv results/compositional/merged_L6genus.tsv
NT=$(( $(wc -l < results/compositional/merged_L6genus.tsv) - 1 )); say "harmonized taxa: $NT"; [ "$NT" -ge 2500 ] && [ "$NT" -le 6000 ] || die "harmonized taxon count $NT outside sanity range 2500-6000"; echo "$NT" > results/harmonized_taxon_count.txt
# master metadata: add NYABG libraries present in the full-depth table but absent from the archived master list (e.g. 310-00E, 310-01E)
run_stage results/metadata_cohort.prev.augmented.flag "augment master metadata with new NYABG ids" python - <<'EOF'
import pandas as pd
hdr=open("results/compositional/merged_L6genus_original.tsv").readline().rstrip("\n").split("\t")[1:]
master=pd.read_csv("results/metadata_cohort.prev.tsv", sep="\t", dtype=str)
with open("data/nyabg/abg_16s_meta.tsv") as fh: ny=[l.split("\t")[0].strip() for l in fh.read().splitlines()[2:] if l.strip()]
missing=[s for s in hdr if s in ny and s not in set(master["sample-id"])]
if missing:
    master=pd.concat([master, pd.DataFrame({"sample-id":missing, "cohort":["nyabg"]*len(missing)})], ignore_index=True)
    master.to_csv("results/metadata_cohort.prev.tsv", sep="\t", index=False)
open("results/metadata_cohort.prev.augmented.flag","w").write("\n".join(missing)+"\n")
print("NYABG ids added to master metadata:", missing)
EOF
run_stage results/metadata_cohort_split.tsv "01 build metadata" python scripts/01_build_metadata.py
NM=$(( $(wc -l < results/metadata_cohort_split.tsv) - 1 )); say "metadata rows: $NM"
python - <<'EOF' | tee -a "$LOG"
import pandas as pd
m=pd.read_csv("results/metadata_cohort_split.tsv",sep="\t",dtype=str); print(m.cohort.value_counts().to_dict())
vc=m.cohort.value_counts(); assert vc.get("control")==6 and vc.get("hmp")==4743 and vc.get("burial") in (70,71,72), f"unexpected cohort counts {vc.to_dict()}"; print("burial n =", vc.get("burial"), "(70 in the 30% run; 72 expected if 310-00E/310-01E now pass filters)")
EOF
[ ${PIPESTATUS[0]} -eq 0 ] || die "metadata check failed"
# EMP human-host-excluded metadata (sensitivity)
run_stage results/emp_reference/metadata_cohort_split_noHumanEMP.tsv "35 EMP exclusion metadata" python scripts/35_filter_metadata.py

say "=== DONE abg00_prep ==="
