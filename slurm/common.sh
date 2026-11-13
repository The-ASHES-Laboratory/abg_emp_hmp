# ---- common preamble (sourced) ----
set -uo pipefail
P=${ABG_P:-${SLURM_SUBMIT_DIR:-$PWD}}  # project root; submit from it or set ABG_P
DRYRUN=${DRYRUN:-0}
export P DRYRUN
cd "$P" || { echo "FATAL: cannot cd $P"; exit 1; }
mkdir -p logs results
LOG="$P/logs/${SLURM_JOB_NAME:-local}_${SLURM_JOB_ID:-0}.log"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
die(){ say "FATAL: $*"; exit 1; }
run_stage(){ local target="$1"; shift; local label="$1"; shift
  if [ -n "$target" ] && [ -s "$target" ]; then say "--- $label SKIPPED (exists: $target) ---"; return 0; fi
  if [ "$DRYRUN" = "1" ]; then say "DRYRUN $label :: $*"; return 0; fi
  say "--- $label  START ---"; "$@" >>"$LOG" 2>&1 || die "$label failed; tail: $(tail -8 "$LOG" | tr '\n' ' | ')"; say "    $label ok"; }
# soft_stage: like run_stage, but a failure is logged as WARNING and the job continues (non-essential integrations)
soft_stage(){ local target="$1"; shift; local label="$1"; shift
  if [ -n "$target" ] && [ -s "$target" ]; then say "--- $label SKIPPED (exists: $target) ---"; return 0; fi
  if [ "$DRYRUN" = "1" ]; then say "DRYRUN(soft) $label :: $*"; return 0; fi
  say "--- $label  START (soft) ---"; if "$@" >>"$LOG" 2>&1; then say "    $label ok"; else say "WARNING: $label failed (non-fatal); tail: $(tail -5 "$LOG" | tr '\n' ' | ')"; echo "$label" >> "$P/results/SOFT_FAILURES.txt"; fi; }
# environments: never source cluster profiles; activate by explicit PATH (envs.sh is written by 00_prep)
[ -f "$P/envs.sh" ] && source "$P/envs.sh"
use_py(){ export PATH="$ENV_PY/bin:$PATH"; export CONDA_PREFIX="$ENV_PY"; unset PYTHONPATH PYTHONHOME; }
use_r(){  export PATH="$ENV_R/bin:$PATH";  export CONDA_PREFIX="$ENV_R";  unset PYTHONPATH PYTHONHOME; export R_LIBS_USER="$P/rlib"; mkdir -p "$P/rlib"; }
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
