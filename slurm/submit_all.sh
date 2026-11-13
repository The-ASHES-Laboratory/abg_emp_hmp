#!/bin/bash
# submit_all.sh -- submit the whole revision pipeline with Slurm dependencies. Run from the project root on Hazel.
set -uo pipefail
P=${ABG_P:-$PWD}; cd "$P" || exit 1; mkdir -p logs
sub(){ local dep="$1"; shift; local out; if [ -n "$dep" ]; then out=$(sbatch --parsable --dependency=afterok:$dep "$@"); else out=$(sbatch --parsable "$@"); fi; echo "${out%%;*}"; }
J5=$(sub "" slurm/05_fulldepth_denoise.sbatch)           || exit 1
J0=$(sub "$J5" slurm/00_prep.sbatch)                    || exit 1
J10=$(sub "$J0" slurm/10_ordination.sbatch)
J20=$(sub "$J0" slurm/20_feast_baseline.sbatch)
J21=$(sub "$J0" slurm/21_feast_depthmatched.sbatch)
J22=$(sub "$J0" slurm/22_feast_emp_excl.sbatch)
J30=$(sub "$J0" slurm/30_isa_networks_decontam.sbatch)
J40=$(sub "$J0" slurm/40_da.sbatch)
J50=$(sub "$J0" slurm/50_classifier.sbatch)
J60=$(sub "$J20:$J30" slurm/60_gradient.sbatch)
J70=$(sub "$J0" slurm/70_contam_sensitivity.sbatch)
printf 'fulldepth\t%s\nprep\t%s\nordination\t%s\nfeast_baseline\t%s\nfeast_depthmatched\t%s\nfeast_emp_excl\t%s\nisa_networks_decontam\t%s\nda\t%s\nclassifier\t%s\ngradient\t%s\ncontam_sensitivity\t%s\n' "$J5" "$J0" "$J10" "$J20" "$J21" "$J22" "$J30" "$J40" "$J50" "$J60" "$J70" | tee jobs.tsv
echo "submitted $(date). Check with: bash slurm/check.sh"
