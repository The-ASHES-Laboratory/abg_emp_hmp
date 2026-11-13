#!/bin/bash
# collect.sh -- bundle results, logs, scripts and provenance into one archive for transfer to the Mac.
set -uo pipefail
P=${ABG_P:-$PWD}; cd "$P" || exit 1
STAMP=$(date +%Y%m%d_%H%M); OUT=${ABG_OUT:-$HOME}/abg_revision_results_$STAMP.tgz
( find results -type f -not -name 'merged_L6genus*' -not -name 'pcoa_coordinates*' -exec sha256sum {} \; ) > results/sha256_manifest.txt
tar czf "$OUT" --exclude='results/compositional/merged_L6genus*' --exclude='results/feast_depth_matched/merged_L6genus_*' results logs scripts scripts_new slurm envs.sh jobs.tsv
echo "ARCHIVE: $OUT ($(du -h "$OUT" | cut -f1))"; echo "scp hpc:$OUT ~/Desktop/"
