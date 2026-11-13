#!/bin/bash
# 44_gradient_variants.sh
# By Carter Clinton, Ph.D.
# Proximity-trend tests (32_gradient_tests.py; 9,999 permutations, seed 42) for each FEAST
# configuration reported in Additional file 5d: depth-matched primary (sources rarefied to 500
# reads), FEAST default coverage, EMP human-host studies and blanks excluded, blank- and
# mock-derived ASVs removed, and burial-matched source depths. Run from the project root.
set -euo pipefail
OUT=${OUT:-results/revision}
META=${META:-data/nyabg/abg_16s_meta.tsv}
TAB=${TAB:-results/compositional/merged_L6genus.tsv}
run() { # key feast_dir table bodysite_dir
  local key=$1 d=$2 tab=$3 bs=${4:-$2}
  python scripts/32_gradient_tests.py --feast-burial "$d/burial_source_props.tsv" --feast-control "$d/control_source_props.tsv" \
    --feast-bodysite "$bs/burial_source_props_by_bodysite.tsv" --nyabg-meta "$META" --table "$tab" \
    --permutations 9999 --seed 42 --out-dir "$OUT/gradient_$key/"
}
run primary_r500     results/feast_depth_matched/r500            "$TAB"
run default_coverage results/feast_split                         "$TAB"
run noHumanEMP_r500  results/feast_depth_matched/r500_noHumanEMP "$TAB"
run contam_r500      results/contam_sensitivity/feast_r500       results/contam_sensitivity/merged_L6genus.tsv results/contam_sensitivity/feast_r500_bodysite
run burialmatched    results/feast_depth_matched/burialmatched   "$TAB"
