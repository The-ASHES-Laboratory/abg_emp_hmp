#!/bin/bash
# check.sh -- one-shot status of all submitted jobs plus the first FATAL line of any failed log.
P=${ABG_P:-$PWD}; cd "$P" || exit 1
IDS=$(cut -f2 jobs.tsv | paste -sd, -)
sacct -j "$IDS" --format=JobID,JobName%16,State%12,Elapsed,MaxRSS,AllocCPUS,NodeList -X
echo "--- FATAL lines (if any) ---"; grep -H "FATAL" logs/*.log 2>/dev/null | head -20
echo "--- outputs present ---"; for f in results/upstream/table.qza results/upstream/taxonomy_silva.qza results/compositional/merged_L6genus_original.tsv results/compositional/merged_L6genus.tsv results/ordination/table2_permanova_permdisp.tsv results/feast_split/burial_source_props.tsv results/diff_abun/da_consensus_oriented.tsv results/sensitivity/isa_results.tsv results/classifier/burial_predictions.csv results/gradient/gradient_tests.tsv results/feast_depth_matched/r500/burial_source_props.tsv results/emp_reference/feast/burial_source_props.tsv; do [ -s "$f" ] && echo "  OK   $f" || echo "  --   $f"; done
