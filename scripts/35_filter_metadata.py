#!/usr/bin/env python3
"""35_filter_metadata.py -- write the EMP human-host-excluded metadata for the sensitivity analyses.
Removes the sample-ids listed in --exclude (2,947 Homo sapiens host samples + 2 sterile-water blanks) from --metadata."""
import argparse, os, pandas as pd
ap=argparse.ArgumentParser(); ap.add_argument("--metadata", default="results/metadata_cohort_split.tsv"); ap.add_argument("--exclude", default="data/emp/emp_exclusion_human_and_blanks.tsv"); ap.add_argument("--out", default="results/emp_reference/metadata_cohort_split_noHumanEMP.tsv")
a=ap.parse_args(); m=pd.read_csv(a.metadata, sep="\t", dtype=str); ex=set(pd.read_csv(a.exclude, sep="\t", dtype=str)["sample-id"])
os.makedirs(os.path.dirname(a.out), exist_ok=True); o=m[~m["sample-id"].isin(ex)]; o.to_csv(a.out, sep="\t", index=False)
print(f"excluded {len(m)-len(o)} samples; remaining {len(o)}; cohorts: {o.cohort.value_counts().to_dict()}")
