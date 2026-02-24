# 24_sequencing_depth.py
# By Carter Clinton, Ph.D.
"""
Sequencing Depth Distribution

Per-cohort depth statistics for supplementary reporting.

Inputs:
  results/compositional/merged_L6genus.tsv
  results/metadata_cohort_split.tsv

Outputs:
  results/sequencing_depth/depth_distribution.tsv
  results/sequencing_depth/depth_summary.tsv
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from utils import read_qiime_tsv


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", default="results/compositional/merged_L6genus.tsv")
    p.add_argument("--metadata", default="results/metadata_cohort_split.tsv")
    p.add_argument("--out-dir", default="results/sequencing_depth/")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading genus table...")
    table = read_qiime_tsv(args.table)
    table = table.apply(pd.to_numeric, errors="coerce").fillna(0)

    print("Loading metadata...")
    meta = pd.read_csv(args.metadata, sep="\t", dtype=str, index_col=0)
    meta = meta[meta.index.isin(table.columns)]

    # Compute per-sample depths
    depths = table.sum(axis=0)

    # Per-sample distribution
    dist_rows = []
    for sid in meta.index:
        if sid in depths.index:
            dist_rows.append({
                "sample_id": sid,
                "cohort": meta.loc[sid, "cohort"],
                "depth": int(depths[sid]),
            })

    dist_df = pd.DataFrame(dist_rows)
    dist_path = os.path.join(args.out_dir, "depth_distribution.tsv")
    dist_df.to_csv(dist_path, sep="\t", index=False)
    print(f"  Wrote {dist_path} ({len(dist_df)} samples)")

    # Summary statistics by cohort
    summary_rows = []
    for cohort in ["burial", "control", "emp", "hmp"]:
        cohort_depths = dist_df[dist_df["cohort"] == cohort]["depth"]
        if len(cohort_depths) == 0:
            continue
        summary_rows.append({
            "cohort": cohort,
            "n_samples": len(cohort_depths),
            "mean": cohort_depths.mean(),
            "median": cohort_depths.median(),
            "sd": cohort_depths.std(),
            "min": cohort_depths.min(),
            "max": cohort_depths.max(),
            "q25": cohort_depths.quantile(0.25),
            "q75": cohort_depths.quantile(0.75),
            "below_500": (cohort_depths < 500).sum(),
            "below_1000": (cohort_depths < 1000).sum(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, "depth_summary.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"  Wrote {summary_path}")

    print("\n=== Sequencing Depth Summary ===")
    for _, row in summary_df.iterrows():
        print(f"  {row['cohort']}: n={row['n_samples']}, "
              f"median={row['median']:.0f}, range=[{row['min']:.0f}-{row['max']:.0f}], "
              f"<500: {row['below_500']}")


if __name__ == "__main__":
    main()
