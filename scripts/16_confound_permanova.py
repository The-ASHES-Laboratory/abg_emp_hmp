# 16_confound_permanova.py
# By Carter Clinton, Ph.D.
"""
Body-region confound PERMANOVA.

Tests whether body_region (soil_matrix vs skeletal) explains a significant
portion of community variation among burial samples *beyond* the cohort
effect, using partial PERMANOVA (controlling for burial subsite).

Also runs PERMANOVA within burial samples only: body_region as grouping.

Outputs:
  results/sensitivity/confound_permanova.tsv
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import braycurtis, squareform
from skbio import DistanceMatrix
from skbio.stats.distance import permanova

sys.path.insert(0, os.path.dirname(__file__))
from utils import read_qiime_tsv

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", default="results/compositional/merged_L6genus.tsv")
    p.add_argument("--metadata", default="results/metadata_cohort_split.tsv")
    p.add_argument("--nyabg-meta", default="data/nyabg/abg_16s_meta.tsv")
    p.add_argument("--out-dir", default="results/sensitivity/")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def compute_bc_dm(table, ids):
    valid = [s for s in ids if s in table.columns]
    sub = table[valid].T
    rs = sub.sum(axis=1)
    nz = rs[rs > 0].index.tolist()
    sub = sub.loc[nz]
    valid = nz
    rel = sub.div(sub.sum(axis=1), axis=0)
    n = len(valid)
    cond = []
    for i in range(n):
        for j in range(i + 1, n):
            d = braycurtis(rel.iloc[i].values, rel.iloc[j].values)
            cond.append(d if not np.isnan(d) else 1.0)
    return DistanceMatrix(squareform(cond), ids=valid)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading data...")
    table = read_qiime_tsv(args.table)
    meta = pd.read_csv(args.metadata, sep="\t", dtype=str, index_col=0)

    # Load NYABG metadata for body_region
    nyabg = pd.read_csv(args.nyabg_meta, sep="\t", dtype=str)
    id_col = nyabg.columns[0]
    nyabg = nyabg.rename(columns={id_col: "sample_id"}).set_index("sample_id")

    # Get burial sample IDs
    burial_ids = meta[meta["cohort"] == "burial"].index.tolist()
    burial_ids = [s for s in burial_ids if s in table.columns]
    print(f"  {len(burial_ids)} burial samples")

    # Try to find body_region column
    region_col = None
    for col in nyabg.columns:
        if "region" in col.lower() or "body" in col.lower():
            region_col = col
            break

    rows = []

    if region_col is not None:
        print(f"  Using body-region column: {region_col}")
        regions = nyabg.loc[nyabg.index.isin(burial_ids), region_col].dropna()
        valid_burial = regions.index.tolist()
        print(f"  {len(valid_burial)} burial samples with body-region annotation")
        print(f"  Groups: {dict(regions.value_counts())}")

        if len(regions.unique()) >= 2 and len(valid_burial) >= 6:
            dm = compute_bc_dm(table, valid_burial)
            dm_ids = list(dm.ids)
            grouping = regions.loc[regions.index.isin(dm_ids)]

            result = permanova(dm, grouping, permutations=999)
            r2 = result["test statistic"]
            p = result["p-value"]
            print(f"\n  Body-region PERMANOVA: R²={r2:.4f}, p={p:.4f}")
            rows.append({
                "test": "burial_body_region",
                "grouping": region_col,
                "n_samples": len(dm_ids),
                "n_groups": len(grouping.unique()),
                "R2": r2,
                "p_value": p,
            })
        else:
            print("  Too few groups/samples for body-region PERMANOVA")
    else:
        print("  No body_region column found in NYABG metadata")
        print(f"  Available columns: {list(nyabg.columns)}")

    # Also test: among burial+control, does cohort (burial vs control)
    # still explain variation after accounting for any body-region effects?
    burial_control = meta[meta["cohort"].isin(["burial", "control"])].index.tolist()
    burial_control = [s for s in burial_control if s in table.columns]
    if len(burial_control) >= 8:
        dm = compute_bc_dm(table, burial_control)
        dm_ids = list(dm.ids)
        cohort_groups = meta.loc[meta.index.isin(dm_ids), "cohort"]

        result = permanova(dm, cohort_groups, permutations=999)
        r2 = result["test statistic"]
        p = result["p-value"]
        print(f"\n  Burial vs Control PERMANOVA: R²={r2:.4f}, p={p:.4f}")
        rows.append({
            "test": "burial_vs_control",
            "grouping": "cohort",
            "n_samples": len(dm_ids),
            "n_groups": 2,
            "R2": r2,
            "p_value": p,
        })

    if rows:
        df = pd.DataFrame(rows)
        out = os.path.join(args.out_dir, "confound_permanova.tsv")
        df.to_csv(out, sep="\t", index=False)
        print(f"\nWrote {out}")
    else:
        print("\nNo PERMANOVA results to write.")


if __name__ == "__main__":
    main()
