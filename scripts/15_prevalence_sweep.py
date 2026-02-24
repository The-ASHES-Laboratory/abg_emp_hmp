# 15_prevalence_sweep.py
# By Carter Clinton, Ph.D.
"""
Prevalence filter sensitivity analysis.

Re-runs IndVal-based taxon source assignments at four prevalence filter
thresholds (1%, 5%, 10%, 25%) to assess how the prevalence cutoff affects
the proportion of taxa assigned to EMP, HMP, or ambiguous.

Output: results/sensitivity/prevalence_sweep.tsv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import read_qiime_tsv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        default="results/compositional/merged_L6genus.tsv",
        help="Merged L6 genus table (TSV)",
    )
    parser.add_argument(
        "--metadata",
        default="results/metadata_cohort_split.tsv",
        help="Metadata TSV with sample-id and cohort columns",
    )
    parser.add_argument(
        "--out",
        default="results/sensitivity/prevalence_sweep.tsv",
        help="Output TSV path",
    )
    return parser.parse_args()


def compute_prevalence(table, sample_ids):
    """Fraction of samples in which each taxon is present (count > 0)."""
    sub = table[table.columns.intersection(sample_ids)]
    if sub.empty:
        return pd.Series(dtype=float)
    return (sub > 0).sum(axis=1) / sub.shape[1]


def assign_taxa(table, cohort_samples, threshold):
    """Filter taxa by prevalence threshold and assign via IndVal scores.

    Parameters
    ----------
    table : DataFrame
        Taxa (rows) x samples (columns) count table.
    cohort_samples : dict
        Mapping of cohort name to list of sample IDs.
    threshold : float
        Minimum prevalence in at least one cohort.

    Returns
    -------
    dict with keys: n_taxa, n_emp, n_hmp, n_ambiguous
    """
    # Compute per-cohort prevalence for every taxon
    prev = {}
    for cohort, sids in cohort_samples.items():
        prev[cohort] = compute_prevalence(table, sids)

    prev_df = pd.DataFrame(prev)

    # Filter: keep taxa with prevalence >= threshold in at least one cohort
    mask = prev_df.max(axis=1) >= threshold
    prev_df = prev_df.loc[mask]

    n_taxa = len(prev_df)
    if n_taxa == 0:
        return {
            "n_taxa": 0,
            "n_emp": 0,
            "n_hmp": 0,
            "n_ambiguous": 0,
        }

    # IndVal-like specificity for EMP vs HMP
    emp_prev = prev_df["emp"].values
    hmp_prev = prev_df["hmp"].values

    denom = emp_prev + hmp_prev
    # Avoid division by zero: taxa absent from both EMP and HMP
    safe_denom = np.where(denom > 0, denom, 1.0)

    indval_emp = emp_prev / safe_denom
    indval_hmp = hmp_prev / safe_denom

    n_emp = int(np.sum(indval_emp > 0.6))
    n_hmp = int(np.sum(indval_hmp > 0.6))
    # Ambiguous: neither EMP nor HMP scores > 0.6
    n_ambiguous = n_taxa - n_emp - n_hmp

    return {
        "n_taxa": n_taxa,
        "n_emp": n_emp,
        "n_hmp": n_hmp,
        "n_ambiguous": n_ambiguous,
    }


def main():
    args = parse_args()

    print("Loading genus table ...", flush=True)
    table = read_qiime_tsv(args.table)
    print(f"  Table shape: {table.shape[0]} taxa x {table.shape[1]} samples", flush=True)

    print("Loading metadata ...", flush=True)
    meta = pd.read_csv(args.metadata, sep="\t", dtype=str)
    # Normalize column name
    id_col = [c for c in meta.columns if c.lower().replace("-", "_") in ("sample_id", "sampleid")][0]
    meta = meta.rename(columns={id_col: "sample_id"})
    meta["cohort"] = meta["cohort"].str.strip().str.lower()

    # Build cohort -> sample list mapping
    cohort_samples = {}
    for cohort in ["burial", "control", "emp", "hmp"]:
        sids = meta.loc[meta["cohort"] == cohort, "sample_id"].tolist()
        # Intersect with table columns
        sids = [s for s in sids if s in table.columns]
        cohort_samples[cohort] = sids
        print(f"  {cohort}: {len(sids)} samples in table", flush=True)

    # Prevalence thresholds to sweep
    thresholds = [0.01, 0.05, 0.10, 0.25]

    results = []
    for thresh in thresholds:
        print(f"Running prevalence threshold = {thresh:.0%} ...", flush=True)
        res = assign_taxa(table, cohort_samples, thresh)
        n = res["n_taxa"]
        row = {
            "threshold": f"{thresh:.2f}",
            "n_taxa": res["n_taxa"],
            "n_emp": res["n_emp"],
            "n_hmp": res["n_hmp"],
            "n_ambiguous": res["n_ambiguous"],
            "pct_emp": f"{100 * res['n_emp'] / n:.1f}" if n > 0 else "0.0",
            "pct_hmp": f"{100 * res['n_hmp'] / n:.1f}" if n > 0 else "0.0",
            "pct_ambiguous": f"{100 * res['n_ambiguous'] / n:.1f}" if n > 0 else "0.0",
        }
        results.append(row)
        print(
            f"  {res['n_taxa']} taxa -> {res['n_emp']} EMP, "
            f"{res['n_hmp']} HMP, {res['n_ambiguous']} ambiguous",
            flush=True,
        )

    # Write output
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, sep="\t", index=False)
    print(f"\nResults written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
