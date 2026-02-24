# 14_rarefaction_sweep.py
# By Carter Clinton, Ph.D.
"""
Rarefaction depth sensitivity analysis.

Rarefies the genus table at multiple sequencing depths (500, 1000, 2000,
5000, 10000) and re-runs IndVal-based taxon source assignments at each
depth.  Samples falling below the depth threshold are dropped.

Output: results/sensitivity/rarefaction_sweep.tsv
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
        default="results/sensitivity/rarefaction_sweep.tsv",
        help="Output TSV path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def rarefy(counts, depth, rng):
    """Rarefy a single sample (1-D array) to a given depth.

    Uses multinomial subsampling without replacement (standard approach).
    Returns None if the sample's total count is below `depth`.
    """
    total = counts.sum()
    if total < depth:
        return None
    p = counts / total
    return rng.multinomial(depth, p)


def rarefy_table(table, depth, rng):
    """Rarefy all samples in a taxa x samples DataFrame.

    Returns a new DataFrame with the same index/columns but rarefied counts.
    Samples below the depth threshold are dropped.
    """
    rarefied = {}
    dropped = 0
    for sid in table.columns:
        counts = table[sid].values.astype(float)
        result = rarefy(counts, depth, rng)
        if result is None:
            dropped += 1
        else:
            rarefied[sid] = result
    if not rarefied:
        return pd.DataFrame(index=table.index), dropped
    out = pd.DataFrame(rarefied, index=table.index)
    return out, dropped


def compute_prevalence(table, sample_ids):
    """Fraction of samples in which each taxon is present (count > 0)."""
    sub = table[table.columns.intersection(sample_ids)]
    if sub.empty:
        return pd.Series(dtype=float)
    return (sub > 0).sum(axis=1) / sub.shape[1]


def assign_taxa(table, cohort_samples):
    """Assign taxa to EMP, HMP, or ambiguous via IndVal scores.

    Uses a default prevalence filter of 10% in at least one cohort (the
    pipeline default), then computes IndVal = prev_cohort / (prev_EMP + prev_HMP).
    """
    prev = {}
    for cohort, sids in cohort_samples.items():
        prev[cohort] = compute_prevalence(table, sids)

    prev_df = pd.DataFrame(prev)
    # Default 10% prevalence filter
    mask = prev_df.max(axis=1) >= 0.10
    prev_df = prev_df.loc[mask]

    n_taxa = len(prev_df)
    if n_taxa == 0:
        return {"n_taxa": 0, "n_emp": 0, "n_hmp": 0, "n_ambiguous": 0}

    emp_prev = prev_df["emp"].values
    hmp_prev = prev_df["hmp"].values
    denom = emp_prev + hmp_prev
    safe_denom = np.where(denom > 0, denom, 1.0)

    indval_emp = emp_prev / safe_denom
    indval_hmp = hmp_prev / safe_denom

    n_emp = int(np.sum(indval_emp > 0.6))
    n_hmp = int(np.sum(indval_hmp > 0.6))
    n_ambiguous = n_taxa - n_emp - n_hmp

    return {
        "n_taxa": n_taxa,
        "n_emp": n_emp,
        "n_hmp": n_hmp,
        "n_ambiguous": n_ambiguous,
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("Loading genus table ...", flush=True)
    table = read_qiime_tsv(args.table)
    print(f"  Table shape: {table.shape[0]} taxa x {table.shape[1]} samples", flush=True)

    print("Loading metadata ...", flush=True)
    meta = pd.read_csv(args.metadata, sep="\t", dtype=str)
    id_col = [c for c in meta.columns if c.lower().replace("-", "_") in ("sample_id", "sampleid")][0]
    meta = meta.rename(columns={id_col: "sample_id"})
    meta["cohort"] = meta["cohort"].str.strip().str.lower()

    # Map sample -> cohort (only samples present in table)
    sample_cohort = {}
    for _, row in meta.iterrows():
        sid = row["sample_id"]
        if sid in table.columns:
            sample_cohort[sid] = row["cohort"]

    depths = [500, 1000, 2000, 5000, 10000]
    results = []

    for depth in depths:
        print(f"\nRarefaction depth = {depth} ...", flush=True)

        # Rarefy
        rarefied, n_dropped = rarefy_table(table, depth, rng)
        print(f"  Dropped {n_dropped} samples below depth", flush=True)

        if rarefied.empty:
            print("  No samples remaining -- skipping", flush=True)
            results.append({
                "depth": depth,
                "n_burial": 0,
                "n_control": 0,
                "n_emp": 0,
                "n_hmp": 0,
                "n_taxa": 0,
                "n_emp_assigned": 0,
                "n_hmp_assigned": 0,
                "n_ambiguous": 0,
            })
            continue

        # Build cohort sample lists from surviving samples
        cohort_samples = {"burial": [], "control": [], "emp": [], "hmp": []}
        for sid in rarefied.columns:
            cohort = sample_cohort.get(sid)
            if cohort in cohort_samples:
                cohort_samples[cohort].append(sid)

        for cohort, sids in cohort_samples.items():
            print(f"  {cohort}: {len(sids)} samples remaining", flush=True)

        # Assign taxa
        res = assign_taxa(rarefied, cohort_samples)
        print(
            f"  {res['n_taxa']} taxa -> {res['n_emp']} EMP, "
            f"{res['n_hmp']} HMP, {res['n_ambiguous']} ambiguous",
            flush=True,
        )

        results.append({
            "depth": depth,
            "n_burial": len(cohort_samples["burial"]),
            "n_control": len(cohort_samples["control"]),
            "n_emp": len(cohort_samples["emp"]),
            "n_hmp": len(cohort_samples["hmp"]),
            "n_taxa": res["n_taxa"],
            "n_emp_assigned": res["n_emp"],
            "n_hmp_assigned": res["n_hmp"],
            "n_ambiguous": res["n_ambiguous"],
        })

    # Write output
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, sep="\t", index=False)
    print(f"\nResults written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
