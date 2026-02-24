# 02_alpha_diversity.py
# By Carter Clinton, Ph.D.
"""
Compute alpha diversity metrics for the NYABG split pipeline.

Reads the genus-level count table (QIIME format), computes four alpha-diversity
metrics per sample (Shannon H', Simpson 1-D, Chao1, Observed OTUs), merges with
cohort metadata, then runs Kruskal-Wallis (global) and pairwise Mann-Whitney U
tests with Benjamini-Hochberg correction and Cliff's delta effect sizes.

Outputs
-------
<out-dir>/alpha_values.tsv   per-sample diversity values + cohort label
<out-dir>/alpha_stats.tsv    statistical test results
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Make sibling modules importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from utils import read_qiime_tsv

# ---------------------------------------------------------------------------
# Alpha-diversity functions (operate on a 1-D array of raw counts)
# ---------------------------------------------------------------------------

def shannon(counts):
    """Shannon entropy H' (natural log)."""
    p = counts[counts > 0] / counts.sum()
    return -np.sum(p * np.log(p))


def simpson(counts):
    """Simpson diversity index (1 - D)."""
    p = counts[counts > 0] / counts.sum()
    return 1.0 - np.sum(p ** 2)


def chao1(counts):
    """Chao1 richness estimator."""
    s_obs = np.sum(counts > 0)
    f1 = np.sum(counts == 1)
    f2 = max(np.sum(counts == 2), 1)  # avoid division by zero
    return s_obs + (f1 ** 2) / (2 * f2)


def observed(counts):
    """Number of observed OTUs (taxa with count > 0)."""
    return np.sum(counts > 0)


# ---------------------------------------------------------------------------
# Cliff's delta effect size
# ---------------------------------------------------------------------------

def cliffs_delta(x, y):
    """
    Compute Cliff's delta, a non-parametric effect size measure.
    Values range from -1 to 1.  |d| < 0.147 negligible, < 0.33 small,
    < 0.474 medium, otherwise large  (Romano et al. 2006).
    """
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan
    # Vectorised pairwise comparison via broadcasting
    # For very large groups this can be memory-heavy; fall back to a
    # loop-based estimator if needed.
    try:
        diff = np.sign(x[:, None] - y[None, :])
        return diff.sum() / (n_x * n_y)
    except MemoryError:
        # Subsample for huge groups
        rng = np.random.default_rng(42)
        max_n = 5000
        xs = rng.choice(x, min(max_n, n_x), replace=False)
        ys = rng.choice(y, min(max_n, n_y), replace=False)
        diff = np.sign(xs[:, None] - ys[None, :])
        return diff.sum() / (len(xs) * len(ys))


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR correction
# ---------------------------------------------------------------------------

def bh_correction(pvals):
    """Return BH-adjusted p-values (q-values) for an array of p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals.copy()
    order = np.argsort(pvals)
    ranked = np.empty_like(pvals)
    ranked[order] = np.arange(1, n + 1)
    qvals = pvals * n / ranked
    # enforce monotonicity (step-up)
    qvals[order[::-1]] = np.minimum.accumulate(qvals[order[::-1]])
    return np.clip(qvals, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute alpha diversity and run cohort comparisons."
    )
    parser.add_argument(
        "--table",
        default="results/compositional/merged_L6genus.tsv",
        help="QIIME-format genus count table (taxa x samples).",
    )
    parser.add_argument(
        "--metadata",
        default="results/metadata_cohort_split.tsv",
        help="Metadata TSV with columns sample_id and cohort.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/alpha_diversity/",
        help="Output directory for result files.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load genus table (taxa x samples) — raw counts, NOT CLR
    # ------------------------------------------------------------------
    print("Loading genus table ...", flush=True)
    table = read_qiime_tsv(args.table)   # taxa (rows) x samples (cols)
    n_taxa, n_samp = table.shape
    print(f"  {n_taxa} taxa x {n_samp} samples loaded.", flush=True)

    # ------------------------------------------------------------------
    # 2. Load metadata and restrict to samples present in the table
    # ------------------------------------------------------------------
    print("Loading metadata ...", flush=True)
    meta = pd.read_csv(args.metadata, sep="\t", dtype=str)
    meta.columns = [c.strip() for c in meta.columns]
    id_col = meta.columns[0]   # first column is sample_id
    meta = meta.rename(columns={id_col: "sample_id"})
    meta = meta.set_index("sample_id")

    common = sorted(set(table.columns) & set(meta.index))
    print(f"  {len(common)} samples in both table and metadata.", flush=True)
    if len(common) == 0:
        sys.exit("ERROR: no overlapping sample IDs between table and metadata.")

    table = table[common]
    meta = meta.loc[common]

    # ------------------------------------------------------------------
    # 3. Compute alpha diversity per sample
    # ------------------------------------------------------------------
    metrics = {
        "shannon": shannon,
        "simpson": simpson,
        "chao1": chao1,
        "observed": observed,
    }

    results = {m: np.empty(len(common)) for m in metrics}

    print(f"Computing alpha diversity for {len(common)} samples ...", flush=True)
    report_interval = max(len(common) // 10, 1)
    for i, sid in enumerate(common):
        counts = table[sid].values.astype(float)
        for mname, mfunc in metrics.items():
            results[mname][i] = mfunc(counts)
        if (i + 1) % report_interval == 0:
            print(f"  processed {i + 1}/{len(common)} samples", flush=True)

    alpha_df = pd.DataFrame(results, index=common)
    alpha_df.index.name = "sample_id"
    alpha_df["cohort"] = meta["cohort"].values

    # Reorder columns: sample_id (index), cohort, then metrics
    alpha_df = alpha_df[["cohort", "shannon", "simpson", "chao1", "observed"]]

    out_values = os.path.join(args.out_dir, "alpha_values.tsv")
    alpha_df.to_csv(out_values, sep="\t")
    print(f"Wrote {out_values} ({len(alpha_df)} rows).", flush=True)

    # ------------------------------------------------------------------
    # 4. Kruskal-Wallis across all 4 cohorts
    # ------------------------------------------------------------------
    print("Running Kruskal-Wallis tests ...", flush=True)
    cohorts = sorted(alpha_df["cohort"].unique())
    print(f"  Cohorts: {cohorts}", flush=True)

    stat_rows = []
    for mname in metrics:
        groups = [alpha_df.loc[alpha_df["cohort"] == c, mname].values for c in cohorts]
        H, p = stats.kruskal(*groups)
        means = {c: g.mean() for c, g in zip(cohorts, groups)}
        stat_rows.append({
            "metric": mname,
            "test": "kruskal_wallis",
            "comparison": "all_cohorts",
            "stat": H,
            "p_value": p,
            "q_value": np.nan,   # filled after BH
            "effect_size": np.nan,
            "mean1": np.nan,
            "mean2": np.nan,
        })

    # ------------------------------------------------------------------
    # 5. Pairwise Mann-Whitney U with Cliff's delta
    # ------------------------------------------------------------------
    pairs = [
        ("burial", "control"),
        ("burial", "emp"),
        ("burial", "hmp"),
        ("control", "emp"),
        ("control", "hmp"),
    ]
    print(f"Running pairwise Mann-Whitney U for {len(pairs)} pairs ...", flush=True)

    for mname in metrics:
        for c1, c2 in pairs:
            x = alpha_df.loc[alpha_df["cohort"] == c1, mname].values
            y = alpha_df.loc[alpha_df["cohort"] == c2, mname].values
            if len(x) == 0 or len(y) == 0:
                print(f"  WARNING: {c1} or {c2} has 0 samples, skipping.", flush=True)
                continue
            U, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            d = cliffs_delta(x, y)
            stat_rows.append({
                "metric": mname,
                "test": "mann_whitney_u",
                "comparison": f"{c1}_vs_{c2}",
                "stat": U,
                "p_value": p,
                "q_value": np.nan,
                "effect_size": d,
                "mean1": x.mean(),
                "mean2": y.mean(),
            })

    # ------------------------------------------------------------------
    # 6. BH correction across all pairwise tests (per metric)
    # ------------------------------------------------------------------
    stats_df = pd.DataFrame(stat_rows)

    # Apply BH correction within each metric for pairwise tests only
    for mname in metrics:
        mask = (stats_df["metric"] == mname) & (stats_df["test"] == "mann_whitney_u")
        if mask.sum() == 0:
            continue
        pvals = stats_df.loc[mask, "p_value"].values
        stats_df.loc[mask, "q_value"] = bh_correction(pvals)

    # Also apply BH to the Kruskal-Wallis p-values (4 tests)
    kw_mask = stats_df["test"] == "kruskal_wallis"
    if kw_mask.sum() > 0:
        stats_df.loc[kw_mask, "q_value"] = bh_correction(
            stats_df.loc[kw_mask, "p_value"].values
        )

    out_stats = os.path.join(args.out_dir, "alpha_stats.tsv")
    stats_df.to_csv(out_stats, sep="\t", index=False)
    print(f"Wrote {out_stats} ({len(stats_df)} rows).", flush=True)

    # ------------------------------------------------------------------
    # 7. Summary to stdout
    # ------------------------------------------------------------------
    print("\n--- Cohort means ---", flush=True)
    for mname in metrics:
        print(f"\n  {mname}:", flush=True)
        for c in cohorts:
            vals = alpha_df.loc[alpha_df["cohort"] == c, mname]
            print(f"    {c:>10s}: mean={vals.mean():.4f}  sd={vals.std():.4f}  n={len(vals)}", flush=True)

    print("\n--- Kruskal-Wallis ---", flush=True)
    for _, row in stats_df[stats_df["test"] == "kruskal_wallis"].iterrows():
        print(f"  {row['metric']:>10s}: H={row['stat']:.2f}  p={row['p_value']:.2e}  q={row['q_value']:.2e}", flush=True)

    print("\n--- Significant pairwise tests (q < 0.05) ---", flush=True)
    sig = stats_df[(stats_df["test"] == "mann_whitney_u") & (stats_df["q_value"] < 0.05)]
    if len(sig) == 0:
        print("  (none)", flush=True)
    else:
        for _, row in sig.iterrows():
            print(
                f"  {row['metric']:>10s} | {row['comparison']:>20s} | "
                f"U={row['stat']:.0f}  q={row['q_value']:.2e}  "
                f"delta={row['effect_size']:.3f}  "
                f"mean1={row['mean1']:.4f}  mean2={row['mean2']:.4f}",
                flush=True,
            )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
