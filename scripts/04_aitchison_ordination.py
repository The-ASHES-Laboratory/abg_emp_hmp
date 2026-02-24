# 04_aitchison_ordination.py
# By Carter Clinton, Ph.D.
"""
Aitchison Distance Ordination & Statistical Tests

Computes Aitchison distance (CLR + Euclidean) PCoA ordination, PERMANOVA,
and ANOSIM with stratified subsampling and replicate stability analysis.
This complements the Bray-Curtis ordination in beta_diversity_ordination.py
by providing a compositionally-aware alternative.

Inputs:
  - results/compositional/merged_L6genus.tsv   (genus-level count table)
  - results/metadata_cohort_split.tsv           (sample metadata with cohort)

Outputs:
  - results/ordination/aitchison_pcoa_coordinates.tsv
  - results/ordination/aitchison_permanova_global.tsv
  - results/ordination/aitchison_permanova_pairwise.tsv
  - results/ordination/aitchison_anosim.tsv
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa
from skbio.stats.distance import permanova, anosim

# Make sibling modules importable
sys.path.insert(0, os.path.dirname(__file__))
from utils import read_qiime_tsv, clr_transform

warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", default="results/compositional/merged_L6genus.tsv",
                   help="QIIME-format genus count table (taxa x samples).")
    p.add_argument("--metadata", default="results/metadata_cohort_split.tsv",
                   help="Metadata TSV with sample_id and cohort columns.")
    p.add_argument("--out-dir", default="results/ordination/",
                   help="Output directory for result files.")
    p.add_argument("--n-subsample", type=int, default=500,
                   help="Number of EMP / HMP samples to subsample per replicate.")
    p.add_argument("--n-replicates", type=int, default=5,
                   help="Number of subsampling replicates.")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed for reproducibility.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Subsampling (mirrors beta_diversity_ordination.py)
# ---------------------------------------------------------------------------

def stratified_subsample(metadata, cohort, subsite_col, n_total, rng):
    """Subsample n_total samples from cohort, stratified by subsite."""
    cohort_meta = metadata[metadata["cohort"] == cohort]
    subsites = cohort_meta[subsite_col].value_counts()
    # Proportional allocation
    n_per_subsite = (subsites / subsites.sum() * n_total).round().astype(int)
    # Ensure total is exactly n_total
    while n_per_subsite.sum() > n_total:
        n_per_subsite[n_per_subsite.idxmax()] -= 1
    while n_per_subsite.sum() < n_total:
        n_per_subsite[n_per_subsite.idxmin()] += 1

    sampled = []
    for subsite, n in n_per_subsite.items():
        pool = cohort_meta[cohort_meta[subsite_col] == subsite].index.tolist()
        n_take = min(n, len(pool))
        sampled.extend(rng.choice(pool, size=n_take, replace=False).tolist())
    return sampled


def get_subsample(metadata, n_sub, rng):
    """Get subsampled IDs: subsample EMP+HMP, keep all burial+control."""
    emp_ids = stratified_subsample(metadata, "emp", "cohort_subsite", n_sub, rng)
    hmp_ids = stratified_subsample(metadata, "hmp", "cohort_subsite", n_sub, rng)
    burial_ids = metadata[metadata["cohort"] == "burial"].index.tolist()
    control_ids = metadata[metadata["cohort"] == "control"].index.tolist()
    return emp_ids + hmp_ids + burial_ids + control_ids


# ---------------------------------------------------------------------------
# Aitchison distance (CLR + Euclidean)
# ---------------------------------------------------------------------------

def compute_aitchison_dm(table, sample_ids):
    """Compute Aitchison distance matrix for selected samples.

    Applies CLR transform (pseudo=0.5) and then Euclidean distance.
    Returns a skbio DistanceMatrix.
    """
    valid = [s for s in sample_ids if s in table.columns]
    sub = table[valid]  # taxa x samples

    # Drop samples with zero total counts
    col_sums = sub.sum(axis=0)
    nonzero = col_sums[col_sums > 0].index.tolist()
    if len(nonzero) < len(valid):
        print(f"  Dropped {len(valid) - len(nonzero)} samples with zero counts",
              flush=True)
    valid = nonzero
    sub = sub[valid]

    # CLR transform (returns taxa x samples DataFrame)
    clr_data = clr_transform(sub, pseudo=0.5)

    # Euclidean distance on CLR-transformed data (samples in rows)
    dists = squareform(pdist(clr_data.T.values, metric="euclidean"))

    dm = DistanceMatrix(dists, ids=valid)
    return dm


# ---------------------------------------------------------------------------
# BH FDR correction
# ---------------------------------------------------------------------------

def bh_adjust(pvals):
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return []
    sorted_idx = np.argsort(pvals)
    adjusted = np.zeros(n)
    for rank, idx in enumerate(sorted_idx):
        adjusted[idx] = pvals[idx] * n / (rank + 1)
    # Enforce monotonicity
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(adjusted[sorted_idx[i]],
                                      adjusted[sorted_idx[i + 1]])
    return np.minimum(adjusted, 1.0)


# ---------------------------------------------------------------------------
# Single replicate
# ---------------------------------------------------------------------------

def run_single_replicate(table, metadata, n_sub, rng):
    """Run one replicate: subsample, CLR+Euclidean distance, PCoA, tests."""
    sample_ids = get_subsample(metadata, n_sub, rng)
    dm = compute_aitchison_dm(table, sample_ids)

    valid = list(dm.ids)
    groups = metadata.loc[metadata.index.isin(valid), "cohort"]
    grouping = pd.Series(groups, index=groups.index)

    # PCoA
    pc = pcoa(dm)
    coords = pc.samples.copy()
    coords.index = valid
    coords["cohort"] = [groups.get(s, "") for s in valid]
    prop_explained = pc.proportion_explained

    # Global PERMANOVA
    perm_global = permanova(dm, grouping, permutations=999)

    # Pairwise PERMANOVA
    cohort_list = sorted(groups.unique())
    pairwise_rows = []
    for i in range(len(cohort_list)):
        for j in range(i + 1, len(cohort_list)):
            c1, c2 = cohort_list[i], cohort_list[j]
            pair_ids = [s for s in valid if groups.get(s) in {c1, c2}]
            if len(pair_ids) < 4:
                continue
            pair_dm = dm.filter(pair_ids)
            pair_groups = grouping.loc[grouping.index.isin(pair_ids)]
            try:
                pair_perm = permanova(pair_dm, pair_groups, permutations=999)
                pseudo_F = pair_perm["test statistic"]
                k = len({c1, c2})
                N = pair_perm["sample size"]
                R2 = pseudo_F * (k - 1) / (pseudo_F * (k - 1) + (N - k))
                pairwise_rows.append({
                    "comparison": f"{c1}_vs_{c2}",
                    "R2": R2,
                    "p_value": pair_perm["p-value"],
                    "n_samples": N,
                })
            except Exception as e:
                print(f"  Skipping pairwise {c1} vs {c2}: {e}", flush=True)

    # ANOSIM
    try:
        anosim_result = anosim(dm, grouping, permutations=999)
        anosim_R = anosim_result["test statistic"]
        anosim_p = anosim_result["p-value"]
    except Exception:
        anosim_R, anosim_p = float("nan"), float("nan")

    global_F = perm_global["test statistic"]
    k_global = len(cohort_list)
    N_global = len(valid)
    global_R2 = global_F * (k_global - 1) / (global_F * (k_global - 1) + (N_global - k_global))

    return {
        "coords": coords,
        "prop_explained": prop_explained,
        "global_R2": global_R2,
        "global_p": perm_global["p-value"],
        "global_F": perm_global["test statistic name"],
        "pairwise": pairwise_rows,
        "anosim_R": anosim_R,
        "anosim_p": anosim_p,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load genus table
    # ------------------------------------------------------------------
    print("Loading genus table ...", flush=True)
    table = read_qiime_tsv(args.table)
    print(f"  {table.shape[0]} taxa x {table.shape[1]} samples", flush=True)

    # ------------------------------------------------------------------
    # 2. Load metadata
    # ------------------------------------------------------------------
    print("Loading metadata ...", flush=True)
    meta = pd.read_csv(args.metadata, sep="\t", dtype=str, index_col=0)
    meta = meta[meta.index.isin(table.columns)]
    cohort_counts = meta["cohort"].value_counts()
    print(f"  {len(meta)} samples in metadata overlap with table", flush=True)
    for c in sorted(cohort_counts.index):
        print(f"    {c}: {cohort_counts[c]}", flush=True)

    # ------------------------------------------------------------------
    # 3. Run replicates
    # ------------------------------------------------------------------
    all_results = []
    for rep in range(args.n_replicates):
        seed = args.seed + rep
        rng = np.random.default_rng(seed)
        print(f"\nReplicate {rep + 1}/{args.n_replicates} (seed={seed}) ...",
              flush=True)
        result = run_single_replicate(table, meta, args.n_subsample, rng)
        all_results.append(result)
        print(f"  Global PERMANOVA: R2={result['global_R2']:.4f}, "
              f"p={result['global_p']:.4f}", flush=True)
        print(f"  ANOSIM: R={result['anosim_R']:.4f}, "
              f"p={result['anosim_p']:.4f}", flush=True)

    # ------------------------------------------------------------------
    # 4. Choose median replicate for coordinate output
    # ------------------------------------------------------------------
    global_R2_vals = [r["global_R2"] for r in all_results]
    median_idx = int(np.argsort(global_R2_vals)[len(global_R2_vals) // 2])
    print(f"\nMedian replicate: {median_idx + 1} "
          f"(R2={global_R2_vals[median_idx]:.4f})", flush=True)

    # Save PCoA coordinates from median replicate
    coords = all_results[median_idx]["coords"]
    # Keep only PC1, PC2, PC3 and cohort
    pc_cols = [c for c in coords.columns if c.startswith("PC") or c == "cohort"]
    pc_keep = []
    for c in pc_cols:
        if c == "cohort":
            pc_keep.append(c)
        elif c in ("PC1", "PC2", "PC3"):
            pc_keep.append(c)
    # Reorder: cohort first, then PCs
    out_coords = coords[["cohort", "PC1", "PC2", "PC3"]].copy()
    out_coords.index.name = "sample_id"
    out_coords.to_csv(
        os.path.join(args.out_dir, "aitchison_pcoa_coordinates.tsv"), sep="\t"
    )
    print(f"Wrote aitchison_pcoa_coordinates.tsv ({len(out_coords)} samples)",
          flush=True)

    # ------------------------------------------------------------------
    # 5. Save global PERMANOVA (per replicate)
    # ------------------------------------------------------------------
    global_rows = []
    for rep, r in enumerate(all_results):
        global_rows.append({
            "replicate": rep + 1,
            "R2": r["global_R2"],
            "p_value": r["global_p"],
            "F_stat": r["global_F"],
        })
    global_df = pd.DataFrame(global_rows)
    global_df.to_csv(
        os.path.join(args.out_dir, "aitchison_permanova_global.tsv"),
        sep="\t", index=False
    )
    print(f"Wrote aitchison_permanova_global.tsv ({len(global_df)} rows)",
          flush=True)

    # ------------------------------------------------------------------
    # 6. Save pairwise PERMANOVA (aggregate across replicates)
    # ------------------------------------------------------------------
    comp_data = {}
    for r in all_results:
        for pw in r["pairwise"]:
            c = pw["comparison"]
            if c not in comp_data:
                comp_data[c] = {"R2": [], "p": []}
            comp_data[c]["R2"].append(pw["R2"])
            comp_data[c]["p"].append(pw["p_value"])

    pw_rows = []
    for comp, vals in sorted(comp_data.items()):
        pw_rows.append({
            "comparison": comp,
            "R2_mean": np.mean(vals["R2"]),
            "R2_sd": np.std(vals["R2"]),
            "p_mean": np.mean(vals["p"]),
            "p_sd": np.std(vals["p"]),
        })
    pw_df = pd.DataFrame(pw_rows)
    if len(pw_df) > 0:
        pw_df["p_BH"] = bh_adjust(pw_df["p_mean"].values)
    pw_df.to_csv(
        os.path.join(args.out_dir, "aitchison_permanova_pairwise.tsv"),
        sep="\t", index=False
    )
    print(f"Wrote aitchison_permanova_pairwise.tsv ({len(pw_df)} rows)",
          flush=True)

    # ------------------------------------------------------------------
    # 7. Save ANOSIM (per replicate)
    # ------------------------------------------------------------------
    anosim_rows = []
    for rep, r in enumerate(all_results):
        anosim_rows.append({
            "replicate": rep + 1,
            "R": r["anosim_R"],
            "p_value": r["anosim_p"],
        })
    anosim_df = pd.DataFrame(anosim_rows)
    anosim_df.to_csv(
        os.path.join(args.out_dir, "aitchison_anosim.tsv"),
        sep="\t", index=False
    )
    print(f"Wrote aitchison_anosim.tsv ({len(anosim_df)} rows)", flush=True)

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("AITCHISON ORDINATION SUMMARY", flush=True)
    print("=" * 60, flush=True)

    print(f"\nGlobal PERMANOVA across {args.n_replicates} replicates:", flush=True)
    print(f"  R2:  mean={np.mean(global_R2_vals):.4f}  "
          f"sd={np.std(global_R2_vals):.4f}  "
          f"range=[{np.min(global_R2_vals):.4f}, {np.max(global_R2_vals):.4f}]",
          flush=True)
    global_p_vals = [r["global_p"] for r in all_results]
    print(f"  p:   mean={np.mean(global_p_vals):.4f}  "
          f"range=[{np.min(global_p_vals):.4f}, {np.max(global_p_vals):.4f}]",
          flush=True)

    print(f"\nANOSIM across {args.n_replicates} replicates:", flush=True)
    anosim_R_vals = [r["anosim_R"] for r in all_results]
    anosim_p_vals = [r["anosim_p"] for r in all_results]
    print(f"  R:   mean={np.nanmean(anosim_R_vals):.4f}  "
          f"sd={np.nanstd(anosim_R_vals):.4f}", flush=True)
    print(f"  p:   mean={np.nanmean(anosim_p_vals):.4f}", flush=True)

    if len(pw_df) > 0:
        print("\nPairwise PERMANOVA (BH-adjusted):", flush=True)
        for _, row in pw_df.iterrows():
            sig = "***" if row["p_BH"] < 0.001 else \
                  "**" if row["p_BH"] < 0.01 else \
                  "*" if row["p_BH"] < 0.05 else "ns"
            print(f"  {row['comparison']:>25s}:  R2={row['R2_mean']:.4f} "
                  f"(sd={row['R2_sd']:.4f})  p_BH={row['p_BH']:.4f} {sig}",
                  flush=True)

    # Proportion explained for median replicate
    prop = all_results[median_idx]["prop_explained"]
    print(f"\nPCoA variance explained (median replicate):", flush=True)
    for i, ax in enumerate(["PC1", "PC2", "PC3"]):
        if i < len(prop):
            print(f"  {ax}: {prop.iloc[i] * 100:.1f}%", flush=True)

    print(f"\nAll Aitchison ordination outputs written to {args.out_dir}",
          flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
