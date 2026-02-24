# 03_beta_ordination.py
# By Carter Clinton, Ph.D.
"""
Beta-Diversity Ordination & PERMANOVA

Computes Bray-Curtis distance, PCoA ordination, and statistical tests
(PERMANOVA, PERMDISP, ANOSIM) with stratified subsampling and replicate
stability analysis.

Inputs:
  - results/compositional/merged_L6genus.tsv
  - results/metadata_cohort_split.tsv

Outputs:
  - results/ordination/pcoa_coordinates.tsv
  - results/ordination/permanova_global.tsv
  - results/ordination/permanova_pairwise.tsv
  - results/ordination/permdisp.tsv
  - results/ordination/anosim.tsv
  - results/ordination/stability_report.tsv
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import braycurtis, squareform
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa
from skbio.stats.distance import permanova, anosim

warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", default="results/compositional/merged_L6genus.tsv")
    p.add_argument("--metadata", default="results/metadata_cohort_split.tsv")
    p.add_argument("--n-subsample", type=int, default=500)
    p.add_argument("--n-replicates", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/ordination/")
    return p.parse_args()


def load_genus_table(path):
    """Load QIIME2-exported genus table, handling preamble lines."""
    skiprows = 0
    with open(path) as f:
        for line in f:
            if line.startswith("#") and not line.startswith("#OTU") and not line.startswith("#Feature"):
                skiprows += 1
            else:
                break
    df = pd.read_csv(path, sep="\t", skiprows=skiprows, index_col=0, low_memory=False)
    df.columns = [str(c).lstrip("#") for c in df.columns]
    if df.columns[-1].lower().startswith("taxonomy"):
        df = df.iloc[:, :-1]
    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


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


def compute_bray_curtis(table, sample_ids):
    """Compute Bray-Curtis distance matrix for selected samples."""
    valid = [s for s in sample_ids if s in table.columns]
    sub = table[valid].T  # samples x taxa
    # Remove samples with zero total counts
    row_sums = sub.sum(axis=1)
    nonzero = row_sums[row_sums > 0].index.tolist()
    if len(nonzero) < len(valid):
        print(f"  Dropped {len(valid) - len(nonzero)} samples with zero counts")
    valid = nonzero
    sub = sub.loc[valid]
    # Normalize to relative abundance (TSS)
    row_sums = sub.sum(axis=1)
    rel = sub.div(row_sums, axis=0)
    # Compute pairwise Bray-Curtis
    n = len(valid)
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            d = braycurtis(rel.iloc[i].values, rel.iloc[j].values)
            # Replace NaN with 1.0 (maximally different) as safety
            if np.isnan(d):
                d = 1.0
            condensed.append(d)
    dm = DistanceMatrix(squareform(condensed), ids=valid)
    return dm


def compute_permdisp(dm, groups, permutations=999):
    """Compute PERMDISP (betadisper equivalent) using centroid distances.

    Returns F-statistic and p-value from permutation test.
    """
    # PCoA to get coordinates
    pc = pcoa(dm)
    coords = pc.samples.values
    ids = list(dm.ids)
    unique_groups = sorted(set(groups.values()))

    # Compute centroids per group
    group_idx = {g: [] for g in unique_groups}
    for i, sid in enumerate(ids):
        g = groups.get(sid)
        if g:
            group_idx[g].append(i)

    centroids = {}
    for g, idxs in group_idx.items():
        centroids[g] = coords[idxs].mean(axis=0)

    # Distances to centroid
    dists = np.zeros(len(ids))
    for i, sid in enumerate(ids):
        g = groups.get(sid)
        if g:
            dists[i] = np.linalg.norm(coords[i] - centroids[g])

    # F-test: between-group variance of distances vs within-group
    k = len(unique_groups)
    N = len(ids)
    grand_mean = dists.mean()
    ss_between = sum(len(group_idx[g]) * (np.mean(dists[group_idx[g]]) - grand_mean) ** 2 for g in unique_groups)
    ss_within = sum(np.sum((dists[group_idx[g]] - np.mean(dists[group_idx[g]])) ** 2) for g in unique_groups)
    f_stat = (ss_between / (k - 1)) / (ss_within / (N - k)) if ss_within > 0 and N > k else 0

    # Permutation test
    count = 0
    for _ in range(permutations):
        perm_groups = np.random.permutation(list(groups.values()))
        perm_gidx = {g: [] for g in unique_groups}
        for i, g in enumerate(perm_groups):
            perm_gidx[g].append(i)
        perm_centroids = {g: coords[idxs].mean(axis=0) for g, idxs in perm_gidx.items()}
        perm_dists = np.array([np.linalg.norm(coords[i] - perm_centroids[perm_groups[i]]) for i in range(N)])
        perm_grand = perm_dists.mean()
        perm_ssb = sum(len(perm_gidx[g]) * (np.mean(perm_dists[perm_gidx[g]]) - perm_grand) ** 2 for g in unique_groups)
        perm_ssw = sum(np.sum((perm_dists[perm_gidx[g]] - np.mean(perm_dists[perm_gidx[g]])) ** 2) for g in unique_groups)
        perm_f = (perm_ssb / (k - 1)) / (perm_ssw / (N - k)) if perm_ssw > 0 and N > k else 0
        if perm_f >= f_stat:
            count += 1
    p_val = (count + 1) / (permutations + 1)
    return f_stat, p_val


def run_single_replicate(table, metadata, n_sub, rng):
    """Run one replicate: subsample, compute distances, run all tests."""
    sample_ids = get_subsample(metadata, n_sub, rng)
    dm = compute_bray_curtis(table, sample_ids)

    valid = list(dm.ids)
    groups = metadata.loc[metadata.index.isin(valid), "cohort"]
    grouping = pd.Series(groups, index=groups.index)

    # PCoA
    pc = pcoa(dm)
    coords = pc.samples.copy()
    coords.index = valid
    coords["cohort"] = [groups.get(s, "") for s in valid]

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
                print(f"  Skipping pairwise {c1} vs {c2}: {e}")

    # ANOSIM
    try:
        anosim_result = anosim(dm, grouping, permutations=999)
        anosim_R = anosim_result["test statistic"]
        anosim_p = anosim_result["p-value"]
    except Exception:
        anosim_R, anosim_p = float("nan"), float("nan")

    # PERMDISP
    groups_dict = {s: groups.get(s, "") for s in valid}
    try:
        permdisp_F, permdisp_p = compute_permdisp(dm, groups_dict, permutations=999)
    except Exception:
        permdisp_F, permdisp_p = float("nan"), float("nan")

    global_F = perm_global["test statistic"]
    k_global = len(cohort_list)
    N_global = len(valid)
    global_R2 = global_F * (k_global - 1) / (global_F * (k_global - 1) + (N_global - k_global))

    return {
        "coords": coords,
        "global_R2": global_R2,
        "global_p": perm_global["p-value"],
        "pairwise": pairwise_rows,
        "anosim_R": anosim_R,
        "anosim_p": anosim_p,
        "permdisp_F": permdisp_F,
        "permdisp_p": permdisp_p,
    }


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
        adjusted[sorted_idx[i]] = min(adjusted[sorted_idx[i]], adjusted[sorted_idx[i + 1]])
    return np.minimum(adjusted, 1.0)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading genus table...")
    table = load_genus_table(args.table)
    print(f"  {table.shape[0]} taxa x {table.shape[1]} samples")

    print("Loading metadata...")
    meta = pd.read_csv(args.metadata, sep="\t", dtype=str, index_col=0)
    # Keep only samples in the table
    meta = meta[meta.index.isin(table.columns)]
    print(f"  {len(meta)} samples in metadata overlap with table")

    all_results = []
    for rep in range(args.n_replicates):
        seed = args.seed + rep
        rng = np.random.default_rng(seed)
        print(f"\nReplicate {rep + 1}/{args.n_replicates} (seed={seed})...")
        result = run_single_replicate(table, meta, args.n_subsample, rng)
        all_results.append(result)
        print(f"  Global PERMANOVA: R²={result['global_R2']:.4f}, p={result['global_p']:.4f}")
        print(f"  ANOSIM: R={result['anosim_R']:.4f}, p={result['anosim_p']:.4f}")
        print(f"  PERMDISP: F={result['permdisp_F']:.4f}, p={result['permdisp_p']:.4f}")

    # Save PCoA coordinates from first replicate
    coords = all_results[0]["coords"]
    coords.to_csv(os.path.join(args.out_dir, "pcoa_coordinates.tsv"), sep="\t")
    print(f"\nWrote pcoa_coordinates.tsv ({len(coords)} samples)")

    # Save global PERMANOVA (mean ± SD across replicates)
    global_R2 = [r["global_R2"] for r in all_results]
    global_p = [r["global_p"] for r in all_results]
    global_df = pd.DataFrame({
        "metric": ["R2", "p_value"],
        "mean": [np.mean(global_R2), np.mean(global_p)],
        "sd": [np.std(global_R2), np.std(global_p)],
        "min": [np.min(global_R2), np.min(global_p)],
        "max": [np.max(global_R2), np.max(global_p)],
    })
    global_df.to_csv(os.path.join(args.out_dir, "permanova_global.tsv"), sep="\t", index=False)

    # Save pairwise PERMANOVA (aggregate across replicates)
    # Collect all pairwise comparisons
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
    # BH-adjust the mean p-values
    if len(pw_df) > 0:
        pw_df["p_BH"] = bh_adjust(pw_df["p_mean"].values)
    pw_df.to_csv(os.path.join(args.out_dir, "permanova_pairwise.tsv"), sep="\t", index=False)

    # PERMDISP
    permdisp_F = [r["permdisp_F"] for r in all_results]
    permdisp_p = [r["permdisp_p"] for r in all_results]
    permdisp_df = pd.DataFrame({
        "metric": ["F_statistic", "p_value"],
        "mean": [np.nanmean(permdisp_F), np.nanmean(permdisp_p)],
        "sd": [np.nanstd(permdisp_F), np.nanstd(permdisp_p)],
    })
    permdisp_df.to_csv(os.path.join(args.out_dir, "permdisp.tsv"), sep="\t", index=False)

    # ANOSIM
    anosim_R = [r["anosim_R"] for r in all_results]
    anosim_p = [r["anosim_p"] for r in all_results]
    anosim_df = pd.DataFrame({
        "metric": ["R_statistic", "p_value"],
        "mean": [np.nanmean(anosim_R), np.nanmean(anosim_p)],
        "sd": [np.nanstd(anosim_R), np.nanstd(anosim_p)],
    })
    anosim_df.to_csv(os.path.join(args.out_dir, "anosim.tsv"), sep="\t", index=False)

    # Stability report
    stability_rows = []
    for comp, vals in sorted(comp_data.items()):
        r2_arr = np.array(vals["R2"])
        cv = np.std(r2_arr) / np.mean(r2_arr) if np.mean(r2_arr) > 0 else float("nan")
        stability_rows.append({
            "comparison": comp,
            "R2_mean": np.mean(r2_arr),
            "R2_sd": np.std(r2_arr),
            "R2_cv": cv,
            "n_replicates": len(r2_arr),
        })
    pd.DataFrame(stability_rows).to_csv(
        os.path.join(args.out_dir, "stability_report.tsv"), sep="\t", index=False
    )

    print(f"\nAll ordination outputs written to {args.out_dir}/")


if __name__ == "__main__":
    main()
