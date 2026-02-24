# 28_burial_vs_control.py
# By Carter Clinton, Ph.D.
"""
Burial vs Control Comparison

Central hypothesis tests comparing burial and control NYABG samples:
1. FEAST EMP fractions: Wilcoxon rank-sum (burial vs control)
2. Network IndVal scores: Compare IndVal_HMP distributions
3. Classifier probabilities: Compare HMP-class probability distributions
4. Beta-diversity distances: Compare burial-to-HMP vs control-to-HMP centroids
5. Taxon assignment enrichment: Fisher's exact test

Reports effect sizes (Cliff's delta) and BH-corrected p-values.
"""

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats


def cliffs_delta(x, y):
    """Compute Cliff's delta effect size (non-parametric)."""
    x, y = np.asarray(x), np.asarray(y)
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan
    more = sum((xi > yj) for xi in x for yj in y)
    less = sum((xi < yj) for xi in x for yj in y)
    return (more - less) / (n_x * n_y)


def bh_correct(pvalues):
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    corrected = [0.0] * n
    prev = 1.0
    for rank, (orig_idx, p) in enumerate(reversed(indexed)):
        adjusted = min(p * n / (n - rank), prev)
        corrected[orig_idx] = adjusted
        prev = adjusted
    return corrected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feast-burial", default="results/feast_split/burial_source_props.tsv")
    ap.add_argument("--feast-control", default="results/feast_split/control_source_props.tsv")
    ap.add_argument("--network-burial", default="results/networks/burial/overall/source_network_taxa.tsv")
    ap.add_argument("--network-control", default="results/networks/control/overall/source_network_taxa.tsv")
    ap.add_argument("--classifier-burial", default="results/classifier/burial_predictions.csv")
    ap.add_argument("--classifier-control", default="results/classifier/control_predictions.csv")
    ap.add_argument("--ordination-dir", default="results/ordination/")
    ap.add_argument("--out-dir", default="results/comparison/")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = []

    # ── 1. FEAST EMP fractions ──────────────────────────────────────────────
    print("=== Test 1: FEAST EMP fractions ===")
    try:
        feast_b = pd.read_csv(args.feast_burial, sep="\t")
        feast_c = pd.read_csv(args.feast_control, sep="\t")
        # Find EMP column (case-insensitive)
        emp_col_b = [c for c in feast_b.columns if c.lower() == "emp"]
        emp_col_c = [c for c in feast_c.columns if c.lower() == "emp"]
        if emp_col_b and emp_col_c:
            burial_emp = feast_b[emp_col_b[0]].dropna().astype(float).values
            control_emp = feast_c[emp_col_c[0]].dropna().astype(float).values
            stat, p = stats.mannwhitneyu(burial_emp, control_emp, alternative="two-sided")
            delta = cliffs_delta(burial_emp, control_emp)
            results.append({
                "test": "feast_emp_fraction",
                "group1": "burial", "group2": "control",
                "n1": len(burial_emp), "n2": len(control_emp),
                "stat": f"{stat:.4f}", "p_value": p,
                "effect_size": f"{delta:.4f}",
                "effect_type": "cliffs_delta",
                "mean1": f"{np.mean(burial_emp):.4f}",
                "mean2": f"{np.mean(control_emp):.4f}",
            })
            print(f"  Burial EMP mean: {np.mean(burial_emp):.4f}, Control: {np.mean(control_emp):.4f}, p={p:.4g}")
        else:
            print("  Warning: EMP column not found in FEAST output")
    except Exception as e:
        print(f"  Skipping: {e}")

    # Also test HMP fractions
    try:
        hmp_col_b = [c for c in feast_b.columns if c.lower() == "hmp"]
        hmp_col_c = [c for c in feast_c.columns if c.lower() == "hmp"]
        if hmp_col_b and hmp_col_c:
            burial_hmp = feast_b[hmp_col_b[0]].dropna().astype(float).values
            control_hmp = feast_c[hmp_col_c[0]].dropna().astype(float).values
            stat, p = stats.mannwhitneyu(burial_hmp, control_hmp, alternative="two-sided")
            delta = cliffs_delta(burial_hmp, control_hmp)
            results.append({
                "test": "feast_hmp_fraction",
                "group1": "burial", "group2": "control",
                "n1": len(burial_hmp), "n2": len(control_hmp),
                "stat": f"{stat:.4f}", "p_value": p,
                "effect_size": f"{delta:.4f}",
                "effect_type": "cliffs_delta",
                "mean1": f"{np.mean(burial_hmp):.4f}",
                "mean2": f"{np.mean(control_hmp):.4f}",
            })
            print(f"  Burial HMP mean: {np.mean(burial_hmp):.4f}, Control: {np.mean(control_hmp):.4f}, p={p:.4g}")
    except Exception as e:
        print(f"  HMP test skipped: {e}")

    # ── 2. Network IndVal HMP scores ────────────────────────────────────────
    print("\n=== Test 2: Network IndVal HMP scores ===")
    try:
        net_b = pd.read_csv(args.network_burial, sep="\t")
        net_c = pd.read_csv(args.network_control, sep="\t")
        # Find IndVal HMP columns
        hmp_indval_cols_b = [c for c in net_b.columns if "indval" in c.lower() and "hmp" in c.lower()]
        hmp_indval_cols_c = [c for c in net_c.columns if "indval" in c.lower() and "hmp" in c.lower()]

        if hmp_indval_cols_b and hmp_indval_cols_c:
            # Merge on taxon key
            key_col = "taxon_key" if "taxon_key" in net_b.columns else net_b.columns[0]
            merged = net_b[[key_col, hmp_indval_cols_b[0]]].merge(
                net_c[[key_col, hmp_indval_cols_c[0]]],
                on=key_col, suffixes=("_burial", "_control")
            ).dropna()
            if len(merged) > 0:
                b_vals = merged.iloc[:, 1].astype(float).values
                c_vals = merged.iloc[:, 2].astype(float).values
                stat, p = stats.wilcoxon(b_vals, c_vals, alternative="two-sided")
                delta = cliffs_delta(b_vals, c_vals)
                results.append({
                    "test": "indval_hmp_paired",
                    "group1": "burial", "group2": "control",
                    "n1": len(b_vals), "n2": len(c_vals),
                    "stat": f"{stat:.4f}", "p_value": p,
                    "effect_size": f"{delta:.4f}",
                    "effect_type": "cliffs_delta",
                    "mean1": f"{np.mean(b_vals):.4f}",
                    "mean2": f"{np.mean(c_vals):.4f}",
                })
                print(f"  {len(merged)} shared taxa, Wilcoxon p={p:.4g}")
        else:
            print("  Warning: IndVal HMP columns not found")
    except Exception as e:
        print(f"  Skipping: {e}")

    # ── 3. Classifier HMP probabilities ─────────────────────────────────────
    print("\n=== Test 3: Classifier HMP probabilities ===")
    try:
        cls_b = pd.read_csv(args.classifier_burial)
        cls_c = pd.read_csv(args.classifier_control)
        for model in ["HMP_RF", "HMP_L1LR"]:
            # Get max HMP class probability per sample
            b_sub = cls_b[cls_b["model"] == model].copy()
            c_sub = cls_c[cls_c["model"] == model].copy()
            if len(b_sub) > 0 and len(c_sub) > 0:
                # Sum probabilities for HMP-related classes per sample
                b_probs = b_sub.groupby("sample_id")["probability"].max().values
                c_probs = c_sub.groupby("sample_id")["probability"].max().values
                stat, p = stats.mannwhitneyu(b_probs, c_probs, alternative="two-sided")
                delta = cliffs_delta(b_probs, c_probs)
                results.append({
                    "test": f"classifier_{model.lower()}_max_prob",
                    "group1": "burial", "group2": "control",
                    "n1": len(b_probs), "n2": len(c_probs),
                    "stat": f"{stat:.4f}", "p_value": p,
                    "effect_size": f"{delta:.4f}",
                    "effect_type": "cliffs_delta",
                    "mean1": f"{np.mean(b_probs):.4f}",
                    "mean2": f"{np.mean(c_probs):.4f}",
                })
                print(f"  {model}: burial mean={np.mean(b_probs):.4f}, control={np.mean(c_probs):.4f}, p={p:.4g}")
    except Exception as e:
        print(f"  Skipping: {e}")

    # ── 4. Beta-diversity distances ─────────────────────────────────────────
    print("\n=== Test 4: Beta-diversity centroid distances ===")
    try:
        pcoa = pd.read_csv(os.path.join(args.ordination_dir, "pcoa_coordinates.tsv"), sep="\t", index_col=0)
        meta = pd.read_csv("results/metadata_cohort_split.tsv", sep="\t")
        meta.columns = [meta.columns[0].replace("#", "").strip()] + list(meta.columns[1:])
        meta = meta.rename(columns={meta.columns[0]: "sample_id"})

        for target_cohort in ["hmp", "emp"]:
            target_ids = set(meta.loc[meta["cohort"] == target_cohort, "sample_id"].astype(str))
            burial_ids = set(meta.loc[meta["cohort"] == "burial", "sample_id"].astype(str))
            control_ids = set(meta.loc[meta["cohort"] == "control", "sample_id"].astype(str))

            # Get centroid of target cohort
            target_in_pcoa = [s for s in target_ids if s in pcoa.index]
            burial_in_pcoa = [s for s in burial_ids if s in pcoa.index]
            control_in_pcoa = [s for s in control_ids if s in pcoa.index]

            if target_in_pcoa and burial_in_pcoa and control_in_pcoa:
                centroid = pcoa.loc[target_in_pcoa].mean(axis=0).values
                b_dists = np.linalg.norm(pcoa.loc[burial_in_pcoa].values - centroid, axis=1)
                c_dists = np.linalg.norm(pcoa.loc[control_in_pcoa].values - centroid, axis=1)
                stat, p = stats.mannwhitneyu(b_dists, c_dists, alternative="two-sided")
                delta = cliffs_delta(b_dists, c_dists)
                results.append({
                    "test": f"pcoa_dist_to_{target_cohort}_centroid",
                    "group1": "burial", "group2": "control",
                    "n1": len(b_dists), "n2": len(c_dists),
                    "stat": f"{stat:.4f}", "p_value": p,
                    "effect_size": f"{delta:.4f}",
                    "effect_type": "cliffs_delta",
                    "mean1": f"{np.mean(b_dists):.4f}",
                    "mean2": f"{np.mean(c_dists):.4f}",
                })
                print(f"  Dist to {target_cohort} centroid: burial={np.mean(b_dists):.4f}, "
                      f"control={np.mean(c_dists):.4f}, p={p:.4g}")
    except Exception as e:
        print(f"  Skipping: {e}")

    # ── 5. Taxon assignment enrichment (Fisher's exact) ─────────────────────
    print("\n=== Test 5: Taxon assignment enrichment ===")
    try:
        net_b = pd.read_csv(args.network_burial, sep="\t")
        net_c = pd.read_csv(args.network_control, sep="\t")
        assign_col = "assignment" if "assignment" in net_b.columns else None
        if assign_col:
            b_counts = net_b[assign_col].value_counts()
            c_counts = net_c[assign_col].value_counts()
            # 2x2: {emp, hmp} x {burial, control}
            b_emp = b_counts.get("emp", 0)
            b_hmp = b_counts.get("hmp", 0)
            c_emp = c_counts.get("emp", 0)
            c_hmp = c_counts.get("hmp", 0)
            table = np.array([[b_emp, b_hmp], [c_emp, c_hmp]])
            odds, p = stats.fisher_exact(table, alternative="two-sided")
            results.append({
                "test": "assignment_enrichment_fisher",
                "group1": "burial", "group2": "control",
                "n1": b_emp + b_hmp, "n2": c_emp + c_hmp,
                "stat": f"{odds:.4f}", "p_value": p,
                "effect_size": f"{odds:.4f}",
                "effect_type": "odds_ratio",
                "mean1": f"emp={b_emp},hmp={b_hmp}",
                "mean2": f"emp={c_emp},hmp={c_hmp}",
            })
            print(f"  Burial: emp={b_emp}, hmp={b_hmp}; Control: emp={c_emp}, hmp={c_hmp}")
            print(f"  Fisher's exact: OR={odds:.4f}, p={p:.4g}")
    except Exception as e:
        print(f"  Skipping: {e}")

    # ── BH correction ───────────────────────────────────────────────────────
    if results:
        pvals = [r["p_value"] for r in results]
        qvals = bh_correct(pvals)
        for r, q in zip(results, qvals):
            r["q_value"] = q
            r["p_value"] = f"{r['p_value']:.6g}"
            r["q_value"] = f"{q:.6g}"

        out_path = os.path.join(args.out_dir, "burial_vs_control_tests.tsv")
        fieldnames = ["test", "group1", "group2", "n1", "n2", "stat",
                       "p_value", "q_value", "effect_size", "effect_type",
                       "mean1", "mean2"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\nWrote {out_path} ({len(results)} tests)")
    else:
        print("\nNo tests completed successfully.")


if __name__ == "__main__":
    main()
