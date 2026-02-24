# 19_cranium_analysis.py
# By Carter Clinton, Ph.D.
"""
Cranium vs Non-Cranium HMP Analysis

Cranium soil shows 73% HMP detection (mean 20.7%) vs 40% (8.2%) for
non-cranial (p=0.027). This script formalizes the analysis with:
  - Mann-Whitney U (one-sided: cranium > non-cranium)
  - Fisher exact test (detection rate)
  - Cliff's delta effect size
  - 4-level spatial gradient: Cranium > Other Body > Soil Matrix > Control
  - Jonckheere-Terpstra ordered trend test for gradient

Inputs:
  data/nyabg/abg_16s_meta.tsv
  results/feast_split/burial_source_props.tsv

Outputs:
  results/cranium/cranium_vs_noncranium.tsv
  results/cranium/cranium_gradient.tsv
"""

import argparse
import os
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nyabg-meta", default="data/nyabg/abg_16s_meta.tsv")
    p.add_argument("--feast-burial", default="results/feast_split/burial_source_props.tsv")
    p.add_argument("--feast-control", default="results/feast_split/control_source_props.tsv")
    p.add_argument("--out-dir", default="results/cranium/")
    return p.parse_args()


def cliffs_delta(x, y):
    """Compute Cliff's delta effect size."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (n_x * n_y)


def jonckheere_terpstra(groups, order):
    """Jonckheere-Terpstra test for ordered alternatives.

    Tests H1: group medians follow the specified order.
    Uses Mann-Whitney U counts between ordered pairs.
    """
    k = len(order)
    J = 0
    total_pairs = 0
    for i in range(k):
        for j in range(i + 1, k):
            g1 = groups.get(order[i], [])
            g2 = groups.get(order[j], [])
            if len(g1) == 0 or len(g2) == 0:
                continue
            # Count how many g2 > g1 (+ 0.5 for ties)
            for x in g1:
                for y in g2:
                    if y > x:
                        J += 1
                    elif y == x:
                        J += 0.5
                    total_pairs += 1

    if total_pairs == 0:
        return J, np.nan

    # Normal approximation
    ns = [len(groups.get(o, [])) for o in order]
    N = sum(ns)
    E_J = (N**2 - sum(n**2 for n in ns)) / 4
    var_num = N**2 * (2*N + 3) - sum(n**2 * (2*n + 3) for n in ns)
    Var_J = var_num / 72
    if Var_J <= 0:
        return J, np.nan
    z = (J - E_J) / np.sqrt(Var_J)
    p = 1 - stats.norm.cdf(z)  # one-sided
    return z, p


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load NYABG metadata
    print("Loading NYABG metadata...")
    nyabg_meta = pd.read_csv(args.nyabg_meta, sep="\t", dtype=str, index_col=0)
    print(f"  {len(nyabg_meta)} samples")

    # Find body region column
    body_col = None
    for candidate in ["body_region", "Body_Region", "BodyRegion", "body_site",
                      "Body Region", "region", "sample_type"]:
        if candidate in nyabg_meta.columns:
            body_col = candidate
            break
    if body_col is None:
        for col in nyabg_meta.columns:
            if "body" in col.lower() or "region" in col.lower():
                body_col = col
                break
    if body_col is None:
        print("ERROR: Cannot find body region column in NYABG metadata")
        print(f"  Available columns: {list(nyabg_meta.columns)}")
        sys.exit(1)
    print(f"  Body region column: '{body_col}'")
    print(f"  Values: {nyabg_meta[body_col].value_counts().to_dict()}")

    # Load FEAST results
    print("Loading FEAST results...")
    feast_burial = pd.read_csv(args.feast_burial, sep="\t", index_col=0)
    print(f"  Burial FEAST: {len(feast_burial)} samples")

    # Load control FEAST if available
    feast_control = None
    try:
        feast_control = pd.read_csv(args.feast_control, sep="\t", index_col=0)
        print(f"  Control FEAST: {len(feast_control)} samples")
    except Exception:
        print("  Control FEAST not loaded")

    # Find HMP proportion column
    hmp_col = None
    for col in feast_burial.columns:
        if "hmp" in col.lower() or "human" in col.lower():
            hmp_col = col
            break
    if hmp_col is None:
        hmp_col = feast_burial.columns[0]
    print(f"  HMP column: '{hmp_col}'")

    # Merge body region with FEAST HMP proportion
    burial_samples = feast_burial.index.tolist()
    data = []
    for sid in burial_samples:
        if sid in nyabg_meta.index:
            region = str(nyabg_meta.loc[sid, body_col]).strip().lower()
            hmp_val = float(feast_burial.loc[sid, hmp_col])
            data.append({"sample_id": sid, "body_region": region, "hmp_proportion": hmp_val})

    df = pd.DataFrame(data)
    print(f"\n  Merged data: {len(df)} samples")

    # Classify into cranium vs non-cranium
    cranium_keywords = ["cranium", "skull", "head", "cranial"]
    df["is_cranium"] = df["body_region"].apply(
        lambda x: any(kw in str(x).lower() for kw in cranium_keywords)
    )

    cranium = df[df["is_cranium"]]
    non_cranium = df[~df["is_cranium"]]
    print(f"  Cranium: {len(cranium)}, Non-cranium: {len(non_cranium)}")

    # === Cranium vs Non-Cranium Tests ===
    results = []

    # Detection rate
    cran_det = (cranium["hmp_proportion"] > 0).sum()
    noncran_det = (non_cranium["hmp_proportion"] > 0).sum()
    contingency = [
        [cran_det, len(cranium) - cran_det],
        [noncran_det, len(non_cranium) - noncran_det],
    ]
    odds, p_fisher = stats.fisher_exact(contingency)
    results.append({
        "test": "Fisher exact (detection rate)",
        "cranium_value": f"{cran_det}/{len(cranium)} ({100*cran_det/max(1,len(cranium)):.1f}%)",
        "noncranium_value": f"{noncran_det}/{len(non_cranium)} ({100*noncran_det/max(1,len(non_cranium)):.1f}%)",
        "statistic": f"OR={odds:.2f}",
        "p_value": p_fisher,
        "effect_size": f"OR={odds:.2f}",
    })

    # Mann-Whitney U (one-sided: cranium > non-cranium)
    cran_hmp = cranium["hmp_proportion"].values
    noncran_hmp = non_cranium["hmp_proportion"].values
    u_stat, p_mw = stats.mannwhitneyu(cran_hmp, noncran_hmp, alternative="greater")
    cd = cliffs_delta(cran_hmp, noncran_hmp)
    results.append({
        "test": "Mann-Whitney U (HMP proportion, one-sided)",
        "cranium_value": f"mean={cran_hmp.mean():.4f}, median={np.median(cran_hmp):.4f}",
        "noncranium_value": f"mean={noncran_hmp.mean():.4f}, median={np.median(noncran_hmp):.4f}",
        "statistic": f"U={u_stat:.0f}",
        "p_value": p_mw,
        "effect_size": f"Cliff's d={cd:.3f}",
    })

    results_df = pd.DataFrame(results)
    results_path = os.path.join(args.out_dir, "cranium_vs_noncranium.tsv")
    results_df.to_csv(results_path, sep="\t", index=False)
    print(f"\nWrote {results_path}")

    # === 4-Level Spatial Gradient ===
    print("\nBuilding 4-level spatial gradient...")

    # Classify body regions into gradient levels
    soil_matrix_keywords = ["soil", "matrix", "sm", "fill", "general"]
    body_other_keywords = ["pelvis", "torso", "leg", "arm", "foot", "hand",
                           "rib", "femur", "vertebr", "sacrum", "tibia",
                           "humerus", "lower", "upper", "body"]

    def classify_gradient(region):
        r = str(region).lower().strip()
        if any(kw in r for kw in cranium_keywords):
            return "Cranium"
        if any(kw in r for kw in soil_matrix_keywords):
            return "Soil Matrix"
        if any(kw in r for kw in body_other_keywords):
            return "Other Body"
        return "Other Body"  # Default for unclassified body-associated

    df["gradient_level"] = df["body_region"].apply(classify_gradient)

    # Add control samples
    control_data = []
    if feast_control is not None:
        for sid in feast_control.index:
            hmp_val = float(feast_control.loc[sid, hmp_col]) if hmp_col in feast_control.columns else 0.0
            control_data.append({
                "sample_id": sid, "body_region": "control",
                "hmp_proportion": hmp_val, "is_cranium": False,
                "gradient_level": "Control",
            })

    if control_data:
        df_full = pd.concat([df, pd.DataFrame(control_data)], ignore_index=True)
    else:
        df_full = df.copy()

    gradient_order = ["Cranium", "Other Body", "Soil Matrix", "Control"]
    gradient_groups = {}
    gradient_rows = []
    for level in gradient_order:
        level_data = df_full[df_full["gradient_level"] == level]
        hmp_vals = level_data["hmp_proportion"].values
        gradient_groups[level] = hmp_vals
        detected = (hmp_vals > 0).sum() if len(hmp_vals) > 0 else 0
        gradient_rows.append({
            "gradient_level": level,
            "n_samples": len(level_data),
            "mean_hmp": np.mean(hmp_vals) if len(hmp_vals) > 0 else np.nan,
            "median_hmp": np.median(hmp_vals) if len(hmp_vals) > 0 else np.nan,
            "sd_hmp": np.std(hmp_vals) if len(hmp_vals) > 0 else np.nan,
            "detection_rate": detected / max(1, len(level_data)),
            "min_hmp": np.min(hmp_vals) if len(hmp_vals) > 0 else np.nan,
            "max_hmp": np.max(hmp_vals) if len(hmp_vals) > 0 else np.nan,
        })

    # Jonckheere-Terpstra ordered trend test
    jt_z, jt_p = jonckheere_terpstra(gradient_groups, gradient_order)
    gradient_rows.append({
        "gradient_level": "TREND_TEST",
        "n_samples": sum(len(v) for v in gradient_groups.values()),
        "mean_hmp": jt_z,
        "median_hmp": jt_p,
        "sd_hmp": np.nan,
        "detection_rate": np.nan,
        "min_hmp": np.nan,
        "max_hmp": np.nan,
    })

    # Pairwise comparisons between adjacent levels
    for i in range(len(gradient_order) - 1):
        g1 = gradient_groups.get(gradient_order[i], [])
        g2 = gradient_groups.get(gradient_order[i + 1], [])
        if len(g1) > 0 and len(g2) > 0:
            u, p = stats.mannwhitneyu(g1, g2, alternative="greater")
            cd = cliffs_delta(g1, g2)
            gradient_rows.append({
                "gradient_level": f"PAIRWISE_{gradient_order[i]}_vs_{gradient_order[i+1]}",
                "n_samples": len(g1) + len(g2),
                "mean_hmp": u,
                "median_hmp": p,
                "sd_hmp": cd,
                "detection_rate": np.nan,
                "min_hmp": np.nan,
                "max_hmp": np.nan,
            })

    gradient_df = pd.DataFrame(gradient_rows)
    gradient_path = os.path.join(args.out_dir, "cranium_gradient.tsv")
    gradient_df.to_csv(gradient_path, sep="\t", index=False)
    print(f"Wrote {gradient_path}")

    # Summary
    print("\n=== Spatial Gradient Summary ===")
    print(f"  Order: {' > '.join(gradient_order)}")
    for level in gradient_order:
        vals = gradient_groups.get(level, [])
        if len(vals) > 0:
            print(f"  {level}: n={len(vals)}, mean={np.mean(vals):.4f}, "
                  f"det={100*(vals>0).sum()/len(vals):.0f}%")
    print(f"  Jonckheere-Terpstra: z={jt_z:.3f}, p={jt_p:.4f}")


if __name__ == "__main__":
    main()
