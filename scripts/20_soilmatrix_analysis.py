# 20_soilmatrix_analysis.py
# By Carter Clinton, Ph.D.
"""
Soil Matrix vs Body-Associated HMP Analysis

Body-associated samples carry more HMP (12.5%) than general burial fill
(6.3%, p=0.024). This script formalizes the comparison with proper
effect sizes.

Inputs:
  data/nyabg/abg_16s_meta.tsv
  results/feast_split/burial_source_props.tsv

Outputs:
  results/soilmatrix/soilmatrix_vs_bodyassoc.tsv
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nyabg-meta", default="data/nyabg/abg_16s_meta.tsv")
    p.add_argument("--feast-burial", default="results/feast_split/burial_source_props.tsv")
    p.add_argument("--out-dir", default="results/soilmatrix/")
    return p.parse_args()


def cliffs_delta(x, y):
    """Compute Cliff's delta effect size."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (n_x * n_y)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load metadata
    print("Loading NYABG metadata...")
    nyabg_meta = pd.read_csv(args.nyabg_meta, sep="\t", dtype=str, index_col=0)

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
        print("ERROR: Cannot find body region column")
        sys.exit(1)
    print(f"  Body region column: '{body_col}'")

    # Load FEAST
    print("Loading FEAST results...")
    feast = pd.read_csv(args.feast_burial, sep="\t", index_col=0)

    # Find HMP column
    hmp_col = None
    for col in feast.columns:
        if "hmp" in col.lower():
            hmp_col = col
            break
    if hmp_col is None:
        hmp_col = feast.columns[0]

    # Classify samples
    soil_matrix_keywords = ["soil", "matrix", "sm", "fill", "general"]

    data = []
    for sid in feast.index:
        if sid not in nyabg_meta.index:
            continue
        region = str(nyabg_meta.loc[sid, body_col]).strip().lower()
        hmp_val = float(feast.loc[sid, hmp_col])
        is_soil_matrix = any(kw in region for kw in soil_matrix_keywords)
        category = "Soil Matrix" if is_soil_matrix else "Body-Associated"
        data.append({
            "sample_id": sid,
            "body_region": region,
            "category": category,
            "hmp_proportion": hmp_val,
        })

    df = pd.DataFrame(data)
    body_assoc = df[df["category"] == "Body-Associated"]["hmp_proportion"].values
    soil_mat = df[df["category"] == "Soil Matrix"]["hmp_proportion"].values

    print(f"  Body-Associated: n={len(body_assoc)}")
    print(f"  Soil Matrix: n={len(soil_mat)}")

    results = []

    # Mann-Whitney U (one-sided: body > soil)
    if len(body_assoc) > 0 and len(soil_mat) > 0:
        u_stat, p_mw = stats.mannwhitneyu(body_assoc, soil_mat, alternative="greater")
        cd = cliffs_delta(body_assoc, soil_mat)

        # Two-sided for reporting
        _, p_two = stats.mannwhitneyu(body_assoc, soil_mat, alternative="two-sided")

        results.append({
            "test": "Mann-Whitney U (one-sided: body > soil)",
            "body_assoc_n": len(body_assoc),
            "body_assoc_mean": np.mean(body_assoc),
            "body_assoc_median": np.median(body_assoc),
            "body_assoc_sd": np.std(body_assoc),
            "soil_matrix_n": len(soil_mat),
            "soil_matrix_mean": np.mean(soil_mat),
            "soil_matrix_median": np.median(soil_mat),
            "soil_matrix_sd": np.std(soil_mat),
            "U_statistic": u_stat,
            "p_value_one_sided": p_mw,
            "p_value_two_sided": p_two,
            "cliffs_delta": cd,
        })

        # Detection rate comparison
        body_det = (body_assoc > 0).sum()
        soil_det = (soil_mat > 0).sum()
        contingency = [
            [body_det, len(body_assoc) - body_det],
            [soil_det, len(soil_mat) - soil_det],
        ]
        odds, p_fisher = stats.fisher_exact(contingency)
        results.append({
            "test": "Fisher exact (detection rate)",
            "body_assoc_n": len(body_assoc),
            "body_assoc_mean": body_det / len(body_assoc),
            "body_assoc_median": np.nan,
            "body_assoc_sd": np.nan,
            "soil_matrix_n": len(soil_mat),
            "soil_matrix_mean": soil_det / len(soil_mat) if len(soil_mat) > 0 else np.nan,
            "soil_matrix_median": np.nan,
            "soil_matrix_sd": np.nan,
            "U_statistic": odds,
            "p_value_one_sided": p_fisher,
            "p_value_two_sided": p_fisher,
            "cliffs_delta": np.nan,
        })

    results_df = pd.DataFrame(results)
    out_path = os.path.join(args.out_dir, "soilmatrix_vs_bodyassoc.tsv")
    results_df.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {out_path}")

    # Summary
    print("\n=== Soil Matrix vs Body-Associated ===")
    if len(body_assoc) > 0:
        print(f"  Body-Associated: mean={np.mean(body_assoc):.4f}, "
              f"median={np.median(body_assoc):.4f}, "
              f"det={100*(body_assoc>0).sum()/len(body_assoc):.1f}%")
    if len(soil_mat) > 0:
        print(f"  Soil Matrix: mean={np.mean(soil_mat):.4f}, "
              f"median={np.median(soil_mat):.4f}, "
              f"det={100*(soil_mat>0).sum()/len(soil_mat):.1f}%")
    if results:
        print(f"  Mann-Whitney p={results[0]['p_value_one_sided']:.4f}, "
              f"Cliff's d={results[0]['cliffs_delta']:.3f}")


if __name__ == "__main__":
    main()
