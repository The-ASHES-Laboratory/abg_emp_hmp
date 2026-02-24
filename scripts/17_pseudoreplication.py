# 17_pseudoreplication.py
# By Carter Clinton, Ph.D.
"""
Pseudoreplication Correction

12 burials contribute multiple samples (47 unique burials -> 70 samples).
This inflates significance for tests assuming independent observations.

This script:
  1. Aggregates to one observation per burial (mean HMP proportion)
  2. Re-runs Fisher exact (detection rate) and Mann-Whitney (HMP proportion)
     at burial level
  3. Re-computes alpha diversity at burial level
  4. Prepares aggregated data for R PERMANOVA with strata

Inputs:
  results/compositional/merged_L6genus.tsv
  results/metadata_cohort_split.tsv
  data/nyabg/abg_16s_meta.tsv
  results/feast_split/burial_source_props.tsv

Outputs:
  results/pseudoreplication/burial_level_aggregation.tsv
  results/pseudoreplication/pseudorep_feast_comparison.tsv
  results/pseudoreplication/pseudorep_alpha_diversity.tsv
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from utils import read_qiime_tsv


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", default="results/compositional/merged_L6genus.tsv")
    p.add_argument("--metadata", default="results/metadata_cohort_split.tsv")
    p.add_argument("--nyabg-meta", default="data/nyabg/abg_16s_meta.tsv")
    p.add_argument("--feast-burial", default="results/feast_split/burial_source_props.tsv")
    p.add_argument("--out-dir", default="results/pseudoreplication/")
    return p.parse_args()


def extract_burial_id(sample_id):
    """Extract burial number from sample ID (e.g., '287A' -> '287')."""
    s = str(sample_id)
    # Strip leading/trailing whitespace
    s = s.strip()
    # Extract leading digits
    digits = ""
    for c in s:
        if c.isdigit():
            digits += c
        else:
            break
    return digits if digits else s


def shannon_diversity(counts):
    """Compute Shannon diversity (H') from count vector."""
    total = counts.sum()
    if total == 0:
        return 0.0
    props = counts[counts > 0] / total
    return -np.sum(props * np.log(props))


def observed_richness(counts):
    """Count non-zero taxa."""
    return (counts > 0).sum()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading genus table...")
    table = read_qiime_tsv(args.table)
    table = table.apply(pd.to_numeric, errors="coerce").fillna(0)
    print(f"  {table.shape[0]} taxa x {table.shape[1]} samples")

    print("Loading cohort metadata...")
    meta = pd.read_csv(args.metadata, sep="\t", dtype=str, index_col=0)
    meta = meta[meta.index.isin(table.columns)]

    # Get burial and control samples
    burial_samples = meta[meta["cohort"] == "burial"].index.tolist()
    control_samples = meta[meta["cohort"] == "control"].index.tolist()
    print(f"  Burial samples: {len(burial_samples)}")
    print(f"  Control samples: {len(control_samples)}")

    # Load NYABG metadata for burial ID mapping
    print("Loading NYABG metadata...")
    try:
        nyabg_meta = pd.read_csv(args.nyabg_meta, sep="\t", dtype=str, index_col=0)
    except Exception:
        nyabg_meta = pd.DataFrame()

    # Map samples to burial IDs
    print("Mapping samples to burial IDs...")
    burial_id_map = {}
    for sid in burial_samples:
        burial_id_map[sid] = extract_burial_id(sid)
    for sid in control_samples:
        burial_id_map[sid] = f"CTRL_{sid}"  # Controls get unique IDs

    # Load FEAST results
    print("Loading FEAST results...")
    feast = None
    try:
        feast = pd.read_csv(args.feast_burial, sep="\t", index_col=0)
        print(f"  FEAST results: {len(feast)} samples")
    except Exception as e:
        print(f"  Could not load FEAST results: {e}")

    # Identify HMP proportion column in FEAST
    hmp_col = None
    if feast is not None:
        for col in feast.columns:
            if "hmp" in col.lower() or "human" in col.lower():
                hmp_col = col
                break
        if hmp_col is None and len(feast.columns) > 0:
            # Try to find any source proportion column
            hmp_col = feast.columns[0]
        print(f"  HMP proportion column: '{hmp_col}'")

    # === Burial-level aggregation ===
    print("\nAggregating to burial level...")
    burial_ids = sorted(set(burial_id_map[s] for s in burial_samples))
    control_ids = sorted(set(burial_id_map[s] for s in control_samples))
    print(f"  Unique burials: {len(burial_ids)}")
    print(f"  Unique controls: {len(control_ids)}")

    agg_rows = []
    for bid in burial_ids + control_ids:
        samples = [s for s, b in burial_id_map.items() if b == bid]
        is_burial = bid in burial_ids
        cohort = "burial" if is_burial else "control"

        # HMP proportion (from FEAST)
        hmp_vals = []
        if feast is not None and hmp_col is not None:
            for s in samples:
                if s in feast.index:
                    val = feast.loc[s, hmp_col]
                    try:
                        hmp_vals.append(float(val))
                    except (ValueError, TypeError):
                        pass

        mean_hmp = np.mean(hmp_vals) if hmp_vals else np.nan
        max_hmp = np.max(hmp_vals) if hmp_vals else np.nan
        detected = any(v > 0 for v in hmp_vals) if hmp_vals else False

        # Alpha diversity (mean across samples within burial)
        shanno_vals = []
        richness_vals = []
        for s in samples:
            if s in table.columns:
                counts = table[s].values
                shanno_vals.append(shannon_diversity(counts))
                richness_vals.append(observed_richness(counts))

        agg_rows.append({
            "burial_id": bid,
            "cohort": cohort,
            "n_samples": len(samples),
            "sample_ids": ",".join(samples),
            "hmp_mean": mean_hmp,
            "hmp_max": max_hmp,
            "hmp_detected": detected,
            "shannon_mean": np.mean(shanno_vals) if shanno_vals else np.nan,
            "richness_mean": np.mean(richness_vals) if richness_vals else np.nan,
        })

    agg_df = pd.DataFrame(agg_rows)
    agg_path = os.path.join(args.out_dir, "burial_level_aggregation.tsv")
    agg_df.to_csv(agg_path, sep="\t", index=False)
    print(f"  Wrote {agg_path} ({len(agg_df)} rows)")

    # === Statistical tests at burial level ===
    print("\nRunning burial-level statistical tests...")
    burial_agg = agg_df[agg_df["cohort"] == "burial"]
    control_agg = agg_df[agg_df["cohort"] == "control"]

    comparison_rows = []

    # Fisher exact: detection rate
    if "hmp_detected" in agg_df.columns:
        burial_det = burial_agg["hmp_detected"].sum()
        burial_nodet = len(burial_agg) - burial_det
        ctrl_det = control_agg["hmp_detected"].sum()
        ctrl_nodet = len(control_agg) - ctrl_det
        contingency = [[burial_det, burial_nodet], [ctrl_det, ctrl_nodet]]
        odds, p_fisher = stats.fisher_exact(contingency)
        comparison_rows.append({
            "test": "Fisher exact (detection rate)",
            "level": "burial",
            "burial_value": f"{burial_det}/{len(burial_agg)} ({100*burial_det/len(burial_agg):.1f}%)",
            "control_value": f"{ctrl_det}/{len(control_agg)} ({100*ctrl_det/max(1,len(control_agg)):.1f}%)",
            "statistic": f"OR={odds:.2f}",
            "p_value": p_fisher,
            "n_burial": len(burial_agg),
            "n_control": len(control_agg),
        })

    # Mann-Whitney U: HMP proportion (mean per burial)
    burial_hmp = burial_agg["hmp_mean"].dropna()
    control_hmp = control_agg["hmp_mean"].dropna()
    if len(burial_hmp) > 0 and len(control_hmp) > 0:
        u_stat, p_mw = stats.mannwhitneyu(burial_hmp, control_hmp, alternative="greater")
        # Rank-biserial correlation as effect size
        n1, n2 = len(burial_hmp), len(control_hmp)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
        comparison_rows.append({
            "test": "Mann-Whitney U (HMP proportion, mean)",
            "level": "burial",
            "burial_value": f"median={burial_hmp.median():.4f}, mean={burial_hmp.mean():.4f}",
            "control_value": f"median={control_hmp.median():.4f}, mean={control_hmp.mean():.4f}",
            "statistic": f"U={u_stat:.0f}, r_rb={rank_biserial:.3f}",
            "p_value": p_mw,
            "n_burial": len(burial_hmp),
            "n_control": len(control_hmp),
        })

    # Mann-Whitney: HMP proportion (max per burial)
    burial_maxhmp = burial_agg["hmp_max"].dropna()
    control_maxhmp = control_agg["hmp_max"].dropna()
    if len(burial_maxhmp) > 0 and len(control_maxhmp) > 0:
        u_stat2, p_mw2 = stats.mannwhitneyu(burial_maxhmp, control_maxhmp, alternative="greater")
        n1, n2 = len(burial_maxhmp), len(control_maxhmp)
        rank_biserial2 = 1 - (2 * u_stat2) / (n1 * n2)
        comparison_rows.append({
            "test": "Mann-Whitney U (HMP proportion, max)",
            "level": "burial",
            "burial_value": f"median={burial_maxhmp.median():.4f}, mean={burial_maxhmp.mean():.4f}",
            "control_value": f"median={control_maxhmp.median():.4f}, mean={control_maxhmp.mean():.4f}",
            "statistic": f"U={u_stat2:.0f}, r_rb={rank_biserial2:.3f}",
            "p_value": p_mw2,
            "n_burial": len(burial_maxhmp),
            "n_control": len(control_maxhmp),
        })

    # Shannon diversity comparison
    burial_shan = burial_agg["shannon_mean"].dropna()
    control_shan = control_agg["shannon_mean"].dropna()
    if len(burial_shan) > 0 and len(control_shan) > 0:
        u_shan, p_shan = stats.mannwhitneyu(burial_shan, control_shan, alternative="two-sided")
        comparison_rows.append({
            "test": "Mann-Whitney U (Shannon diversity)",
            "level": "burial",
            "burial_value": f"median={burial_shan.median():.3f}, mean={burial_shan.mean():.3f}",
            "control_value": f"median={control_shan.median():.3f}, mean={control_shan.mean():.3f}",
            "statistic": f"U={u_shan:.0f}",
            "p_value": p_shan,
            "n_burial": len(burial_shan),
            "n_control": len(control_shan),
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(args.out_dir, "pseudorep_feast_comparison.tsv")
    comp_df.to_csv(comp_path, sep="\t", index=False)
    print(f"  Wrote {comp_path}")

    # Alpha diversity table
    alpha_rows = []
    for _, row in agg_df.iterrows():
        alpha_rows.append({
            "burial_id": row["burial_id"],
            "cohort": row["cohort"],
            "n_samples": row["n_samples"],
            "shannon_mean": row["shannon_mean"],
            "richness_mean": row["richness_mean"],
        })
    alpha_df = pd.DataFrame(alpha_rows)
    alpha_path = os.path.join(args.out_dir, "pseudorep_alpha_diversity.tsv")
    alpha_df.to_csv(alpha_path, sep="\t", index=False)
    print(f"  Wrote {alpha_path}")

    # Save burial ID mapping for R PERMANOVA script
    mapping_rows = []
    for sid, bid in burial_id_map.items():
        mapping_rows.append({"sample_id": sid, "burial_id": bid})
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_path = os.path.join(args.out_dir, "sample_burial_id_map.tsv")
    mapping_df.to_csv(mapping_path, sep="\t", index=False)
    print(f"  Wrote {mapping_path} (for R PERMANOVA with strata)")

    # Summary
    print(f"\n=== Pseudoreplication Summary ===")
    print(f"  Sample level: {len(burial_samples)} burial + {len(control_samples)} control")
    print(f"  Burial level: {len(burial_ids)} burials + {len(control_ids)} controls")
    print(f"  Multi-sample burials: {sum(1 for _, r in burial_agg.iterrows() if r['n_samples'] > 1)}")
    if len(comparison_rows) > 0:
        print(f"\n  Statistical tests (burial level):")
        for row in comparison_rows:
            sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else "ns"
            print(f"    {row['test']}: p={row['p_value']:.4f} {sig}")


if __name__ == "__main__":
    main()
