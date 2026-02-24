# 25_within_burial_variance.py
# By Carter Clinton, Ph.D.
"""
Within-Burial Spatial Heterogeneity

Quantify within-burial HMP variance for burials with multiple samples.
Highlight exemplar burials with large within-burial differences.

Inputs:
  data/nyabg/abg_16s_meta.tsv
  results/feast_split/burial_source_props.tsv

Outputs:
  results/within_burial/within_burial_variance.tsv
  results/within_burial/within_burial_exemplars.tsv
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nyabg-meta", default="data/nyabg/abg_16s_meta.tsv")
    p.add_argument("--feast-burial", default="results/feast_split/burial_source_props.tsv")
    p.add_argument("--out-dir", default="results/within_burial/")
    return p.parse_args()


def extract_burial_id(sample_id):
    """Extract burial number from sample ID."""
    s = str(sample_id).strip()
    digits = ""
    for c in s:
        if c.isdigit():
            digits += c
        else:
            break
    return digits if digits else s


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

    # Load FEAST
    print("Loading FEAST results...")
    feast = pd.read_csv(args.feast_burial, sep="\t", index_col=0)

    hmp_col = None
    for col in feast.columns:
        if "hmp" in col.lower():
            hmp_col = col
            break
    if hmp_col is None:
        hmp_col = feast.columns[0]

    # Map samples to burials
    print("Computing within-burial variance...")
    burial_data = {}  # burial_id -> list of (sample_id, body_region, hmp_proportion)
    for sid in feast.index:
        bid = extract_burial_id(sid)
        hmp_val = float(feast.loc[sid, hmp_col])
        region = nyabg_meta.loc[sid, body_col] if sid in nyabg_meta.index and body_col else "unknown"
        burial_data.setdefault(bid, []).append({
            "sample_id": sid,
            "body_region": str(region).strip(),
            "hmp_proportion": hmp_val,
        })

    # Compute variance for multi-sample burials
    variance_rows = []
    for bid, samples in sorted(burial_data.items()):
        hmp_vals = [s["hmp_proportion"] for s in samples]
        regions = [s["body_region"] for s in samples]
        sample_ids = [s["sample_id"] for s in samples]

        variance_rows.append({
            "burial_id": bid,
            "n_samples": len(samples),
            "sample_ids": ",".join(sample_ids),
            "body_regions": ",".join(regions),
            "hmp_mean": np.mean(hmp_vals),
            "hmp_sd": np.std(hmp_vals) if len(hmp_vals) > 1 else 0,
            "hmp_range": max(hmp_vals) - min(hmp_vals),
            "hmp_max": max(hmp_vals),
            "hmp_min": min(hmp_vals),
            "hmp_cv": np.std(hmp_vals) / np.mean(hmp_vals) if np.mean(hmp_vals) > 0 else np.nan,
        })

    var_df = pd.DataFrame(variance_rows)
    var_path = os.path.join(args.out_dir, "within_burial_variance.tsv")
    var_df.to_csv(var_path, sep="\t", index=False)
    print(f"  Wrote {var_path} ({len(var_df)} burials)")

    # Multi-sample burials only
    multi = var_df[var_df["n_samples"] > 1].copy()
    print(f"\n  Multi-sample burials: {len(multi)}")
    if len(multi) > 0:
        print(f"  Mean within-burial range: {multi['hmp_range'].mean():.4f}")
        print(f"  Max within-burial range: {multi['hmp_range'].max():.4f}")

    # Exemplar burials (largest within-burial differences)
    exemplars = multi.nlargest(min(10, len(multi)), "hmp_range") if len(multi) > 0 else pd.DataFrame()

    # Add per-sample detail for exemplars
    exemplar_detail_rows = []
    for _, ex_row in exemplars.iterrows():
        bid = ex_row["burial_id"]
        for sample in burial_data[bid]:
            exemplar_detail_rows.append({
                "burial_id": bid,
                "sample_id": sample["sample_id"],
                "body_region": sample["body_region"],
                "hmp_proportion": sample["hmp_proportion"],
                "burial_hmp_range": ex_row["hmp_range"],
            })

    exemplar_df = pd.DataFrame(exemplar_detail_rows)
    exemplar_path = os.path.join(args.out_dir, "within_burial_exemplars.tsv")
    exemplar_df.to_csv(exemplar_path, sep="\t", index=False)
    print(f"  Wrote {exemplar_path} ({len(exemplar_df)} samples across {len(exemplars)} exemplar burials)")

    # Summary
    print("\n=== Exemplar Burials (largest within-burial HMP range) ===")
    for _, ex in exemplars.head(5).iterrows():
        print(f"  Burial {ex['burial_id']}: range={ex['hmp_range']:.4f} "
              f"({ex['hmp_min']:.4f}-{ex['hmp_max']:.4f}), "
              f"n={ex['n_samples']}, regions={ex['body_regions']}")


if __name__ == "__main__":
    main()
