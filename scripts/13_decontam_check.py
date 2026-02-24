# 13_decontam_check.py
# By Carter Clinton, Ph.D.
"""
Decontamination Check

Identifies potential contaminant taxa using frequency-based detection
(taxa more abundant in negative controls) and cross-references against
burial-network HMP-indicator taxa.

Note: For full decontam functionality, use the R decontam package.
This script provides a Python-based approximation for taxa flagging.
"""

import argparse
import csv
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="results/compositional/merged_L6genus.tsv")
    ap.add_argument("--nyabg-meta", default="data/nyabg/abg_16s_meta.tsv")
    ap.add_argument("--network-taxa", default="results/networks/burial/overall/source_network_taxa.tsv")
    ap.add_argument("--out-dir", default="results/qc/")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load NYABG metadata to identify negative controls
    print("Loading NYABG metadata...")
    nyabg = pd.read_csv(args.nyabg_meta, sep="\t", comment="#", dtype=str)
    nyabg = nyabg.rename(columns={nyabg.columns[0]: "sample_id"})
    nyabg = nyabg[~nyabg["sample_id"].str.startswith("#", na=False)]

    # Identify negative control samples
    neg_controls = set()
    sample_types = {}
    for _, row in nyabg.iterrows():
        sid = str(row["sample_id"])
        # Look for negative control indicators in any column
        row_str = " ".join(str(v).lower() for v in row.values)
        if any(kw in row_str for kw in ["water", "negative", "blank", "neg control", "extraction"]):
            neg_controls.add(sid)
            sample_types[sid] = "negative"
        elif "zymo" in row_str or "mock" in row_str:
            sample_types[sid] = "mock"
        else:
            sample_types[sid] = "sample"

    print(f"  Found {len(neg_controls)} negative control samples")

    if not neg_controls:
        print("  Warning: No negative controls identified. Using frequency-based detection only.")

    # Load genus table
    print("Loading genus table...")
    with open(args.table) as f:
        header = None
        for line in f:
            if line.startswith("#") and not line.startswith("#OTU") and not line.startswith("#Feature"):
                continue
            header = line.strip().split("\t")
            break

    sample_ids = header[1:]
    nyabg_ids = set(nyabg["sample_id"])
    nyabg_indices = [(i, sid) for i, sid in enumerate(sample_ids) if sid in nyabg_ids]
    neg_indices = [(i, sid) for i, sid in nyabg_indices if sid in neg_controls]
    sample_indices = [(i, sid) for i, sid in nyabg_indices if sid not in neg_controls]

    # Read taxa abundances for NYABG samples
    taxa_data = []
    with open(args.table) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if row[0] == header[0]:
                continue
            taxon = row[0]
            neg_vals = []
            samp_vals = []
            for idx, sid in neg_indices:
                try:
                    neg_vals.append(float(row[idx + 1]))
                except (ValueError, IndexError):
                    neg_vals.append(0.0)
            for idx, sid in sample_indices:
                try:
                    samp_vals.append(float(row[idx + 1]))
                except (ValueError, IndexError):
                    samp_vals.append(0.0)

            if neg_vals or samp_vals:
                neg_mean = np.mean(neg_vals) if neg_vals else 0.0
                samp_mean = np.mean(samp_vals) if samp_vals else 0.0
                neg_prev = sum(v > 0 for v in neg_vals) / len(neg_vals) if neg_vals else 0.0
                samp_prev = sum(v > 0 for v in samp_vals) / len(samp_vals) if samp_vals else 0.0

                # Flag as potential contaminant if more abundant in negatives
                ratio = neg_mean / samp_mean if samp_mean > 0 else float("inf") if neg_mean > 0 else 0
                is_contam = ratio > 1.0 and neg_prev > 0.5

                taxa_data.append({
                    "taxon": taxon,
                    "neg_mean_abund": neg_mean,
                    "sample_mean_abund": samp_mean,
                    "neg_prevalence": neg_prev,
                    "sample_prevalence": samp_prev,
                    "neg_to_sample_ratio": ratio if ratio != float("inf") else 999,
                    "potential_contaminant": is_contam,
                })

    contam_df = pd.DataFrame(taxa_data)
    contam_taxa = set(contam_df.loc[contam_df["potential_contaminant"], "taxon"])
    print(f"  Identified {len(contam_taxa)} potential contaminant taxa")

    # Cross-reference with network HMP-indicator taxa
    print("Cross-referencing with network HMP indicators...")
    try:
        net = pd.read_csv(args.network_taxa, sep="\t")
        hmp_taxa = set()
        if "assignment" in net.columns:
            key_col = "taxon_key" if "taxon_key" in net.columns else net.columns[0]
            hmp_taxa = set(net.loc[net["assignment"] == "hmp", key_col])

        overlap = contam_taxa & hmp_taxa
        print(f"  HMP-indicator taxa: {len(hmp_taxa)}")
        print(f"  Contaminant ∩ HMP-indicator: {len(overlap)}")

        # Flag HMP indicators that are potential contaminants
        contam_df["hmp_indicator"] = contam_df["taxon"].isin(hmp_taxa)
        contam_df["contam_and_hmp"] = contam_df["taxon"].isin(overlap)
    except Exception as e:
        print(f"  Warning: Could not load network taxa: {e}")

    # Write results
    out_path = os.path.join(args.out_dir, "contaminant_check.tsv")
    contam_df.sort_values("neg_to_sample_ratio", ascending=False).to_csv(
        out_path, sep="\t", index=False
    )
    print(f"Wrote {out_path}")

    # Write overlap summary
    summary_path = os.path.join(args.out_dir, "contaminant_hmp_overlap.tsv")
    overlap_rows = contam_df[contam_df.get("contam_and_hmp", False) == True]
    if len(overlap_rows) > 0:
        overlap_rows.to_csv(summary_path, sep="\t", index=False)
        print(f"Wrote {summary_path} ({len(overlap_rows)} overlapping taxa)")
    else:
        print("  No overlap between contaminants and HMP indicators")


if __name__ == "__main__":
    main()
