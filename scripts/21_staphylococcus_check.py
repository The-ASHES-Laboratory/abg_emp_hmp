# 21_staphylococcus_check.py
# By Carter Clinton, Ph.D.
"""
Staphylococcus Contamination Check

Staphylococcus (100% prevalence in controls, 22% in burial) is a likely
handling contaminant. Must verify the skin signal persists without it.

This script:
  1. Reports Staphylococcus prevalence and relative abundance by cohort
  2. Creates a Staphylococcus-excluded genus table
  3. Re-runs L1-LR classifier excluding Staphylococcus

Inputs:
  results/compositional/merged_L6genus.tsv
  results/metadata_cohort_split.tsv

Outputs:
  results/staphylococcus/staph_prevalence_report.tsv
  results/staphylococcus/merged_L6genus_no_staph.tsv
  results/staphylococcus/classifier_no_staph_burial.csv
  results/staphylococcus/staph_impact_summary.tsv
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from utils import read_qiime_tsv, clr_transform


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", default="results/compositional/merged_L6genus.tsv")
    p.add_argument("--metadata", default="results/metadata_cohort_split.tsv")
    p.add_argument("--hmp-meta", default="data/hmp/hmp_metadata.tsv")
    p.add_argument("--hmp-label", default="cohort_subsite")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/staphylococcus/")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading genus table...")
    table = read_qiime_tsv(args.table)
    table = table.apply(pd.to_numeric, errors="coerce").fillna(0)

    print("Loading metadata...")
    meta = pd.read_csv(args.metadata, sep="\t", dtype=str, index_col=0)
    meta = meta[meta.index.isin(table.columns)]

    # === Find Staphylococcus taxa ===
    staph_taxa = [t for t in table.index if "staphylococcus" in str(t).lower()
                  or "g__Staphylococcus" in str(t)]
    print(f"\nStaphylococcus taxa found: {len(staph_taxa)}")
    for t in staph_taxa:
        print(f"  {t}")

    if len(staph_taxa) == 0:
        print("WARNING: No Staphylococcus taxa found. Trying broader search...")
        staph_taxa = [t for t in table.index if "staph" in str(t).lower()]
        print(f"  Broader search: {len(staph_taxa)} taxa")

    # === Prevalence and abundance by cohort ===
    print("\nCalculating Staphylococcus prevalence by cohort...")
    cohorts = ["burial", "control", "emp", "hmp"]
    prevalence_rows = []

    for cohort in cohorts:
        cohort_samples = meta[meta["cohort"] == cohort].index.tolist()
        cohort_samples = [s for s in cohort_samples if s in table.columns]
        if not cohort_samples:
            continue

        cohort_table = table[cohort_samples]
        total_counts = cohort_table.sum(axis=0)

        for taxon in staph_taxa:
            taxon_counts = cohort_table.loc[taxon]
            prevalence = (taxon_counts > 0).sum() / len(cohort_samples)
            rel_abund_among_present = (
                (taxon_counts[taxon_counts > 0] / total_counts[taxon_counts > 0]).mean()
                if (taxon_counts > 0).any() else 0
            )
            mean_rel_abund = (taxon_counts / total_counts).mean()

            prevalence_rows.append({
                "taxon": taxon,
                "cohort": cohort,
                "n_samples": len(cohort_samples),
                "n_present": (taxon_counts > 0).sum(),
                "prevalence": prevalence,
                "mean_rel_abundance": mean_rel_abund,
                "mean_rel_abund_when_present": rel_abund_among_present,
            })

    prev_df = pd.DataFrame(prevalence_rows)
    prev_path = os.path.join(args.out_dir, "staph_prevalence_report.tsv")
    prev_df.to_csv(prev_path, sep="\t", index=False)
    print(f"  Wrote {prev_path}")

    # === Create Staphylococcus-excluded table ===
    print("\nCreating Staphylococcus-excluded table...")
    table_no_staph = table.drop(index=staph_taxa, errors="ignore")
    print(f"  Original: {table.shape[0]} taxa")
    print(f"  After removal: {table_no_staph.shape[0]} taxa")
    print(f"  Removed: {table.shape[0] - table_no_staph.shape[0]} taxa")

    no_staph_path = os.path.join(args.out_dir, "merged_L6genus_no_staph.tsv")
    table_no_staph.to_csv(no_staph_path, sep="\t")
    print(f"  Wrote {no_staph_path}")

    # === Re-run classifier without Staphylococcus ===
    print("\nRe-running classifier without Staphylococcus...")
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier

        # Prepare CLR features without Staphylococcus
        clr_data = pd.DataFrame(
            clr_transform(table_no_staph.values),
            index=table_no_staph.index,
            columns=table_no_staph.columns
        )

        # Use main metadata for HMP body-site labels
        hmp_samples = meta[meta["cohort"] == "hmp"].index.tolist()
        train_ids = sorted(set(clr_data.columns) & set(hmp_samples))

        burial_ids = meta[meta["cohort"] == "burial"].index.tolist()
        predict_ids = [s for s in burial_ids if s in clr_data.columns]

        if len(train_ids) > 0 and len(predict_ids) > 0:
            X_train = clr_data[train_ids].T
            y_train = meta.loc[train_ids, args.hmp_label].astype(str)
            X_pred = clr_data[predict_ids].T

            # L1-LR
            lr = OneVsRestClassifier(
                LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                   max_iter=1000, random_state=args.seed),
                n_jobs=-1
            )
            lr.fit(X_train, y_train)
            lr_proba = pd.DataFrame(lr.predict_proba(X_pred),
                                    index=X_pred.index, columns=lr.classes_)

            # Format output
            rows = []
            for sid in predict_ids:
                for cls in lr_proba.columns:
                    rows.append({
                        "sample_id": sid,
                        "model": "HMP_L1LR_no_staph",
                        "class": cls,
                        "probability": float(lr_proba.loc[sid, cls]),
                    })

            clf_df = pd.DataFrame(rows)
            clf_path = os.path.join(args.out_dir, "classifier_no_staph_burial.csv")
            clf_df.to_csv(clf_path, index=False)
            print(f"  Wrote {clf_path} ({len(clf_df)} rows)")

            # Compute skin signal comparison
            skin_cols = [c for c in lr_proba.columns if "skin" in c.lower()]
            if skin_cols:
                skin_prob = lr_proba[skin_cols].max(axis=1)
                print(f"  Skin signal (no Staph): mean={skin_prob.mean():.4f}, "
                      f"median={skin_prob.median():.4f}")
        else:
            print("  Insufficient overlapping samples for classifier")

    except ImportError as e:
        print(f"  Skipping classifier (missing package): {e}")

    # === Load original classifier results for comparison ===
    print("\nGenerating impact summary...")
    summary_rows = []
    summary_rows.append({
        "metric": "taxa_removed",
        "with_staph": table.shape[0],
        "without_staph": table_no_staph.shape[0],
        "difference": table.shape[0] - table_no_staph.shape[0],
        "note": f"Removed {len(staph_taxa)} Staphylococcus taxa",
    })

    for _, row in prev_df[prev_df["cohort"].isin(["burial", "control"])].iterrows():
        summary_rows.append({
            "metric": f"staph_prevalence_{row['cohort']}",
            "with_staph": row["prevalence"],
            "without_staph": 0.0,
            "difference": row["prevalence"],
            "note": f"Staphylococcus prevalence in {row['cohort']}",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, "staph_impact_summary.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"  Wrote {summary_path}")

    print("\n=== Staphylococcus Check Complete ===")


if __name__ == "__main__":
    main()
