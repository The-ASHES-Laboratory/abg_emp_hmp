# 22_classifier_null.py
# By Carter Clinton, Ph.D.
"""
Classifier Null Distribution

Classify held-out EMP samples against HMP labels to establish what
"chance" confidence looks like for non-human-associated environmental
samples.

Inputs:
  results/compositional/merged_L6genus.tsv
  results/metadata_cohort_split.tsv

Outputs:
  results/classifier_null/classifier_null_distribution.tsv
  results/classifier_null/classifier_null_summary.tsv
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
    p.add_argument("--n-emp-holdout", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/classifier_null/")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    print("Loading genus table...")
    table = read_qiime_tsv(args.table)
    table = table.apply(pd.to_numeric, errors="coerce").fillna(0)

    print("Loading metadata...")
    meta = pd.read_csv(args.metadata, sep="\t", dtype=str, index_col=0)
    meta = meta[meta.index.isin(table.columns)]

    # Prepare CLR features
    print("Computing CLR transform...")
    clr_data = pd.DataFrame(
        clr_transform(table.values),
        index=table.index,
        columns=table.columns
    )

    # Get sample IDs (use main metadata for HMP body-site labels)
    hmp_ids = sorted(set(meta[meta["cohort"] == "hmp"].index) & set(clr_data.columns))
    emp_ids = sorted(set(meta[meta["cohort"] == "emp"].index) & set(clr_data.columns))
    burial_ids = sorted(set(meta[meta["cohort"] == "burial"].index) & set(clr_data.columns))

    print(f"  HMP training: {len(hmp_ids)}")
    print(f"  EMP available: {len(emp_ids)}")
    print(f"  Burial for comparison: {len(burial_ids)}")

    # Hold out EMP samples
    rng = np.random.default_rng(args.seed)
    n_holdout = min(args.n_emp_holdout, len(emp_ids))
    emp_holdout = rng.choice(emp_ids, size=n_holdout, replace=False).tolist()
    print(f"  EMP holdout: {n_holdout}")

    # Train on full HMP
    print("\nTraining L1-LR on HMP...")
    X_train = clr_data[hmp_ids].T
    y_train = meta.loc[hmp_ids, args.hmp_label].astype(str)

    lr = OneVsRestClassifier(
        LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                           max_iter=1000, random_state=args.seed),
        n_jobs=-1
    )
    lr.fit(X_train, y_train)
    print(f"  Trained on {len(hmp_ids)} samples, {y_train.nunique()} classes")

    # Predict EMP holdout
    print("Predicting EMP holdout (null distribution)...")
    X_emp = clr_data[emp_holdout].T
    emp_proba = pd.DataFrame(lr.predict_proba(X_emp), index=X_emp.index, columns=lr.classes_)
    emp_max_conf = emp_proba.max(axis=1)
    emp_predicted_class = emp_proba.idxmax(axis=1)

    # Predict burial (for comparison)
    print("Predicting burial samples...")
    X_burial = clr_data[burial_ids].T
    burial_proba = pd.DataFrame(lr.predict_proba(X_burial), index=X_burial.index, columns=lr.classes_)
    burial_max_conf = burial_proba.max(axis=1)

    # Build null distribution table
    null_rows = []
    for sid in emp_holdout:
        null_rows.append({
            "sample_id": sid,
            "cohort": "emp_holdout",
            "max_confidence": float(emp_max_conf[sid]),
            "predicted_class": emp_predicted_class[sid],
        })
    for sid in burial_ids:
        null_rows.append({
            "sample_id": sid,
            "cohort": "burial",
            "max_confidence": float(burial_max_conf[sid]),
            "predicted_class": burial_proba.loc[sid].idxmax(),
        })

    null_df = pd.DataFrame(null_rows)
    null_path = os.path.join(args.out_dir, "classifier_null_distribution.tsv")
    null_df.to_csv(null_path, sep="\t", index=False)
    print(f"  Wrote {null_path}")

    # Summary statistics
    null_95 = np.percentile(emp_max_conf, 95)
    null_99 = np.percentile(emp_max_conf, 99)
    burial_above_95 = (burial_max_conf > null_95).sum()
    burial_above_99 = (burial_max_conf > null_99).sum()

    summary_rows = [
        {"metric": "emp_null_mean", "value": emp_max_conf.mean()},
        {"metric": "emp_null_median", "value": emp_max_conf.median()},
        {"metric": "emp_null_sd", "value": emp_max_conf.std()},
        {"metric": "emp_null_95th", "value": null_95},
        {"metric": "emp_null_99th", "value": null_99},
        {"metric": "emp_n", "value": len(emp_holdout)},
        {"metric": "burial_mean", "value": burial_max_conf.mean()},
        {"metric": "burial_median", "value": burial_max_conf.median()},
        {"metric": "burial_n", "value": len(burial_ids)},
        {"metric": "burial_above_null_95th", "value": burial_above_95},
        {"metric": "burial_above_null_95th_pct", "value": burial_above_95 / max(1, len(burial_ids))},
        {"metric": "burial_above_null_99th", "value": burial_above_99},
    ]

    # Class distribution in null
    emp_class_dist = emp_predicted_class.value_counts()
    for cls, cnt in emp_class_dist.items():
        summary_rows.append({
            "metric": f"emp_null_predicted_{cls}",
            "value": cnt / len(emp_holdout),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, "classifier_null_summary.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"  Wrote {summary_path}")

    print(f"\n=== Classifier Null Distribution ===")
    print(f"  EMP null: mean={emp_max_conf.mean():.4f}, 95th={null_95:.4f}, 99th={null_99:.4f}")
    print(f"  Burial: mean={burial_max_conf.mean():.4f}")
    print(f"  Burial above null 95th: {burial_above_95}/{len(burial_ids)} "
          f"({100*burial_above_95/max(1,len(burial_ids)):.1f}%)")


if __name__ == "__main__":
    main()
