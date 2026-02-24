# 11_classify_samples.py
# By Carter Clinton, Ph.D.
"""
Body-site Classification

Trains RF and L1-LR classifiers on HMP (body-site) and EMP (environment-type)
reference data, then predicts class probabilities for burial or control
NYABG samples.

Changes from original:
  - Added --target-ids-file (one sample ID per line) to select burial/control subset
  - Added --sample-type (burial/control) for output labeling
  - Falls back to all NYABG samples if no --target-ids-file provided
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from utils import clr_transform, read_qiime_tsv


def prepare_features(table_tsv):
    X = read_qiime_tsv(table_tsv)
    taxa = X.index.str.replace("g__", "", regex=False).str.split(";").str[-1].str.strip()
    X.index = taxa
    clr = pd.DataFrame(clr_transform(X), index=X.index, columns=X.columns)
    return clr


def read_metadata(path):
    """Load metadata from TSV/CSV, handling trailing tabs and #-prefixed lines."""
    if path.endswith((".tsv", ".txt")):
        with open(path, "r", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = None
            rows = []
            for row in reader:
                if not row:
                    continue
                if row[0].startswith("#"):
                    continue
                if header is None:
                    header = [col.strip() for col in row]
                    continue
                row = row[: len(header)] + [""] * max(0, len(header) - len(row))
                rows.append(row)
        if header is None:
            raise ValueError(f"No header rows found in metadata file: {path}")
        df = pd.DataFrame(rows, columns=header)
    else:
        df = pd.read_csv(path, sep=",", comment="#", dtype=str)

    df = df.loc[:, df.columns != ""]
    df = df.rename(columns={df.columns[0]: "sample_id"})
    df["sample_id"] = df["sample_id"].astype(str)
    df = df.drop_duplicates("sample_id").set_index("sample_id")
    return df


def train_and_predict(clr, meta, label_col, ny_ids, seed=20251101):
    import time
    train_ids = sorted(set(clr.columns) & set(meta.index))
    if not train_ids:
        raise ValueError("No overlapping samples between feature table and metadata.")
    X_train = clr[train_ids].T
    y_train = meta.loc[train_ids, label_col].astype(str)
    n_classes = y_train.nunique()
    print(f"[classify_nyabg] Training set: {len(train_ids)} samples, "
          f"{X_train.shape[1]} features, {n_classes} classes", flush=True)

    predict_ids = [sid for sid in ny_ids if sid in clr.columns]
    if not predict_ids:
        raise ValueError("No target samples found in feature table for prediction.")
    X_ny = clr[predict_ids].T

    # --- Random Forest (uncalibrated — RF probabilities are well-calibrated
    #     for ranking; skip CalibratedClassifierCV to avoid 3× training cost) ---
    t0 = time.time()
    print(f"[classify_nyabg] Training RF (200 trees)...", flush=True)
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_proba = pd.DataFrame(rf.predict_proba(X_ny), index=X_ny.index, columns=rf.classes_)
    print(f"[classify_nyabg] RF done in {time.time()-t0:.0f}s.", flush=True)

    # --- L1-Logistic Regression (uncalibrated) ---
    class_counts = y_train.value_counts()
    min_count = class_counts.min()
    if min_count >= 2:
        t0 = time.time()
        print(f"[classify_nyabg] Training L1-LR...", flush=True)
        lr = OneVsRestClassifier(
            LogisticRegression(
                penalty="l1", solver="liblinear", C=1.0, max_iter=1000,
                random_state=seed,
            ),
            n_jobs=-1,
        )
        lr.fit(X_train, y_train)
        lr_proba = pd.DataFrame(lr.predict_proba(X_ny), index=X_ny.index, columns=lr.classes_)
        print(f"[classify_nyabg] LR done in {time.time()-t0:.0f}s.", flush=True)
    else:
        print(f"[classify_nyabg] Min samples/class={min_count}; skipping LR.", flush=True)
        lr = None
        lr_proba = pd.DataFrame()

    return rf, lr, rf_proba, lr_proba


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", required=True)
    ap.add_argument("--nyabg_meta", required=True)
    ap.add_argument("--emp_meta", required=True)
    ap.add_argument("--hmp_meta", required=True)
    ap.add_argument("--hmp_label", required=True)
    ap.add_argument("--emp_label", required=True)
    ap.add_argument("--seed", type=int, default=20251101)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig_cal", required=True)
    ap.add_argument("--fig_imp", required=True)
    # New arguments for split pipeline
    ap.add_argument(
        "--target-ids-file",
        default=None,
        help="File with one sample ID per line (burial or control subset). "
        "If not provided, uses all NYABG samples.",
    )
    ap.add_argument(
        "--sample-type",
        default="nyabg",
        help="Label for target samples (burial/control/nyabg) used in output",
    )
    args = ap.parse_args()

    clr = prepare_features(args.table)
    nyabg = read_metadata(args.nyabg_meta)
    emp = read_metadata(args.emp_meta)
    hmp = read_metadata(args.hmp_meta)

    # Determine target sample IDs
    if args.target_ids_file and os.path.exists(args.target_ids_file):
        with open(args.target_ids_file) as f:
            target_ids = [line.strip() for line in f if line.strip()]
        # Filter to those present in the feature table
        ny_ids = [sid for sid in target_ids if sid in clr.columns]
        print(f"[classify_nyabg] Using {len(ny_ids)} {args.sample_type} samples "
              f"from {args.target_ids_file} ({len(target_ids)} in file)")
    else:
        ny_ids = list(set(clr.columns) & set(nyabg.index))
        print(f"[classify_nyabg] Using all {len(ny_ids)} NYABG samples")

    if not ny_ids:
        raise SystemExit("No target samples found in feature table.")

    _, _, hmp_rf, hmp_lr = train_and_predict(clr, hmp, args.hmp_label, ny_ids, seed=args.seed)
    _, _, emp_rf, emp_lr = train_and_predict(clr, emp, args.emp_label, ny_ids, seed=args.seed)

    rows = []
    for sid in ny_ids:
        for model_name, proba in [
            ("HMP_RF", hmp_rf),
            ("HMP_L1LR", hmp_lr),
            ("EMP_RF", emp_rf),
            ("EMP_L1LR", emp_lr),
        ]:
            if sid in proba.index:
                for cls, p in proba.loc[sid].items():
                    rows.append((sid, args.sample_type, model_name, cls, float(p)))

    out = pd.DataFrame(
        rows, columns=["sample_id", "sample_type", "model", "class", "probability"]
    ).sort_values(["sample_id", "model", "probability"], ascending=[True, True, False])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {len(out)} rows for {len(ny_ids)} {args.sample_type} samples")

    os.makedirs(os.path.dirname(args.fig_cal), exist_ok=True)
    plt.figure()
    plt.title(f"Calibration ({args.sample_type})")
    plt.savefig(args.fig_cal, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.title(f"Feature importance ({args.sample_type})")
    plt.savefig(args.fig_imp, dpi=150, bbox_inches="tight")
    plt.close()
