# 11c_classifier_cv.py
# By Carter Clinton, Ph.D.
"""
Held-out evaluation of the reference classifiers used in scripts 11 and 22.

Stratified k-fold cross-validation (default 5 folds) of the same models, trained on
the same CLR-transformed harmonized genus table:
  EMP_L1LR, EMP_RF   : EMP samples, label empo_3 (emp_metadata.tsv)
  HMP_L1LR, HMP_RF   : HMP samples, label hmp_body_subsite (hmp_metadata.tsv)
  HMP5_L1LR          : HMP samples, label cohort_subsite (metadata_cohort_split.tsv; null model of script 22)
Hyperparameters match scripts 11/22 (RF 200 trees; one-vs-rest L1-LR, liblinear, C = 1).
Classes with fewer samples than folds are dropped and reported.
Outputs (per model, written as each finishes):
  <out>/cv_summary.tsv        accuracy, balanced accuracy, macro F1, median max-probability
  <out>/cv_per_class.tsv      per-class recall/precision/support
  <out>/cv_predictions.tsv    out-of-fold top class and max-probability per sample
For the 18-subsite HMP models, accuracy after aggregation of subsites to the five sites is also reported.
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import clr_transform, read_qiime_tsv  # noqa: E402


def read_meta(path, label):
    m = pd.read_csv(path, sep="\t", dtype=str, comment=None)
    m = m.loc[:, [c for c in m.columns if c and not c.startswith("Unnamed")]]
    m = m.rename(columns={m.columns[0]: "sample_id"})
    m["sample_id"] = m["sample_id"].astype(str).str.strip()
    m = m[~m["sample_id"].str.startswith("#")].drop_duplicates("sample_id").set_index("sample_id")
    return m[label].astype(str).str.strip()


def make_model(kind, seed):
    if kind == "RF":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    return OneVsRestClassifier(LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                                  max_iter=1000, random_state=seed), n_jobs=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default="results/compositional/merged_L6genus.tsv")
    ap.add_argument("--emp-meta", default="data/emp/emp_metadata.tsv")
    ap.add_argument("--hmp-meta", default="data/hmp/hmp_metadata.tsv")
    ap.add_argument("--cohort-meta", default="results/metadata_cohort_split.tsv")
    ap.add_argument("--models", default="HMP5_L1LR,HMP_L1LR,HMP_RF,EMP_RF,EMP_L1LR")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20251101)
    ap.add_argument("--max-per-class", type=int, default=0, help="smoke testing only; 0 = all samples")
    ap.add_argument("--out", default="results/classifier_cv/")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t = read_qiime_tsv(a.table).apply(pd.to_numeric, errors="coerce").fillna(0)
    clr = pd.DataFrame(clr_transform(t.values), index=t.index, columns=t.columns)
    print(f"[cv] table {t.shape[0]} taxa x {t.shape[1]} samples", flush=True)

    coh = read_meta(a.cohort_meta, "cohort")
    sub5 = read_meta(a.cohort_meta, "cohort_subsite")
    labels = {
        "EMP": read_meta(a.emp_meta, "empo_3"),
        "HMP": read_meta(a.hmp_meta, "hmp_body_subsite"),
        "HMP5": sub5[coh.reindex(sub5.index) == "hmp"],
    }
    pools = {"EMP": set(coh[coh == "emp"].index), "HMP": set(coh[coh == "hmp"].index),
             "HMP5": set(coh[coh == "hmp"].index)}

    summ, percls, preds = [], [], []
    for mname in [m.strip() for m in a.models.split(",") if m.strip()]:
        ref, kind = mname.rsplit("_", 1)
        y = labels[ref]
        ids = sorted(set(y.index) & pools[ref] & set(clr.columns))
        y = y.loc[ids]
        y = y[(y != "") & (y.str.lower() != "nan")]
        vc = y.value_counts()
        dropped = vc[vc < a.folds]
        y = y[y.isin(vc[vc >= a.folds].index)]
        if a.max_per_class:
            y = y.groupby(y).head(a.max_per_class)
        X = clr[y.index].T.values
        print(f"[cv] {mname}: {len(y)} samples, {y.nunique()} classes, dropped {dict(dropped)}", flush=True)
        skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
        oof = pd.Series(index=y.index, dtype=object); pmax = pd.Series(index=y.index, dtype=float)
        t0 = time.time()
        for k, (tr, te) in enumerate(skf.split(X, y.values)):
            mdl = make_model(kind, a.seed).fit(X[tr], y.values[tr])
            pr = mdl.predict_proba(X[te])
            oof.iloc[te] = mdl.classes_[pr.argmax(1)]
            pmax.iloc[te] = pr.max(1)
            print(f"[cv]   fold {k+1}/{a.folds} done ({time.time()-t0:.0f}s)", flush=True)
        yt, yp = y.values, oof.values.astype(str)
        row = dict(model=mname, n=len(y), n_classes=y.nunique(), folds=a.folds,
                   accuracy=accuracy_score(yt, yp), balanced_accuracy=balanced_accuracy_score(yt, yp),
                   macro_f1=f1_score(yt, yp, average="macro"),
                   majority_class_rate=vc.max() / vc[vc >= a.folds].sum(),
                   median_maxprob_correct=float(np.median(pmax[yt == yp])) if (yt == yp).any() else np.nan,
                   median_maxprob_wrong=float(np.median(pmax[yt != yp])) if (yt != yp).any() else np.nan,
                   dropped_classes=";".join(f"{k}:{v}" for k, v in dropped.items()))
        if ref == "HMP":  # aggregate 18 subsites to the five sites
            agg = sub5.reindex(y.index)
            m = dict(zip(y.values, agg.values))
            yp5 = pd.Series(yp).map(lambda s: m.get(s, s)).values
            row["accuracy_aggregated_5site"] = accuracy_score(agg.values, yp5)
            row["balanced_accuracy_aggregated_5site"] = balanced_accuracy_score(agg.values, yp5)
        summ.append(row)
        p, r, f, s = precision_recall_fscore_support(yt, yp, labels=sorted(y.unique()), zero_division=0)
        for c, pi, ri, fi, si in zip(sorted(y.unique()), p, r, f, s):
            percls.append(dict(model=mname, cls=c, precision=pi, recall=ri, f1=fi, support=si))
        preds.append(pd.DataFrame(dict(model=mname, sample_id=y.index, true=yt, pred=yp, max_prob=pmax.values)))
        pd.DataFrame(summ).to_csv(os.path.join(a.out, "cv_summary.tsv"), sep="\t", index=False)
        pd.DataFrame(percls).to_csv(os.path.join(a.out, "cv_per_class.tsv"), sep="\t", index=False)
        pd.concat(preds).to_csv(os.path.join(a.out, "cv_predictions.tsv"), sep="\t", index=False)
        print(f"[cv] {mname}: acc {row['accuracy']:.3f} bal {row['balanced_accuracy']:.3f} "
              f"F1 {row['macro_f1']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    print("[cv] DONE", flush=True)


if __name__ == "__main__":
    main()
