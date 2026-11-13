# 43_feast_summaries.py
# By Carter Clinton, Ph.D.
"""Summaries of the FEAST runs reported in the revision (Additional file 11, sheet S5).

Per run: burial and control EMP/HMP/unknown means and SDs, number of samples with HMP >= 1%
and >= 5%, control maximum and Fisher exact test of HMP >= 1% (burial vs control); stability
across the five depth-matched replicates; body-site components (primary and contamination-filtered).
"""
import argparse, os
import pandas as pd
from scipy.stats import fisher_exact

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--results", default="results/")
ap.add_argument("--out-dir", default="results/revision/")
a = ap.parse_args(); R = a.results.rstrip("/") + "/"
runs = {"r500_primary": "feast_depth_matched/r500/", "r500_noHumanEMP": "feast_depth_matched/r500_noHumanEMP/",
        "contam_r500": "contam_sensitivity/feast_r500/", "burialmatched": "feast_depth_matched/burialmatched/",
        "baseline_11read": "feast_split/"}
rows = []
for k, d in runs.items():
    if not os.path.exists(R + d + "burial_source_props.tsv"): print("missing", k); continue
    b = pd.read_csv(R + d + "burial_source_props.tsv", sep="\t"); c = pd.read_csv(R + d + "control_source_props.tsv", sep="\t")
    nb1 = int((b.HMP >= 0.01).sum()); nc1 = int((c.HMP >= 0.01).sum())
    rows.append(dict(run=k, n_burial=len(b), EMP_mean=b.EMP.mean(), EMP_sd=b.EMP.std(), HMP_mean=b.HMP.mean(), HMP_sd=b.HMP.std(),
                     HMP_median=b.HMP.median(), Unk_mean=b.Unknown.mean(), Unk_sd=b.Unknown.std(), n_HMP_ge1pct=nb1,
                     n_HMP_ge5pct=int((b.HMP >= 0.05).sum()), ctrl_EMP_mean=c.EMP.mean(), ctrl_HMP_max=c.HMP.max(),
                     ctrl_Unk_mean=c.Unknown.mean(), ctrl_n_HMP_ge1pct=nc1,
                     fisher_p=fisher_exact([[nb1, len(b) - nb1], [nc1, len(c) - nc1]])[1]))
os.makedirs(a.out_dir, exist_ok=True)
pd.DataFrame(rows).to_csv(a.out_dir + "feast_all_runs_summary.tsv", sep="\t", index=False)
st = R + "feast_depth_matched/r500/stability/"
if os.path.isdir(st):
    reps = [pd.read_csv(st + f"replicate_{i}_burial.tsv", sep="\t").set_index("sample_id") for i in range(1, 6)
            if os.path.exists(st + f"replicate_{i}_burial.tsv")]
    H = pd.concat([r["HMP"] for r in reps], axis=1); U = pd.concat([r["Unknown"] for r in reps], axis=1)
    pd.DataFrame({"replicate": range(1, len(reps) + 1), "HMP_mean": H.mean().values, "Unknown_mean": U.mean().values,
                  "n_HMP_ge1pct": (H >= 0.01).sum().values}).to_csv(a.out_dir + "feast_r500_stability_summary.tsv", sep="\t", index=False)
    cl = (H >= 0.01).sum(1); print("HMP>=1% in all replicates:", int((cl == len(reps)).sum()), "in none:", int((cl == 0).sum()))
bs_rows = []
for lab, f in [("primary_r500_bodysite", "feast_depth_matched/r500/burial_source_props_by_bodysite.tsv"),
               ("contam_r500_bodysite", "contam_sensitivity/feast_r500_bodysite/burial_source_props_by_bodysite.tsv")]:
    if os.path.exists(R + f):
        d = pd.read_csv(R + f, sep="\t"); num = [c for c in d.columns if c != "sample_id"]
        bs_rows.append(dict(run=lab, **{c: d[c].mean() for c in num}, **{c + "_n_ge1pct": int((d[c] >= 0.01).sum()) for c in num}))
if bs_rows: pd.DataFrame(bs_rows).to_csv(a.out_dir + "feast_bodysite_primary_vs_contam.tsv", sep="\t", index=False)
print(pd.DataFrame(rows).set_index("run")[["HMP_mean", "n_HMP_ge1pct", "ctrl_HMP_max", "fisher_p"]].round(4).to_string())
