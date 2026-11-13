# 42_human_genera_per_sample.py
# By Carter Clinton, Ph.D.
"""Per-sample detection of reference-defined human-associated genera (Additional file 11, S11).

Human-associated genera: taxa in the burial source-network table present in HMP with HMP
prevalence >= 10% and higher prevalence in HMP than in EMP. For each burial and control
sample: reads and genera detected in the full-depth genus table, and the mean number of
genera detected over 100 rarefactions to 1,000 reads (seed 42; samples below 1,000 reads blank).
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

def short(t):
    parts = [p for p in t.split(";") if p.strip() and not p.strip().endswith("__")]
    return parts[-1].split("__")[-1] if parts else t

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--genus-table", default="results/upstream/merged_export/feature-table.tsv")
ap.add_argument("--metadata", default="results/metadata_cohort_split.tsv")
ap.add_argument("--network-taxa", default="results/networks/burial/overall/source_network_taxa.tsv")
ap.add_argument("--depth", type=int, default=1000); ap.add_argument("--reps", type=int, default=100)
ap.add_argument("--out", default="results/revision/human_genera_per_sample.tsv")
a = ap.parse_args()
md = pd.read_csv(a.metadata, sep="\t", dtype=str)
nyb = md[md.cohort.isin(["burial", "control"])]["sample-id"].tolist(); coh = md.set_index("sample-id").cohort
with open(a.genus_table) as fh: fh.readline(); hdr = fh.readline().rstrip("\n").split("\t")
T = pd.read_csv(a.genus_table, sep="\t", skiprows=1, usecols=[hdr[0]] + [c for c in hdr if c in set(nyb)], index_col=0)
nb = pd.read_csv(a.network_taxa, sep="\t")
ha = nb[(nb.present_hmp.astype(str).str.strip() == "yes") & (nb.hmp_prevalence >= 0.10) & (nb.hmp_prevalence > nb.emp_prevalence)]
hset = set(ha.taxon) & set(T.index); print("human-associated genera:", len(hset))
rng = np.random.default_rng(42); keep = [c for c in T.columns if T[c].sum() >= a.depth]; res = {c: [] for c in keep}
for _ in range(a.reps):
    for c in keep:
        v = T[c].values.astype(int); idx = np.repeat(np.arange(len(v)), v)
        sub = np.bincount(rng.choice(idx, a.depth, replace=False), minlength=len(v))
        res[c].append(sum(1 for i in np.nonzero(sub)[0] if T.index[i] in hset))
rows = []
for s in T.columns:
    v = T[s]; hv = v[v.index.isin(hset) & (v > 0)].sort_values(ascending=False)
    rows.append(dict(sample=s, cohort=coh[s], reads=int(v.sum()), human_assoc_reads=int(hv.sum()),
                     human_assoc_pct=round(100 * hv.sum() / v.sum(), 3), n_human_assoc_genera_unrarefied=len(hv),
                     n_human_assoc_genera_r1000_mean=(round(float(np.mean(res[s])), 2) if s in res else None),
                     genera_detected="; ".join(f"{short(t)} ({int(n)})" for t, n in hv.items())))
S = pd.DataFrame(rows).sort_values(["cohort", "sample"])
b = S[(S.cohort == "burial")].n_human_assoc_genera_r1000_mean.dropna(); k = S[(S.cohort == "control")].n_human_assoc_genera_r1000_mean.dropna()
print("median r1000 burial", b.median(), "control", k.median(), "Mann-Whitney p", mannwhitneyu(b, k, alternative="two-sided").pvalue)
os.makedirs(os.path.dirname(a.out), exist_ok=True); S.to_csv(a.out, sep="\t", index=False)
