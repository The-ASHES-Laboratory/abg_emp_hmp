#!/usr/bin/env python3
"""34_make_depth_matched_sources.py
Create genus tables in which every EMP and HMP (source) sample is rarefied, without replacement, to
  (a) a fixed depth (--fixed, default 500 reads), or
  (b) a depth drawn (seed) from the empirical burial-sample depth distribution ("burial-matched"),
dropping source samples whose depth is below the target. Burial and control (sink) samples are left unchanged.
Writes <out-dir>/merged_L6genus_r<fixed>.tsv, merged_L6genus_burialmatched.tsv, matching metadata files
(restricted to retained samples) and source_depths.tsv. 07_feast_source_tracking.R is then run on each table.
"""
import argparse, os, sys, importlib.util
import numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); SCRIPTS=os.path.join(os.path.dirname(HERE),"scripts")
spec=importlib.util.spec_from_file_location("beta03", os.path.join(SCRIPTS,"03_beta_ordination.py")); beta03=importlib.util.module_from_spec(spec); spec.loader.exec_module(beta03)
def rarefy(counts, depth, rng):
    counts=np.asarray(counts).round().astype(np.int64); n=counts.sum()
    if n<depth: return None
    if n==depth: return counts
    idx=np.repeat(np.arange(len(counts)), counts); pick=rng.choice(idx, size=depth, replace=False); return np.bincount(pick, minlength=len(counts))
def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="results/compositional/merged_L6genus.tsv"); ap.add_argument("--metadata", default="results/metadata_cohort_split.tsv")
    ap.add_argument("--fixed", type=int, default=500); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out-dir", default="results/feast_depth_matched/")
    a=ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True); rng=np.random.default_rng(a.seed)
    tab=beta03.load_genus_table(a.table); meta=pd.read_csv(a.metadata, sep="\t", dtype=str); meta=meta[meta["sample-id"].isin(tab.columns)]
    coh=meta.set_index("sample-id").cohort; depth=tab.sum(axis=0)
    bur_depths=depth[coh[coh=="burial"].index].values.astype(int); print("burial depth quantiles:", np.percentile(bur_depths,[0,25,50,75,100]))
    src=coh[coh.isin(["emp","hmp"])].index.tolist(); sinks=coh[coh.isin(["burial","control"])].index.tolist()
    for label,targets in (("r%d"%a.fixed, dict.fromkeys(src, a.fixed)), ("burialmatched", dict(zip(src, rng.choice(bur_depths, size=len(src), replace=True))))):
        cols={}; kept=[]; rec=[]
        for s in src:
            r=rarefy(tab[s].values, int(targets[s]), rng); rec.append(dict(sample_id=s, cohort=coh[s], original_depth=int(depth[s]), target_depth=int(targets[s]), retained=r is not None))
            if r is not None: cols[s]=r; kept.append(s)
        out=pd.DataFrame(cols, index=tab.index); out=pd.concat([out, tab[sinks]], axis=1)
        out.index.name="taxon"; out.to_csv(os.path.join(a.out_dir, f"merged_L6genus_{label}.tsv"), sep="\t")
        meta[meta["sample-id"].isin(kept+sinks)].to_csv(os.path.join(a.out_dir, f"metadata_cohort_split_{label}.tsv"), sep="\t", index=False)
        rec=pd.DataFrame(rec); rec.to_csv(os.path.join(a.out_dir, f"source_depths_{label}.tsv"), sep="\t", index=False)
        print(label, "sources retained:", rec.groupby("cohort").retained.sum().to_dict(), "of", rec.groupby("cohort").size().to_dict())
if __name__=="__main__": main()
