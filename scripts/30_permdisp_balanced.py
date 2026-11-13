#!/usr/bin/env python3
"""30_permdisp_balanced.py
(1) PERMDISP (betadisper-equivalent, skbio permdisp, centroid, 999 perms) global and pairwise for Bray-Curtis
    and Aitchison (CLR pseudo=0.5) on the SAME 5 stratified subsamples (seeds 42..46, n=500 EMP + 500 HMP +
    all burial + all control) used by 03_beta_ordination.py / 04_aitchison_ordination.py.
(2) Balanced-design PERMANOVA: --n-draws random draws (seed 42) of 70 EMP soil (cohort_subsite == emp_soil),
    70 HMP (stratified across HMP subsites), all burial, all control; global and pairwise R2 and PERMDISP p.
Outputs under --out-dir: permdisp_bc_global.tsv, permdisp_bc_pairwise.tsv, permdisp_ait_global.tsv,
    permdisp_ait_pairwise.tsv, table2_permanova_permdisp.tsv, balanced_permanova.tsv, balanced_permanova_draws.tsv
"""
import argparse, importlib.util, os, sys, itertools
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
from skbio import DistanceMatrix
from skbio.stats.distance import permanova, permdisp

HERE=os.path.dirname(os.path.abspath(__file__)); SCRIPTS=os.path.join(os.path.dirname(HERE),"scripts")
sys.path.insert(0, SCRIPTS)
spec=importlib.util.spec_from_file_location("beta03", os.path.join(SCRIPTS,"03_beta_ordination.py")); beta03=importlib.util.module_from_spec(spec); spec.loader.exec_module(beta03)
from utils import clr_transform

def dm_bc(table, ids):
    valid=[s for s in ids if s in table.columns]; sub=table[valid].T; sub=sub[sub.sum(axis=1)>0]
    rel=sub.div(sub.sum(axis=1),axis=0); d=squareform(pdist(rel.values, metric="braycurtis")); d=np.nan_to_num(d, nan=1.0)
    return DistanceMatrix(d, ids=sub.index.tolist())
def dm_ait(table, ids):
    valid=[s for s in ids if s in table.columns]; sub=table[valid]; sub=sub.loc[:, sub.sum(axis=0)>0]
    clr=clr_transform(sub, pseudo=0.5); d=squareform(pdist(clr.T.values, metric="euclidean"))
    return DistanceMatrix(d, ids=sub.columns.tolist())
def r2_from_F(F,k,N): return F*(k-1)/(F*(k-1)+(N-k))

def run_tests(dm, groups, perms, pairwise=True):
    g=groups.loc[list(dm.ids)]; out=[]
    pa=permanova(dm, g, permutations=perms); pdisp=permdisp(dm, g, permutations=perms, test="centroid")
    k=g.nunique(); N=len(g)
    out.append(dict(comparison="global", n=N, k=k, pseudo_F=pa["test statistic"], R2=r2_from_F(pa["test statistic"],k,N), p_permanova=pa["p-value"], permdisp_F=pdisp["test statistic"], p_permdisp=pdisp["p-value"]))
    if pairwise:
        for a,b in itertools.combinations(sorted(g.unique()),2):
            ids=g[g.isin([a,b])].index.tolist(); sd=dm.filter(ids); sg=g.loc[ids]
            pa2=permanova(sd, sg, permutations=perms); pd2=permdisp(sd, sg, permutations=perms, test="centroid")
            out.append(dict(comparison=f"{a}_vs_{b}", n=len(ids), k=2, pseudo_F=pa2["test statistic"], R2=r2_from_F(pa2["test statistic"],2,len(ids)), p_permanova=pa2["p-value"], permdisp_F=pd2["test statistic"], p_permdisp=pd2["p-value"]))
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="results/compositional/merged_L6genus.tsv"); ap.add_argument("--metadata", default="results/metadata_cohort_split.tsv")
    ap.add_argument("--n-subsample", type=int, default=500); ap.add_argument("--n-replicates", type=int, default=5); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--permutations", type=int, default=999); ap.add_argument("--n-draws", type=int, default=100); ap.add_argument("--balanced-n", type=int, default=70)
    ap.add_argument("--out-dir", default="results/ordination/")
    a=ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    table=beta03.load_genus_table(a.table)
    meta=pd.read_csv(a.metadata, sep="\t", dtype=str).set_index("sample-id"); meta=meta[meta.index.isin(table.columns)]
    groups=meta["cohort"]
    # (1) PERMDISP on the 03/04 subsamples
    reps={ "bc":[], "ait":[] }
    for r in range(a.n_replicates):
        rng=np.random.default_rng(a.seed+r); ids=beta03.get_subsample(meta, a.n_subsample, rng)
        for name,fn in (("bc",dm_bc),("ait",dm_ait)):
            df=run_tests(fn(table, ids), groups, a.permutations); df["replicate"]=r+1; df["seed"]=a.seed+r; reps[name].append(df)
    for name in reps:
        allr=pd.concat(reps[name]); allr.to_csv(os.path.join(a.out_dir,f"permdisp_{name}_replicates.tsv"), sep="\t", index=False)
        agg=allr.groupby("comparison").agg(n=("n","first"), R2_mean=("R2","mean"), R2_sd=("R2","std"), pseudo_F_mean=("pseudo_F","mean"), p_permanova_mean=("p_permanova","mean"), permdisp_F_mean=("permdisp_F","mean"), permdisp_F_sd=("permdisp_F","std"), p_permdisp_mean=("p_permdisp","mean"), p_permdisp_max=("p_permdisp","max")).reset_index()
        agg[agg.comparison=="global"].to_csv(os.path.join(a.out_dir,f"permdisp_{name}_global.tsv"), sep="\t", index=False)
        agg[agg.comparison!="global"].to_csv(os.path.join(a.out_dir,f"permdisp_{name}_pairwise.tsv"), sep="\t", index=False)
        agg["metric"]="Bray-Curtis" if name=="bc" else "Aitchison"; reps[name]=agg
    t2=pd.concat([reps["bc"],reps["ait"]]); t2.to_csv(os.path.join(a.out_dir,"table2_permanova_permdisp.tsv"), sep="\t", index=False)
    print(t2.to_string(index=False))
    # (2) balanced design
    rng=np.random.default_rng(a.seed); draws=[]
    emp_soil=meta[(meta.cohort=="emp")&(meta.cohort_subsite=="emp_soil")].index.to_numpy(); hmp_meta=meta[meta.cohort=="hmp"]
    bur=meta[meta.cohort=="burial"].index.tolist(); ctl=meta[meta.cohort=="control"].index.tolist()
    for d in range(a.n_draws):
        es=rng.choice(emp_soil, a.balanced_n, replace=False).tolist(); hs=beta03.stratified_subsample(hmp_meta.assign(cohort="hmp"), "hmp", "cohort_subsite", a.balanced_n, rng)
        ids=es+hs+bur+ctl; g=groups.copy(); g.loc[es]="emp_soil"
        for name,fn in (("bc",dm_bc),("ait",dm_ait)):
            df=run_tests(fn(table, ids), g, a.permutations); df["draw"]=d+1; df["metric"]=name; draws.append(df)
    dd=pd.concat(draws); dd.to_csv(os.path.join(a.out_dir,"balanced_permanova_draws.tsv"), sep="\t", index=False)
    q=dd.groupby(["metric","comparison"]).agg(n=("n","first"), R2_mean=("R2","mean"), R2_sd=("R2","std"), R2_p2_5=("R2",lambda x: np.percentile(x,2.5)), R2_p97_5=("R2",lambda x: np.percentile(x,97.5)), frac_p_permanova_le_0_05=("p_permanova",lambda x: np.mean(x<=0.05)), permdisp_p_median=("p_permdisp","median"), frac_p_permdisp_le_0_05=("p_permdisp",lambda x: np.mean(x<=0.05))).reset_index()
    q.to_csv(os.path.join(a.out_dir,"balanced_permanova.tsv"), sep="\t", index=False); print(q.to_string(index=False))
if __name__=="__main__": main()
