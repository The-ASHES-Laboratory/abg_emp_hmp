#!/usr/bin/env python3
"""31_coverage_richness.py
Coverage-based richness (Chao & Jost 2012) on genus counts, as an alternative to Chao1/observed richness
at a fixed rarefaction depth. For burial, control, and the seed-42 stratified 500-sample EMP and HMP
subsamples used by 03_beta_ordination.py:
  - sample coverage C_hat(n) = 1 - f1/n * [ (n-1) f1 / ((n-1) f1 + 2 f2) ]
  - coverage-based rarefaction: richness S(m) at the sample size m whose expected coverage equals the
    target coverage (bisection on the exact expected-coverage formula; interpolation only, m <= n).
Targets: fixed sample coverage of 0.90 and 0.95
(samples whose observed coverage is below a target are excluded from that target and counted).
Tests: Kruskal-Wallis across cohorts; pairwise Mann-Whitney U with Benjamini-Hochberg correction.
Outputs under --out-dir: sample_coverage_per_sample.tsv, coverage_richness_per_sample.tsv,
coverage_richness_summary.tsv, coverage_richness_tests.tsv
"""
import argparse, importlib.util, os, sys, itertools
import numpy as np, pandas as pd
from scipy.special import gammaln
from scipy.stats import kruskal, mannwhitneyu
HERE=os.path.dirname(os.path.abspath(__file__)); SCRIPTS=os.path.join(os.path.dirname(HERE),"scripts"); sys.path.insert(0,SCRIPTS)
spec=importlib.util.spec_from_file_location("beta03", os.path.join(SCRIPTS,"03_beta_ordination.py")); beta03=importlib.util.module_from_spec(spec); spec.loader.exec_module(beta03)

def coverage_obs(x):
    x=x[x>0]; n=x.sum(); f1=(x==1).sum(); f2=(x==2).sum()
    if n==0: return np.nan
    if f1==0: return 1.0
    a=(n-1)*f1/((n-1)*f1+2*f2) if (f1>0 or f2>0) else 0.0
    return 1-(f1/n)*a
def logC(n,k):  # log binomial coefficient, vectorised
    return gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)
def exp_coverage(x,m):
    x=x[x>0]; n=x.sum()
    if m>=n: return coverage_obs(x)
    mask=(n-x)>=m; terms=np.zeros_like(x,dtype=float)
    terms[mask]=np.exp(logC(n-x[mask],m)-logC(n-1,m))
    return 1-np.sum((x/n)*terms)
def exp_richness(x,m):
    x=x[x>0]; n=x.sum()
    if m>=n: return float(len(x))
    mask=(n-x)>=m; terms=np.zeros_like(x,dtype=float); terms[mask]=np.exp(logC(n-x[mask],m)-logC(n,m))
    return len(x)-terms.sum()
def richness_at_coverage(x,target):
    x=x[x>0]; n=int(x.sum())
    if coverage_obs(x)<target-1e-9: return np.nan, np.nan
    lo,hi=1,n
    while hi-lo>1:
        mid=(lo+hi)//2
        if exp_coverage(x,mid)>=target: hi=mid
        else: lo=mid
    m=hi if exp_coverage(x,lo)<target else lo
    return exp_richness(x,m), m

def bh(p):
    p=np.asarray(p,float); n=len(p); o=np.argsort(p); r=np.empty(n); r[o]=np.minimum.accumulate((p[o]*n/np.arange(1,n+1))[::-1])[::-1]; return np.minimum(r,1)

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="results/compositional/merged_L6genus.tsv"); ap.add_argument("--metadata", default="results/metadata_cohort_split.tsv")
    ap.add_argument("--n-subsample", type=int, default=500); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--targets", default="0.90,0.95"); ap.add_argument("--out-dir", default="results/alpha_diversity/")
    a=ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    table=beta03.load_genus_table(a.table); meta=pd.read_csv(a.metadata, sep="\t", dtype=str).set_index("sample-id"); meta=meta[meta.index.isin(table.columns)]
    rng=np.random.default_rng(a.seed); ids=[s for s in beta03.get_subsample(meta, a.n_subsample, rng) if s in table.columns]
    sub=table[ids]; coh=meta.loc[ids,"cohort"]
    rows=[]
    for s in ids:
        x=sub[s].to_numpy(dtype=float).round().astype(int); rows.append(dict(sample_id=s, cohort=coh[s], depth=int(x.sum()), observed=int((x>0).sum()), singletons=int((x==1).sum()), doubletons=int((x==2).sum()), coverage=coverage_obs(x)))
    cov=pd.DataFrame(rows); cov.to_csv(os.path.join(a.out_dir,"sample_coverage_per_sample.tsv"), sep="\t", index=False)
    common=float(np.nanmin(cov.coverage)); targets={f"c{t}":float(t) for t in a.targets.split(",")}  # minimum observed coverage is 1.0 without singletons; not used as a target
    print("coverage per cohort (median):", cov.groupby("cohort").coverage.median().round(4).to_dict(), " common target:", round(common,4))
    out=[]
    for name,t in targets.items():
        for s in ids:
            x=sub[s].to_numpy(dtype=float).round().astype(int); S,m=richness_at_coverage(x,t)
            out.append(dict(sample_id=s, cohort=coh[s], target=name, target_coverage=t, richness_at_coverage=S, m_reads=m))
    rc=pd.DataFrame(out); rc.to_csv(os.path.join(a.out_dir,"coverage_richness_per_sample.tsv"), sep="\t", index=False)
    summ=rc.groupby(["target","target_coverage","cohort"]).agg(n_included=("richness_at_coverage",lambda v: int(v.notna().sum())), n_total=("sample_id","size"), median=("richness_at_coverage","median"), mean=("richness_at_coverage","mean"), sd=("richness_at_coverage","std")).reset_index()
    summ.to_csv(os.path.join(a.out_dir,"coverage_richness_summary.tsv"), sep="\t", index=False); print(summ.to_string(index=False))
    tests=[]
    for name,t in targets.items():
        d=rc[(rc.target==name)&rc.richness_at_coverage.notna()]; groups={c:d[d.cohort==c].richness_at_coverage.values for c in sorted(d.cohort.unique())}
        H,p=kruskal(*[v for v in groups.values() if len(v)>0]); tests.append(dict(target=name, target_coverage=t, test="Kruskal-Wallis", comparison="all", statistic=H, p=p, p_BH=np.nan))
        pr=[]
        for g1,g2 in itertools.combinations(groups,2):
            if len(groups[g1])>1 and len(groups[g2])>1:
                U,pp=mannwhitneyu(groups[g1],groups[g2],alternative="two-sided"); pr.append(dict(target=name,target_coverage=t,test="Mann-Whitney two-sided",comparison=f"{g1}_vs_{g2}",statistic=U,p=pp,median1=np.median(groups[g1]),median2=np.median(groups[g2])))
        if pr:
            pq=bh([r["p"] for r in pr])
            for r,q in zip(pr,pq): r["p_BH"]=q
            tests+=pr
    pd.DataFrame(tests).to_csv(os.path.join(a.out_dir,"coverage_richness_tests.tsv"), sep="\t", index=False); print(pd.DataFrame(tests).to_string(index=False))
if __name__=="__main__": main()
