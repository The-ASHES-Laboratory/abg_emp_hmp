#!/usr/bin/env python3
"""32_gradient_tests.py
Anatomical-proximity gradient re-tests requested by Reviewer 2 (minor 2; major 2) and Reviewer 1 (major 4).
Categories (same keyword rules as 19_cranium_analysis.py): Cranium (body_region contains cranium/skull/head/cranial),
Soil Matrix (contains matrix/soil), Other Body (all other burial samples), Control (control soils).
Tests:
  A. Jonckheere-Terpstra trend, decreasing Cranium > Other Body > Soil Matrix, WITHIN BURIAL SAMPLES ONLY
     (controls excluded), two-sided p by permutation (--permutations) and normal approximation; the published
     4-level version (with controls) is also reported for comparison.
  B. Body-associated (Cranium + Other Body) vs Soil Matrix: Mann-Whitney U two-sided and one-sided, Cliff's d.
  C. Cranium vs non-cranium (within burial): Mann-Whitney two-sided, Cliff's d, Fisher exact on detection.
  D. HMP proportion vs read depth (burial only): Spearman rho; logistic regression of any-HMP on log10 depth (statsmodels if available).
  E. Body-site components (from stratified FEAST by body site): oral+nasal and skin components vs the 3-level
     burial gradient (JT two-sided) and vs depth.
Inputs: FEAST overall table (sample_id, EMP, HMP, Unknown), control table, by-body-site table, NYABG metadata
(QIIME2 format), genus table or depth table for read depth.
Output: <out-dir>/gradient_tests.tsv (one row per test), gradient_group_summary.tsv, gradient_sample_table.tsv
"""
import argparse, os, sys, itertools
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, fisher_exact, norm

CRAN=["cranium","skull","head","cranial"]; MATRIX=["matrix","soil"]
def classify(region):
    r=str(region).lower()
    if any(k in r for k in CRAN): return "Cranium"
    if any(k in r for k in MATRIX): return "Soil Matrix"
    return "Other Body"
def cliffs_d(x,y):
    x=np.asarray(x); y=np.asarray(y); gt=sum((xi>y).sum() for xi in x); lt=sum((xi<y).sum() for xi in x); return (gt-lt)/(len(x)*len(y))
def jt_stat(groups):  # groups ordered list of arrays; JT = sum over i<j of #(x_i < x_j) + 0.5 ties  (increasing alternative)
    J=0.0
    for i,j in itertools.combinations(range(len(groups)),2):
        a=groups[i]; b=groups[j]
        J+=sum(((a_<b).sum()+0.5*(a_==b).sum()) for a_ in a)
    return J
def jt_test(groups, permutations, rng, direction="decreasing"):
    g=[np.asarray(v,float) for v in groups]
    if direction=="decreasing": g=g[::-1]  # test increasing on reversed order
    J=jt_stat(g); n=np.array([len(v) for v in g]); N=n.sum()
    mu=(N**2-(n**2).sum())/4; var=(N**2*(2*N+3)-(n**2*(2*n+3)).sum())/72  # no tie correction in var (conservative-ish); permutation p is primary
    z=(J-mu)/np.sqrt(var) if var>0 else np.nan
    allv=np.concatenate(g); sizes=n; cnt=0; perm_stats=[]
    for _ in range(permutations):
        p=rng.permutation(allv); idx=0; gp=[]
        for s in sizes: gp.append(p[idx:idx+s]); idx+=s
        perm_stats.append(jt_stat(gp))
    perm_stats=np.asarray(perm_stats)
    p_two=(np.sum(np.abs(perm_stats-mu)>=abs(J-mu)-1e-12)+1)/(permutations+1); p_one=(np.sum(perm_stats>=J-1e-12)+1)/(permutations+1)
    return dict(JT=J, z=z, p_normal_two_sided=2*(1-norm.cdf(abs(z))) if not np.isnan(z) else np.nan, p_perm_two_sided=p_two, p_perm_one_sided=p_one)

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feast-burial", default="results/feast_split/burial_source_props.tsv"); ap.add_argument("--feast-control", default="results/feast_split/control_source_props.tsv")
    ap.add_argument("--feast-bodysite", default="results/feast_split/burial_source_props_by_bodysite.tsv"); ap.add_argument("--nyabg-meta", default="data/nyabg/abg_16s_meta.tsv")
    ap.add_argument("--table", default="results/compositional/merged_L6genus.tsv"); ap.add_argument("--depth-table", default=None, help="optional TSV with sample_id, depth; else computed from --table")
    ap.add_argument("--permutations", type=int, default=9999); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out-dir", default="results/gradient/")
    a=ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True); rng=np.random.default_rng(a.seed)
    fb=pd.read_csv(a.feast_burial, sep="\t"); fb=fb.rename(columns={fb.columns[0]:"sample_id"}); fb["sample_id"]=fb.sample_id.astype(str)
    fc=pd.read_csv(a.feast_control, sep="\t"); fc=fc.rename(columns={fc.columns[0]:"sample_id"}); fc["sample_id"]=fc.sample_id.astype(str)
    hcol=[c for c in fb.columns if c.lower()=="hmp"][0]
    with open(a.nyabg_meta) as f: hdr=f.readline().rstrip("\n").lstrip("#").split("\t")
    meta=pd.read_csv(a.nyabg_meta, sep="\t", skiprows=2, header=None, names=hdr, dtype=str, usecols=range(len(hdr))); meta=meta.rename(columns={meta.columns[0]:"sample_id"}).set_index("sample_id")
    bcol=[c for c in meta.columns if c.lower().replace("_","")=="bodyregion"][0]
    burcol=[c for c in meta.columns if c.lower()=="burial"]; burcol=burcol[0] if burcol else None
    # depth
    if a.depth_table: dep=pd.read_csv(a.depth_table, sep="\t", dtype={0:str}); dep=dep.set_index(dep.columns[0]).iloc[:,0]
    else:
        tab=pd.read_csv(a.table, sep="\t", index_col=0, usecols=lambda c: True); dep=tab.sum(axis=0); dep.index=dep.index.astype(str)
    df=fb[["sample_id",hcol]].rename(columns={hcol:"hmp"}).copy(); df["body_region"]=df.sample_id.map(meta[bcol]); df["level"]=df.body_region.apply(classify)
    df["burial_id"]=df.sample_id.map(meta[burcol]) if burcol else np.nan; df["depth"]=df.sample_id.map(dep); df["cohort"]="burial"
    dc=fc[["sample_id",hcol]].rename(columns={hcol:"hmp"}).copy(); dc["body_region"]="control"; dc["level"]="Control"; dc["burial_id"]=np.nan; dc["depth"]=dc.sample_id.map(dep); dc["cohort"]="control"
    allv=pd.concat([df,dc],ignore_index=True); allv.to_csv(os.path.join(a.out_dir,"gradient_sample_table.tsv"), sep="\t", index=False)
    order3=["Cranium","Other Body","Soil Matrix"]; order4=order3+["Control"]
    summ=allv.groupby("level").hmp.agg(n="size", mean="mean", median="median", sd="std", detection_rate=lambda v: (v>0).mean()).reindex(order4).reset_index(); summ.to_csv(os.path.join(a.out_dir,"gradient_group_summary.tsv"), sep="\t", index=False); print(summ.to_string(index=False))
    T=[]
    g3=[df[df.level==l].hmp.values for l in order3]; r=jt_test(g3, a.permutations, rng); T.append(dict(test="JT decreasing trend, within burial (3 levels)", n="/".join(str(len(x)) for x in g3), **r))
    g4=[allv[allv.level==l].hmp.values for l in order4]; r=jt_test(g4, a.permutations, rng); T.append(dict(test="JT decreasing trend, 4 levels incl. controls (published design)", n="/".join(str(len(x)) for x in g4), **r))
    body=df[df.level.isin(["Cranium","Other Body"])].hmp.values; mat=df[df.level=="Soil Matrix"].hmp.values
    U,p2=mannwhitneyu(body,mat,alternative="two-sided"); _,p1=mannwhitneyu(body,mat,alternative="greater")
    T.append(dict(test="Body-associated vs soil matrix: Mann-Whitney", n=f"{len(body)}/{len(mat)}", U=U, p_two_sided=p2, p_one_sided_greater=p1, cliffs_d=cliffs_d(body,mat), median1=np.median(body), median2=np.median(mat), detect1=(body>0).mean(), detect2=(mat>0).mean()))
    cr=df[df.level=="Cranium"].hmp.values; nc=df[df.level!="Cranium"].hmp.values; U,p2=mannwhitneyu(cr,nc,alternative="two-sided"); _,p1=mannwhitneyu(cr,nc,alternative="greater")
    fo,fp=fisher_exact([[(cr>0).sum(),(cr==0).sum()],[(nc>0).sum(),(nc==0).sum()]])
    T.append(dict(test="Cranium vs non-cranium (burial): Mann-Whitney / Fisher detection", n=f"{len(cr)}/{len(nc)}", U=U, p_two_sided=p2, p_one_sided_greater=p1, cliffs_d=cliffs_d(cr,nc), fisher_OR=fo, fisher_p=fp, detect1=(cr>0).mean(), detect2=(nc>0).mean()))
    d2=df.dropna(subset=["depth"]); rho,ps=spearmanr(d2.depth, d2.hmp); T.append(dict(test="Spearman HMP proportion vs read depth (burial)", n=len(d2), rho=rho, p_two_sided=ps))
    for col in ["EMP","Unknown"]:
        if col in fb.columns:
            v=fb.set_index("sample_id").loc[d2.sample_id, col].values; rho,ps=spearmanr(d2.depth, v); T.append(dict(test=f"Spearman {col} proportion vs read depth (burial)", n=len(d2), rho=rho, p_two_sided=ps))
    try:
        import statsmodels.api as sm
        X=sm.add_constant(np.log10(d2.depth.astype(float))); y=(d2.hmp>0).astype(int); fit=sm.Logit(y,X).fit(disp=0)
        T.append(dict(test="Logistic any-HMP ~ log10(depth) (burial)", n=len(d2), coef_log10depth=fit.params.iloc[1], se=fit.bse.iloc[1], p_two_sided=fit.pvalues.iloc[1], odds_ratio_per_log10=float(np.exp(fit.params.iloc[1]))))
        # depth-adjusted rank regression of HMP on level (ordinal 0,1,2) within burial
        lev=d2.level.map({"Cranium":0,"Other Body":1,"Soil Matrix":2}).astype(float); X2=sm.add_constant(pd.DataFrame({"proximity_rank":lev,"log10_depth":np.log10(d2.depth.astype(float))}))
        f2=sm.OLS(d2.hmp.rank(), X2).fit(); T.append(dict(test="Rank(HMP) ~ proximity rank + log10 depth (OLS on ranks, burial)", n=len(d2), coef_proximity=f2.params["proximity_rank"], p_proximity=f2.pvalues["proximity_rank"], coef_log10depth=f2.params["log10_depth"], p_depth=f2.pvalues["log10_depth"]))
    except Exception as e:
        T.append(dict(test="Logistic/OLS depth models", note=f"statsmodels unavailable or failed: {e}"))
    if a.feast_bodysite and os.path.exists(a.feast_bodysite):
        bs=pd.read_csv(a.feast_bodysite, sep="\t"); bs=bs.rename(columns={bs.columns[0]:"sample_id"}); bs["sample_id"]=bs.sample_id.astype(str)
        comp={"oral+nasal":[c for c in bs.columns if c.lower() in ("hmp_oral","hmp_nasal")], "skin":[c for c in bs.columns if c.lower()=="hmp_skin"], "stool+vaginal":[c for c in bs.columns if c.lower() in ("hmp_stool","hmp_vaginal")]}
        bs["level"]=bs.sample_id.map(meta[bcol]).apply(classify); bs["depth"]=bs.sample_id.map(dep)
        for name,cols in comp.items():
            if not cols: continue
            v=bs[cols].sum(axis=1); gg=[v[bs.level==l].values for l in order3]; r=jt_test(gg, a.permutations, rng)
            T.append(dict(test=f"JT decreasing trend, {name} FEAST component, within burial (n={len(bs)})", n="/".join(str(len(x)) for x in gg), means="/".join(f"{np.mean(x):.4f}" for x in gg), detect="/".join(f"{(x>0).mean():.2f}" for x in gg), **r))
            rho,ps=spearmanr(bs.depth, v, nan_policy="omit"); T.append(dict(test=f"Spearman {name} component vs depth", n=int(bs.depth.notna().sum()), rho=rho, p_two_sided=ps))
    out=pd.DataFrame(T); out.to_csv(os.path.join(a.out_dir,"gradient_tests.tsv"), sep="\t", index=False); print(out.to_string(index=False))
if __name__=="__main__": main()
