#!/usr/bin/env python3
"""10b_da_consensus_oriented.py
Cross-method differential-abundance consensus (ANCOM-BC2 ∩ ALDEx2) with EXPLICIT sign orientation.

Why: 08_ancombc2.R sets factor levels c(group2, group1), so ANCOM-BC2 lfc > 0 = enriched in group1
(the first-named group). 09_aldex2.R passes character conditions to aldex(); ALDEx2 (clr_effect.r)
computes diff.btw = level2 - level1 with ALPHABETICAL levels, so effect > 0 = enriched in the
alphabetically LATER label. With labels burial < control < emp < hmp, group1 is alphabetically first in
all five comparisons, so the raw ALDEx2 effect sign is opposite to ANCOM-BC2 for every comparison.
This script re-orients the ALDEx2 effect to the first-named group and records concordance.

Inputs : results/diff_abun/ancombc2_split/ancombc2_{comparison}_{level}.tsv, results/diff_abun/aldex2/aldex2_{comparison}_{level}.tsv
Outputs: <out-dir>/da_consensus_oriented.tsv, da_consensus_oriented_summary.tsv, table1_consensus_taxa.tsv,
         fig6b_effect_sizes.tsv, additional_file10_effect_sizes.tsv
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd

COMPARISONS = [("burial","emp"),("burial","hmp"),("control","emp"),("control","hmp"),("burial","control")]
LEVELS = ["L6","L5"]

def load_ancombc2(path):
    df = pd.read_csv(path, sep="\t")
    if "taxon" not in df.columns: df = df.rename(columns={df.columns[0]:"taxon"})
    q = [c for c in df.columns if c.startswith("q_") and "Intercept" not in c]
    l = [c for c in df.columns if c.startswith("lfc_") and "Intercept" not in c]
    if not q or not l or "error_note" in df.columns and df["error_note"].notna().any() and len(df)<=1:
        return None
    out = df[["taxon", l[0], q[0]]].rename(columns={l[0]:"ancombc2_lfc", q[0]:"ancombc2_q"})
    return out

def load_aldex2(path):
    df = pd.read_csv(path, sep="\t")
    if "taxon" not in df.columns: df = df.rename(columns={df.columns[0]:"taxon"})
    df = df.loc[:, ~df.columns.duplicated()]
    qcol = None
    for cand in ["wi.eBH","we.eBH","q","padj"]:   # wi.eBH first, matching published 10_da_consensus.py
        if cand in df.columns: qcol = cand; break
    if qcol is None:
        for c in df.columns:
            if c.lower().endswith("ebh"): qcol = c; break
    if qcol is None or "effect" not in df.columns: return None
    # Convention check: ALDEx2 writes rab.win.<level1>, rab.win.<level2>; diff.btw is level2 - level1 (clr_effect.r)
    rw=[c for c in df.columns if c.startswith("rab.win.")]
    if len(rw)==2 and "diff.btw" in df.columns:
        lv=sorted([c.replace("rab.win.","") for c in rw]); d=(df[f"rab.win.{lv[1]}"]-df[f"rab.win.{lv[0]}"])
        agree=float(np.mean(np.sign(d)==np.sign(df["diff.btw"])))
        print(f"    [convention check] {os.path.basename(path)}: levels={lv}; sign(rab.win.{lv[1]} - rab.win.{lv[0]}) agrees with sign(diff.btw) in {agree:.1%} of taxa -> effect>0 means higher in '{lv[1]}'")
    return df[["taxon","effect",qcol]].rename(columns={"effect":"aldex2_effect_raw", qcol:"aldex2_q"})

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ancombc2-dir", default="results/diff_abun/ancombc2_split/")
    ap.add_argument("--aldex2-dir", default="results/diff_abun/aldex2/")
    ap.add_argument("--out-dir", default="results/diff_abun/")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    rows=[]; summ=[]
    for g1,g2 in COMPARISONS:
        comp=f"{g1}_vs_{g2}"
        flip = sorted([g1,g2])[0]==g1   # ALDEx2 level1 = alphabetically first; effect = level2-level1
        for lev in LEVELS:
            fa = os.path.join(a.ancombc2_dir, f"ancombc2_{comp}_{lev}.tsv")
            fl = os.path.join(a.aldex2_dir, f"aldex2_{comp}_{lev}.tsv")
            if not (os.path.exists(fa) and os.path.exists(fl)):
                summ.append(dict(comparison=comp, level=lev, n_consensus=np.nan, n_concordant=np.nan, n_discordant=np.nan, note="missing input")); continue
            anc=load_ancombc2(fa); ald=load_aldex2(fl)
            if anc is None or ald is None:
                summ.append(dict(comparison=comp, level=lev, n_consensus=np.nan, n_concordant=np.nan, n_discordant=np.nan, note="unparseable/empty input")); continue
            m = anc.merge(ald, on="taxon", how="inner")
            m = m[(m.ancombc2_q < a.alpha) & (m.aldex2_q < a.alpha)].copy()
            m["aldex2_effect_oriented"] = -m["aldex2_effect_raw"] if flip else m["aldex2_effect_raw"]
            m["aldex2_sign_flipped"] = flip
            m["concordant"] = np.sign(m.ancombc2_lfc) == np.sign(m.aldex2_effect_oriented)
            m["direction_in_first_group"] = np.where(m.ancombc2_lfc>0, "enriched", "depleted")
            m["comparison"]=comp; m["level"]=lev; m["group1"]=g1; m["group2"]=g2
            rows.append(m)
            summ.append(dict(comparison=comp, level=lev, n_consensus=len(m), n_concordant=int(m.concordant.sum()), n_discordant=int((~m.concordant).sum()),
                             note=f"ALDEx2 sign flipped={flip}; discordant: " + ";".join(m.loc[~m.concordant,"taxon"].astype(str).tolist())))
    cons = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    cons.to_csv(os.path.join(a.out_dir,"da_consensus_oriented.tsv"), sep="\t", index=False)
    pd.DataFrame(summ).to_csv(os.path.join(a.out_dir,"da_consensus_oriented_summary.tsv"), sep="\t", index=False)
    if len(cons):
        t1 = cons[["taxon","level","comparison","ancombc2_lfc","ancombc2_q","aldex2_effect_oriented","aldex2_q","direction_in_first_group","concordant"]].sort_values(["comparison","level","ancombc2_lfc"], ascending=[True,True,False])
        t1.to_csv(os.path.join(a.out_dir,"table1_consensus_taxa.tsv"), sep="\t", index=False)
        cons[["taxon","level","comparison","ancombc2_lfc","aldex2_effect_raw","aldex2_effect_oriented","concordant"]].to_csv(os.path.join(a.out_dir,"fig6b_effect_sizes.tsv"), sep="\t", index=False)
        cons.to_csv(os.path.join(a.out_dir,"additional_file10_effect_sizes.tsv"), sep="\t", index=False)
    tot=len(cons); conc=int(cons.concordant.sum()) if tot else 0
    print(f"consensus taxa={tot}; concordant after orientation={conc}; discordant={tot-conc}")
    print(pd.DataFrame(summ)[["comparison","level","n_consensus","n_concordant","n_discordant"]].to_string(index=False))
if __name__=="__main__": main()
