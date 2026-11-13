# 39_blank_genus_table.py
# By Carter Clinton, Ph.D.
"""Genus-level table of NYABG samples plus the touchdown-PCR water control (input to 40).

Collapses the full-depth DADA2 ASV table to the lowest assigned rank (SILVA 138) for the
burial and control samples and the touchdown water blank (Water-neg-TD-plateD).
"""
import argparse, os
import pandas as pd

def lowest_rank(s):
    p = [x.strip() for x in str(s).split(";")]
    p = [x for x in p if x and not x.endswith("__")]
    return p[-1] if p else "Unassigned"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--asv-table", default="results/upstream/table_export/feature-table.tsv")
ap.add_argument("--taxonomy", default="results/upstream/taxonomy_export/taxonomy.tsv")
ap.add_argument("--samples", default="results/gradient_r500/gradient_sample_table.tsv")
ap.add_argument("--blank", default="Water-neg-TD-plateD")
ap.add_argument("--out", default="results/revision/genus_nyabg_plus_blank.tsv")
a = ap.parse_args()
t = pd.read_csv(a.asv_table, sep="\t", skiprows=1, index_col=0)
tx = pd.read_csv(a.taxonomy, sep="\t", index_col=0)["Taxon"]
st = pd.read_csv(a.samples, sep="\t", dtype={"sample_id": str})
cols = [c for c in st.sample_id if c in t.columns] + [a.blank]
G = t[cols].groupby(t.index.map(lambda x: lowest_rank(tx.get(x, "")))).sum()
G = G[G.sum(1) > 0]
os.makedirs(os.path.dirname(a.out), exist_ok=True); G.to_csv(a.out, sep="\t"); print(G.shape)
