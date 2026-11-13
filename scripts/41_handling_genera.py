# 41_handling_genera.py
# By Carter Clinton, Ph.D.
"""Proximity trend of handling-associated genera among burial samples.

Two-sided permutation Jonckheere-Terpstra tests (9,999 permutations, one random stream with
seed 42 in the order below) of genus relative abundance across cranium-associated, other
body-region and soil-matrix samples, for Staphylococcus (all ASVs and excluding the ASV
identical to the mock-community S. aureus sequence), Corynebacterium, Cutibacterium and
Streptococcus; Holm adjustment across the three skin genera.
"""
import argparse, os
import numpy as np, pandas as pd

MOCK_STAPH = "65d43491988bfe557da4d86a5ba25dae"
ORDER = {"Cranium": 0, "Other Body": 1, "Soil Matrix": 2}

def jt(x, g):
    s = 0.0; gs = [x[g == k] for k in range(3)]
    for i in range(3):
        for j in range(i + 1, 3):
            a = gs[i][:, None]; b = gs[j][None, :]; s += (a > b).sum() + 0.5 * (a == b).sum()
    return s

def perm_p(x, g, rng, n=9999):
    obs = jt(x, g); null = np.array([jt(x, rng.permutation(g)) for _ in range(n)]); mu = null.mean()
    return obs, (np.sum(np.abs(null - mu) >= abs(obs - mu)) + 1) / (n + 1)

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--asv-table", default="results/upstream/table_export/feature-table.tsv")
ap.add_argument("--taxonomy", default="results/upstream/taxonomy_export/taxonomy.tsv")
ap.add_argument("--samples", default="results/gradient_r500/gradient_sample_table.tsv")
ap.add_argument("--out", default="results/revision/handling_genera_trend.tsv")
a = ap.parse_args()
t = pd.read_csv(a.asv_table, sep="\t", skiprows=1, index_col=0)
tx = pd.read_csv(a.taxonomy, sep="\t", index_col=0)["Taxon"]
st = pd.read_csv(a.samples, sep="\t", dtype={"sample_id": str})
bur = st[st.cohort == "burial"].sample_id.tolist(); ctl = ["CHP", "CMP", "CP", "DP", "RP", "WMP"]
dep = t[bur].sum(); g = st.set_index("sample_id").loc[bur, "level"].map(ORDER).values
rng = np.random.default_rng(42)
staph = [x for x in t.index if "Staphylococcus" in str(tx.get(x, ""))]
sets = [("Staphylococcus (all ASVs)", staph), ("Staphylococcus (excluding mock-identical ASV)", [x for x in staph if x != MOCK_STAPH])]
for gname in ["Corynebacterium", "Cutibacterium", "Streptococcus"]:
    sets.append((gname, [x for x in t.index if gname in str(tx.get(x, ""))]))
rows = []
for lab, ids in sets:
    x = (t.loc[ids, bur].sum() / dep).values if ids else np.zeros(len(bur))
    obs, p = perm_p(x, g, rng)
    rows.append(dict(genus=lab, n_asvs=len(ids), burial_reads=int(t.loc[ids, bur].values.sum()) if ids else 0,
                     control_reads=int(t.loc[ids, ctl].values.sum()) if ids else 0,
                     detected_cranium=int((x[g == 0] > 0).sum()), detected_other=int((x[g == 1] > 0).sum()),
                     detected_matrix=int((x[g == 2] > 0).sum()), JT=obs, p_perm_two_sided=p))
R = pd.DataFrame(rows)
skin = R[R.genus.isin(["Staphylococcus (all ASVs)", "Corynebacterium", "Cutibacterium"])].sort_values("p_perm_two_sided")
m = len(skin); run = 0; adj = {}
for i, (k, p) in enumerate(zip(skin.genus, skin.p_perm_two_sided)):
    run = max(run, min(1, (m - i) * p)); adj[k] = run
R["p_holm_skin_genera"] = R.genus.map(adj)
os.makedirs(os.path.dirname(a.out), exist_ok=True); R.to_csv(a.out, sep="\t", index=False); print(R.to_string(index=False))
