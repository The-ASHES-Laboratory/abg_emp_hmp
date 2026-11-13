import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); from figstyle_nyabg import *; setup()
"""Fig. 2 (stem fig2_ordination): beta diversity, full-depth genus-level re-analysis.
(a) Bray-Curtis PCoA, (b) Aitchison PCoA, (c) pairwise PERMANOVA R2 for both distances, open marker = PERMDISP p < 0.05.
Inputs (absolute paths, see inputs_manifest.md):
  R/ordination/pcoa_coordinates.tsv            Bray-Curtis PCoA, written by 03_beta_ordination from replicate 1 (seed 42)
  R/ordination/aitchison_pcoa_coordinates.tsv  Aitchison PCoA, written by 04_aitchison_ordination from the median replicate
  R/ordination/table2_permanova_permdisp.tsv   pairwise PERMANOVA R2 + PERMDISP, means over five replicates
  R/ordination/permdisp_bc_replicates.tsv      per-replicate global R2 (to identify the Bray-Curtis median replicate)
  R/metadata_cohort_split.tsv                  used to re-derive the stratified subsamples and identify the replicate in each file
  logs/abg10_ord_1813261.log                   Aitchison PCoA variance explained (median replicate)
Every number drawn or stated is computed here and written to out/fig2_ordination_numbers.json.
"""
import json, re
import pandas as pd
from scipy.stats import chi2
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

R = "results"
LOG = sorted(__import__('glob').glob('logs/abg10_ord_*'))[-1]  # Slurm log of slurm/10_ordination.sbatch
OUT = "figures/out"
COH = ["emp", "hmp", "burial", "control"]          # drawing order (references underneath)

# ---------------- inputs ----------------
bc_all = pd.read_csv(f"{R}/ordination/pcoa_coordinates.tsv", sep="\t", index_col=0)
bc = bc_all[["PC1", "PC2", "cohort"]].copy()
pcs = [c for c in bc_all.columns if c.startswith("PC")]
ait = pd.read_csv(f"{R}/ordination/aitchison_pcoa_coordinates.tsv", sep="\t", index_col=0)
t2 = pd.read_csv(f"{R}/ordination/table2_permanova_permdisp.tsv", sep="\t")
bcrep = pd.read_csv(f"{R}/ordination/permdisp_bc_replicates.tsv", sep="\t")
meta = pd.read_csv(f"{R}/metadata_cohort_split.tsv", sep="\t", index_col=0)
alpha_ids = set(pd.read_csv(f"{R}/alpha_diversity/alpha_values.tsv", sep="\t", usecols=["sample_id"]).sample_id)  # = genus-table samples
meta = meta[meta.index.isin(alpha_ids)]
log = open(LOG).read()

# ---------------- identify replicate in each coordinate file ----------------
def strat(md, cohort, n_total, rng):
    cm = md[md.cohort == cohort]; ss = cm["cohort_subsite"].value_counts()
    n = (ss / ss.sum() * n_total).round().astype(int)
    while n.sum() > n_total: n[n.idxmax()] -= 1
    while n.sum() < n_total: n[n.idxmin()] += 1
    out = []
    for s, k in n.items():
        pool = cm[cm["cohort_subsite"] == s].index.tolist(); out += rng.choice(pool, size=min(k, len(pool)), replace=False).tolist()
    return out
def subsample(seed, n_sub=500):
    rng = np.random.default_rng(seed); e = strat(meta, "emp", n_sub, rng); h = strat(meta, "hmp", n_sub, rng)
    return set(e + h + meta.index[meta.cohort.isin(["burial", "control"])].tolist())
seeds = list(range(42, 47))
rep_bc = [i + 1 for i, s in enumerate(seeds) if subsample(s) == set(bc.index)]
rep_ait = [i + 1 for i, s in enumerate(seeds) if subsample(s) == set(ait.index)]
assert len(rep_bc) == 1 and len(rep_ait) == 1
rep_bc, rep_ait = rep_bc[0], rep_ait[0]
g_bc = bcrep[bcrep.comparison == "global"].sort_values("replicate")
bc_median_rep = int(g_bc.replicate.values[np.argsort(g_bc.R2.values)[len(g_bc) // 2]])
m = re.search(r"04 Aitchison ordination/PERMANOVA  START.*?Median replicate: (\d+) \(R2=([\d.]+)\).*?PCoA variance explained \(median replicate\):\s*\n\s*PC1: ([\d.]+)%\s*\n\s*PC2: ([\d.]+)%", log, re.S)
ait_median_rep_log, ait_pc1, ait_pc2 = int(m.group(1)), float(m.group(3)), float(m.group(4))
assert ait_median_rep_log == rep_ait
bc_block = log[log.index("03 Bray-Curtis ordination/PERMANOVA  START"):log.index("03 Bray-Curtis ordination/PERMANOVA ok")]
bc_var_in_log = "variance explained" in bc_block.lower()
# diagnostic only (not drawn): share of the sum of squared coordinates; equals skbio proportion_explained when
# negative eigenvalues are zeroed before the denominator is taken (behaviour checked on skbio 0.7.3, not on the Hazel build)
ss = (bc_all[pcs] ** 2).sum(); bc_ss_share = (ss / ss.sum()).values[:2] * 100

n_bc = bc.cohort.value_counts().to_dict(); n_ait = ait.cohort.value_counts().to_dict()
assert n_bc == n_ait
n_nyabg = n_bc["burial"] + n_bc["control"]; n_tot = len(bc)

# ---------------- PERMANOVA table ----------------
PAIRS = ["burial_vs_control", "burial_vs_emp", "burial_vs_hmp", "control_vs_emp", "control_vs_hmp", "emp_vs_hmp"]
t2p = t2[t2.comparison.isin(PAIRS)].copy()
t2p["disp_differs"] = t2p.p_permdisp_mean < 0.05
glob = t2[t2.comparison == "global"].set_index("metric")
assert t2p.groupby("metric").size().eq(6).all()

# ---------------- plotting ----------------
fig = plt.figure(figsize=(mm(170), mm(150)))
gs = fig.add_gridspec(1, 2, wspace=0.26, left=0.075, right=0.985, top=0.905, bottom=0.425)
gsc = fig.add_gridspec(1, 1, left=0.135, right=0.985, top=0.295, bottom=0.095)
axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gsc[0, 0])
K = np.sqrt(chi2.ppf(0.95, 2))
STY = {"emp": dict(s=3, alpha=0.30, marker="o", lw=0), "hmp": dict(s=3, alpha=0.30, marker="o", lw=0),
       "burial": dict(s=9, alpha=0.9, marker="o", lw=0.3), "control": dict(s=16, alpha=0.95, marker="D", lw=0.4)}
def ellipse(ax, xy, col):
    mu = xy.mean(0); cov = np.cov(xy.T); w, v = np.linalg.eigh(cov); o = w.argsort()[::-1]; w, v = w[o], v[:, o]
    ang = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
    ax.add_patch(Ellipse(mu, 2 * K * np.sqrt(w[0]), 2 * K * np.sqrt(w[1]), angle=ang, fill=False, ec=col, lw=0.8, ls=(0, (3, 1.5)), zorder=6))
cent = {}
for ax, df, xl, yl, ttl, key in [(axa, bc, "PCo1", "PCo2", "Bray-Curtis", "bc"),
                                (axb, ait, f"PCo1 ({ait_pc1:.1f}%)", f"PCo2 ({ait_pc2:.1f}%)", "Aitchison (CLR)", "ait")]:
    for c in COH:
        d = df[df.cohort == c]
        ax.scatter(d.PC1, d.PC2, c=COL[c], edgecolors="black" if c in ("burial", "control") else "none", zorder=3 if c in ("emp", "hmp") else 4, **STY[c])
    for c in COH:
        xy = df.loc[df.cohort == c, ["PC1", "PC2"]].values
        if c != "control": ellipse(ax, xy, COL[c])
        mu = xy.mean(0); cent[f"{key}_{c}"] = [round(float(mu[0]), 4), round(float(mu[1]), 4)]
        ax.scatter(*mu, s=46 if c != "control" else 40, marker="o" if c != "control" else "D", c=COL[c], edgecolors="black", linewidths=1.0, zorder=8)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl, pad=3)
    ax.axhline(0, color="0.85", lw=0.4, zorder=0); ax.axvline(0, color="0.85", lw=0.4, zorder=0)
    ax.margins(0.04)
letter(axa, "a", dx=-0.16, dy=1.02); letter(axb, "b", dx=-0.16, dy=1.02)

# shared legend for a and b
h = [Line2D([], [], ls="", marker=STY[c]["marker"], ms=4.2, mfc=COL[c], mec="black" if c in ("burial", "control") else COL[c], mew=0.4,
            label=f"{LABEL[c]} (n = {n_bc[c]:,})") for c in ["burial", "control", "emp", "hmp"]]
h += [Line2D([], [], ls="", marker="o", ms=6, mfc="white", mec="black", mew=1.0, label="Cohort centroid"),
      Line2D([], [], color="0.3", lw=0.8, ls=(0, (3, 1.5)), label="95% normal data ellipse")]
fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.53, 0.995), ncol=6, fontsize=6.3, handletextpad=0.3, columnspacing=1.1)

# panel c: dot plot
lab = {"burial_vs_control": "Burial vs control", "burial_vs_emp": "Burial vs EMP", "burial_vs_hmp": "Burial vs HMP",
       "control_vs_emp": "Control vs EMP", "control_vs_hmp": "Control vs HMP", "emp_vs_hmp": "EMP vs HMP"}
MK = {"Bray-Curtis": ("o", +0.17), "Aitchison": ("s", -0.17)}
ypos = {p: len(PAIRS) - 1 - i for i, p in enumerate(PAIRS)}
axc.axhspan(ypos["burial_vs_control"] - 0.45, ypos["burial_vs_control"] + 0.45, color=COL["control"], alpha=0.08, lw=0, zorder=0)
for _, r in t2p.iterrows():
    mk, off = MK[r.metric]; y = ypos[r.comparison] + off
    axc.errorbar(r.R2_mean, y, xerr=r.R2_sd, fmt="none", ecolor="0.2", elinewidth=0.6, capsize=0, zorder=3)
    axc.scatter(r.R2_mean, y, marker=mk, s=20, facecolor="white" if r.disp_differs else "0.2", edgecolor="0.2", linewidths=0.9, zorder=4)
    if r.comparison == "burial_vs_control":
        axc.text(r.R2_mean + 0.0035, y, f"R$^2$ = {r.R2_mean:.3f}", va="center", ha="left", fontsize=6)
axc.set_yticks([ypos[p] for p in PAIRS]); axc.set_yticklabels([lab[p] for p in PAIRS])
axc.get_yticklabels()[0].set_fontweight("bold")
axc.set_ylim(-0.6, len(PAIRS) - 0.4)
xmax = float((t2p.R2_mean + t2p.R2_sd).max()); axc.set_xlim(0, np.ceil(xmax * 1.12 * 50) / 50)
axc.set_xlabel("Pairwise PERMANOVA R$^2$ (mean ± SD of five subsampling replicates)")
axc.set_title("Pairwise PERMANOVA effect size and dispersion", pad=3)
axc.grid(axis="x", color="0.9", lw=0.4); axc.set_axisbelow(True)
hc = [Line2D([], [], ls="", marker="o", ms=4.5, mfc="0.2", mec="0.2", label="Bray-Curtis"),
      Line2D([], [], ls="", marker="s", ms=4.5, mfc="0.2", mec="0.2", label="Aitchison")]
leg = axc.legend(handles=hc, loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=6.2, ncol=2, handletextpad=0.3,
                 columnspacing=1.0, borderaxespad=0.2)
axc.text(0.5, -0.36, "Filled marker: PERMDISP p ≥ 0.05.  Open marker: PERMDISP p < 0.05 (group dispersions differ).",
         transform=axc.transAxes, ha="center", va="top", fontsize=6)
letter(axc, "c", dx=-0.155, dy=1.03)

for ax in (axa, axb, axc):   # keep only tick labels inside the plotted range
    lo, hi = ax.get_xlim(); ax.set_xticks([t for t in ax.get_xticks() if lo <= t <= hi]); ax.set_xlim(lo, hi)
    if ax is not axc:
        lo, hi = ax.get_ylim(); ax.set_yticks([t for t in ax.get_yticks() if lo <= t <= hi]); ax.set_ylim(lo, hi)
rep = save(fig, "fig2_ordination")

numbers = {
    "n_per_cohort_in_coordinates": n_bc, "n_total": n_tot, "n_nyabg": n_nyabg, "n_replicates": len(seeds),
    "bc_coordinates_replicate": rep_bc, "bc_coordinates_seed": seeds[rep_bc - 1], "bc_median_replicate_by_global_R2": bc_median_rep,
    "bc_global_R2_by_replicate": dict(zip(g_bc.replicate.astype(int).astype(str), g_bc.R2.round(4))),
    "ait_coordinates_replicate": rep_ait, "ait_coordinates_seed": seeds[rep_ait - 1],
    "ait_variance_pc1_pct_from_log": ait_pc1, "ait_variance_pc2_pct_from_log": ait_pc2,
    "bc_variance_reported_in_log_or_outputs": bc_var_in_log,
    "bc_ss_share_pc1_pc2_pct_DIAGNOSTIC_NOT_PLOTTED": [round(float(x), 1) for x in bc_ss_share],
    "samples_shared_between_bc_and_ait_files": len(set(bc.index) & set(ait.index)),
    "centroids": cent,
    "ellipse": f"95% normal-theory data ellipse, chi2(2) quantile {chi2.ppf(0.95, 2):.4f}; drawn for burial, EMP, HMP; not for control (n = 6)",
    "permanova_pairwise": t2p[["metric", "comparison", "n", "R2_mean", "R2_sd", "p_permanova_mean", "permdisp_F_mean", "p_permdisp_mean", "p_permdisp_max", "disp_differs"]].round(5).to_dict("records"),
    "permanova_global": {k: {"R2_mean": round(float(glob.loc[k, "R2_mean"]), 4), "R2_sd": round(float(glob.loc[k, "R2_sd"]), 4), "p_permanova_mean": float(glob.loc[k, "p_permanova_mean"]), "p_permdisp_mean": float(glob.loc[k, "p_permdisp_mean"])} for k in glob.index},
    "max_pairwise_p_permanova_mean": float(t2p.p_permanova_mean.max()),
    "max_R2_sd_pairwise": float(t2p.R2_sd.max()),
}
with open(f"{OUT}/fig2_ordination_numbers.json", "w") as f: json.dump(numbers, f, indent=1, default=str)
print(json.dumps(numbers, indent=1, default=str))
print("OVERLAPS:", rep["overlaps"]); print("OUTSIDE:", rep["outside"])
