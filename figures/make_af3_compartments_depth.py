import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); from figstyle_nyabg import *; setup()
"""Additional file 3. (a, b) Full presence-pattern composition of the burial and control source networks;
(c) per-sample sequencing depth by cohort (merged genus-level table) with the 500-read reference-source depth.

Inputs (inputs_manifest.md): R/networks/{burial,control}/overall/source_network_summary.tsv and
source_network_taxa.tsv (presence = detected in >= 1 sample); R/sequencing_depth/depth_distribution.tsv.
"""
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

R = "results"
STEM = "af3_compartments_depth"
DEPTH_LINE = 500
NAME = {"burial": "Burial", "control": "Control", "emp": "EMP", "hmp": "HMP"}
plt.rcParams["hatch.linewidth"] = 0.8

def pattern_order(c):
    return [f"{c}+emp", f"{c}+emp+hmp", f"{c}+hmp", c, "emp", "emp+hmp", "hmp", "none"]
def pname(k, c):
    if k == "none": return "Not detected"
    parts = k.split("+"); s = " + ".join(NAME[p] for p in parts)
    return s + " only" if len(parts) == 1 else s
def pstyle(k, c):
    sink = c in k.split("+"); a = 1.0 if sink else 0.40
    base = k.replace(c, "").strip("+")
    if k == "none": return dict(color=COL["unknown"], alpha=1.0)
    if base == "": return dict(color=COL[c], alpha=a)
    if base == "emp": return dict(color=COL["emp"], alpha=a)
    if base == "hmp": return dict(color=COL["hmp"], alpha=a)
    return dict(color=COL["emp"], alpha=a, hatch="//////", edgecolor=COL["hmp"])

numbers = {}
comp = {}
for c in ["burial", "control"]:
    summ = pd.read_csv(f"{R}/networks/{c}/overall/source_network_summary.tsv", sep="\t")
    d = pd.read_csv(f"{R}/networks/{c}/overall/source_network_taxa.tsv", sep="\t")
    sd = dict(zip(summ.presence_pattern, summ.taxa_count.astype(int)))
    pres = {s: d[f"present_{s}"].astype(str).str.lower().eq("yes") for s in [c, "emp", "hmp"]}
    pat = pd.Series(["+".join([s for s in [c, "emp", "hmp"] if pres[s][i]]) or "none" for i in range(len(d))])
    vc = pat.value_counts().to_dict()
    ks = pattern_order(c)
    for k in ks: assert vc.get(k, 0) == sd.get(k, 0), (c, k, vc.get(k, 0), sd.get(k, 0))
    comp[c] = {k: int(vc.get(k, 0)) for k in ks}
    numbers[f"{c}_patterns"] = comp[c]; numbers[f"{c}_total_taxa"] = int(len(d))
    numbers[f"{c}_sink_present"] = int(pres[c].sum())
    print(c, comp[c], "total", len(d))

# taxa in pattern "none" (absent from sink, EMP and HMP): are they detected in the other NYABG cohort?
OTHER = {"burial": "control", "control": "burial"}
for c in ["burial", "control"]:
    o = OTHER[c]
    d_c = pd.read_csv(f"{R}/networks/{c}/overall/source_network_taxa.tsv", sep="\t")
    d_o = pd.read_csv(f"{R}/networks/{o}/overall/source_network_taxa.tsv", sep="\t").set_index("taxon_key")
    pres_c = pd.concat([d_c[f"present_{s}"].astype(str).str.lower().eq("yes") for s in [c, "emp", "hmp"]], axis=1)
    none_keys = d_c.loc[pres_c.sum(axis=1) == 0, "taxon_key"]
    in_other = int(d_o.loc[none_keys, f"present_{o}"].astype(str).str.lower().eq("yes").sum())
    numbers[f"{c}_none_detected_only_in_{o}"] = in_other
    numbers[f"{c}_none_zero_in_all_four"] = int(len(none_keys) - in_other)
    print(c, "pattern none:", len(none_keys), "| detected only in", o, ":", in_other,
          "| zero in all four cohorts:", len(none_keys) - in_other)

dep = pd.read_csv(f"{R}/sequencing_depth/depth_distribution.tsv", sep="\t")
cohorts = ["burial", "control", "emp", "hmp"]
dstat = {}
for c in cohorts:
    v = dep.loc[dep.cohort == c, "depth"].astype(float)
    dstat[c] = {"n": int(len(v)), "median": int(np.median(v)), "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75)),
                "min": int(v.min()), "max": int(v.max()), "n_below_500": int((v < DEPTH_LINE).sum())}
numbers["depth"] = dstat
print("depth:", dstat)

W, H = 164, 150
fig = plt.figure(figsize=(mm(W), mm(H)))
axA = fig.add_axes([0.165, 0.615, 0.285, 0.33]); axB = fig.add_axes([0.665, 0.615, 0.285, 0.33])
axC = fig.add_axes([0.115, 0.14, 0.86, 0.34])
for ax, c, L in [(axA, "burial", "a"), (axB, "control", "b")]:
    ks = pattern_order(c); vals = [comp[c][k] for k in ks]; tot = sum(vals)
    for i, (k, v) in enumerate(zip(ks, vals)):
        st = pstyle(k, c)
        ax.barh(i, v, height=0.7, color=st["color"], alpha=st["alpha"], hatch=st.get("hatch"),
                edgecolor=st.get("edgecolor", "none"), linewidth=0)
        ax.text(v + max(vals) * 0.02, i, f"{v:,} ({100 * v / tot:.1f}%)", va="center", ha="left", fontsize=6)
    ax.set_yticks(range(len(ks))); ax.set_yticklabels([pname(k, c) for k in ks], fontsize=6.3)
    ax.set_ylim(len(ks) - 0.5, -0.5); ax.set_xlim(0, max(vals) * 1.42)
    ax.axhline(3.5, color="#999999", lw=0.5, ls=":")
    ax.set_xlabel("Taxa (genus level)")
    ax.set_title(f"{NAME[c]} network ({tot:,} taxa)\nn = {dstat[c]['n']} {c} samples", fontsize=7, pad=3)
    letter(ax, L, dx=-0.56, dy=1.10)

# (c) depth: log10 axis drawn in log10 space with decade ticks labelled in reads
xs = np.arange(len(cohorts))
for x, c in zip(xs, cohorts):
    v = np.log10(dep.loc[dep.cohort == c, "depth"].astype(float).values)
    col = COL[c]
    if len(v) >= 20:
        kde = gaussian_kde(v); grid = np.linspace(v.min(), v.max(), 300); dens = kde(grid)
        dens = dens / dens.max() * 0.38
        axC.fill_betweenx(grid, x - dens, x + dens, color=col, alpha=0.30, linewidth=0)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    if len(v) >= 20:
        axC.add_patch(plt.Rectangle((x - 0.06, q1), 0.12, q3 - q1, facecolor="none" if len(v) <= 100 else "white", edgecolor="#222222", lw=0.7, zorder=6))
        axC.plot([x - 0.06, x + 0.06], [med, med], color="#111111", lw=1.1, zorder=7)
    if len(v) <= 100:
        rng = np.random.default_rng(1)
        axC.scatter(x + rng.uniform(-0.17, 0.17, len(v)), v, s=7, color=col, edgecolors="#333333", linewidths=0.25, zorder=5)
        if len(v) < 20:
            axC.plot([x - 0.18, x + 0.18], [med, med], color="#111111", lw=1.1, zorder=6)
axC.axhline(np.log10(DEPTH_LINE), color="#333333", lw=0.7, ls="--", zorder=2)
axC.text(len(cohorts) - 0.52, np.log10(DEPTH_LINE) + 0.05, f"{DEPTH_LINE} reads", ha="right", va="bottom", fontsize=6.3, color="#333333")
lo = int(np.floor(np.log10(dep.depth.min()))); hi = int(np.ceil(np.log10(dep.depth.max())))
axC.set_ylim(lo - 0.05, hi + 0.05)
axC.set_yticks(range(lo, hi + 1)); axC.set_yticklabels([f"{10 ** k:,}" for k in range(lo, hi + 1)])
axC.set_yticks([np.log10(m * 10 ** k) for k in range(lo, hi) for m in range(2, 10)], minor=True)
axC.tick_params(axis="y", which="minor", length=1.8, width=0.4)
axC.set_xticks(xs)
axC.set_xticklabels([f"{NAME[c]}\nn = {dstat[c]['n']:,}\nmedian {dstat[c]['median']:,}\n< {DEPTH_LINE} reads: {dstat[c]['n_below_500']:,}" for c in cohorts], fontsize=6.3)
axC.set_xlim(-0.6, len(cohorts) - 0.4)
axC.set_ylabel("Reads per sample (log scale)")
axC.set_title("Sequencing depth by cohort (merged genus-level table)", fontsize=7, pad=3)
letter(axC, "c", dx=-0.097, dy=1.02)
rep = save(fig, STEM); print("overlap report:", rep)
numbers["overlap_report"] = rep
json.dump(numbers, open(f"figures/out/{STEM}_numbers.json", "w"), indent=1, default=str)
