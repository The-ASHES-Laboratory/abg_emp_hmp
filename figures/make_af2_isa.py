import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); from figstyle_nyabg import *; setup()
"""Additional file 2. Indicator species analysis summary (full-depth run).

Inputs (inputs_manifest.md): R/sensitivity/isa_results.tsv (indicspecies::multipatt, func = "IndVal.g",
999 permutations), filtered to p.value < 0.05; run parameters parsed from job log R/../logs/abg30_net_*.log.
(a) number of significant indicator taxa per indicator group; (b) distribution of the multipatt statistic.
"""
import re, glob, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

R = "results"
STEM = "af2_isa"
ALPHA = 0.05
MULTI = "#8C8C8C"   # multi-cohort indicator groups
NAME = {"burial": "Burial", "control": "Control", "emp": "EMP", "hmp": "HMP"}
UNINF = {"", "uncultured", "unclassified", "unknown", "unidentified", "metagenome", "ambiguous_taxa"}
RANK_TAG = {"p": "p.", "c": "c.", "o": "o.", "f": "f.", "d": "d."}

def parse_tax(s):
    out = {}
    for part in str(s).split(";"):
        m = re.match(r"^([dpcofgs])__(.*)$", part.strip())
        if m: out[m.group(1)] = m.group(2).strip()
    return out
def taxon_label(s):
    t = parse_tax(s)
    if not t: return str(s)
    for r in ["g", "f", "o", "c", "p", "d"]:
        v = t.get(r, "")
        if v.lower() not in UNINF and not v.lower().startswith("uncultured"):
            return v if r == "g" else f"{v} ({RANK_TAG[r]})"
    return "Unclassified"
def gname(g): return " + ".join(NAME[x] for x in g.split("+"))
def gcol(g): return COL[g] if g in NAME else MULTI

isa = pd.read_csv(f"{R}/sensitivity/isa_results.tsv", sep="\t")
sig = isa[isa["p.value"] < ALPHA].copy()
sig["label"] = sig["taxon"].map(taxon_label)
counts = sig.indicator_group.value_counts()
order = list(counts.sort_values(ascending=False, kind="stable").index)
# run parameters from the job log (first ISA block = primary run)
logf = sorted(glob.glob(f"{R}/../logs/abg30_net_*.log"))[-1]
blk = open(logf).read().split("--- 05 ISA noHumanEMP  START ---")[0]
m = re.search(r"Using (\d+) samples: (\d+) burial, (\d+) control, (\d+) EMP, (\d+) HMP", blk)
run = dict(zip(["n_samples", "burial", "control", "emp", "hmp"], map(int, m.groups())))
run["taxa_nonzero"] = int(re.search(r"(\d+) taxa with non-zero counts", blk).group(1))
run["taxa_after_prev_filter"] = int(re.search(r"(\d+) taxa after 5% prevalence filter", blk).group(1))
# cross-check with isa_summary.tsv
summ = pd.read_csv(f"{R}/sensitivity/isa_summary.tsv", sep="\t")
assert dict(zip(summ.indicator_group, summ.n_taxa)) == counts.to_dict()
assert run["taxa_after_prev_filter"] == len(isa)

numbers = {"n_tested": int(len(isa)), "n_significant": int(len(sig)), "n_p_le_0.05": int((isa["p.value"] <= ALPHA).sum()),
           "min_p": float(isa["p.value"].min()), "counts_by_group": counts.to_dict(), "run": run,
           "stat_by_group": {g: {"n": int((sig.indicator_group == g).sum()),
                                 "median": round(float(sig.loc[sig.indicator_group == g, "stat"].median()), 3),
                                 "min": round(float(sig.loc[sig.indicator_group == g, "stat"].min()), 3),
                                 "max": round(float(sig.loc[sig.indicator_group == g, "stat"].max()), 3)} for g in order},
           "stat_overall": {"median": round(float(sig.stat.median()), 3), "min": round(float(sig.stat.min()), 3),
                            "max": round(float(sig.stat.max()), 3)}}
top = {}
for g in ["burial", "control", "emp", "hmp"]:
    s = sig[sig.indicator_group == g].sort_values("stat", ascending=False)
    top[g] = [(l, round(v, 3)) for l, v in zip(s.label.head(5), s.stat.head(5))]
numbers["top_by_group"] = top
print(json.dumps(numbers, indent=1))

W, H = 164, 92
fig = plt.figure(figsize=(mm(W), mm(H)))
axA = fig.add_axes([0.165, 0.11, 0.25, 0.80])
axB = fig.add_axes([0.60, 0.11, 0.34, 0.80])
y = np.arange(len(order))
vals = [counts[g] for g in order]
axA.barh(y, vals, height=0.7, color=[gcol(g) for g in order], edgecolor="none")
for yi, v in zip(y, vals):
    axA.text(v + max(vals) * 0.02, yi, str(v), va="center", ha="left", fontsize=6.5)
axA.set_yticks(y); axA.set_yticklabels([gname(g) for g in order]); axA.invert_yaxis()
axA.set_xlim(0, max(vals) * 1.15); axA.set_ylim(len(order) - 0.5, -0.5)
axA.set_xlabel(f"Indicator taxa (p < {ALPHA:g})")
axA.set_title(f"{len(sig)} of {len(isa)} tested taxa significant", fontsize=7, pad=3)
letter(axA, "a", dx=-0.62, dy=1.01)

rng = np.random.default_rng(42)
for yi, g in zip(y, order):
    v = sig.loc[sig.indicator_group == g, "stat"].values
    if len(v) >= 5:
        axB.boxplot(v, positions=[yi], vert=False, widths=0.55, showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor="none", edgecolor="#333333", linewidth=0.6),
                    medianprops=dict(color="#111111", linewidth=1.0), whiskerprops=dict(linewidth=0.6, color="#333333"),
                    capprops=dict(linewidth=0.6, color="#333333"), zorder=2)
    axB.scatter(v, yi + rng.uniform(-0.22, 0.22, len(v)), s=7, color=gcol(g), alpha=0.75, edgecolors="none", zorder=3)
    axB.text(1.005, yi, f"n = {len(v)}", transform=axB.get_yaxis_transform(), va="center", ha="left", fontsize=6)
axB.set_yticks(y); axB.set_yticklabels([gname(g) for g in order]); axB.set_ylim(len(order) - 0.5, -0.5)
axB.axvline(0.5, color="#999999", lw=0.6, ls="--", zorder=1)
axB.set_xlim(0.25, 1.0); axB.set_xlabel("Indicator value statistic (multipatt, IndVal.g)")
axB.set_title("Statistic distribution by indicator group", fontsize=7, pad=3)
# strongest indicators per single-cohort group are reported in the legend text (numbers["top_by_group"])
letter(axB, "b", dx=-0.36, dy=1.01)
h = [Patch(color=COL[k], label=NAME[k]) for k in ["burial", "control", "emp", "hmp"]] + [Patch(color=MULTI, label="Multi-cohort")]
axB.legend(handles=h, loc="lower right", fontsize=6, ncol=1, handlelength=1.0, borderaxespad=0.3, frameon=True,
           facecolor="white", edgecolor="none", framealpha=0.9)
rep = save(fig, STEM); print("overlap report:", rep)
numbers["overlap_report"] = rep
json.dump(numbers, open(f"figures/out/{STEM}_numbers.json", "w"), indent=1, default=str)
