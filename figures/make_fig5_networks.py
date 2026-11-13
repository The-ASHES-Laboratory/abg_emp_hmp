import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); from figstyle_nyabg import *; setup()
"""Fig. 5. Bipartite source networks (burial, control) rebuilt from the full-depth network tables.

Inputs (inputs_manifest.md): R/networks/{burial,control}/overall/source_network_taxa.tsv,
source_network_edges.tsv, source_network_summary.tsv; R/metadata_cohort_split.tsv; job log
R/../logs/abg30_net_*.log (cohort sizes used by the network builder).

Display rule: taxa detected in the sink cohort (presence = detected in >= 1 sample) are grouped by
compartment (sink + EMP, sink + EMP + HMP, sink + HMP, sink only); within each compartment the
TOP_N taxa with the highest maximum IndVal across the three cohorts are drawn. Every presence edge is
drawn (minimum visible width), edge width scales with the per-source IndVal (specificity x prevalence).
"""
import re, glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

R = "results"
STEM = "fig5_networks"
TOP_N = 8
COHORTS = ["burial", "control"]
RANKS = {"d": "domain", "p": "phylum", "c": "class", "o": "order", "f": "family", "g": "genus"}
RANK_TAG = {"p": "p.", "c": "c.", "o": "o.", "f": "f.", "d": "d."}
UNINF = {"", "uncultured", "unclassified", "unknown", "unidentified", "metagenome", "uncultured_bacterium",
         "uncultured_soil_bacterium", "ambiguous_taxa"}
PHY_SYN = {"Bacteroidetes": "Bacteroidota", "Actinobacteria": "Actinobacteriota", "Fusobacteria": "Fusobacteriota",
           "Acidobacteria": "Acidobacteriota", "Chloroflexota": "Chloroflexi", "Firmicutes_A": "Firmicutes"}
# Phylum fill for taxon nodes (Tol muted; chosen to stay distinct from the cohort/source colours)
PHY_COL = {"Proteobacteria": "#332288", "Firmicutes": "#AA4499", "Actinobacteriota": "#DDCC77",
           "Bacteroidota": "#88CCEE", "Chloroflexi": "#882255", "Acidobacteriota": "#999933",
           "Fusobacteriota": "#CC6677"}
PHY_OTHER = "#DDDDDD"

def parse_tax(s):
    out = {}
    for part in str(s).split(";"):
        part = part.strip()
        m = re.match(r"^([dpcofgs])__(.*)$", part)
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

def phylum_of(s):
    p = parse_tax(s).get("p", "") or "Unknown"
    return PHY_SYN.get(p, p)

def load(c):
    d = pd.read_csv(f"{R}/networks/{c}/overall/source_network_taxa.tsv", sep="\t")
    e = pd.read_csv(f"{R}/networks/{c}/overall/source_network_edges.tsv", sep="\t")
    summ = pd.read_csv(f"{R}/networks/{c}/overall/source_network_summary.tsv", sep="\t")
    for s in ["emp", "hmp", c]:
        d["_" + s] = d[f"present_{s}"].astype(str).str.strip().str.lower().eq("yes")
    d["pat"] = d.apply(lambda r: "+".join([s for s in [c, "emp", "hmp"] if r["_" + s]]) or "none", axis=1)
    d["max_indval"] = d[[f"{s}_indval" for s in [c, "emp", "hmp"]]].max(axis=1)
    d["label"] = d["taxon"].map(taxon_label)
    d["phylum_h"] = d["taxon"].map(phylum_of)
    # cross-check presence-derived compartments against the summary file written by the network builder
    sdict = dict(zip(summ.presence_pattern, summ.taxa_count))
    vc = d.pat.value_counts().to_dict()
    for k, v in sdict.items():
        assert vc.get(k, 0) == v, (c, k, v, vc.get(k))
    # cross-check every presence flag has an edge
    ek = {s: set(e.loc[e.source == s, "taxon_key"]) for s in e.source.unique()}
    for s in ["emp", "hmp", c]:
        assert set(d.loc[d["_" + s], "taxon_key"]) == ek.get(s, set()), (c, s)
    return d, e

# ---- cohort sizes: metadata and network-builder log
md = pd.read_csv(f"{R}/metadata_cohort_split.tsv", sep="\t")
n_meta = md.cohort.value_counts().to_dict()
logf = sorted(glob.glob(f"{R}/../logs/abg30_net_*.log"))[-1]
logtxt = open(logf).read()
n_net = {}
for c in COHORTS:
    blk = logtxt.split(f"--- 06 network {c} overall  START ---")[1]
    m = re.search(r"Cohort counts: (\{[^}]*\})", blk)
    n_net[c] = eval(m.group(1))
print("metadata cohort sizes:", n_meta)
print("network-builder cohort sizes:", n_net)
gst = pd.read_csv(f"{R}/gradient_r500/gradient_sample_table.tsv", sep="\t")
n_burials = int(gst.loc[gst.cohort == "burial", "burial_id"].nunique())
control_sites = sorted(md.loc[md.cohort == "control", md.columns[0]].astype(str))
print("burial samples:", int((gst.cohort == "burial").sum()), "from burials:", n_burials, "| control sites:", control_sites)

numbers = {"n_samples_network": n_net, "top_n_per_compartment": TOP_N, "n_burials": n_burials,
           "control_sites": control_sites, "n_metadata": n_meta}
data = {}
for c in COHORTS:
    d, e = load(c)
    comps = [f"{c}+emp", f"{c}+emp+hmp", f"{c}+hmp", c]
    counts = {k: int((d.pat == k).sum()) for k in comps}
    linked = {s: int(d.loc[d["_" + c], "_" + s].sum()) for s in ["emp", "hmp", c]}
    sel = []
    for k in comps:
        sub = d[d.pat == k].sort_values(["max_indval", "label"], ascending=[False, True]).head(TOP_N)
        sel.append((k, sub))
    data[c] = dict(d=d, comps=comps, counts=counts, linked=linked, sel=sel)
    numbers[c] = {"compartments": counts, "sink_present_total": int(d["_" + c].sum()),
                  "sink_present_taxa_linked_to_source": linked,
                  "displayed_per_compartment": {k: len(s) for k, s in sel},
                  "displayed_taxa": {k: list(s.label) for k, s in sel},
                  "n_edges_drawn": int(sum(int(r["_emp"]) + int(r["_hmp"]) + int(r["_" + c]) for _, s in sel for _, r in s.iterrows()))}
    print(c, "compartments:", counts, "| sink-present:", numbers[c]["sink_present_total"], "| linked:", linked)
    for k, s in sel:
        print("   ", k, len(s), [(l, round(v, 3)) for l, v in zip(s.label, s.max_indval)])
numbers["hmp_edges_control_display"] = int(sum(s["_hmp"].sum() for _, s in data["control"]["sel"]))
print("control HMP edges drawn:", numbers["hmp_edges_control_display"])

# disambiguate duplicate labels within a panel by appending the next informative rank
for c in COHORTS:
    alls = pd.concat([s for _, s in data[c]["sel"]])
    dup = alls.label[alls.label.duplicated(keep=False)]
    for k, s in data[c]["sel"]:
        for i in s.index:
            if s.at[i, "label"] in set(dup):
                t = parse_tax(s.at[i, "taxon"]); bits = [t.get(r, "") for r in ["c", "o", "f"] if t.get(r, "") and t.get(r, "").lower() not in UNINF]
                s.at[i, "label"] = s.at[i, "label"] + (f", {bits[-1]}" if bits else "")

# phyla in display
disp_phy = pd.concat([s for c in COHORTS for _, s in data[c]["sel"]]).phylum_h.value_counts()
print("phyla among displayed taxa:", disp_phy.to_dict())
phy_shown = [p for p in PHY_COL if p in disp_phy.index]

NAME = {"burial": "Burial", "control": "Control"}
def comp_name(k, c):
    parts = k.split("+")
    return " + ".join(NAME[c] if p == c else p.upper() for p in parts) + (" only" if k == c else "")
def comp_style(k, c):
    if k == c: return dict(color=COL[c])
    if k == f"{c}+emp": return dict(color=COL["emp"])
    if k == f"{c}+hmp": return dict(color=COL["hmp"])
    return dict(color=COL["emp"], hatch="//////", edgecolor=COL["hmp"])

SRC_Y = {"emp": 0.90, "sink": 0.50, "hmp": 0.10}
def draw_network(ax, c):
    D = data[c]; rows = []
    for k, s in D["sel"]:
        rows.append(("hdr", k, None))
        if len(s) == 0: rows.append(("none", k, None))
        for _, r in s.iterrows(): rows.append(("tax", k, r))
    n = len(rows); ys = {i: 1 - i / (n - 1) for i in range(n)}
    x0, x1 = 0.0, 1.0
    srcs = {"emp": (COL["emp"], "EMP"), "sink": (COL[c], NAME[c]), "hmp": (COL["hmp"], "HMP")}
    segs, cols, lws = [], [], []
    for i, (kind, k, r) in enumerate(rows):
        if kind != "tax": continue
        for s, key in [("emp", "emp"), ("hmp", "hmp"), ("sink", c)]:
            if r["_" + key]:
                iv = float(r[f"{key}_indval"])
                segs.append([(x0, SRC_Y[s]), (x1, ys[i])]); cols.append(srcs[s][0]); lws.append(0.35 + 2.4 * iv)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, alpha=0.55, zorder=1, capstyle="round"))
    for i, (kind, k, r) in enumerate(rows):
        if kind == "hdr":
            ax.text(x1 - 0.015, ys[i], f"{comp_name(k, c)} ({D['counts'][k]} taxa)", ha="left", va="center",
                    fontsize=6, fontweight="bold", color="#333333")
        elif kind == "none":
            ax.text(x1 + 0.045, ys[i], "no taxa in this compartment", ha="left", va="center", fontsize=5.8,
                    color="#666666")
        else:
            ax.scatter(x1, ys[i], s=16, c=PHY_COL.get(r.phylum_h, PHY_OTHER), edgecolors="#333333", linewidths=0.35, zorder=4)
            ax.text(x1 + 0.045, ys[i], r.label, ha="left", va="center", fontsize=5.8, fontstyle="italic", color="#222222")
    lk = {"emp": D["linked"]["emp"], "sink": D["linked"][c], "hmp": D["linked"]["hmp"]}
    for s, (col, name) in srcs.items():
        ax.scatter(x0, SRC_Y[s], s=150, c=col, edgecolors="#222222", linewidths=0.6, zorder=5)
        ax.text(x0 - 0.06, SRC_Y[s] + 0.012, name, ha="right", va="bottom", fontsize=7.5, fontweight="bold", color=col)
        ax.text(x0 - 0.06, SRC_Y[s] - 0.012, f"{lk[s]} linked taxa", ha="right", va="top", fontsize=6, color="#333333")
    ax.set_xlim(-0.66, 2.24); ax.set_ylim(-0.03, 1.03); ax.axis("off")

def draw_bars(ax, c):
    D = data[c]; ks = D["comps"]; vals = [D["counts"][k] for k in ks]
    for i, (k, v) in enumerate(zip(ks, vals)):
        st = comp_style(k, c)
        ax.barh(i, v, height=0.68, color=st["color"], hatch=st.get("hatch"), edgecolor=st.get("edgecolor", "none"),
                linewidth=0 if "hatch" not in st else 0.0)
        ax.text(v + max(vals) * 0.015, i, str(v), va="center", ha="left", fontsize=6.5)
    ax.set_yticks(range(len(ks))); ax.set_yticklabels([comp_name(k, c) for k in ks], fontsize=6.5)
    ax.invert_yaxis(); ax.set_xlim(0, max(vals) * 1.18); ax.set_xlabel("Taxa (genus level)", fontsize=7)
    ax.set_title(f"{NAME[c]} network compartments (n = {n_net[c][c]} samples)", fontsize=7, pad=3)

plt.rcParams["hatch.linewidth"] = 0.8
W, H = 164, 176   # save() adds 0.1 in padding per side; final PNG <= 170 mm wide
fig = plt.figure(figsize=(mm(W), mm(H)))
axN = [fig.add_axes([0.005, 0.30, 0.49, 0.66]), fig.add_axes([0.505, 0.30, 0.49, 0.66])]
axB = [fig.add_axes([0.17, 0.135, 0.27, 0.105]), fig.add_axes([0.675, 0.135, 0.27, 0.105])]
for j, c in enumerate(COHORTS):
    draw_network(axN[j], c); draw_bars(axB[j], c)
    axN[j].text(0.5, 1.035, f"{NAME[c]} source network (n = {n_net[c][c]} {c} samples)", transform=axN[j].transAxes,
                ha="center", va="bottom", fontsize=8)
letter(axN[0], "a", dx=0.0, dy=1.03); letter(axN[1], "b", dx=0.0, dy=1.03)

# shared legend: sources (edge colour), IndVal width scale, phylum, hatched compartment
axL = fig.add_axes([0.0, 0.0, 1.0, 0.075]); axL.axis("off")
h1 = [Line2D([], [], color=COL[k], lw=2.0, alpha=0.8, label=LABEL[k]) for k in ["burial", "control", "emp", "hmp"]]
leg1 = axL.legend(handles=h1, title="Edge source", loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2,
                  fontsize=6.5, title_fontsize=7, handlelength=1.6, columnspacing=1.0, borderaxespad=0)
axL.add_artist(leg1)
h2 = [Line2D([], [], color="#555555", lw=0.35 + 2.4 * v, alpha=0.7, label=f"{v:.1f}") for v in [0.0, 0.2, 0.5, 0.9]]
leg2 = axL.legend(handles=h2, title="Edge IndVal", loc="upper left", bbox_to_anchor=(0.215, 1.0), ncol=2,
                  fontsize=6.5, title_fontsize=7, handlelength=1.6, columnspacing=1.0, borderaxespad=0)
axL.add_artist(leg2)
h3 = [Line2D([], [], marker="o", ls="", markerfacecolor=PHY_COL[p], markeredgecolor="#333333", markeredgewidth=0.35,
             markersize=4.5, label=p) for p in phy_shown]
h3.append(Line2D([], [], marker="o", ls="", markerfacecolor=PHY_OTHER, markeredgecolor="#333333", markeredgewidth=0.35,
                 markersize=4.5, label="Other phyla"))
leg3 = axL.legend(handles=h3, title="Taxon phylum", loc="upper left", bbox_to_anchor=(0.40, 1.0), ncol=4,
                  fontsize=6.5, title_fontsize=7, handletextpad=0.2, columnspacing=0.9, borderaxespad=0)
for lg in (leg1, leg2, leg3): lg._legend_box.align = "left"

rep = save(fig, STEM)
print("overlap report:", rep)
numbers["overlap_report"] = rep
import json
json.dump(numbers, open(f"figures/out/{STEM}_numbers.json", "w"), indent=1, default=str)
