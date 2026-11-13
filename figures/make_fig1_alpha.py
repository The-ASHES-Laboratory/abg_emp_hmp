import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); from figstyle_nyabg import *; setup()
"""Fig. 1 (stem fig1_alpha): alpha diversity by cohort, full-depth genus-level re-analysis.
(a) Shannon, (b) Gini-Simpson, (c) coverage-standardized genus richness at 95% sample coverage.
Inputs (absolute paths, see inputs_manifest.md):
  R/alpha_diversity/alpha_values.tsv, alpha_stats.tsv (Shannon/Simpson; Mann-Whitney U, BH q per metric)
  R/alpha_diversity/coverage_richness_per_sample.tsv, coverage_richness_tests.tsv, coverage_richness_summary.tsv,
  sample_coverage_per_sample.tsv (target c0.95; interpolation only, samples with observed coverage < 0.95 excluded)
Every number drawn on the figure is computed here and written to out/fig1_alpha_numbers.json.
"""
import json
import pandas as pd
from scipy.stats import gaussian_kde

R = "results"
OUT = "figures/out"
COH = ["burial", "control", "emp", "hmp"]
TARGET = "c0.95"

alpha = pd.read_csv(f"{R}/alpha_diversity/alpha_values.tsv", sep="\t")
astats = pd.read_csv(f"{R}/alpha_diversity/alpha_stats.tsv", sep="\t")
cr = pd.read_csv(f"{R}/alpha_diversity/coverage_richness_per_sample.tsv", sep="\t")
ctests = pd.read_csv(f"{R}/alpha_diversity/coverage_richness_tests.tsv", sep="\t")
csum = pd.read_csv(f"{R}/alpha_diversity/coverage_richness_summary.tsv", sep="\t")
scov = pd.read_csv(f"{R}/alpha_diversity/sample_coverage_per_sample.tsv", sep="\t")

# ---------------- audit of inputs ----------------
N = {}
n_alpha = alpha.cohort.value_counts().reindex(COH).astype(int).to_dict()
c95 = cr[cr.target == TARGET].copy()
c95_inc = c95.dropna(subset=["richness_at_coverage"])
n_c95_total = c95.cohort.value_counts().reindex(COH).astype(int).to_dict()
n_c95 = c95_inc.cohort.value_counts().reindex(COH).astype(int).to_dict()
# cross-check against the pipeline summary
s95 = csum[csum.target == TARGET].set_index("cohort")
assert all(int(s95.loc[c, "n_included"]) == n_c95[c] for c in COH), "per-sample vs summary mismatch"
assert all(int(s95.loc[c, "n_total"]) == n_c95_total[c] for c in COH)
excl = c95[c95.richness_at_coverage.isna()].sample_id
excl_cov = scov.set_index("sample_id").loc[excl, ["cohort", "depth", "coverage"]]
assert (excl_cov.coverage < 0.95).all(), "an excluded sample has coverage >= 0.95"
# samples whose richness at c0.95 is an interpolation below full depth vs at full depth (m == n)
m_eq_n = c95_inc.merge(scov[["sample_id", "depth"]], on="sample_id")
chao_eq_obs_burial = bool(np.allclose(alpha.loc[alpha.cohort == "burial", "chao1"], alpha.loc[alpha.cohort == "burial", "observed"]))

def qfmt(q):
    if q < 0.001:
        e = int(np.floor(np.log10(q))); m = q / 10**e
        return f"q = {m:.1f}×10$^{{{e}}}$"
    if q < 0.01: return f"q = {q:.4f}".rstrip("0")
    return f"q = {q:.2f}" if q >= 0.1 else f"q = {q:.3f}"

def pairs_alpha(metric):
    d = astats[(astats.metric == metric) & (astats.test == "mann_whitney_u")]
    out = []
    for _, r in d.iterrows():
        a, b = r.comparison.split("_vs_"); out.append((a, b, float(r.q_value), float(r.effect_size)))
    return out

def pairs_cov():
    d = ctests[(ctests.target == TARGET) & (ctests.comparison != "all")]
    return [(r.comparison.split("_vs_")[0], r.comparison.split("_vs_")[1], float(r.p_BH), np.nan) for _, r in d.iterrows()]

panels = [
    ("a", "Shannon", "Shannon index (H′)", {c: alpha.loc[alpha.cohort == c, "shannon"].values for c in COH}, pairs_alpha("shannon"), n_alpha),
    ("b", "Simpson", "Simpson index (1 − Σp$^2$)", {c: alpha.loc[alpha.cohort == c, "simpson"].values for c in COH}, pairs_alpha("simpson"), n_alpha),
    ("c", "Coverage-standardized richness", "Genus richness at $\\hat{C}$ = 0.95", {c: c95_inc.loc[c95_inc.cohort == c, "richness_at_coverage"].values for c in COH}, pairs_cov(), n_c95),
]
kw_alpha = astats[(astats.test == "kruskal_wallis")].set_index("metric")
kw_cov = ctests[(ctests.target == TARGET) & (ctests.comparison == "all")].iloc[0]

# ---------------- plotting ----------------
fig, axes = plt.subplots(1, 3, figsize=(mm(170), mm(80)), gridspec_kw=dict(wspace=0.34))
rng = np.random.default_rng(7)
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.19, top=0.86)
numbers = {"n_alpha": n_alpha, "n_c95_total": n_c95_total, "n_c95_included": n_c95,
           "c95_excluded": [{"sample_id": s, "cohort": r.cohort, "depth": int(r.depth), "observed_coverage": round(float(r.coverage), 4)} for s, r in excl_cov.iterrows()],
           "chao1_equals_observed_all_burial": chao_eq_obs_burial, "panels": {}}
for ax, (lt, title, ylab, data, pairs, nmap) in zip(axes, panels):
    pos = np.arange(len(COH))
    ymin = min(v.min() for v in data.values()); ymax = max(v.max() for v in data.values())
    for i, c in enumerate(COH):
        v = data[c]
        # violin (KDE truncated to the data range)
        if c != "control" and len(v) > 2 and np.ptp(v) > 0:
            kde = gaussian_kde(v); yy = np.linspace(v.min(), v.max(), 200); dd = kde(yy); dd = dd / dd.max() * 0.38
            ax.fill_betweenx(yy, i - dd, i + dd, color=COL[c], alpha=0.22, lw=0, zorder=1)
            ax.plot(np.r_[i - dd, (i + dd)[::-1]], np.r_[yy, yy[::-1]], color=COL[c], lw=0.5, zorder=1.1)
        # box (IQR, whiskers 1.5 IQR, no fliers)
        ax.boxplot(v, positions=[i], widths=0.15, showfliers=False, patch_artist=True, zorder=5,
                   boxprops=dict(facecolor=COL[c], edgecolor="black", lw=0.5, alpha=0.85),
                   medianprops=dict(color="black", lw=1.0), whiskerprops=dict(color="black", lw=0.5),
                   capprops=dict(color="black", lw=0.5))
        if c in ("burial", "control"):
            jit = rng.uniform(-0.30, 0.30, len(v)) + i
            ax.scatter(jit, v, s=5 if c == "burial" else 9, marker="o" if c == "burial" else "D", facecolor=COL[c],
                       edgecolor="black", lw=0.3, alpha=0.75, zorder=2)
    ax.set_xticks(pos); ax.set_xticklabels([f"{LABEL[c]}\n{nmap[c]:,}" for c in COH], fontsize=6)
    ax.set_xlabel("Cohort (n samples)", fontsize=6.5, labelpad=2)
    ax.set_xlim(-0.6, len(COH) - 0.4); ax.set_ylabel(ylab); ax.set_title(title, pad=4)
    # brackets: shortest spans first; pairs whose spans do not overlap share a level
    order = sorted(pairs, key=lambda p: (COH.index(p[1]) - COH.index(p[0]), COH.index(p[0])))
    levels = []
    for pr in order:
        x1, x2 = COH.index(pr[0]), COH.index(pr[1])
        for L in levels:
            if all(x2 < COH.index(o[0]) or x1 > COH.index(o[1]) for o in L): L.append(pr); break
        else: levels.append([pr])
    span = ymax - ymin; step = 0.10 * span; y0 = ymax + 0.05 * span
    for k, L in enumerate(levels):
        for (a, b, q, d) in L:
            y = y0 + k * step; x1, x2 = COH.index(a), COH.index(b)
            ax.plot([x1 + 0.04, x1 + 0.04, x2 - 0.04, x2 - 0.04], [y - 0.02 * span, y, y, y - 0.02 * span], color="black", lw=0.5)
            ax.text((x1 + x2) / 2, y + 0.005 * span, qfmt(q), ha="center", va="bottom", fontsize=5.3,
                    color="black" if q < 0.05 else "0.45")
    order = levels
    ax.set_ylim(ymin - 0.04 * span, y0 + len(order) * step + 0.03 * span)
    top = ymax  # keep tick labels within data range
    ax.set_yticks([t for t in ax.get_yticks() if ymin - 0.04 * span - 1e-9 <= t <= top + 0.15 * span])
    ax.spines["left"].set_bounds(ax.get_yticks()[0], ax.get_yticks()[-1])
    letter(ax, lt, dx=-0.27)
    numbers["panels"][lt] = {
        "metric": title,
        "median": {c: round(float(np.median(data[c])), 4) for c in COH},
        "mean": {c: round(float(np.mean(data[c])), 4) for c in COH},
        "pairwise": [{"pair": f"{a}_vs_{b}", "q_BH": float(q), "cliffs_delta": (None if np.isnan(d) else round(d, 3))} for a, b, q, d in pairs],
    }
numbers["panels"]["a"]["kruskal_wallis"] = {"H": float(kw_alpha.loc["shannon", "stat"]), "p": float(kw_alpha.loc["shannon", "p_value"])}
numbers["panels"]["b"]["kruskal_wallis"] = {"H": float(kw_alpha.loc["simpson", "stat"]), "p": float(kw_alpha.loc["simpson", "p_value"])}
numbers["panels"]["c"]["kruskal_wallis"] = {"H": float(kw_cov.statistic), "p": float(kw_cov.p)}
numbers["alpha_stats_pairs_tested"] = sorted(set(astats.loc[astats.test == "mann_whitney_u", "comparison"]))
numbers["c95_m_below_depth_fraction"] = {c: round(float((m_eq_n.loc[m_eq_n.cohort == c, "m_reads"] < m_eq_n.loc[m_eq_n.cohort == c, "depth"]).mean()), 3) for c in COH}

rep = save(fig, "fig1_alpha")
with open(f"{OUT}/fig1_alpha_numbers.json", "w") as f: json.dump(numbers, f, indent=1, default=str)
print(json.dumps(numbers, indent=1, default=str))
print("OVERLAPS:", rep["overlaps"]); print("OUTSIDE:", rep["outside"])
