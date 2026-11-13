import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); from figstyle_nyabg import *; setup()
"""Fig. 6. Differential abundance agreement between ANCOM-BC2 and ALDEx2 (full-depth re-analysis).
(a) Significant taxa per contrast and level, recomputed from the per-method output files
    (ANCOM-BC2 q < 0.05 on the cohort coefficient; ALDEx2 Wilcoxon wi.eBH < 0.05; 'both' = intersection).
(b) ANCOM-BC2 natural-log fold change vs ALDEx2 effect, both oriented to the first-named group,
    for all consensus taxa (da_consensus_oriented.tsv).
Inputs: paths from inputs_manifest.md (R = i4/results, L = results_fd).
"""
import os, re, json
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

R = 'results'
L = 'results/revision'
OUT = 'figures/out'
mpl.rcParams.update({'mathtext.fontset': 'custom', 'mathtext.it': 'Arial:italic', 'mathtext.rm': 'Arial'})

CONTRASTS = ['burial_vs_emp', 'burial_vs_hmp', 'burial_vs_control', 'control_vs_emp', 'control_vs_hmp']
LEVELS = ['L5', 'L6']
GNAME = {'burial': 'Burial', 'control': 'Control', 'emp': 'EMP', 'hmp': 'HMP'}
METH_COL = {'ancombc2': '#7F7F7F', 'aldex2': '#D0D0D0', 'both': '#1A1A1A'}
ALPHA = 0.05


def cname(cmp):
    a, b = cmp.split('_vs_'); return f'{GNAME[a]} vs {GNAME[b] if b != "control" else "control"}'


# ---------------------------------------------------------------- (a) counts from per-method files
rows = []
for cmp in CONTRASTS:
    g1, g2 = cmp.split('_vs_')
    for lev in LEVELS:
        an = pd.read_csv(f'{R}/diff_abun/ancombc2_split/ancombc2_{cmp}_{lev}.tsv', sep='\t')
        al = pd.read_csv(f'{R}/diff_abun/aldex2/aldex2_{cmp}_{lev}.tsv', sep='\t')
        qcol = f'q_cohort{g1}'          # ANCOM-BC2 coefficient for the first-named (non-reference) group
        assert qcol in an.columns, (cmp, lev, an.columns.tolist())
        sa = set(an.loc[an[qcol] < ALPHA, 'taxon']); sb = set(al.loc[al['wi.eBH'] < ALPHA, 'taxon'])
        rows.append(dict(comparison=cmp, level=lev, n_tested_ancombc2=len(an), n_tested_aldex2=len(al),
                         n_tested_both=len(set(an.taxon) & set(al.taxon)),
                         sig_ancombc2=len(sa), sig_aldex2=len(sb), sig_both=len(sa & sb)))
counts = pd.DataFrame(rows)

# ---------------------------------------------------------------- (b) oriented consensus
cons = pd.read_csv(f'{L}/../diff_abun/da_consensus_oriented.tsv', sep='\t')
summ = pd.read_csv(f'{L}/../diff_abun/da_consensus_oriented_summary.tsv', sep='\t')
# audit: recomputed intersections must equal the consensus file and its summary
chk = cons.groupby(['comparison', 'level']).size().rename('n_cons_file')
m = counts.set_index(['comparison', 'level']).join(chk).fillna({'n_cons_file': 0})
m = m.join(summ.set_index(['comparison', 'level'])['n_consensus'])
assert (m.sig_both == m.n_cons_file).all() and (m.sig_both == m.n_consensus).all(), m
# audit: concordance = sign agreement of the two oriented effects
assert (cons.concordant == (np.sign(cons.ancombc2_lfc) == np.sign(cons.aldex2_effect_oriented))).all()
counts.to_csv(f'{OUT}/fig6a_da_counts.tsv', sep='\t', index=False)

N_CONS = len(cons); N_DISC = int((~cons.concordant).sum()); N_CONC = N_CONS - N_DISC
print('Panel a counts:\n', counts.to_string(index=False))
print(f'Panel b: consensus taxa={N_CONS}, concordant={N_CONC}, discordant={N_DISC}')
print('per contrast/level:\n', cons.groupby(['comparison', 'level']).concordant.agg(['size', 'sum']).to_string())


def short_name(tax, lev):
    ranks = [r for r in tax.split(';')]
    names = [(i, r.split('__', 1)[1] if '__' in r else r) for i, r in enumerate(ranks)]
    names = [(i, n) for i, n in names if n and n.lower() not in ('', 'uncultured')]
    i, n = names[-1]
    target = 5 if lev == 'L6' else 4
    latin = re.fullmatch(r'[A-Z][a-z]+', n) is not None
    return rf'$\it{{{n}}}$' if latin else n   # deepest assigned rank (stated in legend)


def place_labels(ax, fig, items, pts_disp, fs=5.6, radii=(10, 14, 19, 25, 32, 42), avoid=()):
    """Greedy deterministic label placement. items: list of (x, y, text, prefer_dx_sign)."""
    r = fig.canvas.get_renderer(); placed = list(avoid)
    axbb = ax.get_window_extent(r)
    angs = np.deg2rad([45, -45, 135, -135, 0, 180, 90, -90, 20, -20, 160, -160, 70, -70, 110, -110])
    for (x, y, s, pref) in items:
        done = False
        for rad in radii:
            if isinstance(pref, (list, tuple)):   # explicit preferred angles (degrees) first
                order = list(np.deg2rad(pref)) + list(angs)
            else:
                order = sorted(angs, key=lambda a: (0 if np.sign(np.cos(a)) == pref else 1))
            for a in order:
                dx, dy = rad * np.cos(a), rad * np.sin(a)
                ha = 'left' if dx > 1 else ('right' if dx < -1 else 'center')
                va = 'bottom' if dy > 1 else ('top' if dy < -1 else 'center')
                t = ax.annotate(s, (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=fs, ha=ha, va=va,
                                annotation_clip=False)
                t.update_positions(r); bb = mpl.text.Text.get_window_extent(t, r).expanded(1.04, 1.10)   # text box only, excluding the leader line
                inside = bb.x0 > axbb.x0 + 2 and bb.x1 < axbb.x1 - 2 and bb.y0 > axbb.y0 + 2 and bb.y1 < axbb.y1 - 2
                hit_pt = np.any((pts_disp[:, 0] > bb.x0 - 3) & (pts_disp[:, 0] < bb.x1 + 3) &
                                (pts_disp[:, 1] > bb.y0 - 3) & (pts_disp[:, 1] < bb.y1 + 3))
                if inside and not hit_pt and not any(bb.overlaps(p) for p in placed):
                    placed.append(bb); done = True
                    # leader line drawn separately (point edge -> nearest edge of the text box), in display coords
                    p0 = ax.transData.transform((x, y)); tb = mpl.text.Text.get_window_extent(t, r)
                    q = np.array([np.clip(p0[0], tb.x0, tb.x1), np.clip(p0[1], tb.y0, tb.y1)])
                    v = q - p0; dist = np.hypot(*v)
                    if dist > 6:
                        p1 = p0 + v / dist * 4.0
                        inv = ax.transData.inverted(); a0, a1 = inv.transform(p1), inv.transform(q - v / dist * 1.0)
                        ax.add_line(Line2D([a0[0], a1[0]], [a0[1], a1[1]], lw=0.4, color='0.35', zorder=5))
                    break
                t.remove()
            if done: break
        if not done:
            print('WARNING: could not place label', s)
    return placed


# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(mm(170), mm(158)))
gs = fig.add_gridspec(2, 1, height_ratios=[0.78, 1.55], hspace=0.36, left=0.075, right=0.985, top=0.965, bottom=0.07)
axA = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1])

# (a)
w = 0.26; xs = []; xt = []; xtl = []
pos = 0.0
for ci, cmp in enumerate(CONTRASTS):
    for lev in LEVELS:
        r_ = counts[(counts.comparison == cmp) & (counts.level == lev)].iloc[0]
        for k, (key, col) in enumerate([('sig_ancombc2', 'ancombc2'), ('sig_aldex2', 'aldex2'), ('sig_both', 'both')]):
            xx = pos + (k - 1) * w; v = int(r_[key])
            axA.bar(xx, v, width=w, color=METH_COL[col], edgecolor='0.15', linewidth=0.4)
            axA.text(xx, v + 1.5, str(v), ha='center', va='bottom', fontsize=5.2)
        xt.append(pos); xtl.append('Family' if lev == 'L5' else 'Genus')
        pos += 1.0
    xs.append(pos - 1.5); pos += 0.45
axA.set_xticks(xt); axA.set_xticklabels(xtl, fontsize=6)
axA.tick_params(axis='x', length=0, pad=2)
for ci, cmp in enumerate(CONTRASTS):
    axA.text(xs[ci], -0.17, cname(cmp), transform=mpl.transforms.blended_transform_factory(axA.transData, axA.transAxes),
             ha='center', va='top', fontsize=7)
for ci in range(len(CONTRASTS) - 1):
    axA.axvline(xs[ci] + 1.225, color='0.85', lw=0.5, zorder=0)
ymax = counts[['sig_ancombc2', 'sig_aldex2']].values.max()
axA.set_ylim(0, ymax * 1.16); axA.set_xlim(-0.6, pos - 0.85)
axA.set_ylabel('Significant taxa (q < 0.05)')
axA.legend(handles=[Patch(facecolor=METH_COL['ancombc2'], edgecolor='0.15', lw=0.4, label='ANCOM-BC2'),
                    Patch(facecolor=METH_COL['aldex2'], edgecolor='0.15', lw=0.4, label='ALDEx2 (Wilcoxon, BH)'),
                    Patch(facecolor=METH_COL['both'], edgecolor='0.15', lw=0.4, label='Both methods (consensus)')],
           loc='upper right', bbox_to_anchor=(1.0, 1.02), ncol=3, fontsize=6.3, handlelength=1.0, columnspacing=1.2)
letter(axA, 'a', dx=-0.065, dy=1.0)

# (b)
CCOL = {'burial_vs_emp': COL['emp'], 'burial_vs_hmp': COL['hmp'], 'burial_vs_control': COL['control'], 'control_vs_hmp': COL['hmp']}
MK = {'L5': 's', 'L6': 'o'}
axB.axhline(0, color='0.4', lw=0.6, zorder=1); axB.axvline(0, color='0.4', lw=0.6, zorder=1)
for cmp in [c for c in CONTRASTS if c in set(cons.comparison)]:
    for lev in LEVELS:
        d = cons[(cons.comparison == cmp) & (cons.level == lev)]
        if d.empty: continue
        filled = cmp.startswith('burial')
        axB.scatter(d.ancombc2_lfc, d.aldex2_effect_oriented, s=16, marker=MK[lev],
                    facecolor=CCOL[cmp] if filled else 'white', edgecolor=CCOL[cmp] if filled else CCOL[cmp],
                    linewidth=0.4 if filled else 1.0, alpha=0.9, zorder=3)
disc = cons[~cons.concordant]
axB.scatter(disc.ancombc2_lfc, disc.aldex2_effect_oriented, s=58, marker='o', facecolor='none', edgecolor='k', linewidth=0.9, zorder=4)
xl = np.abs(cons.ancombc2_lfc).max() * 1.12; yl = np.abs(cons.aldex2_effect_oriented).max() * 1.15
axB.set_xlim(-xl, xl); axB.set_ylim(-yl, yl)
axB.set_xticks([t for t in range(-int(xl), int(xl) + 1)]); axB.set_yticks([t for t in range(-int(yl), int(yl) + 1)])
axB.set_xlabel('ANCOM-BC2 log fold change (natural log)')
axB.set_ylabel('ALDEx2 effect size')
# quadrant annotations
qkw = dict(fontsize=6, color='0.45', transform=axB.transAxes)
axB.text(0.99, 0.985, 'Concordant: higher in first-named group', ha='right', va='top', **qkw)
axB.text(0.01, 0.015, 'Concordant: lower in first-named group', ha='left', va='bottom', **qkw)
axB.text(0.01, 0.985, 'Discordant', ha='left', va='top', **qkw)
axB.text(0.99, 0.015, 'Discordant', ha='right', va='bottom', **qkw)
axB.text(0.99, 0.075, f'Concordant {N_CONC} of {N_CONS}; discordant {N_DISC} of {N_CONS}', ha='right', va='bottom',
         fontsize=6.5, transform=axB.transAxes)
leg_h = [Line2D([], [], ls='', marker='o', ms=4.5, mfc=COL['emp'], mec=COL['emp'], label='Burial vs EMP'),
         Line2D([], [], ls='', marker='o', ms=4.5, mfc=COL['hmp'], mec=COL['hmp'], label='Burial vs HMP'),
         Line2D([], [], ls='', marker='o', ms=4.5, mfc=COL['control'], mec=COL['control'], label='Burial vs control'),
         Line2D([], [], ls='', marker='o', ms=4.5, mfc='white', mec=COL['hmp'], mew=1.0, label='Control vs HMP'),
         Line2D([], [], ls='', marker='s', ms=4.5, mfc='0.6', mec='0.6', label='Family (L5)'),
         Line2D([], [], ls='', marker='o', ms=4.5, mfc='0.6', mec='0.6', label='Genus (L6)'),
         Line2D([], [], ls='', marker='o', ms=7.5, mfc='none', mec='k', mew=0.9, label='Discordant direction')]
axB.legend(handles=leg_h, loc='upper left', bbox_to_anchor=(0.0, 0.94), fontsize=6.3, handletextpad=0.3, labelspacing=0.35)
letter(axB, 'a'.replace('a', 'b'), dx=-0.065, dy=1.0)

fig.canvas.draw()
# labels: all discordant + four largest |LFC|
lab = disc.copy()
big = cons[cons.concordant].assign(a=lambda d: d.ancombc2_lfc.abs()).sort_values('a', ascending=False).head(4)
lab = pd.concat([lab, big.drop(columns='a')])
items = []
for _, r_ in lab.iterrows():
    s = short_name(r_.taxon, r_.level)
    # discordant taxa sit in the lower-right quadrant close together: send labels outward, away from neighbours
    pref = np.sign(r_.ancombc2_lfc) or 1
    if not r_.concordant and r_.ancombc2_lfc > 0:
        pref = {('burial_vs_emp', 'L6'): [100, 120], ('burial_vs_emp', 'L5'): [-120, -135],
                ('burial_vs_hmp', 'L5'): [-10, -30] if 'Clostridia' in r_.taxon else [-60, -45],
                ('burial_vs_hmp', 'L6'): [30, 15] if 'Clostridia' in r_.taxon else [-30, -45]}[(r_.comparison, r_.level)]
    items.append((r_.ancombc2_lfc, r_.aldex2_effect_oriented, s, pref))
pts = axB.transData.transform(cons[['ancombc2_lfc', 'aldex2_effect_oriented']].values)
rr = fig.canvas.get_renderer()
avoid = [t.get_window_extent(rr) for t in axB.texts] + [axB.get_legend().get_window_extent(rr)]
place_labels(axB, fig, items, pts, avoid=avoid)
print('Labelled taxa:'); [print('  ', i[2], round(i[0], 3), round(i[1], 3)) for i in items]

rep = save(fig, 'fig6_da')
print(json.dumps({k: rep[k] for k in ('png', 'overlaps', 'outside')}))
nums = {'counts': counts.to_dict(orient='records'), 'n_consensus': N_CONS, 'n_concordant': N_CONC, 'n_discordant': N_DISC,
        'discordant': disc[['taxon', 'comparison', 'level', 'ancombc2_lfc', 'aldex2_effect_oriented']].round(4).to_dict(orient='records'),
        'largest_abs_lfc_labelled': big[['taxon', 'comparison', 'level', 'ancombc2_lfc', 'aldex2_effect_oriented']].round(4).to_dict(orient='records'),
        'overlaps': rep['overlaps'], 'outside': rep['outside']}
json.dump(nums, open(f'{OUT}/fig6_da_numbers.json', 'w'), indent=1)
