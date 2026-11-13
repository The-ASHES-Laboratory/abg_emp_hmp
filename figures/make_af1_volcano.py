import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); from figstyle_nyabg import *; setup()
"""Additional file 1. Genus-level (L6) volcano plots for ANCOM-BC2 and ALDEx2, burial vs EMP, HMP and control.
Orientation: positive x = higher in burial for both methods.
  ANCOM-BC2: lfc_cohortburial (natural log; burial is the non-reference group) used as is.
  ALDEx2: effect = alphabetically later group minus earlier; burial is earlier in all three contrasts, so effect * -1.
  Verified: sign(oriented effect) == sign(rab.win.burial - rab.win.<comparator>) for every significant taxon.
"""
import re, json
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = 'results'
L = 'results/revision'
OUT = 'figures/out'
mpl.rcParams.update({'mathtext.fontset': 'custom', 'mathtext.it': 'Arial:italic', 'mathtext.rm': 'Arial'})
ALPHA = 0.05; LEV = 'L6'
COMPS = [('emp', 'EMP'), ('hmp', 'HMP'), ('control', 'control')]
NS = '#DADADA'
cons = pd.read_csv(f'{L}/../diff_abun/da_consensus_oriented.tsv', sep='\t')


def short_name(tax):
    names = [r.split('__', 1)[1] if '__' in r else r for r in tax.split(';')]
    names = [n for n in names if n and n.lower() != 'uncultured']
    n = names[-1]
    return rf'$\it{{{n}}}$' if re.fullmatch(r'[A-Z][a-z]+', n) else n


def place_labels(ax, fig, items, pts_disp, fs=5.4, radii=(9, 13, 18, 24, 32, 42), avoid=()):
    r = fig.canvas.get_renderer(); placed = list(avoid); axbb = ax.get_window_extent(r)
    angs = np.deg2rad([45, 135, -45, -135, 0, 180, 90, -90, 20, 160, -20, -160, 70, 110, -70, -110])
    for (x, y, s, pref) in items:
        done = False
        for rad in radii:
            order = sorted(angs, key=lambda a: (0 if np.sign(np.cos(a)) == pref else 1))
            for a in order:
                dx, dy = rad * np.cos(a), rad * np.sin(a)
                ha = 'left' if dx > 1 else ('right' if dx < -1 else 'center')
                va = 'bottom' if dy > 1 else ('top' if dy < -1 else 'center')
                t = ax.annotate(s, (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=fs, ha=ha, va=va,
                                annotation_clip=False)
                t.update_positions(r); bb = mpl.text.Text.get_window_extent(t, r).expanded(1.04, 1.10)
                inside = bb.x0 > axbb.x0 + 2 and bb.x1 < axbb.x1 - 2 and bb.y0 > axbb.y0 + 2 and bb.y1 < axbb.y1 - 2
                mg = 5.0   # display px; exceeds marker radius at figure dpi
                hit = np.any((pts_disp[:, 0] > bb.x0 - mg) & (pts_disp[:, 0] < bb.x1 + mg) &
                             (pts_disp[:, 1] > bb.y0 - mg) & (pts_disp[:, 1] < bb.y1 + mg))
                if inside and not hit and not any(bb.overlaps(p) for p in placed):
                    placed.append(bb); done = True
                    p0 = ax.transData.transform((x, y)); tb = mpl.text.Text.get_window_extent(t, r)
                    q = np.array([np.clip(p0[0], tb.x0, tb.x1), np.clip(p0[1], tb.y0, tb.y1)]); v = q - p0; d = np.hypot(*v)
                    if d > 6:
                        inv = ax.transData.inverted(); a0 = inv.transform(p0 + v / d * 3.5); a1 = inv.transform(q - v / d * 1.0)
                        ax.add_line(Line2D([a0[0], a1[0]], [a0[1], a1[1]], lw=0.4, color='0.35', zorder=5))
                    break
                t.remove()
            if done: break
        if not done: print('WARNING: could not place label', s)


# ------------------------------------------------------------------ data + orientation audit
panels = {}; numbers = {}
for g2, g2lab in COMPS:
    cmp = f'burial_vs_{g2}'
    an = pd.read_csv(f'{R}/diff_abun/ancombc2_split/ancombc2_{cmp}_{LEV}.tsv', sep='\t')
    al = pd.read_csv(f'{R}/diff_abun/aldex2/aldex2_{cmp}_{LEV}.tsv', sep='\t')
    assert 'lfc_cohortburial' in an.columns
    later = max('burial', g2); sgn = 1 if later == 'burial' else -1       # ALDEx2 effect = later - earlier
    al['x'] = sgn * al['effect']; al['q'] = al['wi.eBH']
    rabd = al['rab.win.burial'] - al[f'rab.win.{g2}']
    sig = al.q < ALPHA
    assert (np.sign(al.x[sig]) == np.sign(rabd[sig])).all(), cmp
    an['x'] = an['lfc_cohortburial']; an['q'] = an['q_cohortburial']
    # audit: ANCOM-BC2 burial coefficient vs ALDEx2 within-group CLR median difference over shared taxa
    j = an.set_index('taxon')[['x']].join(pd.Series(rabd.values, index=al.taxon, name='rabd'), how='inner')
    r_an = float(np.corrcoef(j.x, j.rabd)[0, 1])
    ctax = set(cons[(cons.comparison == cmp) & (cons.level == LEV)].taxon)
    for meth, d in (('ancombc2', an), ('aldex2', al)):
        d = d[['taxon', 'x', 'q']].copy()
        d['sig'] = d.q < ALPHA; d['cons'] = d.taxon.isin(ctax)
        qpos = d.q[d.q > 0].min(); d['capped'] = d.q <= 0
        d['y'] = -np.log10(d.q.clip(lower=qpos))
        cap_y = -np.log10(qpos) * 1.04; d.loc[d.capped, 'y'] = cap_y
        panels[(meth, g2)] = d
        numbers[f'{meth}_{cmp}'] = dict(n_tested=len(d), n_sig=int(d.sig.sum()), n_higher_burial=int((d.sig & (d.x > 0)).sum()),
                                       n_higher_comparator=int((d.sig & (d.x < 0)).sum()), n_consensus=int(d.cons.sum()),
                                       n_q_zero_capped=int(d.capped.sum()), min_positive_q=float(qpos))
    numbers[f'audit_{cmp}'] = dict(aldex2_sign_multiplier=sgn, aldex2_sig_sign_matches_rabwin=True,
                                   r_ancombc2_lfc_vs_rabwin_diff=round(r_an, 3))
print(json.dumps(numbers, indent=1))

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(2, 3, figsize=(mm(170), mm(128)))
fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.14, wspace=0.28, hspace=0.42)
letters = iter('abcdef')
for i, (meth, mlab) in enumerate([('ancombc2', 'ANCOM-BC2'), ('aldex2', 'ALDEx2')]):
    for k, (g2, g2lab) in enumerate(COMPS):
        ax = axes[i, k]; d = panels[(meth, g2)]
        ccol = COL[g2]
        col = np.where(~d.sig, NS, np.where(d.x > 0, COL['burial'], ccol))
        order = np.argsort(d.sig.values)  # non-significant drawn first
        for capped, mk in ((False, 'o'), (True, '^')):
            sel = order[(d.capped.values[order] == capped)]
            ax.scatter(d.x.values[sel], d.y.values[sel], s=np.where(d.sig.values[sel], 11, 6), c=col[sel], marker=mk,
                       edgecolor=np.where(d.cons.values[sel], 'k', 'none'), linewidth=0.45, zorder=3)
        ax.axhline(-np.log10(ALPHA), color='0.5', lw=0.5, ls='--', zorder=1)
        ax.axvline(0, color='0.75', lw=0.5, zorder=1)
        xl = np.abs(d.x).max() * 1.18; ax.set_xlim(-xl, xl)
        ax.set_ylim(0, d.y.max() * 1.22)
        lo, hi = ax.get_xlim(); ax.set_xticks([t for t in ax.get_xticks() if lo <= t <= hi])
        lo, hi = ax.get_ylim(); ax.set_yticks([t for t in ax.get_yticks() if lo <= t <= hi * 0.97])
        ax.set_title(f'{mlab}: burial vs {g2lab}', fontsize=7.5, pad=3)
        ax.set_xlabel('Log fold change (natural log)' if meth == 'ancombc2' else 'Effect size (burial minus comparator)', fontsize=7)
        ax.set_ylabel(r'$-\log_{10}$ q' if meth == 'ancombc2' else r'$-\log_{10}$ BH-adjusted Wilcoxon p', fontsize=7)
        nb = numbers[f'{meth}_burial_vs_{g2}']
        ax.text(0.98, 0.98, f'Higher in burial: {nb["n_higher_burial"]}', transform=ax.transAxes, ha='right', va='top',
                fontsize=5.8, color=COL['burial'])
        ax.text(0.02, 0.98, f'Higher in {g2lab}: {nb["n_higher_comparator"]}', transform=ax.transAxes, ha='left', va='top',
                fontsize=5.8, color=ccol)
        letter(ax, next(letters), dx=-0.26, dy=1.03)
        fig.canvas.draw()
        # sparse labels, two per direction: ANCOM-BC2 most significant (ties by |LFC|);
        # ALDEx2 largest |effect| among significant taxa (ALDEx2 q values are heavily tied at the minimum)
        lab = []
        for sgn_ in (1, -1):
            s_ = d[d.sig & (np.sign(d.x) == sgn_)].assign(ax_=lambda z: z.x.abs()).sort_values(['y', 'ax_'] if meth == 'ancombc2' else ['ax_', 'y'], ascending=False).head(2)
            lab += [(r.x, r.y, short_name(r.taxon), sgn_) for r in s_.itertuples()]
        rr = fig.canvas.get_renderer()
        avoid = [t.get_window_extent(rr) for t in ax.texts]
        place_labels(ax, fig, lab, ax.transData.transform(d[['x', 'y']].values), avoid=avoid)
        numbers[f'{meth}_burial_vs_{g2}']['labelled'] = [l[2] for l in lab]

leg = [Line2D([], [], ls='', marker='o', ms=4, mfc=COL['burial'], mec='none', label='Higher in burial (q < 0.05)'),
       Line2D([], [], ls='', marker='o', ms=4, mfc=COL['emp'], mec='none', label='Higher in EMP'),
       Line2D([], [], ls='', marker='o', ms=4, mfc=COL['hmp'], mec='none', label='Higher in HMP'),
       Line2D([], [], ls='', marker='o', ms=4, mfc=COL['control'], mec='none', label='Higher in control'),
       Line2D([], [], ls='', marker='o', ms=3.2, mfc=NS, mec='none', label='Not significant'),
       Line2D([], [], ls='', marker='o', ms=4, mfc='white', mec='k', mew=0.6, label='Consensus taxon (both methods)'),
       Line2D([], [], ls='', marker='^', ms=4, mfc='0.5', mec='none', label='q below numerical precision (plotted at cap)'),
       Line2D([], [], ls='--', color='0.5', lw=0.6, label='q = 0.05')]
fig.legend(handles=leg, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=6.3, handletextpad=0.3, columnspacing=1.4)
rep = save(fig, 'af1_volcano')
print(json.dumps({k: rep[k] for k in ('png', 'overlaps', 'outside')}))
numbers['overlaps'] = rep['overlaps']; numbers['outside'] = rep['outside']
json.dump(numbers, open(f'{OUT}/af1_volcano_numbers.json', 'w'), indent=1)
