import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); from figstyle_nyabg import *; setup()
"""Additional file 6 (stem af6_within_burial): within-burial heterogeneity of depth-matched FEAST HMP attribution
(reference sources rarefied to 500 reads) for burials represented by >= 2 samples.
Recomputed from R/gradient_r500/gradient_sample_table.tsv (hmp column = r500 FEAST HMP proportion; verified against
R/feast_depth_matched/r500/burial_source_props.tsv). R/within_burial/ (default 11-read FEAST) is not used.
(a) per-sample HMP attribution grouped by burial, coloured by proximity level; (b) within-burial range vs burial mean.
"""
import json
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

R = 'results'
g = pd.read_csv(f'{R}/gradient_r500/gradient_sample_table.tsv', sep='\t', dtype={'sample_id': str, 'burial_id': str})
f = pd.read_csv(f'{R}/feast_depth_matched/r500/burial_source_props.tsv', sep='\t', dtype={'sample_id': str}).set_index('sample_id')
gb = g[g.cohort == 'burial'].copy()
assert np.allclose(gb.set_index('sample_id').hmp, f.loc[gb.sample_id, 'HMP'].values)
vc = gb.burial_id.value_counts(); multi = vc[vc > 1].index
m = gb[gb.burial_id.isin(multi)].copy(); m['pct'] = 100 * m.hmp
LEV = ['Cranium', 'Other Body', 'Soil Matrix']; m['lv'] = m.level.map({l: i for i, l in enumerate(LEV)})
S = m.groupby('burial_id').agg(n=('pct', 'size'), mean=('pct', 'mean'), median=('pct', 'median'), lo=('pct', 'min'),
                               hi=('pct', 'max'), nlev=('level', 'nunique'))
S['range'] = S.hi - S.lo
S = S.sort_values('range', ascending=False)

# within-burial ordering among samples from different proximity levels (descriptive)
conc = disc = ties = 0
for b, d in m.groupby('burial_id'):
    for i in range(len(d)):
        for j in range(len(d)):
            a, c = d.iloc[i], d.iloc[j]
            if a.lv < c.lv:
                if a.pct > c.pct: conc += 1
                elif a.pct < c.pct: disc += 1
                else: ties += 1
N = dict(n_burials_total=int(gb.burial_id.nunique()), n_multi=int(len(multi)), n_samples_multi=int(len(m)),
         n_single_sample_burials=int((vc == 1).sum()), n_multi_with_ge2_levels=int((S.nlev >= 2).sum()),
         samples_per_burial=S.n.astype(int).to_dict(), range_pctpts=S.range.round(2).to_dict(), mean_pct=S['mean'].round(2).to_dict(),
         median_range_pctpts=round(float(S.range.median()), 2), n_range_ge10pctpts=int((S.range >= 10).sum()),
         n_samples_ge1pct=int((m.pct >= 1).sum()), n_burials_with_any_ge1pct=int((S.hi >= 1).sum()),
         n_burials_mixed_ge1_and_lt1=int(((S.hi >= 1) & (S.lo < 1)).sum()),
         between_level_pairs=dict(proximal_higher=conc, proximal_lower=disc, ties=ties))
N['level_counts_multi'] = m.level.value_counts().to_dict()
print(json.dumps(N, indent=1))

W, H = mm(170), mm(132)
fig = plt.figure(figsize=(W, H))
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.25, 1], wspace=0.30, left=0.205, right=0.985, top=0.93, bottom=0.08)
axa = fig.add_subplot(gs[0]); axb = fig.add_subplot(gs[1])

# (a) per-sample dot plot grouped by burial (header row per burial)
BR = {'Upper_Body': 'Upper body', 'Lower_Body': 'Lower body', 'Soil_Matrix': 'Soil matrix'}
pretty = lambda s: BR.get(s, s.replace('_', '/'))
y = 0; yt, ytl, band = [], [], []
for k_, (b, r) in enumerate(S.iterrows()):
    d = m[m.burial_id == b].sort_values(['lv', 'pct'], ascending=[True, False])
    y0 = y
    axa.text(0.995, y, f'Burial {b}: n = {int(r.n)}, range {r.range:.1f} pp' if r.range >= 0.1 else f'Burial {b}: n = {int(r.n)}, range {r.range:.2f} pp', transform=axa.get_yaxis_transform(),
             ha='right', va='center', fontsize=6.3, fontweight='bold')
    y += 1
    for _, s in d.iterrows():
        axa.scatter(s.pct, y, s=16, color=PROX[s.level], edgecolor='black', lw=0.3, zorder=3)
        yt.append(y); ytl.append(f'{s.sample_id} ({pretty(s.body_region)})'); y += 1
    axa.hlines(np.arange(y0 + 1, y), -0.02, 100, color='#E6E6E6', lw=0.4, zorder=0)
    band.append((y0 - 0.5, y - 0.5, k_))
    y += 0.4
for lo, hi, k_ in band:
    if k_ % 2 == 0: axa.axhspan(lo, hi, color='#F5F5F5', zorder=-1, lw=0)
axa.set_yticks(yt); axa.set_yticklabels(ytl, fontsize=5.8); axa.tick_params(axis='y', length=0)
axa.set_ylim(y - 0.6, -0.8)
axa.set_xscale('symlog', linthresh=0.1, linscale=0.45); axa.set_xlim(-0.02, 100)
axa.set_xticks([0, 0.1, 1, 10, 100]); axa.set_xticklabels(['0', '0.1', '1', '10', '100'])
axa.set_xticks([mm_ * 10.0 ** e for e in (-2, -1, 0, 1) for mm_ in range(2, 10)], minor=True)
axa.axvline(1, color='black', ls=':', lw=0.6, zorder=1)
axa.set_xlabel('HMP attribution (%)')
axa.spines['left'].set_visible(False)

# (b) range vs mean, size = n samples, filled = burial spans >= 2 proximity levels
sz = lambda n: 10 + 9 * n
for b, r in S.iterrows():
    filled = r.nlev >= 2
    axb.scatter(r['mean'], r.range, s=sz(r.n), facecolor=COL['burial'] if filled else 'white', edgecolor=COL['burial'], lw=0.8, zorder=3)
axb.set_xscale('symlog', linthresh=0.1, linscale=0.45); axb.set_yscale('symlog', linthresh=0.1, linscale=0.45)
for setlim, setticks, setlab in ((axb.set_xlim, axb.set_xticks, axb.set_xticklabels), (axb.set_ylim, axb.set_yticks, axb.set_yticklabels)):
    setlim(-0.02, 100); setticks([0, 0.1, 1, 10, 100]); setlab(['0', '0.1', '1', '10', '100'])
axb.set_xticks([mm_ * 10.0 ** e for e in (-2, -1, 0, 1) for mm_ in range(2, 10)], minor=True)
axb.set_yticks([mm_ * 10.0 ** e for e in (-2, -1, 0, 1) for mm_ in range(2, 10)], minor=True)
# labels: merge labels of points that coincide on the display (near the origin)
fig.canvas.draw()
pts = {b: axb.transData.transform((r['mean'], r.range)) for b, r in S.iterrows()}
groups = []
for b in S.sort_values('mean').index:
    for gr in groups:
        if np.hypot(*(pts[b] - pts[gr[0]])) < mm(2.5) * fig.dpi: gr.append(b); break
    else: groups.append([b])
# greedy label placement: first candidate offset whose box clears placed labels, markers and axes edge
rr = fig.canvas.get_renderer(); placed = []
mk = [axb.transData.transform((r['mean'], r.range)) for _, r in S.iterrows()]
cands = [(6, -2.5, 'left'), (-6, -2.5, 'right'), (0, 6, 'center'), (0, -11, 'center'), (6, 4, 'left'), (6, -9, 'left'), (-6, 4, 'right'), (-6, -9, 'right')]
for gr in sorted(groups, key=lambda gr: -S.loc[gr[0], 'range']):
    b0 = gr[0]; r = S.loc[b0]
    for dx, dy, ha in cands:
        t = axb.annotate(', '.join(f'B{x}' for x in sorted(gr)), (r['mean'], r.range), xytext=(dx, dy), textcoords='offset points',
                         fontsize=6, ha=ha)
        bb = t.get_window_extent(rr).expanded(1.05, 1.15); axbb = axb.get_window_extent(rr)
        hit = any(bb.overlaps(o) for o in placed) or any(bb.contains(*q) for q in mk) or bb.x1 > axbb.x1 or bb.y1 > axbb.y1
        if not hit: placed.append(bb); break
        t.remove()
    else: raise RuntimeError(f'no label slot for {gr}')
axb.set_xlabel('Burial mean HMP attribution (%)'); axb.set_ylabel('Within-burial range (percentage points)')
nn = sorted(S.n.astype(int).unique())
h_size = [Line2D([], [], marker='o', ls='', ms=np.sqrt(sz(n)), mfc='white', mec=COL['burial'], label=f'{n} samples') for n in nn]
h_fill = [Line2D([], [], marker='o', ls='', ms=np.sqrt(sz(2)), mfc=COL['burial'], mec=COL['burial'], label='2 proximity levels'),
          Line2D([], [], marker='o', ls='', ms=np.sqrt(sz(2)), mfc='white', mec=COL['burial'], label='1 proximity level')]
leg1 = axb.legend(handles=h_size, loc='lower right', fontsize=6, title='Samples per burial', title_fontsize=6, labelspacing=0.5,
                  borderaxespad=0.3, handletextpad=0.4)
axb.add_artist(leg1)
axb.legend(handles=h_fill, loc='upper left', fontsize=6, borderaxespad=0.3, handletextpad=0.4)

# proximity legend for (a), above the panel
axa.legend(handles=[Line2D([], [], marker='o', ls='', ms=4.2, mfc=PROX[l], mec='black', mew=0.3, label=PROX_LABEL[l]) for l in LEV]
           + [Line2D([], [], color='black', ls=':', lw=0.6, label='1%')],
           loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=4, fontsize=6.5, borderaxespad=0.2, handletextpad=0.3, columnspacing=1.0)
fig.canvas.draw()
W_in = fig.get_size_inches()[0]
letter(axa, 'a', dx=-(mm(31)) / (axa.get_position().width * W_in), dy=1.045)
letter(axb, 'b', dx=-(mm(11)) / (axb.get_position().width * W_in), dy=1.045)
rep = save(fig, 'af6_within_burial')
print('SAVE', json.dumps(rep))
json.dump(N, open('figures/out/af6_within_burial_numbers.json', 'w'), indent=1)
