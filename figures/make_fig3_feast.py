import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); from figstyle_nyabg import *; setup()
"""Fig. 3 (stem fig3_feast): depth-matched FEAST source attribution (reference sources rarefied to 500 reads).
(a) per-sample stacked bars, burial (n = 72) then control (n = 6), each sorted by EMP proportion descending (submission order)
(b) per-source boxplots with points, burial vs control
(c) mean +/- SD per source
(d) HMP attribution per burial sample across five independent depth-matched FEAST stability replicates
Inputs (inputs_manifest.md): R/feast_depth_matched/r500/{burial,control}_source_props.tsv; R5/feast_depth_matched/r500/stability/replicate_[1-5]_burial.tsv
"""
import json
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

R = 'results'
R5 = 'results'
SRC = ['EMP', 'HMP', 'Unknown']; SKEY = {'EMP': 'emp', 'HMP': 'hmp', 'Unknown': 'unknown'}
THR = 0.01  # stability threshold: HMP attribution >= 1%

rd = lambda p: pd.read_csv(p, sep='\t', dtype={'sample_id': str})
bur = rd(f'{R}/feast_depth_matched/r500/burial_source_props.tsv')
con = rd(f'{R}/feast_depth_matched/r500/control_source_props.tsv')
reps = pd.concat([rd(f'{R5}/feast_depth_matched/r500/stability/replicate_{i}_burial.tsv').set_index('sample_id')['HMP'].rename(f'rep{i}')
                  for i in range(1, 6)], axis=1)
assert len(bur) == 72 and len(con) == 6 and set(reps.index) == set(bur.sample_id)
for d in (bur, con): assert np.allclose(d[SRC].sum(1), 1, atol=1e-6)

# ---------- numbers (all printed for audit) ----------
N = {'n_burial': len(bur), 'n_control': len(con)}
for nm, d in (('burial', bur), ('control', con)):
    for s in SRC:
        N[f'{nm}_{s}_mean_pct'] = round(100 * d[s].mean(), 1); N[f'{nm}_{s}_sd_pct'] = round(100 * d[s].std(ddof=1), 1)
        N[f'{nm}_{s}_median_pct'] = round(100 * d[s].median(), 2)
k = (reps >= THR).sum(1)
N['stab_all5'] = int((k == 5).sum()); N['stab_none'] = int((k == 0).sum()); N['stab_some'] = int(((k > 0) & (k < 5)).sum())
N['stab_k_counts'] = {int(i): int(v) for i, v in k.value_counts().sort_index().items()}
prim = bur.set_index('sample_id')['HMP']
N['primary_ge1pct'] = int((prim >= THR).sum())
N['primary_ge1pct_by_class'] = {'all5': int((prim[k == 5] >= THR).sum()), 'some': int((prim[(k > 0) & (k < 5)] >= THR).sum()),
                                'none': int((prim[k == 0] >= THR).sum())}
rng_ = reps.max(1) - reps.min(1)
N['stab_rep_range_median_pctpts'] = round(100 * rng_.median(), 2); N['stab_rep_range_max_pctpts'] = round(100 * rng_.max(), 1)
N['stab_rep_mean_HMP_pct_by_replicate'] = [round(100 * reps[c].mean(), 1) for c in reps.columns]
# reference sources actually present in the depth-matched (500-read) FEAST input
R2 = 'results'
hdr = open(f'{R2}/feast_depth_matched/merged_L6genus_r500.tsv').readline().rstrip('\n').split('\t')[1:]
md = pd.read_csv(f'{R}/metadata_cohort_split.tsv', sep='\t', dtype=str).set_index('sample-id')['cohort']
N['sources_in_r500_table'] = pd.Series(hdr).map(md).value_counts().to_dict()
N['sources_in_metadata'] = md.value_counts().to_dict()
print(json.dumps(N, indent=1))

def pct(x): return f'{x:.1f}'

# ---------- figure ----------
W, H = mm(170), mm(128)
fig = plt.figure(figsize=(W, H))
gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1.08], hspace=0.42, left=0.07, right=0.985, top=0.93, bottom=0.08)
g0 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], width_ratios=[6.2, 1], wspace=0.02)
g1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], width_ratios=[1.1, 1.05, 1.9], wspace=0.40)
axa = fig.add_subplot(g0[0]); axleg = fig.add_subplot(g0[1]); axleg.axis('off')
axb = fig.add_subplot(g1[0]); axc = fig.add_subplot(g1[1]); axd = fig.add_subplot(g1[2])

def letter_mm(ax, s, off_mm=8.5):
    w = ax.get_position().width * W
    letter(ax, s, dx=-mm(off_mm) / w, dy=1.03)

# (a) stacked bars
bs = bur.sort_values('EMP', ascending=False, kind='mergesort'); cs = con.sort_values('EMP', ascending=False, kind='mergesort')
gap = 2.0
xb = np.arange(len(bs)); xc = len(bs) - 1 + gap + np.arange(len(cs))
for x, d in ((xb, bs), (xc, cs)):
    bottom = np.zeros(len(d))
    for s in SRC:
        axa.bar(x, 100 * d[s].values, bottom=bottom, width=1.0, color=COL[SKEY[s]], edgecolor='none', linewidth=0)
        bottom += 100 * d[s].values
xdiv = len(bs) - 1 + gap / 2
axa.axvline(xdiv, color='black', ls='--', lw=0.7)
axa.set_xlim(-0.6, xc[-1] + 0.6); axa.set_ylim(0, 100); axa.set_xticks([])
axa.set_ylabel('Source attribution (%)'); axa.set_xlabel('Samples (sorted by EMP attribution within group)')
axa.text(np.mean(xb), 101.5, f'Burial (n = {len(bs)})', ha='center', va='bottom', fontsize=7)
axa.text(np.mean(xc), 101.5, f'Control (n = {len(cs)})', ha='center', va='bottom', fontsize=7)
axleg.legend(handles=[Patch(fc=COL[SKEY[s]], ec='none', label=LABEL[SKEY[s]]) for s in SRC], loc='center left',
             handlelength=1.2, borderaxespad=0, fontsize=7)

# (b) boxplots + points
rng = np.random.default_rng(1)
pos = {'burial': -0.19, 'control': 0.19}
for i, s in enumerate(SRC):
    for grp, d in (('burial', bur), ('control', con)):
        v = 100 * d[s].values; x0 = i + pos[grp]
        axb.boxplot(v, positions=[x0], widths=0.3, showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor='none', edgecolor='black', lw=0.6), medianprops=dict(color='black', lw=0.9),
                    whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        axb.scatter(x0 + rng.uniform(-0.1, 0.1, len(v)), v, s=4 if grp == 'burial' else 7, color=COL[grp], alpha=0.8,
                    lw=0.25, edgecolor='white', zorder=3)
axb.set_xticks(range(3)); axb.set_xticklabels([LABEL[SKEY[s]] for s in SRC]); axb.set_xlim(-0.55, 2.55)
axb.set_ylim(-3, 103); axb.set_ylabel('Source attribution (%)')

# (c) mean +/- SD
for i, s in enumerate(SRC):
    for grp, d in (('burial', bur), ('control', con)):
        m, sd = 100 * d[s].mean(), 100 * d[s].std(ddof=1)
        axc.bar(i + pos[grp], m, width=0.36, color=COL[grp], edgecolor='none')
        axc.errorbar(i + pos[grp], m, yerr=sd, color='black', lw=0.6, capsize=1.6, capthick=0.6)
axc.set_xticks(range(3)); axc.set_xticklabels([LABEL[SKEY[s]] for s in SRC]); axc.set_xlim(-0.55, 2.55)
axc.set_ylim(0, 103); axc.set_ylabel('Mean attribution (%)')
axc.legend(handles=[Patch(fc=COL['burial'], label='Burial'), Patch(fc=COL['control'], label='Control')],
           loc='upper right', bbox_to_anchor=(1.02, 1.02), handlelength=1.0, borderaxespad=0.2, fontsize=6.5)

# (d) stability strip / range plot
cls = np.where(k == 5, 'all5', np.where(k == 0, 'none', 'some'))
dd = pd.DataFrame({'k': k, 'cls': cls, 'med': reps.median(1), 'lo': reps.min(1), 'hi': reps.max(1), 'prim': prim.loc[reps.index]})
order = {'all5': 0, 'some': 1, 'none': 2}
dd['o'] = dd.cls.map(order); dd = dd.sort_values(['o', 'k', 'med'], ascending=[True, False, False])
dd['x'] = np.arange(len(dd))
sty = {'all5': dict(c=COL['hmp'], fc=COL['hmp']), 'some': dict(c=COL['hmp'], fc='white'), 'none': dict(c='#8C8C8C', fc='#8C8C8C')}
for sid, r in dd.iterrows():
    st = sty[r.cls]
    axd.vlines(r.x, 100 * r.lo, 100 * r.hi, color=st['c'], lw=0.6, zorder=1)
    axd.scatter(np.full(5, r.x), 100 * reps.loc[sid].values, s=5, facecolor=st['fc'], edgecolor=st['c'], lw=0.4, zorder=2)
    axd.plot([r.x - 0.42, r.x + 0.42], [100 * r.prim] * 2, color='black', lw=0.7, zorder=3)
axd.set_yscale('symlog', linthresh=0.1, linscale=0.45)
axd.set_ylim(-0.02, 100); axd.set_yticks([0, 0.1, 1, 10, 100]); axd.set_yticklabels(['0', '0.1', '1', '10', '100'])
axd.axhline(100 * THR, color='black', ls=':', lw=0.6, zorder=0)
axd.set_xlim(-0.8, len(dd) - 0.2); axd.set_xticks([])
axd.set_ylabel('HMP attribution (%)'); axd.set_xlabel('Burial samples (n = 72)')
lab = {'all5': '5 of 5', 'some': '1 to 4 of 5', 'none': '0 of 5'}
for c_ in ('all5', 'some', 'none'):
    xs = dd.x[dd.cls == c_]
    if c_ != 'none': axd.axvline(xs.max() + 0.5, color='black', lw=0.4, ls='-', alpha=0.5)
    axd.text(xs.mean(), 160, f'{lab[c_]}\nn = {len(xs)}', ha='center', va='bottom', fontsize=6.5, linespacing=1.1)
axd.legend(handles=[Line2D([], [], marker='o', ls='', ms=2.6, mfc='white', mec=COL['hmp'], mew=0.5, label='Replicate'),
                    Line2D([], [], color='black', lw=0.8, label='Primary run'),
                    Line2D([], [], color='black', lw=0.6, ls=':', label='1% threshold')],
           loc='upper right', bbox_to_anchor=(1.0, 0.86), fontsize=6, handlelength=1.4, borderaxespad=0.2, labelspacing=0.3)

fig.canvas.draw()
for ax, s in ((axa, 'a'), (axb, 'b'), (axc, 'c'), (axd, 'd')): letter_mm(ax, s)
rep = save(fig, 'fig3_feast')
print('SAVE', json.dumps(rep))
json.dump(N, open('figures/out/fig3_feast_numbers.json', 'w'), indent=1)
