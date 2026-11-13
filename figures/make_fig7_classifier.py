import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); from figstyle_nyabg import *; setup()
"""Fig. 7. Supervised classification of burial soils against EMP and HMP reference labels (full-depth re-analysis).
(a) EMP EMPO-3 environment type predicted for each burial sample by L1-regularised logistic regression (top class).
(b) HMP body site predicted for burial samples (L1-LR, top subsite class aggregated to body site), with the
    five-site null model (classifier_null.py: one-vs-rest L1-LR trained on the five aggregated HMP sites)
    applied to the same burial samples and to 500 held-out EMP samples.
(c) Maximum posterior probability under the five-site null model: burial (n = 72) vs EMP null (n = 500).
Also writes a table of RF vs L1-LR top-class counts (burial and control soils) for an Additional file.
"""
import json
import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

R = 'results'
OUT = 'figures/out'

HMP_SUBSITE_TO_SITE = {
    'Anterior Nares': 'nasal',
    'Attached Keratinized Gingiva': 'oral', 'Buccal Mucosa': 'oral', 'Hard Palate': 'oral', 'Palatine Tonsils': 'oral',
    'Saliva': 'oral', 'Subgingival Plaque': 'oral', 'Supragingival Plaque': 'oral', 'Throat': 'oral', 'Tongue Dorsum': 'oral',
    'Left Antecubital Fossa': 'skin', 'Right Antecubital Fossa': 'skin', 'Left Retroauricular Crease': 'skin',
    'Right Retroauricular Crease': 'skin',
    'Mid Vagina': 'vaginal', 'Posterior Fornix': 'vaginal', 'Vaginal Introitus': 'vaginal',
    'Stool': 'stool'}


def emp_family(c):
    for k in ('Animal', 'Plant', 'Soil', 'Sediment', 'Water'):
        if c.startswith(k): return 'emp_' + k.lower()
    return 'emp_other'          # Surface (saline / non-saline), Sterile water blank


def top_class(df):
    # top class per sample and model; exact ties broken alphabetically by class name (deterministic)
    d = df.sort_values(['model', 'sample_id', 'probability', 'class'], ascending=[True, True, False, True])
    return d.groupby(['model', 'sample_id'], sort=False).head(1).set_index(['model', 'sample_id'])['class']


bp = pd.read_csv(f'{R}/classifier/burial_predictions.csv', dtype={'sample_id': str})
cp = pd.read_csv(f'{R}/classifier/control_predictions.csv', dtype={'sample_id': str})
nd = pd.read_csv(f'{R}/classifier_null/classifier_null_distribution.tsv', sep='\t', dtype={'sample_id': str})
ns = pd.read_csv(f'{R}/classifier_null/classifier_null_summary.tsv', sep='\t').set_index('metric')['value']

hmp_classes = sorted(bp.loc[bp.model.str.startswith('HMP'), 'class'].unique())
emp_classes = sorted(bp.loc[bp.model.str.startswith('EMP'), 'class'].unique())
assert set(hmp_classes) <= set(HMP_SUBSITE_TO_SITE), set(hmp_classes) - set(HMP_SUBSITE_TO_SITE)
ties = bp.groupby(['model', 'sample_id']).probability.apply(lambda s: int((s == s.max()).sum()))
tie_list = ties[ties > 1]
tb, tc = top_class(bp), top_class(cp)
N_B = bp.sample_id.nunique(); N_C = cp.sample_id.nunique()
assert all(bp.groupby('model').sample_id.nunique() == N_B)

# ------------------------------------------------------------- (a)
a = tb.loc['EMP_L1LR'].value_counts()
print(f'(a) burial n={N_B}; EMP model classes={len(emp_classes)}; predicted classes={len(a)}\n', a.to_string())

# ------------------------------------------------------------- (b)
nb = nd[nd.cohort == 'burial']; ne = nd[nd.cohort == 'emp_holdout']
assert set(nb.sample_id) == set(bp.sample_id) and len(nb) == N_B
N_NULL = len(ne); assert N_NULL == int(ns['emp_n'])
SITES = ['skin', 'oral', 'nasal', 'stool', 'vaginal']
b_main = tb.loc['HMP_L1LR'].map(HMP_SUBSITE_TO_SITE).value_counts().reindex(SITES, fill_value=0)
b_null = nb.predicted_class.str.replace('hmp_', '').value_counts().reindex(SITES, fill_value=0)
e_null = ne.predicted_class.str.replace('hmp_', '').value_counts().reindex(SITES, fill_value=0)
for s in SITES:   # verify against classifier_null_summary.tsv
    assert abs(e_null[s] / N_NULL - ns[f'emp_null_predicted_hmp_{s}']) < 1e-9, s
print('(b) burial HMP_L1LR site counts', b_main.to_dict(), '| burial five-site null model', b_null.to_dict(),
      '| EMP null', e_null.to_dict(), '| EMP null shares', (e_null / N_NULL).round(3).to_dict())

# ------------------------------------------------------------- (c)
p95 = float(np.percentile(ne.max_confidence, 95)); p99 = float(np.percentile(ne.max_confidence, 99))
assert abs(p95 - ns['emp_null_95th']) < 1e-6 and abs(p99 - ns['emp_null_99th']) < 1e-6
n_above = int((nb.max_confidence > p95).sum()); n_above99 = int((nb.max_confidence > p99).sum())
assert n_above == int(ns['burial_above_null_95th']) and n_above99 == int(ns['burial_above_null_99th'])
med_b, med_e = float(nb.max_confidence.median()), float(ne.max_confidence.median())
assert abs(med_b - ns['burial_median']) < 1e-6 and abs(med_e - ns['emp_null_median']) < 1e-6
from scipy.stats import mannwhitneyu
mw = mannwhitneyu(nb.max_confidence, ne.max_confidence, alternative='two-sided')
print(f'(c) null 95th={p95:.4f}, 99th={p99:.4f}; burial above 95th={n_above}/{N_B}, above 99th={n_above99}; '
      f'median burial={med_b:.3f}, EMP null={med_e:.3f}; Mann-Whitney U={mw.statistic:.0f}, p={mw.pvalue:.3g}')

# ------------------------------------------------------------- RF vs L1-LR table
rows = []
for ref, lev, mapper in (('EMP', 'EMPO-3 class', None), ('EMP', 'EMP habitat group', emp_family),
                         ('HMP', 'HMP body subsite', None), ('HMP', 'HMP body site', HMP_SUBSITE_TO_SITE.get)):
    cols = {}
    for who, t in (('burial', tb), ('control', tc)):
        for mdl in ('L1LR', 'RF'):
            s = t.loc[f'{ref}_{mdl}']; s = s.map(mapper) if mapper else s
            cols[f'{who}_{mdl}'] = s.value_counts()
    df = pd.DataFrame(cols).fillna(0).astype(int)
    df.index = [i.replace('emp_', '') if isinstance(i, str) else i for i in df.index]
    df = df.sort_values(['burial_L1LR', 'burial_RF'], ascending=False)
    for k, r_ in df.iterrows():
        rows.append(dict(reference=ref, level=lev, predicted_class=k, **r_.to_dict()))
tab = pd.DataFrame(rows)
tab.to_csv(f'{OUT}/fig7_rf_vs_l1lr_topclass_counts.tsv', sep='\t', index=False)
agree = {}
for ref, mapper, lab in (('EMP', None, 'EMPO-3'), ('EMP', emp_family, 'EMP habitat group'),
                         ('HMP', None, 'HMP subsite'), ('HMP', HMP_SUBSITE_TO_SITE.get, 'HMP site')):
    l1, rf = tb.loc[f'{ref}_L1LR'], tb.loc[f'{ref}_RF'].reindex(tb.loc[f'{ref}_L1LR'].index)
    if mapper: l1, rf = l1.map(mapper), rf.map(mapper)
    agree[lab] = int((l1 == rf).sum())
print('RF vs L1-LR agreement (burial, of', N_B, '):', agree)
print(tab.to_string(index=False))

# ------------------------------------------------------------- figure
fig = plt.figure(figsize=(mm(170), mm(132)))
gs1 = fig.add_gridspec(1, 2, wspace=0.62, left=0.165, right=0.985, top=0.955, bottom=0.585)
gs2 = fig.add_gridspec(1, 1, left=0.075, right=0.985, top=0.45, bottom=0.085)
axA = fig.add_subplot(gs1[0, 0]); axB = fig.add_subplot(gs1[0, 1]); axC = fig.add_subplot(gs2[0, 0])

# (a) horizontal bars, sorted descending, coloured by EMP habitat group
a_ = a.sort_values(ascending=True)
y = np.arange(len(a_))
axA.barh(y, a_.values, color=[EMP_TYPE[emp_family(c)] for c in a_.index], edgecolor='0.2', linewidth=0.4, height=0.72)
for yi, v in zip(y, a_.values):
    axA.text(v + 0.5, yi, f'{v} ({100 * v / N_B:.1f}%)', va='center', ha='left', fontsize=5.8)
axA.set_yticks(y); axA.set_yticklabels(a_.index, fontsize=6.3)
axA.set_xlim(0, a_.max() * 1.42); axA.set_xlabel(f'Burial samples (n = {N_B})')
axA.set_title('EMP environment type (EMPO-3), L1-LR', fontsize=7.5)
fams = [f for f in ['emp_soil', 'emp_water', 'emp_animal', 'emp_sediment', 'emp_plant', 'emp_other']
        if f in {emp_family(c) for c in a_.index}]
axA.legend(handles=[Patch(facecolor=EMP_TYPE[f], edgecolor='0.2', lw=0.4, label=f.replace('emp_', '').capitalize()) for f in fams],
           loc='lower right', fontsize=6, handlelength=0.9, handleheight=0.8, title='Habitat group', title_fontsize=6)
letter(axA, 'a', dx=-0.52, dy=1.03)

# (b) grouped horizontal bars per body site
series = [('Burial, HMP subsite model', b_main / N_B, N_B, COL['burial'], None),
          ('Burial, five-site null model', b_null / N_B, N_B, 'white', COL['burial']),
          ('EMP null, five-site null model', e_null / N_NULL, N_NULL, COL['emp'], None)]
hb = 0.26
ys = np.arange(len(SITES))[::-1]
for j, (lab, share, n, fc, ec) in enumerate(series):
    yy = ys + (1 - j) * hb
    axB.barh(yy, 100 * share.values, height=hb * 0.92, color=fc, edgecolor=ec if ec else '0.2', linewidth=0.9 if ec else 0.4,
             hatch='/////' if ec else None, label=f'{lab} (n = {n})')
    for yi, s_, site in zip(yy, share.values, SITES):
        cnt = int(round(s_ * n))
        axB.text(100 * s_ + 1.2, yi, f'{100 * s_:.1f}%', va='center', ha='left', fontsize=5.3)
axB.set_yticks(ys); axB.set_yticklabels([s.capitalize() for s in SITES], fontsize=6.5)
axB.tick_params(axis='y', pad=9)
for yi, s in zip(ys, SITES):   # site colour key (HMP_SITE), drawn between tick labels and axis
    axB.add_patch(Rectangle((-0.075, yi - 0.17), 0.04, 0.34, transform=mpl.transforms.blended_transform_factory(axB.transAxes, axB.transData),
                            facecolor=HMP_SITE[f'hmp_{s}'], edgecolor='0.3', lw=0.3, clip_on=False))
axB.set_xlim(0, 118); axB.set_xticks([0, 20, 40, 60, 80, 100]); axB.set_xlabel('Samples assigned (%)')
axB.set_ylim(-0.6, len(SITES) - 0.35)
axB.set_title('HMP body site, L1-LR', fontsize=7.5)
axB.legend(loc='lower right', fontsize=5.8, handlelength=1.4, labelspacing=0.3)
letter(axB, 'b', dx=-0.33, dy=1.03)

# (c) maximum posterior probability distributions
bins = np.linspace(0.2, 1.0, 33)
lo = min(nb.max_confidence.min(), ne.max_confidence.min()); assert lo >= bins[0]
we = np.ones(N_NULL) / N_NULL * 100; wb = np.ones(N_B) / N_B * 100
axC.hist(ne.max_confidence, bins=bins, weights=we, color=COL['emp'], alpha=0.35, edgecolor='none', label=f'EMP null (n = {N_NULL})')
axC.hist(ne.max_confidence, bins=bins, weights=we, histtype='step', color=COL['emp'], lw=0.9)
axC.hist(nb.max_confidence, bins=bins, weights=wb, histtype='step', color=COL['burial'], lw=1.4, label=f'Burial (n = {N_B})')
axC.axvline(p95, color='0.15', ls='--', lw=0.8)
ymaxc = axC.get_ylim()[1] * 1.12; rug_lo = -0.085 * ymaxc; axC.set_ylim(rug_lo, ymaxc)
# rug: one tick per burial sample, below the baseline
axC.vlines(nb.max_confidence, rug_lo * 0.9, rug_lo * 0.2, color=COL['burial'], lw=0.5)
axC.axhline(0, color='0.3', lw=0.5)
axC.set_yticks([t for t in axC.get_yticks() if 0 <= t <= ymaxc * 0.97])
axC.text(p95 - 0.004, ymaxc * 0.97, f'EMP null 95th percentile = {p95:.3f}\n{n_above} of {N_B} burial samples above',
         ha='right', va='top', fontsize=6)
axC.plot([med_b], [ymaxc * 0.04], marker='v', color=COL['burial'], ms=4, ls='', mec='white', mew=0.4)
axC.plot([med_e], [ymaxc * 0.04], marker='v', color=COL['emp'], ms=4, ls='', mec='white', mew=0.4)
axC.set_xlim(bins[0], 1.0)
axC.set_xlabel('Maximum posterior probability (five-site HMP null model)')
axC.set_ylabel('Samples per bin (%)')
axC.legend(loc='upper left', fontsize=6.3)
axC.text(0.01, 0.70, f'Medians (triangles): burial {med_b:.3f}, EMP null {med_e:.3f}\nTicks below baseline: individual burial samples',
         transform=axC.transAxes, fontsize=6, va='top')
letter(axC, 'c', dx=-0.058, dy=1.03)

rep = save(fig, 'fig7_classifier')
print(json.dumps({k: rep[k] for k in ('png', 'overlaps', 'outside')}))
nums = dict(n_burial=N_B, n_control=N_C, n_emp_null=N_NULL, emp_model_classes=len(emp_classes), hmp_model_classes=len(hmp_classes),
            a_emp_l1lr_counts=a.to_dict(), b_burial_hmp_l1lr_site=b_main.to_dict(), b_burial_null_model_site=b_null.to_dict(),
            b_emp_null_site=e_null.to_dict(), b_emp_null_share=(e_null / N_NULL).round(3).to_dict(),
            c_null_p95=round(p95, 4), c_null_p99=round(p99, 4), c_burial_above_p95=n_above, c_burial_above_p99=n_above99,
            c_median_burial=round(med_b, 4), c_median_emp_null=round(med_e, 4),
            c_mannwhitney_U=float(mw.statistic), c_mannwhitney_p=float(mw.pvalue),
            rf_l1lr_agreement_burial=agree, top_class_ties=[list(i) for i in tie_list.index],
            control_emp_l1lr=tc.loc['EMP_L1LR'].value_counts().to_dict(),
            control_hmp_l1lr_site=tc.loc['HMP_L1LR'].map(HMP_SUBSITE_TO_SITE).value_counts().to_dict(),
            overlaps=rep['overlaps'], outside=rep['outside'])
json.dump(nums, open(f'{OUT}/fig7_classifier_numbers.json', 'w'), indent=1, default=str)
