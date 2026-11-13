import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); from figstyle_nyabg import *; setup()
"""Fig. 4 (stem fig4_gradient): depth-matched FEAST HMP attribution by anatomical proximity level.
Burial levels Cranium, Other body region, Soil matrix; control soils shown separately (reference only, not part of the tests).
Annotations computed from: R/gradient_r500/gradient_tests.tsv (within-burial 3-level JT, two-sided permutation p;
body-associated vs soil matrix Mann-Whitney two-sided p and Cliff's d) and L/gradient_beta_glmm.tsv (primary r500, beta GLMM row).
y axis: symmetric log (linear below 0.1%, logarithmic above) because most values lie below 1%.
"""
import json
import pandas as pd
from matplotlib.patches import Rectangle

R = 'results'
L = 'results/revision'
g = pd.read_csv(f'{R}/gradient_r500/gradient_sample_table.tsv', sep='\t', dtype={'sample_id': str, 'burial_id': str})
T = pd.read_csv(f'{R}/gradient_r500/gradient_tests.tsv', sep='\t')
G = pd.read_csv(f'{L}/gradient_beta_glmm.tsv', sep='\t')
gs_ = pd.read_csv(f'{R}/gradient_r500/gradient_group_summary.tsv', sep='\t').set_index('level')

LEV = ['Cranium', 'Other Body', 'Soil Matrix']
jt = T[T.test == 'JT decreasing trend, within burial (3 levels)'].iloc[0]
mw = T[T.test == 'Body-associated vs soil matrix: Mann-Whitney'].iloc[0]
gl = G[(G.table == 'primary r500') & G.model.str.startswith('beta GLMM')].iloc[0]
assert jt.n == '15/31/26' and mw.n == '46/26' and int(gl.n) == 72

N = {}
for lv in LEV + ['Control']:
    v = 100 * g.loc[g.level == lv, 'hmp']
    N[f'{lv}_n'] = int(len(v)); N[f'{lv}_median_pct'] = round(float(v.median()), 3); N[f'{lv}_mean_pct'] = round(float(v.mean()), 2)
    N[f'{lv}_sd_pct'] = round(float(v.std(ddof=1)), 2); N[f'{lv}_ge1pct'] = int((v >= 1).sum())
    # cross-check with gradient_group_summary.tsv
    assert abs(gs_.loc[lv, 'median'] * 100 - v.median()) < 1e-9 and int(gs_.loc[lv, 'n']) == len(v)
N['n_burial_samples'] = int((g.cohort == 'burial').sum()); N['n_burials'] = int(g.loc[g.cohort == 'burial', 'burial_id'].nunique())
N['JT_within_burial_3level'] = dict(n=jt.n, JT=float(jt.JT), p_perm_two_sided=float(jt.p_perm_two_sided), p_normal_two_sided=float(jt.p_normal_two_sided))
N['MW_body_vs_soil'] = dict(n=mw.n, U=float(mw.U), p_two_sided=float(mw.p_two_sided), cliffs_d=float(mw.cliffs_d),
                            median_body_pct=round(100 * float(mw.median1), 3), median_soil_pct=round(100 * float(mw.median2), 3))
N['beta_GLMM'] = dict(model=gl.model, estimate=float(gl.estimate), se=float(gl.se), z=float(gl.z), p=float(gl.p), n=int(gl.n))
zi = G[(G.table == 'primary r500') & G.model.str.startswith('ZI-beta')].iloc[0]
N['ZI_beta_GLMM_for_reference'] = dict(estimate=float(zi.estimate), p=float(zi.p))
print(json.dumps(N, indent=1, default=str))

def fp(p): return f'{p:.2g}' if p >= 0.01 else f'{p:.1g}'
def fp3(p): return f'{p:.3f}' if p >= 0.001 else f'{p:.1e}'

W, H = mm(120), mm(95)
fig, ax = plt.subplots(figsize=(W, H)); fig.subplots_adjust(left=0.12, right=0.98, top=0.87, bottom=0.21)
pos = {'Cranium': 0, 'Other Body': 1, 'Soil Matrix': 2, 'Control': 3.45}
ax.add_patch(Rectangle((2.88, -1), 1.2, 101, facecolor='#F2F2F2', edgecolor='none', zorder=0))
rng = np.random.default_rng(4)
for lv, x in pos.items():
    v = 100 * g.loc[g.level == lv, 'hmp'].values
    ax.boxplot(v, positions=[x], widths=0.5, showfliers=False, patch_artist=True, zorder=2,
               boxprops=dict(facecolor='none', edgecolor='black', lw=0.6), medianprops=dict(color='black', lw=1.0),
               whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    ax.scatter(x + rng.uniform(-0.17, 0.17, len(v)), v, s=11, color=PROX[lv], edgecolor='black', lw=0.3, alpha=0.9, zorder=3)
ax.set_yscale('symlog', linthresh=0.1, linscale=0.45)
ax.set_ylim(-0.02, 600); ax.set_yticks([0, 0.1, 1, 10, 100]); ax.set_yticklabels(['0', '0.1', '1', '10', '100'])
ax.spines['left'].set_bounds(0, 100)
ax.set_yticks([m * 10.0 ** e for e in (-2, -1, 0, 1) for m in range(2, 10)], minor=True)
ax.set_xlim(-0.55, 4.05)
ax.set_xticks(list(pos.values()))
ax.set_xticklabels([f'{PROX_LABEL[l]}\n(n = {N[l + "_n"]})' for l in pos])
ax.set_ylabel('HMP attribution (%)')
ax.text(1, -0.2, f'Burial samples ({N["n_burial_samples"]} samples, {N["n_burials"]} burials)', transform=ax.get_xaxis_transform(),
        ha='center', va='top', fontsize=7)
ax.text(3.45, -0.2, 'Reference only', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=7, style='italic')

# bracket: body-associated (Cranium + Other body) vs soil matrix
yb, yt = 130, 230
ax.plot([-0.25, 1.25], [yb, yb], color='black', lw=0.6)
ax.plot([0.5, 0.5, 2.0, 2.0], [yb, yt, yt, yb], color='black', lw=0.6)
ax.text(1.25, yt * 1.15, f'Body-associated vs soil matrix: Mann-Whitney p = {fp3(N["MW_body_vs_soil"]["p_two_sided"])}, '
        f"Cliff's d = {N['MW_body_vs_soil']['cliffs_d']:.2f}", ha='center', va='bottom', fontsize=6.5)
# top annotation block (burial samples only)
ax.text(0.0, 1.085, f'Jonckheere-Terpstra trend, burial samples only (3 levels): two-sided permutation p = {fp3(N["JT_within_burial_3level"]["p_perm_two_sided"])}',
        transform=ax.transAxes, ha='left', va='bottom', fontsize=6.5)
ax.text(0.0, 1.025, f'Beta GLMM, burial random intercept (n = {N["beta_GLMM"]["n"]}), proximity p = {fp3(N["beta_GLMM"]["p"])}',
        transform=ax.transAxes, ha='left', va='bottom', fontsize=6.5)
fig.canvas.draw()
rep = save(fig, 'fig4_gradient')
print('SAVE', json.dumps(rep))
json.dump(N, open('figures/out/fig4_gradient_numbers.json', 'w'), indent=1, default=str)
