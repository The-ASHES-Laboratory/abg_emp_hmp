import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); from figstyle_nyabg import *; setup()
"""Additional file 5 (stem af5_sensitivity): extended sensitivity analyses.
(a) share of prevalence-filtered taxa assigned to EMP, HMP or ambiguous by prevalence threshold (R/sensitivity/prevalence_sweep.tsv)
(b) sample retention by cohort and total taxa retained vs rarefaction depth (R/sensitivity/rarefaction_sweep.tsv;
    cohort totals from R/sequencing_depth/depth_distribution.tsv)
(c) body-region PERMANOVA among burial samples (L/bodyregion_permanova_r.tsv). R2 values reproduced exactly with vegan
    adonis2 on the merged genus table R2/feast_depth_matched/merged_L6genus_r500.tsv (NYABG sinks unrarefied; 72 burial
    samples; Bray-Curtis on relative abundances; Aitchison = Euclidean on CLR with pseudocount 0.5).
(d) Jonckheere-Terpstra decreasing-trend test (Cranium > Other Body > Soil Matrix) among burial samples, two-sided
    permutation p (9,999 permutations, seed 42), for five FEAST variants: L/gradient_<variant>/gradient_tests.tsv written by
    hpc_pkg_fd_snapshot/scripts_new/32_gradient_tests.py; beta GLMM p from L/gradient_beta_glmm.tsv (where available).
"""
import json
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
B = '.'
R = f'results'; R5 = f'results'; L = f'results/revision'
P = pd.read_csv(f'{R}/sensitivity/prevalence_sweep.tsv', sep='\t')
S = pd.read_csv(f'{R}/sensitivity/rarefaction_sweep.tsv', sep='\t')
TOT = pd.read_csv(f'{R}/sequencing_depth/depth_distribution.tsv', sep='\t', dtype={'sample_id': str}).cohort.value_counts().to_dict()
BR = pd.read_csv(f'{L}/bodyregion_permanova.tsv', sep='\t')
G = pd.read_csv(f'{L}/gradient_beta_glmm.tsv', sep='\t')
gs_ = pd.read_csv(f'{R}/gradient_r500/gradient_sample_table.tsv', sep='\t', dtype={'sample_id': str, 'burial_id': str})
gb = gs_[gs_.cohort == 'burial']; vc = gb.burial_id.value_counts()
VAR = [('primary_r500', 'Depth-matched (primary)', 'primary r500', f'{R}/gradient_r500/gradient_tests.tsv'),
       ('default_coverage', 'FEAST default coverage', None, f'{R}/gradient/gradient_tests.tsv'),
       ('noHumanEMP_r500', 'Depth-matched, EMP human-host\nstudies and blanks excluded', None, None),
       ('contam_r500', 'Depth-matched, blank- and\nmock-derived ASVs removed', 'contam-filtered r500', f'{R5}/contam_sensitivity/gradient_r500_bodysite/gradient_tests.tsv'),
       ('burialmatched', 'Burial-matched source depths', None, None)]
JT = []
for key, lab, glmm_key, existing in VAR:
    t = pd.read_csv(f'{L}/gradient_{key}/gradient_tests.tsv', sep='\t')
    r = t[t.test == 'JT decreasing trend, within burial (3 levels)'].iloc[0]
    s = pd.read_csv(f'{L}/gradient_{key}/gradient_group_summary.tsv', sep='\t').set_index('level')
    rec = dict(variant=key, label=lab.replace('\n', ' '), n=r.n, JT=float(r.JT), z=float(r.z), p_perm_two_sided=float(r.p_perm_two_sided),
               p_normal_two_sided=float(r.p_normal_two_sided),
               mean_pct={l: round(100*float(s.loc[l, 'mean']), 2) for l in ['Cranium', 'Other Body', 'Soil Matrix', 'Control']},
               median_pct={l: round(100*float(s.loc[l, 'median']), 3) for l in ['Cranium', 'Other Body', 'Soil Matrix']},
               detection_rate={l: round(float(s.loc[l, 'detection_rate']), 3) for l in ['Cranium', 'Other Body', 'Soil Matrix']})
    if existing:
        e = pd.read_csv(existing, sep='\t'); e = e[e.test == 'JT decreasing trend, within burial (3 levels)'].iloc[0]
        rec['existing_p_perm_two_sided'] = float(e.p_perm_two_sided); rec['matches_existing'] = bool(np.isclose(e.p_perm_two_sided, r.p_perm_two_sided) and e.JT == r.JT)
    if glmm_key:
        for mdl, nm in [('beta GLMM', 'beta_glmm'), ('ZI-beta GLMM', 'zi_beta_glmm')]:
            g = G[(G.table == glmm_key) & G.model.str.startswith(mdl)].iloc[0]
            rec[nm] = dict(estimate=float(g.estimate), se=float(g.se), p=float(g.p), n=int(g.n))
    JT.append(rec)

N = dict(prevalence={f'{int(round(r.threshold*100))}%': dict(n_taxa=int(r.n_taxa), pct_emp=round(100*r.n_emp/r.n_taxa, 1), pct_hmp=round(100*r.n_hmp/r.n_taxa, 1),
                     pct_ambiguous=round(100*r.n_ambiguous/r.n_taxa, 1)) for r in P.itertuples()},
         rarefaction={int(r.depth): dict(burial=f'{r.n_burial}/{TOT["burial"]}', control=f'{r.n_control}/{TOT["control"]}',
                      pct_emp_samples=round(100*r.n_emp/TOT['emp'], 1), pct_hmp_samples=round(100*r.n_hmp/TOT['hmp'], 1), n_taxa=int(r.n_taxa),
                      pct_hmp_taxa=round(100*r.n_hmp_assigned/r.n_taxa, 1)) for r in S.itertuples()},
         cohort_totals={k: int(v) for k, v in TOT.items()},
         bodyregion_permanova=BR.to_dict(orient='records'),
         bodyregion_categories=gb.body_region.value_counts().to_dict(), n_burial_samples=int(len(gb)), n_burials=int(gb.burial_id.nunique()),
         n_burials_ge2=int((vc >= 2).sum()), n_samples_in_burials_ge2=int(vc[vc >= 2].sum()),
         gradient_variants=JT)
print(json.dumps(N, indent=1, default=str))

W, H = mm(170), mm(158)
fig = plt.figure(figsize=(W, H))
top = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.1], height_ratios=[1, 0.8], left=0.075, right=0.985, top=0.955, bottom=0.575, wspace=0.30, hspace=0.18)
axa = fig.add_subplot(top[:, 0]); axb1 = fig.add_subplot(top[0, 1]); axb2 = fig.add_subplot(top[1, 1], sharex=axb1)
axc = fig.add_axes([0.135, 0.065, 0.27, 0.40]); axd = fig.add_axes([0.585, 0.065, 0.40, 0.40])

# (a) stacked shares
x = np.arange(len(P)); bott = np.zeros(len(P))
for c, lab, col in [('emp', 'EMP', COL['emp']), ('hmp', 'HMP', COL['hmp']), ('ambiguous', 'Ambiguous', COL['unknown'])]:
    v = 100 * P[f'n_{c}'].values / P.n_taxa.values
    axa.bar(x, v, 0.62, bottom=bott, color=col, edgecolor='black', linewidth=0.4, label=lab)
    if c != 'ambiguous':
        for xi, b0, vi in zip(x, bott, v): axa.text(xi, b0 + vi/2, f'{vi:.1f}%', ha='center', va='center', fontsize=6, color='white' if c == 'emp' else 'black')
    bott += v
for xi, r in zip(x, P.itertuples()): axa.text(xi, 101.5, f'n = {r.n_taxa:,}\namb. {100*r.n_ambiguous/r.n_taxa:.1f}%', ha='center', va='bottom', fontsize=5.8, linespacing=1.1)
axa.set_xticks(x); axa.set_xticklabels([f'{int(round(t*100))}%' for t in P.threshold])
axa.set_xlabel('Prevalence threshold'); axa.set_ylabel('Share of assigned taxa (%)'); axa.set_ylim(0, 114); axa.set_yticks(range(0, 101, 20))
axa.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=6.5, handlelength=1.0, columnspacing=1.0)
letter(axa, 'a', dx=-0.2, dy=1.0)

# (b) retention and taxa
x = np.arange(len(S))
for coh, mk, ls in [('burial', 'o', '-'), ('control', 'D', '-'), ('emp', 's', '--'), ('hmp', '^', '--')]:
    v = 100 * S[f'n_{coh}'].values / TOT[coh]
    axb1.plot(x, v, ls, marker=mk, ms=3.5, lw=1.0, color=COL[coh], label=f'{LABEL[coh]} (n = {TOT[coh]:,})', mfc=COL[coh] if ls == '-' else 'white')
for xi, r in zip(x, S.itertuples()):
    axb1.text(xi, 109, f'{r.n_burial}/{TOT["burial"]}', ha='center', va='bottom', fontsize=5.8, color=COL['burial'])
    axb1.text(xi, 125, f'{r.n_control}/{TOT["control"]}', ha='center', va='bottom', fontsize=5.8, color=COL['control'])
axb1.set_ylabel('Samples\nretained (%)'); axb1.set_ylim(-8, 140); axb1.set_yticks([0, 25, 50, 75, 100]); axb1.spines['left'].set_bounds(0, 100)
axb1.legend(loc='lower left', fontsize=5.8, handlelength=1.6, ncol=1, borderaxespad=0.2)
plt.setp(axb1.get_xticklabels(), visible=False)
axb2.bar(x, S.n_taxa, 0.55, color='#8C8C8C', edgecolor='black', linewidth=0.4)
for xi, v in zip(x, S.n_taxa): axb2.text(xi, v + 15, f'{v}', ha='center', va='bottom', fontsize=5.8)
axb2.set_ylim(0, S.n_taxa.max() * 1.25); axb2.set_ylabel('Taxa\nretained')
axb2.set_xticks(x); axb2.set_xticklabels([f'{d:,}' for d in S.depth]); axb2.set_xlabel('Rarefaction depth (reads per sample)')
letter(axb1, 'b', dx=-0.2, dy=1.0)

# (c) body-region PERMANOVA
rows = [('body_region', 'BC'), ('body_region', 'Ait'), ('level', 'BC'), ('level', 'Ait')]
FL = {'body_region': 'Body region\n(11 categories)', 'level': 'Proximity level\n(3 levels)'}; DL = {'BC': 'Bray-Curtis', 'Ait': 'Aitchison'}
y = np.array([3.4, 2.6, 1.0, 0.2])
for yi, (f, d) in zip(y, rows):
    r = BR[(BR.factor == f) & (BR.distance == d)].iloc[0]
    axc.barh(yi, r.R2, 0.62, color=COL['burial'] if d == 'BC' else '#DDB57F', edgecolor='black', linewidth=0.4)
    axc.text(r.R2 + 0.005, yi, f'R\u00b2 = {r.R2:.3f}\np = {r.p_unrestricted:.3f}\nblocked p = {r.p_within_burial_blocks:.3f}', va='center', ha='left', fontsize=5.6, linespacing=1.1)
axc.set_yticks(y); axc.set_yticklabels([f'{FL[f]}\n{DL[d]}' if i % 2 == 0 else DL[d] for i, (f, d) in enumerate(rows)], fontsize=6.3)
axc.set_xlim(0, 0.25); axc.set_ylim(-0.4, 4.0); axc.set_xlabel('PERMANOVA R\u00b2 (n = %d burial samples)' % len(gb))
axc.set_xticks([0, 0.1, 0.2])
letter(axc, 'c', dx=-0.46, dy=1.0)

# (d) JT among burial samples across FEAST variants
yy = np.arange(len(JT))[::-1]
for yi, r in zip(yy, JT):
    axd.plot(r['p_perm_two_sided'], yi, 'o', ms=5, color=COL['burial'], zorder=3)
    axd.text(r['p_perm_two_sided'] * 1.18, yi + 0.18, f"{r['p_perm_two_sided']:.4f}", fontsize=5.8, va='bottom', ha='left')
    if 'beta_glmm' in r:
        axd.plot(r['beta_glmm']['p'], yi, 's', ms=4.5, mfc='white', mec='black', mew=0.7, zorder=3)
        axd.plot(r['zi_beta_glmm']['p'], yi, '^', ms=4.5, mfc='white', mec='black', mew=0.7, zorder=3)
axd.axvline(0.05, color='0.4', ls='--', lw=0.6, zorder=1)
axd.set_xscale('log'); axd.set_xlim(0.008, 1.0); axd.set_ylim(-0.6, len(JT) - 0.3)
axd.set_xticks([0.01, 0.05, 0.1, 0.5, 1]); axd.set_xticklabels(['0.01', '0.05', '0.1', '0.5', '1'])
SHORT = {'primary_r500': 'Depth-matched (primary)', 'default_coverage': 'FEAST default coverage',
         'noHumanEMP_r500': 'EMP human-host studies\nand blanks excluded', 'contam_r500': 'Blank- and mock-derived\nASVs removed',
         'burialmatched': 'Burial-matched\nsource depths'}
lab = [SHORT[r['variant']] + ('' if r['n'] == '15/31/26' else f"\n(n = {r['n'].replace('/', ', ')})") for r in JT]
axd.set_yticks(yy); axd.set_yticklabels(lab, fontsize=6.3, linespacing=1.1)
axd.set_xlabel('Two-sided p (log scale)')
h = [Line2D([], [], marker='o', ls='', color=COL['burial'], ms=5, label='JT, permutation'),
     Line2D([], [], marker='s', ls='', mfc='white', mec='black', ms=4.5, label='Beta GLMM'),
     Line2D([], [], marker='^', ls='', mfc='white', mec='black', ms=4.5, label='ZI-beta GLMM')]
axd.legend(handles=h, loc='center right', ncol=1, fontsize=6.3, handlelength=1.0, borderaxespad=0.3)
letter(axd, 'd', dx=-0.40, dy=1.0)
rep = save(fig, 'af5_sensitivity'); print(rep)
json.dump(N, open(f'figures/out/af5_sensitivity_numbers.json', 'w'), indent=1, default=str)
