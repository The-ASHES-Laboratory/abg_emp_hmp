import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); from figstyle_nyabg import *; setup()
"""Additional file 4 (stem af4_sweeps): prevalence filter sweep and rarefaction depth sweep of the taxon source assignment.
Inputs: R/sensitivity/prevalence_sweep.tsv, R/sensitivity/rarefaction_sweep.tsv (scripts 15_prevalence_sweep.py and
14_rarefaction_sweep.py, Hazel job abg10_ord 1813261); cohort totals from R/sequencing_depth/depth_distribution.tsv.
Retention counts are cross-checked against the job log R/../logs/abg10_ord_1813261.log.
Assignment rule (both sweeps): taxa with prevalence >= threshold in at least one cohort (burial, control, EMP, HMP);
EMP-associated if prev_EMP/(prev_EMP+prev_HMP) > 0.6, HMP-associated if prev_HMP/(prev_EMP+prev_HMP) > 0.6, otherwise ambiguous.
Rarefaction sweep: one multinomial rarefaction draw per depth (seed 42), 10% prevalence filter.
"""
import json, re
import pandas as pd
R = 'results'
LOG = sorted(__import__('glob').glob('logs/abg10_ord_*'))[-1]  # Slurm log of slurm/10_ordination.sbatch
P = pd.read_csv(f'{R}/sensitivity/prevalence_sweep.tsv', sep='\t')
S = pd.read_csv(f'{R}/sensitivity/rarefaction_sweep.tsv', sep='\t')
D = pd.read_csv(f'{R}/sequencing_depth/depth_distribution.tsv', sep='\t', dtype={'sample_id': str})
TOT = D.cohort.value_counts().to_dict()

# cross-check sweep table against job log
log = open(LOG).read()
blk = re.findall(r'Rarefaction depth = (\d+) \.\.\.\n(.*?)(?=\n\n)', log, flags=re.S)
for dep, body in blk:
    row = S[S.depth == int(dep)].iloc[0]
    for coh in ['burial', 'control', 'emp', 'hmp']:
        n = int(re.search(rf'{coh}: (\d+) samples remaining', body).group(1)); assert n == row[f'n_{coh}'], (dep, coh)
    m = re.search(r'(\d+) taxa -> (\d+) EMP, (\d+) HMP, (\d+) ambiguous', body).groups()
    assert list(map(int, m)) == [row.n_taxa, row.n_emp_assigned, row.n_hmp_assigned, row.n_ambiguous]
for th, body in re.findall(r'Running prevalence threshold = (\d+)% \.\.\.\n(.*?)\n', log):
    row = P[np.isclose(P.threshold, int(th) / 100)].iloc[0]
    assert list(map(int, re.search(r'(\d+) taxa -> (\d+) EMP, (\d+) HMP, (\d+) ambiguous', body).groups())) == [row.n_taxa, row.n_emp, row.n_hmp, row.n_ambiguous]
print('sweep tables match job log abg10_ord_1813261')

N = dict(cohort_totals={k: int(v) for k, v in TOT.items()},
         prevalence={f'{int(round(r.threshold*100))}%': dict(n_taxa=int(r.n_taxa), n_emp=int(r.n_emp), n_hmp=int(r.n_hmp), n_ambiguous=int(r.n_ambiguous),
                     pct_emp=round(100*r.n_emp/r.n_taxa, 1), pct_hmp=round(100*r.n_hmp/r.n_taxa, 1), pct_ambiguous=round(100*r.n_ambiguous/r.n_taxa, 1)) for r in P.itertuples()},
         rarefaction={int(r.depth): dict(n_burial=int(r.n_burial), n_control=int(r.n_control), n_emp=int(r.n_emp), n_hmp=int(r.n_hmp), n_taxa=int(r.n_taxa),
                      n_emp_assigned=int(r.n_emp_assigned), n_hmp_assigned=int(r.n_hmp_assigned), n_ambiguous=int(r.n_ambiguous),
                      pct_hmp=round(100*r.n_hmp_assigned/r.n_taxa, 1), pct_emp=round(100*r.n_emp_assigned/r.n_taxa, 1)) for r in S.itertuples()})
print(json.dumps(N, indent=1))

W, H = mm(170), mm(72)
fig, (axa, axb) = plt.subplots(1, 2, figsize=(W, H), gridspec_kw=dict(wspace=0.28, left=0.075, right=0.99, top=0.86, bottom=0.25))
cats = [('emp', 'EMP-associated', COL['emp']), ('hmp', 'HMP-associated', COL['hmp']), ('ambiguous', 'Ambiguous', COL['unknown'])]
bw = 0.26
# (a) prevalence sweep
x = np.arange(len(P))
for k, (c, lab, col) in enumerate(cats):
    v = P[f'n_{c}'].values
    axa.bar(x + (k-1)*bw, v, bw, color=col, edgecolor='black', linewidth=0.4, label=lab)
    for xi, vi in zip(x + (k-1)*bw, v): axa.text(xi, vi + 25, f'{vi:,}', ha='center', va='bottom', fontsize=5.5)
axa.set_xticks(x); axa.set_xticklabels([f'{int(round(t*100))}%\n(n = {n:,})' for t, n in zip(P.threshold, P.n_taxa)])
axa.set_xlabel('Prevalence threshold (taxa retained)'); axa.set_ylabel('Number of taxa')
axa.set_ylim(0, P.n_emp.max() * 1.12)
axa.legend(loc='upper right', fontsize=6.5, handlelength=1.0)
letter(axa, 'a', dx=-0.14, dy=1.06)
# (b) rarefaction sweep
x = np.arange(len(S))
for k, (c, lab, col) in enumerate(cats):
    v = S[f'n_{c}_assigned' if c != 'ambiguous' else 'n_ambiguous'].values
    axb.bar(x + (k-1)*bw, v, bw, color=col, edgecolor='black', linewidth=0.4, label=lab)
    for xi, vi in zip(x + (k-1)*bw, v): axb.text(xi, vi + 8, f'{vi}', ha='center', va='bottom', fontsize=5.5)
nb, nc = TOT['burial'], TOT['control']
axb.set_xticks(x); axb.set_xticklabels([f'{d:,}' for d in S.depth])
axb.set_xlabel('Rarefaction depth (reads per sample)'); axb.set_ylabel('Number of taxa')
ytop = S.n_emp_assigned.max() * 1.30; axb.set_ylim(0, ytop)
for xi, r in zip(x, S.itertuples()):
    axb.text(xi, ytop * 0.985, f'Burial {r.n_burial}/{nb}\nControl {r.n_control}/{nc}', ha='center', va='top', fontsize=5.5, color=COL['burial'], linespacing=1.15)
letter(axb, 'b', dx=-0.14, dy=1.06)
h, l = axb.get_legend_handles_labels(); axa.get_legend().remove()
fig.legend(h, l, loc='lower center', bbox_to_anchor=(0.53, 0.0), ncol=3, fontsize=7, handlelength=1.0, columnspacing=1.6)
rep = save(fig, 'af4_sweeps'); print(rep)
json.dump(N, open('figures/out/af4_sweeps_numbers.json', 'w'), indent=1)
