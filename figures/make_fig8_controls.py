import sys; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); from figstyle_nyabg import *; setup()
"""Fig. 8 (stem fig8_controls): negative, positive and mock controls and the blank/mock ASV filter.
Replaces the submitted decontamination figure (control-soil decontam vs HMP indicators, withdrawn as validation).
(a) per-library read depth; (b) touchdown water-blank composition and blank-ASV occurrence in soils;
(c) per-sample fraction of reads removed by the blank/mock ASV filter; (d) headline statistics before/after the filter.
Also writes the Additional file tables on negative controls (library read accounting and filtered-ASV table).
All inputs are read from the paths listed in figures_rev/inputs_manifest.md; every annotated number is computed here."""
import json, re
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

R = 'results'
R5 = 'results'
L = 'results/revision'
U = 'results/upstream'
OUT = 'figures/out'
STEM = 'fig8_controls'
BLANK_TD, BLANK_STD = 'Water-neg-TD-plateD', 'Water-neg-plateD'
MOCK_TD, MOCK_STD = 'Zymo-mock-pos-TD-plateD', 'Zymo-mock-pos-plateD'
POS_SOIL = 'PC'
AMBIG = ['MISC', 's_NA']            # libraries excluded for ambiguous metadata
IMPORT_MIN = 100                    # raw-read import filter (libraries below this were not denoised)
DROP_FRAC, DROP_DEPTH = 0.5, 100    # sample-exclusion rule in slurm/70_contam_sensitivity.sbatch
NUM = {}                            # every number shown in the figure, printed for audit

# ---------------- inputs ----------------
st = pd.read_csv(f'{U}/denoising_stats/stats.tsv', sep='\t', skiprows=[1], dtype={'sample-id': str})
st.columns = [c.strip() for c in st.columns]; st = st.set_index('sample-id')
raw = pd.read_csv(f'{U}/demux_summary/per-sample-fastq-counts.tsv', sep='\t', dtype={'sample ID': str}).set_index('sample ID')
idmap = pd.read_csv(f'{U}/manifest_id_map.tsv', sep='\t', dtype=str).set_index('manifest_id')
md = pd.read_csv(f'{R}/metadata_cohort_split.tsv', sep='\t', dtype=str)
bur = md.loc[md.cohort == 'burial', 'sample-id'].tolist(); ctl = md.loc[md.cohort == 'control', 'sample-id'].tolist()
ft = pd.read_csv(f'{U}/table_export/feature-table.tsv', sep='\t', skiprows=1, index_col=0); ft.columns = ft.columns.astype(str)
tax = pd.read_csv(f'{U}/taxonomy_export/taxonomy.tsv', sep='\t', index_col=0)
rm = pd.read_csv(f'{R}/contam_sensitivity/asvs_to_remove.tsv', sep='\t', index_col=0)
imp = pd.read_csv(f'{R}/contam_sensitivity/sample_impact.tsv', sep='\t', index_col=0); imp.index = imp.index.astype(str)
pw0 = pd.read_csv(f'{R}/ordination/aitchison_permanova_pairwise.tsv', sep='\t').set_index('comparison')
pw1 = pd.read_csv(f'{R}/contam_sensitivity/ordination/aitchison_permanova_pairwise.tsv', sep='\t').set_index('comparison')
fs = pd.read_csv(f'{L}/feast_all_runs_summary.tsv', sep='\t').set_index('run')
g0 = pd.read_csv(f'{R}/gradient_r500/gradient_tests.tsv', sep='\t')
g1 = pd.read_csv(f'{R5}/contam_sensitivity/gradient_r500_bodysite/gradient_tests.tsv', sep='\t')
seqs = {}
with open(f'{U}/rep_seqs_export/dna-sequences.fasta') as fh:
    k = None
    for line in fh:
        line = line.strip()
        if line.startswith('>'): k = line[1:].split()[0]; seqs[k] = ''
        elif k: seqs[k] += line

# ---------------- consistency checks ----------------
assert (ft.sum().round().astype(int) == st.loc[ft.columns, 'non-chimeric']).all(), 'ASV table != non-chimeric counts'
assert (st['input'] == raw.loc[st.index, 'forward sequence count']).all(), 'DADA2 input != demux counts'
assert set(bur + ctl) <= set(st.index) and set(imp.index) == set(bur + ctl)
assert ((imp.depth_after < DROP_DEPTH) | (imp.frac_lost > DROP_FRAC)).eq(imp.drop_sample).all(), 'drop rule mismatch'
removed_import = sorted(set(raw.index) - set(st.index)); assert BLANK_STD in removed_import
assert (raw.loc[removed_import, 'forward sequence count'] < IMPORT_MIN).all()

# ---------------- library table (Additional file) ----------------
def libtype(s):
    if s in bur: return 'Burial soil'
    if s in ctl: return 'Control soil (park)'
    if s == POS_SOIL: return 'Positive-control soil'
    if s == MOCK_TD: return 'ZymoBIOMICS mock, touchdown PCR'
    if s == MOCK_STD: return 'ZymoBIOMICS mock, standard PCR'
    if s == BLANK_TD: return 'Water blank, touchdown PCR'
    if s == BLANK_STD: return 'Water blank, standard PCR'
    if s in AMBIG: return 'Soil library, ambiguous metadata'
    return 'Soil library, failed import'
lib = pd.DataFrame(index=raw.index); lib.index.name = 'library_id'
lib['raw_file_prefix'] = idmap.reindex(lib.index)['raw_prefix']
lib['library_type'] = [libtype(s) for s in lib.index]
lib['raw_read_pairs'] = raw['forward sequence count']
for c in ['input', 'filtered', 'denoised', 'merged', 'non-chimeric']: lib[c] = st[c].reindex(lib.index).astype('Int64')
lib['pct_input_non_chimeric'] = st['percentage of input non-chimeric'].reindex(lib.index)
lib['blank_mock_filter_reads_removed'] = (imp.depth_before - imp.depth_after).reindex(lib.index).astype('Int64')
lib['blank_mock_filter_frac_removed'] = imp.frac_lost.reindex(lib.index).round(4)
def status(s):
    if s in removed_import: return f'removed at import (<{IMPORT_MIN} raw reads); not denoised'
    if s in bur: return 'analysed (primary); excluded from blank/mock-filtered sensitivity set' if imp.loc[s, 'drop_sample'] else 'analysed'
    if s in ctl: return 'analysed'
    if s in AMBIG: return 'denoised; excluded (ambiguous metadata)'
    return 'denoised; QC library (not in analysis)'
lib['status'] = [status(s) for s in lib.index]
order = {t: i for i, t in enumerate(['Burial soil', 'Control soil (park)', 'Positive-control soil', 'ZymoBIOMICS mock, touchdown PCR',
         'ZymoBIOMICS mock, standard PCR', 'Water blank, touchdown PCR', 'Water blank, standard PCR',
         'Soil library, ambiguous metadata', 'Soil library, failed import'])}
lib = lib.assign(_o=lib.library_type.map(order)).sort_values(['_o', 'library_id']).drop(columns='_o')
lib.to_csv(f'{OUT}/{STEM}_library_read_counts.tsv', sep='\t')

# ---------------- filtered ASV table (Additional file) ----------------
def lowest(t):
    parts = [p.strip() for p in str(t).split(';') if p.strip() and not p.strip().endswith('__')]
    g = [p for p in parts if p.startswith('g__')]
    if g: return g[0][3:]
    if not parts or parts[-1][1:3] != '__': return 'Unassigned'
    return parts[-1][3:] + f' ({ {"f": "family", "o": "order", "c": "class", "p": "phylum", "d": "domain"}[parts[-1][0]] })'
tax['label'] = tax.Taxon.map(lowest)
at = pd.DataFrame(index=rm.index); at.index.name = 'asv_id'
at['lowest_rank_taxon'] = tax.label.reindex(at.index); at['silva_taxonomy'] = tax.Taxon.reindex(at.index)
at['classifier_confidence'] = tax.Confidence.reindex(at.index).round(4)
at['flag'] = rm.reason.str.strip(';').str.replace(';', '+')
at['blank_TD_reads'] = ft.loc[at.index, BLANK_TD].astype(int)
at['mock_TD_reads'] = ft.loc[at.index, MOCK_TD].astype(int); at['mock_std_reads'] = ft.loc[at.index, MOCK_STD].astype(int)
at['burial_samples_present'] = (ft.loc[at.index, bur] > 0).sum(1); at['burial_reads'] = ft.loc[at.index, bur].sum(1).astype(int)
at['control_samples_present'] = (ft.loc[at.index, ctl] > 0).sum(1); at['control_reads'] = ft.loc[at.index, ctl].sum(1).astype(int)
at['positive_control_soil_reads'] = ft.loc[at.index, POS_SOIL].astype(int)
at['sequence'] = [seqs.get(a, '') for a in at.index]
assert (at.blank_TD_reads + at.mock_TD_reads + at.mock_std_reads).eq(rm.blank_reads + rm.mock_reads).all()
assert (at.burial_samples_present + at.control_samples_present).eq(rm.soil_samples_present).all()
at = at.sort_values(['blank_TD_reads', 'mock_TD_reads'], ascending=False)
at.to_csv(f'{OUT}/{STEM}_filtered_asvs.tsv', sep='\t')

# ---------------- numbers ----------------
nc = st['non-chimeric']
NUM['n_libraries_imported'] = len(raw); NUM['n_libraries_denoised'] = len(st); NUM['n_removed_import'] = len(removed_import)
NUM['removed_import_ids'] = removed_import
NUM['n_burial'] = len(bur); NUM['n_control'] = len(ctl)
NUM['burial_median_nonchim'] = float(nc[bur].median())
NUM['burial_iqr_nonchim'] = [float(nc[bur].quantile(.25)), float(nc[bur].quantile(.75))]
NUM['burial_range_nonchim'] = [int(nc[bur].min()), int(nc[bur].max())]
NUM['control_median_nonchim'] = float(nc[ctl].median())
NUM['blank_TD_nonchim'] = int(nc[BLANK_TD]); NUM['blank_TD_raw'] = int(raw.loc[BLANK_TD, 'forward sequence count'])
NUM['blank_STD_raw'] = int(raw.loc[BLANK_STD, 'forward sequence count'])
NUM['blank_TD_pct_of_burial_median'] = round(100 * nc[BLANK_TD] / nc[bur].median(), 1)
NUM['blank_TD_burial_percentile'] = round(100 * (nc[bur] < nc[BLANK_TD]).mean(), 1)
NUM['mock_TD_nonchim'] = int(nc[MOCK_TD]); NUM['mock_STD_nonchim'] = int(nc[MOCK_STD]); NUM['pos_soil_nonchim'] = int(nc[POS_SOIL])
bl = ft[BLANK_TD][ft[BLANK_TD] > 0]
gen = (bl.groupby(tax.label.reindex(bl.index)).sum() / bl.sum() * 100).sort_values(ascending=False)
NUM['blank_genus_pct'] = gen.round(1).to_dict(); NUM['n_blank_asvs'] = int(len(bl))
bA = at.loc[bl.index]
NUM['n_blank_asvs_in_burial'] = int((bA.burial_samples_present > 0).sum()); NUM['n_blank_asvs_in_control'] = int((bA.control_samples_present > 0).sum())
absent = bA.index[(bA.burial_samples_present == 0) & (bA.control_samples_present == 0)]
NUM['n_blank_asvs_absent_from_soils'] = int(len(absent)); NUM['pct_blank_reads_absent_from_soils'] = round(100 * bl[absent].sum() / bl.sum(), 1)
NUM['n_blank_asvs_also_in_mock'] = int(((bA.mock_TD_reads + bA.mock_std_reads) > 0).sum())
NUM['n_filtered_asvs_in_pos_soil'] = int((at.positive_control_soil_reads > 0).sum())
NUM['top_burial_removed_asv'] = {at.burial_reads.idxmax(): [at.loc[at.burial_reads.idxmax(), 'lowest_rank_taxon'], int(at.burial_reads.max()), int(at.loc[at.burial_reads.idxmax(), 'burial_samples_present'])]}
NUM['n_filtered_asvs'] = len(at); NUM['n_blank_only'] = int((at.flag == 'blank').sum()); NUM['n_mock_only'] = int((at.flag == 'mock').sum())
NUM['n_blank_and_mock'] = int((at.flag == 'blank+mock').sum())
NUM['pct_burial_reads_removed_total'] = round(100 * (imp.loc[bur, 'depth_before'] - imp.loc[bur, 'depth_after']).sum() / imp.loc[bur, 'depth_before'].sum(), 2)
NUM['burial_frac_removed_median_pct'] = round(100 * imp.loc[bur, 'frac_lost'].median(), 1)
NUM['n_burial_zero_removed'] = int((imp.loc[bur, 'frac_lost'] == 0).sum())
NUM['control_frac_removed_max_pct'] = round(100 * imp.loc[ctl, 'frac_lost'].max(), 1)
NUM['dropped'] = {s: round(100 * imp.loc[s, 'frac_lost'], 1) for s in imp.index[imp.drop_sample]}
NUM['n_burial_after'] = int(len(bur) - imp.loc[bur, 'drop_sample'].sum())
NUM['next_highest_removed'] = {imp.loc[bur][~imp.loc[bur].drop_sample].frac_lost.idxmax(): round(100 * imp.loc[bur][~imp.loc[bur].drop_sample].frac_lost.max(), 1)}
NUM['R2_before'] = float(pw0.loc['burial_vs_control', 'R2_mean']); NUM['R2_after'] = float(pw1.loc['burial_vs_control', 'R2_mean'])
NUM['R2_pBH_before'] = float(pw0.loc['burial_vs_control', 'p_BH']); NUM['R2_pBH_after'] = float(pw1.loc['burial_vs_control', 'p_BH'])
NUM['HMP_mean_before_pct'] = 100 * float(fs.loc['r500_primary', 'HMP_mean']); NUM['HMP_mean_after_pct'] = 100 * float(fs.loc['contam_r500', 'HMP_mean'])
NUM['HMP_n_before'] = int(fs.loc['r500_primary', 'n_burial']); NUM['HMP_n_after'] = int(fs.loc['contam_r500', 'n_burial'])
jt0 = g0[g0.test.str.startswith('JT decreasing trend, within burial')].iloc[0]; jt1 = g1[g1.test.str.startswith('JT decreasing trend, within burial')].iloc[0]
NUM['JT_p_before'] = float(jt0.p_perm_two_sided); NUM['JT_p_after'] = float(jt1.p_perm_two_sided)
NUM['JT_n_before'] = jt0.n; NUM['JT_n_after'] = jt1.n
print(json.dumps(NUM, indent=1, default=str))

# ---------------- figure ----------------
GREY, DGREY = '#8C8C8C', '#4D4D4D'
fig = plt.figure(figsize=(mm(170), mm(124)))
outer = GridSpec(2, 1, figure=fig, height_ratios=[1.05, 1], hspace=0.42, left=0.075, right=0.975, top=0.955, bottom=0.075)
top = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[1.45, 0.38, 0.48], wspace=0.95)
bot = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], width_ratios=[1.0, 0.95], wspace=0.40)
rng = np.random.default_rng(8)

# (a) read depth per library
ax = fig.add_subplot(top[0])
cats = [('Burial', bur, COL['burial'], 'o', True, 'nc'), ('Control', ctl, COL['control'], 'o', True, 'nc'),
        ('PC\nsoil', [POS_SOIL], DGREY, 'D', True, 'nc'),
        ('Mock\nTD', [MOCK_TD], DGREY, 's', True, 'nc'), ('Mock\nstd', [MOCK_STD], DGREY, 's', True, 'nc'),
        ('Blank\nTD', [BLANK_TD], ALARM, 'o', True, 'nc'), ('Blank\nstd', [BLANK_STD], ALARM, 'o', False, 'raw'),
        ('Ambig.', AMBIG, GREY, 'o', True, 'nc'),
        ('Low\nreads', [s for s in removed_import if s != BLANK_STD], GREY, 'o', False, 'raw')]
for i, (lab, ids, c, mk, filled, kind) in enumerate(cats):
    y = (nc.reindex(ids) if kind == 'nc' else raw.loc[ids, 'forward sequence count']).astype(float).values
    x = i + (rng.uniform(-0.28, 0.28, len(y)) if len(y) > 2 else np.linspace(-0.12, 0.12, len(y)) if len(y) == 2 else np.zeros(1))
    ax.scatter(x, y, s=16 if len(y) > 2 else 24, marker=mk, facecolor=c if filled else 'white', edgecolor=c if not filled else 'white',
               linewidth=0.8 if not filled else 0.3, alpha=0.9 if len(y) > 2 else 1, zorder=3)
    if len(ids) > 2:
        ax.hlines(np.median(y), i - 0.36, i + 0.36, color='black', lw=1.0, zorder=4)
ax.set_yscale('log'); ax.set_ylim(5, 6e4); ax.set_yticks([10, 100, 1000, 10000]); ax.minorticks_off()
ax.axhline(NUM['burial_median_nonchim'], color=COL['burial'], ls='--', lw=0.7, zorder=1)
ax.axhline(IMPORT_MIN, color=GREY, ls=':', lw=0.7, zorder=1)
ax.text(len(cats) - 0.45, IMPORT_MIN * 1.12, f'Import filter ({IMPORT_MIN} reads)', fontsize=6, color=DGREY, ha='right', va='bottom')
ax.text(len(cats) - 0.45, NUM['burial_median_nonchim'] / 1.3, f'Burial median\n{NUM["burial_median_nonchim"]:,.0f}', fontsize=6, color=COL['burial'], ha='right', va='top')
ib = [c[0] for c in cats].index('Blank\nTD')
ax.annotate(f'{NUM["blank_TD_nonchim"]:,}', (ib, NUM['blank_TD_nonchim']), xytext=(ib - 0.3, NUM['blank_TD_nonchim'] * 0.33),
            fontsize=6, color=ALARM, ha='right', va='center', arrowprops=dict(arrowstyle='-', lw=0.5, color=ALARM))
iw = ib + 1
ax.annotate(f'{NUM["blank_STD_raw"]} raw', (iw, NUM['blank_STD_raw']), xytext=(iw + 0.22, NUM['blank_STD_raw'] * 0.55),
            fontsize=6, color=ALARM, ha='left', va='center')
nlab = {'Burial': len(bur), 'Control': len(ctl), 'Ambig.': len(AMBIG), 'Low\nreads': len(removed_import) - 1}
ax.set_xticks(range(len(cats)))
ax.set_xticklabels([c[0] + (f'\nn = {nlab[c[0]]}' if c[0] in nlab else '') for c in cats], fontsize=5.6)
ax.set_xlim(-0.6, len(cats) - 0.4)
ax.set_ylabel('Reads per library')
h = [Line2D([], [], marker='o', ls='', mfc=DGREY, mec='white', ms=4, label='Non-chimeric reads (DADA2)'),
     Line2D([], [], marker='o', ls='', mfc='white', mec=DGREY, ms=4, label='Raw input, not denoised')]
ax.legend(handles=h, loc='upper left', fontsize=6, handletextpad=0.2, borderaxespad=0.1, ncol=2, columnspacing=0.8)
letter(ax, 'a', dx=-0.17)

# (b) touchdown-blank composition and blank-ASV occurrence in soils
axg = fig.add_subplot(top[1]); axa = fig.add_subplot(top[2])
gl = list(gen.index)[::-1]
axg.barh(range(len(gl)), gen[gl].values, color=ALARM, height=0.7)
for k, gname in enumerate(gl):
    axg.text(gen[gname] + 2, k, f'{gen[gname]:.1f}', fontsize=6, va='center')
axg.set_yticks(range(len(gl))); axg.set_yticklabels([g.replace(' (family)', '\n(family)') for g in gl], fontsize=6)
for t, gname in zip(axg.get_yticklabels(), gl):
    if '(' not in gname: t.set_style('italic')
axg.set_xlim(0, 100); axg.set_xticks([0, 50, 100]); axg.set_xlabel('Blank reads (%)', fontsize=6.5)
axg.set_title(f'Genus, n = {NUM["n_blank_asvs"]} ASVs', fontsize=7)
letter(axg, 'b', dx=-0.95)
asv = bA.copy(); asv['g'] = asv.lowest_rank_taxon
asv['lab'] = ''; cnt = {}
for a in asv.index:
    g = asv.loc[a, 'g'].replace(' (family)', ' (f.)'); cnt[g] = cnt.get(g, 0) + 1; asv.loc[a, 'lab'] = f'{g} {cnt[g]}'
asv = asv.iloc[::-1]
yy = np.arange(len(asv))
axa.barh(yy + 0.18, asv.burial_samples_present, height=0.36, color=COL['burial'], label=f'Burial (of {len(bur)})')
axa.barh(yy - 0.18, asv.control_samples_present, height=0.36, color=COL['control'], label=f'Control (of {len(ctl)})')
axa.scatter(np.zeros(len(asv)), yy - 0.18, marker='|', color=COL['control'], s=12, lw=1.0, zorder=3)
for k, a in enumerate(asv.index):
    axa.text(asv.loc[a, 'burial_samples_present'] + 0.6, k + 0.18, str(asv.loc[a, 'burial_samples_present']), fontsize=5.5, va='center')
axa.set_yticks(yy); axa.set_yticklabels([l + (' *' if asv.loc[a, 'mock_TD_reads'] + asv.loc[a, 'mock_std_reads'] > 0 else '') for l, a in zip(asv.lab, asv.index)], fontsize=5.6)
for t in axa.get_yticklabels():
    if '(f.)' not in t.get_text(): t.set_style('italic')
axa.set_xlim(0, 30); axa.set_xticks([0, 10, 20, 30]); axa.set_ylim(-0.7, len(asv) - 0.3)
axa.set_xlabel('Samples containing ASV', fontsize=6.5)
axa.set_title('Blank ASVs in soils', fontsize=7)
axa.legend(loc='lower right', fontsize=5.8, handlelength=0.9, handletextpad=0.3, borderaxespad=0.1)

# (c) per-sample fraction of reads removed by the blank/mock ASV filter
ax = fig.add_subplot(bot[0])
sb = imp.loc[bur].sort_values('frac_lost', ascending=False)
xs = np.arange(len(sb))
ax.bar(xs, 100 * sb.frac_lost, width=0.8, color=[ALARM if d else '#E58C8C' for d in sb.drop_sample], edgecolor='none')
xc = len(sb) + 2 + np.arange(len(ctl))
ax.bar(xc, 100 * imp.loc[ctl, 'frac_lost'], width=0.8, color=COL['control'])
ax.scatter(xc, np.zeros(len(ctl)) + 1.2, marker='o', s=6, color=COL['control'], zorder=3)
ax.axhline(100 * DROP_FRAC, color=DGREY, ls='--', lw=0.7)
ax.text(len(sb) + len(ctl) + 2.5, 100 * DROP_FRAC + 2, f'Exclusion threshold ({DROP_FRAC:.0%})', fontsize=6, ha='right', va='bottom', color=DGREY)
for s in sb.index[sb.drop_sample]:
    k = list(sb.index).index(s)
    ax.text(k + 1.2, 100 * sb.loc[s, 'frac_lost'] - 1, f'{s}: {100*sb.loc[s,"frac_lost"]:.1f}%', fontsize=6, color=ALARM, va='top', ha='left')
ax.text(len(sb) * 0.55, 66,
        f'Median burial loss {NUM["burial_frac_removed_median_pct"]:.1f}%\n{NUM["n_burial_zero_removed"]} of {len(bur)} burial samples unaffected\n'
        f'{NUM["pct_burial_reads_removed_total"]:.2f}% of all burial reads removed', fontsize=6, va='bottom', ha='center')
ax.set_xticks([(len(sb) - 1) / 2, xc.mean()]); ax.set_xticklabels([f'Burial samples, ranked (n = {len(bur)})', f'Control\n(n = {len(ctl)})'], fontsize=6.5)
ax.tick_params(axis='x', length=0)
ax.set_xlim(-1, xc[-1] + 1); ax.set_ylim(0, 105)
ax.set_ylabel('Reads removed (%)')
letter(ax, 'c', dx=-0.17)

# (d) headline statistics before / after the filter
sub = GridSpecFromSubplotSpec(3, 1, subplot_spec=bot[1], hspace=1.55)
rows = [('Burial vs control PERMANOVA R² (Aitchison)', NUM['R2_before'], NUM['R2_after'], (0.08, 0.12), '{:.3f}',
         f'n = {len(bur)}/{len(ctl)}', f'n = {NUM["n_burial_after"]}/{len(ctl)}'),
        ('Mean HMP attribution, burial (FEAST, %)', NUM['HMP_mean_before_pct'], NUM['HMP_mean_after_pct'], (2, 6), '{:.1f}',
         f'n = {NUM["HMP_n_before"]}', f'n = {NUM["HMP_n_after"]}'),
        ('JT trend among burial samples, two-sided permutation p', NUM['JT_p_before'], NUM['JT_p_after'], (0, 0.06), '{:.3f}',
         f'n = {NUM["JT_n_before"]}', f'n = {NUM["JT_n_after"]}')]
for k, (title, v0, v1, xl, fmt, n0, n1) in enumerate(rows):
    a = fig.add_subplot(sub[k])
    a.plot([v0, v1], [1, 0], color=GREY, lw=0.6, zorder=1)
    a.scatter([v0], [1], s=26, color='black', zorder=3); a.scatter([v1], [0], s=26, facecolor='white', edgecolor='black', lw=0.9, zorder=3)
    a.text(v0 + (xl[1] - xl[0]) * 0.03, 1, f'{fmt.format(v0)}  ({n0})', fontsize=6, va='center')
    a.text(v1 + (xl[1] - xl[0]) * 0.03, 0, f'{fmt.format(v1)}  ({n1})', fontsize=6, va='center')
    a.set_yticks([1, 0]); a.set_yticklabels(['All ASVs', 'Filtered'], fontsize=6)
    a.set_ylim(-0.6, 1.6); a.set_xlim(*xl); a.locator_params(axis='x', nbins=5)
    if 'p' in title.split()[-1]: a.axvline(0.05, color=GREY, ls=':', lw=0.6)
    a.set_title(title, fontsize=6.5, pad=3)
    if k == 0: letter(a, 'd', dx=-0.26, dy=1.25)

rep = save(fig, STEM)
print('overlap report:', rep)
json.dump(NUM, open(f'{OUT}/{STEM}_numbers.json', 'w'), indent=1, default=str)
