# Key parameters

Analytical parameters that most influence pipeline output, as set in the code used for the revised manuscript (v1.1.0).

## Denoising and taxonomy

| Parameter | Value | Where set |
|---|---|---|
| DADA2 truncation (forward/reverse) | 240/220 nt | `slurm/05_fulldepth_denoise.sbatch` |
| Maximum expected errors; truncation quality | 2/2; 2 | `slurm/05_fulldepth_denoise.sbatch` |
| Read subsampling before denoising | none (full depth); 1.0.x used 30% | `slurm/05_fulldepth_denoise.sbatch` |
| Classifier | QIIME 2 naive Bayes, SILVA 138 99% | `slurm/05_fulldepth_denoise.sbatch` |
| Minimum reads to import a library | 100 | `slurm/05_fulldepth_denoise.sbatch` |

## Cohort subsampling

| Parameter | Value | Where set |
|---|---|---|
| Ordination, PERMANOVA, PERMDISP | 500 EMP and 500 HMP (stratified) plus all NYABG; 5 replicates (seeds 42 to 46) | `03`, `04`, `30` |
| Balanced PERMANOVA | 100 draws of 70 EMP soil and 70 HMP samples plus all NYABG | `30_permdisp_balanced.py` |
| Indicator species | 500 EMP and 500 HMP (random) plus all NYABG; taxa in at least 5% of samples | `05_indicator_species.R` |
| Differential abundance | 500-sample EMP or HMP subsample (seed 42) | `08_ancombc2.R`, `09_aldex2.R` |
| FEAST sources | 100 EMP (by habitat) and 100 HMP (by body site) per sink; 1,000 EM iterations; 5 stability replicates | `07_feast_source_tracking_v2.R` |
| FEAST source depth | every source rarefied to 500 reads without replacement (primary) | `34_make_depth_matched_sources.py` |

## Statistical tests

| Parameter | Value | Where set |
|---|---|---|
| Alpha diversity | unrarefied genus counts; Kruskal-Wallis, two-sided Mann-Whitney U, BH | `02_alpha_diversity.py` |
| Coverage-standardized richness | interpolation to 90% and 95% sample coverage | `31_coverage_richness.py` |
| PERMANOVA and PERMDISP permutations | 999 | `03`, `04`, `30`, `36`, `38` |
| Differential abundance consensus | ANCOM-BC2 q < 0.05 (no pseudo-count sensitivity filter) and ALDEx2 wi.eBH < 0.05; ALDEx2 effects oriented to the first-named group | `10b_da_consensus_oriented.py` |
| Proximity trend | Jonckheere-Terpstra among burial samples, two-sided, 9,999 permutations, seed 42 | `32_gradient_tests.py`, `44` |
| Mixed models | beta and zero-inflated beta (glmmTMB), linear (lme4); proximity rank, log10 depth, burial random intercept | `33`, `37` |

## Decontamination

| Parameter | Value | Where set |
|---|---|---|
| decontam negative set | touchdown-PCR water control (Water-neg-TD-plateD) | `40_decontam_blank.R` |
| decontam thresholds | 0.1 and 0.5 (prevalence method) | `40_decontam_blank.R` |
| Contamination sensitivity | remove every ASV in the water control or mock communities; drop samples losing more than 50% of reads or keeping fewer than 100 | `slurm/70_contam_sensitivity.sbatch` |

## Classification

| Parameter | Value | Where set |
|---|---|---|
| Random forest | 200 trees | `11_classify_samples.py` |
| L1-regularized logistic regression | one-vs-rest, liblinear, C = 1 (no tuning) | `11_classify_samples.py`, `22_classifier_null.py` |
| Training data | all EMP samples (EMPO-3) or all HMP samples (18 subsites); five aggregated HMP sites for the null model | `11`, `22` |
| Held-out evaluation | stratified five-fold cross-validation (seed 20251101) | `11c_classifier_cv.py` |
| Null distribution | 500 EMP samples classified by the five-site HMP model (seed 42) | `22_classifier_null.py` |
