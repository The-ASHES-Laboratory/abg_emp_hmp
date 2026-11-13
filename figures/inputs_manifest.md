# Full-depth inputs for revision figures (paths relative to the project root)
R  = results   (full-depth Hazel outputs, final)
R2 = results   (depth-matched merged table merged_L6genus_r500.tsv; NYABG sinks unrarefied)
R5 = results   (job 72 body-site on contamination-filtered table; burialmatched; r500 stability)
L  = results/revision                  (local recomputations)
U  = results/upstream (full-depth ASV table_export/feature-table.tsv, taxonomy_export/taxonomy.tsv, denoising_stats/stats.tsv)
Metadata: R/metadata_cohort_split.tsv (sample-id, cohort, cohort_subsite); gradient sample table R/gradient_r500/gradient_sample_table.tsv (sample_id, hmp, body_region, level, burial_id, depth, cohort)
Alpha: R/alpha_diversity/alpha_values.tsv; coverage richness R/alpha_diversity/coverage_richness_per_sample.tsv, _tests.tsv, _summary.tsv
Ordination: R/ordination/pcoa_coordinates.tsv (Bray-Curtis), aitchison_pcoa_coordinates.tsv, table2_permanova_permdisp.tsv, balanced_permanova.tsv; PCoA variance (median replicate) Aitchison PC1 23.8%, PC2 9.0% (log abg10_ord); Bray-Curtis % from R/../logs/abg10_ord_1813261.log
FEAST primary (depth-matched, 500-read sources): R/feast_depth_matched/r500/{burial,control}_source_props.tsv, burial_source_props_by_bodysite.tsv, burial_source_props_by_emptype.tsv; stability R5/feast_depth_matched/r500/stability/replicate_[1-5]_burial.tsv
FEAST sensitivity: default 11-read R/feast_split/; EMP-excluded R/feast_depth_matched/r500_noHumanEMP/; contamination-filtered R/contam_sensitivity/feast_r500/ and R5/contam_sensitivity/feast_r500_bodysite/; burial-matched R5/feast_depth_matched/burialmatched/; summary L/feast_all_runs_summary.tsv
Gradient: R/gradient_r500/gradient_tests.tsv, gradient_group_summary.tsv; beta GLMM L/gradient_beta_glmm.tsv; contamination-filtered R5/contam_sensitivity/gradient_r500_bodysite/gradient_tests.tsv
Networks: R/networks/{burial,control}/overall/source_network_taxa.tsv, source_network_summary.tsv, source_network_edges.tsv
ISA: R/sensitivity/isa_results.tsv (filter p.value < 0.05), isa_summary.tsv
DA (oriented, wi.eBH): L/diff_abun/da_consensus_oriented.tsv, da_consensus_oriented_summary.tsv, table1_consensus_taxa.tsv; per-method R/diff_abun/aldex2/*.tsv, R/diff_abun/ancombc2_split/*.tsv (ANCOM-BC2 lfc natural log; lfc_cohort<g> column names the non-reference group; ALDEx2 effect = alphabetically later minus earlier)
Classifier: R/classifier/{burial,control}_predictions.csv (sample_id, sample_type, model, class, probability); R/classifier_null/classifier_null_distribution.tsv, classifier_null_summary.tsv
Contamination: R/contam_sensitivity/asvs_to_remove.tsv, sample_impact.tsv; blank-negative decontam L/decontam_blank_negative_fd.tsv; control-soil decontam R/qc/decontam_results.tsv
Depth: R/sequencing_depth/depth_distribution.tsv; U/denoising_stats/stats.tsv
Within burial: R/within_burial/ (built on 11-read FEAST; recompute from r500 if used)
Paths are relative to the project root (results/, logs/, data/); revision-specific inputs are written to results/revision/ by scripts 36 to 44.
