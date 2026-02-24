# 08_ancombc2.R
# By Carter Clinton, Ph.D.
# ANCOMBC2 Differential Abundance
#
# Runs ANCOMBC2 for 5 pairwise comparisons at L5 (family) and L6 (genus) levels.
# Uses subsampling to address extreme sample-size imbalance.
#
# Comparisons:
#   1. burial vs EMP
#   2. burial vs HMP
#   3. control vs EMP
#   4. control vs HMP
#   5. burial vs control
#
# Outputs:
#   results/diff_abun/ancombc2_split/ancombc2_{comparison}_{level}.tsv
#
# Error handling:
#   - Skips comparisons whose output file already exists (use --force to re-run all)
#   - Falls back to struc_zero=FALSE if TRUE triggers errors
#   - Lowers lib_cut for small-group comparisons (n < 50)
#   - Writes empty results file with error note on unrecoverable failure

suppressPackageStartupMessages({
  library(ANCOMBC)
  library(TreeSummarizedExperiment)
  library(phyloseq)
  library(argparse)
  library(data.table)
})

parser <- ArgumentParser(description = "Run ANCOMBC2 differential abundance")
parser$add_argument("--table", default = "results/compositional/merged_L6genus.tsv")
parser$add_argument("--metadata", default = "results/metadata_cohort_split.tsv")
parser$add_argument("--n-subsample", type = "integer", default = 500L)
parser$add_argument("--seed", type = "integer", default = 42L)
parser$add_argument("--out-dir", default = "results/diff_abun/ancombc2_split/")
parser$add_argument("--force", action = "store_true", default = FALSE,
                    help = "Re-run all comparisons even if output exists")
args <- parser$parse_args()

dir.create(args$out_dir, recursive = TRUE, showWarnings = FALSE)

# ── Load data ─────────────────────────────────────────────────────────────────
cat("Loading genus table...\n")
lines <- readLines(args$table, n = 5)
skip <- 0
for (i in seq_along(lines)) {
  if (grepl("^#", lines[i]) && !grepl("^#OTU|^#Feature", lines[i])) {
    skip <- skip + 1
  } else {
    break
  }
}
counts <- fread(args$table, sep = "\t", skip = skip, header = TRUE, check.names = FALSE)
taxa_col <- names(counts)[1]
taxa_names <- counts[[taxa_col]]
counts[[taxa_col]] <- NULL
if (tolower(tail(names(counts), 1)) == "taxonomy") {
  counts[[tail(names(counts), 1)]] <- NULL
}
count_mat <- as.matrix(counts)
rownames(count_mat) <- taxa_names
count_mat[is.na(count_mat)] <- 0
count_mat <- round(count_mat)

cat("Loading metadata...\n")
meta <- fread(args$metadata, sep = "\t", header = TRUE)
setnames(meta, 1, "sample_id")
meta <- as.data.frame(meta)
rownames(meta) <- meta$sample_id

# ── Helper: subsample ────────────────────────────────────────────────────────
subsample_cohort <- function(meta_df, cohort_val, n_max, seed) {
  sub <- meta_df[meta_df$cohort == cohort_val, ]
  if (nrow(sub) <= n_max) return(rownames(sub))
  set.seed(seed)
  sample(rownames(sub), n_max)
}

# ── Helper: write empty result placeholder on failure ────────────────────────
write_empty_result <- function(outfile, group1, group2, level_label, error_msg) {
  header <- "taxon\tlfc_cohort\tse_cohort\tW_cohort\tp_cohort\tq_cohort\tdiff_cohort\terror_note"
  line <- sprintf("NO_RESULTS\tNA\tNA\tNA\tNA\tNA\tNA\t%s vs %s %s failed: %s",
                  group1, group2, level_label, gsub("\t", " ", error_msg))
  writeLines(c(header, line), outfile)
  cat(sprintf("  Wrote placeholder %s (ANCOMBC2 failed: %s)\n", basename(outfile), error_msg))
}

# ── Run ANCOMBC2 for one comparison with fallback strategies ─────────────────
run_ancombc2_comparison <- function(count_mat, meta_df, group1, group2,
                                     n_sub, seed, level_label) {
  # Get sample IDs
  g1_ids <- subsample_cohort(meta_df, group1, n_sub, seed)
  g2_ids <- subsample_cohort(meta_df, group2, n_sub, seed)
  all_ids <- c(g1_ids, g2_ids)
  valid_ids <- all_ids[all_ids %in% colnames(count_mat)]

  if (length(valid_ids) < 4) {
    cat(sprintf("  Skipping %s vs %s: too few samples (%d)\n", group1, group2, length(valid_ids)))
    return(list(result = NULL, error = "too few samples"))
  }

  sub_counts <- count_mat[, valid_ids, drop = FALSE]
  sub_meta <- meta_df[valid_ids, , drop = FALSE]
  sub_meta$cohort <- factor(sub_meta$cohort, levels = c(group2, group1))

  # Filter taxa: prevalence >= 10% in at least one group
  g1_prev <- rowMeans(sub_counts[, sub_meta$cohort == group1, drop = FALSE] > 0)
  g2_prev <- rowMeans(sub_counts[, sub_meta$cohort == group2, drop = FALSE] > 0)
  keep <- (g1_prev >= 0.10) | (g2_prev >= 0.10)
  sub_counts <- sub_counts[keep, , drop = FALSE]

  if (nrow(sub_counts) < 5) {
    cat(sprintf("  Skipping %s vs %s: too few taxa after filtering (%d)\n",
                group1, group2, nrow(sub_counts)))
    return(list(result = NULL, error = "too few taxa after filtering"))
  }

  n_g1 <- sum(sub_meta$cohort == group1)
  n_g2 <- sum(sub_meta$cohort == group2)
  cat(sprintf("  %s vs %s (%s): %d taxa, %d samples (%d + %d)\n",
              group1, group2, level_label, nrow(sub_counts), ncol(sub_counts),
              n_g1, n_g2))

  # Build TreeSummarizedExperiment
  otu <- otu_table(sub_counts, taxa_are_rows = TRUE)
  sdata <- sample_data(sub_meta)
  ps <- phyloseq(otu, sdata)
  tse <- mia::makeTreeSummarizedExperimentFromPhyloseq(ps)

  # Adaptive lib_cut: lower for small groups to avoid dropping all samples
  min_group_n <- min(n_g1, n_g2)
  adaptive_lib_cut <- if (min_group_n < 50) 100 else 1000

  # Strategy 1: struc_zero=TRUE, adaptive lib_cut
  out <- tryCatch({
    cat(sprintf("    Attempt 1: struc_zero=TRUE, lib_cut=%d\n", adaptive_lib_cut))
    result <- ancombc2(
      data = tse,
      fix_formula = "cohort",
      p_adj_method = "BH",
      alpha = 0.05,
      prv_cut = 0.10,
      lib_cut = adaptive_lib_cut,
      group = "cohort",
      struc_zero = TRUE,
      neg_lb = TRUE,
      pseudo_sens = TRUE
    )
    if (!is.null(result$res) && nrow(result$res) > 0) result$res else NULL
  }, error = function(e) {
    cat(sprintf("    Attempt 1 failed: %s\n", e$message))
    NULL
  })

  if (!is.null(out)) return(list(result = out, error = NULL))

  # Strategy 2: struc_zero=FALSE (avoids structural zero detection bug)
  out <- tryCatch({
    cat(sprintf("    Attempt 2: struc_zero=FALSE, lib_cut=%d\n", adaptive_lib_cut))
    result <- ancombc2(
      data = tse,
      fix_formula = "cohort",
      p_adj_method = "BH",
      alpha = 0.05,
      prv_cut = 0.10,
      lib_cut = adaptive_lib_cut,
      group = "cohort",
      struc_zero = FALSE,
      neg_lb = TRUE,
      pseudo_sens = TRUE
    )
    if (!is.null(result$res) && nrow(result$res) > 0) result$res else NULL
  }, error = function(e) {
    cat(sprintf("    Attempt 2 failed: %s\n", e$message))
    NULL
  })

  if (!is.null(out)) return(list(result = out, error = NULL))

  # Strategy 3: minimal settings (no struc_zero, no neg_lb, lib_cut=0)
  out <- tryCatch({
    cat("    Attempt 3: minimal settings (struc_zero=FALSE, neg_lb=FALSE, lib_cut=0)\n")
    result <- ancombc2(
      data = tse,
      fix_formula = "cohort",
      p_adj_method = "BH",
      alpha = 0.05,
      prv_cut = 0.10,
      lib_cut = 0,
      group = "cohort",
      struc_zero = FALSE,
      neg_lb = FALSE,
      pseudo_sens = FALSE
    )
    if (!is.null(result$res) && nrow(result$res) > 0) result$res else NULL
  }, error = function(e) {
    cat(sprintf("    Attempt 3 failed: %s\n", e$message))
    NULL
  })

  if (!is.null(out)) return(list(result = out, error = NULL))

  return(list(result = NULL, error = "all 3 attempts failed"))
}

# ── Define comparisons ───────────────────────────────────────────────────────
comparisons <- list(
  c("burial", "emp"),
  c("burial", "hmp"),
  c("control", "emp"),
  c("control", "hmp"),
  c("burial", "control")
)

# ── Run all comparisons at L6 (genus) ────────────────────────────────────────
cat("\n=== ANCOMBC2 at L6 (genus) ===\n")
for (comp in comparisons) {
  g1 <- comp[1]
  g2 <- comp[2]
  outfile <- file.path(args$out_dir, sprintf("ancombc2_%s_vs_%s_L6.tsv", g1, g2))

  # Skip if output already exists (unless --force)
  if (!args$force && file.exists(outfile) && file.info(outfile)$size > 100) {
    cat(sprintf("  %s vs %s (L6): output exists, skipping (use --force to re-run)\n", g1, g2))
    next
  }

  n_sub <- if (g1 %in% c("burial", "control") && g2 %in% c("burial", "control")) {
    10000L  # no subsampling for burial vs control
  } else {
    args$n_subsample
  }

  res <- run_ancombc2_comparison(count_mat, meta, g1, g2, n_sub, args$seed, "L6")
  if (!is.null(res$result)) {
    fwrite(res$result, outfile, sep = "\t")
    cat(sprintf("  Wrote %s (%d rows)\n", basename(outfile), nrow(res$result)))
  } else {
    write_empty_result(outfile, g1, g2, "L6", res$error)
  }
}

# ── Run all comparisons at L5 (family) ───────────────────────────────────────
# Aggregate to family level
cat("\n=== Aggregating to L5 (family) ===\n")
family_names <- sapply(strsplit(taxa_names, ";"), function(x) {
  fam <- grep("^f__", x, value = TRUE)
  if (length(fam) > 0) paste(x[1:which(x == fam[1])], collapse = ";") else paste(x, collapse = ";")
})
# Group by family
fam_groups <- split(seq_along(family_names), family_names)
count_mat_L5 <- do.call(rbind, lapply(fam_groups, function(idx) {
  if (length(idx) == 1) return(count_mat[idx, , drop = FALSE])
  colSums(count_mat[idx, , drop = FALSE])
}))
rownames(count_mat_L5) <- names(fam_groups)

cat(sprintf("  %d families\n", nrow(count_mat_L5)))

cat("\n=== ANCOMBC2 at L5 (family) ===\n")
for (comp in comparisons) {
  g1 <- comp[1]
  g2 <- comp[2]
  outfile <- file.path(args$out_dir, sprintf("ancombc2_%s_vs_%s_L5.tsv", g1, g2))

  # Skip if output already exists (unless --force)
  if (!args$force && file.exists(outfile) && file.info(outfile)$size > 100) {
    cat(sprintf("  %s vs %s (L5): output exists, skipping (use --force to re-run)\n", g1, g2))
    next
  }

  n_sub <- if (g1 %in% c("burial", "control") && g2 %in% c("burial", "control")) {
    10000L
  } else {
    args$n_subsample
  }

  res <- run_ancombc2_comparison(count_mat_L5, meta, g1, g2, n_sub, args$seed, "L5")
  if (!is.null(res$result)) {
    fwrite(res$result, outfile, sep = "\t")
    cat(sprintf("  Wrote %s (%d rows)\n", basename(outfile), nrow(res$result)))
  } else {
    write_empty_result(outfile, g1, g2, "L5", res$error)
  }
}

cat("\n=== ANCOMBC2 analysis complete ===\n")
