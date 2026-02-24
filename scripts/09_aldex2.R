# 09_aldex2.R
# By Carter Clinton, Ph.D.
# ALDEx2 Differential Abundance
#
# Runs ALDEx2 for 5 pairwise comparisons at L5 (family) and L6 (genus) levels.
# Same comparisons as ANCOMBC2 for cross-method validation.
#
# Outputs:
#   results/diff_abun/aldex2/aldex2_{comparison}_{level}.tsv

suppressPackageStartupMessages({
  library(ALDEx2)
  library(argparse)
  library(data.table)
})

parser <- ArgumentParser(description = "Run ALDEx2 differential abundance")
parser$add_argument("--table", default = "results/compositional/merged_L6genus.tsv")
parser$add_argument("--metadata", default = "results/metadata_cohort_split.tsv")
parser$add_argument("--n-subsample", type = "integer", default = 500L)
parser$add_argument("--mc-samples", type = "integer", default = 128L)
parser$add_argument("--seed", type = "integer", default = 42L)
parser$add_argument("--out-dir", default = "results/diff_abun/aldex2/")
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

# ── Helper ────────────────────────────────────────────────────────────────────
subsample_cohort <- function(meta_df, cohort_val, n_max, seed) {
  sub <- meta_df[meta_df$cohort == cohort_val, ]
  if (nrow(sub) <= n_max) return(rownames(sub))
  set.seed(seed)
  sample(rownames(sub), n_max)
}

# ── Run ALDEx2 for one comparison ────────────────────────────────────────────
run_aldex2_comparison <- function(count_mat, meta_df, group1, group2,
                                   n_sub, mc_samples, seed, level_label) {
  g1_ids <- subsample_cohort(meta_df, group1, n_sub, seed)
  g2_ids <- subsample_cohort(meta_df, group2, n_sub, seed)
  all_ids <- c(g1_ids, g2_ids)
  valid_ids <- all_ids[all_ids %in% colnames(count_mat)]

  if (length(valid_ids) < 4) {
    cat(sprintf("  Skipping %s vs %s: too few samples\n", group1, group2))
    return(NULL)
  }

  sub_counts <- count_mat[, valid_ids, drop = FALSE]
  sub_meta <- meta_df[valid_ids, , drop = FALSE]

  # Filter taxa: prevalence >= 10% in at least one group
  g1_prev <- rowMeans(sub_counts[, sub_meta$cohort == group1, drop = FALSE] > 0)
  g2_prev <- rowMeans(sub_counts[, sub_meta$cohort == group2, drop = FALSE] > 0)
  keep <- (g1_prev >= 0.10) | (g2_prev >= 0.10)
  sub_counts <- sub_counts[keep, , drop = FALSE]

  # ALDEx2 needs integer counts
  sub_counts <- apply(sub_counts, 2, as.integer)
  rownames(sub_counts) <- rownames(count_mat)[keep]

  # Remove taxa with zero total
  row_sums <- rowSums(sub_counts)
  sub_counts <- sub_counts[row_sums > 0, , drop = FALSE]

  if (nrow(sub_counts) < 5) {
    cat(sprintf("  Skipping %s vs %s: too few taxa after filtering\n", group1, group2))
    return(NULL)
  }

  conditions <- sub_meta$cohort
  cat(sprintf("  %s vs %s (%s): %d taxa, %d samples (%d + %d)\n",
              group1, group2, level_label, nrow(sub_counts), ncol(sub_counts),
              sum(conditions == group1), sum(conditions == group2)))

  tryCatch({
    set.seed(seed)
    x <- aldex(
      reads = sub_counts,
      conditions = conditions,
      mc.samples = mc_samples,
      test = "t",
      effect = TRUE,
      denom = "all"
    )
    x$taxon <- rownames(x)
    return(x)
  }, error = function(e) {
    cat(sprintf("  ALDEx2 error: %s\n", e$message))
    return(NULL)
  })
}

# ── Comparisons ──────────────────────────────────────────────────────────────
comparisons <- list(
  c("burial", "emp"),
  c("burial", "hmp"),
  c("control", "emp"),
  c("control", "hmp"),
  c("burial", "control")
)

# ── L6 (genus) ───────────────────────────────────────────────────────────────
cat("\n=== ALDEx2 at L6 (genus) ===\n")
for (comp in comparisons) {
  g1 <- comp[1]; g2 <- comp[2]
  n_sub <- if (g1 %in% c("burial", "control") && g2 %in% c("burial", "control")) {
    10000L
  } else {
    args$n_subsample
  }

  result <- run_aldex2_comparison(count_mat, meta, g1, g2, n_sub,
                                   args$mc_samples, args$seed, "L6")
  if (!is.null(result)) {
    outfile <- file.path(args$out_dir, sprintf("aldex2_%s_vs_%s_L6.tsv", g1, g2))
    fwrite(result, outfile, sep = "\t")
    cat(sprintf("  Wrote %s (%d rows)\n", basename(outfile), nrow(result)))
  }
}

# ── L5 (family) ──────────────────────────────────────────────────────────────
cat("\n=== Aggregating to L5 (family) ===\n")
family_names <- sapply(strsplit(taxa_names, ";"), function(x) {
  fam <- grep("^f__", x, value = TRUE)
  if (length(fam) > 0) paste(x[1:which(x == fam[1])], collapse = ";") else paste(x, collapse = ";")
})
fam_groups <- split(seq_along(family_names), family_names)
count_mat_L5 <- do.call(rbind, lapply(fam_groups, function(idx) {
  if (length(idx) == 1) return(count_mat[idx, , drop = FALSE])
  colSums(count_mat[idx, , drop = FALSE])
}))
rownames(count_mat_L5) <- names(fam_groups)
cat(sprintf("  %d families\n", nrow(count_mat_L5)))

cat("\n=== ALDEx2 at L5 (family) ===\n")
for (comp in comparisons) {
  g1 <- comp[1]; g2 <- comp[2]
  n_sub <- if (g1 %in% c("burial", "control") && g2 %in% c("burial", "control")) {
    10000L
  } else {
    args$n_subsample
  }

  result <- run_aldex2_comparison(count_mat_L5, meta, g1, g2, n_sub,
                                   args$mc_samples, args$seed, "L5")
  if (!is.null(result)) {
    outfile <- file.path(args$out_dir, sprintf("aldex2_%s_vs_%s_L5.tsv", g1, g2))
    fwrite(result, outfile, sep = "\t")
    cat(sprintf("  Wrote %s (%d rows)\n", basename(outfile), nrow(result)))
  }
}

cat("\n=== ALDEx2 analysis complete ===\n")
