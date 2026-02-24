# 18_pseudorep_permanova.R
# By Carter Clinton, Ph.D.
# Pseudoreplication-Corrected PERMANOVA
#
# Re-runs PERMANOVA with strata=burial_id using vegan::adonis2().
# skbio's permanova() does not support strata, so this must be done in R.
#
# Uses the burial ID mapping produced by 02a_pseudoreplication.py.
#
# Inputs:
#   results/compositional/merged_L6genus.tsv
#   results/metadata_cohort_split.tsv
#   results/pseudoreplication/sample_burial_id_map.tsv
#
# Outputs:
#   results/pseudoreplication/pseudorep_permanova.tsv

suppressPackageStartupMessages({
  library(vegan)
  library(data.table)
  library(argparse)
})

parser <- ArgumentParser(description = "PERMANOVA with strata for pseudoreplication correction")
parser$add_argument("--table", default = "results/compositional/merged_L6genus.tsv")
parser$add_argument("--metadata", default = "results/metadata_cohort_split.tsv")
parser$add_argument("--burial-map", default = "results/pseudoreplication/sample_burial_id_map.tsv")
parser$add_argument("--n-subsample", type = "integer", default = 500L)
parser$add_argument("--seed", type = "integer", default = 42L)
parser$add_argument("--out-dir", default = "results/pseudoreplication/")
args <- parser$parse_args()

dir.create(args$out_dir, recursive = TRUE, showWarnings = FALSE)

# ── Load genus table ──────────────────────────────────────────────────────────
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
cat(sprintf("  %d taxa x %d samples\n", nrow(count_mat), ncol(count_mat)))

# ── Load metadata ─────────────────────────────────────────────────────────────
cat("Loading metadata...\n")
meta <- fread(args$metadata, sep = "\t", header = TRUE)
setnames(meta, 1, "sample_id")
meta <- as.data.frame(meta)
rownames(meta) <- meta$sample_id
meta <- meta[meta$sample_id %in% colnames(count_mat), ]
cat(sprintf("  %d samples\n", nrow(meta)))

# ── Load burial ID mapping ───────────────────────────────────────────────────
cat("Loading burial ID mapping...\n")
burial_map <- fread(args$burial_map, sep = "\t", header = TRUE)
burial_map <- as.data.frame(burial_map)
cat(sprintf("  %d sample-burial mappings\n", nrow(burial_map)))

# ── Subsample EMP/HMP ────────────────────────────────────────────────────────
set.seed(args$seed)
emp_ids <- meta$sample_id[meta$cohort == "emp"]
hmp_ids <- meta$sample_id[meta$cohort == "hmp"]
burial_ids <- meta$sample_id[meta$cohort == "burial"]
control_ids <- meta$sample_id[meta$cohort == "control"]

if (length(emp_ids) > args$n_subsample) {
  emp_ids <- sample(emp_ids, args$n_subsample)
}
if (length(hmp_ids) > args$n_subsample) {
  hmp_ids <- sample(hmp_ids, args$n_subsample)
}

all_ids <- c(burial_ids, control_ids, emp_ids, hmp_ids)
all_ids <- all_ids[all_ids %in% colnames(count_mat)]

# ── Prepare community matrix (samples x taxa) ────────────────────────────────
comm <- t(count_mat[, all_ids])
# Remove zero-sum taxa
taxa_sums <- colSums(comm)
comm <- comm[, taxa_sums > 0]
cat(sprintf("Community matrix: %d samples x %d taxa\n", nrow(comm), ncol(comm)))

# Remove zero-sum samples (no reads after filtering)
row_totals <- rowSums(comm)
zero_samples <- row_totals == 0
if (any(zero_samples)) {
  cat(sprintf("  Removing %d samples with zero total reads\n", sum(zero_samples)))
  comm <- comm[!zero_samples, ]
  row_totals <- rowSums(comm)
}

# Relative abundance (TSS)
comm_rel <- comm / row_totals
# Replace any remaining NaN/NA with 0
comm_rel[is.na(comm_rel)] <- 0

# ── Compute Bray-Curtis distance ─────────────────────────────────────────────
cat("Computing Bray-Curtis distance...\n")
bc_dist <- vegdist(comm_rel, method = "bray")

# ── Build grouping and strata vectors ─────────────────────────────────────────
cohort_vec <- meta[rownames(comm), "cohort"]

# Build strata: burial_id for burial/control samples, unique ID for EMP/HMP
strata_vec <- character(nrow(comm))
for (i in seq_len(nrow(comm))) {
  sid <- rownames(comm)[i]
  matched <- burial_map$burial_id[burial_map$sample_id == sid]
  if (length(matched) > 0) {
    strata_vec[i] <- matched[1]
  } else {
    # EMP/HMP samples — each is its own stratum (independent)
    strata_vec[i] <- sid
  }
}

# ── PERMANOVA without strata (standard) ──────────────────────────────────────
cat("Running PERMANOVA without strata...\n")
perm_no_strata <- adonis2(bc_dist ~ cohort_vec, permutations = 999)
cat("  Without strata:\n")
print(perm_no_strata)

# ── PERMANOVA with strata ────────────────────────────────────────────────────
cat("\nRunning PERMANOVA with strata (burial_id)...\n")
strata_factor <- factor(strata_vec)

# Use strata in permutations
perm_ctrl <- how(nperm = 999, blocks = strata_factor)
perm_with_strata <- adonis2(bc_dist ~ cohort_vec, permutations = perm_ctrl)
cat("  With strata:\n")
print(perm_with_strata)

# ── Extract and save results ─────────────────────────────────────────────────
results <- data.frame(
  analysis = c("no_strata", "with_strata_burial_id"),
  R2 = c(perm_no_strata$R2[1], perm_with_strata$R2[1]),
  F_statistic = c(perm_no_strata$F[1], perm_with_strata$F[1]),
  p_value = c(perm_no_strata$`Pr(>F)`[1], perm_with_strata$`Pr(>F)`[1]),
  df = c(perm_no_strata$Df[1], perm_with_strata$Df[1]),
  n_samples = c(nrow(comm), nrow(comm)),
  n_burials = c(
    length(unique(strata_vec[cohort_vec %in% c("burial", "control")])),
    length(unique(strata_vec[cohort_vec %in% c("burial", "control")]))
  ),
  n_unique_strata = c(NA, length(unique(strata_vec))),
  stringsAsFactors = FALSE
)

out_path <- file.path(args$out_dir, "pseudorep_permanova.tsv")
fwrite(results, out_path, sep = "\t")
cat(sprintf("\nWrote %s\n", out_path))

# ── Pairwise with strata ─────────────────────────────────────────────────────
cat("\nRunning pairwise PERMANOVA with strata...\n")
cohorts <- sort(unique(cohort_vec))
pw_rows <- list()
for (i in seq_along(cohorts)) {
  for (j in seq(i + 1, length(cohorts))) {
    if (j > length(cohorts)) break
    c1 <- cohorts[i]
    c2 <- cohorts[j]
    mask <- cohort_vec %in% c(c1, c2)
    if (sum(mask) < 4) next

    sub_dist <- as.dist(as.matrix(bc_dist)[mask, mask])
    sub_cohort <- cohort_vec[mask]
    sub_strata <- factor(strata_vec[mask])

    tryCatch({
      sub_ctrl <- how(nperm = 999, blocks = sub_strata)
      pw_result <- adonis2(sub_dist ~ sub_cohort, permutations = sub_ctrl)
      pw_rows[[length(pw_rows) + 1]] <- data.frame(
        comparison = paste0(c1, "_vs_", c2),
        R2 = pw_result$R2[1],
        F_statistic = pw_result$F[1],
        p_value = pw_result$`Pr(>F)`[1],
        n_samples = sum(mask),
        stringsAsFactors = FALSE
      )
      cat(sprintf("  %s vs %s: R2=%.4f, F=%.2f, p=%.4f\n",
                  c1, c2, pw_result$R2[1], pw_result$F[1], pw_result$`Pr(>F)`[1]))
    }, error = function(e) {
      cat(sprintf("  Skipping %s vs %s: %s\n", c1, c2, e$message))
    })
  }
}

if (length(pw_rows) > 0) {
  pw_df <- do.call(rbind, pw_rows)
  # BH correction
  pw_df$p_BH <- p.adjust(pw_df$p_value, method = "BH")
  pw_path <- file.path(args$out_dir, "pseudorep_permanova_pairwise.tsv")
  fwrite(pw_df, pw_path, sep = "\t")
  cat(sprintf("Wrote %s\n", pw_path))
}

cat("\n=== Pseudoreplication PERMANOVA complete ===\n")
