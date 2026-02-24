# 05_indicator_species.R
# By Carter Clinton, Ph.D.
# Indicator Species Analysis (ISA) using indicspecies::multipatt
#
# Identifies taxa significantly associated with specific cohorts (burial,
# control, EMP, HMP) or cohort combinations using the IndVal methodology.
#
# Outputs:
#   results/sensitivity/isa_results.tsv       — full ISA results
#   results/sensitivity/isa_summary.tsv       — summary counts per group

suppressPackageStartupMessages({
  library(indicspecies)
  library(data.table)
  library(argparse)
})

parser <- ArgumentParser(description = "Indicator Species Analysis")
parser$add_argument("--table", default = "results/compositional/merged_L6genus.tsv")
parser$add_argument("--metadata", default = "results/metadata_cohort_split.tsv")
parser$add_argument("--n-subsample", type = "integer", default = 500L)
parser$add_argument("--seed", type = "integer", default = 42L)
parser$add_argument("--out-dir", default = "results/sensitivity/")
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

cat("Loading metadata...\n")
meta <- fread(args$metadata, sep = "\t", header = TRUE)
setnames(meta, 1, "sample_id")
meta <- as.data.frame(meta)
rownames(meta) <- meta$sample_id

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
cat(sprintf("Using %d samples: %d burial, %d control, %d EMP, %d HMP\n",
            length(all_ids), length(burial_ids), length(control_ids),
            length(emp_ids), length(hmp_ids)))

# ── Prepare matrices ─────────────────────────────────────────────────────────
sub_counts <- count_mat[, all_ids]
# Transpose: samples x taxa (required by multipatt)
comm <- t(sub_counts)

# Remove zero-sum taxa
taxa_sums <- colSums(comm)
comm <- comm[, taxa_sums > 0]
cat(sprintf("  %d taxa with non-zero counts\n", ncol(comm)))

# Prevalence filter: present in >= 5% of samples
prevalence <- colMeans(comm > 0)
comm <- comm[, prevalence >= 0.05]
cat(sprintf("  %d taxa after 5%% prevalence filter\n", ncol(comm)))

# Group vector
groups <- meta[all_ids, "cohort"]

# ── Run multipatt ─────────────────────────────────────────────────────────────
cat("Running indicator species analysis (this may take a few minutes)...\n")
isa <- multipatt(comm, groups, func = "IndVal.g", control = how(nperm = 999))

cat("Done.\n")

# ── Extract results ──────────────────────────────────────────────────────────
isa_summary <- isa$sign
isa_summary$taxon <- rownames(isa_summary)

# Add indicator group name
isa_summary$indicator_group <- apply(isa_summary[, grepl("^s\\.", names(isa_summary))], 1,
  function(x) {
    cols <- names(x)[x == 1]
    # Strip "s." prefix
    groups <- gsub("^s\\.", "", cols)
    paste(groups, collapse = "+")
  }
)

# Sort by p-value
isa_summary <- isa_summary[order(isa_summary$p.value), ]

# Write full results
out_full <- file.path(args$out_dir, "isa_results.tsv")
fwrite(isa_summary, out_full, sep = "\t")
cat(sprintf("Wrote %s (%d taxa)\n", basename(out_full), nrow(isa_summary)))

# Significant indicators (p < 0.05)
sig <- isa_summary[!is.na(isa_summary$p.value) & isa_summary$p.value < 0.05, ]
cat(sprintf("\n%d significant indicator taxa (p < 0.05):\n", nrow(sig)))

# Summary by indicator group
if (nrow(sig) > 0) {
  summary_tab <- as.data.frame(table(sig$indicator_group))
  names(summary_tab) <- c("indicator_group", "n_taxa")
  summary_tab <- summary_tab[order(-summary_tab$n_taxa), ]
  print(summary_tab, row.names = FALSE)

  out_summary <- file.path(args$out_dir, "isa_summary.tsv")
  fwrite(summary_tab, out_summary, sep = "\t")
  cat(sprintf("\nWrote %s\n", basename(out_summary)))
} else {
  cat("No significant indicator taxa found.\n")
}

cat("\n=== ISA complete ===\n")
