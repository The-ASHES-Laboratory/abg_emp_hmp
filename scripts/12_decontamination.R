# 12_decontamination.R
# By Carter Clinton, Ph.D.
# Decontamination analysis using R decontam package.
#
# Identifies potential contaminant taxa using frequency-based and
# prevalence-based methods, comparing burial samples against controls.
# Cross-references against burial-network HMP-indicator taxa.
#
# Outputs:
#   results/qc/decontam_results.tsv     — per-taxon contamination scores
#   results/qc/decontam_summary.tsv     — summary of contaminant overlap
#   results/qc/contaminant_taxa.tsv     — list of flagged contaminants

suppressPackageStartupMessages({
  library(decontam)
  library(phyloseq)
  library(data.table)
  library(argparse)
})

parser <- ArgumentParser(description = "Decontamination with R decontam")
parser$add_argument("--table", default = "results/compositional/merged_L6genus.tsv")
parser$add_argument("--metadata", default = "results/metadata_cohort_split.tsv")
parser$add_argument("--nyabg-meta", default = "data/nyabg/abg_16s_meta.tsv")
parser$add_argument("--network-taxa",
                    default = "results/networks/burial/overall/source_network_taxa.tsv")
parser$add_argument("--threshold", type = "double", default = 0.1)
parser$add_argument("--out-dir", default = "results/qc/")
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
count_mat <- round(count_mat)

# ── Load metadata ─────────────────────────────────────────────────────────────
cat("Loading metadata...\n")
meta <- fread(args$metadata, sep = "\t", header = TRUE)
setnames(meta, 1, "sample_id")
meta <- as.data.frame(meta)
rownames(meta) <- meta$sample_id

# ── Focus on burial + control NYABG samples only ─────────────────────────────
nyabg_ids <- meta$sample_id[meta$cohort %in% c("burial", "control")]
nyabg_ids <- nyabg_ids[nyabg_ids %in% colnames(count_mat)]
cat(sprintf("  %d NYABG samples (burial + control)\n", length(nyabg_ids)))

if (length(nyabg_ids) < 5) {
  stop("Too few NYABG samples for decontamination analysis.")
}

sub_counts <- count_mat[, nyabg_ids]
sub_meta <- meta[nyabg_ids, ]

# Remove taxa with zero counts across all NYABG samples
taxa_sums <- rowSums(sub_counts)
sub_counts <- sub_counts[taxa_sums > 0, ]
cat(sprintf("  %d taxa with non-zero counts in NYABG\n", nrow(sub_counts)))

# ── Build phyloseq object ────────────────────────────────────────────────────
otu <- otu_table(sub_counts, taxa_are_rows = TRUE)
sdata <- sample_data(sub_meta)

# Mark controls as "negative" for decontam
# Controls = environmental soil not near burials = expected to have less
# human-associated contamination
sdata$is_neg <- sdata$cohort == "control"
ps <- phyloseq(otu, sdata)

# ── Run decontam: prevalence method ──────────────────────────────────────────
cat("Running decontam (prevalence method)...\n")
contamdf_prev <- isContaminant(ps, method = "prevalence", neg = "is_neg",
                                threshold = args$threshold)
n_contam_prev <- sum(contamdf_prev$contaminant, na.rm = TRUE)
cat(sprintf("  Prevalence method: %d contaminants identified (threshold=%.2f)\n",
            n_contam_prev, args$threshold))

# ── Run decontam: frequency method (if DNA concentration available) ──────────
# Try to load NYABG-specific metadata with DNA concentration
contamdf_freq <- NULL
tryCatch({
  nyabg_meta <- fread(args$nyabg_meta, sep = "\t", header = TRUE)
  setnames(nyabg_meta, 1, "sample_id")
  nyabg_meta <- as.data.frame(nyabg_meta)
  rownames(nyabg_meta) <- nyabg_meta$sample_id

  # Look for DNA concentration column
  conc_col <- NULL
  for (col in names(nyabg_meta)) {
    if (grepl("conc|concentration|dna_quant|qubit", col, ignore.case = TRUE)) {
      conc_col <- col
      break
    }
  }

  if (!is.null(conc_col)) {
    cat(sprintf("  Found DNA concentration column: %s\n", conc_col))
    conc <- as.numeric(nyabg_meta[nyabg_ids, conc_col])
    if (sum(!is.na(conc)) >= 5) {
      sample_data(ps)$quant <- conc
      contamdf_freq <- isContaminant(ps, method = "frequency", conc = "quant",
                                      threshold = args$threshold)
      n_contam_freq <- sum(contamdf_freq$contaminant, na.rm = TRUE)
      cat(sprintf("  Frequency method: %d contaminants identified\n", n_contam_freq))
    } else {
      cat("  Too few non-NA concentration values for frequency method.\n")
    }
  } else {
    cat("  No DNA concentration column found; skipping frequency method.\n")
  }
}, error = function(e) {
  cat(sprintf("  Could not load NYABG metadata for frequency method: %s\n", e$message))
})

# ── Combine results ──────────────────────────────────────────────────────────
results <- data.frame(
  taxon = rownames(contamdf_prev),
  prev_p = contamdf_prev$p,
  prev_contaminant = contamdf_prev$contaminant,
  stringsAsFactors = FALSE
)

if (!is.null(contamdf_freq)) {
  results$freq_p <- contamdf_freq$p
  results$freq_contaminant <- contamdf_freq$contaminant
  results$either_contaminant <- results$prev_contaminant | results$freq_contaminant
} else {
  results$freq_p <- NA
  results$freq_contaminant <- NA
  results$either_contaminant <- results$prev_contaminant
}

# Sort by prevalence p-value
results <- results[order(results$prev_p), ]

# Write full results
out_results <- file.path(args$out_dir, "decontam_results.tsv")
fwrite(results, out_results, sep = "\t")
cat(sprintf("\nWrote %s (%d taxa)\n", basename(out_results), nrow(results)))

# ── Contaminant list ─────────────────────────────────────────────────────────
contaminants <- results[results$either_contaminant == TRUE & !is.na(results$either_contaminant), ]
out_contam <- file.path(args$out_dir, "contaminant_taxa.tsv")
fwrite(contaminants, out_contam, sep = "\t")
cat(sprintf("Wrote %s (%d contaminant taxa)\n", basename(out_contam), nrow(contaminants)))

# ── Cross-reference with network HMP indicators ─────────────────────────────
cat("\nCross-referencing with burial network taxa...\n")
if (file.exists(args$network_taxa)) {
  net_taxa <- fread(args$network_taxa, sep = "\t", header = TRUE)
  # Find HMP-indicator taxa in network
  hmp_col <- NULL
  for (col in names(net_taxa)) {
    if (grepl("hmp", col, ignore.case = TRUE)) {
      hmp_col <- col
      break
    }
  }

  if (!is.null(hmp_col)) {
    # Get taxa assigned to HMP (or burial+hmp)
    hmp_taxa <- net_taxa[[1]][grepl("hmp", net_taxa[[hmp_col]], ignore.case = TRUE) |
                               grepl("hmp", net_taxa$presence_pattern, ignore.case = TRUE)]
    if (length(hmp_taxa) == 0 && "presence_pattern" %in% names(net_taxa)) {
      hmp_taxa <- net_taxa[[1]][grepl("hmp", net_taxa$presence_pattern)]
    }
    cat(sprintf("  %d HMP-indicator taxa in network\n", length(hmp_taxa)))

    overlap <- intersect(contaminants$taxon, hmp_taxa)
    cat(sprintf("  %d overlap between contaminants and HMP indicators\n", length(overlap)))

    summary_df <- data.frame(
      metric = c("total_contaminants", "hmp_indicator_taxa",
                 "overlap_contam_hmp", "overlap_fraction"),
      value = c(nrow(contaminants), length(hmp_taxa),
                length(overlap),
                ifelse(nrow(contaminants) > 0,
                       round(length(overlap) / nrow(contaminants), 3), 0))
    )
  } else {
    summary_df <- data.frame(
      metric = c("total_contaminants", "note"),
      value = c(nrow(contaminants), "no HMP column in network taxa")
    )
  }
} else {
  cat("  Network taxa file not found.\n")
  summary_df <- data.frame(
    metric = c("total_contaminants", "note"),
    value = c(nrow(contaminants), "network taxa file not found")
  )
}

out_summary <- file.path(args$out_dir, "decontam_summary.tsv")
fwrite(summary_df, out_summary, sep = "\t")
cat(sprintf("Wrote %s\n", basename(out_summary)))

cat("\n=== Decontamination analysis complete ===\n")
