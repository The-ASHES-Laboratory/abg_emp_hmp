# 07_feast_source_tracking.R
# By Carter Clinton, Ph.D.
#
# FEAST Microbial Source Tracking
#
# Runs FEAST source tracking with burial and control NYABG samples as sinks,
# EMP and HMP as sources. Includes overall, body-site-stratified, and
# environment-type-stratified analyses.
#
# Outputs:
#   results/feast_split/burial_source_props.tsv
#   results/feast_split/control_source_props.tsv
#   results/feast_split/burial_source_props_by_bodysite.tsv
#   results/feast_split/burial_source_props_by_emptype.tsv
#   results/feast_split/burial_hmp_only_test.tsv
#   results/feast_split/stability/replicate_{1-5}_burial.tsv

library(FEAST)
suppressPackageStartupMessages({
  library(argparse)
  library(data.table)
})

parser <- ArgumentParser(description = "Run FEAST source tracking (split pipeline)")
parser$add_argument("--table", default = "results/compositional/merged_L6genus.tsv",
                    help = "Merged genus table")
parser$add_argument("--metadata", default = "results/metadata_cohort_split.tsv",
                    help = "Split metadata with cohort and cohort_subsite columns")
parser$add_argument("--n-subsample", type = "integer", default = 100L,
                    help = "Number of EMP/HMP samples to subsample per source (FEAST works best with <200)")
parser$add_argument("--n-replicates", type = "integer", default = 5L,
                    help = "Number of subsampling replicates")
parser$add_argument("--seed", type = "integer", default = 42L)
parser$add_argument("--em-iterations", type = "integer", default = 1000L)
parser$add_argument("--out-dir", default = "results/feast_split/")
args <- parser$parse_args()

dir.create(args$out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(args$out_dir, "stability"), showWarnings = FALSE)

# ── Load genus table ──────────────────────────────────────────────────────────
cat("Loading genus table...\n")
# Skip comment lines that aren't the header
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
# Drop taxonomy column if present
if (tolower(tail(names(counts), 1)) == "taxonomy") {
  counts[[tail(names(counts), 1)]] <- NULL
}
count_matrix <- as.matrix(counts)
rownames(count_matrix) <- taxa_names
# Round to integers for FEAST
count_matrix[is.na(count_matrix)] <- 0
count_matrix <- round(count_matrix)
cat(sprintf("  %d taxa x %d samples\n", nrow(count_matrix), ncol(count_matrix)))

# ── Load metadata ─────────────────────────────────────────────────────────────
cat("Loading metadata...\n")
meta <- fread(args$metadata, sep = "\t", header = TRUE)
setnames(meta, 1, "sample_id")
meta <- meta[meta$sample_id %in% colnames(count_matrix), ]
cat(sprintf("  %d samples in metadata\n", nrow(meta)))

# ── Helper functions ──────────────────────────────────────────────────────────
stratified_subsample <- function(meta_df, cohort_val, subsite_col, n_total, seed) {
  sub <- meta_df[meta_df$cohort == cohort_val, ]
  if (nrow(sub) <= n_total) return(sub$sample_id)

  set.seed(seed)
  subsites <- table(sub[[subsite_col]])
  n_per <- round(subsites / sum(subsites) * n_total)
  # Adjust to exact n_total
  while (sum(n_per) > n_total) n_per[which.max(n_per)] <- n_per[which.max(n_per)] - 1
  while (sum(n_per) < n_total) n_per[which.min(n_per)] <- n_per[which.min(n_per)] + 1

  sampled <- c()
  for (ss in names(n_per)) {
    pool <- sub$sample_id[sub[[subsite_col]] == ss]
    n_take <- min(n_per[ss], length(pool))
    sampled <- c(sampled, sample(pool, n_take))
  }
  return(sampled)
}

run_feast_for_sinks <- function(sink_ids, source_ids, source_labels, count_mat,
                                 em_iters = 1000, temp_dir = tempdir()) {
  # source_labels: named vector mapping source sample-id -> source name (e.g., "EMP", "HMP")
  #
  # FEAST returns a list where the first elements form a proportions matrix.
  # Columns are named "sampleID_Env" for each source sample + "Unknown".
  # We aggregate per-sample proportions by source environment name.
  #
  # FEAST also writes to dir_path/outfile_source_contributions_matrix.txt

  results <- list()
  n_sinks <- length(sink_ids)
  unique_sources <- sort(unique(source_labels))

  for (idx in seq_along(sink_ids)) {
    sink_id <- sink_ids[idx]
    if (idx %% 10 == 1) cat(sprintf("  Processing sink %d/%d (%s)\n", idx, n_sinks, sink_id))

    all_ids <- c(sink_id, source_ids)
    valid_ids <- all_ids[all_ids %in% colnames(count_mat)]
    if (length(valid_ids) < 3 || !(sink_id %in% valid_ids)) {
      cat(sprintf("  Skipping %s: too few valid samples\n", sink_id))
      next
    }

    # Extract count submatrix: taxa x samples, then transpose to samples x taxa
    sub_counts <- count_mat[, valid_ids, drop = FALSE]
    sub_mat <- as.matrix(t(sub_counts))  # samples x taxa
    storage.mode(sub_mat) <- "integer"

    # Check for zero-count samples and remove them (except the sink)
    row_sums <- rowSums(sub_mat)
    zero_rows <- names(which(row_sums == 0))
    if (sink_id %in% zero_rows) {
      cat(sprintf("  Skipping %s: sink has zero counts\n", sink_id))
      next
    }
    if (length(zero_rows) > 0) {
      valid_ids <- valid_ids[!valid_ids %in% zero_rows]
      sub_mat <- sub_mat[valid_ids, , drop = FALSE]
    }
    if (sum(valid_ids != sink_id) < 2) {
      cat(sprintf("  Skipping %s: too few non-zero source samples\n", sink_id))
      next
    }

    # Build FEAST metadata — all samples share id=1 (flag=0 uses all sources)
    feast_meta <- data.frame(
      Env = character(length(valid_ids)),
      SourceSink = character(length(valid_ids)),
      id = rep(1L, length(valid_ids)),
      stringsAsFactors = FALSE
    )
    rownames(feast_meta) <- valid_ids

    for (i in seq_along(valid_ids)) {
      sid <- valid_ids[i]
      if (sid == sink_id) {
        feast_meta$Env[i] <- sink_id
        feast_meta$SourceSink[i] <- "Sink"
      } else {
        feast_meta$Env[i] <- source_labels[sid]
        feast_meta$SourceSink[i] <- "Source"
      }
    }

    outfile_base <- paste0("feast_", gsub("[^a-zA-Z0-9]", "_", sink_id))

    tryCatch({
      # Save current working directory (FEAST does setwd!)
      old_wd <- getwd()

      feast_out <- FEAST(
        C = sub_mat,
        metadata = feast_meta,
        different_sources_flag = 0,
        dir_path = temp_dir,
        outfile = outfile_base,
        COVERAGE = NULL,
        EM_iterations = em_iters
      )

      # Restore working directory
      setwd(old_wd)

      # FEAST returns list = c(proportions_mat_columns..., FEAST_output_fields...).
      # The proportions matrix is also written to a file. Read it from file.
      out_file <- file.path(temp_dir, paste0(outfile_base, "_source_contributions_matrix.txt"))
      if (file.exists(out_file)) {
        prop_df <- read.table(out_file, sep = "\t", header = TRUE, check.names = FALSE)
        # prop_df: rows = sinks (1 row), cols = "sampleID_Env" per source + "Unknown"
        # Aggregate by source environment name
        agg <- c()
        for (src_env in unique_sources) {
          # Find columns matching this env (format: sampleID_Env)
          matching_cols <- grep(paste0("_", src_env, "$"), colnames(prop_df), value = TRUE)
          if (length(matching_cols) > 0) {
            agg[src_env] <- sum(as.numeric(prop_df[1, matching_cols]), na.rm = TRUE)
          } else {
            agg[src_env] <- 0
          }
        }
        # Unknown column
        if ("Unknown" %in% colnames(prop_df)) {
          agg["Unknown"] <- as.numeric(prop_df[1, "Unknown"])
        } else {
          agg["Unknown"] <- max(0, 1 - sum(agg))
        }

        results[[sink_id]] <- c(sample_id = sink_id, agg)
        # Clean up temp file
        file.remove(out_file)
      } else {
        cat(sprintf("  Warning: no output file for %s\n", sink_id))
      }
    }, error = function(e) {
      # Restore working directory on error too
      tryCatch(setwd(old_wd), error = function(e2) NULL)
      cat(sprintf("  FEAST error for %s: %s\n", sink_id, e$message))
    })
  }

  if (length(results) == 0) return(data.frame())
  out <- do.call(rbind, lapply(results, function(x) as.data.frame(t(x), stringsAsFactors = FALSE)))
  # Ensure numeric columns
  for (col in names(out)) {
    if (col != "sample_id") out[[col]] <- as.numeric(out[[col]])
  }
  return(out)
}

# ── 1. Overall FEAST (burial sinks, EMP+HMP sources) ─────────────────────────
cat("\n=== Overall FEAST (burial) ===\n")
burial_ids <- meta$sample_id[meta$cohort == "burial"]
control_ids <- meta$sample_id[meta$cohort == "control"]

# Use first replicate for main results
set.seed(args$seed)
emp_sub <- stratified_subsample(meta, "emp", "cohort_subsite", args$n_subsample, args$seed)
hmp_sub <- stratified_subsample(meta, "hmp", "cohort_subsite", args$n_subsample, args$seed)
source_ids <- c(emp_sub, hmp_sub)

# Create source labels
source_labels <- setNames(
  c(rep("EMP", length(emp_sub)), rep("HMP", length(hmp_sub))),
  c(emp_sub, hmp_sub)
)

burial_props <- run_feast_for_sinks(
  burial_ids, source_ids, source_labels, count_matrix,
  em_iters = args$em_iterations
)
if (nrow(burial_props) > 0) {
  fwrite(burial_props, file.path(args$out_dir, "burial_source_props.tsv"), sep = "\t")
  cat(sprintf("  Wrote burial_source_props.tsv (%d samples)\n", nrow(burial_props)))
}

# ── 2. Control sinks ─────────────────────────────────────────────────────────
cat("\n=== Overall FEAST (control) ===\n")
control_props <- run_feast_for_sinks(
  control_ids, source_ids, source_labels, count_matrix,
  em_iters = args$em_iterations
)
if (nrow(control_props) > 0) {
  fwrite(control_props, file.path(args$out_dir, "control_source_props.tsv"), sep = "\t")
  cat(sprintf("  Wrote control_source_props.tsv (%d samples)\n", nrow(control_props)))
}

# ── 3. Body-site-stratified FEAST ────────────────────────────────────────────
cat("\n=== Body-site-stratified FEAST ===\n")
hmp_subsites <- c("hmp_oral", "hmp_skin", "hmp_stool", "hmp_vaginal", "hmp_nasal")
bodysite_source_ids <- c(emp_sub)
bodysite_labels <- setNames(rep("EMP", length(emp_sub)), emp_sub)

for (site in hmp_subsites) {
  site_samples <- meta$sample_id[meta$cohort_subsite == site]
  site_sub <- if (length(site_samples) > 100) {
    sample(site_samples, 100)
  } else {
    site_samples
  }
  bodysite_source_ids <- c(bodysite_source_ids, site_sub)
  bodysite_labels <- c(bodysite_labels, setNames(rep(site, length(site_sub)), site_sub))
}

burial_bodysite <- run_feast_for_sinks(
  burial_ids, bodysite_source_ids, bodysite_labels, count_matrix,
  em_iters = args$em_iterations
)
if (nrow(burial_bodysite) > 0) {
  fwrite(burial_bodysite, file.path(args$out_dir, "burial_source_props_by_bodysite.tsv"), sep = "\t")
  cat(sprintf("  Wrote burial_source_props_by_bodysite.tsv (%d samples)\n", nrow(burial_bodysite)))
}

# ── 4. EMP-type-stratified FEAST ─────────────────────────────────────────────
cat("\n=== EMP-type-stratified FEAST ===\n")
emp_types <- c("emp_soil", "emp_water", "emp_sediment", "emp_animal", "emp_plant", "emp_other")
emptype_source_ids <- c(hmp_sub)
emptype_labels <- setNames(rep("HMP", length(hmp_sub)), hmp_sub)

for (etype in emp_types) {
  etype_samples <- meta$sample_id[meta$cohort_subsite == etype]
  etype_sub <- if (length(etype_samples) > 100) {
    sample(etype_samples, 100)
  } else {
    etype_samples
  }
  emptype_source_ids <- c(emptype_source_ids, etype_sub)
  emptype_labels <- c(emptype_labels, setNames(rep(etype, length(etype_sub)), etype_sub))
}

burial_emptype <- run_feast_for_sinks(
  burial_ids, emptype_source_ids, emptype_labels, count_matrix,
  em_iters = args$em_iterations
)
if (nrow(burial_emptype) > 0) {
  fwrite(burial_emptype, file.path(args$out_dir, "burial_source_props_by_emptype.tsv"), sep = "\t")
  cat(sprintf("  Wrote burial_source_props_by_emptype.tsv (%d samples)\n", nrow(burial_emptype)))
}

# ── 5. Diagnostic: HMP-only source ──────────────────────────────────────────
cat("\n=== Diagnostic FEAST (HMP-only source) ===\n")
hmp_only_labels <- setNames(rep("HMP", length(hmp_sub)), hmp_sub)
burial_hmp_only <- run_feast_for_sinks(
  burial_ids, hmp_sub, hmp_only_labels, count_matrix,
  em_iters = args$em_iterations
)
if (nrow(burial_hmp_only) > 0) {
  fwrite(burial_hmp_only, file.path(args$out_dir, "burial_hmp_only_test.tsv"), sep = "\t")
  cat(sprintf("  Wrote burial_hmp_only_test.tsv (%d samples)\n", nrow(burial_hmp_only)))
}

# ── 6. Stability replicates ─────────────────────────────────────────────────
cat("\n=== Stability replicates ===\n")
for (rep in seq_len(args$n_replicates)) {
  rep_seed <- args$seed + rep
  cat(sprintf("  Replicate %d (seed=%d)...\n", rep, rep_seed))

  emp_rep <- stratified_subsample(meta, "emp", "cohort_subsite", args$n_subsample, rep_seed)
  hmp_rep <- stratified_subsample(meta, "hmp", "cohort_subsite", args$n_subsample, rep_seed)
  rep_source_ids <- c(emp_rep, hmp_rep)
  rep_labels <- setNames(
    c(rep("EMP", length(emp_rep)), rep("HMP", length(hmp_rep))),
    c(emp_rep, hmp_rep)
  )

  rep_result <- run_feast_for_sinks(
    burial_ids, rep_source_ids, rep_labels, count_matrix,
    em_iters = args$em_iterations
  )
  if (nrow(rep_result) > 0) {
    fwrite(rep_result, file.path(args$out_dir, "stability",
                                  sprintf("replicate_%d_burial.tsv", rep)), sep = "\t")
  }
}

cat("\n=== FEAST analysis complete ===\n")
