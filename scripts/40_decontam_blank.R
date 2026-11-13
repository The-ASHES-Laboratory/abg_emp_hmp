# 40_decontam_blank.R
# By Carter Clinton, Ph.D.
# Prevalence-based decontam (thresholds 0.1 and 0.5) with the touchdown-PCR water control as
# the negative set, on the genus table from 39_blank_genus_table.py. Replaces the control-soil
# negative set of 12_decontamination.R, which cannot flag taxa that are rare in control soils.
suppressMessages(library(decontam))
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "rev_args.R"))
a <- parse_args_kv(list(table = "results/revision/genus_nyabg_plus_blank.tsv", blank = "Water-neg-TD-plateD",
                        out = "results/revision/decontam_blank_negative.tsv"))
G <- read.delim(a$table, row.names = 1, check.names = FALSE)
X <- t(as.matrix(G)); neg <- rownames(X) == a$blank
r <- do.call(rbind, lapply(c(0.1, 0.5), function(th) {
  d <- isContaminant(X, neg = neg, method = "prevalence", threshold = th)
  data.frame(threshold = th, taxon = rownames(d)[d$contaminant]) }))
print(r); dir.create(dirname(a$out), showWarnings = FALSE, recursive = TRUE)
write.table(r, a$out, sep = "\t", quote = FALSE, row.names = FALSE)
