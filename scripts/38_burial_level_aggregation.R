# 38_burial_level_aggregation.R
# By Carter Clinton, Ph.D.
# Pseudoreplication control: one composite profile per burial (mean relative abundance and
# mean CLR across the burial's samples), PERMANOVA and PERMDISP of 47 burial composites vs
# 6 controls (999 permutations), and burial-level Fisher tests of HMP attribution >= 1%.
# Replaces the strata PERMANOVA of 18_pseudorep_permanova.R (one stratum per reference sample).
# Usage: Rscript scripts/38_burial_level_aggregation.R samples=... table=... feast=... out=...
suppressMessages({library(vegan); library(data.table)})
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "rev_args.R"))
a <- parse_args_kv(list(samples = "results/gradient_r500/gradient_sample_table.tsv",
                        table = "results/feast_depth_matched/merged_L6genus_r500.tsv",
                        feast = "results/feast_depth_matched/r500/burial_source_props.tsv",
                        out = "results/revision/burial_level_aggregation.tsv"))
st <- read.delim(a$samples, stringsAsFactors = FALSE)
hdr <- names(fread(a$table, nrows = 0)); keep <- c(hdr[1], intersect(hdr, st$sample_id))
tab <- fread(a$table, select = keep); m <- t(as.matrix(tab[, -1])); colnames(m) <- tab[[1]]; m <- m[, colSums(m) > 0]
st <- st[match(rownames(m), st$sample_id), ]
st$unit <- ifelse(st$cohort == "burial", paste0("B", st$burial_id), st$sample_id)
rel <- m / rowSums(m); clr <- t(apply(m + 0.5, 1, function(x) log(x) - mean(log(x))))
agg <- function(X) rowsum(X, st$unit) / as.vector(table(st$unit)[rownames(rowsum(X, st$unit))])
relA <- agg(rel); clrA <- agg(clr); grp <- ifelse(grepl("^B", rownames(relA)), "burial", "control")
set.seed(42); out <- list()
for (dn in c("BC", "Ait")) { D <- if (dn == "BC") vegdist(relA, "bray") else dist(clrA)
  pa <- adonis2(D ~ grp, permutations = 999); bd <- permutest(betadisper(D, grp), permutations = 999)
  out[[dn]] <- data.frame(level = "burial-aggregated", distance = dn, n_burial_units = sum(grp == "burial"),
    n_control = sum(grp == "control"), R2 = pa$R2[1], F = pa$F[1], p = pa$`Pr(>F)`[1],
    permdisp_F = bd$tab$F[1], permdisp_p = bd$tab$`Pr(>F)`[1]) }
res <- do.call(rbind, out)
fb <- read.delim(a$feast); fb$burial <- st$burial_id[match(fb$sample_id, st$sample_id)]
bl <- aggregate(HMP ~ burial, fb, max); nb <- nrow(bl); k <- sum(bl$HMP >= 0.01); ft <- fisher.test(matrix(c(k, nb - k, 0, 6), 2))
blm <- aggregate(HMP ~ burial, fb, mean); k2 <- sum(blm$HMP >= 0.01); ft2 <- fisher.test(matrix(c(k2, nb - k2, 0, 6), 2))
res$fisher_note <- sprintf("burial max HMP>=1%%: %d/%d vs 0/6, p=%.3f; burial mean: %d/%d, p=%.3f", k, nb, ft$p.value, k2, nb, ft2$p.value)
print(res, digits = 3)
dir.create(dirname(a$out), showWarnings = FALSE, recursive = TRUE)
write.table(res, a$out, sep = "\t", quote = FALSE, row.names = FALSE)
