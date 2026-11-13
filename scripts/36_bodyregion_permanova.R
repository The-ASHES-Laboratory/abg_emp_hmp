# 36_bodyregion_permanova.R
# By Carter Clinton, Ph.D.
# PERMANOVA of burial-sample composition by recorded body region (11 categories) and by
# proximity level (3), Bray-Curtis on relative abundance and Aitchison (CLR, pseudocount 0.5),
# 999 permutations, unrestricted and with permutations restricted within burials.
# Replaces the body-region test of 16_confound_permanova.py (which matched no samples).
# Usage: Rscript scripts/36_bodyregion_permanova.R samples=... table=... out=...
suppressMessages({library(vegan); library(data.table)})
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "rev_args.R"))
a <- parse_args_kv(list(samples = "results/gradient_r500/gradient_sample_table.tsv",
                        table = "results/feast_depth_matched/merged_L6genus_r500.tsv",
                        out = "results/revision/bodyregion_permanova.tsv"))
st <- read.delim(a$samples, stringsAsFactors = FALSE); st <- st[st$cohort == "burial", ]
hdr <- names(fread(a$table, nrows = 0)); keep <- c(hdr[1], intersect(hdr, st$sample_id))
tab <- fread(a$table, select = keep)
m <- as.matrix(tab[, -1]); rownames(m) <- tab[[1]]; m <- t(m); m <- m[, colSums(m) > 0]
st <- st[match(rownames(m), st$sample_id), ]
rel <- m / rowSums(m); bc <- vegdist(rel, "bray")
clr <- t(apply(m + 0.5, 1, function(x) log(x) - mean(log(x)))); ait <- dist(clr)
set.seed(42); res <- list()
for (dn in c("BC", "Ait")) { D <- if (dn == "BC") bc else ait
  for (v in c("body_region", "level")) {
    a1 <- adonis2(D ~ st[[v]], permutations = 999)
    h <- how(nperm = 999); setBlocks(h) <- factor(st$burial_id)
    a2 <- adonis2(D ~ st[[v]], permutations = h)
    res[[length(res) + 1]] <- data.frame(distance = dn, factor = v, df = a1$Df[1], R2 = a1$R2[1], F = a1$F[1],
      p_unrestricted = a1$`Pr(>F)`[1], p_within_burial_blocks = a2$`Pr(>F)`[1]) } }
out <- do.call(rbind, res); print(out, digits = 4)
dir.create(dirname(a$out), showWarnings = FALSE, recursive = TRUE)
write.table(out, a$out, sep = "\t", quote = FALSE, row.names = FALSE)
