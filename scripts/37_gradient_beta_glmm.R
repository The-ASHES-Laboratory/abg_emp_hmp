# 37_gradient_beta_glmm.R
# By Carter Clinton, Ph.D.
# Beta and zero-inflated beta GLMMs (glmmTMB) of FEAST HMP proportion on proximity rank
# (cranium = 1, other body = 2, soil matrix = 3) and log10 read depth, burial random intercept.
# Beta model: proportions squeezed after Smithson and Verkuilen (2006). ZI-beta: intercept-only
# zero inflation (the covariate ZI model in 33_gradient_glmm.R is not identifiable with one zero).
# Usage: Rscript scripts/37_gradient_beta_glmm.R primary=... contam=... out=...
suppressMessages(library(glmmTMB))
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "rev_args.R"))
a <- parse_args_kv(list(primary = "results/gradient_r500/gradient_sample_table.tsv",
                        contam = "results/contam_sensitivity/gradient_r500/gradient_sample_table.tsv",
                        out = "results/revision/gradient_beta_glmm.tsv"))
fit <- function(f, lab) {
  d <- read.delim(f, stringsAsFactors = FALSE); d <- d[d$cohort == "burial" & !is.na(d$depth), ]
  d$prox_rank <- match(d$level, c("Cranium", "Other Body", "Soil Matrix")); d$log10depth <- log10(d$depth)
  n <- nrow(d); d$y <- (d$hmp * (n - 1) + 0.5) / n
  m1 <- glmmTMB(y ~ prox_rank + log10depth + (1 | burial_id), family = beta_family(), data = d)
  m2 <- glmmTMB(hmp ~ prox_rank + log10depth + (1 | burial_id), ziformula = ~1, family = beta_family(),
                data = transform(d, hmp = pmin(hmp, 1 - 1e-6)))
  rbind(
    data.frame(table = lab, model = "beta GLMM (Smithson-Verkuilen squeeze), burial RE",
      t(summary(m1)$coefficients$cond["prox_rank", ]), re_var = as.numeric(VarCorr(m1)$cond$burial_id), n = n, exact_zeros = sum(d$hmp == 0)),
    data.frame(table = lab, model = "ZI-beta GLMM (intercept-only ZI), burial RE",
      t(summary(m2)$coefficients$cond["prox_rank", ]), re_var = as.numeric(VarCorr(m2)$cond$burial_id), n = n, exact_zeros = sum(d$hmp == 0)))
}
res <- rbind(fit(a$primary, "primary r500"), fit(a$contam, "contam-filtered r500"))
names(res)[3:6] <- c("estimate", "se", "z", "p"); print(res, digits = 3)
dir.create(dirname(a$out), showWarnings = FALSE, recursive = TRUE)
write.table(res, a$out, sep = "\t", quote = FALSE, row.names = FALSE)
