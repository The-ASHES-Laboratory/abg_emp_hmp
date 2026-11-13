#!/usr/bin/env Rscript
# 33_gradient_glmm.R
# Mixed models for the anatomical-proximity gradient with burial identity as a random intercept
# (Reviewer 2 minor 3) and read depth as a covariate (Reviewer 2 major 2).
# Models (burial samples only; levels Cranium < Other Body < Soil Matrix coded 0,1,2 as an ordinal trend
# and also as a factor):
#   M1 glmmTMB zero-inflated beta: hmp ~ proximity + log10 depth + (1|burial), zi ~ proximity + log10 depth   [if glmmTMB available]
#   M2 two-part fallback / check: glmer binomial any-HMP ~ proximity + log10depth + (1|burial);
#      lmer logit(hmp) ~ proximity + log10depth + (1|burial) on positive samples
#   M3 lmer asin(sqrt(hmp)) ~ proximity + log10depth + (1|burial) on all burial samples
# Input: gradient_sample_table.tsv from 32_gradient_tests.py.  Output: <out-dir>/gradient_glmm.tsv, gradient_glmm_models.txt
suppressPackageStartupMessages({library(argparse)})
p <- ArgumentParser(); p$add_argument("--sample-table", default="results/gradient/gradient_sample_table.tsv"); p$add_argument("--out-dir", default="results/gradient/")
a <- p$parse_args(); dir.create(a$out_dir, showWarnings=FALSE, recursive=TRUE)
d <- read.delim(a$sample_table, stringsAsFactors=FALSE); d <- d[d$cohort=="burial" & !is.na(d$depth), ]
d$prox <- factor(d$level, levels=c("Cranium","Other Body","Soil Matrix")); d$prox_rank <- as.numeric(d$prox)-1; d$log10depth <- log10(d$depth)
d$burial_id <- ifelse(is.na(d$burial_id) | d$burial_id=="", paste0("single_", d$sample_id), d$burial_id); d$any <- as.integer(d$hmp>0)
eps <- 1e-4; d$hmp_sq <- pmin(pmax(d$hmp, eps), 1-eps)   # squeeze for beta
rows <- list(); sink(file.path(a$out_dir,"gradient_glmm_models.txt"))
cat("n burial samples:", nrow(d), " n burials (random-effect levels):", length(unique(d$burial_id)), "\n"); print(table(d$prox))
addrow <- function(model, term, est, se, z, pval, note="") rows[[length(rows)+1]] <<- data.frame(model=model, term=term, estimate=est, se=se, statistic=z, p=pval, note=note)
if (requireNamespace("glmmTMB", quietly=TRUE)) {
  library(glmmTMB)
  for (form in c("rank","factor")) {
    f <- if (form=="rank") hmp_sq ~ prox_rank + log10depth + (1|burial_id) else hmp_sq ~ prox + log10depth + (1|burial_id)
    zf <- if (form=="rank") ~ prox_rank + log10depth else ~ prox + log10depth
    m <- tryCatch(glmmTMB(f, ziformula=zf, family=beta_family(), data=d), error=function(e) e)
    if (inherits(m,"error")) { cat("M1 glmmTMB ZI-beta (",form,") failed:", conditionMessage(m), "\n"); addrow(paste0("M1 ZI-beta glmmTMB (",form,")"), "FAILED", NA, NA, NA, NA, conditionMessage(m)); next }
    cat("\n==== M1 glmmTMB zero-inflated beta (", form, ") ====\n"); print(summary(m))
    s <- summary(m)$coefficients
    for (part in names(s)) { cf <- s[[part]]; if (is.null(cf)) next; for (i in seq_len(nrow(cf))) addrow(paste0("M1 ZI-beta glmmTMB (",form,")"), paste0(part,":",rownames(cf)[i]), cf[i,1], cf[i,2], cf[i,3], cf[i,4]) }
    vc <- VarCorr(m)$cond$burial_id; addrow(paste0("M1 ZI-beta glmmTMB (",form,")"), "var(burial random intercept, cond)", as.numeric(vc[1,1]), NA, NA, NA)
  }
} else { cat("glmmTMB not available\n"); addrow("M1 ZI-beta glmmTMB", "NOT RUN", NA, NA, NA, NA, "glmmTMB not installed") }
if (requireNamespace("lme4", quietly=TRUE)) {
  library(lme4)
  g <- tryCatch(glmer(any ~ prox_rank + log10depth + (1|burial_id), family=binomial, data=d, control=glmerControl(optimizer="bobyqa")), error=function(e) e)
  if (!inherits(g,"error")) { cat("\n==== M2a glmer binomial any-HMP ====\n"); print(summary(g)); cf <- summary(g)$coefficients; for (i in seq_len(nrow(cf))) addrow("M2a glmer binomial any-HMP", rownames(cf)[i], cf[i,1], cf[i,2], cf[i,3], cf[i,4]); addrow("M2a glmer binomial any-HMP","var(burial)", as.numeric(VarCorr(g)$burial_id[1,1]),NA,NA,NA) } else addrow("M2a glmer binomial any-HMP","FAILED",NA,NA,NA,NA,conditionMessage(g))
  dp <- d[d$hmp>0,]; dp$logit <- qlogis(pmin(pmax(dp$hmp,eps),1-eps))
  if (nrow(dp) > 8) { l <- tryCatch(lmer(logit ~ prox_rank + log10depth + (1|burial_id), data=dp, REML=TRUE), error=function(e) e)
    if (!inherits(l,"error")) { cat("\n==== M2b lmer logit(HMP) | HMP>0 ====\n"); print(summary(l)); cf <- summary(l)$coefficients; tv <- cf[, "t value"]; pv <- 2*pnorm(-abs(tv)); for (i in seq_len(nrow(cf))) addrow("M2b lmer logit(HMP) positives", rownames(cf)[i], cf[i,1], cf[i,2], tv[i], pv[i], "p from normal approx of t") } }
  l3 <- tryCatch(lmer(asin(sqrt(hmp)) ~ prox_rank + log10depth + (1|burial_id), data=d, REML=TRUE), error=function(e) e)
  if (!inherits(l3,"error")) { cat("\n==== M3 lmer asin-sqrt(HMP), all burial ====\n"); print(summary(l3)); cf <- summary(l3)$coefficients; tv <- cf[, "t value"]; pv <- 2*pnorm(-abs(tv)); for (i in seq_len(nrow(cf))) addrow("M3 lmer asin-sqrt all", rownames(cf)[i], cf[i,1], cf[i,2], tv[i], pv[i], "p from normal approx of t"); addrow("M3 lmer asin-sqrt all","var(burial)", as.numeric(VarCorr(l3)$burial_id[1,1]),NA,NA,NA) }
}
sink(); out <- do.call(rbind, rows); write.table(out, file.path(a$out_dir,"gradient_glmm.tsv"), sep="\t", quote=FALSE, row.names=FALSE); print(out)
