# rev_args.R: minimal key=value argument parser shared by the revision R scripts.
# By Carter Clinton, Ph.D.
parse_args_kv <- function(defaults) {
  a <- commandArgs(trailingOnly = TRUE)
  for (x in a) { kv <- strsplit(sub("^--", "", x), "=", fixed = TRUE)[[1]]
    if (length(kv) == 2 && kv[1] %in% names(defaults)) defaults[[kv[1]]] <- kv[2] }
  defaults
}
