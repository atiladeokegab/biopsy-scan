# The Charville v2 permutation screen as a competent R user would write it (bench/r_baselines.py).
# Rscript survival_perm.R <dir> <cores>; <dir> holds dims.txt (n C B), X.f64 (n x C, column-major, NaN = missing),
# time.f64, event.f64, strata.i32 and P.i32 (n x B, 0-based: shuffle b gives patient i the outcome of P[i, b]).
# Writes z.f64 (observed z per pattern), null.f64 (each shuffle's maxT) and elapsed.txt (seconds, compute only).
suppressMessages({library(survival); library(parallel)})
a <- commandArgs(TRUE); dir <- a[1]; cores <- as.integer(a[2])
d <- scan(file.path(dir, "dims.txt"), quiet = TRUE); n <- d[1]; C <- d[2]; B <- d[3]
rd <- function(f, what, k) readBin(file.path(dir, f), what, k, size = if (is.integer(what)) 4L else 8L)
X <- matrix(rd("X.f64", double(), n * C), n, C)
tt <- rd("time.f64", double(), n); ee <- rd("event.f64", double(), n); ss <- rd("strata.i32", integer(), n)
P <- matrix(rd("P.i32", integer(), n * B), n, B) + 1L

t0 <- proc.time()[["elapsed"]]
# Null stratified Cox fit (Breslow ties): its martingale residuals are the score's per-patient weights.
resid0 <- function(t, e, s) as.vector(residuals(coxph(Surv(t, e) ~ strata(s), ties = "breslow"), type = "martingale"))
key <- apply(is.na(X), 2, function(m) paste(which(m), collapse = ","))
groups <- lapply(split(seq_len(C), key), function(cols) {  # one group per missingness mask: complete cases
  keep <- !is.na(X[, cols[1]]); s <- factor(ss[keep])
  Xc <- X[keep, cols, drop = FALSE]
  Xc <- Xc - rowsum(Xc, s)[as.integer(s), , drop = FALSE] / as.vector(table(s))[as.integer(s)]  # centre within strata
  ns <- as.vector(table(s))
  list(cols = cols, keep = keep, s = s, Xc = Xc, Sxx = rowsum(Xc^2, s) * ifelse(ns >= 2, 1 / (ns - 1), 0),
       r = resid0(tt[keep], ee[keep], s), full = all(keep))
})
zstat <- function(g, R) {  # R: complete cases x shuffles; V is the exact within-stratum permutation variance
  V <- crossprod(g$Sxx, rowsum(R^2, g$s)); U <- crossprod(g$Xc, R)
  ifelse(V > 0, U / sqrt(pmax(V, 0)), 0)
}
z <- numeric(C)
for (g in groups) z[g$cols] <- zstat(g, matrix(g$r))
chunks <- split(seq_len(B), cut(seq_len(B), cores, labels = FALSE))
null <- unlist(mclapply(chunks, function(bs) {
  m <- numeric(length(bs))
  for (g in groups) {
    R <- if (g$full) matrix(g$r[P[, bs]], ncol = length(bs))  # same risk sets: shuffled outcomes shuffle residuals
         else vapply(bs, function(b) { o <- P[, b]; resid0(tt[o][g$keep], ee[o][g$keep], g$s) }, numeric(sum(g$keep)))
    m <- pmax(m, apply(abs(zstat(g, R)), 2, max))
  }
  m
}, mc.cores = cores), use.names = FALSE)
elapsed <- proc.time()[["elapsed"]] - t0
writeBin(z, file.path(dir, "z.f64")); writeBin(null, file.path(dir, "null.f64"))
writeLines(format(elapsed, digits = 10), file.path(dir, "elapsed.txt"))
