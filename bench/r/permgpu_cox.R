# permGPU's Cox score permutation test on the same X (bench/r_baselines.py); it takes no strata.
# Rscript permgpu_cox.R <dir> <B> <seed>; reads dims.txt, X.f64, time.f64, event.f64 as survival_perm.R.
# Writes permgpu_stat.f64, permgpu_unadjp.f64, permgpu_fwerp.f64 and permgpu_elapsed.txt (the permgpu() call).
suppressMessages({library(Biobase); library(permGPU)})
a <- commandArgs(TRUE); dir <- a[1]; B <- as.integer(a[2]); seed <- as.integer(a[3])
d <- scan(file.path(dir, "dims.txt"), quiet = TRUE); n <- d[1]; C <- d[2]
rd <- function(f, k) readBin(file.path(dir, f), double(), k, size = 8L)
X <- matrix(rd("X.f64", n * C), n, C)
pheno <- data.frame(time = rd("time.f64", n), event = rd("event.f64", n), row.names = sprintf("p%03d", seq_len(n)))
expr <- t(X); rownames(expr) <- sprintf("f%04d", seq_len(C))
es <- makeExprSet(as.data.frame(expr), pheno)
set.seed(seed)
t0 <- proc.time()[["elapsed"]]
res <- permgpu(es, "time", "event", "cox", B)
elapsed <- proc.time()[["elapsed"]] - t0
writeBin(as.double(res$stat), file.path(dir, "permgpu_stat.f64"))
writeBin(as.double(res$unadjp), file.path(dir, "permgpu_unadjp.f64"))
writeBin(as.double(res$fwerp), file.path(dir, "permgpu_fwerp.f64"))
writeLines(format(elapsed, digits = 10), file.path(dir, "permgpu_elapsed.txt"))
