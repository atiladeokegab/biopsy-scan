# Negative result: why Numba with 8 threads lost to single-thread NumPy

`bench/results/cpu-ladder.json` shows Numba at 8 threads slower than the NumPy reference:
3.26 s against 2.98 s per study over 100 Charville studies. This note explains that result with
controlled measurements. All the numbers are in `bench/results/numba-negative.json`, produced by
`bench/numba_negative.py`. Every claim below is labelled MEASURED or ESTIMATE.

## What we measured

The workload is the real Charville cohort with the ladder's design: 159 patients, 2 regions each,
a 400 µm crop and eps 0.1. That makes 291 region calls per study; the median region has 2,192
cells, 55% of them visible after the crop.

Method:
- Each configuration ran in a fresh process.
- BLAS, OMP and MKL threads were pinned to 1 unless the row tests them.
- The seeds were fixed. Seed 0 was a warm-up, and the first call (JIT compile or cache load) was timed separately and excluded.
- NumPy and Numba were alternated within each run.
- Output parity was asserted for every study.
- The 1-minute load average was recorded around every timing.

The VM was heavily shared throughout: load average 48 to 174 on 8 CPUs, from other team jobs.

MEASURED, seconds per study. Each run timed 2 studies (seeds 1 and 2), and every row is the mean
of those 2 studies. The 8-thread omp row was timed in three separate runs, 1 thread and PASSIVE in
two, and the other rows in one. The NumPy and Numba columns come from the first run shown; the
ratio and launch columns give every run, separated by slashes. The ratio is Numba over NumPy,
paired within a run.

| Configuration | NumPy | Numba | Ratio | Empty parallel launch | Numba CPU time |
|---|---|---|---|---|---|
| 1 thread | 21.7 | 5.35 | 0.25 / 0.61 | 3.8 / 3.4 µs | 1.4 s |
| 2 threads | 14.0 | 11.1 | 0.79 | 6.2 µs | 3.3 s |
| 8 threads, omp layer (the ladder's setting) | 22.0 | 56.7 | 2.57 / 2.06 / 2.87 | 47,000 to 52,000 µs | 24.5 to 27.3 s |
| 8 threads, BLAS threads unpinned | 9.4 | 50.5 | 5.35 | 46,000 µs | 26.1 s |
| 8 threads, workqueue layer | 15.1 | 6.84 | 0.45 | 4,500 µs | 1.7 s |
| 8 threads, omp, `OMP_WAIT_POLICY=PASSIVE` | 8.72 | 2.48 | 0.28 / 0.42 | 193 / 855 µs | 1.4 to 1.5 s |

An earlier run (3 seeds, load 45 to 84) gave the same pattern with the thread count: Numba took
8.0, 12.7, 28.1 and 53.0 s per study at 1, 2, 4 and 8 threads, while NumPy stayed at 19 to 21 s.
That run stopped after 4 configurations.

## The cause it supports

The cause is GNU OpenMP busy-waiting on an oversubscribed machine (MEASURED). Numba's default
`omp` threading layer leaves its idle workers spinning. Each study makes 291 short parallel calls
(6 ms of single-thread work each), and each call cannot finish until all 8 workers have been
scheduled. With the run queue at 9 to 20 times the CPU count:
- An empty parallel launch cost about 50 ms at 8 threads, against 4 µs at 1 thread.
- The process burned 27 s of CPU per study for 1.4 s of useful work.

The same 8 threads with passive waiting were 2.4 to 3.5 times faster than NumPy. They did the
same work with the same memory traffic and the same outputs.

The other candidates:
- **JIT compile time counted in the timing:** ruled out (MEASURED). The ladder already excluded the first call, and here it is timed separately.
- **Numba threads multiplied by BLAS threads:** ruled out (MEASURED). Unpinning BLAS changed nothing beyond load noise, and the timed path calls no BLAS.
- **Memory bandwidth:** ruled out (MEASURED). Passive waiting removed the loss without changing memory traffic, and an empty kernel reproduces the launch cost.
- **Race-safe reduction:** ruled out (MEASURED, code). `_counts` writes only `out[i]` inside `prange(i)`; there is no reduction, atomic or shared accumulator.
- **Load average:** a necessary condition (MEASURED). Spinning is cheap on an idle machine and costly when other processes want the CPUs.
- **Too little work per thread:** a secondary factor (MEASURED). About 150 visible cells per thread per call is small. Even with workqueue, 8 threads did not beat 1 thread under this load.

Whether spinning workers also slow the NumPy reference in the same process is not established
(ESTIMATE). The NumPy times next to active-wait Numba were 22.0 s, 18.2 s and 9.7 s in three
runs; next to passive-wait Numba they were 8.7 s and 12.0 s. The load moved too much between
runs to separate the two effects.

## What would fix it

1. Set `OMP_WAIT_POLICY=PASSIVE`, or `NUMBA_THREADING_LAYER=workqueue`, for every Numba run on a shared machine (MEASURED: 2.48 s against 8.72 s, and 5.02 s against 11.96 s, per study at 8 threads).
2. Or run Numba with 1 thread per study and spread whole studies across cores (MEASURED: 5.35 s against 21.7 s, and 4.45 s against 7.29 s, per study at 1 thread).
3. Batch a study's 291 regions into one `prange` call, so each study pays one fork-join instead of 291 (ESTIMATE: not built).
4. Re-run `bench/real_cpu.py` with fix 1 on a quiet host (load below 8) before quoting any 100-study CPU speed-up (ESTIMATE). The per-study comparison has now been rerun on a dedicated host; see below. The ladder's 8-thread rows were taken at load 10 to 26 with active waiting, so they describe the scheduler, not the kernel.

The absolute seconds here are not comparable with `cpu-ladder.json` because the load differed.
Only the paired ratios within a row are.

## Quiet-host rerun on Modal

`bench/modal_quiet_cpu.py` reran the same comparison on a Modal container with 8 cores reserved,
16 GiB and no GPU. The container saw 24 CPUs; Numba ran 8 threads. It used the same worker, design and seeds (warm-up seed 0, timed seeds 1 and 2). There
were 5 interleaved rounds; each round started one fresh process per Numba mode, in rotated order.
The raw data is in `bench/results/modal-quiet-cpu-20261121T180327Z.json`. Each cell below is the median over 10 studies, with the range.

| Numba mode, 8 threads | Numba s/study | NumPy s/study (same process) | Numba / NumPy | Empty launch |
|---|---|---|---|---|
| omp, default (active wait) | 0.76 (0.70 to 0.81) | 3.12 (2.87 to 4.75) | 0.24 (0.16 to 0.26) | 13 µs |
| omp, `OMP_WAIT_POLICY=PASSIVE` | 1.13 (0.97 to 1.23) | 3.09 (2.85 to 3.41) | 0.36 (0.31 to 0.40) | 101 µs |
| workqueue | 1.22 (1.14 to 1.27) | 3.00 (2.88 to 3.42) | 0.40 (0.35 to 0.42) | 227 µs |

All of these are MEASURED. On a host with no competing load, Numba with 8 threads was
2.5 to 4.1 times faster than NumPy per study, by mode: 4.1 times with omp default, 2.8 with
PASSIVE and 2.5 with workqueue. Parity held in every study. The default
active wait was the fastest: with no competition for the cores, spinning costs nothing and
hands work over quickest. That confirms the cause above. Active waiting is the best choice on a
dedicated host and the worst on an oversubscribed one. PASSIVE is the safe default when the host is shared.

The container runs under Modal's sandbox, which reports a load average of 0 and hides the CPU
model. "Quiet" therefore means 8 reserved cores, not an observed idle host (MEASURED: Numba's
range across 10 studies was 11% to 23% of the median, by mode). The run cost about $0.05 of container
time, logged in `bench/results/modal-spend-numba.json`.
