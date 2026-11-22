# Submission: Preflight (Team Dry Lab)

Thousands of studies, simulated in seconds, before you image a slide.

SoTA Scientific Parallelism Hackathon, King's Cross, 21-22 Nov 2026. Requirements: [docs/PRD.md](PRD.md).

## Deadlines (UTC)

| Deadline | UTC |
|---|---|
| Code freeze | 2026-11-22 16:00Z |
| Submit | 2026-11-22 17:00Z |

## Format

A live 5-minute pitch to the judges, including a live demo of about 60 s, with the demo video as backup.

## Checklist

| Item | Where | Status |
|---|---|---|
| Deck | PDF exported at freeze to `docs/assets/preflight-deck.pdf` | not yet exported |
| Live demo | https://atiladeokegab.github.io/sota-parallelism-nov21/ | live (PR #36 merged; the page answered HTTP 200 at 22:16Z on 21 Nov) |
| Offline fallback | `make demo` (checks `demo/data`, then serves `demo/` locally) | in integration |
| Demo video (backup) | `docs/film/` | recorded (proof-v5); attached at submission |
| README | `README.md`: what it is, how to run it, a screenshot from the final run | not yet refreshed |

## Run order on stage

1. Open the page.
2. Set patients and regions per patient.
3. Press "Run the studies". The race plays only on this press; changing the design clears the lanes (#48).
4. Let the five-lane race finish. The fastest lane, the fused kernel, runs on 8 CPU cores, not the GPU.
5. Point at the "our cohort" marker on the chart.

## Judging criteria and evidence

Status is read from the repo as it is now: MEASURED (a results file or a passing test in `make smoke`), ESTIMATE (a
simulated rate with its interval, or arithmetic on measured rows), SYNTHETIC (model-generated outcomes), PENDING (the
code or file is not in `integration` yet), UNMEASURED (a run that cannot answer the question). Numbers are quoted
only from results files; every row, with its source key, is in [docs/JUDGING.md](JUDGING.md).

| Criterion | PRD items | Evidence | Status |
|---|---|---|---|
| Correctness and assumptions | R-1, R-2, R-3, R-8 | `core/test_spine.py`, `stats/test_reference.py`, `tests/test_score_contract.py`, `tests/test_reimage_contract.py`, `reimage/test_backends.py`, `audit/test_audit.py` (all in `make smoke`) | MEASURED on CPU and the CUDA simulator; on a real A100, 39 passed, 0 failed (`bench/results/modal-a100-20261121T173127Z.json`, #75), MEASURED. Simulator against the real cohort's survival curve (#119, `bench/results/real-check-charville.json`): the difference interval contains zero at all three sizes; outcome model only, not imaging noise (counts MEASURED; outcomes SYNTHETIC; power ESTIMATE) |
| Fundamental reshaping | R-4, R-6 | Shared-threshold shortcut: `atlas/test_simulation.py` (#45); fleet sharding, bit-identical to one device: `reimage/test_fleet.py` (#34); fleet timing: 8 A100s against 1, 6.3× compute span, median of 2 fresh repeats (range 6.18–6.42×, same GPU model; 1 attempt failed the same-model check and is excluded; `bench/results/sweep-repeats.json`, #134), superseding #75's single sweep (6.6×) | TESTED; fleet MEASURED |
| Meaningful performance improvement | R-5, R-6 | Headline, all on 8 CPU cores: 583× against off-the-shelf squidpy + lifelines (24.4 s against 0.042 s per study, 8 workers each; cross-run, same container class, until the same-host rerun lands; instrumented, so it includes our reference recount and agreement checks; different statistics, top pattern the same in 14 of 50 studies; #132, `bench/results/external-baseline-20261122T043823Z.json`); 21× against an optimised lab-method rewrite (0.881 s against 0.0419 s per evaluated study, cross-run; 14.0× per whole design point, 1,761.88 s against 125.73 s compute wall; the fused kernel re-images, scoring and the cutoff are a separate step, and the cutoff and calls are outside the timing; #120, `bench/results/modal-design-20261121T205536Z.json`); 5.5× fair 8 against 8 (compute per decided study, calibration charged; Numba arm; #134). The race (#113, `bench/results/modal-design-20261121T194458Z.json`): against the lab way on the same 8 cores, Preflight's Numba lane is 7.0× per evaluated study and 4.70× per design point (estimating FWER and power with each procedure, calibration included); hybrid ties Numba; different test procedures, one run per arm. Matched workflow comparison through scoring (#134, `bench/results/sweep-repeats.json`; Numba arm; excludes data load, the threshold calculation and the final calls): 40.0×, median of 3 fresh repeats (range 37.3–40.2×), 25.3× with setup; 1 lab core against 8 Preflight cores; calls agree on 89 of 100 in every repeat; supersedes #123's single sweep (42.1×). Fair, 8 against 8 (#134, Numba arm, not the fused kernel behind 21×): 5.5× (range 4.6–5.6×), 3.7× with setup; calls agree on 362 of 400. CPU strong scaling (#135): 28.0× from 1 to 64 cores, median of 3 repeats (range 24.9–33.1×); #115's single sweep (38.2×) is superseded. Off the shelf (#132, `bench/results/external-baseline-20261122T043823Z.json`): squidpy + lifelines on 8 cores 24.4 s per study (MEASURED); 27.7× and 583× against the lab and fused arms, cross-run, same container class; what a lab would run off the shelf, not Preflight's speedup: different statistics, and the top pattern matches in 14 of 50 studies. Repeat spreads are median and min–max, descriptive, not confidence intervals. Rerun the race on your own CPU: `make race-local` (#140; it fetches the Zenodo data first through `make data`). It runs the same lab and fused arms as #113 and #120 on a small slice, 8 studies per stream on every core by default (`make race-local RACE_STUDIES=20 RACE_WORKERS=4` to change). Its timings are MEASURED on that machine, not the Modal figures, and it times the work without reproducing the decisions. GPU kernel 3.19× (#75) | MEASURED (compute); rates ESTIMATE; outcomes SYNTHETIC; #132 ratios cross-run |
| Scientific value | R-4, R-7, R-8 | Schoenfeld formula 80% against simulated 27% [22, 32] at 200 patients (`bench/results/vs-schoenfeld-300.json`, #53); about 524 patients for 80% (`bench/results/power-at-scale.json`, #83); head and neck: about 234 patients for 80% (pointwise 95% bounds cross 80% at 200 and 265, not a confidence interval for n; resampled from 34 UPMC discovery patients; not comparable with Charville's 524; `bench/results/hn-power-upmc-discovery.json`, #133); the shared-threshold procedure had lower estimated power, 15.3% against the lab's 20.3% (#113), and event-count calibration does not close it, +0.2 points [−0.7, +1.1] (#117); Charville v2 maxT p = 0.098 over 3,780 patterns (`results/charville-v2.json`, #47); UPMC locked test inconclusive at 18% simulated power (`results/upmc-rediscovery.json`, #109) | Schoenfeld formula ANALYTIC; simulated power, power at scale, head and neck sample size (outcomes SYNTHETIC) and calibration ESTIMATE; Charville MEASURED, exploratory; UPMC UNMEASURED (underpowered) |

The demo's answer grid, `demo/data/grid.json`, is the Atlas output: 48 design points; per point, 1,000 alternative studies for power, 10,000 null
studies for false alarms and 10,000 calibration studies for the threshold (`meta.budget`); power ESTIMATE, compute
MEASURED, outcomes SYNTHETIC.

## Hard lines

- No invented numbers. Every figure carries MEASURED, SYNTHETIC, UNMEASURED, ESTIMATE or ANALYTIC.
- Charville is the only patient-level cohort; UPMC is descriptive apart from one pre-registered locked test (R-16), which is UNMEASURED (underpowered) and inconclusive: `results/upmc-rediscovery.json`.
- Real-data results are exploratory: their outcomes have already been seen.
- Every speed-up names its baseline, hardware and scope: the lab way (a 1,000-shuffle test in every study) for the race
  and the matched run, one A100 for the fleet, the NumPy reference for the kernel. Two different test procedures are
  never called equal work.
