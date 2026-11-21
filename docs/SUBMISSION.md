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
4. Let the CPU-vs-GPU race finish.
5. Point at the "our cohort" marker on the chart.

## Judging criteria and evidence

Status is read from the repo as it is now: MEASURED (a results file or a passing test in `make smoke`), ESTIMATE (a
simulated rate with its interval, or arithmetic on measured rows), SYNTHETIC (model-generated outcomes), PENDING (the
code or file is not in `integration` yet), UNMEASURED (a run that cannot answer the question). Numbers are quoted
only from results files; every row, with its source key, is in [docs/JUDGING.md](JUDGING.md).

| Criterion | PRD items | Evidence | Status |
|---|---|---|---|
| Correctness and assumptions | R-1, R-2, R-3, R-8 | `core/test_spine.py`, `stats/test_reference.py`, `tests/test_score_contract.py`, `tests/test_reimage_contract.py`, `reimage/test_backends.py`, `audit/test_audit.py` (all in `make smoke`) | MEASURED on CPU and the CUDA simulator; on a real A100, 39 passed, 0 failed (`bench/results/modal-a100-20261121T173127Z.json`, #75), MEASURED |
| Fundamental reshaping | R-4, R-6 | Shared-threshold shortcut: `atlas/test_simulation.py` (#45); fleet sharding, bit-identical to one device: `reimage/test_fleet.py` (#34); fleet timing: 1,000 studies, 843.7 s on 1 A100 against 128.2 s on 8, 6.6× compute span, one sweep (`bench/results/modal-a100-20261121T183709Z.json`, #75) | TESTED; fleet MEASURED |
| Meaningful performance improvement | R-5, R-6 | The race (#113, `bench/results/modal-design-20261121T194458Z.json`): against the lab way on the same 8 cores, Preflight's Numba lane is 7.0× per evaluated study and 4.70× per design point (calibration paid); hybrid ties Numba; different test procedures, one run per arm. Matched end to end (#123, `bench/results/matched-e2e-20261121T201925Z.json`): 42.1× per decided study through scoring, 26.7× with setup, 1 lab core against 8 Preflight cores; calls agree on 89 of 100. GPU kernel 3.19× (#75) | MEASURED (compute); rates ESTIMATE |
| Scientific value | R-4, R-7, R-8 | Schoenfeld formula 80% against simulated 27% [22, 32] at 200 patients (`bench/results/vs-schoenfeld-300.json`, #53); about 524 patients for 80% (`bench/results/power-at-scale.json`, #83); the speed costs power, 15.3% against the lab's 20.3% (#113), and event-count calibration does not close it, +0.2 points [−0.7, +1.1] (#117); Charville v2 maxT p = 0.098 over 3,780 patterns (`results/charville-v2.json`, #47); UPMC locked test inconclusive at 18% simulated power (`results/upmc-rediscovery.json`, #109) | Schoenfeld, power and calibration ESTIMATE; Charville MEASURED, exploratory; UPMC UNMEASURED (underpowered) |

The demo's answer grid, `demo/data/grid.json`, is the Atlas output: 48 design points, 1,000 studies each; power
ESTIMATE, compute MEASURED, outcomes SYNTHETIC.

## Hard lines

- No invented numbers. Every figure carries MEASURED, SYNTHETIC, UNMEASURED, ESTIMATE or ANALYTIC.
- Charville is the only patient-level cohort; UPMC is descriptive apart from one pre-registered locked test (R-16), which is UNMEASURED (underpowered) and inconclusive: `results/upmc-rediscovery.json`.
- Real-data results are exploratory: their outcomes have already been seen.
- Every speed-up names its baseline, hardware and scope: the lab way (a 1,000-shuffle test in every study) for the race
  and the matched run, one A100 for the fleet, the NumPy reference for the kernel. Two different test procedures are
  never called equal work.
