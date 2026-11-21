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
| Live demo | https://atiladeokegab.github.io/sota-parallelism-nov21/ | live once PR #36 merges |
| Offline fallback | `make demo` (checks `demo/data`, then serves `demo/` locally) | in integration |
| Demo video (backup) | `docs/film/` | not yet recorded |
| README | `README.md`: what it is, how to run it, a screenshot from the final run | not yet refreshed |

## Run order on stage

1. Open the page.
2. Set patients and regions per patient.
3. Press "Run the studies". The race plays only on this press; changing the design clears the lanes (#48).
4. Let the CPU-vs-GPU race finish.
5. Point at the "our cohort" marker on the chart.

## Judging criteria and evidence

Status is read from the repo as it is now: MEASURED (a results file or a passing test in `make smoke`), PENDING (the
code or file is not in `integration` yet), UNMEASURED (needs hardware or a run we have not done). Numbers are quoted
only from results files.

| Criterion | PRD items | Evidence | Status |
|---|---|---|---|
| Correctness and assumptions | R-1, R-2, R-3, R-8 | `core/test_spine.py`, `stats/test_reference.py`, `tests/test_score_contract.py`, `tests/test_reimage_contract.py`, `reimage/test_backends.py`, `audit/test_audit.py` (all in `make smoke`) | MEASURED on CPU and the CUDA simulator; agreement on a real GPU UNMEASURED |
| Fundamental reshaping | R-4, R-6 | Shared-threshold shortcut and its nested check: `atlas/test_simulation.py` (branch `5-study-atlas`); fleet sharding, bit-identical to one device: `reimage/test_fleet.py` (PR #34) | PENDING |
| Meaningful performance improvement | R-5, R-6 | `bench/results/ladder-amd-epyc-7b13.json` (SYNTHETIC tissue, AMD EPYC 7B13, 8 logical CPUs): only the NumPy reference rung is timed. Rung 0 (per-study cKDTree recount plus permutation test), Numba all-core and GPU rungs, and fleet studies/s are not in a results file | NumPy reference MEASURED; every speed-up UNMEASURED |
| Scientific value | R-4, R-7, R-8 | `audit/results/charville.json` (patient-level, 159 patients, 38 events, exploratory, already seen); `audit/results/upmc.json` (descriptive only); atlas output and `bench/results/vs-schoenfeld.json` (Schoenfeld vs simulated power) not yet in integration | Audit MEASURED; atlas and Schoenfeld PENDING |

The demo's answer grid, `demo/data/grid.json`, is labelled SYNTHETIC (a toy attenuation model) until the atlas output
replaces it.

## Hard lines

- No invented numbers. Every figure carries MEASURED, SYNTHETIC, UNMEASURED, ESTIMATE or ANALYTIC.
- Charville is the only patient-level cohort; UPMC is descriptive only.
- Real-data results are exploratory: their outcomes have already been seen.
- Every speed-up is quoted against rung 0, what a lab runs today.
