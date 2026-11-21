# SoTA Scientific Parallelism Hackathon (King's Cross, 21-22 Nov 2026)

A GPU engine that re-images thousands of simulated spatial-biomarker studies cell by cell from real tumour tissue (UPMC CODEX head and neck; Charville, the Stanford colorectal cohort), so a lab can see, before it images a single slide, which study designs (patients, regions per patient, panel, imaging error) can actually find a survival-linked spatial pattern like Helsinki's stroma finding. The reshaping (nesting, region pools, one shared null threshold per design point) and a speedup ladder that separates algorithm from hardware against the best adequate CPU method.

Official rules: the event brief

Smoke: `make smoke`

The product that ships is `main`: the last commit that passed the smoke check.

Board: https://github.com/users/atiladeokegab/projects/15

## Deadlines

Your agent checks these every session against the UTC column. GitHub milestones keep only the
date, so this UTC column is the clock, never the milestone.

| Deadline | Event time | UTC |
|---|---|---|
| spine | 2026-11-21T13:00+00:00 | 2026-11-21T13:00Z |
| P1 green or cut | 2026-11-21T20:00+00:00 | 2026-11-21T20:00Z |
| code freeze | 2026-11-22T16:00+00:00 | 2026-11-22T16:00Z |
| submit | 2026-11-22T17:00+00:00 | 2026-11-22T17:00Z |

## Judging criteria

- Correctness and assumptions
- Fundamental reshaping
- Meaningful performance improvement
- Scientific value

## Team

| Name | GitHub | Role | Bring |
|---|---|---|---|
| Atilade | @atiladeokegab | lead, designer, electronic engineering, AI agents, product, the pitch | the idea, the agents |
| Faisal | @FHL-08 | to confirm in the room | to confirm |

The lead's agents: Zeus, Prometheus, Atlas, Hephaestus. When this page or a review says "the lead", it may be one of them acting for the lead.
