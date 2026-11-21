"""Strong scaling of build-9's Numba arm (bench/design_batch.py): pure helpers for bench/modal_scaling.py.

The same fixed null studies (design_batch seeds, stream "null", i = 0..n-1) run at each core count, one
Modal CPU container per count, workers = cores, cpu=(cores, cores) as a hard limit. Reported: studies wall,
wall per study and the speedup vs 1 core (a ratio of measured walls). Parity: every count's X sha256 and
z sha256 per study equal the 8-core point's, bit for bit. Anything short of that is FAILED with no timings.
"""
from __future__ import annotations

import hashlib

import numpy as np

CORES = (1, 2, 4, 8, 16, 32, 64)
REFERENCE_CORES = 8  # build-9's numba arm ran 8 workers
# modal.com/pricing read 2026-11-21T17:36Z (gpu/modal_app.py PRICING): preemptible Function list price, USD/s.
CPU_CORE_USD_S, MEMORY_GIB_USD_S = 0.0000131, 0.00000222
CAP_USD = 5.0


def parse_cores(text: str) -> tuple[int, ...]:
    """Distinct whole core counts in 1..64, sorted, including 1 (the speedup base) and 8 (the parity base)."""
    try:
        cores = [int(c) for c in text.split(",") if c.strip()]
    except ValueError:
        raise SystemExit(f"--cores must be whole numbers: {text!r}") from None
    if len(set(cores)) != len(cores) or not all(1 <= c <= 64 for c in cores):
        raise SystemExit(f"--cores must be distinct counts in 1..64: {text!r}")
    if not {1, REFERENCE_CORES} <= set(cores):
        raise SystemExit(f"--cores must include 1 and {REFERENCE_CORES}: {text!r}")
    return tuple(sorted(cores))


def memory_gib(cores: int) -> int:
    return max(16, 2 * cores)


def timeout_s(cores: int, studies: int) -> int:
    # shortcut: ~1 CPU-s per study (build-9's numba arm: 2,936 CPU-s re-imaging for 3,000 studies) x 3, plus 15 min
    return int(900 + 3 * studies / cores)


def usd_per_s(cores: int) -> float:
    return cores * CPU_CORE_USD_S + memory_gib(cores) * MEMORY_GIB_USD_S


def ceiling_usd(cores: tuple[int, ...], studies: int) -> float:
    """Every container running to its timeout: the most this sweep can bill."""
    return round(sum(timeout_s(c, studies) * usd_per_s(c) for c in cores), 2)


def check_cap(cores: tuple[int, ...], studies: int, spent_usd: float, cap_usd: float = CAP_USD) -> float:
    ceiling = ceiling_usd(cores, studies)
    if spent_usd + ceiling > cap_usd:
        raise SystemExit(f"ceiling ${ceiling} + ${spent_usd} already spent > ${cap_usd} cap: ask the lead first")
    return ceiling


def z_sha(z) -> str:
    return hashlib.sha256(np.ascontiguousarray(z, dtype=np.float64).tobytes()).hexdigest()


def parity(points: dict) -> dict:
    """Each core count's per-study X sha256 and z sha256 against the 8-core point's."""
    ref = {r["i"]: r for r in points[REFERENCE_CORES]["records"]}
    out = {"reference_cores": REFERENCE_CORES, "basis": "sha256 of X and of z per study, bit for bit", "by_cores": {}}
    for c, p in points.items():
        recs = {r["i"]: r for r in p["records"]}
        same_set = set(recs) == set(ref)
        out["by_cores"][c] = {"studies": len(recs), "same_studies": same_set,
                              "x_identical": same_set and all(recs[i]["x_sha"] == ref[i]["x_sha"] for i in ref),
                              "z_identical": same_set and all(recs[i]["z_sha"] == ref[i]["z_sha"] for i in ref)}
    out["all"] = all(v["same_studies"] and v["x_identical"] and v["z_identical"] for v in out["by_cores"].values())
    return out


def prior_parity(records: list, prior: dict) -> dict:
    """The 8-core point against build-9's committed numba arm: X sha256 and maxT per shared null study."""
    old = {s["i"]: s for s in prior["arms"]["numba"]["studies"] if s["stream"] == "null"}
    common = [r for r in records if r["i"] in old]
    return {"x_identical": bool(common) and all(r["x_sha"] == old[r["i"]]["x_sha"] for r in common),
            "maxT_identical": bool(common) and all(r["maxT"] == old[r["i"]]["maxT"] for r in common),
            "compared": len(common)}


def curve(points: dict) -> list:
    """Wall per study and the speedup vs 1 core, from each point's in-container studies wall."""
    base = points[1]["studies_wall_s"]
    return [{"cores": c, "workers": p["workers"], "studies": len(p["records"]), "studies_wall_s": p["studies_wall_s"],
             "wall_per_study_s": p["studies_wall_s"] / len(p["records"]), "speedup_vs_1_core": base / p["studies_wall_s"]}
            for c, p in sorted(points.items())]


def result(points: dict, cores: tuple[int, ...], errors: dict, prior: dict | None) -> dict:
    """MEASURED only when every count ran, parity holds and workers == cores; else FAILED with no timings."""
    failures = [f"cores {c}: {e}" for c, e in errors.items()]
    failures += [f"cores {c}: missing" for c in cores if c not in points and c not in errors]
    failures += [f"cores {c}: workers {p['workers']}" for c, p in points.items() if p["workers"] != c]
    par = parity(points) if not failures else {"all": False}
    if not failures and not par["all"]:
        failures.append("parity: X or z differs from the 8-core point")
    if prior is not None and REFERENCE_CORES in points:
        par["against_build9_8core"] = prior_parity(points[REFERENCE_CORES]["records"], prior)
    out = {"status": "FAILED" if failures else "MEASURED", "failures": failures, "parity": par}
    if not failures:
        out["curve"] = curve(points)
    return out
