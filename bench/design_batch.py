"""The batched design point: S Charville studies at one design, three arms, FWER and power.

Design: bench.real_cpu.DESIGN (159 patients drawn with replacement, 2 regions of each, a 400 um field
of view, 10% mis-calls). Outcomes: atlas.simulation's Weibull model (#5, pinned), planted on the
first usable truth feature at HR 2 per source SD for the alternative, HR 1 for the nulls.

Three seed streams, disjoint by construction (SeedSequence([ROOT, stream, i])): `calibration`
nulls set the batched arms' one design threshold (stats.threshold), `null` studies measure FWER and
`alternative` studies measure power. Each study's stream gives its image, survival and censoring
seeds, so every arm sees the same studies.

Arms:
  lab    per study: fresh cKDTree re-imaging, then a 1,000-shuffle conditional maxT test
         (stats.permutation_test); call = p <= alpha. Needs no calibration studies.
  numba  per study: CSR Numba re-imaging, then the R-15 Cox score (screen_batch, NumPy).
  gpu    per study: CSR re-imaging with the numba.cuda kernel; then every study's Cox score
         in blocks on the GPU (stats.screen.screen_studies).
  hybrid the numba arm's re-imaging on the CPU pool feeding the gpu arm's batched GPU scoring.
  fused  the numba arm with reimage.fused: mis-call, crop and counting in one pass over visible focal cells.
CPU stages run studies in a pool of single-thread workers; the gpu arm is one process.
Parity: X bit-identical across arms (sha256 per study) and z of gpu and hybrid vs numba within 1e-9.
"""
from __future__ import annotations

import hashlib
import math
import os
from time import perf_counter

import numpy as np

ROOT = 20261121
STREAMS = {"calibration": 101, "null": 202, "alternative": 303}
ALPHA, HR, LAB_RESAMPLES = 0.05, 2.0, 1000
Z_TOL = 1e-9
CPU_REIMAGE = ("numba", "hybrid", "fused")  # Numba CSR re-imaging on the CPU pool
CPU_SCORE = ("numba", "fused")  # per-study NumPy screen_batch in the pool
GPU_SCORE = ("gpu", "hybrid")  # batched screen_studies on the GPU
# atlas (#5) is not on this branch yet: its outcome model is read from this commit (ATLAS_SIMULATION = a copy).
ATLAS_PIN = ("79a77926aa0c07fc2c85574736f0f1396767ded0", "atlas/simulation.py")


def simulation():
    """atlas.simulation from this tree once #5 lands, else the pinned copy at $ATLAS_SIMULATION."""
    try:
        from atlas import simulation as module
        return module
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("atlas_simulation_pinned", os.environ["ATLAS_SIMULATION"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def design():
    from bench.real_cpu import DESIGN
    return dict(DESIGN)


def seeds(stream: str, i: int) -> tuple[int, int, int]:
    """(image, survival, censoring) seeds of study i in a stream."""
    return tuple(int(v) for v in np.random.SeedSequence([ROOT, STREAMS[stream], i]).generate_state(3, np.uint64))


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided 95% Wilson score interval for k successes in n."""
    if not 0 <= k <= n or n < 1:
        raise ValueError("need 0 <= k <= n and n >= 1")
    p, d = k / n, 1 + z * z / n
    mid, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, mid - half), min(1.0, mid + half)


def rate(flags) -> dict:
    flags = np.asarray(flags, bool)
    lo, hi = wilson(int(flags.sum()), len(flags))
    return {"rate": float(flags.mean()), "k": int(flags.sum()), "n": len(flags), "wilson95": [lo, hi]}


def sha(X) -> str:
    return hashlib.sha256(np.ascontiguousarray(X, dtype=np.float64).tobytes()).hexdigest()


# ---- per-process state (fork-shared with pool workers) ----
_S: dict = {}


def prepare(arm: str, cells: dict, plan: dict) -> dict:
    """Outcome models, every study's draws and outcomes, and (CSR arms) the cohort's adjacency, once."""
    from core.reimage_ref import draw
    from core.truth import truth

    t = perf_counter()
    sim = simulation()
    fit_model, outcomes = sim.fit_model, sim.outcomes
    reference = truth(cells)
    models = {"null": fit_model(cells, reference, hr=1.0), "alternative": fit_model(cells, reference, hr=HR)}
    d = design()
    studies = {}
    for stream, n in plan.items():
        rows = []
        for i in range(n):
            img, surv, cens = seeds(stream, i)
            drawn = draw(cells, d, img)
            model = models["alternative" if stream == "alternative" else "null"]
            time, event, strata = outcomes(model, drawn["patients"], surv, cens)
            rows.append({"i": i, "image_seed": img, "time": time, "event": event.astype(float), "strata": strata,
                         "regions": sorted({r for look in drawn["looks"] for r, _, _ in look})})
        studies[stream] = rows
    prep_s = perf_counter() - t
    t = perf_counter()
    if arm == "lab":
        cells["_nbr"] = {r: None for r in range(len(cells["patient_of_region"]))}  # the tree path builds no CSR
    else:
        from core.truth import layout, neighbours
        L = layout(cells)
        need = sorted({r for rows in studies.values() for s in rows for r in s["regions"]})
        cells["_nbr"] = {r: neighbours(L["xy"](r)) for r in need}
        from reimage import cpu, gpu
        module = cpu if arm in CPU_REIMAGE else gpu  # fused.prepare is cpu.prepare
        _S["prepared"] = {id(cells["_nbr"][r]): module.prepare(cells["_nbr"][r], L["K"]) for r in need}
    adjacency_s = perf_counter() - t
    _S.update(cells=cells, studies=studies, arm=arm, plant=models["alternative"]["feature"])
    return {"prep_s": prep_s, "adjacency_s": adjacency_s, "plant_feature": models["alternative"]["feature"],
            "expected_event_fraction": {k: m["expected_event_fraction"] for k, m in models.items()}}


def _kernel(arm: str):
    if arm == "lab":
        from bench.lab_today import tree_region
        return tree_region
    from reimage import cpu, fused, gpu
    module = fused if arm == "fused" else cpu if arm in CPU_REIMAGE else gpu
    prepared = _S["prepared"]  # adjacency and CSR built once per cohort in prepare()

    def region(nbr, xy, types, focal, K, **kw):
        return module.reimage_region(nbr, xy, types, focal, K, prepared=prepared[id(nbr)], **kw)
    return region


def one_study(job: tuple[str, int]) -> dict:
    """Re-image one study and (CPU arms) score it; returns the record and stage seconds."""
    from core.reimage_ref import reimage
    stream, i = job
    s = _S["studies"][stream][i]
    kernel = _S.get("kernel") or _S.setdefault("kernel", _kernel(_S["arm"]))
    t = perf_counter()
    X = reimage(_S["cells"], design(), s["image_seed"], region_kernel=kernel)
    rec = {"stream": stream, "i": i, "x_sha": sha(X), "reimage_s": perf_counter() - t}
    t = perf_counter()
    if _S["arm"] == "lab":
        from stats import permutation_test
        r = permutation_test(X, s["time"], s["event"], s["strata"], n_resamples=LAB_RESAMPLES, seed=s["image_seed"])
        rec.update(p=float(r["p"]), maxT=float(r["maxT"]), test_s=perf_counter() - t)
    elif _S["arm"] in CPU_SCORE:
        from stats import screen_batch
        z = screen_batch(X, s["time"][None], s["event"][None], strata=s["strata"], backend="numpy")["z"][0]
        rec.update(z=z, score_s=perf_counter() - t)
    else:
        rec["X"] = X
    return rec


def warm(arm: str) -> float:
    """JIT, CUDA context and BLAS warm-up on the first planned study, recomputed later; excluded from timings."""
    from core.reimage_ref import reimage
    first = next(rows[0] for rows in _S["studies"].values() if rows)
    t = perf_counter()
    X = reimage(_S["cells"], design(), first["image_seed"], region_kernel=_S.setdefault("kernel", _kernel(arm)))
    if arm == "gpu":
        warm_score(first, X)
    return perf_counter() - t


def warm_score(first: dict, X) -> None:
    from stats.screen import screen_studies
    screen_studies(X[None], first["time"][None], first["event"][None], first["strata"][None], backend="gpu")


def run_arm(arm: str, cells: dict, plan: dict, workers: int = 8, block: int = 128) -> dict:
    """Time one arm over the plan {stream: studies}; studies run in stream order."""
    import multiprocessing as mp

    setup = prepare(arm, cells, plan)
    if arm == "lab":
        plan = {k: v for k, v in plan.items() if k != "calibration"}  # the lab calibrates per study
    warm_s = warm(arm) if arm != "lab" else None  # before the fork: workers inherit the JIT (hybrid: no CUDA yet)
    jobs = [(stream, i) for stream, n in plan.items() for i in range(n)]
    t = perf_counter()
    if arm == "gpu":
        recs = [one_study(j) for j in jobs]
    else:
        from concurrent.futures import ProcessPoolExecutor  # a dead worker raises instead of hanging, unlike mp.Pool
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"), initializer=_single_thread) as pool:
            recs = list(pool.map(one_study, jobs))
    studies_wall = perf_counter() - t
    score_wall = warm_score_s = None
    if arm == "hybrid":  # CUDA starts only after the pool has closed: forking a CUDA process is unsafe
        first = next(rows[0] for rows in _S["studies"].values() if rows)
        t = perf_counter()
        warm_score(first, recs[0]["X"])  # jobs[0] is the first planned study
        warm_score_s = perf_counter() - t
    if arm in GPU_SCORE:
        from stats.screen import screen_studies
        t = perf_counter()
        for stream in plan:
            rows = [r for r in recs if r["stream"] == stream]
            src = _S["studies"][stream]
            out = screen_studies(np.stack([r.pop("X") for r in rows]), np.stack([src[r["i"]]["time"] for r in rows]),
                                 np.stack([src[r["i"]]["event"] for r in rows]),
                                 np.stack([src[r["i"]]["strata"] for r in rows]), backend="gpu", block=block)
            for r, z in zip(rows, out["z"]):
                r["z"] = z
        score_wall = perf_counter() - t
    return {"arm": arm, "setup": setup, "warm_s": warm_s, "workers": 1 if arm == "gpu" else workers,
            "warm_score_s": warm_score_s,
            "studies_wall_s": studies_wall, "batched_score_wall_s": score_wall, "records": recs}


def _single_thread():
    os.environ.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    try:
        import numba
        numba.set_num_threads(1)
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def summarize(arm_out: dict, plant: int) -> dict:
    """FWER and power with Wilson 95% intervals; batched arms use one threshold from the calibration stream."""
    from stats import threshold
    recs = arm_out["records"]
    by = {s: sorted((r for r in recs if r["stream"] == s), key=lambda r: r["i"]) for s in STREAMS}
    out = {"studies": {s: len(v) for s, v in by.items()}}
    if arm_out["arm"] == "lab":
        call = {s: [r["p"] <= ALPHA for r in by[s]] for s in ("null", "alternative")}
        out["rule"] = f"per-study conditional maxT, {LAB_RESAMPLES} within-strata shuffles, p <= {ALPHA}"
    else:
        cut = float(threshold({"alpha": ALPHA}, np.stack([r["z"] for r in by["calibration"]])))
        call = {s: [float(np.abs(r["z"]).max()) > cut for r in by[s]] for s in ("null", "alternative")}
        out["rule"] = f"maxT > design threshold from {len(by['calibration'])} independent calibration nulls"
        out["threshold"] = cut if np.isfinite(cut) else None  # None: too few calibration studies, nothing rejects
        out["identification"] = rate([c and int(np.argmax(np.abs(r["z"]))) == plant
                                      for c, r in zip(call["alternative"], by["alternative"])])
    out["fwer"], out["power"] = rate(call["null"]), rate(call["alternative"])
    return out


def parity(arms: dict) -> dict:
    """X bit-identical across every arm on shared studies; numba and gpu z within Z_TOL."""
    keyed = {a: {(r["stream"], r["i"]): r for r in o["records"]} for a, o in arms.items()}
    names = list(keyed)
    x_ok, x_n = True, 0
    for a in names[1:]:
        for k, r in keyed[a].items():
            if k in keyed[names[0]]:
                x_n += 1
                x_ok &= r["x_sha"] == keyed[names[0]][k]["x_sha"]
    res = {"x_bitwise": bool(x_ok and x_n), "x_compared": x_n}
    z_ok = True
    for other in (a for a in GPU_SCORE if a in keyed and "numba" in keyed):
        common = sorted(set(keyed["numba"]) & set(keyed[other]))
        dz = max(float(np.max(np.abs(keyed["numba"][k]["z"] - keyed[other][k]["z"])
                       / np.maximum(1.0, np.abs(keyed["numba"][k]["z"])))) for k in common)
        res[f"z_{other}_vs_numba"] = {"max_rel_diff": dz, "within_tol": bool(dz <= Z_TOL), "compared": len(common)}
        z_ok &= dz <= Z_TOL
    if "numba" in keyed and any(a in keyed for a in GPU_SCORE):
        res["z_tol"] = Z_TOL
    res["all"] = bool(res["x_bitwise"] and z_ok)
    return res


# Per-study stage times are perf_counter wall seconds inside single-thread pool workers, summed over studies:
# worker wall, not CPU time. Files before 2026-11-21T20:20Z named them *_cpu_s.
LEGACY = {"reimage_cpu_s": "reimage_worker_wall_s", "scoring_cpu_s": "scoring_worker_wall_s",
          "scoring_and_calibration_cpu_s": "scoring_and_calibration_worker_wall_s"}
COMPUTE_SCOPE = ("compute_wall_s = re-imaging every evaluated study (the pool's wall for CPU-reimaged arms) plus "
                 "batched scoring; excludes data load, outcome prep, adjacency, warm-up and the local threshold-and-"
                 "calls summary, which end_to_end_in_container_s adds back (all but the summary)")


def stage_times(arm_out: dict) -> dict:
    recs, arm = arm_out["records"], arm_out["arm"]
    st = {"prep_s": arm_out["setup"]["prep_s"], "adjacency_s": arm_out["setup"]["adjacency_s"],
          "reimage_worker_wall_s": sum(r["reimage_s"] for r in recs), "studies_wall_s": arm_out["studies_wall_s"]}
    if arm == "lab":
        st["scoring_and_calibration_worker_wall_s"] = sum(r["test_s"] for r in recs)
    elif arm in CPU_SCORE:
        st["scoring_worker_wall_s"] = sum(r["score_s"] for r in recs)
        st["calibration_s"] = "the calibration stream's studies, inside reimage and scoring"
    else:
        st["scoring_wall_s"] = arm_out["batched_score_wall_s"]
        st["calibration_s"] = "the calibration stream's studies, inside reimage and scoring"
    st["evaluated_studies"] = len(recs)
    return complete(st, arm_out.get("load_s"), arm_out["warm_s"], arm_out.get("warm_score_s"))


def complete(st: dict, load_s, warm_s, warm_score_s) -> dict:
    """Legacy names renamed; the compute scope and the in-container end-to-end time derived from recorded stages."""
    st = {LEGACY.get(k, k): v for k, v in st.items()}
    st["compute_wall_s"] = st["studies_wall_s"] + (st.get("scoring_wall_s") or 0)
    st["wall_per_study_s"] = st["compute_wall_s"] / st["evaluated_studies"]
    st["end_to_end_in_container_s"] = (st["compute_wall_s"] + st["prep_s"] + st["adjacency_s"]
                                       + (load_s or 0) + (warm_s or 0) + (warm_score_s or 0))
    st["scope"] = COMPUTE_SCOPE
    return st


def prior_parity(records: list, prior_arms: dict) -> dict:
    """A new arm against a committed result's arms: X sha256 per shared study; batched maxT within Z_TOL."""
    new = {(r["stream"], r["i"]): r for r in records}
    out, ok = {}, True
    for a, o in prior_arms.items():
        old = {(s["stream"], s["i"]): s for s in o["studies"]}
        common = [k for k in new if k in old]
        e = {"x_bitwise": bool(common) and all(new[k]["x_sha"] == old[k]["x_sha"] for k in common),
             "x_compared": len(common)}
        if a != "lab" and common:  # the lab's maxT is its permutation statistic, not compared
            d = max(abs(float(np.abs(new[k]["z"]).max()) - old[k]["maxT"]) / max(1.0, old[k]["maxT"]) for k in common)
            e.update(maxT_max_rel_diff=d, maxT_within_tol=bool(d <= Z_TOL))
        ok &= e["x_bitwise"] and e.get("maxT_within_tol", True)
        out[a] = e
    out["all"] = bool(ok)
    return out


def split(arm: str, stages: dict) -> dict:
    """Re-imaging vs scoring. GPU-scored arms: wall of each stage. CPU arms: worker wall summed over studies."""
    if arm in GPU_SCORE:
        r, sc, basis = stages["studies_wall_s"], stages["scoring_wall_s"], "wall seconds of each stage"
    else:
        r = stages["reimage_worker_wall_s"]
        basis = "worker wall seconds (perf_counter in single-thread workers) summed over studies; not CPU time"
        sc = stages["scoring_worker_wall_s"] if arm in CPU_SCORE else stages["scoring_and_calibration_worker_wall_s"]
    return {"reimage_s": r, "scoring_s": sc, "basis": basis, "reimage_share": r / (r + sc),
            "dominant": "re-imaging" if r >= sc else "scoring"}


def headline(arms: dict) -> str:
    """Which stage dominates, and wall per study, from each arm's split and stages."""
    dom = {a: v["split"]["dominant"] for a, v in arms.items()}
    shares = ", ".join(f"{a} {100 * v['split']['reimage_share']:.1f}%" for a, v in arms.items())
    walls = ", ".join(f"{a} {v['stages']['wall_per_study_s']:.3f} s" for a, v in arms.items())
    stage = dom[next(iter(dom))] if len(set(dom.values())) == 1 else "mixed"
    lead = f"{stage.capitalize()} dominates every arm" if stage != "mixed" else f"Dominant stage differs by arm: {dom}"
    return (f"{lead} (re-imaging share: {shares}). Compute wall per evaluated study (re-imaging and scoring only; "
            f"load, prep, adjacency, warm-up and the summary excluded): {walls}.")
