"""Recalibrate the frozen race maxima, without re-imaging or changing the statistic.

uv run python -m bench.event_count_calibration --baseline <frozen race JSON> --data-dir <cached data>

Primary: E <= 38 versus E > 38, fixed from the source's 38/159 event fraction before evaluation.
Secondary: exact event count. Sparse or unseen calibration bins never reject. Both use only stream101;
streams202/303 evaluate null rejection/power on the original study IDs. The lab statistic is different.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from stats.reference import threshold

ROOT = Path(__file__).resolve().parents[1]
ALPHA, SPLIT, PATIENTS = 0.05, 38, 159
SEED, STREAMS = 20261121, {"calibration": 101, "null": 202, "alternative": 303}
BASELINE_SHA = "5010130eec31466a67c1993d3246cc589cda5542d057470f79584a7298c29ad9"
SOURCE_SHA = "2ca08fb03ed2ecd51e552a5ab1bbe6caede58f54c07971ed35af08c10db4aa1b"
GRAMMAR_SHA = "772f6a845266470456af96f92d0573cc1bec72432ad74d789439baff066c2156"
REFERENCE_SHA = "b9ec05a23cc25ddf89f3772f61df540cd67d8597d9f3e55b19239f97e26e0e25"
PINS = {
    "atlas/simulation.py": "b3503f0ad72ae53bc039067b6c8b4bf8eb9be409c52103867c90b3e28582f39a",
    "core/reimage_ref.py": "ca0436dfad0691325d0fadf2d8624dfcddf8e25224a58b7f6ad87f1da53e1143",
    "core/truth.py": "64586fb870a504563a89966833a602ff865925cb99ae70a0c82d27c6d4277af8",
    "core/grammar.py": "1e830dda303a1071999dfee8de7065bc8341445e732b487e89a9efc6baa8aaf2",
}


def _validate(records):
    seen = set()
    for r in records:
        key = (r["stream"], r["i"])
        if (r["stream"] not in STREAMS or type(r["i"]) is not int or r["i"] < 0 or key in seen
                or type(r["events"]) is not int or not 0 <= r["events"] <= PATIENTS
                or isinstance(r["maxT"], bool) or not math.isfinite(r["maxT"]) or r["maxT"] < 0):
            raise ValueError("invalid or duplicate study identity, event count or maximum")
        seen.add(key)


def _bin(events, mode):
    if mode == "pooled":
        return "all"
    if mode == "two_bins":
        return "low" if events <= SPLIT else "high"
    if mode == "exact":
        return str(events)
    raise ValueError("mode must be pooled, two_bins or exact")


def fit_thresholds(calibration, mode="two_bins"):
    """Fit independent calibration rows only; None serializes an infinite, non-rejecting cutoff."""
    _validate(calibration)
    if not calibration or any(r["stream"] != "calibration" for r in calibration):
        raise ValueError("fit requires calibration rows only")
    groups = {"all": []} if mode == "pooled" else {"low": [], "high": []} if mode == "two_bins" else {}
    for r in calibration:
        groups.setdefault(_bin(r["events"], mode), []).append(r["maxT"])
    table = {}
    for key, maxima in sorted(groups.items()):
        n = len(maxima)
        rank = math.ceil((n + 1) * (1 - ALPHA))
        cut = threshold({"alpha": ALPHA}, np.array(maxima)[:, None]) if n >= 2 else math.inf
        table[key] = {"n": n, "rank": rank, "threshold": float(cut) if math.isfinite(cut) else None}
    return {"mode": mode, "split": SPLIT if mode == "two_bins" else None, "bins": table}


def reject(table, record):
    _validate([record])
    cut = table["bins"].get(_bin(record["events"], table["mode"]), {}).get("threshold")
    return cut is not None and record["maxT"] > cut  # strict: ties do not reject


def rate(flags):
    """Wilson 95% interval for observed rejection frequency, conditional on the fitted cutoffs."""
    n, k = len(flags), int(np.sum(flags))
    if not n:
        raise ValueError("a rate needs at least one study")
    p, z = k / n, 1.959963984540054
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": k, "n": n, "rate": p, "wilson95": [max(0.0, mid - half), min(1.0, mid + half)]}


def paired_difference(flags, baseline):
    """Seeded paired bootstrap over evaluation IDs; calibration is held fixed."""
    delta = np.asarray(flags, int) - np.asarray(baseline, int)
    counts = np.array([np.sum(delta == v) for v in (-1, 0, 1)])
    draws = np.random.default_rng(404).multinomial(len(delta), counts / len(delta), size=20000)
    lo, hi = np.quantile((draws[:, 2] - draws[:, 0]) / len(delta), [0.025, 0.975])
    return {"wins": int(counts[2]), "losses": int(counts[0]), "n": len(delta), "difference": float(delta.mean()),
            "paired_bootstrap95": [float(lo), float(hi)], "bootstrap_seed": 404, "bootstrap_draws": 20000}


def analyze(records):
    _validate(records)
    by = {s: sorted((r for r in records if r["stream"] == s), key=lambda r: r["i"]) for s in STREAMS}
    for s, rows in by.items():
        if not rows or [r["i"] for r in rows] != list(range(len(rows))):
            raise ValueError(f"{s}: missing or noncontiguous study IDs")
    for s in ("null", "alternative"):
        if any(isinstance(r.get("lab_p"), bool) or not isinstance(r.get("lab_p"), (int, float))
               or not math.isfinite(r["lab_p"]) or not 0 <= r["lab_p"] <= 1 for r in by[s]):
            raise ValueError("invalid or missing paired lab p-value")
    tables = {m: fit_thresholds(by["calibration"], m) for m in ("pooled", "two_bins", "exact")}
    calls = {m: {s: [reject(t, r) for r in by[s]] for s in ("null", "alternative")} for m, t in tables.items()}
    calls["lab"] = {s: [r["lab_p"] <= ALPHA for r in by[s]] for s in ("null", "alternative")}
    methods = {}
    for name, stages in calls.items():
        out = {"fwer": rate(stages["null"]), "power": rate(stages["alternative"])}
        if name in tables:
            out["calibration"] = tables[name]
        if name in ("two_bins", "exact"):
            out["paired_vs_pooled"] = {s: paired_difference(stages[s], calls["pooled"][s]) for s in stages}
            out["paired_vs_lab"] = {s: paired_difference(stages[s], calls["lab"][s]) for s in stages}
            keys = sorted({_bin(r["events"], name) for r in by["null"]})
            out["null_by_bin"] = {k: rate([c for r, c in zip(by["null"], stages["null"])
                                           if _bin(r["events"], name) == k]) for k in keys}
            out["unsupported_evaluation"] = {
                s: sum(tables[name]["bins"].get(_bin(r["events"], name), {}).get("threshold") is None for r in by[s])
                for s in stages}
        methods[name] = out
    return {"primary": "two_bins", "secondary": "exact", "methods": methods,
            "event_counts": {s: {"n": len(rows), "total": sum(r["events"] for r in rows),
                                  "min": min(r["events"] for r in rows), "max": max(r["events"] for r in rows)}
                             for s, rows in by.items()}}


def reconstruct(baseline_path, data_dir):
    """Reproduce outcome draws from hash-verified source/model; reuse the original score maxima."""
    from atlas import simulation
    from atlas.artifacts import array_hash, source_metadata
    from core.cohorts import load
    from core.grammar import grammar, grammar_hash
    from core.reimage_ref import draw

    raw = Path(baseline_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASELINE_SHA:
        raise ValueError("baseline is not the frozen race JSON")
    baseline = json.loads(raw)
    for rel, want in PINS.items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != want:
            raise ValueError(f"{rel}: generating code differs from the pinned race")
    if not (Path(data_dir) / "charville_cells.npz").is_file():
        raise ValueError("requires the existing Charville cache; never fetches or builds source data")
    cells = load("charville", data_dir=Path(data_dir))
    source = source_metadata(cells)
    if source["source_hash"] != SOURCE_SHA or grammar_hash(grammar(cells["type_names"])) != GRAMMAR_SHA:
        raise ValueError("Charville source or grammar differs from the frozen source")
    with np.load(Path(data_dir) / "atlas-cache/charville-reference.npz", allow_pickle=False) as cache:
        reference = cache["X"]
        if (str(cache["source_hash"]), str(cache["grammar_hash"])) != (SOURCE_SHA, GRAMMAR_SHA):
            raise ValueError("reference cache belongs to another source/grammar")
    if (reference.shape != (PATIENTS, 630) or reference.dtype != np.float64
            or hashlib.sha256(reference.tobytes()).hexdigest() != REFERENCE_SHA):
        raise ValueError("reference values differ from the frozen model")
    models = {s: simulation.fit_model(cells, reference, hr=2.0 if s == "alternative" else 1.0)
              for s in ("null", "alternative")}
    if any(m["feature"] != 0 or not math.isclose(m["expected_event_fraction"], 38 / 159, abs_tol=1e-12)
           for m in models.values()):
        raise ValueError("outcome model differs from the race setup")
    keyed = {a: {(r["stream"], r["i"]): r for r in baseline["arms"][a]["studies"]}
             for a in ("lab", "numba", "gpu")}
    expected = {(s, i) for s in STREAMS for i in range(1000)}
    if (len(baseline["arms"]["numba"]["studies"]) != 3000 or set(keyed["numba"]) != expected
            or len(baseline["arms"]["gpu"]["studies"]) != 3000 or set(keyed["gpu"]) != expected
            or len(baseline["arms"]["lab"]["studies"]) != 2000
            or set(keyed["lab"]) != {(s, i) for s, i in expected if s != "calibration"}):
        raise ValueError("baseline stream IDs are duplicated, missing or unexpected")
    records = []
    for stream, stream_seed in STREAMS.items():
        model = models["alternative" if stream == "alternative" else "null"]
        for i in range(1000):
            key = stream, i
            original = keyed["numba"][key]
            gpu = keyed["gpu"][key]
            if original["x_sha"] != gpu["x_sha"] or not math.isclose(original["maxT"], gpu["maxT"], abs_tol=1e-9):
                raise ValueError("paired CPU/GPU study identities or statistics disagree")
            img, surv, cens = (int(v) for v in np.random.SeedSequence([SEED, stream_seed, i]).generate_state(3, np.uint64))
            # Patient sampling is the first RNG operation in the pinned draw; later region/crop draws cannot affect it.
            patients = np.random.default_rng(img).integers(0, PATIENTS, PATIENTS)
            if i in (0, 999):
                d = {k: baseline["design"][k] for k in ("patients", "regions_per_patient", "fov_um", "eps")}
                if not np.array_equal(patients, draw(cells, d, img)["patients"]):
                    raise ValueError("shortcut patient sampling differs from the producer draw")
            time, event, strata = simulation.outcomes(model, patients, surv, cens)
            record = {"stream": stream, "i": i, "events": int(event.sum()), "maxT": original["maxT"],
                      "x_sha": original["x_sha"], "image_seed": img, "survival_seed": surv, "censor_seed": cens,
                      "patients_hash": array_hash(patients),
                      "outcomes_hash": hashlib.sha256("".join(array_hash(a) for a in (time, event, strata)).encode()).hexdigest(),
                      "events_by_stratum": {str(int(g)): int(event[strata == g].sum()) for g in np.unique(strata)}}
            if stream != "calibration":
                lab = keyed["lab"][key]
                if lab["x_sha"] != original["x_sha"]:
                    raise ValueError("lab and screen studies are not paired")
                record["lab_p"] = lab["p"]
            records.append(record)
    return records, {"baseline_sha256": BASELINE_SHA, "baseline_producer": baseline["source_commit"],
                     "baseline_url": "https://github.com/atiladeokegab/sota-parallelism-nov21/blob/c4a766edf18705e164eb476ce00efee6881aae18/bench/results/modal-design-20261121T185537Z.json",
                     "generating_file_sha256": PINS, "source": source, "grammar_sha256": GRAMMAR_SHA,
                     "reference_sha256": REFERENCE_SHA, "seed": SEED, "streams": STREAMS,
                     "models": {k: {f: m[f] for f in ("feature", "source_mean", "source_sd", "shape", "hr", "log_scale",
                                                      "expected_event_fraction", "censoring_tail_policy")}
                                for k, m in models.items()},
                     "reconstruction_limit": "Original race omitted outcome hashes/event counts. Counts are reconstructed from pinned code and verified source, not compared with original saved outcomes. Six patient draws match the full producer draw."}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "bench/results/event-count-calibration.json")
    args = ap.parse_args(argv)
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    started = perf_counter()
    records, provenance = reconstruct(args.baseline, args.data_dir)
    report = analyze(records)
    if [(report["methods"][m]["fwer"]["k"], report["methods"][m]["power"]["k"]) for m in ("pooled", "lab")] != [(46, 153), (50, 203)]:
        raise ValueError("reused records do not reproduce the original two comparison arms")
    if report["methods"]["pooled"]["calibration"]["bins"]["all"]["threshold"] != 4.280528180733615:
        raise ValueError("pooled threshold does not reproduce the original calibration")
    revision = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    report.update(schema="event-count-calibration/1", status="MEASURED", labels={"outcomes": "SYNTHETIC", "power": "ESTIMATE", "fwer": "ESTIMATE", "counts_and_compute": "MEASURED"},
                  provenance=provenance, records=records,
                  execution={"source_commit": revision or "unknown", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             "wall_s": perf_counter() - started, "machine": platform.node(), "affinity_cores": 1,
                             "python": platform.python_version(), "software": {k: version(k) for k in ("numpy", "scipy")},
                             "modal_runs": 0, "modal_spend_usd": 0},
                  assumptions=["Empirical Charville source, Weibull HR2/sourceSD alternative; source repeated with replacement.",
                               "Primary bins E<=38 versus E>38 and secondary exact-E rule fixed before evaluation; no method selected on observed performance.",
                               "Within each fixed bin, ceil((n+1)*.95) rank and strict >; undersupported/unseen bins never reject.",
                               "Null exchangeability yields model-based global-null (weak) FWER control averaged over calibration randomness; strong FWER is not established.",
                               "Wilson and paired bootstrap intervals condition on empirical source, outcome model and realized calibration; exclude source and calibration-fitting uncertainty.",
                               "Lab per-study permutation uses a different variance/statistic. Its power gap cannot be attributed solely to threshold calibration.",
                               "An interval containing .05 does not prove FWER<=.05; this is a single synthetic study-design experiment."])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tmp.replace(args.out)
    print(json.dumps({m: {k: report["methods"][m][k] for k in ("power", "fwer")} for m in report["methods"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
