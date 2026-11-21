"""Run cohort-specific Monte Carlo study rehearsals and save completed points."""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from core.cohorts import load
from core.grammar import grammar, grammar_hash
from core.truth import truth

from .artifacts import charts, save_tables, source_metadata
from .runner import DEFAULT_BUDGET, run_point


def reference_matrix(cells, path, metadata):
    fingerprint = grammar_hash(grammar(cells["type_names"]))
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["source_hash"]) == metadata["source_hash"] and str(cached["grammar_hash"]) == fingerprint:
                return cached["X"], fingerprint
    X = truth(cells)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, X=X, source_hash=metadata["source_hash"], grammar_hash=fingerprint)
    temporary.replace(path)
    return X, fingerprint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", nargs="+", choices=["charville", "upmc"], default=["charville", "upmc"])
    parser.add_argument("--patients", nargs="+", type=int)
    parser.add_argument("--regions", nargs="+", type=int)
    parser.add_argument("--fov-um", nargs="+", help="numeric sides or full; defaults to full for the first timing slice")
    parser.add_argument("--eps", nargs="+", type=float, default=[0.1])
    parser.add_argument("--hr", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--backend", choices=["numpy", "cpu", "gpu"], default="cpu")
    parser.add_argument("--out", type=Path, default=Path("atlas/results"))
    parser.add_argument("--cache", type=Path, default=Path("data/atlas-cache"))
    for name, value in DEFAULT_BUDGET.items():
        parser.add_argument("--" + name, type=int, default=value)
    args = parser.parse_args(argv)
    budget = {key: getattr(args, key) for key in DEFAULT_BUDGET}
    for cohort in args.cohorts:
        cells = load(cohort)
        meta = source_metadata(cells)
        reference, fingerprint = reference_matrix(cells, args.cache / f"{cohort}-reference.npz", meta)
        n = meta["source_n"]
        patients = sorted(set(args.patients or [max(2, round(n / 2)), n, 2 * n, 3 * n]))
        requested_regions = args.regions or [1, 2, 4, 8]
        if any(k < 1 for k in requested_regions) or any(p < 2 for p in patients):
            parser.error("patients must be >= 2 and region counts >= 1")
        regions = sorted({min(k, meta["max_regions_per_patient"]) for k in requested_regions})
        fov = list(dict.fromkeys(None if x == "full" else float(x) for x in (args.fov_um or ["full"])))
        if any(f is not None and (not np.isfinite(f) or f <= 0) for f in fov):
            parser.error("field-of-view sides must be positive and finite, or full")
        designs = [dict(cohort=cohort, patients=p, regions_per_patient=k, fov_um=f,
                        eps=e, backend=args.backend)
                   for p, k, f, e in itertools.product(patients, regions, fov, args.eps)]
        nested_points = {0, len(designs) // 2, len(designs) - 1}
        metadata = {"cohort": cohort, **meta, "grammar_hash": fingerprint,
                    "budget": budget, "hr_per_source_sd": args.hr, "seed": args.seed,
                    "backend": args.backend, "numpy_version": np.__version__,
                    "inference": "model-based Monte Carlo under the empirical verified source; no clinical claim",
                    "planned_fov_um": ([200, 300, 400, 500, 600, None] if cohort == "charville"
                                       else [200, 400, 600, 800, 1000, None]),
                    "completed_fov_um": fov, "censoring_tail": "administrative horizon at stratum maximum follow-up"}
        summaries, records = [], []
        out = args.out / cohort
        for i, design in enumerate(designs):
            start = time.perf_counter()
            point_budget = {**budget, "nested": budget["nested"] if i in nested_points else 0}
            summary, trials, _ = run_point(cells, reference, design, hr=args.hr, budget=point_budget, seed=args.seed)
            summary.update(grammar_hash=fingerprint, source_extrapolation=design["patients"] > n,
                           resample_ratio=design["patients"] / n, runtime_s=time.perf_counter() - start)
            summaries.append(summary)
            records.extend(trials)
            metadata["completed_designs"] = len(summaries)
            save_tables(summaries, records, metadata, out)
            print(json.dumps({"cohort": cohort, "point": i + 1, "of": len(designs),
                              "power": summary["power"], "fwer": summary["fwer"],
                              "runtime_s": summary["runtime_s"]}), flush=True)
        charts(summaries, out, cohort=cohort, fov_um=fov[0], eps=args.eps[0], hr=args.hr)


if __name__ == "__main__":
    main()
