"""Run cohort-specific Monte Carlo study rehearsals and save completed points."""

import argparse
import hashlib
import itertools
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from core.cohorts import load
from core.grammar import grammar, grammar_hash
from core.truth import layout, neighbours, truth

from .artifacts import (
    array_hash,
    charts,
    code_fingerprint,
    comparison_charts,
    generator_provenance,
    save_point,
    save_tables,
    source_metadata,
)
from .bank import build_bank
from .runner import DEFAULT_BUDGET, design_id, run_point


def reference_matrix(cells, path, metadata):
    fingerprint = grammar_hash(grammar(cells["type_names"]))
    provenance = json.dumps(generator_provenance(("core",), "numpy"), sort_keys=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            if (str(cached["source_hash"]) == metadata["source_hash"] and str(cached["grammar_hash"]) == fingerprint
                    and "generator" in cached and str(cached["generator"]) == provenance
                    and "hash" in cached and str(cached["hash"]) == array_hash(cached["X"])):
                return cached["X"], fingerprint
    X = truth(cells)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, X=X, source_hash=metadata["source_hash"], grammar_hash=fingerprint,
             generator=provenance, hash=array_hash(X))
    temporary.replace(path)
    return X, fingerprint


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", nargs="+", choices=["charville", "upmc"], default=["charville", "upmc"])
    parser.add_argument("--patients", nargs="+", type=int)
    parser.add_argument("--regions", nargs="+", type=int)
    parser.add_argument("--fov-um", nargs="+", help="numeric sides or full; defaults to full for the first timing slice")
    parser.add_argument("--eps", nargs="+", type=float, default=[0.1])
    parser.add_argument("--hr", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--backend", choices=["numpy", "cpu", "gpu"], default="cpu")
    parser.add_argument("--bank-draws", type=int, default=64, help="finite-bank approximation; 0 uses fresh direct imaging")
    parser.add_argument("--workers", type=int, default=1, help="concurrent imaging groups; NumPy backend only")
    parser.add_argument("--out", type=Path, default=Path("atlas/results"))
    parser.add_argument("--cache", type=Path, default=Path("data/atlas-cache"))
    for name, value in DEFAULT_BUDGET.items():
        parser.add_argument("--" + name, type=int, default=value)
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.workers > 1 and args.backend != "numpy"):
        parser.error("workers must be positive; concurrent groups require backend numpy")
    if args.bank_draws != 0 and args.bank_draws < 2:
        parser.error("bank-draws must be 0 (direct) or >= 2")
    args.eps = list(dict.fromkeys(args.eps))
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
        designs = [{"cohort": cohort, "patients": p, "regions_per_patient": k, "fov_um": f,
                    "eps": e, "backend": args.backend}
                   for p, k, f, e in itertools.product(patients, regions, fov, args.eps)]
        nested_points = {0, len(designs) // 2, len(designs) - 1}
        metadata = {"cohort": cohort, "result_label": "SYNTHETIC", **meta, "grammar_hash": fingerprint,
                    "budget": budget, "hr_per_source_sd": args.hr, "seed": args.seed,
                    "backend": args.backend, "numpy_version": np.__version__,
                    "inference": "model-based Monte Carlo under the empirical verified source; no clinical claim",
                    "planned_fov_um": ([200, 300, 400, 500, 600, None] if cohort == "charville"
                                       else [200, 400, 600, 800, 1000, None]),
                    "requested_fov_um": fov, "censoring_tail": "administrative horizon at stratum maximum follow-up",
                    "mc_interval_scope": "conditional on source, acquisition bank and realized calibration; excludes source and bank-construction uncertainty",
                    "method_check_scope": "paired differences with conservative 95% bounds; interval containing zero does not establish equivalence",
                    "direct_check_scope": "bank and direct imaging use the same bank-derived cutoff"}
        metadata["code_fingerprint"] = code_fingerprint()
        metadata["reference_generator"] = generator_provenance(("core",), "numpy")
        metadata["inference_scope"] = ("descriptive verified-source simulation only" if cohort == "upmc"
                                        else "empirical-source study-design simulation")
        out = args.out / cohort
        out.mkdir(parents=True, exist_ok=True)
        manifest = {**metadata, "designs": designs, "bank_draws": args.bank_draws,
                    "checked_design_indices": sorted(nested_points)}
        run_id = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        manifest_path = out / "run.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text())["run_id"] != run_id:
                raise ValueError("output directory belongs to a different run; choose a new --out")
        else:
            if any(out.iterdir()):
                raise ValueError("output directory has no run manifest; choose a new --out")
            temporary = manifest_path.with_suffix(".tmp.json")
            temporary.write_text(json.dumps({"run_id": run_id, **manifest}, indent=2) + "\n")
            temporary.replace(manifest_path)
        metadata["run_id"] = run_id
        # Warm once before threads share the immutable neighbour arrays. Otherwise
        # concurrent first acquisitions could build duplicate large source graphs.
        if args.workers > 1:
            L = layout(cells)
            cache = cells.setdefault("_nbr", {})
            for r in range(len(cells["patient_of_region"])):
                if r not in cache:
                    cache[r] = neighbours(L["xy"](r))
        groups = {}
        for i, design in enumerate(designs):
            key = (design["regions_per_patient"], design["fov_um"], design["eps"])
            groups.setdefault(key, []).append((i, design))

        total = len(designs)

        def run_group(group, cells=cells, reference=reference, fingerprint=fingerprint, out=out,
                      n=n, nested_points=nested_points, cohort=cohort, total=total):
            # A bank is shared across N within this group and released on return.
            bank = (build_bank(cells, group[0][1], args.bank_draws, args.seed, args.cache,
                               batch_size=budget["batch"], grammar_hash=fingerprint) if args.bank_draws else None)
            completed = []
            for i, original in group:
                design = {**original, "acquisition_bank": bank} if bank is not None else original
                marker = out / "points" / f"{design_id(design)}.parquet"
                evidence = out / "studies" / marker.name
                if marker.exists() and evidence.exists():
                    summary = pd.read_parquet(marker).iloc[0].to_dict()
                    if bank is None or summary["bank_hash"] == bank["hash"]:
                        completed.append((i, summary))
                        continue
                start = time.perf_counter()
                point_budget = {**budget, "nested": budget["nested"] if i in nested_points else 0,
                                "direct": budget["direct"] if i in nested_points else 0}
                summary, trials, _ = run_point(cells, reference, design, hr=args.hr, budget=point_budget, seed=args.seed)
                summary.update(grammar_hash=fingerprint, source_extrapolation=design["patients"] > n,
                               resample_ratio=design["patients"] / n, runtime_s=time.perf_counter() - start,
                               bank_load_or_build_s=0 if bank is None else bank["load_or_build_s"])
                save_point(summary, trials, out)
                completed.append((i, summary))
                print(json.dumps({"cohort": cohort, "point": i + 1, "of": total,
                                  "power": summary["power"], "fwer": summary["fwer"],
                                  "runtime_s": summary["runtime_s"]}), flush=True)
            return completed

        indexed = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_group, group) for group in groups.values()]
            for future in as_completed(futures):
                indexed.update(future.result())
                summaries = [indexed[i] for i in sorted(indexed)]
                metadata["completed_designs"] = len(summaries)
                metadata["completed_fov_um"] = [f for f in fov if any(row["fov_um"] == f for row in summaries)]
                metadata["measurement_model"] = summaries[0]["measurement_model"]
                metadata["bank_draws"] = args.bank_draws
                save_tables(summaries, None, metadata, out)
        summaries = [indexed[i] for i in sorted(indexed)]
        for f in fov:
            for e in args.eps:
                label = "full" if f is None else f"{f:g}"
                chart_dir = out / f"fov-{label}-eps-{e:g}"
                charts(summaries, chart_dir, cohort=cohort, fov_um=f, eps=e, hr=args.hr)
        comparison_charts(summaries, out, cohort=cohort)


if __name__ == "__main__":
    main()
