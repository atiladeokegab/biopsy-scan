"""Single-core lab-style rung: uv run python -m bench.lab_today.

Fresh cKDTree recounts and a 1000-permutation test for three studies. Update only
the lab_today field of the CPU result, keeping its extrapolation explicitly estimated.
"""

import json
import os
import statistics
import subprocess
from time import perf_counter

import numpy as np

from bench.real_cpu import DESIGN, OUTPUT, save
from core.cohorts import load
from core.grammar import grammar, grammar_hash
from core.reimage_ref import crop_mask, draw, miscall, reimage
from core.truth import truth_region
from stats import permutation_test


def tree_region(_nbr, xy, types, focal, K, crop=None, M=None, u=None):
    observed = miscall(types, M, u)
    visible = crop_mask(xy, crop)
    return truth_region(xy[visible], observed[visible], focal, K)


def main():
    report = json.loads(OUTPUT.read_text())
    if report.get("design") != DESIGN or report.get("cohort", {}).get("name") != "charville":
        raise ValueError("CPU result must contain the same Charville design")
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {affinity[-1]})
    cells = load("charville")
    assert len(cells["time"]) == 159
    if report["cohort"]["grammar_hash"] != grammar_hash(grammar(cells["type_names"])):
        raise ValueError("CPU result and legacy baseline have different feature grammars")
    # The tree callback builds its own cropped-region tree; do not build a CSR too.
    cells["_nbr"] = {r: None for r in range(len(cells["patient_of_region"]))}
    reference_cells = {key: value for key, value in cells.items() if key != "_nbr"}
    rows = []
    for seed in range(3):
        start = perf_counter()
        X = reimage(cells, DESIGN, seed, region_kernel=tree_region)
        feature_s = perf_counter() - start
        patients = draw(cells, DESIGN, seed)["patients"]
        infer_start = perf_counter()
        result = permutation_test(X, cells["time"][patients], cells["event"][patients],
                                  cells["strata"][patients], n_resamples=1000, seed=seed)
        full_s = perf_counter() - start
        inference_s = perf_counter() - infer_start
        assert len(result["null_z"]) == 1000
        # Validation cost is outside the timed legacy pipeline.
        reference = reimage(reference_cells, DESIGN, seed)
        assert np.array_equal(X, reference, equal_nan=True), f"tree/reference parity failed at {seed}"
        rows.append(dict(seed=seed, feature_s=feature_s, permutation_s=inference_s,
                         full_s=full_s, parity="exact, including NaN placement"))
        print(json.dumps(rows[-1]), flush=True)
    mean_s = statistics.mean(row["full_s"] for row in rows)
    report["lab_today"] = {
        "rung": 0, "status": "MEASURED-CPU", "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "scope": "Fresh cKDTree counts per cropped/relabelled region, patient feature aggregation, 1000 conditional permutations per study",
        "single_core_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {key: os.environ.get(key) for key in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")},
        "measured_studies": 3, "permutations_per_study": 1000, "rows": rows,
        "mean_s_per_study": mean_s,
        "extrapolation": {"status": "ESTIMATE", "studies": 1000, "seconds": mean_s * 1000,
                          "basis": "mean of three measured studies multiplied by 1000; not a measured run"},
        "comparison_scope": "Full legacy inference differs from feature-only rows; do not quote their quotient as a matched end-to-end speedup",
        "inference_note": "Inherited resampled outcomes are a timing workload; no scientific p-value inference is claimed",
    }
    save(report)
    print(f"MEASURED-CPU: {mean_s:.3f} s/study; ESTIMATE: {mean_s * 1000:.3f} s for 1000 studies", flush=True)


if __name__ == "__main__":
    main()
