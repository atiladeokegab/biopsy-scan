"""Run the frozen Preflight Basel pilot or reserved comparison on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from audit.external_basel import FEATURE, SIDES_UM, SOURCE_DOI, SOURCE_MD5, align_outcomes, prepare_cells, private_path
from audit.rediscovery import FitFailed, _cox
from core.reimage_ref import crop_mask
from core.truth import layout, neighbours
from scripts.check_external_validation_split import _read_split, check_outcome_opening

PROTOCOL_SHA256 = "7acf85e5d93508697882b6ff6ae4801a1d15a7d1254532ae1aa124115f7f955d"
SEED = 20261123
DRAWS = 200
BLOCKS = 10
PATIENTS = (60, 30)


def _sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _rng(partition, index):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED, partition, index])))


def _field_fraction(edges, focal, t_cells, visible):
    keep = visible[focal]
    if not keep.any():
        return float("nan")
    weights = visible.astype(np.int32)
    total = (edges @ weights)[keep]
    t_count = (edges @ (weights * t_cells))[keep]
    return float(np.divide(t_count, total, out=np.zeros(len(total)), where=total > 0).mean())


def acquisition_schedule(n, partition):
    orders = np.empty((DRAWS, n), dtype=int)
    uniforms = np.empty((DRAWS, n, 2))
    for draw in range(DRAWS):
        rng = _rng(partition, draw)
        orders[draw] = rng.permutation(n)
        uniforms[draw] = rng.random((n, 2))
    return orders, uniforms


def acquisition_features(cells, indices, uniforms):
    """Precompute paired features per draw and patient for reuse in block refits."""
    dense = np.empty(len(indices))
    crops = np.empty((2, DRAWS, len(indices)))
    geometry = layout(cells)
    for patient, region in enumerate(indices):
        xy, types = geometry["xy"](region), geometry["types"](region)
        focal, t_cells = np.flatnonzero(types == 0), types == 1
        nbr = neighbours(xy, 50.0)
        rows = np.repeat(np.arange(len(xy)), np.diff(nbr["ptr"]))
        keep = nbr["dist"] <= 50.0
        edges = csr_matrix((np.ones(keep.sum(), dtype=np.int32), (rows[keep], nbr["idx"][keep])),
                           shape=(len(xy), len(xy)))[focal]
        dense[patient] = _field_fraction(edges, focal, t_cells, crop_mask(xy, None))
        for arm, side in enumerate(SIDES_UM):
            offsets = uniforms[:, patient] * (cells["image_size_um"][region] - side)
            for draw, (x0, y0) in enumerate(offsets):
                crops[arm, draw, patient] = _field_fraction(edges, focal, t_cells, crop_mask(xy, (x0, y0, side)))
    return dense, crops


def event_ceiling(orders, event, include):
    # B is nested within A, so only B's observed event count can bind this ceiling.
    return sum(int(event[order[include[order]][:PATIENTS[1]]].sum() >= 5) for order in orders)


def _fit(x, time, event, minimum_events):
    keep = np.isfinite(x)
    count, events = int(keep.sum()), int(event[keep].sum())
    estimate = None
    if events >= minimum_events:
        try:
            estimate = _cox(x[keep], time[keep], event[keep], None)
        except FitFailed:
            pass
    return estimate, count, events


def _comparison(dense, crops, orders, time, event, include):
    reference, n, events = _fit(dense[include], time[include], event[include], 10)
    counts = np.zeros((2, DRAWS, 2), dtype=int)
    log_hr = np.full((2, DRAWS), np.nan)
    differences = []
    for draw, order in enumerate(orders):
        order = order[include[order]]
        if len(order) < PATIENTS[0]:
            raise ValueError("the 60-patient acquisition does not fit after block deletion")
        for arm, size in enumerate(PATIENTS):
            sampled = order[:size]
            estimate, measured, observed = _fit(crops[arm, draw, sampled], time[sampled], event[sampled], 5)
            counts[arm, draw] = (measured, observed)
            if estimate is not None:
                log_hr[arm, draw] = estimate[0]
        a, b = log_hr[:, draw]
        if reference is not None and np.isfinite((a, b)).all():
            differences.append(abs(a - reference[0]) - abs(b - reference[0]))
    return reference, n, events, counts, log_hr, np.asarray(differences)


def compare(dense, crops, orders, time, event, partition, scale):
    mean, sd = scale
    dense = (dense - mean) / sd
    crops = (crops - mean) / sd
    include = np.ones(len(dense), dtype=bool)
    reference, count, events, counts, log_hr, differences = _comparison(dense, crops, orders, time, event, include)
    interval = conditional = average = None
    leave_out = []
    if len(differences) >= 140:
        average = float(differences.mean())
        sample = _rng(partition, DRAWS).choice(differences, size=(2000, len(differences)))
        conditional = np.quantile(sample.mean(axis=1), [0.025, 0.975]).tolist()
        block = np.arange(len(dense)) % BLOCKS
        for omitted in range(BLOCKS):
            *_, delta = _comparison(dense, crops, orders, time, event, block != omitted)
            leave_out.append(float(delta.mean()) if len(delta) >= 140 else None)
        if all(value is not None for value in leave_out):
            estimates = np.asarray(leave_out)
            se = np.sqrt((BLOCKS - 1) / BLOCKS * np.square(estimates - estimates.mean()).sum())
            interval = [float(average - 2.262 * se), float(average + 2.262 * se)]

    dense_summary = {"measured": count, "events": events, "log_hr": None, "ci95": None}
    if reference is not None:
        beta, se = reference
        dense_summary.update(log_hr=beta, ci95=[beta - 1.959963984540054 * se, beta + 1.959963984540054 * se])
    arms = {}
    for arm, name in enumerate(("A", "B")):
        coeff = log_hr[arm, np.isfinite(log_hr[arm])]
        arms[name] = {"patients_per_draw": PATIENTS[arm], "side_um": SIDES_UM[arm], "area_mm2": 5.4,
                      "measured_fraction": float(counts[arm, :, 0].mean() / PATIENTS[arm]),
                      "measured_min_median_max": np.quantile(counts[arm, :, 0], [0, 0.5, 1]).tolist(),
                      "events_min_median_max": np.quantile(counts[arm, :, 1], [0, 0.5, 1]).tolist(),
                      "log_hr_min_median_max": np.quantile(coeff, [0, 0.5, 1]).tolist() if len(coeff) else None,
                      "failed_fits": DRAWS - len(coeff)}
    return {"dense": dense_summary, "arms": arms, "valid_pairs": len(differences),
            "failed_pair_fraction": 1 - len(differences) / DRAWS, "mean_error_a_minus_b": average,
            "conditional_ci95": conditional, "patient_ci95": interval, "leave_block_out": leave_out,
            "preferred": None if average in (None, 0) else ("A" if average < 0 else "B")}


def _supports(result, recommendation):
    if result["preferred"] is None or recommendation is None:
        return "inconclusive"
    if result["preferred"] != recommendation:
        return "contrary directional ranking"
    bounds = (result["patient_ci95"], result["conditional_ci95"])
    same_sign = all(ci is not None and (ci[1] < 0 if recommendation == "A" else ci[0] > 0) for ci in bounds)
    dense = result["dense"]["ci95"]
    dense_has_direction = dense is not None and (dense[0] > 0 or dense[1] < 0)
    return "supports pilot recommendation" if same_sign and dense_has_direction else "inconclusive"


def run(geometry, manifest_path, protocol, outcomes, receipt=None, pilot_path=None):
    manifest_path, geometry = private_path(manifest_path), private_path(geometry)
    provenance = {"protocol_sha256": _sha256(protocol), "split_sha256": _sha256(manifest_path),
                  "geometry_sha256": _sha256(geometry), "source_doi": SOURCE_DOI, "source_md5": SOURCE_MD5}
    if provenance["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("unexpected protocol revision")
    pilot = None
    if receipt is None:
        manifest = _read_split(manifest_path.read_bytes())
        partition, code = "pilot", 1
    else:
        pilot = json.loads(Path(pilot_path).read_bytes())
        if pilot["partition"] != "pilot" or pilot["provenance"] != provenance:
            raise ValueError("the pilot checkpoint does not match the frozen inputs")
        manifest = check_outcome_opening(manifest_path, protocol, private_path(receipt))
        if pilot["preferred"] not in ("A", "B"):
            raise ValueError("reserved analysis requires a pilot recommendation")
        partition, code = "reserved", 2

    if min(len(manifest[name]) for name in ("pilot", "reserved")) < 67:
        raise ValueError("each partition requires at least 67 eligible patients")
    cells = prepare_cells(pd.read_parquet(geometry))
    ids = manifest[partition]
    by_patient = {name: index for index, name in enumerate(cells["patient_names"])}
    if not set(manifest["eligible"]).issubset(by_patient):
        raise ValueError("eligible patients must all have a fitting archived image")
    endpoint = pd.read_csv(private_path(outcomes), dtype={"patient_id": str}, keep_default_na=False)
    time, event = align_outcomes(endpoint, ids)
    orders, uniforms = acquisition_schedule(len(ids), code)
    event_limit = event_ceiling(orders, event, np.ones(len(ids), dtype=bool))
    result = {"partition": partition, "patients": len(ids), "feature": FEATURE, "provenance": provenance,
              "observed_events": int(event.sum()), "event_eligible_pair_ceiling": event_limit}
    if event_limit < 140 or event.sum() < 10:
        blocks = np.arange(len(ids)) % BLOCKS
        return {**result, "decision": "inconclusive", "preferred": None, "acquisition_draws": DRAWS,
                "minimum_usable_pairs": 140, "small_arm_patients": PATIENTS[1],
                "minimum_arm_events": 5, "minimum_dense_events": 10,
                "leave_block_out_event_ceilings": [event_ceiling(orders, event, blocks != b) for b in range(BLOCKS)]}

    dense, crops = acquisition_features(cells, [by_patient[patient] for patient in ids], uniforms)
    if pilot is None:
        measured = dense[np.isfinite(dense)]
        scale = [float(measured.mean()), float(measured.std())] if len(measured) else []
    else:
        scale = pilot["pilot_scale"]
    if len(scale) != 2 or not np.isfinite(scale).all() or scale[1] <= 0:
        raise ValueError("pilot dense biomarker does not have an estimable scale")
    result.update(compare(dense, crops, orders, time, event, code, scale), pilot_scale=scale)
    if pilot is not None:
        result["decision"] = _supports(result, pilot["preferred"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--pilot", type=Path)
    args = parser.parse_args()
    if (args.receipt is None) != (args.pilot is None):
        parser.error("reserved analysis requires both the opening receipt and the pilot checkpoint")
    try:
        result = run(args.geometry, args.manifest, args.protocol, args.outcomes, args.receipt, args.pilot)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Basel validation failed: {type(exc).__name__}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
