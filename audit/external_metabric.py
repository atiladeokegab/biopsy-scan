"""Run the frozen METABRIC comparison with separate clinical custody."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
from PIL import Image

from audit.external_basel import FEATURE, SIDES_UM, TYPE_NAMES, private_path
from audit.external_validation import (
    BLOCKS, _sha256, _supports, acquisition_features, acquisition_schedule, compare, event_ceiling,
)
from scripts.check_external_validation_split import (
    METABRIC_PROTOCOL_SHA256, METABRIC_SPLIT_SHA256, check_metabric_outcome_opening, read_metabric_split,
)

PROCESSED = (910_666_172, "f69910a57b0ea4957db12e18aa064263")
MASKS = (71_831_315, "7e60187f7d1bebb8f9280f373d1a7a6f")
SELECTION_SHA256 = "e25a68412716ba78d01d9863bc5c93d22a1597221307e19d542b1c88db7637c2"
PILOT_ENDPOINTS_SHA256 = "ef330f4eb12dc1a07794ba2fec7e4ff1877302a588b016ac0c01ec7062d22467"
DESIGN = (90, 45)
PATIENT = "metabric_id"
IMAGE = "ImageNumber"
CELL = "ObjectNumber"
XY = ["Location_Center_X", "Location_Center_Y"]
FLAGS = ["is_epithelial", "is_tumour", "is_normal", "is_dcis"]
PHENOTYPE = "cellPhenotype"
T_TYPES = {"CD4^{+} T cells", "CD8^{+} T cells"}


def _verify_archive(path, expected):
    path = private_path(path)
    with path.open("rb") as archive:
        if path.stat().st_size != expected[0] or hashlib.file_digest(archive, "md5").hexdigest() != expected[1]:
            raise ValueError("METABRIC archive size or checksum does not match")
    return path


def _read_selection(path, expected_ids):
    path = private_path(path)
    if _sha256(path) != SELECTION_SHA256:
        raise ValueError("unexpected METABRIC selected-images revision")
    selection = pd.read_csv(path, dtype={PATIENT: str}, keep_default_na=False)
    if set(selection) != {PATIENT, IMAGE, "width", "height"} or list(selection[PATIENT]) != expected_ids:
        raise ValueError("selected images must match the ordered eligible patients")
    fields = [IMAGE, "width", "height"]
    if not selection[fields].ge(1).all().all() or not selection[fields].mod(1).eq(0).all().all():
        raise ValueError("invalid selected image size or key")
    if not selection[IMAGE].is_unique or selection[["width", "height"]].lt(SIDES_UM[-1]).any().any():
        raise ValueError("selected images must be unique and fit both fields")
    return selection


def _verify_masks(archive, selection):
    pattern = re.compile(r"MB([0-9]+)_([0-9]+)_CellMask\.tiff")
    expected = {image: (patient, width, height) for patient, image, width, height in
                selection[[PATIENT, IMAGE, "width", "height"]].itertuples(index=False, name=None)}
    seen = set()
    for member in archive.namelist():
        match = pattern.fullmatch(Path(member).name)
        if match is None or int(match[2]) not in expected:
            continue
        image = int(match[2])
        with archive.open(member) as file, Image.open(file) as mask:
            if image in seen or expected[image] != ("MB-" + match[1], *mask.size):
                raise ValueError("selected image does not match its corrected mask")
        seen.add(image)
    if seen != set(expected):
        raise ValueError("selected image missing from corrected masks")


def _prepare_cells(cells, selection):
    matched = cells.merge(selection, on=[PATIENT, IMAGE], validate="many_to_one")
    if len(matched) != len(cells) or set(matched[IMAGE]) != set(selection[IMAGE]):
        raise ValueError("processed cells do not match the selected patient/image keys")
    if (matched.duplicated([IMAGE, CELL]).any() or not (matched[CELL].ge(1) & matched[CELL].mod(1).eq(0)).all()
            or not matched[FLAGS].isin([0, 1]).all().all() or not matched[PHENOTYPE].str.strip().fillna("").ne("").all()):
        raise ValueError("invalid processed cell key, phenotype or tissue flag")
    if not (np.isfinite(matched[XY]).all(axis=1) & matched[XY].ge(0).all(axis=1)
            & matched[XY[0]].lt(matched.width) & matched[XY[1]].lt(matched.height)).all():
        raise ValueError("centroid outside corrected mask bounds")

    focal = matched[FLAGS].eq([1, 1, 0, 0]).all(axis=1)
    t_cells = matched[PHENOTYPE].str.strip().isin(T_TYPES)
    if (focal & t_cells).any() or set(matched.loc[focal, IMAGE]) != set(selection[IMAGE]):
        raise ValueError("unexpected tumour focal and T-cell annotations")
    matched["cell_type"] = np.select([focal, t_cells], [0, 1], default=2)
    order = {image: idx for idx, image in enumerate(selection[IMAGE])}
    matched["region_id"] = matched[IMAGE].map(order)
    ordered = matched.sort_values(["region_id", CELL])
    return {"region_id": ordered["region_id"].to_numpy(np.int64),
            "x": ordered[XY[0]].to_numpy(float), "y": ordered[XY[1]].to_numpy(float),
            "cell_type": ordered["cell_type"].to_numpy(np.int64), "type_names": list(TYPE_NAMES),
            "patient_of_region": np.arange(len(selection)),
            "image_size_um": selection[["width", "height"]].to_numpy(float)}


def load_geometry(processed, masks, selection):
    with ZipFile(_verify_archive(masks, MASKS)) as archive:
        _verify_masks(archive, selection)
    with ZipFile(_verify_archive(processed, PROCESSED)) as archive:
        (member,) = [name for name in archive.namelist() if Path(name).name == "SingleCells.csv"
                     and not name.startswith("__MACOSX/")]
        with archive.open(member) as cells:
            columns = [PATIENT, IMAGE, CELL, *XY, *FLAGS, PHENOTYPE]
            selected = set(selection[IMAGE])
            chunks = [chunk[chunk[IMAGE].isin(selected)] for chunk in
                      pd.read_csv(cells, usecols=columns, dtype={PATIENT: str, PHENOTYPE: str}, chunksize=100_000)]
    return _prepare_cells(pd.concat(chunks, ignore_index=True), selection)


def align_endpoints(frame, ids):
    if set(frame) != {PATIENT, "yearsToStatus", "DeathBreast", "ERStatus"}:
        raise ValueError("unexpected METABRIC endpoint schema")
    if not frame[PATIENT].is_unique or set(frame[PATIENT]) != set(ids):
        raise ValueError("endpoints must match exactly the requested METABRIC partition")
    aligned = frame.set_index(PATIENT).loc[ids]
    time = pd.to_numeric(aligned["yearsToStatus"], errors="raise").to_numpy(float)
    event = pd.to_numeric(aligned["DeathBreast"], errors="raise")
    strata = aligned["ERStatus"]
    if not ((np.isfinite(time) & (time > 0)).all() and event.isin([0, 1]).all()
            and strata.isin(["pos", "neg"]).all()):
        raise ValueError("invalid METABRIC follow-up, event or ER stratum")
    return time, event.to_numpy(bool), strata.eq("pos").to_numpy(np.int8)


def _pilot_gate(orders, event):
    blocks = np.arange(len(event)) % BLOCKS
    kept = [np.ones(len(event), dtype=bool), *(blocks != block for block in range(BLOCKS))]
    ceilings = [event_ceiling(orders, event, mask, small_arm=DESIGN[1]) for mask in kept]
    dense_pass = all(event[mask].sum() >= 10 for mask in kept)
    return {"event_eligible_pairs": ceilings[0], "block_event_eligible_pairs": ceilings[1:],
            "dense_floor_pass": dense_pass, "passed": dense_pass and min(ceilings) >= 140}


def run(processed, masks, selection_path, manifest_path, protocol, endpoints, receipt=None, pilot_path=None):
    manifest_path = private_path(manifest_path)
    provenance = {"protocol_sha256": METABRIC_PROTOCOL_SHA256, "split_sha256": METABRIC_SPLIT_SHA256,
                  "selection_sha256": SELECTION_SHA256, "processed_md5": PROCESSED[1], "masks_md5": MASKS[1],
                  "pilot_endpoints_sha256": PILOT_ENDPOINTS_SHA256}
    pilot = None
    if receipt is None:
        manifest = read_metabric_split(manifest_path, protocol)
        partition, code = "pilot", 1
    else:
        manifest, pilot = check_metabric_outcome_opening(manifest_path, protocol, private_path(receipt),
                                                       private_path(pilot_path))
        if pilot.get("provenance") != provenance:
            raise ValueError("pilot provenance does not match the frozen inputs")
        scale = np.asarray(pilot.get("pilot_scale"), dtype=float)
        if scale.shape != (2,) or not np.isfinite(scale).all() or scale[1] <= 0:
            raise ValueError("invalid frozen pilot scale")
        partition, code = "reserved", 2

    endpoints = private_path(endpoints)
    if pilot is None and _sha256(endpoints) != PILOT_ENDPOINTS_SHA256:
        raise ValueError("unexpected METABRIC pilot endpoint revision")
    selection = _read_selection(selection_path, manifest["eligible"])
    ids = manifest[partition]
    time, event, strata = align_endpoints(pd.read_csv(endpoints, dtype={PATIENT: str}, keep_default_na=False), ids)
    orders, uniforms = acquisition_schedule(len(ids), code)
    result = {"partition": partition, "patients": len(ids), "feature": FEATURE, "provenance": provenance}
    if pilot is None:
        gate = _pilot_gate(orders, event)
        result["count_gate"] = gate
        if not gate["passed"]:
            return dict(result, preferred=None, decision="inconclusive")

    cells = load_geometry(processed, masks, selection)
    by_patient = {patient: idx for idx, patient in enumerate(selection[PATIENT])}
    dense, crops = acquisition_features(cells, [by_patient[patient] for patient in ids], uniforms)
    if pilot is None:
        scale = np.array([dense.mean(), dense.std(ddof=0)])
        if not np.isfinite(scale).all() or scale[1] <= 0:
            return dict(result, preferred=None, decision="inconclusive", reason="unusable pilot scale")
    result.update(compare(dense, crops, orders, time, event, code, scale, patients=DESIGN, strata=strata),
                  pilot_scale=scale.tolist())
    if pilot is not None:
        result["decision"] = _supports(result, pilot["preferred"])
    elif result["preferred"] is None:
        result["decision"] = "inconclusive"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("processed", "masks", "selection", "manifest", "protocol", "endpoints"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--pilot", type=Path)
    args = parser.parse_args()
    if (args.receipt is None) != (args.pilot is None):
        parser.error("reserved analysis requires both an opening receipt and a pilot checkpoint")
    try:
        result = run(args.processed, args.masks, args.selection, args.manifest, args.protocol, args.endpoints,
                     args.receipt, args.pilot)
    except (OSError, ValueError, KeyError, TypeError, BadZipFile) as exc:
        parser.exit(1, f"METABRIC validation failed: {type(exc).__name__}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
