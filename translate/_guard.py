"""Shared non-leakage, provenance, negative-index validation, and R-16 rule helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._stats import stable_hash

_TRUSTED_LOADER_TOKEN = object()


class LockedSplitError(PermissionError):
    """R-17 must never read or use locked held-out patients."""


def _validate_real_cohort_manifest(cohort: str, cohort_data: dict[str, Any]) -> None:
    """Fail-closed active lock & manifest verification for real-cohort payloads."""
    from core import cohorts

    lock_file = getattr(cohorts, "LOCK_FILE", None)
    if lock_file is None or not Path(lock_file).exists():
        raise LockedSplitError("fail-closed: cohorts.LOCK_FILE must exist for real-cohort R-17 access")

    if cohort_data.get("_trusted_loader_token") is _TRUSTED_LOADER_TOKEN:
        return

    data_dir = Path(cohort_data.get("data_dir") or cohorts.DATA_DIR)
    region_names = list(cohort_data.get("region_names") or ())
    if not region_names:
        raise LockedSplitError("fail-closed: real-cohort payload missing region_names for manifest validation")
    try:
        locked = cohorts.locked_regions(cohort, data_dir) if cohort == "upmc" else set()
        if cohort == "upmc" and not locked:
            raise LockedSplitError("fail-closed: cohorts.locked_regions returned empty set for UPMC")
        if locked and (set(region_names) & locked):
            raise LockedSplitError("fail-closed: real-cohort payload contains locked regions")
        manifest = cohorts.load_regions(cohort, data_dir=data_dir, identity="case", split="discovery")
    except LockedSplitError:
        raise
    except Exception as exc:
        raise LockedSplitError(f"fail-closed: could not verify real-cohort discovery manifest ({exc})") from exc

    allowed_regions = set(manifest["region_id"].astype(str))
    if not set(region_names).issubset(allowed_regions):
        raise LockedSplitError("fail-closed: real-cohort regions outside verified discovery manifest")
    patient_names = list(cohort_data.get("patient_names") or ())
    if patient_names and not set(patient_names).issubset(set(manifest["patient_id"].astype(str))):
        raise LockedSplitError("fail-closed: real-cohort patient_names outside verified discovery manifest")


def validate_discovery_cohort(
    cohort_data: dict[str, Any],
    *,
    split: str = "discovery",
) -> None:
    """Validate split provenance, real-data lock verification, and 1D non-negative aligned indices."""
    if split != "discovery":
        raise LockedSplitError(f"R-17 must run on split='discovery' only, got {split!r}")
    data_split = cohort_data.get("split")
    if data_split != "discovery":
        raise LockedSplitError(f"cohort_data split must be explicitly 'discovery', got {data_split!r}")
    cohort_name = str(cohort_data.get("cohort", ""))
    if cohort_name in ("upmc", "charville"):
        _validate_real_cohort_manifest(cohort_name, cohort_data)
    locked_pats = set(cohort_data.get("locked_patients", ()))
    if locked_pats and (locked_pats & set(cohort_data.get("patient_names", ()))):
        raise LockedSplitError("cohort_data includes locked patient IDs; refusing R-17 execution")

    x = np.asarray(cohort_data["x"], dtype=np.float64)
    y = np.asarray(cohort_data["y"], dtype=np.float64)
    ct_raw = np.asarray(cohort_data["cell_type"])
    rid_raw = np.asarray(cohort_data["region_id"])
    pat_raw = np.asarray(cohort_data["patient_of_region"])
    st_raw = np.asarray(cohort_data["strata"])
    for label, raw_arr in (
        ("cell_type", ct_raw),
        ("region_id", rid_raw),
        ("patient_of_region", pat_raw),
        ("strata", st_raw),
    ):
        if not np.issubdtype(raw_arr.dtype, np.integer):
            raise ValueError(f"{label} must have an integer dtype, got {raw_arr.dtype}")

    cell_type = ct_raw.astype(np.int64)
    markers = np.asarray(cohort_data["markers"], dtype=np.float64)
    region_id = rid_raw.astype(np.int64)
    pat_of_reg = pat_raw.astype(np.int64)
    time = np.asarray(cohort_data["time"], dtype=np.float64)
    event = np.asarray(cohort_data["event"], dtype=bool)
    strata = st_raw.astype(np.int64)
    type_names = list(cohort_data["type_names"])
    marker_names = list(cohort_data["marker_names"])

    for label, vec in (
        ("x", x),
        ("y", y),
        ("cell_type", cell_type),
        ("region_id", region_id),
        ("patient_of_region", pat_of_reg),
        ("time", time),
        ("event", event),
        ("strata", strata),
    ):
        if vec.ndim != 1:
            raise ValueError(f"{label} must be 1D, got shape {vec.shape}")

    n_cells = len(x)
    n_reg = len(pat_of_reg)
    n_pat = len(time)
    if (
        len(y) != n_cells
        or len(cell_type) != n_cells
        or len(region_id) != n_cells
        or markers.ndim != 2
        or markers.shape != (n_cells, len(marker_names))
    ):
        raise ValueError("inconsistent cell array lengths or marker columns")
    if len(event) != n_pat or len(strata) != n_pat:
        raise ValueError("inconsistent patient endpoint/strata array lengths")
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(markers).all() and np.isfinite(time).all()):
        raise ValueError("non-finite coordinates, markers, or survival times in cohort_data")
    if n_cells and (region_id.min() < 0 or region_id.max() >= n_reg):
        raise ValueError(f"region_id contains negative or out-of-range indices (range [0, {n_reg}))")
    if n_reg and (pat_of_reg.min() < 0 or pat_of_reg.max() >= n_pat):
        raise ValueError(f"patient_of_region contains negative or out-of-range indices (range [0, {n_pat}))")
    if n_cells and (cell_type.min() < 0 or cell_type.max() >= len(type_names)):
        raise ValueError(f"cell_type contains negative or out-of-range codes (range [0, {len(type_names)}))")


def resolve_r16_rule(
    cohort_data: dict[str, Any],
    r16_rule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the frozen R-16 biomarker definition and verify cohort expressibility."""
    if r16_rule is None:
        from audit import rediscovery

        rule = dict(rediscovery.RULE)
        rule_sha = rediscovery.RULE_SHA256
    else:
        rule = dict(r16_rule)
        rule_sha = str(rule.get("rule_sha256") or stable_hash(rule))

    out = {**rule, "rule_sha256": rule_sha}
    if not rule.get("expressible", True):
        out["expressible"] = False
        out["reason"] = str(rule.get("reason") or "R-16 candidate reported unsupported by cohort")
        return out

    focal = rule.get("focal")
    neighbour = rule.get("neighbour")
    type_names = list(cohort_data["type_names"])
    marker_names = set(cohort_data.get("marker_names", ()))
    req_markers = list(rule.get("required_markers") or rule.get("markers") or ())
    if req_markers and marker_names:
        missing_mk = [mk for mk in req_markers if mk not in marker_names]
        if missing_mk:
            cohort_label = str(cohort_data.get("cohort", "cohort"))
            out["expressible"] = False
            out["reason"] = f"not testable on {cohort_label}: missing markers {missing_mk}"
            return out
    if not focal or not neighbour:
        raise ValueError("expressible R-16 rule must specify 'focal' and 'neighbour'")
    if focal not in type_names or neighbour not in type_names:
        missing = [t for t in (focal, neighbour) if t not in type_names]
        out["expressible"] = False
        out["reason"] = f"R-16 types absent from cohort: {missing}"
        return out

    radius_um = float(rule.get("radius_um", 25.0))
    stat = str(rule.get("statistic", "fraction"))
    if radius_um <= 0 or stat not in ("fraction", "enrichment"):
        raise ValueError(f"unsupported R-16 radius={radius_um} or statistic={stat!r}")
    out["radius_um"] = radius_um
    out["statistic"] = stat
    out["focal_code"] = type_names.index(focal)
    out["nbr_code"] = type_names.index(neighbour)
    return out
