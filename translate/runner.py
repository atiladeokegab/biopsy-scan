"""R-17 Translate runner: fail-closed discovery-only loading and Step 1 + Step 2 orchestration."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from typing import Any

import numpy as np
import pandas as pd

if __name__ == "__main__" and str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit.schema import ALREADY_SEEN_DISCLOSURE
from core import cohorts
from translate._guard import (
    _TRUSTED_LOADER_TOKEN,
    LockedSplitError,
    resolve_r16_rule,
    validate_discovery_cohort,
)
from translate.formats import simulate_formats
from translate.panels import run_panel_cv

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results" / "upmc-translate.json"


def _load_upmc_discovery_markers(data_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load UPMC discovery-only M=22 markers cleanly; fail-closed if lock file/split missing."""
    sig = inspect.signature(cohorts.load_regions)
    if "split" not in sig.parameters or not hasattr(cohorts, "locked_regions"):
        return None, "UPMC discovery-only locked split (R-13) is not merged on this branch"

    lock_file = getattr(cohorts, "LOCK_FILE", None)
    if lock_file is None or not Path(lock_file).exists():
        raise LockedSplitError("fail-closed: cohorts.LOCK_FILE must exist for UPMC discovery load")

    labels_path = Path(data_dir) / "upmc_labels.csv"
    zip_path = Path(data_dir) / "upmc_raw_data.zip"
    if not labels_path.exists() or not zip_path.exists():
        return None, "UPMC raw files missing in data_dir (run on Modal with mounted data)"

    locked = cohorts.locked_regions("upmc", data_dir)
    if not locked:
        raise LockedSplitError("fail-closed: cohorts.locked_regions returned empty set for UPMC")

    reg = cohorts.load_regions("upmc", data_dir=Path(data_dir), identity="case", split="discovery")
    if set(reg["region_id"]) & locked:
        raise LockedSplitError("fail-closed: discovery regions overlap locked held-out regions")

    pts = cohorts.load_patients(reg)
    pts = pts[pts.usable].reset_index(drop=True)
    pidx = {p: i for i, p in enumerate(pts.patient_id)}
    reg = reg[reg.patient_id.isin(pidx)].reset_index(drop=True)
    region_ids = list(reg.region_id)
    if set(region_ids) & locked:
        raise LockedSplitError("fail-closed: usable discovery regions overlap locked held-out regions")

    cells_sig = inspect.signature(cohorts.load_cells)
    if "split" not in cells_sig.parameters:
        raise LockedSplitError("fail-closed: cohorts.load_cells lacks mandatory split parameter")
    c = cohorts.load_cells("upmc", region_ids, Path(data_dir), split="discovery")
    reg_ptr = np.asarray(c["region_ptr"], dtype=np.int64)
    reg_counts = np.diff(reg_ptr)
    rid = np.repeat(np.arange(len(reg), dtype=np.int32), reg_counts)
    type_names = list(c["type_names"])

    chunks = []
    marker_names: list[str] | None = None
    with zipfile.ZipFile(zip_path) as zf:
        for r_idx, r in enumerate(region_ids):
            start, end = int(reg_ptr[r_idx]), int(reg_ptr[r_idx + 1])
            cd = pd.read_csv(zf.open(f"raw_data/{r}.cell_data.csv"), usecols=["CELL_ID", "X", "Y"])
            ct = pd.read_csv(zf.open(f"raw_data/{r}.cell_types.csv"), usecols=["CELL_ID", "CELL_TYPE"])
            if len(cd) != (end - start):
                raise ValueError(
                    f"{r}: geometry cell count {end - start} != raw cell_data count {len(cd)}"
                )
            ex = pd.read_csv(zf.open(f"raw_data/{r}.expression.csv"))
            cols = [col for col in ex.columns if col != "CELL_ID"]
            if marker_names is None:
                marker_names = cols
            elif marker_names != cols:
                raise ValueError(f"{r}: inconsistent expression marker headers")
            merged = (
                cd.merge(ct, on="CELL_ID", how="inner", sort=False, validate="one_to_one")
                .merge(ex, on="CELL_ID", how="inner", sort=False, validate="one_to_one")
            )
            if len(merged) != len(cd) or not np.array_equal(
                merged["CELL_ID"].to_numpy(), cd["CELL_ID"].to_numpy()
            ):
                raise ValueError(f"{r}: cell alignment/order mismatch between cell_data, cell_types, and expression")
            expected_xy = merged[["X", "Y"]].to_numpy(np.float64) * cohorts.PIXEL_UM
            if not np.allclose(c["xy_um"][start:end], expected_xy, rtol=0.0, atol=1e-6):
                raise ValueError(f"{r}: cached geometry coordinates do not match raw cell_data row order")
            cached_types = [type_names[code] for code in c["cell_type"][start:end]]
            if cached_types != merged["CELL_TYPE"].astype(str).tolist():
                raise ValueError(f"{r}: cached cell types do not match raw cell_types row order")
            vals = merged[marker_names].to_numpy(np.float64)
            if not np.isfinite(vals).all():
                raise ValueError(f"{r}: non-finite marker intensity in discovery cells")
            chunks.append(vals)

    markers = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, len(marker_names or [])), np.float64)
    return {
        "cohort": "upmc",
        "split": "discovery",
        "data_dir": str(data_dir),
        "_trusted_loader_token": _TRUSTED_LOADER_TOKEN,
        "x": c["xy_um"][:, 0].astype(np.float64),
        "y": c["xy_um"][:, 1].astype(np.float64),
        "cell_type": c["cell_type"].astype(np.int16),
        "markers": markers,
        "marker_names": marker_names or [],
        "type_names": list(c["type_names"]),
        "region_id": rid,
        "patient_of_region": reg.patient_id.map(pidx).to_numpy(np.int32),
        "time": pts.time_days.to_numpy(np.float64),
        "event": pts.event.to_numpy(bool),
        "strata": pts.stratum.to_numpy(np.int32),
        "region_names": region_ids,
        "patient_names": list(pts.patient_id),
    }, None


def _code_revision() -> str:
    root = Path(__file__).resolve().parents[1]
    res = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return res.stdout.strip() if res.returncode == 0 and res.stdout.strip() else "unknown"


def run_translate(
    cohort_data: dict[str, Any] | None = None,
    *,
    data_dir: Path | None = None,
    out_path: Path | None = DEFAULT_OUT,
    split: str = "discovery",
    r16_rule: dict[str, Any] | None = None,
    seed: int = 0,
    n_outer: int = 5,
    n_inner: int = 3,
    n_boot: int = 400,
    n_draws: int = 8,
    misreg_grid: tuple[float, ...] = (0.0, 25.0, 50.0, 75.0, 100.0),
    tile_grid: tuple[float, ...] = (100.0, 150.0, 200.0),
) -> dict[str, Any]:
    """Run R-17 Translate steps 1-2 strictly on discovery patients and write JSON."""
    t0 = time.perf_counter()
    if split != "discovery":
        raise LockedSplitError(f"R-17 must run on split='discovery' only, got {split!r}")

    unmeasured_reason: str | None = None
    if cohort_data is None:
        resolved_dir = cohorts.DATA_DIR if data_dir is None else Path(data_dir)
        cohort_data, unmeasured_reason = _load_upmc_discovery_markers(resolved_dir)
    else:
        validate_discovery_cohort(cohort_data, split=split)

    if unmeasured_reason is not None or cohort_data is None:
        from audit import rediscovery

        rule_meta = dict(r16_rule) if r16_rule is not None else {**rediscovery.RULE, "rule_sha256": rediscovery.RULE_SHA256}
        reason = unmeasured_reason or "discovery cohort unavailable"
        record: dict[str, Any] = {
            "schema_version": 1,
            "cohort": "upmc",
            "split": "discovery",
            "locked_patients_used": False,
            "exploratory": True,
            "already_seen": True,
            "already_seen_disclosure": ALREADY_SEEN_DISCLOSURE,
            "clinical_claims": False,
            "seed": int(seed),
            "code_revision": _code_revision(),
            "status": "UNMEASURED",
            "reason": reason,
            "m": 0,
            "r16_rule": rule_meta,
            "step1_panels": {
                "status": "UNMEASURED",
                "reason": reason,
                "r16_rule_sha256": rule_meta.get("rule_sha256"),
                "m": 0,
                "n_panels": 0,
                "panels": [],
            },
            "step2_formats": {
                "status": "UNMEASURED",
                "reason": reason,
                "r16_rule_sha256": rule_meta.get("rule_sha256"),
                "single_slide": {"status": "UNMEASURED", "reason": reason},
                "serial_section": {str(int(d)): {"misreg_um": float(d), "status": "UNMEASURED", "reason": reason} for d in misreg_grid},
                "tiles": {str(int(t)): {"tile_um": float(t), "status": "UNMEASURED", "reason": reason} for t in tile_grid},
            },
            "labels": {
                "step1_panels": "UNMEASURED",
                "step2_formats": "UNMEASURED",
            },
            "wall_s": time.perf_counter() - t0,
        }
    else:
        resolved_rule = resolve_r16_rule(cohort_data, r16_rule)
        step1 = run_panel_cv(
            cohort_data, split=split, n_outer=n_outer, n_inner=n_inner, n_boot=n_boot, seed=seed, r16_rule=resolved_rule,
        )
        step2 = simulate_formats(
            cohort_data,
            split=split,
            step1_result=step1,
            misreg_grid=misreg_grid,
            tile_grid=tile_grid,
            n_draws=n_draws,
            n_outer=n_outer,
            n_boot=n_boot,
            seed=seed,
            r16_rule=resolved_rule,
        )
        clean_step1 = {k: v for k, v in step1.items() if not k.startswith("_")}
        overall_status = "MEASURED" if step1["status"] == "MEASURED" and step2["status"] == "MEASURED" else "UNMEASURED"
        record = {
            "schema_version": 1,
            "cohort": str(cohort_data.get("cohort", "synthetic")),
            "split": "discovery",
            "locked_patients_used": False,
            "exploratory": True,
            "already_seen": True,
            "already_seen_disclosure": ALREADY_SEEN_DISCLOSURE,
            "clinical_claims": False,
            "seed": int(seed),
            "code_revision": _code_revision(),
            "status": overall_status,
            "reason": step1.get("reason") or step2.get("reason"),
            "m": int(step1["m"]),
            "n_panels": int(step1["n_panels"]),
            "n_patients": int(step1["n_patients"]),
            "n_events": int(step1["n_events"]),
            "r16_rule": resolved_rule,
            "step1_panels": clean_step1,
            "step2_formats": step2,
            "labels": {
                "step1_panels": step1["status"],
                "step2_formats": step2["status"],
            },
            "wall_s": time.perf_counter() - t0,
        }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R-17 Translate steps 1-2 on UPMC discovery split")
    parser.add_argument("--data-dir", type=Path, default=cohorts.DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rec = run_translate(data_dir=args.data_dir, out_path=args.out, seed=args.seed)
    print(f"upmc-translate: status={rec['status']}, m={rec['m']}, reason={rec.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
