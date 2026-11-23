"""Import the pinned Jackson/Fischer Basel tumour geometry.

The private geometry export contains no follow-up fields. The validation runner
checks the frozen split and opening receipt before requesting reserved outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EXPORTER = REPO / "scripts" / "export_external_basel.R"
SOURCE_DOI = "10.5281/zenodo.20621983"
SOURCE_MD5 = "ddbc54e6e8a604a8febabd5a8ad2cd1a"
ARCHIVES = {1_248_797_988: SOURCE_MD5, 283_028_201: "c704430733435c8866438362edec4274"}
# Jackson and Fischer et al. (2020) Methods equate four pixels to four micrometres.
PIXEL_UM = 1.0
TYPE_NAMES = ("Tumor", "T cell", "Other")
FEATURE = "Tumor|T cell|50um|fraction"
SIDES_UM = (300.0, 300.0 * np.sqrt(2))
EVENT = "death by primary disease"
CENSORED = ("alive", "alive w metastases", "death")
GEOMETRY = ("patient_cohort", "patient_id", "image_name", "cell_number",
            "image_width", "image_height", "cell_x", "cell_y", "cell_metacluster")


def private_path(path: Path) -> Path:
    path = path.resolve()
    # A coded archive, manifest or cell table must not be stored in a Git checkout.
    for parent in (path, *path.parents):
        marker = parent / ".git"
        if marker.is_file() or (marker / "HEAD").is_file():
            raise ValueError("keep coded Basel files outside Git checkouts")
    return path


def verify_archive(path: Path) -> Path:
    path = private_path(path)
    checksum = ARCHIVES.get(path.stat().st_size)
    if checksum is None:
        raise ValueError("unexpected size for the pinned Basel archive")
    with path.open("rb") as src:
        if hashlib.file_digest(src, "md5").hexdigest() != checksum:
            raise ValueError("the Basel archive does not match the published checksum")
    return path


def export_geometry(archive: Path, output: Path, rscript: str = "Rscript") -> None:
    archive, output = verify_archive(archive), private_path(output)
    if output == archive:
        raise ValueError("geometry output must differ from the source archive")
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        csv = Path(scratch) / "geometry.csv"
        subprocess.run([rscript, "--vanilla", str(EXPORTER), str(archive), str(csv)],
                       check=True, timeout=90)
        frame = pd.read_csv(csv, dtype={"patient_id": str, "image_name": str}, keep_default_na=False)
        prepare_cells(frame)
        target = Path(scratch) / "geometry.parquet"
        frame.to_parquet(target, index=False)
        target.chmod(0o600)
        os.replace(target, output)


def prepare_cells(frame: pd.DataFrame) -> dict:
    """Build the engine contract in sorted patient/image/cell order, with reported bounds."""
    if set(frame) != set(GEOMETRY) or frame.empty or frame[list(GEOMETRY)].isna().any().any():
        raise ValueError("incomplete or unexpected geometry schema")
    if not frame["patient_cohort"].eq("Basel").all():
        raise ValueError("expected only Basel images")
    frame = frame.copy()
    for field in ("patient_id", "image_name"):
        if not frame[field].map(lambda item: isinstance(item, str) and bool(item.strip()) and item == item.strip()).all():
            raise ValueError("invalid patient or image key")
    for field in ("image_width", "image_height", "cell_x", "cell_y", "cell_number", "cell_metacluster"):
        frame[field] = pd.to_numeric(frame[field], errors="raise")
        if not np.isfinite(frame[field]).all():
            raise ValueError(f"nonfinite {field}")
    if not (frame["cell_number"].ge(1) & frame["cell_number"].mod(1).eq(0)).all():
        raise ValueError("cell identifiers must be positive integers")
    if frame.duplicated(["image_name", "cell_number"]).any():
        raise ValueError("duplicate image/cell key")

    image_fields = ["image_name", "patient_id", "image_width", "image_height"]
    images = frame[image_fields].drop_duplicates()
    if images["image_name"].duplicated().any():
        raise ValueError("conflicting patient or size for an image")
    if not (frame["image_width"].gt(0) & frame["image_height"].gt(0)
            & frame["cell_x"].ge(0) & frame["cell_x"].lt(frame["image_width"])
            & frame["cell_y"].ge(0) & frame["cell_y"].lt(frame["image_height"])).all():
        raise ValueError("centroid outside reported image bounds")
    clusters = frame["cell_metacluster"]
    if not clusters.isin(range(1, 28)).all():
        raise ValueError("unknown Basel metacluster")

    fits = images[["image_width", "image_height"]].min(axis=1) * PIXEL_UM >= SIDES_UM[-1]
    images = images.loc[fits].assign(area=lambda rows: rows["image_width"] * rows["image_height"])
    images = (images.sort_values(["patient_id", "area", "image_name"], ascending=[True, False, True])
              .drop_duplicates("patient_id"))
    frame = frame[frame["image_name"].isin(images["image_name"])].copy()
    clusters = frame["cell_metacluster"]

    # The published metaclusters 14–26 are tumours, 3 and 5 are pure T cells.
    # Retain all remaining cells in the denominator of the neighbor fraction.
    frame["cell_type"] = np.select([clusters.between(14, 26), clusters.isin([3, 5])], [0, 1], default=2)
    frame["region_id"] = frame["image_name"].map({name: i for i, name in enumerate(images["image_name"])})
    ordered = frame.sort_values(["region_id", "cell_number"])
    return {"region_id": ordered["region_id"].to_numpy(np.int64),
            "x": ordered["cell_x"].to_numpy(float) * PIXEL_UM,
            "y": ordered["cell_y"].to_numpy(float) * PIXEL_UM,
            "cell_type": ordered["cell_type"].to_numpy(np.int64),
            "type_names": list(TYPE_NAMES),
            "region_names": images["image_name"].tolist(), "patient_names": images["patient_id"].tolist(),
            "patient_of_region": np.arange(len(images)),
            "image_size_um": images[["image_width", "image_height"]].to_numpy(float) * PIXEL_UM}


def align_outcomes(frame: pd.DataFrame, requested: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if set(frame) != {"patient_id", "patient_status", "patient_OS_months"}:
        raise ValueError("unexpected clinical outcome schema")
    if frame["patient_id"].duplicated().any() or len(requested) != len(set(requested)):
        raise ValueError("duplicate or conflicting outcome records")
    if set(frame["patient_id"]) != set(requested):
        raise ValueError("outcomes must match exactly the requested patient IDs")
    aligned = frame.set_index("patient_id").loc[requested]
    status = aligned["patient_status"].str.strip().str.lower()
    months = pd.to_numeric(aligned["patient_OS_months"], errors="raise").to_numpy(float)
    if not status.isin((EVENT, *CENSORED)).all() or not (np.isfinite(months) & (months > 0)).all():
        raise ValueError("unknown status or unusable disease-specific follow-up")
    return months, status.eq(EVENT).to_numpy(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the pinned Basel geometry into a private Parquet file.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    export_geometry(args.archive, args.geometry, args.rscript)


if __name__ == "__main__":
    main()
