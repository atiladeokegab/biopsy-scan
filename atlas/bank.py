"""Finite patient-acquisition banks: an explicit approximation to fresh imaging."""

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from .artifacts import source_metadata

LOG = logging.getLogger(__name__)


def imaging_descriptor(design):
    values = {key: design.get(key, 0.0 if key == "eps" else None)
              for key in ("cohort", "regions_per_patient", "fov_um", "eps")}
    return {key: value.item() if isinstance(value, np.generic) else value for key, value in values.items()}


def bind_design(design):
    """Give public calibration and evaluation the same scientific identity."""
    bank = design.get("acquisition_bank")
    if bank is not None and bank["imaging"] != imaging_descriptor(design):
        raise ValueError("acquisition bank does not match the imaging design")
    return {**design, "alpha": design.get("alpha", 0.05), "eps": design.get("eps", 0.0),
            "measurement_model": "direct" if bank is None else "finite-bank approximation",
            "bank_key": None if bank is None else bank["key"]}


def build_bank(cells, design, draws, seed, cache_dir, *, batch_size=8, grammar_hash=""):
    """Image each source patient draws times, without generated-outcome selection.

    A study later draws a source patient and an independent uniform bank index
    for each sampled row. Correlations among all features and missing-focal NaNs
    stay together. Calibration is conditional on this finite empirical imaging
    distribution; direct paired checks quantify the approximation gap.
    """
    from reimage import run

    if not isinstance(draws, int) or isinstance(draws, bool) or draws < 2:
        raise ValueError("bank draws must be an integer >= 2")
    source = source_metadata(cells)
    description = {"source_hash": source["source_hash"], "grammar_hash": grammar_hash,
                   "draws": draws, "seed": seed,
                   **imaging_descriptor(design)}
    description = {k: v.item() if isinstance(v, np.generic) else v for k, v in description.items()}
    key = hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / f"bank-{key}.npz"
    start = time.perf_counter()
    cached = path.exists()
    if cached:
        with np.load(path, allow_pickle=False) as saved:
            X = saved["X"]
    else:
        rows = []
        for begin in range(0, draws, batch_size):
            LOG.info("bank %s acquisitions %s-%s/%s", key[:8], begin + 1, min(begin + batch_size, draws), draws)
            seeds = np.array([np.random.SeedSequence([seed, int(key[:8], 16), 505, i])
                              .generate_state(1, dtype=np.uint64)[0]
                              for i in range(begin, min(begin + batch_size, draws))], dtype=np.uint64)
            rows.append(run({**design, "cells": cells, "patients": None}, seeds))
        X = np.concatenate(rows)
    if (X.ndim != 3 or X.shape[:2] != (draws, source["source_n"])
            or X.dtype != np.float64 or np.isinf(X).any()):
        raise ValueError("bank does not match source dimensions/dtype")
    if not cached:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, X=X)
        temporary.replace(path)
    return {"X": X, "key": key, "draws": draws, "seed": seed, "source_hash": source["source_hash"],
            "imaging": imaging_descriptor(design),
            "hash": hashlib.sha256(np.ascontiguousarray(X).view(np.uint8)).hexdigest(),
            "load_or_build_s": time.perf_counter() - start, "cached": cached}


def bank_batch(bank, seeds, patients):
    rows = []
    X = bank["X"]
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        p = rng.integers(0, X.shape[1], patients)  # same first patient draw as the core
        acquisition = rng.integers(0, X.shape[0], patients)
        rows.append(X[acquisition, p])
    return np.stack(rows)
