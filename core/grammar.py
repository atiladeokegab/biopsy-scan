"""The frozen feature grammar and its hash.

Feature c = (focal type t, neighbour type j, radius r, statistic s), flattened in that order:
c = ((t * K + j) * R + r) * S + s, with t over the cohort's tumour types, j over every cell type.
Per focal cell: fraction = n_j(r) / n(r) (0 when no neighbours), enrichment = fraction / p_j,
p_j = share of type j among the region's visible cells (0 when p_j = 0). Patient feature = mean
over the patient's focal cells of type t (all regions pooled: focal-cell weighted).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from core.cohorts import EXPRESSION_CHANNELS, EXPRESSION_TRANSFORM, PIXEL_UM

RADII_UM = (25.0, 50.0, 100.0)
STATS = ("fraction", "enrichment")
GRAMMAR_VERSION = 1
GRAMMAR_V2_VERSION = 2


def is_tumour(name: str) -> bool:
    return name.lower().startswith("tumor")


def grammar(type_names: list[str]) -> dict:
    """The grammar for a cohort's sorted type list; focal = the tumour types."""
    focal = [i for i, n in enumerate(type_names) if is_tumour(n)]
    return {"version": GRAMMAR_VERSION, "type_names": [str(n) for n in type_names], "focal_types": focal,
            "radii_um": list(RADII_UM), "stats": list(STATS), "order": "t,j,r,s",
            "weighting": "focal-cell mean, regions pooled", "pixel_um": PIXEL_UM}


def grammar_v2(type_names: list[str], channels: tuple[str, ...] | list[str] = EXPRESSION_CHANNELS) -> dict:
    """Grammar v2: frozen v1 features plus v2 per-channel neighbour expression/morphology means."""
    focal = [i for i, n in enumerate(type_names) if is_tumour(n)]
    return {
        "version": GRAMMAR_V2_VERSION,
        "type_names": [str(n) for n in type_names],
        "focal_types": focal,
        "radii_um": list(RADII_UM),
        "stats": list(STATS),
        "channels": [str(ch) for ch in channels],
        "transform": EXPRESSION_TRANSFORM,
        "order": "v1(t,j,r,s)_then_expr(t,j,r,m)",
        "weighting": "focal-cell mean, regions pooled",
        "pixel_um": PIXEL_UM,
    }


def n_features(g: dict) -> int:
    base = len(g["focal_types"]) * len(g["type_names"]) * len(g["radii_um"]) * len(g["stats"])
    if "channels" in g:
        return base + len(g["focal_types"]) * len(g["type_names"]) * len(g["radii_um"]) * len(g["channels"])
    return base


def grammar_hash(g: dict) -> str:
    return hashlib.sha256(json.dumps(g, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def feature_names(g: dict) -> list[str]:
    tn = g["type_names"]
    names = [f"{tn[t]}|{tn[j]}|{int(r)}um|{s}" for t in g["focal_types"] for j in range(len(tn))
             for r in g["radii_um"] for s in g["stats"]]
    if "channels" in g:
        names.extend(
            f"{tn[t]}|{tn[j]}|{int(r)}um|{ch}"
            for t in g["focal_types"]
            for j in range(len(tn))
            for r in g["radii_um"]
            for ch in g["channels"]
        )
    return names


def features(S: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Patient features (P, C) from summed region outputs S (P, T, K, R, 2) and focal counts m (P, T).

    A patient with no focal cells of type t gets NaN for that t's features.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        X = S / m[:, :, None, None, None]
    return X.reshape(len(S), -1)


def features_v2(S_counts: np.ndarray, S_expr: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Patient v1+v2 features (P, C_v1 + C_expr) from region sums and focal counts m (P, T)."""
    x1 = features(S_counts, m)
    with np.errstate(invalid="ignore", divide="ignore"):
        x2 = (S_expr / m[:, :, None, None, None]).reshape(len(S_expr), -1)
    return np.concatenate([x1, x2], axis=1)
