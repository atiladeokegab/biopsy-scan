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

RADII_UM = (25.0, 50.0, 100.0)
STATS = ("fraction", "enrichment")
GRAMMAR_VERSION = 1


def is_tumour(name: str) -> bool:
    return name.lower().startswith("tumor")


def grammar(type_names: list[str]) -> dict:
    """The grammar for a cohort's sorted type list; focal = the tumour types."""
    focal = [i for i, n in enumerate(type_names) if is_tumour(n)]
    return {"version": GRAMMAR_VERSION, "type_names": list(type_names), "focal_types": focal,
            "radii_um": list(RADII_UM), "stats": list(STATS), "order": "t,j,r,s",
            "weighting": "focal-cell mean, regions pooled"}


def n_features(g: dict) -> int:
    return len(g["focal_types"]) * len(g["type_names"]) * len(g["radii_um"]) * len(g["stats"])


def grammar_hash(g: dict) -> str:
    return hashlib.sha256(json.dumps(g, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def feature_names(g: dict) -> list[str]:
    tn = g["type_names"]
    return [f"{tn[t]}|{tn[j]}|{int(r)}um|{s}" for t in g["focal_types"] for j in range(len(tn))
            for r in g["radii_um"] for s in g["stats"]]


def features(S: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Patient features (P, C) from summed region outputs S (P, T, K, R, 2) and focal counts m (P, T).

    A patient with no focal cells of type t gets NaN for that t's features.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        X = S / m[:, :, None, None, None]
    return X.reshape(len(S), -1)
