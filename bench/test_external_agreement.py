"""bench.external_agreement: squidpy's graph + the grammar = our X; lifelines' Efron score = our score z.

squidpy and lifelines are not project dependencies (lifelines needs pandas<3): these skip without them and run with
the Modal image's pins (bench/modal_external.py PINS), e.g. uv run --with squidpy==1.8.2 --with lifelines==0.30.3
--with anndata==0.12.19 --with pandas==2.3.3.
"""

import numpy as np
import pytest

from bench import external_agreement as ea
from bench import external_baseline as eb
from bench.test_external_baseline import DESIGN, _cells


def test_squidpy_graph_with_the_grammar_gives_our_features():
    pytest.importorskip("squidpy")
    from core.truth import layout
    cells = _cells()
    L = layout(cells)
    sc = eb.study_cells(cells, DESIGN, 5)
    Xg = np.stack([ea.graph_features(looks, L["focal"], L["K"])[0] for looks in sc])
    assert np.allclose(Xg, eb.our_features(sc, L["focal"], L["K"]), rtol=0, atol=1e-12, equal_nan=True)
    # the second row is squidpy's own nhood_enrichment count share (external_baseline's fraction slots)
    Xp = ea.graph_features(sc[0], L["focal"], L["K"])[1]
    Xs = eb.squidpy_features(sc[0], cells["type_names"], L["focal"], seed=0)
    assert np.allclose(Xp[0::2], Xs[0::2], equal_nan=True)


def test_lifelines_efron_score_is_our_stratified_score_z_without_ties():
    pytest.importorskip("lifelines")
    from stats import screen_batch
    rng = np.random.default_rng(3)
    n = 120
    X = rng.normal(size=(n, 4))
    X[:7, 2] = np.nan
    time = rng.exponential(np.exp(-0.6 * X[:, 0]))  # continuous: no tied event times
    event = (rng.random(n) < 0.7).astype(float)
    strata = np.repeat([0, 1, 2], n // 3)
    z = ea.lifelines_score(X, time, event, strata)
    zo = screen_batch(X, time[None], event[None], strata=strata, backend="numpy")["z"][0]
    assert np.allclose(z, zo, atol=1e-9) and z[0] > 3


def test_summarize_counts_top1_per_stream_and_config():
    rng = np.random.default_rng(0)
    X, z = rng.random((10, 6)), rng.normal(size=6)
    same, other = ea.compare(X, z, X, z), ea.compare(X, -z[::-1], X, z)
    recs = [{"stream": "null", "configs": {"m": same}}, {"stream": "alternative", "configs": {"m": other}}]
    s = ea.summarize(recs)["m"]
    assert s["null"]["top1_same"] == {"k": 1, "of": 1} and s["all"]["top1_same"]["of"] == 2
    assert same["x_max_abs_diff"] == 0 and same["z_spearman"] == pytest.approx(1)
