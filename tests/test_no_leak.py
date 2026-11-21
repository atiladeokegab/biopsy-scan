"""A-13.3@1: discovery-only patients govern every discovery step; locked values never reach its outputs (offline)."""
import json
import shutil
import zipfile

import numpy as np
import pytest

from core import cohorts, heldout
from core.test_holdout import HEAD, ROWS, data_dir  # noqa: F401  (the shared invented fixture)


def test_discovery_loaders_never_return_locked_regions(data_dir):
    locked = cohorts.locked_regions(data_dir=data_dir)
    assert locked
    for ident in ("case", "tuple", "tuple+case", "patient"):
        assert not locked & set(cohorts.load_regions("upmc", data_dir, ident).region_id)
    assert not locked & set(cohorts.load_cells("upmc", data_dir=data_dir)["region_ids"])
    assert set(cohorts.load_cells("upmc", data_dir=data_dir, split="locked")["region_ids"]) == locked


def test_discovery_reads_an_archive_with_no_locked_files(data_dir):
    """Cold (builds its cache) and warm (reads it), discovery never opens a locked region's files."""
    locked = cohorts.locked_regions(data_dir=data_dir)
    zp = data_dir / "upmc_raw_data.zip"
    with zipfile.ZipFile(zp) as z:
        keep = {n: z.read(n) for n in z.namelist() if n.split("/")[1].split(".")[0] not in locked}
    m = cohorts.EXPRESSION_MARKERS
    with zipfile.ZipFile(zp, "w") as z:
        for n, b in keep.items():
            z.writestr(n, b)
            if n.endswith(".cell_data.csv"):
                r = n[:-len(".cell_data.csv")]
                z.writestr(f"{r}.expression.csv", "\n".join(",".join(v) for v in (["CELL_ID", *m], ["1", *["0.5"] * len(m)],
                                                                                   ["2", *["1.5"] * len(m)])) + "\n")
                z.writestr(f"{r}.cell_features.csv", "CELL_ID,SIZE\n1,10\n2,20\n")
    for _ in ("cold", "warm"):
        c = cohorts.load_cells("upmc", data_dir=data_dir)
        ch = cohorts.load_channels("upmc", data_dir=data_dir)
        assert c["region_ids"] == ch["region_ids"] and not locked & set(c["region_ids"])
        assert len(c["cell_type"]) == len(ch["channels_raw"]) == 2 * len(c["region_ids"])
    with pytest.raises(KeyError):  # the locked side really is missing from this archive
        cohorts.load_cells("upmc", data_dir=data_dir, split="locked")


def test_guard_raises_when_discovery_asks_for_a_locked_region(data_dir):
    r = sorted(cohorts.locked_regions(data_dir=data_dir))[0]
    for loader in (cohorts.load_cells, cohorts.load_channels):
        with pytest.raises(cohorts.LockedPatientError):
            loader("upmc", [r], data_dir)
    assert cohorts.load_cells("upmc", [r], data_dir, split="locked")["region_ids"] == [r]


def test_without_the_guard_the_locked_region_leaks(data_dir, monkeypatch):
    """The red case the guard exists for: with no lock the same request returns the locked cells."""
    r = sorted(cohorts.locked_regions(data_dir=data_dir))[0]
    monkeypatch.setattr(cohorts, "LOCK_FILE", None)
    assert cohorts.load_cells("upmc", [r], data_dir)["region_ids"] == [r]


def test_split_all_reads_both_sides(data_dir):
    every = set(cohorts.load_regions("upmc", data_dir, "tuple", split="all").region_id)
    assert cohorts.locked_regions(data_dir=data_dir) < every and len(every) == len(ROWS)
    r = sorted(cohorts.locked_regions(data_dir=data_dir))[0]
    assert cohorts.load_cells("upmc", [r], data_dir, split="all")["region_ids"] == [r]


def test_labels_other_than_the_locked_ones_are_refused(data_dir):
    with open(data_dir / "upmc_labels.csv", "a") as f:
        f.write("UPMC_c007_v001_r001_reg099,NED,1,0,0,0,9999,a,NA\n")
    for split in cohorts.SPLITS[:2]:
        with pytest.raises(ValueError, match="labels the held-out split was locked on"):
            cohorts.load_regions("upmc", data_dir, split=split)
    with pytest.raises(ValueError, match="labels the held-out split was locked on"):
        cohorts.load_cells("upmc", data_dir=data_dir)


def test_search_entry_point_cannot_reach_locked_regions(data_dir):
    """The real-data search (audit.runner) takes no split, so it can only ever load the discovery side."""
    import inspect
    from dataclasses import replace

    from audit import runner
    assert "split" not in inspect.signature(runner.run_cohort_audit).parameters
    with pytest.raises(TypeError):
        runner.run_cohort_audit("upmc", split="all", out_dir=None, data_dir=data_dir)
    seen = []

    def spy(cohort, **kw):
        seen.append(kw)
        raise StopIteration
    with pytest.raises(StopIteration):
        runner.run_cohort_audit("upmc", out_dir=None, data_dir=data_dir, core_api=replace(runner.load_default_core(), load=spy))
    assert seen == [{"data_dir": data_dir}]                          # no split, so cohorts.load's default
    assert inspect.signature(cohorts.load).parameters["split"].default == "discovery"


def test_only_the_grid_path_passes_split_all():
    """split="all" bypasses the lock: no code but scripts/export_grid.py may pass it."""
    users = {p.relative_to(cohorts.ROOT).as_posix() for p in cohorts.ROOT.rglob("*.py")
             if not {".venv", ".worktrees", "node_modules"} & set(p.relative_to(cohorts.ROOT).parts) and not p.name.startswith("test_")
             and ('split="all"' in p.read_text() or 'SPLIT = "all"' in p.read_text())}
    assert users == {"scripts/export_grid.py"}


def _cohort(d, locked_values=None, n_patients=16):
    """Case-numbered patients with real-looking cells; locked_values perturbs every locked region's
    outcome and cells (same lock, re-hashed for the new labels) to check nothing of them reaches discovery."""
    rows = [(f"UPMC_c00{1 + (p % 2) * 3}_v001_r001_reg{p * 2 + k:03d}", p) for p in range(n_patients) for k in (0, 1)]
    lab, cells = [], {}
    for r, p in rows:
        ev, day = p % 3 == 0, 300 + 97 * p
        if locked_values and r in locked_values:
            ev, day = not ev, day + 41
        lab.append(f"{r},{'DOD' if ev else 'NED'},0,0,0,{int(ev)},{day},a,HN-{p:03d} A_1\n")
        rng = np.random.default_rng(int(r[-3:]) + (1000 if locked_values and r in locked_values else 0))
        n = int(rng.integers(150, 250))
        types = ["Tumor", "Stroma", "B cell", "CD8 T cell"] + (["Locked-only type"] if locked_values and r in locked_values else [])
        cells[r] = (rng.random((n, 2)) * 800, rng.choice(types, n))
    d.mkdir()
    (d / "upmc_labels.csv").write_text(HEAD + "".join(lab))
    with zipfile.ZipFile(d / "upmc_raw_data.zip", "w") as z:
        for r, (xy, ty) in cells.items():
            ids = np.arange(len(ty))
            z.writestr(f"raw_data/{r}.cell_data.csv", "CELL_ID,X,Y\n" + "".join(f"{i},{x:.3f},{y:.3f}\n" for i, (x, y) in zip(ids, xy)))
            z.writestr(f"raw_data/{r}.cell_types.csv", "CELL_ID,CELL_TYPE\n" + "".join(f"{i},{t}\n" for i, t in zip(ids, ty)))
    return d


def _discovery_outputs(d):
    from audit import runner
    from core.truth import truth
    X = truth(cohorts.load("upmc", d))
    out = runner.run_cohort_audit("upmc", out_dir=None, data_dir=d, n_resamples=199, seed=0)
    return X.tobytes(), json.dumps(out, sort_keys=True, allow_nan=True)  # no wall-clock fields in either


def test_changing_locked_patients_leaves_discovery_outputs_byte_identical(tmp_path, monkeypatch):
    a = _cohort(tmp_path / "a")
    m = heldout.build(a / "upmc_labels.csv")
    locked = set(m["locked_regions"])
    assert locked and m["counts"]["discovery"]["patient_events"] > 0
    b = _cohort(tmp_path / "b", locked_values=locked)
    mb = dict(m, labels_sha256=heldout.hashlib.sha256((b / "upmc_labels.csv").read_bytes()).hexdigest())
    mb["sha256"] = cohorts.manifest_hash(mb)
    for d, man in ((a, m), (b, mb)):
        (d / "lock.json").write_text(json.dumps(man))
    shutil.copytree(a, tmp_path / "a_open")
    shutil.copytree(b, tmp_path / "b_open")

    def run(d):
        monkeypatch.setattr(cohorts, "LOCK_FILE", d / "lock.json")
        return _discovery_outputs(d)
    assert run(a) == run(b)                                   # candidates, statistics, ranks: byte for byte
    monkeypatch.setattr(cohorts, "LOCK_FILE", None)           # red case: without the lock the change shows
    assert _discovery_outputs(tmp_path / "a_open") != _discovery_outputs(tmp_path / "b_open")
