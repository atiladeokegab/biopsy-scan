"""Lock the UPMC held-out patients before any search, and write the manifest the loaders guard on.

The split unit is the clinical tuple (core.cohorts.UPMC_TUPLE), with tuples linked through a shared case
number merged into one unit, so a whole patient stays on one side under every identity reading: the case
number, the tuple, the two tuples that hold two case numbers, a case whose regions disagree on the tuple,
and the unmapped regions. About FRACTION of the units in each event stratum (survival_status, any-cause
death in the unit) are locked, drawn with the frozen SEED.

Run: make heldout (after make data). Rebuilding reproduces the committed manifest or exits 1; it
never overwrites a different lock.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # python -I drops the cwd

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core import cohorts  # noqa: E402
from core.cohorts import UPMC_TUPLE, manifest_hash, upmc_patient_ids  # noqa: E402

SEED = 20261121
FRACTION = 0.3


def linked_units(lab: pd.DataFrame) -> pd.Series:
    """Each region's split unit: its clinical tuple, or the smallest tuple of those linked through shared case numbers."""
    case = lab.sample_label_visualizer.where(lab.sample_label_visualizer != "NA", "").str.split(" ").str[0]
    tup = lab[UPMC_TUPLE].agg("|".join, axis=1)
    key = case.where(case != "", "T:" + tup)
    unit = tup
    while True:  # propagate the smallest tuple through case and tuple links until nothing changes
        new = unit.groupby(key).transform("min").groupby(tup).transform("min")
        if new.equals(unit):
            return unit
        unit = new


def build(labels: Path, seed: int = SEED, fraction: float = FRACTION) -> dict:
    """The manifest for one UPMC labels file: the grouping counts, the conflicts and the locked side."""
    raw = Path(labels).read_bytes()
    lab = pd.read_csv(labels, dtype=str, keep_default_na=False)
    pid, conflicts = upmc_patient_ids(lab)
    unit = linked_units(lab)
    units = lab.survival_status.groupby(unit).max().rename("event").rename_axis("unit").reset_index()
    rng = np.random.default_rng(seed)
    locked_units, strata = set(), {}
    for ev, u in units.groupby("event", sort=True):
        k = int(np.floor(fraction * len(u) + 0.5))
        locked_units |= set(rng.permutation(u.unit.to_numpy())[:k])
        strata[ev] = {"units": len(u), "locked": k}
    lock = unit.isin(locked_units)
    mapped, is_tuple = pid != "", pid.str.startswith("T:")
    side = lambda m: {"units": int(unit[m].nunique()), "patients": int(pid[m & mapped].nunique()),  # noqa: E731
                      "patient_events": int(pid[m & mapped & (lab.survival_status == "1")].nunique()),
                      "regions": int(m.sum()), "unmapped_regions": int((m & ~mapped).sum())}
    m = {
        "cohort": "upmc",
        "labels_sha256": hashlib.sha256(raw).hexdigest(),
        "rule": "core.cohorts.upmc_patient_ids: the case number where present; an NA region is T:<clinical tuple> "
                "when no case-numbered region shares it; conflicts are unmapped and listed",
        "unit": "clinical tuple: " + "|".join(UPMC_TUPLE) + "; tuples sharing a case number form one unit",
        "stratified_by": "survival_status",
        "seed": seed,
        "fraction": fraction,
        "counts": {
            "regions": len(lab),
            "case_patients": int(pid[mapped & ~is_tuple].nunique()),
            "case_regions": int((mapped & ~is_tuple).sum()),
            "tuple_patients": int(pid[is_tuple].nunique()),
            "tuple_regions": int(is_tuple.sum()),
            "patients": int(pid[mapped].nunique()),
            "regions_mapped": int(mapped.sum()),
            "regions_unmapped": int((~mapped).sum()),
            "patient_events": int(pid[mapped & (lab.survival_status == "1")].nunique()),
            "units": len(units),
            "strata": strata,
            "locked": side(lock),
            "discovery": side(~lock),
        },
        "conflicts": conflicts,
        "locked_patients": sorted(set(pid[lock & mapped])),
        "locked_regions": sorted(lab.region_id[lock]),
    }
    m["sha256"] = manifest_hash(m)
    return m


def main(labels: Path = cohorts.DATA_DIR / "upmc_labels.csv", out: Path = cohorts.ROOT / "core" / "upmc_locked.json") -> int:
    m = build(labels)
    text = json.dumps(m, indent=1, sort_keys=True) + "\n"
    if out.exists() and out.read_text() != text:
        print(f"{out} is locked and differs from this rebuild; refusing to overwrite", file=sys.stderr)
        return 1
    verb = "reproduced" if out.exists() else "wrote"
    out.write_text(text)
    print(f"{verb} {out}: sha256 {m['sha256']}")
    print(json.dumps(m["counts"], indent=1))
    for c in m["conflicts"]:
        print(f"conflict: {c['kind']}: {c['group']} ({len(c['regions'])} regions, {', '.join(c['coverslips'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main(*map(Path, sys.argv[1:])))
