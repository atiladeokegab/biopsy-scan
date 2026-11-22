import numpy as np
import pytest

from bench.modal_fleet_scaling import MODEL, FLEET, split_batches, summarize, validate_point, workload


def _parts(batches, n):
    return [
        {"gpu": {"model": MODEL, "count": 1, "uuid": f"GPU-{i}", "modal_task_id": f"task-{i}"},
         "batches": {mode: {"seeds": seeds, "features": np.array(seeds)[:, None],
                             "compute_s": 5.0, "compute_window": [10 + i, 15 + i]}
                     for mode, seeds in shard.items()}}
        for i, shard in enumerate(split_batches(batches, n))
    ]


def test_strong_weak_shares_only_identical_work_and_checks_all_features():
    base = workload(1, 0)
    _, reference = validate_point(_parts(base, 1), 1, base)
    full = workload(16, 0)
    parts = _parts(full, 16)
    rows, _ = validate_point(parts, 16, full, reference)
    assert rows['weak']['shared_with_strong'] and rows['strong']['studies'] == 256
    assert rows['strong']['compute_span_s'] == 20  # includes stagger between workers

    small = workload(4, 0)
    shards = _parts(small, 4)
    rows, _ = validate_point(shards, 4, small, reference)
    assert rows['weak']['studies'] == 64
    shards[-1]['batches']['weak']['features'][-1] = -1
    with pytest.raises(ValueError, match="features differ"):
        validate_point(shards, 4, small, reference)
    parts[1]['gpu']['uuid'] = parts[0]['gpu']['uuid']
    with pytest.raises(ValueError, match="distinct"):
        validate_point(parts, 16, full, reference)


def test_efficiency_uses_paired_throughput_baseline():
    repeats = [{str(n): {m: {'studies_per_hour': base * n / divisor, 'compute_span_s': 2}
                         for m in ('strong', 'weak')}
                for n in FLEET} for base, divisor in ((100, 1), (200, 2))]
    for repeat in repeats:
        repeat['1']['strong']['studies_per_hour'] = 100
        repeat['1']['weak']['studies_per_hour'] = 100
    assert summarize(repeats)['weak']['16']['efficiency'] == {'median': 1, 'min': 1, 'max': 1}
