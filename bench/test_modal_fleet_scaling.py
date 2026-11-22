import numpy as np
import pytest

from bench.modal_fleet_scaling import MODEL, FLEET, split_batches, summarize, validate_point, workload


def _parts(batches, n):
    return [
        {"gpu": {"model": MODEL, "count": 1, "uuid": f"GPU-{i}", "modal_task_id": f"task-{i}"},
         "batches": {mode: {"seeds": seeds, "features": np.array(seeds)[:, None, None],
                             "compute_s": 5.0, "compute_window": [10 + i / 10, 15 + i / 10]}
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
    assert rows['strong']['compute_span_s'] == 6.5  # includes stagger between workers

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

    shards = _parts(small, 4)
    shards[-1]['batches']['strong']['compute_window'] = [16, 21]
    with pytest.raises(ValueError, match="common overlap"):
        validate_point(shards, 4, small, reference)


def test_efficiency_uses_paired_throughput_baseline():
    repeats = [{str(n): {m: {'studies_per_hour': base * n / divisor, 'compute_span_s': 2}
                         for m in ('strong', 'weak')}
                for n in FLEET} for base, divisor in ((100, 1), (200, 2))]
    repeats[1]['1']['strong']['studies_per_hour'] = 200
    repeats[1]['1']['weak']['studies_per_hour'] = 200
    assert summarize(repeats)['weak']['16']['efficiency'] == {'median': .75, 'min': .5, 'max': 1}


def test_validation_accepts_actual_reimage_output():
    from bench.fleet import synthetic_cohort
    from reimage import run

    cells, _ = synthetic_cohort(n_patients=2, cells=24)
    batches = {'strong': [0, 1], 'weak': [0]}
    parts = _parts(batches, 1)
    design = {'cells': cells, 'patients': 2, 'regions_per_patient': 2, 'fov_um': 400, 'eps': .1}
    for mode, seeds in batches.items():
        parts[0]['batches'][mode]['features'] = run(design, seeds)
    rows, reference = validate_point(parts, 1, batches)
    assert reference.shape[:2] == (2, 2)
    assert rows['weak']['exact_features_vs_one_gpu']
