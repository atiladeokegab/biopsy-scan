import json

import pytest

from bench import modal_h100_design as h100


def test_all_historical_gpu_studies_match_fused_reference():
    receipt = json.loads(h100.ARCHIVE.read_text())
    reference = h100.archived_reference(receipt)
    gpu = receipt['arms']['gpu']['studies']
    parity = h100.validate_records(gpu, reference)
    assert parity['studies_compared'] == 3000 and parity['max_abs_maxT_delta'] < 1e-9

    with pytest.raises(ValueError, match='missing, duplicated'):
        h100.validate_records(gpu[:-1] + gpu[:1], reference)
    with pytest.raises(ValueError, match='feature hash'):
        h100.validate_records([gpu[0] | {'x_sha': 'different'}, *gpu[1:]], reference)
    with pytest.raises(ValueError, match='absolute 1e-9'):
        h100.validate_records([gpu[0] | {'maxT': gpu[0]['maxT'] + 1e-7}, *gpu[1:]], reference)


def test_gate_aborts_before_work_if_workers_are_missing(monkeypatch):
    ticks = iter((0.0, 101.0, 102.0))
    monkeypatch.setattr(h100.time, "perf_counter", lambda: next(ticks))
    ready = {}
    with pytest.raises(TimeoutError, match="only 1 of 2.*allocated"):
        h100._rendezvous(ready, "first", 2, prepared=False)
    with pytest.raises(TimeoutError, match="failed its allocated gate"):
        h100._rendezvous(ready, "late", 2, prepared=False)


def test_abort_between_reads_cannot_be_counted_as_a_worker():
    class AbortDuringRead(dict):
        def get(self, key, default=None):
            value = super().get(key, default)
            self["aborted"] = True
            return value

        def keys(self):
            self["aborted"] = True
            return super().keys()

        len = dict.__len__

    with pytest.raises(TimeoutError, match="failed its allocated gate"):
        h100._rendezvous(AbortDuringRead(), "first", 2, prepared=False)


def test_preparation_gate_counts_only_prepared_workers(monkeypatch):
    ticks = iter((0.0, 101.0))
    monkeypatch.setattr(h100.time, "perf_counter", lambda: next(ticks))
    ready = {"allocated:first": True, "allocated:second": True, "unrelated": True}
    with pytest.raises(TimeoutError, match="only 1 of 2.*prepared"):
        h100._rendezvous(ready, "first", 2, prepared=True)
