import hashlib
import json

import pytest

from scripts.check_external_validation_split import check_outcome_opening, check_split, split_patients


def test_split_is_stable_and_disjoint():
    ids = [f"synthetic-{i}" for i in range(9)]
    pilot, reserved = split_patients(ids)
    assert len(pilot) == 3
    assert len(reserved) == 6
    assert set(pilot).isdisjoint(reserved)
    assert set(pilot + reserved) == set(ids)
    assert split_patients(ids[::-1]) == (pilot, reserved)
    check_split(ids, pilot, reserved)


def test_split_rejects_overlap_and_different_assignment():
    ids = [f"synthetic-{i}" for i in range(9)]
    pilot, reserved = split_patients(ids)
    with pytest.raises(ValueError, match="overlap"):
        check_split(ids, pilot, [pilot[0], *reserved])
    with pytest.raises(ValueError, match="frozen patient assignment"):
        check_split(ids, pilot[::-1], reserved)


@pytest.mark.parametrize("ids", [["a", "b", "b"], ["a", "b", " c"], ["a", "b", ""]])
def test_split_rejects_duplicate_or_noncanonical_ids(ids):
    with pytest.raises(ValueError):
        split_patients(ids)


def test_outcome_opening_binds_protocol_and_manifest(tmp_path):
    ids = ["synthetic-a", "synthetic-b", "synthetic-c"]
    pilot, reserved = split_patients(ids)
    manifest_value = {"eligible": ids, "pilot": pilot, "reserved": reserved}
    manifest = tmp_path / "split.json"
    protocol = tmp_path / "protocol.md"
    receipt = tmp_path / "receipt.json"
    manifest.write_text(json.dumps(manifest_value))
    protocol.write_text("Frozen protocol\n")

    def issue_receipt():
        receipt.write_text(json.dumps({
            "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
            "split_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }))

    issue_receipt()
    assert check_outcome_opening(manifest, protocol, receipt) == manifest_value
    protocol.write_text("Different protocol\n")
    with pytest.raises(ValueError, match="receipt does not match"):
        check_outcome_opening(manifest, protocol, receipt)
    protocol.write_text("Frozen protocol\n")
    manifest.write_text(json.dumps({**manifest_value, "event": [False, True, True]}))
    with pytest.raises(ValueError, match="receipt does not match"):
        check_outcome_opening(manifest, protocol, receipt)
    issue_receipt()
    with pytest.raises(ValueError, match="unexpected fields"):
        check_outcome_opening(manifest, protocol, receipt)
