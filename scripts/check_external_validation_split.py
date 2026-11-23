"""Validate Preflight cohort splits and bind outcome opening to reviewed inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import re

SPLIT_NAMESPACE = b"preflight-basel-external-v1\0" + b"20261123\0"
SPLIT_KEYS = {"eligible", "pilot", "reserved"}
RECEIPT_KEYS = {"protocol_sha256", "split_sha256"}
METABRIC_NAMESPACE = b"preflight-metabric-external-v2\0" + b"20261123\0"
METABRIC_PROTOCOL_SHA256 = "8e2e1e83a8ecb0c5f650561bbcc7042b6c2625dfca67ecc83b147da97bce2661"
METABRIC_SPLIT_SHA256 = "920e4759f9c10fd0bf71b4cf6178104b97cb8d1512c459ecff7ae8cd87089529"


def _unique_ids(name, patients):
    if not isinstance(patients, (list, tuple)) or not patients:
        raise ValueError(f"{name} must be a nonempty list of patient IDs")
    if any(not isinstance(patient, str) or not patient or patient != patient.strip()
           or not patient.isprintable() for patient in patients):
        raise ValueError(f"{name} contains a noncanonical patient ID")
    if len(set(patients)) != len(patients):
        raise ValueError(f"{name} contains duplicate patient IDs")


def split_patients(eligible_ids, *, namespace=SPLIT_NAMESPACE):
    """Return the frozen one-third pilot split in deterministic patient order."""
    _unique_ids("eligible", eligible_ids)
    if len(eligible_ids) < 3:
        raise ValueError("the split requires at least three eligible patients")
    ordered = sorted(
        eligible_ids,
        key=lambda patient: (
            hashlib.sha256(namespace + patient.encode("utf-8")).digest(),
            patient,
        ),
    )
    return ordered[:len(ordered) // 3], ordered[len(ordered) // 3:]


def check_split(eligible_ids, pilot_ids, reserved_ids, *, namespace=SPLIT_NAMESPACE):
    """Reject overlap, duplicate IDs and changes to the preregistered split."""
    expected_pilot, expected_reserved = split_patients(eligible_ids, namespace=namespace)
    _unique_ids("pilot", pilot_ids)
    _unique_ids("reserved", reserved_ids)
    if set(pilot_ids).intersection(reserved_ids):
        raise ValueError("pilot and reserved patient IDs overlap")
    if list(pilot_ids) != expected_pilot or list(reserved_ids) != expected_reserved:
        raise ValueError("manifest does not match the frozen patient assignment")


def _json_fields(data, expected, label):
    def unique_keys(pairs):
        fields = dict(pairs)
        if len(fields) != len(pairs):
            raise ValueError(f"{label} contains duplicate keys")
        return fields

    value = json.loads(data, object_pairs_hook=unique_keys)
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} has unexpected fields")
    return value


def _read_split(data, *, namespace=SPLIT_NAMESPACE):
    manifest = _json_fields(data, SPLIT_KEYS, "split manifest")
    check_split(manifest["eligible"], manifest["pilot"], manifest["reserved"], namespace=namespace)
    return manifest


def check_outcome_opening(manifest_path, protocol_path, receipt_path):
    """Check the reviewer receipt before the caller reads reserved outcome values."""
    manifest_bytes = Path(manifest_path).read_bytes()
    protocol_bytes = Path(protocol_path).read_bytes()
    receipt = _json_fields(Path(receipt_path).read_bytes(), RECEIPT_KEYS, "opening receipt")
    expected = {
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "split_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    if receipt != expected:
        raise ValueError("opening receipt does not match the protocol and split files")
    return _read_split(manifest_bytes)


def _read_metabric_split(manifest_bytes, protocol_bytes):
    if hashlib.sha256(protocol_bytes).hexdigest() != METABRIC_PROTOCOL_SHA256:
        raise ValueError("unexpected METABRIC protocol revision")
    if hashlib.sha256(manifest_bytes).hexdigest() != METABRIC_SPLIT_SHA256:
        raise ValueError("unexpected METABRIC split revision")
    manifest = _read_split(manifest_bytes, namespace=METABRIC_NAMESPACE)
    if manifest["eligible"] != manifest["pilot"] + manifest["reserved"]:
        raise ValueError("METABRIC eligible IDs must be in frozen hash order")
    if any(re.fullmatch(r"MB-[0-9]+", patient) is None for patient in manifest["eligible"]):
        raise ValueError("METABRIC manifest contains a noncanonical patient ID")
    if min(len(manifest["pilot"]), len(manifest["reserved"])) < 100:
        raise ValueError("METABRIC requires at least 100 patients per partition")
    return manifest


def read_metabric_split(manifest_path, protocol_path):
    """Check the frozen protocol and ordered manifest before the pilot."""
    return _read_metabric_split(Path(manifest_path).read_bytes(), Path(protocol_path).read_bytes())


def check_metabric_outcome_opening(manifest_path, protocol_path, receipt_path, pilot_checkpoint_path):
    """Check the METABRIC opening receipt and return the verified manifest and pilot."""
    manifest_bytes = Path(manifest_path).read_bytes()
    protocol_bytes = Path(protocol_path).read_bytes()
    pilot_bytes = Path(pilot_checkpoint_path).read_bytes()
    receipt = _json_fields(Path(receipt_path).read_bytes(), RECEIPT_KEYS | {"pilot_sha256"}, "opening receipt")
    expected = {
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "split_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "pilot_sha256": hashlib.sha256(pilot_bytes).hexdigest(),
    }
    if receipt != expected:
        raise ValueError("METABRIC opening receipt does not match the protocol, split and pilot files")
    manifest = _read_metabric_split(manifest_bytes, protocol_bytes)
    pilot = json.loads(pilot_bytes)
    if not isinstance(pilot, dict) or pilot.get("partition") != "pilot" or pilot.get("preferred") not in ("A", "B"):
        raise ValueError("METABRIC opening requires a pilot recommendation")
    return manifest, pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if (args.protocol is None) != (args.receipt is None):
        parser.error("--protocol and --receipt must be supplied together")
    try:
        manifest = (
            check_outcome_opening(args.manifest, args.protocol, args.receipt)
            if args.receipt else _read_split(args.manifest.read_bytes())
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Split check failed: {exc}\n")
    print(json.dumps({key: len(manifest[key]) for key in ("eligible", "pilot", "reserved")}))


if __name__ == "__main__":
    main()
