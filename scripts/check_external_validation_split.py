"""Validate the Preflight Basel split and bind outcome opening to reviewed inputs."""

import argparse
import hashlib
import json
from pathlib import Path

SPLIT_NAMESPACE = b"preflight-basel-external-v1\0" + b"20261123\0"
SPLIT_KEYS = {"eligible", "pilot", "reserved"}
RECEIPT_KEYS = {"protocol_sha256", "split_sha256"}


def _unique_ids(name, patients):
    if not isinstance(patients, (list, tuple)) or not patients:
        raise ValueError(f"{name} must be a nonempty list of patient IDs")
    if any(not isinstance(patient, str) or not patient or patient != patient.strip()
           or not patient.isprintable() for patient in patients):
        raise ValueError(f"{name} contains a noncanonical patient ID")
    if len(set(patients)) != len(patients):
        raise ValueError(f"{name} contains duplicate patient IDs")


def split_patients(eligible_ids):
    """Return the frozen one-third pilot split in deterministic patient order."""
    _unique_ids("eligible", eligible_ids)
    if len(eligible_ids) < 3:
        raise ValueError("the split requires at least three eligible patients")
    ordered = sorted(
        eligible_ids,
        key=lambda patient: (
            hashlib.sha256(SPLIT_NAMESPACE + patient.encode("utf-8")).digest(),
            patient,
        ),
    )
    return ordered[:len(ordered) // 3], ordered[len(ordered) // 3:]


def check_split(eligible_ids, pilot_ids, reserved_ids):
    """Reject overlap, duplicate IDs and changes to the preregistered split."""
    expected_pilot, expected_reserved = split_patients(eligible_ids)
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


def _read_split(data):
    manifest = _json_fields(data, SPLIT_KEYS, "split manifest")
    check_split(manifest["eligible"], manifest["pilot"], manifest["reserved"])
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
