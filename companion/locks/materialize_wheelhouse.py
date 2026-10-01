"""Download one committed wheel manifest into a new verified wheelhouse."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import validate as lock_validator

FAILURE_RECEIPT = "failed-download.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _retain_failure(
    *,
    profile: str,
    destination: Path,
    row: dict[str, object],
    partial_path: Path,
    error: BaseException,
) -> None:
    partial_exists = partial_path.is_file()
    actual_bytes = partial_path.stat().st_size if partial_exists else 0
    actual_sha256 = (
        _sha256_file(partial_path)
        if partial_exists
        else hashlib.sha256(b"").hexdigest()
    )
    receipt = {
        "status": "FAIL",
        "profile": profile,
        "package": {
            "name": row["name"],
            "version": row["version"],
            "filename": row["filename"],
            "source_index": row["source_index"],
            "url": row["url"],
        },
        "expected": {"bytes": row["bytes"], "sha256": row["sha256"]},
        "actual": {
            "bytes": actual_bytes,
            "sha256": actual_sha256,
            "partial_file": partial_path.name if partial_exists else None,
        },
        "reason": {"type": type(error).__name__, "message": str(error)},
    }
    receipt_path = destination / FAILURE_RECEIPT
    pending_path = destination / f".{FAILURE_RECEIPT}.tmp"
    pending_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    pending_path.replace(receipt_path)


def materialize(profile: str, destination: Path) -> dict[str, int]:
    lock_validator.validate()
    lock_validator.require(
        profile in {*lock_validator.PROFILES, "legacy-test"},
        f"unknown wheelhouse profile: {profile}",
    )
    lock_validator.require(
        not destination.exists(),
        f"destination must not already exist: {destination}",
    )
    destination.mkdir(parents=True)
    manifest = json.loads(
        (lock_validator.LOCKS / f"{profile}.wheels.json").read_text(encoding="utf-8")
    )
    for row in manifest["packages"]:
        final_path = destination / row["filename"]
        partial_path = destination / f"{row['filename']}.part"
        request = urllib.request.Request(
            row["url"],
            headers={"User-Agent": "llm-foundations-wheelhouse-builder/1"},
        )
        digest = hashlib.sha256()
        byte_count = 0
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                with partial_path.open("xb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                        byte_count += len(chunk)
            lock_validator.require(
                byte_count == row["bytes"],
                f"{profile}: {row['filename']} byte count",
            )
            lock_validator.require(
                digest.hexdigest() == row["sha256"],
                f"{profile}: {row['filename']} SHA-256",
            )
            partial_path.replace(final_path)
        except BaseException as error:
            try:
                _retain_failure(
                    profile=profile,
                    destination=destination,
                    row=row,
                    partial_path=partial_path,
                    error=error,
                )
            except Exception as receipt_error:
                raise RuntimeError(
                    f"{error}; failed to retain download receipt: {receipt_error}"
                ) from error
            raise
        print(f"{row['filename']} {byte_count} {digest.hexdigest()}")
    return lock_validator.validate_wheelhouse(profile, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile")
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    try:
        summary = materialize(arguments.profile, arguments.destination)
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps({"status": "PASS", "wheelhouse": summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
