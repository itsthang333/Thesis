from __future__ import annotations

"""Fail-closed materialization of canonical UTF-8 LF text artifacts."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def validate_sha256(value: str, *, name: str) -> str:
    value = value.strip().lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def canonical_lf_bytes(payload: bytes) -> bytes:
    text = payload.decode("utf-8", errors="strict")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def materialize_canonical_lf(
    input_path: Path,
    output_path: Path,
    *,
    expected_input_sha256: str,
    expected_output_sha256: str,
) -> dict[str, Any]:
    expected_input = validate_sha256(
        expected_input_sha256,
        name="expected input SHA-256",
    )
    expected_output = validate_sha256(
        expected_output_sha256,
        name="expected output SHA-256",
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"Canonical-LF input does not exist: {input_path}")
    if output_path.exists():
        raise FileExistsError(f"Canonical-LF output already exists: {output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(
            f"Canonical-LF output parent does not exist: {output_path.parent}"
        )
    raw = input_path.read_bytes()
    raw_sha256 = sha256_bytes(raw)
    if raw_sha256 != expected_input:
        raise ValueError(
            f"Canonical-LF input SHA-256 mismatch: expected {expected_input}, got {raw_sha256}"
        )
    canonical = canonical_lf_bytes(raw)
    canonical_sha256 = sha256_bytes(canonical)
    if canonical_sha256 != expected_output:
        raise ValueError(
            "Canonical-LF output SHA-256 mismatch before write: "
            f"expected {expected_output}, got {canonical_sha256}"
        )
    with output_path.open("xb") as handle:
        handle.write(canonical)
    physical_sha256 = sha256_bytes(output_path.read_bytes())
    if physical_sha256 != expected_output:
        raise RuntimeError("Canonical-LF physical output differs after write")
    return {
        "status": "CANONICAL_LF_MATERIALIZATION_PASS",
        "input_path": str(input_path.resolve()),
        "input_bytes": len(raw),
        "input_sha256": raw_sha256,
        "output_path": str(output_path.resolve()),
        "output_bytes": len(canonical),
        "output_sha256": physical_sha256,
        "crlf_pairs_removed": raw.count(b"\r\n"),
        "lone_cr_removed": raw.replace(b"\r\n", b"").count(b"\r"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-output-sha256", required=True)
    args = parser.parse_args()
    result = materialize_canonical_lf(
        args.input,
        args.output,
        expected_input_sha256=args.expected_input_sha256,
        expected_output_sha256=args.expected_output_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
