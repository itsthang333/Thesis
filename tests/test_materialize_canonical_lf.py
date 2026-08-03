from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from project.materialize_canonical_lf import (
    canonical_lf_bytes,
    materialize_canonical_lf,
)


def _sha(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def test_materializes_verified_crlf_and_lone_cr_to_lf(tmp_path: Path) -> None:
    raw = b'{\r\n  "status": "ready"\r}\r\n'
    canonical = b'{\n  "status": "ready"\n}\n'
    source = tmp_path / "raw.json"
    output = tmp_path / "canonical.json"
    source.write_bytes(raw)

    audit = materialize_canonical_lf(
        source,
        output,
        expected_input_sha256=_sha(raw),
        expected_output_sha256=_sha(canonical),
    )

    assert output.read_bytes() == canonical
    assert audit["status"] == "CANONICAL_LF_MATERIALIZATION_PASS"
    assert audit["crlf_pairs_removed"] == 2
    assert audit["lone_cr_removed"] == 1
    assert audit["output_sha256"] == _sha(canonical)


def test_lf_input_is_idempotent(tmp_path: Path) -> None:
    raw = b"alpha\nbeta\n"
    source = tmp_path / "raw.txt"
    output = tmp_path / "canonical.txt"
    source.write_bytes(raw)
    materialize_canonical_lf(
        source,
        output,
        expected_input_sha256=_sha(raw),
        expected_output_sha256=_sha(raw),
    )
    assert output.read_bytes() == raw


def test_rejects_wrong_input_hash_before_output(tmp_path: Path) -> None:
    source = tmp_path / "raw.txt"
    output = tmp_path / "canonical.txt"
    source.write_bytes(b"alpha\r\n")
    with pytest.raises(ValueError, match="input SHA-256 mismatch"):
        materialize_canonical_lf(
            source,
            output,
            expected_input_sha256="0" * 64,
            expected_output_sha256=_sha(b"alpha\n"),
        )
    assert not output.exists()


def test_rejects_wrong_output_hash_before_output(tmp_path: Path) -> None:
    raw = b"alpha\r\n"
    source = tmp_path / "raw.txt"
    output = tmp_path / "canonical.txt"
    source.write_bytes(raw)
    with pytest.raises(ValueError, match="output SHA-256 mismatch before write"):
        materialize_canonical_lf(
            source,
            output,
            expected_input_sha256=_sha(raw),
            expected_output_sha256="0" * 64,
        )
    assert not output.exists()


def test_refuses_existing_output(tmp_path: Path) -> None:
    raw = b"alpha\r\n"
    source = tmp_path / "raw.txt"
    output = tmp_path / "canonical.txt"
    source.write_bytes(raw)
    output.write_bytes(b"do-not-overwrite")
    with pytest.raises(FileExistsError):
        materialize_canonical_lf(
            source,
            output,
            expected_input_sha256=_sha(raw),
            expected_output_sha256=_sha(canonical_lf_bytes(raw)),
        )
    assert output.read_bytes() == b"do-not-overwrite"


def test_rejects_invalid_utf8(tmp_path: Path) -> None:
    raw = b"\xff\r\n"
    source = tmp_path / "raw.bin"
    output = tmp_path / "canonical.bin"
    source.write_bytes(raw)
    with pytest.raises(UnicodeDecodeError):
        materialize_canonical_lf(
            source,
            output,
            expected_input_sha256=_sha(raw),
            expected_output_sha256="0" * 64,
        )
    assert not output.exists()
