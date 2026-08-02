from __future__ import annotations

import importlib.util
from pathlib import Path
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project/kaggle_wrappers/run_skelex_reconstruction_selector_s8_audit_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("s8_audit_transport_wrapper", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_safe_extract_rejects_path_traversal_before_write(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    archive = tmp_path / module.TRANSPORT_ARCHIVE_NAME
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", b"escape")
    monkeypatch.setattr(module, "TRANSPORT_ARCHIVE_SHA256", module.hash_file(archive))
    monkeypatch.setattr(module, "TRANSPORT_ARCHIVE_FILE_COUNT", 1)
    monkeypatch.setattr(module, "TRANSPORT_ARCHIVE_UNCOMPRESSED_BYTES", 6)
    with pytest.raises(RuntimeError, match="Unsafe"):
        module.safe_extract_transport_archive(archive, tmp_path / "extract")
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "extract").exists()


def test_safe_extract_and_pair_verification_use_exact_payload(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    payload = tmp_path / "payload"
    root = payload / "producer"
    root.mkdir(parents=True)
    required = {
        "prediction_pair_freeze.json": module.PAIR_FREEZE_SHA256,
        "run_manifest.json": module.RUN_MANIFEST_SHA256,
        "gt_blind_diagnostics.json": module.DIAGNOSTICS_SHA256,
        "reconstruction_evidence/evidence_manifest.json": module.EVIDENCE_MANIFEST_SHA256,
    }
    for arm, hashes in module.ARMS.items():
        required[f"{arm}/prediction_freeze.json"] = hashes["freeze"]
        required[f"{arm}/candidate_scores/candidate_score_manifest.csv"] = hashes["scores"]
        required[f"{arm}/predictions/prediction_manifest.csv"] = hashes["predictions"]
    contents: dict[str, bytes] = {}
    for index, relative in enumerate(required):
        contents[relative] = f"fixture-{index}".encode()
    archive = tmp_path / module.TRANSPORT_ARCHIVE_NAME
    with zipfile.ZipFile(archive, "w") as handle:
        for relative, content in contents.items():
            handle.writestr(f"producer/{relative}", content)
    monkeypatch.setattr(module, "TRANSPORT_ARCHIVE_SHA256", module.hash_file(archive))
    monkeypatch.setattr(module, "TRANSPORT_ARCHIVE_FILE_COUNT", len(contents))
    monkeypatch.setattr(module, "TRANSPORT_ARCHIVE_UNCOMPRESSED_BYTES", sum(map(len, contents.values())))
    monkeypatch.setattr(module, "PAIR_FREEZE_SHA256", module.sha256(contents["prediction_pair_freeze.json"]).hexdigest())
    monkeypatch.setattr(module, "RUN_MANIFEST_SHA256", module.sha256(contents["run_manifest.json"]).hexdigest())
    monkeypatch.setattr(module, "DIAGNOSTICS_SHA256", module.sha256(contents["gt_blind_diagnostics.json"]).hexdigest())
    monkeypatch.setattr(module, "EVIDENCE_MANIFEST_SHA256", module.sha256(contents["reconstruction_evidence/evidence_manifest.json"]).hexdigest())
    for arm, hashes in module.ARMS.items():
        hashes["freeze"] = module.sha256(contents[f"{arm}/prediction_freeze.json"]).hexdigest()
        hashes["scores"] = module.sha256(contents[f"{arm}/candidate_scores/candidate_score_manifest.csv"]).hexdigest()
        hashes["predictions"] = module.sha256(contents[f"{arm}/predictions/prediction_manifest.csv"]).hexdigest()
    verified, transport = module.find_and_verify_producer_output(tmp_path, tmp_path / "extract")
    assert verified.name == "producer"
    assert transport["mode"] == "exact_archive_safe_extract"
    assert module.hash_file(verified / "prediction_pair_freeze.json") == module.PAIR_FREEZE_SHA256
