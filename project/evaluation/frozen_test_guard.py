from __future__ import annotations

"""Fail-closed guard for any access to the locked BTXRD test split."""

import hashlib
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


def _canonical_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_test_config(
    frozen_config: str | Path | None,
    *,
    split: str,
    split_manifest: str | Path | None = None,
    requested_artifacts: dict[str, str | Path] | None = None,
    checkpoint_any_of: tuple[str, ...] = (),
    requested_checkpoint: str | Path | None = None,
) -> dict[str, object] | None:
    if split != "test":
        return None
    if frozen_config is None:
        raise ValueError("--frozen-config is required before any test-split access")
    path = Path(frozen_config).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    recorded = document.get("freeze_sha256")
    payload = {key: value for key, value in document.items() if key != "freeze_sha256"}
    actual = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if recorded != actual:
        raise ValueError("Frozen config checksum mismatch")
    if document.get("status") != "final":
        raise ValueError("Test access requires a frozen config with status='final'")
    if int(document.get("schema_version", -1)) not in {3, 4}:
        raise ValueError("Test access requires frozen-config schema v3 or v4")

    source = document.get("source") or {}
    expected_commit = str(source.get("git_commit", ""))
    if expected_commit:
        current_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        if current_commit != expected_commit:
            raise ValueError(
                f"Frozen commit {expected_commit} does not match current commit {current_commit}"
            )

    for key in (
        "split_manifest",
        "classifier_checkpoint",
        "classifier_budget_audit",
        "sam_checkpoint",
        "unet_checkpoint",
        "supervised_unet_checkpoint",
    ):
        artifact = document.get(key)
        if not artifact:
            continue
        artifacts = artifact if isinstance(artifact, list) else [artifact]
        for item in artifacts:
            artifact_path = Path(str(item["path"]))
            if not artifact_path.is_file():
                raise FileNotFoundError(f"Frozen artifact is missing: {artifact_path}")
            if _sha256_file(artifact_path) != item.get("sha256"):
                raise ValueError(f"Frozen artifact hash mismatch: {key}")

    if split_manifest is not None:
        requested = Path(split_manifest).resolve()
        frozen_manifest = document.get("split_manifest") or {}
        if str(requested) != str(Path(str(frozen_manifest.get("path", ""))).resolve()):
            raise ValueError("Requested split manifest path differs from frozen config")
        if _sha256_file(requested) != frozen_manifest.get("sha256"):
            raise ValueError("Requested split manifest hash differs from frozen config")
    for key, requested_path_raw in (requested_artifacts or {}).items():
        requested_path = Path(requested_path_raw).resolve()
        frozen_artifact = document.get(key)
        if key == "classifier_checkpoint" and document.get("classifier_checkpoints"):
            frozen_artifact = document["classifier_checkpoints"]
        if frozen_artifact is None:
            frozen_artifact = (document.get("artifacts") or {}).get(key)
        candidates = frozen_artifact if isinstance(frozen_artifact, list) else [frozen_artifact or {}]
        if not any(
            requested_path == Path(str(item.get("path", ""))).resolve()
            and _sha256_file(requested_path) == item.get("sha256")
            for item in candidates
        ):
            raise ValueError(f"Requested artifact differs from frozen config: {key}")
    if requested_checkpoint is not None:
        requested_path = Path(requested_checkpoint).resolve()
        allowed = [document.get(key) or {} for key in checkpoint_any_of]
        if not any(
            requested_path == Path(str(item.get("path", ""))).resolve()
            and _sha256_file(requested_path) == item.get("sha256")
            for item in allowed
        ):
            raise ValueError("Requested checkpoint is not one of the frozen U-Net checkpoints")
    return document
