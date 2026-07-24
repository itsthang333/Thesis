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


def sha256_source_file(path: Path) -> str:
    """Hash text source with canonical LF newlines for cross-OS portability."""
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def verify_frozen_test_config(
    frozen_config: str | Path | None,
    *,
    split: str,
    split_manifest: str | Path | None = None,
    requested_artifacts: dict[str, str | Path] | None = None,
    checkpoint_any_of: tuple[str, ...] = (),
    requested_checkpoint: str | Path | None = None,
    requested_threshold: float | None = None,
    requested_image_size: int | None = None,
    requested_stage: str | None = None,
    validate_document_only: bool = False,
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
    schema_version = int(document.get("schema_version", -1))
    if schema_version not in {3, 4}:
        raise ValueError("Test access requires frozen-config schema v3 or v4")

    if schema_version == 4:
        source = document.get("source") or {}
        source_files = source.get("files") or []
        if not source_files:
            raise ValueError("Frozen-config schema v4 requires checksum-bound source files")
        for item in source_files:
            relative = Path(str(item.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe frozen source path: {relative}")
            source_path = (REPO_ROOT / relative).resolve()
            try:
                source_path.relative_to(REPO_ROOT.resolve())
            except ValueError as exc:
                raise ValueError(f"Frozen source path escapes repository: {relative}") from exc
            if not source_path.is_file():
                raise FileNotFoundError(f"Frozen source file is missing: {source_path}")
            if sha256_source_file(source_path) != item.get("sha256"):
                raise ValueError(f"Frozen source hash mismatch: {relative.as_posix()}")

        if validate_document_only:
            return document

        allowed_stages = set(document.get("allowed_test_stages") or ())
        if not requested_stage or requested_stage not in allowed_stages:
            raise ValueError(
                f"Frozen config permits only {sorted(allowed_stages)} on test; "
                f"requested stage={requested_stage!r}"
            )

        def verify_portable_artifact(
            key: str,
            requested_path_raw: str | Path | None,
        ) -> None:
            if requested_path_raw is None:
                return
            requested_path = Path(requested_path_raw).resolve()
            if not requested_path.is_file():
                raise FileNotFoundError(requested_path)
            frozen_artifact = document.get(key) or {}
            if _sha256_file(requested_path) != frozen_artifact.get("sha256"):
                raise ValueError(f"Requested artifact hash differs from frozen config: {key}")
            expected_bytes = frozen_artifact.get("bytes")
            if expected_bytes is not None and requested_path.stat().st_size != int(expected_bytes):
                raise ValueError(f"Requested artifact size differs from frozen config: {key}")

        verify_portable_artifact("split_manifest", split_manifest)
        for key, requested_path_raw in (requested_artifacts or {}).items():
            verify_portable_artifact(key, requested_path_raw)
        if requested_checkpoint is not None:
            requested_path = Path(requested_checkpoint).resolve()
            allowed = [document.get(key) or {} for key in checkpoint_any_of]
            requested_hash = _sha256_file(requested_path)
            requested_bytes = requested_path.stat().st_size
            if not any(
                requested_hash == item.get("sha256")
                and (
                    item.get("bytes") is None
                    or requested_bytes == int(item.get("bytes"))
                )
                for item in allowed
            ):
                raise ValueError("Requested checkpoint is not the frozen official WSSS checkpoint")

        evaluation = document.get("evaluation") or {}
        if requested_threshold is None or not np_isclose(
            requested_threshold, evaluation.get("threshold")
        ):
            raise ValueError(
                f"Test threshold must equal frozen value {evaluation.get('threshold')!r}"
            )
        if requested_image_size is None or int(requested_image_size) != int(
            evaluation.get("image_size", -1)
        ):
            raise ValueError(
                f"Test image size must equal frozen value {evaluation.get('image_size')!r}"
            )
        if evaluation.get("threshold_selection_partition") != "val":
            raise ValueError("Frozen threshold must be selected on validation")
        if evaluation.get("threshold_sweep_forbidden") is not True:
            raise ValueError("Frozen config must explicitly forbid threshold sweeping on test")
        return document

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
        artifact_path = Path(str(artifact["path"]))
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Frozen artifact is missing: {artifact_path}")
        if _sha256_file(artifact_path) != artifact.get("sha256"):
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
        frozen_artifact = document.get(key) or {}
        if requested_path != Path(str(frozen_artifact.get("path", ""))).resolve():
            raise ValueError(f"Requested artifact path differs from frozen config: {key}")
        if _sha256_file(requested_path) != frozen_artifact.get("sha256"):
            raise ValueError(f"Requested artifact hash differs from frozen config: {key}")
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


def np_isclose(left: object, right: object, *, tolerance: float = 1e-12) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False
