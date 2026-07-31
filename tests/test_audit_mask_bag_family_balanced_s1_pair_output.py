from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "project" / "audit_mask_bag_family_balanced_s1_pair_output.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("s1_pair_output_auditor", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_family_evidence(root: Path):
    family_root = root / "candidate_families"
    rows: list[dict[str, object]] = []
    expected: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index in range(2):
        image_id = f"image_{index}.jpeg"
        indices = np.asarray([2, 5, 9], dtype=np.int64)
        families = np.asarray([0, 0, 1], dtype=np.int64)
        relative = Path("families") / f"{index}.npz"
        path = family_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            schema_version=np.asarray(1, dtype=np.int64),
            candidate_indices=indices,
            family_ids=families,
        )
        rows.append(
            {
                "image_id": image_id,
                "group_id": f"group_{index}",
                "tumor": index,
                "candidate_payload_sha256": f"{index + 1:064x}",
                "candidate_count": 3,
                "family_count": 2,
                "family_path": str(relative),
                "family_sha256": _sha256(path),
            }
        )
        expected[image_id] = (indices, families)
    manifest = family_root / "candidate_family_manifest.csv"
    _write_csv(manifest, rows)
    return _sha256(manifest), expected


def _write_arm(root: Path, mode: str, expected):
    module = _load_module()
    prediction_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for index, (image_id, (indices, families)) in enumerate(expected.items()):
        logits = np.asarray([-0.2 + index, 0.7 + index, 0.1], dtype=np.float32)
        winner = int(np.argmax(logits))
        score_relative = Path("scores") / f"{index}.npz"
        score_path = root / "candidate_scores" / score_relative
        score_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            score_path,
            schema_version=np.asarray(1, dtype=np.int32),
            candidate_indices=indices,
            candidate_logits=logits,
        )
        bag_logit = (
            module._smooth_pool(logits)
            if mode == "standard"
            else module._family_balanced_pool(logits, families)
        )
        probability = module._sigmoid(bag_logit)
        values = np.zeros((4, 4), dtype=np.float16)
        values[: index + 1, :2] = np.float16(probability)
        map_relative = Path("maps") / f"{index}.npy"
        map_path = root / "predictions" / map_relative
        map_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(map_path, values, allow_pickle=False)
        common = {
            "image_id": image_id,
            "group_id": f"group_{index}",
            "tumor": index,
            "candidate_payload_sha256": f"{index + 1:064x}",
            "candidate_count": 3,
            "selected_candidate_index": int(indices[winner]),
            "selected_candidate_logit": float(logits[winner]),
        }
        score_rows.append(
            {**common, "score_path": str(score_relative), "score_sha256": _sha256(score_path)}
        )
        prediction_rows.append(
            {
                **common,
                "candidate_logit_tta": "mean_original_aligned_horizontal_flip",
                "bag_logit": bag_logit,
                "bag_probability": probability,
                "selected_area_ratio": float((values > 0).mean()),
                "fallback_count": 0,
                "map_path": str(map_relative),
                "map_sha256": _sha256(map_path),
            }
        )
    prediction_manifest = root / "predictions" / "prediction_manifest.csv"
    score_manifest = root / "candidate_scores" / "candidate_score_manifest.csv"
    _write_csv(prediction_manifest, prediction_rows)
    _write_csv(score_manifest, score_rows)
    return {
        "prediction_manifest_sha256": _sha256(prediction_manifest),
        "candidate_score_manifest_sha256": _sha256(score_manifest),
    }


def test_s1_auditor_is_gt_blind_and_evaluator_free() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    lowered = source.lower()
    for forbidden in (
        "datasets.factory",
        "segmentation_dataset",
        "annotation_name",
        'split="test"',
        "candidate_quality",
        "oracle_candidate",
        "evaluate_mask_bag_selector_arm",
    ):
        assert forbidden not in lowered
    assert '"validation_gt_read": False' in source
    assert '"consumer_trained": False' in source
    assert '"test_evaluated": False' in source


def test_family_balanced_pool_is_independently_hierarchical() -> None:
    module = _load_module()
    logits = np.asarray([2.0, 2.0, 0.0], dtype=np.float32)
    families = np.asarray([0, 0, 1], dtype=np.int64)
    family_zero = module._smooth_pool(logits[:2])
    expected = module._smooth_pool(np.asarray([family_zero, logits[2]]))
    assert module._family_balanced_pool(logits, families) == pytest.approx(expected)
    assert module._family_balanced_pool(logits, families) != pytest.approx(
        module._smooth_pool(logits)
    )


def test_float32_reduction_tolerance_is_base_or_four_ulps() -> None:
    module = _load_module()
    assert module._float32_reduction_atol(0.25) == pytest.approx(2.0e-6)
    expected = -10.748272689228461
    tolerance = module._float32_reduction_atol(expected)
    assert tolerance == pytest.approx(
        4.0 * abs(float(np.spacing(np.float32(expected))))
    )
    module._close(
        -10.748274803161621,
        expected,
        name="GPU float32 hierarchical reduction",
        atol=tolerance,
    )
    with pytest.raises(ValueError, match="differs"):
        module._close(
            expected + 8.0 * abs(float(np.spacing(np.float32(expected)))),
            expected,
            name="corrupted hierarchical reduction",
            atol=tolerance,
        )


@pytest.mark.parametrize("mode", ["standard", "family_balanced"])
def test_each_arm_pool_scores_selection_and_maps_are_recomputed(tmp_path: Path, mode: str) -> None:
    module = _load_module()
    manifest_sha, expected = _write_family_evidence(tmp_path)
    families, _bytes = module._verify_family_evidence(
        tmp_path, manifest_sha, expected_validation=2
    )
    arm_root = tmp_path / mode
    freeze = _write_arm(arm_root, mode, expected)
    result = module._verify_arm_validation(
        arm_root,
        freeze,
        families,
        mode=mode,
        expected_validation=2,
        expected_map_shape=(4, 4),
    )
    assert result["physical_validation_maps_verified"] == 2
    assert result["physical_candidate_score_payloads_verified"] == 2


def test_family_manifest_rejects_payload_hash_mismatch(tmp_path: Path) -> None:
    module = _load_module()
    manifest_sha, _expected = _write_family_evidence(tmp_path)
    path = tmp_path / "candidate_families" / "families" / "0.npz"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        module._verify_family_evidence(tmp_path, manifest_sha, expected_validation=2)


def test_launch_binding_must_freeze_protocol_runtime_sources() -> None:
    module = _load_module()
    hashes = {path: "a" * 64 for path in module.REQUIRED_RUNTIME_SOURCES}
    protocol = {"scientific_source": {"canonical_lf_source_hashes": hashes}}
    binding = {
        "schema_version": 1,
        "status": "FROZEN_PRELAUNCH",
        "protocol_sha256": module.PROTOCOL_SHA256,
        "scientific_source_commit": module.SOURCE_COMMIT,
        "kernel": "owner/kernel",
        "kernel_version": 1,
        "checkout_commit": "b" * 40,
        "bound_wrapper_sha256": "c" * 64,
        "runtime_source_hashes": hashes,
    }
    assert module._verify_launch_binding(binding, protocol) == hashes
    binding["status"] = "DRAFT"
    with pytest.raises(ValueError, match="binding contract"):
        module._verify_launch_binding(binding, protocol)


def test_history_requires_exact_ordered_fixed_epochs(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "history.json"
    path.write_text(
        __import__("json").dumps(
            [{"epoch": epoch, "loss": 1.0 / epoch} for epoch in range(1, 17)]
        ),
        encoding="utf-8",
    )
    module._verify_history(path)
    path.write_text('[{"epoch": 1}]', encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 16"):
        module._verify_history(path)
