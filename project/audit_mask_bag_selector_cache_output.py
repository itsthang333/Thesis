from __future__ import annotations

"""Independent GT-blind audit for a completed mask-bag selector cache."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from models.mask_bag_selector_cache_io import (
    PackedCandidateMasks,
    load_selector_cache_record,
)


GIT_SPLIT_SHA256 = (
    "43662d5d7969ae2a5bc61c6a0de3e0c392debef19c98d809f7d9bdfd0abb2fa8"
)
FROZEN_SPLIT_SHA256 = (
    "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
)
MODEL_HASHES = {
    "config.json": "89daf9751d9576d586dedf9543c1083211611fa3a36908db7a799b3ce7c68ede",
    "preprocessor_config.json": "c537fc995c30e2353f07253899618d60e9eae3d5f82473778602c007c6523b56",
    "model.safetensors": "dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae",
}
PROJECTION_SHA256 = (
    "5cbb6846ca1b185fda50b0843951985422ce7e7782fd897639e3238cf9b567ec"
)
TRAIN_CANDIDATE_MANIFEST_SHA256 = (
    "ad3b52d626a46ba92325113a4742aba710167db86f759c77500a76ab280458d1"
)
TRAIN_PSEUDO_MANIFEST_SHA256 = (
    "5aec58ce402da70189c2776453f614e21e5b46fde36b408fc7198c7eeee5dc21"
)
VAL_CANDIDATE_MANIFEST_SHA256 = (
    "3e9396f532c793258919a1d99aa3dcef00523436c853207b8d7123e5dc133090"
)
VAL_PSEUDO_MANIFEST_SHA256 = (
    "286d1fce0bcbd0f96a15b6b386ad27a0edac3500a63c5b87e16f9075d6c6320e"
)
BASELINE_FREEZE_SHA256 = (
    "ec346276d41da7f81d7b4181ee773f5dc962dab70942303d11085804029e3ec3"
)
BASELINE_CHECKPOINT_SHA256 = (
    "58b82642dfa6723e2ec8293687be0096ccfbd26163222aa0b32db01b2d0e1069"
)
BASELINE_MANIFEST_SHA256 = (
    "a810e1fcc4c4422d207eb020a70313caf5d3402bf30c277331247a30555678ee"
)
BASELINE_SOURCE_COMMIT = "fda732941664e67d4b87a8c3cba071b6979b2214"
BASELINE_PROTOCOL_SHA256 = (
    "4aadd1bbd57689147c7db8130bb5c76fab7b79c7e8d92a8bf4f51474fe45b555"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, *, name: str) -> str:
    result = value.strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return result


def _validate_commit(value: str, *, name: str) -> str:
    result = value.strip().lower()
    if len(result) != 40 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase 40-character Git commit")
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _read_split(
    path: Path,
    *,
    expected_frozen_sha256: str,
    expected_counts: Mapping[str, int],
) -> dict[str, list[dict[str, str]]]:
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    expected = _validate_sha256(expected_frozen_sha256, name="split SHA-256")
    if actual == GIT_SPLIT_SHA256:
        if b"\r" in payload:
            raise ValueError("Canonical Git split contains carriage returns")
        if hashlib.sha256(payload.replace(b"\n", b"\r\n")).hexdigest() != expected:
            raise ValueError("Canonical Git split does not reconstruct frozen bytes")
    elif actual != expected:
        raise ValueError("Split manifest SHA-256 mismatch")
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    selected: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "val"):
        current = sorted(
            (
                dict(row)
                for row in rows
                if row.get("split") == split and row.get("eligible") == "1"
            ),
            key=lambda row: row["image_id"],
        )
        if len(current) != expected_counts[split]:
            raise ValueError(f"Frozen {split} cohort mismatch")
        ids = [row["image_id"] for row in current]
        if len(ids) != len(set(ids)) or any(row["tumor"] not in {"0", "1"} for row in current):
            raise ValueError(f"Frozen {split} identities/labels are invalid")
        selected[split] = current
    return selected


def _candidate_index(
    path: Path,
    *,
    expected_sha256: str,
    expected_image_ids: set[str],
) -> dict[str, dict[str, str]]:
    if sha256_file(path) != _validate_sha256(expected_sha256, name="candidate manifest SHA-256"):
        raise ValueError("Candidate manifest SHA-256 mismatch")
    rows = _read_csv(path)
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        image_id = Path(row["image_name"]).name
        if image_id in indexed:
            raise ValueError("Candidate manifest contains duplicate images")
        _validate_sha256(row["diagnostic_sha256"], name="candidate payload SHA-256")
        if int(row["candidate_count"]) <= 0:
            raise ValueError("Candidate manifest contains an empty gallery")
        indexed[image_id] = row
    if set(indexed) != expected_image_ids:
        raise ValueError("Candidate manifest cohort differs from split")
    return indexed


def _family_ids(payload: Mapping[str, object]) -> np.ndarray:
    components = np.asarray(payload["component_ids"])
    modes = np.asarray(payload["prompt_modes"])
    sources = np.asarray(payload["proposal_source_ids"])
    fallback = np.asarray(payload["fallback_flags"]).astype(bool)
    if not (
        components.ndim == modes.ndim == sources.ndim == fallback.ndim == 1
        and components.shape == modes.shape == sources.shape == fallback.shape
    ):
        raise ValueError("Candidate family provenance arrays do not align")
    keys = [
        ("fallback", "fallback", -1)
        if bool(is_fallback)
        else (str(source), str(mode), int(component))
        for component, mode, source, is_fallback in zip(
            components, modes, sources, fallback
        )
    ]
    mapping = {key: index for index, key in enumerate(sorted(set(keys)))}
    return np.asarray([mapping[key] for key in keys], dtype=np.int32)


def _unpack_masks(record: PackedCandidateMasks) -> np.ndarray:
    bits = record.height * record.width
    unpacked = np.unpackbits(np.asarray(record.packed), axis=1, count=bits)
    return unpacked.reshape(record.candidate_count, record.height, record.width).astype(bool)


def _shape_and_geometry(
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count, height, width = masks.shape
    flat = masks.reshape(count, -1).astype(np.float32)
    areas = flat.sum(axis=1)
    if np.any(areas <= 0):
        raise ValueError("Validation cache contains an empty candidate mask")
    shape_rows: list[list[float]] = []
    for mask in masks:
        y, x = np.nonzero(mask)
        box_height = int(y.max() - y.min() + 1)
        box_width = int(x.max() - x.min() + 1)
        area = int(mask.sum())
        box_area = box_height * box_width
        shape_rows.append(
            [
                area / float(height * width),
                box_area / float(height * width),
                area / float(box_area),
                float(np.log((box_width / width) / (box_height / height))),
            ]
        )
    intersections = flat @ flat.T
    unions = areas[:, None] + areas[None, :] - intersections
    iou = intersections / np.maximum(unions, 1.0)
    containment = intersections / np.maximum(
        np.minimum(areas[:, None], areas[None, :]), 1.0
    )
    coordinates = np.indices((height, width), dtype=np.float32)
    centroid_y = np.einsum("nhw,hw->n", masks, coordinates[0]) / areas
    centroid_x = np.einsum("nhw,hw->n", masks, coordinates[1]) / areas
    delta_y = (centroid_y[:, None] - centroid_y[None, :]) / max(1.0, height - 1.0)
    delta_x = (centroid_x[:, None] - centroid_x[None, :]) / max(1.0, width - 1.0)
    distance = np.sqrt(delta_y**2 + delta_x**2)
    return (
        np.asarray(shape_rows, dtype=np.float32),
        iou.astype(np.float32),
        containment.astype(np.float32),
        distance.astype(np.float32),
    )


def _verify_reproduction(
    cache_root: Path,
    baseline_root: Path,
    *,
    expected_images: int,
    tolerance: float,
) -> dict[str, object]:
    baseline_rows = _read_csv(baseline_root / "predictions" / "prediction_manifest.csv")
    reproduced_root = cache_root / "baseline_reproduction" / "predictions"
    reproduced_rows = _read_csv(reproduced_root / "prediction_manifest.csv")
    if len(baseline_rows) != expected_images or len(reproduced_rows) != expected_images:
        raise ValueError("Baseline reproduction cohort mismatch")
    indexed = {row["image_id"]: row for row in reproduced_rows}
    if len(indexed) != expected_images:
        raise ValueError("Baseline reproduction has duplicate images")
    maxima = {"selected": 0.0, "bag": 0.0, "probability": 0.0}
    fields = (
        "group_id",
        "tumor",
        "candidate_payload_sha256",
        "candidate_count",
        "selected_candidate_index",
        "candidate_logit_tta",
        "fallback_count",
        "map_path",
        "map_sha256",
    )
    physical_maps = 0
    for baseline in baseline_rows:
        current = indexed.get(baseline["image_id"])
        if current is None or any(current[field] != baseline[field] for field in fields):
            raise ValueError(f"Baseline reproduction identity mismatch: {baseline['image_id']}")
        maxima["selected"] = max(
            maxima["selected"],
            abs(float(current["selected_candidate_logit"]) - float(baseline["selected_candidate_logit"])),
        )
        maxima["bag"] = max(
            maxima["bag"], abs(float(current["bag_logit"]) - float(baseline["bag_logit"]))
        )
        maxima["probability"] = max(
            maxima["probability"],
            abs(float(current["bag_probability"]) - float(baseline["bag_probability"])),
        )
        baseline_map_path = baseline_root / "predictions" / baseline["map_path"]
        map_path = reproduced_root / current["map_path"]
        if (
            not baseline_map_path.is_file()
            or sha256_file(baseline_map_path) != baseline["map_sha256"]
            or not map_path.is_file()
            or sha256_file(map_path) != current["map_sha256"]
        ):
            raise ValueError(f"Reproduced physical map mismatch: {baseline['image_id']}")
        physical_maps += 1
    if any(value > tolerance for value in maxima.values()):
        raise ValueError("Baseline reproduction numerical tolerance exceeded")
    return {
        "validation_images": expected_images,
        "selected_indices_exact": expected_images,
        "map_hashes_exact": physical_maps,
        "maximum_selected_logit_delta": maxima["selected"],
        "maximum_bag_logit_delta": maxima["bag"],
        "maximum_bag_probability_delta": maxima["probability"],
        "logit_tolerance": tolerance,
        "reproduction_manifest_sha256": sha256_file(
            reproduced_root / "prediction_manifest.csv"
        ),
    }


def audit_cache_output(
    *,
    cache_root: Path,
    expected_cache_freeze_sha256: str,
    expected_wrapper_audit_sha256: str,
    expected_source_commit: str,
    expected_protocol_sha256: str,
    split_manifest: Path,
    expected_split_sha256: str,
    train_candidate_manifest: Path,
    expected_train_candidate_manifest_sha256: str,
    val_candidate_manifest: Path,
    expected_val_candidate_manifest_sha256: str,
    expected_train_pseudo_manifest_sha256: str,
    expected_val_pseudo_manifest_sha256: str,
    baseline_root: Path,
    expected_baseline_freeze_sha256: str,
    expected_baseline_checkpoint_sha256: str,
    expected_baseline_manifest_sha256: str,
    expected_counts: Mapping[str, int],
    expected_model_hashes: Mapping[str, str],
    expected_projection_sha256: str,
    output_path: Path | None = None,
) -> dict[str, object]:
    freeze_path = cache_root / "selector_cache_freeze.json"
    wrapper_audit_path = cache_root / "wrapper_output_audit.json"
    manifest_path = cache_root / "selector_cache_manifest.csv"
    reproduction_path = cache_root / "baseline_reproduction_audit.json"
    run_manifest_path = cache_root / "run_manifest.json"
    if sha256_file(freeze_path) != _validate_sha256(
        expected_cache_freeze_sha256, name="cache freeze SHA-256"
    ):
        raise ValueError("Selector-cache freeze SHA-256 mismatch")
    if sha256_file(wrapper_audit_path) != _validate_sha256(
        expected_wrapper_audit_sha256, name="wrapper audit SHA-256"
    ):
        raise ValueError("Selector-cache wrapper audit SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    wrapper_audit = json.loads(wrapper_audit_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    reproduction = json.loads(reproduction_path.read_text(encoding="utf-8"))
    source_commit = _validate_commit(expected_source_commit, name="source commit")
    protocol_sha256 = _validate_sha256(expected_protocol_sha256, name="protocol SHA-256")
    counts = {"train": int(expected_counts["train"]), "val": int(expected_counts["val"])}
    if (
        freeze.get("source_commit") != source_commit
        or freeze.get("protocol_sha256") != protocol_sha256
        or freeze.get("split_sha256") != expected_split_sha256
        or freeze.get("projection_sha256") != expected_projection_sha256
        or freeze.get("baseline_source_commit") != BASELINE_SOURCE_COMMIT
        or freeze.get("baseline_protocol_sha256") != BASELINE_PROTOCOL_SHA256
        or freeze.get("model_snapshot", {}).get("config.json", {}).get("sha256")
        != expected_model_hashes["config.json"]
        or freeze.get("model_snapshot", {}).get("preprocessor_config.json", {}).get("sha256")
        != expected_model_hashes["preprocessor_config.json"]
        or freeze.get("model_snapshot", {}).get("model.safetensors", {}).get("sha256")
        != expected_model_hashes["model.safetensors"]
        or freeze.get("train_candidate_manifest_sha256")
        != expected_train_candidate_manifest_sha256
        or freeze.get("train_pseudo_manifest_sha256")
        != expected_train_pseudo_manifest_sha256
        or freeze.get("val_candidate_manifest_sha256")
        != expected_val_candidate_manifest_sha256
        or freeze.get("val_pseudo_manifest_sha256") != expected_val_pseudo_manifest_sha256
        or freeze.get("baseline_prediction_freeze_sha256")
        != expected_baseline_freeze_sha256
        or freeze.get("baseline_checkpoint_sha256") != expected_baseline_checkpoint_sha256
        or freeze.get("baseline_prediction_manifest_sha256")
        != expected_baseline_manifest_sha256
        or freeze.get("cohort") != {"train": counts["train"], "validation": counts["val"]}
        or freeze.get("validation_selected_indices_reproduced") != counts["val"]
        or freeze.get("validation_map_hashes_reproduced") != counts["val"]
        or freeze.get("train_masks_discarded") is not True
        or freeze.get("validation_masks_bitpacked") is not True
        or freeze.get("affinity_features_cached") is not True
        or freeze.get("affinity_feature_dim") != 24
        or freeze.get("validation_gt_read") is not False
        or freeze.get("consumer_trained") is not False
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("Selector-cache freeze contract mismatch")
    if (
        sha256_file(manifest_path) != freeze.get("selector_cache_manifest_sha256")
        or sha256_file(reproduction_path) != freeze.get("baseline_reproduction_audit_sha256")
        or sha256_file(baseline_root / "prediction_freeze.json")
        != expected_baseline_freeze_sha256
        or sha256_file(baseline_root / "rad_dino_mask_bag_mil.pt")
        != expected_baseline_checkpoint_sha256
        or sha256_file(baseline_root / "predictions" / "prediction_manifest.csv")
        != expected_baseline_manifest_sha256
    ):
        raise ValueError("Selector-cache/baseline physical root hash mismatch")

    baseline_freeze = json.loads(
        (baseline_root / "prediction_freeze.json").read_text(encoding="utf-8")
    )
    if (
        baseline_freeze.get("source_commit") != BASELINE_SOURCE_COMMIT
        or baseline_freeze.get("protocol_sha256") != BASELINE_PROTOCOL_SHA256
        or baseline_freeze.get("split_sha256") != expected_split_sha256
        or baseline_freeze.get("checkpoint_sha256")
        != expected_baseline_checkpoint_sha256
        or baseline_freeze.get("prediction_manifest_sha256")
        != expected_baseline_manifest_sha256
        or baseline_freeze.get("train_candidate_manifest_sha256")
        != expected_train_candidate_manifest_sha256
        or baseline_freeze.get("train_pseudo_manifest_sha256")
        != expected_train_pseudo_manifest_sha256
        or baseline_freeze.get("val_candidate_manifest_sha256")
        != expected_val_candidate_manifest_sha256
        or baseline_freeze.get("val_pseudo_manifest_sha256")
        != expected_val_pseudo_manifest_sha256
        or baseline_freeze.get("validation_gt_read") is not False
        or baseline_freeze.get("consumer_trained") is not False
        or baseline_freeze.get("test_evaluated") is not False
    ):
        raise ValueError("Frozen baseline provenance contract mismatch")

    split_rows = _read_split(
        split_manifest,
        expected_frozen_sha256=expected_split_sha256,
        expected_counts=counts,
    )
    candidate_rows = {
        "train": _candidate_index(
            train_candidate_manifest,
            expected_sha256=expected_train_candidate_manifest_sha256,
            expected_image_ids={row["image_id"] for row in split_rows["train"]},
        ),
        "val": _candidate_index(
            val_candidate_manifest,
            expected_sha256=expected_val_candidate_manifest_sha256,
            expected_image_ids={row["image_id"] for row in split_rows["val"]},
        ),
    }
    expected: dict[tuple[str, str], dict[str, str]] = {}
    for split in ("train", "val"):
        for row in split_rows[split]:
            candidate = candidate_rows[split][row["image_id"]]
            expected[(split, row["image_id"])] = {
                "group_id": row["group_id"],
                "tumor": row["tumor"],
                "candidate_payload_sha256": candidate["diagnostic_sha256"],
                "source_candidate_count": candidate["candidate_count"],
            }

    manifest_rows = _read_csv(manifest_path)
    indexed = {(row["split"], row["image_id"]): row for row in manifest_rows}
    if len(manifest_rows) != len(indexed) or set(indexed) != set(expected):
        raise ValueError("Selector-cache manifest cohort/identity mismatch")
    physical_records = 0
    physical_bytes = 0
    total_candidates = {"train": 0, "val": 0}
    validation_geometry_records = 0
    for key in sorted(expected):
        split, image_id = key
        row = indexed[key]
        expected_row = expected[key]
        if (
            row["group_id"] != expected_row["group_id"]
            or row["tumor"] != expected_row["tumor"]
            or row["candidate_payload_sha256"] != expected_row["candidate_payload_sha256"]
        ):
            raise ValueError(f"Selector-cache provenance mismatch: {split}/{image_id}")
        cache_path = cache_root / row["cache_path"]
        payload = load_selector_cache_record(
            cache_path,
            expected_sha256=row["cache_sha256"],
            require_packed_masks=split == "val",
        )
        descriptors = np.asarray(payload["descriptors"])
        affinity = np.asarray(payload["affinity_features"])
        indices = np.asarray(payload["candidate_indices"])
        count = descriptors.shape[0]
        if (
            int(row["candidate_count"]) != count
            or int(row["descriptor_dim"]) != 128
            or descriptors.shape[1] != 128
            or int(row["affinity_dim"]) != 24
            or affinity.shape != (count, 24)
            or bool(int(row["packed_masks_included"])) != (split == "val")
            or int(indices[-1]) >= int(expected_row["source_candidate_count"])
            or not np.array_equal(np.asarray(payload["family_ids"]), _family_ids(payload))
        ):
            raise ValueError(f"Selector-cache content/manifest mismatch: {split}/{image_id}")
        if split == "val":
            packed = payload.get("packed_masks")
            if not isinstance(packed, PackedCandidateMasks):
                raise ValueError("Validation cache lacks packed masks")
            masks = _unpack_masks(packed)
            shape, iou, containment, distance = _shape_and_geometry(masks)
            comparisons = (
                (shape, np.asarray(payload["shape_features"])),
                (iou, np.asarray(payload["pairwise_iou"])),
                (containment, np.asarray(payload["pairwise_containment"])),
                (distance, np.asarray(payload["pairwise_distance"])),
            )
            if any(not np.allclose(first, second, rtol=0.0, atol=1.0e-6) for first, second in comparisons):
                raise ValueError(f"Validation packed-mask geometry mismatch: {image_id}")
            validation_geometry_records += 1
        physical_records += 1
        physical_bytes += cache_path.stat().st_size
        total_candidates[split] += count

    independent_reproduction = _verify_reproduction(
        cache_root,
        baseline_root,
        expected_images=counts["val"],
        tolerance=5.0e-6,
    )
    if reproduction != independent_reproduction:
        raise ValueError("Stored baseline reproduction audit differs from independent audit")
    if (
        wrapper_audit.get("scientific_source_commit") != source_commit
        or wrapper_audit.get("protocol_sha256") != protocol_sha256
        or wrapper_audit.get("selector_cache_freeze_sha256")
        != expected_cache_freeze_sha256
        or wrapper_audit.get("selector_cache_manifest_sha256") != sha256_file(manifest_path)
        or wrapper_audit.get("baseline_reproduction_audit_sha256")
        != sha256_file(reproduction_path)
        or wrapper_audit.get("run_manifest_sha256") != sha256_file(run_manifest_path)
        or wrapper_audit.get("physical_cache_records_verified") != sum(counts.values())
        or wrapper_audit.get("cohort") != counts
        or wrapper_audit.get("validation_gt_read") is not False
        or wrapper_audit.get("consumer_trained") is not False
        or wrapper_audit.get("test_evaluated") is not False
    ):
        raise ValueError("Selector-cache wrapper audit contract mismatch")
    t4x2 = wrapper_audit.get("t4x2", {})
    convolution_checksums = t4x2.get("real_convolution_checksums", [])
    if (
        t4x2.get("cuda_device_count") != 2
        or len(t4x2.get("cuda_device_names", [])) != 2
        or not all("T4" in name for name in t4x2["cuda_device_names"])
        or len(convolution_checksums) != 2
        or not np.isfinite(np.asarray(convolution_checksums, dtype=np.float64)).all()
    ):
        raise ValueError("Selector-cache wrapper T4x2 evidence mismatch")
    runtime = run_manifest.get("runtime", {})
    expected_cache_summary = {
        "schema_version": 2,
        "records": sum(counts.values()),
        "train_records": counts["train"],
        "validation_records": counts["val"],
        "manifest_sha256": sha256_file(manifest_path),
    }
    if (
        run_manifest.get("cache_freeze_sha256") != expected_cache_freeze_sha256
        or run_manifest.get("cache") != expected_cache_summary
        or run_manifest.get("baseline_reproduction") != independent_reproduction
        or run_manifest.get("validation_gt_read") is not False
        or run_manifest.get("consumer_trained") is not False
        or run_manifest.get("test_evaluated") is not False
        or runtime.get("cuda_device_count") != 2
        or len(runtime.get("cuda_device_names", [])) != 2
        or not all("T4" in name for name in runtime["cuda_device_names"])
        or runtime.get("encoder_data_parallel") is not True
    ):
        raise ValueError("Selector-cache run manifest/runtime mismatch")

    audit = {
        "audit_id": "independent_mask_bag_selector_cache_output_v1",
        "cache_freeze_sha256": sha256_file(freeze_path),
        "wrapper_output_audit_sha256": sha256_file(wrapper_audit_path),
        "cache_manifest_sha256": sha256_file(manifest_path),
        "baseline_reproduction_audit_sha256": sha256_file(reproduction_path),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "source_commit": source_commit,
        "protocol_sha256": protocol_sha256,
        "physical_cache_records_verified": physical_records,
        "physical_cache_bytes_verified": physical_bytes,
        "candidate_counts": total_candidates,
        "validation_packed_mask_geometry_records_verified": validation_geometry_records,
        "baseline_reproduction": independent_reproduction,
        "cohort": counts,
        "training_labels": "image_level_only",
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
    }
    if output_path is not None:
        if output_path.exists():
            raise FileExistsError(f"Independent audit output already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--expected-cache-freeze-sha256", required=True)
    parser.add_argument("--expected-wrapper-audit-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--train-candidate-manifest", type=Path, required=True)
    parser.add_argument("--val-candidate-manifest", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_cache_output(
        cache_root=args.cache_root,
        expected_cache_freeze_sha256=args.expected_cache_freeze_sha256,
        expected_wrapper_audit_sha256=args.expected_wrapper_audit_sha256,
        expected_source_commit=args.expected_source_commit,
        expected_protocol_sha256=args.expected_protocol_sha256,
        split_manifest=args.split_manifest,
        expected_split_sha256=FROZEN_SPLIT_SHA256,
        train_candidate_manifest=args.train_candidate_manifest,
        expected_train_candidate_manifest_sha256=TRAIN_CANDIDATE_MANIFEST_SHA256,
        val_candidate_manifest=args.val_candidate_manifest,
        expected_val_candidate_manifest_sha256=VAL_CANDIDATE_MANIFEST_SHA256,
        expected_train_pseudo_manifest_sha256=TRAIN_PSEUDO_MANIFEST_SHA256,
        expected_val_pseudo_manifest_sha256=VAL_PSEUDO_MANIFEST_SHA256,
        baseline_root=args.baseline_root,
        expected_baseline_freeze_sha256=BASELINE_FREEZE_SHA256,
        expected_baseline_checkpoint_sha256=BASELINE_CHECKPOINT_SHA256,
        expected_baseline_manifest_sha256=BASELINE_MANIFEST_SHA256,
        expected_counts={"train": 2981, "val": 371},
        expected_model_hashes=MODEL_HASHES,
        expected_projection_sha256=PROJECTION_SHA256,
        output_path=args.output,
    )
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
