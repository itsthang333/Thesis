from __future__ import annotations

"""Freeze the four X4 inference-label arms before validation GT is opened."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from frozen_io import load_split_rows_without_annotations, sha256_file
from x4_contract import (
    CANONICAL_SPLIT_SHA256,
    GATE_ARMS,
    STUDENT_SEEDS,
    load_x4_protocol,
)


GATE_THRESHOLD = 0.5


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def gate_uses_direct_mask(
    arm: str,
    *,
    known_tumor: int,
    binary_probability: float,
    ten_class_probability: float,
) -> bool | None:
    """Return the direct-mask gate decision; ``None`` denotes the student arm."""

    if arm == "known_binary_label":
        return bool(known_tumor)
    if arm == "binary_predicted_gate":
        return binary_probability >= GATE_THRESHOLD
    if arm == "ten_class_predicted_gate":
        return ten_class_probability >= GATE_THRESHOLD
    if arm == "label_free_rich_gallery_student":
        return None
    raise ValueError(f"unknown X4 gate arm: {arm}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=STUDENT_SEEDS, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--expected-selection-freeze-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--binary-predictions", type=Path, required=True)
    parser.add_argument("--expected-binary-predictions-sha256", required=True)
    parser.add_argument("--ten-class-predictions", type=Path, required=True)
    parser.add_argument("--expected-ten-class-predictions-sha256", required=True)
    parser.add_argument("--student-prediction-root", type=Path, required=True)
    parser.add_argument("--expected-student-freeze-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _indexed_classifier_predictions(
    path: Path,
    expected_sha256: str,
    split_by_id: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"classifier predictions SHA-256 mismatch: {path}")
    rows = read_csv(path)
    indexed = {row["image_id"]: row for row in rows}
    if len(rows) != 371 or len(indexed) != 371 or set(indexed) != set(split_by_id):
        raise ValueError("classifier prediction cohort differs from canonical validation")
    for image_id, row in indexed.items():
        if int(row["tumor"]) != int(split_by_id[image_id]["tumor"]):
            raise ValueError(f"classifier label provenance differs: {image_id}")
        probability = float(row["tumor_probability"])
        if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise ValueError(f"invalid classifier tumor probability: {image_id}")
    return indexed


def _verify_student_bundle(
    root: Path,
    expected_freeze_sha256: str,
    *,
    seed: int,
    protocol_sha: str,
    expected_ids: set[str],
) -> dict[str, dict[str, str]]:
    freeze_path = root / "prediction_freeze.json"
    manifest_path = root / "prediction_manifest.csv"
    if sha256_file(freeze_path) != expected_freeze_sha256:
        raise ValueError("Rich-Gallery student freeze SHA-256 mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("schema_version") != 1
        or freeze.get("stage") != "x4_student_prediction_freeze_v1"
        or freeze.get("arm") != "rich_gallery"
        or int(freeze.get("seed", -1)) != seed
        or freeze.get("split_sha256") != CANONICAL_SPLIT_SHA256
        or freeze.get("x4_protocol_sha256") != protocol_sha
        or int(freeze.get("images", -1)) != 371
        or freeze.get("prediction_manifest_sha256") != sha256_file(manifest_path)
        or freeze.get("predictions_frozen_before_spatial_ground_truth") is not True
        or freeze.get("validation_annotations_read") != 0
        or freeze.get("test_images_read") != 0
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("Rich-Gallery student bundle violates X4 Stage A")
    rows = read_csv(manifest_path)
    indexed = {row["image_id"]: row for row in rows}
    if len(rows) != 371 or len(indexed) != 371 or set(indexed) != expected_ids:
        raise ValueError("Rich-Gallery student prediction cohort differs")
    for image_id, row in indexed.items():
        mask_path = root / row["mask_path"]
        if sha256_file(mask_path) != row["mask_sha256"]:
            raise ValueError(f"Rich-Gallery student mask changed: {image_id}")
    return indexed


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if sha256_file(args.split_manifest) != CANONICAL_SPLIT_SHA256:
        raise ValueError("X4 canonical split SHA-256 mismatch")
    _, protocol_sha = load_x4_protocol(Path(__file__).resolve().parents[1])
    split_rows = load_split_rows_without_annotations(
        args.split_manifest,
        expected_sha256=CANONICAL_SPLIT_SHA256,
        split="val",
        allow_test=False,
    )
    split_by_id = {row["image_id"]: row for row in split_rows}
    if len(split_rows) != 371 or sum(int(row["tumor"]) for row in split_rows) != 184:
        raise ValueError("X4 gate freeze requires canonical 371/184 validation")

    selection_freeze_path = args.selection_root / "prediction_freeze.json"
    selection_manifest_path = args.selection_root / "selection_manifest.csv"
    if sha256_file(selection_freeze_path) != args.expected_selection_freeze_sha256:
        raise ValueError("direct Rich-Gallery selection freeze SHA-256 mismatch")
    selection_freeze = json.loads(selection_freeze_path.read_text(encoding="utf-8"))
    if (
        selection_freeze.get("stage") != "final_rich_gallery_choice_freeze_v1"
        or selection_freeze.get("cohort_split") != "val"
        or selection_freeze.get("split_sha256") != CANONICAL_SPLIT_SHA256
        or int(selection_freeze.get("images", -1)) != 371
        or selection_freeze.get("candidate_choices_frozen_before_spatial_gt") is not True
        or selection_freeze.get("spatial_ground_truth_used") is not False
        or selection_freeze.get("validation_gt_read") is not False
        or selection_freeze.get("test_images_read") != 0
        or selection_freeze.get("test_evaluated") is not False
        or selection_freeze.get("selection_manifest_sha256")
        != sha256_file(selection_manifest_path)
    ):
        raise ValueError("direct Rich-Gallery selection violates the annotation boundary")
    selection_rows = read_csv(selection_manifest_path)
    selection_by_id = {row["image_id"]: row for row in selection_rows}
    if len(selection_rows) != 371 or set(selection_by_id) != set(split_by_id):
        raise ValueError("direct Rich-Gallery selection cohort differs")

    binary = _indexed_classifier_predictions(
        args.binary_predictions,
        args.expected_binary_predictions_sha256,
        split_by_id,
    )
    ten_class = _indexed_classifier_predictions(
        args.ten_class_predictions,
        args.expected_ten_class_predictions_sha256,
        split_by_id,
    )
    student = _verify_student_bundle(
        args.student_prediction_root,
        args.expected_student_freeze_sha256,
        seed=args.seed,
        protocol_sha=protocol_sha,
        expected_ids=set(split_by_id),
    )

    args.output_dir.mkdir(parents=True)
    for arm in GATE_ARMS:
        (args.output_dir / "masks" / arm).mkdir(parents=True)
    manifest_rows: list[dict[str, object]] = []
    for split_row in split_rows:
        image_id = split_row["image_id"]
        selection = selection_by_id[image_id]
        candidate_path = (
            args.candidate_root / "candidate_diagnostics" / f"{Path(image_id).stem}.npz"
        )
        if sha256_file(candidate_path) != selection["candidate_payload_sha256"]:
            raise ValueError(f"candidate payload changed: {image_id}")
        with np.load(candidate_path, allow_pickle=False) as payload:
            candidate_masks = payload["sam_masks"].astype(bool)
        selected_index = int(selection["selected_candidate_index"])
        if selected_index < 0 or selected_index >= len(candidate_masks):
            raise ValueError(f"selected candidate index is invalid: {image_id}")
        direct_mask = candidate_masks[selected_index]
        binary_probability = float(binary[image_id]["tumor_probability"])
        ten_probability = float(ten_class[image_id]["tumor_probability"])

        with Image.open(args.student_prediction_root / student[image_id]["mask_path"]) as handle:
            student_mask = np.asarray(handle.convert("L")) > 0
        for arm in GATE_ARMS:
            decision = gate_uses_direct_mask(
                arm,
                known_tumor=int(split_row["tumor"]),
                binary_probability=binary_probability,
                ten_class_probability=ten_probability,
            )
            if decision is None:
                mask = student_mask
                probability = ""
                gate_positive = ""
                source = "rich_gallery_student"
            else:
                mask = direct_mask if decision else np.zeros_like(direct_mask)
                probability = (
                    float(split_row["tumor"])
                    if arm == "known_binary_label"
                    else binary_probability
                    if arm == "binary_predicted_gate"
                    else ten_probability
                )
                gate_positive = int(decision)
                source = selection["selected_source"] if decision else "empty_gate"
            relative = Path("masks") / arm / f"{Path(image_id).stem}.png"
            mask_path = args.output_dir / relative
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path, optimize=True)
            manifest_rows.append(
                {
                    "image_id": image_id,
                    "group_id": split_row["group_id"],
                    "tumor": split_row["tumor"],
                    "arm": arm,
                    "seed": args.seed,
                    "gate_threshold": "" if decision is None else GATE_THRESHOLD,
                    "gate_probability": probability,
                    "gate_positive": gate_positive,
                    "source": source,
                    "mask_path": relative.as_posix(),
                    "mask_height": int(mask.shape[0]),
                    "mask_width": int(mask.shape[1]),
                    "positive_pixels": int(mask.sum()),
                    "mask_sha256": sha256_file(mask_path),
                }
            )
    if len(manifest_rows) != 371 * len(GATE_ARMS):
        raise RuntimeError("X4 gate prediction cohort is incomplete")
    manifest_path = args.output_dir / "prediction_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    freeze = {
        "schema_version": 1,
        "stage": "x4_gate_prediction_freeze_v1",
        "seed": args.seed,
        "arms": list(GATE_ARMS),
        "split": "val",
        "split_sha256": CANONICAL_SPLIT_SHA256,
        "x4_protocol_sha256": protocol_sha,
        "selection_freeze_sha256": args.expected_selection_freeze_sha256,
        "binary_predictions_sha256": args.expected_binary_predictions_sha256,
        "ten_class_predictions_sha256": args.expected_ten_class_predictions_sha256,
        "student_freeze_sha256": args.expected_student_freeze_sha256,
        "classifier_gate_threshold": GATE_THRESHOLD,
        "images_per_arm": 371,
        "tumor_images": 184,
        "normal_images": 187,
        "prediction_manifest_sha256": sha256_file(manifest_path),
        "predictions_frozen_before_spatial_ground_truth": True,
        "spatial_ground_truth_used": False,
        "validation_annotations_read": 0,
        "test_images_read": 0,
        "test_evaluated": False,
    }
    freeze_path = args.output_dir / "prediction_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**freeze, "prediction_freeze_sha256": sha256_file(freeze_path)}, indent=2))


if __name__ == "__main__":
    main()
