from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_COMMIT = "de02acb59900cc64f7bdf649d20286d6219af82c"
EXPECTED_WRAPPER_SHA256 = (
    "e70928fc44485b02cd74874ae9398cfcb7f846e28baf82be9d2231f0403cc0c1"
)
EXPECTED_SPLIT_SHA256 = (
    "43662d5d7969ae2a5bc61c6a0de3e0c392debef19c98d809f7d9bdfd0abb2fa8"
)
EXPECTED_MODEL_ID = (
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)
EXPECTED_OPEN_CLIP_VERSION = "2.32.0"
EXPECTED_TRANSFORMERS_VERSION = "4.35.2"
EXPECTED_PROMPTS = {
    "tumor": [
        "A bone radiograph showing a bone tumor.",
        "An x-ray image with a bone neoplasm.",
        "An x-ray image showing an abnormal bone lesion.",
    ],
    "normal": [
        "A normal bone radiograph without a tumor.",
        "An x-ray image of healthy bone.",
        "An x-ray image without a bone lesion.",
    ],
}
EXPECTED_SCORE_PER_CLASS = 32
EXPECTED_SALIENCY_PER_CLASS = 4
EXPECTED_REPEAT_ATOL = 1e-5
EXPECTED_DYNAMIC_RANGE_MIN = 1e-6
EXPECTED_SALIENCY_METHOD = (
    "channelwise mean absolute gradient-times-activation at frozen visual "
    "transformer block 11 norm2, tumor-prompt mean minus normal-prompt mean"
)
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_train_rows(split_manifest: Path) -> list[dict[str, str]]:
    if sha256_file(split_manifest) != EXPECTED_SPLIT_SHA256:
        raise ValueError("Frozen split manifest SHA-256 mismatch")
    with split_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("split") == "train" and row.get("eligible") == "1"
        ]
    if len(rows) != 2981:
        raise ValueError(f"Unexpected eligible train population: {len(rows)}")
    if sum(int(row["tumor"]) for row in rows) != 1488:
        raise ValueError("Unexpected eligible train tumor population")
    return rows


def expected_sample_ids(
    train_rows: list[dict[str, str]],
) -> tuple[list[str], list[str]]:
    by_class: dict[int, list[dict[str, str]]] = {}
    for label in (0, 1):
        by_class[label] = sorted(
            (row for row in train_rows if int(row["tumor"]) == label),
            key=lambda row: hashlib.sha256(row["image_id"].encode()).hexdigest(),
        )
        if len(by_class[label]) < EXPECTED_SCORE_PER_CLASS:
            raise ValueError(f"Insufficient rows for image label {label}")
    score = [
        row["image_id"]
        for label in (0, 1)
        for row in by_class[label][:EXPECTED_SCORE_PER_CLASS]
    ]
    saliency = [
        row["image_id"]
        for label in (0, 1)
        for row in by_class[label][:EXPECTED_SALIENCY_PER_CLASS]
    ]
    return score, saliency


def require_finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite")
    return number


def close(actual: float, expected: float, *, atol: float = 1e-12) -> bool:
    return abs(actual - expected) <= atol


def validate_payloads(
    predeclared: dict[str, Any],
    result: dict[str, Any],
    train_rows: list[dict[str, str]],
) -> dict[str, Any]:
    prompt_sha = sha256_json(EXPECTED_PROMPTS)
    if predeclared.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        raise ValueError("Predeclared source commit mismatch")
    if result.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        raise ValueError("Result source commit mismatch")
    if predeclared.get("model_id") != EXPECTED_MODEL_ID:
        raise ValueError("Predeclared BiomedCLIP model mismatch")
    if result.get("model_id") != EXPECTED_MODEL_ID:
        raise ValueError("Result BiomedCLIP model mismatch")
    if predeclared.get("open_clip_torch") != EXPECTED_OPEN_CLIP_VERSION:
        raise ValueError("Predeclared open_clip version mismatch")
    if predeclared.get("transformers") != EXPECTED_TRANSFORMERS_VERSION:
        raise ValueError("Predeclared transformers version mismatch")
    if result.get("environment", {}).get("open_clip") != EXPECTED_OPEN_CLIP_VERSION:
        raise ValueError("Runtime open_clip version mismatch")
    if result.get("environment", {}).get("transformers") != EXPECTED_TRANSFORMERS_VERSION:
        raise ValueError("Runtime transformers version mismatch")
    if predeclared.get("prompts") != EXPECTED_PROMPTS:
        raise ValueError("Frozen prompt text mismatch")
    if predeclared.get("saliency") != EXPECTED_SALIENCY_METHOD:
        raise ValueError("Frozen saliency reduction mismatch")
    if predeclared.get("prompt_sha256") != prompt_sha:
        raise ValueError("Predeclared prompt SHA-256 mismatch")
    if result.get("prompt_sha256") != prompt_sha:
        raise ValueError("Result prompt SHA-256 mismatch")

    for payload_name, payload in (("predeclared", predeclared), ("result", result)):
        if payload.get("validation_masks_read") is not False:
            raise ValueError(f"{payload_name} did not lock validation masks")
        if payload.get("test_evaluated") is not False:
            raise ValueError(f"{payload_name} did not lock test")
    if result.get("status") != "PASS":
        raise ValueError("Cloud implementation gate did not pass")

    population = result.get("population", {})
    expected_population = {
        "score_images": 2 * EXPECTED_SCORE_PER_CLASS,
        "saliency_images": 2 * EXPECTED_SALIENCY_PER_CLASS,
        "source_split": "train",
        "validation_images": 0,
        "test_images": 0,
    }
    if population != expected_population:
        raise ValueError(f"Smoke population mismatch: {population}")

    weights = result.get("model_weight_files")
    if not isinstance(weights, list) or not weights:
        raise ValueError("Physical BiomedCLIP weight hashes are absent")
    for index, weight in enumerate(weights):
        if not isinstance(weight, dict):
            raise ValueError(f"Invalid weight record {index}")
        if int(weight.get("bytes", 0)) <= 1_000_000:
            raise ValueError(f"Weight record {index} is not a physical checkpoint")
        if not HEX_SHA256.fullmatch(str(weight.get("sha256", ""))):
            raise ValueError(f"Weight record {index} SHA-256 is invalid")

    expected_score_ids, expected_saliency_ids = expected_sample_ids(train_rows)
    label_by_id = {row["image_id"]: int(row["tumor"]) for row in train_rows}
    score_diagnostic = result.get("score_diagnostic", {})
    score_rows = score_diagnostic.get("rows")
    if not isinstance(score_rows, list):
        raise ValueError("Score rows are absent")
    actual_score_ids = [row.get("image_id") for row in score_rows]
    if actual_score_ids != expected_score_ids:
        raise ValueError("Score sample identities/order differ from the frozen rule")
    score_values = []
    score_labels = []
    for index, row in enumerate(score_rows):
        image_id = str(row["image_id"])
        label = int(row.get("image_label", -1))
        if label != label_by_id[image_id]:
            raise ValueError(f"Score image label mismatch at row {index}")
        score_labels.append(label)
        score_values.append(require_finite(row.get("score"), f"score row {index}"))
    means = {
        label: sum(score for score, row_label in zip(score_values, score_labels) if row_label == label)
        / sum(row_label == label for row_label in score_labels)
        for label in (0, 1)
    }
    overall_mean = sum(score_values) / len(score_values)
    overall_std = math.sqrt(
        sum((score - overall_mean) ** 2 for score in score_values) / len(score_values)
    )
    recomputed = {
        "tumor_mean": means[1],
        "normal_mean": means[0],
        "tumor_minus_normal": means[1] - means[0],
        "overall_std": overall_std,
    }
    for key, expected in recomputed.items():
        actual = require_finite(score_diagnostic.get(key), f"score diagnostic {key}")
        if not close(actual, expected):
            raise ValueError(f"Score diagnostic {key} was not reproduced")

    saliency_diagnostic = result.get("saliency_diagnostic", {})
    if saliency_diagnostic.get("target_layer") != "model.visual.trunk.blocks[11].norm2":
        raise ValueError("Saliency target layer mismatch")
    repeat_delta = require_finite(
        saliency_diagnostic.get("repeat_max_abs_delta"), "repeat delta"
    )
    if repeat_delta > EXPECTED_REPEAT_ATOL:
        raise ValueError("Saliency repeatability gate failed")
    saliency_rows = saliency_diagnostic.get("rows")
    if not isinstance(saliency_rows, list):
        raise ValueError("Saliency rows are absent")
    if [row.get("image_id") for row in saliency_rows] != expected_saliency_ids:
        raise ValueError("Saliency sample identities/order differ from the frozen rule")
    dynamic_ranges = []
    for index, row in enumerate(saliency_rows):
        image_id = str(row["image_id"])
        if int(row.get("image_label", -1)) != label_by_id[image_id]:
            raise ValueError(f"Saliency image label mismatch at row {index}")
        if row.get("finite") is not True:
            raise ValueError(f"Saliency finite flag failed at row {index}")
        minimum = require_finite(row.get("minimum"), f"saliency minimum {index}")
        maximum = require_finite(row.get("maximum"), f"saliency maximum {index}")
        mean = require_finite(row.get("mean"), f"saliency mean {index}")
        dynamic_range = require_finite(
            row.get("dynamic_range"), f"saliency dynamic range {index}"
        )
        require_finite(row.get("contrast_score"), f"saliency score {index}")
        if minimum < 0 or maximum < minimum or not minimum <= mean <= maximum:
            raise ValueError(f"Saliency summary bounds invalid at row {index}")
        if not close(dynamic_range, maximum - minimum, atol=1e-10):
            raise ValueError(f"Saliency dynamic range mismatch at row {index}")
        if dynamic_range <= EXPECTED_DYNAMIC_RANGE_MIN:
            raise ValueError(f"Saliency is constant at row {index}")
        shape = row.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or int(shape[0]) <= 0
            or int(shape[0]) != int(shape[1])
        ):
            raise ValueError(f"Saliency shape invalid at row {index}")
        dynamic_ranges.append(dynamic_range)

    return {
        "status": "PASS",
        "decision": "IMPLEMENTATION_GATE_PASS",
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "model_id": EXPECTED_MODEL_ID,
        "model_weight_files": weights,
        "prompt_sha256": prompt_sha,
        "population": expected_population,
        "score_diagnostic_recomputed": recomputed,
        "saliency_dynamic_range_min": min(dynamic_ranges),
        "repeat_max_abs_delta": repeat_delta,
        "validation_masks_read": False,
        "test_evaluated": False,
    }


def audit(
    root: Path,
    split_manifest: Path,
    wrapper: Path,
) -> dict[str, Any]:
    wrapper_hash = sha256_file(wrapper)
    if wrapper_hash != EXPECTED_WRAPPER_SHA256:
        raise ValueError("BiomedCLIP smoke wrapper SHA-256 mismatch")
    predeclared_path = root / "predeclared_method.json"
    result_path = root / "smoke_result.json"
    if not predeclared_path.is_file() or not result_path.is_file():
        raise FileNotFoundError("BiomedCLIP smoke evidence is incomplete")
    predeclared = json.loads(predeclared_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    audited = validate_payloads(predeclared, result, load_train_rows(split_manifest))
    audited["wrapper_sha256"] = wrapper_hash
    audited["downloaded_artifact_sha256"] = {
        "predeclared_method": sha256_file(predeclared_path),
        "smoke_result": sha256_file(result_path),
    }
    return audited


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args.root, args.split_manifest, args.wrapper)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
