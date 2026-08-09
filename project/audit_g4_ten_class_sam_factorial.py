from __future__ import annotations

"""Independent output audit for one ten-class x SAM factorial seed."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from run_g4_e1_downstream import E1_SHA
from run_g4_e3_sam_backbone import G1_SHA, SAM_SHA, SPLIT_SHA
from run_g4_ten_class_sam_factorial import PROTOCOL_SHA, SUPPORTED_SAM


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require_no_test(payload: dict[str, object], name: str) -> None:
    if payload.get("test_evaluated") is not False or int(payload.get("test_images_read", -1)) != 0:
        raise ValueError(f"{name} did not prove zero test access")


def audit(root: Path, seed: int, model_type: str) -> dict[str, object]:
    if seed not in E1_SHA["ten_class"] or model_type not in SUPPORTED_SAM:
        raise ValueError("unsupported factorial arm")
    paths = {
        "result": root / "summary.json",
        "anchor": root / "anchor" / "candidate_supply_manifest.json",
        "addition": root / "addition" / "candidate_supply_manifest.json",
        "gallery": root / "gallery" / "gallery_merge_contract.json",
        "gallery_manifest": root / "gallery" / "candidate_diagnostics_manifest.csv",
        "scores": root / "scores" / "diagnostic_freeze.json",
        "choices": root / "choices" / "prediction_freeze.json",
        "selection": root / "choices" / "selection_manifest.csv",
        "evaluation": root / "evaluation" / "summary.json",
        "per_image": root / "evaluation" / "per_image.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    payload = {name: read_json(path) for name, path in paths.items() if path.suffix == ".json"}
    for name in ("result", "anchor", "addition", "gallery", "scores", "choices", "evaluation"):
        require_no_test(payload[name], name)

    result = payload["result"]
    anchor = payload["anchor"]
    addition = payload["addition"]
    gallery = payload["gallery"]
    scores = payload["scores"]
    choices = payload["choices"]
    evaluation = payload["evaluation"]
    selections = read_csv(paths["selection"])
    per_image = read_csv(paths["per_image"])
    gallery_rows = read_csv(paths["gallery_manifest"])
    if (
        result.get("study") != "G4 ten-class x SAM factorial"
        or int(result.get("seed", -1)) != seed
        or result.get("classifier_checkpoint_sha256") != E1_SHA["ten_class"][seed]
        or result.get("sam_model_type") != model_type
        or result.get("sam_checkpoint_sha256") != SAM_SHA[model_type]
        or result.get("protocol_sha256") != PROTOCOL_SHA
        or result.get("split_sha256") != SPLIT_SHA
        or result.get("choices_frozen_before_spatial_gt") is not True
        or int(result.get("spatial_annotations_opened", -1)) != 184
        or result.get("evaluation_summary_sha256") != sha256(paths["evaluation"])
    ):
        raise ValueError("root factorial contract differs")
    if (
        anchor.get("classifier_checkpoint_sha256") != E1_SHA["ten_class"][seed]
        or anchor.get("sam_checkpoint_sha256") != SAM_SHA[model_type]
        or anchor.get("protocol_sha256") != PROTOCOL_SHA
        or anchor.get("split_sha256") != SPLIT_SHA
        or addition.get("sam_checkpoint_sha256") != SAM_SHA[model_type]
        or addition.get("protocol_sha256") != PROTOCOL_SHA
        or gallery.get("protocol_sha256") != PROTOCOL_SHA
        or gallery.get("split_sha256") != SPLIT_SHA
        or scores.get("baseline_checkpoint_sha256") != G1_SHA
        or scores.get("protocol_sha256") != PROTOCOL_SHA
        or choices.get("candidate_choices_frozen_before_spatial_gt") is not True
        or choices.get("spatial_ground_truth_used") is not False
        or evaluation.get("candidate_choices_frozen_before_spatial_gt") is not True
        or evaluation.get("split") != "val"
        or evaluation.get("split_sha256") != SPLIT_SHA
    ):
        raise ValueError("factorial stage provenance differs")
    summary = evaluation.get("summary", {})
    if (
        len(gallery_rows) != 371
        or len(selections) != 371
        or len(per_image) != 184
        or len({row["image_id"] for row in selections}) != 371
        or len({row["image_id"] for row in per_image}) != 184
        or int(summary.get("overall", {}).get("n", -1)) != 184
        or int(summary.get("small", {}).get("n", -1)) != 94
        or int(summary.get("medium", {}).get("n", -1)) != 72
        or int(summary.get("large", {}).get("n", -1)) != 18
        or result.get("summary") != summary
    ):
        raise ValueError("factorial cohort or metrics differ")
    return {
        "schema_version": 1,
        "study": "independent G4 ten-class x SAM factorial audit",
        "pass": True,
        "seed": seed,
        "sam_model_type": model_type,
        "classifier_checkpoint_sha256": E1_SHA["ten_class"][seed],
        "sam_checkpoint_sha256": SAM_SHA[model_type],
        "protocol_sha256": PROTOCOL_SHA,
        "split_sha256": SPLIT_SHA,
        "summary": summary,
        "selection_manifest_sha256": sha256(paths["selection"]),
        "per_image_sha256": sha256(paths["per_image"]),
        "images": 371,
        "tumor_images": 184,
        "normal_images": 187,
        "test_images_read": 0,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sam-model-type", choices=SUPPORTED_SAM, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root.resolve(), args.seed, args.sam_model_type)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": True, "seed": args.seed, "overall": report["summary"]["overall"]}, indent=2))


if __name__ == "__main__":
    main()
