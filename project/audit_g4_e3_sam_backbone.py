from __future__ import annotations

"""Independent provenance/output audit for one completed G4 E3 SAM arm."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


SPLIT_SHA = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
PROTOCOL_SHA = "c65e6771cc6e68fe51de39c19374cffab35180259e8eed40eead7eed4ff6fb74"
SOURCE_COMMIT = "14d7eb1c91132157df8ec5e6d6d1a7056031e7f8"
SAM_SHA = {
    "vit_b": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
    "vit_l": "3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622",
    "vit_h": "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
}


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


def _assert_no_test(
    payload: dict[str, object], *, name: str, require_images_read: bool = True
) -> None:
    if payload.get("test_evaluated") is not False:
        raise ValueError(f"{name} accessed/evaluated test")
    if require_images_read and payload.get("test_images_read") != 0:
        raise ValueError(f"{name} did not prove zero test images read")


def audit(root: Path, model_type: str) -> dict[str, object]:
    if model_type not in SAM_SHA:
        raise ValueError(f"unsupported SAM model type: {model_type}")
    root = root.resolve()
    result_path = root / "summary.json"
    evaluation_path = root / "evaluation" / "summary.json"
    evaluation_audit_path = root / "evaluation" / "evaluation_audit.json"
    gallery_summary_path = root / "gallery" / "candidate_diagnostics_summary.json"
    gallery_contract_path = root / "gallery" / "gallery_merge_contract.json"
    gallery_manifest_path = root / "gallery" / "candidate_diagnostics_manifest.csv"
    choice_freeze_path = root / "choices" / "prediction_freeze.json"
    selection_path = root / "choices" / "selection_manifest.csv"
    required = (
        result_path,
        evaluation_path,
        evaluation_audit_path,
        gallery_summary_path,
        gallery_contract_path,
        gallery_manifest_path,
        choice_freeze_path,
        selection_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"E3 output is incomplete: {missing}")

    result = read_json(result_path)
    evaluation = read_json(evaluation_path)
    evaluation_audit = read_json(evaluation_audit_path)
    gallery_summary = read_json(gallery_summary_path)
    gallery_contract = read_json(gallery_contract_path)
    choice_freeze = read_json(choice_freeze_path)
    anchor = read_json(root / "anchor" / "candidate_supply_manifest.json")
    addition = read_json(root / "addition" / "candidate_supply_manifest.json")
    for name, payload in (
        ("result", result),
        ("evaluation", evaluation),
        ("anchor", anchor),
        ("addition", addition),
        ("choices", choice_freeze),
    ):
        _assert_no_test(payload, name=name)
    # This redundant hash/reproduction receipt predates the explicit
    # ``test_images_read`` field.  The evaluator summary above is the
    # authoritative fail-closed counter, while this receipt must still state
    # explicitly that test evaluation was disabled.
    _assert_no_test(
        evaluation_audit, name="evaluation audit", require_images_read=False
    )

    if (
        result.get("study") != "G4 E3 matched end-to-end SAM-v1 backbone ablation"
        or result.get("sam_model_type") != model_type
        or result.get("sam_checkpoint_sha256") != SAM_SHA[model_type]
        or result.get("protocol_sha256") != PROTOCOL_SHA
        or result.get("source_commit") != SOURCE_COMMIT
        or result.get("split_sha256") != SPLIT_SHA
        or result.get("choices_frozen_before_spatial_gt") is not True
        or int(result.get("spatial_annotations_opened", -1)) != 184
        or result.get("evaluation_summary_sha256") != sha256(evaluation_path)
    ):
        raise ValueError("E3 root result/provenance contract differs")
    if (
        gallery_summary.get("complete") is not True
        or gallery_summary.get("prediction_first") is not True
        or gallery_summary.get("ground_truth_loaded_during_generation") is not False
        or gallery_summary.get("split") != "val"
        or int(gallery_summary.get("manifest_rows", -1)) != 371
        or gallery_summary.get("manifest_sha256") != sha256(gallery_manifest_path)
        or gallery_contract.get("split_sha256") != SPLIT_SHA
        or gallery_contract.get("protocol_sha256") != PROTOCOL_SHA
        or gallery_contract.get("output_manifest_sha256") != gallery_summary.get("manifest_sha256")
        or gallery_contract.get("validation_gt_read") is not False
        or gallery_contract.get("test_evaluated") is not False
    ):
        raise ValueError("E3 merged gallery contract differs")

    gallery_rows = read_csv(gallery_manifest_path)
    selection_rows = read_csv(selection_path)
    gallery_ids = [str(row.get("image_name", "")) for row in gallery_rows]
    selection_ids = [str(row.get("image_id", "")) for row in selection_rows]
    if (
        len(gallery_rows) != 371
        or len(selection_rows) != 371
        or len(set(gallery_ids)) != 371
        or len(set(selection_ids)) != 371
        or {Path(value).stem.casefold() for value in gallery_ids}
        != {Path(value).stem.casefold() for value in selection_ids}
    ):
        raise ValueError("E3 gallery/selection cohort differs")
    if (
        choice_freeze.get("candidate_choices_frozen_before_spatial_gt") is not True
        or choice_freeze.get("candidate_choices_frozen_before_validation_gt") is not True
        or choice_freeze.get("spatial_ground_truth_used") is not False
        or choice_freeze.get("cohort_split") != "val"
        or choice_freeze.get("split_sha256") != SPLIT_SHA
        or int(choice_freeze.get("images", -1)) != 371
        or int(choice_freeze.get("tumor_images", -1)) != 184
        or int(choice_freeze.get("normal_images", -1)) != 187
        or choice_freeze.get("candidate_manifest_sha256") != gallery_summary.get("manifest_sha256")
        or choice_freeze.get("selection_manifest_sha256") != sha256(selection_path)
    ):
        raise ValueError("E3 choice freeze differs")

    summary = evaluation.get("summary", {})
    if (
        evaluation.get("split") != "val"
        or evaluation.get("split_sha256") != SPLIT_SHA
        or evaluation.get("validation_ablation") is not True
        or evaluation.get("candidate_choices_frozen_before_spatial_gt") is not True
        or int(evaluation.get("spatial_annotations_opened", -1)) != 184
        or not isinstance(summary, dict)
        or int(summary.get("overall", {}).get("n", -1)) != 184
        or int(summary.get("small", {}).get("n", -1)) != 94
        or int(summary.get("medium", {}).get("n", -1)) != 72
        or int(summary.get("large", {}).get("n", -1)) != 18
        or result.get("summary") != summary
        or evaluation_audit.get("pass") is not True
        # E3 intentionally changes the SAM backbone, so the evaluator runs in
        # validation-ablation mode and must not claim locked-baseline
        # reproduction.  Exact B reproduction is checked numerically in the
        # paired E3 comparison, not by mislabelling every arm as the baseline.
        or evaluation_audit.get("overall_dice_reproduced") is not False
        or evaluation_audit.get("summary_sha256") != sha256(evaluation_path)
    ):
        raise ValueError("E3 spatial evaluation contract differs")

    resources = result.get("resource_metrics", {})
    required_resources = (
        "candidate_generation_elapsed_seconds",
        "candidate_generation_seconds_per_image_per_supply_sum",
        "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes",
        "merged_gallery_bytes",
        "total_arm_elapsed_seconds",
    )
    if not isinstance(resources, dict) or any(
        key not in resources
        or not isinstance(resources[key], (int, float))
        or not math.isfinite(float(resources[key]))
        or float(resources[key]) <= 0
        for key in required_resources
    ):
        raise ValueError("E3 resource telemetry is absent/non-positive")

    diagnostic_files = list((root / "gallery" / "candidate_diagnostics").glob("*.npz"))
    if len(diagnostic_files) != 371:
        raise ValueError("E3 merged gallery does not contain exactly 371 payloads")
    return {
        "schema_version": 1,
        "study": "independent G4 E3 SAM backbone output audit",
        "pass": True,
        "sam_model_type": model_type,
        "sam_checkpoint_sha256": SAM_SHA[model_type],
        "source_commit": SOURCE_COMMIT,
        "protocol_sha256": PROTOCOL_SHA,
        "split_sha256": SPLIT_SHA,
        "images": 371,
        "tumor_images": 184,
        "normal_images": 187,
        "evaluation_summary_sha256": sha256(evaluation_path),
        "selection_manifest_sha256": sha256(selection_path),
        "candidate_manifest_sha256": sha256(gallery_manifest_path),
        "result_summary_sha256": sha256(result_path),
        "summary": summary,
        "resource_metrics": resources,
        "spatial_annotations_opened": 184,
        "test_images_read": 0,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sam-model-type", choices=tuple(SAM_SHA), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("E3 independent audit output already exists")
    report = audit(args.root, args.sam_model_type)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "pass": True,
        "sam_model_type": args.sam_model_type,
        "audit_sha256": sha256(args.output),
        "overall": report["summary"]["overall"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
