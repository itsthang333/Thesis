from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED = {
    "train_segmentation.py": {
        "reference": "5c940cf86ff95d395b320d9f973b625d3d15b5296820606a654a404259891fd4",
        "current": "50f1e0f2f768e5bc2c85514f2ec2c3107a26f5f2ea13775efab62d85ffd5ee4b",
        "diff": "071b5eef9cf0d5bf5fdaab87598e7f5bb4dbc56d223c9fea38efedbd0b70199a",
    },
    "models/unet.py": {
        "reference": "aa867b3c95aeed4e906dd03203283bee7d5dd717e1144ba626bd41150c88bf64",
        "current": "aa867b3c95aeed4e906dd03203283bee7d5dd717e1144ba626bd41150c88bf64",
        "diff": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    },
    "models/losses.py": {
        "reference": "330b1d4eb536c078ea73f4ea68b1ba34ea3392704c104dd99b5f7950a8d51e7c",
        "current": "330b1d4eb536c078ea73f4ea68b1ba34ea3392704c104dd99b5f7950a8d51e7c",
        "diff": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    },
    "datasets/btxrd.py": {
        "reference": "e0b78250577804092549bce848476c55f1ed432530238c6a1b070642af720ae9",
        "current": "d8f0804be4e81cdb4d58e4673708c1067eb7d9b49b42bb78cb6051188c156001",
        "diff": "14e401bf65cb518175b20e5fa17efa6cfbff700443ffa68baa108391d7713992",
    },
    "evaluate_unet.py": {
        "reference": "61cf37093f5e4d335d47d33c2d2e7e6c14cd712badc761afc27aeed30f0860eb",
        "current": "61cf37093f5e4d335d47d33c2d2e7e6c14cd712badc761afc27aeed30f0860eb",
        "diff": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    },
    "evaluation/segmentation_metrics.py": {
        "reference": "070ebb9f1092dac5fe87ff7c1acc7470d5834ce4e1dae0373b8ef60783cf314a",
        "current": "070ebb9f1092dac5fe87ff7c1acc7470d5834ce4e1dae0373b8ef60783cf314a",
        "diff": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    },
}


def canonical_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unified_diff_sha256(
    reference_data: bytes,
    current_data: bytes,
    relative: str,
) -> str:
    diff = "".join(
        difflib.unified_diff(
            reference_data.decode("utf-8").splitlines(keepends=True),
            current_data.decode("utf-8").splitlines(keepends=True),
            fromfile=f"reference/{relative}",
            tofile=f"current/{relative}",
        )
    ).encode("utf-8")
    return sha256(diff)


def audit(reference_project: Path, current_project: Path) -> dict[str, Any]:
    file_evidence: dict[str, Any] = {}
    for relative, expected in EXPECTED.items():
        reference_path = reference_project / relative
        current_path = current_project / relative
        reference_data = canonical_bytes(reference_path)
        current_data = canonical_bytes(current_path)
        actual = {
            "reference_canonical_lf_sha256": sha256(reference_data),
            "current_canonical_lf_sha256": sha256(current_data),
            "unified_diff_canonical_lf_sha256": unified_diff_sha256(
                reference_data,
                current_data,
                relative,
            ),
        }
        if actual["reference_canonical_lf_sha256"] != expected["reference"]:
            raise ValueError(f"Frozen reference source changed: {relative}")
        if actual["current_canonical_lf_sha256"] != expected["current"]:
            raise ValueError(f"Current paired-consumer source changed: {relative}")
        if actual["unified_diff_canonical_lf_sha256"] != expected["diff"]:
            raise ValueError(f"Paired-consumer source diff changed: {relative}")
        actual["status"] = "PASS"
        file_evidence[relative] = actual

    return {
        "status": "PASS",
        "audit_role": (
            "static equivalence of frozen GT consumer and current paired WSL "
            "consumer; no training or evaluation was executed"
        ),
        "file_evidence": file_evidence,
        "bit_identical_core": {
            "architecture": True,
            "loss": True,
            "evaluation_entrypoint": True,
            "segmentation_metrics": True,
        },
        "reviewed_non_behavioral_or_label_source_changes": {
            "train_segmentation.py": (
                "adds fail-closed paired-reference validation and embeds its "
                "provenance in resolved config/checkpoints; optimizer, AMP, "
                "DataLoaders, seed, model construction, loss and train/val "
                "loops are unchanged"
            ),
            "datasets/btxrd.py": (
                "relaxes only pseudo-manifest source-grid equality and records "
                "source/consumer resize provenance; GT mask construction, "
                "image transforms, mask transforms and augmentation are unchanged"
            ),
        },
        "paired_contract_conclusion": {
            "consumer_behavior_drift_detected": False,
            "only_allowed_scientific_difference": (
                "training mask source: GT polygons in reference arm versus "
                "image-label-only pseudo masks in experimental arm"
            ),
            "validation_gt_usage": "evaluation/model selection only",
            "test_evaluated": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-project", type=Path, required=True)
    parser.add_argument("--current-project", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(
        args.reference_project.resolve(),
        args.current_project.resolve(),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
