from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "project" / "tools"
spec = importlib.util.spec_from_file_location(
    "audit_biomedclip_smoke_under_test",
    TOOLS / "audit_biomedclip_smoke.py",
)
assert spec is not None and spec.loader is not None
AUDIT = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = AUDIT
spec.loader.exec_module(AUDIT)


def make_train_rows() -> list[dict[str, str]]:
    return [
        {"image_id": f"normal-{index:03d}.jpeg", "tumor": "0"}
        for index in range(40)
    ] + [
        {"image_id": f"tumor-{index:03d}.jpeg", "tumor": "1"}
        for index in range(40)
    ]


def make_valid_payloads():
    train_rows = make_train_rows()
    score_ids, saliency_ids = AUDIT.expected_sample_ids(train_rows)
    labels = {row["image_id"]: int(row["tumor"]) for row in train_rows}
    score_rows = [
        {
            "image_id": image_id,
            "image_label": labels[image_id],
            "score": 0.2 * labels[image_id] + index * 1e-4,
        }
        for index, image_id in enumerate(score_ids)
    ]
    values = [row["score"] for row in score_rows]
    tumor = [row["score"] for row in score_rows if row["image_label"] == 1]
    normal = [row["score"] for row in score_rows if row["image_label"] == 0]
    overall_mean = sum(values) / len(values)
    score_diagnostic = {
        "tumor_mean": sum(tumor) / len(tumor),
        "normal_mean": sum(normal) / len(normal),
        "tumor_minus_normal": sum(tumor) / len(tumor) - sum(normal) / len(normal),
        "overall_std": (
            sum((value - overall_mean) ** 2 for value in values) / len(values)
        )
        ** 0.5,
        "rows": score_rows,
    }
    saliency_rows = [
        {
            "image_id": image_id,
            "image_label": labels[image_id],
            "contrast_score": 0.1,
            "shape": [14, 14],
            "minimum": 0.0,
            "maximum": 0.1,
            "mean": 0.02,
            "dynamic_range": 0.1,
            "finite": True,
        }
        for image_id in saliency_ids
    ]
    prompt_sha = AUDIT.sha256_json(AUDIT.EXPECTED_PROMPTS)
    predeclared = {
        "source_commit": AUDIT.EXPECTED_SOURCE_COMMIT,
        "model_id": AUDIT.EXPECTED_MODEL_ID,
        "open_clip_torch": AUDIT.EXPECTED_OPEN_CLIP_VERSION,
        "transformers": AUDIT.EXPECTED_TRANSFORMERS_VERSION,
        "prompts": copy.deepcopy(AUDIT.EXPECTED_PROMPTS),
        "prompt_sha256": prompt_sha,
        "validation_masks_read": False,
        "test_evaluated": False,
    }
    result = {
        "status": "PASS",
        "source_commit": AUDIT.EXPECTED_SOURCE_COMMIT,
        "model_id": AUDIT.EXPECTED_MODEL_ID,
        "prompt_sha256": prompt_sha,
        "model_weight_files": [
            {"name": "weights.bin", "bytes": 2_000_000, "sha256": "a" * 64}
        ],
        "population": {
            "score_images": 64,
            "saliency_images": 8,
            "source_split": "train",
            "validation_images": 0,
            "test_images": 0,
        },
        "score_diagnostic": score_diagnostic,
        "saliency_diagnostic": {
            "target_layer": "model.visual.trunk.blocks[11].norm2",
            "repeat_max_abs_delta": 0.0,
            "rows": saliency_rows,
        },
        "environment": {
            "open_clip": AUDIT.EXPECTED_OPEN_CLIP_VERSION,
            "transformers": AUDIT.EXPECTED_TRANSFORMERS_VERSION,
        },
        "validation_masks_read": False,
        "test_evaluated": False,
    }
    return predeclared, result, train_rows


class BiomedClipSmokeAuditTests(unittest.TestCase):
    def test_valid_payload_passes(self) -> None:
        predeclared, result, rows = make_valid_payloads()
        audited = AUDIT.validate_payloads(predeclared, result, rows)
        self.assertEqual(audited["status"], "PASS")

    def test_changed_prompt_fails_closed(self) -> None:
        predeclared, result, rows = make_valid_payloads()
        predeclared["prompts"]["tumor"][0] = "changed"
        with self.assertRaisesRegex(ValueError, "prompt text"):
            AUDIT.validate_payloads(predeclared, result, rows)

    def test_validation_access_fails_closed(self) -> None:
        predeclared, result, rows = make_valid_payloads()
        result["population"]["validation_images"] = 1
        with self.assertRaisesRegex(ValueError, "population"):
            AUDIT.validate_payloads(predeclared, result, rows)

    def test_wrong_sample_order_fails_closed(self) -> None:
        predeclared, result, rows = make_valid_payloads()
        result["score_diagnostic"]["rows"][0], result["score_diagnostic"]["rows"][1] = (
            result["score_diagnostic"]["rows"][1],
            result["score_diagnostic"]["rows"][0],
        )
        with self.assertRaisesRegex(ValueError, "identities/order"):
            AUDIT.validate_payloads(predeclared, result, rows)

    def test_constant_saliency_fails_closed(self) -> None:
        predeclared, result, rows = make_valid_payloads()
        result["saliency_diagnostic"]["rows"][0]["maximum"] = 0.0
        result["saliency_diagnostic"]["rows"][0]["mean"] = 0.0
        result["saliency_diagnostic"]["rows"][0]["dynamic_range"] = 0.0
        with self.assertRaisesRegex(ValueError, "constant"):
            AUDIT.validate_payloads(predeclared, result, rows)


if __name__ == "__main__":
    unittest.main()
