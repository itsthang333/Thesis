from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1] / "project"
sys.path.insert(0, str(PROJECT))

from analyze_frozen_gallery_union_oracle import analyze  # noqa: E402


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class FrozenGalleryUnionOracleTests(unittest.TestCase):
    def _write_source(
        self,
        root: Path,
        name: str,
        values: list[float],
        *,
        with_groups: bool,
        test_evaluated: bool = False,
    ) -> dict[str, str]:
        csv_path = root / f"{name}.csv"
        fieldnames = ["image_name", "oracle_best_single_dice"]
        if with_groups:
            fieldnames.append("size_group")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, value in enumerate(values):
                row = {
                    "image_name": f"IMG{index:06d}.jpeg",
                    "oracle_best_single_dice": value,
                }
                if with_groups:
                    row["size_group"] = (
                        "small" if index < 94 else "medium" if index < 166 else "large"
                    )
                writer.writerow(row)
        contract = root / f"{name}.json"
        contract.write_text(
            json.dumps({"test_evaluated": test_evaluated}) + "\n",
            encoding="utf-8",
        )
        return {
            "name": name,
            "prompt_quality_csv": csv_path.name,
            "prompt_quality_sha256": _sha(csv_path),
            "contract_json": contract.name,
            "contract_sha256": _sha(contract),
        }

    def test_selects_small_priority_pair_after_all_metric_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = [0.41] * 184
            small = [0.7] * 94 + [0.41] * 90
            extent = [0.41] * 94 + [0.8] * 90
            spec = {
                "schema_version": 1,
                "sources": [
                    self._write_source(root, "base", base, with_groups=True),
                    self._write_source(root, "small", small, with_groups=False),
                    self._write_source(root, "extent", extent, with_groups=False),
                ],
                "fully_reference": {
                    "overall": 0.4,
                    "small": 0.4,
                    "medium": 0.4,
                    "large": 0.4,
                },
                "required_anchor_source": "base",
                "test_evaluated": False,
            }
            result = analyze(spec, root)
            self.assertEqual(
                result["recommended_minimal_pair"]["sources"], ["base", "small"]
            )
            self.assertEqual(result["cohort"], {"tumor": 184, "small": 94, "medium": 72, "large": 18})
            self.assertFalse(result["test_evaluated"])

    def test_rejects_source_contract_that_opened_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._write_source(
                root, "bad", [0.5] * 184, with_groups=True, test_evaluated=True
            )
            spec = {
                "schema_version": 1,
                "sources": [source, self._write_source(root, "ok", [0.6] * 184, with_groups=False)],
                "fully_reference": {key: 0.4 for key in ("overall", "small", "medium", "large")},
                "required_anchor_source": "bad",
                "test_evaluated": False,
            }
            with self.assertRaisesRegex(ValueError, "test locked"):
                analyze(spec, root)

    def test_rejects_tampered_prompt_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._write_source(root, "a", [0.5] * 184, with_groups=True)
            (root / "a.csv").write_text("tampered\n", encoding="utf-8")
            spec = {
                "schema_version": 1,
                "sources": [source],
                "fully_reference": {key: 0.4 for key in ("overall", "small", "medium", "large")},
                "required_anchor_source": "a",
                "test_evaluated": False,
            }
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                analyze(spec, root)


if __name__ == "__main__":
    unittest.main()
