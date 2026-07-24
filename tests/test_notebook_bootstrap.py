from __future__ import annotations

import json
import unittest
from pathlib import Path


def cell_source_text(cell: dict[str, object]) -> str:
    """Normalize both valid nbformat source encodings to one text string."""
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(line) for line in source)
    return str(source)


class NotebookBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "thesis_final.ipynb"
        cls.notebook = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
        cls.code_sources = [
            cell_source_text(cell)
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        ]

    def test_bootstrap_precedes_pipeline_initialization(self) -> None:
        self.assertIn("# Cell 0 - Kaggle bootstrap", self.code_sources[0])
        self.assertIn("# Cell 1 - verification", self.code_sources[1])

    def test_bootstrap_provisions_all_external_inputs(self) -> None:
        source = self.code_sources[0]
        for required in (
            "'git', 'clone'",
            "'--branch', REPOSITORY_BRANCH",
            "official_wsss_frozen_test.json",
            "EXPECTED_SPLIT_SHA256",
            "EXPECTED_CHECKPOINT_SHA256",
        ):
            self.assertIn(required, source)

    def test_default_dataset_path_is_the_requested_kaggle_input(self) -> None:
        self.assertIn(
            "/kaggle/input/datasets/itsthang333/btxrd-raw/BTXRD",
            self.code_sources[0],
        )

    def test_notebook_has_exactly_one_test_entrypoint(self) -> None:
        combined = "\n".join(self.code_sources)
        self.assertEqual(combined.count("'--split', 'test'"), 1)
        self.assertNotIn("--threshold-grid", combined)
        self.assertNotIn("supervised_oracle", combined)
        self.assertNotIn("generate_pseudo_masks.py", combined)
        self.assertIn("BTXRD_RUN_LOCKED_TEST", combined)


if __name__ == "__main__":
    unittest.main()
