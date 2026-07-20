from __future__ import annotations

import json
import unittest
from pathlib import Path


class NotebookBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "thesis_final.ipynb"
        cls.notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cls.code_sources = [
            cell.get("source", "")
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        ]

    def test_bootstrap_precedes_pipeline_initialization(self) -> None:
        self.assertIn("# Cell 0 - Kaggle bootstrap", self.code_sources[0])
        self.assertIn("# Cell 1 - immutable inputs", self.code_sources[1])

    def test_bootstrap_provisions_all_external_inputs(self) -> None:
        source = self.code_sources[0]
        for required in (
            '"git", "clone"',
            '"--branch", REPOSITORY_BRANCH',
            'os.environ["BTXRD_GIT_COMMIT"]',
            '"pip", "install"',
            "sam_vit_b_01ec64.pth",
            "build_btxrd_split_manifest.py",
            'os.environ["BTXRD_SPLIT_MANIFEST"]',
        ):
            self.assertIn(required, source)

    def test_default_dataset_path_is_the_requested_kaggle_input(self) -> None:
        self.assertIn(
            "/kaggle/input/datasets/wanwin/data-btxrd/BTXRD",
            self.code_sources[0],
        )


if __name__ == "__main__":
    unittest.main()
