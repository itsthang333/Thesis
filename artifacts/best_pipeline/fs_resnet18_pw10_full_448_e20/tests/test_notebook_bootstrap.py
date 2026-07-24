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

    def test_kaggle_defaults_to_hybrid_profile(self) -> None:
        combined = "\n".join(self.code_sources)
        self.assertIn(
            'os.environ.get("BTXRD_PIPELINE_PROFILE", "btxrd_hybrid")',
            combined,
        )


class ColabNotebookBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        notebook_path = Path(__file__).resolve().parents[1] / "thesis_final_colab.ipynb"
        cls.notebook = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
        cls.code_sources = [
            cell_source_text(cell)
            for cell in cls.notebook["cells"]
            if cell.get("cell_type") == "code"
        ]

    def test_colab_gpu_metadata_and_bootstrap_order(self) -> None:
        self.assertEqual(self.notebook["metadata"]["accelerator"], "GPU")
        self.assertEqual(self.notebook["metadata"]["colab"]["gpuType"], "T4")
        self.assertIn("# Cell 0 - Google Colab bootstrap", self.code_sources[0])
        self.assertIn("# Cell 1 - immutable inputs", self.code_sources[1])

    def test_colab_bootstrap_provisions_external_inputs(self) -> None:
        source = self.code_sources[0]
        for required in (
            'Path("/content")',
            '"git", "clone"',
            'DRIVE_DATASET_ROOT = Path("/content/drive/MyDrive/Thesis/BTXRD")',
            'drive.mount("/content/drive")',
            'dataset_root / "images"',
            'dataset_root / "Annotations"',
            'dataset_root / "dataset.xlsx"',
            "sam_vit_b_01ec64.pth",
            "build_btxrd_split_manifest.py",
            'os.environ.setdefault("BTXRD_DISABLE_TQDM", "1")',
        ):
            self.assertIn(required, source)
        self.assertNotIn("kagglehub.dataset_download", source)

    def test_colab_notebook_has_final_drive_persistence(self) -> None:
        source = self.code_sources[-1]
        self.assertIn('drive.mount("/content/drive")', source)
        self.assertIn("shutil.copytree(OUTPUT_ROOT, drive_destination)", source)
        self.assertIn("Refusing to overwrite", source)

    def test_colab_defaults_to_hybrid_profile(self) -> None:
        combined = "\n".join(self.code_sources)
        self.assertIn(
            'os.environ.get("BTXRD_PIPELINE_PROFILE", "btxrd_hybrid")',
            combined,
        )


if __name__ == "__main__":
    unittest.main()
