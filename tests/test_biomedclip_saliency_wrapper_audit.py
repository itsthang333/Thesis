from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "project" / "tools"
spec = importlib.util.spec_from_file_location(
    "audit_biomedclip_saliency_wrapper_under_test",
    TOOLS / "audit_biomedclip_saliency_wrapper.py",
)
assert spec is not None and spec.loader is not None
AUDIT = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = AUDIT
spec.loader.exec_module(AUDIT)
WRAPPER = (
    ROOT
    / "tmp"
    / "kaggle"
    / "biomedclip_tiled_saliency_val_v1"
    / "run_biomedclip_tiled_saliency_val.py"
)


class BiomedClipSaliencyWrapperAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = WRAPPER.read_text(encoding="utf-8")

    def test_frozen_wrapper_passes(self) -> None:
        result = AUDIT.validate_wrapper_source(
            self.source,
            wrapper_sha256=AUDIT.EXPECTED_WRAPPER_SHA256,
        )
        self.assertEqual(result["status"], "PASS")

    def test_wrapper_hash_drift_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "wrapper SHA-256"):
            AUDIT.validate_wrapper_source(self.source, wrapper_sha256="0" * 64)

    def test_annotation_token_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden token"):
            AUDIT.validate_wrapper_source(
                self.source + "\n# Annotations\n",
                wrapper_sha256=AUDIT.EXPECTED_WRAPPER_SHA256,
            )

    def test_target_layer_drift_fails_closed(self) -> None:
        changed = self.source.replace(
            '"target_layer": "model.visual.trunk.blocks[11].norm1"',
            '"target_layer": "model.visual.trunk.blocks[11].norm2"',
        )
        with self.assertRaisesRegex(ValueError, "target-layer"):
            AUDIT.validate_wrapper_source(
                changed,
                wrapper_sha256=AUDIT.EXPECTED_WRAPPER_SHA256,
            )


if __name__ == "__main__":
    unittest.main()
