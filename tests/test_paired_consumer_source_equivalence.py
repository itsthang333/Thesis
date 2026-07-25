from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "project" / "tools"
spec = importlib.util.spec_from_file_location(
    "audit_paired_consumer_source_equivalence_under_test",
    TOOLS / "audit_paired_consumer_source_equivalence.py",
)
assert spec is not None and spec.loader is not None
AUDIT = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = AUDIT
spec.loader.exec_module(AUDIT)
REFERENCE = (
    ROOT
    / "artifacts"
    / "best_pipeline"
    / "fs_resnet18_pw10_full_448_e20"
    / "project"
)


class PairedConsumerSourceEquivalenceTests(unittest.TestCase):
    def test_current_consumer_matches_reviewed_equivalence_contract(self) -> None:
        result = AUDIT.audit(REFERENCE, ROOT / "project")
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(
            result["paired_contract_conclusion"][
                "consumer_behavior_drift_detected"
            ]
        )

    def test_model_source_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "project"
            shutil.copytree(ROOT / "project", current)
            model = current / "models" / "unet.py"
            model.write_text(
                model.read_text(encoding="utf-8") + "\n# mutation\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "Current paired-consumer source changed",
            ):
                AUDIT.audit(REFERENCE, current)


if __name__ == "__main__":
    unittest.main()
