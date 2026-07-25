from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "project" / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location(
    "audit_source_consensus_under_test",
    TOOLS / "audit_source_consensus.py",
)
assert spec is not None and spec.loader is not None
AUDIT = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = AUDIT
spec.loader.exec_module(AUDIT)


class SourceConsensusAuditTests(unittest.TestCase):
    def valid_contract(self):
        predeclared = {
            "frozen": {
                "source_consensus_weights": dict(AUDIT.EXPECTED_WEIGHTS),
                "support_clip": "unchanged CAM-only support with kernel 5",
            }
        }
        manifest = {
            "source_commit": AUDIT.EXPECTED_SOURCE_COMMIT,
            "source_hashes": dict(AUDIT.EXPECTED_SOURCE_HASHES),
            "commands": {
                "generate": [
                    "python",
                    "--proposal-teacher-segmentation-checkpoint",
                    "teacher.pt",
                    "--selection-method",
                    "source_consensus",
                ]
            },
        }
        return predeclared, manifest

    def test_exact_source_consensus_contract_passes(self) -> None:
        predeclared, manifest = self.valid_contract()
        result = AUDIT.validate_source_consensus_contract(predeclared, manifest)
        self.assertEqual(result["status"], "PASS")

    def test_changed_weight_fails_closed(self) -> None:
        predeclared, manifest = self.valid_contract()
        predeclared["frozen"]["source_consensus_weights"]["cam_density"] = 0.30
        with self.assertRaisesRegex(ValueError, "weights differ"):
            AUDIT.validate_source_consensus_contract(predeclared, manifest)

    def test_non_consensus_command_fails_closed(self) -> None:
        predeclared, manifest = self.valid_contract()
        manifest["commands"]["generate"][-1] = "coverage_mass_sam"
        with self.assertRaisesRegex(ValueError, "did not use"):
            AUDIT.validate_source_consensus_contract(predeclared, manifest)


if __name__ == "__main__":
    unittest.main()
