from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "project" / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location(
    "audit_biomedclip_protocol_under_test",
    TOOLS / "audit_biomedclip_protocol.py",
)
assert spec is not None and spec.loader is not None
AUDIT = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = AUDIT
spec.loader.exec_module(AUDIT)


class BiomedClipProtocolAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(
            (ROOT / "artifacts" / "research_protocols" / "biomedclip_tiled_val_v1.json").read_text(
                encoding="utf-8"
            )
        )

    def test_frozen_protocol_passes(self) -> None:
        AUDIT.validate_protocol(copy.deepcopy(self.protocol))

    def test_population_drift_fails_closed(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["population"]["subgroups"]["small_lt_1pct"] = 93
        with self.assertRaisesRegex(ValueError, "population"):
            AUDIT.validate_protocol(protocol)

    def test_oracle_gate_cannot_authorize_train_masks(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["promotion_gates"]["direct_train_pseudo_mask_generation"] = protocol[
            "promotion_gates"
        ]["localization_source_to_selector_research"]
        with self.assertRaisesRegex(ValueError, "direct-promotion"):
            AUDIT.validate_protocol(protocol)

    def test_test_unlock_fails_closed(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["test_evaluated"] = True
        with self.assertRaisesRegex(ValueError, "lock test"):
            AUDIT.validate_protocol(protocol)


if __name__ == "__main__":
    unittest.main()
