from __future__ import annotations

import unittest

from project.generate_pseudo_masks import external_saliency_test_authorized


class FinalWsssExternalSaliencyUnlockTests(unittest.TestCase):
    def test_validation_remains_available_without_test_lock(self) -> None:
        self.assertTrue(external_saliency_test_authorized("val", None))

    def test_test_remains_locked_without_exact_wsss_scope(self) -> None:
        self.assertFalse(external_saliency_test_authorized("test", None))
        self.assertFalse(external_saliency_test_authorized("test", {"scope": "joint"}))

    def test_wsss_prediction_lock_authorizes_test_proposal_generation(self) -> None:
        self.assertTrue(
            external_saliency_test_authorized(
                "test",
                {"status": "final", "scope": "wsss_prediction_only"},
            )
        )


if __name__ == "__main__":
    unittest.main()
