from __future__ import annotations

import unittest

from project.generate_pseudo_masks import (
    external_saliency_image_label_target_authorized,
    external_saliency_test_authorized,
)


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

    def test_binary_and_collapsed_ten_class_are_the_same_authorized_event(self) -> None:
        self.assertTrue(
            external_saliency_image_label_target_authorized(
                ["tumor"], cam_target_class="ground_truth", cam_aggregation="class"
            )
        )
        self.assertTrue(
            external_saliency_image_label_target_authorized(
                ["tumor_type"],
                cam_target_class="ground_truth",
                cam_aggregation="tumor_log_odds",
            )
        )

    def test_subtype_or_mismatched_targets_remain_rejected(self) -> None:
        for columns, target, aggregation in (
            (["tumor_type"], "ground_truth", "class"),
            (["tumor_type"], "predicted", "tumor_log_odds"),
            (["tumor"], "ground_truth", "tumor_log_odds"),
            (["tumor"], "predicted", "class"),
        ):
            self.assertFalse(
                external_saliency_image_label_target_authorized(
                    columns,
                    cam_target_class=target,
                    cam_aggregation=aggregation,
                )
            )


if __name__ == "__main__":
    unittest.main()
