from __future__ import annotations

"""Run the same-gallery B4 class-contrast BAS selector probe.

The implementation deliberately reuses the already audited BAS training and
same-gallery transport machinery from the never-launched B1 static runner.  B4
changes only the predeclared scientific variable: a 448-pixel tumor-vs-normal
BAS contrast rank is added to the trusted Geometry-v3/upstream equal-rank
architecture. No collaborator gallery, checkpoint, prediction, or output is an
input.
"""

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_bas_candidate_descriptor_core as base
from datasets.btxrd_image_label_only import (
    build_image_label_only_classification_dataset,
)


EXPERIMENT_ID = "EXP-20260801-codex-b4-same-gallery-bas-semantic-v1"
RUN_ID = "btxrd_same_gallery_bas_semantic_b4_v1"
CONTROL_ARM = "geometry_v3_plus_upstream_equal_rank"
SEMANTIC_ARM = "geometry_v3_plus_upstream_plus_class_contrast_bas"
EXTRA_PROVENANCE = {
    "input_size": 448,
    "semantic_map": "tumor_over_tumor_plus_normal",
    "control_formula": "mean_rank_geometry_v3_upstream",
    "semantic_formula": "mean_rank_geometry_v3_upstream_class_contrast_bas",
}


@torch.inference_mode()
def _classify_and_all_localization(
    model: base.BASResNet50Localizer,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read both existing BAS class maps without changing the frozen model."""

    x = model.stem(images)
    x = model.layer1(x)
    x = model.layer2(x)
    stage3 = model.layer3(x)
    maps = model.localization_head(stage3)
    class_maps = model.classifier_head(
        model.layer4(base.F.max_pool2d(stage3, kernel_size=2))
    )
    logits = base.F.adaptive_avg_pool2d(class_maps, 1).flatten(1)
    if not torch.isfinite(logits).all() or not torch.isfinite(maps).all():
        raise RuntimeError("B4 BAS inference output is non-finite")
    return logits, maps


@torch.inference_mode()
def _validation_activations(
    model: base.BASResNet50Localizer,
    loader: DataLoader,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Freeze raw class maps and parameter-free tumor/normal competition."""

    model.eval()
    activations: dict[str, np.ndarray] = {}
    labels: list[int] = []
    probabilities: list[float] = []
    tumor_ranges: list[float] = []
    contrast_ranges: list[float] = []
    for images, targets, image_ids in loader:
        images = images.cuda(non_blocking=True)
        logits, maps = _classify_and_all_localization(model, images)
        flip_logits, flip_maps = _classify_and_all_localization(
            model,
            torch.flip(images, dims=(-1,))
        )
        logits = 0.5 * (logits + flip_logits)
        maps = 0.5 * (maps + torch.flip(flip_maps, dims=(-1,)))
        probabilities_batch = torch.softmax(logits.float(), dim=1)[:, 1]
        normal = maps[:, 0].float()
        tumor = maps[:, 1].float()
        contrast = tumor / (tumor + normal).clamp_min(1.0e-8)
        for index, image_id in enumerate(image_ids):
            key = str(image_id)
            value = torch.stack(
                (normal[index], tumor[index], contrast[index]), dim=0
            ).cpu().numpy().astype(np.float32)
            if key in activations or not np.isfinite(value).all():
                raise RuntimeError(f"invalid/duplicate B4 activation: {key}")
            activations[key] = value
            label = int(targets[index].reshape(-1)[0].item())
            labels.append(label)
            probabilities.append(float(probabilities_batch[index].cpu()))
            if label == 1:
                tumor_ranges.append(float(np.ptp(value[1])))
                contrast_ranges.append(float(np.ptp(value[2])))
    if len(activations) != 371 or len(tumor_ranges) != 184:
        raise RuntimeError("B4 validation activation cohort mismatch")
    metrics = base._binary_metrics(
        np.asarray(labels, dtype=np.int8),
        np.asarray(probabilities, dtype=np.float64),
    )
    tumor_nondegenerate = sum(value > 1.0e-4 for value in tumor_ranges)
    contrast_nondegenerate = sum(value > 1.0e-4 for value in contrast_ranges)
    gate: dict[str, Any] = {
        **metrics,
        "validation_images": 371,
        "validation_tumor_images": 184,
        "finite_activation_maps": 371,
        "tumor_nondegenerate_activation_maps": tumor_nondegenerate,
        "tumor_nondegenerate_fraction": tumor_nondegenerate / 184.0,
        "contrast_nondegenerate_activation_maps": contrast_nondegenerate,
        "contrast_nondegenerate_fraction": contrast_nondegenerate / 184.0,
        "minimum_auroc": 0.75,
        "minimum_sensitivity": 0.60,
        "minimum_specificity": 0.60,
        "minimum_tumor_nondegenerate_fraction": 0.95,
        "minimum_contrast_nondegenerate_fraction": 0.95,
    }
    gate["operational_gate_pass"] = bool(
        metrics["auroc"] >= 0.75
        and metrics["sensitivity"] >= 0.60
        and metrics["specificity"] >= 0.60
        and tumor_nondegenerate / 184.0 >= 0.95
        and contrast_nondegenerate / 184.0 >= 0.95
    )
    return activations, gate


def _score_arms(
    output_dir: Path,
    records: list[dict[str, Any]],
    base_scored: list[dict[str, Any]],
    baseline_rows: list[dict[str, str]],
    activations: dict[str, np.ndarray],
    candidate_root: Path,
    candidate_rows: dict[str, dict[str, str]],
) -> tuple[dict[str, list[dict[str, Any]]], str, dict[str, float | int]]:
    """Score the trusted two-rank control and one class-contrast BAS arm."""

    accepted = {row["image_id"]: row for row in baseline_rows}
    evidence_root = output_dir / "activation_evidence"
    evidence_root.mkdir(parents=True, exist_ok=False)
    evidence_rows: list[dict[str, object]] = []
    arms: dict[str, list[dict[str, Any]]] = {
        CONTROL_ARM: [],
        SEMANTIC_ARM: [],
    }
    correlations: list[float] = []
    changed_selections = 0
    if len(records) != len(base_scored):
        raise ValueError("B4 activation/base records do not align")

    for index, (record, scored) in enumerate(zip(records, base_scored)):
        image_id = str(record["image_id"])
        masks = base.unpack_candidate_masks(record["packed_masks"]).astype(np.float32)
        activation = np.asarray(activations[image_id], dtype=np.float32)
        if activation.ndim != 3 or activation.shape[0] != 3:
            raise ValueError(f"B4 activation contract mismatch: {image_id}")
        activation_tensor = torch.from_numpy(activation[2])[None, None]
        mask_tensor = torch.from_numpy(masks)[None]
        valid = torch.ones((1, masks.shape[0]), dtype=torch.bool)
        coverage, purity, harmonic = base.candidate_activation_evidence(
            activation_tensor,
            mask_tensor,
            valid,
        )
        contrast_rank = base.within_bag_percentile_ranks(harmonic, valid)[0]
        base_logits = torch.from_numpy(
            np.asarray(scored["base_candidate_logits"], dtype=np.float32)
        )[None]
        base_rank = base.within_bag_percentile_ranks(base_logits, valid)[0]

        # Candidate diagnostics are used only to prove that cache rows still
        # refer to the immutable same-gallery payload; no upstream score enters
        # either arm.
        candidate_row = candidate_rows[Path(image_id).stem]
        candidate_path = candidate_root / candidate_row["diagnostic_path"]
        if (
            base.sha256_file(candidate_path) != candidate_row["diagnostic_sha256"]
            or candidate_row["diagnostic_sha256"]
            != record["candidate_payload_sha256"]
        ):
            raise ValueError(f"B4 candidate provenance mismatch: {image_id}")

        with np.load(candidate_path, allow_pickle=False) as candidate_payload:
            all_upstream = np.asarray(
                candidate_payload["selection_scores"], dtype=np.float32
            )
        kept = np.asarray(record["candidate_indices"], dtype=np.int64)
        if (
            all_upstream.ndim != 1
            or np.any(kept < 0)
            or np.any(kept >= len(all_upstream))
            or not np.isfinite(all_upstream).all()
        ):
            raise ValueError(f"B4 upstream score alignment mismatch: {image_id}")
        upstream = torch.from_numpy(all_upstream[kept])[None]
        upstream_rank = base.within_bag_percentile_ranks(upstream, valid)[0]
        control_rank = base.equal_rank_aggregate(
            (base_logits, upstream), valid
        )[0]
        semantic_rank = base.equal_rank_aggregate(
            (base_logits, upstream, harmonic), valid
        )[0]

        if len(control_rank) > 1:
            correlation = float(
                np.corrcoef(
                    control_rank.numpy().astype(np.float64),
                    contrast_rank.numpy().astype(np.float64),
                )[0, 1]
            )
            if np.isfinite(correlation):
                correlations.append(correlation)
        changed_selections += int(
            int(torch.argmax(control_rank)) != int(torch.argmax(semantic_rank))
        )

        relative = Path(f"{index:04d}_{Path(image_id).stem}.npz")
        evidence_path = evidence_root / relative
        np.savez_compressed(
            evidence_path,
            activation=activation[2],
            normal_activation=activation[0],
            tumor_activation=activation[1],
            class_contrast_activation=activation[2],
            candidate_indices=np.asarray(record["candidate_indices"], dtype=np.int32),
            coverage=coverage[0].numpy().astype(np.float32),
            purity=purity[0].numpy().astype(np.float32),
            harmonic=harmonic[0].numpy().astype(np.float32),
            activation_rank=contrast_rank.numpy().astype(np.float32),
            baseline_logits=np.asarray(
                scored["base_candidate_logits"], dtype=np.float32
            ),
            baseline_rank=base_rank.numpy().astype(np.float32),
            upstream_scores=upstream[0].numpy().astype(np.float32),
            upstream_rank=upstream_rank.numpy().astype(np.float32),
            control_rank=control_rank.numpy().astype(np.float32),
            semantic_rank=semantic_rank.numpy().astype(np.float32),
        )
        evidence_rows.append(
            {
                "image_id": image_id,
                "group_id": record["group_id"],
                "tumor": record["label"],
                "candidate_count": masks.shape[0],
                "evidence_path": str(relative),
                "evidence_sha256": base.sha256_file(evidence_path),
                "tumor_activation_range": float(np.ptp(activation[1])),
                "class_contrast_activation_range": float(np.ptp(activation[2])),
                "baseline_logits_sha256": sha256(
                    np.asarray(scored["base_candidate_logits"], dtype=np.float32).tobytes()
                ).hexdigest(),
            }
        )
        common = {
            "image_id": image_id,
            "bag_logit": float(accepted[image_id]["bag_logit"]),
            "bag_probability": float(accepted[image_id]["bag_probability"]),
        }
        arms[CONTROL_ARM].append(
            {**common, "candidate_logits": control_rank.numpy().astype(np.float32)}
        )
        arms[SEMANTIC_ARM].append(
            {**common, "candidate_logits": semantic_rank.numpy().astype(np.float32)}
        )

    if not correlations:
        raise RuntimeError("B4 BAS/Geometry-v3 rank correlation is undefined")
    diagnostics: dict[str, float | int] = {
        "mean_contrast_baseline_rank_correlation": float(np.mean(correlations)),
        "correlation_images": len(correlations),
        "semantic_changed_selections": changed_selections,
        "semantic_changed_selection_fraction": changed_selections / 371.0,
    }
    return (
        arms,
        base._write_csv(evidence_root / "activation_manifest.csv", evidence_rows),
        diagnostics,
    )


def main() -> None:
    # These are frozen B4 protocol constants, not runtime-tunable options.
    base.EXPECTED_IMAGE_SIZE = 448
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.RUN_ID = RUN_ID
    base.REQUIRE_DIAGNOSTIC_PASS_TO_FREEZE = False
    base.CORRELATION_KEY = "mean_contrast_baseline_rank_correlation"
    base.CHANGE_FRACTION_KEY = "semantic_changed_selection_fraction"
    base.MAXIMUM_CORRELATION_FIELD = "maximum_mean_contrast_baseline_rank_correlation"
    base.MAXIMUM_CORRELATION = 0.95
    base.MINIMUM_CHANGE_FRACTION = 0.05
    base.EXTRA_PROVENANCE = EXTRA_PROVENANCE
    base.build_classification_dataset = build_image_label_only_classification_dataset
    base._score_arms = _score_arms
    base._validation_activations = _validation_activations
    base.main()


if __name__ == "__main__":
    main()
