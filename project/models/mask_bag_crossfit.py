from __future__ import annotations

"""Group-preserving cross-fit helpers for image-label-only proposal MIL."""

import hashlib
import json
from typing import Iterable

import numpy as np


def assign_group_stratified_folds(
    image_labels: np.ndarray,
    group_ids: np.ndarray,
    *,
    fold_count: int,
    seed: int,
) -> np.ndarray:
    """Assign whole groups while balancing image counts within each class."""

    labels = np.asarray(image_labels).reshape(-1)
    groups = np.asarray(group_ids).astype("U128").reshape(-1)
    if labels.shape != groups.shape or labels.size == 0:
        raise ValueError("image_labels and group_ids must be aligned vectors")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("image_labels must be binary")
    labels = labels.astype(np.int8)
    if np.any(groups == ""):
        raise ValueError("group_ids must be nonempty")
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")

    unique_groups = np.unique(groups)
    group_records: list[tuple[str, int, int]] = []
    for group in unique_groups:
        members = groups == group
        group_labels = np.unique(labels[members])
        if len(group_labels) != 1:
            raise ValueError(f"group {group!r} contains mixed image labels")
        group_records.append(
            (str(group), int(group_labels[0]), int(np.count_nonzero(members)))
        )

    class_group_counts = {
        label: sum(record[1] == label for record in group_records)
        for label in (0, 1)
    }
    if any(count < fold_count for count in class_group_counts.values()):
        raise ValueError("each class must contain at least one group per fold")

    rng = np.random.default_rng(seed)
    fold_tie_order = rng.permutation(fold_count)
    fold_tie_rank = np.empty(fold_count, dtype=np.int64)
    fold_tie_rank[fold_tie_order] = np.arange(fold_count)
    fold_rows = np.zeros((fold_count, 2), dtype=np.int64)
    fold_groups = np.zeros((fold_count, 2), dtype=np.int64)
    group_to_fold: dict[str, int] = {}

    for label in (0, 1):
        records = [record for record in group_records if record[1] == label]
        random_ties = {
            group: float(value)
            for (group, _label, _size), value in zip(
                records,
                rng.random(len(records)),
                strict=True,
            )
        }
        records.sort(key=lambda record: (-record[2], random_ties[record[0]]))
        for group, _label, size in records:
            target = min(
                range(fold_count),
                key=lambda fold: (
                    int(fold_rows[fold, label]),
                    int(fold_groups[fold, label]),
                    int(fold_rows[fold].sum()),
                    int(fold_tie_rank[fold]),
                ),
            )
            group_to_fold[group] = target
            fold_rows[target, label] += size
            fold_groups[target, label] += 1

    assignments = np.asarray(
        [group_to_fold[str(group)] for group in groups],
        dtype=np.int32,
    )
    for group in unique_groups:
        if len(np.unique(assignments[groups == group])) != 1:
            raise RuntimeError("cross-fit assignment split a group")
    for fold in range(fold_count):
        fold_labels = labels[assignments == fold]
        if set(np.unique(fold_labels).tolist()) != {0, 1}:
            raise RuntimeError("cross-fit fold lacks one image class")
    return assignments


def crossfit_assignment_manifest(
    image_ids: Iterable[str],
    group_ids: Iterable[str],
    image_labels: Iterable[int],
    fold_ids: Iterable[int],
) -> dict[str, object]:
    """Create a deterministic hashable summary without candidate targets."""

    images = np.asarray(list(image_ids), dtype="U256")
    groups = np.asarray(list(group_ids), dtype="U128")
    labels = np.asarray(list(image_labels), dtype=np.int8)
    folds = np.asarray(list(fold_ids), dtype=np.int32)
    if not (
        images.ndim
        == groups.ndim
        == labels.ndim
        == folds.ndim
        == 1
        and len(images) == len(groups) == len(labels) == len(folds)
        and len(images) > 0
    ):
        raise ValueError("cross-fit manifest arrays must be aligned vectors")
    if len(np.unique(images)) != len(images) or np.any(images == ""):
        raise ValueError("image_ids must be unique and nonempty")
    if not np.isin(labels, (0, 1)).all() or np.any(folds < 0):
        raise ValueError("cross-fit labels/folds are invalid")
    for group in np.unique(groups):
        if len(np.unique(folds[groups == group])) != 1:
            raise ValueError("cross-fit manifest splits a group")

    rows = sorted(
        (
            {
                "image_id": str(image),
                "group_id": str(group),
                "image_label": int(label),
                "heldout_fold": int(fold),
            }
            for image, group, label, fold in zip(
                images,
                groups,
                labels,
                folds,
                strict=True,
            )
        ),
        key=lambda row: row["image_id"],
    )
    row_payload = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    fold_summary = []
    for fold in sorted(np.unique(folds).tolist()):
        heldout = folds == fold
        fold_summary.append(
            {
                "fold": int(fold),
                "images": int(heldout.sum()),
                "groups": int(len(np.unique(groups[heldout]))),
                "normal_images": int(((labels == 0) & heldout).sum()),
                "tumor_images": int(((labels == 1) & heldout).sum()),
            }
        )
    return {
        "schema_version": 1,
        "rows": len(rows),
        "groups": int(len(np.unique(groups))),
        "folds": int(len(np.unique(folds))),
        "row_payload_sha256": hashlib.sha256(row_payload).hexdigest(),
        "fold_summary": fold_summary,
    }


def audit_crossfit_training_exclusion(
    group_ids: Iterable[str],
    heldout_fold_ids: Iterable[int],
    training_groups_by_fold: dict[int, Iterable[str]],
) -> dict[str, object]:
    """Prove each target-producing model excluded its held-out groups."""

    groups = np.asarray(list(group_ids), dtype="U128")
    folds = np.asarray(list(heldout_fold_ids), dtype=np.int32)
    if groups.ndim != 1 or folds.shape != groups.shape or len(groups) == 0:
        raise ValueError("group_ids and heldout_fold_ids must align")
    if np.any(groups == "") or np.any(folds < 0):
        raise ValueError("cross-fit exclusion inputs are invalid")

    records: list[dict[str, int]] = []
    for fold in sorted(np.unique(folds).tolist()):
        if int(fold) not in training_groups_by_fold:
            raise ValueError(f"missing training-group manifest for fold {fold}")
        heldout_groups = set(groups[folds == fold].tolist())
        training_groups = {
            str(group) for group in training_groups_by_fold[int(fold)]
        }
        overlap = heldout_groups & training_groups
        if overlap:
            raise RuntimeError(
                f"fold {fold} target producer trained on held-out groups"
            )
        records.append(
            {
                "fold": int(fold),
                "heldout_groups": len(heldout_groups),
                "training_groups": len(training_groups),
                "overlap": 0,
            }
        )
    return {
        "complete": True,
        "group_overlap": 0,
        "folds": records,
    }


__all__ = [
    "assign_group_stratified_folds",
    "audit_crossfit_training_exclusion",
    "crossfit_assignment_manifest",
]
