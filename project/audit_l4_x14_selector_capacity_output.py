from __future__ import annotations

"""Independent fail-closed audit for L4 X14 Stage-A output."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


SEEDS = (42, 43, 44)
ARCHITECTURES = ("linear", "one_hidden", "two_hidden")
BASELINE_ARM = "E8__R7"
UPSTREAM_ARM = "X14__upstream"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def expected_models() -> set[str]:
    return {
        f"{architecture}__seed{seed}"
        for seed in SEEDS
        for architecture in ARCHITECTURES
    }


def expected_arms() -> set[str]:
    return {
        BASELINE_ARM,
        UPSTREAM_ARM,
        *(
            f"X14__{architecture}_{mode}__seed{seed}"
            for seed in SEEDS
            for architecture in ARCHITECTURES
            for mode in ("only", "r7")
        ),
    }


def require_finite(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {name}")
    return result


def audit(
    root: Path,
    *,
    expected_source_commit: str,
    expected_protocol_sha256: str,
    expected_split_sha256: str,
    expected_candidate_manifest_sha256: str,
) -> dict[str, object]:
    paths = {
        "freeze": root / "g4_choice_freeze.json",
        "choices": root / "g4_choices.csv",
        "histories": root / "training_histories.json",
        "metrics": root / "inner_holdout_image_label_metrics.json",
        "inner": root / "x14_inner_split.csv",
        "manifest": root / "run_manifest.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing X14 artifacts: {missing}")

    freeze = read_json(paths["freeze"])
    arms = expected_arms()
    models = expected_models()
    if (
        freeze.get("stage") != "l4_x14_selector_capacity_choice_freeze_v1"
        or freeze.get("study") != "L4 X14 matched selector-capacity diagnostic"
        or freeze.get("source_commit") != expected_source_commit
        or freeze.get("protocol_sha256") != expected_protocol_sha256
        or freeze.get("split_sha256") != expected_split_sha256
        or freeze.get("candidate_manifest_sha256") != expected_candidate_manifest_sha256
        or freeze.get("baseline_arm") != BASELINE_ARM
        or int(freeze.get("images", -1)) != 371
        or int(freeze.get("tumor_images", -1)) != 184
        or int(freeze.get("selection_rows", -1)) != 371 * len(arms)
        or freeze.get("seeds") != list(SEEDS)
        or freeze.get("architectures") != list(ARCHITECTURES)
        or set(freeze.get("arms", [])) != arms
        or int(freeze.get("baseline_r7_exact_matches", -1)) != 371
        or freeze.get("same_descriptor_cache") is not True
        or freeze.get("same_inner_split") is not True
        or freeze.get("same_mil_objective_optimizer_epochs") is not True
        or freeze.get("no_best_epoch_selection") is not True
        or freeze.get("candidate_choices_frozen_before_spatial_gt") is not True
        or freeze.get("spatial_ground_truth_used") is not False
        or freeze.get("validation_gt_read") is not False
        or int(freeze.get("test_images_read", -1)) != 0
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("X14 freeze contract differs")
    if (
        freeze.get("choices_sha256") != sha256(paths["choices"])
        or freeze.get("training_histories_sha256") != sha256(paths["histories"])
        or freeze.get("inner_holdout_metrics_sha256") != sha256(paths["metrics"])
        or freeze.get("inner_split_sha256") != sha256(paths["inner"])
    ):
        raise ValueError("X14 frozen artifact hash differs")

    parameter_counts = freeze.get("architecture_trainable_parameters")
    if not isinstance(parameter_counts, dict) or set(parameter_counts) != set(ARCHITECTURES):
        raise ValueError("X14 parameter-count receipt differs")
    ordered = [int(parameter_counts[name]) for name in ARCHITECTURES]
    if not ordered[0] < ordered[1] < ordered[2]:
        raise ValueError("X14 capacity is not strictly ordered")

    checkpoints = freeze.get("checkpoint_sha256")
    if not isinstance(checkpoints, dict) or set(checkpoints) != models:
        raise ValueError("X14 checkpoint set differs")
    for name, digest in checkpoints.items():
        path = root / "checkpoints" / f"{name}.pt"
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"X14 checkpoint differs: {name}")

    histories = read_json(paths["histories"])
    if set(histories) != models:
        raise ValueError("X14 history set differs")
    for name, rows in histories.items():
        if not isinstance(rows, list) or [int(row["epoch"]) for row in rows] != list(range(1, 17)):
            raise ValueError(f"X14 fixed epoch history differs: {name}")
        for row in rows:
            for metric in ("total", "image", "instance", "consistency"):
                require_finite(row[metric], f"{name}/{metric}")

    metrics = read_json(paths["metrics"])
    if set(metrics) != models:
        raise ValueError("X14 inner-holdout metric set differs")
    holdout_n: set[int] = set()
    holdout_positive: set[int] = set()
    for name, row in metrics.items():
        holdout_n.add(int(row["n"]))
        holdout_positive.add(int(row["positive"]))
        for metric in ("auroc", "average_precision_auprc", "f1", "brier_score"):
            require_finite(row[metric], f"{name}/{metric}")
    if len(holdout_n) != 1 or len(holdout_positive) != 1:
        raise ValueError("X14 architectures saw different holdout populations")

    inner_rows = read_csv(paths["inner"])
    if len(inner_rows) != 2981 or len({row["image_id"] for row in inner_rows}) != 2981:
        raise ValueError("X14 inner split population differs")
    roles_by_group: dict[str, set[str]] = defaultdict(set)
    for row in inner_rows:
        roles_by_group[row["group_id"]].add(row["inner_role"])
    if any(len(roles) != 1 for roles in roles_by_group.values()):
        raise ValueError("X14 inner split leaks groups")

    choices = read_csv(paths["choices"])
    if len(choices) != 371 * len(arms):
        raise ValueError("X14 choice row count differs")
    by_image: dict[str, set[str]] = defaultdict(set)
    tumor_by_image: dict[str, int] = {}
    for row in choices:
        image_id, arm = row["image_id"], row["arm"]
        if arm not in arms or arm in by_image[image_id]:
            raise ValueError("X14 choice matrix has duplicate or unknown arm")
        by_image[image_id].add(arm)
        tumor_by_image.setdefault(image_id, int(row["tumor"]))
        if tumor_by_image[image_id] != int(row["tumor"]):
            raise ValueError("X14 tumor label differs across arms")
        if int(row["selected_candidate_index"]) < 0:
            raise ValueError("X14 selected an invalid candidate")
        require_finite(row["selected_upstream_score"], "selected upstream")
        require_finite(row["selected_g1_logit"], "selected selector logit")
    if (
        len(by_image) != 371
        or sum(tumor_by_image.values()) != 184
        or any(image_arms != arms for image_arms in by_image.values())
    ):
        raise ValueError("X14 choice matrix is incomplete")

    manifest = read_json(paths["manifest"])
    if (
        manifest.get("study") != freeze["study"]
        or manifest.get("source_commit") != expected_source_commit
        or manifest.get("protocol_sha256") != expected_protocol_sha256
        or manifest.get("split_sha256") != expected_split_sha256
        or manifest.get("choice_freeze_sha256") != sha256(paths["freeze"])
        or manifest.get("training", {}).get("seeds") != list(SEEDS)
        or int(manifest.get("training", {}).get("epochs", -1)) != 16
        or manifest.get("validation_gt_read") is not False
        or int(manifest.get("test_images_read", -1)) != 0
        or manifest.get("test_evaluated") is not False
    ):
        raise ValueError("X14 run manifest differs")

    return {
        "pass": True,
        "images": 371,
        "tumor_images": 184,
        "arms": len(arms),
        "models": len(models),
        "selection_rows": len(choices),
        "inner_train_images": sum(row["inner_role"] == "inner_train" for row in inner_rows),
        "inner_holdout_images": sum(row["inner_role"] == "inner_holdout" for row in inner_rows),
        "choice_freeze_sha256": sha256(paths["freeze"]),
        "validation_gt_read": False,
        "test_images_read": 0,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.output_root,
        expected_source_commit=args.expected_source_commit,
        expected_protocol_sha256=args.expected_protocol_sha256,
        expected_split_sha256=args.expected_split_sha256,
        expected_candidate_manifest_sha256=args.expected_candidate_manifest_sha256,
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
