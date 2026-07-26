from __future__ import annotations

"""Independent post-freeze audit for the affinity-guided proposal selector."""

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from datasets.btxrd import BTXRDSegmentationDataset
from pseudo.affinity_selector_input import (
    load_affinity_selector_contract,
    load_affinity_selector_map,
)


EXPECTED = {
    "wrapper_sha256": "7a736598a9dd9791dc3271c9176c64ab3cf7f5da59b7c399ae39c4c979ad1190",
    "checkout_commit": "ca9462f13588243c0e490c2f18564038e49fd857",
    "implementation_commit": "56c01f241bda4b80183918517999f7ddbb37fc55",
    "protocol_sha256": "07bd490d309a850daeba1d00590d36968360aa7a70bebe3d418feb7c44ffadf7",
    "split_sha256": "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c",
    "baseline_per_image_sha256": (
        "59bead9162ff90851087c67ecac7f1bc8d9133e7c6a8aebb2f3db6e6606d7b05"
    ),
    "baseline_prompt_sha256": (
        "200dbc4172dcd7e5bd7c2c0a23734725925e3153d73b52a2796c4f3fcda5ab9a"
    ),
    "affinity_manifest_sha256": (
        "c066744a2acf3df3a078f4b25a973ea21bf140adeb88f713c90de8886a53fc42"
    ),
    "affinity_package_sha256": (
        "f9fb008a19c9f0ed1cb5ffb3f04f4f35227562614ca1dd6655fdaff6373167e9"
    ),
    "affinity_freeze_sha256": (
        "ed8b323dfaddf8fc9b5f7061a49dbc74f67187a45fb55c73ebdb46f36f9ff4ad"
    ),
    "affinity_source_commit": "38b5bb4b9d7a846862443b442ff406f0ab41d3bd",
    "affinity_protocol_sha256": (
        "53e2bb82ef35862b6c3e20387edbe60776f9d1ba46da516b9d5116db3fa2e7cf"
    ),
    "affinity_checkpoint_sha256": (
        "c5f9278de813396628fffe8360f09c786f1b74750861c23192d568405535b0d3"
    ),
}
EXPECTED_COUNTS = {"images": 371, "tumor": 184, "normal": 187}
EXPECTED_SUBGROUPS = {"small": 94, "medium": 72, "large": 18}
RANK_PERCENTILES = [80, 85, 90, 95, 97, 99]
OPERATIONAL_GOALS = {
    "overall": 0.340240391925425,
    "small": 0.17895493248574226,
    "medium": 0.5124417783635557,
    "large": 0.4937033565801355,
}
PSEUDO_QUALITY_FLOORS = {
    "overall": 0.2887922448972941,
    "small": 0.14660840079958384,
    "medium": 0.43761542706270595,
    "large": 0.4360151465236896,
}
GALLERY_INVARIANT_PROMPT_FIELDS = (
    "foreground_iou",
    "foreground_recall",
    "foreground_precision",
    "point_hit_rate",
    "negative_rejection_rate",
    "box_recall",
    "box_precision",
    "oracle_best_single_dice",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline-per-image", type=Path, required=True)
    parser.add_argument("--baseline-prompt-quality", type=Path, required=True)
    parser.add_argument("--affinity-input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def index_rows(
    rows: list[dict[str, str]], key: str
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if not value or value in indexed:
            raise ValueError(f"Missing or duplicate {key}: {value!r}")
        indexed[value] = row
    return indexed


def assert_close(
    actual: float,
    expected: float,
    name: str,
    *,
    atol: float = 5.0e-12,
) -> None:
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError(f"{name}: non-finite value")
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol):
        raise ValueError(f"{name}: {actual} != {expected}")


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def subgroup(area: float) -> str:
    return "small" if area < 0.01 else ("medium" if area < 0.05 else "large")


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * float(np.logical_and(prediction, target).sum()) / denominator


def audit_static_wrapper(root: Path) -> dict[str, Any]:
    wrapper = root / "provenance" / "wrapper.py"
    if not wrapper.is_file() or sha256_file(wrapper) != EXPECTED["wrapper_sha256"]:
        raise ValueError("Runtime wrapper SHA-256 mismatch")
    source = wrapper.read_text(encoding="utf-8")
    ast.parse(source)
    if "__PROTOCOL_" in source or "__AFFINITY_" in source:
        raise ValueError("Runtime wrapper contains an unresolved placeholder")
    order = {
        "generate": source.index("run(generate, project, env, log)"),
        "freeze": source.index("freeze_path.write_text"),
        "evaluate_diagnostics": source.index(
            "run(evaluate_diagnostics, project, env, log)"
        ),
        "evaluate_final": source.index("run(evaluate_final, project, env, log)"),
        "baseline_gt_artifacts": source.index("baseline_per = find_unique"),
        "compare": source.index("run(compare, project, env, log)"),
    }
    if not (
        order["generate"]
        < order["freeze"]
        < order["evaluate_diagnostics"]
        < order["evaluate_final"]
        < order["baseline_gt_artifacts"]
        < order["compare"]
    ):
        raise ValueError(f"Runtime wrapper prediction/GT order drift: {order}")
    cleanup = source[source.index("for directory in (") :]
    if 'pseudo / "masks"' in cleanup:
        raise ValueError("Runtime wrapper deletes final masks needed for independent audit")
    return {
        "wrapper_sha256": EXPECTED["wrapper_sha256"],
        "ast_parse": "PASS",
        "prediction_freeze_before_gt": "PASS",
        "selected_masks_preserved": "PASS",
        "order_offsets": order,
    }


def audit_protocol(protocol: Path) -> dict[str, Any]:
    if sha256_file(protocol) != EXPECTED["protocol_sha256"]:
        raise ValueError("Protocol SHA-256 mismatch")
    parsed = read_json(protocol)
    if (
        parsed.get("status")
        != "predeclared_before_any_affinity_guided_selector_prediction"
        or parsed.get("source_lock", {}).get("implementation_commit")
        != EXPECTED["implementation_commit"]
        or parsed.get("consumer_trained") is not False
        or parsed.get("test_evaluated") is not False
    ):
        raise ValueError("Predeclared protocol contract mismatch")
    return {
        "protocol_sha256": EXPECTED["protocol_sha256"],
        "status": parsed["status"],
        "consumer_trained": False,
        "test_evaluated": False,
    }


def audit_affinity_input(root: Path) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    manifest = root / "prediction_manifest.csv"
    package = root / "selector_input_manifest.json"
    freeze = root / "prediction_freeze.json"
    indexed, contract = load_affinity_selector_contract(
        manifest_path=manifest,
        package_metadata_path=package,
        prediction_freeze_path=freeze,
        expected_manifest_sha256=EXPECTED["affinity_manifest_sha256"],
        expected_package_metadata_sha256=EXPECTED["affinity_package_sha256"],
        expected_prediction_freeze_sha256=EXPECTED["affinity_freeze_sha256"],
        expected_source_commit=EXPECTED["affinity_source_commit"],
        expected_protocol_sha256=EXPECTED["affinity_protocol_sha256"],
        expected_checkpoint_sha256=EXPECTED["affinity_checkpoint_sha256"],
        split="val",
        split_manifest_sha256=EXPECTED["split_sha256"],
        image_size=320,
    )
    total_bytes = 0
    for image_id, row in indexed.items():
        load_affinity_selector_map(
            row,
            root=root,
            expected_image_id=image_id,
            expected_group_id=row["group_id"],
            expected_image_label=int(row["tumor"]),
            image_size=320,
        )
        total_bytes += (root / row["map_path"]).stat().st_size
    if (
        len(indexed) != 371
        or sum(int(row["tumor"]) for row in indexed.values()) != 184
        or total_bytes != 76028288
    ):
        raise ValueError("Affinity selector physical input cohort/bytes mismatch")
    return indexed, {
        **(contract or {}),
        "physical_maps_verified": len(indexed),
        "map_bytes": total_bytes,
    }


def audit_frozen_output_before_gt(
    root: Path,
    affinity_rows: dict[str, dict[str, str]],
) -> tuple[dict[str, Path], dict[str, Any]]:
    paths = {
        "run_manifest": root / "run_manifest.json",
        "cloud_audit": root / "independent_downstream_audit.json",
        "comparison": root / "comparison.json",
        "freeze": root / "prediction_freeze.json",
        "pseudo_manifest": root / "pseudo_masks" / "pseudo_mask_manifest.csv",
        "pseudo_summary": root / "pseudo_masks" / "pseudo_mask_summary.json",
        "run_metadata": root / "pseudo_masks" / "run_metadata.json",
        "candidate_manifest": (
            root / "pseudo_masks" / "candidate_diagnostics_manifest.csv"
        ),
        "candidate_summary": (
            root / "pseudo_masks" / "candidate_diagnostics_summary.json"
        ),
        "candidate_per_image": root / "final_evaluation" / "per_image.csv",
        "candidate_summary_metrics": root / "final_evaluation" / "summary.json",
        "prompt_quality": root / "diagnostic_evaluation" / "prompt_quality.csv",
        "prediction_first_audit": (
            root / "diagnostic_evaluation" / "prediction_first_audit.json"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Selector output is incomplete: {missing}")

    freeze = read_json(paths["freeze"])
    pseudo_summary = read_json(paths["pseudo_summary"])
    candidate_summary = read_json(paths["candidate_summary"])
    metadata = read_json(paths["run_metadata"])
    manifest = read_json(paths["run_manifest"])
    cloud_audit = read_json(paths["cloud_audit"])
    pseudo_hash = sha256_file(paths["pseudo_manifest"])
    candidate_hash = sha256_file(paths["candidate_manifest"])
    if (
        freeze.get("status") != "FROZEN_BEFORE_GT"
        or freeze.get("pseudo_manifest_sha256") != pseudo_hash
        or freeze.get("candidate_manifest_sha256") != candidate_hash
        or int(freeze.get("pseudo_rows", -1)) != 371
        or int(freeze.get("candidate_tumor_rows", -1)) != 184
        or freeze.get("validation_gt_read") is not False
        or freeze.get("test_evaluated") is not False
    ):
        raise ValueError("Prediction-freeze contract mismatch")
    if (
        pseudo_summary.get("complete") is not True
        or int(pseudo_summary.get("manifest_rows", -1)) != 371
        or pseudo_summary.get("manifest_sha256") != pseudo_hash
        or candidate_summary.get("complete") is not True
        or candidate_summary.get("prediction_first") is not True
        or int(candidate_summary.get("manifest_rows", -1)) != 184
        or candidate_summary.get("manifest_sha256") != candidate_hash
        or candidate_summary.get("pseudo_manifest_sha256") != pseudo_hash
        or candidate_summary.get("selection_method") != "affinity_rank_single"
        or int(candidate_summary.get("support_clip_kernel", 0)) != -1
    ):
        raise ValueError("Frozen pseudo/candidate manifest summaries mismatch")
    if (
        metadata.get("selection_method") != "affinity_rank_single"
        or metadata.get("best_per_component") is not False
        or int(metadata.get("fusion_topk", 0)) != 1
        or int(metadata.get("support_clip_kernel", 0)) != -1
        or int(metadata.get("closing_kernel", -1)) != 0
        or int(metadata.get("opening_kernel", -1)) != 0
        or int(metadata.get("min_size", -1)) != 1
        or int(metadata.get("max_hole_area", -1)) != 0
        or float(metadata.get("guidance_threshold", -1)) != 0.0
    ):
        raise ValueError("Frozen affinity-selector metadata drift")
    affinity_contract = metadata.get("affinity_selector", {})
    if (
        affinity_contract.get("manifest_sha256")
        != EXPECTED["affinity_manifest_sha256"]
        or affinity_contract.get("package_metadata_sha256")
        != EXPECTED["affinity_package_sha256"]
        or affinity_contract.get("prediction_freeze_sha256")
        != EXPECTED["affinity_freeze_sha256"]
        or affinity_contract.get("contains_validation_gt_derived_metrics")
        is not False
        or affinity_contract.get("test_evaluated") is not False
    ):
        raise ValueError("Frozen affinity-selector provenance drift")

    pseudo_rows = index_rows(read_csv(paths["pseudo_manifest"]), "image_name")
    diagnostic_rows = index_rows(read_csv(paths["candidate_manifest"]), "image_name")
    if set(pseudo_rows) != set(affinity_rows) or len(pseudo_rows) != 371:
        raise ValueError("Pseudo/affinity cohorts differ")
    tumor_names = {
        image_id for image_id, row in affinity_rows.items() if row["tumor"] == "1"
    }
    if set(diagnostic_rows) != tumor_names or len(diagnostic_rows) != 184:
        raise ValueError("Candidate diagnostics do not cover all tumors")

    mask_bytes = 0
    empty_masks = 0
    for image_name, row in pseudo_rows.items():
        affinity_hash = affinity_rows[image_name]["map_sha256"]
        if row.get("affinity_selector_map_sha256") != affinity_hash:
            raise ValueError(f"Pseudo affinity provenance mismatch: {image_name}")
        relative = Path(row["mask_path"])
        if relative != Path("masks") / f"{Path(image_name).stem}.png":
            raise ValueError(f"Unexpected final-mask path: {image_name}")
        mask_path = (root / "pseudo_masks" / relative).resolve()
        if not mask_path.is_file() or sha256_file(mask_path) != row["mask_sha256"]:
            raise ValueError(f"Final-mask hash mismatch: {image_name}")
        with Image.open(mask_path) as image:
            values = np.asarray(image.convert("L"))
        if values.shape != (320, 320) or not set(np.unique(values)).issubset({0, 255}):
            raise ValueError(f"Final-mask schema mismatch: {image_name}")
        foreground = int(np.count_nonzero(values))
        if foreground != int(row["mask_foreground_pixels"]):
            raise ValueError(f"Final-mask foreground mismatch: {image_name}")
        empty_masks += int(foreground == 0)
        mask_bytes += mask_path.stat().st_size
        if image_name in tumor_names:
            if (
                int(row["selected_candidates"]) != 1
                or int(row["affinity_supports"]) != len(RANK_PERCENTILES)
                or int(row["affinity_selected_percentile"])
                not in set(RANK_PERCENTILES)
            ):
                raise ValueError(f"Raw single-candidate contract mismatch: {image_name}")
            assert_close(
                float(row["selected_area_ratio"]),
                float(row["final_area_ratio"]),
                f"{image_name}/unchanged_boundary_area",
                atol=1.0e-12,
            )
    for image_name, row in diagnostic_rows.items():
        if (
            row.get("affinity_selector_map_sha256")
            != affinity_rows[image_name]["map_sha256"]
        ):
            raise ValueError(f"Diagnostic affinity provenance mismatch: {image_name}")

    artifact_hashes = manifest.get("artifact_hashes", {})
    expected_artifacts = {
        "prediction_freeze": paths["freeze"],
        "pseudo_manifest": paths["pseudo_manifest"],
        "candidate_manifest": paths["candidate_manifest"],
        "candidate_prompt_quality": paths["prompt_quality"],
        "candidate_per_image": paths["candidate_per_image"],
        "comparison": paths["comparison"],
        "independent_audit": paths["cloud_audit"],
    }
    for key, path in expected_artifacts.items():
        if artifact_hashes.get(key) != sha256_file(path):
            raise ValueError(f"Run-manifest artifact hash mismatch: {key}")
    if (
        manifest.get("source_commit") != EXPECTED["checkout_commit"]
        or manifest.get("implementation_commit") != EXPECTED["implementation_commit"]
        or manifest.get("protocol_sha256") != EXPECTED["protocol_sha256"]
        or manifest.get("wrapper_sha256") != EXPECTED["wrapper_sha256"]
        or manifest.get("validation_gt_read_only_after_prediction_freeze") is not True
        or manifest.get("consumer_trained") is not False
        or manifest.get("test_evaluated") is not False
        or cloud_audit.get("status") != "PASS"
        or cloud_audit.get("consumer_trained") is not False
        or cloud_audit.get("test_evaluated") is not False
    ):
        raise ValueError("Run/cloud audit provenance or supervision contract mismatch")

    return paths, {
        "prediction_freeze_sha256": sha256_file(paths["freeze"]),
        "pseudo_manifest_sha256": pseudo_hash,
        "candidate_manifest_sha256": candidate_hash,
        "physical_final_masks_verified": len(pseudo_rows),
        "physical_final_mask_bytes": mask_bytes,
        "empty_final_masks": empty_masks,
        "prediction_artifacts_verified_before_gt": True,
        "consumer_trained": False,
        "test_evaluated": False,
    }


def recompute_final_metrics_after_freeze(
    *,
    paths: dict[str, Path],
    dataset_root: Path,
    split_manifest: Path,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    # This function is deliberately called only after every physical output
    # mask and its manifest/freeze binding has passed the no-GT audit above.
    dataset = BTXRDSegmentationDataset(
        root=dataset_root,
        split="val",
        image_size=320,
        augment=False,
        split_manifest=split_manifest,
    )
    gt_by_name: dict[str, np.ndarray] = {}
    for index in range(len(dataset)):
        _image, target, name = dataset[index]
        gt_by_name[str(name)] = target[0].numpy() > 0.5

    pseudo_rows = index_rows(read_csv(paths["pseudo_manifest"]), "image_name")
    reported_rows = read_csv(paths["candidate_per_image"])
    reported = index_rows(reported_rows, "image_name")
    if set(gt_by_name) != set(pseudo_rows) or set(reported) != set(pseudo_rows):
        raise ValueError("GT, final-mask and reported cohorts differ")

    recomputed: list[dict[str, Any]] = []
    for image_name in sorted(pseudo_rows):
        row = pseudo_rows[image_name]
        target = gt_by_name[image_name]
        with Image.open(paths["pseudo_manifest"].parent / row["mask_path"]) as image:
            prediction = np.asarray(image.convert("L")) > 0
        area = float(target.mean())
        value = dice(prediction, target)
        item = {
            "image_name": image_name,
            "group_id": reported[image_name]["group_id"],
            "gt_positive": bool(target.any()),
            "gt_area_ratio": area,
            "size_group": subgroup(area) if target.any() else "normal",
            "dice": value,
            "pred_area_ratio": float(prediction.mean()),
        }
        claimed = reported[image_name]
        assert_close(value, float(claimed["dice"]), f"{image_name}/dice")
        assert_close(area, float(claimed["gt_area_ratio"]), f"{image_name}/gt_area")
        assert_close(
            item["pred_area_ratio"],
            float(claimed["pred_area_ratio"]),
            f"{image_name}/pred_area",
        )
        claimed_positive = str(claimed["gt_positive"]).casefold() in {"1", "true"}
        if claimed_positive != item["gt_positive"]:
            raise ValueError(f"Reported GT-positive drift: {image_name}")
        recomputed.append(item)

    tumors = [row for row in recomputed if row["gt_positive"]]
    counts = {
        name: sum(row["size_group"] == name for row in tumors)
        for name in ("small", "medium", "large")
    }
    if (
        len(recomputed) != 371
        or len(tumors) != 184
        or len(recomputed) - len(tumors) != 187
        or counts != EXPECTED_SUBGROUPS
    ):
        raise ValueError("Independent GT cohort/subgroup mismatch")
    means = {
        "overall": statistics.fmean(row["dice"] for row in tumors),
        **{
            name: statistics.fmean(
                row["dice"] for row in tumors if row["size_group"] == name
            )
            for name in ("small", "medium", "large")
        },
    }
    summary = read_json(paths["candidate_summary_metrics"])
    assert_close(means["overall"], float(summary["mean_tumor_dice"]), "summary/overall")
    return recomputed, means


def paired_report(
    pairs: list[dict[str, Any]],
    *,
    baseline_key: str,
    candidate_key: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    baseline_values: list[float] = []
    candidate_values: list[float] = []
    for row in pairs:
        baseline = float(row[baseline_key])
        candidate = float(row[candidate_key])
        grouped.setdefault(str(row["group_id"]), []).append(candidate - baseline)
        baseline_values.append(baseline)
        candidate_values.append(candidate)
    group_ids = sorted(grouped)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled: list[float] = []
        for _ in group_ids:
            sampled.extend(grouped[rng.choice(group_ids)])
        samples.append(statistics.fmean(sampled))
    baseline_mean = statistics.fmean(baseline_values)
    candidate_mean = statistics.fmean(candidate_values)
    return {
        "images": len(pairs),
        "groups": len(group_ids),
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "mean_delta": candidate_mean - baseline_mean,
        "ci95_low": percentile(samples, 0.025),
        "ci95_high": percentile(samples, 0.975),
        "iterations": iterations,
        "seed": seed,
    }


def audit_post_freeze_comparison(
    *,
    paths: dict[str, Path],
    recomputed: list[dict[str, Any]],
    baseline_per_image: Path,
    baseline_prompt_quality: Path,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if sha256_file(baseline_per_image) != EXPECTED["baseline_per_image_sha256"]:
        raise ValueError("Same-gallery baseline per-image SHA-256 mismatch")
    if sha256_file(baseline_prompt_quality) != EXPECTED["baseline_prompt_sha256"]:
        raise ValueError("Same-gallery baseline prompt-quality SHA-256 mismatch")
    baseline = index_rows(read_csv(baseline_per_image), "image_name")
    candidate = index_rows(read_csv(paths["candidate_per_image"]), "image_name")
    recomputed_index = {row["image_name"]: row for row in recomputed}
    baseline_prompt = index_rows(read_csv(baseline_prompt_quality), "image_name")
    candidate_prompt = index_rows(read_csv(paths["prompt_quality"]), "image_name")
    tumor_names = {name for name, row in recomputed_index.items() if row["gt_positive"]}
    if set(baseline) != set(candidate) or set(baseline_prompt) != tumor_names:
        raise ValueError("Same-gallery baseline cohort mismatch")
    if set(candidate_prompt) != tumor_names:
        raise ValueError("Candidate prompt diagnostics do not cover all tumors")

    max_gallery_metric_delta = 0.0
    pairs: list[dict[str, Any]] = []
    for name in sorted(tumor_names):
        base_row = baseline[name]
        candidate_row = candidate[name]
        if (
            base_row["group_id"] != candidate_row["group_id"]
            or candidate_row["group_id"] != recomputed_index[name]["group_id"]
        ):
            raise ValueError(f"Frozen group drift: {name}")
        assert_close(
            float(candidate_row["dice"]),
            float(recomputed_index[name]["dice"]),
            f"{name}/independent_final_dice",
        )
        for field in GALLERY_INVARIANT_PROMPT_FIELDS:
            delta = abs(
                float(candidate_prompt[name][field])
                - float(baseline_prompt[name][field])
            )
            max_gallery_metric_delta = max(max_gallery_metric_delta, delta)
        assert_close(
            float(candidate_prompt[name]["selected_dice"]),
            float(candidate_prompt[name]["final_dice"]),
            f"{name}/raw_selected_equals_final",
            atol=2.0e-7,
        )
        assert_close(
            float(candidate_prompt[name]["final_dice"]),
            float(candidate_row["dice"]),
            f"{name}/prompt_final_equals_metric",
            atol=2.0e-7,
        )
        pairs.append(
            {
                "image_name": name,
                "group_id": candidate_row["group_id"],
                "size_group": recomputed_index[name]["size_group"],
                "baseline_final": float(base_row["dice"]),
                "candidate_final": float(candidate_row["dice"]),
                "baseline_oracle": float(
                    baseline_prompt[name]["oracle_best_single_dice"]
                ),
                "candidate_oracle": float(
                    candidate_prompt[name]["oracle_best_single_dice"]
                ),
            }
        )
    if max_gallery_metric_delta > 2.0e-7:
        raise ValueError(
            "Regenerated proposal gallery differs from the frozen same-gallery baseline"
        )

    reports: dict[str, Any] = {}
    for metric, baseline_key, candidate_key in (
        ("final_dice", "baseline_final", "candidate_final"),
        ("oracle_best_single_dice", "baseline_oracle", "candidate_oracle"),
    ):
        reports[metric] = {
            "overall": paired_report(
                pairs,
                baseline_key=baseline_key,
                candidate_key=candidate_key,
                iterations=iterations,
                seed=seed,
            ),
            "subgroups": {
                name: paired_report(
                    [row for row in pairs if row["size_group"] == name],
                    baseline_key=baseline_key,
                    candidate_key=candidate_key,
                    iterations=iterations,
                    seed=seed,
                )
                for name in ("small", "medium", "large")
            },
        }

    cloud = read_json(paths["comparison"])
    for metric in reports:
        for name in ("overall", "small", "medium", "large"):
            actual = (
                reports[metric]["overall"]
                if name == "overall"
                else reports[metric]["subgroups"][name]
            )
            claimed = (
                cloud["metrics"][metric]["overall"]
                if name == "overall"
                else cloud["metrics"][metric]["subgroups"][name]
            )
            for field in (
                "baseline_mean",
                "candidate_mean",
                "mean_delta",
                "ci95_low",
                "ci95_high",
            ):
                assert_close(
                    float(actual[field]),
                    float(claimed[field]),
                    f"comparison/{metric}/{name}/{field}",
                )
            if actual["images"] != claimed["images"] or actual["groups"] != claimed["groups"]:
                raise ValueError(f"Comparison cohort drift: {metric}/{name}")

    oracle = reports["oracle_best_single_dice"]
    final = reports["final_dice"]
    oracle_gate = (
        oracle["overall"]["candidate_mean"] >= OPERATIONAL_GOALS["overall"]
        and all(
            oracle["subgroups"][name]["candidate_mean"] >= OPERATIONAL_GOALS[name]
            for name in ("small", "medium", "large")
        )
    )
    statistical_gate = (
        final["overall"]["ci95_low"] > 0.0
        and all(
            final["subgroups"][name]["mean_delta"] >= 0.0
            for name in ("small", "medium", "large")
        )
    )
    quality_gate = (
        final["overall"]["candidate_mean"] >= PSEUDO_QUALITY_FLOORS["overall"]
        and all(
            final["subgroups"][name]["candidate_mean"]
            >= PSEUDO_QUALITY_FLOORS[name]
            for name in ("small", "medium", "large")
        )
    )
    direct_gate = oracle_gate and statistical_gate and quality_gate
    decision = (
        "AUTHORIZE_SEPARATE_TRAIN_PSEUDO_MASK_PROTOCOL"
        if direct_gate
        else "RETAIN_SELECTOR_RESEARCH_ONLY"
        if oracle_gate or statistical_gate
        else "REJECT"
    )
    if cloud.get("decision") != decision:
        raise ValueError("Cloud and independent selector decisions differ")
    expected_gate_status = {
        "candidate_oracle_above_operational_goals": "PASS" if oracle_gate else "FAIL",
        "statistical_improvement": "PASS" if statistical_gate else "FAIL",
        "halfway_to_operational_goal_quality": "PASS" if quality_gate else "FAIL",
        "authorize_train_pseudo_mask_protocol": "PASS" if direct_gate else "FAIL",
    }
    for key, expected in expected_gate_status.items():
        if cloud.get("protocol_promotion_gates", {}).get(key) != expected:
            raise ValueError(f"Cloud gate status mismatch: {key}")
    return {
        "same_gallery_max_invariant_metric_abs_delta": max_gallery_metric_delta,
        "metrics": reports,
        "gates": expected_gate_status,
        "decision": decision,
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations != 10_000 or args.bootstrap_seed != 42:
        raise ValueError("This frozen audit requires 10000 replicates and seed 42")
    root = args.root.resolve()
    static = audit_static_wrapper(root)
    protocol = audit_protocol(args.protocol.resolve())
    if sha256_file(args.split_manifest.resolve()) != EXPECTED["split_sha256"]:
        raise ValueError("Frozen split SHA-256 mismatch")
    affinity_rows, affinity = audit_affinity_input(args.affinity_input_root.resolve())
    paths, frozen = audit_frozen_output_before_gt(root, affinity_rows)
    recomputed, means = recompute_final_metrics_after_freeze(
        paths=paths,
        dataset_root=args.dataset_root.resolve(),
        split_manifest=args.split_manifest.resolve(),
    )
    comparison = audit_post_freeze_comparison(
        paths=paths,
        recomputed=recomputed,
        baseline_per_image=args.baseline_per_image.resolve(),
        baseline_prompt_quality=args.baseline_prompt_quality.resolve(),
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    result = {
        "status": "PASS",
        "audit_role": "independent post-freeze affinity-guided selector audit",
        "static_wrapper": static,
        "protocol": protocol,
        "affinity_input": affinity,
        "frozen_output": frozen,
        "cohort": EXPECTED_COUNTS,
        "subgroups": EXPECTED_SUBGROUPS,
        "independent_final_dice": means,
        "comparison": comparison,
        "complete_misses_included": True,
        "validation_gt_read_only_after_prediction_freeze": True,
        "bootstrap": {
            "unit": "complete frozen validation group",
            "iterations": args.bootstrap_iterations,
            "seed": args.bootstrap_seed,
        },
        "consumer_trained": False,
        "test_evaluated": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
