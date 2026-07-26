from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUTPUT = WORK / "btxrd_prompt_source_graph_selector_val_v1"
SOURCE_REPOSITORY = "https://github.com/itsthang333/Thesis.git"
SOURCE_COMMIT = "e8233ae48f62a526647ea5dba13a482d50f2c111"
IMPLEMENTATION_COMMIT = "abdcaca482d676b45a902ad3d927c832877a39d4"
EXPECTED_PROTOCOL_SHA256 = "f6fbc130ebd353ac8ba59552dff87ceed04a70895d936bdae648afe45cf8c50e"
EXPECTED_GIT_SPLIT_SHA256 = "43662d5d7969ae2a5bc61c6a0de3e0c392debef19c98d809f7d9bdfd0abb2fa8"
EXPECTED_SPLIT_SHA256 = "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
EXPECTED_CLASSIFIER_SHA256 = "f62d3702541ec3e6571751ddda22dab4c723943397471d3897500da1620304c5"
EXPECTED_SAM_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
EXPECTED_SALIENCY_MANIFEST_SHA256 = "72360237a12802c06ea5da8cecde7dbb87d4fef7a9dbf358f09e080384267bf1"
EXPECTED_SALIENCY_METADATA_SHA256 = "06109965042b9c433126101f7e32c609c8069312af58b2d0e6693ad4e77ccc4b"
EXPECTED_SALIENCY_AUDIT_SHA256 = "9afafa07681a83f7b31b08c9cceb79635d2dd58af53c7a8a9561f7ae01de21cf"
EXPECTED_SALIENCY_SOURCE_COMMIT = "95fc1c24ce8387c3ef211b4a0b71f6275f4e8b68"
EXPECTED_BIOMED_WEIGHT_SHA256 = "52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be"
EXPECTED_BASELINE_PER_IMAGE_SHA256 = "fe5cf247cd236799de9e279db342314c11ff65fdb065cda26986c302efd05540"
EXPECTED_BASELINE_PROMPT_SHA256 = "d1b570ae3a6287fdaf7fc5c28aea864d6883e5c57037542b39b17c4c6ea995e4"
EXPECTED_BASELINE_PSEUDO_MANIFEST_SHA256 = "1c7f5fe96c51fc86b984d4e86a5300655784461aeecc15178435deaa85943f1c"
SEGMENT_ANYTHING_COMMIT = "6fdee8f2727f4506cfbbe553e23b895e27956588"
EXPECTED_SOURCE_SHA256 = {
    "generate_pseudo_masks.py": "c958a38f76ae1f55c85fb62951a8442028c5ac011c839998ea65bbd2416be185",
    "pseudo/mask_selection.py": "21f4897691c71190a5c3d81f17de431c658d6421baec5ddfa20a2c581a965126",
    "evaluate_saved_candidate_diagnostics.py": "a746da95b0e8bb6e8afde24d8ddee95d0b04692a4ee725aa4c57d0e627173f6a",
    "pseudo/candidate_diagnostics.py": "e65c7402f9934e4a3cea918f4d5c707a759f1f78ef66a5a793c2c60ec51e6cc2",
    "tools/compare_candidate_diagnostics.py": "ef5e07b7cc8e5495aa22f8b2de877d83040bd89a908d1c609876de205a6c2ad8",
    "evaluate_pseudo_masks.py": "acae150b0932ab54290f93d98849d63e71de1c996eb8d12fe305ee5f0672db2d",
    "pseudo/manifest.py": "48467c99dd7257722ea08f1012fa723ce4bd19221965a453bbee682df0f7e404",
    "../tests/test_prompt_source_graph_selector.py": "8b36c0f57d9c7ffda745c38ee30cdac47803abf401d2148ac14387d0ed2fa027",
    "../tests/test_external_saliency_proposal_gallery.py": "94dda50ccd5155451a4c803fc28fdb939796fedb4b4ad3db1731068331491dfe",
}
EXPECTED_COUNTS = {"images": 371, "tumor": 184, "normal": 187}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], cwd: Path, env: dict[str, str], log: Path) -> None:
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def checkout_source() -> tuple[Path, Path, Path]:
    repository = WORK / "thesis_source"
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", SOURCE_REPOSITORY, str(repository)],
        check=True,
    )
    subprocess.run(["git", "-C", str(repository), "checkout", "--detach", SOURCE_COMMIT], check=True)
    actual = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != SOURCE_COMMIT:
        raise RuntimeError(f"Source commit mismatch: {actual}")
    project = repository / "project"
    for relative, expected in EXPECTED_SOURCE_SHA256.items():
        if sha256(project / relative) != expected:
            raise RuntimeError(f"Source SHA-256 mismatch: {relative}")
    protocol = (
        repository
        / "artifacts"
        / "research_protocols"
        / "prompt_source_graph_selector_val_v1.json"
    )
    if sha256(protocol) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Predeclared downstream protocol SHA-256 mismatch")
    parsed = json.loads(protocol.read_text(encoding="utf-8"))
    if (
        parsed.get("status")
        != "predeclared_before_any_prompt_source_graph_prediction"
        or parsed.get("source_lock", {}).get("implementation_commit") != IMPLEMENTATION_COMMIT
        or parsed.get("test_evaluated") is not False
    ):
        raise RuntimeError("Downstream protocol contract mismatch")
    git_split = repository / "artifacts" / "data_audit" / "split_manifest.csv"
    if sha256(git_split) != EXPECTED_GIT_SPLIT_SHA256:
        raise RuntimeError("Canonical Git split mismatch")
    split = WORK / "frozen_split_manifest.csv"
    split.write_bytes(git_split.read_bytes().replace(b"\n", b"\r\n"))
    if sha256(split) != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("Reconstructed frozen split mismatch")
    return project.resolve(), split.resolve(), protocol.resolve()


def find_btxrd_root() -> Path:
    for candidate in [INPUT / "btxrd-raw" / "BTXRD", *sorted(INPUT.glob("**/BTXRD"))]:
        if (candidate / "images").is_dir() and (candidate / "Annotations").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("BTXRD root not found")


def find_unique(name: str, expected_hash: str) -> Path:
    matches = [
        path.resolve()
        for path in INPUT.rglob(name)
        if path.is_file() and sha256(path) == expected_hash
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} with hash {expected_hash}, got {matches}")
    return matches[0]


def audit_split(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("split") == "val" and row.get("eligible") == "1"
        ]
    result = {
        "images": len(rows),
        "tumor": sum(int(row["tumor"]) for row in rows),
        "normal": sum(1 - int(row["tumor"]) for row in rows),
    }
    if result != EXPECTED_COUNTS or len({row["image_id"] for row in rows}) != 371:
        raise RuntimeError(f"Frozen validation cohort mismatch: {result}")
    return result


def audit_saliency_input() -> tuple[Path, Path, Path, dict[str, object]]:
    manifest = find_unique("saliency_manifest.csv", EXPECTED_SALIENCY_MANIFEST_SHA256)
    metadata = find_unique("run_metadata.json", EXPECTED_SALIENCY_METADATA_SHA256)
    audit = find_unique("independent_saliency_audit.json", EXPECTED_SALIENCY_AUDIT_SHA256)
    if manifest.parent != metadata.parent:
        raise RuntimeError("Saliency manifest/metadata roots differ")
    parsed_metadata = json.loads(metadata.read_text(encoding="utf-8"))
    parsed_audit = json.loads(audit.read_text(encoding="utf-8"))
    if (
        parsed_metadata.get("validation_gt_read") is not False
        or parsed_metadata.get("test_evaluated") is not False
        or parsed_audit.get("status") != "PASS"
        or parsed_audit.get("validation_gt_read") is not False
        or parsed_audit.get("test_evaluated") is not False
    ):
        raise RuntimeError("Saliency no-GT/no-test audit failed")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 371 or len({row["image_id"] for row in rows}) != 371:
        raise RuntimeError("Saliency map cohort mismatch")
    tumors = sum(int(row["tumor_image_label"]) for row in rows)
    if tumors != 184:
        raise RuntimeError("Saliency tumor cohort mismatch")
    for row in rows:
        map_path = (manifest.parent / row["map_path"]).resolve()
        try:
            map_path.relative_to(manifest.parent.resolve())
        except ValueError as error:
            raise RuntimeError("Saliency map escapes frozen root") from error
        if not map_path.is_file() or sha256(map_path) != row["map_sha256"]:
            raise RuntimeError(f"Physical saliency map hash mismatch: {row['image_id']}")
    evidence = {
        "manifest_sha256": sha256(manifest),
        "metadata_sha256": sha256(metadata),
        "audit_sha256": sha256(audit),
        "rows": len(rows),
        "tumor": tumors,
        "normal": len(rows) - tumors,
        "physical_map_hashes_verified": len(rows),
        "validation_gt_read": False,
        "test_evaluated": False,
    }
    return manifest, metadata, audit, evidence


def generation_command(
    project: Path,
    data: Path,
    split: Path,
    classifier: Path,
    sam: Path,
    saliency_manifest: Path,
    saliency_metadata: Path,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(project / "generate_pseudo_masks.py"),
        "--pipeline-profile", "default",
        "--data-root", str(data),
        "--split", "val",
        "--split-manifest", str(split),
        "--classifier-checkpoint", str(classifier),
        "--sam-checkpoint", str(sam),
        "--external-saliency-manifest", str(saliency_manifest),
        "--external-saliency-run-metadata", str(saliency_metadata),
        "--external-saliency-expected-manifest-sha256", EXPECTED_SALIENCY_MANIFEST_SHA256,
        "--external-saliency-expected-metadata-sha256", EXPECTED_SALIENCY_METADATA_SHA256,
        "--external-saliency-expected-source-commit", EXPECTED_SALIENCY_SOURCE_COMMIT,
        "--external-saliency-expected-model-weight-sha256", EXPECTED_BIOMED_WEIGHT_SHA256,
        "--external-saliency-role", "proposal_gallery",
        "--classifier-device", "cuda",
        "--sam-device", "cuda",
        "--target-columns", "tumor",
        "--image-size", "320",
        "--sam-image-size", "512",
        "--batch-size", "1",
        "--num-workers", "2",
        "--output-dir", str(output),
        "--process-all",
        "--save-visuals-limit", "0",
        "--confidence-threshold", "0.5",
        "--cam-tta-flip",
        "--cam-percentile", "90",
        "--cam-percentile-ensemble",
        "--cam-percentile-values", "85,90,95",
        "--max-points", "5",
        "--min-component-area", "100",
        "--mask-score-threshold", "0.4",
        "--seed-percentile", "82",
        "--support-percentile", "55",
        "--morphology-fusion-mode", "components",
        "--sam-prompt-mode", "box_point",
        "--sam-prompt-ensemble",
        "--max-components", "3",
        "--all-cam-components",
        "--points-per-component", "5",
        "--bbox-padding-ratio", "0.02",
        "--negative-points-per-component", "4",
        "--prompt-border-margin", "2",
        "--max-box-area-ratio", "0.35",
        "--selection-method", "prompt_source_graph",
        "--fusion-topk", "1",
        "--component-topk", "3",
        "--support-clip-kernel", "5",
        "--closing-kernel", "0",
        "--opening-kernel", "0",
        "--min-size", "40",
        "--max-hole-area", "0",
        "--guidance-threshold", "0.4",
        "--preprocessing-mode", "none",
        "--low-score-policy", "empty",
        "--cam-target-class", "ground_truth",
        "--save-candidate-diagnostics",
    ]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    provenance = OUTPUT / "provenance"
    provenance.mkdir()
    shutil.copy2(Path(__file__), provenance / "wrapper.py")
    started = datetime.now(timezone.utc)
    clock = time.monotonic()
    log = OUTPUT / "kernel.log"
    project, split, protocol = checkout_source()
    split_audit = audit_split(split)
    data = find_btxrd_root()
    classifier = find_unique("best_classifier.pt", EXPECTED_CLASSIFIER_SHA256)
    sam = find_unique("sam_vit_b_01ec64.pth", EXPECTED_SAM_SHA256)
    baseline_per = find_unique("per_image.csv", EXPECTED_BASELINE_PER_IMAGE_SHA256)
    baseline_prompt = find_unique("prompt_quality.csv", EXPECTED_BASELINE_PROMPT_SHA256)
    baseline_pseudo_manifest = find_unique(
        "pseudo_mask_manifest.csv", EXPECTED_BASELINE_PSEUDO_MANIFEST_SHA256
    )
    saliency_manifest, saliency_metadata, saliency_audit, saliency_evidence = (
        audit_saliency_input()
    )

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(project),
            "PYTHONUNBUFFERED": "1",
            "BTXRD_DISABLE_TQDM": "1",
        }
    )
    if not __import__("torch").cuda.is_available():
        raise RuntimeError("Downstream validation requires a Kaggle GPU")
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--no-deps",
            f"git+https://github.com/facebookresearch/segment-anything.git@{SEGMENT_ANYTHING_COMMIT}",
        ],
        project.parent,
        env,
        log,
    )
    run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_external_saliency_proposal_gallery.py",
        ],
        project.parent,
        env,
        log,
    )
    run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_candidate_diagnostics.py",
        ],
        project.parent,
        env,
        log,
    )
    run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_prompt_source_graph_selector.py",
        ],
        project.parent,
        env,
        log,
    )

    pseudo = OUTPUT / "pseudo_masks"
    diagnostic_evaluation = OUTPUT / "diagnostic_evaluation"
    final_evaluation = OUTPUT / "final_evaluation"
    diagnostic_evaluation.mkdir(parents=True)
    final_evaluation.mkdir(parents=True)
    generate = generation_command(
        project,
        data,
        split,
        classifier,
        sam,
        saliency_manifest,
        saliency_metadata,
        pseudo,
    )
    run(generate, project, env, log)

    run_metadata = json.loads((pseudo / "run_metadata.json").read_text())
    parameters = run_metadata
    if (
        parameters.get("external_saliency_role") != "proposal_gallery"
        or parameters.get("external_saliency_semantics")
        != (
            "append_component_sam_proposals; prompt-stable "
            "source-consensus graph selector; layercam support "
            "and post-processing unchanged"
        )
        or parameters.get("cam_tta_flip") is not True
        or parameters.get("selection_method") != "prompt_source_graph"
        or parameters.get("support_clip_kernel") != 5
    ):
        raise RuntimeError("Prompt/source graph generation contract mismatch")
    with baseline_pseudo_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        baseline_rows = {
            row["image_name"]: row for row in csv.DictReader(handle)
            if row.get("true_tumor") == "1"
        }
    with (pseudo / "pseudo_mask_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        candidate_rows = {
            row["image_name"]: row for row in csv.DictReader(handle)
            if row.get("true_tumor") == "1"
        }
    if set(baseline_rows) != set(candidate_rows) or len(candidate_rows) != 184:
        raise RuntimeError("Proposal-gallery tumor cohort differs from baseline")
    images_with_external_components = 0
    for image_name, candidate_row in candidate_rows.items():
        baseline_row = baseline_rows[image_name]
        if int(candidate_row["cam_morphology_components"]) != int(
            baseline_row["morphology_components"]
        ):
            raise RuntimeError(f"LayerCAM component gallery drift: {image_name}")
        external_count = int(candidate_row["external_saliency_components"])
        if int(candidate_row["morphology_components"]) != (
            int(candidate_row["cam_morphology_components"]) + external_count
        ):
            raise RuntimeError(f"Proposal source accounting mismatch: {image_name}")
        if int(candidate_row["sam_candidate_count"]) < int(
            baseline_row["sam_candidate_count"]
        ):
            raise RuntimeError(f"Baseline SAM gallery was not preserved: {image_name}")
        images_with_external_components += int(external_count > 0)
    if images_with_external_components == 0:
        raise RuntimeError("External saliency appended no proposal components")

    pseudo_summary = json.loads((pseudo / "pseudo_mask_summary.json").read_text())
    candidate_summary = json.loads(
        (pseudo / "candidate_diagnostics_summary.json").read_text()
    )
    if candidate_summary.get("selection_method") != "prompt_source_graph":
        raise RuntimeError("Candidate summary selector mismatch")
    with (pseudo / "candidate_diagnostics_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        diagnostic_rows = list(csv.DictReader(handle))
    if len(diagnostic_rows) != 184:
        raise RuntimeError("Schema-v2 diagnostic cohort is incomplete")
    source_label_counts = {"layercam": 0, "external_saliency": 0}
    prompt_label_counts = {"box_point": 0, "point": 0, "box": 0}
    for row in diagnostic_rows:
        diagnostic_path = pseudo / row["diagnostic_path"]
        if not diagnostic_path.is_file() or sha256(diagnostic_path) != row["diagnostic_sha256"]:
            raise RuntimeError(f"Diagnostic physical hash mismatch: {row['image_name']}")
        with np.load(diagnostic_path, allow_pickle=False) as payload:
            if int(payload["schema_version"][0]) != 2:
                raise RuntimeError(f"Diagnostic schema is not v2: {row['image_name']}")
            candidate_count = int(payload["sam_masks"].shape[0])
            if (
                len(payload["proposal_source_ids"]) != candidate_count
                or len(payload["prompt_modes"]) != candidate_count
                or len(payload["component_ids"]) != candidate_count
                or int(row["candidate_count"]) != candidate_count
            ):
                raise RuntimeError(
                    f"Diagnostic candidate provenance is unaligned: {row['image_name']}"
                )
            source_values = set(str(value) for value in payload["proposal_source_ids"])
            if not {"layercam", "external_saliency"}.issubset(source_values):
                raise RuntimeError(
                    f"Both proposal sources are not represented: {row['image_name']}"
                )
            unexpected_sources = source_values - {"layercam", "external_saliency"}
            if unexpected_sources:
                raise RuntimeError(
                    f"Unexpected proposal source labels: {unexpected_sources}"
                )
            prompt_values = set(str(value) for value in payload["prompt_modes"])
            if not {"box_point", "point"}.issubset(prompt_values):
                raise RuntimeError(
                    f"Required prompt fallback is absent: {row['image_name']}"
                )
            unexpected_prompts = prompt_values - {"box_point", "point", "box"}
            if unexpected_prompts:
                raise RuntimeError(f"Unexpected prompt labels: {unexpected_prompts}")
            for source in source_values:
                source_label_counts[source] += 1
            for prompt in prompt_values:
                prompt_label_counts[prompt] += 1
    pseudo_hash = str(pseudo_summary["manifest_sha256"])
    candidate_hash = str(candidate_summary["manifest_sha256"])
    freeze = {
        "status": "FROZEN_BEFORE_GT",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "pseudo_manifest_sha256": pseudo_hash,
        "candidate_manifest_sha256": candidate_hash,
        "pseudo_rows": pseudo_summary["manifest_rows"],
        "candidate_tumor_rows": candidate_summary["manifest_rows"],
        "candidate_diagnostic_schema": 2,
        "proposal_source_label_image_counts": source_label_counts,
        "prompt_label_image_counts": prompt_label_counts,
        "validation_gt_read": False,
        "test_evaluated": False,
    }
    freeze_path = OUTPUT / "prediction_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")

    evaluate_diagnostics = [
        sys.executable,
        str(project / "evaluate_saved_candidate_diagnostics.py"),
        "--data-root", str(data),
        "--split", "val",
        "--split-manifest", str(split),
        "--pseudo-output-dir", str(pseudo),
        "--expected-pseudo-manifest-sha256", pseudo_hash,
        "--expected-candidate-manifest-sha256", candidate_hash,
        "--output-dir", str(diagnostic_evaluation),
    ]
    run(evaluate_diagnostics, project, env, log)
    evaluate_final = [
        sys.executable,
        str(project / "evaluate_pseudo_masks.py"),
        "--data-root", str(data),
        "--split", "val",
        "--split-manifest", str(split),
        "--pred-mask-root", str(pseudo / "masks"),
        "--image-size", "320",
        "--batch-size", "8",
        "--num-workers", "2",
        "--output-csv", str(final_evaluation / "per_image.csv"),
        "--output-json", str(final_evaluation / "summary.json"),
        "--bootstrap-iterations", "2000",
        "--bootstrap-seed", "42",
    ]
    run(evaluate_final, project, env, log)

    candidate_per_hash = sha256(final_evaluation / "per_image.csv")
    candidate_prompt_hash = sha256(diagnostic_evaluation / "prompt_quality.csv")
    comparison_path = OUTPUT / "comparison.json"
    compare = [
        sys.executable,
        str(project / "tools" / "compare_candidate_diagnostics.py"),
        "--baseline-per-image", str(baseline_per),
        "--baseline-prompt-quality", str(baseline_prompt),
        "--candidate-per-image", str(final_evaluation / "per_image.csv"),
        "--candidate-prompt-quality", str(diagnostic_evaluation / "prompt_quality.csv"),
        "--expected-baseline-per-image-sha256", EXPECTED_BASELINE_PER_IMAGE_SHA256,
        "--expected-baseline-prompt-quality-sha256", EXPECTED_BASELINE_PROMPT_SHA256,
        "--expected-candidate-per-image-sha256", candidate_per_hash,
        "--expected-candidate-prompt-quality-sha256", candidate_prompt_hash,
        "--iterations", "10000",
        "--seed", "42",
        "--output-json", str(comparison_path),
    ]
    run(compare, project, env, log)
    comparison = json.loads(comparison_path.read_text())
    final_report = comparison["metrics"]["final_dice"]
    direct_gate = (
        final_report["overall"]["ci95_low"] > 0.0
        and all(
            final_report["subgroups"][group]["mean_delta"] >= 0.0
            for group in ("small", "medium", "large")
        )
    )
    comparison["protocol_promotion_gates"] = {
        "direct_train_pseudo_mask_generation": "PASS" if direct_gate else "FAIL",
        "paired_unet_consumer_training": "AUTHORIZED" if direct_gate else "FORBIDDEN",
    }
    comparison["decision"] = "DIRECT_PROMOTE" if direct_gate else "REJECT_NO_CONSUMER"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )

    independent_audit = {
        "status": "PASS",
        "prediction_freeze_sha256": sha256(freeze_path),
        "prediction_artifacts_verified_before_gt_load": True,
        "pseudo_manifest_sha256": pseudo_hash,
        "candidate_manifest_sha256": candidate_hash,
        "candidate_prompt_quality_sha256": candidate_prompt_hash,
        "candidate_per_image_sha256": candidate_per_hash,
        "diagnostic_prediction_first_audit_sha256": sha256(
            diagnostic_evaluation / "prediction_first_audit.json"
        ),
        "complete_misses_included": True,
        "baseline_layercam_component_counts_preserved": True,
        "baseline_sam_candidate_counts_not_reduced": True,
        "tumor_images_with_external_components": images_with_external_components,
        "candidate_diagnostic_schema": 2,
        "proposal_source_label_image_counts": source_label_counts,
        "prompt_label_image_counts": prompt_label_counts,
        "prompt_source_provenance_verified_before_gt": True,
        "cohort": {"images": 371, "tumor": 184, "normal": 187},
        "subgroups": {"small": 94, "medium": 72, "large": 18},
        "validation_gt_read_only_after_prediction_freeze": True,
        "test_evaluated": False,
        "decision": comparison["decision"],
    }
    audit_path = OUTPUT / "independent_downstream_audit.json"
    audit_path.write_text(json.dumps(independent_audit, indent=2, sort_keys=True) + "\n")

    manifest = {
        "run_id": "btxrd_prompt_source_graph_selector_val_v1",
        "source_commit": SOURCE_COMMIT,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "source_hashes": EXPECTED_SOURCE_SHA256,
        "protocol_sha256": sha256(protocol),
        "wrapper_sha256": sha256(Path(__file__)),
        "split_audit": split_audit,
        "saliency_input_audit": saliency_evidence,
        "proposal_gallery_audit": {
            "baseline_layercam_component_counts_preserved": True,
            "baseline_sam_candidate_counts_not_reduced": True,
            "tumor_images_with_external_components": images_with_external_components,
            "candidate_diagnostic_schema": 2,
            "proposal_source_label_image_counts": source_label_counts,
            "prompt_label_image_counts": prompt_label_counts,
        },
        "classifier_sha256": sha256(classifier),
        "sam_sha256": sha256(sam),
        "commands": {
            "generate_without_gt": generate,
            "evaluate_frozen_candidates": evaluate_diagnostics,
            "evaluate_frozen_final_masks": evaluate_final,
            "paired_comparison": compare,
        },
        "artifact_hashes": {
            "prediction_freeze": sha256(freeze_path),
            "pseudo_manifest": pseudo_hash,
            "candidate_manifest": candidate_hash,
            "candidate_prompt_quality": candidate_prompt_hash,
            "candidate_per_image": candidate_per_hash,
            "comparison": sha256(comparison_path),
            "independent_audit": sha256(audit_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "gpu": __import__("torch").cuda.get_device_name(0),
        },
        "decision": comparison["decision"],
        "validation_gt_read_only_after_prediction_freeze": True,
        "test_evaluated": False,
        "elapsed_seconds": time.monotonic() - clock,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    for directory in (
        pseudo / "masks",
        pseudo / "overlays",
        pseudo / "debug",
        pseudo / "candidate_diagnostics",
    ):
        if directory.is_dir():
            shutil.rmtree(directory)
    shutil.rmtree(WORK / "thesis_source")
    split.unlink(missing_ok=True)
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
