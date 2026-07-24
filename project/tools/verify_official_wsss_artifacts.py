from __future__ import annotations

"""Verify the compact official WSSS thesis evidence without loading models."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    official = root / "artifacts" / "official_wsss"
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, **details: object) -> None:
        checks.append({"name": name, "passed": bool(condition), **details})

    selection = json_file(official / "SELECTION.json")
    split_path = root / "artifacts" / "data_audit" / "split_manifest.csv"
    split_rows = csv_rows(split_path)
    split_counts: dict[str, int] = {}
    for row in split_rows:
        if row.get("eligible") == "1":
            split = str(row["split"])
            split_counts[split] = split_counts.get(split, 0) + 1
    check(
        "split_manifest_sha256",
        sha256_file(split_path) == selection["split_manifest_sha256"],
        sha256=sha256_file(split_path),
    )
    check(
        "split_counts",
        split_counts == {"train": 2981, "val": 371, "test": 373},
        counts=split_counts,
    )
    check(
        "selection_is_wsss",
        str(selection.get("supervision", "")).startswith("binary image-level"),
    )
    check("test_remains_locked", selection.get("test_evaluated") is False)

    train_manifest = official / "pseudo_masks" / "train" / "pseudo_mask_manifest.csv"
    train_summary = json_file(
        official / "pseudo_masks" / "train" / "pseudo_mask_summary.json"
    )
    check(
        "train_pseudo_manifest_hash",
        sha256_file(train_manifest) == selection["train_pseudo_mask_manifest_sha256"],
        sha256=sha256_file(train_manifest),
    )
    check(
        "train_pseudo_manifest_complete",
        len(csv_rows(train_manifest)) == 2981
        and train_summary.get("complete") is True
        and train_summary.get("manifest_rows") == 2981,
    )
    check(
        "train_pseudo_status_accounting",
        sum(int(value) for value in dict(train_summary["statuses"]).values()) == 2981,
        statuses=train_summary["statuses"],
    )

    pseudo_summary = json_file(
        official / "pseudo_masks" / "validation" / "evaluation" / "summary.json"
    )
    check(
        "validation_pseudo_population",
        (
            pseudo_summary.get("images"),
            pseudo_summary.get("tumor_images"),
            pseudo_summary.get("normal_images"),
        )
        == (371, 184, 187),
    )
    check(
        "validation_pseudo_dice",
        abs(float(pseudo_summary["mean_tumor_dice"]) - 0.23433922219069755)
        <= 1e-12,
    )

    segmenter_summary = json_file(official / "segmenter" / "evaluation" / "summary.json")
    segmenter_rows = csv_rows(official / "segmenter" / "evaluation" / "per_image.csv")
    tumor_dice = [
        float(row["dice"]) for row in segmenter_rows if row.get("group") == "tumor"
    ]
    check(
        "segmenter_population",
        len(segmenter_rows) == 371 and len(tumor_dice) == 184,
    )
    check(
        "segmenter_per_image_reproduces_summary",
        abs(sum(tumor_dice) / len(tumor_dice) - float(segmenter_summary["mean_tumor_dice"]))
        <= 1e-12,
    )
    check(
        "segmenter_selection_matches_summary",
        abs(
            float(selection["evaluation"]["mean_tumor_dice"])
            - float(segmenter_summary["mean_tumor_dice"])
        )
        <= 1e-12
        and float(selection["evaluation"]["threshold"])
        == float(segmenter_summary["threshold"]),
    )
    bootstrap = json_file(official / "segmenter" / "evaluation" / "bootstrap.json")
    dice_interval = bootstrap["intervals"]["mean_tumor_dice"]
    check(
        "segmenter_bootstrap_interval",
        float(dice_interval["ci95_low"])
        <= float(segmenter_summary["mean_tumor_dice"])
        <= float(dice_interval["ci95_high"]),
        ci95=[dice_interval["ci95_low"], dice_interval["ci95_high"]],
    )

    pointer = json_file(official / "checkpoints" / "checkpoint_pointer.json")
    checkpoint_pointer = pointer["artifacts"]["official_wsss_segmenter"]
    check(
        "official_checkpoint_pointer",
        checkpoint_pointer["sha256"] == selection["checkpoint"]["sha256"]
        and checkpoint_pointer["bytes"] == selection["checkpoint"]["bytes"],
        sha256=checkpoint_pointer["sha256"],
        bytes=checkpoint_pointer["bytes"],
    )
    fs_pointer = json_file(
        root
        / "artifacts"
        / "diagnostics"
        / "fully_supervised_upper_bound"
        / "pointer.json"
    )
    check(
        "fully_supervised_is_not_official",
        fs_pointer.get("official_wsss") is False
        and "diagnostic" in str(fs_pointer.get("status", "")),
    )

    rejected_tokens = ("coverage_mass_sam_causal", "classifier_candidate_causal_scores")
    execution_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted((root / "project").rglob("*.py"))
        if path.resolve() != Path(__file__).resolve()
    )
    check(
        "rejected_causal_selector_absent_from_execution_paths",
        not any(token in execution_text for token in rejected_tokens),
    )

    failures = [item for item in checks if not item["passed"]]
    result = {
        "verifier": "official_wsss_artifacts_v1",
        "passed": not failures,
        "checks": len(checks),
        "failures": failures,
        "results": checks,
    }
    output = args.output or official / "audit_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": result["passed"],
        "checks": result["checks"],
        "failures": len(failures),
        "output": str(output),
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
