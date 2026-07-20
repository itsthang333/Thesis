from __future__ import annotations

"""Build an immutable, group-aware BTXRD split manifest without editing BTXRD.

BTXRD does not publish patient/lesion identifiers even though the dataset paper
states that several images can be different views of the same lesion.  This
tool therefore creates a *heuristic* case grouping from consecutive image IDs
whose stable clinical metadata match.  The grouping is deliberately labelled
as heuristic in every artifact; it must not be described as a verified patient
or lesion partition.

The tool also removes exact byte-identical duplicate images from the eligible
set, validates every image/annotation, and records any source-metadata repair.
All outputs are derived artifacts.  Source images, annotations and metadata are
never modified.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANATOMY_COLUMNS = (
    "hand",
    "ulna",
    "radius",
    "humerus",
    "foot",
    "tibia",
    "fibula",
    "femur",
    "hip bone",
    "ankle-joint",
    "knee-joint",
    "hip-joint",
    "wrist-joint",
    "elbow-joint",
    "shoulder-joint",
)
TUMOR_TYPE_COLUMNS = (
    "osteochondroma",
    "multiple osteochondromas",
    "simple bone cyst",
    "giant cell tumor",
    "osteofibroma",
    "synovial osteochondroma",
    "other bt",
    "osteosarcoma",
    "other mt",
)
TUMOR_TYPE_CLASS_NAMES = ("normal",) + TUMOR_TYPE_COLUMNS
VIEW_COLUMNS = ("frontal", "lateral", "oblique")
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIOS = (0.8, 0.1, 0.1)
BUILDER_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit BTXRD and build a read-only, group-aware split manifest"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "dataset_audit" / "btxrd_group_v1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS[0])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_RATIOS[1])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS[2])
    parser.add_argument(
        "--phash-distance",
        type=int,
        default=4,
        help="Maximum 64-bit pHash Hamming distance for diagnostic near-duplicate candidates",
    )
    parser.add_argument(
        "--skip-phash",
        action="store_true",
        help="Skip perceptual near-duplicate diagnostics (exact SHA-256 dedup still runs)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace an existing derived manifest; source BTXRD is never changed",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_text_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def git_state() -> dict[str, object]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        status = run("status", "--short")
        return {
            "branch": run("branch", "--show-current"),
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(status),
            "status_short": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _normalise_cell(value: object) -> object:
    if value is None:
        return ""
    try:
        if bool(np.isnan(value)):  # type: ignore[arg-type]
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_metadata(dataset_root: Path) -> tuple[list[dict[str, object]], Path]:
    csv_path = dataset_root / "dataset.csv"
    xlsx_path = dataset_root / "dataset.xlsx"
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)], csv_path
    if xlsx_path.is_file():
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise ImportError("Reading dataset.xlsx requires pandas and openpyxl") from exc
        frame = pd.read_excel(xlsx_path)
        rows = [
            {str(key): _normalise_cell(value) for key, value in row.items()}
            for row in frame.to_dict(orient="records")
        ]
        return rows, xlsx_path
    raise FileNotFoundError(f"No dataset.csv or dataset.xlsx under {dataset_root}")


def binary_flag(row: dict[str, object], column: str, image_id: str) -> int:
    if column not in row:
        raise ValueError(f"{image_id}: missing metadata column {column!r}")
    try:
        value = float(row[column])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{image_id}: {column!r} must be 0 or 1, got {row[column]!r}") from exc
    if not math.isfinite(value) or value not in (0.0, 1.0):
        raise ValueError(f"{image_id}: {column!r} must be 0 or 1, got {row[column]!r}")
    return int(value)


def annotation_labels(annotation_path: Path) -> set[str]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    return {
        str(shape.get("label", "")).strip().lower()
        for shape in payload.get("shapes", [])
        if shape.get("shape_type") == "polygon" and str(shape.get("label", "")).strip()
    }


def resolve_tumor_type(
    row: dict[str, object],
    image_id: str,
    annotation_path: Path,
) -> tuple[int, str]:
    matches = [
        index
        for index, column in enumerate(TUMOR_TYPE_COLUMNS, start=1)
        if binary_flag(row, column, image_id)
    ]
    tumor = binary_flag(row, "tumor", image_id)
    if not tumor:
        if matches:
            raise ValueError(
                f"{image_id}: tumor=0 but tumor-type flags are active: "
                f"{[TUMOR_TYPE_CLASS_NAMES[index] for index in matches]}"
            )
        return 0, ""
    if tumor and len(matches) == 1:
        return matches[0], ""
    if tumor and len(matches) > 1 and annotation_path.is_file():
        labels = annotation_labels(annotation_path)
        matching_labels = [
            index for index in matches if TUMOR_TYPE_CLASS_NAMES[index].lower() in labels
        ]
        if len(matching_labels) == 1:
            selected = matching_labels[0]
            original = ",".join(TUMOR_TYPE_CLASS_NAMES[index] for index in matches)
            correction = (
                f"source flags [{original}] resolved to {TUMOR_TYPE_CLASS_NAMES[selected]} "
                "from the matching LabelMe class label"
            )
            return selected, correction
    raise ValueError(
        f"{image_id}: tumor={tumor} but tumor-type flags resolve to "
        f"{[TUMOR_TYPE_CLASS_NAMES[index] for index in matches]}"
    )


def single_flag_name(
    row: dict[str, object], columns: Sequence[str], image_id: str, kind: str
) -> str:
    matches = [column for column in columns if binary_flag(row, column, image_id)]
    if len(matches) != 1:
        raise ValueError(f"{image_id}: expected exactly one {kind}, got {matches}")
    return matches[0]


def multi_flag_names(
    row: dict[str, object], columns: Sequence[str], image_id: str, kind: str
) -> str:
    """Return all active metadata flags in stable source-column order.

    BTXRD's anatomy fields are not mutually exclusive: a radiograph can cover
    multiple bones or a bone and its joint.  Normal images intentionally have
    no anatomy flag.  Treating these columns as a single categorical field
    would reject valid rows and would make the audit disagree with the source
    metadata semantics.
    """
    matches = [column for column in columns if binary_flag(row, column, image_id)]
    return "|".join(matches)


def perceptual_hash(path: Path) -> int:
    """Return a 64-bit pHash implemented with a deterministic 2-D DCT."""
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
            dtype=np.float64,
        )
    n = 32
    positions = np.arange(n, dtype=np.float64)
    frequencies = positions[:, None]
    transform = np.cos((math.pi / n) * (positions + 0.5) * frequencies)
    transform[0] *= 1.0 / math.sqrt(2.0)
    transform *= math.sqrt(2.0 / n)
    low = (transform @ array @ transform.T)[:8, :8]
    threshold = float(np.median(low.flat[1:]))
    value = 0
    for coefficient in low.flat:
        value = (value << 1) | int(coefficient > threshold)
    return value


def audit_source_rows(
    rows: list[dict[str, object]],
    dataset_root: Path,
    compute_phash: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "Annotations"
    seen_ids: set[str] = set()
    audited: list[dict[str, object]] = []
    unreadable_images: list[dict[str, str]] = []
    invalid_annotations: list[dict[str, str]] = []

    for row_number, row in enumerate(rows, start=2):
        image_id = str(row.get("image_id", "")).strip()
        if not image_id:
            raise ValueError(f"metadata row {row_number} has no image_id")
        if image_id in seen_ids:
            raise ValueError(f"metadata contains duplicate image_id {image_id!r}")
        seen_ids.add(image_id)
        image_path = images_dir / image_id
        if not image_path.is_file():
            raise FileNotFoundError(f"metadata references missing image: {image_path}")
        annotation_path = annotations_dir / f"{Path(image_id).stem}.json"

        tumor = binary_flag(row, "tumor", image_id)
        benign = binary_flag(row, "benign", image_id)
        malignant = binary_flag(row, "malignant", image_id)
        if benign + malignant != tumor:
            raise ValueError(
                f"{image_id}: benign + malignant must equal tumor, got "
                f"{benign} + {malignant} != {tumor}"
            )
        tumor_type, correction = resolve_tumor_type(row, image_id, annotation_path)
        anatomy = multi_flag_names(row, ANATOMY_COLUMNS, image_id, "anatomy")
        view = single_flag_name(row, VIEW_COLUMNS, image_id, "view")
        center = int(float(row["center"]))
        age = str(row["age"]).strip()
        gender = str(row["gender"]).strip()

        if tumor and not annotation_path.is_file():
            raise FileNotFoundError(f"tumor image is missing annotation: {annotation_path}")
        if annotation_path.is_file():
            try:
                payload = json.loads(annotation_path.read_text(encoding="utf-8"))
                if Path(str(payload.get("imagePath", ""))).name != image_id:
                    invalid_annotations.append(
                        {"annotation": annotation_path.name, "reason": "imagePath mismatch"}
                    )
            except Exception as exc:  # noqa: BLE001 - audit records all source failures
                invalid_annotations.append(
                    {"annotation": annotation_path.name, "reason": f"{type(exc).__name__}: {exc}"}
                )

        try:
            with Image.open(image_path) as image:
                width, height = image.size
                image_format = str(image.format)
                image.verify()
            sha256 = sha256_file(image_path)
            phash = f"{perceptual_hash(image_path):016x}" if compute_phash else ""
        except Exception as exc:  # noqa: BLE001 - audit records all source failures
            unreadable_images.append(
                {"image_id": image_id, "reason": f"{type(exc).__name__}: {exc}"}
            )
            continue

        audited.append(
            {
                "source_row": row_number,
                "image_id": image_id,
                "numeric_image_id": int("".join(character for character in Path(image_id).stem if character.isdigit())),
                "center": center,
                "age": age,
                "gender": gender,
                "anatomy": anatomy,
                "view": view,
                "tumor": tumor,
                "benign": benign,
                "malignant": malignant,
                "tumor_type": tumor_type,
                "tumor_type_name": TUMOR_TYPE_CLASS_NAMES[tumor_type],
                "metadata_correction": correction,
                "bytes": image_path.stat().st_size,
                "width": width,
                "height": height,
                "image_format": image_format,
                "image_sha256": sha256,
                "phash64": phash,
                "annotation_sha256": sha256_file(annotation_path) if annotation_path.is_file() else "",
            }
        )

    source_images = {path.name for path in images_dir.iterdir() if path.is_file()}
    source_annotations = {path.stem for path in annotations_dir.glob("*.json")}
    metadata_stems = {Path(image_id).stem for image_id in seen_ids}
    tumor_stems = {Path(row["image_id"]).stem for row in audited if row["tumor"]}
    inventory = {
        "metadata_rows": len(rows),
        "source_images": len(source_images),
        "source_annotations": len(source_annotations),
        "missing_metadata_images": sorted(seen_ids - source_images),
        "unused_source_images": sorted(source_images - seen_ids),
        "missing_tumor_annotations": sorted(tumor_stems - source_annotations),
        "annotations_not_in_metadata": sorted(source_annotations - metadata_stems),
        "unreadable_images": unreadable_images,
        "invalid_annotations": invalid_annotations,
    }
    if unreadable_images:
        raise RuntimeError(f"{len(unreadable_images)} source images are unreadable")
    return audited, inventory


def mark_exact_duplicates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_hash: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_hash[str(row["image_sha256"])].append(row)

    reports: list[dict[str, object]] = []
    for image_hash, group in sorted(by_hash.items()):
        for row in group:
            row["eligible"] = 1
            row["exclusion_reason"] = ""
            row["duplicate_of"] = ""
        if len(group) == 1:
            continue

        annotated = [row for row in group if row["annotation_sha256"]]
        if annotated and len({int(row["tumor"]) for row in group}) > 1:
            representative = min(annotated, key=lambda row: str(row["image_id"]))
            rule = "prefer annotated tumor row for conflicting tumor/normal exact duplicate"
        else:
            representative = min(group, key=lambda row: str(row["image_id"]))
            rule = "lexicographically first image_id"

        metadata_keys = (
            "center", "age", "gender", "anatomy", "view", "tumor", "benign",
            "malignant", "tumor_type",
        )
        signatures = {tuple(row[key] for key in metadata_keys) for row in group}
        conflict = len(signatures) > 1
        for row in group:
            if row is representative:
                continue
            row["eligible"] = 0
            row["exclusion_reason"] = (
                "exact_duplicate_conflicting_metadata" if conflict else "exact_duplicate"
            )
            row["duplicate_of"] = representative["image_id"]

        reports.append(
            {
                "image_sha256": image_hash,
                "image_ids": [row["image_id"] for row in group],
                "representative": representative["image_id"],
                "selection_rule": rule,
                "metadata_conflict": conflict,
                "metadata": [
                    {key: row[key] for key in ("image_id", *metadata_keys)} for row in group
                ],
            }
        )
    return reports


def assign_heuristic_groups(rows: list[dict[str, object]]) -> None:
    stable_keys = (
        "center", "age", "gender", "anatomy", "tumor", "benign", "malignant", "tumor_type"
    )
    ordered = sorted(rows, key=lambda row: int(row["numeric_image_id"]))
    group_number = 0
    previous_signature: tuple[object, ...] | None = None
    previous_numeric_id: int | None = None
    for row in ordered:
        signature = tuple(row[key] for key in stable_keys)
        numeric_id = int(row["numeric_image_id"])
        if (
            previous_signature != signature
            or previous_numeric_id is None
            or numeric_id != previous_numeric_id + 1
        ):
            group_number += 1
        row["group_id"] = f"btxrd-heuristic-{group_number:06d}"
        row["group_source"] = "consecutive_image_id_plus_stable_metadata_excluding_view"
        previous_signature = signature
        previous_numeric_id = numeric_id

    sizes = Counter(str(row["group_id"]) for row in rows if row["eligible"])
    for row in rows:
        row["group_size"] = sizes.get(str(row["group_id"]), 0)
        row["group_confidence"] = "heuristic" if row["group_size"] > 1 else "singleton"


def assign_group_splits(
    rows: list[dict[str, object]],
    ratios: tuple[float, float, float],
    seed: int,
) -> None:
    if any(ratio < 0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError(f"split ratios must be non-negative and sum to 1, got {ratios}")

    eligible = [row for row in rows if row["eligible"]]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in eligible:
        grouped[str(row["group_id"])].append(row)
    for group_id, group_rows in grouped.items():
        labels = {int(row["tumor_type"]) for row in group_rows}
        if len(labels) != 1:
            raise ValueError(f"heuristic group {group_id} spans tumor types {sorted(labels)}")

    groups_by_class: dict[int, list[tuple[str, list[dict[str, object]]]]] = defaultdict(list)
    for group_id, group_rows in grouped.items():
        groups_by_class[int(group_rows[0]["tumor_type"])].append((group_id, group_rows))

    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    for class_index in range(len(TUMOR_TYPE_CLASS_NAMES)):
        class_groups = groups_by_class.get(class_index, [])
        rng.shuffle(class_groups)
        class_groups.sort(key=lambda item: len(item[1]), reverse=True)
        total_images = sum(len(group_rows) for _, group_rows in class_groups)
        targets = {
            split_name: total_images * ratio
            for split_name, ratio in zip(SPLIT_NAMES, ratios)
        }
        counts = {split_name: 0 for split_name in SPLIT_NAMES}

        unassigned = list(class_groups)
        if len(unassigned) >= 3:
            smallest = sorted(unassigned, key=lambda item: (len(item[1]), item[0]))
            val_group = smallest[0]
            test_group = smallest[1]
            for split_name, selected in (("val", val_group), ("test", test_group)):
                assignments[selected[0]] = split_name
                counts[split_name] += len(selected[1])
                unassigned.remove(selected)
        elif len(unassigned) == 2:
            test_group = min(unassigned, key=lambda item: (len(item[1]), item[0]))
            assignments[test_group[0]] = "test"
            counts["test"] += len(test_group[1])
            unassigned.remove(test_group)

        for group_id, group_rows in unassigned:
            deficits = {
                split_name: targets[split_name] - counts[split_name]
                for split_name in SPLIT_NAMES
            }
            best_deficit = max(deficits.values())
            candidates = [name for name, deficit in deficits.items() if deficit == best_deficit]
            split_name = rng.choice(candidates)
            assignments[group_id] = split_name
            counts[split_name] += len(group_rows)

    for row in rows:
        row["split"] = assignments[str(row["group_id"])] if row["eligible"] else "excluded"


def validate_manifest_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    eligible = [row for row in rows if row["eligible"]]
    image_ids = [str(row["image_id"]) for row in eligible]
    if len(image_ids) != len(set(image_ids)):
        raise AssertionError("eligible manifest contains duplicate image_id")

    group_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    for row in eligible:
        group_splits[str(row["group_id"])].add(str(row["split"]))
        hash_splits[str(row["image_sha256"])].add(str(row["split"]))
    leaking_groups = {key: sorted(value) for key, value in group_splits.items() if len(value) > 1}
    leaking_hashes = {key: sorted(value) for key, value in hash_splits.items() if len(value) > 1}
    if leaking_groups:
        raise AssertionError(f"group leakage detected: {list(leaking_groups.items())[:5]}")
    if leaking_hashes:
        raise AssertionError(f"exact duplicate leakage detected: {list(leaking_hashes.items())[:5]}")

    return {
        "eligible_images": len(eligible),
        "excluded_images": len(rows) - len(eligible),
        "groups": len(group_splits),
        "group_overlap_count": 0,
        "exact_hash_overlap_count": 0,
    }


def split_distributions(rows: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for split_name in (*SPLIT_NAMES, "excluded"):
        split_rows = [row for row in rows if row["split"] == split_name]
        output[split_name] = {
            "images": len(split_rows),
            "groups": len({row["group_id"] for row in split_rows}) if split_name != "excluded" else 0,
            "tumor_type": dict(Counter(str(row["tumor_type_name"]) for row in split_rows)),
            "center": dict(Counter(str(row["center"]) for row in split_rows)),
            "anatomy": dict(Counter(str(row["anatomy"]) for row in split_rows)),
            "view": dict(Counter(str(row["view"]) for row in split_rows)),
        }
    return output


def near_duplicate_candidates(
    rows: list[dict[str, object]], maximum_distance: int
) -> list[dict[str, object]]:
    eligible = [row for row in rows if row["eligible"] and row["phash64"]]
    candidates: list[dict[str, object]] = []
    for left, right in itertools.combinations(eligible, 2):
        distance = (int(str(left["phash64"]), 16) ^ int(str(right["phash64"]), 16)).bit_count()
        if distance <= maximum_distance and left["image_sha256"] != right["image_sha256"]:
            candidates.append(
                {
                    "left": left["image_id"],
                    "right": right["image_id"],
                    "phash_hamming": distance,
                    "same_group": left["group_id"] == right["group_id"],
                    "same_stable_metadata": all(
                        left[key] == right[key]
                        for key in ("center", "age", "gender", "anatomy", "tumor_type")
                    ),
                    "left_split": left["split"],
                    "right_split": right["split"],
                }
            )
    return candidates


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "Annotations"
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        raise FileNotFoundError(
            f"Expected BTXRD images/ and Annotations/ under {dataset_root}"
        )

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "split_manifest.csv"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(
            f"Refusing to replace immutable manifest {manifest_path}. Use --force only after "
            "marking every dependent checkpoint/result stale."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows, metadata_path = load_metadata(dataset_root)
    rows, inventory = audit_source_rows(
        source_rows,
        dataset_root=dataset_root,
        compute_phash=not args.skip_phash,
    )
    exact_duplicate_report = mark_exact_duplicates(rows)
    assign_heuristic_groups(rows)
    assign_group_splits(rows, ratios=ratios, seed=args.seed)
    integrity = validate_manifest_rows(rows)
    near_candidates = (
        near_duplicate_candidates(rows, args.phash_distance) if not args.skip_phash else []
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    repository = git_state()
    dataset_table_sha256 = sha256_file(metadata_path)
    image_file_list_sha256 = hash_text_lines(
        f"{row['image_id']}\t{row['bytes']}\t{row['image_sha256']}" for row in rows
    )
    annotation_manifest_sha256 = hash_text_lines(
        f"{row['image_id']}\t{row['annotation_sha256']}"
        for row in rows
        if row["annotation_sha256"]
    )

    for row in rows:
        row.update(
            {
                "split_seed": args.seed,
                "split_ratios": ",".join(str(value) for value in ratios),
                "split_algorithm": "heuristic-consecutive-metadata-group-greedy-v1",
                "grouping_limitation": (
                    "heuristic only; BTXRD publishes no patient/lesion/case identifier"
                ),
                "dataset_table": metadata_path.name,
                "dataset_table_sha256": dataset_table_sha256,
                "git_commit": repository.get("commit", ""),
            }
        )

    manifest_fields = (
        "image_id",
        "group_id",
        "group_source",
        "group_confidence",
        "group_size",
        "split",
        "eligible",
        "exclusion_reason",
        "duplicate_of",
        "tumor",
        "benign",
        "malignant",
        "tumor_type",
        "tumor_type_name",
        "center",
        "age",
        "gender",
        "anatomy",
        "view",
        "image_sha256",
        "annotation_sha256",
        "bytes",
        "width",
        "height",
        "image_format",
        "metadata_correction",
        "split_seed",
        "split_ratios",
        "split_algorithm",
        "grouping_limitation",
        "dataset_table",
        "dataset_table_sha256",
        "git_commit",
    )
    write_csv(manifest_path, rows, manifest_fields)
    write_csv(
        output_dir / "image_hash_manifest.csv",
        rows,
        ("image_id", "bytes", "width", "height", "image_format", "image_sha256", "phash64"),
    )
    (output_dir / "exact_duplicate_report.json").write_text(
        json.dumps(exact_duplicate_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "near_duplicate_candidates.json").write_text(
        json.dumps(near_candidates, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest_sha256 = sha256_file(manifest_path)
    summary = {
        "builder_version": BUILDER_VERSION,
        "generated_at_utc": generated_at,
        "dataset_root": str(dataset_root),
        "metadata_path": str(metadata_path),
        "dataset_table_sha256": dataset_table_sha256,
        "image_file_list_sha256": image_file_list_sha256,
        "annotation_manifest_sha256": annotation_manifest_sha256,
        "split_manifest_sha256": manifest_sha256,
        "repository": repository,
        "split_seed": args.seed,
        "split_ratios": ratios,
        "split_algorithm": "heuristic-consecutive-metadata-group-greedy-v1",
        "partition_unit": "heuristic case group",
        "partition_limitation": (
            "The official BTXRD release contains no patient, lesion, case, study or accession ID. "
            "Groups are inferred from consecutive image IDs and identical stable metadata while "
            "excluding view. They reduce obvious multi-view leakage but do not prove patient- or "
            "lesion-independent evaluation."
        ),
        "inventory": inventory,
        "integrity": integrity,
        "metadata_corrections": [
            {"image_id": row["image_id"], "correction": row["metadata_correction"]}
            for row in rows
            if row["metadata_correction"]
        ],
        "exact_duplicate_groups": len(exact_duplicate_report),
        "exact_duplicate_conflict_groups": sum(
            bool(report["metadata_conflict"]) for report in exact_duplicate_report
        ),
        "near_duplicate_candidates": len(near_candidates),
        "group_size_distribution": dict(
            sorted(Counter(int(row["group_size"]) for row in rows if row["eligible"]).items())
        ),
        "split_distributions": split_distributions(rows),
    }
    (output_dir / "dataset_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
