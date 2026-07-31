from __future__ import annotations

"""Audit post-freeze proposal-gallery union ceilings on BTXRD validation.

This is a diagnostic only.  It combines already-frozen candidate galleries by
taking the per-image maximum oracle value.  It never constructs a deployable
router and must not be used as an inference-time selection rule.
"""

import argparse
import csv
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np


GROUPS = ("overall", "small", "medium", "large")


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _require_hash(path: Path, expected: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"Frozen artifact hash mismatch: {path}")


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("test_evaluated") is not False:
        raise ValueError(f"Source contract does not keep test locked: {path}")
    return payload


def _load_oracles(path: Path) -> tuple[dict[str, float], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values: dict[str, float] = {}
    size_groups: dict[str, str] = {}
    for row in rows:
        image_name = str(row.get("image_name", "")).strip()
        if not image_name or image_name in values:
            raise ValueError(f"Duplicate or empty image_name in {path}")
        value = float(row["oracle_best_single_dice"])
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid oracle value for {image_name} in {path}")
        values[image_name] = value
        if row.get("size_group"):
            group = str(row["size_group"]).strip().lower()
            if group not in {"small", "medium", "large"}:
                raise ValueError(f"Invalid size group {group!r} for {image_name}")
            size_groups[image_name] = group
    if len(values) != 184:
        raise ValueError(f"Expected 184 tumor rows in {path}, found {len(values)}")
    return values, size_groups


def summarize(values: dict[str, float], size_groups: dict[str, str]) -> dict[str, float]:
    ids = sorted(values)
    result = {"overall": float(np.mean([values[key] for key in ids]))}
    for group in GROUPS[1:]:
        members = [values[key] for key in ids if size_groups[key] == group]
        result[group] = float(np.mean(members))
    return result


def analyze(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    if int(spec.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported analysis specification")
    if spec.get("test_evaluated") is not False:
        raise ValueError("Union analysis must keep test locked")

    sources: dict[str, dict[str, float]] = {}
    size_groups: dict[str, str] | None = None
    source_evidence: dict[str, dict[str, Any]] = {}
    for source in spec["sources"]:
        name = str(source["name"])
        if not name or name in sources:
            raise ValueError(f"Duplicate or empty source name: {name!r}")
        csv_path = _resolve(root, source["prompt_quality_csv"])
        contract_path = _resolve(root, source["contract_json"])
        _require_hash(csv_path, source["prompt_quality_sha256"])
        _require_hash(contract_path, source["contract_sha256"])
        _load_contract(contract_path)
        values, groups = _load_oracles(csv_path)
        if groups:
            if len(groups) != 184:
                raise ValueError("Size reference must cover all 184 tumor images")
            if size_groups is not None and groups != size_groups:
                raise ValueError("Frozen size-group assignments disagree")
            size_groups = groups
        sources[name] = values
        source_evidence[name] = {
            "prompt_quality_csv": source["prompt_quality_csv"],
            "prompt_quality_sha256": source["prompt_quality_sha256"],
            "contract_json": source["contract_json"],
            "contract_sha256": source["contract_sha256"],
        }

    if size_groups is None:
        raise ValueError("No source supplies frozen size-group assignments")
    expected_ids = set(next(iter(sources.values())))
    if any(set(values) != expected_ids for values in sources.values()):
        raise ValueError("Proposal sources do not cover the same 184 tumor images")
    counts = {
        group: sum(value == group for value in size_groups.values())
        for group in GROUPS[1:]
    }
    if counts != {"small": 94, "medium": 72, "large": 18}:
        raise ValueError(f"Frozen subgroup counts differ: {counts}")

    fully = {key: float(spec["fully_reference"][key]) for key in GROUPS}
    rows: list[dict[str, Any]] = []
    names = sorted(sources)
    for width in range(1, len(names) + 1):
        for members in combinations(names, width):
            union = {
                image_id: max(sources[name][image_id] for name in members)
                for image_id in expected_ids
            }
            metrics = summarize(union, size_groups)
            rows.append(
                {
                    "sources": list(members),
                    "source_count": width,
                    "metrics": metrics,
                    "delta_vs_fully": {
                        key: metrics[key] - fully[key] for key in GROUPS
                    },
                    "oracle_exceeds_fully_all_metrics": all(
                        metrics[key] >= fully[key] for key in GROUPS
                    ),
                }
            )

    singles = [row for row in rows if row["source_count"] == 1]
    pairs = [row for row in rows if row["source_count"] == 2]
    feasible_pairs = [row for row in pairs if row["oracle_exceeds_fully_all_metrics"]]
    if not feasible_pairs:
        raise RuntimeError("No two-source union exceeds the frozen fully reference")
    anchor = str(spec.get("required_anchor_source", ""))
    if anchor not in sources:
        raise ValueError("required_anchor_source must name one frozen source")
    anchored_pairs = [row for row in feasible_pairs if anchor in row["sources"]]
    if not anchored_pairs:
        raise RuntimeError("No anchored two-source union exceeds the fully reference")
    best_overall_pair = max(
        pairs,
        key=lambda row: (row["metrics"]["overall"], row["metrics"]["small"]),
    )
    small_priority_pair = max(
        anchored_pairs,
        key=lambda row: (row["metrics"]["small"], row["metrics"]["overall"]),
    )
    all_union = next(row for row in rows if row["source_count"] == len(names))
    return {
        "schema_version": 1,
        "analysis_role": "post-freeze feasibility diagnostic; not a deployable router",
        "cohort": {"tumor": 184, **counts},
        "source_evidence": source_evidence,
        "fully_reference": fully,
        "single_galleries": sorted(
            singles, key=lambda row: row["metrics"]["overall"], reverse=True
        ),
        "best_overall_pair": best_overall_pair,
        "recommended_minimal_pair": {
            **small_priority_pair,
            "selection_rule": (
                "among two-source unions containing the required current-pipeline "
                "anchor and whose oracle exceeds the frozen fully reference in "
                "overall/small/medium/large, maximize the predeclared small-lesion "
                "oracle, then overall oracle"
            ),
            "required_anchor_source": anchor,
        },
        "all_source_union": all_union,
        "all_combinations": rows,
        "algorithmic_use": (
            "append the recommended sources unconditionally before image-label-only "
            "selection; per-image oracle winners and validation size groups are "
            "forbidden inference inputs"
        ),
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    result = analyze(spec, spec_path.parents[2])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
