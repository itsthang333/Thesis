from __future__ import annotations

"""GT-blind inventory audit for the frozen S8 producer transport package."""

import argparse
import hashlib
import json
from pathlib import Path


PAIR_FREEZE = "b2cfd59fb01046f445d098790efa5a0fdc649bbc80f565439ba51c5cd453fa00"
RUN_MANIFEST = "5bb136f8b6f7a6a173abacce2faf0aad1b7caf9e087adcfd158655d71ff7c510"
DIAGNOSTICS = "98ceacd4a3dd1c32d42105c8cddc436d0e8256dd596c3b250f18fa2e39ecc569"
EVIDENCE_MANIFEST = "be391e11deef6a02a32c85c3bdc861cb05dd357313d89b9177c3bcdfb850cf55"
CONTROL_FREEZE = "aa8acefaafe9c453dd08e1cba8d71b4a0aae25ecf16662774aebf53d01cf2ccd"
CONTROL_SCORES = "6d08db6c7fd29d5ed2bc7ff57133cd9fdea1adc65ec797f9c3f0ee424b3452fc"
CONTROL_PREDICTIONS = "f66f7370b8b93ddadd1b7eef134ae5735c0a8f6743c97de021eab331d06666bf"
PRIMARY_FREEZE = "a96e8a0bf88201c8b3fffb30c7ec41fd0abb2c3e4b7e9430408665b01dc2a148"
PRIMARY_SCORES = "0f66c5ff54ea44f778bc62fb0e71145191e0189bdcc0ac419f87d0b1ec566de3"
PRIMARY_PREDICTIONS = "285c026853bb5e91482140e03cc2d47d418f4172716aec0aad32e97e5b9d7309"
DIRECT_LOG = "0ed4c39bcee22a559de3c642bb72a5b77ce148c50e8260bf0d89d5dd47d82a79"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path) -> tuple[list[dict[str, object]], str]:
    rows: list[dict[str, object]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    payload = "".join(
        f"{row['path']}\t{row['bytes']}\t{row['sha256']}\n" for row in rows
    ).encode("utf-8")
    return rows, hashlib.sha256(payload).hexdigest()


def require_hash(root: Path, relative: str, expected: str) -> None:
    path = root / relative
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"hash mismatch: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--direct-log", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    log = args.direct_log.resolve()
    if not root.is_dir() or not log.is_file():
        raise RuntimeError("missing frozen producer root or direct log")
    rows, inventory_sha = inventory(root)
    expected = {
        "prediction_pair_freeze.json": PAIR_FREEZE,
        "run_manifest.json": RUN_MANIFEST,
        "gt_blind_diagnostics.json": DIAGNOSTICS,
        "reconstruction_evidence/evidence_manifest.json": EVIDENCE_MANIFEST,
        "geometry_v3_plus_upstream_equal_rank/prediction_freeze.json": CONTROL_FREEZE,
        "geometry_v3_plus_upstream_equal_rank/candidate_scores/candidate_score_manifest.csv": CONTROL_SCORES,
        "geometry_v3_plus_upstream_equal_rank/predictions/prediction_manifest.csv": CONTROL_PREDICTIONS,
        "geometry_v3_plus_upstream_plus_skelex_reconstruction_rerank/prediction_freeze.json": PRIMARY_FREEZE,
        "geometry_v3_plus_upstream_plus_skelex_reconstruction_rerank/candidate_scores/candidate_score_manifest.csv": PRIMARY_SCORES,
        "geometry_v3_plus_upstream_plus_skelex_reconstruction_rerank/predictions/prediction_manifest.csv": PRIMARY_PREDICTIONS,
    }
    for relative, expected_hash in expected.items():
        require_hash(root, relative, expected_hash)
    evidence = list((root / "reconstruction_evidence").glob("*.npz"))
    control_scores = list((root / "geometry_v3_plus_upstream_equal_rank/candidate_scores/scores").glob("*.npz"))
    primary_scores = list((root / "geometry_v3_plus_upstream_plus_skelex_reconstruction_rerank/candidate_scores/scores").glob("*.npz"))
    control_maps = list((root / "geometry_v3_plus_upstream_equal_rank/predictions/maps").glob("*.npy"))
    primary_maps = list((root / "geometry_v3_plus_upstream_plus_skelex_reconstruction_rerank/predictions/maps").glob("*.npy"))
    counts = {
        "evidence_npz": len(evidence),
        "control_score_npz": len(control_scores),
        "primary_score_npz": len(primary_scores),
        "control_prediction_npy": len(control_maps),
        "primary_prediction_npy": len(primary_maps),
    }
    if counts != {
        "evidence_npz": 371,
        "control_score_npz": 371,
        "primary_score_npz": 371,
        "control_prediction_npy": 371,
        "primary_prediction_npy": 371,
    }:
        raise RuntimeError(f"unexpected frozen output counts: {counts}")
    forbidden = ("validation_gt", "ground_truth", "annotation", "test_image", "test_mask")
    forbidden_paths = [
        row["path"] for row in rows
        if any(token in str(row["path"]).lower() for token in forbidden)
        and str(row["path"]).lower() != "gt_blind_diagnostics.json"
    ]
    if forbidden_paths:
        raise RuntimeError(f"forbidden GT/test-like paths: {forbidden_paths[:5]}")
    payload = {
        "schema_version": 1,
        "status": "GT_BLIND_FROZEN_PRODUCER_TRANSPORT_AUDIT_PASS",
        "source_root": str(args.output_root),
        "direct_log": {"path": str(args.direct_log), "bytes": log.stat().st_size, "sha256": sha256(log)},
        "expected_direct_log_sha256": DIRECT_LOG,
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "inventory_sha256": inventory_sha,
        "required_hashes": expected,
        "counts": counts,
        "forbidden_gt_test_paths": forbidden_paths,
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
        "notes": "This is transport provenance for already-frozen GT-blind producer output; it creates no prediction and performs no evaluator action.",
    }
    if sha256(log) != DIRECT_LOG:
        raise RuntimeError("direct log hash mismatch")
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
