from __future__ import annotations

"""Fail-closed binding of one accepted selector cache into the frozen R1 wrapper."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


UNBOUND_TEMPLATE_SHA256 = (
    "4b886f91aa01b15c18a1a0105db11a31f62233635e214e9bbd406e5712d05044"
)
CORRECTED_CACHE_SOURCE_COMMIT = "c0e38628069ff3bedd4493c4ff004b75bd32e008"
CORRECTED_CACHE_PROTOCOL_SHA256 = (
    "be9c91b53926eda2f8bf8dba894385f03dc6accd15d3fa9646da0d1a17a635f2"
)
R1_SOURCE_COMMIT = "d66c89958baa3344dbbaae6030a9ccd8ecec7b3a"
R1_PROTOCOL_SHA256 = (
    "dded9c638e142576fedf0ae4c8102fdf64198744a4949707865e50b7081f312b"
)
PENDING = "PENDING_TERMINAL_SELECTOR_CACHE_GATE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, *, name: str) -> str:
    result = value.strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return result


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _literal_assignments(source: str) -> dict[str, object]:
    tree = ast.parse(source)
    values: dict[str, object] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    return values


def _require_safety(payload: Mapping[str, object], *, name: str) -> None:
    if (
        payload.get("validation_gt_read") is not False
        or payload.get("consumer_trained") is not False
        or payload.get("test_evaluated") is not False
    ):
        raise ValueError(f"{name} safety boundary mismatch")


def _verify_cache_contract(
    cache_root: Path,
    independent_audit_path: Path,
    *,
    expected_independent_audit_sha256: str,
) -> dict[str, str]:
    independent_hash = _validate_sha256(
        expected_independent_audit_sha256,
        name="independent audit SHA-256",
    )
    if sha256_file(independent_audit_path) != independent_hash:
        raise ValueError("Independent cache audit SHA-256 mismatch")
    independent = _json(independent_audit_path)
    freeze_path = cache_root / "selector_cache_freeze.json"
    wrapper_audit_path = cache_root / "wrapper_output_audit.json"
    freeze_hash = sha256_file(freeze_path)
    wrapper_audit_hash = sha256_file(wrapper_audit_path)
    freeze = _json(freeze_path)
    wrapper_audit = _json(wrapper_audit_path)

    reproduction = independent.get("baseline_reproduction", {})
    if not isinstance(reproduction, dict):
        raise ValueError("Independent cache audit lacks baseline reproduction")
    if (
        independent.get("audit_id")
        != "independent_mask_bag_selector_cache_output_v1"
        or independent.get("cache_freeze_sha256") != freeze_hash
        or independent.get("wrapper_output_audit_sha256") != wrapper_audit_hash
        or independent.get("source_commit") != CORRECTED_CACHE_SOURCE_COMMIT
        or independent.get("protocol_sha256") != CORRECTED_CACHE_PROTOCOL_SHA256
        or independent.get("physical_cache_records_verified") != 3352
        or independent.get("validation_packed_mask_geometry_records_verified") != 371
        or independent.get("cohort") != {"train": 2981, "val": 371}
        or reproduction.get("validation_images") != 371
        or reproduction.get("selected_indices_exact") != 371
        or reproduction.get("map_hashes_exact") != 371
        or independent.get("training_labels") != "image_level_only"
    ):
        raise ValueError("Independent selector-cache audit contract mismatch")
    _require_safety(independent, name="independent audit")

    if (
        freeze.get("source_commit") != CORRECTED_CACHE_SOURCE_COMMIT
        or freeze.get("protocol_sha256") != CORRECTED_CACHE_PROTOCOL_SHA256
        or freeze.get("cohort") != {"train": 2981, "validation": 371}
        or freeze.get("validation_selected_indices_reproduced") != 371
        or freeze.get("validation_map_hashes_reproduced") != 371
        or freeze.get("train_masks_discarded") is not True
        or freeze.get("validation_masks_bitpacked") is not True
        or freeze.get("affinity_features_cached") is not True
        or freeze.get("affinity_feature_dim") != 24
    ):
        raise ValueError("Selector-cache freeze contract mismatch")
    _require_safety(freeze, name="cache freeze")

    if (
        wrapper_audit.get("scientific_source_commit")
        != CORRECTED_CACHE_SOURCE_COMMIT
        or wrapper_audit.get("protocol_sha256")
        != CORRECTED_CACHE_PROTOCOL_SHA256
        or wrapper_audit.get("selector_cache_freeze_sha256") != freeze_hash
        or wrapper_audit.get("physical_cache_records_verified") != 3352
        or wrapper_audit.get("cohort") != {"train": 2981, "val": 371}
        or wrapper_audit.get("validation_selected_indices_reproduced") != 371
        or wrapper_audit.get("validation_map_hashes_reproduced") != 371
    ):
        raise ValueError("Selector-cache wrapper audit contract mismatch")
    _require_safety(wrapper_audit, name="cache wrapper audit")
    return {
        "cache_freeze_sha256": freeze_hash,
        "cache_wrapper_audit_sha256": wrapper_audit_hash,
        "independent_audit_sha256": independent_hash,
    }


def bind_r1_cache(
    *,
    template_path: Path,
    cache_root: Path,
    independent_audit_path: Path,
    expected_independent_audit_sha256: str,
    output_wrapper_path: Path,
    output_audit_path: Path,
    expected_template_sha256: str = UNBOUND_TEMPLATE_SHA256,
) -> dict[str, object]:
    template_hash = _validate_sha256(expected_template_sha256, name="template SHA-256")
    if sha256_file(template_path) != template_hash:
        raise ValueError("Unbound R1 wrapper template SHA-256 mismatch")
    if output_wrapper_path.resolve() == output_audit_path.resolve():
        raise ValueError("Bound wrapper and binding audit outputs must differ")
    if output_wrapper_path.exists() or output_audit_path.exists():
        raise FileExistsError("Bound wrapper/audit output already exists")

    source_bytes = template_path.read_bytes()
    source = source_bytes.decode("utf-8")
    assignments = _literal_assignments(source)
    expected_assignments = {
        "SCIENTIFIC_SOURCE_COMMIT": R1_SOURCE_COMMIT,
        "PROTOCOL_SHA256": R1_PROTOCOL_SHA256,
        "CACHE_PROTOCOL_SHA256": CORRECTED_CACHE_PROTOCOL_SHA256,
        "CACHE_SCIENTIFIC_SOURCE_COMMIT": CORRECTED_CACHE_SOURCE_COMMIT,
        "CACHE_BINDING_READY": False,
        "CACHE_FREEZE_SHA256": PENDING,
        "CACHE_WRAPPER_AUDIT_SHA256": PENDING,
    }
    if any(assignments.get(key) != value for key, value in expected_assignments.items()):
        raise ValueError("Unbound R1 wrapper constants differ from the frozen template")

    cache = _verify_cache_contract(
        cache_root,
        independent_audit_path,
        expected_independent_audit_sha256=expected_independent_audit_sha256,
    )
    replacements = {
        b"CACHE_BINDING_READY = False": b"CACHE_BINDING_READY = True",
        f'CACHE_FREEZE_SHA256 = "{PENDING}"'.encode(): (
            f'CACHE_FREEZE_SHA256 = "{cache["cache_freeze_sha256"]}"'.encode()
        ),
        f'CACHE_WRAPPER_AUDIT_SHA256 = "{PENDING}"'.encode(): (
            "CACHE_WRAPPER_AUDIT_SHA256 = "
            f'"{cache["cache_wrapper_audit_sha256"]}"'
        ).encode(),
    }
    bound_bytes = source_bytes
    for old, new in replacements.items():
        if bound_bytes.count(old) != 1:
            raise ValueError(f"Frozen binding target does not occur exactly once: {old!r}")
        bound_bytes = bound_bytes.replace(old, new)

    reversed_bytes = bound_bytes
    for old, new in reversed(tuple(replacements.items())):
        if reversed_bytes.count(new) != 1:
            raise ValueError(f"Bound replacement does not occur exactly once: {new!r}")
        reversed_bytes = reversed_bytes.replace(new, old)
    if reversed_bytes != source_bytes:
        raise RuntimeError("Bound wrapper differs outside the three allowed replacements")

    bound_assignments = _literal_assignments(bound_bytes.decode("utf-8"))
    if (
        bound_assignments.get("CACHE_BINDING_READY") is not True
        or bound_assignments.get("CACHE_FREEZE_SHA256")
        != cache["cache_freeze_sha256"]
        or bound_assignments.get("CACHE_WRAPPER_AUDIT_SHA256")
        != cache["cache_wrapper_audit_sha256"]
    ):
        raise RuntimeError("Bound R1 wrapper constants are invalid")

    bound_hash = hashlib.sha256(bound_bytes).hexdigest()
    audit: dict[str, object] = {
        "audit_id": "mask_bag_normal_prototype_r1_cache_binding_v1",
        "status": "BOUND_NOT_LAUNCHED",
        "binder_source_sha256": sha256_file(Path(__file__)),
        "unbound_template_sha256": template_hash,
        "bound_wrapper_sha256": bound_hash,
        **cache,
        "cache_source_commit": CORRECTED_CACHE_SOURCE_COMMIT,
        "cache_protocol_sha256": CORRECTED_CACHE_PROTOCOL_SHA256,
        "r1_source_commit": R1_SOURCE_COMMIT,
        "r1_protocol_sha256": R1_PROTOCOL_SHA256,
        "exact_wrapper_replacements": 3,
        "inverse_byte_reconstruction_sha256": hashlib.sha256(reversed_bytes).hexdigest(),
        "training_labels": "image_level_only",
        "validation_gt_read": False,
        "consumer_trained": False,
        "test_evaluated": False,
        "kernel_launched": False,
    }
    output_wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    output_audit_path.parent.mkdir(parents=True, exist_ok=True)
    output_wrapper_path.write_bytes(bound_bytes)
    output_audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if sha256_file(output_wrapper_path) != bound_hash:
        raise RuntimeError("Written bound wrapper hash mismatch")
    if _json(output_audit_path) != audit:
        raise RuntimeError("Written cache-binding audit content mismatch")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--independent-audit", type=Path, required=True)
    parser.add_argument("--expected-independent-audit-sha256", required=True)
    parser.add_argument("--output-wrapper", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = bind_r1_cache(
        template_path=args.template,
        cache_root=args.cache_root,
        independent_audit_path=args.independent_audit,
        expected_independent_audit_sha256=args.expected_independent_audit_sha256,
        output_wrapper_path=args.output_wrapper,
        output_audit_path=args.output_audit,
    )
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
