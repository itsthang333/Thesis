from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_WRAPPER_SHA256 = (
    "e34171b67f0bca069d488eaa47610ae6cbc6415e2ac6a9f45f306eb1a8b02a30"
)
EXPECTED_REPOSITORY_COMMIT = "26c22db7c3af18c32f66b1cd3b4c3987a10ded19"
EXPECTED_IMPLEMENTATION_COMMIT = "95fc1c24ce8387c3ef211b4a0b71f6275f4e8b68"
EXPECTED_PROTOCOL_SHA256 = (
    "9f5b2250d4a82fa8d546f3dd1dd3c7b477235addb9ab93d709637dc376015ea8"
)
EXPECTED_PROTOCOL_AUDIT_SHA256 = (
    "b009cad423567d3b98f7d6d31e8418695cf6bf7734b7669d3e40750fc31e0c5e"
)
EXPECTED_SPLIT_SHA256 = (
    "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
)
EXPECTED_WEIGHT_SHA256 = (
    "52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be"
)
FORBIDDEN_SOURCE_TOKENS = (
    "evaluate_pseudo_masks.py",
    "build_segmentation_dataset",
    "Annotations",
    "evaluate_prompt_quality",
    "ground_truth",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def literal_assignments(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    result: dict[str, Any] = {}

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in result:
            return result[node.id]
        if isinstance(node, ast.List):
            return [evaluate(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(evaluate(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            return {
                evaluate(key): evaluate(value)
                for key, value in zip(node.keys, node.values)
            }
        raise ValueError(f"Unsupported predeclaration expression: {type(node).__name__}")

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            result[target.id] = evaluate(node.value)
        except (TypeError, ValueError):
            continue
    return result


def validate_wrapper_source(source: str, *, wrapper_sha256: str) -> dict[str, Any]:
    if wrapper_sha256 != EXPECTED_WRAPPER_SHA256:
        raise ValueError("Saliency-only wrapper SHA-256 mismatch")
    assignments = literal_assignments(source)
    expected_constants = {
        "SOURCE_REPOSITORY_COMMIT": EXPECTED_REPOSITORY_COMMIT,
        "IMPLEMENTATION_COMMIT": EXPECTED_IMPLEMENTATION_COMMIT,
        "EXPECTED_FROZEN_SPLIT_SHA256": EXPECTED_SPLIT_SHA256,
        "EXPECTED_MODEL_WEIGHT_SHA256": EXPECTED_WEIGHT_SHA256,
        "EXPECTED_PROTOCOL_SHA256": EXPECTED_PROTOCOL_SHA256,
        "EXPECTED_PROTOCOL_AUDIT_SHA256": EXPECTED_PROTOCOL_AUDIT_SHA256,
    }
    for name, expected in expected_constants.items():
        if assignments.get(name) != expected:
            raise ValueError(f"Saliency-only wrapper constant drift: {name}")
    predeclared = assignments.get("PREDECLARED")
    if not isinstance(predeclared, dict):
        raise ValueError("Saliency-only wrapper predeclaration is absent")
    if predeclared.get("stage") != "prediction-first full-validation saliency generation only":
        raise ValueError("Saliency-only wrapper stage mismatch")
    if predeclared.get("validation_gt_read") is not False:
        raise ValueError("Saliency-only wrapper did not lock validation GT")
    if predeclared.get("test_evaluated") is not False:
        raise ValueError("Saliency-only wrapper did not lock test")
    if predeclared.get("method", {}).get("target_layer") != (
        "model.visual.trunk.blocks[11].norm1"
    ):
        raise ValueError("Saliency-only wrapper target-layer mismatch")
    for token in FORBIDDEN_SOURCE_TOKENS:
        if token in source:
            raise ValueError(f"Saliency-only wrapper contains forbidden token: {token}")
    for required in (
        "generate_biomedclip_saliency.py",
        "audit_biomedclip_saliency.py",
        "--expected-model-weight-sha256",
        "--source-commit",
        '"validation_gt_read": False',
        '"test_evaluated": False',
    ):
        if required not in source:
            raise ValueError(f"Saliency-only wrapper omits required contract: {required}")
    return {
        "status": "PASS",
        "wrapper_sha256": wrapper_sha256,
        "source_repository_commit": EXPECTED_REPOSITORY_COMMIT,
        "implementation_commit": EXPECTED_IMPLEMENTATION_COMMIT,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "protocol_audit_sha256": EXPECTED_PROTOCOL_AUDIT_SHA256,
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "model_weight_sha256": EXPECTED_WEIGHT_SHA256,
        "validation_gt_read": False,
        "test_evaluated": False,
    }


def audit(wrapper: Path, protocol: Path, protocol_audit: Path) -> dict[str, Any]:
    if sha256_file(protocol) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Local protocol SHA-256 mismatch")
    if sha256_file(protocol_audit) != EXPECTED_PROTOCOL_AUDIT_SHA256:
        raise ValueError("Local protocol-audit SHA-256 mismatch")
    source = wrapper.read_text(encoding="utf-8")
    return validate_wrapper_source(source, wrapper_sha256=sha256_file(wrapper))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-audit", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args.wrapper, args.protocol, args.protocol_audit)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
