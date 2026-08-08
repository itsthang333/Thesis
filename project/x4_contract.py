from __future__ import annotations

"""Immutable constants shared by the X4 thesis experiments."""

from pathlib import Path
import hashlib
import json


CANONICAL_SPLIT_SHA256 = (
    "85511ee1bd1339c7b6b4f527acc504869da935997fd6b2485042edd619193c8c"
)
RESNET18_IMAGENET1K_V1_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)
X4_PROTOCOL_RELATIVE_PATH = Path("artifacts/final_pipeline/x4/x4_protocol.json")
STUDENT_SEEDS = (42, 43, 44)
STUDENT_ARMS = (
    "cam",
    "puzzlecam",
    "s2c",
    "rich_gallery",
    "fully_supervised",
)
PSEUDO_STUDENT_ARMS = STUDENT_ARMS[:-1]
GATE_ARMS = (
    "known_binary_label",
    "binary_predicted_gate",
    "ten_class_predicted_gate",
    "label_free_rich_gallery_student",
)
THRESHOLD_GRID = tuple(round(value / 100.0, 2) for value in range(10, 91, 5))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_x4_protocol(repo_root: Path) -> tuple[dict[str, object], str]:
    path = repo_root / X4_PROTOCOL_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("split_sha256") != CANONICAL_SPLIT_SHA256
        or tuple(payload.get("student_seeds", ())) != STUDENT_SEEDS
        or tuple(payload.get("student_arms", ())) != STUDENT_ARMS
        or tuple(payload.get("threshold_grid", ())) != THRESHOLD_GRID
        or payload.get("test_evaluated") is not False
    ):
        raise ValueError("X4 protocol differs from the immutable code contract")
    return payload, sha256_file(path)
