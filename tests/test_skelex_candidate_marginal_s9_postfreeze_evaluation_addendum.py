from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
ADDENDUM = ROOT / (
    "artifacts/research_protocols/"
    "skelex_candidate_marginal_s9_v1_postfreeze_evaluation_addendum.json"
)
PROTOCOL = ROOT / "artifacts/research_protocols/skelex_candidate_marginal_s9_v1.json"


def _canonical_sha(path: Path) -> str:
    return sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_s9_postfreeze_addendum_closes_sources_and_pre_gt_boundary() -> None:
    payload = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    assert payload["status"] == "FROZEN_BEFORE_VALIDATION_GT"
    assert payload["bootstrap_replicates"] == 10_000
    assert payload["bootstrap_seed"] == 20261205
    assert payload["validation_gt_read"] is False
    assert payload["consumer_trained"] is False
    assert payload["test_evaluated"] is False
    assert payload["post_hoc_rescue_or_sweep_authorized"] is False
    assert sha256(PROTOCOL.read_bytes()).hexdigest() == payload[
        "scientific_protocol_sha256"
    ]
    source = payload["source"]
    for name in (
        "evaluator",
        "dynamic_readiness",
        "dynamic_readiness_test",
        "matched_decision",
        "matched_decision_test",
    ):
        path = ROOT / source[name]
        assert _canonical_sha(path) == source[f"{name}_canonical_lf_sha256"]
    commit = payload["decision_source_commit"]
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=True,
    )


def test_s9_postfreeze_addendum_freezes_exact_mechanism_and_operational_gates() -> None:
    payload = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    assert all(payload["mechanism_gate"].values())
    assert payload["operational_gate"]["minimum_dice"] == {
        "overall": 0.34024039,
        "small": 0.17895493,
        "medium": 0.51244178,
        "large": 0.49370336,
    }
    assert payload["dynamic_hash_boundary"][
        "readiness_must_be_committed_and_visible_central_before_evaluation"
    ] is True
    assert payload["required_order"][2].startswith("freeze dynamic output hashes")

