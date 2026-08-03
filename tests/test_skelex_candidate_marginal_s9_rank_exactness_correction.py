from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "artifacts/research_protocols/skelex_candidate_marginal_s9_v1.json"
CORRECTION = ROOT / (
    "artifacts/research_protocols/"
    "skelex_candidate_marginal_s9_v1_rank_exactness_correction.json"
)
ERROR_AUDIT = ROOT / (
    "artifacts/kaggle/skelex_candidate_marginal_s9_v1/"
    "kernel_version1_error_audit.json"
)


def _canonical_sha(path: Path) -> str:
    return sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_s9_rank_correction_closes_exact_old_and_new_source_hashes() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    correction = json.loads(CORRECTION.read_text(encoding="utf-8"))
    assert correction["status"] == (
        "IMPLEMENTATION_ONLY_CORRECTION_BEFORE_PREDICTION_FREEZE"
    )
    assert sha256(PROTOCOL.read_bytes()).hexdigest() == correction[
        "scientific_protocol_sha256"
    ]
    assert sha256(ERROR_AUDIT.read_bytes()).hexdigest() == correction[
        "failure_audit_sha256"
    ]
    assert correction["kernel_version_failed"] == 1
    assert correction["kernel_version_next"] == 2
    for relative, expected in correction["corrected_source_hashes"].items():
        assert _canonical_sha(ROOT / relative) == expected
        assert protocol["canonical_lf_source_hashes"][relative] == correction[
            "superseded_protocol_hashes"
        ][relative]


def test_s9_rank_correction_is_implementation_only_and_pre_gt() -> None:
    payload = json.loads(CORRECTION.read_text(encoding="utf-8"))
    assert all(payload["scientific_contract_unchanged"].values())
    assert payload["implementation_change"][
        "duplicate_numpy_float64_arm_composition_removed"
    ] is True
    assert payload["implementation_change"][
        "exact_vector_and_argmax_regression_candidate_counts"
    ] == [1, 81]
    assert payload["partial_version1_outputs_reused_for_prediction"] is False
    assert payload["post_hoc_scientific_rescue_or_sweep_authorized"] is False
    assert payload["validation_gt_read"] is False
    assert payload["consumer_trained"] is False
    assert payload["test_evaluated"] is False


def test_s9_rank_correction_commits_are_real_and_ordered() -> None:
    payload = json.loads(CORRECTION.read_text(encoding="utf-8"))
    for ancestor, descendant in (
        (payload["scientific_source_commit"], payload["correction_source_commit"]),
        (payload["failure_analysis_commit"], payload["correction_source_commit"]),
        (payload["checklist_guard_commit"], payload["correction_source_commit"]),
    ):
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=ROOT,
            check=True,
        )
