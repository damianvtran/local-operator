"""Evidence finalization and immutable terminal tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.evidence.models import (
    ActionBatchPayload,
    EnvironmentStepPayload,
    FinalizationIntent,
    LifecycleTransitionPayload,
    ModelRequestPayload,
    ObservationPayload,
    OutcomeDraft,
    ScoreArtifact,
    ScoringResultPayload,
)
from local_operator.evaluation.evidence.store import (
    EvidenceBundleInvalid,
    EvidenceTerminal,
    EvidenceWriter,
)
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.receipts import RedactionSet
from tests.unit.evaluation.evidence.test_models import (
    DIGEST,
    OTHER_DIGEST,
    ROUTE,
    manifest,
)


def _append_completed(writer: EvidenceWriter, *, monotonic_ns: int, wall_time_ms: int) -> None:
    writer.record_final_lifecycle(
        LifecycleTransitionPayload(
            previous_state_id=None,
            state_id=OTHER_DIGEST,
            state="completed",
            finalization_id="final",
            preflight_seal_id=DIGEST,
            commitment_id=OTHER_DIGEST,
            reconciliation_id=DIGEST,
            reconciliation_reportable=True,
            score_id=ScoreArtifact(status="scored", binary=0).score_id,
            cleanup_result_id=OTHER_DIGEST,
            rescue_required=False,
        ),
        monotonic_ns=monotonic_ns,
        wall_time_ms=wall_time_ms,
    )


def _finalized(root: Path) -> tuple[EvidenceWriter, ScoreArtifact]:
    writer = EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(()))
    writer.begin_finalization(
        "final",
        "score-op",
        FinalizationIntent(kind="score", scorer_id="scorer", scorer_version="1"),
        monotonic_ns=1,
        wall_time_ms=2,
    )
    score = ScoreArtifact(status="scored", binary=0)
    writer.record_scoring_result(
        ScoringResultPayload(finalization_id="final", scoring_operation_id="score-op", score=score),
        monotonic_ns=2,
        wall_time_ms=3,
    )
    _append_completed(writer, monotonic_ns=3, wall_time_ms=4)
    return writer, score


def _draft(score: ScoreArtifact) -> OutcomeDraft:
    return OutcomeDraft(
        finalization_id="final",
        preflight_seal_id=DIGEST,
        commitment_id=OTHER_DIGEST,
        reconciliation_id=DIGEST,
        cleanup_result_id=OTHER_DIGEST,
        result=score,
        reportability_label="reportable",
        comparability_label="comparable",
        ended_wall_time_ms=5,
    )


def test_seal_recomputes_all_assertions_and_verifies_terminal(tmp_path: Path) -> None:
    writer, score = _finalized(tmp_path / "bundle")
    try:
        outcome = writer.seal(_draft(score))
        with pytest.raises(EvidenceTerminal):
            writer.seal(_draft(score))
        with pytest.raises(EvidenceTerminal):
            writer.abandon("operator_abandoned", "too-late")
    finally:
        writer.close()
    report = verify_bundle(tmp_path / "bundle")
    assert report.valid
    assert report.terminal_state == "sealed"
    assert report.outcome == outcome
    assert report.outcome is not None and report.outcome.result.binary == 0


def test_three_observation_cycles_seal_as_complete_graph(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    writer = EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(()))
    try:
        action_artifact = writer.publish_artifact(b"safe", media_type="text/plain")
        writer.append(
            "observation",
            ObservationPayload(observation_id="observation-0", sequence=0),
            monotonic_ns=1,
            wall_time_ms=1,
        )
        for index in range(2):
            writer.append(
                "action_batch",
                ActionBatchPayload(
                    action_batch_id=f"batch-{index}",
                    observation_id=f"observation-{index}",
                    action_count=1,
                    action_artifact=action_artifact,
                ),
                monotonic_ns=2 + index * 3,
                wall_time_ms=2 + index * 3,
            )
            writer.append(
                "environment_step",
                EnvironmentStepPayload(
                    step_id=f"step-{index}",
                    action_batch_id=f"batch-{index}",
                    receipt_id=DIGEST,
                    input_observation_id=f"observation-{index}",
                    output_observation_id=f"observation-{index + 1}",
                    terminated=index == 1,
                    truncated=False,
                ),
                monotonic_ns=3 + index * 3,
                wall_time_ms=3 + index * 3,
            )
            writer.append(
                "observation",
                ObservationPayload(observation_id=f"observation-{index + 1}", sequence=index + 1),
                monotonic_ns=4 + index * 3,
                wall_time_ms=4 + index * 3,
            )
        writer.begin_finalization(
            "final",
            "score-op",
            FinalizationIntent(kind="score", scorer_id="scorer", scorer_version="1"),
            monotonic_ns=8,
            wall_time_ms=8,
        )
        score = ScoreArtifact(status="scored", binary=1)
        writer.record_scoring_result(
            ScoringResultPayload(
                finalization_id="final", scoring_operation_id="score-op", score=score
            ),
            monotonic_ns=9,
            wall_time_ms=9,
        )
        writer.record_final_lifecycle(
            LifecycleTransitionPayload(
                previous_state_id=None,
                state_id=OTHER_DIGEST,
                state="completed",
                finalization_id="final",
                preflight_seal_id=DIGEST,
                commitment_id=OTHER_DIGEST,
                reconciliation_id=DIGEST,
                reconciliation_reportable=True,
                score_id=score.score_id,
                cleanup_result_id=OTHER_DIGEST,
                rescue_required=False,
            ),
            monotonic_ns=10,
            wall_time_ms=10,
        )
        outcome = writer.seal(_draft(score))
    finally:
        writer.close()
    report = verify_bundle(root)
    assert report.valid and outcome.counters.environment_step_count == 2


def test_seal_rejects_orphan_model_request(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    writer = EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(()))
    try:
        writer.append(
            "model_request",
            ModelRequestPayload(
                request_id="orphan",
                requested_route=ROUTE,
                tool_schema_digest=DIGEST,
                input_tokens=1,
                message_count=1,
                tool_count=0,
            ),
            monotonic_ns=1,
            wall_time_ms=1,
        )
        writer.begin_finalization(
            "final",
            "score-op",
            FinalizationIntent(kind="score", scorer_id="scorer", scorer_version="1"),
            monotonic_ns=2,
            wall_time_ms=2,
        )
        score = ScoreArtifact(status="scored", binary=0)
        writer.record_scoring_result(
            ScoringResultPayload(
                finalization_id="final", scoring_operation_id="score-op", score=score
            ),
            monotonic_ns=3,
            wall_time_ms=3,
        )
        _append_completed(writer, monotonic_ns=4, wall_time_ms=4)
        with pytest.raises(EvidenceBundleInvalid, match="independent seal verification"):
            writer.seal(_draft(score))
    finally:
        writer.close()
    assert "receipt_binding_invalid" in {issue.code for issue in verify_bundle(root).issues}


def test_seal_rejects_forged_finalization_and_lifecycle_score(tmp_path: Path) -> None:
    writer, score = _finalized(tmp_path / "finalization")
    try:
        bad = _draft(score).model_dump(mode="json")
        bad["finalization_id"] = "forged"
        with pytest.raises(EvidenceBundleInvalid, match="durable finalization"):
            writer.seal(OutcomeDraft.model_validate(bad, strict=True))
    finally:
        writer.close()

    root = tmp_path / "lifecycle-score"
    writer = EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(()))
    try:
        writer.begin_finalization(
            "final",
            "score-op",
            FinalizationIntent(kind="score", scorer_id="scorer", scorer_version="1"),
            monotonic_ns=1,
            wall_time_ms=2,
        )
        score = ScoreArtifact(status="scored", binary=0)
        writer.record_scoring_result(
            ScoringResultPayload(
                finalization_id="final", scoring_operation_id="score-op", score=score
            ),
            monotonic_ns=2,
            wall_time_ms=3,
        )
        writer.record_final_lifecycle(
            LifecycleTransitionPayload(
                previous_state_id=None,
                state_id=OTHER_DIGEST,
                state="completed",
                finalization_id="final",
                preflight_seal_id=DIGEST,
                commitment_id=OTHER_DIGEST,
                reconciliation_id=DIGEST,
                reconciliation_reportable=True,
                score_id=ScoreArtifact(status="scored", binary=1).score_id,
                cleanup_result_id=OTHER_DIGEST,
                rescue_required=False,
            ),
            monotonic_ns=3,
            wall_time_ms=4,
        )
        with pytest.raises(EvidenceBundleInvalid, match="lifecycle receipts"):
            writer.seal(_draft(score))
    finally:
        writer.close()


def test_seal_rejects_draft_receipt_or_score_disagreement(tmp_path: Path) -> None:
    writer, score = _finalized(tmp_path / "bundle")
    try:
        bad = _draft(score).model_dump(mode="json")
        bad["commitment_id"] = DIGEST
        with pytest.raises(EvidenceBundleInvalid, match="lifecycle receipts"):
            writer.seal(OutcomeDraft.model_validate(bad, strict=True))
        bad = _draft(score).model_dump(mode="json")
        bad["result"] = ScoreArtifact(status="scored", binary=1).model_dump(mode="json")
        with pytest.raises(EvidenceBundleInvalid, match="durable score"):
            writer.seal(OutcomeDraft.model_validate(bad, strict=True))
    finally:
        writer.close()


def test_outcome_publication_before_state_update_still_derives_sealed(tmp_path: Path) -> None:
    class FailStateWrite:
        def __init__(self) -> None:
            self.fail = False

        def write(self, fd: int, data: bytes) -> int:
            if self.fail and b'"state":"sealed"' in data:
                raise OSError("state-cutpoint")
            return os.write(fd, data)

        def fsync(self, fd: int) -> None:
            os.fsync(fd)

        def link(self, src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int) -> None:
            os.link(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

        def unlink(self, path: str, *, dir_fd: int) -> None:
            os.unlink(path, dir_fd=dir_fd)

    calls = FailStateWrite()
    writer = EvidenceWriter.create(
        tmp_path / "bundle",
        manifest(),
        RedactionSet.from_resolved_values(()),
        syscalls=calls,
    )
    writer.begin_finalization(
        "final",
        "score-op",
        FinalizationIntent(kind="score", scorer_id="scorer", scorer_version="1"),
        monotonic_ns=1,
        wall_time_ms=2,
    )
    score = ScoreArtifact(status="scored", binary=0)
    writer.record_scoring_result(
        ScoringResultPayload(finalization_id="final", scoring_operation_id="score-op", score=score),
        monotonic_ns=2,
        wall_time_ms=3,
    )
    _append_completed(writer, monotonic_ns=3, wall_time_ms=4)
    calls.fail = True
    try:
        with pytest.raises(OSError, match="state-cutpoint"):
            writer.seal(_draft(score))
    finally:
        writer.close()
    report = verify_bundle(tmp_path / "bundle")
    assert report.valid
    assert report.terminal_state == "sealed"
    assert "state_stale" in {issue.code for issue in report.issues}


def test_infrastructure_failure_seals_unscored_not_binary_zero(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    writer = EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(()))
    try:
        writer.begin_finalization(
            "final",
            None,
            FinalizationIntent(kind="unscored"),
        )
        score = ScoreArtifact(status="unscored", reason="infrastructure_failure")
        writer.record_final_lifecycle(
            LifecycleTransitionPayload(
                previous_state_id=None,
                state_id=OTHER_DIGEST,
                state="completed",
                finalization_id="final",
                preflight_seal_id=DIGEST,
                commitment_id=OTHER_DIGEST,
                reconciliation_id=DIGEST,
                reconciliation_reportable=True,
                score_id=score.score_id,
                cleanup_result_id=OTHER_DIGEST,
                rescue_required=False,
            )
        )
        draft = OutcomeDraft(
            finalization_id="final",
            preflight_seal_id=DIGEST,
            commitment_id=OTHER_DIGEST,
            reconciliation_id=DIGEST,
            cleanup_result_id=OTHER_DIGEST,
            result=score,
            reportability_label="infrastructure_failure",
            comparability_label="comparable",
            ended_wall_time_ms=5,
        )
        outcome = writer.seal(draft)
    finally:
        writer.close()
    assert outcome.result.status == "unscored"
    assert outcome.result.binary is None
    assert verify_bundle(root).valid


def test_forged_outcome_artifact_cannot_bless_unreferenced_file(tmp_path: Path) -> None:
    writer, score = _finalized(tmp_path / "bundle")
    try:
        outcome = writer.seal(_draft(score))
    finally:
        writer.close()
    root = tmp_path / "bundle"
    extra = b"unreferenced"
    digest = __import__("hashlib").sha256(extra).hexdigest()
    (root / "artifacts" / digest).write_bytes(extra)
    data = outcome.model_dump(mode="json")
    data["artifacts"].append(
        {"sha256": digest, "media_type": "text/plain", "byte_count": len(extra)}
    )
    data["evidence_root"] = "0" * 64
    forged = type(outcome).model_validate(data, strict=True)
    (root / "outcome.json").write_bytes(forged.to_canonical_json())
    report = verify_bundle(root)
    assert "artifact_unreferenced" in {issue.code for issue in report.issues}
    assert "outcome_mismatch" in {issue.code for issue in report.issues}


def test_terminal_conflict_and_outcome_tamper_are_detected(tmp_path: Path) -> None:
    writer, score = _finalized(tmp_path / "bundle")
    try:
        writer.seal(_draft(score))
    finally:
        writer.close()
    root = tmp_path / "bundle"
    (root / "abandonment.json").write_bytes(b"{}")
    assert "terminal_conflict" in {issue.code for issue in verify_bundle(root).issues}
    (root / "abandonment.json").unlink()
    data: dict[str, Any] = json.loads((root / "outcome.json").read_bytes())
    data["event_count"] += 1
    (root / "outcome.json").write_text(
        json.dumps(data, separators=(",", ":"), sort_keys=True), encoding="utf-8"
    )
    report = verify_bundle(root)
    assert not report.valid
    assert {
        "outcome_noncanonical",
        "outcome_invalid",
        "outcome_mismatch",
        "counter_mismatch",
    } & {issue.code for issue in report.issues}
