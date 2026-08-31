"""Evidence finalization and immutable terminal tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import pytest

from local_operator.evaluation.evidence.models import (
    ActionBatchPayload,
    BudgetCommitmentPayload,
    CleanupPayload,
    EnvironmentStepPayload,
    FinalizationIntent,
    LifecycleTransitionPayload,
    ModelRequestPayload,
    ObservationPayload,
    OutcomeDraft,
    PreflightPayload,
    ReconciliationPayload,
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


def _append_provenance(
    writer: EvidenceWriter, *, monotonic_ns: int = 1, wall_time_ms: int = 1
) -> tuple[int, int]:
    writer.append(
        "preflight",
        PreflightPayload(
            sealed_preflight_id=DIGEST,
            plan_id=DIGEST,
            receipt_ids=(OTHER_DIGEST,),
            passed=True,
        ),
        monotonic_ns=monotonic_ns,
        wall_time_ms=wall_time_ms,
    )
    writer.append(
        "budget_commitment",
        BudgetCommitmentPayload(
            commitment_id=OTHER_DIGEST,
            budget_id=OTHER_DIGEST,
            reservation_ids=(DIGEST,),
            reserved_summary_digest=DIGEST,
        ),
        monotonic_ns=monotonic_ns + 1,
        wall_time_ms=wall_time_ms + 1,
    )
    return monotonic_ns + 2, wall_time_ms + 2


def _append_final_receipts(
    writer: EvidenceWriter, *, monotonic_ns: int, wall_time_ms: int
) -> tuple[int, int]:
    writer.record_reconciliation(
        ReconciliationPayload(
            reconciliation_id=DIGEST,
            budget_id=OTHER_DIGEST,
            commitment_id=OTHER_DIGEST,
            reportable=True,
            provider_cost_microusd=0,
            environment_cost_microusd=0,
            total_cost_microusd=0,
        ),
        monotonic_ns=monotonic_ns,
        wall_time_ms=wall_time_ms,
    )
    writer.record_cleanup(
        CleanupPayload(
            cleanup_result_id=OTHER_DIGEST,
            cleanup_plan_id=DIGEST,
            receipt_ids=(DIGEST,),
            rescue_required=False,
        ),
        monotonic_ns=monotonic_ns + 1,
        wall_time_ms=wall_time_ms + 1,
    )
    return monotonic_ns + 2, wall_time_ms + 2


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
    monotonic_ns, wall_time_ms = _append_provenance(writer)
    writer.begin_finalization(
        "final",
        "score-op",
        FinalizationIntent(kind="score", scorer_id="scorer", scorer_version="1"),
        monotonic_ns=monotonic_ns,
        wall_time_ms=wall_time_ms,
    )
    score = ScoreArtifact(status="scored", binary=0)
    writer.record_scoring_result(
        ScoringResultPayload(finalization_id="final", scoring_operation_id="score-op", score=score),
        monotonic_ns=monotonic_ns + 1,
        wall_time_ms=wall_time_ms + 1,
    )
    monotonic_ns, wall_time_ms = _append_final_receipts(
        writer, monotonic_ns=monotonic_ns + 2, wall_time_ms=wall_time_ms + 2
    )
    _append_completed(writer, monotonic_ns=monotonic_ns, wall_time_ms=wall_time_ms)
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
        _append_provenance(writer, monotonic_ns=0, wall_time_ms=0)
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
        _append_final_receipts(writer, monotonic_ns=10, wall_time_ms=10)
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
            monotonic_ns=12,
            wall_time_ms=12,
        )
        outcome = writer.seal(_draft(score))
    finally:
        writer.close()
    report = verify_bundle(root)
    assert report.valid and outcome.counters.environment_step_count == 2


def test_finish_action_allows_terminal_after_nonterminal_step(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    writer = EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(()))
    try:
        _append_provenance(writer, monotonic_ns=0, wall_time_ms=0)
        action_artifact = writer.publish_artifact(b"safe", media_type="text/plain")
        writer.append(
            "observation",
            ObservationPayload(observation_id="observation-0", sequence=0),
            monotonic_ns=1,
            wall_time_ms=1,
        )
        writer.append(
            "action_batch",
            ActionBatchPayload(
                action_batch_id="batch-0",
                observation_id="observation-0",
                action_count=1,
                action_artifact=action_artifact,
            ),
            monotonic_ns=2,
            wall_time_ms=2,
        )
        writer.append(
            "environment_step",
            EnvironmentStepPayload(
                step_id="step-0",
                action_batch_id="batch-0",
                receipt_id=DIGEST,
                input_observation_id="observation-0",
                output_observation_id="observation-1",
                terminated=False,
                truncated=False,
            ),
            monotonic_ns=3,
            wall_time_ms=3,
        )
        writer.append(
            "observation",
            ObservationPayload(observation_id="observation-1", sequence=1),
            monotonic_ns=4,
            wall_time_ms=4,
        )
        writer.append(
            "action_batch",
            ActionBatchPayload(
                action_batch_id="finish",
                observation_id="observation-1",
                action_count=1,
                action_artifact=action_artifact,
                terminal="finish",
            ),
            monotonic_ns=5,
            wall_time_ms=5,
        )
        writer.begin_finalization(
            "final",
            "score-op",
            FinalizationIntent(kind="score", scorer_id="scorer", scorer_version="1"),
            monotonic_ns=6,
            wall_time_ms=6,
        )
        score = ScoreArtifact(status="scored", binary=1)
        writer.record_scoring_result(
            ScoringResultPayload(
                finalization_id="final", scoring_operation_id="score-op", score=score
            ),
            monotonic_ns=7,
            wall_time_ms=7,
        )
        _append_final_receipts(writer, monotonic_ns=8, wall_time_ms=8)
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
    assert outcome.result.binary == 1 and verify_bundle(root).valid


@pytest.mark.parametrize(
    ("missing", "replacement"),
    [
        ("preflight", None),
        ("budget_commitment", None),
        ("reconciliation", None),
        ("cleanup", None),
        ("preflight", "wrong_plan"),
        ("cleanup", "wrong_plan"),
    ],
)
def test_seal_rejects_missing_or_mismatched_provenance(
    tmp_path: Path, missing: str, replacement: str | None
) -> None:
    root = tmp_path / "bundle"
    writer, score = _finalized(root)
    writer.close()
    lines = (root / "events.jsonl").read_bytes().splitlines()
    selected = []
    for line in lines:
        decoded = json.loads(line)
        if decoded["kind"] == missing:
            if replacement == "wrong_plan":
                key = "plan_id" if missing == "preflight" else "cleanup_plan_id"
                decoded["payload"][key] = "f" * 64
                decoded["event_id"] = "0" * 64
                from local_operator.evaluation.evidence.models import EventRecord

                selected.append(
                    EventRecord.model_validate(decoded, strict=True).to_canonical_json()
                )
            continue
        selected.append(line)
    (root / "events.jsonl").write_bytes(b"\n".join(selected) + b"\n")
    report = verify_bundle(root)
    assert not report.valid
    issue_codes = {issue.code for issue in report.issues}
    if replacement == "wrong_plan" or missing in {"preflight", "budget_commitment"}:
        assert "receipt_binding_invalid" in issue_codes
    else:
        assert {"event_sequence_mismatch", "event_chain_mismatch"} & issue_codes
    with pytest.raises(EvidenceBundleInvalid):
        EvidenceWriter.open_for_abandon(root, RedactionSet.from_resolved_values(()))


def test_seal_rejects_orphan_model_request(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    writer = EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(()))
    try:
        _append_provenance(writer, monotonic_ns=0, wall_time_ms=0)
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
        _append_final_receipts(writer, monotonic_ns=4, wall_time_ms=4)
        _append_completed(writer, monotonic_ns=6, wall_time_ms=6)
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
        _append_provenance(writer, monotonic_ns=0, wall_time_ms=0)
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
            monotonic_ns=2,
            wall_time_ms=3,
        )
        _append_final_receipts(writer, monotonic_ns=3, wall_time_ms=4)
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
            monotonic_ns=5,
            wall_time_ms=6,
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
    _append_provenance(writer, monotonic_ns=0, wall_time_ms=0)
    writer.begin_finalization(
        "final",
        "score-op",
        FinalizationIntent(kind="score", scorer_id="scorer", scorer_version="1"),
        monotonic_ns=2,
        wall_time_ms=2,
    )
    score = ScoreArtifact(status="scored", binary=0)
    writer.record_scoring_result(
        ScoringResultPayload(finalization_id="final", scoring_operation_id="score-op", score=score),
        monotonic_ns=2,
        wall_time_ms=3,
    )
    _append_final_receipts(writer, monotonic_ns=4, wall_time_ms=4)
    _append_completed(writer, monotonic_ns=6, wall_time_ms=6)
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


@pytest.mark.parametrize(
    ("state", "reason", "failure_kind"),
    [
        ("failed", "crash", "crash"),
        ("cancelled", "cancelled", "cancelled"),
    ],
)
def test_post_running_terminal_without_score_keeps_reconcile_cleanup_order(
    tmp_path: Path,
    state: Literal["failed", "cancelled"],
    reason: Literal["crash", "cancelled"],
    failure_kind: Literal["crash", "cancelled"],
) -> None:
    root = tmp_path / "bundle"
    writer = EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(()))
    try:
        _append_provenance(writer, monotonic_ns=0, wall_time_ms=0)
        writer.append(
            "observation",
            ObservationPayload(observation_id="observation-0", sequence=0),
            monotonic_ns=1,
            wall_time_ms=1,
        )
        writer.begin_finalization(
            "final",
            None,
            FinalizationIntent(kind="unscored"),
            monotonic_ns=2,
            wall_time_ms=2,
        )
        score = ScoreArtifact(status="unscored", reason=reason)
        _append_final_receipts(writer, monotonic_ns=3, wall_time_ms=3)
        writer.record_final_lifecycle(
            LifecycleTransitionPayload(
                previous_state_id=None,
                state_id=OTHER_DIGEST,
                state=state,
                finalization_id="final",
                preflight_seal_id=DIGEST,
                commitment_id=OTHER_DIGEST,
                reconciliation_id=DIGEST,
                reconciliation_reportable=True,
                score_id=score.score_id,
                cleanup_result_id=OTHER_DIGEST,
                rescue_required=False,
                failure_kind=failure_kind,
            ),
            monotonic_ns=5,
            wall_time_ms=5,
        )
    finally:
        writer.close()
    report = verify_bundle(root)
    assert report.valid
    assert [event.kind for event in report.events][-4:] == [
        "finalization_start",
        "reconciliation",
        "cleanup",
        "lifecycle_transition",
    ]


def test_early_preflight_failure_has_no_unused_authorities(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    writer = EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(()))
    try:
        writer.append(
            "preflight",
            PreflightPayload(
                sealed_preflight_id=DIGEST,
                plan_id=DIGEST,
                receipt_ids=(),
                passed=False,
            ),
            monotonic_ns=0,
            wall_time_ms=0,
        )
        # Early failure has no budget commitment, scorer, reconciliation, or cleanup.
        writer.begin_finalization(
            "final",
            None,
            FinalizationIntent(kind="unscored"),
            monotonic_ns=2,
            wall_time_ms=2,
        )
        writer.record_final_lifecycle(
            LifecycleTransitionPayload(
                previous_state_id=None,
                state_id=OTHER_DIGEST,
                state="failed",
                finalization_id="final",
                preflight_seal_id=DIGEST,
                failure_kind="preflight",
            ),
            monotonic_ns=3,
            wall_time_ms=3,
        )
    finally:
        writer.close()
    report = verify_bundle(root)
    assert report.valid
    assert [event.kind for event in report.events] == [
        "preflight",
        "finalization_start",
        "lifecycle_transition",
    ]


def test_infrastructure_failure_seals_unscored_not_binary_zero(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    writer = EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(()))
    try:
        _append_provenance(writer, monotonic_ns=0, wall_time_ms=0)
        writer.begin_finalization(
            "final",
            None,
            FinalizationIntent(kind="unscored"),
            monotonic_ns=1,
            wall_time_ms=1,
        )
        score = ScoreArtifact(status="unscored", reason="infrastructure_failure")
        _append_final_receipts(writer, monotonic_ns=2, wall_time_ms=2)
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
                failure_kind="crash",
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


@pytest.mark.parametrize(
    "mutation",
    [
        {"finalization_id": "forged"},
        {"intent": "unscored", "scoring_operation_id": None},
        {"scoring_operation_id": "forged-op"},
    ],
)
def test_sealed_verifier_rejects_forged_finalization_start(
    tmp_path: Path, mutation: dict[str, Any]
) -> None:
    root = tmp_path / "bundle"
    writer, score = _finalized(root)
    try:
        writer.seal(_draft(score))
    finally:
        writer.close()
    lines = (root / "events.jsonl").read_bytes().splitlines()
    records = [json.loads(line) for line in lines]
    start_index = next(
        index for index, record in enumerate(records) if record["kind"] == "finalization_start"
    )
    records[start_index]["payload"].update(mutation)
    records[start_index]["payload"]["intent_digest"] = "0" * 64
    previous = manifest().manifest_digest
    rewritten = []
    from local_operator.evaluation.evidence.models import EventRecord

    for sequence, record in enumerate(records):
        record.update(
            {
                "sequence": sequence,
                "previous_event_sha256": previous,
                "event_id": "0" * 64,
            }
        )
        event = EventRecord.model_validate(record, strict=True)
        rewritten.append(event.to_canonical_json())
        previous = event.event_id
    (root / "events.jsonl").write_bytes(b"\n".join(rewritten) + b"\n")
    report = verify_bundle(root)
    assert not report.valid
    assert "finalization_invalid" in {issue.code for issue in report.issues}


def test_sealed_verifier_rejects_invalid_finalization_start_digest(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    writer, score = _finalized(root)
    try:
        writer.seal(_draft(score))
    finally:
        writer.close()
    lines = (root / "events.jsonl").read_bytes().splitlines()
    start = next(json.loads(line) for line in lines if b'"kind":"finalization_start"' in line)
    start["payload"]["intent_digest"] = "f" * 64
    with pytest.raises(ValueError, match="finalization start identity"):
        from local_operator.evaluation.evidence.models import FinalizationStartPayload

        FinalizationStartPayload.model_validate(start["payload"], strict=True)


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
