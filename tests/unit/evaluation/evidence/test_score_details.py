"""Only the active scoring receipt may publish artifacts during finalization."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from local_operator.evaluation.evidence.models import (
    EvidenceArtifactRef,
    FinalizationIntent,
    LifecycleTransitionPayload,
    ScoreArtifact,
    ScoringResultPayload,
)
from local_operator.evaluation.evidence.store import (
    EvidenceBundleInvalid,
    EvidenceRecoveryOnly,
    EvidenceTerminal,
    EvidenceWriter,
)
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.receipts import RedactionSet
from tests.unit.evaluation.evidence.test_finalization import (
    _append_final_receipts,
    _append_provenance,
    _draft,
)
from tests.unit.evaluation.evidence.test_models import DIGEST, OTHER_DIGEST, manifest
from tests.unit.evaluation.evidence.test_store import _OSCallsForTest


def _payload(data: bytes, *, media_type: str = "application/json") -> ScoringResultPayload:
    return ScoringResultPayload(
        finalization_id="final",
        scoring_operation_id="score-op",
        score=ScoreArtifact(
            status="scored",
            binary=0,
            partial_ppm=500_000,
            details=EvidenceArtifactRef(
                sha256=hashlib.sha256(data).hexdigest(), media_type=media_type, byte_count=len(data)
            ),
        ),
    )


def _complete(writer: EvidenceWriter, score: ScoreArtifact) -> None:
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
        monotonic_ns=7,
        wall_time_ms=7,
    )


def _begin(writer: EvidenceWriter) -> None:
    _append_provenance(writer)
    writer.begin_finalization(
        "final",
        "score-op",
        FinalizationIntent(kind="score", scorer_id="scorer", scorer_version="1"),
        monotonic_ns=3,
        wall_time_ms=3,
    )


def test_scoring_detail_seals_unchanged_and_only_once(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    data = b'{"score":0.5,"summary":[true,false]}'
    payload = _payload(data)
    with EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(())) as writer:
        _begin(writer)
        with pytest.raises(EvidenceTerminal):
            writer.publish_artifact(data, media_type="application/json")
        writer.record_scoring_result(payload, details_source=data, monotonic_ns=4, wall_time_ms=4)
        with pytest.raises(EvidenceTerminal):
            writer.record_scoring_result(payload, details_source=data)
        _append_final_receipts(writer, monotonic_ns=5, wall_time_ms=5)
        _complete(writer, payload.score)
        writer.seal(_draft(payload.score))
    report = verify_bundle(root)
    assert report.valid, [i.code for i in report.issues]
    assert report.outcome is not None and report.outcome.result == payload.score
    assert payload.score.details is not None
    assert (root / "artifacts" / payload.score.details.sha256).read_bytes() == data


@pytest.mark.parametrize("case", ["digest", "count", "media", "secret", "operation", "no-ref"])
def test_detail_rejection_is_nonmutating_and_retryable(tmp_path: Path, case: str) -> None:
    root = tmp_path / "bundle"
    data = b'{"score":0.5}'
    payload = _payload(data)
    source = data
    if case == "digest":
        source = b'{"score":0.6}'
    elif case == "count":
        ref = payload.score.details
        assert ref is not None
        score = ScoreArtifact(
            status="scored",
            binary=0,
            details=EvidenceArtifactRef(
                sha256=ref.sha256, media_type=ref.media_type, byte_count=len(data) + 1
            ),
        )
        payload = ScoringResultPayload(
            finalization_id="final", scoring_operation_id="score-op", score=score
        )
    elif case == "media":
        source = b"not-json"
        payload = _payload(source)
    elif case == "secret":
        source = b'{"secret":"canary-score-evidence-47329"}'
        payload = _payload(source)
    elif case == "operation":
        payload = payload.model_copy(update={"scoring_operation_id": "wrong-op"})
    elif case == "no-ref":
        payload = ScoringResultPayload(
            finalization_id="final",
            scoring_operation_id="score-op",
            score=ScoreArtifact(status="scored", binary=0),
        )
    redactions = RedactionSet.from_resolved_values(("canary-score-evidence-47329",))
    with EvidenceWriter.create(root, manifest(), redactions) as writer:
        _begin(writer)
        before = (root / "events.jsonl").read_bytes()
        with pytest.raises(EvidenceBundleInvalid):
            writer.record_scoring_result(payload, details_source=source)
        assert (root / "events.jsonl").read_bytes() == before
        assert list((root / "artifacts").iterdir()) == []
        # Retrying publication is safe after a deterministic rejection; it does
        # not rerun the external evaluator or mint a second scoring operation.
        writer.record_scoring_result(_payload(data), details_source=data)
        assert verify_bundle(root).valid


def test_missing_details_cannot_seal(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    payload = _payload(b'{"score":0.5}')
    with EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(())) as writer:
        _begin(writer)
        # Compatibility: callers may still prepublish while open. A ref alone
        # is not proof of those bytes, and the independent seal gate catches it.
        writer.record_scoring_result(payload, monotonic_ns=4, wall_time_ms=4)
        _append_final_receipts(writer, monotonic_ns=5, wall_time_ms=5)
        _complete(writer, payload.score)
        with pytest.raises(EvidenceBundleInvalid):
            writer.seal(_draft(payload.score))
    assert not verify_bundle(root).valid


@pytest.mark.parametrize("cutpoint", [1, 2, 3])
def test_score_publication_fsync_cutpoints_are_recovery_only(tmp_path: Path, cutpoint: int) -> None:
    class CutpointCalls(_OSCallsForTest):
        armed = False
        calls = 0

        def fsync(self, fd: int) -> None:
            if self.armed:
                self.calls += 1
                if self.calls == cutpoint:
                    raise OSError("score-detail-cutpoint")
            super().fsync(fd)

    calls = CutpointCalls()
    root = tmp_path / "bundle"
    data = b'{"score":0.5}'
    payload = _payload(data)
    with EvidenceWriter.create(
        root, manifest(), RedactionSet.from_resolved_values(()), syscalls=calls
    ) as writer:
        _begin(writer)
        calls.armed = True
        with pytest.raises(OSError, match="score-detail-cutpoint"):
            writer.record_scoring_result(payload, details_source=data)
        assert calls.calls == cutpoint
        with pytest.raises(EvidenceRecoveryOnly):
            writer.record_scoring_result(payload, details_source=data)
        with pytest.raises(EvidenceRecoveryOnly):
            writer.seal(_draft(payload.score))
    assert not (root / "outcome.json").exists()
