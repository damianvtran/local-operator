"""Crash-safe evidence writer tests."""

from __future__ import annotations

import multiprocessing
import os
import signal
import threading
from multiprocessing.synchronize import Event as ProcessEvent
from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.evidence.models import (
    CancelPayload,
    FinalizationIntent,
    ObservationPayload,
    ScoreArtifact,
    ScoringResultPayload,
)
from local_operator.evaluation.evidence.store import (
    EvidenceBundleBusy,
    EvidenceBundleInvalid,
    EvidenceRecoveryOnly,
    EvidenceTerminal,
    EvidenceWriter,
)
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.receipts import RedactionSet
from tests.unit.evaluation.evidence.test_models import DIGEST, manifest


def redactions(*values: str) -> RedactionSet:
    return RedactionSet.from_resolved_values(values)


def _hold_writer(root: str, ready: ProcessEvent, release: ProcessEvent) -> None:
    with EvidenceWriter.create(root, manifest(), redactions()):
        ready.set()
        assert release.wait(10)


def _hold_finalizing(root: str, ready: ProcessEvent, release: ProcessEvent) -> None:
    with EvidenceWriter.create(root, manifest(), redactions()) as writer:
        writer.begin_finalization(
            "final",
            "score-op",
            FinalizationIntent(kind="score", scorer_id="scorer", scorer_version="1"),
            monotonic_ns=1,
            wall_time_ms=2,
        )
        ready.set()
        assert release.wait(30)


def test_create_append_artifact_abandon_and_verify_real_files(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), redactions()) as writer:
        ref = writer.publish_artifact(b"safe", media_type="text/plain")
        event = writer.append(
            "observation",
            ObservationPayload(observation_id="observation", sequence=0, artifacts=(ref,)),
            monotonic_ns=1,
            wall_time_ms=2,
        )
        abandonment = writer.abandon("operator_abandoned", "test-complete")
    assert (root / "manifest.json").stat().st_mode & 0o777 == 0o600
    assert (root / "artifacts" / ref.sha256).read_bytes() == b"safe"
    assert (root / "events.jsonl").read_bytes() == event.to_canonical_json() + b"\n"
    report = verify_bundle(root)
    assert report.valid
    assert report.terminal_state == "abandoned"
    assert report.abandonment == abandonment
    assert report.artifacts == (ref,)


def test_existing_manifest_is_exact_and_execution_never_resumes(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    writer = EvidenceWriter.create(root, manifest(), redactions())
    writer.close()
    with EvidenceWriter.create(root, manifest(), redactions()) as reopened:
        reopened.append(
            "cancel",
            CancelPayload(cancellation_id="cancel", source="operator", diagnostic_code="x"),
            monotonic_ns=1,
            wall_time_ms=2,
        )
    with pytest.raises(EvidenceBundleInvalid, match="cannot resume"):
        EvidenceWriter.create(root, manifest(), redactions())
    changed_data = manifest().model_dump(mode="json")
    changed_data.update(
        {"episode_id": "different", "manifest_digest": "0" * 64, "bundle_id": "0" * 64}
    )
    from local_operator.evaluation.evidence.models import EvidenceManifest

    changed = EvidenceManifest.model_validate(changed_data, strict=True)
    with pytest.raises(EvidenceBundleInvalid, match="manifest"):
        EvidenceWriter.create(root, changed, redactions())


def test_second_process_writer_is_rejected_without_timing_sleep(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(target=_hold_writer, args=(str(tmp_path / "bundle"), ready, release))
    process.start()
    assert ready.wait(10)
    try:
        with pytest.raises(EvidenceBundleBusy):
            EvidenceWriter.create(tmp_path / "bundle", manifest(), redactions())
    finally:
        release.set()
        process.join(10)
    assert process.exitcode == 0


def test_sigkill_owner_death_allows_only_verified_abandonment(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    root = tmp_path / "bundle"
    process = context.Process(target=_hold_finalizing, args=(str(root), ready, release))
    process.start()
    assert ready.wait(10)
    assert process.pid is not None
    os.kill(process.pid, signal.SIGKILL)
    process.join(10)
    assert process.exitcode == -signal.SIGKILL
    with EvidenceWriter.open_for_abandon(root, redactions()) as recovered:
        with pytest.raises(EvidenceRecoveryOnly):
            recovered.append(
                "cancel",
                CancelPayload(cancellation_id="cancel", source="operator", diagnostic_code="x"),
            )
        with pytest.raises(EvidenceRecoveryOnly):
            recovered.begin_finalization("retry", None, FinalizationIntent(kind="unscored"))
        abandonment = recovered.abandon("ambiguous_finalization", "owner-died")
    assert abandonment.reason == "ambiguous_finalization"
    assert verify_bundle(root).terminal_state == "abandoned"


def test_append_and_finalize_share_one_lock_order_without_corruption(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    writer = EvidenceWriter.create(root, manifest(), redactions())
    barrier = threading.Barrier(3)
    results: list[str] = []

    def append() -> None:
        barrier.wait()
        try:
            writer.append(
                "cancel",
                CancelPayload(
                    cancellation_id="cancel",
                    source="operator",
                    diagnostic_code="requested",
                ),
            )
            results.append("append")
        except EvidenceTerminal:
            results.append("append-closed")

    def finalize() -> None:
        barrier.wait()
        writer.begin_finalization(
            "final",
            None,
            FinalizationIntent(kind="unscored"),
        )
        results.append("finalize")

    threads = [threading.Thread(target=append), threading.Thread(target=finalize)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()
    writer.close()
    assert "finalize" in results
    assert set(results) in ({"finalize", "append"}, {"finalize", "append-closed"})
    report = verify_bundle(root)
    assert report.valid and report.terminal_state == "finalizing"


def test_finalizing_marker_precedes_scoring_start_and_closes_execution(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), redactions()) as writer:
        start = writer.begin_finalization(
            "final",
            "score-op",
            FinalizationIntent(kind="score", scorer_id="scorer", scorer_version="1"),
            monotonic_ns=1,
            wall_time_ms=2,
        )
        assert start is not None and start.kind == "scoring_start"
        with pytest.raises(EvidenceTerminal):
            writer.append(
                "cancel",
                CancelPayload(cancellation_id="cancel", source="operator", diagnostic_code="x"),
            )
        score = ScoreArtifact(status="scored", binary=0)
        writer.record_scoring_result(
            ScoringResultPayload(
                finalization_id="final", scoring_operation_id="score-op", score=score
            ),
            monotonic_ns=2,
            wall_time_ms=3,
        )
        with pytest.raises(EvidenceTerminal, match="exactly once"):
            writer.record_scoring_result(
                ScoringResultPayload(
                    finalization_id="final", scoring_operation_id="score-op", score=score
                )
            )


def test_redaction_rejects_all_encoded_canaries_without_echo(tmp_path: Path) -> None:
    secret = "very-secret-value"
    variants = [
        secret,
        "dmVyeS1zZWNyZXQtdmFsdWU=",
        "dmVyeS1zZWNyZXQtdmFsdWU",
        "very-secret-value".encode().hex(),
        "very-secret-value".encode().hex().upper(),
        "very-secret-value".replace("-", "%2D"),
    ]
    with EvidenceWriter.create(tmp_path / "bundle", manifest(), redactions(secret)) as writer:
        for index, variant in enumerate(variants):
            with pytest.raises(EvidenceBundleInvalid) as error:
                writer.append(
                    "error",
                    {
                        "error_id": f"error-{index}",
                        "category": "internal",
                        "diagnostic_code": "redacted",
                        "detail_artifact": None,
                        "retryable": False,
                        "unexpected": variant,
                    },
                )
            assert secret not in str(error.value)
        for index, variant in enumerate(variants):
            with pytest.raises(EvidenceBundleInvalid) as error:
                writer.publish_artifact(variant.encode(), media_type="text/plain")
            assert secret not in str(error.value)
    assert (tmp_path / "bundle" / "events.jsonl").read_bytes() == b""
    assert list((tmp_path / "bundle" / "artifacts").iterdir()) == []


def test_invalid_artifact_expectations_and_media_leave_no_target(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), redactions()) as writer:
        for kwargs in (
            {"expected_sha256": DIGEST},
            {"expected_byte_count": 99},
            {"media_type": "application/json"},
            {"media_type": "image/png"},
        ):
            options: dict[str, Any] = {"media_type": "text/plain", **kwargs}
            with pytest.raises(EvidenceBundleInvalid):
                writer.publish_artifact(b"not-json-or-png", **options)
    assert list((root / "artifacts").iterdir()) == []


def test_journal_fsync_failure_does_not_advance_memory_head(tmp_path: Path) -> None:
    class FailFirstJournalFsync:
        def __init__(self) -> None:
            self.failed = False
            self.armed = False

        def write(self, fd: int, data: bytes) -> int:
            return os.write(fd, data)

        def fsync(self, fd: int) -> None:
            if self.armed and not self.failed:
                self.failed = True
                raise OSError("cutpoint")
            os.fsync(fd)

        def link(self, src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int) -> None:
            os.link(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

        def unlink(self, path: str, *, dir_fd: int) -> None:
            os.unlink(path, dir_fd=dir_fd)

    calls = FailFirstJournalFsync()
    with EvidenceWriter.create(
        tmp_path / "bundle", manifest(), redactions(), syscalls=calls
    ) as writer:
        calls.armed = True
        initial = (writer._sequence, writer._head)  # white-box crash invariant
        with pytest.raises(OSError, match="cutpoint"):
            writer.append(
                "cancel",
                CancelPayload(cancellation_id="cancel", source="operator", diagnostic_code="x"),
                monotonic_ns=1,
                wall_time_ms=2,
            )
        assert (writer._sequence, writer._head) == initial


def test_death_between_finalizing_marker_and_scoring_start_is_abandon_only(
    tmp_path: Path,
) -> None:
    class FailScoringStartWrite:
        def write(self, fd: int, data: bytes) -> int:
            if b'"kind":"scoring_start"' in data:
                raise OSError("journal-cutpoint")
            return os.write(fd, data)

        def fsync(self, fd: int) -> None:
            os.fsync(fd)

        def link(self, src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int) -> None:
            os.link(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

        def unlink(self, path: str, *, dir_fd: int) -> None:
            os.unlink(path, dir_fd=dir_fd)

    root = tmp_path / "bundle"
    writer = EvidenceWriter.create(root, manifest(), redactions(), syscalls=FailScoringStartWrite())
    with pytest.raises(OSError, match="journal-cutpoint"):
        writer.begin_finalization(
            "final",
            "score-op",
            FinalizationIntent(kind="score", scorer_id="scorer", scorer_version="1"),
        )
    writer.close()
    assert verify_bundle(root).terminal_state == "finalizing"
    with EvidenceWriter.open_for_abandon(root, redactions()) as recovered:
        with pytest.raises(EvidenceRecoveryOnly):
            recovered.record_scoring_result(
                ScoringResultPayload(
                    finalization_id="final",
                    scoring_operation_id="score-op",
                    score=ScoreArtifact(status="scored", binary=1),
                )
            )
        recovered.abandon("ambiguous_finalization", "start-not-durable")
    assert verify_bundle(root).terminal_state == "abandoned"


def test_hardlinked_artifact_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), redactions()) as writer:
        ref = writer.publish_artifact(b"safe", media_type="text/plain")
        os.link(root / "artifacts" / ref.sha256, tmp_path / "second-link")
        writer.append(
            "observation",
            ObservationPayload(observation_id="observation", sequence=0, artifacts=(ref,)),
            monotonic_ns=1,
            wall_time_ms=2,
        )
    report = verify_bundle(root)
    assert "artifact_unsafe" in {issue.code for issue in report.issues}
