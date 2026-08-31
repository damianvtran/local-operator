"""Crash-safe evidence writer tests."""

from __future__ import annotations

import base64
import multiprocessing
import os
import signal
import stat
import threading
import tracemalloc
from multiprocessing.synchronize import Event as ProcessEvent
from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.evidence.models import (
    BudgetCommitmentPayload,
    CancelPayload,
    FinalizationIntent,
    ObservationPayload,
    PreflightPayload,
    ScoreArtifact,
    ScoringResultPayload,
)
from local_operator.evaluation.evidence.store import (
    MAX_ARTIFACT_BYTES,
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


def _append_authority(writer: EvidenceWriter, *, timestamp: int = 0) -> None:
    writer.append(
        "preflight",
        PreflightPayload(
            sealed_preflight_id=DIGEST,
            plan_id=DIGEST,
            receipt_ids=(),
            passed=True,
        ),
        monotonic_ns=timestamp,
        wall_time_ms=timestamp,
    )
    writer.append(
        "budget_commitment",
        BudgetCommitmentPayload(
            commitment_id=DIGEST,
            budget_id=manifest().budget_id,
            reservation_ids=(),
            reserved_summary_digest=DIGEST,
        ),
        monotonic_ns=timestamp,
        wall_time_ms=timestamp,
    )


class _OSCallsForTest:
    def write(self, fd: int, data: bytes) -> int:
        return os.write(fd, data)

    def fsync(self, fd: int) -> None:
        os.fsync(fd)

    def link(self, src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int) -> None:
        os.link(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    def unlink(self, path: str, *, dir_fd: int) -> None:
        os.unlink(path, dir_fd=dir_fd)

    def close(self, fd: int) -> None:
        os.close(fd)


def _hold_writer(root: str, ready: ProcessEvent, release: ProcessEvent) -> None:
    with EvidenceWriter.create(root, manifest(), redactions()):
        ready.set()
        assert release.wait(10)


def _hold_finalizing(root: str, ready: ProcessEvent, release: ProcessEvent) -> None:
    with EvidenceWriter.create(root, manifest(), redactions()) as writer:
        _append_authority(writer)
        writer.begin_finalization(
            "final",
            "score-op",
            FinalizationIntent(kind="score", scorer_id="scorer", scorer_version="1"),
            monotonic_ns=1,
            wall_time_ms=2,
        )
        ready.set()
        assert release.wait(30)


def test_immutable_publication_fsyncs_target_and_temp_removal(tmp_path: Path) -> None:
    class RecordingCalls(_OSCallsForTest):
        def __init__(self) -> None:
            self.order: list[str] = []

        def write(self, fd: int, data: bytes) -> int:
            self.order.append("write")
            return super().write(fd, data)

        def fsync(self, fd: int) -> None:
            self.order.append("fsync")
            super().fsync(fd)

        def link(self, src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int) -> None:
            self.order.append("link")
            super().link(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

        def unlink(self, path: str, *, dir_fd: int) -> None:
            self.order.append("unlink")
            super().unlink(path, dir_fd=dir_fd)

    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    calls = RecordingCalls()
    try:
        EvidenceWriter._create_immutable(fd, "manifest.json", b"{}", calls)
    finally:
        os.close(fd)
    assert calls.order == ["write", "fsync", "link", "fsync", "unlink", "fsync"]
    assert sorted(path.name for path in root.iterdir()) == ["manifest.json"]


@pytest.mark.parametrize("failure_index", [1, 2])
def test_immutable_cleanup_cutpoints_never_claim_success(
    tmp_path: Path, failure_index: int
) -> None:
    class CleanupCutpoint(_OSCallsForTest):
        def __init__(self) -> None:
            self.directory_fsyncs = 0

        def fsync(self, fd: int) -> None:
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                self.directory_fsyncs += 1
                if failure_index == 2 and self.directory_fsyncs == 2:
                    raise OSError("directory-cutpoint")
            super().fsync(fd)

        def unlink(self, path: str, *, dir_fd: int) -> None:
            if failure_index == 1 and self.directory_fsyncs == 1:
                raise OSError("unlink-cutpoint")
            super().unlink(path, dir_fd=dir_fd)

    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(EvidenceBundleInvalid):
            EvidenceWriter._create_immutable(fd, "outcome.json", b"{}", CleanupCutpoint())
    finally:
        os.close(fd)
    assert (root / "outcome.json").exists()


def test_create_append_artifact_abandon_and_verify_real_files(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), redactions()) as writer:
        ref = writer.publish_artifact(b"safe", media_type="text/plain")
        writer.append(
            "preflight",
            PreflightPayload(
                sealed_preflight_id=DIGEST,
                plan_id=DIGEST,
                receipt_ids=(),
                passed=False,
            ),
            monotonic_ns=1,
            wall_time_ms=2,
        )
        event = writer.append(
            "error",
            {
                "error_id": "preflight",
                "category": "infrastructure",
                "diagnostic_code": "failed",
                "detail_artifact": ref.model_dump(mode="json"),
                "retryable": False,
            },
            monotonic_ns=2,
            wall_time_ms=3,
        )
        abandonment = writer.abandon("preflight_failure", "test-complete")
    assert (root / "manifest.json").stat().st_mode & 0o777 == 0o600
    assert (root / "artifacts" / ref.sha256).read_bytes() == b"safe"
    assert (root / "events.jsonl").read_bytes().endswith(event.to_canonical_json() + b"\n")
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
    _append_authority(writer)
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


def test_invalid_finalization_leaves_open_bytes_and_allows_retry(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), redactions()) as writer:
        before_state = (root / "state.json").read_bytes()
        before_journal = (root / "events.jsonl").read_bytes()
        before_head = (writer._sequence, writer._head)
        with pytest.raises(EvidenceBundleInvalid, match="finalization start.*phase"):
            writer.begin_finalization(
                "final",
                "score-op",
                FinalizationIntent(kind="score", scorer_id="scorer", scorer_version="1"),
                monotonic_ns=1,
                wall_time_ms=1,
            )
        assert (root / "state.json").read_bytes() == before_state
        assert (root / "events.jsonl").read_bytes() == before_journal
        assert (writer._sequence, writer._head) == before_head
        _append_authority(writer)
        event = writer.begin_finalization(
            "final",
            "score-op",
            FinalizationIntent(kind="score", scorer_id="scorer", scorer_version="1"),
            monotonic_ns=1,
            wall_time_ms=1,
        )
        assert event is not None and event.kind == "scoring_start"


def test_invalid_intent_does_not_mutate_open_bundle(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), redactions()) as writer:
        _append_authority(writer)
        before = (
            (root / "state.json").read_bytes(),
            (root / "events.jsonl").read_bytes(),
            writer._sequence,
            writer._head,
        )
        with pytest.raises(EvidenceBundleInvalid, match="operation ID"):
            writer.begin_finalization(
                "final",
                None,
                FinalizationIntent(kind="score", scorer_id="scorer", scorer_version="1"),
            )
        assert before == (
            (root / "state.json").read_bytes(),
            (root / "events.jsonl").read_bytes(),
            writer._sequence,
            writer._head,
        )


def test_finalizing_marker_precedes_scoring_start_and_closes_execution(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), redactions()) as writer:
        _append_authority(writer)
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


def test_binary_artifacts_reject_whitespace_encoded_canaries(tmp_path: Path) -> None:
    secret = "very-secret-value"
    standard = base64.b64encode(secret.encode()).decode()
    urlsafe = base64.urlsafe_b64encode(secret.encode()).decode()
    hexed = secret.encode().hex()
    variants = (
        "\n".join(standard[index : index + 4] for index in range(0, len(standard), 4)),
        "\t".join(urlsafe[index : index + 3] for index in range(0, len(urlsafe), 3)),
        " ".join(hexed[index : index + 2] for index in range(0, len(hexed), 2)),
    )
    with EvidenceWriter.create(tmp_path / "bundle", manifest(), redactions(secret)) as writer:
        for variant in variants:
            with pytest.raises(EvidenceBundleInvalid) as error:
                writer.publish_artifact(
                    b"\x00header\x00" + variant.encode() + b"\x00tail",
                    media_type="application/octet-stream",
                )
            assert secret not in str(error.value)
    assert list((tmp_path / "bundle" / "artifacts").iterdir()) == []


@pytest.mark.parametrize(
    "payload",
    [
        b"x" * 70_000 + b"very-secret-value" + b"tail",
        b"prefix-very%2dsecret-value-tail",
        b"prefix-%76%65%72%79%2D%73%65%63%72%65%74%2D%76%61%6C%75%65-tail",
    ],
)
def test_scanner_rejects_reviewer_payloads_in_one_large_feed(
    tmp_path: Path, payload: bytes
) -> None:
    with EvidenceWriter.create(
        tmp_path / "bundle", manifest(), redactions("very-secret-value")
    ) as writer:
        with pytest.raises(EvidenceBundleInvalid):
            writer.publish_artifact(payload, media_type="application/octet-stream")


def test_percent_scanner_preserves_state_across_every_split(tmp_path: Path) -> None:
    payload = b"prefix-%76%65%72%79%2Dsecret-value-tail"
    for split in range(1, len(payload)):
        root = tmp_path / f"bundle-{split}"
        with EvidenceWriter.create(root, manifest(), redactions("very-secret-value")) as writer:
            with pytest.raises(EvidenceBundleInvalid):
                writer.publish_artifact(
                    (payload[:split], payload[split:]),
                    media_type="application/octet-stream",
                )


def test_percent_scanner_incomplete_escape_is_literal_and_case_sensitive(
    tmp_path: Path,
) -> None:
    with EvidenceWriter.create(
        tmp_path / "bundle", manifest(), redactions("Very-Secret-Value")
    ) as writer:
        writer.publish_artifact(
            (b"lowercase-very-secret-value-%", b"2"),
            media_type="application/octet-stream",
        )


@pytest.mark.parametrize("suffix", [b"%", b"%7"])
def test_terminal_incomplete_percent_is_flushed_as_literal(tmp_path: Path, suffix: bytes) -> None:
    with EvidenceWriter.create(
        tmp_path / "bundle", manifest(), redactions("benign-value")
    ) as writer:
        ref = writer.publish_artifact(b"benign" + suffix, media_type="application/octet-stream")
    assert ref.byte_count == len(b"benign" + suffix)


def test_fully_encoded_prefix_plus_trailing_percent_still_rejects(
    tmp_path: Path,
) -> None:
    payload = b"%76%65%72%79%2D%73%65%63%72%65%74%2D%76%61%6C%75%65%"
    with EvidenceWriter.create(
        tmp_path / "bundle", manifest(), redactions("very-secret-value")
    ) as writer:
        with pytest.raises(EvidenceBundleInvalid):
            writer.publish_artifact(payload, media_type="application/octet-stream")
    assert list((tmp_path / "bundle" / "artifacts").iterdir()) == []


def test_scanner_finish_is_idempotent_and_feed_after_finish_rejects(
    tmp_path: Path,
) -> None:
    from local_operator.evaluation.evidence.store import _RedactionScanner

    with EvidenceWriter.create(
        tmp_path / "bundle", manifest(), redactions("very-secret-value")
    ) as writer:
        scanner = _RedactionScanner(writer)
        scanner.feed(b"safe%")
        scanner.finish()
        retained = scanner.retained_bytes
        scanner.finish()
        assert scanner.retained_bytes == retained
        with pytest.raises(EvidenceBundleInvalid, match="finalized"):
            scanner.feed(b"later")


def test_publish_calls_scanner_finish_before_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.evaluation.evidence import store

    called = False
    original = store._RedactionScanner.finish

    def finish(self: Any) -> None:
        nonlocal called
        called = True
        original(self)

    monkeypatch.setattr(store._RedactionScanner, "finish", finish)
    with EvidenceWriter.create(tmp_path / "bundle", manifest(), redactions()) as writer:
        writer.publish_artifact(b"safe", media_type="application/octet-stream")
    assert called


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
    _append_authority(writer)
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


def test_journal_failure_poison_blocks_every_later_authority(tmp_path: Path) -> None:
    class PartialThenFail:
        def __init__(self) -> None:
            self.armed = False
            self.wrote_partial = False

        def write(self, fd: int, data: bytes) -> int:
            if self.armed and not self.wrote_partial:
                self.wrote_partial = True
                return os.write(fd, data[:7])
            if self.armed:
                raise OSError("journal-failure")
            return os.write(fd, data)

        def fsync(self, fd: int) -> None:
            os.fsync(fd)

        def link(self, src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int) -> None:
            os.link(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

        def unlink(self, path: str, *, dir_fd: int) -> None:
            os.unlink(path, dir_fd=dir_fd)

    calls = PartialThenFail()
    root = tmp_path / "bundle"
    writer = EvidenceWriter.create(root, manifest(), redactions(), syscalls=calls)
    calls.armed = True
    initial = (writer._sequence, writer._head)
    with pytest.raises(OSError, match="journal-failure"):
        writer.append(
            "cancel",
            CancelPayload(cancellation_id="cancel", source="operator", diagnostic_code="x"),
        )
    partial = (root / "events.jsonl").read_bytes()
    assert partial and not partial.endswith(b"\n")
    assert (writer._sequence, writer._head) == initial
    with pytest.raises(EvidenceRecoveryOnly):
        writer.append(
            "cancel",
            CancelPayload(cancellation_id="later", source="operator", diagnostic_code="x"),
        )
    with pytest.raises(EvidenceRecoveryOnly):
        writer.begin_finalization("final", None, FinalizationIntent(kind="unscored"))
    with pytest.raises(EvidenceRecoveryOnly):
        writer.seal(_outcome_draft_for_poison())
    assert (root / "events.jsonl").read_bytes() == partial
    writer.close()


def _outcome_draft_for_poison() -> Any:
    from local_operator.evaluation.evidence.models import OutcomeDraft

    return OutcomeDraft(
        finalization_id="final",
        preflight_seal_id=DIGEST,
        commitment_id=DIGEST,
        reconciliation_id=DIGEST,
        cleanup_result_id=DIGEST,
        result=ScoreArtifact(status="unscored", reason="crash"),
        reportability_label="unscored",
        comparability_label="comparable",
        ended_wall_time_ms=1,
    )


def test_fork_child_cannot_use_or_close_parent_writer(tmp_path: Path) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork start method unavailable")
    root = tmp_path / "bundle"
    writer = EvidenceWriter.create(root, manifest(), redactions())
    context = multiprocessing.get_context("fork")
    result = context.Queue()

    def child() -> None:
        try:
            writer.append(
                "cancel",
                CancelPayload(cancellation_id="child", source="operator", diagnostic_code="x"),
            )
        except EvidenceRecoveryOnly:
            result.put("rejected")
        finally:
            writer.close()

    process = context.Process(target=child)
    process.start()
    process.join(10)
    assert process.exitcode == 0
    assert result.get(timeout=5) == "rejected"
    writer.append(
        "cancel",
        CancelPayload(cancellation_id="parent", source="operator", diagnostic_code="x"),
        monotonic_ns=1,
        wall_time_ms=2,
    )
    writer.close()
    report = verify_bundle(root)
    assert report.valid and len(report.events) == 1


@pytest.mark.parametrize("terminal", ["outcome.json", "abandonment.json"])
def test_create_never_reopens_empty_terminal_bundle(tmp_path: Path, terminal: str) -> None:
    root = tmp_path / "bundle"
    writer = EvidenceWriter.create(root, manifest(), redactions())
    writer.close()
    state_before = (root / "state.json").read_bytes()
    (root / terminal).write_bytes(b"{}")
    with pytest.raises(EvidenceTerminal):
        EvidenceWriter.create(root, manifest(), redactions())
    assert (root / "state.json").read_bytes() == state_before
    assert (root / "events.jsonl").read_bytes() == b""


def test_unsafe_root_or_artifact_permissions_are_rejected(tmp_path: Path) -> None:
    unsafe_root = tmp_path / "unsafe-root"
    unsafe_root.mkdir(mode=0o700)
    unsafe_root.chmod(0o777)
    with pytest.raises(EvidenceBundleInvalid, match="permissions"):
        EvidenceWriter.create(unsafe_root, manifest(), redactions())

    root = tmp_path / "bundle"
    writer = EvidenceWriter.create(root, manifest(), redactions())
    writer.close()
    (root / "artifacts").chmod(0o777)
    with pytest.raises(EvidenceBundleInvalid, match="permissions"):
        EvidenceWriter.create(root, manifest(), redactions())


def test_closed_writer_fork_does_not_close_reused_descriptors(tmp_path: Path) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork start method unavailable")
    writer = EvidenceWriter.create(tmp_path / "bundle", manifest(), redactions())
    former_fds = (writer._events_fd, writer._artifacts_fd, writer._lock_fd, writer._root_fd)
    writer.close()
    replacements: dict[int, int] = {}
    opened: list[int] = []
    try:
        while set(former_fds) - set(replacements):
            fd = os.open("/dev/null", os.O_RDONLY)
            opened.append(fd)
            if fd in former_fds:
                replacements[fd] = fd
        context = multiprocessing.get_context("fork")
        child = context.Process(target=lambda: None)
        child.start()
        child.join(10)
        assert child.exitcode == 0
        for fd in replacements:
            os.fstat(fd)
    finally:
        for fd in opened:
            os.close(fd)


def test_nested_at_fork_close_does_not_deadlock(tmp_path: Path) -> None:
    script = r"""
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from local_operator.evaluation.evidence.store import EvidenceWriter
from local_operator.evaluation.receipts import RedactionSet
from tests.unit.evaluation.evidence.test_models import manifest
root=Path(tempfile.mkdtemp())/'bundle'
first=EvidenceWriter.create(root,manifest(),RedactionSet.from_resolved_values(()))
def nested(): first.close()
os.register_at_fork(before=nested)
pid=os.fork()
if pid == 0: os._exit(0)
_, status=os.waitpid(pid,0)
print(os.waitstatus_to_exitcode(status))
"""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", script, str(Path.cwd())],
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    assert result.stdout.strip() == "0"


def test_streaming_redaction_memory_is_bounded_for_64mib(tmp_path: Path) -> None:
    chunk = b"x" * (1024 * 1024)
    scanner_peak = 0

    def source() -> Any:
        nonlocal scanner_peak
        for _ in range(64):
            yield chunk
            _current, peak = tracemalloc.get_traced_memory()
            scanner_peak = max(scanner_peak, peak)

    tracemalloc.start()
    try:
        with EvidenceWriter.create(
            tmp_path / "bundle", manifest(), redactions("very-secret-value")
        ) as writer:
            ref = writer.publish_artifact(source(), media_type="application/octet-stream")
    finally:
        tracemalloc.stop()
    assert ref.byte_count == 64 * 1024 * 1024
    assert scanner_peak < 32 * 1024 * 1024


def test_streaming_redaction_rss_delta_is_bounded_in_fresh_process() -> None:
    script = r"""
import resource, tempfile
from pathlib import Path
from local_operator.evaluation.evidence.store import EvidenceWriter
from local_operator.evaluation.receipts import RedactionSet
from tests.unit.evaluation.evidence.test_models import manifest
before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
root=Path(tempfile.mkdtemp())/'bundle'
chunk=b'x'*(1024*1024)
redactions=RedactionSet.from_resolved_values(('very-secret-value',))
with EvidenceWriter.create(root,manifest(),redactions) as writer:
 writer.publish_artifact(
  (chunk for _ in range(64)),media_type='application/octet-stream'
 )
after=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(after-before)
"""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        timeout=120,
        check=True,
    )
    # Darwin reports bytes; Linux reports KiB.
    delta = int(result.stdout.strip())
    delta_bytes = delta if sys.platform == "darwin" else delta * 1024
    assert delta_bytes < 64 * 1024 * 1024


def test_streaming_redaction_catches_chunk_boundary_canary(tmp_path: Path) -> None:
    with EvidenceWriter.create(
        tmp_path / "bundle", manifest(), redactions("very-secret-value")
    ) as writer:
        with pytest.raises(EvidenceBundleInvalid):
            writer.publish_artifact(
                (b"prefix-very-sec", b"ret-value-suffix"),
                media_type="application/octet-stream",
            )


def test_artifact_limit_rejects_before_unbounded_write(tmp_path: Path) -> None:
    with EvidenceWriter.create(tmp_path / "bundle", manifest(), redactions()) as writer:
        with pytest.raises(EvidenceBundleInvalid, match="maximum"):
            writer.publish_artifact(
                (b"x" * (1024 * 1024) for _ in range(MAX_ARTIFACT_BYTES // (1024 * 1024) + 1)),
                media_type="application/octet-stream",
            )


def test_redaction_unlink_failure_preserves_generic_error_and_poisons(
    tmp_path: Path,
) -> None:
    class UnlinkFails:
        def write(self, fd: int, data: bytes) -> int:
            return os.write(fd, data)

        def fsync(self, fd: int) -> None:
            os.fsync(fd)

        def link(self, src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int) -> None:
            os.link(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

        def unlink(self, path: str, *, dir_fd: int) -> None:
            raise OSError("unlink-failure")

    secret = "very-secret-value"
    root = tmp_path / "bundle"
    writer = EvidenceWriter.create(root, manifest(), redactions(secret), syscalls=UnlinkFails())
    with pytest.raises(EvidenceBundleInvalid) as error:
        writer.publish_artifact(secret.encode(), media_type="text/plain")
    assert secret not in str(error.value)
    with pytest.raises(EvidenceRecoveryOnly):
        writer.append(
            "cancel",
            CancelPayload(cancellation_id="later", source="operator", diagnostic_code="x"),
        )
    assert any(path.name.startswith(".artifact.") for path in (root / "artifacts").iterdir())
    writer.close()
    assert not verify_bundle(root).valid


def test_temp_close_failure_still_unlinks_and_poisons(tmp_path: Path) -> None:
    class CloseFails(_OSCallsForTest):
        def close(self, fd: int) -> None:
            os.close(fd)
            raise OSError("temp-close-failure")

    root = tmp_path / "bundle"
    writer = EvidenceWriter.create(root, manifest(), redactions(), syscalls=CloseFails())
    with pytest.raises(EvidenceBundleInvalid):
        writer.publish_artifact(b"safe", media_type="text/plain", expected_sha256=DIGEST)
    assert list((root / "artifacts").iterdir()) == []
    with pytest.raises(EvidenceRecoveryOnly):
        writer.begin_finalization("final", None, FinalizationIntent(kind="unscored"))
    writer.close()


def test_temp_close_and_unlink_failures_both_attempted_and_poison(
    tmp_path: Path,
) -> None:
    calls = {"close": 0, "unlink": 0}

    class UnlinkFails:
        def write(self, fd: int, data: bytes) -> int:
            return os.write(fd, data)

        def fsync(self, fd: int) -> None:
            os.fsync(fd)

        def link(self, src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int) -> None:
            os.link(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

        def unlink(self, path: str, *, dir_fd: int) -> None:
            calls["unlink"] += 1
            raise OSError("unlink-failure")

        def close(self, fd: int) -> None:
            calls["close"] += 1
            os.close(fd)
            raise OSError("close-failure")

    writer = EvidenceWriter.create(
        tmp_path / "bundle", manifest(), redactions(), syscalls=UnlinkFails()
    )
    with pytest.raises(EvidenceRecoveryOnly):
        writer.publish_artifact(b"safe", media_type="text/plain")
    assert calls == {"close": 1, "unlink": 1}
    with pytest.raises(EvidenceRecoveryOnly):
        writer.append(
            "cancel",
            CancelPayload(cancellation_id="later", source="operator", diagnostic_code="x"),
        )
    writer.close()


def test_content_rejection_cleanup_keeps_writer_reusable(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), redactions("very-secret-value")) as writer:
        with pytest.raises(EvidenceBundleInvalid):
            writer.publish_artifact(b"very-secret-value", media_type="application/octet-stream")
        writer.append(
            "cancel",
            CancelPayload(cancellation_id="later", source="operator", diagnostic_code="x"),
            monotonic_ns=1,
            wall_time_ms=2,
        )
    assert list((root / "artifacts").iterdir()) == []
    assert verify_bundle(root).valid


def test_ambiguous_artifact_link_poisons_writer(tmp_path: Path) -> None:
    class LinkFails:
        def write(self, fd: int, data: bytes) -> int:
            return os.write(fd, data)

        def fsync(self, fd: int) -> None:
            os.fsync(fd)

        def link(self, src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int) -> None:
            raise OSError("link-ambiguous")

        def unlink(self, path: str, *, dir_fd: int) -> None:
            os.unlink(path, dir_fd=dir_fd)

    writer = EvidenceWriter.create(
        tmp_path / "bundle", manifest(), redactions(), syscalls=LinkFails()
    )
    with pytest.raises(OSError, match="link-ambiguous"):
        writer.publish_artifact(b"safe", media_type="text/plain")
    with pytest.raises(EvidenceRecoveryOnly):
        writer.append(
            "cancel",
            CancelPayload(cancellation_id="cancel", source="operator", diagnostic_code="x"),
        )
    writer.close()


def test_close_attempts_all_fds_and_releases_lock_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    writer = EvidenceWriter.create(root, manifest(), redactions())
    events_fd = writer._events_fd
    real_close = os.close
    failed = False

    def close(fd: int) -> None:
        nonlocal failed
        if fd == events_fd and not failed:
            failed = True
            real_close(fd)
            raise OSError("close-failure")
        real_close(fd)

    monkeypatch.setattr(os, "close", close)
    with pytest.raises(OSError, match="close-failure"):
        writer.close()
    assert all(
        getattr(writer, name) == -1
        for name in ("_events_fd", "_artifacts_fd", "_lock_fd", "_root_fd")
    )
    writer.close()
    monkeypatch.setattr(os, "close", real_close)
    with EvidenceWriter.create(root, manifest(), redactions()):
        pass


def test_hardlinked_artifact_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), redactions()) as writer:
        _append_authority(writer)
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
