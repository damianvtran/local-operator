"""Owner audit-before-bytes, exercised through the existing paired-relay path.

The positive and failure controls use only APIs present on the actual base:
they must fail because unaudited content escaped, not because a new API is absent.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import audit, files, store
from tests.unit.network.test_mesh_file_pull import (
    SESSION,
    Devices,
    _in_store,
    _pad,
    _pull,
)
from tests.unit.network.test_mesh_file_pull import devices as _devices_fixture
from tests.unit.network.test_mesh_file_pull import mesh as _mesh_fixture
from tests.unit.network.test_mesh_file_pull import pair as _pair_fixture

# Expose the original fixture objects without wrapping their scope or dependencies.
devices = _devices_fixture
mesh = _mesh_fixture
pair = _pair_fixture


def _admissions(root: Path) -> list[dict[str, Any]]:
    path = store.audit_path(root)
    if not path.exists():
        return []
    return [
        row
        for line in path.read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line)).get("event") == "session_file_read_admitted"
    ]


def _fd_matches(fd: int, path: Path) -> bool:
    if not path.exists():
        return False
    opened, named = os.fstat(fd), path.stat()
    return (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino)


def _event(reference: str = "scratchpad://audit.bin") -> audit.AuditEvent:
    return audit.AuditEvent(
        event="session_file_read_admitted",
        actor="synthetic-requester",
        session_id=SESSION,
        detail={"reference": reference, "admitted_bytes": 7, "offset": 0, "decision": "allow"},
    )


def _fixture_source(owner: Any, payload: bytes, attachment: bool) -> str:
    if not attachment:
        (_pad(owner) / "audit.bin").write_bytes(payload)
        return "scratchpad://audit.bin"
    from local_operator.session.attachments import AttachmentStore

    item = AttachmentStore(Path(owner.root) / "attachments").put_bytes(
        payload, "application/octet-stream"
    )
    assert item is not None
    transcript = Path(owner.root) / "sessions" / SESSION / "transcript.jsonl"
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "user", "content": [{"attachment": item.digest}]}) + "\n")
    return f"attachment:{item.digest}"


@pytest.mark.parametrize("attachment", [False, True])
@pytest.mark.parametrize("phase", ["stat", "verify"])
@pytest.mark.parametrize("fail_audit", [False, True])
def test_hash_phase_requires_durable_owner_read_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attachment: bool,
    phase: str,
    fail_audit: bool,
) -> None:
    """Hashing consumes owner bytes even when no FILE payload is returned.

    Keep this direct-handler discriminator socket-free: the paired-relay controls
    separately establish authenticated dispatch. Observe the old behavior before
    asserting, so a baseline failure reports real target reads and the digest or
    match response, rather than failing merely because a new helper is absent.
    """
    from types import SimpleNamespace

    from local_operator.network.types import MeshRefusal

    root = tmp_path / "owner"
    (root / "sessions" / SESSION / "scratchpad").mkdir(parents=True)
    owner: Any = SimpleNamespace(root=root, audit=audit.AuditLog(root), _own_label=lambda: "owner")
    payload = b"synthetic hash admission payload"
    reference = _fixture_source(owner, payload, attachment)
    target = (
        root / "attachments" / f"{reference.removeprefix('attachment:')}.bin"
        if attachment
        else _pad(owner) / "audit.bin"
    )
    size, mtime = target.stat().st_size, target.stat().st_mtime_ns
    prefix = 7
    expected = size if phase == "stat" else prefix
    frame: dict[str, Any] = {
        "session_id": SESSION,
        "phase": phase,
        "expect_bytes": size,
        "expect_mtime_ns": mtime,
        "prefix_bytes": prefix,
        "prefix_digest": "sha256:" + hashlib.sha256(payload[:prefix]).hexdigest(),
        # These untrusted fields must not override the real serving operation.
        "expected_file_phase": "fetch",
        "audit_detail": {"phase": "fetch"},
        ("attachment" if attachment else "path"): (
            reference.removeprefix("attachment:") if attachment else reference
        ),
    }
    link: Any = SimpleNamespace(
        device_id="synthetic-requester", network_id="synthetic-network", epoch=1
    )
    real_required, real_pread, real_fsync = owner.audit.record_required, os.pread, os.fsync
    attempts: list[tuple[dict[str, Any], dict[str, Any]]] = []
    reads: list[tuple[int, int]] = []
    durable: set[int] = set()
    directory_durable: set[int] = set()
    reads_admitted: list[bool] = []

    def required(event: audit.AuditEvent, **context: Any) -> bool:
        # The trusted expected operation is separate from mutable event detail;
        # a phaseless hash row must not silently pass as a historical fetch row.
        attempts.append((event.detail.copy(), context.copy()))
        return False if fail_audit else real_required(event, **context)

    def synced(fd: int) -> None:
        real_fsync(fd)
        if _fd_matches(fd, store.audit_path(root)):
            durable.update(row["seq"] for row in _admissions(root))
        if _fd_matches(fd, store.audit_path(root).parent):
            directory_durable.update(
                row["seq"] for row in _admissions(root) if row["seq"] in durable
            )

    def pread(fd: int, count: int, offset: int) -> bytes:
        if _fd_matches(fd, target):
            reads.append((offset, count))
            rows = _admissions(root)
            reads_admitted.append(bool(rows and rows[-1]["seq"] in directory_durable))
        return real_pread(fd, count, offset)

    monkeypatch.setattr(owner.audit, "record_required", required)
    monkeypatch.setattr(os, "fsync", synced)
    monkeypatch.setattr(os, "pread", pread)
    reply = None
    refusal = None
    try:
        try:
            reply = files.make_handler(owner)(link, frame)
        except MeshRefusal as error:
            refusal = error.code
        observed = {"attempts": attempts, "reads": reads, "reply": reply, "refusal": refusal}
        if fail_audit:
            assert refusal == "audit_unavailable" and not reads and reply is None, observed
        else:
            assert attempts == [
                (
                    {
                        "reference": reference,
                        "admitted_bytes": expected,
                        "offset": 0,
                        "decision": "allow",
                        "phase": phase,
                    },
                    {"expected_file_phase": phase},
                )
            ], observed
            assert reads == [(0, expected)] and reads_admitted == [True], observed
            assert reply is not None and refusal is None, observed
            if phase == "stat":
                assert reply["sha256"] == "sha256:" + hashlib.sha256(payload).hexdigest()
            else:
                assert reply == {"match": True}
    finally:
        owner.audit.close()


@pytest.mark.parametrize("phase", ["fetch", "stat", "verify"])
def test_closed_file_read_variants_preserve_legacy_fetch(tmp_path: Path, phase: str) -> None:
    log = audit.AuditLog(tmp_path)
    event = _event()
    if phase != "fetch":
        event.detail["phase"] = phase
    try:
        assert log.record_required(event, expected_file_phase=phase)
        rows = log.tail()
        assert rows[-1]["detail"] == event.detail
        assert set(rows[-1]["detail"]) == {"reference", "offset", "admitted_bytes", "decision"} | (
            {"phase"} if phase != "fetch" else set()
        )
        # This historical/future-fetch selector must not count present null as
        # fetch. The rows are planned work, never delivery/completion receipts.
        candidates = rows + [{"detail": {"phase": None}}, {"detail": {"phase": "unknown"}}]
        fetch = [row for row in candidates if "phase" not in row["detail"]]
        assert len(fetch) == (1 if phase == "fetch" else 0)
    finally:
        log.close()


@pytest.mark.parametrize("expected_phase", ["unknown", "", None, True, "verify"])
def test_hash_context_cannot_accept_legacy_fetch_shape(
    tmp_path: Path,
    expected_phase: Any,
) -> None:
    log = audit.AuditLog(tmp_path)
    try:
        assert not log.record_required(_event(), expected_file_phase=expected_phase)
        assert log.recorded_through == 0 and not _admissions(tmp_path)
    finally:
        log.close()


def _damage_read_detail(detail: dict[str, Any], damage: str) -> None:
    if damage == "missing_phase":
        if "phase" in detail:
            del detail["phase"]
        else:
            detail["phase"] = None
    elif damage.startswith("phase_"):
        detail["phase"] = {
            "phase_null": None,
            "phase_unknown": "unknown",
            "phase_empty": "",
            "phase_bool": True,
            "phase_fetch": "fetch",
        }[damage]
    elif damage == "extra":
        detail["extra"] = "untrusted"
    elif damage == "truncated":
        detail["truncated"] = True
    elif damage == "missing_offset":
        del detail["offset"]
    elif damage == "count_bool":
        detail["admitted_bytes"] = True
    elif damage == "count_float":
        detail["admitted_bytes"] = 1.0
    elif damage == "offset_bool":
        detail["offset"] = False
    elif damage == "offset_float":
        detail["offset"] = 0.0
    elif damage == "count_negative":
        detail["admitted_bytes"] = -1
    elif damage == "offset_negative":
        detail["offset"] = -1
    elif damage == "reference_bool":
        detail["reference"] = True
    elif damage == "decision_bool":
        detail["decision"] = True
    elif damage == "reference_control":
        detail["reference"] = "scratchpad://a\nfile"
    else:
        raise AssertionError(damage)


@pytest.mark.parametrize("phase", ["fetch", "stat", "verify"])
@pytest.mark.parametrize("stage", ["raw", "rendered", "mutating_renderer", "durable"])
@pytest.mark.parametrize(
    "damage",
    [
        "missing_phase",
        "phase_null",
        "phase_unknown",
        "phase_empty",
        "phase_bool",
        "phase_fetch",
        "extra",
        "truncated",
        "missing_offset",
        "count_bool",
        "count_float",
        "offset_bool",
        "offset_float",
        "count_negative",
        "offset_negative",
        "reference_bool",
        "decision_bool",
        "reference_control",
    ],
)
def test_file_read_rejects_invalid_candidate_without_sequence_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    stage: str,
    damage: str,
) -> None:
    log = audit.AuditLog(tmp_path)
    event = _event()
    event.detail["admitted_bytes"] = 1
    if phase != "fetch":
        event.detail["phase"] = phase
    real_render, real_sync = log._render, log.sync

    def rendered(item: audit.AuditEvent) -> str:
        if stage == "mutating_renderer":
            # Mutation must not redefine the independently captured expectation.
            _damage_read_detail(item.detail, damage)
            return real_render(item)
        row = json.loads(real_render(item))
        _damage_read_detail(row["detail"], damage)
        return json.dumps(row) + "\n"

    def synced() -> None:
        real_sync()
        row = json.loads(log._required_line or "{}")
        _damage_read_detail(row["detail"], damage)
        log._required_line = json.dumps(row) + "\n"

    try:
        with monkeypatch.context() as patch:
            if stage == "raw":
                _damage_read_detail(event.detail, damage)
            elif stage == "durable":
                patch.setattr(log, "sync", synced)
            else:
                patch.setattr(log, "_render", rendered)
            assert not log.record_required(event, expected_file_phase=phase)
        if stage != "durable":
            assert not _admissions(tmp_path), "invalid rendered hash became an apparent fetch row"
            assert log.recorded_through == 0 and log.records_written == 0
        else:
            # The persisted original row is valid; corrupting its acknowledgement
            # must still refuse the target read. Do not pretend append was undone.
            assert len(_admissions(tmp_path)) == 1
        next_event = _event()
        assert log.record_required(next_event)
        seqs = [row["seq"] for row in _admissions(tmp_path)]
        assert seqs == ([1] if stage != "durable" else [1, 2])
    finally:
        log.close()


@pytest.mark.parametrize("phase", ["stat", "verify"])
@pytest.mark.parametrize("damage", ["missing_phase", "truncated", "count_float"])
def test_sanitizer_cannot_redefine_expected_hash_detail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    damage: str,
) -> None:
    log = audit.AuditLog(tmp_path)
    event = _event()
    event.detail.update(phase=phase, admitted_bytes=1)
    original = audit._sanitize_detail

    def sanitize(name: str, detail: dict[str, Any]) -> dict[str, Any]:
        _damage_read_detail(detail, damage)
        return original(name, detail)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(audit, "_sanitize_detail", sanitize)
            assert not log.record_required(event, expected_file_phase=phase)
        assert not _admissions(tmp_path) and log.recorded_through == 0
        assert log.record_required(_event())
        assert _admissions(tmp_path)[-1]["seq"] == 1
    finally:
        log.close()


@pytest.mark.parametrize("phase", ["stat", "verify"])
def test_hash_reference_budgets_final_phase_without_target_content(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    # Find the deterministic serialized-budget boundary, not a guessed length.
    references = ["scratchpad://" + "😀" * n for n in range(1, 200)]
    reference = next(
        ref
        for ref in references
        if audit.file_read_reference(ref, 0, 1) == ref
        and audit.file_read_reference(ref, 0, 1, phase=phase) != ref
    )
    real_hash = hashlib.sha256
    hashed: list[bytes] = []

    def sha256(value: bytes = b"") -> Any:
        hashed.append(value)
        return real_hash(value)

    monkeypatch.setattr(audit.hashlib, "sha256", sha256)
    fallback = audit.file_read_reference(reference, 0, 1, phase=phase)
    assert hashed == [reference.encode("utf-8")]
    assert fallback == "reference:sha256:" + real_hash(reference.encode()).hexdigest()
    detail = {
        "reference": fallback,
        "offset": 0,
        "admitted_bytes": 1,
        "decision": "allow",
        "phase": phase,
    }
    assert audit._sanitize_detail("session_file_read_admitted", detail) == detail
    with pytest.raises(ValueError):
        audit.file_read_reference(reference, False, 1, phase=phase)
    with pytest.raises(ValueError):
        audit.file_read_reference(reference, 0, 1, phase="unknown")


@pytest.mark.parametrize(
    "phase,payload,prefix,expected",
    [
        ("stat", b"", 0, "empty_file"),
        ("verify", b"abc", 0, False),
        ("verify", b"abc", -1, False),
        ("verify", b"abc", 4, False),
        ("verify", b"abc", "invalid", "bad_request"),
    ],
)
def test_hash_no_read_branches_need_no_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    payload: bytes,
    prefix: Any,
    expected: Any,
) -> None:
    from local_operator.network.types import MeshRefusal

    pad = tmp_path / "sessions" / SESSION / "scratchpad"
    pad.mkdir(parents=True)
    target = pad / "empty.bin"
    target.write_bytes(payload)

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no-read branch attempted a hash or audit")

    monkeypatch.setattr(files, "_hash_fd", unexpected)
    frame = {
        "phase": phase,
        "path": "scratchpad://empty.bin",
        "prefix_bytes": prefix,
        "expect_bytes": len(payload),
        "expect_mtime_ns": target.stat().st_mtime_ns,
    }
    if type(expected) is str:
        with pytest.raises(MeshRefusal) as error:
            files.serve_file(tmp_path, SESSION, frame, admit=unexpected)
        assert error.value.code == expected
    else:
        assert files.serve_file(tmp_path, SESSION, frame, admit=unexpected) == {"match": False}


@pytest.mark.parametrize("attachment", [False, True])
@pytest.mark.parametrize("phase", ["stat", "verify"])
@pytest.mark.parametrize(
    "failure",
    [
        "disabled",
        "degraded",
        "append",
        "fsync",
        "directory",
        "readback",
        "renderer",
        "sanitizer",
    ],
)
def test_hash_durable_faults_prevent_target_read_and_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attachment: bool,
    phase: str,
    failure: str,
) -> None:
    from types import SimpleNamespace

    from local_operator.network.types import MeshRefusal

    root = tmp_path / "owner"
    (root / "sessions" / SESSION / "scratchpad").mkdir(parents=True)
    owner: Any = SimpleNamespace(root=root, audit=audit.AuditLog(root), _own_label=lambda: "owner")
    reference = _fixture_source(owner, b"synthetic failure payload", attachment)
    target = (
        root / "attachments" / f"{reference.removeprefix('attachment:')}.bin"
        if attachment
        else _pad(owner) / "audit.bin"
    )
    frame = {
        "phase": phase,
        "session_id": SESSION,
        "prefix_bytes": 7,
        "expect_bytes": target.stat().st_size,
        "expect_mtime_ns": target.stat().st_mtime_ns,
        ("attachment" if attachment else "path"): (
            reference.removeprefix("attachment:") if attachment else reference
        ),
    }
    link: Any = SimpleNamespace(device_id="synthetic-requester", network_id="synthetic", epoch=1)
    real_open, real_fsync, real_sync = Path.open, os.fsync, owner.audit.sync
    real_pread, real_render = os.pread, owner.audit._render
    real_sanitize, real_required = audit._sanitize_detail, owner.audit.record_required
    reads: list[int] = []
    injected: list[str] = []
    admitting = False

    def opened(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if failure == "append" and path == store.audit_path(root) and "a" in mode:
            injected.append(failure)
            raise OSError("injected hash admission append failure")
        return real_open(path, mode, *args, **kwargs)

    def fsync(fd: int) -> None:
        if (failure == "fsync" and _fd_matches(fd, store.audit_path(root))) or (
            failure == "directory" and _fd_matches(fd, store.audit_path(root).parent)
        ):
            injected.append(failure)
            raise OSError("injected hash admission fsync failure")
        real_fsync(fd)

    def sync() -> None:
        real_sync()
        if failure == "readback":
            injected.append(failure)
            row = json.loads(owner.audit._required_line)
            del row["detail"]["phase"]
            owner.audit._required_line = json.dumps(row)

    def render(event: audit.AuditEvent) -> str:
        row = json.loads(real_render(event))
        if failure == "renderer":
            injected.append(failure)
            del row["detail"]["phase"]
        return json.dumps(row) + "\n"

    def required(event: audit.AuditEvent, **context: Any) -> bool:
        nonlocal admitting
        admitting = True
        try:
            return real_required(event, **context)
        finally:
            admitting = False

    def sanitize(name: str, detail: dict[str, Any]) -> dict[str, Any]:
        if failure == "sanitizer" and admitting and name == "session_file_read_admitted":
            injected.append(failure)
            detail = {key: value for key, value in detail.items() if key != "phase"}
        return real_sanitize(name, detail)

    def pread(fd: int, count: int, offset: int) -> bytes:
        if _fd_matches(fd, target):
            reads.append(count)
        return real_pread(fd, count, offset)

    monkeypatch.setattr(owner.audit, "record_required", required)
    monkeypatch.setattr(owner.audit, "_render", render)
    monkeypatch.setattr(audit, "_sanitize_detail", sanitize)
    monkeypatch.setattr(Path, "open", opened)
    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(owner.audit, "sync", sync)
    monkeypatch.setattr(os, "pread", pread)
    if failure == "disabled":
        owner.audit._enabled = False
    if failure == "degraded":
        owner.audit.degraded = True
    try:
        with pytest.raises(MeshRefusal) as error:
            files.make_handler(owner)(link, frame)
        assert error.value.code == "audit_unavailable"
        assert reads == []
        if failure in {"append", "fsync", "directory", "readback", "renderer", "sanitizer"}:
            assert injected == [failure], "fault missed the new hash admission"
    finally:
        owner.audit.close()


@pytest.mark.parametrize("attachment", [False, True])
def test_owner_commits_before_source_read_and_first_received_content(
    mesh: Devices,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    attachment: bool,
) -> None:
    owner, requester, _host, _port = mesh
    payload = b"a" * (files.FILE_CHUNK_BYTES + 19)
    source = "scratchpad://audit.bin"
    reference = source
    if attachment:
        from local_operator.session.attachments import AttachmentStore

        item = AttachmentStore(Path(owner.root) / "attachments").put_bytes(
            payload, "application/octet-stream"
        )
        assert item is not None
        source = reference = f"attachment:{item.digest}"
        transcript = Path(owner.root) / "sessions" / SESSION / "transcript.jsonl"
        with transcript.open("a", encoding="utf-8") as handle:
            row = {"type": "user", "content": [{"attachment": item.digest}]}
            handle.write(json.dumps(row) + "\n")
    else:
        (_pad(owner) / "audit.bin").write_bytes(payload)

    committed: set[int] = set()
    directory_committed: set[int] = set()
    real_fsync = os.fsync
    real_read = files._read_span
    real_ask = files._Asker.ask
    received = bytearray()

    def synced(fd: int) -> None:
        real_fsync(fd)
        path = store.audit_path(owner.root)
        if _fd_matches(fd, path):
            committed.update(row["seq"] for row in _admissions(owner.root))
        if _fd_matches(fd, path.parent):
            directory_committed.update(
                row["seq"] for row in _admissions(owner.root) if row["seq"] in committed
            )

    def check(offset: int, count: int) -> None:
        rows = _admissions(owner.root)
        assert rows, "content escaped without an owner audit row"
        row = rows[-1]
        assert row["seq"] in committed, "owner row was appended but not fsynced"
        assert row["seq"] in directory_committed, "owner row's name was not durably published"
        assert row["actor"] == requester.identity.device_id
        assert row["session_id"] == SESSION
        assert row["outcome"] == "ok"
        assert row["detail"] == {
            "reference": reference,
            "offset": offset,
            "admitted_bytes": count,
            "decision": "allow",
        }
        assert "requesting_session" not in row["detail"], "no trusted requesting session is known"

    def read(fd: int, offset: int, limit: int) -> bytes:
        check(offset, limit)
        return real_read(fd, offset, limit)

    def ask(self: Any, **fields: Any) -> dict[str, Any]:
        reply = real_ask(self, **fields)
        if fields.get("phase") == "fetch":
            data = base64.b64decode(reply["data"])
            check(int(reply["offset"]), len(data))
            received.extend(data)
        return reply

    monkeypatch.setattr(os, "fsync", synced)
    monkeypatch.setattr(files, "_read_span", read)
    monkeypatch.setattr(files._Asker, "ask", ask)
    rc, receipt = _pull(capsys, source)
    assert rc == 0, receipt
    assert received == payload
    # Presence, not truthiness: null/unknown phase is never a fetch row.
    rows = [row for row in _admissions(owner.root) if "phase" not in row["detail"]]
    assert len(rows) == 2
    assert sum(row["detail"]["admitted_bytes"] for row in rows) == len(payload)
    assert not _admissions(requester.root), "the required receipt belongs to the owner"


@pytest.mark.parametrize("attachment", [False, True])
@pytest.mark.parametrize("failure", ["append", "fsync", "directory", "disabled", "degraded"])
def test_failed_durable_audit_refuses_with_zero_content_bytes(
    mesh: Devices,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
    attachment: bool,
) -> None:
    owner, requester, _host, _port = mesh
    payload = b"must not leave owner"
    source = _fixture_source(owner, payload, attachment)
    source_reads: list[int] = []
    received: list[bytes] = []
    real_open, real_fsync = Path.open, os.fsync
    real_read, real_ask = files._read_span, files._Asker.ask
    audit_path = store.audit_path(owner.root)
    real_required = owner.audit.record_required
    active = False
    expected_seq = 0
    fault_rows: list[int] = []
    entry_health: list[bool] = []

    def required(event: audit.AuditEvent, **context: Any) -> bool:
        nonlocal active, expected_seq
        # Arm AFTER fresh-link lifecycle writes. A previously degraded writer
        # refusing is not evidence that THIS admission's append/fsync failed.
        with owner.audit._write_lock:  # noqa: SLF001 — same atomic writer boundary
            entry_health.append(not owner.audit.degraded)
            expected_seq = owner.audit.recorded_through + 1
            active = True
            try:
                return real_required(event, **context)
            finally:
                active = False

    class AppendFault:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def __enter__(self) -> AppendFault:
            self.handle.__enter__()
            return self

        def __exit__(self, *args: Any) -> Any:
            return self.handle.__exit__(*args)

        def fileno(self) -> int:
            return self.handle.fileno()

        def write(self, text: str) -> Any:
            rows = [json.loads(line) for line in text.splitlines()]
            if active and any(
                row.get("event") == "session_file_read_admitted" and row["seq"] == expected_seq
                for row in rows
            ):
                fault_rows.append(expected_seq)
                raise OSError("injected NEW admission append failure")
            return self.handle.write(text)

    def open_file(path: Path, *args: Any, **kwargs: Any) -> Any:
        handle = real_open(path, *args, **kwargs)
        if path == audit_path and args and args[0] == "a":
            return AppendFault(handle)
        return handle

    def fsync(fd: int) -> None:
        target = audit_path.parent if failure == "directory" else audit_path
        if active and _fd_matches(fd, target):
            rows = _admissions(owner.root)
            assert rows and rows[-1]["seq"] == expected_seq, "fault missed NEW admission"
            fault_rows.append(expected_seq)
            raise OSError("injected NEW admission fsync failure")
        real_fsync(fd)

    def read(fd: int, offset: int, limit: int) -> bytes:
        source_reads.append(limit)
        return real_read(fd, offset, limit)

    def ask(self: Any, **fields: Any) -> dict[str, Any]:
        reply = real_ask(self, **fields)
        if fields.get("phase") == "fetch" and reply.get("data"):
            received.append(base64.b64decode(reply["data"]))
        return reply

    if failure == "append":
        monkeypatch.setattr(Path, "open", open_file)
    elif failure in {"fsync", "directory"}:
        monkeypatch.setattr(os, "fsync", fsync)
    elif failure == "disabled":
        monkeypatch.setattr(owner.audit, "_enabled", False)
    else:
        monkeypatch.setattr(owner.audit, "degraded", True)
    monkeypatch.setattr(owner.audit, "record_required", required)
    monkeypatch.setattr(files, "_read_span", read)
    monkeypatch.setattr(files._Asker, "ask", ask)
    rc, receipt = _pull(capsys, source)
    if failure in {"append", "fsync", "directory"}:
        assert entry_health == [True], "fault was armed on an already-degraded writer"
        assert fault_rows == [expected_seq], "fault never hit NEW admission's operation"
    # Refusal/control frames are allowed; FILE data and source content reads are not.
    assert received == [], "file content arrived despite a failed durable audit"
    assert source_reads == [], "content was read for egress before audit admission"
    assert rc == 1 and receipt["code"] == "audit_unavailable", receipt
    assert not _in_store(requester, payload)
    assert not any(
        path.stat().st_size for path in files.staging_root(requester.root).glob("*.part")
    )


def test_admitted_count_is_a_bounded_plan_not_a_completion_claim(
    mesh: Devices,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    owner, _requester, _host, _port = mesh
    payload = b"planned bytes"
    (_pad(owner) / "audit.bin").write_bytes(payload)

    def cut_after_admission(fd: int, offset: int, limit: int) -> bytes:
        rows = _admissions(owner.root)
        assert rows and rows[-1]["detail"]["admitted_bytes"] == len(payload)
        raise OSError("injected source read failure after admission")

    monkeypatch.setattr(files, "_read_span", cut_after_admission)
    rc, _receipt = _pull(capsys, "scratchpad://audit.bin")
    assert rc == 1
    assert _admissions(owner.root)[-1]["detail"]["decision"] == "allow"


@pytest.mark.parametrize("kind", ["long", "control", "unicode"])
@pytest.mark.parametrize("fail_audit", [False, True])
def test_fallback_reference_never_reads_fetch_content_before_admission(
    mesh: Devices,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kind: str,
    fail_audit: bool,
) -> None:
    owner, _requester, _host, _port = mesh
    relative = {
        "long": "/".join(["x" * 80] * 3) + "/audit.bin",
        "control": "audit\tname.bin",
        # Exact R1 path: legal UTF-8 components, only 186 reference characters,
        # but the existing ASCII-escaped detail budget is exceeded.
        "unicode": "/".join(["😀" * 50] * 3 + ["😀" * 20]),
    }[kind]
    target = _pad(owner) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = b"long reference"
    target.write_bytes(payload)
    reference = "scratchpad://" + relative
    expected_reference = "reference:sha256:" + hashlib.sha256(reference.encode()).hexdigest()
    real_required, real_pread = owner.audit.record_required, os.pread
    real_ask = files._Asker.ask
    active_read = False
    admitted = False
    fetch_reads: list[int] = []
    received: list[bytes] = []

    def required(event: audit.AuditEvent, **context: Any) -> bool:
        nonlocal admitted
        if fail_audit:
            return False
        admitted = real_required(event, **context)
        if admitted:
            row = _admissions(owner.root)[-1]
            expected_detail = {
                "reference": expected_reference,
                "offset": 0,
                "admitted_bytes": len(payload),
                "decision": "allow",
            }
            if context["expected_file_phase"] != "fetch":
                expected_detail["phase"] = context["expected_file_phase"]
            assert row["detail"] == expected_detail, "EXACT fields must precede every target read"
            assert row["actor"] == mesh[1].identity.device_id
            assert row["session_id"] == SESSION
        return admitted

    def pread(fd: int, count: int, offset: int) -> bytes:
        if active_read and _fd_matches(fd, target):
            fetch_reads.append(count)
            assert admitted, "fallback hashed source content BEFORE admission"
        return real_pread(fd, count, offset)

    real_handler = owner._handlers["net_file_get"]  # noqa: SLF001

    def handler(link: Any, frame: dict[str, Any]) -> dict[str, Any]:
        nonlocal active_read, admitted
        # #4228 covered fetch only. A intentionally extends the same observation
        # to hash phases; an earlier operation's receipt cannot admit this one.
        active_read = frame.get("phase") in {"stat", "verify", "fetch"}
        admitted = False
        try:
            result = real_handler(link, frame)
            assert isinstance(result, dict)
            return result
        finally:
            active_read = False

    def ask(self: Any, **fields: Any) -> dict[str, Any]:
        reply = real_ask(self, **fields)
        if fields.get("phase") == "fetch" and reply.get("data"):
            received.append(base64.b64decode(reply["data"]))
        return reply

    monkeypatch.setattr(owner.audit, "record_required", required)
    monkeypatch.setattr(os, "pread", pread)
    monkeypatch.setitem(owner._handlers, "net_file_get", handler)  # noqa: SLF001
    monkeypatch.setattr(files._Asker, "ask", ask)
    # urlsplit strips literal tabs; percent-encoding reaches the legal filename
    # while the owner's canonical reference still contains the actual character.
    rc, receipt = _pull(capsys, reference.replace("\t", "%09"))
    if fail_audit:
        assert rc == 1 and receipt["code"] == "audit_unavailable", receipt
        assert fetch_reads == [] and received == []
        assert not _in_store(mesh[1], payload)
    else:
        assert rc == 0, receipt
        assert fetch_reads and b"".join(received) == payload


@pytest.mark.parametrize(
    "offset, requested, expected", [(0, 99, 7), (5, 99, 2), (7, 99, 0), (9, 99, 0)]
)
def test_admission_count_clips_to_owner_observed_eof(
    mesh: Devices,
    offset: int,
    requested: int,
    expected: int,
) -> None:
    owner, requester, _host, _port = mesh
    target = _pad(owner) / "audit.bin"
    target.write_bytes(b"1234567")
    handler = files.make_handler(owner)
    # Invoke the owner handler, retaining its real durable writer. The link's
    # authenticated attributes are all this handler consumes after dispatch.
    from types import SimpleNamespace

    link: Any = SimpleNamespace(
        device_id=requester.identity.device_id,
        network_id="synthetic-network",
        epoch=1,
    )
    frame = {
        "session_id": SESSION,
        "path": "scratchpad://audit.bin",
        "phase": "fetch",
        "offset": offset,
        "limit": requested,
        "expect_bytes": target.stat().st_size,
        "expect_mtime_ns": target.stat().st_mtime_ns,
    }
    reply = handler(link, frame)
    assert reply["bytes"] == expected
    row = _admissions(owner.root)[-1]
    assert row["detail"]["admitted_bytes"] == expected
    assert row["detail"]["offset"] == offset


def test_metadata_index_never_reads_content_or_makes_a_nested_mesh_request(
    mesh: Devices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.network import file_index

    owner, _requester, _host, _port = mesh

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("metadata listing attempted content/audit/mesh access")

    monkeypatch.setattr(files, "_read_span", unexpected)
    monkeypatch.setattr(owner.audit, "record_required", unexpected)
    monkeypatch.setattr(owner, "_ensure_link", unexpected)
    # The ready user-attachment index is metadata-only, even when no content
    # writer is available. Serving it must not recursively request a peer.
    handler = file_index.make_handler(owner)
    unused_link: Any = None
    result = handler(
        unused_link, {"session_id": SESSION, "phase": "list", "index": "user_attachments"}
    )
    assert result["state"] == "ready"
    assert not _admissions(owner.root)


def test_buffer_acceptance_is_not_a_durable_acknowledgement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = audit.AuditLog(tmp_path)
    log.record(audit.AuditEvent(event="panic_raised"))
    before = log.syncs
    monkeypatch.setattr(log, "flush", lambda: None)
    assert not log.record_required(_event())
    assert log.syncs > before, "fsync of an older row alone is not acknowledgement"
    assert not _admissions(tmp_path)


def test_required_writer_returns_only_after_durable_commit(tmp_path: Path) -> None:
    log = audit.AuditLog(tmp_path)
    event = audit.AuditEvent(
        event="session_file_read_admitted",
        actor="synthetic-requester",
        session_id=SESSION,
        detail={
            "reference": "scratchpad://audit.bin",
            "admitted_bytes": 7,
            "offset": 0,
            "decision": "allow",
        },
    )
    assert log.record_required(event)
    assert log.syncs >= 1
    assert _admissions(tmp_path)[-1]["detail"] == event.detail
    log.degraded = True
    assert not log.record_required(event)
    assert len(_admissions(tmp_path)) == 1
    log.close()


@pytest.mark.parametrize(
    "reference",
    [
        "scratchpad://" + "/".join(["😀" * 50] * 3 + ["😀" * 20]),
        "scratchpad://a\nname.bin",
        "scratchpad://" + "x" * 300,
    ],
)
def test_required_ack_refuses_details_changed_by_sanitizer(
    tmp_path: Path,
    reference: str,
) -> None:
    log = audit.AuditLog(tmp_path)
    assert not log.record_required(_event(reference)), "sanitizer lost required audit fields"
    assert not _admissions(tmp_path)


def test_required_first_creation_syncs_discoverable_directory_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "new-parent" / "owner"
    log = audit.AuditLog(root)
    path = store.audit_path(root)
    # Each newly created name must be persisted by syncing its PARENT. The
    # already-existing tmp_path bounds the chain; no live config is involved.
    needed = {path.parent, root, root.parent, tmp_path}
    synced: set[Path] = set()
    real_fsync = os.fsync

    def fsync(fd: int) -> None:
        real_fsync(fd)
        synced.update(directory for directory in needed if _fd_matches(fd, directory))

    monkeypatch.setattr(os, "fsync", fsync)
    assert log.record_required(_event())
    assert synced == needed, "file fsync alone does not publish a newly created name"
    assert _admissions(root)[-1]["detail"] == _event().detail


@pytest.mark.parametrize("rotation", ["before", "after"])
def test_required_ack_tracks_exact_row_through_healthy_rotation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rotation: str,
) -> None:
    log = audit.AuditLog(tmp_path)
    path = store.audit_path(tmp_path)
    for _ in range(20):
        log.record(audit.AuditEvent(event="link_opened", detail={"role": "x" * 200}))
    log.flush()
    log._max_bytes = path.stat().st_size + (1 if rotation == "after" else 0)  # noqa: SLF001
    expected_seq = log.recorded_through + 1
    archive_synced: set[Path] = set()
    archive_published: set[Path] = set()
    real_fsync, real_unlink = os.fsync, Path.unlink

    def fsync(fd: int) -> None:
        for archive in path.parent.glob("audit.jsonl.*.gz"):
            if _fd_matches(fd, archive):
                # Reading through EOF checks that gzip.close already wrote the
                # trailer, not merely that a partial compressed stream exists.
                with gzip.open(archive, "rt", encoding="utf-8") as handle:
                    assert [json.loads(line) for line in handle]
                real_fsync(fd)
                archive_synced.add(archive)
                return
        real_fsync(fd)
        if _fd_matches(fd, path.parent):
            archive_published.update(archive_synced)

    def unlink(named: Path, *args: Any, **kwargs: Any) -> None:
        if named == path:
            assert (
                archive_synced and archive_published == archive_synced
            ), "retired prior live copy before COMPLETE archive and directory fsync"
        real_unlink(named, *args, **kwargs)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(Path, "unlink", unlink)
    assert log.record_required(_event()), "healthy rotation rejected exact requested row"
    rows = []
    for archive in path.parent.glob("audit.jsonl.*.gz"):
        with gzip.open(archive, "rt", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle)
    rows.extend(json.loads(line) for line in path.read_text().splitlines())
    matching = [row for row in rows if row["seq"] == expected_seq]
    assert len(matching) == 1 and matching[0]["detail"] == _event().detail
    assert archive_synced == archive_published


def test_healthy_postappend_rotation_acknowledges_admission_not_bookkeeping(
    tmp_path: Path,
) -> None:
    log = audit.AuditLog(tmp_path)
    log.record(audit.AuditEvent(event="panic_raised"))
    log._max_bytes = store.audit_path(tmp_path).stat().st_size + 1  # noqa: SLF001
    requested_seq = log.recorded_through + 1
    assert log.record_required(_event()), "later audit_rotated sequence is not this admission"
    assert log.recorded_through > requested_seq, "control did not cross a rotation boundary"


def test_fsync_of_replacement_inode_is_not_admission_ack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = audit.AuditLog(tmp_path)
    path = store.audit_path(tmp_path)
    log.record(audit.AuditEvent(event="panic_raised"))
    log._max_bytes = path.stat().st_size + 1  # noqa: SLF001
    replacement_syncs: list[int] = []
    real_fsync = os.fsync

    def wrong_generation(size: int) -> None:
        # Negative control for receipt identity, not fake counters: actually
        # move the unsynced admission away, create and fsync ONLY a fresh inode.
        path.rename(path.with_name("unpublished-prior-copy"))
        log._write_rotation_record(1, size)  # noqa: SLF001 — existing writer API
        log.sync()

    def fsync(fd: int) -> None:
        real_fsync(fd)
        if _fd_matches(fd, path):
            replacement_syncs.append(os.fstat(fd).st_ino)

    monkeypatch.setattr(log, "_rotate", wrong_generation)
    monkeypatch.setattr(os, "fsync", fsync)
    assert not log.record_required(_event()), "fsynced unrelated replacement inode"
    assert replacement_syncs, "negative control never synced the WRONG generation"
    assert not _admissions(tmp_path)


def _live_audit_copy(path: Path) -> tuple[tuple[int, int], bytes] | None:
    """Read the actual live inode and bytes while the caller owns the writer lock."""
    try:
        with path.open("rb") as handle:
            info = os.fstat(handle.fileno())
            return (info.st_dev, info.st_ino), handle.read()
    except FileNotFoundError:
        return None


def _assert_prior_copy_witness(
    original_identity: tuple[int, int],
    at_fault: tuple[tuple[int, int], bytes] | None,
    at_return: tuple[tuple[int, int], bytes] | None,
    expected_admission: dict[str, Any],
) -> None:
    """Pure witness validation, not a substitute for observing the real fault.

    A later healthy rotation may retain the row elsewhere, but cannot repair
    retirement of its original live copy BEFORE the failed archive publication.
    Both snapshots must still name that inode and its exact admission contents.
    """
    for snapshot in (at_fault, at_return):
        assert snapshot is not None, "prior live copy retired before archive was durable"
        identity, content = snapshot
        assert identity == original_identity, "prior live inode was replaced at failure boundary"
        rows = [json.loads(line) for line in content.splitlines()]
        matching = [row for row in rows if row.get("seq") == expected_admission["seq"]]
        assert len(matching) == 1 and all(
            matching[0].get(key) == value for key, value in expected_admission.items()
        ), "original live copy is missing the exact admission row"
    assert at_return == at_fault, "prior live contents changed before failed admission returned"


@pytest.mark.parametrize("attachment", [False, True])
@pytest.mark.parametrize("failure", ["archive", "archive_directory"])
def test_archive_failure_preserves_prior_copy_and_refuses_content(
    mesh: Devices,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    attachment: bool,
    failure: str,
) -> None:
    owner, requester, _host, _port = mesh
    payload = b"archive must commit before these bytes leave"
    source = _fixture_source(owner, payload, attachment)
    path = store.audit_path(owner.root)
    real_required, real_fsync = owner.audit.record_required, os.fsync
    real_read, real_ask = files._read_span, files._Asker.ask
    expected_seq = 0
    active = False
    archive_synced = False
    original_identity = (0, 0)
    at_fault: tuple[tuple[int, int], bytes] | None = None
    expected_admission: dict[str, Any] = {}
    boundary_checks: list[str] = []
    fault_rows: list[int] = []
    source_reads: list[int] = []
    received: list[bytes] = []

    def required(event: audit.AuditEvent, **context: Any) -> bool:
        nonlocal active, expected_seq, original_identity, expected_admission
        with owner.audit._write_lock:  # noqa: SLF001
            assert not owner.audit.degraded, "fault armed on a previously degraded writer"
            expected_seq = owner.audit.recorded_through + 1
            original = path.stat()
            original_identity = (original.st_dev, original.st_ino)
            expected_admission = {
                "seq": expected_seq,
                "event": event.event,
                "actor": event.actor,
                "session_id": event.session_id,
                "detail": dict(event.detail),
            }
            # Pre-append size is below the boundary; THIS row crosses it.
            owner.audit._max_bytes = original.st_size + 1  # noqa: SLF001
            active = True
            try:
                result = real_required(event, **context)
                assert result is False, "faulted admission unexpectedly acknowledged"
                # Capture BEFORE releasing the existing lock: later lifecycle
                # writes may legitimately rotate this inode after refusal.
                at_return = _live_audit_copy(path)
                _assert_prior_copy_witness(
                    original_identity, at_fault, at_return, expected_admission
                )
                boundary_checks.append("return")
                return result
            finally:
                active = False

    def capture_fault_boundary() -> None:
        nonlocal at_fault
        at_fault = _live_audit_copy(path)
        _assert_prior_copy_witness(original_identity, at_fault, at_fault, expected_admission)
        boundary_checks.append("fault")
        fault_rows.append(expected_seq)

    def fsync(fd: int) -> None:
        nonlocal archive_synced
        if active:
            for archive in path.parent.glob("audit.jsonl.*.gz"):
                if not _fd_matches(fd, archive):
                    continue
                with gzip.open(archive, "rt", encoding="utf-8") as handle:
                    rows = [json.loads(line) for line in handle]
                assert any(row["seq"] == expected_seq for row in rows)
                if failure == "archive":
                    capture_fault_boundary()
                    raise OSError("injected admission archive fsync failure")
                real_fsync(fd)
                archive_synced = True
                return
            if (
                failure == "archive_directory"
                and archive_synced
                and not fault_rows
                and _fd_matches(fd, path.parent)
            ):
                capture_fault_boundary()
                raise OSError("injected admission archive publication fsync failure")
        real_fsync(fd)

    def read(fd: int, offset: int, limit: int) -> bytes:
        source_reads.append(limit)
        return real_read(fd, offset, limit)

    def ask(self: Any, **fields: Any) -> dict[str, Any]:
        reply = real_ask(self, **fields)
        if fields.get("phase") == "fetch" and reply.get("data"):
            received.append(base64.b64decode(reply["data"]))
        return reply

    monkeypatch.setattr(owner.audit, "record_required", required)
    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(files, "_read_span", read)
    monkeypatch.setattr(files._Asker, "ask", ask)
    rc, receipt = _pull(capsys, source)
    assert fault_rows == [expected_seq], "fault missed the NEW admission's archive"
    assert rc == 1 and receipt["code"] == "audit_unavailable", receipt
    assert source_reads == [] and received == []
    assert not _in_store(requester, payload)
    assert not any(p.stat().st_size for p in files.staging_root(requester.root).glob("*.part"))
    assert boundary_checks == ["fault", "return"], "missing locked fault/return witness"
    assert at_fault is not None
    original_rows = [json.loads(line) for line in at_fault[1].splitlines()]
    original_row = next(row for row in original_rows if row["seq"] == expected_seq)
    # This is CONTINUED retention, separately from the stronger immediate live
    # copy witnesses above. Lifecycle traffic may have safely rotated it by now.
    with owner.audit._write_lock:  # noqa: SLF001
        retained_rows = []
        for generation in [path, *path.parent.glob("audit.jsonl.*.gz")]:
            if generation.suffix == ".gz":
                with gzip.open(generation, "rt", encoding="utf-8") as handle:
                    retained_rows.extend(json.loads(line) for line in handle)
            elif generation.exists():
                retained_rows.extend(
                    json.loads(line) for line in generation.read_text().splitlines()
                )
        assert original_row in retained_rows, "exact admission was lost after refusal"


def test_archive_failure_boundary_witness_on_synthetic_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the same witness without a relay translating assertion failures.

    A separately granted counterfactual packet replaces ONLY this fresh log's
    rotation behavior with a deliberately record-dropping test double. Its raw
    RED must come from the real missing row at the locked failure boundary, not
    a fabricated snapshot or an import/setup error. The normal writer passes.
    """
    log = audit.AuditLog(tmp_path)
    log.record(audit.AuditEvent(event="panic_raised"))
    path = store.audit_path(tmp_path)
    original = path.stat()
    identity = (original.st_dev, original.st_ino)
    event = _event()
    expected_seq = log.recorded_through + 1
    expected = {
        "seq": expected_seq,
        "event": event.event,
        "actor": event.actor,
        "session_id": event.session_id,
        "detail": dict(event.detail),
    }
    at_fault: tuple[tuple[int, int], bytes] | None = None
    fault_rows: list[int] = []
    real_fsync = os.fsync

    def fsync(fd: int) -> None:
        nonlocal at_fault
        for archive in path.parent.glob("audit.jsonl.*.gz"):
            if not _fd_matches(fd, archive):
                continue
            with gzip.open(archive, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            assert any(row["seq"] == expected_seq for row in rows), "fault missed admission"
            at_fault = _live_audit_copy(path)
            _assert_prior_copy_witness(identity, at_fault, at_fault, expected)
            fault_rows.append(expected_seq)
            raise OSError("injected synthetic admission archive fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    with log._write_lock:  # noqa: SLF001
        log._max_bytes = original.st_size + 1  # noqa: SLF001
        assert log.record_required(event) is False
        assert fault_rows == [expected_seq], "fault did not hit exact real append generation"
        _assert_prior_copy_witness(identity, at_fault, _live_audit_copy(path), expected)


@pytest.mark.parametrize(
    "damage",
    [
        "retired_at_fault",
        "retired_at_return",
        "wrong_inode",
        "wrong_sequence",
        "wrong_row",
        "changed_bytes",
    ],
)
def test_prior_copy_witness_rejects_invalid_boundary_snapshots(damage: str) -> None:
    # These are pure validator controls, NOT synthetic evidence of disk I/O.
    # The real paired-relay test above captures actual file descriptors/bytes.
    expected = {"seq": 7, "event": "session_file_read_admitted", "detail": {"offset": 0}}
    identity = (11, 22)
    content = (json.dumps(expected) + "\n").encode()
    at_fault: tuple[tuple[int, int], bytes] | None = (identity, content)
    at_return: tuple[tuple[int, int], bytes] | None = (identity, content)
    if damage == "retired_at_fault":
        at_fault = None
    elif damage == "retired_at_return":
        at_return = None
    elif damage == "wrong_inode":
        at_return = ((11, 23), content)
    elif damage == "wrong_sequence":
        at_return = (identity, json.dumps({**expected, "seq": 8}).encode())
    elif damage == "wrong_row":
        at_return = (identity, json.dumps({**expected, "detail": {"offset": 1}}).encode())
    else:
        at_return = (identity, content.replace(b": ", b":  "))
    with pytest.raises(AssertionError):
        _assert_prior_copy_witness(identity, at_fault, at_return, expected)
