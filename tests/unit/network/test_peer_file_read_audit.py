"""Owner audit-before-bytes, exercised through the existing paired-relay path.

The positive and failure controls use only APIs present on the actual base:
they must fail because unaudited content escaped, not because a new API is absent.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import audit, files, store
from tests.unit.network.test_mesh_file_pull import (  # noqa: F401 — fixture dependencies
    SESSION,
    Devices,
    _in_store,
    _pad,
    _pull,
    devices,
    mesh,
    pair,
)


def _admissions(root: Path) -> list[dict[str, Any]]:
    path = store.audit_path(root)
    if not path.exists():
        return []
    return [
        row
        for line in path.read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line)).get("event") == "session_file_read_admitted"
    ]


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
    real_fsync = os.fsync
    real_read = files._read_span
    real_ask = files._Asker.ask
    received = bytearray()

    def synced(fd: int) -> None:
        real_fsync(fd)
        path = store.audit_path(owner.root)
        if path.exists() and os.fstat(fd).st_ino == path.stat().st_ino:
            committed.update(row["seq"] for row in _admissions(owner.root))

    def check(offset: int, count: int) -> None:
        rows = _admissions(owner.root)
        assert rows, "content escaped without an owner audit row"
        row = rows[-1]
        assert row["seq"] in committed, "owner row was appended but not fsynced"
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
    rows = _admissions(owner.root)
    assert len(rows) == 2
    assert sum(row["detail"]["admitted_bytes"] for row in rows) == len(payload)
    assert not _admissions(requester.root), "the required receipt belongs to the owner"


@pytest.mark.parametrize("failure", ["append", "fsync", "disabled", "degraded"])
def test_failed_durable_audit_refuses_with_zero_content_bytes(
    mesh: Devices,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    owner, requester, _host, _port = mesh
    payload = b"must not leave owner"
    (_pad(owner) / "audit.bin").write_bytes(payload)
    source_reads: list[int] = []
    received: list[bytes] = []
    real_open, real_fsync = Path.open, os.fsync
    real_read, real_ask = files._read_span, files._Asker.ask
    audit_path = store.audit_path(owner.root)

    def open_file(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == audit_path and args and args[0] == "a":
            raise OSError("injected owner audit append failure")
        return real_open(path, *args, **kwargs)

    def fsync(fd: int) -> None:
        if audit_path.exists() and os.fstat(fd).st_ino == audit_path.stat().st_ino:
            raise OSError("injected owner audit fsync failure")
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
    elif failure == "fsync":
        monkeypatch.setattr(os, "fsync", fsync)
    elif failure == "disabled":
        monkeypatch.setattr(owner.audit, "_enabled", False)
    else:
        monkeypatch.setattr(owner.audit, "degraded", True)
    monkeypatch.setattr(files, "_read_span", read)
    monkeypatch.setattr(files._Asker, "ask", ask)
    rc, receipt = _pull(capsys, "scratchpad://audit.bin")
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


def test_long_pad_reference_is_not_silently_truncated(
    mesh: Devices,
    capsys: pytest.CaptureFixture[str],
) -> None:
    owner, _requester, _host, _port = mesh
    relative = "/".join(["x" * 80] * 3) + "/audit.bin"
    target = _pad(owner) / relative
    target.parent.mkdir(parents=True)
    payload = b"long reference"
    target.write_bytes(payload)
    rc, receipt = _pull(capsys, "scratchpad://" + relative)
    assert rc == 0, receipt
    assert _admissions(owner.root)[-1]["detail"]["reference"] == (
        "sha256:" + hashlib.sha256(payload).hexdigest()
    )


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
    assert not log.record_required(audit.AuditEvent(event="session_file_read_admitted"))
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
