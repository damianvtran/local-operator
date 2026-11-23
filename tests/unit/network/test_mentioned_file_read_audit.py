"""P3 persisted-byte acknowledgements; real writer/files, no transport claims."""

from __future__ import annotations

import dataclasses
import gzip
import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor

import pytest

from local_operator.network import audit


def identity(**changes):
    digest = "sha256:" + "a" * 64
    binding = {
        "subject_issuer": "https://synthetic.invalid",
        "subject_id": "synthetic-account",
        "owner_key_digest": digest,
        "requester_key_digest": digest,
        "native_account_digest": digest,
        "anchor_digest": digest,
        "owner_admission_digest": digest,
        "requester_admission_digest": digest,
        "association": "a" * 64,
        "authority_epoch": "synthetic-authority",
        "generation": 1,
        "journal_revision": 2,
        "deny_generation": 0,
        "evidence_digest": digest,
        "relay_instance": "synthetic-relay",
    }
    read = {
        "owner_key_digest": digest,
        "network_id": "synthetic-network",
        "network_epoch": 1,
        "link_id": "synthetic-link",
        "transcript_hash": "a" * 64,
        "request_id": "synthetic-request",
        "session_id": "synthetic-session",
        "source_row_id": "synthetic-source-row",
        "provenance_id": "a" * 32,
        "proof_digest": digest,
        "file_version": "establish:" + "b" * 32,
        "phase": "fetch",
        "offset": 0,
        "length": 7,
        "transfer_nonce": "c" * 32,
    }
    values = dict(
        operation="establish",
        binding_json=json.dumps(binding),
        read_binding_json=json.dumps(read),
        source_stamp=(1, 2, 3, 4, 7, 1, 0o100600),
        nonce="b" * 32,
    )
    return audit.MentionedFileAuditIdentity(**{**values, **changes})


def test_receipt_is_actual_new_durable_disk_row(tmp_path):
    log = audit.AuditLog(tmp_path)
    assert not log._path.exists()
    expected = identity()
    receipt = log.record_mentioned_file_required(expected)
    assert type(receipt) is audit.MentionedFileAdmissionReceipt
    assert log._path.read_bytes() == receipt.record
    info = log._path.stat()
    assert (info.st_dev, info.st_ino) == (receipt.device, receipt.inode)
    row = json.loads(receipt.record)
    assert row["detail"] == expected.detail()
    assert row["event"] == "mentioned_file_read_admitted"
    assert row["seq"] == receipt.sequence == 1
    assert receipt.generation == 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(receipt, "sequence", 4)


@pytest.mark.parametrize(
    "corruption", ["omit", "substitute", "truncate", "inode", "duplicate_conflict"]
)
def test_actual_disk_corruption_refused_while_intended_line_and_seq_unchanged(
    tmp_path, monkeypatch, corruption
):
    log = audit.AuditLog(tmp_path)
    original = log.record
    observations = []

    def corrupt(event):
        original(event)
        observations.append((log._required_line, log._seq))
        path = log._required_path
        assert path is not None
        if corruption == "omit":
            path.write_bytes(b"")
        elif corruption == "truncate":
            path.write_bytes(path.read_bytes()[:-3])
        else:
            row = json.loads(path.read_bytes())
            row["detail"]["read_binding"]["offset"] = 99
            raw = json.dumps(row).encode() + b"\n"
            if corruption == "duplicate_conflict":
                raw = path.read_bytes() + raw
            if corruption == "inode":
                replacement = path.with_suffix(".replacement")
                replacement.write_bytes(raw)
                os.replace(replacement, path)
            else:
                path.write_bytes(raw)
        assert (log._required_line, log._seq) == observations[-1]

    monkeypatch.setattr(log, "record", corrupt)
    assert log.record_mentioned_file_required(identity()) is None
    assert observations and observations[0][1] == 1
    assert observations[0][0] is not None
    assert json.loads(observations[0][0])["detail"]["read_binding"]["offset"] == 0


@pytest.mark.parametrize(
    "failure", ["disabled", "degraded", "append", "file_fsync", "dir_fsync", "readback"]
)
def test_every_durability_failure_refuses(tmp_path, monkeypatch, failure):
    log = audit.AuditLog(tmp_path, enabled=failure != "disabled")
    if failure == "degraded":
        log.degraded = True
    if failure == "append":
        monkeypatch.setattr(log, "flush", lambda: None)
    if failure in {"file_fsync", "dir_fsync"}:
        real_fsync = os.fsync

        def fsync(fd):
            directory = stat.S_ISDIR(os.fstat(fd).st_mode)
            if directory == (failure == "dir_fsync"):
                raise OSError("synthetic fsync failure")
            real_fsync(fd)

        monkeypatch.setattr(os, "fsync", fsync)
    if failure == "readback":
        real_open = os.open

        def opened(path, flags, *args, **kwargs):
            if path == log._path and flags & os.O_NOFOLLOW:
                raise OSError("synthetic readback failure")
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", opened)
    assert log.record_mentioned_file_required(identity()) is None


@pytest.mark.parametrize("when", ["before", "after"])
def test_rotation_receipt_reads_exact_archived_generation(tmp_path, when):
    log = audit.AuditLog(tmp_path)
    log.record(audit.AuditEvent(event="panic_raised"))
    log._max_bytes = log._path.stat().st_size + (1 if when == "after" else 0)
    receipt = log.record_mentioned_file_required(identity())
    assert receipt is not None
    archive = log._path.with_name(f"{log._path.name}.{receipt.generation}.gz")
    assert receipt.generation > 0
    with gzip.open(archive, "rb") as handle:
        assert receipt.record in list(handle)
    info = archive.stat()
    assert (info.st_dev, info.st_ino) == (receipt.device, receipt.inode)


@pytest.mark.parametrize(
    "section,key",
    [
        ("read_binding", k)
        for k in (
            "offset",
            "length",
            "phase",
            "request_id",
            "transfer_nonce",
            "link_id",
            "source_row_id",
            "proof_digest",
            "file_version",
        )
    ]
    + [
        ("binding", k)
        for k in ("subject_id", "requester_key_digest", "generation", "journal_revision")
    ],
)
def test_rendered_identity_substitution_never_admitted(tmp_path, monkeypatch, section, key):
    log = audit.AuditLog(tmp_path)
    real = audit._sanitize_detail

    def sanitize(event, detail):
        result = real(event, detail)
        if event == "mentioned_file_read_admitted":
            value = result[section][key]
            result[section][key] = value + 1 if type(value) is int else value + "different"
        return result

    monkeypatch.setattr(audit, "_sanitize_detail", sanitize)
    assert log.record_mentioned_file_required(identity()) is None


def test_signed_posix_timestamps_are_exact_metadata_not_an_age_policy(tmp_path):
    log = audit.AuditLog(tmp_path)
    expected = identity(source_stamp=(1, 2, -3, -4, 7, 1, 0o100600))
    receipt = log.record_mentioned_file_required(expected)
    assert receipt is not None
    assert json.loads(receipt.record)["detail"]["source_stamp"] == [1, 2, -3, -4, 7, 1, 0o100600]


def test_legacy_api_cannot_admit_p3_without_readback(tmp_path):
    log = audit.AuditLog(tmp_path)
    with pytest.raises(ValueError, match="P3 requires"):
        log.record_required(
            audit.AuditEvent(event="mentioned_file_read_admitted", detail=identity().detail())
        )
    assert not log._path.exists()


def test_corrupt_archive_cannot_borrow_rotated_durability_ack(tmp_path, monkeypatch):
    log = audit.AuditLog(tmp_path, max_bytes=1)
    real = log.record

    def record(event):
        real(event)
        path = log._required_path
        assert path is not None
        assert path.name.endswith(".gz")
        with gzip.open(path, "rb") as handle:
            lines = list(handle)
        intended = log._required_line
        assert intended is not None
        row = json.loads(intended)
        row["detail"]["binding"]["generation"] += 1
        with gzip.open(path, "wb") as handle:
            for line in lines:
                handle.write(
                    json.dumps(row).encode() + b"\n" if line == intended.encode() else line
                )
        assert log._required_line == intended and log._required_durable

    monkeypatch.setattr(log, "record", record)
    assert log.record_mentioned_file_required(identity()) is None


def test_concurrent_strict_and_legacy_share_sequence_and_writer(tmp_path):
    log = audit.AuditLog(tmp_path)

    def write(i):
        if i % 2:
            return log.record_mentioned_file_required(identity()) is not None
        return log.record_required(
            audit.AuditEvent(
                event="session_file_read_admitted",
                detail={
                    "reference": "scratchpad://synthetic.bin",
                    "offset": i,
                    "admitted_bytes": 7,
                    "decision": "allow",
                },
            )
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(write, range(20)))
    rows = [json.loads(line) for line in log._path.read_text().splitlines()]
    assert [row["seq"] for row in rows] == list(range(1, 21))
    legacy = [r for r in rows if r["event"] == "session_file_read_admitted"]
    assert len(legacy) == 10
    assert all(
        set(r["detail"]) == {"reference", "offset", "admitted_bytes", "decision"} for r in legacy
    )


@pytest.mark.parametrize(
    "change",
    [
        {"operation": "unknown"},
        {"operation": "fetch"},  # An establishment marker is not a sealed version.
        {"source_stamp": (1, 2)},
        {"source_stamp": [1, 2, 3, 4, 7, 1, 0o100600]},
        {"source_stamp": (1, 2, 3, 4, 7, 2, 0o100600)},
        {"source_stamp": (1, 2, 3, 4, 7, 1, 0o120777)},
        {"nonce": "not-a-nonce"},
        {"read_binding_json": "{}"},
        {"binding_json": "{}"},
    ],
)
def test_closed_schema_is_not_generic_telemetry(tmp_path, change):
    log = audit.AuditLog(tmp_path)
    assert log.record_mentioned_file_required(identity(**change)) is None
    assert not log._path.exists()
