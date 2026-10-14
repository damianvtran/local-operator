"""Durable update intent/recovery on owned roots, not installation/handoff proof."""

from pathlib import Path
import os

import pytest

from local_operator import install_transaction as tx


def start(root: Path):
    entry = getattr(tx, "start_operation", None)
    assert callable(entry), "immutable normal routes need durable operation persistence"
    return entry(root, baseline="A", target="B", route="default-cli")


def test_operation_survives_reread_with_candidate_and_partial_receipts(tmp_path):
    operation = start(tmp_path)
    operation.advance("preparing", candidate="owned-B")
    operation.advance("partial", receipts=[{"unit": "serve", "state": "unsupported"}])
    row = tx.operation_status(tmp_path, operation.identifier)
    assert row["baseline"] == "A" and row["target"] == "B"
    assert row["candidate"] == "owned-B" and row["phase"] == "partial"
    assert row["receipts"] == [{"unit": "serve", "state": "unsupported"}]
    assert row["owner_state"] == "live"


def test_observation_never_creates_root_or_acquires_lock(tmp_path):
    assert callable(
        getattr(tx, "operation_status", None)
    ), "status must be a read-only operation observer"
    root = tmp_path / "absent"
    assert tx.operation_status(root, "0" * 32) is None
    assert not root.exists()


def test_operation_directory_symlink_cannot_write_outside_root(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    root = tmp_path / "owned"
    root.mkdir()
    (root / ".update-operations").symlink_to(external)
    with pytest.raises(tx.InstallTransactionError, match="operation"):
        start(root)
    assert list(external.iterdir()) == []


def test_observer_preserves_all_record_and_lock_bytes(tmp_path):
    operation = start(tmp_path)

    def snapshot():
        return {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in tmp_path.rglob("*")
            if path.is_file()
        }

    before = snapshot()
    assert tx.operation_status(tmp_path, operation.identifier)["phase"] == "queued"
    after = snapshot()
    assert before == after


def test_pending_observation_cannot_be_promoted_to_complete(tmp_path):
    operation = start(tmp_path)
    with pytest.raises(tx.InstallTransactionError, match="completion contract"):
        operation.advance("complete", receipts=[])
    assert tx.operation_status(tmp_path, operation.identifier)["phase"] == "queued"


def test_operation_record_symlink_is_unknown_without_touching_sentinel(tmp_path):
    operation = start(tmp_path)
    record = tmp_path / tx.OPERATION_DIR / f"{operation.identifier}.json"
    external = tmp_path / "sentinel"
    external.write_text("not recovery authority")
    record.unlink()
    record.symlink_to(external)
    with pytest.raises(tx.InstallTransactionError, match="unknown"):
        tx.operation_status(tmp_path, operation.identifier)
    assert external.read_text() == "not recovery authority"


def test_interrupted_owner_identity_is_not_permission_to_complete(tmp_path, monkeypatch):
    operation = start(tmp_path)
    monkeypatch.setattr(tx, "owner_alive", lambda _record: None)
    row = tx.operation_status(tmp_path, operation.identifier)
    assert row["owner_state"] == "unknown"
    assert row["phase"] == "queued"


def test_occurrence_running_then_terminal_round_trips_durably(tmp_path) -> None:
    operation = tx.start_operation(tmp_path, baseline="A", target="B", route="latest")
    tx.record_occurrence(
        tmp_path, operation.identifier, "sched-1", kind="scheduler", state="running"
    )
    row = tx.occurrence_status(tmp_path, operation.identifier)
    assert row["sched-1"]["state"] == "running" and row["sched-1"]["terminal"] == ""
    assert row["sched-1"]["birth"] and row["sched-1"]["started_by_pid"] == os.getpid()
    tx.record_occurrence(
        tmp_path,
        operation.identifier,
        "sched-1",
        kind="scheduler",
        state="terminal",
        terminal="done",
    )
    row = tx.occurrence_status(tmp_path, operation.identifier)
    assert row["sched-1"]["state"] == "terminal" and row["sched-1"]["terminal"] == "done"


def test_occurrence_map_absent_reads_none_not_empty(tmp_path) -> None:
    operation = tx.start_operation(tmp_path, baseline="A", target="B", route="latest")
    # No occurrences written yet: unreadable/absent must be None, NOT {} — the
    # successor treats None as "unknown" and {} as "nothing in flight".
    assert tx.occurrence_status(tmp_path, operation.identifier) is None
    tx.record_occurrence(tmp_path, operation.identifier, "r-1", kind="relay", state="running")
    assert set(tx.occurrence_status(tmp_path, operation.identifier)) == {"r-1"}


def test_occurrence_rejects_bad_kind_state_and_terminal(tmp_path) -> None:
    operation = tx.start_operation(tmp_path, baseline="A", target="B", route="latest")
    with pytest.raises(tx.InstallTransactionError):
        tx.record_occurrence(tmp_path, operation.identifier, "x", kind="wat", state="running")
    with pytest.raises(tx.InstallTransactionError):
        tx.record_occurrence(tmp_path, operation.identifier, "x", kind="scheduler", state="")
    with pytest.raises(tx.InstallTransactionError):
        tx.record_occurrence(
            tmp_path, operation.identifier, "x", kind="scheduler", state="terminal"
        )
