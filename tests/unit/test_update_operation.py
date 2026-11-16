"""Durable update intent/recovery on owned roots, not installation/handoff proof."""

import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator import install_transaction as tx


def start(root: Path) -> tx.UpdateOperation:
    """The durable-operation entry point, typed for the checker.

    The ``getattr``/``callable`` probe is what pins "the immutable normal routes
    have a durable operation entry"; it returns ``object`` to the analyzer, so the
    result is cast to the real type instead of leaving every ``.advance``/
    ``.identifier`` below untyped (pyright reported those as attribute errors).
    """
    entry = getattr(tx, "start_operation", None)
    assert callable(entry), "immutable normal routes need durable operation persistence"
    return cast(tx.UpdateOperation, entry(root, baseline="A", target="B", route="default-cli"))


def _row(root: Path, identifier: str) -> dict[str, Any]:
    """``operation_status`` non-None, narrowed once for every read below.

    The status reader answers ``None`` for an absent/unreadable record; a test
    that asserts fields on it genuinely requires the non-None case, so the assert
    lives here rather than being repeated (and forgotten) at each call site.
    """
    row = tx.operation_status(root, identifier)
    assert row is not None
    return row


def _occ_row(root: Path, identifier: str) -> dict[str, dict[str, Any]]:
    """``occurrence_status`` non-None, narrowed for the map reads below."""
    row = tx.occurrence_status(root, identifier)
    assert row is not None
    return row


@pytest.mark.parametrize(
    "data",
    [
        b'{"padding":"' + b"x" * (64 * 1024) + b'"}',
        ('{"padding":"' + "é" * (32 * 1024) + '"}').encode("utf-8"),
        b'{"padding":"\xff"}',
        b'{"nested":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}",
    ],
)
def test_recovery_reader_rejects_oversized_or_invalid_utf8_without_mutation(tmp_path, data):
    record = tmp_path / "legacy.json"
    record.write_bytes(data)
    before = record.stat().st_mtime_ns
    assert tx.read_record(record) is None
    assert record.read_bytes() == data
    assert record.stat().st_mtime_ns == before


def _nested(depth: int) -> bytes:
    """A dict whose deepest container sits at exactly ``depth`` levels."""
    return b'{"a":' + b"[" * (depth - 1) + b"0" + b"]" * (depth - 1) + b"}"


def test_recovery_reader_enforces_its_own_nesting_bound(tmp_path):
    """The bound is explicit, not the interpreter's recursion limit.

    Python 3.12+ parses a 2000-deep array without RecursionError, so a reader
    relying on that limit accepted it; the explicit bound refuses at 33 on every
    interpreter and accepts the deepest legitimate shape (32).
    """
    record = tmp_path / "depth.json"
    record.write_bytes(_nested(tx.MAX_RECOVERY_RECORD_DEPTH))
    assert tx.read_record(record) is not None
    record.write_bytes(_nested(tx.MAX_RECOVERY_RECORD_DEPTH + 1))
    assert tx.read_record(record) is None
    assert tx.parse_record_bytes(_nested(tx.MAX_RECOVERY_RECORD_DEPTH + 1)) is None


def test_depth_scan_is_linear_on_an_unterminated_escaped_quote_run():
    """Round 1, M1: a string-masking regex took 13.4 s on exactly this shape.

    65,533 B, under the 64 KiB cap: one quote then 32,766 escaped quotes, never
    closed. The single-pass scanner refuses it in milliseconds; the bound below
    is ~100x the measured cost so a slow CI host cannot flake, yet far under
    the quadratic behaviour it guards against.
    """
    import time

    data = b'"' + b'\\"' * 32766
    assert len(data) == 65_533
    started = time.perf_counter()
    assert tx.parse_record_bytes(data) is None
    assert time.perf_counter() - started < 2.0
    wrapped = b'{"s":' + data + b"}"
    started = time.perf_counter()
    assert tx.parse_record_bytes(wrapped) is None
    assert time.perf_counter() - started < 2.0


def test_escaped_backslash_ends_before_the_closing_quote():
    """``"\\\\"`` is a closed string: brackets after it are structure again."""
    deep = b'{"s":"\\\\","x":' + b"[" * 40 + b"0" + b"]" * 40 + b"}"
    assert tx.parse_record_bytes(deep) is None
    shallow = b'{"s":"\\\\","x":[[0]]}'
    assert tx.parse_record_bytes(shallow) == {"s": "\\", "x": [[0]]}


def test_recovery_reader_ignores_brackets_inside_strings(tmp_path):
    value = {"detail": "[[[[" * 40 + '\\"{{{{' * 20, "operation_id": "0" * 32}
    record = tmp_path / "strings.json"
    record.write_text(json.dumps(value))
    assert tx.read_record(record) == value


def test_recovery_reader_accepts_exact_byte_boundary(tmp_path):
    # The writer permits exactly 64 KiB; rejecting it here would strand valid
    # existing recovery evidence after upgrading the controller.
    data = b'{"padding":"' + b"x" * (64 * 1024 - 14) + b'"}'
    assert len(data) == 64 * 1024
    record = tmp_path / "boundary.json"
    record.write_bytes(data)
    assert tx.read_record(record) == {"padding": "x" * (64 * 1024 - 14)}


def test_recovery_reader_protects_hardlinks_and_special_files(tmp_path):
    sentinel = tmp_path / "sentinel"
    sentinel.write_text('{"operation_id":"' + "0" * 32 + '"}')
    linked = tmp_path / "linked.json"
    os.link(sentinel, linked)
    assert tx.read_record(linked) is None
    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo)
    assert tx.read_record(fifo) is None
    assert sentinel.read_text() == '{"operation_id":"' + "0" * 32 + '"}'


def test_recovery_reader_bounds_growth_after_fstat(tmp_path, monkeypatch):
    record = tmp_path / "growing.json"
    record.write_bytes(b'{"padding":"')
    original = os.fstat

    def grow_after_stat(descriptor):
        sample = original(descriptor)
        with record.open("ab") as stream:
            stream.write(b"x" * (64 * 1024) + b'"}')
        return sample

    def must_not_parse(_data):
        pytest.fail("oversized bytes reached the JSON parser")

    monkeypatch.setattr(tx.os, "fstat", grow_after_stat)
    monkeypatch.setattr(tx.json, "loads", must_not_parse)
    assert tx.read_record(record) is None


def test_capacity_refusal_keeps_the_existing_256_boundary(tmp_path):
    directory = tmp_path / tx.OPERATION_DIR
    directory.mkdir()
    # Admission has always counted ALL entries, not just parseable JSON. Do not
    # make foreign or unfinished names into spare slots in the diagnostic fix.
    for index in range(255):
        (directory / f"unknown-{index}").write_text("protected")
    operation = start(tmp_path)
    assert len(list(directory.iterdir())) == 256
    assert _row(tmp_path, operation.identifier)["phase"] == "queued"
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(tx.RecoveryCapacityError) as caught:
        start(tmp_path)
    assert caught.value.code == "update_recovery_capacity"
    assert caught.value.active_limit == 256
    assert caught.value.observed_at_least == 256
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


def test_incident_shaped_256_real_legacy_records_remain_protected(tmp_path):
    # This histogram comes from the operator-supplied incident, NOT a claim
    # about the contents or eligibility of the real Mac's recovery store.
    phases = ["partial"] * 181 + ["promoted"] * 67 + ["failed"] * 7 + ["prepared"]
    operations = []
    for phase in phases:
        operation = start(tmp_path)
        operation.advance(phase)
        operations.append(operation)
    directory = tmp_path / tx.OPERATION_DIR
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in directory.iterdir()
    }
    assert len(before) == 256
    report = tx.recovery_capacity_status(tmp_path)
    assert report["scan_complete"] is True
    assert report["observed_entries"] == report["protected_entries"] == 256
    assert report["protected_reasons"] == {"writer_finality_unproven": 256}
    assert report["retirable_entries"] == 0
    with pytest.raises(tx.RecoveryCapacityError):
        start(tmp_path)
    assert before == {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in directory.iterdir()
    }
    for operation in operations:
        assert _row(tmp_path, operation.identifier)["retirement"]["eligible"] is False


@pytest.mark.parametrize("phase", sorted(tx._OPERATION_PHASES))
def test_legacy_phases_and_dead_owners_remain_protected(tmp_path, monkeypatch, phase):
    operation = start(tmp_path)
    operation.advance(phase)
    monkeypatch.setattr(tx, "owner_alive", lambda _row: False)
    status = _row(tmp_path, operation.identifier)
    assert status["owner_state"] == "gone"
    assert status["phase"] == phase
    assert status["retirement"] == {"eligible": False, "reason": "writer_finality_unproven"}
    assert tx.recovery_capacity_status(tmp_path)["retirable_entries"] == 0
    # Existing ID lookup and late writer paths must still work. A dry run does
    # not take ownership from an old controller or move its durable record.
    operation.advance("partial", receipts=[{"state": "pending"}])
    assert _row(tmp_path, operation.identifier)["receipts"] == [{"state": "pending"}]


def test_capacity_dry_run_does_not_trust_a_claimed_terminal_seal(tmp_path):
    operation = start(tmp_path)
    record = tmp_path / tx.OPERATION_DIR / f"{operation.identifier}.json"
    row = json.loads(record.read_text())
    row["terminal_seal"] = {"version": 1, "central_services_complete": True}
    record.write_text(json.dumps(row))
    status = _row(tmp_path, operation.identifier)
    assert status["retirement"] == {"eligible": False, "reason": "unverified_terminal_seal"}
    report = tx.recovery_capacity_status(tmp_path)
    assert report["protected_reasons"] == {"unverified_terminal_seal": 1}
    assert report["retirable_entries"] == 0


def test_capacity_dry_run_never_creates_an_absent_root(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("a dry run must not acquire a transaction or probe owner liveness")

    monkeypatch.setattr(tx, "transaction", forbidden)
    monkeypatch.setattr(tx, "owner_alive", forbidden)
    root = tmp_path / "absent"
    assert tx.recovery_capacity_status(root) == {
        "active_limit": 256,
        "observed_entries": 0,
        "namespace_entries": {tx.OPERATION_DIR: 0, tx.V2_OPERATION_DIR: 0},
        "at_limit": False,
        "scan_complete": True,
        "protected_entries": 0,
        "retirable_entries": 0,
        "protected_reasons": {},
        "mutation_performed": False,
    }
    assert not root.exists()


def test_capacity_dry_run_protects_unsafe_unknown_and_legacy_entries(tmp_path, monkeypatch):
    operation = start(tmp_path)
    directory = tmp_path / tx.OPERATION_DIR
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("private unrelated bytes")
    (directory / ("1" * 32 + ".json")).symlink_to(sentinel)
    (directory / ("2" * 32 + ".json")).write_text("not JSON")
    (directory / "foreign").mkdir()
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    def forbidden(*_args, **_kwargs):
        pytest.fail("dry run acquired a transaction or probed owner liveness")

    monkeypatch.setattr(tx, "transaction", forbidden)
    monkeypatch.setattr(tx, "owner_alive", forbidden)
    report = tx.recovery_capacity_status(tmp_path)
    assert report["observed_entries"] == report["protected_entries"] == 4
    assert report["scan_complete"] is True
    assert report["retirable_entries"] == 0
    assert report["protected_reasons"] == {
        "writer_finality_unproven": 1,
        "unreadable_or_unsafe_record": 2,
        "unknown_namespace_entry": 1,
    }
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert (directory / f"{operation.identifier}.json").exists()


def test_capacity_dry_run_stops_after_the_first_overflow_entry(tmp_path, monkeypatch):
    directory = tmp_path / tx.OPERATION_DIR
    directory.mkdir()
    for index in range(300):
        (directory / f"{index:032x}.json").write_text(json.dumps({"operation_id": f"{index:032x}"}))
    original = tx.read_record
    reads = []

    def counted(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(tx, "read_record", counted)
    report = tx.recovery_capacity_status(tmp_path)
    assert report["observed_entries"] == report["protected_entries"] == 257
    assert report["scan_complete"] is False
    assert report["retirable_entries"] == 0
    assert len(reads) == 256
    assert sum(report["protected_reasons"].values()) == 257
    assert len(list(directory.iterdir())) == 300


def test_operation_survives_reread_with_candidate_and_partial_receipts(tmp_path):
    operation = start(tmp_path)
    operation.advance("preparing", candidate="owned-B")
    operation.advance("partial", receipts=[{"unit": "serve", "state": "unsupported"}])
    row = _row(tmp_path, operation.identifier)
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
    assert _row(tmp_path, operation.identifier)["phase"] == "queued"
    after = snapshot()
    assert before == after


def test_pending_observation_cannot_be_promoted_to_complete(tmp_path):
    operation = start(tmp_path)
    with pytest.raises(tx.InstallTransactionError, match="completion contract"):
        operation.advance("complete", receipts=[])
    assert _row(tmp_path, operation.identifier)["phase"] == "queued"


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
    row = _row(tmp_path, operation.identifier)
    assert row["owner_state"] == "unknown"
    assert row["phase"] == "queued"


def test_occurrence_running_then_terminal_round_trips_durably(tmp_path) -> None:
    operation = tx.start_operation(tmp_path, baseline="A", target="B", route="latest")
    tx.record_occurrence(
        tmp_path, operation.identifier, "sched-1", kind="scheduler", state="running"
    )
    row = _occ_row(tmp_path, operation.identifier)
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
    row = _occ_row(tmp_path, operation.identifier)
    assert row["sched-1"]["state"] == "terminal" and row["sched-1"]["terminal"] == "done"


def test_occurrence_map_absent_reads_none_not_empty(tmp_path) -> None:
    operation = tx.start_operation(tmp_path, baseline="A", target="B", route="latest")
    # No occurrences written yet: unreadable/absent must be None, NOT {} — the
    # successor treats None as "unknown" and {} as "nothing in flight".
    assert tx.occurrence_status(tmp_path, operation.identifier) is None
    tx.record_occurrence(tmp_path, operation.identifier, "r-1", kind="relay", state="running")
    assert set(_occ_row(tmp_path, operation.identifier)) == {"r-1"}


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
