"""Guard L1 outbox: durable, bounded, idempotent, inert.

Each test opens a FRESH ``GuardOutbox`` (or a raw sqlite3 connection) to read
back, so "durable" means committed to the file, not cached in an object.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import stat
from contextlib import closing
from pathlib import Path

import pytest

from local_operator.guard import decisions as gd
from local_operator.guard.outbox import (
    DECISION_TTL_DAYS,
    GUARD_OUTBOX_SCHEMA_VERSION,
    MAX_OUTBOX_ROWS,
    GuardOutbox,
    GuardOutboxRefused,
    OutboxDecisionSink,
    guard_outbox_path,
)

TODAY = dt.date(2026, 11, 27)


def _rec(outcome: str = "block", day: dt.date = TODAY, **kw: str) -> gd.GuardDecisionRecord:
    return gd.GuardDecisionRecord(outcome, "custom", "shell", day=day.isoformat(), **kw)


def test_construction_writes_nothing(tmp_path: Path) -> None:
    # A non-Team install that never appends must leave no artefact behind.
    GuardOutbox(tmp_path)
    assert not (tmp_path / "guard").exists()


def test_append_is_durable_across_instances_and_owner_only(tmp_path: Path) -> None:
    records = [_rec("block"), _rec("pass"), _rec("allow")]
    box = GuardOutbox(tmp_path)
    assert [box.append(r, today=TODAY).status for r in records] == ["stored"] * 3

    reopened = GuardOutbox(tmp_path)
    assert reopened.pending() == records  # same values, insertion order
    path = guard_outbox_path(tmp_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with closing(sqlite3.connect(path)) as raw:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == GUARD_OUTBOX_SCHEMA_VERSION
        assert raw.execute("SELECT COUNT(*) FROM decision_records").fetchone()[0] == 3


def test_append_is_idempotent_and_refuses_a_changed_meaning(tmp_path: Path) -> None:
    box = GuardOutbox(tmp_path)
    record = _rec("block")
    assert box.append(record, today=TODAY).status == "stored"
    assert box.append(record, today=TODAY).status == "duplicate"
    assert GuardOutbox(tmp_path).count() == 1

    changed = _rec("allow", record_id=record.record_id)
    with pytest.raises(GuardOutboxRefused) as refused:
        box.append(changed, today=TODAY)
    assert refused.value.code == "record_conflict"
    assert GuardOutbox(tmp_path).pending() == [record]


def test_size_is_bounded_and_spent_rows_make_room_first(tmp_path: Path) -> None:
    box = GuardOutbox(tmp_path, max_rows=4)
    stale = _rec("block", day=TODAY - dt.timedelta(days=DECISION_TTL_DAYS + 1))
    sent = _rec("warn")
    live = [_rec("pass"), _rec("allow")]
    for r in [stale, sent, *live]:
        assert box.append(r, today=TODAY).status == "stored"
    assert box.mark_sent([sent.record_id]) == 1

    # Full: the expired and the sent row give way; live rows are kept.
    newcomer = _rec("block")
    assert box.append(newcomer, today=TODAY).status == "stored"
    assert box.count() == 3
    assert box.pending() == [*live, newcomer]

    # Full of rows that still count: the NEW record is the one dropped.
    assert box.append(_rec("pass"), today=TODAY).status == "stored"
    assert box.append(_rec("warn"), today=TODAY).status == "dropped"
    assert GuardOutbox(tmp_path).count() == 4


def test_default_cap_matches_its_derivation() -> None:
    # 4 outcomes x 4 rule classes x c=20 per day x 3-day TTL, doubled.
    assert len(gd.OUTCOMES) * len(gd.RULE_CLASSES) * 20 * DECISION_TTL_DAYS * 2 <= MAX_OUTBOX_ROWS


def test_prune_drops_sent_and_expired(tmp_path: Path) -> None:
    box = GuardOutbox(tmp_path)
    old = _rec(day=TODAY - dt.timedelta(days=DECISION_TTL_DAYS + 1))
    edge = _rec(day=TODAY - dt.timedelta(days=DECISION_TTL_DAYS))
    sent = _rec()
    for r in (old, edge, sent):
        box.append(r, today=TODAY)
    box.mark_sent([sent.record_id])
    assert box.prune(today=TODAY) == 2
    assert box.pending() == [edge]


def test_unknown_generation_is_refused_not_recreated(tmp_path: Path) -> None:
    path = guard_outbox_path(tmp_path)
    path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(path)) as raw:
        raw.execute("CREATE TABLE someone_elses (x)")
        raw.commit()
    before = path.read_bytes()
    with pytest.raises(GuardOutboxRefused) as refused:
        GuardOutbox(tmp_path).append(_rec(), today=TODAY)
    assert refused.value.code == "outbox_damaged"
    assert path.read_bytes() == before


def test_a_row_edited_on_disk_to_a_non_enum_is_refused_on_read(tmp_path: Path) -> None:
    box = GuardOutbox(tmp_path)
    box.append(_rec(), today=TODAY)
    with closing(sqlite3.connect(guard_outbox_path(tmp_path))) as raw:
        raw.execute("UPDATE decision_records SET outcome = 'rm -rf /Users/x'")
        raw.commit()
    with pytest.raises(ValueError):
        box.pending()


def test_outbox_sink_through_emit_decision(tmp_path: Path) -> None:
    box = GuardOutbox(tmp_path)
    previous = gd.set_decision_sink(OutboxDecisionSink(box))
    try:
        gd.emit_decision("pass", "dangerous_delete", "shell")
    finally:
        gd.set_decision_sink(previous)
    (record,) = GuardOutbox(tmp_path).pending()
    assert (record.outcome, record.rule_class, record.tool_class) == (
        "pass",
        "dangerous_delete",
        "shell",
    )


def test_permissions_are_reasserted_on_every_open(tmp_path: Path) -> None:
    """L1 MINOR-3 (folded into L2a): a loosened file or directory is tightened again."""
    import os

    from local_operator.guard.decisions import GuardDecisionRecord

    outbox = GuardOutbox(tmp_path)
    outbox.append(GuardDecisionRecord("block", "custom", "shell"))
    os.chmod(outbox.path, 0o644)
    os.chmod(outbox.path.parent, 0o755)
    outbox.append(GuardDecisionRecord("warn", "custom", "shell"))
    assert outbox.path.stat().st_mode & 0o777 == 0o600
    assert outbox.path.parent.stat().st_mode & 0o777 == 0o700


def test_a_file_that_is_not_sqlite_is_the_damaged_refusal(tmp_path: Path) -> None:
    """L1 NIT-2: never a raw ``sqlite3.DatabaseError``; the file is left untouched."""
    from local_operator.guard.decisions import GuardDecisionRecord

    outbox = GuardOutbox(tmp_path)
    outbox.path.parent.mkdir(parents=True)
    outbox.path.write_bytes(b"not a database" * 200)
    with pytest.raises(GuardOutboxRefused) as refused:
        outbox.append(GuardDecisionRecord("block", "custom", "shell"))
    assert refused.value.code == "outbox_damaged"
    assert outbox.path.read_bytes().startswith(b"not a database")
