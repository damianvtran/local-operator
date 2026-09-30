"""The trigger consume protocol: one row per record, settled only after persist.

THE CONTRACT UNDER TEST (design §3.2): the engine is the ONE writer of her
schedule list, so a pending trigger record is consumed HERE — re-verified
against the authoritative staleness rule, armed as at most ONE
``aida-trigger-*`` row, and settled only after the caller's persist lands. The
tests drive the same seams production does (``reconcile`` over an in-memory
list, ``consume_triggers`` over a fabricated record, the session's
settle-after-persist hook over a stub transcript) rather than private helpers.

Fail-proof notes: the idempotence test re-consumes with the row PRESENT (a
version that re-armed unconditionally would show two rows); the settle test
checks the record file, so a version that deleted the record up front would
fail it; the class-hold test flips the same input the active test arms from.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.aida import proactive
from local_operator.harness.wake_types import WakeSchedule
from local_operator.wakes import store as wake_store
from local_operator.wakes import triggers
from tests.unit.aida.conftest import write_config

SESSION_ID = "aida00000001"
NOW_MS = int(time.time() * 1000)


def _root_with_row(root: Path, *, stale: bool = True) -> Path:
    """An isolated root: her state, her session, and one project row."""
    session_dir = root / "sessions" / SESSION_ID
    session_dir.mkdir(parents=True)
    (session_dir / "transcript.jsonl").write_text("")
    (session_dir / "attachment.json").write_text(
        json.dumps({"team": "", "agent": "aida", "goal": ""})
    )
    (root / "aida").mkdir(exist_ok=True)
    (root / "aida" / "state.json").write_text(
        json.dumps({"schema_version": 1, "session_id": SESSION_ID})
    )
    stamp = time.time() - (6 * 3600 if stale else 60)
    (root / "projects").mkdir(exist_ok=True)
    (root / "projects" / "atlas.json").write_text(
        json.dumps(
            {
                "id": "atlas",
                "name": "atlas-migration",
                "title": "Atlas migration",
                "status": "active",
                "progress": "did a thing",
                "progress_updated_at": stamp,
                "sessions": [],
            }
        )
    )
    return root


def _record(
    root: Path, instances: list[dict[str, Any]], *, updated_at: int | None = None
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "target": SESSION_ID,
        "noted_at_ms": NOW_MS,
        "updated_at_ms": NOW_MS if updated_at is None else updated_at,
        "next_attempt_ms": 0,
        "attempts": 0,
        "instances": instances,
    }
    path = triggers.pending_path(root, SESSION_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload


def _instance_entry(key: str = "atlas") -> dict[str, Any]:
    return {
        "source": "project_staleness",
        "key": key,
        "fingerprint": [key, "active", int(time.time()) - 6 * 3600],
        "age_s": 21600,
        "payload": {
            "display_name": "Atlas migration",
            "status": "active",
            "progress_age_s": 21600,
            "sessions": [],
        },
    }


# -- consume ---------------------------------------------------------------------


def test_consume_arms_exactly_one_row_and_re_consume_is_idempotent(
    isolated_root: Path,
) -> None:
    root = _root_with_row(isolated_root)
    record = _record(root, [_instance_entry()])

    out = proactive.consume_triggers([], config_dir=root, session_id=SESSION_ID, now_ms=NOW_MS)
    assert out.changed is True
    assert len(out.schedules) == 1
    row = out.schedules[0]
    assert row.id == triggers.trigger_row_id(record["instances"])
    assert row.id.startswith("aida-trigger-")
    assert row.next_due_at == NOW_MS
    assert row.every_ms is None
    assert "Atlas migration" in row.message
    assert out.settle is not None and out.settle[0] == SESSION_ID

    # Re-consume with the row present: NO second row; skip-and-settle token.
    again = proactive.consume_triggers(
        list(out.schedules), config_dir=root, session_id=SESSION_ID, now_ms=NOW_MS + 1
    )
    assert again.changed is False
    assert len(again.schedules) == 1
    assert again.settle is not None


def test_a_refresh_between_record_and_consume_drops_the_instance(isolated_root: Path) -> None:
    root = _root_with_row(isolated_root)
    _record(root, [_instance_entry()])
    # The refresh lands after the record was written.
    row = json.loads((root / "projects" / "atlas.json").read_text())
    row["progress_updated_at"] = time.time()
    (root / "projects" / "atlas.json").write_text(json.dumps(row))

    out = proactive.consume_triggers([], config_dir=root, session_id=SESSION_ID, now_ms=NOW_MS)
    assert out.changed is False
    assert out.schedules == []
    assert out.settle is not None  # the record settles even though nothing armed
    assert triggers.read_pending_record(root, SESSION_ID) is not None  # not yet — caller settles


def test_a_missing_project_row_drops_the_instance(isolated_root: Path) -> None:
    root = _root_with_row(isolated_root)
    _record(root, [_instance_entry(key="ghost")])
    out = proactive.consume_triggers([], config_dir=root, session_id=SESSION_ID, now_ms=NOW_MS)
    assert out.changed is False and out.settle is not None


def test_an_unknown_source_is_left_alone(isolated_root: Path) -> None:
    root = _root_with_row(isolated_root)
    entry = _instance_entry()
    entry["source"] = "some_future_source"
    _record(root, [entry])
    out = proactive.consume_triggers([], config_dir=root, session_id=SESSION_ID, now_ms=NOW_MS)
    assert out.changed is False
    assert out.settle is not None and out.settle[1] == ()  # nothing verified ⇒ nothing settled


def test_the_cap_skips_with_a_note_and_does_not_settle(isolated_root: Path) -> None:
    root = _root_with_row(isolated_root)
    _record(root, [_instance_entry()])
    full = [
        WakeSchedule(id=f"w{i}", message="m", next_due_at=NOW_MS, created_at=NOW_MS)
        for i in range(16)
    ]
    out = proactive.consume_triggers(full, config_dir=root, session_id=SESSION_ID, now_ms=NOW_MS)
    assert out.changed is False
    assert len(out.schedules) == 16
    assert out.notes and "full" in out.notes[0]
    assert out.settle is None
    assert triggers.read_pending_record(root, SESSION_ID) is not None


# -- reconcile integration -------------------------------------------------------


def test_reconcile_appends_the_row_and_returns_the_settle_token(isolated_root: Path) -> None:
    root = _root_with_row(isolated_root)
    entry = _instance_entry()
    _record(root, [entry])

    result = proactive.reconcile([], config_dir=root, session_id=SESSION_ID, class_reactive=False)
    assert result.changed is True
    # The cadence/greeting rows arm in the same pass; the check-in must be
    # among them, with the deterministic id.
    assert triggers.trigger_row_id([entry]) in [row.id for row in result.schedules]
    assert result.settle is not None


def test_reconcile_on_a_hold_leaves_the_record(isolated_root: Path) -> None:
    root = _root_with_row(isolated_root)
    _record(root, [_instance_entry()])

    result = proactive.reconcile([], config_dir=root, session_id=SESSION_ID, class_reactive=True)
    assert result.changed is False
    assert result.settle is None
    assert triggers.read_pending_record(root, SESSION_ID) is not None

    write_config(root, {"aida": {"cadence": {"paused": True}}})
    result = proactive.reconcile([], config_dir=root, session_id=SESSION_ID, class_reactive=False)
    assert result.changed is False
    assert triggers.read_pending_record(root, SESSION_ID) is not None


def test_consume_skips_while_the_master_switch_is_off(isolated_root: Path) -> None:
    """Round 1 (R2): the master switch was creation-only — a record pending
    when it flipped still armed a row and fired a real check-in.

    ``consume_triggers`` now asks the same gate the pass and the supervisor
    use; the record is LEFT (nothing settles), so a re-enable inside its TTL
    reconsiders it and the TTL drops it otherwise.
    """
    root = _root_with_row(isolated_root)
    _record(root, [_instance_entry()])
    settings = root / "wakes" / "triggers" / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(
        json.dumps({"schema_version": 1, "values": {"wakes.triggers.enabled": False}})
    )

    out = proactive.consume_triggers([], config_dir=root, session_id=SESSION_ID, now_ms=NOW_MS)
    assert out.schedules == []
    assert out.settle is None
    assert triggers.read_pending_record(root, SESSION_ID) is not None

    settings.write_text(
        json.dumps({"schema_version": 1, "values": {"wakes.triggers.enabled": True}})
    )
    out = proactive.consume_triggers([], config_dir=root, session_id=SESSION_ID, now_ms=NOW_MS)
    assert len(out.schedules) == 1
    assert out.schedules[0].id.startswith("aida-trigger-")
    assert out.settle is not None


# -- settle and the session seams -------------------------------------------------


def test_settle_triggers_is_compare_and_delete(isolated_root: Path) -> None:
    root = _root_with_row(isolated_root)
    record = _record(root, [_instance_entry()])

    assert (
        proactive.settle_triggers(
            root,
            SESSION_ID,
            (SESSION_ID, (("project_staleness", "atlas"),), record["updated_at_ms"] - 1),
        )
        is False
    )
    assert triggers.read_pending_record(root, SESSION_ID) is not None
    assert (
        proactive.settle_triggers(
            root,
            SESSION_ID,
            (SESSION_ID, (("project_staleness", "atlas"),), record["updated_at_ms"]),
        )
        is True
    )
    assert triggers.read_pending_record(root, SESSION_ID) is None
    # A token for another target is refused.
    assert proactive.settle_triggers(root, SESSION_ID, ("someone-else", (("s", "k"),), 0)) is False


@pytest.mark.asyncio
async def test_the_session_settle_hook_journals_then_deletes(isolated_root: Path) -> None:
    root = _root_with_row(isolated_root)
    record = _record(root, [_instance_entry()])

    class _Transcript:
        def __init__(self) -> None:
            self.entries: list[tuple[str, dict[str, Any]]] = []

        async def append_custom(self, kind: str, payload: dict[str, Any]) -> None:
            self.entries.append((kind, payload))

    class _Stub:
        _session_id = SESSION_ID
        _aida_duty = True
        _aida_pending_trigger_settle: tuple[str, tuple[tuple[str, str], ...], int] | None = (
            SESSION_ID,
            (("project_staleness", "atlas"),),
            record["updated_at_ms"],
        )

        def __init__(self) -> None:
            self._transcript = _Transcript()

    stub = _Stub()
    from local_operator.session.session import Session

    # The unbound method is exercised against the stub on purpose (it only
    # needs ``_transcript``, ``_session_id`` and the stash), so the cast is
    # the honest spelling: this is not a real Session.
    await Session._aida_settle_triggers_after_persist(cast("Any", stub))

    kind, payload = stub._transcript.entries[0]
    assert kind == proactive.TRIGGER_CUSTOM_ENTRY_TYPE
    assert payload["target"] == SESSION_ID
    assert payload["instances"] == [["project_staleness", "atlas"]]
    assert stub._aida_pending_trigger_settle is None
    assert triggers.read_pending_record(root, SESSION_ID) is None


def test_has_pending_triggers_is_a_stat(isolated_root: Path) -> None:
    root = _root_with_row(isolated_root)
    assert proactive.has_pending_triggers(root, SESSION_ID) is False
    _record(root, [_instance_entry()])
    assert proactive.has_pending_triggers(root, SESSION_ID) is True


# -- message and label ------------------------------------------------------------


def test_the_message_summarises_overflow_and_clips_user_text() -> None:
    instances = [
        _instance_entry(),
        {
            **_instance_entry(key="billing"),
            "payload": {**_instance_entry()["payload"], "display_name": "x" * 500},
        },
    ]
    message = proactive.compose_trigger_message(instances, overflow=3)
    assert "and 3 more stale project(s)." in message
    assert "x" * 400 not in message  # clipped, never the raw field
    assert "message each linked session" in message
    assert "do not update the records yourself" in message


def test_the_trigger_row_gets_a_readable_label() -> None:
    label = proactive.wake_display_label("aida-trigger-1234abcd")
    assert "1234abcd" not in label
    assert "check-in" in label


# -- helpers the design pins ------------------------------------------------------


def test_consume_never_touches_a_missing_record(isolated_root: Path) -> None:
    root = _root_with_row(isolated_root)
    out = proactive.consume_triggers([], config_dir=root, session_id=SESSION_ID, now_ms=NOW_MS)
    assert out.changed is False and out.settle is None and out.schedules == []
    assert wake_store.read_entry(root, SESSION_ID) is None  # nothing engaged, nothing written
