"""The proactive cadence engine: timing, holds, the escalation budget.

Everything here calls the engine the way its callers do — ``reconcile`` over an
in-memory list (the session's path), ``ensure_armed``/``pause``/``resume`` over
the files (the boot and desktop paths) — so the tests pin the same seams
production uses rather than private helpers.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from local_operator.aida import proactive, state
from local_operator.harness.wake_types import WakeSchedule
from local_operator.wakes import store as wake_store
from tests.unit.aida.conftest import mark_met, write_config

SESSION_ID = "0123456789ab"


def _pin_cadence_away_from_now(root: Path, *, hours: float = 12) -> None:
    """Pin ``aida.cadence.at`` to a wall-clock time far from every extra due.

    WHY THIS EXISTS (a real CI failure, not hygiene): the spacing floor
    measures an extra against the CADENCE too, and the cadence due is a
    wall-clock occurrence — so a suite running inside the 90 minutes before
    09:00 local (CI's 03:39 UTC run did exactly this) refuses an ``in 4h``
    extra outright and these tests read red for the hour rather than for the
    code. Pinning the cadence ~12 h out keeps every due in play farther apart
    than the floor at any hour the suite can run.
    """
    at = datetime.fromtimestamp((int(time.time() * 1000) + int(hours * 3_600_000)) / 1000)
    write_config(root, {"aida": {"cadence": {"at": at.strftime("%H:%M")}}})


def _root_with_session(root: Path) -> Path:
    # The ATTACHMENT is part of a real creation: ``aida.bootstrap`` writes
    # ``agent="aida"`` beside the directory, and the effective action class is
    # read through it (``action_class.session_action_class``). A helper that
    # skips it builds a session no production path can produce — a classless
    # one, which reads reactive and (correctly) arms nothing.
    from local_operator.resume import write_session_attachment

    session_dir = root / "sessions" / SESSION_ID
    session_dir.mkdir(parents=True)
    write_session_attachment(session_dir, team="", agent="aida", goal="")
    state.update_state(root, session_id=SESSION_ID)
    # The steady state these cadence tests model: she has met the operator.
    mark_met(root)
    return root


def _entry(root: Path, schedules: list[dict[str, Any]]) -> dict[str, Any]:
    wake_store.write_entry(
        root, SESSION_ID, cwd=str(root / "sessions" / SESSION_ID), schedules=schedules
    )
    return wake_store.read_entry(root, SESSION_ID) or {}


def _row(wake_id: str, due_ms: int, message: str = "x") -> dict[str, Any]:
    return {
        "id": wake_id,
        "message": message,
        "next_due_at": due_ms,
        "every_ms": None,
        "created_at": due_ms - 1000,
        "fired_count": 0,
        "limit": None,
        "until_at": None,
        "request_id": None,
    }


# -- timing ------------------------------------------------------------------


def test_next_cadence_is_the_next_local_occurrence() -> None:
    day = datetime(2026, 9, 27, 0, 0, 0)
    at_0800 = int((day + timedelta(hours=8)).timestamp() * 1000)
    at_0900 = int((day + timedelta(hours=9)).timestamp() * 1000)
    at_2300 = int((day + timedelta(hours=23)).timestamp() * 1000)

    assert proactive.next_cadence_ms(at_0800, "09:00") == at_0900
    # Exactly ON the cadence time rolls to tomorrow: "next" means after now,
    # and firing at the same millisecond the user is reading the setting would
    # make a just-set time appear to fire immediately.
    assert proactive.next_cadence_ms(at_0900, "09:00") == at_0900 + 86_400_000
    assert proactive.next_cadence_ms(at_2300, "09:00") == at_0900 + 86_400_000


def test_cadence_at_parses_or_falls_back() -> None:
    now = 1_790_000_000_000
    assert proactive.next_cadence_ms(now, "07:15") == proactive.next_cadence_ms(now, "07:15")
    # An unusable value falls back to the default rather than raising: the
    # cadence must survive a hand-edited config.
    assert proactive.next_cadence_ms(now, "banana") == proactive.next_cadence_ms(
        now, proactive.DEFAULT_CADENCE_AT
    )


def test_the_default_check_in_is_0830_local() -> None:
    """The packaged default, pinned by LITERAL: 08:30 local, so her update
    lands at the start of the work day.

    Only the default moves — an install with ``aida.cadence.at`` set keeps its
    configured time (``test_policy_reads_the_config_and_the_env_switch``
    covers that side). Pinned literally here because every other test reaches
    the constant symbolically and a silent drift would pass them all.
    """
    day = datetime(2026, 9, 28, 0, 0, 0)
    at_0800 = int((day + timedelta(hours=8)).timestamp() * 1000)
    at_0830 = int((day + timedelta(hours=8, minutes=30)).timestamp() * 1000)
    assert proactive.DEFAULT_CADENCE_AT == "08:30"
    assert proactive.next_cadence_ms(at_0800, proactive.DEFAULT_CADENCE_AT) == at_0830


# -- policy ------------------------------------------------------------------


def test_policy_defaults_match_the_registry(isolated_root: Path) -> None:
    policy = proactive.policy(isolated_root)
    assert policy == proactive.CadencePolicy(
        enabled=proactive.DEFAULT_ENABLED,
        paused=proactive.DEFAULT_PAUSED,
        at=proactive.DEFAULT_CADENCE_AT,
        max_extra_per_day=proactive.DEFAULT_MAX_EXTRA_PER_DAY,
        min_gap_minutes=proactive.DEFAULT_MIN_GAP_MINUTES,
    )


def test_policy_reads_the_config_and_the_env_switch(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_config(
        isolated_root,
        {"aida": {"cadence": {"at": "07:30", "paused": True, "max_extra_per_day": 0}}},
    )
    policy = proactive.policy(isolated_root)
    assert policy.at == "07:30" and policy.paused and policy.max_extra_per_day == 0

    monkeypatch.setenv("LOCAL_OPERATOR_NO_AIDA", "1")
    assert proactive.policy(isolated_root).enabled is False


# -- load/fire holds ---------------------------------------------------------


def test_filter_on_load_drops_only_aida_rows_while_held(isolated_root: Path) -> None:
    rows = [
        WakeSchedule(id="w1", message="user", next_due_at=1),
        WakeSchedule(id="aida-cadence", message="x", next_due_at=2),
    ]
    assert [row.id for row in proactive.filter_on_load(rows, config_dir=isolated_root)] == [
        "w1",
        "aida-cadence",
    ]

    write_config(isolated_root, {"aida": {"cadence": {"paused": True}}})
    assert [row.id for row in proactive.filter_on_load(rows, config_dir=isolated_root)] == ["w1"]
    assert proactive.hold_active(isolated_root) is True

    # Disabled drops them too — the same rows, for the harder switch.
    write_config(isolated_root, {"aida": {"cadence": {"paused": False}, "enabled": False}})
    assert [row.id for row in proactive.filter_on_load(rows, config_dir=isolated_root)] == ["w1"]
    # (enabled=false is not "held" — it is disabled; both drop the rows.)
    assert proactive.delivery_allowed(isolated_root) is False


def test_delivery_allowed_says_no_while_paused(isolated_root: Path) -> None:
    assert proactive.delivery_allowed(isolated_root) is True
    write_config(isolated_root, {"aida": {"cadence": {"paused": True}}})
    assert proactive.delivery_allowed(isolated_root) is False


# -- reconcile ---------------------------------------------------------------


def test_reconcile_ensures_one_cadence_row(isolated_root: Path) -> None:
    _root_with_session(isolated_root)
    result = proactive.reconcile([], config_dir=isolated_root, session_id=SESSION_ID)
    assert [row.id for row in result.schedules] == [proactive.CADENCE_ID]
    assert result.changed is True

    again = proactive.reconcile(result.schedules, config_dir=isolated_root, session_id=SESSION_ID)
    assert again.changed is False
    assert len([row for row in again.schedules if row.id == proactive.CADENCE_ID]) == 1


def test_reconcile_drops_aida_rows_while_paused(isolated_root: Path) -> None:
    _root_with_session(isolated_root)
    write_config(isolated_root, {"aida": {"cadence": {"paused": True}}})
    rows = [
        WakeSchedule(id="w1", message="user", next_due_at=1),
        WakeSchedule(id="aida-cadence", message="x", next_due_at=2),
        WakeSchedule(id="aida-extra-1", message="y", next_due_at=3),
    ]
    result = proactive.reconcile(rows, config_dir=isolated_root, session_id=SESSION_ID)
    assert [row.id for row in result.schedules] == ["w1"]


def test_escalation_tray_arms_within_budget_and_records_it(isolated_root: Path) -> None:
    _root_with_session(isolated_root)
    _pin_cadence_away_from_now(isolated_root)
    now = int(time.time() * 1000)
    state.write_json(
        state.escalate_path(isolated_root),
        {"wakes": [{"in": "4h", "message": "check the deploy"}]},
    )
    result = proactive.reconcile([], config_dir=isolated_root, session_id=SESSION_ID, now_ms=now)
    ids = sorted(row.id for row in result.schedules)
    assert ids == ["aida-cadence", "aida-extra-1"]
    # The tray was consumed exactly once.
    assert not state.escalate_path(isolated_root).exists()
    ledger = (state.read_state(isolated_root) or {}).get("extras") or {}
    assert ledger.get("armed") == 1

    # Second request within budget but inside the spacing floor → refused
    # with an observable note, not silently.
    state.write_json(
        state.escalate_path(isolated_root),
        {"wakes": [{"in": "4h", "message": "again"}]},
    )
    second = proactive.reconcile(
        result.schedules, config_dir=isolated_root, session_id=SESSION_ID, now_ms=now
    )
    assert sorted(row.id for row in second.schedules) == ids
    assert second.notes and "spacing" in second.notes[0]


def test_escalation_budget_zero_disables_escalation(isolated_root: Path) -> None:
    _root_with_session(isolated_root)
    write_config(isolated_root, {"aida": {"cadence": {"max_extra_per_day": 0}}})
    state.write_json(state.escalate_path(isolated_root), {"wakes": [{"in": "4h"}]})
    result = proactive.reconcile([], config_dir=isolated_root, session_id=SESSION_ID)
    assert [row.id for row in result.schedules] == [proactive.CADENCE_ID]
    assert result.notes and "refused" in result.notes[0]


def test_escalation_string_shorthand(isolated_root: Path) -> None:
    _root_with_session(isolated_root)
    _pin_cadence_away_from_now(isolated_root)
    now = int(time.time() * 1000)
    state.write_json(state.escalate_path(isolated_root), {"wakes": ["in 4h", "at 23:59"]})
    result = proactive.reconcile([], config_dir=isolated_root, session_id=SESSION_ID, now_ms=now)
    extras = [row for row in result.schedules if row.id.startswith(proactive.EXTRA_ID_PREFIX)]
    # With the cadence pinned away, "in 4h" always clears the floor; "at 23:59"
    # may still be inside the floor of the cadence or the first extra depending
    # on the clock, which is why only the first is asserted.
    assert extras and extras[0].message == proactive.DEFAULT_EXTRA_MESSAGE


# -- pause / resume over the files ------------------------------------------


@pytest.mark.asyncio
async def test_pause_invokes_config_hold_and_supervisor_marker(isolated_root: Path) -> None:
    _root_with_session(isolated_root)
    now = int(time.time() * 1000)
    # Seeded through the SAME writer production uses (``arm.py``), because the
    # transcript is the authority for schedule state and the cancel path reads
    # it: stamping the index alone would describe a store no real process can
    # produce, and the cancel would (correctly) find no such row.
    from local_operator.wakes.arm import arm_wake

    # A surviving user row keeps the entry alive, which is the case held_at
    # exists for: without one, cancelling every aida row removes the entry and
    # there is nothing left to hold.
    await arm_wake(isolated_root, SESSION_ID, {"message": "user", "in": "2h"}, now_ms=now)
    await arm_wake(
        isolated_root,
        SESSION_ID,
        {"message": "cadence", "in": "30m"},
        wake_id=proactive.CADENCE_ID,
        now_ms=now,
    )

    outcome = await proactive.pause(isolated_root, SESSION_ID, now_ms=now)

    assert outcome.cancelled == (proactive.CADENCE_ID,)
    assert outcome.held is True
    policy = proactive.policy(isolated_root)
    assert policy.paused is True
    entry = wake_store.read_entry(isolated_root, SESSION_ID) or {}
    assert entry.get("held_at") == now
    assert [row["id"] for row in entry["schedules"]] == ["w1"]
    assert wake_store.is_held(entry) is True

    word = await proactive.resume(isolated_root, SESSION_ID, now_ms=now)
    assert word in {"armed", "present"}
    assert proactive.policy(isolated_root).paused is False
    entry = wake_store.read_entry(isolated_root, SESSION_ID) or {}
    assert "held_at" not in entry
    assert proactive.CADENCE_ID in [row["id"] for row in entry["schedules"]]


@pytest.mark.asyncio
async def test_ensure_armed_words(isolated_root: Path) -> None:
    # No session on disk → nothing to arm against.
    assert await proactive.ensure_armed(isolated_root, SESSION_ID) == "no-session"

    _root_with_session(isolated_root)
    assert await proactive.ensure_armed(isolated_root, SESSION_ID) == "armed"
    assert await proactive.ensure_armed(isolated_root, SESSION_ID) == "present"

    write_config(isolated_root, {"aida": {"cadence": {"paused": True}}})
    assert await proactive.ensure_armed(isolated_root, SESSION_ID) == "paused"

    write_config(isolated_root, {"aida": {"enabled": False}})
    assert await proactive.ensure_armed(isolated_root, SESSION_ID) == "disabled"


def test_status_reports_what_is_on_disk(isolated_root: Path) -> None:
    _root_with_session(isolated_root)
    st = proactive.status(isolated_root)
    assert st["enabled"] is True and st["paused"] is False
    assert st["session_id"] == SESSION_ID
    assert st["cadence_due_at"] is None  # nothing armed yet
    assert st["max_extra_per_day"] == proactive.DEFAULT_MAX_EXTRA_PER_DAY


# -- the tray's failure endings (review round 1, M1) --------------------------


@pytest.mark.asyncio
async def test_a_live_owner_mid_drain_puts_the_unarmed_requests_back(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M1a: a 503 restores the tray instead of destroying it.

    The sweep used to consume the tray, hit the owner refusal, and break with a
    comment promising the owner's reconcile would pick the rest up — a
    reconcile that reads the FILE this sweep had already unlinked. The requests
    vanished: no row, no note, nothing on disk.
    """
    _root_with_session(isolated_root)
    _pin_cadence_away_from_now(isolated_root)
    now = int(time.time() * 1000)
    _entry(isolated_root, [_row(proactive.CADENCE_ID, now + 60_000)])
    tray = {"wakes": [{"in": "4h", "message": "check the deploy"}]}
    state.write_json(state.escalate_path(isolated_root), tray)
    # A live owner, through the same predicate `arm.py` refuses on.
    monkeypatch.setattr("local_operator.wakes.supervisor.wedged_runtime", lambda *a: None)
    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda *a: (None, os.getpid()),
    )

    word = await proactive.ensure_armed(isolated_root, SESSION_ID, now_ms=now)

    assert word == "present", "the cadence row was already there"
    restored = json.loads(state.escalate_path(isolated_root).read_text())
    assert restored["wakes"] == tray["wakes"], "the refused request must go back"
    entry = wake_store.read_entry(isolated_root, SESSION_ID) or {}
    assert [row["id"] for row in entry.get("schedules") or []] == [proactive.CADENCE_ID]


def test_a_contended_lock_leaves_the_tray_unread(isolated_root: Path) -> None:
    """M1b: a contended lock must not eat the batch it could not process.

    The first cut consumed the tray BEFORE ``state.locked``, so a contended
    acquisition lost the whole batch while the note it produced claimed the
    opposite — "left unread" about a file that was already gone.
    """
    _root_with_session(isolated_root)
    tray = {"wakes": [{"in": "4h", "message": "keep me"}]}
    state.write_json(state.escalate_path(isolated_root), tray)
    holder = state.wake_lock(isolated_root)
    holder.acquire()
    try:
        result = proactive.reconcile([], config_dir=isolated_root, session_id=SESSION_ID)
    finally:
        holder.release()

    assert "left unread" in " ".join(result.notes), result.notes
    on_disk = json.loads(state.escalate_path(isolated_root).read_text())
    assert on_disk["wakes"] == tray["wakes"], "the note must be true: the file is still there"
    assert [row.id for row in result.schedules] == [proactive.CADENCE_ID]


@pytest.mark.asyncio
async def test_extras_armed_in_one_drain_respect_the_spacing_floor(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q1: the external drain enforces `min_gap_minutes` across its own arming.

    The in-session reconcile checks each new extra against the growing `kept`
    list; the external drain measured only against the rows it read at entry,
    so `in 1m` + `in 2m` both armed one minute apart under a 90-minute floor —
    two writers, two answers to a bound the README states once.
    """
    _root_with_session(isolated_root)
    _pin_cadence_away_from_now(isolated_root)
    now = int(time.time() * 1000)
    # The cadence row itself is parked six hours out as well — belt and braces,
    # because the drain's spacing check reads the CONFIG's next occurrence.
    _entry(isolated_root, [_row(proactive.CADENCE_ID, now + 6 * 60 * 60 * 1000)])
    monkeypatch.setattr("local_operator.wakes.supervisor.wedged_runtime", lambda *a: None)
    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record", lambda *a: (None, None)
    )
    state.write_json(
        state.escalate_path(isolated_root),
        {"wakes": [{"in": "1m", "message": "one"}, {"in": "2m", "message": "two"}]},
    )

    notes = await proactive._drain_tray_external(
        isolated_root, SESSION_ID, proactive.policy(isolated_root), now
    )

    entry = wake_store.read_entry(isolated_root, SESSION_ID) or {}
    ids = [row["id"] for row in entry.get("schedules") or []]
    assert ids.count("aida-extra-1") == 1
    assert "aida-extra-2" not in ids, "a second extra inside the floor must be refused"
    assert any("spacing" in note for note in notes), notes


@pytest.mark.asyncio
async def test_drain_notes_surface_at_the_default_log_level(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Q3: the drain's notes must be visible on a DEFAULT server.

    ``lop serve`` configures its console logging at the platform default
    (WARNING), so an INFO-emitting drain answered "the bound is observable
    rather than silent" with lines nobody could see. Pinned at the LEVEL, not
    the text: a record below WARNING would pass a naive capture and still be
    filtered by the default daemon — the exact round-2 miss.
    """
    _root_with_session(isolated_root)
    _pin_cadence_away_from_now(isolated_root)
    now = int(time.time() * 1000)
    _entry(isolated_root, [_row(proactive.CADENCE_ID, now + 60_000)])
    state.write_json(state.escalate_path(isolated_root), {"wakes": ["bogus"]})
    monkeypatch.setattr("local_operator.wakes.supervisor.wedged_runtime", lambda *a: None)
    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record", lambda *a: (None, None)
    )

    with caplog.at_level(logging.NOTSET, logger="local_operator.aida.proactive"):
        notes = await proactive._drain_tray_external(
            isolated_root, SESSION_ID, proactive.policy(isolated_root), now
        )

    assert notes and "invalid" in notes[0], notes
    matching = [
        record
        for record in caplog.records
        if record.name == "local_operator.aida.proactive" and "invalid" in record.getMessage()
    ]
    assert matching, [record.name for record in caplog.records]
    assert all(record.levelno >= logging.WARNING for record in matching), [
        record.levelno for record in matching
    ]


@pytest.mark.asyncio
async def test_a_mid_drain_refusal_restores_from_the_exact_position(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NIT-1: duplicates in one tray batch must not re-restore a processed one.

    ``requests.index(request)`` resolves to the FIRST EQUAL item, so a tray
    holding two identical requests would put back BOTH when the second one's
    arm was refused mid-drain — a redundant re-arm of one already handled.
    ``enumerate`` keeps the loop position as the remainder boundary. The
    spacing floor is pinned to 0 so the duplicate reaches its arm attempt
    (with the default 90-minute gap the floor refuses equal-due duplicates
    before any arm, which is why this needed its own test).
    """
    from local_operator.wakes.arm import WakeWriteError

    _root_with_session(isolated_root)
    _pin_cadence_away_from_now(isolated_root)
    write_config(isolated_root, {"aida": {"cadence": {"min_gap_minutes": 0}}})
    now = int(time.time() * 1000)
    _entry(isolated_root, [_row(proactive.CADENCE_ID, now + 60_000)])
    duplicate = {"in": "5m", "message": "the same request twice"}
    state.write_json(state.escalate_path(isolated_root), {"wakes": [duplicate, dict(duplicate)]})

    attempted: list[str] = []

    async def fake_arm(_root, _session_id, _payload, *, wake_id=None, now_ms=None):
        attempted.append(str(wake_id))
        if len(attempted) == 1:
            return None
        raise WakeWriteError("an owner appeared", status=503, code="wake_owner_busy")

    monkeypatch.setattr("local_operator.wakes.arm.arm_wake", fake_arm)

    notes = await proactive._drain_tray_external(
        isolated_root, SESSION_ID, proactive.policy(isolated_root), now
    )

    assert attempted == ["aida-extra-1", "aida-extra-2"], attempted
    assert any("live session" in note for note in notes), notes
    restored = json.loads(state.escalate_path(isolated_root).read_text())
    assert restored["wakes"] == [
        duplicate
    ], "exactly the refused second entry goes back — not the processed first one too"
