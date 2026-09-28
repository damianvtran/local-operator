"""The proactive cadence engine: timing, holds, the escalation budget.

Everything here calls the engine the way its callers do — ``reconcile`` over an
in-memory list (the session's path), ``ensure_armed``/``pause``/``resume`` over
the files (the boot and desktop paths) — so the tests pin the same seams
production uses rather than private helpers.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from local_operator.aida import proactive, state
from local_operator.harness.wake_types import WakeSchedule
from local_operator.wakes import store as wake_store
from tests.unit.aida.conftest import write_config

SESSION_ID = "0123456789ab"


def _root_with_session(root: Path) -> Path:
    (root / "sessions" / SESSION_ID).mkdir(parents=True)
    state.update_state(root, session_id=SESSION_ID)
    return root


def _entry(root: Path, schedules: list[dict]) -> dict:
    wake_store.write_entry(
        root, SESSION_ID, cwd=str(root / "sessions" / SESSION_ID), schedules=schedules
    )
    return wake_store.read_entry(root, SESSION_ID) or {}


def _row(wake_id: str, due_ms: int, message: str = "x") -> dict:
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
    now = int(time.time() * 1000)
    state.write_json(state.escalate_path(isolated_root), {"wakes": ["in 4h", "at 23:59"]})
    result = proactive.reconcile([], config_dir=isolated_root, session_id=SESSION_ID, now_ms=now)
    extras = [row for row in result.schedules if row.id.startswith(proactive.EXTRA_ID_PREFIX)]
    # "at 23:59" may be inside the spacing floor of 09:00 depending on the
    # clock; at least the "in 4h" one must arm.
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
