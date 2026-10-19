"""``librarian.proactive`` — the cadence policy, the bounded extras and the holds."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pytest

from local_operator.harness.wake_types import WakeSchedule
from local_operator.librarian import proactive, state
from tests.unit.aida.conftest import write_config

SID = "abc123456789"


def _seed(root: Path, *, session_id: str = SID, make_dir: bool = True) -> str:
    state.update_state(root, session_id=session_id)
    if make_dir:
        (root / "sessions" / session_id).mkdir(parents=True)
    return session_id


def test_policy_defaults_match_the_design() -> None:
    assert proactive.DEFAULT_CADENCE_AT == "03:30"
    assert proactive.DEFAULT_MAX_EXTRA_PER_DAY == 1
    assert proactive.DEFAULT_MIN_GAP_MINUTES == 720
    assert proactive.DEFAULT_ENABLED is True


def test_policy_reads_the_config_section(isolated_root: Path) -> None:
    write_config(
        isolated_root,
        {
            "librarian": {
                "enabled": False,
                "cadence": {
                    "at": "05:00",
                    "paused": True,
                    "max_extra_per_day": 0,
                    "min_gap_minutes": 30,
                },
            }
        },
    )
    pol = proactive.policy(isolated_root)
    assert (pol.enabled, pol.at, pol.paused, pol.max_extra_per_day, pol.min_gap_minutes) == (
        False,
        "05:00",
        True,
        0,
        30,
    )


def test_env_switch_disables_the_policy(isolated_root: Path, monkeypatch) -> None:
    monkeypatch.setenv(state.ENV_DISABLE, "1")
    assert proactive.policy(isolated_root).enabled is False
    assert proactive.hold_active(isolated_root) is True
    assert proactive.delivery_allowed(isolated_root) is False


def test_next_cadence_is_the_next_local_clock_occurrence() -> None:
    # 2023-11-14 12:00 in the runner's local zone.
    now = int(datetime(2023, 11, 14, 12, 0).timestamp() * 1000)
    due = proactive.next_cadence_ms(now, "03:30")
    expected = int(datetime(2023, 11, 15, 3, 30).timestamp() * 1000)
    assert due == expected
    # An unparseable time falls back to the default rather than raising.
    assert proactive.next_cadence_ms(now, "not-a-time") == due


def test_parse_cadence_at_rejects_out_of_range() -> None:
    assert proactive.parse_cadence_at("03:30") == (3, 30)
    assert proactive.parse_cadence_at("24:00") is None
    assert proactive.parse_cadence_at("3") is None


def test_reconcile_untouched_for_a_foreign_session(isolated_root: Path, monkeypatch) -> None:
    _seed(isolated_root, session_id="other-session")
    rows = [WakeSchedule(id="w1", message="user row", next_due_at=1, created_at=1)]
    result = proactive.reconcile(rows, config_dir=isolated_root, session_id="not-hers")
    assert result.schedules == rows and result.changed is False and not result.notes


def test_reconcile_arms_one_cadence_and_is_idempotent(isolated_root: Path) -> None:
    _seed(isolated_root)
    first = proactive.reconcile([], config_dir=isolated_root, session_id=SID)
    assert [row.id for row in first.schedules] == ["librarian-cadence"]
    assert first.changed is True
    assert first.schedules[0].notify is True
    second = proactive.reconcile(first.schedules, config_dir=isolated_root, session_id=SID)
    assert second.schedules == first.schedules
    assert second.changed is False
    # The due is mirrored for the one-catch-up rule.
    assert state.read_state(isolated_root)["cadence_due_ms"] == first.schedules[0].next_due_at


def test_reconcile_keeps_user_rows(isolated_root: Path) -> None:
    _seed(isolated_root)
    rows = [
        WakeSchedule(id="w1", message="mine", next_due_at=2, created_at=1),
        WakeSchedule(id="librarian-extra-1", message="extra", next_due_at=3, created_at=1),
    ]
    result = proactive.reconcile(rows, config_dir=isolated_root, session_id=SID)
    ids = [row.id for row in result.schedules]
    assert ids.count("w1") == 1 and "librarian-extra-1" in ids and "librarian-cadence" in ids


def test_reconcile_drops_engine_rows_when_paused(isolated_root: Path) -> None:
    _seed(isolated_root)
    write_config(isolated_root, {"librarian": {"cadence": {"paused": True}}})
    rows = [
        WakeSchedule(id="w1", message="mine", next_due_at=2, created_at=1),
        WakeSchedule(id="librarian-cadence", message="pass", next_due_at=3, created_at=1),
    ]
    result = proactive.reconcile(rows, config_dir=isolated_root, session_id=SID)
    assert [row.id for row in result.schedules] == ["w1"]
    assert result.changed is True


def test_one_catch_up_after_a_missed_day(isolated_root: Path) -> None:
    _seed(isolated_root)
    now = int(time.time() * 1000)
    state.update_state(isolated_root, cadence_due_ms=now - 2 * 24 * 3_600_000)
    result = proactive.reconcile([], config_dir=isolated_root, session_id=SID)
    row = result.schedules[0]
    assert row.id == "librarian-cadence"
    assert row.next_due_at <= now + 5_000, "a missed day is caught up now, not tomorrow"
    # And the catch-up is recorded, so the next pass does not stack another.
    assert state.read_state(isolated_root)["cadence_due_ms"] == row.next_due_at
    again = proactive.reconcile(result.schedules, config_dir=isolated_root, session_id=SID)
    assert again.schedules == result.schedules and again.changed is False


def test_filter_on_load_drops_rows_only_while_held(isolated_root: Path) -> None:
    rows = [
        WakeSchedule(id="w1", message="mine", next_due_at=2, created_at=1),
        WakeSchedule(id="librarian-cadence", message="pass", next_due_at=3, created_at=1),
    ]
    assert [r.id for r in proactive.filter_on_load(rows, isolated_root)] == [
        "w1",
        "librarian-cadence",
    ]
    write_config(isolated_root, {"librarian": {"cadence": {"paused": True}}})
    assert [r.id for r in proactive.filter_on_load(rows, isolated_root)] == ["w1"]


def test_extras_are_bounded_by_the_daily_budget(isolated_root: Path) -> None:
    _seed(isolated_root)
    write_config(isolated_root, {"librarian": {"cadence": {"min_gap_minutes": 0}}})
    state.extras_path(isolated_root).write_text(
        json.dumps({"requests": [{"in": "1h", "message": "look at the deck"}]})
    )
    result = proactive.reconcile([], config_dir=isolated_root, session_id=SID)
    extras = [row for row in result.schedules if row.id.startswith("librarian-extra-")]
    assert len(extras) == 1 and extras[0].message.startswith("look at the deck")
    assert proactive.QUIET_REPLY in extras[0].message
    assert not state.extras_path(isolated_root).exists()
    # A second request in the same local day is refused, and the note says why.
    state.extras_path(isolated_root).write_text(
        json.dumps({"requests": [{"in": "2h", "message": "again"}]})
    )
    second = proactive.reconcile(result.schedules, config_dir=isolated_root, session_id=SID)
    assert not [row for row in second.schedules if row.id == "librarian-extra-2"]
    assert any("budget exhausted" in note for note in second.notes)


@pytest.mark.asyncio
async def test_pause_writes_the_key_cancels_rows_and_stamps_held(isolated_root: Path) -> None:
    _seed(isolated_root)
    # Arm the row through the real external writer first: a hand-written index
    # entry has no transcript, and the cancel path (rightly) refuses a session
    # whose authoritative rows it cannot read.
    assert await proactive.ensure_armed(isolated_root, SID) == "armed"
    outcome = await proactive.pause(isolated_root, SID)
    assert proactive.policy(isolated_root).paused is True
    assert state.read_state(isolated_root)["paused_at"] is not None
    assert "librarian-cadence" in outcome.cancelled
    # The rowless-entry rule: cancelling her only row removes the index entry
    # entirely, so there is nothing left to stamp ``held_at`` on. ``held`` is
    # therefore False in this shape and the config key is the authority the
    # supervisor's own pause bit reads — the same answer Aida's pause gives.
    entry_path = isolated_root / "wakes" / f"{SID}.json"
    assert not entry_path.exists()

    await proactive.resume(isolated_root, SID)
    assert proactive.policy(isolated_root).paused is False
    entry = json.loads(entry_path.read_text())
    assert [row["id"] for row in entry["schedules"]] == ["librarian-cadence"]


@pytest.mark.asyncio
async def test_run_now_arms_one_row_due_now(isolated_root: Path) -> None:
    _seed(isolated_root)
    assert await proactive.run_now(isolated_root, SID) == "armed"
    entry = json.loads((isolated_root / "wakes" / f"{SID}.json").read_text())
    extra = [row for row in entry["schedules"] if row["id"].startswith("librarian-extra-")]
    assert len(extra) == 1
    assert extra[0]["next_due_at"] <= int(time.time() * 1000) + 5_000


@pytest.mark.asyncio
async def test_ensure_armed_is_idempotent_for_a_session_with_no_owner(
    isolated_root: Path,
) -> None:
    _seed(isolated_root)
    assert await proactive.ensure_armed(isolated_root, SID) == "armed"
    assert await proactive.ensure_armed(isolated_root, SID) == "present"


@pytest.mark.asyncio
async def test_ensure_armed_reports_the_holds(isolated_root: Path) -> None:
    _seed(isolated_root)
    write_config(isolated_root, {"librarian": {"cadence": {"paused": True}}})
    assert await proactive.ensure_armed(isolated_root, SID) == "paused"
    write_config(isolated_root, {"librarian": {"enabled": False}})
    assert await proactive.ensure_armed(isolated_root, SID) == "disabled"
    assert await proactive.ensure_armed(isolated_root, "") == "no-session"


def test_wake_label_follows_the_configured_name(isolated_root: Path) -> None:
    write_config(isolated_root, {"librarian": {"name": "Archivist"}})
    assert proactive.wake_display_label("librarian-cadence") == "Archivist's library pass"
    assert proactive.wake_display_label("librarian-extra-2") == "Archivist's extra library pass"
    assert proactive.wake_display_label("w7") == "w7"


def test_status_is_read_only(isolated_root: Path) -> None:
    _seed(isolated_root)
    before = sorted(p.name for p in isolated_root.rglob("*"))
    payload = proactive.status(isolated_root)
    assert payload["session_id"] == SID
    assert payload["last_run_at"] is None
    assert payload["max_extra_per_day"] == 1
    assert sorted(p.name for p in isolated_root.rglob("*")) == before


@pytest.mark.asyncio
async def test_external_extras_respect_the_floor_and_pause_removes_them(
    isolated_root: Path,
) -> None:
    """The external drain is held to the same 720-minute floor as the reconcile.

    Measured regression before this: a tray request "in 1h" armed a generic
    ``w1`` row minutes from the cadence, and pause — which drops only
    ``librarian-*`` rows — left it firing. The fake clock makes both halves
    deterministic: the sub-floor request must arm nothing, the allowed one must
    be a named ``librarian-extra-N``, and pause must cancel it.
    """
    base = int(datetime(2023, 11, 14, 12, 0).timestamp() * 1000)
    _seed(isolated_root)
    assert await proactive.ensure_armed(isolated_root, SID, now_ms=base) == "armed"

    # 20:00 is 7.5 h from the 03:30 cadence — inside the 12 h floor.
    state.extras_path(isolated_root).write_text(
        json.dumps({"requests": [{"in": "8h", "message": "too soon"}]})
    )
    assert await proactive.ensure_armed(isolated_root, SID, now_ms=base) == "present"
    entry = json.loads((isolated_root / "wakes" / f"{SID}.json").read_text())
    assert [row["id"] for row in entry["schedules"]] == [proactive.CADENCE_ID]

    # 18:00 the next day is 14.5 h from the cadence — outside the floor.
    state.extras_path(isolated_root).write_text(
        json.dumps({"requests": [{"in": "30h", "message": "organise a batch"}]})
    )
    await proactive.ensure_armed(isolated_root, SID, now_ms=base)
    entry = json.loads((isolated_root / "wakes" / f"{SID}.json").read_text())
    extras = [row for row in entry["schedules"] if row["id"].startswith("librarian-extra-")]
    assert [row["id"] for row in extras] == ["librarian-extra-1"]
    assert extras[0]["next_due_at"] == base + 30 * 3_600_000
    assert proactive.QUIET_REPLY in extras[0]["message"]

    outcome = await proactive.pause(isolated_root, SID, now_ms=base)
    assert set(outcome.cancelled) == {"librarian-cadence", "librarian-extra-1"}
    entry_path = isolated_root / "wakes" / f"{SID}.json"
    if entry_path.exists():
        entry = json.loads(entry_path.read_text())
        assert not [row for row in entry["schedules"] if proactive.is_librarian_row(row["id"])]
