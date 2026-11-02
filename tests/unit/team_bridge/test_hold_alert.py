"""The long-hold alert: ONE owner notification when screening holds age past a threshold.

WHY. A conversation message held in screening is invisible to the person it is
for: the agent sees ``body: null`` and the owner sees nothing, for hours if the
screening route is down. ``Store.long_holds_readonly`` counts the conversation
holds older than the threshold and ``hold_alert.tick`` (called from the
connector loop) raises, revises and clears one keyed notification for it.

Pinned here: the store reader counts only conversation holds once per message and
skips the two owner-action waits; the alert is contentless (a count and the
threshold, no id, sender, body or age); a repeat inside the throttle does no
notification I/O; an unchanged count writes no new revision; a drained queue
sends exactly one low-priority "cleared" update to the session that holds the alert
(after a restart, a session change or a switch-off too, and never for a pruned row);
and nothing in the tick (or its
hook in the loop) can kill the connector.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from local_operator.notifications.store import NotificationStore
from local_operator.team_bridge import hold_alert
from local_operator.team_bridge.connector import loop as loop_mod
from local_operator.team_bridge.settings import (
    DEFAULT_HOLD_ALERT_AFTER_MINUTES,
    TeamBridgeSettings,
)
from local_operator.team_bridge.store import OwnerReceipt, Store
from tests.unit.team_bridge.test_screening_status import _composite, _owner, _raw_hold

KEY = "team-bridge-long-hold:" + _owner().endpoint_id
HOUR = 3600.0


@pytest.fixture(autouse=True)
def _fresh_process_state() -> Any:
    hold_alert._last_tick.clear()
    hold_alert._raised.clear()
    yield
    hold_alert._last_tick.clear()
    hold_alert._raised.clear()


def _enrolled(root: Path, *, session: str | None = "sess_cos") -> Store:
    store = Store(root)
    store.commit_enrollment(
        _owner(), OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id=session
    )
    return store


def _threshold(root: Path, value: Any) -> None:
    (root / "config.yml").write_text(
        yaml.safe_dump({"values": {"team_bridge": {"hold_alert_after_minutes": value}}}),
        encoding="utf-8",
    )


class Clock:
    """The injectable monotonic clock the throttle reads."""

    def __init__(self) -> None:
        self.value = 1000.0

    def __call__(self) -> float:
        return self.value


def _items(root: Path) -> list[dict[str, Any]]:
    return NotificationStore(root).list_items(state="all")[0]


async def _tick(root: Path, clock: Clock, now: float) -> str:
    return await hold_alert.tick(_owner(), root, now=now, monotonic=clock)


# -- Store.long_holds_readonly -------------------------------------------------


def test_long_holds_counts_old_conversation_holds_once_per_message(tmp_path: Path) -> None:
    store = _enrolled(tmp_path)
    now = 10 * HOUR
    _raw_hold(tmp_path, _composite("bcmsg_old1"), "provider_error", created_at=now - 3 * HOUR)
    # The same message under both spellings counts once, at the earliest time.
    _raw_hold(tmp_path, _composite("bcmsg_old2", 2), "provider_error", created_at=now - 2 * HOUR)
    _raw_hold(tmp_path, "bcmsg_old2", "provider_error", created_at=now - 5 * HOUR)
    _raw_hold(tmp_path, _composite("bcmsg_young"), "provider_error", created_at=now - 60)

    found = store.long_holds_readonly(_owner(), older_than_s=HOUR, now=now)

    assert found == {"count": 2}


def test_long_holds_excludes_owner_action_waits_pair_ids_and_damaged_rows(tmp_path: Path) -> None:
    store = _enrolled(tmp_path)
    now = 10 * HOUR
    old = now - 3 * HOUR
    _raw_hold(tmp_path, _composite("bcmsg_cos"), "open_chief_of_staff", created_at=old)
    _raw_hold(tmp_path, _composite("bcmsg_off"), "bridge_off", created_at=old)
    _raw_hold(tmp_path, "bmsg_pair1", "provider_error", created_at=old)
    _raw_hold(tmp_path, _composite("bcmsg_bad"), "provider_error", created_at="not-a-number")
    _raw_hold(tmp_path, _composite("bcmsg_kept"), "model_hold", created_at=old)

    found = store.long_holds_readonly(_owner(), older_than_s=HOUR, now=now)

    assert found == {"count": 1}


def test_long_holds_empty_absent_and_fenced_answer_zero(tmp_path: Path) -> None:
    zero = {"count": 0}
    assert Store(tmp_path / "absent").long_holds_readonly(_owner(), older_than_s=1) == zero
    assert not (tmp_path / "absent").exists()

    store = _enrolled(tmp_path)
    _raw_hold(tmp_path, _composite("bcmsg_a"), "model_hold", created_at=1.0)
    stale = _owner()
    object.__setattr__(stale, "enrollment_generation", 9)
    assert store.long_holds_readonly(stale, older_than_s=1, now=10 * HOUR) == zero
    assert store.long_holds_readonly(_owner(), older_than_s=1, now=10 * HOUR)["count"] == 1


# -- the threshold setting -----------------------------------------------------


@pytest.mark.parametrize("raw", ["soon", -1, 10081, True, False, 1.5, None, [60], {"a": 1}])
def test_unreadable_or_out_of_range_thresholds_read_the_default(raw: Any) -> None:
    settings = TeamBridgeSettings.from_values({"team_bridge": {"hold_alert_after_minutes": raw}})
    assert settings.hold_alert_after_minutes == DEFAULT_HOLD_ALERT_AFTER_MINUTES == 60


@pytest.mark.parametrize("raw", [0, 1, 45, 10080])
def test_in_range_thresholds_are_kept(raw: int) -> None:
    settings = TeamBridgeSettings.from_values({"team_bridge": {"hold_alert_after_minutes": raw}})
    assert settings.hold_alert_after_minutes == raw


def test_an_absent_threshold_reads_sixty() -> None:
    assert TeamBridgeSettings.from_values({}).hold_alert_after_minutes == 60


# -- tick ----------------------------------------------------------------------


def test_below_threshold_holds_raise_nothing(tmp_path: Path) -> None:
    _enrolled(tmp_path)
    now = 10 * HOUR
    _raw_hold(tmp_path, _composite("bcmsg_a"), "provider_error", created_at=now - 600)

    assert asyncio.run(_tick(tmp_path, Clock(), now)) == "none_held"
    assert not (tmp_path / "notifications.db").exists()


def test_an_old_hold_raises_exactly_one_contentless_notification(tmp_path: Path) -> None:
    _enrolled(tmp_path)
    now = 10 * HOUR
    secret_id = "bcmsg_abc123def456"
    _raw_hold(tmp_path, _composite(secret_id), "provider_error", created_at=now - 5 * HOUR)
    _raw_hold(tmp_path, _composite("bcmsg_second"), "provider_error", created_at=now - 2 * HOUR)

    assert asyncio.run(_tick(tmp_path, Clock(), now)) == "raised"

    [item] = _items(tmp_path)
    assert item["key"] == KEY
    assert item["session_id"] == "sess_cos"
    assert item["priority"] == "normal"
    assert "2" in item["body_md"] and "60" in item["body_md"]
    text = f"{item['title']} {item['body_md']}"
    for forbidden in (secret_id, "bcmsg_", "bcv_", "5 hours", "300", "acc_", "provider_error"):
        assert forbidden not in text


def test_a_repeat_inside_the_throttle_does_no_notification_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enrolled(tmp_path)
    now = 10 * HOUR
    _raw_hold(tmp_path, _composite("bcmsg_a"), "provider_error", created_at=now - 3 * HOUR)
    clock = Clock()
    assert asyncio.run(_tick(tmp_path, clock, now)) == "raised"

    calls: list[Any] = []
    monkeypatch.setattr(NotificationStore, "notify", lambda *a, **k: calls.append(k))
    monkeypatch.setattr(Store, "long_holds_readonly", lambda *a, **k: calls.append("store"))
    clock.value += hold_alert.THROTTLE_S - 1

    assert asyncio.run(_tick(tmp_path, clock, now)) == "throttled"
    assert calls == []


def test_the_same_count_after_the_gap_writes_no_new_revision(tmp_path: Path) -> None:
    _enrolled(tmp_path)
    now = 10 * HOUR
    _raw_hold(tmp_path, _composite("bcmsg_a"), "provider_error", created_at=now - 3 * HOUR)
    clock = Clock()
    asyncio.run(_tick(tmp_path, clock, now))
    [first] = _items(tmp_path)

    clock.value += hold_alert.THROTTLE_S
    # Older holds must not move the content: the body carries no age.
    assert asyncio.run(_tick(tmp_path, clock, now + HOUR)) == "unchanged"

    [again] = _items(tmp_path)
    assert again["revision"] == first["revision"]
    assert again["content_hash"] == first["content_hash"]


def test_a_changed_count_revises_the_same_item(tmp_path: Path) -> None:
    _enrolled(tmp_path)
    now = 10 * HOUR
    _raw_hold(tmp_path, _composite("bcmsg_a"), "provider_error", created_at=now - 3 * HOUR)
    clock = Clock()
    asyncio.run(_tick(tmp_path, clock, now))
    _raw_hold(tmp_path, _composite("bcmsg_b"), "provider_error", created_at=now - 3 * HOUR)

    clock.value += hold_alert.THROTTLE_S
    assert asyncio.run(_tick(tmp_path, clock, now)) == "updated"

    [item] = _items(tmp_path)
    assert item["revision"] == 2 and "2" in item["body_md"]


def test_a_drained_queue_sends_one_low_cleared_update_then_nothing(tmp_path: Path) -> None:
    store = _enrolled(tmp_path)
    now = 10 * HOUR
    stored = _composite("bcmsg_a")
    _raw_hold(tmp_path, stored, "provider_error", created_at=now - 3 * HOUR)
    clock = Clock()
    asyncio.run(_tick(tmp_path, clock, now))
    store.delete_hold(_owner(), stored)

    clock.value += hold_alert.THROTTLE_S
    assert asyncio.run(_tick(tmp_path, clock, now)) == "cleared"
    [cleared] = _items(tmp_path)
    assert cleared["priority"] == "low" and cleared["revision"] == 2

    clock.value += hold_alert.THROTTLE_S
    assert asyncio.run(_tick(tmp_path, clock, now)) == "none_held"
    assert _items(tmp_path)[0]["revision"] == 2


def test_nothing_standing_and_nothing_held_sends_nothing(tmp_path: Path) -> None:
    """With no alert in the inbox there is nothing to clear: no row is created."""

    _enrolled(tmp_path)
    assert asyncio.run(_tick(tmp_path, Clock(), 10 * HOUR)) == "none_held"
    assert not (tmp_path / "notifications.db").exists()


def _raise_then_forget(tmp_path: Path, store: Store, clock: Clock, now: float) -> str:
    """Raise the alert, then drop every in-process record (a restart)."""

    stored = _composite("bcmsg_a")
    _raw_hold(tmp_path, stored, "provider_error", created_at=now - 3 * HOUR)
    assert asyncio.run(_tick(tmp_path, clock, now)) == "raised"
    hold_alert._raised.clear()
    hold_alert._last_tick.clear()
    return stored


def test_a_restart_clears_an_alert_the_previous_process_left_standing(tmp_path: Path) -> None:
    store = _enrolled(tmp_path)
    now = 10 * HOUR
    clock = Clock()
    stored = _raise_then_forget(tmp_path, store, clock, now)
    store.delete_hold(_owner(), stored)

    assert asyncio.run(_tick(tmp_path, clock, now)) == "cleared"

    [item] = _items(tmp_path)
    assert item["priority"] == "low" and item["revision"] == 2
    clock.value += hold_alert.THROTTLE_S
    assert asyncio.run(_tick(tmp_path, clock, now)) == "none_held"  # an already-low item stays
    assert _items(tmp_path)[0]["revision"] == 2


def test_disabling_the_alert_clears_the_one_this_process_raised(tmp_path: Path) -> None:
    _enrolled(tmp_path)
    now = 10 * HOUR
    _raw_hold(tmp_path, _composite("bcmsg_a"), "provider_error", created_at=now - 3 * HOUR)
    clock = Clock()
    assert asyncio.run(_tick(tmp_path, clock, now)) == "raised"

    _threshold(tmp_path, 0)
    clock.value += hold_alert.THROTTLE_S
    assert asyncio.run(_tick(tmp_path, clock, now)) == "disabled"

    [item] = _items(tmp_path)
    assert item["priority"] == "low" and item["revision"] == 2
    assert hold_alert._raised == {}


def test_disabling_after_a_restart_clears_a_standing_alert_too(tmp_path: Path) -> None:
    store = _enrolled(tmp_path)
    now = 10 * HOUR
    clock = Clock()
    _raise_then_forget(tmp_path, store, clock, now)

    _threshold(tmp_path, 0)
    assert asyncio.run(_tick(tmp_path, clock, now)) == "disabled"

    assert _items(tmp_path)[0]["priority"] == "low"


def test_the_clear_goes_to_the_session_that_raised_it_after_the_designation_moves(
    tmp_path: Path,
) -> None:
    store = _enrolled(tmp_path, session="sess_old")
    now = 10 * HOUR
    stored = _composite("bcmsg_a")
    _raw_hold(tmp_path, stored, "provider_error", created_at=now - 3 * HOUR)
    clock = Clock()
    assert asyncio.run(_tick(tmp_path, clock, now)) == "raised"

    store.update_designated_session(_owner(), "sess_new")
    store.delete_hold(_owner(), stored)
    clock.value += hold_alert.THROTTLE_S
    assert asyncio.run(_tick(tmp_path, clock, now)) == "cleared"

    [item] = _items(tmp_path)
    assert item["session_id"] == "sess_old" and item["priority"] == "low"


def test_a_restart_after_the_designation_moved_still_clears_the_old_sessions_alert(
    tmp_path: Path,
) -> None:
    # Q3: the alert stands in sess_old; the process dies (no record), the
    # designation moves to sess_new and the holds drain.
    store = _enrolled(tmp_path, session="sess_old")
    now = 10 * HOUR
    clock = Clock()
    stored = _raise_then_forget(tmp_path, store, clock, now)
    store.update_designated_session(_owner(), "sess_new")
    store.delete_hold(_owner(), stored)

    assert asyncio.run(_tick(tmp_path, clock, now)) == "cleared"

    [item] = _items(tmp_path)  # nothing was created in sess_new
    assert item["session_id"] == "sess_old" and item["priority"] == "low"
    clock.value += hold_alert.THROTTLE_S
    assert asyncio.run(_tick(tmp_path, clock, now)) == "none_held"


def test_disabling_after_the_designation_moved_clears_the_old_sessions_alert(
    tmp_path: Path,
) -> None:
    store = _enrolled(tmp_path, session="sess_old")
    now = 10 * HOUR
    clock = Clock()
    _raise_then_forget(tmp_path, store, clock, now)
    store.update_designated_session(_owner(), "sess_new")
    _threshold(tmp_path, 0)

    assert asyncio.run(_tick(tmp_path, clock, now)) == "disabled"

    [item] = _items(tmp_path)
    assert item["session_id"] == "sess_old" and item["priority"] == "low"


def test_a_restart_after_a_move_leaves_a_dismissed_old_alert_dismissed(tmp_path: Path) -> None:
    store = _enrolled(tmp_path, session="sess_old")
    now = 10 * HOUR
    clock = Clock()
    stored = _raise_then_forget(tmp_path, store, clock, now)
    [raised] = _items(tmp_path)
    NotificationStore(tmp_path).apply("dismiss", ids=[(raised["id"], raised["revision"])])
    store.update_designated_session(_owner(), "sess_new")
    store.delete_hold(_owner(), stored)

    asyncio.run(_tick(tmp_path, clock, now))

    [item] = NotificationStore(tmp_path).list_items(state="dismissed")[0]
    assert item["session_id"] == "sess_old" and item["priority"] == "low"
    assert NotificationStore(tmp_path).list_items(state="unread")[0] == []


def test_a_session_change_while_held_moves_the_alert_and_clears_the_old_one(
    tmp_path: Path,
) -> None:
    store = _enrolled(tmp_path, session="sess_old")
    now = 10 * HOUR
    _raw_hold(tmp_path, _composite("bcmsg_a"), "provider_error", created_at=now - 3 * HOUR)
    clock = Clock()
    asyncio.run(_tick(tmp_path, clock, now))

    store.update_designated_session(_owner(), "sess_new")
    clock.value += hold_alert.THROTTLE_S
    assert asyncio.run(_tick(tmp_path, clock, now)) == "raised"

    by_session = {item["session_id"]: item for item in _items(tmp_path)}
    assert by_session["sess_new"]["priority"] == "normal"
    assert by_session["sess_old"]["priority"] == "low"


def test_a_pruned_alert_is_not_recreated_as_a_stray_cleared_item(tmp_path: Path) -> None:
    store = _enrolled(tmp_path)
    now = 10 * HOUR
    clock = Clock()
    stored = _raise_then_forget(tmp_path, store, clock, now)
    hold_alert._raised[_owner().endpoint_id] = "sess_cos"  # this process still believes it
    notifications = NotificationStore(tmp_path)
    with notifications._connect() as conn:  # simulate the retention prune of the row
        conn.execute("DELETE FROM items")
    store.delete_hold(_owner(), stored)

    assert asyncio.run(_tick(tmp_path, clock, now)) == "none_held"
    assert _items(tmp_path) == []
    assert hold_alert._raised == {}


def test_a_dismissed_alert_is_still_cleared_quietly_without_reviving(tmp_path: Path) -> None:
    store = _enrolled(tmp_path)
    now = 10 * HOUR
    clock = Clock()
    stored = _raise_then_forget(tmp_path, store, clock, now)
    notifications = NotificationStore(tmp_path)
    [item] = _items(tmp_path)
    notifications.apply("dismiss", ids=[(item["id"], item["revision"])])
    store.delete_hold(_owner(), stored)

    assert asyncio.run(_tick(tmp_path, clock, now)) == "cleared"

    assert notifications.list_items(state="unread")[0] == []  # still dismissed


def test_a_later_incident_revives_a_dismissed_alert(tmp_path: Path) -> None:
    store = _enrolled(tmp_path)
    now = 10 * HOUR
    stored = _composite("bcmsg_a")
    _raw_hold(tmp_path, stored, "provider_error", created_at=now - 3 * HOUR)
    clock = Clock()
    asyncio.run(_tick(tmp_path, clock, now))
    clock.value += hold_alert.THROTTLE_S
    store.delete_hold(_owner(), stored)
    asyncio.run(_tick(tmp_path, clock, now))  # cleared (low)
    notifications = NotificationStore(tmp_path)
    [item] = _items(tmp_path)
    notifications.apply("dismiss", ids=[(item["id"], item["revision"])])

    _raw_hold(tmp_path, _composite("bcmsg_c"), "provider_error", created_at=now - 3 * HOUR)
    clock.value += hold_alert.THROTTLE_S
    assert asyncio.run(_tick(tmp_path, clock, now)) == "raised"

    [revived] = notifications.list_items(state="unread")[0]
    assert revived["priority"] == "normal" and revived["key"] == KEY


def test_threshold_zero_disables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _enrolled(tmp_path)
    now = 10 * HOUR
    _raw_hold(tmp_path, _composite("bcmsg_a"), "provider_error", created_at=now - 30 * HOUR)
    _threshold(tmp_path, 0)

    assert asyncio.run(_tick(tmp_path, Clock(), now)) == "disabled"
    # Nothing standing to clear, so nothing is written (and nothing is created).
    assert not (tmp_path / "notifications.db").exists()


def test_the_threshold_is_read_fresh_on_every_run(tmp_path: Path) -> None:
    _enrolled(tmp_path)
    now = 10 * HOUR
    _raw_hold(tmp_path, _composite("bcmsg_a"), "provider_error", created_at=now - 30 * 60)
    clock = Clock()
    assert asyncio.run(_tick(tmp_path, clock, now)) == "none_held"

    _threshold(tmp_path, 10)
    clock.value += hold_alert.THROTTLE_S
    assert asyncio.run(_tick(tmp_path, clock, now)) == "raised"
    assert "10" in _items(tmp_path)[0]["body_md"]


@pytest.mark.parametrize("raw", ["soon", -5, 10081, True])
def test_a_garbage_threshold_acts_as_sixty_minutes(tmp_path: Path, raw: Any) -> None:
    _enrolled(tmp_path)
    now = 10 * HOUR
    _raw_hold(tmp_path, _composite("bcmsg_a"), "provider_error", created_at=now - 90 * 60)
    _threshold(tmp_path, raw)

    assert asyncio.run(_tick(tmp_path, Clock(), now)) == "raised"
    assert "60" in _items(tmp_path)[0]["body_md"]


def test_no_designated_session_sends_nothing(tmp_path: Path) -> None:
    _enrolled(tmp_path, session=None)
    now = 10 * HOUR
    _raw_hold(tmp_path, _composite("bcmsg_a"), "provider_error", created_at=now - 3 * HOUR)

    assert asyncio.run(_tick(tmp_path, Clock(), now)) == "no_session"
    assert not (tmp_path / "notifications.db").exists()


def test_a_notification_failure_is_a_closed_outcome_never_an_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enrolled(tmp_path)
    now = 10 * HOUR
    _raw_hold(tmp_path, _composite("bcmsg_a"), "provider_error", created_at=now - 3 * HOUR)

    def boom(*_: Any, **__: Any) -> Any:
        raise OSError("disk says CANARY-PATH")

    monkeypatch.setattr(NotificationStore, "notify", boom)

    assert asyncio.run(_tick(tmp_path, Clock(), now)) == "failed"


def test_a_store_failure_is_a_closed_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enrolled(tmp_path)

    def boom(*_: Any, **__: Any) -> Any:
        raise RuntimeError("damaged")

    monkeypatch.setattr(Store, "long_holds_readonly", boom)

    assert asyncio.run(_tick(tmp_path, Clock(), 10 * HOUR)) == "failed"


# -- the hook in the connector loop --------------------------------------------


def test_run_forever_survives_a_failing_hold_alert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _enrolled(tmp_path)
    owner = _owner()
    ticks: list[str] = []
    polls: list[float] = []
    stop = asyncio.Event()

    async def exploding_tick(*_: Any, **__: Any) -> str:
        ticks.append("tick")
        raise RuntimeError("CANARY-SECRET must not be logged")

    async def iteration(owner_: Any, **__: Any) -> dict[str, Any]:
        polls.append(time.monotonic())
        stop.set()
        return {"events": 0, "reissued": False, "resync": False, "handled": []}

    async def access_provider() -> Any:
        return object()

    monkeypatch.setattr(hold_alert, "tick", exploding_tick)
    monkeypatch.setattr(loop_mod, "run_iteration", iteration)

    async def drive() -> Any:
        return await asyncio.wait_for(
            loop_mod.run_forever(
                owner,
                config_dir=tmp_path,
                auth_store=None,
                access_provider=access_provider,
                owner_source=store.active_owner,
                stop=stop,
            ),
            10,
        )

    assert asyncio.run(drive()) is None
    assert ticks == ["tick"] and len(polls) == 1  # the iteration still ran after the failure
