"""Elapsed cadence through real registry, writer, Session and scheduler seams.

No provider/native app is needed: scripted sessions exercise accepted delivery
and transcript/index commits. Frozen epoch clocks distinguish an owed tick from
mere absence, and negative controls protect independent wake ownership.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from local_operator import settings_io
from local_operator.aida import proactive, state
from local_operator.config import ConfigManager
from local_operator.harness.wake_types import DueWake, WakeSchedule
from local_operator.session.session import WAKE_SCHEDULES_CUSTOM_TYPE
from local_operator.wakes import arm, store
from tests.unit.aida.conftest import mark_met, write_config
from tests.unit.aida.test_aida_session_hooks import make_session, seed_rows

HOUR = 3_600_000
NOW = 1_800_000_000_000
SID = "11223344aabb"
KEY = "aida.cadence.interval_hours"


def configure(root: Path, hours: int = 4) -> None:
    from local_operator.resume import write_session_attachment

    directory = root / "sessions" / SID
    directory.mkdir(parents=True, exist_ok=True)
    write_session_attachment(directory, team="", agent="aida", goal="")
    state.update_state(root, session_id=SID)
    mark_met(root)
    settings_io.write_setting(ConfigManager(config_dir=root), settings_io.BY_KEY[KEY], hours)


def row(root: Path, now: int = NOW, hours: int = 4) -> WakeSchedule:
    return proactive.cadence_schedule(now, config_dir=root, interval_hours=hours)


def cadence(rows: list[WakeSchedule]) -> WakeSchedule:
    return next(item for item in rows if item.id == proactive.CADENCE_ID)


def indexed_cadence(root: Path, session_id: str) -> dict[str, Any]:
    """The cadence row from the wake INDEX, narrowed for typing."""
    entry = store.read_entry(root, session_id) or {}
    return next(row for row in entry.get("schedules", []) if row.get("id") == proactive.CADENCE_ID)


def transcript_cadence(root: Path, session_id: str) -> WakeSchedule:
    """The authoritative transcript row, narrowed for typing."""
    current = proactive._transcript_cadence(root, session_id)
    assert current is not None
    return current


@pytest.mark.parametrize("invalid", [-1, 169, True, 4.5, "4"])
def test_registry_and_hand_edited_policy_share_bounds(isolated_root: Path, invalid: object) -> None:
    setting = settings_io.BY_KEY[KEY]
    assert settings_io.validate(setting, invalid) is not None
    write_config(isolated_root, {"aida": {"cadence": {"interval_hours": invalid}}})
    assert proactive.policy(isolated_root).interval_hours == 0


def test_registry_copy_reaches_tui_rows_and_english_fallback(isolated_root: Path) -> None:
    from local_operator.i18n import messages
    from local_operator.i18n.keys import wire_aida_cadence
    from local_operator.tui.widgets.settings_view import SettingsView

    setting = settings_io.BY_KEY[KEY]
    view = SettingsView(ConfigManager(config_dir=isolated_root))
    actual = next(
        item.setting
        for item in view._build_rows()
        if item.setting is not None and item.setting.key == KEY
    )
    assert actual is setting
    assert settings_io._t(wire_aida_cadence.interval_hours_label()) == setting.label
    for locale in ("en", "fr", "unknown"):
        assert (
            messages.render(
                "wire.aida_cadence.interval_hours.label",
                {},
                locale=locale,
            )
            == setting.label
        )
    section = next(section for section in settings_io.SECTIONS if section.name == "aida")
    assert section.scope is settings_io.Scope.LIVE


@pytest.mark.asyncio
async def test_actual_settings_api_numeric_row_and_rejections(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from local_operator.server.routes import settings as settings_routes

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "synthetic-test-token")
    app = FastAPI()
    app.state.config_manager = ConfigManager(config_dir=isolated_root)
    app.include_router(settings_routes.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer synthetic-test-token"},
    ) as client:
        response = await client.get("/v1/settings")
        assert response.status_code == 200
        actual = next(item for item in response.json()["result"]["settings"] if item["key"] == KEY)
        bounds = (actual["kind"], actual["value"], actual["minimum"], actual["maximum"])
        assert bounds == ("int", 0, 0, 168)
        assert actual["label"] == settings_io.BY_KEY[KEY].label
        response = await client.patch(f"/v1/settings/{KEY}", json={"value": 4})
        assert response.status_code == 200 and response.json()["result"]["value"] == 4
        assert proactive.policy(isolated_root).interval_hours == 4
        for invalid in (-1, 169, True, 4.5):
            response = await client.patch(f"/v1/settings/{KEY}", json={"value": invalid})
            assert response.status_code == 422
            assert proactive.policy(isolated_root).interval_hours == 4


def test_pending_due_is_the_spacing_authority_before_recovery_arm(isolated_root: Path) -> None:
    configure(isolated_root)
    first = row(isolated_root)
    assert (
        proactive._cadence_due(
            [],
            proactive.policy(isolated_root),
            NOW + 3 * HOUR,
            config_dir=isolated_root,
        )
        == first.next_due_at
    )


def test_registry_default_and_nested_consumer(isolated_root: Path) -> None:
    assert proactive.policy(isolated_root).interval_hours == 0
    setting = settings_io.BY_KEY[KEY]
    assert setting.path == ("aida", "cadence", "interval_hours")
    assert (setting.default, setting.minimum, setting.maximum) == (0, 0, 168)
    configure(isolated_root)
    assert proactive.policy(isolated_root).interval_hours == 4
    assert settings_io.read_setting(ConfigManager(config_dir=isolated_root), setting) == 4
    assert setting.label == "Check-in interval (hours)"


@pytest.mark.parametrize("date", [(2026, 3, 8), (2026, 11, 1)])
def test_interval_is_elapsed_across_both_dst_edges(date: tuple[int, int, int]) -> None:
    local = datetime(*date, 0, 30, tzinfo=ZoneInfo("America/New_York"))
    anchor = int(local.timestamp() * 1000)
    due = proactive.next_cadence_ms(anchor, interval_hours=4, anchor_ms=anchor)
    assert due - anchor == 4 * HOUR
    assert datetime.fromtimestamp(due / 1000, local.tzinfo).hour in (3, 5)


@pytest.mark.parametrize(
    "origin",
    [
        "foreign",
        "aida-cadence:v2:4:1:2",
        "aida-cadence:v1:4:0:1",
        "aida-cadence:v1:4:0:14400000:extra",
    ],
)
def test_unknown_or_malformed_origin_preserves_present_due(
    isolated_root: Path,
    origin: str,
) -> None:
    present = WakeSchedule(
        id=proactive.CADENCE_ID, message="owed", next_due_at=NOW + HOUR, request_id=origin
    )
    assert proactive.cadence_phase(present) is None
    assert (
        proactive.cadence_schedule(
            NOW,
            config_dir=isolated_root,
            interval_hours=4,
            existing=present,
        )
        is present
    )


def test_first_arm_failure_reuses_pending_due_not_a_new_anchor(isolated_root: Path) -> None:
    first = row(isolated_root)
    recovered = row(isolated_root, NOW + 3 * HOUR)
    assert recovered.next_due_at == first.next_due_at
    assert recovered.request_id == first.request_id
    overdue = row(isolated_root, NOW + 13 * HOUR)
    assert proactive.cadence_phase(overdue) == proactive.cadence_phase(first)
    assert overdue.next_due_at == NOW + 13 * HOUR + 1000


def test_only_accepted_retirement_advances_and_events_do_not_slide(isolated_root: Path) -> None:
    configure(isolated_root)
    first = row(isolated_root)
    independent = [
        WakeSchedule(id=key, message=key, next_due_at=NOW + 30 * HOUR)
        for key in ("w1", "aida-extra-1", "aida-greeting", "project-event")
    ]
    proactive.remember_cadence(isolated_root, [first])
    for offset in (0, 1, 3, 5, 13):
        result = proactive.reconcile(
            [*independent, first],
            config_dir=isolated_root,
            session_id=SID,
            now_ms=NOW + offset * HOUR,
        )
        assert cadence(result.schedules) == first
        assert [item for item in result.schedules if item.id != first.id] == independent
        assert result.changed is False
    # A plain removal is not a receipt; an owed pending due catches up once.
    absent = proactive.reconcile(
        independent, config_dir=isolated_root, session_id=SID, now_ms=NOW + 13 * HOUR
    )
    assert proactive.cadence_phase(cadence(absent.schedules)) == proactive.cadence_phase(first)
    delivered = proactive.reconcile(
        independent,
        config_dir=isolated_root,
        session_id=SID,
        now_ms=NOW + 13 * HOUR,
        cadence_fired=first,
    )
    following = cadence(delivered.schedules)
    assert following.next_due_at == NOW + 16 * HOUR
    assert following.request_id != first.request_id
    assert [item for item in delivered.schedules if item.id != first.id] == independent


@pytest.mark.asyncio
async def test_external_registry_to_transcript_and_index_idempotent_readback(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(isolated_root)
    independent = WakeSchedule(
        id="w1",
        message="continuous fallback",
        every_ms=4 * HOUR,
        next_due_at=NOW + HOUR,
        created_at=NOW,
    )
    await seed_rows(isolated_root, SID, [independent])
    monkeypatch.setattr(arm, "_ensure_supervisor", lambda *_: "test-no-install")
    assert await proactive.ensure_armed(isolated_root, SID, now_ms=NOW) == "armed"
    first = proactive._transcript_cadence(isolated_root, SID)
    assert first is not None and first.next_due_at == NOW + 4 * HOUR
    assert first.every_ms is None
    for offset in (HOUR, 3 * HOUR, 13 * HOUR):
        assert await proactive.ensure_armed(isolated_root, SID, now_ms=NOW + offset) == "present"
        assert proactive._transcript_cadence(isolated_root, SID) == first
    indexed = store.read_entry(isolated_root, SID)
    assert indexed is not None
    assert indexed["schedules"][0] == independent.model_dump()
    assert indexed["schedules"][1] == first.model_dump()
    # An index-only crash repair must not change the owed transcript occurrence.
    store.entry_path(isolated_root, SID).unlink()
    assert await proactive.ensure_armed(isolated_root, SID, now_ms=NOW + 20 * HOUR) == "present"
    assert transcript_cadence(isolated_root, SID) == first
    entry = store.read_entry(isolated_root, SID) or {}
    assert entry["schedules"][0] == independent.model_dump()


@pytest.mark.asyncio
async def test_stale_index_presence_cannot_hide_authoritative_absence(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(isolated_root)
    first = row(isolated_root)
    await seed_rows(isolated_root, SID, [])
    store.write_entry(isolated_root, SID, cwd=str(isolated_root), schedules=[first])
    monkeypatch.setattr(arm, "_ensure_supervisor", lambda *_: "test-no-install")
    assert await proactive.ensure_armed(isolated_root, SID, now_ms=NOW + HOUR) == "armed"
    assert transcript_cadence(isolated_root, SID).request_id == first.request_id
    assert indexed_cadence(isolated_root, SID)["next_due_at"] == first.next_due_at


@pytest.mark.asyncio
async def test_interval_registry_write_applies_through_live_config_watcher(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from local_operator.config_watch import ConfigWatcher

    configure(isolated_root)
    first = row(isolated_root)
    fallback = WakeSchedule(
        id="w1", message="continuous", every_ms=4 * HOUR, next_due_at=NOW + 100 * HOUR
    )
    await seed_rows(isolated_root, SID, [fallback, first])
    monkeypatch.setattr(time, "time", lambda: NOW / 1000)
    session = make_session(isolated_root, SID)
    watcher = ConfigWatcher(isolated_root)
    unsubscribe = watcher.subscribe(session._apply_config_change)
    try:
        settings_io.write_setting(
            ConfigManager(config_dir=isolated_root),
            settings_io.BY_KEY[KEY],
            6,
        )
        change = watcher.poll_now()
        assert change is not None and KEY in change.changed_keys
        for _ in range(400):
            # Wait on BOTH faces of the truth: the transcript commit and the
            # derived index, so a transient transcript-first/index-second
            # window is not mistaken for a failure to converge.
            index = store.read_entry(isolated_root, SID) or {}
            indexed = next(
                (item for item in index.get("schedules", []) if item["id"] == proactive.CADENCE_ID),
                None,
            )
            current = proactive._transcript_cadence(isolated_root, SID)
            if (
                current is not None
                and current.next_due_at == NOW + 6 * HOUR
                and indexed == current.model_dump()
            ):
                break
            await asyncio.sleep(0)
        assert cadence(list(session._wake.schedules)).next_due_at == NOW + 6 * HOUR
        assert transcript_cadence(isolated_root, SID).next_due_at == NOW + 6 * HOUR
        entry = store.read_entry(isolated_root, SID) or {}
        assert entry["schedules"][0] == fallback.model_dump()
    finally:
        unsubscribe()
        await session.dispose()


@pytest.mark.asyncio
async def test_pause_resume_keeps_phase_and_skips_suppressed_intervals(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(isolated_root)
    first = row(isolated_root)
    fallback = WakeSchedule(id="w1", message="fallback", next_due_at=NOW + HOUR, every_ms=4 * HOUR)
    await seed_rows(isolated_root, SID, [fallback, first])
    monkeypatch.setattr(arm, "_ensure_supervisor", lambda *_: "test-no-install")
    await proactive.pause(isolated_root, SID, now_ms=NOW + HOUR)
    assert proactive._transcript_cadence(isolated_root, SID) is None
    entry = store.read_entry(isolated_root, SID) or {}
    assert entry["schedules"] == [fallback.model_dump()]
    assert await proactive.resume(isolated_root, SID, now_ms=NOW + 13 * HOUR) == "armed"
    resumed = transcript_cadence(isolated_root, SID)
    assert resumed.next_due_at == NOW + 16 * HOUR
    assert proactive.cadence_phase(resumed) is not None
    phase = proactive.cadence_phase(resumed)
    assert phase is not None and phase["anchor_ms"] == NOW
    entry = store.read_entry(isolated_root, SID) or {}
    assert entry["schedules"][0] == fallback.model_dump()


@pytest.mark.asyncio
async def test_real_session_accept_retire_failed_persist_and_crash_mirror(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(isolated_root)
    first = row(isolated_root)
    proactive.remember_cadence(isolated_root, [first])
    await seed_rows(isolated_root, SID, [first])
    monkeypatch.setattr(time, "time", lambda: (NOW + 13 * HOUR) / 1000)
    session = make_session(isolated_root, SID)
    monkeypatch.setattr(session, "_prompt_messages", AsyncMock())
    try:
        # Normal delivery accepts this exact arm; no synthetic fired flag.
        await session._deliver_wake(DueWake(schedule=first, occurrence=1))
        assert session._aida_cadence_delivered == first
        append = session._transcript.append_custom
        monkeypatch.setattr(
            session._transcript,
            "append_custom",
            AsyncMock(side_effect=OSError("disk")),
        )
        with pytest.raises(OSError, match="disk"):
            await session._persist_wake_schedules([])
        assert session._aida_cadence_delivered == first
        assert proactive._transcript_cadence(isolated_root, SID) == first
        monkeypatch.setattr(session._transcript, "append_custom", append)
        mirror = proactive.remember_cadence

        def fail_mirror(*_args: object, **_kwargs: object) -> None:
            raise OSError("mirror")

        monkeypatch.setattr(proactive, "remember_cadence", fail_mirror)
        await session._persist_wake_schedules([])
        following = transcript_cadence(isolated_root, SID)
        assert following.next_due_at == NOW + 16 * HOUR
        assert session._aida_cadence_delivered is None
        monkeypatch.setattr(proactive, "remember_cadence", mirror)
        # The authoritative row wins over the stale sidecar after a restart.
        assert (
            await proactive.ensure_armed(
                isolated_root,
                SID,
                now_ms=NOW + 14 * HOUR,
            )
            == "present"
        )
        assert proactive._transcript_cadence(isolated_root, SID) == following
        saved = state.read_state(isolated_root)
        assert saved is not None and saved["cadence_phase"] == proactive.cadence_phase(following)
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_mode_switch_due_and_busy_keeps_accepted_origin_until_retirement(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(isolated_root)
    first = row(isolated_root)
    await seed_rows(isolated_root, SID, [first])
    monkeypatch.setattr(time, "time", lambda: (NOW + 13 * HOUR) / 1000)
    session = make_session(isolated_root, SID)
    monkeypatch.setattr(session, "_prompt_messages", AsyncMock())
    try:
        configure(isolated_root, 6)
        await session._aida_reconcile_now()
        # Scheduler loaded overdue row into grace; original receipt remains owed.
        current = cadence(list(session._wake.schedules))
        assert current.request_id == first.request_id
        session._is_streaming = True
        await session._deliver_wake(DueWake(schedule=current, occurrence=1))
        assert session._aida_cadence_delivered is not None
        assert session._aida_cadence_delivered.request_id == first.request_id
        await session._persist_wake_schedules([])
        following = transcript_cadence(isolated_root, SID)
        assert following.next_due_at == NOW + 19 * HOUR
        assert proactive.cadence_phase(following) is not None
        phase = proactive.cadence_phase(following)
        assert phase is not None and phase["interval_hours"] == 6
        assert phase["anchor_ms"] == NOW + 13 * HOUR
    finally:
        session._is_streaming = False
        await session.dispose()


@pytest.mark.asyncio
async def test_catchup_take_and_phantom_retirement_advance_once(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(isolated_root)
    first = row(isolated_root)
    await seed_rows(isolated_root, SID, [first])
    monkeypatch.setattr(time, "time", lambda: (NOW + 13 * HOUR) / 1000)
    session = make_session(isolated_root, SID)
    try:
        session._resume_grace_ends_ms = 0
        message = session._take_resume_catchup()
        assert message is not None
        assert session._aida_cadence_delivered == first
        assert session._take_resume_catchup() is None
        await session._persist_wake_schedules([])
        following = proactive._transcript_cadence(isolated_root, SID)
        assert following is not None and following.next_due_at == NOW + 16 * HOUR
        await session._persist_wake_schedules([following])
        assert proactive._transcript_cadence(isolated_root, SID) == following
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_real_scheduler_catchup_pump_keeps_elapsed_grid(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(isolated_root)
    first = row(isolated_root)
    await seed_rows(isolated_root, SID, [first])
    clock = [NOW + 13 * HOUR]
    monkeypatch.setattr(time, "time", lambda: clock[0] / 1000)
    session = make_session(isolated_root, SID)
    monkeypatch.setattr(session, "_ensure_wake_supervisor", lambda: None)
    try:
        clock[0] = int(session._resume_grace_ends_ms) + 1
        assert session._take_resume_catchup() is not None
        await session._wake.pump()
        following = transcript_cadence(isolated_root, SID)
        assert following.next_due_at == NOW + 16 * HOUR
        index = store.read_entry(isolated_root, SID) or {}
        assert index["schedules"] == [following.model_dump()]
        assert session._aida_cadence_delivered is None
    finally:
        await session.dispose()


def test_interval_daily_mode_switch_preserves_owed_and_daily_at(isolated_root: Path) -> None:
    configure(isolated_root)
    first = row(isolated_root)
    write_config(isolated_root, {"aida": {"cadence": {"interval_hours": 0, "at": "09:30"}}})
    future = proactive.reconcile([first], config_dir=isolated_root, session_id=SID, now_ms=NOW)
    assert cadence(future.schedules).next_due_at == proactive.next_cadence_ms(NOW, "09:30")
    owed = proactive.reconcile(
        [first], config_dir=isolated_root, session_id=SID, now_ms=NOW + 13 * HOUR
    )
    assert cadence(owed.schedules) == first
    accepted = proactive.reconcile(
        [], config_dir=isolated_root, session_id=SID, now_ms=NOW + 13 * HOUR, cadence_fired=first
    )
    assert cadence(accepted.schedules).next_due_at == proactive.next_cadence_ms(
        NOW + 13 * HOUR,
        "09:30",
    )


@pytest.mark.asyncio
async def test_stale_accepted_origin_cannot_spend_replacement(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure(isolated_root)
    first = row(isolated_root)
    await seed_rows(isolated_root, SID, [first])
    monkeypatch.setattr(time, "time", lambda: (NOW + HOUR) / 1000)
    session = make_session(isolated_root, SID)
    try:
        replacement = proactive.cadence_schedule(
            NOW + HOUR,
            config_dir=isolated_root,
            interval_hours=6,
        )
        await session._transcript.append_custom(
            WAKE_SCHEDULES_CUSTOM_TYPE,
            {"schedules": [replacement.model_dump()]},
        )
        session._aida_cadence_delivered = first
        configure(isolated_root, 6)
        await session._persist_wake_schedules([])
        assert transcript_cadence(isolated_root, SID) == replacement
        assert session._aida_cadence_delivered == first
    finally:
        await session.dispose()
