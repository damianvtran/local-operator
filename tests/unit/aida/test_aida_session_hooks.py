"""The session-side hooks: load hold, persist reconcile, fire-time guard.

These build a REAL ``Session`` over the aida session directory — the same
construction the resume path performs — so what is pinned is the behaviour of
the three seams production relies on, not a private helper's contract.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from local_operator.aida import proactive, state
from local_operator.harness.types import ModelSpec
from local_operator.harness.wake_types import DueWake, WakeSchedule
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.wakes import store as wake_store
from tests.e2e.harness import ScriptedStream, text_turn
from tests.unit.aida.conftest import mark_met, write_config

MODEL = ModelSpec(provider="test", model_id="aida-model", context_window=100_000)


def make_session(root: Path, session_id: str) -> Session:
    """A real session over ``root/sessions/<id>``, pointed at ``root``.

    Carries the attachment a real ``aida.bootstrap`` writes (``agent="aida"``):
    the effective action class is read through it, and a session without one
    reads reactive — which would quietly disable the very hooks these tests
    exercise. For the id that is NOT in ``state.json`` the attachment changes
    nothing (``is_aida_session`` compares ids).
    """
    from local_operator.resume import write_session_attachment

    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    write_session_attachment(directory, team="", agent="aida", goal="")
    return Session(
        model=MODEL,
        model_source="config",
        stream_fn=ScriptedStream([text_turn("ok") for _ in range(4)]),
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: [],
        cwd=str(directory.parent),
    )


async def seed_rows(root: Path, session_id: str, rows: list[WakeSchedule]) -> None:
    """Seed schedule state through BOTH stores production writes together.

    The transcript entry is the authority the session loads from; the derived
    index is what the supervisor reads. Seeding only one would describe a store
    no real writer can produce.
    """
    from local_operator.session.session import WAKE_SCHEDULES_CUSTOM_TYPE

    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    await Transcript(directory).append_custom(
        WAKE_SCHEDULES_CUSTOM_TYPE, {"schedules": [row.model_dump() for row in rows]}
    )
    wake_store.write_entry(root, session_id, cwd=str(directory), schedules=rows)


@pytest.mark.asyncio
async def test_load_hold_drops_aida_rows_while_paused(isolated_root: Path) -> None:
    session_id = "aaaa11112222"
    state.update_state(isolated_root, session_id=session_id)
    now = int(time.time() * 1000)
    await seed_rows(
        isolated_root,
        session_id,
        [
            WakeSchedule(id="w1", message="user", next_due_at=now + 3_600_000),
            WakeSchedule(id="aida-cadence", message="cadence", next_due_at=now + 60_000),
        ],
    )
    write_config(isolated_root, {"aida": {"cadence": {"paused": True}}})

    session = make_session(isolated_root, session_id)
    try:
        assert session._aida_duty is True
        assert [row.id for row in session._wake.schedules] == ["w1"]
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_other_sessions_are_not_aida_sessions(isolated_root: Path) -> None:
    state.update_state(isolated_root, session_id="aaaa11112222")
    session = make_session(isolated_root, "bbbb33334444")
    try:
        assert session._aida_duty is False
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_persist_reconcile_rearms_the_cadence_and_drops_while_paused(
    isolated_root: Path,
) -> None:
    session_id = "cccc55556666"
    state.update_state(isolated_root, session_id=session_id)
    mark_met(isolated_root)
    session = make_session(isolated_root, session_id)
    try:
        assert session._aida_duty is True
        # Active: a persist with no aida rows gains the next cadence row (this
        # is how a fired one-shot is re-armed for tomorrow — the pump persists
        # the advanced list right after a delivery).
        await session._persist_wake_schedules([])
        entry = wake_store.read_entry(isolated_root, session_id) or {}
        assert [row["id"] for row in entry.get("schedules") or []] == [proactive.CADENCE_ID]

        # Paused: the same persist drops them again.
        write_config(isolated_root, {"aida": {"cadence": {"paused": True}}})
        await session._persist_wake_schedules(
            [
                WakeSchedule(
                    id=proactive.CADENCE_ID,
                    message="x",
                    next_due_at=int(time.time() * 1000) + 1000,
                )
            ]
        )
        entry = wake_store.read_entry(isolated_root, session_id)
        assert not (entry or {}).get("schedules")
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_fire_time_guard_drops_a_held_wake(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = "dddd77778888"
    state.update_state(isolated_root, session_id=session_id)
    session = make_session(isolated_root, session_id)
    prompts = AsyncMock()
    monkeypatch.setattr(session, "_prompt_messages", prompts)
    try:
        row = WakeSchedule(id=proactive.CADENCE_ID, message="cadence", next_due_at=1)
        due = DueWake(schedule=row, occurrence=1)

        # Active: the hook proceeds to the prompt path.
        await session._deliver_wake(due)
        assert prompts.called

        # Held: dropped at the door, so a pause that lands between a due time
        # and its delivery cannot send one last proactive turn.
        prompts.reset_mock()
        write_config(isolated_root, {"aida": {"cadence": {"paused": True}}})
        session._wake_fired_since_persist = False
        await session._deliver_wake(due)
        assert not prompts.called

        # A non-Aida row is never suppressed by her pause.
        other = WakeSchedule(id="w1", message="user", next_due_at=1)
        await session._deliver_wake(DueWake(schedule=other, occurrence=1))
        assert prompts.called
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_config_write_reaches_her_live_session(isolated_root: Path) -> None:
    """The live-apply proof the LIVE scope label promises, on HER session.

    A pause written through another ``ConfigManager`` must reach a session that
    is already running — that is the mechanism ``/aida pause`` relies on when
    her session lives in a different process (the desktop, a second terminal),
    and the reason the settings scope label is not a painted lie. The delivery
    goes through the platform's own registry-key diff
    (``ConfigWatcher`` -> ``Session._apply_config_change``), whose spawned
    reconcile is awaited here on a bounded loop of event-loop turns — a turn
    count, not a sleep-for, so contention cannot make it flaky.
    """
    from local_operator import settings_io
    from local_operator.config import ConfigManager
    from local_operator.config_watch import ConfigWatcher

    session_id = "eeee99990000"
    state.update_state(isolated_root, session_id=session_id)
    now = int(time.time() * 1000)
    await seed_rows(
        isolated_root,
        session_id,
        [WakeSchedule(id=proactive.CADENCE_ID, message="cadence", next_due_at=now + 60_000)],
    )
    session = make_session(isolated_root, session_id)
    watcher = ConfigWatcher(isolated_root)
    unsubscribe = watcher.subscribe(session._apply_config_change)
    try:
        assert [row.id for row in session._wake.schedules] == [proactive.CADENCE_ID]

        setting = settings_io.BY_KEY["aida.cadence.paused"]
        settings_io.write_setting(ConfigManager(config_dir=isolated_root), setting, True)
        change = watcher.poll_now()
        assert change is not None and "aida.cadence.paused" in change.changed_keys

        # WAIT ON THE DURABLE FACT, not on its in-memory shadow. The live
        # list empties first and the persist is the task's very next await, so
        # a loop that watches only the live list can be satisfied a turn
        # before the index write lands — which is exactly how this test read
        # red in a loaded batch run while passing in isolation. The subject of
        # this seam is the durable hold, so the loop waits for both.
        for _ in range(400):
            in_memory = not any(proactive.is_aida_row(row.id) for row in session._wake.schedules)
            index = wake_store.read_entry(isolated_root, session_id) or {}
            if in_memory and not index.get("schedules"):
                break
            await asyncio.sleep(0.01)
        assert [row.id for row in session._wake.schedules] == []
        # ...and the drop is durable: the persist that followed the reconcile
        # wrote the empty list to the index, so a restart keeps the hold.
        assert not (wake_store.read_entry(isolated_root, session_id) or {}).get("schedules")
    finally:
        unsubscribe()
        await session.dispose()


@pytest.mark.asyncio
async def test_renaming_her_session_syncs_the_config_name(isolated_root: Path) -> None:
    """Direction one (the operator's report): a rename of HER conversation
    stores ``aida.name``.

    The hook lives in ``Session.set_conversation_name`` — the ONE writer the
    TUI's ``/title``, the runtime's ``/rename``, the desktop and the phone all
    funnel through — so every rename gesture syncs config, not just one.
    """
    from local_operator.config import ConfigManager

    session_id = "ab12ab12ab12"
    state.update_state(isolated_root, session_id=session_id)
    session = make_session(isolated_root, session_id)
    try:
        assert session.set_conversation_name("Boss", user_set=True) == "Boss"
        assert ConfigManager(config_dir=isolated_root).get_nested_value(("aida", "name")) == "Boss"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_other_sessions_do_not_touch_the_name(isolated_root: Path) -> None:
    """A rename of ANY other conversation must not rename her."""
    from local_operator.config import ConfigManager

    state.update_state(isolated_root, session_id="ab12ab12ab12")
    session = make_session(isolated_root, "cd34cd34cd34")
    try:
        assert session._aida_duty is False
        assert session.set_conversation_name("Not her", user_set=True) == "Not her"
        assert (
            ConfigManager(config_dir=isolated_root).get_nested_value(("aida", "name"), None) is None
        )
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_generated_titles_never_rewrite_the_name(isolated_root: Path) -> None:
    """The auto-namer (``user_set=False``) is not the operator naming her.

    Adopting a generated title would let a fresh install's first turn
    silently rewrite a configured name.
    """
    from local_operator.config import ConfigManager

    session_id = "ef56ef56ef56"
    state.update_state(isolated_root, session_id=session_id)
    session = make_session(isolated_root, session_id)
    try:
        stored = session.set_conversation_name("Auto generated title", user_set=False)
        assert stored == "Auto generated title"
        assert (
            ConfigManager(config_dir=isolated_root).get_nested_value(("aida", "name"), None) is None
        )
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_name_write_reaches_her_live_session(isolated_root: Path) -> None:
    """Direction two, live: ``aida.name`` → the open session re-titles itself.

    The same registry-key-diff seam as the pause test above, and the subject
    is the fact every surface reads: ``session.conversation_name``. The
    rename must go through the ordinary writer, so the user-set claim moves
    with it (a generated title can never displace it afterwards).
    """
    from local_operator import settings_io
    from local_operator.config import ConfigManager
    from local_operator.config_watch import ConfigWatcher

    session_id = "1212ababcdcd"
    state.update_state(isolated_root, session_id=session_id)
    session = make_session(isolated_root, session_id)
    watcher = ConfigWatcher(isolated_root)
    unsubscribe = watcher.subscribe(session._apply_config_change)
    try:
        setting = settings_io.BY_KEY["aida.name"]
        settings_io.write_setting(ConfigManager(config_dir=isolated_root), setting, "Sovereign")
        change = watcher.poll_now()
        assert change is not None and "aida.name" in change.changed_keys

        for _ in range(400):
            if session.conversation_name == "Sovereign":
                break
            await asyncio.sleep(0.01)
        assert session.conversation_name == "Sovereign"
        assert session.conversation_name_state.user_set is True
    finally:
        unsubscribe()
        await session.dispose()


async def _wait_for_reply(session: Any) -> None:
    """Block until the session holds an assistant message — on its EVENTS.

    Re-tested after each event the session publishes, so it waits exactly as
    long as the turn takes. It replaced a ``range(200)`` x 10 ms poll that went
    red once in this lane at host load ~17 (AGENTS.md, "Wait on the event, never
    on the clock"); the ceiling below is a deadlock guard, not a budget.
    """
    changed = asyncio.Event()
    unsubscribe = session.subscribe(lambda *_: changed.set())

    def _replied() -> bool:
        # BOTH stores, because the assertions read both: ``history()`` is the
        # in-memory list the turn appends to first, and the desktop's history
        # window reads the TRANSCRIPT, whose durable append lands a beat later.
        # Waiting on the first alone left the window empty (measured: the 10 ms
        # poll this replaced only passed because it gave the writer that beat).
        in_memory = any(getattr(m, "role", "") == "assistant" for m in session.history())
        durable = any(
            (e.payload.get("message") or e.payload).get("role") == "assistant"
            for e in session._transcript.entries()
            if isinstance(e.payload, dict)
        )
        return in_memory and durable

    async def _loop() -> None:
        while True:
            changed.clear()
            if _replied():
                return
            await changed.wait()

    try:
        await asyncio.wait_for(_loop(), 120.0)
    except asyncio.TimeoutError:
        raise AssertionError("no assistant reply was published: wedged, not slow") from None
    finally:
        unsubscribe()


@pytest.mark.asyncio
async def test_the_greeting_is_hidden_on_every_surface_and_stamps_delivery(
    isolated_root: Path,
    attended_surface: None,
) -> None:
    """Her reply is the FIRST VISIBLE row; the trigger reaches only the model.

    Drives the real delivery (``Session._deliver_wake``) over a real
    transcript with a scripted provider, then reads the conversation back the
    way each surface does: the model's request (must carry the trigger), the
    desktop's history window, and the phone's fold (neither may show it). The
    ledger moves to ``delivered`` at THIS fire, not at arm time (audit
    A1/A3/A4).
    """
    from local_operator.aida import onboarding
    from local_operator.harness.types import WakeDeliveredEvent
    from local_operator.mobile.projection import fold_messages_to_entries
    from local_operator.session.history_window import display_window

    session_id = "eeee99990000"
    state.update_state(isolated_root, session_id=session_id)
    onboarding.request_greeting(isolated_root, "tui")
    onboarding.mark_greeted(isolated_root, 1)
    session = make_session(isolated_root, session_id)
    # The session's provider double is the scripted stream the harness built;
    # cast because the session's own annotation is the callable it was given.
    stream = cast(Any, session._stream_fn)
    events: list[object] = []
    session.subscribe(events.append)
    try:
        trigger = onboarding.greeting_message(isolated_root, surface="tui")
        row = WakeSchedule(
            id=onboarding.GREETING_WAKE_ID, message=trigger, next_due_at=1, hidden=True
        )
        await session._deliver_wake(DueWake(schedule=row, occurrence=1, final=True))
        await _wait_for_reply(session)

        # The model read the trigger.
        assert stream.requests, "her greeting turn never reached the provider"
        assert trigger in str(stream.requests[0])
        # No receipt card was emitted for the live surfaces.
        assert not [e for e in events if isinstance(e, WakeDeliveredEvent)]
        # The ledger moved at the fire.
        assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_DELIVERED
        assert onboarding.greeted_at(isolated_root) is not None

        # Desktop history window: her reply is the first (and only) row.
        transcript = session._transcript
        page = display_window(
            transcript,
            conversation_id=session_id,
            owner_epoch="epoch",
            through_id=transcript.entries()[-1].id,
        )
        visible = [m for m in page.messages if trigger in str(getattr(m, "details", "") or m)]
        assert not visible, page.messages
        roles = [getattr(m, "role", getattr(m, "custom_type", "")) for m in page.messages]
        assert roles and roles[0] == "assistant", roles

        # Phone fold: no wake notice, no user row; her text is first.
        entries = fold_messages_to_entries(session.history())
        kinds = [e.kind for e in entries]
        assert "notice" not in kinds and "user" not in kinds, kinds
        assert entries and entries[0].kind == "assistant"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_unattended_fire_withholds_the_greeting_and_keeps_the_request(
    isolated_root: Path,
    headless_surface: None,
) -> None:
    """A requested greeting must not land in a runtime with nobody attached.

    Review round 1, R-3: the state machine gated the REQUEST, so a person who
    asked for the greeting and quit before its due time left a due row that the
    next headless runtime (a wake-supervisor engagement, ``lop exec``, the phone
    daemon) would take — delivering into a turn nobody can read and stamping
    ``delivered``, which spends the greeting and starts the cadence it gates.

    What is pinned: the fire is WITHHELD (no turn, no provider request, no
    transcript row, no receipt), the ledger never reaches ``delivered``, and it
    returns to ``requested`` so the next attended moment's reconcile arms it
    again — then that attended fire does land.
    """
    from local_operator.aida import onboarding
    from local_operator.harness.types import WakeDeliveredEvent

    session_id = "ffff88880000"
    state.update_state(isolated_root, session_id=session_id)
    onboarding.request_greeting(isolated_root, "tui")
    onboarding.mark_greeted(isolated_root, 1)
    session = make_session(isolated_root, session_id)
    stream = cast(Any, session._stream_fn)
    events: list[object] = []
    session.subscribe(events.append)
    try:
        row = WakeSchedule(
            id=onboarding.GREETING_WAKE_ID,
            message=onboarding.greeting_message(isolated_root, surface="tui"),
            next_due_at=1,
            hidden=True,
        )
        await session._deliver_wake(DueWake(schedule=row, occurrence=1, final=True))
        for _ in range(40):
            if session.history():
                break
            await asyncio.sleep(0.01)

        # Nothing reached the model, the transcript or a front end.
        assert not stream.requests, "a withheld greeting must not run a turn"
        assert not session.history(), session.history()
        assert not [e for e in events if isinstance(e, WakeDeliveredEvent)]
        # And the ledger still owes the person who asked for it: not delivered,
        # not spent, ready for the next attended moment to arm.
        assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_REQUESTED
        assert onboarding.greeted_at(isolated_root) is None
        # ``requested`` is the only state ``proactive.reconcile``/resume arm from
        # (``greeting_armable`` = requested AND a resolvable provider); the
        # provider half is not modelled here — this root has no provider config,
        # so the arm gate's other term is covered by the reconcile tests.
        assert onboarding.greeting_record(isolated_root)["state"] == onboarding.GREETING_REQUESTED

        # The next ATTENDED fire does land — the withholding cost no greeting.
        session_attended = make_session(isolated_root, session_id)
        attended_stream = cast(Any, session_attended._stream_fn)
        try:
            from local_operator.aida import activation

            original = activation.human_surface_present
            activation.human_surface_present = lambda: True
            try:
                await session_attended._deliver_wake(
                    DueWake(schedule=row, occurrence=2, final=True)
                )
                await _wait_for_reply(session_attended)
            finally:
                activation.human_surface_present = original
            assert attended_stream.requests, "the attended fire must reach the provider"
            assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_DELIVERED
        finally:
            await session_attended.dispose()
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_runtime_probe_decides_and_a_detached_tui_is_attended(
    isolated_root: Path,
    headless_surface: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-3, second half: the gate reads the RUNTIME's connection table.

    The first fire-time gate asked ``activation.human_surface_present`` — "does
    THIS PROCESS have a tty or the desktop token". A TUI's session runs in a
    detached runtime child spawned with ``stdin=DEVNULL``, so that answered
    "nobody" for the most common surface the greeting is requested from, and
    the greeting would have been withheld forever in exactly the case it exists
    for. The ``headless_surface`` fixture reproduces that process-level answer
    here; the runtime probe (what ``serving`` installs from
    ``RuntimeServer.attended_surfaces``) must override it both ways.
    """
    from local_operator.aida import onboarding

    session_id = "ffff88881111"
    state.update_state(isolated_root, session_id=session_id)
    session = make_session(isolated_root, session_id)
    try:
        # No probe installed: the in-process fallback, which this fixture says
        # is headless.
        assert session._aida_greeting_may_land() is False
        # A runtime with a LOCAL attach (the TUI) installed its probe: attended,
        # even though this process has no tty — the case the first gate lost.
        session._aida_attended_probe = lambda: True
        assert session._aida_greeting_may_land() is True
        # A runtime with nobody local (exec, supervisor, phone relay): withheld.
        session._aida_attended_probe = lambda: False
        assert session._aida_greeting_may_land() is False

        # The doorbell: a request the withhold put back is re-armed the moment a
        # local surface arrives, and nothing happens when nothing is owed.
        calls: list[str] = []

        async def _fake_reconcile() -> None:
            calls.append("reconcile")

        session._aida_duty = True
        session._aida_reconcile_now = _fake_reconcile  # type: ignore[method-assign]
        session.aida_attended()  # owed: nothing to re-arm
        await asyncio.sleep(0.01)
        assert calls == []
        # ``fresh_install`` needs a provider; this root has no provider config,
        # and the request's precondition is not what this test is about.
        monkeypatch.setattr(onboarding, "provider_configured", lambda _root: True)
        onboarding.request_greeting(isolated_root, "tui")
        assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_REQUESTED
        session.aida_attended()
        for _ in range(50):
            if calls:
                break
            await asyncio.sleep(0.01)
        assert calls == ["reconcile"]
    finally:
        await session.dispose()


def test_reconcile_does_not_arm_a_requested_greeting_for_nobody(
    isolated_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without this the withhold is a loop: re-armed due-now, withheld, again.

    ``Session._persist_wake_schedules`` reconciles on every persist, so a
    headless runtime that withheld the greeting would arm it again on its next
    write and fire it straight into the same withhold. The arm takes the same
    attendance answer as the fire.
    """
    from local_operator.aida import onboarding, proactive

    session_id = "ffff88882222"
    state.update_state(isolated_root, session_id=session_id)
    monkeypatch.setattr(onboarding, "provider_configured", lambda _root: True)
    onboarding.request_greeting(isolated_root, "tui")
    assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_REQUESTED

    unattended = proactive.reconcile(
        [], config_dir=isolated_root, session_id=session_id, now_ms=1_000, attended=False
    )
    assert not [r for r in unattended.schedules if r.id == onboarding.GREETING_WAKE_ID]
    assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_REQUESTED

    attended = proactive.reconcile(
        [], config_dir=isolated_root, session_id=session_id, now_ms=2_000, attended=True
    )
    rows = [r for r in attended.schedules if r.id == onboarding.GREETING_WAKE_ID]
    assert len(rows) == 1 and rows[0].hidden is True
    assert onboarding.greeting_state(isolated_root) == onboarding.GREETING_ARMED
