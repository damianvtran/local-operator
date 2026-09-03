"""The cold viewer: a session you are looking at but not running.

``lop`` boots into this state. There is no runtime, deliberately none is
started, and the first message is what brings one into existence. These tests
pin the three properties that makes safe:

1. Opening a session creates NOTHING — no process, no directory, no lease.
2. A cold viewer still renders: durable history, the configured model, and any
   scheduled wakes come from disk rather than from an owner.
3. The first mutating call engages a runtime and attaches to it, once, even
   when several arrive together.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from local_operator.session.remote import RemoteSession
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from tests.unit.session.runtime.test_server import FakeHandle

SESSION_ID = "coldviewer01"


async def _never():
    raise AssertionError("takeover was not expected")


def _seed_transcript(config_dir: Path, session_id: str) -> Path:
    directory = config_dir / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    return directory


@pytest.mark.asyncio
async def test_a_cold_viewer_creates_no_process_and_no_directory(
    tmp_path: Path, monkeypatch
) -> None:
    """Opening a terminal is not work, and must not cost a session directory."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)

    viewer = await RemoteSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        assert viewer.is_cold is True
        assert viewer.session_id == SESSION_ID
        # Nothing on disk, and nothing published.
        assert list((tmp_path / "sessions").iterdir()) == []
        assert registry.scan(tmp_path) == []
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_cold_viewer_renders_durable_history_without_an_owner(
    tmp_path: Path, monkeypatch
) -> None:
    """`--resume` of a session nobody is running still shows the conversation."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)

    from local_operator.harness.types import Message
    from local_operator.session.transcript import Transcript

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("what did we decide?"))
    await transcript.append_message(Message.assistant("we decided to ship it"))

    viewer = await RemoteSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        texts = [getattr(message, "text", "") for message in viewer.history()]
        assert "what did we decide?" in texts
        assert "we decided to ship it" in texts
        # Canonical state exists and names this session, so every widget reads
        # a cold session through the same path it reads an attached one.
        assert viewer.frontend_state.session_id == SESSION_ID
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_cold_viewer_shows_scheduled_wakes_from_the_index(
    tmp_path: Path, monkeypatch
) -> None:
    """A cold session's wakes are real and the picker/panel must see them."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_transcript(tmp_path, SESSION_ID)

    from local_operator.wakes.store import write_entry

    write_entry(
        tmp_path,
        SESSION_ID,
        cwd=str(tmp_path),
        schedules=[
            {
                "id": "wake-1",
                "message": "check the deploy",
                "next_due_at": 4_102_444_800_000,
                "created_at": 1,
            }
        ],
    )

    viewer = await RemoteSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        wakes = viewer.frontend_state.wakes
        assert [wake.id for wake in wakes] == ["wake-1"]
        assert wakes[0].message == "check the deploy"
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_the_first_prompt_binds_the_viewer_to_a_runtime(tmp_path: Path, monkeypatch) -> None:
    """The cold-to-attached seam, against a REAL server over a real socket.

    ``engage_runtime`` is stubbed to start an in-process ``RuntimeServer``
    instead of spawning a python: the seam under test is the viewer's, and a
    real subprocess would bring a provider and ~1.2 s of construction with it.
    Everything after the engage — the record scan, the dial, the canonical
    sync, the history boundary — is production code.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_transcript(tmp_path, "s1")

    handle = FakeHandle()
    server = RuntimeServer(handle, kind="tui")
    engagements = 0

    async def fake_engage(session_id, cwd, work, *, config_dir, deadline_s=30.0):  # noqa: ANN001
        nonlocal engagements
        engagements += 1
        server.start()
        # A real runtime claims the transcript, and ``find_owner_record``
        # consults that liveness marker before trusting any record. Writing it
        # is part of standing in for the process, not test scaffolding.
        marker = config_dir / "sessions" / session_id / ".session.pid"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(os.getpid()), encoding="utf-8")
        for _ in range(200):
            rows = registry.scan(config_dir)
            if rows and rows[0][1] == "live":
                return None
            await asyncio.sleep(0.02)
        raise AssertionError("the fake runtime never published")

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", fake_engage)

    viewer = await RemoteSession.cold(
        "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        assert viewer.is_cold is True

        await viewer.prompt("start working")

        assert viewer.is_cold is False, "the first prompt must bind the viewer"
        assert engagements == 1
        assert handle.calls[-1][0] == "prompt"
    finally:
        await viewer.dispose()
        server.close()


@pytest.mark.asyncio
async def test_concurrent_first_writes_engage_exactly_one_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    """A prompt racing the speculative warm engage must not start two runtimes.

    The composer fires a warm engage on the first keystroke and the user can
    submit before it lands, so this race is the NORMAL case rather than an
    exotic one. ``_ensure_bound``'s lock is what makes them share one runtime.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_transcript(tmp_path, "s1")

    handle = FakeHandle()
    server = RuntimeServer(handle, kind="tui")
    engagements = 0

    async def fake_engage(session_id, cwd, work, *, config_dir, deadline_s=30.0):  # noqa: ANN001
        nonlocal engagements
        engagements += 1
        # A real engage takes time; without that delay the lock is never
        # actually contended and the test would prove nothing.
        await asyncio.sleep(0.2)
        server.start()
        # A real runtime claims the transcript, and ``find_owner_record``
        # consults that liveness marker before trusting any record. Writing it
        # is part of standing in for the process, not test scaffolding.
        marker = config_dir / "sessions" / session_id / ".session.pid"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(os.getpid()), encoding="utf-8")
        for _ in range(200):
            rows = registry.scan(config_dir)
            if rows and rows[0][1] == "live":
                return None
            await asyncio.sleep(0.02)
        raise AssertionError("the fake runtime never published")

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", fake_engage)

    viewer = await RemoteSession.cold(
        "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        await asyncio.gather(
            viewer.prompt("first"),
            viewer.prompt("second"),
            viewer.prompt("third"),
        )
        assert engagements == 1, "concurrent first writes engaged more than one runtime"
        prompts = [call for call in handle.calls if call[0] == "prompt"]
        assert len(prompts) == 3, "every prompt must still reach the one runtime"
    finally:
        await viewer.dispose()
        server.close()


@pytest.mark.asyncio
async def test_a_draft_warms_the_runtime_before_the_message_is_sent(
    tmp_path: Path, monkeypatch
) -> None:
    """The speculative engage, driven through the REAL composer.

    The seam is ``Editor.edit`` — the documented funnel every buffer mutation
    passes through — and NOT a key handler on the App. An earlier attempt
    overrode ``App._on_key``, which sits in Textual's dispatch path: it broke
    key handling in 190 tests across settings, todo, analytics and selection,
    because intercepting there stops the widgets that bind their own keys from
    ever seeing them.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_transcript(tmp_path, "s1")

    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.editor import Editor

    engaged = asyncio.Event()

    async def fake_engage(session_id, cwd, work, *, config_dir, deadline_s=30.0):  # noqa: ANN001
        engaged.set()
        raise ConnectionError("no runtime in this test")

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", fake_engage)

    viewer = await RemoteSession.cold(
        "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )

    async def factory():
        return viewer

    app = OperatorApp(factory)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            # Let the boot worker adopt the viewer: the app only holds it after
            # the factory it awaits has returned.
            for _ in range(30):
                await pilot.pause()
            assert app._session is viewer, "the app never adopted the cold viewer"
            assert app._warm_engage_started is False, "an idle viewer must not engage"

            editor = app.query_one(Editor)
            editor.focus()
            await pilot.press("h")
            # Waited on the ENGAGE, not on the flag. The draft signal is a
            # posted message so it lands a tick later, and the flag it sets is
            # deliberately self-clearing: a warm-up that fails must leave the
            # next real message free to try again, so asserting on the flag
            # would be asserting on a value the failure path correctly resets.
            for _ in range(100):
                await pilot.pause()
                if engaged.is_set():
                    break
            assert engaged.is_set(), "the first keystroke never engaged a runtime"

            # And the failure is SILENT: a speculative warm-up the user did not
            # ask for must not paint an error over their draft.
            assert editor.text == "h", "the warm-up disturbed the draft"
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_model_default_persists_for_a_local_runtime(tmp_path: Path, monkeypatch) -> None:
    """`/model … d` must keep working once EVERY session is remote.

    The refusal was keyed on ``is_remote``, which meant "somebody else's
    session" before the viewer model and means "any session at all" after it.
    Left as it was, a user on their own machine \u2014 whose runtime is a child
    process on that same machine \u2014 was told to run the command "on the
    terminal whose launches it should govern", which was the terminal they
    were already sitting at.

    The question the refusal always meant is narrower: would this machine's
    config write reach the runtime? A runtime that published a record HERE is
    local, whatever transport talks to it.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_transcript(tmp_path, "s1")

    from local_operator.tui.app import OperatorApp

    viewer = await RemoteSession.cold(
        "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )

    async def factory():
        return viewer

    app = OperatorApp(factory)

    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(30):
            await pilot.pause()

        # No record published: the runtime is not reachable from this config,
        # so persisting here would write the wrong machine's config.
        assert app._session_runs_elsewhere() is True

        # A record for this session id is what "local" MEANS.
        from local_operator.session.runtime import registry
        from local_operator.session.runtime.types import SessionRecord

        registry.publish(
            SessionRecord(
                pid=os.getpid(),
                kind="daemon",
                session_id="s1",
                conversation_name="local one",
                cwd=str(tmp_path),
                model_label="m",
                control_port=1,
                control_key="k" * 16,
            ),
            tmp_path,
        )

        assert app._session_runs_elsewhere() is False, (
            "a runtime that published its record on THIS machine is local, "
            "so /model default must persist rather than refuse"
        )
