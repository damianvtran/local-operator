"""The cold viewer: a session you are looking at but not running.

``lop`` boots into this state, and the TUI engages a runtime from it as soon
as the session is adopted (``OperatorApp._engage_runtime_eagerly``) so the
band is complete before anything is typed. The viewer FACADE itself still
starts nothing: these tests pin the three properties that keep the cold state
safe for every caller that is not the TUI, and for the TUI between mount and
the engage landing:

1. Constructing a cold viewer creates NOTHING — no process, no directory, no
   lease. Only an engage does, and an unused one is handed back
   (``tests/unit/session/runtime/test_eager_runtime.py``).
2. A cold viewer still renders: durable history, the configured model, and any
   scheduled wakes come from disk rather than from an owner.
3. A mutating call engages a runtime and attaches to it, once, even when
   several arrive together — and again after a failed engage.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from local_operator.session.attached import AttachedSession
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from tests.unit.session.runtime.test_server import FakeHandle

#: Upper bound on an awaited event, never a budget to sleep through.
DEADLOCK_GUARD_S = 30.0

SESSION_ID = "coldviewer01"


async def _never():
    raise AssertionError("takeover was not expected")


def _seed_transcript(config_dir: Path, session_id: str) -> Path:
    directory = config_dir / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    return directory


def _configure_provider(config_dir: Path) -> None:
    """Name a provider and model, the way a real machine's config does.

    The TUI's engage is gated on the viewer having a resolvable model (a
    first-run screen with an empty config must not spawn a runtime that
    exits rc=2), so any test that expects the app to ENGAGE has to look like
    a configured machine, not an empty temp dir.
    """
    from local_operator.config import ConfigManager

    ConfigManager(config_dir=config_dir).update_config(
        {"hosting": "anthropic", "model_name": "claude-opus-5"}
    )


@pytest.mark.asyncio
async def test_a_cold_viewer_creates_no_process_and_no_directory(
    tmp_path: Path, monkeypatch
) -> None:
    """Opening a terminal is not work, and must not cost a session directory."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)

    viewer = await AttachedSession.cold(
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

    viewer = await AttachedSession.cold(
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
async def test_a_cold_viewer_rehydrates_a_pasted_image_from_the_shared_store(
    tmp_path: Path, monkeypatch
) -> None:
    """An externalized image must replay with its BYTES, not an empty block.

    The cold replay used to resolve attachment digests against a per-session
    store (``<config>/sessions/<id>``) that nothing ever writes to, so a user
    screenshot that is on disk and intact replayed as the "image unavailable —
    no longer in the transcript" receipt. The journal's own config dir is where
    the write path externalizes to, and that is the only root a replay can
    resolve against.

    Asserted through ``AttachedSession.cold`` — the real cold seam that boots a
    viewer over a seeded journal — rather than ``replay_entries`` directly.

    The reader's ENVIRONMENT deliberately differs from the owning config dir:
    ``LOCAL_OPERATOR_CONFIG_DIR`` points at a second, empty directory while the
    session is read with ``config_dir=<owner>``. A regression that resolved the
    replay against ``AttachmentStore()`` (the env default) would therefore
    resolve nothing and fail here, which is the wiring this pins — the bug's own
    shape was a root that looked plausible and was not the writer's.
    """
    owner_cfg = tmp_path / "owner"
    reader_env = tmp_path / "reader-env"
    directory = _seed_transcript(owner_cfg, SESSION_ID)

    import base64
    import json

    from local_operator.harness.types import ImageContent, Message
    from local_operator.session.attachments import (
        AttachmentStore,
        store_for_transcript_dir,
    )
    from local_operator.session.transcript import Transcript

    # Write under the OWNING config dir, so the bytes land in
    # <owner>/attachments — the store derived from the session directory.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(owner_cfg))

    # Above the 1 KB externalization floor, so the row references the store
    # rather than carrying the bytes inline — an inline row resolves with no
    # store at all and would pass on the buggy tree.
    image = ImageContent(
        data=base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4096).decode("ascii"),
        mime_type="image/png",
    )
    transcript = Transcript(directory)
    await transcript.append_message(Message.user("look at this [Image #1]", images=[image]))

    rows = [
        json.loads(line)
        for line in (directory / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert any(
        block.get("attachment")
        for row in rows
        for block in (row.get("payload", {}).get("content") or [])
        if isinstance(block, dict)
    ), "precondition: the image row must be externalized to the store"

    # Now hand the reader a DIFFERENT env config dir than the one that owns the
    # journal. The roots must disagree, or the test cannot see a rewire.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(reader_env))
    assert store_for_transcript_dir(directory).root == owner_cfg / "attachments"
    assert AttachmentStore().root == reader_env / "attachments"
    assert store_for_transcript_dir(directory).root != AttachmentStore().root

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=owner_cfg, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        blocks = [
            block
            for message in viewer.history()
            for block in (getattr(message, "content", None) or [])
            if isinstance(block, ImageContent)
        ]
        assert len(blocks) == 1
        # NON-EMPTY and byte-identical to the paste: a block that merely
        # EXISTS is exactly what the bug produced.
        assert blocks[0].data == image.data
        assert len(blocks[0].data) > 0
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_cold_viewer_degrades_when_the_store_sidecar_is_not_an_object(
    tmp_path: Path, monkeypatch
) -> None:
    """A damaged sidecar must degrade into the viewer, never raise.

    ``AttachmentStore.get`` promises "degrade to a placeholder, never raise" —
    and so does ``transcript._resolve_attachments``, which is the only caller on
    this path. A sidecar that parses as JSON but is not an object (``[]``,
    ``null``, a bare number) has no ``mime_type`` key, so an unguarded
    ``meta.get`` raised AttributeError straight through the cold replay. It
    needs a damaged or hand-edited store, which is why this is the guard rather
    than a live defect: the property under test is that the replay survives the
    same class of damage as a truncated sidecar already did.
    """
    import base64

    from local_operator.harness.types import ImageContent, Message
    from local_operator.session.transcript import Transcript

    owner_cfg = tmp_path / "owner"
    directory = _seed_transcript(owner_cfg, SESSION_ID)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(owner_cfg))

    image = ImageContent(
        data=base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4096).decode("ascii"),
        mime_type="image/png",
    )
    await Transcript(directory).append_message(
        Message.user("look at this [Image #1]", images=[image])
    )

    sidecars = list((owner_cfg / "attachments").glob("*.json"))
    assert len(sidecars) == 1, "precondition: the write path stored one sidecar"
    sidecars[0].write_text("[]", encoding="utf-8")

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=owner_cfg, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        history = viewer.history()
        blocks = [
            block
            for message in history
            for block in (getattr(message, "content", None) or [])
            if isinstance(block, ImageContent)
        ]
        assert len(blocks) == 1
        # The honest receipt, not a raise and not a blanked transcript.
        assert blocks[0].data == ""
        assert any("look at this [Image #1]" in getattr(m, "text", "") for m in history)
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

    viewer = await AttachedSession.cold(
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

    async def fake_engage(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        nonlocal engagements
        engagements += 1
        server.start()
        # A real runtime claims the transcript, and ``find_runtime_record``
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

    viewer = await AttachedSession.cold(
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

    async def fake_engage(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        nonlocal engagements
        engagements += 1
        # A real engage takes time; without that delay the lock is never
        # actually contended and the test would prove nothing.
        await asyncio.sleep(0.2)
        server.start()
        # A real runtime claims the transcript, and ``find_runtime_record``
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

    viewer = await AttachedSession.cold(
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
    """The draft engage RETRIES after a failed mount engage.

    The seam is ``Editor.edit`` — the documented funnel every buffer mutation
    passes through — and NOT a key handler on the App. An earlier attempt
    overrode ``App._on_key``, which sits in Textual's dispatch path: it broke
    key handling in 190 tests across settings, todo, analytics and selection,
    because intercepting there stops the widgets that bind their own keys from
    ever seeing them.

    The app now also engages at MOUNT (see the eager-start test below), so this
    no longer asserts that an idle viewer stays cold — it asserts the property
    the keystroke path still owns: ``engage_runtime`` here always raises, which
    clears the latch, and the first keystroke must then try AGAIN rather than
    leaving the viewer stranded on a failed start.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_transcript(tmp_path, "s1")
    _configure_provider(tmp_path)

    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.editor import Editor

    engaged = asyncio.Event()
    settled = asyncio.Event()

    async def fake_engage(
        session_id, cwd, work, *, config_dir, deadline_s=30.0, preempt=None, preempt_budget_s=0.0
    ):  # noqa: ANN001
        engaged.set()
        raise ConnectionError("no runtime in this test")

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", fake_engage)

    viewer = await AttachedSession.cold(
        "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )

    async def factory():
        return viewer

    app = OperatorApp(factory)
    # `_set_starting(False)` is the worker's `finally`; it is the observable
    # edge for "the failed engage has fully unwound", which is when the latch
    # is guaranteed clear again.
    real_set_starting = app._set_starting

    def observed_set_starting(starting: bool) -> None:
        real_set_starting(starting)
        if not starting:
            settled.set()

    monkeypatch.setattr(app, "_set_starting", observed_set_starting)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            # Let the boot worker adopt the viewer: the app only holds it after
            # the factory it awaits has returned.
            for _ in range(30):
                await pilot.pause()
            assert app._session is viewer, "the app never adopted the cold viewer"
            # The MOUNT engage already ran and failed (the stub always raises),
            # which is what leaves the latch clear for the keystroke below to
            # retry. Waiting for that failure to land first is what makes the
            # retry assertion mean something rather than passing on the mount
            # attempt's own event.
            await asyncio.wait_for(engaged.wait(), timeout=DEADLOCK_GUARD_S)
            # The latch is cleared by the worker's `except`, which runs after
            # the stub raised — a later loop turn than the event. Wait for
            # THAT edge too, through the band state the same `finally` clears.
            await asyncio.wait_for(settled.wait(), timeout=DEADLOCK_GUARD_S)
            await pilot.pause()
            assert app._warm_engage_started is False, "a failed engage must clear the latch"
            engaged.clear()
            settled.clear()

            editor = app.query_one(Editor)
            editor.focus()
            await pilot.press("h")
            # Waited on the ENGAGE, not on the flag. The draft signal is a
            # posted message so it lands a tick later, and the flag it sets is
            # deliberately self-clearing: a warm-up that fails must leave the
            # next real message free to try again, so asserting on the flag
            # would be asserting on a value the failure path correctly resets.
            await asyncio.wait_for(engaged.wait(), timeout=DEADLOCK_GUARD_S)
            await pilot.pause()

            # And the failure is SILENT: a speculative warm-up the user did not
            # ask for must not paint an error over their draft.
            assert editor.text == "h", "the warm-up disturbed the draft"
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_model_default_persists_for_a_local_runtime(tmp_path: Path, monkeypatch) -> None:
    """`/model … d` must keep working once EVERY session is remote.

    The refusal was keyed on a transport flag, which meant "somebody else's
    session" before the viewer model and "any session at all" after it.
    Left as it was, a user on their own machine \u2014 whose runtime is a child
    process on that same machine \u2014 was told to run the command "on the
    terminal whose launches it should govern", which was the terminal they
    were already sitting at.

    The question the refusal always meant is narrower: would this machine's
    config write reach the runtime? A runtime that published a record HERE is
    local, whatever transport talks to it.

    The COLD half of this changed in #624 (review round 1, R1): this test
    used to assert that a cold viewer with no record is "elsewhere", which is
    the state every fresh `lop` boots into — so `/model default` and the
    picker's `d` refused the local user before their first message. A cold
    viewer has no runtime at all, so there is nothing anywhere for the wrong
    config to govern; it is known-local, not unknown. The record half below
    is unchanged.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_transcript(tmp_path, "s1")

    from local_operator.tui.app import OperatorApp

    viewer = await AttachedSession.cold(
        "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )

    async def factory():
        return viewer

    app = OperatorApp(factory)

    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(30):
            await pilot.pause()

        # No record published AND no runtime: the cold viewer `lop` boots
        # into. The next runtime this terminal starts is exactly what the
        # write is for (#624 R1) — not "elsewhere".
        assert viewer.is_cold is True
        assert app._session_runs_elsewhere() is False

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


@pytest.mark.asyncio
async def test_a_viewer_defers_naming_to_the_runtime(tmp_path: Path, monkeypatch) -> None:
    """Naming belongs to the process that owns the provider.

    `ServingSessionHandle.prompt` calls its own `_maybe_name_conversation`, so
    the title is generated beside the transcript that stores it. A viewer must
    not race that — and cannot: `AttachedSession.complete_once` raises by
    construction ("provider errands run on the session owner"), so leaving the
    viewer's naming worker enabled started a worker on every first message
    that could only ever fail.

    The PROVISIONAL name still shows, because that is local and needs no
    provider call: it is what stops the tab reading `lo › <cwd>` for the whole
    length of the opening turn.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_transcript(tmp_path, "s1")

    from local_operator.tui.app import OperatorApp

    viewer = await AttachedSession.cold(
        "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )

    async def factory():
        return viewer

    app = OperatorApp(factory)
    started: list[str] = []

    async def fail_if_called(text, complete):  # noqa: ANN001
        started.append(text)
        raise AssertionError("a viewer must not run a provider naming errand")

    monkeypatch.setattr("local_operator.session.naming.generate_title", fail_if_called)

    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(30):
            await pilot.pause()

        app._maybe_name_conversation("refactor the YAML loader please")
        for _ in range(10):
            await pilot.pause()

    assert started == [], "the viewer started a naming errand it cannot complete"


@pytest.mark.asyncio
async def test_a_cold_viewer_restores_the_roster_and_todos_from_disk(
    tmp_path: Path, monkeypatch
) -> None:
    """The details are on the FIRST frame, not after a runtime starts.

    The regression this guards shipped in the release that made the TUI a
    viewer: the old in-process TUI restored the subagent roster and the todo
    list at boot (``Session.__init__`` → ``_load_subagent_roster`` /
    ``_load_todo_snapshot``), and the viewer path synthesised canonical state
    from config and the wake index only. So a resumed session opened with an
    empty subagent panel and no todos and stayed that way indefinitely — the
    details were not slow to arrive, they were never going to arrive.

    Asserted on the state the widgets read, and asserted BEFORE anything binds
    a runtime, because "after the first message" is exactly the behaviour that
    was wrong.

    The roster SIDECAR is written too, because a restored row's identity comes
    from it: ``job1``'s record is what makes the row addressable (its
    ``session_id`` is the child's own, and the run sidebar's child reader opens
    a child by it), which is QA round 1's Q1-1 on the surface the renderer
    reads.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)

    from local_operator.harness.types import Message
    from local_operator.session.frontend_state import (
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        FrontendModelSpec,
        FrontendSessionState,
        JobState,
        TodoItemState,
        TodoPhaseState,
    )
    from local_operator.session.transcript import Transcript

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("run the audit"))
    durable = FrontendSessionState(
        session_id=SESSION_ID,
        epoch="previous-owner",
        conversation_title="Article search rollout",
        goal="ship the rollout",
        jobs=[JobState(id="job1", type="task", label="auditor", status="succeeded")],
        todos=[
            TodoPhaseState(
                name="Verification", items=[TodoItemState(text="run the gate", status="pending")]
            )
        ],
        cumulative_parent_cost=12.5,
        context_tokens=322_546,
        context_window=1_000_000,
        # The context reading is only restored when the checkpoint's model
        # matches the one configured now — a token count measured against one
        # window means nothing divided by another (design round 1, D1). The
        # config this test writes names no model, so state the identity that
        # makes the reading interpretable.
        selected_model=FrontendModelSpec(provider="", model_id=""),
    )
    await transcript.append_custom(
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        {"checkpoint_id": "c1", "state": durable.model_dump(mode="json")},
    )
    child_id = "abcdef012345"
    child_dir = tmp_path / "sessions" / child_id
    child_dir.mkdir(parents=True, exist_ok=True)
    from local_operator.resume import ORIGIN_SUBAGENT, mark_session_origin
    from local_operator.session.session import (
        SUBAGENT_ROSTER_SIDECAR,
        _write_roster_sidecar,
    )

    mark_session_origin(child_dir, ORIGIN_SUBAGENT, label="auditor")
    _write_roster_sidecar(
        directory / SUBAGENT_ROSTER_SIDECAR,
        {
            "version": 1,
            "generation": 1,
            "jobs": [],
            "accounting": [],
            "records": [
                {
                    "job_id": "job1",
                    "label": "auditor",
                    "session_dir": str(child_dir),
                    "outcome": "completed",
                }
            ],
        },
    )

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        assert viewer.is_cold is True, "restoring details must not start a runtime"
        state = viewer.frontend_state
        assert [job.id for job in state.jobs] == ["job1"]
        # A restored row has no in-process trajectory; the panel needs to know
        # that rather than rendering the child as an empty live job.
        assert state.jobs[0].restored is True
        # And it addresses its child exactly as a LIVE row does: the comms node
        # the runtime would have stamped these from is gone, but the roster
        # record it was resolved from carries the directory.
        assert state.jobs[0].session_id == child_id
        assert state.jobs[0].session_dir == str(child_dir)
        assert [item.text for phase in state.todos for item in phase.items] == ["run the gate"]
        assert state.conversation_title == "Article search rollout"
        assert state.goal == "ship the rollout"
        # Spend and occupancy are the conversation's, not this process's: a
        # resumed session that already cost money must not open reading zero.
        assert state.cumulative_parent_cost == 12.5
        assert state.context_tokens == 322_546
        # The model still comes from THIS process's config, not the stale copy
        # in the checkpoint — the config may have changed since.
        assert state.cwd == str(tmp_path)
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_cold_viewer_opens_when_the_checkpoint_is_unreadable(
    tmp_path: Path, monkeypatch
) -> None:
    """A bad status row must never stop a conversation from opening."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)

    from local_operator.harness.types import Message
    from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE
    from local_operator.session.transcript import Transcript

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("still here?"))
    await transcript.append_custom(
        FRONTEND_CHECKPOINT_CUSTOM_TYPE, {"checkpoint_id": "c1", "state": {"jobs": "not-a-list"}}
    )

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        assert viewer.frontend_state.session_id == SESSION_ID
        assert list(viewer.frontend_state.jobs) == []
        assert [getattr(m, "text", "") for m in viewer.history()] == ["still here?"]
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_cold_viewer_prefers_the_live_wake_index_over_the_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    """MCP servers come from the checkpoint; wakes deliberately do not.

    Both are chrome the pre-0.46.0 path showed and the viewer dropped (review
    round 1, C5) — but they have different sources of truth and restoring them
    the same way would be wrong. There is no live MCP manager to ask while
    cold, so the durable copy is the only thing that can populate that panel.
    The wake INDEX, by contrast, is a derived file a supervisor rewrites
    without opening the session, so it is fresher than any checkpoint:
    overwriting it with the durable copy would re-show a wake that already
    fired.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)

    from local_operator.harness.types import Message
    from local_operator.session.frontend_state import (
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        FrontendSessionState,
        McpServerState,
        WakeState,
    )
    from local_operator.session.transcript import Transcript
    from local_operator.wakes.store import write_entry

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("hello"))

    # The index holds the CURRENT truth: one wake, still scheduled.
    write_entry(
        tmp_path,
        SESSION_ID,
        cwd=str(tmp_path),
        schedules=[
            {
                "id": "w-live",
                "message": "the wake that is still pending",
                "next_due_at": 4_102_444_800_000,
                "created_at": 1,
            }
        ],
    )

    durable = FrontendSessionState(
        session_id=SESSION_ID,
        epoch="previous-owner",
        mcp_servers=[McpServerState(name="linear", status="connected")],
        # A stale wake the supervisor has since fired and removed from the index.
        wakes=[WakeState(id="w-stale", message="already fired", next_due_at=1)],
    )
    await transcript.append_custom(
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        {"checkpoint_id": "c1", "state": durable.model_dump(mode="json")},
    )

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        state = viewer.frontend_state
        assert [server.name for server in state.mcp_servers] == ["linear"]
        assert [wake.id for wake in state.wakes] == ["w-live"], (
            "the live wake index must win over the checkpoint, or a fired wake "
            "reappears on every resume"
        )
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_restored_context_reading_keeps_the_window_it_was_measured_against(
    tmp_path: Path, monkeypatch
) -> None:
    """The cold frame must divide the restored tokens by a REAL window.

    The cold spec is built from ``config.yml``, which names a provider and a
    model but carries no metadata. Before the cold pair was resolved at all,
    ``ModelSpec``'s 128k DEFAULT window applied while the restored token count
    had been measured against the 1M window the runtime really had — and
    ``_context_window`` reads the effective spec, so a resumed session painted
    ``268.2%/128k`` (design round 1, D1) on the one surface whose job is to say
    how much room is left.

    The divisor is asserted, and NOT a ceiling on the percentage. ``32.3%`` here
    is a property of this fixture: the band paints over-budget readings on
    purpose (``context_spelling(900_000, 872_000)`` is ``103.2%``), because the
    percentage is what says the NEXT request overflows — which a stale
    denominator would hide (review round 2, minor 2).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)

    from local_operator.config import ConfigManager
    from local_operator.harness.types import Message
    from local_operator.session.frontend_state import (
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        FrontendModelSpec,
        FrontendSessionState,
    )
    from local_operator.session.transcript import Transcript

    config = ConfigManager(config_dir=tmp_path)
    config.update_config({"hosting": "anthropic", "model_name": "claude-opus-5"})

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("hello"))
    durable = FrontendSessionState(
        session_id=SESSION_ID,
        epoch="previous-owner",
        context_tokens=322_546,
        context_window=1_000_000,
        selected_model=FrontendModelSpec(
            provider="anthropic", model_id="claude-opus-5", context_window=1_000_000
        ),
    )
    await transcript.append_custom(
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        {"checkpoint_id": "c1", "state": durable.model_dump(mode="json")},
    )

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        state = viewer.frontend_state
        spec = state.effective_model or state.selected_model
        assert spec is not None
        window = int(spec.context_window or 0)
        assert window == 1_000_000, "the window the tokens were measured against must survive"
        assert state.context_tokens is not None
        assert state.context_window == 1_000_000, (
            "the restored reading's denominator is the checkpoint's where this process "
            "resolved no budget of its own (``anthropic`` here: the fresh spec states "
            "neither ``default_context_window`` nor ``max_context_window``)"
        )
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_restored_reading_is_dropped_when_the_model_changed(
    tmp_path: Path, monkeypatch
) -> None:
    """A count measured against another window is not convertible, so it goes.

    The other direction of D1: if the user switched models since the
    checkpoint, the stored numerator and the current denominator describe
    different things. There is no honest conversion, so the reading is dropped
    and the band renders an unknown context rather than a confident wrong
    percentage. The next real turn supplies a live one.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)

    from local_operator.config import ConfigManager
    from local_operator.harness.types import Message
    from local_operator.session.frontend_state import (
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        FrontendModelSpec,
        FrontendSessionState,
    )
    from local_operator.session.transcript import Transcript

    config = ConfigManager(config_dir=tmp_path)
    config.update_config({"hosting": "openai", "model_name": "gpt-5.6-sol"})

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("hello"))
    durable = FrontendSessionState(
        session_id=SESSION_ID,
        epoch="previous-owner",
        context_tokens=900_000,
        context_window=1_000_000,
        selected_model=FrontendModelSpec(
            provider="anthropic", model_id="claude-opus-5", context_window=1_000_000
        ),
    )
    await transcript.append_custom(
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        {"checkpoint_id": "c1", "state": durable.model_dump(mode="json")},
    )

    viewer = await AttachedSession.cold(
        SESSION_ID,
        config_dir=tmp_path,
        cwd=str(tmp_path),
        takeover_factory=_never,
        initial_model=FrontendModelSpec(provider="openai", model_id="gpt-5.6-sol"),
        model_selection_override=True,
    )
    try:
        state = viewer.frontend_state
        assert state.context_tokens is None, (
            "a reading measured against another model's window must be dropped, "
            "not divided by the new one"
        )
        spec = state.effective_model or state.selected_model
        assert spec is not None and spec.model_id == "gpt-5.6-sol"
    finally:
        await viewer.dispose()


def test_the_restore_adopts_a_local_window_with_the_tokens_measured_against_it() -> None:
    """Finding 1: the numerator and its denominator come from ONE source.

    A local endpoint's spec is METADATA-RICH by construction
    (``providers/local.py`` sets ``context_metadata_resolved``), so the window
    branch's old gate — "adopt the checkpoint's window unless the fresh spec has
    resolved metadata" — stopped firing for the whole local population the day the
    cold pair started being resolved. That is a numerator from the checkpoint under
    a denominator from the endpoint: 20,000 tokens from a 32,768 checkpoint read as
    ``488.2%`` of the endpoint's current 4,096 (review round 1, minor 1).

    The numerator and its denominator come from ONE source, and on this
    population that source is the checkpoint: the 4,096 is
    ``DEFAULT_LOCAL_CONTEXT``, the route default for a tag no listing row
    describes, so the fresh spec states no budget of its own and the checkpoint's
    window is the only real number in the process
    (``_fresh_spec_states_a_budget``, whose local carve-out is keyed on the
    provider KIND — ``LOCAL_PROVIDER_IDS``, the set ``build_model_spec`` routes to
    ``local_model_spec`` — because a route FILL and an answer leave the same
    fields behind and only the server's own evidence lands in
    ``default_context_window``/``max_context_window``). The account-scoped
    population is the one that refuses — see
    ``test_the_restore_refuses_a_stale_window_the_account_has_answered_for`` and
    ``test_the_restore_refuses_a_stale_window_on_an_api_key_after_an_oauth_run`` —
    and the assertions below check both readers rather than only the spec, which is
    the part that makes a paint order unable to mix them.

    NOT distinguished here, and not distinguishable from a cold frame: whether a
    4,096 the endpoint genuinely ANSWERED can be told from this route default.
    Both write the same fields (``providers/local.py``), so the discrimination
    belongs to the model layer (design round 2, D2 — deferred on the PR).
    """
    from local_operator.session.frontend_state import (
        FrontendModelSpec,
        FrontendSessionState,
    )

    # The local shape: the endpoint's answer for THIS model (the fallback working
    # budget when no listing row describes it), carried with the resolved flag.
    state = FrontendSessionState(
        session_id="coldlocal01",
        epoch="cold",
        selected_model=FrontendModelSpec(
            provider="ollama",
            model_id="qwen3:32b",
            context_window=4_096,
            context_metadata_resolved=True,
        ),
        effective_model=FrontendModelSpec(
            provider="ollama",
            model_id="qwen3:32b",
            context_window=4_096,
            context_metadata_resolved=True,
        ),
    )
    durable = FrontendSessionState(
        session_id="coldlocal01",
        epoch="previous-owner",
        context_tokens=20_000,
        context_window=32_768,
        selected_model=FrontendModelSpec(
            provider="ollama", model_id="qwen3:32b", context_window=32_768
        ),
    )

    context = AttachedSession._consistent_context(state, durable)
    specs = AttachedSession._restored_model_specs(state, durable)

    window = specs["selected_model"].context_window
    assert window == 32_768, (
        "the checkpoint's window is the one those tokens were measured against, so "
        "adopting its numerator means adopting its denominator too"
    )
    assert specs["effective_model"].context_window == window, (
        "the band reads the EFFECTIVE spec for its label and window, so both halves "
        "of the state must carry the same number"
    )
    assert (
        context["context_tokens"] == 20_000 and context["context_window"] == window
    ), "the state-level reading is the same pair, so no paint order can mix them"
    fresh = state.selected_model
    assert fresh is not None
    assert fresh.default_context_window is None and fresh.max_context_window is None, (
        "precondition: an uncovered local tag's route default is not a BUDGET, so this "
        "process resolved none and the checkpoint's window is the only real denominator "
        "in it. That is the half of the rule that ADOPTS; the account-scoped half "
        "refuses, and "
        "test_the_restore_refuses_a_stale_window_the_account_has_answered_for pins it "
        "(review round 2, blocker 1)."
    )
    from local_operator.tui.widgets.status_line import context_spelling

    assert context_spelling(20_000, window) == "61.0%/32.8k", (
        "the local cell's own reading, unchanged by the round-3 widening: the carve-out "
        "exists so this frame does not become the 488.2%/4k a route fill would paint"
    )


@pytest.mark.parametrize(
    "fresh_name",
    [
        # Naming refuses an id handed back as a name ...
        "zzz-9",
        # ... and a listing name that belongs to ANOTHER curated model, so a route
        # cannot borrow a direct provider's marketing string.
        "Claude Opus 5",
    ],
)
def test_a_name_naming_refuses_is_adopted_from_the_checkpoint(fresh_name: str) -> None:
    """Finding 2: ask naming's rule, do not re-derive one of its refusals.

    The adoption gate used to re-implement naming's ID-ECHO refusal alone, so a
    fresh name refused for one of the rule's other reasons — an ambiguity, a
    borrowed curated name — blocked the adoption forever: the first frame kept the
    bare id while the conversation's own checkpoint held a name naming would have
    accepted (review round 1, minor 2). The gate now asks ``model_label`` whether
    the render resolved a name at all, which is the same question the band answers
    when it paints.
    """
    from local_operator.model.naming import model_label
    from local_operator.session.frontend_state import (
        FrontendModelSpec,
        FrontendSessionState,
    )

    provider, model_id = "deepseek", "zzz-9"
    assert (
        model_label(provider, model_id, fresh_name).full == f"{provider}/{model_id}"
    ), "precondition: naming refuses this name, so the first frame paints the selector"
    state = FrontendSessionState(
        session_id="coldname01",
        epoch="cold",
        selected_model=FrontendModelSpec(
            provider=provider, model_id=model_id, display_name=fresh_name
        ),
    )
    durable = FrontendSessionState(
        session_id="coldname01",
        epoch="previous-owner",
        selected_model=FrontendModelSpec(
            provider=provider, model_id=model_id, display_name="DeepSeek Nine"
        ),
    )

    specs = AttachedSession._restored_model_specs(state, durable)

    assert specs["selected_model"].display_name == "DeepSeek Nine", (
        "the conversation's own record of its identity is the only name this "
        "process could not resolve itself"
    )
    assert model_label(provider, model_id, specs["selected_model"].display_name).full == (
        "DeepSeek Nine"
    ), "the adopted name is one the band will actually print"


def test_a_name_naming_resolves_is_never_replaced_by_the_checkpoint() -> None:
    """The adoption is a FILL, not an override: naming's own answer wins.

    The mirror of the test above. A fresh resolution that DID produce a name is a
    fact about this model read from the catalogue this process can see, and an old
    checkpoint's name must not displace it — the band would then caption the model
    with a string the current registry does not answer to.
    """
    from local_operator.session.frontend_state import (
        FrontendModelSpec,
        FrontendSessionState,
    )

    state = FrontendSessionState(
        session_id="coldname02",
        epoch="cold",
        selected_model=FrontendModelSpec(
            provider="deepseek", model_id="deepseek-flash", display_name="DeepSeek Flash"
        ),
    )
    durable = FrontendSessionState(
        session_id="coldname02",
        epoch="previous-owner",
        selected_model=FrontendModelSpec(
            provider="deepseek", model_id="deepseek-flash", display_name="DeepSeek V4 Flash"
        ),
    )

    specs = AttachedSession._restored_model_specs(state, durable)

    assert (
        specs["selected_model"].display_name == "DeepSeek Flash"
    ), "a resolved name is this process's own reading, not a gap to fill"


def test_a_pinned_fallback_spec_is_never_patched_with_the_selections_name() -> None:
    """Finding 5: the update is derived from the SELECTED model, so it follows it.

    ``stored`` is the checkpoint's selected model, and an effective spec can name a
    pinned fallback route instead. The cold state sets both fields from one object,
    so the divergence is an invariant to enforce rather than a reachable path — but
    an unenforced invariant is one refactor away from captioning a fallback with the
    selection's identity, on the one segment the band reads for its label.

    Asserted on the WINDOW, which this restore always has once ``_restored_pair``
    matches and the checkpoint carries one, so the test pins the GATE rather than
    the naming rules — those have the two tests above, and this one must not depend
    on what this environment's catalogue happens to call `deepseek/deepseek-flash`.
    """
    from local_operator.session.frontend_state import (
        FrontendModelSpec,
        FrontendSessionState,
    )

    state = FrontendSessionState(
        session_id="coldroute01",
        epoch="cold",
        selected_model=FrontendModelSpec(provider="deepseek", model_id="deepseek-flash"),
        # The route actually serving, which is not the selection.
        effective_model=FrontendModelSpec(
            provider="openrouter",
            model_id="deepseek/flash-v4",
            display_name="DeepSeek V4 Flash",
            context_window=262_144,
        ),
    )
    durable = FrontendSessionState(
        session_id="coldroute01",
        epoch="previous-owner",
        context_tokens=20_000,
        context_window=1_000_000,
        selected_model=FrontendModelSpec(
            provider="deepseek",
            model_id="deepseek-flash",
            context_window=1_000_000,
            display_name="DeepSeek Flash",
        ),
    )

    specs = AttachedSession._restored_model_specs(state, durable)

    assert (
        specs["selected_model"].context_window == 1_000_000
    ), "precondition: the same-model gate adopts the checkpoint's window"
    assert (
        specs["effective_model"].context_window == 262_144
    ), "a spec that names another model must not take the selection's denominator"
    assert (
        specs["effective_model"].display_name == "DeepSeek V4 Flash"
    ), "nor its name — the band reads the effective spec for its label"


@pytest.mark.asyncio
async def test_a_cold_viewer_never_paints_a_job_as_running(tmp_path: Path, monkeypatch) -> None:
    """With no runtime, nothing is running — and the roster must say so.

    The persisted roster records what a job's state WAS when it was written,
    and a session whose terminal closed mid-run persists ``running`` rows by
    design (observed on 8 sessions on the reference machine). Restored
    verbatim they paint a spinner for a child that cannot be working and the
    band counts them as live activity (UX round 1, U1) — worse than the empty
    panel this change replaces, because it is confidently wrong and invites a
    cancel that finds nothing.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)

    from local_operator.harness.types import Message
    from local_operator.session.frontend_state import (
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        FrontendSessionState,
        JobState,
    )
    from local_operator.session.transcript import Transcript

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("go"))
    durable = FrontendSessionState(
        session_id=SESSION_ID,
        epoch="previous-owner",
        jobs=[
            JobState(id="was-running", type="task", label="cut off", status="running"),
            JobState(
                id="was-parked", type="task", label="never started", status="running", queued=True
            ),
            JobState(id="finished", type="task", label="done", status="completed"),
        ],
    )
    await transcript.append_custom(
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        {"checkpoint_id": "c1", "state": durable.model_dump(mode="json")},
    )

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        jobs = {job.id: job for job in viewer.frontend_state.jobs}
        assert not any(
            job.status == "running" for job in jobs.values()
        ), "no job can be running when no runtime is alive"
        # Cut off mid-run: shown as interrupted so the panel can offer to
        # resume the child from its own transcript.
        assert jobs["was-running"].status == "interrupted"
        assert jobs["was-running"].restored is True
        # Parked and never started: it has no transcript to resume, so an
        # `interrupted` row would invite a resume that finds nothing.
        assert "was-parked" not in jobs
        # A settled fact the last runtime recorded is not relitigated.
        assert jobs["finished"].status == "completed"
        # And the band's own count agrees.
        running = sum(1 for job in jobs.values() if job.status == "running" and not job.queued)
        assert running == 0
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_an_unreadable_checkpoint_degrades_loudly(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """A corrupt status row must not silently reproduce the pre-fix experience.

    Falling back leaves exactly the empty roster this change exists to fix, so
    at DEBUG it is indistinguishable from the original bug and the next report
    gets re-diagnosed from scratch (UX round 1, U5). The open still succeeds.
    """
    import logging

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)

    from local_operator.harness.types import Message
    from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE
    from local_operator.session.transcript import Transcript

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("still here?"))
    await transcript.append_custom(
        FRONTEND_CHECKPOINT_CUSTOM_TYPE, {"checkpoint_id": "c1", "state": {"jobs": "not-a-list"}}
    )

    with caplog.at_level(logging.WARNING, logger="local_operator.session.attached"):
        viewer = await AttachedSession.cold(
            SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
        )
    try:
        assert any(
            "checkpoint unreadable" in record.getMessage() for record in caplog.records
        ), "an unreadable checkpoint must be reported at WARNING, not buried at DEBUG"
        # And the user is told, not just the log.
        assert viewer.degraded_reason
        # The conversation still opens: a status row never costs the history.
        assert [getattr(m, "text", "") for m in viewer.history()] == ["still here?"]
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_the_restored_roster_unions_the_sidecar_and_the_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    """Neither store alone is the full roster, so the restore reads both.

    The sidecar is written on every roster move and the checkpoint at turn
    end, so they disagree whenever a child settles after the last turn
    boundary. On the reference session they differ in BOTH directions — 18
    rows against 17, two children only the sidecar knows and one only the
    checkpoint knows — so a resume that read either alone dropped a child that
    really ran (UX round 1, U4).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)

    import json

    from local_operator.harness.types import Message
    from local_operator.session.frontend_state import (
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        FrontendSessionState,
        JobState,
    )
    from local_operator.session.session import SUBAGENT_ROSTER_SIDECAR
    from local_operator.session.transcript import Transcript

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("go"))
    durable = FrontendSessionState(
        session_id=SESSION_ID,
        epoch="previous-owner",
        jobs=[
            JobState(id="shared", type="task", label="in both", status="completed"),
            JobState(id="checkpoint-only", type="task", label="older store", status="completed"),
        ],
    )
    await transcript.append_custom(
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        {"checkpoint_id": "c1", "state": durable.model_dump(mode="json")},
    )
    (directory / SUBAGENT_ROSTER_SIDECAR).write_text(
        json.dumps(
            {
                "version": 1,
                "generation": 4,
                "records": [],
                "jobs": [
                    # The sidecar is fresher: its copy of the shared row wins.
                    {"id": "shared", "type": "task", "label": "in both", "status": "failed"},
                    {
                        "id": "sidecar-only",
                        "type": "task",
                        "label": "settled after the turn",
                        "status": "completed",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        jobs = {job.id: job for job in viewer.frontend_state.jobs}
        assert set(jobs) == {
            "shared",
            "checkpoint-only",
            "sidecar-only",
        }, "a child recorded by only one store still ran; the roster is the union"
        # Where both know a row, the fresher store wins.
        assert jobs["shared"].status == "failed"
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_cold_fork_does_not_list_its_parents_children(tmp_path: Path, monkeypatch) -> None:
    """#573 on the cold path: the checkpoint a fork inherits is the parent's.

    ``fork.EXCLUDED_SIDECARS`` keeps the roster sidecar out of a fork precisely
    so it does not list children it cannot address, and the checkpoint was
    smuggling equivalent rows past that exclusion. The rest of the durable
    state — title, spend, occupancy — IS the conversation's and still restores,
    and the identity the viewer reports is its own directory's.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)

    from local_operator.harness.types import Message
    from local_operator.session.frontend_state import (
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        FrontendModelSpec,
        FrontendSessionState,
        JobState,
    )
    from local_operator.session.transcript import Transcript

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("go"))
    inherited = FrontendSessionState(
        session_id="parent000001",
        epoch="parent-owner",
        conversation_title="Parent's title",
        jobs=[JobState(id="parent-child", type="task", label="auditor", status="succeeded")],
        cumulative_parent_cost=4.5,
        selected_model=FrontendModelSpec(provider="", model_id=""),
    )
    await transcript.append_custom(
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        {"checkpoint_id": "cp-parent", "state": inherited.model_dump(mode="json")},
    )
    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        state = viewer.frontend_state
        assert state.session_id == SESSION_ID
        assert state.jobs == ()
        assert state.conversation_title == "Parent's title"
        assert state.cumulative_parent_cost == 4.5
    finally:
        await viewer.dispose()


def _stamped(context_tokens: int, *, provider: str = "anthropic", model_id: str = "claude-opus-5"):
    """An assistant turn carrying the provider receipt a real turn persists."""
    from local_operator.harness.types import Usage

    return Usage(
        input_tokens=1_000,
        output_tokens=120,
        context_tokens=context_tokens,
        provider=provider,
        model_id=model_id,
    )


@pytest.mark.asyncio
async def test_a_checkpoint_less_resume_seeds_the_readings_from_the_transcript(
    tmp_path: Path, monkeypatch
) -> None:
    """The operator's report, reproduced: a cold conversation reads its own history.

    A desktop detaches between requests, so the durable frontend checkpoint —
    the cold path's only source of accounting — is usually absent on this
    surface. The conversation then opened with an empty context and no spend for
    a session already deep into its window, and stayed that way until the user
    spent a whole turn. The transcript holds the readings; nothing else did.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)
    _configure_provider(tmp_path)

    from local_operator.harness.types import Message
    from local_operator.session.frontend_state import CostKnowledge
    from local_operator.session.transcript import Transcript

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("how far along are we?"))
    await transcript.append_message(Message.assistant("deep in it", usage=_stamped(322_546)))

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        state = viewer.frontend_state
        assert viewer.is_cold is True, "seeding accounting must not start a runtime"
        assert state.context_tokens == 322_546
        assert (
            state.context_is_estimate is False
        ), "a provider receipt is exact, so it must not be replaced by the local estimate"
        assert state.cumulative_parent_cost is not None
        assert state.cost_knowledge == CostKnowledge.FLOOR
        assert state.last_usage is not None and state.last_usage.context_tokens == 322_546
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_the_seeded_cost_is_a_marked_floor_not_an_exact_total(
    tmp_path: Path, monkeypatch
) -> None:
    """One receipt prices ONE point in time, so the total is a floor.

    Summing the transcript's receipts would double-count a growing context (each
    reading already includes the previous ones), and calling the single newest
    receipt the lifetime total would hide every dollar spent before it. ``floor``
    is the only honest label, and the cost chip already prints its mark.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)
    _configure_provider(tmp_path)

    from local_operator.harness.types import Message
    from local_operator.session.frontend_state import CostKnowledge
    from local_operator.session.transcript import Transcript
    from local_operator.tui.costs import turn_cost

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("first"))
    await transcript.append_message(Message.assistant("old", usage=_stamped(90_000)))
    await transcript.append_message(Message.user("second"))
    newest = _stamped(322_546)
    await transcript.append_message(Message.assistant("new", usage=newest))

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        state = viewer.frontend_state
        assert state.cost_knowledge == CostKnowledge.FLOOR
        newest_cost = turn_cost("anthropic/claude-opus-5", newest)
        older_cost = turn_cost("anthropic/claude-opus-5", _stamped(90_000))
        assert (
            newest_cost is not None and older_cost is not None
        ), "precondition: both readings must be priceable, or the comparison is vacuous"
        assert state.cumulative_parent_cost == newest_cost
        assert (
            state.cumulative_parent_cost != older_cost + newest_cost
        ), "the floor prices the newest reading, it does not sum the transcript"
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_seeded_reading_carries_the_denominator_the_band_divides_by(
    tmp_path: Path, monkeypatch
) -> None:
    """The band's denominator and its numerator must arrive together.

    The window the strip divides by is the EFFECTIVE SPEC's — ``tui.app._context_window``
    reads exactly that on every later paint, deliberately, because the percentage
    predicts when the next request overflows. The state's own ``context_window`` was
    the stricter sibling of that reader: it took a denominator only from
    ``usage_seed.reading_window``, which additionally requires the reading to be
    ATTRIBUTABLE (a receipt for this model) and the metadata to be ACCOUNT-resolved.
    So on a config-only spec the numerator was published with the field left unset,
    and ``StatusLine.update`` reads ``None`` as LEAVE-ALONE: the first frames kept
    whatever the previous session had painted. The operator's own report is that
    frame — ``287.5k/—`` for two paints before the spec's own refresh landed 18ms
    later (QA round 1, Q1), and ``224.6%/128k`` when the outgoing session had a
    smaller window (Q2, the ``/resume`` path).

    Carrying the spec's own number makes the two agree by construction. The refusal
    that survives is the VALUE one, shared with ``reading_window``
    (``usage_seed.denominator_window``): a placeholder window is still never a
    denominator for a real reading — see
    ``test_a_cold_openai_session_does_not_divide_by_the_placeholder_window``, which
    asserts exactly that on the 128k placeholder and must keep doing so.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)
    _configure_provider(tmp_path)

    from local_operator.harness.types import Message
    from local_operator.session.transcript import Transcript

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("hello"))
    await transcript.append_message(Message.assistant("hi", usage=_stamped(322_546)))

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        state = viewer.frontend_state
        spec = state.selected_model
        assert spec is not None and spec.context_metadata_resolved is False, (
            "precondition: this config-only spec has no resolved window, so there is "
            "no denominator to vouch for"
        )
        assert spec.context_window == 1_000_000, (
            "precondition: the spec carries the MODEL's own window out of its metadata "
            "row, not a bare default (the config path resolves the pair so the effort "
            "ladder and level are answerable)"
        )
        assert state.context_window == spec.context_window, (
            "the state must carry the same denominator the band divides by on every "
            "later paint, or the numerator and its window arrive a paint apart"
        )
        assert state.context_tokens == 322_546, "the numerator is still a fact"
        # NOT asserted here, deliberately: that 322_546/1_000_000 is under 1.0.
        # Over-budget readings are something the band paints ON PURPOSE —
        # ``context_spelling(900_000, 872_000)`` starts ``103.2%`` and is asserted
        # as correct in ``tests/unit/model/test_openai_context.py`` — because the
        # percentage is what says the NEXT request overflows, which is precisely
        # the reading a stale denominator hides. The property this frame owes is
        # the one asserted above: ONE denominator, carried with its numerator
        # (review round 2, minor 2).
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_receipts_before_the_newest_prune_are_not_seeded(tmp_path: Path, monkeypatch) -> None:
    """A reading the pass invalidated must not be restored as the current one.

    The pruning counterpart of the compaction boundary: a blanked tool result
    shrinks the live context without leaving a marker, so a receipt from before
    the shrink describes a context that no longer exists (measured: 640_000
    restored for a real 31_715). The seed runs the SAME positional scan the owner
    runs, so it is refused here too — while a reading taken AFTER the prune is
    still seeded.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)
    _configure_provider(tmp_path)

    from local_operator.harness.types import Message, TextContent, ToolCall
    from local_operator.session.transcript import Transcript

    transcript = Transcript(directory)
    call = ToolCall(name="read", arguments={"path": "big.py"})
    await transcript.append_message(Message.user("read it"))
    await transcript.append_message(Message.assistant("reading", usage=_stamped(640_000)))
    tool_row = Message(
        role="tool",
        tool_call_id=call.id,
        tool_name="read",
        content=[TextContent(text="X" * 30_000)],
        provider_payload={"details": {"path": "big.py"}},
    )
    await transcript.append_message(tool_row)
    await transcript.append_prune(tool_row.id, "[pruned]")

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        state = viewer.frontend_state
        assert state.context_tokens is None, "every receipt here predates the prune"
        assert state.cumulative_parent_cost is None
        assert state.last_usage is None
    finally:
        await viewer.dispose()

    # A turn taken after the prune is a reading the shrink did not invalidate.
    await transcript.append_message(Message.user("what now?"))
    await transcript.append_message(Message.assistant("carry on", usage=_stamped(31_715)))
    resumed = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        assert resumed.frontend_state.context_tokens == 31_715
    finally:
        await resumed.dispose()


@pytest.mark.asyncio
async def test_a_transcript_with_no_receipts_seeds_nothing(tmp_path: Path, monkeypatch) -> None:
    """``None`` is not ``0``: an empty reading must stay empty.

    A brand-new conversation, a provider that reports no usage, and a history of
    nothing but user messages all look the same on disk. None of them justifies
    a confident ``0``, which the strip would render as a real reading of an empty
    context rather than as "nothing reported yet".
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)
    _configure_provider(tmp_path)

    from local_operator.harness.types import Message
    from local_operator.session.frontend_state import CostKnowledge
    from local_operator.session.transcript import Transcript

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("first words"))
    await transcript.append_message(Message.assistant("no receipt on this row"))

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        state = viewer.frontend_state
        assert state.context_tokens is None
        assert state.cumulative_parent_cost is None
        assert state.last_usage is None
        assert state.cost_knowledge == CostKnowledge.UNKNOWN
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_checkpoint_that_carried_accounting_wins_over_the_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    """The seed fills nulls; it never overwrites what the last turn end knew.

    The checkpoint is the conversation's LAST turn-end state, while a receipt is
    one point in time, and the checkpoint's figures were already reconciled by
    the owner that wrote them. A transcript that also carries older receipts must
    not drag those figures backwards.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)
    _configure_provider(tmp_path)

    from local_operator.harness.types import Message
    from local_operator.session.frontend_state import (
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        CostKnowledge,
        FrontendModelSpec,
        FrontendSessionState,
    )
    from local_operator.session.transcript import Transcript

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("hello"))
    await transcript.append_message(Message.assistant("hi", usage=_stamped(90_000)))
    durable = FrontendSessionState(
        session_id=SESSION_ID,
        epoch="previous-owner",
        context_tokens=322_546,
        context_window=1_000_000,
        cumulative_parent_cost=12.5,
        cost_knowledge=CostKnowledge.EXACT,
        selected_model=FrontendModelSpec(
            provider="anthropic", model_id="claude-opus-5", context_window=1_000_000
        ),
    )
    await transcript.append_custom(
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        {"checkpoint_id": "c1", "state": durable.model_dump(mode="json")},
    )

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        state = viewer.frontend_state
        assert state.context_tokens == 322_546
        assert state.cumulative_parent_cost == 12.5
        assert state.cost_knowledge == CostKnowledge.EXACT
        assert state.context_window == 1_000_000
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_cold_openai_session_does_not_divide_by_the_placeholder_window(
    tmp_path: Path, monkeypatch
) -> None:
    """B1 end to end: an openai spec whose window is the 128k PLACEHOLDER.

    This is the ordinary openai shape on a machine whose credential cannot be
    resolved: ``context_spec_for_access`` returns ``UNKNOWN_CONTEXT_WINDOW``
    together with ``context_metadata_resolved: True``. Trusting that flag divided
    a real 322_546-token receipt by 128_000 and printed a measured ``252.0%/128k``
    — a confidently wrong reading where the cold path previously showed none.

    The numerator is still a fact (the receipt names this model), and the floor
    cost is still the receipt's own, so this test fails on the WINDOW assertion if
    the placeholder guard is removed.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)
    from local_operator.config import ConfigManager

    ConfigManager(config_dir=tmp_path).update_config(
        {"hosting": "openai", "model_name": "gpt-5.6-sol"}
    )

    from local_operator.harness.types import Message
    from local_operator.model.configure import UNKNOWN_CONTEXT_WINDOW
    from local_operator.session.frontend_state import CostKnowledge
    from local_operator.session.transcript import Transcript

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("how far along are we?"))
    await transcript.append_message(
        Message.assistant(
            "deep in it",
            usage=_stamped(322_546, provider="openai", model_id="gpt-5.6-sol"),
        )
    )

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        state = viewer.frontend_state
        spec = state.selected_model
        assert (
            spec is not None and spec.context_window == UNKNOWN_CONTEXT_WINDOW
        ), "precondition: this is the placeholder pair the guard exists for"
        assert spec.context_metadata_resolved is True, "precondition: the flag alone is not enough"
        assert state.context_tokens == 322_546
        assert state.context_window is None, (
            "a placeholder window is not a denominator, however resolved the metadata claims "
            "to be"
        )
        assert state.cost_knowledge == CostKnowledge.FLOOR
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_checkpoint_window_is_not_replaced_under_its_own_tokens(
    tmp_path: Path, monkeypatch
) -> None:
    """M1: the window write obeys the method's own fill-only contract.

    A checkpoint's ``context_tokens``/``context_window`` pair is self-consistent —
    measured together, against the same budget. Importing a fresh denominator
    under the stored numerator computes a percentage the tokens were never
    measured on (a checkpoint at 500_000/1_050_000 printed as 390.6% of the fresh
    window), which is the same class of wrong reading as B1 reached from the other
    direction.

    The fresh spec here is the UNRESOLVED-ACCOUNT shape —
    ``context_spec_for_access``'s own output for an account it could not resolve
    (the 128k placeholder, both provenance fields absent) — because that is a
    population where this process resolved no budget and the checkpoint's window IS
    the frame's denominator. The synthetic ``400_000`` spec this test used to stub
    is no longer that population: a resolved NON-placeholder window on a non-local
    route is now read as an answer of its own
    (``_fresh_spec_states_a_budget``, review round 3 blocker 1), so it belongs to
    the refusal cells above, not to the fill-only one.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)
    from local_operator.config import ConfigManager

    ConfigManager(config_dir=tmp_path).update_config(
        {"hosting": "openai", "model_name": "gpt-5.6-sol"}
    )

    # A spec that states NO budget of its own, without a value written out here:
    # the real resolver's answer for an account it cannot reach.
    from local_operator.model.configure import context_spec_for_access
    from local_operator.session.frontend_state import FrontendModelSpec
    from local_operator.session.usage_seed import denominator_window

    unresolved = context_spec_for_access(
        FrontendModelSpec(provider="openai", model_id="gpt-5.6-sol"), None, {}
    )
    assert denominator_window(unresolved) is None, (
        "precondition: the placeholder is not a denominator, so nothing here can be mistaken "
        "for an answer of this process's own"
    )

    async def _unresolved(config_dir, model, *, stickiness_key):
        return FrontendModelSpec(**unresolved.model_dump(mode="python"))

    monkeypatch.setattr("local_operator.session.attached.resolve_context_metadata", _unresolved)

    from local_operator.harness.types import Message
    from local_operator.session.frontend_state import (
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        CostKnowledge,
        FrontendSessionState,
    )
    from local_operator.session.transcript import Transcript

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("hello"))
    await transcript.append_message(
        Message.assistant("hi", usage=_stamped(322_546, provider="openai", model_id="gpt-5.6-sol"))
    )
    durable = FrontendSessionState(
        session_id=SESSION_ID,
        epoch="previous-owner",
        context_tokens=500_000,
        context_window=1_050_000,
        cost_knowledge=CostKnowledge.EXACT,
        cumulative_parent_cost=12.5,
        selected_model=FrontendModelSpec(
            provider="openai", model_id="gpt-5.6-sol", context_window=1_050_000
        ),
    )
    await transcript.append_custom(
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        {"checkpoint_id": "c1", "state": durable.model_dump(mode="json")},
    )

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        state = viewer.frontend_state
        assert state.context_tokens == 500_000, "the checkpoint's numerator wins"
        assert state.context_window == 1_050_000, (
            "the checkpoint's own denominator must survive; replacing it computes a percentage "
            "its tokens were never measured on"
        )
        assert state.cumulative_parent_cost == 12.5
        assert state.cost_knowledge == CostKnowledge.EXACT
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "checkpoint_tokens",
        "checkpoint_window",
        "account_window",
        "account_default",
        "account_max",
        "expected_band",
    ),
    [
        pytest.param(250_000, 272_000, 872_000, 272_000, 872_000, "28.7%/872k", id="account-grew"),
        pytest.param(
            400_000, 872_000, 272_000, 272_000, 272_000, "147.1%/272k", id="account-shrank"
        ),
    ],
)
async def test_the_restore_refuses_a_stale_window_the_account_has_answered_for(
    tmp_path: Path,
    monkeypatch,
    checkpoint_tokens: int,
    checkpoint_window: int,
    account_window: int,
    account_default: int,
    account_max: int,
    expected_band: str,
) -> None:
    """Blocker 1 end to end: the account's CURRENT window governs the first frame.

    A checkpoint written under a 272k budget whose account now answers 872k — and
    the opt-out direction, a checkpoint under 872k whose account now answers 272k
    — is the shape where adopting the checkpoint's window splits the first frame
    from the frame the runtime paints moments later. Measured through both
    streams' own paths while the gate was absent: ``cold 110.3%/272k`` against
    ``live 34.4%/872k`` when the account grew, and ``cold 45.9%/872k`` against
    ``live 147.1%/272k`` when it shrank.

    BOTH DIRECTIONS ARE PINNED, on the literal band string, because only one of
    them looks wrong: the SHRANK cell is the HIDING direction — ``45.9%/872k`` is
    calm where the conversation is over budget, so a refusal that fired only when
    the fresh window is LARGER would leave the suite green while the surface that
    exists to warn stayed quiet (review round 2, blocker 1; design round 2, D1;
    review round 3, minor 2).

    The discrimination is ``_fresh_spec_states_a_budget``, on the value rule the
    band itself applies: the fresh spec carries a window
    ``usage_seed.denominator_window`` VOUCHES (these cells state one on
    ``context_window`` and both provenance fields), so the model layer answered
    for this pair and the checkpoint's window is the stale one. Only the WINDOW
    moves — the numerator is the conversation's own reading and stays, which is
    the pairing the runtime publishes on attach
    (``frontend_state.refresh_from_session``: ``receipt_context or
    current.context_tokens`` beside the effective spec's own window).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)
    from local_operator.config import ConfigManager

    ConfigManager(config_dir=tmp_path).update_config(
        {"hosting": "openai", "model_name": "gpt-5.6-sol"}
    )

    # The account's own answer, on BOTH provenance fields: that is one of the two
    # shapes in which the model layer says it resolved this pair, and it is the one
    # the round-3 blocker is NOT about (the API-key population below states the
    # window on ``context_window`` alone).
    async def _account_answered(config_dir, model, *, stickiness_key):
        return model.model_copy(
            update={
                "context_window": account_window,
                "default_context_window": account_default,
                "max_context_window": account_max,
                "context_metadata_resolved": True,
            }
        )

    monkeypatch.setattr(
        "local_operator.session.attached.resolve_context_metadata", _account_answered
    )

    from local_operator.harness.types import Message
    from local_operator.session.frontend_state import (
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        FrontendModelSpec,
        FrontendSessionState,
    )
    from local_operator.session.transcript import Transcript
    from local_operator.tui.widgets.status_line import context_spelling

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("how far along are we?"))
    await transcript.append_message(
        Message.assistant(
            "deep in it",
            usage=_stamped(checkpoint_tokens, provider="openai", model_id="gpt-5.6-sol"),
        )
    )
    # A self-consistent pair, as a real turn-end checkpoint writes: the tokens are
    # 91.9% of the window they were measured against.
    durable = FrontendSessionState(
        session_id=SESSION_ID,
        epoch="previous-owner",
        context_tokens=checkpoint_tokens,
        context_window=checkpoint_window,
        selected_model=FrontendModelSpec(
            provider="openai", model_id="gpt-5.6-sol", context_window=checkpoint_window
        ),
    )
    await transcript.append_custom(
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        {"checkpoint_id": "c1", "state": durable.model_dump(mode="json")},
    )

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        state = viewer.frontend_state
        spec = state.effective_model or state.selected_model
        tokens, window = state.context_tokens, state.context_window
        assert tokens is not None and window is not None, (
            "the restored pair is the conversation's own reading: a numerator and the window "
            "it is divided by"
        )
        assert tokens == checkpoint_tokens, "the restored numerator is the conversation's own"
        assert spec is not None and spec.context_window == account_window, (
            "the account's current window is what this spec resolved, so the checkpoint's stale "
            f"{checkpoint_window // 1000}k must not replace it — that replacement is the "
            "first-frame reading the live frame contradicts"
        )
        assert window == account_window, (
            "and the state-level field follows it, because that is the window the seed's "
            "denominator rule vouches (``usage_seed.denominator_window``) — one band, one "
            "denominator, whichever paint lands last"
        )
        assert context_spelling(tokens, window) == expected_band, (
            "the literal first-frame reading: the shrank cell is the one that HIDES an "
            "over-budget conversation behind a calm percentage"
        )
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_the_restore_refuses_a_stale_window_on_an_api_key_after_an_oauth_run(
    tmp_path: Path, monkeypatch
) -> None:
    """Blocker 1's population: a pair whose only budget is ``context_window``.

    The SAME PAIR resumed on a different CREDENTIAL is still the same model, so
    ``_restored_pair`` matches and the checkpoint's reading is a candidate for
    restoration. What differs is the budget: a Codex/OAuth-served run held 272,000
    tokens of room, and an OpenAI API key resolves the public row's 1,050,000.

    This population is what a gate keyed on ``default_context_window``/
    ``max_context_window`` cannot see, and structurally rather than as an edge of
    the credential matrix: the shipped catalogue states a window through
    ``context_window`` alone (0 of its 120 rows set either provenance field), so
    for a complete first-hand row — ``_listing_can_correct`` is False for
    ``openai/gpt-5.6-sol``, i.e. no listing can ever correct it — those two fields
    stay ``None`` on an API key for good. ``denominator_window`` VOUCHES the
    1,050,000, so it is a budget and not the placeholder: this process resolved an
    answer of its own, and restoring the checkpoint's 272,000 first-painted
    ``110.3%/272k`` where the attach frame paints ``28.6%/1.1M`` — verbatim the
    string round 2 was filed on (review round 3, blocker 1).

    The mirror direction (an API-key checkpoint resumed on an OAuth account whose
    opt-out resolves 272k WITH both provenance fields) is covered by the
    parametrised test above, which is why the one-directional hole was easy to
    miss.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)
    from local_operator.config import ConfigManager
    from local_operator.model import discovery
    from local_operator.model.configure import context_spec_for_access
    from local_operator.providers.auth_store import OAuthAccess
    from local_operator.session.frontend_state import FrontendModelSpec
    from local_operator.session.usage_seed import denominator_window

    # No listing leg is answerable for a complete first-hand row, which is also the
    # production state for the 93 shipped rows that carry no listing at all.
    monkeypatch.setattr(discovery, "available_models", lambda *args, **kwargs: ([], "static"))
    ConfigManager(config_dir=tmp_path).update_config(
        {"hosting": "openai", "model_name": "gpt-5.6-sol"}
    )

    # The resolver's own answer for this pair on an API key, computed by the real
    # function rather than written out here; only the credential lookup (which a
    # unit test cannot have) is supplied for it.
    access = OAuthAccess("sk-public", 0, kind="api_key")
    settings = ConfigManager(config_dir=tmp_path).get_config().values
    fresh = context_spec_for_access(
        FrontendModelSpec(provider="openai", model_id="gpt-5.6-sol"), access, settings
    )
    assert fresh.context_window == 1_050_000, "precondition: the public row's own window"
    assert fresh.default_context_window is None and fresh.max_context_window is None, (
        "precondition: the ONLY budget this resolution states is ``context_window``, which is "
        "what the narrower gate read as `nothing answered`"
    )
    assert denominator_window(fresh) == 1_050_000, (
        "precondition: the band VOUCHES this window, so it is a budget rather than the 128k "
        "placeholder — the whole basis for calling it an answer of this process's own"
    )

    async def _api_key_account(config_dir, model, *, stickiness_key):
        return FrontendModelSpec(**fresh.model_dump(mode="python"))

    monkeypatch.setattr(
        "local_operator.session.attached.resolve_context_metadata", _api_key_account
    )

    from local_operator.harness.types import Message
    from local_operator.session.frontend_state import (
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        FrontendSessionState,
    )
    from local_operator.session.transcript import Transcript
    from local_operator.tui.widgets.status_line import context_spelling

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("how far along are we?"))
    await transcript.append_message(
        Message.assistant(
            "deep in it", usage=_stamped(300_000, provider="openai", model_id="gpt-5.6-sol")
        )
    )
    durable = FrontendSessionState(
        session_id=SESSION_ID,
        epoch="previous-owner",
        context_tokens=300_000,
        context_window=272_000,
        selected_model=FrontendModelSpec(
            provider="openai", model_id="gpt-5.6-sol", context_window=272_000
        ),
    )
    await transcript.append_custom(
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        {"checkpoint_id": "c1", "state": durable.model_dump(mode="json")},
    )

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        state = viewer.frontend_state
        assert state.context_tokens == 300_000, "the numerator stays: it is the conversation's own"
        assert state.context_window == 1_050_000, (
            "the API key's own window is the one this frame must divide by; the OAuth run's "
            "272,000 is stale and only the narrowed gate let it through"
        )
        assert context_spelling(state.context_tokens, state.context_window) == "28.6%/1.1M", (
            "the literal reading the attach frame also paints — the cold frame used to say "
            "110.3%/272k here"
        )
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_the_restore_refuses_a_stale_window_for_a_pair_with_no_listing_leg(
    tmp_path: Path, monkeypatch
) -> None:
    """Blocker 1's second population: a shipped row no listing can correct.

    ``anthropic/claude-opus-4-20250514`` is one of the 93 shipped rows with no
    listing leg, so the fresh spec is the registry's own 200,000 with both
    provenance fields absent — permanent, not an outage. A checkpoint written when
    that row said 1,000,000 (or written on a route whose answer exceeded the row's)
    therefore restored a window eight times the process's own, and the frame read
    ``25.0%/1M`` where the attach frame paints ``125.0%/200k`` — the HIDING
    direction, on the surface whose whole purpose is to warn before the next
    request overflows.

    Nothing about the account is involved here: the pair's OWN row answers, and the
    gate has to read that answer off ``context_window`` to see it.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)
    from local_operator.config import ConfigManager
    from local_operator.model import discovery
    from local_operator.model.configure import build_model_spec
    from local_operator.session.frontend_state import FrontendModelSpec
    from local_operator.session.usage_seed import denominator_window

    monkeypatch.setattr(discovery, "available_models", lambda *args, **kwargs: ([], "static"))
    ConfigManager(config_dir=tmp_path).update_config(
        {"hosting": "anthropic", "model_name": "claude-opus-4-20250514"}
    )

    fresh = build_model_spec("anthropic", "claude-opus-4-20250514")
    assert fresh.context_window == 200_000, "precondition: the shipped row's own window"
    assert (
        denominator_window(fresh) == 200_000
    ), "precondition: vouched, so the row's window is a budget this process resolved"

    from local_operator.harness.types import Message
    from local_operator.session.frontend_state import (
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        FrontendSessionState,
    )
    from local_operator.session.transcript import Transcript
    from local_operator.tui.widgets.status_line import context_spelling

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("how far along are we?"))
    await transcript.append_message(
        Message.assistant(
            "deep in it",
            usage=_stamped(250_000, provider="anthropic", model_id="claude-opus-4-20250514"),
        )
    )
    durable = FrontendSessionState(
        session_id=SESSION_ID,
        epoch="previous-owner",
        context_tokens=250_000,
        context_window=1_000_000,
        selected_model=FrontendModelSpec(
            provider="anthropic", model_id="claude-opus-4-20250514", context_window=1_000_000
        ),
    )
    await transcript.append_custom(
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        {"checkpoint_id": "c1", "state": durable.model_dump(mode="json")},
    )

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        state = viewer.frontend_state
        assert state.context_tokens == 250_000, "the numerator stays"
        assert (
            state.context_window == 200_000
        ), "the row's own window governs, so the over-budget conversation reads over budget"
        assert (
            context_spelling(state.context_tokens, state.context_window) == "125.0%/200k"
        ), "the literal reading: adopting the checkpoint would paint 25.0%/1M and hide it"
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_seeded_floor_is_priced_on_the_receipts_own_model(
    tmp_path: Path, monkeypatch
) -> None:
    """m3: a receipt from another model was BILLED at that model's rates.

    The conversation here runs on anthropic; the newest receipt was served by
    openai. The reading is the spend that actually happened, so it is priced on the
    model that served it — pricing it on the session's model reported 0.008 where
    the receipt's own stamp gives 0.0064.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)
    _configure_provider(tmp_path)

    from local_operator.harness.types import Message
    from local_operator.session.frontend_state import CostKnowledge
    from local_operator.session.transcript import Transcript
    from local_operator.tui.costs import turn_cost

    receipt = _stamped(322_546, provider="openai", model_id="gpt-5.6-sol")
    transcript = Transcript(directory)
    await transcript.append_message(Message.user("hello"))
    await transcript.append_message(Message.assistant("hi", usage=receipt))

    own = turn_cost("openai/gpt-5.6-sol", receipt)
    session_model = turn_cost("anthropic/claude-opus-5", receipt)
    assert own is not None and session_model is not None, "precondition: both must be priceable"
    assert own != session_model, "precondition: the two rate tables differ"

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        state = viewer.frontend_state
        assert state.context_tokens is None, "the numerator was measured on another model"
        assert state.cost_knowledge == CostKnowledge.FLOOR
        assert state.cumulative_parent_cost == own
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_an_unattributable_receipt_seeds_no_cost(tmp_path: Path, monkeypatch) -> None:
    """m3: no stamp and no saved selection means the receipt is not billable to anyone.

    Before this, an unattributable reading was still priced at whatever model the
    session happens to be configured with — a number with no evidence behind it.
    ``None`` means no chip, which is the honest answer.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)
    _configure_provider(tmp_path)

    from local_operator.harness.types import Message, Usage
    from local_operator.session.frontend_state import CostKnowledge
    from local_operator.session.transcript import Transcript

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("hello"))
    await transcript.append_message(
        Message.assistant(
            "hi", usage=Usage(input_tokens=1_000, output_tokens=50, context_tokens=90_000)
        )
    )

    viewer = await AttachedSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        state = viewer.frontend_state
        assert state.last_usage is not None, "the receipt itself is still shown"
        assert state.context_tokens is None
        assert state.cumulative_parent_cost is None
        assert state.cost_knowledge == CostKnowledge.UNKNOWN
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_cursor_cut_read_leaves_the_cold_seed_unset(tmp_path: Path, monkeypatch) -> None:
    """n1: the seed follows the same cut as the replay it rides on.

    ``_read_transcript`` stashes the accounting fallback from the rows it read,
    but a ``through_id`` cursor discards rows ABOVE the cursor from that replay. A
    receipt from those rows describes a window this viewer is not showing, so the
    seed is skipped there rather than a second copy of the cut rule being written
    beside the replay's — a second boundary is exactly the disagreement the shared
    scan exists to prevent.

    Cold passes no cursor today, which is why this pins the SHAPE: a future
    ``want_checkpoint=True`` caller that negotiates a window must not seed from
    receipts the window excludes.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _seed_transcript(tmp_path, SESSION_ID)
    _configure_provider(tmp_path)

    from local_operator.harness.types import Message
    from local_operator.session.transcript import Transcript

    transcript = Transcript(directory)
    await transcript.append_message(Message.user("older turn"))
    cursor = await transcript.append_message(
        Message.assistant("receipt below the cursor", usage=_stamped(90_000))
    )
    await transcript.append_message(Message.user("newer turn"))
    await transcript.append_message(
        Message.assistant("receipt above the cursor", usage=_stamped(322_546))
    )

    # Built the way ``cold()`` builds one, WITHOUT its consume-and-clear step:
    # the stash is explicitly one-shot, so the read has to be observed directly.
    viewer = AttachedSession(
        config_dir=tmp_path,
        session_id=SESSION_ID,
        takeover_factory=_never,
        surface="terminal",
    )
    try:
        # An uncut read seeds from every receipt in the file...
        await viewer._load_history(None, want_checkpoint=True)
        seeded = viewer._cold_seed_usage
        assert seeded is not None and seeded.context_tokens == 322_546

        # ...a read cut at a row BEFORE the newest receipt must not seed from it.
        await viewer._load_history(cursor.id, want_checkpoint=True)
        assert (
            viewer._cold_seed_usage is None
        ), "a receipt above the cursor is not part of the window this viewer replays"

        # And the gate is the cut alone: the same read without one seeds again.
        await viewer._load_history(None, want_checkpoint=True)
        restored = viewer._cold_seed_usage
        assert restored is not None and restored.context_tokens == 322_546
    finally:
        await viewer.dispose()
