"""Aida's TUI surfaces, driven through the real app rather than the helpers.

Two things this file exists for, both learned the hard way:

* the PINNED-FIRST picker claim is about the row PIPELINE (``_cmd_resume``
  builds rows, overlays live state, then hands them to the screen). A test of
  the reorder function alone proved the function and shipped nothing — the
  first cut called it from nowhere, and only a rendered picker showed her
  last. So these tests type ``/resume`` and read the screen.
* the receipts are the feature's whole visible surface for pause/resume/status
  (decision (c): no extra chrome), so they are asserted as the text a user
  reads, on the transcript, in a booted app.
"""

from __future__ import annotations

import asyncio

import pytest

from local_operator.tui.widgets.editor import Editor
from tests.unit.tui.test_app_pilot import (
    FakeSession,
    _factory,
    _resume_factory,
    _seed_session,
    _transcript_text,
)


def _boot(tmp_path, monkeypatch, *, resume_boots=None):
    """An app on an isolated root, with the real config-dir resolution."""
    from local_operator.tui.app import OperatorApp

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    app = OperatorApp(
        lambda: _factory(FakeSession()),
        resume_factory=_resume_factory(resume_boots if resume_boots is not None else []),
    )
    return app


async def _settle(pilot, seconds: float = 2.0) -> None:
    """Pause for a bounded WALL-CLOCK grace, not an iteration count.

    WHY DEADLINES RATHER THAN LOOP BUDGETS (round-1 remediation): the first
    `/aida` open in a process pulls `session_factory` in cold while checking
    whether a greeting is owed (~0.8-2.3 s measured here), so a budget counted
    in event-loop iterations made a test's outcome depend on which test in an
    xdist worker happened to pay that import — a worker-order change flipped
    whole tests. These loops mean "give the worker enough time"; a wall-clock
    bound says exactly that and is order-independent.
    """
    import time as _time

    deadline = _time.monotonic() + seconds
    while _time.monotonic() < deadline:
        await pilot.pause()


async def _until(pilot, predicate, seconds: float = 20.0) -> None:
    """Pause until ``predicate()`` holds, or the deadline; assert after."""
    import time as _time

    deadline = _time.monotonic() + seconds
    while _time.monotonic() < deadline:
        await pilot.pause()
        if predicate():
            return


@pytest.mark.asyncio
async def test_the_picker_leads_with_her_pinned_row(tmp_path, monkeypatch) -> None:
    """R27's picker half, through the pipeline that actually builds the rows.

    Three real sessions: hers (created and pinned by ``ensure_session``) plus
    two ordinary ones whose transcripts are FRESHER than her directory's
    mtime, so a pure-recency order would put her last — which is exactly what
    the first cut of this feature did.
    """
    import os

    from local_operator import aida as aida_pkg

    her_id = await aida_pkg.ensure_session(tmp_path)
    assert her_id is not None, "the isolate must be enabled by default"
    _seed_session(tmp_path, "aaaaaaaaaaa1", prompt="fix the flaky login test")
    _seed_session(tmp_path, "bbbbbbbbbbb2", prompt="draft the release notes")
    # Her directory is made the OLDEST on purpose: the picker's native order is
    # ``(-mtime, id)``, so without the pin she would land last (and the id
    # tiebreak would not save her either — '6' sorts before 'a'). The pin is
    # therefore the only reason she can be first, which is what R27 claims.
    old = os.stat(tmp_path / "sessions" / her_id).st_mtime - 3600
    os.utime(tmp_path / "sessions" / her_id, (old, old))

    from local_operator.tui.widgets.session_picker import SessionPickerScreen

    app = _boot(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        editor.text = "/resume"
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        await pilot.press("enter")
        await _until(
            pilot,
            lambda: isinstance(app.screen, SessionPickerScreen) and bool(app.screen._all),
        )

        picker = app.screen
        assert isinstance(picker, SessionPickerScreen)
        ids = [row.id for row in picker._all]
        assert ids[0] == her_id, (
            "the pinned conversation leads the picker; a pure-recency order "
            f"would list her last. Order was {ids}"
        )
        # The rest keep their order — the reorder moves pinned rows, it does
        # not sort the remainder. In recency order the seeds are newer than
        # her directory, so both follow her.
        assert ids[1:] == ["bbbbbbbbbbb2", "aaaaaaaaaaa1"], ids
        # And the FIRST painted row is hers, not just the first in a list.
        painted = "\n".join(picker.render_lines_for_test())
        assert painted.index("Aida") < painted.index("fix the flaky"), painted


@pytest.mark.asyncio
async def test_the_pause_resume_status_receipts(tmp_path, monkeypatch) -> None:
    """``/aida pause``, ``resume`` and ``status`` each answer in one line.

    The pause receipt must also be true on disk (config flipped, hold marker
    written) — a receipt that only says the words would pass a transcript
    assertion and hold nothing.
    """
    from local_operator import aida as aida_pkg
    from local_operator.config import ConfigManager
    from tests.unit.aida.conftest import mark_met

    # Steady state: she has met the operator, so her ensure arms the cadence.
    mark_met(tmp_path)
    her_id = await aida_pkg.ensure_session(tmp_path)
    assert her_id is not None
    # The cadence arm is what gives ``pause`` an index entry to stamp: her
    # ensure writes ``wakes/<id>.json`` with the ``aida-cadence`` one-shot.
    from local_operator.wakes import store as wake_store

    assert wake_store.read_entry(tmp_path, her_id), "ensure must arm the cadence"

    app = _boot(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)

        async def run(command: str) -> str:
            editor.focus()
            editor.text = command
            editor.move_cursor(editor._end_of_buffer())
            await pilot.pause()
            await pilot.press("enter")
            await _settle(pilot, 3.0)
            return _transcript_text(app)

        body = await run("/aida status")
        # The identity mark rides every receipt she mints (design round 1, D1)
        # and the budget reads as prose, not "0/2" jargon (D3).
        assert "\u21c8 Aida: active" in body, body
        assert "next check-in" in body, body
        # The pair rides non-breaking spaces so a narrow column cannot split
        # "0 of" from its ceiling (D3's copy, measured in the rendered frame).
        assert "extra check-ins used today: 0\u00a0of\u00a02" in body, body

        body = await run("/aida pause")
        assert "\u21c8 Aida: paused" in body, body
        # ``get_nested_value``, not ``get_config_value``: the settings registry
        # stores this under the nested path (("aida","cadence","paused")), and
        # the dotted string is a verbatim top-level key that nothing writes —
        # the mismatch config.py's own docstring warns about.
        assert (
            ConfigManager(config_dir=tmp_path).get_nested_value(("aida", "cadence", "paused"))
            is True
        )
        # Pause CANCELS her rows (the freeze: remembered in state for resume);
        # the held_at marker itself is exercised where it exists for a reason —
        # against a surviving owner row (test_aida_proactive's pause test).
        held = wake_store.read_entry(tmp_path, her_id) or {}
        assert [row["id"] for row in held.get("schedules") or []] == [], held

        body = await run("/aida resume")
        assert "\u21c8 Aida: active again" in body, body
        again = wake_store.read_entry(tmp_path, her_id) or {}
        assert "held_at" not in again, again
        assert [row["id"] for row in again.get("schedules") or []] == ["aida-cadence"], again
        assert (
            ConfigManager(config_dir=tmp_path).get_nested_value(("aida", "cadence", "paused"))
            is False
        )

        body = await run("/aida =pause now")  # the escape sends her the WORD
        assert "\u21c8 Aida: paused" not in body, "the `=` escape must not be parsed as the verb"


@pytest.mark.asyncio
async def test_a_request_rides_the_adoption_onto_her_conversation(tmp_path, monkeypatch) -> None:
    """``/aida <request>`` types ONCE: the text is delivered to her session.

    The stash-then-consume seam (``_pending_aida_prompt`` →
    ``_submit_aida_prompt``) is the one place a request can be silently
    dropped, so the assertion is on the SESSION's recorded prompt, not on a
    receipt: a dropped request leaves a perfectly quiet screen.
    """
    # The app resolves ``config_dir()`` from the ENV, so the isolate has to be
    # the env's root BEFORE the app is built: with the two out of step, the
    # app creates a SECOND Aida next to the one this test made — which is how
    # this test first failed, and it is the whole reason every test here goes
    # through ``_boot``'s env or sets it before construction.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    from local_operator import aida as aida_pkg

    her_id = await aida_pkg.ensure_session(tmp_path)

    built: list[FakeSession] = []
    boots: list[str | None] = []

    class HerSession(FakeSession):
        """A fake whose ``session_id`` is the one it was booted FOR.

        ``FakeSession.session_id`` is a read-only property returning ``"sess"``
        — assignment raises — and the adoption seam compares exactly this
        attribute, so the id has to be the boot's.
        """

        def __init__(self, sid: str) -> None:
            super().__init__()
            self._sid = sid

        @property
        def session_id(self) -> str:
            return self._sid

    async def resume_factory(session_id):
        boots.append(session_id)
        session = HerSession(session_id or "")
        built.append(session)
        return session

    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=resume_factory)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # Wait for the boot session before sending: a command handled while the
        # boot is in flight sees `_session is None` and answers "still
        # starting" instead of opening her — the race this test hit
        # deterministically under load (round-1 remediation).
        await _until(pilot, lambda: bool(app._conversation_id()))
        editor = app.query_one(Editor)
        editor.focus()
        editor.text = "/aida summarise everything in flight"
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        await pilot.press("enter")
        await _until(pilot, lambda: built and built[-1].prompts)

        assert boots == [her_id], f"the factory must be asked for HER conversation: {boots}"
        assert built[0].prompts == ["summarise everything in flight"], built[0].prompts
        assert app._conversation_id() == her_id


def _viewer_for(her_id: str):
    """A faked AttachedSession: FakeSession with HER id, recording prompts."""
    from tests.unit.tui.test_app_pilot import FakeSession

    class HerViewer(FakeSession):
        @property
        def session_id(self) -> str:  # type: ignore[override]
            return her_id

    return HerViewer()


@pytest.mark.asyncio
async def test_her_request_survives_an_attach_onto_a_live_owner(tmp_path, monkeypatch) -> None:
    """U1: the BLOCKER's success half — `/aida hello` lands on a live her.

    The attach path (`_resume_session` → `_attach_or_refuse` →
    `_adopt_built_viewer`) spends neither prompt seam, so a request typed
    against an already-running Aida switched the conversation and dropped the
    words: no transcript row, no notice, nothing on disk. This drives that
    exact seam and asserts the session RECORDS the prompt.
    """
    from local_operator import aida as aida_pkg
    from local_operator.session.attached import AttachedSession
    from tests.unit.tui.test_resume_connect_retry import _app, _record, _running

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    her_id = await aida_pkg.ensure_session(tmp_path)
    assert her_id is not None

    viewer = _viewer_for(her_id)
    record = _record(90909, "her conversation")

    async def cold(*_args, **_kwargs):
        raise ConnectionError("no cold paint on this fake")

    async def connect(*_args, **_kwargs):
        return viewer

    monkeypatch.setattr("local_operator.session.attached.AttachedSession.cold", cold)
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)
    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 90909),
    )
    assert AttachedSession is not None  # imported for the patch targets' sake

    app = _app(monkeypatch, tmp_path)

    async with _running(app):
        # INSIDE the running app, because booting it adopts a session of its
        # own — and every adoption spends-or-drops the stash. A user types
        # `/aida hello` long after boot; setting it before `_running` measured
        # the boot instead of the attach (this test's first cut did exactly
        # that and passed the wrong way).
        app._pending_aida_prompt = (her_id, "hello", None)
        await app._attach_or_refuse(tmp_path, her_id)
        for _ in range(60):
            if viewer.prompts:
                break
            await asyncio.sleep(0)

        assert app._pending_aida_prompt is None, "the stash must be spent, not stranded"
        assert viewer.prompts == ["hello"], "the request must reach HER session"


@pytest.mark.asyncio
async def test_a_refused_attach_drops_the_request_with_a_notice(tmp_path, monkeypatch) -> None:
    """U1's other ending: when the transition cannot happen, say so.

    A request that outlives a failed transition would fire on some LATER
    adoption — words typed minutes ago delivered into a fresh conversation.
    It is dropped, and the drop is narrated rather than silent.
    """
    from local_operator import aida as aida_pkg
    from tests.unit.tui.test_resume_connect_retry import (
        _app,
        _notices,
        _record,
        _running,
    )

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    her_id = await aida_pkg.ensure_session(tmp_path)
    assert her_id is not None

    record = _record(90909, "her conversation")

    async def cold(*_args, **_kwargs):
        raise ConnectionError("no cold paint on this fake")

    async def connect(*_args, **_kwargs):
        raise ConnectionError("the runtime is not responding")

    monkeypatch.setattr("local_operator.session.attached.AttachedSession.cold", cold)
    monkeypatch.setattr("local_operator.session.attached.AttachedSession.connect", connect)
    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record",
        lambda _root, _concrete: (record, 90909),
    )

    app = _app(monkeypatch, tmp_path)

    async with _running(app):
        # After boot, for the reason the success test documents.
        app._pending_aida_prompt = (her_id, "hello", None)
        await app._attach_or_refuse(tmp_path, her_id)

        assert app._pending_aida_prompt is None, "a stale stash must not fire later"
        assert any("Request not sent" in text for text in _notices(app)), _notices(app)


@pytest.mark.asyncio
async def test_a_reserved_word_counts_only_as_the_whole_argument(tmp_path, monkeypatch) -> None:
    """The cross-host grammar (UI review round 1, MINOR-3): word, or message.

    ``/aida pause and think`` used to split on the first space and run the
    control word, silently DROPPING "and think" — a typed request, gone, under
    a receipt about pausing. The word ALONE is the control; anything longer is
    a message for her, verbatim; ``=`` still escapes the word itself (the
    ``/team =chart`` precedent).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    from local_operator import aida as aida_pkg
    from local_operator.config import ConfigManager

    await aida_pkg.ensure_session(tmp_path)

    built: list[FakeSession] = []

    class HerSession(FakeSession):
        def __init__(self, sid: str) -> None:
            super().__init__()
            self._sid = sid

        @property
        def session_id(self) -> str:  # type: ignore[override]
            return self._sid

    async def resume_factory(session_id):
        session = HerSession(session_id or "")
        built.append(session)
        return session

    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=resume_factory)

    def paused() -> bool:
        got = ConfigManager(config_dir=tmp_path).get_nested_value(("aida", "cadence", "paused"))
        return got is True

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # Wait for the boot session before sending (see the sibling note in
        # `test_a_request_rides_the_adoption_onto_her_conversation`): a command
        # handled mid-boot sees no session and answers "still starting"
        # instead of reaching her.
        await _until(pilot, lambda: bool(app._conversation_id()))
        editor = app.query_one(Editor)

        async def send(command: str, *, wait_for_prompt: bool) -> None:
            editor.focus()
            editor.text = command
            editor.move_cursor(editor._end_of_buffer())
            await pilot.pause()
            before = len(built[-1].prompts) if built else 0
            await pilot.press("enter")
            if wait_for_prompt:
                # Wait for THIS send's prompt, not merely for a non-empty
                # history: after the first message, `built[-1].prompts` is
                # already populated and the old condition returned before the
                # new prompt landed (a race that read as a fast test).
                await _until(pilot, lambda: built and len(built[-1].prompts) > before)
            else:
                await _settle(pilot, 2.0)

        # Words with more in them are a MESSAGE — never the control, and never
        # silently truncated.
        await send("/aida pause and think", wait_for_prompt=True)
        assert paused() is False, "a message must not run the control word"
        assert built[-1].prompts == ["pause and think"], built[-1].prompts

        # The word ALONE is the control.
        await send("/aida pause", wait_for_prompt=False)
        assert paused() is True

        # `=` sends the literal word as a message (escape sigil consumed, like
        # `/team =chart`); the control stays untriggered.
        await send("/aida =pause", wait_for_prompt=True)
        assert paused() is True, "the escape must not run the control"
        assert built[-1].prompts == ["pause and think", "pause"], built[-1].prompts


@pytest.mark.asyncio
async def test_the_rename_verb_updates_both_stores(tmp_path, monkeypatch) -> None:
    """``/aida rename <name>``: config, her stored title, and a receipt.

    The entry point the operator asked for, through the real handler: the
    config key (what every surface reads) and her session's stored title
    (what the picker, the sidebar and a resume read) must move TOGETHER, the
    receipt must name the stored value, and the escape must still reach her
    as a message without renaming anything. The rename made while HER
    CONVERSATION IS OPEN (the ``=rename nope`` send opened it) must NOT call
    the session setter: on the attached lane its rename RPC is refused
    ("/rename is terminal-only here") and leaves an unretrieved task
    exception per rename (UX round 1, U2) — the live re-title rides the
    config watcher, pinned at session level in ``test_aida_session_hooks``.
    """
    from local_operator import aida as aida_pkg
    from local_operator.config import ConfigManager
    from local_operator.resume import stored_session_title

    her_id = await aida_pkg.ensure_session(tmp_path)
    assert her_id is not None

    built: list[FakeSession] = []
    setter_calls: list[str] = []

    class HerSession(FakeSession):
        def __init__(self, sid: str) -> None:
            super().__init__()
            self._sid = sid

        @property
        def session_id(self) -> str:  # type: ignore[override]
            return self._sid

    async def resume_factory(session_id):
        session = HerSession(session_id or "")
        built.append(session)
        return session

    # The real config-dir resolution, like ``_boot`` sets up: the rename
    # worker and the receipts resolve THROUGH ``paths.config_dir()``, so the
    # env must point at this test's root or the write lands elsewhere.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=resume_factory)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)

        async def send(command: str) -> str:
            await _await_session(app, pilot)
            editor.focus()
            editor.text = command
            editor.move_cursor(editor._end_of_buffer())
            await pilot.pause()
            # Enter on an open list completes the highlighted row instead of
            # submitting (the ``test_approvals_ux`` rule): these sends are
            # about the RECEIPTS, so an exact word that matches a row (bare
            # ``rename``, ``status``) is submitted deliberately with the list
            # dismissed first. The list itself is pinned by
            # ``test_the_space_after_aida_offers_the_reserved_words``.
            if editor.picker.is_open():
                await pilot.press("escape")
                await pilot.pause()
            before = _transcript_text(app)
            await pilot.press("enter")
            # Progress-based settlement (review round 1, N2 / QA Q2): a fixed
            # turn count lost the cold-boot race.
            for _ in range(400):
                await pilot.pause()
                if _transcript_text(app) != before:
                    break
            for _ in range(40):
                await pilot.pause()
            return _transcript_text(app)

        # Bare `rename` reports; the word WITH a name renames.
        body = await send("/aida rename")
        assert "Aida: /aida rename <name> renames her everywhere." in body, body
        body = await send("/aida rename Sovereign")
        assert "renamed: Sovereign" in body, body

        # A receipt minted AFTER the rename speaks the new name...
        body = await send("/aida status")
        assert "\u21c8 Sovereign: active" in body, body

        # ...and the escape still sends her the literal words, unparsed —
        # which also OPENS her conversation, the lane U2 is about.
        await send("/aida =rename nope")
        assert built and built[-1].prompts and "rename nope" in built[-1].prompts[-1]
        assert app._conversation_id() == her_id

        # Rename while her conversation is OPEN here: the stores move and the
        # session setter must NOT be called (U2). The recorder mirrors the
        # real signature so a stray call cannot pass unnoticed.
        session = built[-1]
        original = session.set_conversation_name

        def record(text: str, *, user_set: bool = True) -> str:
            setter_calls.append(text)
            return original(text, user_set=user_set)

        session.set_conversation_name = record  # type: ignore[method-assign]
        body = await send("/aida rename Vega")
        assert "renamed: Vega" in body, body
        body = await send("/aida rename")
        assert "Vega: /aida rename <name> renames her everywhere." in body, body

    # BOTH stores moved: the config key every surface reads, and the stored
    # title the picker/sidebar/resume read.
    assert ConfigManager(config_dir=tmp_path).get_nested_value(("aida", "name")) == "Vega"
    assert stored_session_title(tmp_path / "sessions" / her_id) == "Vega"
    assert setter_calls == [], setter_calls


@pytest.mark.asyncio
async def test_the_rename_verb_refuses_a_bad_name(tmp_path, monkeypatch) -> None:
    """An invalid name is refused with the registry's own sentence, and
    nothing is written — neither the config nor her title."""
    from local_operator import aida as aida_pkg
    from local_operator.config import ConfigManager
    from local_operator.resume import stored_session_title

    her_id = await aida_pkg.ensure_session(tmp_path)
    assert her_id is not None

    app = _boot(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # Wait out the cold-boot race before dispatching (N2 / QA Q2): a
        # refusal read off a dropped command must still be the RENAME's refusal.
        await _await_session(app, pilot)
        editor = app.query_one(Editor)
        editor.focus()
        editor.text = "/aida rename " + "x" * 81
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        before = _transcript_text(app)
        await pilot.press("enter")
        # Progress-based settlement, like ``_send`` (N2 / QA Q2).
        for _ in range(400):
            await pilot.pause()
            if _transcript_text(app) != before:
                break
        for _ in range(40):
            await pilot.pause()
        body = _transcript_text(app)
        assert "not renamed: at most 80 characters" in body, body

    assert ConfigManager(config_dir=tmp_path).get_nested_value(("aida", "name"), None) is None
    assert stored_session_title(tmp_path / "sessions" / her_id) == "Aida"


async def _await_session(app, pilot, *, cap: int = 900) -> None:
    """Wait out the cold-boot race before sending ``/aida …`` (N2 / QA Q2).

    ``OperatorApp`` refuses slash commands while its first session is still
    starting (the ``_cmd_aida`` head guard), and on a loaded host that window
    is longer than any fixed pause count — the command is dropped and the
    test reads ``! session is still starting…``. Wait on the STATE.
    """
    for _ in range(cap):
        if app._session is not None:
            break
        await pilot.pause()
    assert app._session is not None, "the boot session never arrived"


async def _send(app, pilot, command: str, *, until=None) -> str:
    """Type ``command`` into the composer and return the transcript after.

    Dismisses an open argument list first — Enter on an open list completes
    the highlighted row instead of submitting (the ``test_approvals_ux``
    rule), which bare words like ``status`` now match. The list's own
    behaviour is pinned separately.

    Settlement is PROGRESS-based, not clock-based (review round 1, N2 / QA
    Q2): on a loaded host a fixed turn count races the session boot, so the
    helper waits for the transcript to move (or ``until`` to go true), then
    gives the follow-up paint a few turns.
    """
    editor = app.query_one(Editor)
    await _await_session(app, pilot)
    editor.focus()
    editor.text = command
    editor.move_cursor(editor._end_of_buffer())
    await pilot.pause()
    if editor.picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    before = _transcript_text(app)
    await pilot.press("enter")
    for _ in range(400):
        await pilot.pause()
        if until() if until is not None else _transcript_text(app) != before:
            break
    for _ in range(40):
        await pilot.pause()
    return _transcript_text(app)


@pytest.mark.asyncio
async def test_the_space_after_aida_offers_the_reserved_words(tmp_path, monkeypatch) -> None:
    """UX round 1, U1: ``/aida<space>`` opens HER words, not the provider list.

    The provider fall-through is wrong for every ``/aida`` argument — the
    free-form half is prose for her — and it also hid the ``rename`` verb
    from anyone who never learned to type it. The rows are sourced from
    ``AIDA_SUBCOMMANDS`` plus ``rename``, the words the handler itself
    parses, so the picker cannot drift from the grammar.
    """
    from local_operator.slash_commands import AIDA_SUBCOMMANDS
    from local_operator.tui.widgets.command_picker import PickerMode

    app = _boot(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        editor.text = "/aida "
        editor.move_cursor(editor._end_of_buffer())
        editor._sync_picker()
        await pilot.pause()
        picker = editor.picker
        assert picker.mode is PickerMode.ARGUMENT, "the argument list did not open"
        names = [name for name, _ in picker.suggestions()]
        assert names == [*AIDA_SUBCOMMANDS, "rename"], names


@pytest.mark.asyncio
async def test_renaming_her_conversation_says_she_is_renamed_too(tmp_path, monkeypatch) -> None:
    """UX round 1, U3: ``/title`` on HER conversation names the side effect.

    The config row (``applied: aida.name``) is the audit trail; the rename's
    own receipt is where the expectation forms, so it says the product-wide
    half out loud — the sentence that stops "renamed the thread" from
    reading as "renamed only this thread". ``_aida_duty`` is the same gate
    the config sync reads, so the clause cannot claim a rename that did not
    sync.
    """
    from local_operator import aida as aida_pkg

    her_id = await aida_pkg.ensure_session(tmp_path)
    assert her_id is not None

    built: list[FakeSession] = []

    class HerSession(FakeSession):
        def __init__(self, sid: str) -> None:
            super().__init__()
            self._sid = sid
            self._aida_duty = True

        @property
        def session_id(self) -> str:  # type: ignore[override]
            return self._sid

    async def resume_factory(session_id):
        session = HerSession(session_id or "")
        built.append(session)
        return session

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=resume_factory)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _send(app, pilot, "/aida hello")
        for _ in range(200):
            await pilot.pause()
            if built:
                break
        assert built and app._conversation_id() == her_id
        body = await _send(app, pilot, "/title Vega")
        assert "renamed: Vega — she is now called Vega everywhere" in body, body


@pytest.mark.asyncio
async def test_the_launcher_receipt_names_the_configured_name(tmp_path, monkeypatch) -> None:
    """Review R1-M1 / UX U4: the launcher-less receipt read the packaged name.

    Only reachable when ``OperatorApp`` is built without ``resume_factory``
    (test/embedding construct — the shipped CLI always provides one), which
    is exactly why it survived the first sweep.
    """
    from local_operator import settings_io
    from local_operator.config import ConfigManager
    from local_operator.tui.app import OperatorApp

    settings_io.write_setting(
        ConfigManager(config_dir=tmp_path), settings_io.BY_KEY["aida.name"], "Nova"
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        body = await _send(
            app,
            pilot,
            "/aida",
            until=lambda: "requires a session-capable" in _transcript_text(app),
        )
        assert "opening Nova requires a session-capable launcher" in body, body


@pytest.mark.asyncio
async def test_a_settings_rename_reconciles_her_stored_title(tmp_path, monkeypatch) -> None:
    """Review R1-M3: a non-``/aida`` rename must not leave the pinned row stale.

    The reported gap: with her session CLOSED and a TUI running, a rename
    written outside ``/aida rename`` (the /settings page, ``lop config
    edit``, the desktop PATCH) reached the config and nothing else — the
    sidebar row and the picker, which render the STORED title, kept the old
    name until the next ``ensure_session``. The app's config-change listener
    now reconciles the stored title on the same delivery.

    The boot ensure is awaited BEFORE the write, so the only actor left when
    the write lands is the watcher seam under test (nothing else reconciles
    spontaneously; the title is asserted un-moved first).
    """
    from local_operator import aida as aida_pkg
    from local_operator import settings_io
    from local_operator.config import ConfigManager
    from local_operator.resume import stored_session_title

    her_id = await aida_pkg.ensure_session(tmp_path)
    assert her_id is not None
    assert stored_session_title(tmp_path / "sessions" / her_id) == "Aida"

    app = _boot(tmp_path, monkeypatch)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        if app._aida_boot_task is not None:
            await app._aida_boot_task
        for _ in range(20):
            await pilot.pause()
        assert stored_session_title(tmp_path / "sessions" / her_id) == "Aida"

        # The /settings page's own writer; ``notify_local`` hands it to this
        # process's watcher exactly as the page does in production.
        settings_io.write_setting(
            ConfigManager(config_dir=tmp_path), settings_io.BY_KEY["aida.name"], "Vega"
        )
        # Give the delivery turns to arrive, then drain the listener's worker
        # rather than racing its timing.
        for _ in range(60):
            await pilot.pause()
        await app.workers.wait_for_complete()
        assert stored_session_title(tmp_path / "sessions" / her_id) == "Vega"


@pytest.mark.asyncio
async def test_opening_her_conversation_with_a_framework_toast_up_survives(
    tmp_path, monkeypatch
) -> None:
    """QA round 1, Q1: the framework's notification toast shares the TYPE NAME
    ``Toast`` — and Textual resolves a class selector by type name, so
    ``_reset_band_for_swap``'s ``self.query(Toast)`` used to hand it the
    framework's widget, which has no ``withdraw``, and opening her
    conversation while one was up killed the app. The command palette's
    Screenshot notice is the production source; the toast is mounted directly
    because ``App.notify`` does not paint one in headless runs.

    Both halves are driven: the real ``/aida`` flow that crashed, and the
    swap helper itself with the foreign toast still on screen.
    """
    from textual.notifications import Notification
    from textual.widgets._toast import Toast as FrameworkToast

    from local_operator import aida as aida_pkg

    her_id = await aida_pkg.ensure_session(tmp_path)
    assert her_id is not None

    boots: list[str | None] = []
    app = _boot(tmp_path, monkeypatch, resume_boots=boots)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        foreign = FrameworkToast(Notification(message="Saved screenshot", timeout=100))
        await app.screen.mount(foreign)
        await pilot.pause()

        await _send(app, pilot, "/aida")
        for _ in range(200):
            await pilot.pause()
            if boots:
                break
        assert app.is_running, "the app died during the session transition"
        # The swap ran: the app asked for HER id (the fake's own id is canned).
        assert boots and boots[-1] == her_id, boots

        # The exact helper the swap runs, with the foreign toast still up.
        app._reset_band_for_swap()
        assert app.is_running
        assert list(app.query(FrameworkToast)), "the foreign toast was not touched"


@pytest.mark.asyncio
async def test_aida_with_no_provider_opens_her_view_at_the_cue(tmp_path, monkeypatch) -> None:
    """R28: in the setup state, `/aida` renders HER view at the provider cue.

    The defect this replaces: with no provider, `/aida` answered "session is
    still starting…" — a promise of a session that cannot arrive until
    `/login` succeeds. The view instead frames the screen as hers, carries the
    shared cue, creates her durable session (the bootstrap works without a
    provider, and R7 needs the SAME one after login), and refuses a typed send
    with the same cue instead of the generic waiting line.
    """
    from local_operator.session_factory import HostingNotConfiguredError
    from tests.unit.tui.test_app_pilot import _await_setup_state

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    async def _no_hosting_factory():
        raise HostingNotConfiguredError("Hosting platform is not configured.")

    from local_operator.tui.app import OperatorApp

    app = OperatorApp(_no_hosting_factory, resume_factory=_resume_factory([]))
    async with app.run_test(size=(100, 30)) as pilot:
        await _await_setup_state(app, pilot)
        editor = app.query_one(Editor)

        # The SPLASH arm's refusal (before `/aida` opens her view): the shared
        # cue carries the diagnosis, so the cue appears
        # exactly ONCE — the prefix this PR used to add put it at both ends of
        # one line (design review round 2, NIT).
        editor.focus()
        editor.text = "hello"
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        await pilot.press("enter")
        await _until(pilot, lambda: "Not sent" in _transcript_text(app))
        # The recommended command, named once (audit D1/U2: Radient first).
        assert _transcript_text(app).count("connect an AI account") == 1
        assert _transcript_text(app).count("/login radient") == 1

        async def send(line: str, until: object = None) -> None:
            editor.focus()
            editor.text = line
            editor.move_cursor(editor._end_of_buffer())
            await pilot.pause()
            await pilot.press("enter")
            if callable(until):
                await _until(pilot, until)
            else:
                await _settle(pilot, 3.0)

        # The predicate must be a fact ONLY her view can satisfy. The splash
        # block above already put the no-provider cue in the transcript,
        # so waiting on the cue returned on its first pause — while
        # `_aida_open_without_provider` still had `ensure_session()` to run —
        # and the `"Aida" in body` assertion below raced the render. Waiting
        # on her own words restores the self-synchronising shape this block
        # used to have (review round 3, R3-F1: 3 failed / 4 runs at load
        # 150-270 with the cue as the predicate, mechanism in this diff).
        await send("/aida", lambda: "chief of staff" in _transcript_text(app))
        assert app._aida_setup_view is True
        assert "/login radient" in (app._splash_notice or "")
        body = _transcript_text(app)
        assert "Aida" in body
        assert "connect an ai account" in body.lower()
        assert "still starting" not in body
        # D1: no introduction promise — the greeting is gated on
        # `first_run_pending` and an install with conversations never gets it,
        # so the sentence was false there.
        assert "introduce herself" not in body

        # Her durable session exists (created WITHOUT a provider), so the
        # conversation opened after `/login` is the same one (R7).
        from local_operator.aida import state as aida_state

        her_id = aida_state.session_id_of(tmp_path)
        assert her_id and (tmp_path / "sessions" / her_id).is_dir()

        # A typed send is refused with the SAME cue, and says nothing was sent.
        await send("hello there", lambda: "can't reply yet" in _transcript_text(app))
        body = _transcript_text(app)
        assert "can't reply yet" in body
        assert "your message was not sent" in body
        assert "/login radient" in body
        assert "still starting" not in body

        # `/aida <text>` opens the view and says the request was not sent.
        # Asserted on the unwrapped fragment: the block wraps at 100 columns,
        # so the sentence arrives with a newline inside it.
        await send("/aida book me a flight", lambda: "say it again once" in _transcript_text(app))
        body = _transcript_text(app)
        assert "say it again once" in body


@pytest.mark.asyncio
async def test_a_typed_message_at_the_setup_splash_names_login(tmp_path, monkeypatch) -> None:
    """U3: with no provider, the refusal must name the route that CONNECTS one.

    The launcher's shared startup map says "Settings > Providers" — desktop
    vocabulary on a terminal, naming a surface the TUI does not have (its
    `/settings` Providers section holds provider OPTIONS, not a place to
    connect one). In the setup state there is no session and no turn, so the
    answer is the shared cue; this pins the splash half, and her view's half is
    pinned above.
    """
    from local_operator.session_factory import HostingNotConfiguredError
    from tests.unit.tui.test_app_pilot import _await_setup_state

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    async def _no_hosting_factory():
        raise HostingNotConfiguredError("Hosting platform is not configured.")

    from local_operator.tui.app import OperatorApp

    app = OperatorApp(_no_hosting_factory, resume_factory=_resume_factory([]))
    async with app.run_test(size=(100, 30)) as pilot:
        await _await_setup_state(app, pilot)
        editor = app.query_one(Editor)
        editor.focus()
        editor.text = "hello"
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        await pilot.press("enter")
        await _until(pilot, lambda: "/login radient" in _transcript_text(app))

        body = _transcript_text(app)
        assert "/login radient" in body
        assert "still starting" not in body
        assert "Settings > Providers" not in body


@pytest.mark.asyncio
async def test_first_run_login_opens_her_conversation_and_arms_the_greeting(
    tmp_path, monkeypatch
) -> None:
    """R26: finishing setup on a fresh install opens HER conversation.

    Driven through the real `/login` flow: the setup state leaves, the routing
    predicate is true (no human conversations, a provider now resolves), the
    rebuild targets her id, and the greeting row is already armed when the
    reload lands. An existing install fails the predicate and boots as before
    — pinned in `tests/unit/aida/test_aida_onboarding.py` on the predicate
    itself, which is the half a fake-session test cannot see.
    """
    from local_operator.session_factory import HostingNotConfiguredError
    from tests.unit.tui.test_app_pilot import FakeProviderController, _await_setup_state

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    from local_operator import aida as aida_pkg

    her_id = await aida_pkg.ensure_session(tmp_path)
    assert her_id is not None

    boots: list[str | None] = []

    class HerSession(FakeSession):
        def __init__(self, sid: str) -> None:
            super().__init__()
            self._sid = sid

        @property
        def session_id(self) -> str:
            return self._sid

    async def resume_factory(session_id):
        boots.append(session_id)
        return HerSession(session_id or "")

    async def _no_hosting_factory():
        raise HostingNotConfiguredError("Hosting platform is not configured.")

    from local_operator.tui.app import OperatorApp

    app = OperatorApp(
        _no_hosting_factory,
        resume_factory=resume_factory,
        provider_controller=FakeProviderController(),
    )
    async with app.run_test(size=(100, 30)) as pilot:
        await _await_setup_state(app, pilot)
        editor = app.query_one(Editor)
        editor.focus()
        editor.text = "/login deepseek"
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        await pilot.press("enter")
        await _until(pilot, lambda: boots and app._conversation_id() == her_id)

        assert boots == [her_id], f"the rebuild must target HER conversation: {boots}"
        assert app._conversation_id() == her_id

    # The greeting was armed BEFORE the reload, so it fires into her session.
    from local_operator.wakes import store as wake_store

    entry = wake_store.read_entry(tmp_path, her_id) or {}
    ids = [row["id"] for row in entry.get("schedules") or []]
    assert "aida-greeting" in ids, entry


@pytest.mark.asyncio
async def test_opening_her_conversation_arms_the_owed_greeting(tmp_path, monkeypatch) -> None:
    """R20/R22 on the OTHER first contact: `/aida` on a provider-present install.

    The setup seam only runs when a `/login` ENDS a setup state; an install
    that already resolves a provider boots to a normal conversation and meets
    her from `/aida` (or the picker). Nothing armed the greeting on that path
    — first contact got no greeting, and the earliest natural fire was the
    next 09:00 cadence (review round 1, U2).

    The boot-materialised directory below is the shipped shape this test host
    originally could not see: a normal launch leaves one session directory
    (lease + pid, no transcript) behind, and while existence alone counted as
    "the operator has conversations" the predicate read false on a fresh
    install's first boot — this test failed on that head and passes with the
    engagement-discounting predicate (UX review round 2, U4).
    """
    from local_operator.config import ConfigManager

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    ConfigManager(config_dir=tmp_path).update_config({"hosting": "test", "model_name": "mock"})

    booted = tmp_path / "sessions" / "bootmaterialised0001"
    booted.mkdir(parents=True)
    (booted / ".execution-lease").write_text('{"pid": 1}', encoding="utf-8")
    (booted / ".session.pid").write_text("1", encoding="utf-8")

    from local_operator import aida as aida_pkg

    her_id = await aida_pkg.ensure_session(tmp_path)
    assert her_id

    boots: list[str | None] = []

    class HerSession(FakeSession):
        @property
        def session_id(self) -> str:  # type: ignore[override]
            return her_id

    async def resume_factory(session_id):
        boots.append(session_id)
        return HerSession()

    from local_operator.tui.app import OperatorApp

    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=resume_factory)
    entry = {}
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # Wait for the boot session before sending (the mid-boot race above).
        await _until(pilot, lambda: bool(app._conversation_id()))
        editor = app.query_one(Editor)
        editor.focus()
        editor.text = "/aida"
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        await pilot.press("enter")

        from local_operator.wakes import store as wake_store

        def _armed() -> bool:
            entry = wake_store.read_entry(tmp_path, her_id) or {}
            ids = [row.get("id") for row in entry.get("schedules") or []]
            return "aida-greeting" in ids

        await _until(pilot, _armed)
        entry = wake_store.read_entry(tmp_path, her_id) or {}

    ids = [row.get("id") for row in entry.get("schedules") or []]
    assert "aida-greeting" in ids, entry
    assert boots == [her_id], boots
    from local_operator.aida import onboarding

    # Armed by an ATTENDED surface (the TUI), hidden, and not yet delivered:
    # ``greeted_at`` now means "delivered", stamped at the actual fire.
    assert onboarding.greeting_state(tmp_path) == onboarding.GREETING_ARMED
    assert onboarding.greeting_record(tmp_path)["surface"] == "tui"
    assert onboarding.greeted_at(tmp_path) is None
    row = next(r for r in entry.get("schedules") or [] if r.get("id") == "aida-greeting")
    assert row.get("hidden") is True


# --------------------------------------------------------------------------- #
# First-run onboarding (Lane B): boot routing, the setup composer, /credential
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_first_boot_with_a_provider_already_configured_opens_her(
    tmp_path, monkeypatch
) -> None:
    """A8: a provider set up BEFORE the first launch (env key, `lop login`,
    the desktop) never passes the setup state, so the post-login seam never
    ran and the user met an empty chat. The boot now routes to her, requests
    the greeting from this ATTENDED surface, and arms it hidden."""
    from local_operator.aida import onboarding
    from local_operator.config import ConfigManager
    from local_operator.tui.app import OperatorApp
    from local_operator.wakes import store as wake_store

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    ConfigManager(config_dir=tmp_path).update_config({"hosting": "test", "model_name": "mock"})

    boots: list[str | None] = []

    async def resume_factory(session_id):
        boots.append(session_id)

        class Hers(FakeSession):
            @property
            def session_id(self) -> str:  # type: ignore[override]
                return session_id or ""

        return Hers()

    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=resume_factory)
    async with app.run_test(size=(100, 30)) as pilot:
        await _until(pilot, lambda: bool(boots))
        her_id = onboarding.state.session_id_of(tmp_path)
        assert her_id and boots == [her_id], boots

        def _armed() -> bool:
            entry = wake_store.read_entry(tmp_path, her_id) or {}
            return any(r.get("id") == "aida-greeting" for r in entry.get("schedules") or [])

        await _until(pilot, _armed)
    assert onboarding.greeting_state(tmp_path) == onboarding.GREETING_ARMED
    assert onboarding.greeting_record(tmp_path)["surface"] == "tui"


@pytest.mark.asyncio
async def test_an_existing_install_boots_as_before_and_is_never_greeted(
    tmp_path, monkeypatch
) -> None:
    """R22 under the boot route: conversations exist, so no reroute, no greeting,
    and the ledger records ``skipped`` so the question is never asked again."""
    from local_operator.aida import onboarding
    from local_operator.config import ConfigManager

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    ConfigManager(config_dir=tmp_path).update_config({"hosting": "test", "model_name": "mock"})
    _seed_session(tmp_path, "aaaaaaaaaaa1", prompt="fix the flaky login test")

    boots: list[str | None] = []
    app = _boot(tmp_path, monkeypatch, resume_boots=boots)
    async with app.run_test(size=(100, 30)) as pilot:
        await _until(pilot, lambda: bool(app._conversation_id()))
        await _settle(pilot, 1.0)
    assert boots == []
    assert onboarding.greeting_state(tmp_path) == onboarding.GREETING_SKIPPED


@pytest.mark.asyncio
async def test_the_setup_composer_names_the_command_and_credential_refuses(
    tmp_path, monkeypatch
) -> None:
    """D7 + U11: the setup placeholder is the instruction (painted in the
    stronger ink via ``-setup``), and ``/credential`` refuses up front with the
    `/login` remedy instead of walking the user into a masked paste."""
    from local_operator.session_factory import HostingNotConfiguredError
    from local_operator.tui.app import SETUP_PLACEHOLDER, OperatorApp
    from tests.unit.tui.test_app_pilot import FakeProviderController, _await_setup_state

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    async def _no_hosting():
        raise HostingNotConfiguredError("Hosting platform is not configured.")

    app = OperatorApp(_no_hosting, provider_controller=FakeProviderController())
    async with app.run_test(size=(100, 30)) as pilot:
        await _await_setup_state(app, pilot)
        editor = app.query_one(Editor)
        assert editor.placeholder == SETUP_PLACEHOLDER == "Type /login radient to begin"
        assert editor.has_class("-setup")
        editor.focus()
        editor.text = "/credential GITHUB_TOKEN"
        editor.move_cursor(editor._end_of_buffer())
        await pilot.pause()
        await pilot.press("enter")
        await _until(pilot, lambda: "/credential stores secrets" in _transcript_text(app))
        assert "/login radient" in _transcript_text(app)
        # Leaving the state takes the class and the placeholder with it.
        app._setup_state = False
        assert not editor.has_class("-setup")
        assert editor.placeholder != SETUP_PLACEHOLDER
