"""The ``@path`` feature over a REAL ``Session``, through the assembled app.

WHY THIS FILE EXISTS apart from ``test_at_submit.py`` and ``test_at_picker.py``:
those drive the composer over a ``FakeSession``, which is the right weight for
composer behaviour and the wrong one for this feature's dangerous half. Two
defects in the first review round shipped GREEN precisely because the fakes
lacked the machinery under test — a missing approval gate, and a turn-lock test
that passed vacuously because the fake had no ``_turn_lock`` at all. Neither is
visible from a double that does not have the part.

So every test here runs the production seam end to end: composer →
``on_editor_submitted`` → ``Session.prompt`` → ``expand_references`` → the
recorded provider request, over a real ``Session``, a real ``Transcript`` and a
real ``OperatorApp``.

TWO RULES this file has to obey, both learned the hard way.

* **Do not inject an approval gate.** ``_adopt_session`` REPLACES an adopted
  session's handler with the app's interactive one (``app.py:9124``), so a
  recording gate installed at construction is silently discarded and the card
  then waits forever for a keypress nobody sends. The test that needs a gate
  ARMS the session (``yolo=False``) and lets the app install the real one —
  which is also the only way to be testing the shipped routing rather than a
  double's idea of it.
* **Bound waits in LOOP TURNS, never seconds** (AGENTS.md, "Wait on the event,
  never on the clock"). There is no publication to subscribe to here — the work
  is an in-process coroutine the app itself scheduled — so :func:`_settle`
  pumps the pilot and re-tests the predicate each turn. A wedge then fails as an
  assertion naming what never arrived, instead of blocking a suite that has no
  pytest-timeout to stop it.

Every secret here is a PLACEHOLDER marker (``SECRET_KEY=MARKER_ENV_777``). No
test reads, writes or asserts on a real credential, and the markers are what make
"the secret did not reach the model" a byte-level claim rather than a guess.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from local_operator.references import REFERENCE_BLOCK_OPEN
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.approval import ApprovalPrompt
from local_operator.tui.widgets.command_picker import PickerMode
from local_operator.tui.widgets.editor import Editor, EditorSubmitted, PastedText
from tests.e2e.harness import (
    TEST_MODEL,
    ScriptedStream,
    build_session,
    dispose_quietly,
    drain,
    text_turn,
    transcript_text,
    wait_for_adoption,
)
from tests.unit.harness.test_comms import MAX_PUMP_TURNS

#: Workspace bodies. Each carries a unique marker so "reached the model" and
#: "stayed out of the model" are both byte-level facts, never an inference from
#: a length or a substring that could appear by accident.
README_BODY = "# Title\n\nMARKER_README_913 is the sentinel line.\n"
AUTH_BODY = "def login():\n    return MARKER_AUTH_42\n"
#: A placeholder, not a credential. The value exists only so a leak is
#: detectable by a string that cannot occur anywhere else in the process.
ENV_BODY = "SECRET_KEY=MARKER_ENV_777\n"


def _sent_to_model(stream: ScriptedStream) -> str:
    """Every text block of every message the provider was actually handed.

    Reads the RECORDED REQUESTS rather than anything the app believes it sent,
    because the request is the only artifact the model ever sees. A test that
    asserted on the composer's text would pass on an expansion that never
    reached the wire.
    """
    chunks: list[str] = []
    for request in stream.requests:
        for message in request.messages:
            for block in getattr(message, "content", None) or []:
                text = getattr(block, "text", None)
                if text:
                    chunks.append(text)
    return "\n".join(chunks)


def _armed_session(directory: Path, stream: Any, cwd: Path) -> Session:
    """A real session with the approval gate ARMED (``yolo=False``).

    ``build_session`` hardcodes ``yolo=True``, and under yolo the resolver's
    contract is auto-approve (``references.py``'s ``_approved`` returns True on
    a ``None`` gate), so a sensitive path would be read without asking and the
    card under test would never mount.

    No gate is passed here, deliberately — see the module docstring. The app
    installs its OWN interactive one when it adopts the session, and that is the
    production seam under test; a gate injected here is silently replaced by it.
    """
    from local_operator.variables import VariableStore

    return Session(
        model=TEST_MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda *_args: [],
        yolo=False,
        cwd=str(cwd),
        variables=VariableStore(cwd=str(cwd)),
    )


async def _factory(session: Session) -> Session:
    """The app's session factory, already resolved."""
    return session


async def _settle(pilot: Any, predicate: Callable[[], bool], turns: int = MAX_PUMP_TURNS) -> bool:
    """Pump the pilot until ``predicate`` holds; report whether it ever did.

    Bounded in LOOP TURNS rather than seconds: a turn count survives contention
    that a wall-clock budget does not, and the predicate is re-tested after
    every pump, so the wait lasts exactly as long as the work does. The bound is
    a backstop that turns a wedge into a named assertion failure — this suite
    has no pytest-timeout, so an unbounded await here would hang the shard.
    """
    for _ in range(turns):
        if predicate():
            return True
        await pilot.pause()
    return predicate()


async def _type_and_submit(pilot: Any, app: OperatorApp, text: str) -> None:
    """Put ``text`` in the real composer and press Enter.

    Types into the buffer rather than posting an ``EditorSubmitted``, so the
    picker sync, the paste splice and the submit routing all run as they do for
    a user. The Escape is conditional because a trailing ``@token`` leaves the
    FILE picker open, and an open picker owns Enter — submitting through it
    would complete a row instead of sending the draft.

    Deliberately does NOT wait for the provider: what "done" means differs per
    test (a turn, an aside, an approval card), and a wait baked in here would
    be a guess the caller cannot see. Each test settles on its own predicate.
    """
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    editor.load_text(text)
    editor.move_cursor(editor._end_of_buffer())
    await pilot.pause()
    if editor.picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("enter")


async def _type_keystrokes(pilot: Any, app: OperatorApp, text: str) -> None:
    """Type ``text`` into the real composer ONE KEY AT A TIME, pressing nothing else.

    Deliberately separate from :func:`_type_and_submit`, which loads the buffer
    wholesale and ESCAPES out of an open picker first. That escape hatch is the
    reason the feature's headline flow shipped broken: a draft ending in a
    reference leaves the FILE list open, and an open FILE list owns Enter, so a
    helper that dismisses the list before pressing Enter can never observe
    whether the user's own Enter sends the message. Its own docstring admitted
    the hazard and the 43-test composer suite still went green over it (QA
    round 1, Q-1).

    So anything asserting on what an ORDINARY keystroke does has to arrive the
    way a user arrives: real ``pilot.press`` per character, no ``load_text``,
    no Escape, no pre-parked caret. ``load_text`` is what the other helper uses
    and it is right there, because those tests are about the submit path rather
    than about the keystroke that reaches it.
    """
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    for ch in text:
        await pilot.press("space" if ch == " " else ch)
    await pilot.pause()


@pytest.mark.asyncio
async def test_reference_reaches_the_model_expanded_and_the_row_stays_short(
    tmp_path: Path,
) -> None:
    """The feature's whole promise, at the seam: model gets the body, user sees
    the token.

    Both halves are asserted, and the transcript half is pinned POSITIVELY as
    well as negatively. ``transcript_text`` returns ``""`` when the screen has
    no transcript view, so a bare "the marker is absent" would pass on an empty
    screen — the vacuous-pass shape that let a turn-lock test ship green against
    a fake with no lock. Asserting the typed line IS on screen closes it.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    (workspace / "README.md").write_text(README_BODY, encoding="utf-8")
    stream = ScriptedStream([text_turn("Summarised.")])
    session = build_session(tmp_path / "session", stream, cwd=workspace)
    session.set_conversation_name("at-realpath-expanded", user_set=True)
    app = OperatorApp(lambda: _factory(session))
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await wait_for_adoption(app, pilot)
            await drain(pilot)
            await _type_and_submit(pilot, app, "summarise @README.md")
            assert await _settle(
                pilot, lambda: bool(stream.requests)
            ), "the provider was never called"
            await drain(pilot, cycles=20)

            sent = _sent_to_model(stream)
            assert "MARKER_README_913" in sent, "the file body never reached the model"
            assert REFERENCE_BLOCK_OPEN in sent, "the body was not wrapped in a reference block"
            # The operator's own sentence still leads: the block is appended,
            # never substituted in place.
            assert "summarise @README.md" in sent, "the typed sentence was lost"

            painted = transcript_text(app)
            assert "summarise @README.md" in painted, (
                "the transcript never showed the typed line, so the absence "
                f"below proves nothing; screen was: {painted!r}"
            )
            assert "MARKER_README_913" not in painted, (
                "the file body leaked into the transcript row — the row is "
                "supposed to stay the typed line"
            )
    finally:
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_sensitive_env_mounts_a_real_card_and_a_denial_keeps_the_secret_out(
    tmp_path: Path,
) -> None:
    """A sensitive in-workspace ``.env`` escalates to the REAL card, and a
    denial keeps the secret off the wire.

    ``.env`` is inside the workspace, so the ``inside and not sensitive``
    short-circuit in ``_approved`` is the only thing standing between it and an
    unasked read. This is the test that a ``FakeSession`` cannot write: the gate
    it would record is not the gate the app installs.

    The absence assertion is guarded by a presence one — the turn must still have
    run — so "the secret is not in the payload" cannot pass on a payload that
    was never sent.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    (workspace / ".env").write_text(ENV_BODY, encoding="utf-8")
    stream = ScriptedStream([text_turn("Sure.")])
    session = _armed_session(tmp_path / "session", stream, workspace)
    session.set_conversation_name("at-realpath-denied", user_set=True)
    app = OperatorApp(lambda: _factory(session))
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await wait_for_adoption(app, pilot)
            await drain(pilot)
            editor = app.query_one(Editor)
            editor.focus()
            await pilot.pause()
            editor.load_text("what is in @.env ?")
            editor.move_cursor(editor._end_of_buffer())
            await pilot.pause()
            if editor.picker.is_open():
                await pilot.press("escape")
                await pilot.pause()
            await pilot.press("enter")

            # The card the deny-list escalation must mount. Bounded in loop
            # turns: a gate that never fires fails here by NAME rather than
            # hanging a suite with no timeout to stop it.
            mounted = await _settle(pilot, lambda: bool(app.query(ApprovalPrompt)))
            assert mounted, (
                "no approval card mounted for a sensitive in-workspace .env — "
                "the deny-list escalation did not reach the app's gate"
            )
            await pilot.press("n")  # deny
            settled = await _settle(pilot, lambda: bool(stream.requests))
            await drain(pilot, cycles=20)

            assert settled, "the turn never ran after the denial, so nothing was proven"
            sent = _sent_to_model(stream)
            assert "MARKER_ENV_777" not in sent, "the denied secret reached the model"
            # A denial degrades to the verbatim token plus a notice, never to a
            # swallowed request: the user's question still goes.
            assert ".env" in sent, "the denied token did not degrade to its verbatim form"
    finally:
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_aside_declining_gate_refuses_a_sensitive_path_and_still_settles(
    tmp_path: Path,
) -> None:
    """The aside's DECLINING gate, on a real session: refuses the secret, and
    the aside still answers.

    ``_aside_worker`` expands under a gate that refuses rather than asks,
    because asking would cancel this very worker — ``request_tool_approval`` →
    ``_close_aside`` → cancels the ``aside`` worker group, which is where the
    expansion awaits. If that ruling breaks the symptom is a HANG or a silently
    discarded question, so the load-bearing assertion is that the aside's model
    call happened at all.

    The settled answer is read back through ``fork_messages``, which also pins
    the display/sent split on this surface: the panel keeps the TYPED question,
    not the expanded text handed to the model.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    (workspace / ".env").write_text(ENV_BODY, encoding="utf-8")
    (workspace / "auth.py").write_text(AUTH_BODY, encoding="utf-8")
    stream = ScriptedStream([text_turn("Aside answer.")])
    session = build_session(tmp_path / "session", stream, cwd=workspace)
    session.set_conversation_name("at-realpath-aside", user_set=True)
    app = OperatorApp(lambda: _factory(session))
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await wait_for_adoption(app, pilot)
            await drain(pilot)
            await _type_and_submit(pilot, app, "/btw what is in @.env ?")
            panel = app._aside_panel()
            settled = await _settle(
                pilot, lambda: panel is not None and bool(panel.fork_messages())
            )
            await drain(pilot, cycles=20)

            assert settled, (
                "the aside never settled — a self-cancelled worker discards the "
                "question in silence, which reads as a flake rather than a denial"
            )
            assert stream.requests, "the aside never reached the model (self-cancelled?)"
            sent = _sent_to_model(stream)
            assert "MARKER_ENV_777" not in sent, "the declined secret reached the model"
            assert ".env" in sent, "the declined token did not degrade to its verbatim form"
            # The TYPED question is what a fork would adopt, and the expanded
            # payload is not: `_aside_worker` expands only the text it hands the
            # model.
            assert panel is not None
            assert panel.fork_messages() == [
                ("what is in @.env ?", "Aside answer.")
            ], "the aside panel did not keep the typed question and the answer"
    finally:
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_paste_chip_beside_a_reference_delivers_both_payloads(tmp_path: Path) -> None:
    """A collapsed paste and an ``@path`` in one draft: both payloads go, and the
    chip does not.

    The two expansions are independent and ordered — ``on_editor_submitted``
    splices pastes into the REQUEST, then ``Session.prompt`` expands references
    over the result. A regression in either half shows up here and nowhere
    else, because this is the only test that puts them in the same draft.

    Posted as an ``EditorSubmitted`` rather than typed: a paste chip is minted
    by the clipboard handler, and the property under test is what the submit
    handler does with one already in the buffer.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    (workspace / "auth.py").write_text(AUTH_BODY, encoding="utf-8")
    stream = ScriptedStream([text_turn("Compared.")])
    session = build_session(tmp_path / "session", stream, cwd=workspace)
    session.set_conversation_name("at-realpath-paste", user_set=True)
    app = OperatorApp(lambda: _factory(session))
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await wait_for_adoption(app, pilot)
            await drain(pilot)
            pasted = PastedText("PASTED_PAYLOAD_77\nline2\nline3", "[Paste #1, 3 lines]")
            app.post_message(
                EditorSubmitted(
                    "[Paste #1, 3 lines] compare with @auth.py", attachments={1: pasted}
                )
            )
            settled = await _settle(pilot, lambda: bool(stream.requests))
            await drain(pilot, cycles=20)

            assert settled, "the provider was never called"
            sent = _sent_to_model(stream)
            assert "PASTED_PAYLOAD_77" in sent, "the pasted payload was lost"
            assert "MARKER_AUTH_42" in sent, "the @path body was lost"
            assert (
                "[Paste #1, 3 lines]" not in sent
            ), "the literal chip was sent — the payload was never spliced in"
    finally:
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_picker_opens_in_file_mode_and_lists_the_workspace(tmp_path: Path) -> None:
    """A bare ``@`` opens the picker in FILE mode, listing the workspace.

    FILE is a fourth mode rather than a reuse of SKILL, and the distinction is
    observable in exactly this shape: ``README.md`` ranks despite being
    uppercase and ``src/`` despite carrying no skill evidence at all, both of
    which ``skill_suggestions``' lowercase-evidence gate would reject. Asserting
    the ROWS rather than only the mode is what makes that difference a fact.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    (workspace / "README.md").write_text(README_BODY, encoding="utf-8")
    (workspace / "auth.py").write_text(AUTH_BODY, encoding="utf-8")
    (workspace / "src").mkdir()
    stream = ScriptedStream([text_turn("ok")])
    session = build_session(tmp_path / "session", stream, cwd=workspace)
    session.set_conversation_name("at-realpath-picker", user_set=True)
    app = OperatorApp(lambda: _factory(session))
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await wait_for_adoption(app, pilot)
            await drain(pilot)
            editor = app.query_one(Editor)
            editor.focus()
            await pilot.pause()
            editor.load_text("look at @")
            editor.move_cursor(editor._end_of_buffer())

            opened = await _settle(pilot, lambda: editor.picker.is_open())
            assert opened, "the picker never opened on a trailing @"
            mode = editor.picker.mode
            assert mode is PickerMode.FILE, f"picker mode is {mode}, not FILE"
            rows = [name for name, _ in editor.picker.suggestions()]
            assert rows, "the picker opened but listed nothing"
            missing = {"README.md", "auth.py", "src/"} - set(rows)
            listed = " | ".join(rows)
            assert not missing, f"workspace entries not listed: {sorted(missing)} of {listed}"
    finally:
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_a_draft_ending_in_a_reference_SENDS_on_the_first_enter(tmp_path: Path) -> None:
    """THE HEADLINE FLOW, typed for real: `summarise @README.md`, Enter, sent.

    The regression this exists to catch shipped GREEN through a 43-test composer
    suite, because every one of those tests reached the submit path through a
    helper that pressed Escape first. Escape closed the very list whose Enter
    behaviour was broken, so no test ever asked the question a user asks: I have
    finished typing, Enter, did it go?

    It did not. A FILE completion inserts no trailing space (a path may continue,
    `@src/` being one keystroke from `@src/app.py`), so the token stayed open,
    the list re-opened on the name it had just completed, and every Enter
    re-completed the same row: measured on the base, `summarise @README.md` +
    Enter x3 left the buffer byte-identical with `requests=0` (QA round 1, Q-1).
    A draft ending in a reference could not be sent at all, by any number of
    presses, until the user pressed Escape or typed on.

    Asserted on the RECORDED REQUEST, so "the message went" is a fact about what
    the model was handed rather than about what the app believes it sent, and on
    the painted row, so the strip half is held to the same keystrokes.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    (workspace / "README.md").write_text(README_BODY, encoding="utf-8")
    stream = ScriptedStream([text_turn("Summarised.")])
    session = build_session(tmp_path / "session", stream, cwd=workspace)
    session.set_conversation_name("at-realpath-type-enter", user_set=True)
    app = OperatorApp(lambda: _factory(session))
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await wait_for_adoption(app, pilot)
            await drain(pilot)
            editor = app.query_one(Editor)
            await _type_keystrokes(pilot, app, "summarise @README.md")
            assert editor.picker.is_open(), (
                "fixture never reached the open file list, so this test would "
                "pass without ever exercising the Enter routing it is about"
            )

            await pilot.press("enter")
            settled = await _settle(pilot, lambda: bool(stream.requests))
            await drain(pilot, cycles=20)

            assert settled, (
                "the FIRST Enter after typing a complete reference sent nothing "
                "— the draft is unsendable and the list re-completes forever"
            )
            sent = _sent_to_model(stream)
            assert "MARKER_README_913" in sent, "the file body never reached the model"
            assert "summarise @README.md" in sent, "the typed sentence was lost"
            assert editor.text == "", "the buffer was not cleared, so it never submitted"
            painted = transcript_text(app)
            assert "summarise @README.md" in painted
            assert "MARKER_README_913" not in painted, "the body leaked into the row"
    finally:
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_an_unfinished_token_still_takes_the_second_enter_to_send(tmp_path: Path) -> None:
    """The mid-path rule survives: the FIRST Enter completes, the second sends.

    The fix for the Q-1 trap must not become the mis-send the original no-submit
    rule existed to prevent. `@READ` is not what the row says (`README.md`), so
    accepting it CHANGES the buffer and the keystroke stays a completion — that
    is the `@src/`-is-one-keystroke-from-`@src/app.py` case, and submitting a
    directory the user was still typing past has no undo once the turn is
    dispatched.

    Only a row the buffer ALREADY holds sends, which is what makes the trap's
    escape hatch and the mid-path rule two rules rather than a contradiction.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    (workspace / "README.md").write_text(README_BODY, encoding="utf-8")
    stream = ScriptedStream([text_turn("ok")])
    session = build_session(tmp_path / "session", stream, cwd=workspace)
    session.set_conversation_name("at-realpath-two-enter", user_set=True)
    app = OperatorApp(lambda: _factory(session))
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await wait_for_adoption(app, pilot)
            await drain(pilot)
            editor = app.query_one(Editor)
            await _type_keystrokes(pilot, app, "summarise @READ")
            assert editor.picker.is_open(), "fixture never opened the file list"

            await pilot.press("enter")
            await drain(pilot, cycles=20)
            assert not stream.requests, (
                "the first Enter SENT an unfinished token — the completion was "
                "skipped and a directory-ish path went to the model"
            )
            assert (
                editor.text == "summarise @README.md"
            ), f"the row was not completed into the buffer: {editor.text!r}"

            await pilot.press("enter")
            settled = await _settle(pilot, lambda: bool(stream.requests))
            assert settled, "the completed token still could not be sent"
            assert "MARKER_README_913" in _sent_to_model(stream)
    finally:
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_prose_naming_no_path_is_sent_verbatim_and_never_rewritten(tmp_path: Path) -> None:
    """`glab mr create --assignee @me`: no list, no rewrite, verbatim on the wire.

    The PR body names this exact sentence as a token that must pass through
    untouched, and on the WIRE it always did — the resolver calls `@me` prose.
    The COMPOSER disagreed: a subsequence matcher reached `README.md` from `me`,
    that kept the FILE list open, and an open list owns Enter, so Enter silently
    rewrote the operator's sentence into `--assignee @README.md` and sent
    nothing at all (QA round 1, Q-2). Measured on the base: `requests=0`, buffer
    mutated, second Enter also inert.

    Both halves are asserted because either alone is passable by accident — the
    picker must not open, AND the sentence must arrive as typed.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    (workspace / "README.md").write_text(README_BODY, encoding="utf-8")
    (workspace / "my file.txt").write_text("spaced\n", encoding="utf-8")
    stream = ScriptedStream([text_turn("ok")])
    session = build_session(tmp_path / "session", stream, cwd=workspace)
    session.set_conversation_name("at-realpath-prose", user_set=True)
    app = OperatorApp(lambda: _factory(session))
    draft = "run glab mr create --assignee @me"
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await wait_for_adoption(app, pilot)
            await drain(pilot)
            editor = app.query_one(Editor)
            await _type_keystrokes(pilot, app, draft)

            assert not editor.picker.is_open(), (
                "a subsequence row opened the FILE list for a token the resolver "
                "calls prose, which is what makes Enter rewrite it"
            )

            await pilot.press("enter")
            settled = await _settle(pilot, lambda: bool(stream.requests))
            assert settled, "the prose draft could not be sent"
            assert (
                editor.text != "run glab mr create --assignee @README.md"
            ), "the operator's prose was rewritten into a filename"
            sent = _sent_to_model(stream)
            assert draft in sent, "the sentence did not reach the model as typed"
            assert (
                "MARKER_README_913" not in sent
            ), "a file the operator never referenced reached the model"
    finally:
        await dispose_quietly(session)
