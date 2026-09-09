"""A prompt whose bind fails gives the text back instead of dropping it.

**Why this file exists.** Incident ``dea45f5bdae2`` (2026-09-08): a message
typed into a live viewer hit a busy/retiring runtime, the bind exhausted its
sync envelope, and ``RuntimeUnresponsiveError`` — a real ``ConnectionError``
whose text matched none of ``_RUNTIME_GONE_MARKERS`` — fell through the turn
worker's ladder to the generic arm, which printed a transient-looking notice
and DROPPED the text. The transcript proves it: the message never arrived and
the user retyped a variant four minutes later.

The defect was the SHAPE, not the instance: data preservation was gated on an
allowlist of failure prose, fail-open on the user's data. These tests pin the
inverted contract — restore unless delivery is positively known — and the one
assertion that makes the old shape structurally unable to return: an
UNRECOGNISED ``ConnectionError`` must restore too. A marker-based gate cannot
pass that test by construction.

Separation of concerns is pinned alongside: restoration is universal, the
NOTICE is per-case, and ``_go_cold()`` (a real side effect: dropping a live
binding) stays gated on the markers so a merely busy runtime keeps its
binding.

The object under test is the one `lop` builds — a real ``OperatorApp`` over a
real ``RemoteSession.cold`` with a minted id, in an isolated config dir —
driven through the real editor submit path, exactly as
``tests/unit/tui/test_cold_slash_binds.py`` drives the slash seam.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest
from textual import events

from local_operator.harness.types import ImageContent
from local_operator.session.remote import (
    _SYNC_UNRESPONSIVE_REASON,
    RemoteSession,
    RuntimeUnresponsiveError,
)

try:  # the contract constant landed WITH the fix
    from local_operator.session.remote import _PROMPT_DELIVERED_ATTR
except ImportError:  # pre-fix tree: read the attribute by its documented name
    _PROMPT_DELIVERED_ATTR = "prompt_delivered"
from local_operator.session.runtime.launch import (
    ActionableConnectionError,
    RuntimeStartupError,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Attachment, Editor
from local_operator.tui.widgets.transcript import NoticeBlock
from tests.unit.tui.test_app_pilot import _transcript_text
from tests.unit.tui.test_cold_slash_binds import _cold_app

MESSAGE = "I downloaded the store version of the extension, can you try it"


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated config dir with a configured provider, as every `lop` has.

    Same shape as ``test_cold_slash_binds.isolated`` (defined locally rather
    than imported: a fixture imported from a sibling module collides with the
    ``isolated`` PARAMETER every test here takes, which flake8 rightly flags).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "sessions").mkdir()
    (tmp_path / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
    )
    return tmp_path


def _refuse_engages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the mount engage from spawning a real runtime process.

    The rig boots a cold viewer against a config that looks configured, so the
    mount engage would call the REAL ``engage_runtime``. Refusing instantly
    keeps the facade cold and the test fast; the failure is silent by design
    (``_start_runtime_engage`` logs and clears its latch), which is the state
    a real `lop` is in during its first engage window anyway.
    """

    async def _refuse(*args: Any, **kwargs: Any) -> None:
        raise ConnectionError("owner socket unreachable: [Errno 61] Connect call failed")

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", _refuse)


async def _boot(app: OperatorApp, pilot: Any) -> RemoteSession:
    for _ in range(200):
        await pilot.pause()
        if app._session is not None:
            break
    session = app._session
    assert isinstance(session, RemoteSession)
    assert session.is_cold and session.session_id, "the shape lop boots into"
    return session


async def _until(pilot: Any, predicate: Any, *, timeout: float = 10.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        await pilot.pause()
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return False


async def _type_and_enter(app: OperatorApp, pilot: Any, line: str) -> None:
    """The real submit: one paste event, then Enter, no typing in between."""
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    app.post_message(events.Paste(line))
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


def _fail_the_bind_with(session: RemoteSession, error: BaseException) -> None:
    """The incident seam: the bind raises inside ``RemoteSession.prompt``.

    ``prompt()`` calls ``_ensure_bound()`` first, so a bind that cannot
    complete raises exactly here — off the wire, before any frame exists.
    """

    async def failing(*args: Any, **kwargs: Any) -> None:
        raise error

    session._ensure_bound = failing  # type: ignore[method-assign]


def _notices(app: OperatorApp) -> list[str]:
    return [block.text() for block in app.query(NoticeBlock)]


def _one_line(text: str) -> str:
    """Collapse a transcript read, because `_transcript_text` hard-wraps rows.

    An assertion like ``"back in the composer" in text`` must survive the wrap
    the terminal width imposes; joining on whitespace makes the check about
    WORDS, not line breaks.
    """
    return " ".join(text.split())


async def _settle(app: OperatorApp, pilot: Any) -> None:
    settled = await _until(pilot, lambda: not app._interaction.active_workers and _notices(app))
    assert settled, f"the turn worker never settled; transcript:\n{_transcript_text(app)}"


def _cold_rig(isolated: Path) -> OperatorApp:
    return _cold_app(isolated, uuid.uuid4().hex[:12])


@pytest.mark.asyncio
async def test_the_incident_prompt_comes_back_to_the_composer(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact incident: a busy runtime eats the bind, not the message.

    This is the test that would have caught ``dea45f5bdae2`` — it FAILS on the
    pre-fix tree, where the composer ends empty and the only trace of the
    message is a "the runtime is not responding" notice that reads as status,
    not as loss.
    """
    _refuse_engages(monkeypatch)
    app = _cold_rig(isolated)
    async with app.run_test(size=(110, 30)) as pilot:
        session = await _boot(app, pilot)
        _fail_the_bind_with(session, RuntimeUnresponsiveError(_SYNC_UNRESPONSIVE_REASON))
        await _type_and_enter(app, pilot, MESSAGE)
        await _settle(app, pilot)

        assert (
            app.query_one(Editor).text == MESSAGE
        ), "the typed text must be back in the composer, one keystroke from resend"
        text = _one_line(_transcript_text(app))
        assert "your message is back in the composer" in text, text
        assert "it is still running" in text, "the runtime IS alive — a busy owner is not a crash"


@pytest.mark.asyncio
async def test_an_unrecognised_connection_error_still_restores(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE ANTI-ROT TEST. Wording nobody has written yet must not eat text.

    This is the assertion the old allowlist shape cannot pass: it restores on
    failure UNLESS delivery is positively known, so a reworded transport
    message, a new error class, or an exotic ``OSError`` all preserve the
    user's data by default. If this test fails, someone has re-gated data
    preservation on failure prose.
    """
    _refuse_engages(monkeypatch)
    app = _cold_rig(isolated)
    async with app.run_test(size=(110, 30)) as pilot:
        session = await _boot(app, pilot)
        _fail_the_bind_with(session, ConnectionError("beam flew off the cantilever at 03:14"))
        await _type_and_enter(app, pilot, MESSAGE)
        await _settle(app, pilot)

        assert app.query_one(Editor).text == MESSAGE
        text = _one_line(_transcript_text(app))
        assert (
            "beam flew off the cantilever at 03:14" in text
        ), "the raw reason is still relayed — honesty about the failure is not lost"
        assert "your message is back in the composer" in text, text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "label"),
    [
        (RuntimeUnresponsiveError(_SYNC_UNRESPONSIVE_REASON), "busy runtime"),
        (
            ConnectionError(
                "owner socket unreachable: [Errno 61] Connect call failed " "('127.0.0.1', 51753)"
            ),
            "dead runtime (marker match)",
        ),
        (ConnectionError("this session was stopped"), "stopped session"),
        (ConnectionError("some wording nobody has written yet"), "unrecognised"),
        (ConnectionError("owner closed the connection mid-sync"), "marker: closed"),
        (ActionableConnectionError("no API key is configured for test"), "actionable"),
        (RuntimeStartupError("runtime exited before its first sync"), "startup failure"),
    ],
)
async def test_every_failure_this_path_can_produce_restores_the_text(
    isolated: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException, label: str
) -> None:
    """Restoration is UNIVERSAL; only the notice is per-case.

    The sweep is the design's contract stated as a property: whatever raises
    out of ``session.prompt()`` before the ACK boundary, the composer ends
    holding the user's words. Copy assertions live in the per-case tests; a
    failure here is a data-loss regression, not a wording one.
    """
    _refuse_engages(monkeypatch)
    app = _cold_rig(isolated)
    async with app.run_test(size=(110, 30)) as pilot:
        session = await _boot(app, pilot)
        if label == "stopped session":
            app._stopped_session_id = session.session_id
        _fail_the_bind_with(session, error)
        await _type_and_enter(app, pilot, MESSAGE)
        await _settle(app, pilot)

        assert app.query_one(Editor).text == MESSAGE, f"{label}: text was lost"
        assert not app._interaction.unsent, f"{label}: nothing should need the overflow queue"


@pytest.mark.asyncio
async def test_a_delivered_prompt_does_not_restore(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one case where restoring would be wrong: the ACK already came.

    ``_PROMPT_DELIVERED_ATTR`` is the positive delivery assertion set past the
    ACK boundary in ``RemoteSession.prompt``. A failure carrying it belongs to
    a prompt the session already owns; putting the text back would invite a
    double send. No live production raise sets it today (the post-ACK slice of
    ``prompt()`` is empty) — this pins the gate so the first one that appears
    behaves correctly.
    """
    _refuse_engages(monkeypatch)
    app = _cold_rig(isolated)
    delivered = ConnectionError("post-ack bookkeeping failed")
    setattr(delivered, _PROMPT_DELIVERED_ATTR, True)

    async def delivered_then_failed(*args: Any, **kwargs: Any) -> None:
        raise delivered

    async with app.run_test(size=(110, 30)) as pilot:
        session = await _boot(app, pilot)
        session.prompt = delivered_then_failed  # type: ignore[method-assign]
        await _type_and_enter(app, pilot, MESSAGE)
        await _settle(app, pilot)

        assert app.query_one(Editor).text == "", "a delivered prompt must not come back"
        assert not app._interaction.unsent
        assert "back in the composer" not in _transcript_text(
            app
        ), "the notice must not claim a restore that did not happen"


@pytest.mark.asyncio
async def test_a_dead_runtime_drops_the_binding_a_busy_one_keeps_it(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The separation: markers gate ``_go_cold()``, never the user's data.

    Both failures restore. Only the genuinely dead runtime may drop the live
    binding — firing ``_go_cold()`` for a busy owner would throw away a
    binding to a runtime that is still serving the session. This is the test
    that stops a future "simplification" from re-merging the two questions.
    """
    from local_operator.tui.app import _is_runtime_gone

    # Sanity on the predicate the separation depends on, at the source.
    assert _is_runtime_gone(
        ConnectionError(
            "owner socket unreachable: [Errno 61] Connect call failed ('127.0.0.1', 51753)"
        )
    )
    assert not _is_runtime_gone(RuntimeUnresponsiveError(_SYNC_UNRESPONSIVE_REASON))

    for error, expect_cold in (
        (
            ConnectionError(
                "owner socket unreachable: [Errno 61] Connect call failed " "('127.0.0.1', 51753)"
            ),
            True,
        ),
        (RuntimeUnresponsiveError(_SYNC_UNRESPONSIVE_REASON), False),
    ):
        _refuse_engages(monkeypatch)
        app = _cold_rig(isolated)
        async with app.run_test(size=(110, 30)) as pilot:
            session = await _boot(app, pilot)
            went_cold: list[bool] = []
            real_go_cold = session._go_cold

            def spy_cold(*args: Any, **kwargs: Any) -> Any:
                went_cold.append(True)
                return real_go_cold(*args, **kwargs)

            session._go_cold = spy_cold  # type: ignore[method-assign]
            _fail_the_bind_with(session, error)
            await _type_and_enter(app, pilot, MESSAGE)
            await _settle(app, pilot)

            assert app.query_one(Editor).text == MESSAGE, "both cases restore the text"
            assert bool(went_cold) is expect_cold, (
                f"`_go_cold` fired={bool(went_cold)} for {type(error).__name__}; "
                "a busy runtime must keep its binding"
            )


@pytest.mark.asyncio
async def test_a_stopped_session_notice_names_the_unsent_text(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stopped arm keeps its own copy — and now its own restore too."""
    _refuse_engages(monkeypatch)
    app = _cold_rig(isolated)
    async with app.run_test(size=(110, 30)) as pilot:
        session = await _boot(app, pilot)
        app._stopped_session_id = session.session_id
        _fail_the_bind_with(session, ConnectionError("this session was stopped"))
        await _type_and_enter(app, pilot, MESSAGE)
        await _settle(app, pilot)

        assert app.query_one(Editor).text == MESSAGE
        text = _one_line(_transcript_text(app))
        assert "this session was stopped — your message was not sent" in text, text
        assert "/resume" in text, "the way back is named"


@pytest.mark.asyncio
async def test_a_dead_runtime_notice_points_at_the_resend(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The marker arm's copy: stopped runtime, text in the composer, resend."""
    _refuse_engages(monkeypatch)
    app = _cold_rig(isolated)
    async with app.run_test(size=(110, 30)) as pilot:
        session = await _boot(app, pilot)
        _fail_the_bind_with(
            session,
            ConnectionError(
                "owner socket unreachable: [Errno 61] Connect call failed " "('127.0.0.1', 51753)"
            ),
        )
        await _type_and_enter(app, pilot, MESSAGE)
        await _settle(app, pilot)

        text = _one_line(_transcript_text(app))
        assert "this session's runtime stopped" in text, text
        assert "your message is back in the composer" in text, text
        assert "send it again to start a new one" in text, text


@pytest.mark.asyncio
async def test_images_come_back_with_their_markers(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An image-bearing prompt restores its attachment, not just its words.

    With no ``accepted`` draft, the restore re-synthesises the ``[Image #N]``
    citation from the marker indices the text already carries, so the picture
    comes back cited rather than orphaned.
    """
    _refuse_engages(monkeypatch)
    app = _cold_rig(isolated)
    image = ImageContent(data="aGVsbG8=", mime_type="image/png")
    async with app.run_test(size=(110, 30)) as pilot:
        session = await _boot(app, pilot)
        _fail_the_bind_with(session, RuntimeUnresponsiveError(_SYNC_UNRESPONSIVE_REASON))
        app._submit_prompt("look at this", images=[image])
        await _settle(app, pilot)

        editor = app.query_one(Editor)
        assert editor.text == "look at this\n[Image #1]", repr(editor.text)
        attachments = editor.attachments()
        attachment = attachments.get(1)
        # `attachments()` is the `Marked` union; the runtime check is the
        # assertion that a pasted IMAGE came back, not a collapsed text.
        assert isinstance(attachment, Attachment), "the restored citation must be the image"
        assert attachment.image is image, "and still point at the pasted bytes"


@pytest.mark.asyncio
async def test_a_collapsed_paste_restores_as_the_chip_line(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``accepted`` rides the restore: the chip line, not the expanded payload.

    The hazard ``_submit_prompt`` records: a collapsed paste has the EXPANDED
    text as its payload; restoring that instead of the chip would make the
    user delete by hand exactly what the collapse existed to spare them. The
    original attachment map must survive verbatim too.
    """
    from local_operator.tui.session_interaction import SessionDraft

    _refuse_engages(monkeypatch)
    app = _cold_rig(isolated)
    image = ImageContent(data="aGVsbG8=", mime_type="image/png")
    # The chip carries its citation: `adopt_attachments` keeps only markers
    # the restored text cites, and a real collapsed paste keeps them on the
    # typed line while the payload below carries the expanded bytes.
    chip = "the log line [Image #1]"
    expanded = f"{chip}\n[500 lines of pasted payload]"
    accepted = SessionDraft(text=chip, attachments={1: Attachment(image, "[Image #1]")})
    async with app.run_test(size=(110, 30)) as pilot:
        session = await _boot(app, pilot)
        _fail_the_bind_with(session, RuntimeUnresponsiveError(_SYNC_UNRESPONSIVE_REASON))
        source = app._interaction
        # The same hand-off the editor's submit path makes (``submitted_draft``
        # → ``_start_turn_for(accepted=...)``).
        source.turn.submitted_draft = accepted
        app._submit_prompt(expanded, images=[image])
        await _settle(app, pilot)

        editor = app.query_one(Editor)
        assert editor.text == chip, repr(editor.text)
        restored = editor.attachments().get(1)
        assert isinstance(restored, Attachment)
        assert restored.marker == "[Image #1]"
        assert restored.image is image, "the attachment map round-trips verbatim"


@pytest.mark.asyncio
async def test_a_failed_foreground_bind_does_not_leave_connecting_on_the_splash(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cold splash stops claiming progress after the attempt ended.

    The stuck-`connecting…` half of the incident: after a failed bind nothing
    is in flight, the band has already stopped saying `starting…`, and the
    splash is the one surface still promising work. It must answer with the
    failure's own sentence (the one `_bind_then_dispatch` already ships) and
    a model row that says `not connected`.
    """
    from local_operator.tui.widgets.welcome import WelcomeView

    _refuse_engages(monkeypatch)
    app = _cold_rig(isolated)
    async with app.run_test(size=(110, 30)) as pilot:
        session = await _boot(app, pilot)
        _fail_the_bind_with(session, RuntimeUnresponsiveError(_SYNC_UNRESPONSIVE_REASON))
        app._bind_then_dispatch("/goal test the fix")
        appeared = await _until(pilot, lambda: app._splash_unbound)
        assert appeared, "the failed engage must mark the splash"

        assert app._splash_notice == (
            "could not reach this session's runtime in time — it is still "
            "running; try that again in a moment"
        ), app._splash_notice
        welcome = app.query_one(WelcomeView)
        welcome.refresh_info()
        assert welcome._info.unbound, "the splash's info snapshot carries the state"
        assert welcome._info.notice == app._splash_notice


def test_the_model_row_ladder_says_not_connected_only_when_unbound() -> None:
    """The pure ladder: pending by default, setup first, unbound over pending.

    Pinned at the function level because the word is chosen at render time —
    a widget-level assertion would only see it after a repaint, and the ladder
    is where a regression would live.
    """
    from local_operator.tui.widgets.welcome import (
        _PRIORITY_MODEL,
        WelcomeInfo,
        _status_rows,
    )

    def model_word(info: WelcomeInfo) -> str:
        # The model row is the one ladder row at `_PRIORITY_MODEL`; its plain
        # text is the word the splash paints.
        words = [t.plain for p, t in _status_rows(info, 80) if p == _PRIORITY_MODEL]
        assert words, "the ladder always emits a model row"
        return words[-1]

    assert model_word(WelcomeInfo()) == "connecting…"
    assert model_word(WelcomeInfo(setup=True)) == "setup"
    assert model_word(WelcomeInfo(unbound=True)) == "not connected"
    # A failed engage outranks the cold facade's PROVISIONAL label: that label
    # is the viewer's own copy from local config, not state an owner
    # confirmed, and after a failed bind it must not read as connected. The
    # band takes the same word, per the two-surfaces-one-word rule (D10's
    # shape, applied to state).
    assert model_word(WelcomeInfo(model_label="test/mock", unbound=True)) == "not connected"
    # With no failure the label wins as it always has (D10 proper: the splash
    # names the model the band names).
    assert model_word(WelcomeInfo(model_label="test/mock")) == "test/mock"
