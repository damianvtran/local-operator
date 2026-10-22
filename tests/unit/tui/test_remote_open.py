"""Slice V's TUI surfaces: open a peer's session, `/move --to`, remote lifecycle.

The mesh itself is proven over real relays in ``tests/unit/network/
test_remote_viewer.py`` (prompt/rename/slash/stop landing on the peer's runtime)
and ``test_mobility*.py`` (the move protocol). What these pin is the APP's half —
that the real ``OperatorApp`` (the one that loads the stylesheet) routes each act
to the mesh seam, keeps its local behaviour byte-identical when nothing is remote,
and says what happened in the design's words.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network.mobility import MOVE_RESULT_PHASES
from local_operator.resume import SessionRow
from local_operator.session.catalog import CatalogEntry
from local_operator.session.runtime.registry import scan as record_scan
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_move import (
    AMBIGUOUS_MOVE,
    MOVE_PHASE_ORDER,
    MoveTo,
    parse_move_to,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory

# ---------------------------------------------------------------------------
# the grammar
# ---------------------------------------------------------------------------


def test_the_phase_order_is_the_contracts() -> None:
    assert MOVE_PHASE_ORDER == tuple(MOVE_RESULT_PHASES)


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        # No `--to`: today's path form, untouched — even an id-shaped word.
        ("", None),
        ("~/src", None),
        ("9f3ac1e0b7d2", None),
        ("dir with spaces", None),
        # The mobility forms.
        ("--to pixel-8", MoveTo(to="pixel-8")),
        ("9f3ac1e0b7d2 --to local", MoveTo(session_id="9f3ac1e0b7d2", to="local")),
        ("--to=build-box --keep", MoveTo(to="build-box", keep=True)),
        ("--keep 9f3ac1e0b7d2 --to local", MoveTo("9f3ac1e0b7d2", "local", True)),
        # The queued form (§5.4): a bare flag, anywhere, in either spelling —
        # and combinable with ``--keep``, which the CLI answers by ignoring the
        # queue (a copy retires nothing, so there is no safe point to wait for).
        ("--to pixel-8 --queue", MoveTo(to="pixel-8", queue=True)),
        ("--queue --to=build-box", MoveTo(to="build-box", queue=True)),
        ("--queue 9f3ac1e0b7d2 --to local", MoveTo("9f3ac1e0b7d2", "local", queue=True)),
        ("--queue --keep --to local", MoveTo(to="local", keep=True, queue=True)),
        # Refusals, each by name.
        ("--to", MoveTo(error="--to needs a device: /move --to <peer|local>")),
        ("~/src --to pixel-8", MoveTo(error=AMBIGUOUS_MOVE)),
        ("./x --to local", MoveTo(error=AMBIGUOUS_MOVE)),
        ("a b --to local", MoveTo(error=AMBIGUOUS_MOVE)),
        # LOH-9 S2: the role-definition opt-out is accepted anywhere, and the
        # "only ..." refusal names it so a reader is not told the flag does not
        # exist by an error that omits it.
        ("--to pixel-8 --drop-session-roles", MoveTo(to="pixel-8", drop_session_roles=True)),
        (
            "--drop-session-roles 9f3ac1e0b7d2 --to local",
            MoveTo("9f3ac1e0b7d2", "local", drop_session_roles=True),
        ),
        (
            "--to x --force",
            MoveTo(
                error=(
                    "/move --to takes only --keep, --queue and --drop-session-roles, "
                    "not '--force'"
                )
            ),
        ),
        # ``--wait`` is refused BY NAME, naming the shell route that can honour it:
        # the producer's busy sentence sends readers to a shell for this flag, and a
        # bare "unknown flag" here would be the papercut one surface over.
        (
            "--to x --wait 30",
            MoveTo(
                error=(
                    "this view cannot wait — from a shell: "
                    "`lop sessions move <id> --to <to> --wait <seconds>`"
                )
            ),
        ),
        (
            "--wait=30 --to x",
            MoveTo(
                error=(
                    "this view cannot wait — from a shell: "
                    "`lop sessions move <id> --to <to> --wait <seconds>`"
                )
            ),
        ),
    ],
)
def test_the_discriminant_is_the_presence_of_to(arg: str, expected: MoveTo | None) -> None:
    assert parse_move_to(arg) == expected


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _row(sid: str, *, remote: bool = False, reachable: bool = True) -> SessionRow:
    return SessionRow(
        sid,
        time.time(),
        f"Session {sid}",
        locality="remote" if remote else "",
        owner_device="d_peer" if remote else "",
        owner_device_name="pixel-8" if remote else "",
        reachable=reachable,
        unreachable_reason="" if reachable else "connect_failed:ConnectionRefusedError",
    )


def _notices(app: Any) -> list[str]:
    from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView

    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


def _no_resume(_id: str | None = None) -> Any:
    """A resume launcher that is present (so `/resume`/`/new` are offered) and inert."""
    return _factory(FakeSession())


def _recording(into: list[Any]) -> Any:
    def factory(_id: str | None = None) -> Any:
        into.append(_id)
        return _factory(FakeSession())

    return factory


async def _booted(pilot: Any, app: OperatorApp) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


# ---------------------------------------------------------------------------
# opening a peer's session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_picking_a_peer_row_opens_it_instead_of_announcing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding 2: the pick OPENS the session through the remote seam."""
    opened: list[str] = []
    monkeypatch.setattr(
        "local_operator.session.remote_open.remote_row_for",
        lambda sid, root: _row(sid, remote=True) if sid == "peer1" else None,
    )

    async def fake_open(session_id: str, **kwargs: Any) -> Any:
        opened.append(session_id)
        return FakeSession()

    monkeypatch.setattr("local_operator.session.remote_open.open_remote_viewer", fake_open)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        adopted: list[Any] = []

        async def record(remote: Any) -> None:
            adopted.append(remote)

        app._adopt_built_viewer = record  # type: ignore[method-assign]
        from local_operator.tui.widgets.session_sidebar import SessionSidebar

        app.post_message(SessionSidebar.Selected("peer1"))
        for _ in range(20):
            await pilot.pause()
            if adopted:
                break
        assert opened == ["peer1"]
        assert adopted, "the peer's viewer was never adopted"
        shown = " ".join(_notices(app))
        assert "opened peer1 on pixel-8" in shown, shown
        assert "--engage warms it" not in shown, "the old announce-only sentence is back"
        # Nothing local was started for a peer's id.
        assert app._sidebar_navigation.requested_id == ""


@pytest.mark.asyncio
async def test_an_unreachable_peer_row_is_refused_with_its_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "local_operator.session.remote_open.remote_row_for",
        lambda sid, root: _row(sid, remote=True, reachable=False),
    )
    opened: list[str] = []

    async def fake_open(session_id: str, **kwargs: Any) -> Any:
        opened.append(session_id)
        return None

    monkeypatch.setattr("local_operator.session.remote_open.open_remote_viewer", fake_open)
    launched: list[Any] = []
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_recording(launched))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        app._run_slash_command("/resume peer1")
        await pilot.pause()
        await pilot.pause()
        shown = " ".join(_notices(app))
        assert "peer1 is on pixel-8, which is unreachable" in shown, shown
        assert "ConnectionRefusedError" not in shown, "the raw token reached the user"
        assert "/network doctor pixel-8" in shown, shown
        assert opened == []
    assert launched == [], "a peer's id reached the LOCAL resume factory"


@pytest.mark.asyncio
async def test_ctrl_shift_down_traverses_past_a_peer_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review r9 MINOR 1: local → peer → local must be walkable by the shortcut.

    The pick of the peer row is the refusal/open arm; the next press must step
    from IT, not from the attached local row before it (which recomputed the same
    peer row forever).
    """
    monkeypatch.setattr(
        "local_operator.session.remote_open.remote_row_for",
        lambda sid, root: _row(sid, remote=True, reachable=False) if sid == "peer" else None,
    )
    entries = [
        CatalogEntry(_row("aaaa")),
        CatalogEntry(_row("peer", remote=True, reachable=False)),
        CatalogEntry(_row("bbbb")),
    ]
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        picked: list[str] = []
        real_select = app._select_sidebar_session

        def spy(session_id: str) -> Any:
            picked.append(session_id)
            if session_id == "peer":
                return real_select(session_id)
            return None

        app._select_sidebar_session = spy  # type: ignore[method-assign]
        monkeypatch.setattr(type(app._session), "session_id", "aaaa", raising=False)
        app._sidebar_navigation.intend("")
        app._switch_session_from(entries, 1)
        await pilot.pause()
        await pilot.pause()
        app._switch_session_from(entries, 1)
        await pilot.pause()
        assert picked == ["peer", "bbbb"], picked


# ---------------------------------------------------------------------------
# quitting never ends a remote runtime
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_leaving_a_remote_session_never_offers_its_runtime_back() -> None:
    """Cell 1.2: the TUI's pristine offer is skipped for a runtime on another device."""
    asked: list[str] = []

    class Remote(FakeSession):
        runtime_locality = "another-machine"

        async def retire_if_unused(self) -> str:
            asked.append("remote")
            return "retired"

    class Local(FakeSession):
        runtime_locality = "this-machine"

        async def retire_if_unused(self) -> str:
            asked.append("local")
            return "kept"

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        await app._retire_unused_runtime(Remote())
        await app._retire_unused_runtime(Local())
    assert asked == ["local"], "a remote runtime was offered back on leave"


# ---------------------------------------------------------------------------
# /move --to
# ---------------------------------------------------------------------------


_COMMITTED = {
    "ok": True,
    "session_id": "sess1",
    "new_session_id": "sess1",
    "mode": "move",
    "from_device": {"device_id": "d_me", "name": "laptop"},
    "to_device": {"device_id": "d_peer", "name": "pixel-8"},
    "phase": "done",
    "phases": [{"phase": p, "at": 0.0} for p in MOVE_PHASE_ORDER],
}


@pytest.mark.asyncio
async def test_moving_the_current_session_leaves_it_first_then_reopens_it_remote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding 9: detach → commit → reopen attached-remote, in that order.

    ``_leave_for_move`` is stubbed so the ORDER is observable. The REOPEN is not:
    it runs the app's own ``_select_sidebar_session`` → ``SessionNavigation``, and
    the observable is that navigation's own ``committed_id``, plus the source the
    prepare obtained for the id. Round 1 pinned this over a stub of both ends
    (MINOR 2), which asserted the order of calls the test itself made: a reopen
    naming the wrong id, or none at all, would have passed.
    """
    order: list[str] = []

    def fake_move(
        session_id: str,
        to: str,
        *,
        keep: bool = False,
        queue: bool = False,
        drop_session_roles: bool = False,
    ) -> dict[str, Any]:
        order.append(f"move {session_id} {to} keep={keep}")
        return dict(_COMMITTED)

    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake_move)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        monkeypatch.setattr(type(app._session), "session_id", "sess1", raising=False)

        async def leave() -> None:
            order.append("leave")

        app._leave_for_move = leave  # type: ignore[method-assign]
        app._run_slash_command("/move --to pixel-8")
        for _ in range(30):
            await pilot.pause()
            if app._sidebar_navigation.committed_id == "sess1":
                break
        assert order == ["leave", "move sess1 pixel-8 keep=False"], order
        # THE REOPEN, THROUGH THE APP'S OWN NAVIGATION.
        assert app._sidebar_navigation.committed_id == "sess1"
        shown = " ".join(_notices(app))
        assert "✓ prepared  ✓ handing off  ✓ committed  ✓ done" in shown, shown
        assert "Moved sess1 to pixel-8. It runs there now." in shown, shown
        # U10: a finished move rides the SAME muted ink as the queued receipt
        # for the same request — `info`'s dim token made them two event classes.
        from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView

        finished = next(
            block
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, NoticeBlock) and "Moved sess1 to pixel-8" in block.text()
        )
        # RENDERED ink, not the private token (round 3): read the object the
        # transcript paints and compare its colour with the theme's `muted`.
        assert _rendered_in_muted(finished), finished.renderable


#: The five refusals the move contract can produce AFTER the current session has
#: already been left, as ``(code, message, phase_reached, changed, tail)``. The
#: shapes are the ones ``network.mobility`` builds: ``deadline_exceeded`` is the
#: one that arrives with ``changed=True`` (``_move_refusal``'s ``changed=bool(
#: reached and reached != "prepared")``), and its own message says the handoff was
#: already committed — which is why a tail claiming "it did not move" contradicted
#: the same sentence (round 2, R2-3).
_MOVE_REFUSALS: tuple[tuple[str, str, str | None, bool, str], ...] = (
    (
        "busy",
        "This session is working right now — try again when the turn finishes",
        None,
        False,
        " You are back on sess1 — it did not move.",
    ),
    (
        "digest_mismatch",
        "the copy did not verify; it was rolled back",
        "handing_off",
        False,
        # U7: a refusal that REACHED a phase carries the producer's sentence and
        # no placement tail — the copy failure's own wording owns the state now.
        "",
    ),
    (
        "relay_unavailable",
        "this device's relay is not running",
        None,
        False,
        " You are back on sess1 — it did not move.",
    ),
    (
        "session_unreachable",
        "the session's runtime did not answer",
        None,
        False,
        " You are back on sess1 — it did not move.",
    ),
    (
        "deadline_exceeded",
        "pixel-8 did not finish taking sess1 in time. This device has already "
        "committed the handoff, so ask again rather than retrying from scratch",
        "committed",
        True,
        " You are back on sess1; this move may already have gone through, so check "
        "before asking again.",
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "message", "phase_reached", "changed", "tail"),
    _MOVE_REFUSALS,
    ids=[cell[0] for cell in _MOVE_REFUSALS],
)
async def test_a_refused_move_of_the_current_session_puts_the_user_back_on_it(
    monkeypatch: pytest.MonkeyPatch,
    code: str,
    message: str,
    phase_reached: str | None,
    changed: bool,
    tail: str,
) -> None:
    """V3: leaving first is right; every refusal after it owes a way back — TRUE about the state.

    The ordered pair is genuinely required — an attached viewer blocks the owner's
    exclusive retire — so the CURRENT session is left before the CLI runs. Then
    every refusal the CLI produces returned without reopening it, because
    ``_publish_move_result`` reopened only on success. Measured against the real
    app on round 1's head: ``order == ['leave', 'move …']``, ``reopened == []``,
    the app sitting on the fresh local replacement — while the receipt said
    "Nothing changed", which is false about a screen the user can see.

    ALL FIVE CODES ARE CELLS (round 2, R2-3 asked for the ``changed=True`` one): the
    way back has to hold for each, and the tail has to be true about each. A refusal
    that arrives with ``changed=True`` says the move may already have gone through,
    because ``network.mobility`` says exactly that in the message the same receipt
    carries — a tail claiming "it did not move" would contradict it in one sentence.

    Nothing on either side of the sequence is stubbed: the real ``_leave_for_move``
    does the detaching and the real ``_select_sidebar_session`` is the way back, so
    the assertion is on the app's own navigation completing on the id it was on
    (``committed_id``), not on calls this test made itself.
    """

    def fake_move(
        session_id: str,
        to: str,
        *,
        keep: bool = False,
        queue: bool = False,
        drop_session_roles: bool = False,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "code": code,
            "message": message,
            "session_id": session_id,
            "phase_reached": phase_reached,
            "changed": changed,
        }

    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake_move)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        monkeypatch.setattr(type(app._session), "session_id", "sess1", raising=False)
        app._run_slash_command("/move --to pixel-8")
        for _ in range(40):
            await pilot.pause()
            if any("Could not move" in notice for notice in _notices(app)):
                break
        for _ in range(40):
            await pilot.pause()
            if app._sidebar_navigation.committed_id == "sess1":
                break
        shown = " ".join(_notices(app))
        # THE WAY BACK: the app's own navigation completed on the session the user
        # was in, so the refusal did not leave them on the fresh local
        # replacement. Before round 1's fix this id was never reached — the refusal
        # branch returned before the reopen, which only the success path had.
        assert app._sidebar_navigation.committed_id == "sess1"
        assert "sess1" in app._sidebar_sources, list(app._sidebar_sources)
        # AND THE SENTENCE IS TRUE ABOUT IT. "Nothing changed" is never true here:
        # locally the TUI did leave.
        assert f"Could not move sess1: {message}.{tail}" in shown, shown
        assert "Nothing changed." not in shown, shown
        if changed or not tail:
            # R3-1: when the expected tail is empty, the substring assert above is
            # satisfied by the prefix alone, so the absence of the old tail is the
            # half that actually proves it was dropped.
            assert "it did not move" not in shown, shown


@pytest.mark.asyncio
async def test_a_refused_move_says_so_and_reopens_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_move(
        session_id: str,
        to: str,
        *,
        keep: bool = False,
        queue: bool = False,
        drop_session_roles: bool = False,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "code": "busy",
            "message": "This session is working right now — try again when the turn finishes",
            "session_id": session_id,
            "phase_reached": None,
            "changed": False,
        }

    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake_move)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        reopened: list[str] = []
        app._select_sidebar_session = reopened.append  # type: ignore[method-assign]
        app._run_slash_command("/move other1 --to pixel-8")
        for _ in range(30):
            await pilot.pause()
            if any("Could not move" in n for n in _notices(app)):
                break
        shown = " ".join(_notices(app))
        assert (
            "Could not move other1: This session is working right now — try again when the "
            "turn finishes. Nothing changed." in shown
        ), shown
        assert reopened == []
        # A refusal before any phase is ONE sentence: the empty phase row is gone.
        assert not any(n.startswith("moving other1") for n in _notices(app)), _notices(app)


@pytest.mark.asyncio
async def test_a_busy_refusal_names_the_shell_route_its_wait_belongs_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The producer's busy sentence, rendered here, sends ``--wait`` to a shell.

    The papercut (2026-09-29): mid-turn ``/move … --to`` printed "…try again when
    the turn finishes, or pass --wait <seconds> to re-check" — and typing that
    flag at this composer refused it (the composer carries only ``--keep`` and
    ``--queue``), so
    the printed remedy was one this surface could not run. The refusal is the
    PRODUCER's sentence (``mobility._busy_sentence``) rendered verbatim, so the
    frame is where the fix has to hold: this cell feeds the real producer's
    output through the real app and asserts the rendered notice names the shell
    route.
    """
    from local_operator.network import mobility

    def fake_move(
        session_id: str,
        to: str,
        *,
        keep: bool = False,
        queue: bool = False,
        drop_session_roles: bool = False,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "code": "busy",
            "message": mobility._busy_sentence("busy"),
            "session_id": session_id,
            "phase_reached": None,
            "changed": False,
        }

    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake_move)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        app._run_slash_command("/move other1 --to pixel-8")
        for _ in range(30):
            await pilot.pause()
            if any("Could not move" in n for n in _notices(app)):
                break
        shown = " ".join(_notices(app))
        assert "Could not move other1:" in shown, shown
        # THE REMEDY NAMES ITS SURFACE, in the rendered sentence itself — and it is
        # the producer's words that arrive, not a composer rewrite of them.
        assert "from a shell" in shown, shown
        assert mobility._busy_sentence("busy") in shown, shown


@pytest.mark.asyncio
async def test_a_move_of_a_session_this_tui_holds_in_the_sidebar_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M's deferral, TUI half: this terminal's own parked viewer blocks the move.

    Refused BEFORE anything runs, naming THIS terminal; once the source is
    released the same command proceeds.
    """
    ran: list[str] = []

    def fake_move(
        session_id: str,
        to: str,
        *,
        keep: bool = False,
        queue: bool = False,
        drop_session_roles: bool = False,
    ) -> dict[str, Any]:
        ran.append(session_id)
        return {**_COMMITTED, "session_id": session_id, "new_session_id": session_id}

    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake_move)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        app._select_sidebar_session = lambda _sid: None  # type: ignore[method-assign]

        class Held:
            retired = False

        app._sidebar_sources["held1"] = Held()  # type: ignore[assignment]
        app._run_slash_command("/move held1 --to pixel-8")
        await pilot.pause()
        await pilot.pause()
        shown = " ".join(_notices(app))
        assert "this terminal is still holding it open in the sidebar" in shown, shown
        assert ran == []

        # RELEASED: the sidebar lets the source go, and the same move proceeds.
        app._sidebar_sources.pop("held1")
        app._run_slash_command("/move held1 --to pixel-8")
        for _ in range(30):
            await pilot.pause()
            if ran:
                break
        assert ran == ["held1"]


# ---------------------------------------------------------------------------
# the target is checked before the session is left (mesh-move-outcome, clause 2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_recall_from_an_unreachable_peer_never_leaves_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clause 2, the operator's own report: leave nothing, name the device.

    ``remote_row_and_silence`` answers the shape the sidebar's read really has:
    the row, carrying the reason it cannot answer. The app must refuse BEFORE
    the phase row and BEFORE ``_leave_for_move`` — that order IS the fix; the
    old order left first and asked after, which is what put the operator back
    on the same remote conversation.
    """
    monkeypatch.setattr(
        "local_operator.session.remote_open.remote_row_and_silence",
        lambda sid, root: (_row(sid, remote=True, reachable=False), ()),
    )
    ran: list[str] = []

    def fake_move(
        session_id: str,
        to: str,
        *,
        keep: bool = False,
        queue: bool = False,
        drop_session_roles: bool = False,
    ) -> dict[str, Any]:
        ran.append(f"move {session_id} {to}")
        return dict(_COMMITTED)

    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake_move)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        monkeypatch.setattr(type(app._session), "session_id", "sess1", raising=False)
        left: list[str] = []

        async def leave() -> None:
            left.append("left")

        app._leave_for_move = leave  # type: ignore[method-assign]
        app._run_slash_command("/move --to local")
        for _ in range(40):
            await pilot.pause()
            if any("unreachable" in notice for notice in _notices(app)):
                break
        shown = " ".join(_notices(app))
        assert left == [], "the session was left before the peer answered"
        assert ran == [], "the CLI was run for a target that cannot take it"
        assert "sess1 is on pixel-8, which is unreachable" in shown, shown
        # NOTHING CLAIMED THE MOVE STARTED: no phase row was ever painted.
        assert not any(
            notice.startswith(("requesting a move", "moving sess1")) for notice in _notices(app)
        ), _notices(app)
        # THE REFUSAL RELEASES THE LOCK: the "a move is already running" guard
        # is reset by the worker's own ``finally``, so the next request runs.
        app._run_slash_command("/move --to pixel-8")
        for _ in range(40):
            await pilot.pause()
            if ran:
                break
        assert ran == ["move sess1 pixel-8"], ran


@pytest.mark.asyncio
async def test_a_recall_where_a_device_stayed_silent_names_it_and_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resolution miss WITH silence is not "nobody holds it": name the device.

    The sentence reports what the devices did (did not answer), never asserts
    absence, and its tail is true about the screen: the user is still in the
    session because nothing was left.
    """
    from local_operator.session.peer_rows import UnansweredPeer

    monkeypatch.setattr(
        "local_operator.session.remote_open.remote_row_and_silence",
        lambda sid, root: (
            None,
            (UnansweredPeer("d_cloud", "cloud-node-1", "connect_failed"),),
        ),
    )
    ran: list[str] = []

    def fake_move(
        session_id: str,
        to: str,
        *,
        keep: bool = False,
        queue: bool = False,
        drop_session_roles: bool = False,
    ) -> dict[str, Any]:
        ran.append(f"move {session_id} {to}")
        return dict(_COMMITTED)

    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake_move)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        monkeypatch.setattr(type(app._session), "session_id", "sess1", raising=False)
        left: list[str] = []

        async def leave() -> None:
            left.append("left")

        app._leave_for_move = leave  # type: ignore[method-assign]
        app._run_slash_command("/move --to local")
        for _ in range(40):
            await pilot.pause()
            if any(
                "did not answer" in notice or "not answering" in notice for notice in _notices(app)
            ):
                break
        shown = " ".join(_notices(app))
        assert left == []
        assert ran == []
        assert (
            "cloud-node-1 is not answering right now, so sess1 could not be handed over. "
            "Nothing changed; you are still in it." in shown
        ), shown
        assert not any(
            notice.startswith(("requesting a move", "moving sess1")) for notice in _notices(app)
        ), _notices(app)


@pytest.mark.asyncio
async def test_a_recall_from_a_peer_that_answers_still_leaves_and_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-check is a GATE, not a new refusal: an answering peer proceeds.

    The leave still comes first (the attached viewer blocks the owner's retire),
    and the CLI still runs — the pre-check must not turn a working recall into a
    no-op.
    """
    monkeypatch.setattr(
        "local_operator.session.remote_open.remote_row_and_silence",
        lambda sid, root: (_row(sid, remote=True), ()),
    )
    order: list[str] = []

    def fake_move(
        session_id: str,
        to: str,
        *,
        keep: bool = False,
        queue: bool = False,
        drop_session_roles: bool = False,
    ) -> dict[str, Any]:
        order.append(f"move {session_id} {to}")
        return dict(_COMMITTED)

    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake_move)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        monkeypatch.setattr(type(app._session), "session_id", "sess1", raising=False)

        async def leave() -> None:
            order.append("leave")

        app._leave_for_move = leave  # type: ignore[method-assign]
        app._select_sidebar_session = lambda _sid: None  # type: ignore[method-assign]
        app._run_slash_command("/move --to local")
        for _ in range(40):
            await pilot.pause()
            if order[-1:] == ["move sess1 local"]:
                break
        assert order == ["leave", "move sess1 local"], order
        shown = " ".join(_notices(app))
        assert "Moved sess1 home from laptop" in shown, shown


def test_an_offload_to_a_peer_that_is_not_answering_is_refused_and_one_that_cannot_resolve_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The offload half of the pre-check, against its two real answers.

    A peer this device knows and the relay reports silent is a refusal — the
    move would otherwise leave a session behind for a device that is not there.
    A token this device cannot resolve is NOT a refusal: the CLI resolves the
    destination, and a TUI that guessed would refuse a spelling the relay
    accepts.
    """
    from local_operator.network.peers import KnownPeer
    from local_operator.session.peer_rows import UnansweredPeer
    from local_operator.tui.app import _offload_target_refusal

    monkeypatch.setattr(
        "local_operator.network.peers.known_peers",
        lambda root=None: [KnownPeer("d_cloud", "cloud-node-1")],
    )
    monkeypatch.setattr(
        "local_operator.session.peer_rows.read_listing",
        lambda root=None, **kwargs: (
            (),
            (UnansweredPeer("d_cloud", "cloud-node-1", "connect_failed"),),
        ),
    )
    assert _offload_target_refusal("cloud-node-1", "other1", Path("/nowhere"), leaving=False) == (
        "cloud-node-1 is not answering right now, so other1 could not be handed over. "
        "Nothing changed; it is still here."
    )
    # A session the user is IN gets the tail about the screen they are on.
    assert _offload_target_refusal("cloud-node-1", "other1", Path("/nowhere"), leaving=True) == (
        "cloud-node-1 is not answering right now, so other1 could not be handed over. "
        "Nothing changed; you are still in it."
    )
    assert _offload_target_refusal("elsewhere", "other1", Path("/nowhere"), leaving=False) is None
    # A peer that ANSWERED is not in the silence list, so it may take the move.
    monkeypatch.setattr(
        "local_operator.session.peer_rows.read_listing",
        lambda root=None, **kwargs: ((), ()),
    )
    assert (
        _offload_target_refusal("cloud-node-1", "other1", Path("/nowhere"), leaving=False) is None
    )


def test_the_offload_pre_check_reads_the_listing_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1-4: the offload half forces a read, exactly like the recall half.

    A TTL-cached listing answers "who was silent when the sidebar last polled",
    and refusing a move on that refuses on a fact that may have changed since —
    the asymmetry the recall half already avoided by reading on a miss. The
    forced read is the assertion: a cache hit could hide a peer that has
    answered, and the two halves of one command must be equally fresh.
    """
    from local_operator.network.peers import KnownPeer
    from local_operator.tui.app import _offload_target_refusal

    seen: list[dict[str, Any]] = []

    def listing(root: Any = None, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return (), ()

    monkeypatch.setattr(
        "local_operator.network.peers.known_peers",
        lambda root=None: [KnownPeer("d_cloud", "cloud-node-1")],
    )
    monkeypatch.setattr("local_operator.session.peer_rows.read_listing", listing)
    assert (
        _offload_target_refusal("cloud-node-1", "other1", Path("/nowhere"), leaving=False) is None
    )
    assert seen == [{"ttl_s": 0}], seen


@pytest.mark.asyncio
async def test_the_initial_phase_line_says_the_request_is_being_made() -> None:
    """Clause 2's wording: "requesting" until a phase proves something started."""
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        before = app._move_phase_text("sess1", MoveTo(to="local"), [])
        assert before.startswith("requesting a move of sess1 to local"), before
        assert "moving" not in before, before
        # The four step marks survive the tense change.
        assert "· prepared" in before and "· done" in before, before
        assert "✓" not in before, before
        # Once a phase is present the move exists and the line says so.
        started = app._move_phase_text("sess1", MoveTo(to="local"), ["prepared"])
        assert started.startswith("moving sess1 to local"), started
        assert "✓ prepared" in started, started
        assert "· handing off" in started, started


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "phase", "reopens"),
    [("in_progress", "prepared", False), ("completed", "done", True)],
    ids=["in-progress", "completed"],
)
async def test_only_a_completed_move_says_moved_and_reopens(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    phase: str,
    reopens: bool,
) -> None:
    """Clause 1: ``ok`` is acceptance, not arrival — gate on the verified state.

    An ``ok`` receipt whose destination has only reached ``prepared`` is still
    moving: it must NOT reopen the destination (the conversation is not there
    yet) and must say so. The completed row is the control — same fake, same
    command — so the gate is what the difference measures.
    """
    reached = list(MOVE_PHASE_ORDER[: MOVE_PHASE_ORDER.index(phase) + 1])

    def fake_move(
        session_id: str,
        to: str,
        *,
        keep: bool = False,
        queue: bool = False,
        drop_session_roles: bool = False,
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "session_id": session_id,
            "new_session_id": session_id,
            "mode": "move",
            "from_device": {"device_id": "d_me", "name": "laptop"},
            "to_device": {"device_id": "d_peer", "name": "pixel-8"},
            "phase": phase,
            "state": state,
            "phases": [{"phase": item, "at": 0.0} for item in reached],
        }

    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake_move)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        reopened: list[str] = []
        app._select_sidebar_session = reopened.append  # type: ignore[method-assign]
        app._run_slash_command("/move other1 --to pixel-8")
        for _ in range(40):
            await pilot.pause()
            if any(notice.startswith(("Moved", "The move of")) for notice in _notices(app)):
                break
        shown = " ".join(_notices(app))
        if reopens:
            assert reopened == ["other1"], reopened
            assert "Moved other1 to pixel-8. It runs there now." in shown, shown
        else:
            assert reopened == [], "an in-progress move reopened the destination"
            assert (
                "The move of other1 to pixel-8 is still in progress — preparing the handoff. "
                "It has not landed there yet." in shown
            ), shown
            assert "Moved other1" not in shown, shown


# ---------------------------------------------------------------------------
# the queued move (§5.4)
# ---------------------------------------------------------------------------


@dataclass
class _QueuedFake:
    """A ``run_session_move`` stand-in answering the queued receipt.

    The payload mirrors ``mobility._queued_receipt``'s shape — a SUCCESS that
    carries the record in ``queue`` and the queue's own phases — because that
    block is exactly what ``_publish_move_result`` tells the two apart by.
    """

    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self,
        session_id: str,
        to: str,
        *,
        keep: bool = False,
        queue: bool = False,
        drop_session_roles: bool = False,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "session_id": session_id,
                "to": to,
                "keep": keep,
                "queue": queue,
                "drop_session_roles": drop_session_roles,
            }
        )
        # The QUEUED tense (round 4): the product's notice is future here, and the
        # fake must be the same sentence or the test cannot catch a tense slip.
        notice = (
            "1 session role/team definition will be stashed on laptop at "
            "/tmp/config/dropped-session-roles/other1 when this queued move "
            "runs — a mesh move does not carry them"
        )
        carry = {"session_roles": 1, "session_roles_notice": notice} if drop_session_roles else {}
        return {
            "ok": True,
            "session_id": session_id,
            "new_session_id": session_id,
            "mode": "move",
            "from_device": {"device_id": "d_me", "name": "laptop"},
            "to_device": {"device_id": "d_peer", "name": "pixel-8"},
            "phase": "queued",
            "phases": [{"phase": "queued", "at": 0.0}],
            "queue": {
                "phase": "queued",
                "phases": [{"phase": "queued", "at": 0.0}],
                "to_device": "d_peer",
                "to_name": "pixel-8",
                "detail": "",
                "code": "",
                "request_id": "cli-rq-1",
                "updated_at": 0.0,
                "drop_session_roles": drop_session_roles,
                "session_roles": 1 if drop_session_roles else 0,
                "session_roles_notice": notice if drop_session_roles else "",
            },
            **({"carry": carry} if carry else {}),
        }


def _rendered_in_muted(block: object) -> bool:
    """Whether the block's RENDERABLE paints in the theme's `muted` ink.

    Reads the painted object (text style + span styles), not the block's private
    `_token`: the finding asked for the rendered ink, and a token assertion
    passes even if the theme mapping changes the colour underneath it.
    """
    from local_operator.tui import theme as theme_mod

    text = getattr(block, "renderable", None)
    target = theme_mod.semantic_color("muted")
    styles = [getattr(text, "style", None)] + [span.style for span in getattr(text, "spans", [])]
    for style in styles:
        if style is None:
            continue
        if isinstance(style, str):
            # A token span resolves through the theme; `muted` is the token whose
            # ink this check is about.
            if style == "muted":
                return True
            continue
        if str(getattr(getattr(style, "color", None), "name", "")) == target:
            return True
    return False


@pytest.mark.asyncio
async def test_a_queued_move_passes_the_flag_and_renders_the_queue_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--queue`` travels, and the receipt is the queue's own (§5.4).

    The queued answer is a SUCCESS that commits nothing: the notice must say the
    move runs at the next safe point, name a PRODUCT cancel route (design review
    round 1, D1 — the composer carries no cancel verb, so the receipt says "ask
    Local Operator" rather than a terminal command this surface would refuse),
    and must NOT paint the move's phase row as if steps had been walked.
    """
    fake = _QueuedFake()
    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        reopened: list[str] = []
        app._select_sidebar_session = reopened.append  # type: ignore[method-assign]
        app._run_slash_command("/move other1 --to pixel-8 --queue")
        for _ in range(30):
            await pilot.pause()
            if any("Queued a move" in n for n in _notices(app)):
                break
        assert fake.calls == [
            {
                "session_id": "other1",
                "to": "pixel-8",
                "keep": False,
                "queue": True,
                "drop_session_roles": False,
            }
        ]
        shown = " ".join(_notices(app))
        assert "Queued a move of other1 to pixel-8 — waiting for a safe point." in shown, shown
        assert "ask Local Operator to cancel the queued move" in shown, shown
        # THE RETIRED SHAPE (design review round 1, D1): the receipt used to
        # hand the reader the CLI's own ``--cancel-queued`` line — a spelling
        # belonging to another surface's vocabulary (design §2.9).
        assert "cancel-queued" not in shown, shown
        # And the paint (design review round 1, D2): a multi-line receipt rides
        # `note`/`muted`, never `info`/`dim` — 3.77:1 on light paper, under the
        # 4.5:1 AA floor.
        from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView

        receipt = next(
            block
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, NoticeBlock) and "Queued a move of other1" in block.text()
        )
        assert _rendered_in_muted(receipt), receipt.renderable
        assert reopened == []
        # No phase row: a queued move has walked no move phase yet.
        assert not any(n.startswith("moving other1") for n in _notices(app)), _notices(app)


@pytest.mark.asyncio
async def test_a_queued_drop_move_reports_the_stash(monkeypatch: pytest.MonkeyPatch) -> None:
    """M4/QA round 2: a queued DROP receipts its stash, like the run will.

    The flag travels on the call, and the receipt prints the carry line the run
    will honour — a queued drop that said nothing left the user with no
    statement about definitions they are about to lose from this conversation.
    """
    fake = _QueuedFake()
    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        app._run_slash_command("/move other1 --to pixel-8 --queue --drop-session-roles")
        for _ in range(30):
            await pilot.pause()
            if any("Queued a move" in n for n in _notices(app)):
                break
        assert fake.calls[0]["drop_session_roles"] is True, fake.calls
        shown = " ".join(_notices(app))
        assert "dropped-session-roles/other1" in shown, shown
        assert "will be stashed on laptop" in shown, shown
        assert ".." not in shown, shown


@pytest.mark.asyncio
async def test_a_queued_move_stands_the_live_turn_guard_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--queue`` answers the guard instead of tripping it (§5.4).

    Without the flag the same request is refused by "a turn is still running";
    the queue is the shape that WAITS the turn out, so a request carrying it must
    travel — the merged queue path converts exactly this blocker into a durable
    intent (``mobility._source_prepare``'s enqueue branch).
    """
    fake = _QueuedFake()
    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        monkeypatch.setattr(type(app._session), "session_id", "sess1", raising=False)
        order: list[str] = []

        async def leave() -> None:
            order.append("leave")

        app._leave_for_move = leave  # type: ignore[method-assign]
        app._turn_is_live = lambda: True  # type: ignore[method-assign]
        app._run_slash_command("/move --to pixel-8 --queue")
        for _ in range(30):
            await pilot.pause()
            if fake.calls:
                break
        # THE QUEUED REQUEST STILL LEAVES FIRST (the runtime cannot be retired
        # from under this viewer) — the guard stood down, the leave did not.
        assert order == ["leave"], order
        assert fake.calls == [
            {
                "session_id": "sess1",
                "to": "pixel-8",
                "keep": False,
                "queue": True,
                "drop_session_roles": False,
            }
        ]
        assert not any("a turn is still running" in n for n in _notices(app)), _notices(app)


@pytest.mark.asyncio
async def test_a_queued_move_stands_the_sidebar_hold_guard_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hold that refuses the bare move is WAITED OUT with ``--queue``.

    A sidebar source is an attached viewer; the queue accepts a viewed source
    and announces clients at the safe point, so refusing a queued request here
    would dead-end the flag's own use case (the cell above covers the bare
    refusal, which must stay).
    """
    fake = _QueuedFake()
    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        app._select_sidebar_session = lambda _sid: None  # type: ignore[method-assign]

        class Held:
            retired = False

        app._sidebar_sources["held1"] = Held()  # type: ignore[assignment]
        app._run_slash_command("/move held1 --to pixel-8 --queue")
        for _ in range(30):
            await pilot.pause()
            if fake.calls:
                break
        assert fake.calls == [
            {
                "session_id": "held1",
                "to": "pixel-8",
                "keep": False,
                "queue": True,
                "drop_session_roles": False,
            }
        ]
        assert any("Queued a move of held1" in n for n in _notices(app)), _notices(app)


@dataclass(frozen=True)
class _PeerFacts:
    """Enough owner facts for ``_remote_owner_facts`` to name the home device."""

    device_id: str = "d_peer"
    device_name: str = "pixel-8"


class _OwnedRemote(FakeSession):
    """A viewer of a PEER's conversation: an id, and the device that holds it.

    ``runtime_locality`` is the one fact ``_remote_owner_facts`` branches on, and
    the owner's ``facts`` block is the same source the band and the prompt router
    read — a test double for the facade, not a second lookup invented here.
    """

    runtime_locality = "another-machine"

    def __init__(self, session_id: str = "ux-session") -> None:
        super().__init__()
        self._sid = session_id
        self._owner = SimpleNamespace(facts=_PeerFacts())

    @property
    def session_id(self) -> str:
        return self._sid


def _queued_recall(session_id: str = "ux-session") -> dict[str, Any]:
    """The receipt a RECALL that had to wait answers with (``mobility``'s shape).

    A recall runs on the destination, so ``from_device`` is the peer that still
    holds the conversation and ``to_device`` is this device — the opposite of
    ``_QueuedFake``, which is shaped for an offload.
    """
    return {
        "ok": True,
        "session_id": session_id,
        "new_session_id": session_id,
        "mode": "move",
        "from_device": {"device_id": "d_peer", "name": "pixel-8"},
        "to_device": {"device_id": "d_me", "name": "laptop"},
        "phase": "queued",
        "phases": [{"phase": "queued", "at": 0.0}],
        "queue": {
            "phase": "queued",
            "phases": [{"phase": "queued", "at": 0.0}],
            "to_device": "d_me",
            "to_name": "laptop",
            "detail": "",
            "code": "",
            "request_id": "cli-rq-recall",
            "updated_at": 0.0,
        },
        "state": "in_progress",
    }


def _patch_the_peer(monkeypatch: pytest.MonkeyPatch, *, sid: str = "ux-session") -> list[str]:
    """Make ``sid`` a reachable peer session and record every remote open."""
    monkeypatch.setattr(
        "local_operator.session.remote_open.remote_row_and_silence",
        lambda session_id, root: (
            (_row(session_id, remote=True), ()) if session_id == sid else (None, ())
        ),
    )
    monkeypatch.setattr(
        "local_operator.session.remote_open.remote_row_for",
        lambda session_id, root: _row(session_id, remote=True) if session_id == sid else None,
    )
    opened: list[str] = []

    async def fake_open(session_id: str, **kwargs: Any) -> Any:
        opened.append(session_id)
        return _OwnedRemote(session_id)

    monkeypatch.setattr("local_operator.session.remote_open.open_remote_viewer", fake_open)
    return opened


@pytest.mark.asyncio
async def test_the_target_check_is_visible_while_it_is_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U1(a): the mesh read can block for the probe budget — the screen must say so.

    Before the fix nothing was appended until the read answered, so the one state
    a user is most likely to retry (a move that appears to have done nothing) was
    also the one state with no feedback at all.
    """
    gate = asyncio.Event()
    ran: list[Any] = []

    async def held_check(session_id: str, request: MoveTo, *, leaving: bool) -> str | None:
        await gate.wait()
        return "other1 is on pixel-8, which is unreachable."

    monkeypatch.setattr(
        "local_operator.tui.app.run_session_move",
        lambda *args, **kwargs: ran.append(args) or dict(_COMMITTED),
    )
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        app._move_target_refusal = held_check  # type: ignore[method-assign]
        app._run_slash_command("/move other1 --to pixel-8")
        for _ in range(30):
            await pilot.pause()
            if any("checking whether" in notice for notice in _notices(app)):
                break
        assert "checking whether pixel-8 can take other1…" in _notices(app), _notices(app)
        # R2-3 / D2-1: the held-probe row is prose the user reads while the
        # fan-out can block, so it carries `note`'s muted ink (7.18:1 light) and
        # not `info`'s dim (3.77:1 light) the design round measured below AA.
        # Read the PAINTED block, not the token: the finding was about ink.
        from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView

        checking = next(
            block
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, NoticeBlock)
            and block.text().startswith("checking whether pixel-8 can take other1")
        )
        assert _rendered_in_muted(checking), checking.renderable
        # NOTHING CLAIMS THE MOVE STARTED while the check is held: the phase row
        # is built but not painted, because no target has accepted yet.
        assert not any(n.startswith("requesting a move") for n in _notices(app)), _notices(app)
        gate.set()
        for _ in range(30):
            await pilot.pause()
            if any("unreachable" in notice for notice in _notices(app)):
                break
        shown = _notices(app)
        assert not any("checking whether" in n for n in shown), shown
        assert any("unreachable" in n for n in shown), shown
        assert not any(n.startswith("requesting a move") for n in shown), shown
        assert ran == [], "the CLI ran after a refused pre-check"


@pytest.mark.asyncio
async def test_the_phase_row_is_readable_in_the_session_the_leave_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U1(b) + D1: the pending row lives on the SURVIVING transcript, at AA ink.

    The row was appended to the transcript the real ``_leave_for_move`` then
    replaced, so the whole pending request ran on an empty new conversation
    (measured: ``notices=[]``). Nothing here stubs the leave or the reload — the
    CLI is held open so the frame can be read while the request is in flight.
    """
    started = threading.Event()
    release = threading.Event()

    def held_move(session_id: str, to: str, **kwargs: Any) -> dict[str, Any]:
        started.set()
        if not release.wait(30):
            raise AssertionError("the test never released the CLI")
        return dict(_COMMITTED)

    monkeypatch.setattr("local_operator.tui.app.run_session_move", held_move)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        monkeypatch.setattr(type(app._session), "session_id", "sess1", raising=False)
        before = app._session
        app._run_slash_command("/move --to pixel-8")
        for _ in range(80):
            await pilot.pause()
            if started.is_set():
                break
        assert started.is_set(), "the CLI never ran"
        # THE REAL LEAVE RAN — a different session object is current — and the
        # row is on the transcript the user is now looking at.
        assert app._session is not before
        shown = " ".join(_notices(app))
        assert (
            "requesting a move of sess1 to pixel-8 — waiting for pixel-8 to accept" in shown
        ), shown
        from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView

        pending = next(
            block
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, NoticeBlock)
            and block.text().startswith("requesting a move of sess1")
        )
        # D1: `note`'s muted ink, not `info`'s dim (3.77:1 on light paper).
        assert _rendered_in_muted(pending), pending.renderable
        release.set()
        for _ in range(80):
            await pilot.pause()
            if any("Moved sess1 to pixel-8" in notice for notice in _notices(app)):
                break
        assert any("Moved sess1 to pixel-8" in n for n in _notices(app)), _notices(app)


@pytest.mark.asyncio
async def test_a_queued_recall_survives_the_real_remote_reopen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2: the queue receipt is on the FINAL screen, not only "opened … on …".

    The real adoption path runs here (``_select_sidebar_session`` →
    ``_open_remote_session`` → ``_adopt_built_viewer``); only the peer's own
    transport seams are faked. Before the fix the receipt was painted into the
    fresh local transcript and then replaced: the user lost the one durable
    answer to "did my move happen?" — and its cancel route with it.
    """
    opened = _patch_the_peer(monkeypatch)
    monkeypatch.setattr("local_operator.tui.app.run_session_move", lambda *a, **k: _queued_recall())
    app = OperatorApp(lambda: _factory(_OwnedRemote()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        app._run_slash_command("/move --to local --queue")
        for _ in range(80):
            await pilot.pause()
            if any("opened ux-session" in notice for notice in _notices(app)):
                break
        shown = " ".join(_notices(app))
        assert "opened ux-session on pixel-8" in shown, shown
        # THE RECEIPT, in the design's own words, on the reopened conversation.
        assert "Queued a move of ux-session to this device." in shown, shown
        assert "Nothing has moved yet; you are still on pixel-8." in shown, shown
        assert "Ask Local Operator to cancel the queued move before it starts." in shown, shown
        # The window-disconnection explanation survives after it.
        assert "Windows open on it are told first" in shown, shown
        assert opened == ["ux-session"], opened
        assert "Moved ux-session" not in shown, shown


@pytest.mark.asyncio
async def test_a_queued_recall_that_cannot_reopen_still_says_it_is_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2-1: an ACCEPTED queue must not vanish because the way back failed.

    The real leave, the real ``_publish_move_result`` and the real
    ``_reopen_where_the_session_lives`` run; only the peer's transport fails —
    the link drops while opening, which is the shape the UX round measured (its
    final frame held nothing but "could not open … link dropped"). The receipt
    itself is destination-scoped and must NOT be posted on the fresh local
    conversation, so what carries the accepted facts there is the fallback: the
    same receipt with the "you are still on <peer>" claim removed.
    """
    opened = _patch_the_peer(monkeypatch)

    async def dropped_link(session_id: str, **kwargs: Any) -> Any:
        opened.append(session_id)
        raise ConnectionError("synthetic peer link dropped during reopen")

    monkeypatch.setattr("local_operator.session.remote_open.open_remote_viewer", dropped_link)
    monkeypatch.setattr("local_operator.tui.app.run_session_move", lambda *a, **k: _queued_recall())
    app = OperatorApp(lambda: _factory(_OwnedRemote()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        app._run_slash_command("/move --to local --queue")
        for _ in range(80):
            await pilot.pause()
            if any("could not reopen" in notice for notice in _notices(app)):
                break
        shown = " ".join(_notices(app))
        # The failure really happened, on the id the move was for.
        assert "could not open ux-session on pixel-8" in shown, shown
        # THE ACCEPTED OUTCOME SURVIVES, and it is still true on this screen:
        # queued, nothing moved yet, and the cancel route before it starts.
        assert "Queued a move of ux-session to this device." in shown, shown
        assert "Nothing has moved yet." in shown, shown
        assert "could not reopen the conversation on pixel-8" in shown, shown
        assert "ask Local Operator to cancel the queued move before it starts." in shown, shown
        # THE STALE-NAVIGATION GUARD STILL HOLDS: the destination-scoped body
        # would claim this window is in the peer's conversation. It is not.
        assert "you are still on" not in shown, shown
        assert "Moved ux-session" not in shown, shown
        assert opened == ["ux-session"], opened
        # SINGLE USE: the fallback was spent, so no later transition can repeat it.
        assert app._move_notice_carry is None, app._move_notice_carry


@pytest.mark.asyncio
async def test_a_late_failed_recall_keeps_its_refusal_when_the_reopen_drops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2-1's refusal neighbour: the OTHER fallback shape, on the same failed seam.

    A recall whose peer went silent is REFUSED — nothing moved — and the reopen is
    only the return trip to say so. When that trip fails, the pre-fix code dropped
    the receipt on the same branch the queued cell above pins, leaving only
    "could not open …". The fallback states what happened (nothing), where the
    session is (still on the peer), and why this screen cannot show it — and it
    drops the destination-scoped "you are still in the conversation on …" claim,
    which would be false here.
    """
    opened = _patch_the_peer(monkeypatch)

    async def dropped_link(session_id: str, **kwargs: Any) -> Any:
        opened.append(session_id)
        raise ConnectionError("synthetic peer link dropped during reopen")

    def fake_move(session_id: str, to: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "code": "unreachable",
            "message": "pixel-8 is not answering right now; nothing was changed",
            "session_id": session_id,
            "phase_reached": None,
            "changed": False,
        }

    monkeypatch.setattr("local_operator.session.remote_open.open_remote_viewer", dropped_link)
    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake_move)
    app = OperatorApp(lambda: _factory(_OwnedRemote()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        app._run_slash_command("/move --to local")
        for _ in range(80):
            await pilot.pause()
            if any("could not reopen" in notice for notice in _notices(app)):
                break
        shown = " ".join(_notices(app))
        assert "could not open ux-session on pixel-8" in shown, shown
        assert (
            "Could not move ux-session: pixel-8 did not answer. Nothing moved. This window "
            "could not reopen the conversation on pixel-8; it is still there." in shown
        ), shown
        assert "you are still in the conversation on pixel-8" not in shown, shown
        assert app._move_notice_carry is None, app._move_notice_carry


@pytest.mark.asyncio
async def test_a_late_failed_recall_is_readable_after_the_remote_reopen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2: a refusal that arrives AFTER the leave is not erased by the way back.

    Same real adoption; the producer's ``unreachable`` refusal is the one the UX
    measured. The sentence must name the device, say nothing moved, and say where
    the composer still sends — on the final screen.
    """
    opened = _patch_the_peer(monkeypatch)

    def fake_move(session_id: str, to: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "code": "unreachable",
            "message": "pixel-8 is not answering right now; nothing was changed",
            "session_id": session_id,
            "phase_reached": None,
            "changed": False,
        }

    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake_move)
    app = OperatorApp(lambda: _factory(_OwnedRemote()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        app._run_slash_command("/move --to local")
        for _ in range(80):
            await pilot.pause()
            if any("opened ux-session" in notice for notice in _notices(app)):
                break
        shown = " ".join(_notices(app))
        assert "opened ux-session on pixel-8" in shown, shown
        assert (
            "Could not move ux-session: pixel-8 did not answer. Nothing moved; you are "
            "still in the conversation on pixel-8." in shown
        ), shown
        assert opened == ["ux-session"], opened


@pytest.mark.asyncio
async def test_a_reached_copy_failure_keeps_the_producers_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U7: once a phase was reached, the producer's paused/restart sentence IS the notice.

    A copy failure has a state and a remedy the generic "did not answer / Nothing
    moved" fallback cannot know: the copy was discarded, the source still holds the
    conversation paused, and its relay has to be restarted. The reopen still runs,
    and the notice must not replace that with the late-failed-recall wording.
    """
    opened = _patch_the_peer(monkeypatch)
    producer = (
        "device-a stopped answering during the copy (no answer on its link). "
        "device-b discarded its partial copy; device-a still has the conversation, "
        "paused for this move. On device-a, restart the relay (`lop network restart`), "
        "then run the move again."
    )

    def fake_move(session_id: str, to: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "code": "unreachable",
            "message": producer,
            "session_id": session_id,
            "phase_reached": "prepared",
            "changed": False,
        }

    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake_move)
    app = OperatorApp(lambda: _factory(_OwnedRemote()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        app._run_slash_command("/move --to local")
        for _ in range(80):
            await pilot.pause()
            if any("paused for this move" in notice for notice in _notices(app)):
                break
        shown = " ".join(_notices(app))
        assert "paused for this move" in shown, shown
        assert "restart the relay" in shown, shown
        assert "did not answer" not in shown, shown
        assert "Nothing moved" not in shown, shown
        assert "it did not move" not in shown, shown
        assert "nothing was changed" not in shown, shown
        assert opened == ["ux-session"], opened


@pytest.mark.asyncio
async def test_a_reached_copy_failure_that_cannot_reopen_keeps_the_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U11, the link-drop spelling: a failed reopen may not rewrite the state.

    The reopen still runs and the link dies while opening. Before this branch the
    fallback was the late-failed-recall one, which said the peer "did not answer /
    Nothing moved" — hiding that the copy was discarded and the source is paused.
    The producer's sentence stays; only the placement-neutral "could not reopen"
    fact is appended.
    """
    opened = _patch_the_peer(monkeypatch)
    producer = (
        "device-a stopped answering during the copy (no answer on its link). "
        "device-b discarded its partial copy; device-a still has the conversation, "
        "paused for this move. On device-a, restart the relay (`lop network restart`), "
        "then run the move again."
    )

    async def dropped_link(session_id: str, **kwargs: Any) -> Any:
        opened.append(session_id)
        raise ConnectionError("synthetic peer link dropped during reopen")

    def fake_move(session_id: str, to: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "code": "unreachable",
            "message": producer,
            "session_id": session_id,
            "phase_reached": "prepared",
            "changed": False,
        }

    monkeypatch.setattr("local_operator.session.remote_open.open_remote_viewer", dropped_link)
    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake_move)
    app = OperatorApp(lambda: _factory(_OwnedRemote()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        app._run_slash_command("/move --to local")
        for _ in range(80):
            await pilot.pause()
            if any("could not reopen" in notice for notice in _notices(app)):
                break
        move_notice = next(
            notice for notice in _notices(app) if notice.startswith("Could not move ux-session:")
        )
        assert "paused for this move" in move_notice, move_notice
        assert "restart the relay" in move_notice, move_notice
        assert (
            "This window could not reopen the conversation on pixel-8; it is still there."
            in move_notice
        ), move_notice
        assert "did not answer" not in move_notice, move_notice
        assert "Nothing moved" not in move_notice, move_notice
        assert "Nothing changed" not in move_notice, move_notice
        assert opened == ["ux-session"], opened


@pytest.mark.asyncio
async def test_a_reached_copy_failure_whose_listing_goes_stale_keeps_the_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U11, the listing spelling: the peer's row turns unreachable at reopen time.

    The reopen is refused before it opens anything, so the fallback is the only
    notice left. It must carry the producer's paused/restart sentence, not the
    generic "did not answer / Nothing moved" wording.
    """
    producer = (
        "device-a stopped answering during the copy (no answer on its link). "
        "device-b discarded its partial copy; device-a still has the conversation, "
        "paused for this move. On device-a, restart the relay (`lop network restart`), "
        "then run the move again."
    )
    state = {"moved": False}

    def row_and_silence(session_id: str, root: Any) -> tuple[Any, tuple[Any, ...]]:
        if session_id != "ux-session":
            return None, ()
        return _row(session_id, remote=True, reachable=not state["moved"]), ()

    opened: list[str] = []

    async def fake_open(session_id: str, **kwargs: Any) -> Any:
        opened.append(session_id)
        return _OwnedRemote(session_id)

    def fake_move(session_id: str, to: str, **kwargs: Any) -> dict[str, Any]:
        state["moved"] = True
        return {
            "ok": False,
            "code": "unreachable",
            "message": producer,
            "session_id": session_id,
            "phase_reached": "prepared",
            "changed": False,
        }

    monkeypatch.setattr(
        "local_operator.session.remote_open.remote_row_and_silence", row_and_silence
    )
    monkeypatch.setattr(
        "local_operator.session.remote_open.remote_row_for",
        lambda sid, root: (
            _row(sid, remote=True, reachable=not state["moved"]) if sid == "ux-session" else None
        ),
    )
    monkeypatch.setattr("local_operator.session.remote_open.open_remote_viewer", fake_open)
    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake_move)
    app = OperatorApp(lambda: _factory(_OwnedRemote()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        app._run_slash_command("/move --to local")
        for _ in range(80):
            await pilot.pause()
            if any("could not reopen" in notice for notice in _notices(app)):
                break
        move_notice = next(
            notice for notice in _notices(app) if notice.startswith("Could not move ux-session:")
        )
        assert "paused for this move" in move_notice, move_notice
        assert "restart the relay" in move_notice, move_notice
        assert (
            "This window could not reopen the conversation on pixel-8; it is still there."
            in move_notice
        ), move_notice
        # THE PICK'S OWN "which is unreachable (it did not answer)" row is a
        # different fact about the link; what may not appear is the MOVE notice
        # using that as a replacement for the producer's state.
        assert "did not answer" not in move_notice, move_notice
        assert "Nothing moved" not in move_notice, move_notice
        assert "Nothing changed" not in move_notice, move_notice
        assert opened == [], opened


@pytest.mark.asyncio
async def test_a_refusal_still_reaches_a_local_reopen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2's local CONTROL cell: the done-callback seam delivers, and delivers once.

    A local id returns the sidebar's navigation task (not ``None`` like a peer's
    id), and the carry is spent by that task's done callback. This fixture keeps
    the source current — the same-id pick commits without a swap — so it passes
    against the pre-fix code as well; what it pins is that the local seam does
    not DROP the carry and does not post it twice. The remote cells above are
    the discriminating ones for the defect itself.
    """

    def fake_move(session_id: str, to: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "code": "relay_unavailable",
            "message": "this device's relay is not running",
            "session_id": session_id,
            "phase_reached": None,
            "changed": False,
        }

    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake_move)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        monkeypatch.setattr(type(app._session), "session_id", "sess1", raising=False)
        app._run_slash_command("/move --to pixel-8")
        for _ in range(80):
            await pilot.pause()
            if app._sidebar_navigation.committed_id == "sess1":
                break
        for _ in range(20):
            await pilot.pause()
            if any("Could not move" in notice for notice in _notices(app)):
                break
        shown = " ".join(_notices(app))
        assert "Could not move sess1: this device's relay is not running." in shown, shown
        assert "You are back on sess1 — it did not move." in shown, shown
        # SINGLE USE: the seam that posts must consume the carry, or a second
        # transition would print the same refusal again.
        repeats = [n for n in _notices(app) if "this device's relay is not running" in n]
        assert len(repeats) == 1, repeats


@pytest.mark.asyncio
async def test_a_duplicate_refusal_says_where_the_composer_now_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U3: "copy here" and the visible composer are two different conversations."""

    def fake_move(session_id: str, to: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "code": "duplicate",
            "message": "both this device and pixel-8 list sess1",
            "session_id": session_id,
            "phase_reached": "committed",
            "changed": True,
        }

    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake_move)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        monkeypatch.setattr(type(app._session), "session_id", "sess1", raising=False)
        reopened: list[str] = []
        app._select_sidebar_session = reopened.append  # type: ignore[method-assign]
        app._run_slash_command("/move --to pixel-8")
        for _ in range(60):
            await pilot.pause()
            if any("Could not move" in notice for notice in _notices(app)):
                break
        shown = " ".join(_notices(app))
        assert (
            "This window is now on a new local conversation, not either copy of sess1. "
            "Wait for cleanup to finish before reopening sess1." in shown
        ), shown
        assert reopened == [], "an ambiguous duplicate was reopened automatically"


@pytest.mark.asyncio
async def test_an_unconfirmed_recall_keeps_the_local_copy_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1-2/Q-1: a lost commit reply must not claim the user is back on the source.

    The producer keeps the verified copy in staging and says a retry is unsafe;
    the source may already have handed the id away, so the old ``changed=True``
    tail — "You are back on <id>" — is a claim nobody can support. The recall
    branch must stay on the fresh local conversation (where the copy will land)
    and must not navigate to a source that may be gone.
    """

    def fake_move(session_id: str, to: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "code": "unconfirmed",
            "message": (
                f"the verified copy of {session_id} is kept under /staging/{session_id}: "
                "pixel-8 did not answer the commit"
            ),
            "session_id": session_id,
            "phase_reached": "handing_off",
            "changed": True,
        }

    async def no_refusal(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake_move)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        monkeypatch.setattr(type(app._session), "session_id", "sess1", raising=False)
        monkeypatch.setattr(app, "_move_target_refusal", no_refusal)
        reopened: list[str] = []
        app._select_sidebar_session = reopened.append  # type: ignore[method-assign]
        app._run_slash_command("/move --to local")
        for _ in range(60):
            await pilot.pause()
            if any("Could not move" in notice for notice in _notices(app)):
                break
        shown = " ".join(_notices(app))
        assert "the verified copy of sess1 is kept under /staging/sess1" in shown, shown
        assert (
            "This window is on a new local conversation; the verified copy is held here "
            "and will finish when the other device answers." in shown
        ), shown
        assert "You are back on sess1" not in shown, shown
        assert reopened == [], "an unconfirmed recall reopened a source that may be gone"


@pytest.mark.asyncio
async def test_an_in_progress_move_after_the_leave_says_where_the_composer_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U3: an unlanded handoff says what is happening and where input now goes."""

    def fake_move(session_id: str, to: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "session_id": session_id,
            "new_session_id": session_id,
            "mode": "move",
            "from_device": {"device_id": "d_me", "name": "laptop"},
            "to_device": {"device_id": "d_peer", "name": "pixel-8"},
            "phase": "handing_off",
            "state": "in_progress",
            "phases": [{"phase": "handing_off", "at": 0.0}],
        }

    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake_move)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        monkeypatch.setattr(type(app._session), "session_id", "sess1", raising=False)
        reopened: list[str] = []
        app._select_sidebar_session = reopened.append  # type: ignore[method-assign]
        app._run_slash_command("/move --to pixel-8")
        for _ in range(60):
            await pilot.pause()
            if any("still in progress" in notice for notice in _notices(app)):
                break
        shown = " ".join(_notices(app))
        assert (
            "The move of sess1 to pixel-8 is still in progress — finishing the handoff. "
            "This window is on a new local conversation; sess1 is not ready to reopen yet." in shown
        ), shown
        assert "handing_off" not in shown, shown
        assert "Moved sess1" not in shown, shown
        assert reopened == [], "an unlanded destination was reopened"


@pytest.mark.asyncio
async def test_the_tui_reads_the_shared_receipt_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N1: the TUI must ask ``mobility.move_receipt_state``, not re-derive it.

    A completed-looking receipt is answered ``in_progress`` by the ONE function
    both surfaces read. The local derivation this replaces (``state`` plus a
    phase gate) agreed with the CLI only for as long as nobody changed either
    vocabulary — this proves the TUI has stopped having its own.
    """
    monkeypatch.setattr(
        "local_operator.network.mobility.move_receipt_state", lambda result: "in_progress"
    )
    monkeypatch.setattr("local_operator.tui.app.run_session_move", lambda *a, **k: dict(_COMMITTED))
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        reopened: list[str] = []
        app._select_sidebar_session = reopened.append  # type: ignore[method-assign]
        app._run_slash_command("/move other1 --to pixel-8")
        for _ in range(60):
            await pilot.pause()
            if any("still in progress" in notice for notice in _notices(app)):
                break
        shown = " ".join(_notices(app))
        assert "The move of other1 to pixel-8 is still in progress" in shown, shown
        assert "Moved other1" not in shown, shown
        assert reopened == [], reopened


@pytest.mark.asyncio
async def test_a_viewed_refusal_offers_the_queue_spelling_this_surface_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``viewed_elsewhere`` gains "Move at the next step" (§5.4).

    The producer's own sentence says "queue the move"; this surface must then
    spell the flag IT runs — the papercut the ``--wait`` clause was fixed for,
    one code over.
    """
    from local_operator.session.runtime.types import VIEWED_MOVE_REFUSAL

    def fake_move(
        session_id: str,
        to: str,
        *,
        keep: bool = False,
        queue: bool = False,
        drop_session_roles: bool = False,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "code": "viewed_elsewhere",
            "message": VIEWED_MOVE_REFUSAL,
            "session_id": session_id,
            "phase_reached": None,
            "changed": False,
        }

    monkeypatch.setattr("local_operator.tui.app.run_session_move", fake_move)
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        app._run_slash_command("/move other1 --to pixel-8")
        for _ in range(30):
            await pilot.pause()
            if any("Could not move" in n for n in _notices(app)):
                break
        shown = " ".join(_notices(app))
        assert VIEWED_MOVE_REFUSAL in shown, shown
        assert "To queue it from here: /move other1 --to pixel-8 --queue." in shown, shown


@pytest.mark.asyncio
async def test_a_path_move_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """R16 topology 0: `/move <path>` never reaches the mesh."""
    monkeypatch.setattr(
        "local_operator.tui.app.run_session_move",
        lambda *a, **k: pytest.fail("a path move reached the mesh"),
    )
    scratch = tmp_path_factory.mktemp("move-target")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        applied: list[str] = []
        app._apply_move = lambda raw, notice: applied.append(raw)  # type: ignore[method-assign]
        app._run_slash_command(f"/move {scratch}")
        await pilot.pause()
        assert applied == [str(scratch)]


# ---------------------------------------------------------------------------
# /archive and /delete on a remote current session
# ---------------------------------------------------------------------------


class _RemoteCurrent(FakeSession):
    runtime_locality = "another-machine"

    class _Owner:
        class facts:  # noqa: N801 — mirrors RemoteOwner.facts' attribute shape
            device_id = "d_peer"
            device_name = "pixel-8"

    _owner = _Owner()


@pytest.mark.asyncio
async def test_remote_archive_and_delete_run_on_the_peer_and_print_its_words(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    scratch = tmp_path_factory.mktemp("viewer-root")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(scratch))
    # ``archive_change``'s peer-side branch is reached only when the LOCAL lane is
    # NOT taken, so the assertion below is worth something only if the local lane
    # could have run: a transcript for this session is what
    # ``_resumable_session_id`` gates on, and without one the local lane would
    # return early and "the writer was never called" would be vacuously true.
    (scratch / "sessions" / "sess").mkdir(parents=True)
    (scratch / "sessions" / "sess" / "transcript.jsonl").write_text("", encoding="utf-8")
    calls: list[tuple[str, str, bool]] = []
    local_writes: list[tuple[Any, ...]] = []
    refusal = "this conversation is open in a running session (pid 4242); stop it first"

    # The LOCAL writer, spied rather than inferred from its absence: a remote id
    # must never reach it (§8.3; round 1, MINOR 3).
    monkeypatch.setattr(
        "local_operator.session.archived.archive_change",
        lambda *args, **kwargs: local_writes.append((args, kwargs)) or (False, []),
    )

    def fake_lifecycle(
        session_id: str, *, action: str, peer: str, confirmed: bool = False, root: Any = None
    ) -> dict[str, Any]:
        calls.append((action, peer, confirmed))
        if action == "delete" and confirmed:
            return {"ok": False, "code": "refused", "message": refusal}
        return {"ok": True, "message": f"{action} ok on the owner"}

    monkeypatch.setattr("local_operator.network.mobility.lifecycle", fake_lifecycle)
    app = OperatorApp(lambda: _factory(_RemoteCurrent()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        for command in ("/archive", "/delete", "/delete yes"):
            app._run_slash_command(command)
            for _ in range(20):
                await pilot.pause()
                if len(calls) >= 1 + ("delete" in command) + (command == "/delete yes"):
                    break
            await pilot.pause()
        shown = " ".join(_notices(app))
        assert calls == [
            ("archive", "d_peer", False),
            ("delete", "d_peer", False),
            ("delete", "d_peer", True),
        ], calls
        assert "archive ok on the owner (pixel-8)" in shown, shown
        assert "Nothing was deleted. /delete yes deletes it on pixel-8." in shown, shown
        # THE OWNER'S REFUSAL, VERBATIM.
        assert f"pixel-8 refused: {refusal}" in shown, shown
    # AND THIS DEVICE RAN NO LOCAL ARCHIVE WRITE for a remote id (§8.3). The
    # writer is asserted, not a file's absence: a file-absence check cannot tell
    # "the local writer was never called" from "the local writer was called and
    # wrote nothing", which is the failure §8.3 names (round 1, MINOR 3).
    assert local_writes == [], local_writes


@pytest.mark.asyncio
async def test_a_relayed_connection_without_delete_cannot_archive_through_this_apps_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V2, the TUI-HOSTED twin of the runtime's gate.

    A session is owned either by a detached runtime or by this app, and BOTH
    route a follower's slash command through ``run_slash_authoritative`` →
    ``_slash_result``. A gate in one host only moves the escalation to whichever
    host the operator happens to be running, so this pins the app's own dispatch
    against a real connection's resolved set — and pins the negative too, since a
    gate that refused the VERB rather than the CAPABILITY would take away the
    archive from a connection that legitimately may do it.
    """
    writes: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        "local_operator.session.archived.archive_change",
        lambda *args, **kwargs: writes.append((args, kwargs)) or (False, []),
    )
    app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_no_resume)
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(pilot, app)
        # A ``drive`` role's resolved set, from the vocabulary itself.
        drive = frozenset({"list", "view", "prompt", "steer", "stop", "slash"})
        for command in ("archive", "unarchive", "delete"):
            refused = await app.run_slash_authoritative(
                command, "yes", locality="remote", capabilities=drive
            )
            assert refused["kind"] == "notice", refused
            assert refused["style"] == "warning", refused
            assert command in refused["text"], refused
            assert "delete" in refused["text"], refused
        assert writes == [], writes

        # AND THE HALF THAT IS NOT IMPLIED BY THE LOCALITY (round 3, R3-3): a
        # relayed connection that forwarded NO capability set at all — the keyword
        # left out, which is exactly what a caller that has resolved nothing sends —
        # must be refused too. ``None`` means "not said", not "said none", and until
        # this cell existed the mutation
        # ``may_run_delete_scoped_slash = lambda loc, caps: loc == "local" or caps is
        # None or "delete" in caps`` passed every delete-gate test in the suite: the
        # routed cells above always carried a set, and the terminal carrier hides
        # the gap behind its allowlist.
        for command in ("archive", "unarchive", "delete"):
            refused = await app.run_slash_authoritative(command, "yes", locality="remote")
            assert refused["kind"] == "notice", refused
            assert refused["style"] == "warning", refused
            assert "delete" in refused["text"], refused
        assert writes == [], writes

        # THE NEGATIVE CONTROL: the same command from a connection that DID
        # resolve ``delete`` reaches the handler. (The fake session has no
        # transcript, so the handler's own answer here is "nothing saved yet" —
        # what this asserts is that the GATE let it through.)
        allowed = await app.run_slash_authoritative(
            "archive", "", locality="remote", capabilities=drive | {"delete"}
        )
        assert allowed["kind"] == "notice", allowed
        assert "capability" not in allowed["text"], allowed


async def _phone_dial(record: Any, *, locality: Any = "remote") -> tuple[Any, Any]:
    """One connection carrying the PHONE's own auth frame.

    ``mobile/daemon.py`` authenticates with ``{"key": …, "locality": "remote"}`` and
    no capabilities at all, so that is what this sends — the shape the round-3
    review used to show the allowlist turning the operator's phone away.

    ``locality`` is ``Any`` because the V4-2 cells send values a dialer should never
    send (``0``, ``""``, ``"LOCAL"``) to prove the reader refuses to guess: only the
    exact expected value may take the local lane.

    The record is selected by PID rather than by position: ``registry.scan()`` lists
    every live runtime on this machine, and dialling the first one reaches a
    FOREIGN session's socket (measured) instead of this test's host.
    """
    deadline = asyncio.get_running_loop().time() + 20
    target = None
    while asyncio.get_running_loop().time() < deadline:
        for candidate, liveness in record_scan():
            if liveness == "live" and candidate.pid == os.getpid():
                target = candidate
                break
        if target is not None:
            break
        await asyncio.sleep(0.05)
    seen = [(r.pid, live) for r, live in record_scan()]
    assert (
        target is not None
    ), f"this process published no live runtime record: scanned {seen}, mine={os.getpid()}"
    auth: dict[str, Any] = {"key": target.control_key}
    if locality is not None:
        auth["locality"] = locality
    reader = writer = None
    for _ in range(30):
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", target.control_port, limit=1 << 20
            )
            break
        except OSError:
            # The record is published before the listener finishes binding.
            await asyncio.sleep(0.1)
    assert reader is not None and writer is not None, "the host never accepted a dial"
    writer.write(json.dumps(auth).encode() + b"\n")
    await writer.drain()
    welcome = await asyncio.wait_for(reader.readline(), timeout=10)
    assert json.loads(welcome)["op"] == "projection", welcome
    # SETTLE BEFORE SENDING: the runtime refuses every non-priority op while this
    # connection's canonical sync is still pending ("this viewer is still
    # connecting … retry once the interface has connected"), and that flag clears
    # in the BIND task's ``finally``. The refusal it answers with carries no
    # ``req``, so a request/reply caller waits forever for a reply that was sent —
    # measured as the first three cells of a fresh host, in whichever order they
    # were asked. Reading until the pushes stop is what the round-2 stream tests
    # do for the same reason.
    for _ in range(40):
        try:
            await asyncio.wait_for(reader.readline(), timeout=0.25)
        except (asyncio.TimeoutError, TimeoutError):
            break
    return reader, writer


async def _phone_call(
    pilot: Any, reader: Any, writer: Any, command: str, args: str, req: int
) -> str:
    """Send one ``slash`` frame and pump the APP's loop until the receipt lands.

    The receipt waits on ``TuiSessionHandle._on_app`` — a hop onto the app's own
    loop — so a caller that merely awaits the socket starves the task it is waiting
    for. Measured on a fresh host: the first three frames of the first connection
    timed out at 20 s each that way, and answered immediately when the wait pumped
    the pilot.
    """
    writer.write(
        json.dumps({"op": "slash", "req": req, "command": command, "args": args}).encode() + b"\n"
    )
    await writer.drain()
    for _ in range(400):
        await pilot.pause()
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=0.05)
        except (asyncio.TimeoutError, TimeoutError):
            continue
        frame = json.loads(raw.decode("utf-8", "replace"))
        # ``ack`` IS the reply for this op: ``slash`` is not a payload op, so the
        # receipt travels in ``detail`` rather than in ``result`` (measured — a
        # reader that matched only ``result``/``error`` waited forever for a reply
        # that had already arrived).
        if frame.get("req") == req and frame.get("op") in ("result", "error", "ack"):
            payload = frame.get(
                "result", frame.get("data", frame.get("detail", frame.get("message")))
            )
            return payload if isinstance(payload, str) else json.dumps(payload)
    raise AssertionError("no reply arrived")


@pytest.mark.asyncio
async def test_a_phone_shaped_connection_runs_its_slash_commands_through_the_tui_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3-1 through the REAL host: the phone's own auth frame, server, handle and app.

    The two round-3 halves, in one measured cell: what the phone must KEEP
    (``/rename``, ``/model``, ``/mcp login``, ``/stop``) and what it must not reach
    (``/move``, ``/archive``, ``/delete``, ``/exit``, ``/update``). The lanes are
    the design decision — the owner's TERMINAL via ``_run_slash_command``, or the
    per-verb-gated DISPATCHER via ``run_slash_authoritative`` — and the receipts are
    what the caller sees.

    A local caller is measured in the same run as the control: the phone's shape
    routes, an unmarked loopback connection still types into the terminal.
    """
    from local_operator.mobile.tui_handle import TuiSessionHandle
    from local_operator.session.runtime.server import RuntimeServer

    session = FakeSession()
    moved: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        "local_operator.tui.app.run_session_move",
        lambda *args, **kwargs: moved.append((args, kwargs)),
    )
    writes: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        "local_operator.session.archived.archive_change",
        lambda *args, **kwargs: writes.append((args, kwargs)) or (False, []),
    )
    stop_all: list[str] = []
    app = OperatorApp(lambda: _factory(session), resume_factory=_no_resume)
    # THIS IS THE INSTRUMENT THAT MATTERS FOR V4-1: the fan-out is an ACTION, and a
    # receipt cannot show whether it ran, so the method itself is replaced. A relayed
    # `/stop all` that reaches it is a machine-wide stop in production.
    monkeypatch.setattr(app, "_stop_all", lambda notice: stop_all.append("ARMED"))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(400):
            await pilot.pause()
            if app._session is session:
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("the app never adopted the session")

        terminal: list[str] = []
        dispatched: list[tuple[str, Any, Any]] = []
        real_terminal = app._run_slash_command

        def spy_terminal(line: str, attachments: Any = None) -> Any:
            terminal.append(line)
            return real_terminal(line, attachments)

        app._run_slash_command = spy_terminal  # type: ignore[method-assign]
        real_dispatch = app.run_slash_authoritative

        async def spy_dispatch(command: str, args: str, images: Any = None, **kw: Any) -> Any:
            dispatched.append((command, kw.get("locality"), kw.get("capabilities")))
            return await real_dispatch(command, args, images, **kw)

        app.run_slash_authoritative = spy_dispatch  # type: ignore[method-assign]

        handle = TuiSessionHandle(app)
        server = RuntimeServer(handle, kind="tui")
        server.start()
        try:
            assert await server.wait_until_published(timeout=15), "no record published"
            reader, writer = await _phone_dial(record=None)
            req = 500
            try:
                # THE FOUR THE REVIEW NAMED, and the lane each must reach. The
                # EFFECT is asserted where the app can produce one (the rename
                # lands on the session), and the LANE everywhere, because the
                # lanes are what the routing decides; the receipts are the app's
                # own words either way (``/rename``'s is "renamed: …", not a
                # transport string, measured).
                for command, args, lane in (
                    ("rename", "probe title", "dispatcher"),
                    ("model", "", "dispatcher"),
                    ("mcp", "login notion", "dispatcher"),
                    ("stop", "", "terminal"),
                ):
                    terminal.clear()
                    dispatched.clear()
                    req += 1
                    receipt = await _phone_call(pilot, reader, writer, command, args, req)
                    assert receipt, (command, receipt)
                    if lane == "terminal":
                        assert terminal == [f"/{command}"], (command, terminal)
                        assert dispatched == [], (command, dispatched)
                    else:
                        assert command in [row[0] for row in dispatched], (command, dispatched)
                        assert terminal == [], (command, terminal)
                    # NEVER the carrier's own words: no refusal of this gate's
                    # reached these verbs.
                    assert "not available" not in receipt or command == "rename", (command, receipt)
                # The rename really landed on the session the phone was watching.
                assert session.conversation_name == "probe title", session.conversation_name

                # V4-1: THE ARGUMENT FORMS OF `/stop` ARE NOT THE PHONE'S. This lane
                # hands over the whole LINE, and `/stop`'s local handler is
                # argument-sensitive: `all` arms the machine-wide fan-out and <pid>
                # reaches another session. Both must fall out of the terminal lane and
                # be refused — asserted on the fan-out's own method, because a receipt
                # cannot prove an action did not run.
                for args, expect in (
                    ("all", "attached terminal"),
                    ("ALL", "attached terminal"),
                    ("62181", "attached terminal"),
                    ("--all", "attached terminal"),
                ):
                    terminal.clear()
                    dispatched.clear()
                    req += 1
                    receipt = await _phone_call(pilot, reader, writer, "stop", args, req)
                    assert not receipt.startswith("ran "), (args, receipt)
                    assert expect in receipt, (args, receipt)
                    assert terminal == [], (args, terminal)
                    assert [row[0] for row in dispatched] == ["stop"], (args, dispatched)
                assert stop_all == [], stop_all
                # THE POSITIVE CONTROL for the same rule: the BARE form is the kill
                # switch for the session this carrier is attached to and still reaches
                # the terminal lane (asserted with the four verbs above), so V4-1 was
                # fixed by scoping the ARGUMENT, not by taking `/stop` from the phone.

                # AND THE FIVE THAT MUST BE REFUSED. The delete-scoped pair is
                # refused by the capability gate before a lane is chosen; `move`,
                # `exit` and `update` reach the dispatcher and come back with the
                # app's OWN sentence, which is what makes the phone's "no" honest
                # rather than a second copy of it here.
                terminal.clear()
                dispatched.clear()
                for command, args, expect in (
                    ("move", "abc --to evil", "attached terminal"),
                    ("exit", "", "attached terminal"),
                    ("update", "", "attached terminal"),
                    ("archive", "", "delete"),
                    ("delete", "yes", "delete"),
                ):
                    req += 1
                    receipt = await _phone_call(pilot, reader, writer, command, args, req)
                    assert not receipt.startswith("ran "), receipt
                    assert expect in receipt, (command, receipt)
                assert moved == [], moved
                assert terminal == [], terminal
                # ``/move`` reaches the dispatcher and dies there (no branch);
                # the delete-scoped pair never reaches a lane at all.
                assert [row[0] for row in dispatched] == ["move", "exit", "update"], dispatched
            finally:
                writer.close()

            # V4-2: ONLY THE EXACT VALUE TAKES THE LOCAL LANE. Every other spelling a
            # dialer could send must be read as RELAYED — the old reading handed them
            # the widest lane, `/move` and the whole local verb set included. Each cell
            # asks for the one command that PROVES the lane: `/move` runs in the owner's
            # terminal iff the connection was read as local.
            for spelling in ("loopback", "", 0, "LOCAL", "Local", "remote "):
                reader3, writer3 = await _phone_dial(record=None, locality=spelling)
                try:
                    terminal.clear()
                    dispatched.clear()
                    moved.clear()
                    req += 1
                    receipt = await _phone_call(
                        pilot, reader3, writer3, "move", "abc --to evil", req
                    )
                    assert moved == [], (spelling, moved)
                    assert terminal == [], (spelling, terminal)
                    assert not receipt.startswith("ran "), (spelling, receipt)
                    assert "attached terminal" in receipt, (spelling, receipt)
                    # ... and the capability gate still answers the delete-scoped verb.
                    req += 1
                    refused = await _phone_call(pilot, reader3, writer3, "archive", "", req)
                    assert "delete" in refused and "nothing was changed" in refused, refused
                finally:
                    writer3.close()
                assert writes == [], writes

            # THE CONTROL: an unmarked loopback connection is still LOCAL, and every
            # one of those same commands runs in the terminal exactly as before.
            reader2, writer2 = await _phone_dial(record=None, locality=None)
            try:
                terminal.clear()
                dispatched.clear()
                for command, args in (("rename", "local title"), ("move", "abc --to evil")):
                    req += 1
                    receipt = await _phone_call(pilot, reader2, writer2, command, args, req)
                    assert receipt, (command, receipt)
                assert dispatched == [], dispatched
                # Both commands a PHONE is refused ran here. The rename's own
                # effect is asserted on the phone's cell above, where the
                # dispatcher's producer answers synchronously; this control is
                # about the LANE, which is what the local reading decides.
                assert terminal == ["/rename local title", "/move abc --to evil"], terminal

            finally:
                writer2.close()

            # R3-2: the ROUTED entry's own default is fail-closed too, and this is
            # the carrier that feeds the app's delete gate. Omitting ``locality``
            # entirely — the shape a forgotten forward takes — must not be read as
            # local. Mutate that default back to ``locality: str = "local"`` on
            # ``TuiSessionHandle.run_slash_authoritative`` and this cell writes the
            # archive instead of answering with the capability sentence.
            denied = await handle.run_slash_authoritative("archive", "", None)
            assert denied["style"] == "warning", denied
            assert "delete" in denied["text"], denied
            assert writes == [], writes
        finally:
            server.close()
