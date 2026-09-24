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
import time
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
        # Refusals, each by name.
        ("--to", MoveTo(error="--to needs a device: /move --to <peer|local>")),
        ("~/src --to pixel-8", MoveTo(error=AMBIGUOUS_MOVE)),
        ("./x --to local", MoveTo(error=AMBIGUOUS_MOVE)),
        ("a b --to local", MoveTo(error=AMBIGUOUS_MOVE)),
        ("--to x --force", MoveTo(error="/move --to takes only --keep, not '--force'")),
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

    def fake_move(session_id: str, to: str, *, keep: bool = False) -> dict[str, Any]:
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
        " You are back on sess1 — it did not move.",
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

    def fake_move(session_id: str, to: str, *, keep: bool = False) -> dict[str, Any]:
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
        if changed:
            assert "it did not move" not in shown, shown


@pytest.mark.asyncio
async def test_a_refused_move_says_so_and_reopens_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_move(session_id: str, to: str, *, keep: bool = False) -> dict[str, Any]:
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
async def test_a_move_of_a_session_this_tui_holds_in_the_sidebar_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M's deferral, TUI half: this terminal's own parked viewer blocks the move.

    Refused BEFORE anything runs, naming THIS terminal; once the source is
    released the same command proceeds.
    """
    ran: list[str] = []

    def fake_move(session_id: str, to: str, *, keep: bool = False) -> dict[str, Any]:
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


async def _phone_dial(record: Any, *, locality: str | None = "remote") -> tuple[Any, Any]:
    """One connection carrying the PHONE's own auth frame.

    ``mobile/daemon.py`` authenticates with ``{"key": …, "locality": "remote"}`` and
    no capabilities at all, so that is what this sends — the shape the round-3
    review used to show the allowlist turning the operator's phone away.

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
    app = OperatorApp(lambda: _factory(session), resume_factory=_no_resume)
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
