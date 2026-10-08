"""A park on ANOTHER device reaches this origin — once per episode.

The peer's runtime owns the park; nothing is pushed to this machine, so the
origin's notice is edge-detected off the sidebar poll's own rows
(``session/peer_rows.park_edges``). These tests drive the REAL app (the one
that loads ``local_operator.tcss``) because the questions are what a user sees
and what a machine hears:

* one card per EPISODE, withdrawn when the park clears — a card left standing
  after the answer lands contradicts the very thing that ended the park;
* the watch is ALWAYS ON: the sidebar's refresh is the fast path, and a slow
  always-on tick keeps looking while the list is CLOSED — the app's default
  state, where hanging the notice off the sidebar's paused timer would mean a
  park waits, unbounded, until somebody opens the list (agent review round 1,
  MAJOR-1);
* a device that does NOT answer is not a clear: its episode survives the
  outage and is not re-announced on recovery (agent review round 1, MINOR-1);
  a whole-RELAY failure is the documented residual — indistinguishable from an
  all-clear at this seam, so it reads as one (see `park_edges`);
* the OS banner goes through the app's existing notifier, naming the DEVICE,
  and stays quiet while the user is looking at the terminal (the notifier's own
  focus gate, unchanged);
* while THIS app is attached to the parked session neither fires: the gate card
  is already on screen with the device hint on it, and the app must not toast
  over its own dock card;
* nothing is written to the attention store — a park has no token, and the
  store's rule is that no automatic path acknowledges (design note §2).

``test_tunnel_park_notice.py`` is the sibling suite (a park of THIS machine's
connector); this one is the mesh half.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.resume import SessionRow
from local_operator.tui.app import REMOTE_PARK_POLL_S, OperatorApp
from local_operator.tui.notify import remote_park_banner, remote_park_card
from local_operator.tui.widgets.toast import Toast
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_notify_wiring import RecordingNotifier


@pytest.fixture(autouse=True)
def isolate_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A headless pilot that inherits the operator's CMUX_* variables can rename
    # their real cmux workspaces; HOME is redirected too because the config dir
    # alone leaves the cache pointed at the real home (AGENTS.md, "Isolating a
    # run").
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda self: None)


def _parked(
    session_id: str = "s_parked",
    device_id: str = "d_bb",
    *,
    name: str = "Backfill the audit log",
    kind: str | None = "approval",
    device_name: str = "demo-laptop",
    live_state: str = "busy",
) -> SessionRow:
    """One live parked peer row — the shape ``peer_session_rows`` hands the app."""
    return SessionRow(
        id=session_id,
        mtime=0.0,
        name=name,
        pending=kind,
        live_state=live_state,
        locality="remote",
        owner_device=device_id,
        owner_device_name=device_name,
    )


class AttachedFake(FakeSession):
    """A fake whose session is a PEER's — the attached origin viewer."""

    runtime_locality = "another-machine"

    def __init__(
        self,
        *,
        session_id: str = "s_parked",
        device_id: str = "d_bb",
        device_name: str = "demo-laptop",
    ) -> None:
        super().__init__()
        self._session_id = session_id
        self._owner = SimpleNamespace(
            facts=SimpleNamespace(device_id=device_id, device_name=device_name)
        )

    @property
    def session_id(self) -> str:
        return self._session_id


async def _boot(pilot: Any, app: OperatorApp) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


def _toast(app: OperatorApp) -> Toast:
    return app.query_one(Toast)


async def _stage_card(app: OperatorApp, pilot: Any) -> asyncio.Task[Any]:
    """Put the app's OWN approval card on screen for the current gate.

    The REAL path rather than a hand-set ``_approval``: the park notice yields
    to the card the app would actually have mounted, so the cells drive the
    same mount the operator sees — and the same one the fix's predicate reads.
    Returns the parked gate task for the caller to cancel on the way out.
    """
    session = cast(Any, app._session)
    assert session is not None
    session.pending_gate = SimpleNamespace(
        kind="approval", request_id="req-attached", question_index=0
    )
    task = asyncio.create_task(app.request_tool_approval("bash", "echo park-check"))
    for _ in range(80):
        await pilot.pause()
        if app._approval is not None and app._approval.is_mounted:
            break
    assert app._approval is not None and app._approval.is_mounted, "the card never mounted"
    return task


@pytest.mark.asyncio
async def test_a_park_is_announced_once_and_withdrawn_when_it_clears() -> None:
    """One card, the device-named remedy, and it leaves when the park does."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        toast = _toast(app)
        rows = (_parked(),)

        app._note_remote_parks(rows)
        await pilot.pause()
        assert toast.display, "a park on a peer must say so"
        assert toast.message == remote_park_card(
            "demo-laptop", "approval", name="Backfill the audit log"
        )
        assert (
            toast.message.splitlines()[0] == "Backfill the audit log"
        ), "the card must name the conversation so two parks on one device differ"
        assert "Waiting for approval on demo-laptop" in toast.message
        # The repaved tail (D3): the card keeps the older-build DIAGNOSIS and drops
        # the two backticked verbs, so the pin names the product action instead of
        # the retired `lop network ready --peer …` shape (Q7).
        assert "updating it is what fixes that" in toast.message
        assert "ask Local Operator to set up operator authority there" in toast.message

        # The same episode, polled again: `show` re-arms its own dismissal
        # timer, so re-raising on every tick would hold the card forever.
        generation = toast.generation
        app._note_remote_parks(rows)
        await pilot.pause()
        assert toast.generation == generation, "a second poll re-raised the card"

        # The park clears (answered): the card leaves with it.
        app._note_remote_parks(())
        await pilot.pause()
        assert not toast.display, "an answered park left its card standing"

        # A SECOND park is a second episode.
        app._note_remote_parks(rows)
        await pilot.pause()
        assert toast.display and toast.generation > generation


@pytest.mark.asyncio
async def test_the_banner_names_the_device_and_the_conversation() -> None:
    """The OS leg rides the app's notifier, with the device in the body."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        notifier = RecordingNotifier()
        app._notifier = notifier  # type: ignore[assignment]

        app._note_remote_parks((_parked(),))
        await pilot.pause()

        assert notifier.kinds == ["approval"]
        assert notifier.bodies == [remote_park_banner("demo-laptop", "approval")]
        assert notifier.labels == ["Backfill the audit log"]


@pytest.mark.asyncio
async def test_an_attached_viewer_is_not_told_about_its_own_card() -> None:
    """THIS app attached to the parked session WITH its gate card up: silent.

    The premise is the CARD: it is on screen (with the device hint on it), so
    a toast would be the app interrupting itself — and the peer's own ladder
    is what stops ITS banner, because a viewer is watching. The episode is
    still consumed: the user is looking at the surface the notice would point
    them to. The card is staged through the real mount (``_stage_card``), not
    assumed: with no card up this app must be TOLD (the F-A cells below).
    """
    app = OperatorApp(lambda: _factory(AttachedFake()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        notifier = RecordingNotifier()
        app._notifier = notifier  # type: ignore[assignment]
        toast = _toast(app)
        gate_task = await _stage_card(app, pilot)
        # Staging the card fires the app's own LOCAL waiting edge (the band's
        # waiting phase) — that one belongs to the card, not to the park. What
        # must not happen is a SECOND, park-shaped notice on top of it.
        before = list(notifier.kinds)

        app._note_remote_parks((_parked(),))
        await pilot.pause()

        assert not toast.display, "the app toasted over its own parked gate"
        assert notifier.kinds == before, "the app notified about the session it is in"
        assert app._remote_park_episodes, "the suppressed episode was not consumed"

        gate_task.cancel()
        await asyncio.gather(gate_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_an_attached_viewer_without_its_card_is_still_told() -> None:
    """F-A (beat-2): the follow view suppressed the notice with NO card on screen.

    The auto-follow after a slash move attaches this app to the moved session,
    and the notice used to yield to ATTACHMENT ALONE. The follow view rendered
    no gate card at all — the facade held no pending gate, the band read
    ``No owner`` — so the origin learned nothing while the peer sat parked for
    minutes. Suppression must mean "the gate card is already on screen"; with
    no card to point at, the park is announced, card and banner.
    """
    app = OperatorApp(lambda: _factory(AttachedFake()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        notifier = RecordingNotifier()
        app._notifier = notifier  # type: ignore[assignment]
        toast = _toast(app)
        session = app._session
        assert session is not None
        assert getattr(session, "pending_gate", None) is None, (
            "this cell is about the no-gate follow shape; the gate-present shape "
            "has its own cell below"
        )

        app._note_remote_parks((_parked(),))
        await pilot.pause()

        assert toast.display, "an attached viewer with no gate card still needs the park"
        assert "Waiting for approval on demo-laptop" in toast.message
        assert (
            "Its gate card has not reached this view — deny it here once it does." in toast.message
        ), (
            "the attached reader must not be told to open the session it is looking at "
            "(design round 1, D1)"
        )
        assert notifier.kinds == ["approval"], "the banner half must fire too"


@pytest.mark.asyncio
async def test_an_attached_viewer_whose_card_never_mounted_is_still_told() -> None:
    """F-A, the other half: a pending gate the app never mounted must not suppress.

    The suppression premise is the card, so a facade that HOLDS the gate with
    no card widget on screen — a mount dropped by the ladder, a view not yet
    composed, a re-arm still owed — is not "already on screen": the notice
    fires rather than being swallowed by a card that never arrived.
    """
    app = OperatorApp(lambda: _factory(AttachedFake()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        toast = _toast(app)
        session = cast(Any, app._session)
        assert session is not None
        session.pending_gate = SimpleNamespace(
            kind="approval", request_id="req-ghost", question_index=0
        )

        app._note_remote_parks((_parked(),))
        await pilot.pause()

        assert toast.display, "a pending gate with no card suppressed the park notice"


@pytest.mark.asyncio
async def test_notifications_off_is_card_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The card needs no delivery path, so it survives the kill switch."""
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)

        app._note_remote_parks((_parked(),))
        await pilot.pause()
        assert _toast(app).display, "the card is the surface, not the banner"


@pytest.mark.asyncio
async def test_a_park_writes_nothing_to_the_attention_store(tmp_path: Path) -> None:
    """A park has no token, and no automatic path may acknowledge (design §2).

    The store holds COMPLETION receipts, bound to explicit-ack watermarks; a
    park is a transient notice plus the row's own durable ``pending``. Pinned
    as an absence because the tempting shortcut — writing a receipt so the park
    "survives a dismiss" — is real design work with its own ack semantics.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._note_remote_parks((_parked(),))
        await pilot.pause()
        assert _toast(app).display
        assert not (tmp_path / "config" / "attention.db").exists()


@pytest.mark.asyncio
async def test_the_sidebar_poll_is_the_detector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The hook the whole feature rides: the poll's own rows, no second read.

    ``read_listing`` is what ``_refresh_sidebar`` already calls; the park
    detector reads edges off its return value, so a listing that paints the
    peer rows is also what announces its park. This cell drives the REAL
    refresh (the method the 2 s timer calls) rather than ``_note_remote_parks``
    directly, which is the wiring a rename or a return-shape change would
    break silently.
    """
    from local_operator.session import peer_rows as peer_rows_module

    rows = (_parked(),)
    monkeypatch.setattr(peer_rows_module, "read_listing", lambda *a, **k: (rows, ()))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        # The refresh still owns the OPEN case; opening the list is what starts
        # it (the same resume the user's keybinding does). The CLOSED case is
        # the always-on watch's, driven in its own cell below.
        app._session_sidebar.set_open(True)
        app._refresh_sidebar()
        for _ in range(60):
            await pilot.pause()
            if _toast(app).display:
                break
        assert _toast(app).display, "the poll never announced the park"
        assert "Waiting for approval on demo-laptop" in _toast(app).message


def _silent_relay(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A relay whose one device stays silent, and a clock that outlasts the TTL.

    The injected clock is the shape that discriminates (R-1 / R2-1): a pair of
    ``peer_session_rows`` + ``unanswered_peers`` calls is only ONE read while the
    first finishes inside the cache TTL, and a silent peer is exactly the listing
    that spends it — so the second call re-dials. Every layer above the catalogue
    is production code.
    """
    from local_operator.network import projection, store
    from local_operator.session import peer_rows as peer_rows_module
    from tests.unit.server.test_desktop_remote_open import _JumpingClock
    from tests.unit.session.test_peer_rows import _Catalog, _Facts

    peer_rows_module.clear_cache()
    catalog = _Catalog(
        [_Facts("d_bb", "demo-laptop", reachable=False, reason="connect_failed:Refused")], []
    )
    monkeypatch.setattr(store, "find_own_relay", lambda root=None: object())
    monkeypatch.setattr(projection, "RelayPeerCatalog", lambda root: catalog)
    monkeypatch.setattr(peer_rows_module, "time", _JumpingClock())
    return catalog


@pytest.mark.asyncio
async def test_the_closed_list_watch_reads_the_listing_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2-1: ``_read_remote_parks`` hands the detector ONE read's rows AND silence."""
    catalog = _silent_relay(monkeypatch)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        seen: list[tuple[Any, Any]] = []
        monkeypatch.setattr(
            app, "_note_remote_parks", lambda rows, unanswered=(): seen.append((rows, unanswered))
        )
        catalog.calls = 0

        await app._read_remote_parks()

    assert catalog.calls == 1, "the watch issued a SECOND fan-out for one tick"
    ((rows, unanswered),) = seen
    assert rows == ()
    assert [peer.device_id for peer in unanswered] == ["d_bb"]


@pytest.mark.asyncio
async def test_the_sidebar_refresh_reads_the_listing_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2-1: the 2 s refresh is ONE fan-out, and its silence rides the same read."""
    catalog = _silent_relay(monkeypatch)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        seen: list[tuple[Any, Any]] = []
        monkeypatch.setattr(
            app, "_note_remote_parks", lambda rows, unanswered=(): seen.append((rows, unanswered))
        )
        app._session_sidebar.set_open(True)
        if app._sidebar_timer is not None:
            app._sidebar_timer.pause()
        catalog.calls = 0
        app._refresh_sidebar()
        for _ in range(60):
            await pilot.pause()
            if seen:
                break

    assert seen, "the refresh never reached the park detector"
    assert catalog.calls == 1, "the refresh issued a SECOND fan-out"
    assert [peer.device_id for peer in seen[0][1]] == ["d_bb"]


@pytest.mark.asyncio
async def test_a_park_while_the_list_is_closed_is_announced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The always-on watch: the DEFAULT configuration must still hear the park.

    The sidebar's poll is registered paused and pauses with the list, so with
    the app's default closed sidebar the notice used to wait for a keypress
    that might never come (agent review round 1, MAJOR-1). This drives the
    always-on tick's own method — the one the 5 s interval calls — with the
    list CLOSED, and then proves the open-list refresh does not announce the
    same episode a second time.
    """
    from local_operator.session import peer_rows as peer_rows_module

    rows = (_parked(),)
    monkeypatch.setattr(peer_rows_module, "read_listing", lambda *a, **k: (rows, ()))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        toast = _toast(app)
        assert not app._session_sidebar.display, "this cell is about the closed list"

        app._watch_remote_parks()
        for _ in range(60):
            await pilot.pause()
            if toast.display:
                break
        assert toast.display, "a park reached a closed-list origin and nothing said so"
        assert "Waiting for approval on demo-laptop" in toast.message
        assert not app._remote_park_watch_pending, "the watch left its read flag set"

        # Opening the list must not re-announce the episode the watch already
        # delivered: one notice per episode however the rows arrive.
        generation = toast.generation
        app._session_sidebar.set_open(True)
        app._refresh_sidebar()
        for _ in range(40):
            await pilot.pause()
        assert toast.generation == generation, "the open list re-raised the watch's card"


@pytest.mark.asyncio
async def test_the_always_on_watch_is_armed_and_fires_on_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 5 s watch is ARMED BY THE APP and fires without a caller (round 2).

    The cell above drives the watch's own method; this one proves the app
    actually SCHEDULES it: the arming is intercepted on the way to
    ``set_interval`` — identity against the live attribute, the shape
    ``test_attention.py`` holds its tick with, so a wrapper cannot make the pin
    vacuous — and then the card must arrive from the app's OWN tick, with
    ``_watch_remote_parks`` never called by the test. Delete the
    ``set_interval(REMOTE_PARK_POLL_S, ...)`` line in ``on_mount`` and this
    fails while the other still passes, which is exactly how round-1's
    MAJOR-1 was silent in the default configuration.
    """
    from local_operator.session import peer_rows as peer_rows_module

    rows = (_parked(),)
    monkeypatch.setattr(peer_rows_module, "read_listing", lambda *a, **k: (rows, ()))

    armed: list[tuple[float, dict[str, Any]]] = []
    original = OperatorApp.set_interval

    def spy(self: OperatorApp, *args: Any, **kwargs: Any) -> Any:
        callback = args[1] if len(args) > 1 else kwargs.get("callback")
        if callback is not None and callback == getattr(self, "_watch_remote_parks", None):
            armed.append((args[0] if args else kwargs.get("interval", 0.0), kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(OperatorApp, "set_interval", spy)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        toast = _toast(app)
        assert not app._session_sidebar.display, "this cell is about the closed list"

        assert armed, "the always-on park watch is not armed on mount"
        ((interval, kwargs),) = armed
        assert interval == REMOTE_PARK_POLL_S == 5.0
        assert not kwargs.get(
            "pause"
        ), "armed paused — it would never fire while the list is closed"

        # The app's own tick, no caller: wait past the interval (with slack for
        # a loaded host) and require the card to arrive from the timer alone.
        for _ in range(200):
            await pilot.pause(0.1)
            if toast.display:
                break
        assert toast.display, "no card arrived from the app's own 5 s watch"


@pytest.mark.asyncio
async def test_a_silent_device_does_not_withdraw_or_re_announce_its_park() -> None:
    """A non-answer is not the answer landing (agent review round 1, MINOR-1).

    ``peer_session_rows`` returns ``()`` for a refused or timed-out relay and
    for a peer marked unreachable; treating that as "the park cleared" withdrew
    the card and then re-announced the same live park as a SECOND episode when
    the peer recovered. The roster of silent devices rides with the rows, and
    the episode must survive the outage: card stays, no new card, and the
    recovery with the park gone is the one real clear.
    """
    from local_operator.session.peer_rows import UnansweredPeer

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        toast = _toast(app)
        rows = (_parked(),)

        app._note_remote_parks(rows)
        await pilot.pause()
        assert toast.display
        generation = toast.generation

        silent = (UnansweredPeer(device_id="d_bb", name="demo-laptop", reason="no answer"),)
        app._note_remote_parks((), silent)
        await pilot.pause()
        assert toast.display, "a refused read withdrew a card for a park that may still be live"
        assert toast.generation == generation, "a refused read re-raised the card"
        assert app._remote_park_episodes, "a refused read dropped the episode"

        # The peer answers again and the park is gone: THAT is the clear.
        app._note_remote_parks((), ())
        await pilot.pause()
        assert not toast.display, "the answered clear did not withdraw the card"

        # A re-park after a REAL clear is a second episode — pinned here so the
        # silence carry above cannot accidentally mute a genuine second park.
        app._note_remote_parks(rows)
        await pilot.pause()
        assert toast.display
        assert toast.generation > generation, "a re-park after a real clear re-arms"


@pytest.mark.asyncio
async def test_a_stored_unread_completion_raises_no_card() -> None:
    """The §2 discriminator, end to end: a stored row is not a park.

    The shape a producer that predates the 2026-10-07 correction shipped: a
    stored row (``live_state: ""``) carrying ``pending: "ask"`` minted from an
    unread completion. The producer no longer mints it and the row reader
    drops it (``network.types.row_needs_claim``), but a row that already
    carries the legacy claim must still not page a person for a turn that
    finished hours ago and needs nobody.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        stored = _parked(kind="ask", live_state="")
        app._note_remote_parks((stored,))
        await pilot.pause()
        assert not _toast(app).display
        assert app._remote_park_episodes == {}
