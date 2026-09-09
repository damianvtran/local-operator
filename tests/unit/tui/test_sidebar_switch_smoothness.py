"""The sidebar switch must land in ONE settled state, not converge into one.

The operator's report was four symptoms of a single unsmooth switch: the
composer changed size mid-switch, tool traces mounted into an already-visible
view and shuffled it upward, the scroll position bounced, and the whole thing
felt slow. Four root causes were measured behind them, and each assertion below
pins one of them.

WHY THERE IS NOT A SINGLE MILLISECOND IN THIS FILE. AGENTS.md, "Timing, flakes,
and how to assert that something is fast", is explicit: a ceiling calibrated on
a dev box flakes on CI, where the same sites measured 206-413 ms against a local
36-38 ms. The facts this change is really about are structural and do not move
with machine load:

* the readiness gate is never refused, so no switch pays a recovery relayout;
* no painted frame shows the composer at a height neither session asked for;
* no history rows mount into the view after it becomes visible;
* the switch paints exactly one scroll position.

Each is a fact about WHAT the app did, not how long it took, so it fails
deterministically when the code regresses and never when the box is busy.

The rig drives the real prepare/commit pair through `_switch` from
``test_sidebar_swap_reset`` — only the owner lease is stubbed; the prepared
replay, the parked outgoing view, the commit, the adopt, the reveal and the
readiness gate are all production code.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest
from textual._compositor import LayoutUpdate
from textual.widget import Widget

from local_operator.tui.app import OperatorApp
from local_operator.tui.session_interaction import SessionInteraction
from local_operator.tui.widgets.transcript import TranscriptView
from tests.unit.tui.test_app_pilot import _factory
from tests.unit.tui.test_sidebar_swap_reset import SidebarRemote, _message, _switch


@pytest.fixture(autouse=True)
def isolated_switch(tmp_path, monkeypatch):
    # A headless TUI test must never inherit the operator's multiplexer
    # identity: an inherited CMUX_WORKSPACE_ID has previously let a headless run
    # rename his real cmux workspaces.
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_terminal_title", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_multiplexer_broadcast", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_herdr_reporter", lambda _self: None)


def _conversation(session_id: str, turns: int) -> SidebarRemote:
    """A transcript several viewports deep, so the resume fill has work to do.

    The under-fill that causes the post-reveal shuffle is a property of a
    conversation LONGER than the prepared window (`bound = max(12, height//2)`),
    so a short fixture cannot exercise it at all.

    Every message carries an ``id``: `PreparedReplay` derives a block's
    navigation anchor from the projecting message's id, so a fixture without one
    produces a transcript with no anchors and cannot exercise the saved-position
    path at all.
    """
    history = []
    for turn in range(turns):
        user = _message("user", f"question {turn} about the switch path")
        user.id = f"{session_id}-u{turn}"
        assistant = _message("assistant", f"answer {turn}, with some detail to render")
        assistant.id = f"{session_id}-a{turn}"
        history.extend((user, assistant))
    return SidebarRemote(session_id, history=history)


async def _switch_bound(app: OperatorApp, pilot, remote: SidebarRemote) -> None:
    """`_switch`, with the source registered the way the REAL lease registers it.

    `_adopt_session` looks a source up by session identity through
    ``_interactions[id(session)]``. A fixture that skips that registration makes
    the app adopt a DIFFERENT `SessionInteraction` than the one the readiness
    gate is bound to, so `_is_current(source)` is False for the whole switch —
    `post_display_hook` returns before its recovery branch and the gate can
    never be satisfied. The bench harness documents the same trap, where it
    reads as a 15 s timeout on every cold switch.

    Anything asserting on the GATE has to go through here; `_switch` alone is
    enough for assertions about widgets and drafts.
    """
    source = app._sidebar_sources.get(remote.session_id)
    if source is None:
        source = SessionInteraction(remote)
        app._sidebar_sources[remote.session_id] = source
    app._interactions[id(remote)] = source
    await _switch(app, pilot, remote)


class _FrameRecorder:
    """Every painted frame, with the geometry the complaints are about.

    `App._display` is the single funnel every compositor refresh passes
    through, so this is the app's own paint sequence rather than a poll. It is
    installed on the INSTANCE, never on the class, so a failure cannot leak into
    a sibling test.
    """

    def __init__(self, app: OperatorApp) -> None:
        self.app = app
        self.frames: list[dict] = []
        self.armed = False
        self.revealed = False
        self._real_display = app._display

        def display(screen, renderable):  # type: ignore[no-untyped-def]
            out = self._real_display(screen, renderable)
            if self.armed:
                self.frames.append(self._sample(renderable))
            return out

        app._display = display  # type: ignore[method-assign]

    def _sample(self, renderable) -> dict:  # type: ignore[no-untyped-def]
        sample: dict = {"layout": isinstance(renderable, LayoutUpdate)}
        try:
            sample["editor_height"] = int(self.app._editor().outer_size.height)
        except Exception:  # noqa: BLE001 - a mid-swap query can find nothing
            sample["editor_height"] = None
        if self.revealed:
            try:
                view = self.app._transcript_view()
                sample["scroll_y"] = float(view.scroll_y)
                sample["blocks"] = len(view.blocks())
            except Exception:  # noqa: BLE001
                pass
        return sample

    @property
    def editor_heights(self) -> list[int]:
        return [f["editor_height"] for f in self.frames if f.get("editor_height") is not None]

    @property
    def scroll_positions(self) -> list[float]:
        return [f["scroll_y"] for f in self.frames if "scroll_y" in f]


@pytest.mark.asyncio
async def test_switch_never_refuses_the_readiness_gate() -> None:
    """RC1: the first post-commit paint must be evidence the gate can accept.

    `_sidebar_displayed_frame` is only recorded for a `LayoutUpdate`, and the
    arming refresh used to be satisfied by a partial `ChopsUpdate` — so the gate
    refused on the first paint of EVERY switch (21/21 measured) and
    `post_display_hook` bought a second full-screen relayout to recover. The fix
    is to make the arming refresh PRODUCE a full update, never to let the gate
    accept weaker evidence: `_sidebar_gate_surface_ready` still requires a real
    painted compositor map covering the target surfaces.

    `_sidebar_gate_recoveries` counts entries into that recovery branch. The
    branch is deliberately KEPT — the container-lands-before-its-children case
    it was written for is real — so this asserts it stays dead, not that it is
    gone.
    """
    home = SidebarRemote("home-session")
    targets = [_conversation(f"target-{i}", 40) for i in range(3)]

    app = OperatorApp(lambda: _factory(home))
    with patch("local_operator.session.remote.RemoteSession", SidebarRemote):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()

            for remote in targets:
                await _switch_bound(app, pilot, remote)
            # And back again: a return leg hits the retained-presentation path,
            # which reaches the gate by a different route than a first visit.
            for remote in targets:
                await _switch_bound(app, pilot, remote)

            # The gate must actually have been REACHED, or a zero below would
            # mean "never armed" rather than "never refused".
            assert app._sidebar_gate_reached > 0, "no switch reached the readiness gate"
            assert app._sidebar_gate_recoveries == 0, (
                "the readiness gate was refused and paid a recovery relayout; "
                "the arming refresh is no longer producing a full LayoutUpdate"
            )


@pytest.mark.asyncio
async def test_the_composer_never_paints_a_height_neither_session_asked_for() -> None:
    """RC3: no off-state composer frame.

    Both sessions carry the SAME multi-line draft, so the steady-state composer
    height is identical before and after. Any other height in the painted
    sequence is therefore transition jitter and not a legitimate per-session
    draft handoff — which is exactly what emptying the buffer at `pending`
    produced: the composer collapsed to one row, the docked input shrank with
    it, and the transcript above reflowed into the vacated rows.
    """
    home = SidebarRemote("home-session")
    first = _conversation("draft-a", 30)
    second = _conversation("draft-b", 30)
    draft = "first line of the draft\nsecond line\nthird line"

    app = OperatorApp(lambda: _factory(home))
    with patch("local_operator.session.remote.RemoteSession", SidebarRemote):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()
            await _switch(app, pilot, first)
            for _ in range(20):
                await pilot.pause()

            # Give BOTH sides the same draft: the one the editor is holding, and
            # the one the target will be handed at commit.
            editor = app._editor()
            editor.load_text(draft)
            for _ in range(20):
                await pilot.pause()
            incoming = app._sidebar_sources.get(second.session_id)
            if incoming is None:
                incoming = SessionInteraction(second)
                app._sidebar_sources[second.session_id] = incoming
            incoming.draft.text = draft

            settled_height = int(editor.outer_size.height)
            assert settled_height > 1, "the fixture draft must not be one row"

            recorder = _FrameRecorder(app)
            recorder.armed = True
            # The pending hook is what a click runs before the navigation task;
            # it is where the buffer used to be emptied.
            app._sidebar_navigation_pending(second.session_id)
            for _ in range(20):
                await pilot.pause()
            await _switch(app, pilot, second)
            recorder.armed = False

            off_state = [h for h in recorder.editor_heights if h != settled_height]
            assert not off_state, (
                f"the composer painted {sorted(set(off_state))} rows during a switch whose "
                f"steady state is {settled_height} on both sides"
            )


@pytest.mark.asyncio
async def test_no_history_rows_mount_into_the_visible_view() -> None:
    """RC4: the resume fill runs before reveal, so nothing arrives afterwards.

    The prepared window under-fills the viewport by construction, so the fill
    fired on essentially every switch and mounted 24 further rows into a view
    the user was already looking at — 53% of the conversation appearing after
    the fact, which is the "traces shuffle upward" report.

    Counted on the VISIBLE view only: rows appended to a parked offscreen replay
    are the preparation doing its job.
    """
    home = SidebarRemote("home-session")
    target = _conversation("deep-session", 60)

    app = OperatorApp(lambda: _factory(home))
    mounted_after_reveal: list[int] = []
    revealed = {"yes": False}

    real_insert = TranscriptView.insert_blocks

    def insert_blocks(self, index, blocks, **kwargs):  # type: ignore[no-untyped-def]
        if revealed["yes"]:
            try:
                if self is app._transcript_view():
                    mounted_after_reveal.append(len(list(blocks)))
            except Exception:  # noqa: BLE001
                pass
        return real_insert(self, index, blocks, **kwargs)

    with patch("local_operator.session.remote.RemoteSession", SidebarRemote), patch.object(
        TranscriptView, "insert_blocks", insert_blocks
    ):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()

            real_commit = app._commit_sidebar_session

            def commit(session_id, prepared, generation):  # type: ignore[no-untyped-def]
                out = real_commit(session_id, prepared, generation)
                revealed["yes"] = True
                return out

            app._commit_sidebar_session = commit  # type: ignore[method-assign]
            await _switch(app, pilot, target)
            for _ in range(60):
                await pilot.pause()

            assert not mounted_after_reveal, (
                f"{sum(mounted_after_reveal)} history rows mounted into the visible view "
                "after it was revealed; the pre-reveal fill is not reaching its goal"
            )


@pytest.mark.asyncio
async def test_a_switch_paints_exactly_one_scroll_position() -> None:
    """RC4/§6.1(6): the scroll lands once, with no bounce.

    Two painted positions is the operator's "scroll position jitter": the view
    appeared at one offset and was dragged to another a frame or two later, by
    `_size_updated` reacting to rows arriving after reveal. With the fill landed
    before reveal there is nothing left to move it, so the sequence is
    single-valued.

    The session is left at TAIL, which is the case the operator called out
    explicitly: it must land on the most recent message, cleanly.
    """
    home = SidebarRemote("home-session")
    target = _conversation("tail-session", 60)

    app = OperatorApp(lambda: _factory(home))
    with patch("local_operator.session.remote.RemoteSession", SidebarRemote):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()

            recorder = _FrameRecorder(app)
            real_commit = app._commit_sidebar_session

            def commit(session_id, prepared, generation):  # type: ignore[no-untyped-def]
                out = real_commit(session_id, prepared, generation)
                recorder.revealed = True
                return out

            app._commit_sidebar_session = commit  # type: ignore[method-assign]
            recorder.armed = True
            await _switch(app, pilot, target)
            for _ in range(60):
                await pilot.pause()
            recorder.armed = False

            positions = recorder.scroll_positions
            assert positions, "no painted frame carried a scroll position"
            assert len(set(positions)) == 1, (
                f"the switch painted {sorted(set(positions))} — the view moved after it was "
                "already on screen"
            )
            view = app._transcript_view()
            assert view.is_following_tail, "a session left at tail did not land at the tail"


@pytest.mark.asyncio
async def test_leaving_at_tail_clears_the_saved_anchor() -> None:
    """§5.3: `following_tail` and a stale anchor must not both be recorded.

    A reader who scrolls up and then scrolls BACK to the tail wrote an anchor on
    the way up and nothing cleared it on the way down, so the draft claimed both
    "I am following the tail" and "restore me to this block at offset -1". Every
    consumer today tests the flag first, which is what makes it latent rather
    than visible — and is exactly why it is worth pinning: the next reader of
    `scroll_anchor_id` that forgets the flag lands the user at a position they
    left, with a nonsense offset that clamps into silent wrongness.
    """
    home = SidebarRemote("home-session")
    target = _conversation("anchor-session", 60)

    app = OperatorApp(lambda: _factory(home))
    with patch("local_operator.session.remote.RemoteSession", SidebarRemote):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()
            await _switch(app, pilot, target)
            for _ in range(40):
                await pilot.pause()

            view = app._transcript_view()
            source = app._interaction

            # Scroll UP, which is what writes an anchor into the draft.
            view.scroll_to(y=max(0.0, view.max_scroll_y - 8), animate=False, immediate=True)
            app._transcript_scrolled(True)
            for _ in range(20):
                await pilot.pause()
            app._capture_sidebar_scroll(source)
            assert source.draft.following_tail is False
            assert source.draft.scroll_anchor_id, "the fixture never produced an anchor to go stale"

            # Now scroll back DOWN to the tail and leave.
            view.scroll_to(y=view.max_scroll_y, animate=False, immediate=True)
            app._transcript_scrolled(False)
            for _ in range(20):
                await pilot.pause()
            app._capture_sidebar_scroll(source)

            assert source.draft.following_tail is True
            assert source.draft.scroll_anchor_id == "", (
                "a session left at the tail kept a stale scroll anchor"
            )
            assert source.draft.scroll_offset == 0


@pytest.mark.asyncio
async def test_text_typed_during_a_transition_belongs_to_the_target() -> None:
    """RC3's attribution contract, which the frozen prefix must not change.

    The composer no longer empties at `pending`, so the buffer visibly holds the
    OUTGOING draft while the switch runs. What the user types lands after it,
    and `_take_sidebar_transition_buffer` subtracts the frozen prefix so the
    suffix — and only the suffix — is attributed to the session being opened.
    """
    home = SidebarRemote("home-session")
    first = _conversation("attribution-a", 20)
    second = _conversation("attribution-b", 20)

    app = OperatorApp(lambda: _factory(home))
    with patch("local_operator.session.remote.RemoteSession", SidebarRemote):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()
            await _switch(app, pilot, first)
            for _ in range(20):
                await pilot.pause()

            editor = app._editor()
            editor.load_text("outgoing draft")
            for _ in range(10):
                await pilot.pause()
            outgoing = app._interaction

            app._sidebar_navigation_pending(second.session_id)
            for _ in range(10):
                await pilot.pause()

            # The buffer still PAINTS the outgoing draft — that is the whole
            # point — and the snapshot has been taken.
            assert editor.text == "outgoing draft"
            assert outgoing.draft.text == "outgoing draft"

            editor.load_text("outgoing draft and what I typed after clicking")
            for _ in range(10):
                await pilot.pause()

            typed, _attachments = app._take_sidebar_transition_buffer()
            assert typed == " and what I typed after clicking", (
                "the frozen prefix was not subtracted; the outgoing draft would be "
                "attributed to the target"
            )


@pytest.mark.asyncio
async def test_an_abandoned_transition_restores_the_draft_exactly_once() -> None:
    """The failure leg: the user stays put, so text is theirs — and is not doubled.

    `_abandon_sidebar_transition` assembles `draft.text + typed`. With the
    buffer no longer emptied at `pending` it holds both halves already, so
    taking the buffer BEFORE clearing the prefix is what keeps the restore from
    concatenating the outgoing draft onto itself.
    """
    home = SidebarRemote("home-session")
    first = _conversation("abandon-a", 20)

    app = OperatorApp(lambda: _factory(home))
    with patch("local_operator.session.remote.RemoteSession", SidebarRemote):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()
            await _switch(app, pilot, first)
            for _ in range(20):
                await pilot.pause()

            editor = app._editor()
            editor.load_text("half a sentence")
            for _ in range(10):
                await pilot.pause()

            app._sidebar_navigation_pending("some-other-session")
            for _ in range(10):
                await pilot.pause()
            editor.load_text("half a sentence, and the rest")
            for _ in range(10):
                await pilot.pause()

            # `pending("")` with no commit in between: the switch failed.
            app._sidebar_navigation_pending("")
            for _ in range(10):
                await pilot.pause()

            assert editor.text == "half a sentence, and the rest", (
                "the abandoned transition did not restore the draft exactly once"
            )
            assert app._sidebar_transition_from is None
            assert app._sidebar_transition_prefix == ""


@pytest.mark.asyncio
async def test_a_click_burst_keeps_the_original_snapshot() -> None:
    """Re-entering with a transition already open must not re-snapshot.

    A burst of clicks calls the pending hook repeatedly. The first call owns the
    snapshot; a later one would otherwise capture a buffer that already contains
    transition typing and record it as the outgoing session's draft.
    """
    home = SidebarRemote("home-session")
    first = _conversation("burst-a", 20)

    app = OperatorApp(lambda: _factory(home))
    with patch("local_operator.session.remote.RemoteSession", SidebarRemote):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()
            await _switch(app, pilot, first)
            for _ in range(20):
                await pilot.pause()

            editor = app._editor()
            editor.load_text("original draft")
            for _ in range(10):
                await pilot.pause()
            outgoing = app._interaction

            app._sidebar_navigation_pending("target-one")
            for _ in range(10):
                await pilot.pause()
            editor.load_text("original draft plus typing")
            for _ in range(10):
                await pilot.pause()
            # Second click of the burst, transition already open.
            app._sidebar_navigation_pending("target-two")
            for _ in range(10):
                await pilot.pause()

            assert outgoing.draft.text == "original draft", (
                "a re-entrant click overwrote the frozen snapshot"
            )
            assert app._sidebar_transition_prefix == "original draft"
            typed, _ = app._take_sidebar_transition_buffer()
            assert typed == " plus typing"


@pytest.mark.asyncio
async def test_the_parked_view_stays_non_interactive() -> None:
    """RC2: skipping the stylesheet cascade must not make a parked view live.

    `Widget.disabled` does two jobs — interactivity gating, which reads the
    field, and appearance, which is the whole-subtree `Stylesheet.apply` that
    `watch_disabled` triggers. Only the second is skipped, and it is invisible
    on a view parked at `offset: 100vw`. The first is what keeps a clipped
    transcript from taking focus, scroll or mouse events, so it is asserted
    directly rather than trusted.
    """
    home = SidebarRemote("home-session")
    first = _conversation("park-a", 20)
    second = _conversation("park-b", 20)

    app = OperatorApp(lambda: _factory(home))
    with patch("local_operator.session.remote.RemoteSession", SidebarRemote):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()
            await _switch(app, pilot, first)
            for _ in range(20):
                await pilot.pause()
            parked = app._transcript_view()

            await _switch(app, pilot, second)
            for _ in range(40):
                await pilot.pause()
            live = app._transcript_view()
            assert live is not parked, "the switch did not change the visible transcript"

            assert parked.disabled is True
            assert parked._self_or_ancestors_disabled is True
            assert parked.focusable is False
            assert parked.allow_vertical_scroll is False

            # And the revealed view is genuinely interactive again: an un-park
            # that only skipped the cascade would leave the field set.
            assert live.disabled is False
            assert live.focusable is True
            assert live.allow_vertical_scroll is True


def test_parking_a_transcript_skips_the_stylesheet_cascade() -> None:
    """The cost half of RC2, asserted structurally rather than in milliseconds.

    `watch_disabled` re-applies CSS across the whole subtree — 131 node
    applications on a real transcript, 10.2 ms of the 18.8 ms commit, inside a
    section that deliberately never awaits. The fact that matters is that the
    watcher does not run at all, which is a property of the call and not of the
    machine: `set_reactive` writes the value without invoking watchers.
    """
    from local_operator.tui.app import _set_transcript_parked

    calls: list[bool] = []

    class _Probe(Widget):
        def watch_disabled(self, disabled: bool) -> None:  # pragma: no cover - must not run
            calls.append(disabled)

    probe = _Probe()
    _set_transcript_parked(probe, True)
    assert probe.disabled is True
    _set_transcript_parked(probe, False)
    assert probe.disabled is False
    assert calls == [], "the disabled watcher ran, re-applying the stylesheet subtree"


@pytest.mark.asyncio
async def test_a_failed_pre_reveal_fill_does_not_invalidate_the_preparation() -> None:
    """The pre-reveal fill runs after the invalidation check, so it may not raise.

    `PreparationInvalidated` releases the prepared widgets and re-prepares. The
    pre-reveal fill runs inside the commit, AFTER ownership has already moved,
    so a raise out of `_prefill_resume_before_reveal` would tear down a
    presentation that is already live.

    SCOPE OF THIS GUARD, stated so it is not read as more than it is: it covers
    the SYNCHRONOUS pre-reveal pass this change added. The ordinary deferred
    fill scheduled afterwards still propagates a projection failure the way it
    always has — `_fill_resume_until_scrollable` re-raises by design, to keep
    the page retryable rather than silently losing its cursor — and that path is
    reached from a `call_after_refresh` callback, long after the commit has
    returned, so it cannot reach the invalidation retry either. Changing it is
    not in this change's scope.
    """
    home = SidebarRemote("home-session")
    target = _conversation("raising-session", 60)

    app = OperatorApp(lambda: _factory(home))
    with patch("local_operator.session.remote.RemoteSession", SidebarRemote):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()

            calls: list[int] = []

            def explode(*_args, **_kwargs):
                calls.append(1)
                raise RuntimeError("projection failed")

            # Only the synchronous pre-reveal pass is exercised: it is called
            # directly, exactly as the commit calls it, with the deferred
            # follow-up stubbed out so a later raise cannot be mistaken for
            # this one.
            app._mount_older_resume_page = explode  # type: ignore[method-assign]
            app._start_resume_fill = lambda **_kw: None  # type: ignore[method-assign]

            await _switch(app, pilot, target)
            for _ in range(40):
                await pilot.pause()

            # The commit completed and the conversation is usable.
            assert app._session is not None
            assert app._session.session_id == target.session_id
            assert app._transcript_view().blocks()
            assert calls, "the pre-reveal fill never reached the projection at all"
