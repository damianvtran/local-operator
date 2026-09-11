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
* the switch paints exactly one scroll position;
* no painted frame shows an empty conversation.

Each is a fact about WHAT the app did, not how long it took, so it fails
deterministically when the code regresses and never when the box is busy.

TWO OF THESE ARE VIEWPORT-CONDITIONAL, and the sizes below are therefore
load-bearing rather than incidental (QA rounds 1 and 2, Q1/Q2/Q4):

* **Zero gate refusals needs BOTH enough columns and enough rows.** An earlier
  version of this note said ">=41 columns" and QA round 2 replied "no, it is
  height"; both are half right, and the single-axis phrasing is what sent a
  reader to the wrong axis. Measured on this rig, sweeping one axis at a time:

  ===========  ============  ===========================================
  size         recoveries    what it falsifies
  ===========  ============  ===========================================
  41x30        0             --
  40x30        5             not "any height >= 22 passes"
  120x22       0             --
  120x20       9             not "any width >= 41 passes"
  41x20        13            wide enough, still refuses -> height matters
  30x36        13            tall enough, still refuses -> width matters
  ===========  ============  ===========================================

  So it is a viewport FLOOR on both axes, not a threshold on either one: at 30
  rows the boundary sits between 40 and 41 columns, and at 120 columns it sits
  between 20 and 22 rows. Below it the gate refuses again
  (``TAIL_BLOCK_UNPAINTED`` and a 2-row ``TAIL_BLOCK_OVERFLOWS_CONTENT``) and
  the recovery branch does exactly the job it exists for. The BASE behaves
  identically in every cell measured, so this is reach, not regression, and the
  switch still completes and lands correctly at every size tried.
* **Exactly one painted scroll position holds at >=70 columns.** Below that the
  switch paints two positions 1-2 rows apart, with ZERO post-reveal inserts --
  a final geometry settle, not content arriving late. The base at the same
  sizes paints ``[15.0, 63.0]`` / ``[13.0, 14.0, 62.0]`` with 24 rows mounting
  after reveal, so the operator's actual complaint (the large upward shuffle)
  is gone at every width.

If either test fails at a SMALLER viewport than the one it declares, that is a
genuine regression at that size -- do not "fix" it by enlarging the fixture. If
one fails at the declared size, check both axes before concluding anything: the
history of this note is two confident single-axis claims that were each wrong.

The rig drives the real prepare/commit pair through `_switch` from
``test_sidebar_swap_reset`` — only the owner lease is stubbed; the prepared
replay, the parked outgoing view, the commit, the adopt, the reveal and the
readiness gate are all production code.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from textual._compositor import LayoutUpdate
from textual.widget import Widget

from local_operator.tui.app import OperatorApp, _suppress_intermediate_paint
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
        self.frames: list[dict[str, object]] = []
        self.armed = False
        self.revealed = False
        self._real_display = app._display

        def display(screen, renderable):  # type: ignore[no-untyped-def]
            out = self._real_display(screen, renderable)
            if self.armed:
                self.frames.append(self._sample(renderable))
            return out

        app._display = display  # type: ignore[method-assign]

    def _sample(self, renderable) -> dict[str, object]:  # type: ignore[no-untyped-def]
        sample: dict[str, object] = {"layout": isinstance(renderable, LayoutUpdate)}
        try:
            sample["editor_height"] = int(self.app._editor().outer_size.height)
        except Exception:  # noqa: BLE001 - a mid-swap query can find nothing
            sample["editor_height"] = None
        if self.revealed:
            try:
                view = self.app._transcript_view()
                sample["scroll_y"] = float(view.scroll_y)
                blocks = view.blocks()
                sample["blocks"] = len(blocks)
                # BLOCKS THE USER CAN ACTUALLY SEE, not blocks mounted. The two
                # diverge exactly when a frame is laid out against geometry it
                # is not drawn in, which is the D1 blank frame: 46 mounted, 0
                # intersecting the content region.
                content = view.content_region
                sample["blocks_in_view"] = sum(
                    1
                    for block in blocks
                    if (region := getattr(block, "region", None)) is not None
                    and region.area
                    and content.overlaps(region)
                )
            except Exception:  # noqa: BLE001
                pass
        return sample

    @property
    def editor_heights(self) -> list[int]:
        return [
            int(height) for f in self.frames if isinstance(height := f.get("editor_height"), int)
        ]

    @property
    def scroll_positions(self) -> list[float]:
        return [
            float(scroll)
            for f in self.frames
            if isinstance(scroll := f.get("scroll_y"), (int, float))
        ]

    @property
    def blank_frames(self) -> list[int]:
        """Indices of painted frames that had rows mounted but none visible."""
        return [
            index
            for index, f in enumerate(self.frames)
            if isinstance(in_view := f.get("blocks_in_view"), int)
            and in_view == 0
            and isinstance(mounted := f.get("blocks"), int)
            and mounted > 0
        ]


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
    with patch("local_operator.session.attached.AttachedSession", SidebarRemote):
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
    with patch("local_operator.session.attached.AttachedSession", SidebarRemote):
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
@pytest.mark.parametrize("size", [(100, 30), (200, 50)])
async def test_no_history_rows_mount_into_the_visible_view(size: tuple[int, int]) -> None:
    """RC4: the resume fill runs before reveal, so nothing arrives afterwards.

    The prepared window under-fills the viewport by construction, so the fill
    fired on essentially every switch and mounted 24 further rows into a view
    the user was already looking at — 53% of the conversation appearing after
    the fact, which is the "traces shuffle upward" report.

    Counted on the VISIBLE view only: rows appended to a parked offscreen replay
    are the preparation doing its job.

    TWO TERMINAL HEIGHTS, because the pre-reveal pass mounts exactly ONE page
    and "one page is enough" is a claim about geometry, not a law (agent review
    round 1, R3). A taller terminal needs more rows to fill the same viewport,
    so 200x50 is where the bound would break first if `RESUME_PAGE_MESSAGES`
    ever stopped covering it; the 30-row case is the one the audit measured
    (21 blocks against a 27-row viewport).
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

    with (
        patch("local_operator.session.attached.AttachedSession", SidebarRemote),
        patch.object(TranscriptView, "insert_blocks", insert_blocks),
    ):
        async with app.run_test(size=size) as pilot:
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
    with patch("local_operator.session.attached.AttachedSession", SidebarRemote):
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
    with patch("local_operator.session.attached.AttachedSession", SidebarRemote):
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
            assert (
                source.draft.scroll_anchor_id == ""
            ), "a session left at the tail kept a stale scroll anchor"
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
    with patch("local_operator.session.attached.AttachedSession", SidebarRemote):
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
    with patch("local_operator.session.attached.AttachedSession", SidebarRemote):
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

            assert (
                editor.text == "half a sentence, and the rest"
            ), "the abandoned transition did not restore the draft exactly once"
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
    with patch("local_operator.session.attached.AttachedSession", SidebarRemote):
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

            assert (
                outgoing.draft.text == "original draft"
            ), "a re-entrant click overwrote the frozen snapshot"
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
    with patch("local_operator.session.attached.AttachedSession", SidebarRemote):
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
    applications on a real transcript, inside a section that deliberately never
    awaits. The fact that matters is that the watcher does not run ON THE PARK,
    which is a property of the call and not of the machine: `set_reactive`
    writes the value without invoking watchers. (The millisecond figures behind
    that count were taken on a loaded box and are indicative only; the count is
    the invariant.)

    ONLY THE PARK. The un-park deliberately DOES run the watcher — see
    `test_a_revealed_transcript_is_never_left_dimmed` for why skipping it left
    revealed transcripts at `opacity: 0.7`. The saving this asserts is on the
    outgoing 45-81 block transcript, which is where the cost was measured.
    """
    from local_operator.tui.app import _set_transcript_parked

    calls: list[bool] = []

    class _Probe(Widget):
        def watch_disabled(self, disabled: bool) -> None:  # pragma: no cover - must not run
            calls.append(disabled)

    probe = _Probe()
    _set_transcript_parked(probe, True)
    assert probe.disabled is True
    assert calls == [], "the disabled watcher ran on the park, re-applying the subtree"
    _set_transcript_parked(probe, False)
    assert probe.disabled is False


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
    with patch("local_operator.session.attached.AttachedSession", SidebarRemote):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()

            calls: list[int] = []

            def explode(*_args, **_kwargs):
                calls.append(1)
                raise RuntimeError("projection failed")

            # The deferred fill is RECORDED, not stubbed to a no-op. Stubbing it
            # out was this test's blind spot (agent review round 1, R2): with
            # the call swallowed, the test still passed while `chained = True`
            # was being set BEFORE the mount, so a raising mount left the
            # `finally`'s `if not chained:` already False and the documented
            # degradation path never ran. Recording it means the promise in the
            # docstring -- "a projection failure must degrade to the deferred
            # fill" -- is what is actually asserted.
            # Recorded and then STOPPED. Letting the real deferred fill run
            # would re-enter the same stubbed `_mount_older_resume_page` and
            # raise from `_fill_resume_until_scrollable`, which re-raises by
            # design (see the scope note above) -- a second, unrelated failure
            # on a path this test does not own. What R2 is about is whether the
            # degradation path is REACHED, so that is what is recorded.
            deferred: list[int] = []

            def record_start(**_kwargs):
                deferred.append(1)

            app._mount_older_resume_page = explode  # type: ignore[method-assign]
            app._start_resume_fill = record_start  # type: ignore[method-assign]

            await _switch(app, pilot, target)
            for _ in range(40):
                await pilot.pause()

            # The commit completed and the conversation is usable.
            assert app._session is not None
            assert app._session.session_id == target.session_id
            assert app._transcript_view().blocks()
            assert calls, "the pre-reveal fill never reached the projection at all"
            assert deferred, (
                "the pre-reveal fill raised and the deferred fill never started; "
                "the documented degradation path is dead (see R2)"
            )


@pytest.mark.asyncio
async def test_no_painted_frame_shows_an_empty_conversation() -> None:
    """D1: the switch must never paint a frame with rows mounted and none visible.

    The pre-reveal fill needs a synchronous layout so a freshly mounted page can
    author its height before the readiness gate reads the painted map. But
    `Screen._refresh_layout()` ends in `_compositor_refresh()`, which paints
    immediately — and at that moment the transcript has been revealed while the
    composer still occupies the OUTGOING draft's rows. Content laid out against
    a box that is about to shrink falls outside the viewport, so the frame
    showed an EMPTY conversation: 46 blocks mounted, 0 intersecting the content
    region, at `scroll_y = 84 = max_scroll_y` immediately before a settled 82.

    On a warm switch that was the FIRST painted frame — the fastest, most
    common switch opening on a blank screen — and at 80x24 the whole screen
    blanked, sidebar included (design review round 1, D1). The base never
    painted such a frame, so this was a regression introduced by the fix, and a
    blank flash is a worse unpolish signal than the shuffle it replaced.

    WHAT THIS ASSERTS. Both halves of the defect, directly: no painted frame
    inside the commit shows rows mounted with none in view, AND the commit runs
    no painting refresh at all. The second implies the first here, but they fail
    with different diagnostics -- the frame assertion names the symptom the user
    reported, the refresh assertion names the mechanism -- so both are kept.

    AN EARLIER VERSION OF THIS DOCSTRING WAS WRONG, and the correction is worth
    recording because it would otherwise talk the next author out of the
    stronger assertion. It claimed the commit "paints NOTHING" under `run_test`
    and that a frame-sequence assertion would therefore be vacuous. That was a
    measurement taken with the fix LIVE: zero paints in the commit is precisely
    what the fix produces, and reading that as "the harness cannot see paints"
    confused the fix working with the rig being blind (QA round 2, Q5). With
    the suppression neutered -- the real `_compositor_refresh` restored inside
    the stand-in, every other line untouched -- this rig reports
    `displays_in_commit=1` with `mounted=44, in_view=0`. The frame sequence is
    observable and non-vacuous, so it is asserted below.

    The rendered-frame evidence still lives on the PR (design round 1 D1 and its
    remediation; QA round 2 measured 0 blank frames across 18 cells on head
    against 11 on the baseline). AGENTS.md is explicit that a green test is not
    visual evidence: this test is the structural half of that pair, not a
    replacement for it.
    """
    home = SidebarRemote("home-session")
    target = _conversation("blank-frame-session", 60)

    app = OperatorApp(lambda: _factory(home))
    with patch("local_operator.session.attached.AttachedSession", SidebarRemote):
        # 120x36, not the 100x30 the sibling tests use: with a 3-row outgoing
        # draft the smaller terminal leaves a 16-row viewport that the prepared
        # window ALREADY fills, so the pre-fill correctly early-returns and the
        # settle this guard is about never runs (measured: scroll_y 17 against
        # a 16-row viewport, 18 blocks in and 18 out). At 120x36 the same draft
        # leaves a 25-row viewport and the fill mounts 20 -> 44 blocks.
        async with app.run_test(size=(120, 36)) as pilot:
            for _ in range(20):
                await pilot.pause()

            # A multi-row outgoing draft against the target's empty one: the
            # stale box only exists while the composer is about to CHANGE
            # height, so this is what makes the dock shrink across the commit.
            editor = app._editor()
            editor.load_text("outgoing draft line one\nline two\nline three")
            for _ in range(20):
                await pilot.pause()
            assert int(editor.outer_size.height) > 1, (
                "the outgoing draft must make the composer taller than the "
                "incoming session's, or there is no height change to lag"
            )

            # Observe the COMPOSITOR REFRESH, which is the thing that paints,
            # rather than the layout that calls it: the pre-fill reaches
            # `_refresh_layout` by two routes (the first-visit viewport probe
            # and the post-mount settle) and the fix guards both by rebinding
            # this attribute for the duration of the call.
            layouts: list[int] = []
            paints: list[int] = []
            screen = app.screen
            real_layout = screen._refresh_layout
            in_commit = {"yes": False}

            def refresh_layout(*args, **kwargs):  # type: ignore[no-untyped-def]
                if in_commit["yes"]:
                    layouts.append(1)
                    if screen._compositor_refresh is not _suppress_intermediate_paint:
                        paints.append(1)
                return real_layout(*args, **kwargs)

            screen._refresh_layout = refresh_layout  # type: ignore[method-assign]

            # THE SYMPTOM ITSELF: every frame painted while the commit is on the
            # stack, scored the way the user experiences it -- rows mounted but
            # none intersecting the content region is a conversation the user
            # sees as empty.
            commit_frames: list[dict[str, int]] = []
            real_display = app._display

            def display(target_screen, renderable):  # type: ignore[no-untyped-def]
                out = real_display(target_screen, renderable)
                if in_commit["yes"]:
                    try:
                        view = app._transcript_view()
                        blocks = view.blocks()
                        content = view.content_region
                        commit_frames.append(
                            {
                                "mounted": len(blocks),
                                "in_view": sum(
                                    1
                                    for block in blocks
                                    if (region := getattr(block, "region", None)) is not None
                                    and region.area
                                    and content.overlaps(region)
                                ),
                            }
                        )
                    except Exception:  # noqa: BLE001 - a mid-swap query can find nothing
                        pass
                return out

            app._display = display  # type: ignore[method-assign]
            real_commit = app._commit_sidebar_session

            def commit(session_id, prepared, generation):  # type: ignore[no-untyped-def]
                in_commit["yes"] = True
                try:
                    return real_commit(session_id, prepared, generation)
                finally:
                    in_commit["yes"] = False

            app._commit_sidebar_session = commit  # type: ignore[method-assign]
            await _switch(app, pilot, target)
            for _ in range(60):
                await pilot.pause()

            assert layouts, (
                "the commit ran no synchronous layout at all; the pre-reveal "
                "settle this asserts about is not happening"
            )
            assert app._resume_fill_active or app._transcript_view().blocks(), (
                "the pre-reveal fill never engaged, so the layouts counted above "
                "are not the ones this guard is about"
            )
            assert not paints, (
                f"{len(paints)} of {len(layouts)} synchronous layouts inside the commit "
                "would have painted; the switch can put a half-arranged frame "
                "(revealed transcript, outgoing-sized composer) on screen"
            )
            blank = [
                index
                for index, frame in enumerate(commit_frames)
                if frame["in_view"] == 0 and frame["mounted"] > 0
            ]
            assert not blank, (
                f"frames {blank} of {len(commit_frames)} painted inside the commit showed "
                f"rows mounted with none in view ({[commit_frames[i] for i in blank]}); "
                "the user sees an empty conversation mid-switch"
            )


@pytest.mark.asyncio
async def test_a_revealed_transcript_is_never_left_dimmed() -> None:
    """R1: un-parking must run the stylesheet cascade, even though parking skips it.

    `_set_transcript_parked` skips `watch_disabled` to avoid a whole-subtree
    re-apply inside the frozen commit. That is safe in one direction only: the
    park declines to dim a view nobody can see, but skipping the UN-park cascade
    would decline to UN-dim one the user is about to look at.

    The rule that dims it already exists — Textual's own
    `*:disabled:can-focus { opacity: 0.7 }` matches because
    `TranscriptView.can_focus` is True. Any cascade while the view is parked (a
    terminal resize, a theme change) applies it, and without a watcher on the
    un-park the view is revealed still carrying 0.7 and never self-heals: the
    reviewer measured it surviving a reveal, a keystroke, a scroll and a further
    switch, visibly dimming the body text in the rendered palette.

    Asserted here as the appearance half that
    `test_the_parked_view_stays_non_interactive` deliberately does not cover.
    """
    from local_operator.tui.app import _set_transcript_parked

    applied: list[bool] = []

    class _Probe(Widget):
        can_focus = True

        def watch_disabled(self, disabled: bool) -> None:
            applied.append(disabled)

    probe = _Probe()
    _set_transcript_parked(probe, True)
    assert probe.disabled is True
    assert applied == [], "parking ran the cascade it exists to skip"

    _set_transcript_parked(probe, False)
    assert probe.disabled is False
    assert applied == [False], (
        "un-parking did not run the disabled watcher, so a view dimmed by a "
        "cascade while parked would stay dimmed after it is revealed"
    )
