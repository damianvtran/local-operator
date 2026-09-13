"""A refused readiness frame must not leave a switch wedged on its saved anchor.

WHY THE ROWS IN THIS FIXTURE WRAP, and why that is load-bearing rather than
decoration. A saved position (`following_tail=False` plus an anchor) reaches the
commit's post-reveal `restore_revealed_anchor`, which scrolls the reader back
onto the anchor. With rows that re-wrap at the revealed width — ~10 screen rows
each, the shape `_commit_sidebar_session`'s own comment calls out — the
pre-reveal measurement of that anchor no longer survives the reveal, the reader
is left at the tail with the anchor row off-screen, and the readiness gate
refuses every frame. `post_display_hook`'s recovery branch bought one relayout
and nothing else, so the next frame was refused identically and the loop ran
until `_await_sidebar_frame`'s 15 s timer. MEASURED pre-fix on this rig: the
switch never commits — 340-552 recoveries inside the 5 s backstop below (498
over a longer pump), the reader at `scroll_y == max_scroll_y == 133` and the
anchor at `y=-131` — and through the e2e rig, 246 refusals / 240 recoveries over
15.8 s ending in `SurfaceNotReady`. Short rows do NOT reproduce it (3
recoveries, commits) — that is the difference between a row that
fits where it was measured and one that does not — so a future editor who
"simplifies" the fixture to short messages deletes the regression, not the
noise. `OperatorApp._reassert_sidebar_anchor` is the half that ends the wedge:
a refused frame re-asserts the reader's saved position, so a following frame has
the anchor painted and the gate accepts it. Post-fix the same switch commits
with 8 recoveries and the anchor row back on screen (`y=2`).

WHY THERE IS A FORCED-REFUSAL CASE AS WELL. The wrapped geometry alone wedges
the switch, which is the load-free reproduction this test mostly relies on. On
CI the same defect was reached the other way round — an INCIDENTAL first-frame
refusal (about one run in twenty, which no local run can be asked to arrange)
landing on that same geometry, after which the recovery loop re-armed itself out
of it. The forced case injects those refusals at the gate so that path is pinned
deterministically too.

WHAT EACH CASE PINS, in the order the failure demands:

* the switch COMMITS — the gate is satisfied, not merely re-armed;
* the reader is ON its saved anchor (the anchor row overlaps the content
  region), which is the user-visible half;
* recoveries stay bounded. The ceiling separates "converged" from "spun" by
  more than an order of magnitude rather than pinning a frame count: a healthy
  switch spends 0, this fixture post-fix spends 8, and the pre-fix loop spends
  hundreds and never commits at all.

WHAT THIS RIG DOES NOT PIN, MEASURED: it pins the CONVERGENCE PROPERTY, not the
layer that produces the bad geometry. Neutering the commit's reveal-time restore
(`self.call_after_refresh(restore_revealed_anchor)` -> `pass`, then restored)
leaves this rig GREEN — 2 passed — because the synchronous re-assert alone lands
the reader on the anchor. So do not read a pass here as covering that line: the
reveal-time restore's necessity is guarded by the e2e `wrapped` case
(`tests/e2e/test_sidebar_display_e2e.py`), which is the slower witness, and a
case that pins the layer would have to make the re-assert unable to run. Both
layers are wanted — the after-refresh restore keeps the healthy switch's frame
count at 0 — but only one of them is pinned here, and saying so is cheaper than
letting the next reader believe otherwise.

Run it with ``env -u NO_COLOR TERM=xterm-256color`` like the rest of the TUI
suite; the fixture below unsets every inherited ``CMUX_*`` variable, because an
inherited ``CMUX_WORKSPACE_ID`` has previously let a headless run rename the
operator's real cmux workspaces.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.session_interaction import SessionDraft, SessionInteraction
from tests.unit.tui.test_app_pilot import _factory
from tests.unit.tui.test_sidebar_swap_reset import SidebarRemote, _message

#: Gate recoveries one switch may spend before this counts as a spin.
#:
#: Same sizing argument as `_GATE_RECOVERY_CEILING` in the sibling sidebar e2e
#: files: separate the two populations by orders of magnitude rather than pin an
#: exact frame count, which legitimately varies with which paint carries the
#: completed compositor map. Measured here: 0 for a switch with no saved anchor,
#: 8 for this fixture once it converges, 498+ for the pre-fix loop, which cannot
#: commit at all.
_RECOVERY_CEILING = 40

#: Gate checks refused before the production gate is consulted, standing in for
#: the incidental first-frame refusal a loaded runner produces. Three rather
#: than one: an e2e run measured a SINGLE refused frame healing by itself when
#: the refusal landed before the reveal, so one is not a reliable trigger.
_FORCED_REFUSALS = 3

#: Wall-clock bound on the switch itself, so a non-converging switch fails this
#: test in seconds instead of sitting out `_await_sidebar_frame`'s 15 s timer.
#: A BACKSTOP, not an assertion: the pump ends the moment the switch does.
_SETTLE_BACKSTOP_S = 5.0


@pytest.fixture(autouse=True)
def isolated_anchor_switch(tmp_path, monkeypatch):
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


def _wrapped_conversation(session_id: str, turns: int) -> SidebarRemote:
    """A conversation whose assistant rows are many screen rows tall.

    Every message carries an ``id``: `PreparedReplay` derives a block's
    navigation anchor from the projecting message's id, so a fixture without one
    has no anchors and cannot exercise the saved-position path at all.
    """
    history: list[object] = []
    for turn in range(turns):
        user = _message("user", f"question {turn} about the switch path")
        user.id = f"{session_id}-u{turn}"
        assistant = _message(
            "assistant",
            f"answer {turn}, with some detail to render " + ("wrapped content " * 35),
        )
        assistant.id = f"{session_id}-a{turn}"
        history.extend((user, assistant))
    return SidebarRemote(session_id, history=history)


@pytest.mark.asyncio
@pytest.mark.parametrize("refused_frames", [0, _FORCED_REFUSALS])
async def test_a_refused_frame_still_lands_the_reader_on_its_saved_anchor(
    refused_frames: int,
) -> None:
    home = SidebarRemote("home-session")
    target = _wrapped_conversation("anchored-target", 20)
    anchor_id = target.history()[0].id
    assert anchor_id, "the fixture must anchor on a real message id"

    app = OperatorApp(lambda: _factory(home))
    with patch("local_operator.session.attached.AttachedSession", SidebarRemote):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(20):
                await pilot.pause()

            # The REAL lease registers the source this way and seeds its draft
            # out of the store (`_lease_sidebar_source`); both are stubbed here
            # because leasing reaches for an owner record on disk, and a rig
            # that skips the registration makes `_is_current(source)` false for
            # the whole switch, so `post_display_hook` never reaches its
            # recovery branch and the gate can never be satisfied.
            source = SessionInteraction(target)
            app._sidebar_sources[target.session_id] = source
            app._interactions[id(target)] = source
            await app._sidebar_drafts.put(
                target.session_id,
                SessionDraft(following_tail=False, scroll_anchor_id=anchor_id),
            )

            async def lease(_session_id, *, speculative=False):
                source.draft = await app._sidebar_drafts.get(target.session_id)
                source.preparations += 1
                return source

            app._lease_sidebar_source = lease  # type: ignore[method-assign]

            production_gate = app._sidebar_gate_surface_ready
            refusals_left = refused_frames

            def gate(candidate):  # type: ignore[no-untyped-def]
                nonlocal refusals_left
                if refusals_left:
                    refusals_left -= 1
                    return False
                return production_gate(candidate)

            app._sidebar_gate_surface_ready = gate  # type: ignore[method-assign]

            recoveries_before = app._sidebar_gate_recoveries
            task = app._sidebar_navigation.select(target.session_id)
            deadline = asyncio.get_running_loop().time() + _SETTLE_BACKSTOP_S
            while not task.done() and asyncio.get_running_loop().time() < deadline:
                await pilot.pause()

            spent = app._sidebar_gate_recoveries - recoveries_before
            assert refusals_left == 0, "the gate was never reached with a frame armed"
            assert task.done(), (
                f"the switch never finished: {spent} recoveries in {_SETTLE_BACKSTOP_S}s, "
                "so a refused frame is still re-arming itself instead of converging"
            )
            assert (
                app._sidebar_navigation.committed_id == target.session_id
            ), "the switch re-armed the gate but never satisfied it"

            view = app._transcript_view()
            anchor = next(
                (block for block in view.blocks() if block.navigation_anchor_id == anchor_id),
                None,
            )
            assert anchor is not None, "the anchored row is not in the committed view"
            assert anchor.region.overlaps(view.content_region), (
                "the switch committed with the reader left off its saved anchor "
                f"(anchor at y={anchor.region.y}, content region {view.content_region})"
            )
            assert spent <= _RECOVERY_CEILING, (
                f"a refused frame cost {spent} recovery relayouts; the recovery "
                "branch is re-arming instead of re-asserting the saved position"
            )
