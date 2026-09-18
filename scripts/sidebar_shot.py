"""Capture the session sidebar over a populated transcript, for visual validation.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/sidebar_shot.py OUT.svg [COLSxROWS]

Seeds ONE fixed catalog covering every state the list can draw — a live turn,
an idle runtime, an attached viewer, an armed wake, a dormant wake, a parked
gate, a wedged runtime, cold rows, and (see ``UNSEEN_ROWS``) rows carrying an
unacknowledged completion of each kind including one that has since been
resumed — so a single frame answers both questions this change is about:

* **Which glyph each state draws**, across every mark the list can produce.
* **How much of a title survives.** The names here are real-length generated
  titles, so the frame shows the title budget rather than a curated short one.

**What this script can and cannot prove.** It renders from a FIXED catalog of
``SessionRow``s, so it exercises the display layer only. The rows named
"…(bg job)" and "…(subagent)" carry ``live_state="idle"`` because that is what
the fixed runtime now publishes for them — which means they render ``●`` in
this script against BOTH trees, and a before/after pair of these frames is not
evidence for the activity fix (round 1, D3). The change those rows stand for
happens upstream, where ``ServingSessionHandle`` decides whether to publish
``busy`` at all; it is proven by
``tests/unit/session/runtime/test_activity_vs_residency.py`` and by the live
probe described on the PR, not here. To see the pre-fix rendering of those
rows, set ``LO_SIDEBAR_SHOT_STALE_BUSY=1``, which restores the state the OLD
record would have published for a session holding background work.

The frame is deterministic: the spinner is pinned to a known frame and the
catalog is fixed, so before/after captures differ only where the change does.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textual.pilot import Pilot  # noqa: E402

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.resume import SessionRow  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.session_catalog import (  # noqa: E402
    CatalogEntry,
    SidebarSettings,
)
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.session_sidebar import SessionSidebar  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

NOW = time.time()

#: How many frames to wait for the app's catalog poll to land before calling it
#: stuck. The wait itself is on the poll's own pending flag, never on the clock
#: (`_await_sidebar_poll`); this only bounds a poll that never returns.
POLL_SETTLE_FRAMES = 500

#: ``(id, title, age_minutes, live_state, pending, wakes, dormant)``.
#: Titles are real generated-length names, several of them sharing a prefix,
#: because "Article-search-svc s…" vs "Article search servi…" is exactly the
#: discrimination the list failed to support at the base width.
#: Rows whose live_state models a session HOLDING BACKGROUND WORK with a
#: finished conversation. Under the old runtime these published ``busy``; under
#: the fixed one they publish idle. ``LO_SIDEBAR_SHOT_STALE_BUSY=1`` renders
#: them the old way so the two can be compared in one tree.
STALE_BUSY = os.environ.get("LO_SIDEBAR_SHOT_STALE_BUSY") == "1"
HOLDING_WORK = "busy" if STALE_BUSY else "idle"

ROWS = [
    ("aaaaaaaaaaa1", "Fix sidebar activity indicator accuracy", 1, "busy", None, 0, False),
    ("aaaaaaaaaaa2", "Update Provider Onboarding and OAuth UX", 4, "attached", None, 0, False),
    ("aaaaaaaaaaa3", "Article-search-svc schema review (bg job)", 11, HOLDING_WORK, None, 0, False),
    ("aaaaaaaaaaa4", "Article search service integration rollout", 6, "idle", None, 0, False),
    ("aaaaaaaaaaa5", "OSWorld benchmark evaluation (subagent)", 8, HOLDING_WORK, None, 0, False),
    ("aaaaaaaaaaa6", "Auto-update inactive session runtimes", 13, "idle", None, 2, False),
    ("aaaaaaaaaaa7", "Debugging session cost and naming drift", 43, "idle", None, 1, True),
    # The two states the round-1 design findings were about. Neither was in the
    # fixture when those findings were raised, so the frames could not show the
    # bugs OR their fixes and the reviewer had to build a scratch harness to
    # see them (round 2). A capture script that cannot frame the states it was
    # reviewed for is the same defect as D3 in a different place.
    #
    #   attached + armed wake -> must draw the ATTACHED mark, not the wake (D2)
    #   cold + dormant wake   -> draws the wake mark, so the words must follow
    #                            it: "Stopped (N wakes dormant)" (D1)
    ("aaaaaaaaaab4", "Review provider onboarding copy", 7, "attached", None, 2, False),
    ("aaaaaaaaaab5", "Stopped nightly catalogue refresh", 120, "", None, 1, True),
    ("aaaaaaaaaaa8", "Add Flavia's Adverse Media Case", 3, "idle", "approval", 0, False),
    ("aaaaaaaaaaa9", "Review and merge open provider MRs", 52, "wedged", None, 0, False),
    ("aaaaaaaaaab1", "Toggleable Sidebar for Session Switching", 49, "", None, 3, False),
    ("aaaaaaaaaab2", "Mark Focused TUI Session as Read", 300, "", None, 0, False),
    ("aaaaaaaaaab3", "Address Local Operator packaging review", 35, "", None, 0, False),
]

#: Rows carrying an UNACKNOWLEDGED completion, as
#: ``(id, title, age_minutes, live_state, completion_kind)``.
#:
#: Kept as a separate table because `unseen` is a dimension the fixture above
#: does not model at all — it is layered on by the attention store, not by the
#: runtime record — and widening every tuple in ROWS to carry two more fields
#: would touch fourteen rows to describe four.
#:
#: These are the states this change is about, and none of them could be framed
#: before: the fixture had no unseen row whatsoever, so neither the shared
#: `✗` nor the stale-mark bug was visible in any capture. The last entry is
#: the operator's reported defect exactly — a session that finished a turn
#: unread and has since been RESUMED and is working again. Before the fix it
#: painted `✗` over its own spinner; after, the spinner wins and the mark
#: returns when the turn ends.
UNSEEN_ROWS = [
    ("aaaaaaaaaac1", "Backfill customer cube contacts", 9, "idle", "complete"),
    ("aaaaaaaaaac2", "Sanctions feed reconciliation run", 16, "idle", "error"),
    ("aaaaaaaaaac3", "Nightly PEP screening sweep", 22, "idle", "interrupted"),
    ("aaaaaaaaaac4", "Resumed adverse-media enrichment", 2, "busy", "interrupted"),
]


#: Hidden-population rows for the ⌥ layer. `(id, label, role, age)` — a sub row
#: is identified by what it was delegated to do, never by the session name its
#: runtime generated, so these carry no `name` at all. The third has an EMPTY
#: label so the frame also shows the degraded `role only` form.
SUBAGENT_ROWS = [
    ("aaaaaaaaaad1", "section the sidebar", "coder", 4),
    ("aaaaaaaaaad2", "audit the poll cost", "reviewer", 11),
    ("aaaaaaaaaad3", "", "scout", 19),
]

#: Pinned in the capture. One ACTIVE row, to show that a pin LIFTS a row out of
#: the section it ranked into, and one SUBAGENT row, to show a pinned agent run
#: staying visible in ★ Pinned while the layer itself is off.
PINNED_IDS = ("aaaaaaaaaaa3", "aaaaaaaaaad2")

#: What the footer chip reports. Larger than the seeded rows on purpose: the
#: chip counts the whole hidden population, not the capped slice on screen.
#: ``LO_SIDEBAR_SHOT_TOTAL`` overrides it so the capped frame (`⌥1k+`, above
#: 999) can be captured from the same fixture; the default reproduces the
#: existing frames byte-identically.
SUBAGENT_TOTAL = int(os.environ.get("LO_SIDEBAR_SHOT_TOTAL") or 438)

#: The session this capture is ATTACHED to — the row the app paints as current,
#: bold on its own ground. The fixture declares it, and the seeded app session
#: is made to match it (see `_AttachedFixtureSession`), because the app's own
#: catalog poll re-derives `current_id` from the session on every refresh.
CURRENT_ID = "aaaaaaaaaaa1"

#: Keys pressed AFTER the rows are seeded and the spinner pinned, as a
#: comma-separated list (``ctrl+o``). This is how a CHORD is captured as the
#: user actually fires it rather than by setting the state it produces by
#: hand — a frame of the outcome is not evidence that the binding reaches it.
#: Pressed last so the seeded catalog is already in place. Unset presses
#: nothing.
#:
#: A chord capture is also a FOCUSED one (see `FOCUS_LIST` below), and it
#: FAILS if the chord left the sidebar unchanged: a frame indistinguishable
#: from the same capture with the chord omitted is not evidence of the chord.
_CHORD_ENV = os.environ.get("LO_SIDEBAR_SHOT_CHORD") or ""
CHORDS = tuple(part.strip() for part in _CHORD_ENV.split(",") if part.strip())

#: The capture opens the sidebar but leaves the keyboard in the composer, which
#: is the unfocused footer (`f9 focus`). ``LO_SIDEBAR_SHOT_FOCUS=1`` presses
#: `f9` as well, so the focused lead (`esc return`) can be captured — the only
#: way to see the D12 fix on a real frame. Default unset, i.e. unfocused.
#:
#: A CHORD implies it, and could not be captured without it: `ctrl+a`/`ctrl+o`
#: are sidebar-SCOPED, so a press from the composer resolves to no binding at
#: all and the knob then reports a chord that never had a chance to run as a
#: chord that did not work (review round 3, MAJOR 1a). The frame a chord
#: capture produces therefore carries the FOCUSED lead.
FOCUS_LIST = os.environ.get("LO_SIDEBAR_SHOT_FOCUS") == "1" or bool(CHORDS)

#: Which rows are pinned. ``LO_SIDEBAR_SHOT_PINS`` is a comma-separated id
#: list, or the literal ``none`` for an unpinned list — the baseline a user
#: sees before pinning anything, which the fixed ``PINNED_IDS`` above cannot
#: frame. Unset reproduces ``PINNED_IDS`` byte-identically.
_PINS_ENV = os.environ.get("LO_SIDEBAR_SHOT_PINS")
if _PINS_ENV is None:
    PINS: tuple[str, ...] = PINNED_IDS
elif _PINS_ENV.strip().lower() in {"none", ""}:
    PINS = ()
else:
    PINS = tuple(part.strip() for part in _PINS_ENV.split(",") if part.strip())

#: The ⌥ layer. ``LO_SIDEBAR_SHOT_LAYER=0`` captures it OFF — the default
#: state of `tui.sidebar_show_subagents`, and the only way to frame that the
#: footer chip counts the hidden population whether or not the layer is drawn.
#: Unset leaves it ON, as the existing frames have it.
SHOW_SUBAGENTS = os.environ.get("LO_SIDEBAR_SHOT_LAYER") != "0"

#: Which row the cursor sits on. ``LO_SIDEBAR_SHOT_CURSOR`` takes an id so a
#: pinned row can be put under the cursor and the `›`-displaces-`★` handoff
#: captured. Unset keeps the current/cursor row at the first seeded id.
CURSOR_ID = os.environ.get("LO_SIDEBAR_SHOT_CURSOR") or CURRENT_ID

#: ``LO_SIDEBAR_SHOT_NEUTRAL_CWD=1`` runs the app from the isolated HOME so the
#: status band paints ``~`` instead of the operator's real checkout path. The
#: sidebar itself is unaffected; this only keeps a published frame free of a
#: personal absolute path, which matters at the WIDE widths where the band has
#: room to spell the whole thing out. Same fixture device as `pages_shot.py`,
#: whose note calls a stable cwd fixture data rather than a layout change.
NEUTRAL_CWD = os.environ.get("LO_SIDEBAR_SHOT_NEUTRAL_CWD") == "1"


class _AttachedFixtureSession(FakeSession):
    """The fake app session, wearing the fixture's attached row id.

    The app's own catalog poll writes `current_id` from the ATTACHED session
    (`_refresh_sidebar`), and `current_id` is what the painter bolds as the
    current row (`session_sidebar.py`, `current = entry.id == self.current_id`).
    The stock fake answers `"sess"` — no row of this fixture — so a chord that
    triggers a poll would drop the current row's ground from the frame, a
    difference the chord did not cause and the chord-off frame would not share.
    In the product the attached session IS a row of the list; this makes the
    two agree, so the fixture's `CURRENT_ID` is derived by the app rather than
    remembered by the script.
    """

    @property
    def session_id(self) -> str:
        return CURRENT_ID


def _entries(show_subagents: bool = SHOW_SUBAGENTS) -> list[CatalogEntry]:
    entries = [
        CatalogEntry(
            SessionRow(
                id=session_id,
                mtime=NOW - age * 60,
                name=name,
                live_state=state,
                pending=pending,
                wakes=wakes,
                wakes_dormant=dormant,
            )
        )
        for session_id, name, age, state, pending, wakes, dormant in ROWS
    ]
    entries += [
        CatalogEntry(
            SessionRow(id=session_id, mtime=NOW - age * 60, name=name, live_state=state),
            unseen=True,
            completion_kind=kind,
        )
        for session_id, name, age, state, kind in UNSEEN_ROWS
    ]
    # What the LAYER-OFF catalog actually supplies. `load_catalog` takes
    # `include_subagents=show_subagents`, and with it False the hidden
    # population is filtered out at the LOAD site -- except for PINNED ids,
    # which are exempt from that filter (and from the cap) so a pinned run
    # still resolves. Handing all three rows in regardless would render a
    # layer-off frame that the product never produces.
    subagent_rows = [row for row in SUBAGENT_ROWS if show_subagents or row[0] in PINS]
    entries += [
        CatalogEntry(
            SessionRow(id=session_id, mtime=NOW - age * 60, name=""),
            subagent=True,
            label=label,
            agent=agent,
        )
        for session_id, label, agent, age in subagent_rows
    ]
    return entries


def _serve_fixture_to_app_poll() -> list[str]:
    """Answer the app's own catalog re-poll from the fixture the list is seeded with.

    `ctrl+a`/`ctrl+o` post `SubagentLayerToggled`, and the app answers by
    re-polling the catalog off disk (`app.py`, `_refresh_sidebar`). An isolated
    capture has no store behind it, so that poll returns EMPTY and replaces the
    seeded list with it — resetting `current_id`, `cursor_id` and the pins, on
    the way to a frame with no caret anywhere and no `★ Pinned` section, which
    is a state the product does not produce (review round 3, MAJOR 1b).

    Handing that poll the fixture is what lets the chord's OWN re-poll land for
    real: the jump `ctrl+o` arms is landed by the `set_entries` this poll
    delivers, against the rows a real store holding this fixture would supply.
    Only the store the refresh path reads and the pin store the pin chord
    writes are redirected — the path itself (generation check, `current_id`,
    `set_entries`, `set_pins`) runs unchanged, so nothing here hand-sets the
    state a chord produces, which is the thing this knob's contract forbids.

    Returns the pin store itself, so the caller can assert the frame's pins
    against the STORE's rather than against the seed: a chord is allowed to
    change the pinned set (`f10` does), and only a reset behind its back is the
    defect this knob has to be able to see.
    """
    from local_operator.tui import session_catalog, sidebar_pins

    def load_catalog(
        directory: Path,
        limit: int | None = None,
        *,
        include_subagents: bool = False,
        pinned_hidden_ids: Sequence[str] = (),
    ) -> list[CatalogEntry]:
        # `_entries` already models both arguments the refresh path passes:
        # `include_subagents` by filtering the hidden population, and the
        # pinned-id exemption by keeping a pinned subagent row in either layer.
        # `pinned_hidden_ids` is the set `read_pins` below answers with, so
        # there is no third case to model.
        return _entries(include_subagents)

    fixture_pins = list(PINS)

    def read_pins(directory: Path) -> list[str]:
        return list(fixture_pins)

    def toggle_pin(directory: Path, session_id: str) -> bool:
        """`sidebar_pins.toggle_pin`'s contract, over the fixture's own store.

        Pin when absent, unpin when present, newest first, returning the new
        state. `f10` reaches `action_toggle_pin`, which reads and writes pins;
        a reader that always answered `PINS` would swallow the toggle, and the
        capture would report a key that demonstrably fired as one that never
        reached its action. The real function writes the capture's isolated
        config file, which the fixture store does not include, so its own
        `read_pins` answers `[]` and every toggle there would look like a pin.
        """
        if session_id in fixture_pins:
            fixture_pins.remove(session_id)
            return False
        fixture_pins.insert(0, session_id)
        return True

    session_catalog.load_catalog = load_catalog
    session_catalog.subagent_population = lambda directory: SUBAGENT_TOTAL
    sidebar_pins.read_pins = read_pins
    sidebar_pins.toggle_pin = toggle_pin
    return fixture_pins


async def _await_sidebar_poll(app: OperatorApp, pilot: Pilot[None]) -> None:
    """Return once the app's catalog poll has landed and the list has repainted.

    The poll is a worker: `_refresh_sidebar` raises `_sidebar_refresh_pending`
    synchronously and the worker lowers it in its `finally`, so the flag IS the
    delivery. Waiting on it is waiting on the event rather than on the clock
    (AGENTS.md, "Wait on the event, never on the clock"); a fixed number of
    pauses is a guess that holds on a fast machine and not on a slow one.
    """
    for _ in range(POLL_SETTLE_FRAMES):
        await pilot.pause()
        if not app._sidebar_refresh_pending:
            return
    raise AssertionError(
        f"the sidebar catalog poll did not land within {POLL_SETTLE_FRAMES} frames"
    )


def _pin_spinner(sidebar: SessionSidebar) -> None:
    """Hold the spinner at a known phase, and fail if it drifted.

    Pausing the animation timer is NOT enough, and a frame pinned that way was
    in fact unstable across runs (design review round 2, D6: six runs, two
    distinct glyphs). `_sync_animation()` runs from `set_entries` and from
    several other paths, and it RE-RESUMES the timer whenever any visible row
    is busy — so a tick can still land during the settling pauses and advance
    `_frame` out from under the pin.

    Stopping the timer outright and clearing the handle is what actually holds:
    `_sync_animation` returns early when `_timer is None`, so nothing can
    restart it, and `_frame` stays exactly where it is put. A chord, and the
    poll it triggers, both run `set_entries`, so this is called again after
    each of them rather than once at the start.
    """
    if sidebar._timer is not None:
        sidebar._timer.stop()
        sidebar._timer = None
    sidebar._frame = 2
    sidebar.refresh()
    # Assert the pin rather than trusting it: a capture script whose
    # determinism claim is false silently poisons every future comparison
    # captured from it.
    assert sidebar._frame == 2, f"spinner phase drifted to {sidebar._frame}"


def _widget_state(sidebar: SessionSidebar) -> tuple[object, ...]:
    """Everything a chord under capture is allowed to move, as one comparison.

    Focus is in here because it is DRAWN — the row caret is
    `self.has_focus and entry.id == self.cursor_id`, and the footer lead flips
    to `esc return` — so `f9` is a chord whose effect this has to be able to
    see rather than report as a chord that never reached its binding.
    """
    return (
        sidebar.has_focus,
        sidebar.show_subagents,
        sidebar.cursor_id,
        sidebar._offset,
        tuple(sidebar._pins),
        len(sidebar.entries),
    )


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))

    if NEUTRAL_CWD:
        os.environ["HOME"] = str(Path(os.environ["HOME"]).resolve())
        os.chdir(os.environ["HOME"])

    app = OperatorApp(lambda: _factory(_AttachedFixtureSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app._sidebar_settings = SidebarSettings(False, "left")
        # Seed a conversation so the frame shows the list BESIDE something,
        # which is the only way the width trade-off is visible at all.
        for turn in range(1, 7):
            app._append_block(UserBlock(f"Turn {turn}: what should we do about the stale rows?"))
            prose = AssistantBlock()
            prose.update_text(
                f"Answer {turn}: the audit log still has every row, so a backfill "
                "is possible. Nothing else reads that column today, which is why "
                "dropping it is on the table at all."
            )
            app._append_block(prose)
        await pilot.pause()

        await pilot.press("ctrl+b")
        await pilot.pause()
        if FOCUS_LIST:
            # BEFORE the rows are handed in, not after: focusing runs
            # `_set_sidebar_open`, which takes a fresh population count and
            # re-reads the store, and that read would replace the seeded rows
            # with the empty real one. Seeding last is what every other piece
            # of state here already does for the same reason.
            await pilot.press("f9")
            await pilot.pause()
        sidebar = app._session_sidebar
        # The ⌥ layer ON, so the capture shows all four sections at once. The
        # rows are handed in directly (as every other row here is) rather than
        # loaded, so this never touches the developer's real store.
        sidebar.show_subagents = SHOW_SUBAGENTS
        sidebar.set_pins(PINS)
        sidebar.set_subagent_total(SUBAGENT_TOTAL)
        sidebar.set_entries(_entries())
        sidebar.current_id = CURRENT_ID
        sidebar.cursor_id = CURSOR_ID
        # Pin the animation: a capture is only comparable frame-to-frame if the
        # spinner is at a known phase in both. See `_pin_spinner`.
        _pin_spinner(sidebar)
        await pilot.pause()
        await pilot.pause()

        if CHORDS:
            # No routine poll may be in flight when the chord is pressed: the
            # chord's own `_refresh_sidebar()` is DROPPED while one is pending
            # (`app.py`, `if ... or self._sidebar_refresh_pending: return`), and
            # a dropped re-poll is a jump with no rows to land on. Settle first.
            fixture_pins = _serve_fixture_to_app_poll()
            await _await_sidebar_poll(app, pilot)
            _pin_spinner(sidebar)
            before = _widget_state(sidebar)

            for chord in CHORDS:
                await pilot.press(chord)
                await pilot.pause()

            # The chord's own re-poll is the delivery under capture — including
            # for `ctrl+o`, whose jump is landed by the `set_entries` that
            # carries its rows. Wait for it rather than for a fixed pause.
            await _await_sidebar_poll(app, pilot)
            _pin_spinner(sidebar)
            await pilot.pause()
            # A chord capture is evidence only if the frame it produced can be
            # told apart from the same capture with the chord omitted. Assert
            # THAT rather than the layer flag, because the flag cannot tell the
            # two apart: with `LAYER=1` it is already True before the press, so
            # an assertion on it passes whether or not the binding ever ran,
            # and two captures differing by nothing but the chord came out
            # byte-identical (review round 3, MAJOR 1c).
            assert _widget_state(sidebar) != before, (
                f"chords {CHORDS!r} left the sidebar unchanged in this "
                f"configuration ({before}): the binding never reached its "
                "action, or the chord's effects cancel out (a toggle pressed "
                "twice). Either way the frame is not evidence of the chord."
            )
            # The two properties the round found this frame WITHOUT, and the
            # reason a frame from the broken knob was not one the product
            # produces. `cursor_id` names a row of the list, or no row carries
            # the caret the painter draws with `entry.id == self.cursor_id`;
            # the pins survive the re-poll, or the `★ Pinned` section is gone.
            members = {entry.id for entry in sidebar.entries}
            assert sidebar.cursor_id in members, (
                f"the chord left the caret on {sidebar.cursor_id!r}, which is "
                f"no row of the {len(members)}-row list: the frame would draw "
                "no caret at all"
            )
            assert tuple(sidebar._pins) == tuple(fixture_pins), (
                f"the sidebar shows pins {sidebar._pins!r} against the store's "
                f"{tuple(fixture_pins)!r}: the poll reset them behind the "
                "chord's back, and the frame would lose its ★ Pinned rows"
            )
            # A frame whose caret row is off the DRAWN page paints no caret at
            # all, and a reader cannot tell that frame apart from a chord that
            # never ran. It is reachable, and not a chord that failed: the
            # delivery answering the chord re-windows the list (the ⌥ section
            # adds chrome, so `page_size` drops) and a row ranked below the new
            # window is painted as `+N more pinned — scroll` instead — the
            # accepted consequence `_display_rows` documents. Say it, and say
            # what to do about it, rather than failing a capture of a state the
            # product really produces.
            if sidebar.cursor_id not in {entry.id for entry in sidebar.visible_entries}:
                print(
                    f"note: the caret row {sidebar.cursor_id} is outside the drawn page "
                    f"(page_size {sidebar.page_size} of {len(sidebar.entries)} rows, offset "
                    f"{sidebar._offset}), so this frame paints no caret. The delivery that "
                    "answers the chord re-windows the list; capture at a taller size, or set "
                    "LO_SIDEBAR_SHOT_PINS=none to capture the jump landing on the delivery.",
                    file=sys.stderr,
                )

        save_capture(app, out)
        conversation = app.query_one("#session-conversation")
        print(
            f"terminal={size[0]}x{size[1]} "
            f"sidebar_outer={sidebar.region.width} "
            f"sidebar_content={sidebar.content_region.width} "
            f"conversation={conversation.size.width} "
            f"virtual={app.screen.virtual_size} actual={app.screen.size} "
            f"scrollbar={app.screen.show_vertical_scrollbar}"
        )


asyncio.run(main())
