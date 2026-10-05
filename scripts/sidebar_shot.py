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

The seeded prose is a fixture, not a subject: it is mounted without
``finalize_text()``, so it is a STREAMING block and deliberately paints no rail
(``AssistantBlock._rail_cols``) — a missing bar beside it is that, not a defect.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textual import events  # noqa: E402
from textual.geometry import Offset  # noqa: E402
from textual.pilot import Pilot  # noqa: E402

from scripts.visual_capture import (  # noqa: E402
    isolate_capture,
    refuse_flag_shaped_argument,
    save_capture,
)

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
#: ``LO_SIDEBAR_SHOT_MESH=<config-root>`` sources the mesh tier from the REAL
#: producer — ``session.peer_rows.peer_session_rows`` against that root's own
#: relay — instead of the hand-stamped ``PEER_ROWS`` below.
#:
#: WHY THIS EXISTS (QA round 10, Q-R10-1). The published frame used to be the
#: ONLY rendering of a peer tier this tree could produce: ``PEER_ROWS`` is
#: stamped by hand, and its comment recorded that "the projection that will
#: produce them is not in this tree yet". The projection arrived in this branch
#: and did not work — ``RelayPeerCatalog._call`` read the reply ENVELOPE, so a
#: device with a peer answering held zero remote rows and the sidebar painted
#: ``No conversations yet``. A screenshot captured from a fixture cannot show
#: that, and did not. With this knob the frame is the surface the product
#: produces, from a real relay's answer.
#:
#: ``scripts/mesh_sidebar_shot.py`` builds the two-device mesh and invokes this
#: script with the knob set; the fixture stays the default so the deterministic
#: goldens and every other gallery frame are unchanged.
MESH_ROOT = os.environ.get("LO_SIDEBAR_SHOT_MESH") or ""

#: The mesh (R6) rows: one live peer's two sessions and one UNREACHABLE peer's
#: session. Used when ``LO_SIDEBAR_SHOT_MESH`` is unset; the published README
#: frame is captured from the REAL producer instead (see above).
#:
#: Two devices, not one, because the pair is what the MERGE is about: their rows
#: share the ordinary bins with no per-device heading between them (operator
#: convergence, 2026-10-05), and a single-device fixture cannot tell that apart
#: from a device that is simply the only peer.
PEER_ROWS = [
    (
        "bbbbbbbbbbb1",
        "Mesh transport identity design",
        3,
        "busy",
        "d_1a2b3c4d",
        "radiant-m4",
        True,
        "",
    ),
    (
        "bbbbbbbbbbb2",
        "Federated catalogue fan-out",
        26,
        "idle",
        "d_1a2b3c4d",
        "radiant-m4",
        True,
        "",
    ),
    (
        "ccccccccccc1",
        "Phone portal deploy check",
        58,
        # COLD on purpose (state "", not "idle"): the merged bins are what
        # these captures are about, and a cold peer row is the half that files
        # into `Previous Sessions` among this device's own — the interleaving a
        # headless frame cannot show if every fixture peer is live.
        "",
        "d_77aa88bb",
        "pixel-8",
        False,
        "connect_failed:ConnectionRefusedError",
    ),
]

#: The network each fixture device belongs to, by the NAME the tooltip's clause
#: reads. The real producer resolves these from this device's membership record
#: (``session/peer_rows._network_names``); the fixture stamps the resolved name
#: because rendering is what these captures are of, and the ``peers-hover`` case
#: asserts the clause is ON the frame.
PEER_NETWORKS = {"d_1a2b3c4d": "devmesh", "d_77aa88bb": "studio"}

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

#: A POINTER GESTURE on the footer's `⌥N` chip: ``hover`` moves the pointer
#: onto its cells, ``click`` presses them (issue #1357 principle 5). The cells
#: are located in the PAINTED footer (`⌥N` at its end), not through widget
#: internals, so one command captures both halves of a before/after pair: on a
#: tree without the control both gestures are no-ops and the frame is the
#: before, while on the changed tree hover shows the affordance and click the
#: flipped layer. The click answers its own re-poll from the fixture (the same
#: discipline the chord knob keeps). Unset performs no gesture.
CHIP_GESTURE = os.environ.get("LO_SIDEBAR_SHOT_CHIP") or ""

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


def _entries(show_subagents: bool = SHOW_SUBAGENTS, peers: bool = False) -> list[CatalogEntry]:
    # EVERY row carries a birth, local ones included (QA round 1, Q-1): the
    # product stamps every candidate's ``created_at`` from the store
    # (`session_catalog.load_catalog` / `session_created_at`), and the fixture
    # left it at zero — so a same-tier remote row with a claim sorted ABOVE
    # newer local rows in these frames, an interleaving artifact the merged-bins
    # claim must not be judged on. Same clock as ``mtime``: the label and the
    # position then agree by construction.
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
                created_at=NOW - age * 60,
            )
        )
        for session_id, name, age, state, pending, wakes, dormant in ROWS
    ]
    entries += [
        CatalogEntry(
            SessionRow(
                id=session_id,
                mtime=NOW - age * 60,
                name=name,
                live_state=state,
                created_at=NOW - age * 60,
            ),
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
            SessionRow(id=session_id, mtime=NOW - age * 60, name="", created_at=NOW - age * 60),
            subagent=True,
            label=label,
            agent=agent,
        )
        for session_id, label, agent, age in subagent_rows
    ]
    if peers:
        entries += [CatalogEntry(row) for row in _remote_rows()]
    return entries


def _remote_rows() -> list[SessionRow]:
    """The mesh tier: the REAL producer's rows when a mesh is named, else the fixture.

    The producer is called with the mesh's own config root, so this covers the
    whole production path — this device's relay, its control socket, the
    fan-out to each peer over a live link, and the unwrap that was missing
    (Q-R10-1). It comes back empty when anything on that path is broken, which
    is the point: a frame that cannot show a remote row is the failure the
    published screenshot used to hide.
    """
    if MESH_ROOT:
        from local_operator.session.peer_rows import peer_session_rows

        return list(peer_session_rows(Path(MESH_ROOT)))
    return [
        SessionRow(
            session_id,
            NOW - age * 60,
            name,
            live_state=state,
            locality="remote",
            owner_device=device,
            owner_device_name=label,
            owner_network_name=PEER_NETWORKS.get(device, ""),
            reachable=reachable,
            unreachable_reason=reason,
            # The producer stamps the peer's ``started`` claim here (see
            # ``session/peer_rows``), which is what orders a remote row among
            # this device's own; the fixture stamps the same field from the
            # same clock its ``mtime`` uses, so its frames order the way a
            # real mesh's would.
            created_at=NOW - age * 60,
        )
        for session_id, name, age, state, device, label, reachable, reason in PEER_ROWS
    ]


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


async def _forward_mouse(
    app: OperatorApp,
    pilot: Pilot[None],
    widget: Any,
    event_classes: Sequence[type],
    offset: tuple[int, int],
    button: int = 1,
) -> None:
    """Deliver pilot-shaped mouse events at a widget-relative ``offset``.

    FOR THE FULL-HEIGHT (DOCKED) SIDEBAR THE PILOT CANNOT REACH THE FOOTER.
    ``pilot.hover``/``pilot.click`` refuse any target outside
    ``screen.size.region`` — a region that starts at the screen's ORIGIN — and
    with the app's one-cell screen inset the docked sidebar's footer (the
    widget's last content line) sits exactly one row below that check region.
    A real terminal clicks that row fine; the overlay drawer is height-clamped
    above the input dock, so its footer sits INSIDE the region and the plain
    pilot does reach it. The seam is used uniformly anyway — these are the
    same events pilot builds and the same delivery it performs
    (``app.mouse_position`` + ``screen._forward_event``, pilot.py
    ``_post_mouse_events``); only the bounds pre-check is skipped.
    """
    x = widget.region.x + offset[0]
    y = widget.region.y + offset[1]
    app.mouse_position = Offset(x, y)
    for event_class in event_classes:
        kwargs: dict[str, Any] = {"chain": 1} if event_class is events.Click else {}
        event = event_class(
            widget=widget,
            x=x,
            y=y,
            delta_x=0,
            delta_y=0,
            button=0 if event_class is events.MouseMove else button,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=x,
            screen_y=y,
            **kwargs,
        )
        app.screen._forward_event(event)
        await pilot.pause()


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


#: The locality marks the sidebar paints, SPELLED rather than imported (the
#: failure direction the sibling guard documents: a rename makes a guard REFUSE
#: a frame rather than pass one it should not). `↗` = remote, `↛` = remote and
#: unreachable — the AT-REST cue (design round 1, D2), which is exactly the kind
#: of fact that fails quietly in a frame (a missing glyph still looks like a
#: list). See `session_sidebar.REMOTE_MARK` for the pairing's rationale.
REMOTE_GLYPH = "↗"
UNREACHABLE_GLYPH = "↛"


def _require_unreachable_cue(sidebar: Any) -> None:
    """Refuse a mesh frame whose drawn rows do not carry the locality marks.

    One glyph per DRAWN remote row, read off the painted line (`render`) —
    `↗` when reachable, `↛` when not — so the unreachable state is decodable at
    REST (design round 1, D2: Textual tooltips are mouse-only, so hover may not
    be the only channel) and so a regression that drops the mark state cannot
    ship a frame that still looks like a list. An off-page row is not charged;
    a drawn one cannot be skipped.
    """
    drawn = sidebar._display_rows()
    lines = sidebar.render().plain.splitlines()
    offset = 0 if sidebar._draws_section_headers(drawn) else 1
    charged = 0
    for index, (kind, entry) in enumerate(drawn):
        if kind != "entry" or entry is None or not entry.row.is_remote:
            continue
        charged += 1
        painted = lines[index + offset] if index + offset < len(lines) else ""
        wanted = UNREACHABLE_GLYPH if not entry.row.reachable else REMOTE_GLYPH
        if wanted not in painted:
            raise SystemExit(
                f"the drawn remote row {entry.id!r} reads {painted!r}: {wanted!r} "
                "is not in its cell — the at-rest locality cue this frame exists "
                "to show is missing or wrong"
            )
    if not charged:
        raise SystemExit(
            "no remote row is on the drawn page: the marks this guard reads have "
            "nothing to paint — the merged-bins guard should have caught this"
        )


def _require_unreachable_tooltip_line(sidebar: Any) -> None:
    """Refuse an unreachable-hover frame whose tooltip lacks the reason line.

    DESIGN ROUND 1, D4: the reason is its own terse line (`unreachable · <why>`)
    under an intact device clause. The failure is quiet — a tooltip that
    dropped the line still looks like a tooltip — so the guard reads the
    widget's own description for the prefix.
    """
    description = sidebar.tooltip or ""
    if "unreachable · " not in description:
        raise SystemExit(
            f"the tooltip reads {description!r}: no `unreachable · ` line — the "
            "reason the case exists to show is missing or fused into another line"
        )


def _require_local_hover_no_clause(sidebar: Any) -> None:
    """Refuse a LOCAL-row hover frame whose tooltip gained a device clause.

    N1's scoping claim: `on <device> · <network>` is for rows that live
    elsewhere; a local row's tooltip is the three lines it has always been.
    Read off the widget's own description, the same source the remote hover
    guard reads.
    """
    description = sidebar.tooltip or ""
    stray = [
        line for line in description.splitlines() if line.startswith("on ") or "unreachable" in line
    ]
    if stray:
        raise SystemExit(
            f"the LOCAL row's tooltip reads {description!r}: {stray!r} — the "
            "device clause belongs to remote rows only, and this case exists to "
            "show that it stays off local ones"
        )
    from textual.widgets import Tooltip

    tooltip_widget = sidebar.screen.get_child_by_type(Tooltip)
    if tooltip_widget is None or not tooltip_widget.display:
        raise SystemExit(
            "the description is set but the Tooltip widget is not displayed: the "
            "frame would show the bare row the case exists to explain"
        )


def _require_pinned_remote(sidebar: Any, pinned_id: str) -> None:
    """Refuse a frame in which the pinned REMOTE row is not on the Pinned tier.

    N1's other half of "first-class": the pin's lift is the ordinary one, so a
    pinned remote row paints under `★ Pinned` WITH its locality mark — the
    row-kind and the section both read off the frame's own state, not off the
    seed.
    """
    ranks = {entry.id: sidebar._section_of(entry) for entry in sidebar.entries}
    if ranks.get(pinned_id) != 0:
        raise SystemExit(
            f"the pinned remote row {pinned_id!r} ranks {ranks.get(pinned_id)!r}: "
            "the pin placed it outside `★ Pinned`"
        )
    lines = sidebar.render().plain.splitlines()
    for index, line in enumerate(lines):
        entry = sidebar._entry_at(index)
        if entry is not None and entry.id == pinned_id:
            if "★" not in line or REMOTE_GLYPH not in line:
                raise SystemExit(
                    f"the pinned remote row reads {line!r}: the pin's `★` and/or "
                    f"the `{REMOTE_GLYPH}` mark are not on its painted line"
                )
            return
    raise SystemExit(
        f"the pinned remote row {pinned_id!r} is not on any painted line: the "
        "frame cannot show what the case exists for"
    )


def _require_hover_device_clause(sidebar: Any, entry: Any) -> None:
    """Refuse a hover frame whose tooltip does not read the device AND network.

    THE CASE THIS GUARDS (``peers-hover``, operator convergence 2026-10-05): the
    clause ``on <device> · <network>`` is the convention's readable half — with
    the per-device heading retired it is the ONE place a row names the machine
    and the network it was projected through. Both facts fail QUIETLY in a frame:
    a missing clause renders as a shorter tooltip that still looks like a
    tooltip. So the guard reads the widget's own description — what
    ``_show_tooltip_now`` just painted — and the Tooltip widget's visibility,
    and refuses a frame that carries neither.
    """
    from textual.widgets import Tooltip

    description = sidebar.tooltip or ""
    label = entry.row.owner_label
    network = entry.row.owner_network_name
    if not network or f"on {label}" not in description or f"· {network}" not in description:
        raise SystemExit(
            f"the tooltip reads {description!r}: the device · network clause this "
            "case exists for is missing — and an EMPTY network field is refused "
            "too (review round 1, MINOR-2): the check used to short-circuit when "
            "the row carried no network, which is exactly the regression "
            "(fixture stops stamping) this guard exists to catch"
        )
    tooltip_widget = sidebar.screen.get_child_by_type(Tooltip)
    if tooltip_widget is None or not tooltip_widget.display:
        raise SystemExit(
            "the description is set but the Tooltip widget is not displayed: the "
            "frame would show the bare row the case exists to explain"
        )


def _require_merged_bins(sidebar: Any) -> None:
    """Refuse a mesh frame that is not the merged-bins state it was asked for.

    THE PAIR THIS CASE'S BEFORE/AFTER RESTS ON (operator convergence,
    2026-10-05): on the retired tree every remote row filed under its own
    ``⇄ <device>`` heading; the changed tree files them into the ordinary bins.
    The guard is on the ONE property that changed — no section outside the four
    tier names — and on the mesh tier being ON the drawn page at all, because
    "no stray heading" passes vacuously on a frame where the rows fell below
    the fold, and a guard that cannot fail refutes nothing.
    """
    tier = {"header:pinned", "header:active", "header:previous", "header:subagent"}
    kinds = [kind for kind, _entry in sidebar._display_rows()]
    stray = [kind for kind in kinds if kind.startswith("header:") and kind not in tier]
    if stray:
        raise SystemExit(
            f"the frame paints {stray!r}: a section outside the four tier names is "
            "the per-device peer section this case exists to show REMOVED — remote "
            "rows are first-class now, so re-capture against the changed tree"
        )
    drawn = {entry.row.id for _kind, entry in sidebar._display_rows() if entry is not None}
    wanted = {row.id for row in _remote_rows()}
    if not drawn & wanted:
        raise SystemExit(
            f"no remote row of {sorted(wanted)!r} is on the drawn page: the frame "
            "would show this device's own sessions only, and the no-stray-section "
            "check would pass vacuously"
        )


async def main() -> None:
    global CURSOR_ID, FOCUS_LIST
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: sidebar_shot.py OUT.svg [COLSxROWS] "
            "[peers|peers-focus|peers-hover|peers-hover-unreachable|peers-hover-local|peers-pinned]"
        )
    # Before it is used as a path: a mistyped flag here writes a file called
    # ``--help.svg`` into the working directory (see the helper's docstring).
    refuse_flag_shaped_argument(sys.argv[1], what="OUT")
    out = sys.argv[1]
    size = (100, 30)
    # ``peers`` is the VARIANT and the size is positional, in either order: the
    # gallery passes `out peers 100x30`, a person types `out 100x30`. A token
    # containing an `x` is a size; anything else names the variant.
    peers = False
    focus = FOCUS_LIST
    hover = ""
    pin_remote = False
    for arg in sys.argv[2:]:
        refuse_flag_shaped_argument(arg, what="argument")
        if "x" in arg:
            cols, rows = arg.split("x")
            size = (int(cols), int(rows))
        elif arg == "peers":
            peers = True
        elif arg == "peers-focus":
            # THE SETTLING FRAME design round 1's D4 asked for: the peers variant
            # with the list FOCUSED and the cursor ON a remote row, which is the
            # only state that shows the caret and the locality mark on one row.
            # Both knobs already existed (`LO_SIDEBAR_SHOT_FOCUS`,
            # `LO_SIDEBAR_SHOT_CURSOR`) and were never combined into a CASE, which
            # is exactly why every frame of that round was unfocused and the
            # interaction went unverified — a knob the gallery cannot reach is not
            # evidence.
            peers = True
            focus = True
        elif arg == "peers-hover":
            # THE HOVER FRAME (operator convergence, 2026-10-05): the pointer
            # resting on a remote row with its tooltip up, which is the ONLY
            # place the row names the device AND its network once the per-device
            # heading is gone — the desktop sibling reads the same two facts on
            # its hover/accessible name. Tooltips are off in every other capture
            # (`run_test(tooltips=False)` is Textual's default), so without this
            # case the clause had no durable frame at all, and a guard
            # (`_require_hover_device_clause`) refuses one that lacks it.
            peers = True
            hover = "remote"
        elif arg == "peers-hover-unreachable":
            # DESIGN ROUND 1, D4: the unreachable tooltip in its own register —
            # the device clause intact, the reason on its own line — which is
            # the frame the fused-sentence critique was written against.
            peers = True
            hover = "unreachable"
        elif arg == "peers-hover-local":
            # DESIGN ROUND 1, N1: the tooltip's SCOPING is the claim — a LOCAL
            # row must NOT gain the device · network clause — and a frame that
            # never hovers one cannot show that it doesn't.
            peers = True
            hover = "local"
        elif arg == "peers-pinned":
            # DESIGN ROUND 1, N1: a REMOTE row inside the `★ Pinned` bin, which
            # is half of "first-class": the pin's lift works on a remote row
            # exactly as on a local one, and no other case shows it.
            peers = True
            pin_remote = True
        else:
            raise SystemExit(
                f"unknown argument {arg!r}: expected a WxH size, 'peers', 'peers-focus', "
                "'peers-hover', 'peers-hover-unreachable', 'peers-hover-local' or "
                "'peers-pinned'"
            )
    FOCUS_LIST = focus
    if focus and peers:
        # The caret goes on a REMOTE row — the only state that shows the caret
        # and the locality mark on one row — whichever source supplied them.
        remote = _remote_rows()
        CURSOR_ID = remote[0].id if remote else CURSOR_ID
    pins = PINS
    pinned_remote_id = ""
    if pin_remote:
        remote = _remote_rows()
        if not remote:
            raise SystemExit(
                "peers-pinned: no remote rows to pin — a frame without one cannot "
                "show the pin's lift on a remote row"
            )
        pinned_remote_id = remote[0].id
        pins = (*PINS, pinned_remote_id)

    if NEUTRAL_CWD:
        os.environ["HOME"] = str(Path(os.environ["HOME"]).resolve())
        os.chdir(os.environ["HOME"])

    # THE NO_COLOR MODALITY (design round 1, N1): Textual reads NO_COLOR at App
    # CONSTRUCTION (`app.no_color`, which installs its monochrome filter), and
    # `isolate_capture` pops the variable precisely so ordinary captures keep
    # their colour — so the knob re-adds it INSIDE the isolated world and the
    # marks and bins get a frame with colour stripped. Visual only; nothing
    # about layout changes.
    if os.environ.get("LO_SIDEBAR_SHOT_NO_COLOR") == "1":
        os.environ["NO_COLOR"] = "1"
    app = OperatorApp(lambda: _factory(_AttachedFixtureSession()))
    async with app.run_test(size=size, tooltips=bool(hover)) as pilot:
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
        sidebar.set_pins(pins)
        sidebar.set_subagent_total(SUBAGENT_TOTAL)
        sidebar.set_entries(_entries(peers=peers))
        sidebar.current_id = CURRENT_ID
        sidebar.cursor_id = CURSOR_ID
        # Pin the animation: a capture is only comparable frame-to-frame if the
        # spinner is at a known phase in both. See `_pin_spinner`.
        _pin_spinner(sidebar)
        await pilot.pause()
        await pilot.pause()

        hovered = None
        if hover:
            # The pointer rests on the first row of the kind the case is about
            # that the frame DRAWS (an off-page one cannot be hovered), and the
            # tooltip is taken up through the same two-step every other tooltip
            # capture uses: the app's own delay, then the widget's restore for
            # the in-row move Textual's clock cannot complete headlessly (see
            # `_show_tooltip_now`).
            drawn = sidebar._display_rows()
            predicates = {
                "remote": lambda row: row.is_remote and row.reachable,
                "unreachable": lambda row: row.is_remote and not row.reachable,
                "local": lambda row: not row.is_remote,
            }
            wanted = predicates[hover]
            found = next(
                (
                    i
                    for i, (kind, entry) in enumerate(drawn)
                    if kind == "entry" and entry is not None and wanted(entry.row)
                ),
                None,
            )
            if found is None:
                raise SystemExit(
                    f"no {hover} row on the drawn page: the hover this case exists "
                    "for has nothing to land on. Capture at 100x45 or taller."
                )
            hovered = drawn[found][1]
            hover_y = found if sidebar._draws_section_headers(drawn) else found + 1
            await pilot.hover(sidebar, offset=(8, hover_y))
            await asyncio.sleep(float(app.TOOLTIP_DELAY) + 0.2)
            await pilot.pause()
            sidebar._show_tooltip_now()
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

        if CHIP_GESTURE:
            # THE CHIP AS A CONTROL. `⌥N` lives in the footer, and its cells
            # are read off the PAINTED line so the same command runs on a tree
            # with and without the control — the before half of a pair
            # gestures at a chip that is a marker, and does nothing.
            if CHIP_GESTURE not in {"hover", "click"}:
                raise SystemExit(
                    f"LO_SIDEBAR_SHOT_CHIP={CHIP_GESTURE!r}: expected 'hover' or 'click'"
                )
            footer = sidebar.render().plain.splitlines()[-1]
            match = re.search(r"⌥(?:1k\+|\d+)\s*$", footer)
            if match is None:
                raise SystemExit(f"no ⌥ chip on the footer to gesture at: {footer!r}")
            offset = (
                int(sidebar.styles.padding.left) + match.start(),
                sidebar.size.height - 1,
            )
            was = sidebar.show_subagents
            if CHIP_GESTURE == "hover":
                await _forward_mouse(app, pilot, sidebar, [events.MouseMove], offset, button=0)
            else:
                # Answer the re-poll the click's own `SubagentLayerToggled`
                # posts from the fixture, exactly as the chord block does. On
                # a tree without the control nothing is posted and
                # `_await_sidebar_poll` returns on its first pause.
                _serve_fixture_to_app_poll()
                await _await_sidebar_poll(app, pilot)
                await _forward_mouse(
                    app,
                    pilot,
                    sidebar,
                    [events.MouseDown, events.MouseUp, events.Click],
                    offset,
                )
                await pilot.pause()
                await _await_sidebar_poll(app, pilot)
                _pin_spinner(sidebar)
            await pilot.pause()
            await pilot.pause()
            print(
                f"chip gesture: {CHIP_GESTURE} at footer col {match.start()} "
                f"-> show_subagents {was}->{sidebar.show_subagents}"
            )

        if peers:
            # BEFORE the write, so a frame that is not the state it claims is
            # never on disk to be read as evidence (the census `new_remote_shot`
            # grew for the splash mark, applied to the merged bins and the
            # at-rest locality cues).
            _require_merged_bins(sidebar)
            _require_unreachable_cue(sidebar)
        if hover == "remote" and hovered is not None:
            _require_hover_device_clause(sidebar, hovered)
        if hover == "unreachable" and hovered is not None:
            _require_hover_device_clause(sidebar, hovered)
            _require_unreachable_tooltip_line(sidebar)
        if hover == "local":
            _require_local_hover_no_clause(sidebar)
        if pin_remote and pinned_remote_id:
            _require_pinned_remote(sidebar, pinned_remote_id)
        save_capture(app, out)
        conversation = app.query_one("#session-conversation")
        # THE LIST STATE, as a number rather than something to infer from pixels
        # (design round 1, D5b). A frame that paints nine of twenty-four rows and
        # stops halfway down the list cannot be read without knowing WHERE in the
        # order the window sits and WHY it is that tall — the earlier frame's
        # report said only the painted range, so "the store grew and the list got
        # shorter" and "the window is a page of a taller list" were
        # indistinguishable. All four numbers are already computed here for the
        # chord assertion; this is them, on the artifact.
        list_state = {
            "entries": len(sidebar.entries),
            "offset": sidebar._offset,
            "page_size": sidebar.page_size,
            "visible": len(sidebar.visible_entries),
            "sections": len({sidebar._section_key(e) for e in sidebar.entries}),
            "cursor_id": sidebar.cursor_id,
            "has_focus": sidebar.has_focus,
        }
        Path(f"{out}.list-state.json").write_text(json.dumps(list_state, indent=2) + "\n")
        print(
            f"terminal={size[0]}x{size[1]} "
            f"sidebar_outer={sidebar.region.width} "
            f"sidebar_content={sidebar.content_region.width} "
            f"conversation={conversation.size.width} "
            f"virtual={app.screen.virtual_size} actual={app.screen.size} "
            f"scrollbar={app.screen.show_vertical_scrollbar}"
        )
        print("list_state=" + json.dumps(list_state, sort_keys=True))


if __name__ == "__main__":
    # GUARDED, or importing this module to reuse its fixtures executes the
    # capture against the IMPORTER's argv (design round 4, D31: `unknown
    # argument 'two': expected a WxH size, 'peers' or 'peers-focus'`, from a
    # round that only wanted the fixture). Every other shot script in the tree
    # guards its entry point; this one did not, which is what made the rig
    # unusable as a library.
    asyncio.run(main())
