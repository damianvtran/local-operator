"""Render the subagent page in the duplicated-notice state.

Drives the REAL ``OperatorApp`` (so ``local_operator.tcss`` applies) and puts
the page in exactly the state the operator screenshotted: a busy child whose
trajectory has reached ``TRAJECTORY_CAP``, with one error notice that the
front-eviction keeps re-keying on every 1 Hz refresh.

Usage:
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python /tmp/subagent_dupe_shot.py out.svg
Run from the repo root (or a worktree of it) — ``tests.`` must import.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any, cast

sys.path.insert(0, os.getcwd())

# Shimmer sweeps the working line at 30 fps, so the glyph under the cursor
# differs between two captures taken milliseconds apart. The widget has a
# supported still-frame mode for exactly this (WorkingBlock: "when shimmer is
# disabled the line falls back to a static dim marker so the running state
# stays legible in a still frame"), which is the right switch to use rather
# than reaching into the widget's timers. Set before importing the app, since
# the flag is read at render time through settings/env.
os.environ["LOCAL_OPERATOR_NO_SHIMMER"] = "1"

from local_operator.harness.comms import SubagentComms  # noqa: E402
from local_operator.session.session import Session  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.subagent_panel import job_elapsed  # noqa: E402
from local_operator.tui.widgets.subagent_view import (  # noqa: E402
    TRAJECTORY_MAX_EVENTS as CAP,
)
from local_operator.tui.widgets.subagent_view import SubagentView  # noqa: E402
from local_operator.tui.widgets.tool_card import ToolCard  # noqa: E402
from local_operator.tui.widgets.transcript import WorkingBlock  # noqa: E402
from tests.unit.tui.test_band_panels import (  # noqa: E402
    FakeSession,
    _async_factory,
    _fake_jobs,
    _Job,
)

ERR = "Invalid arguments: argument 'edits' does not match type array"

#: The child's current activity, shown on the working line. Kept in one place
#: because it must be handed to BOTH the explicit ``show()`` and the job the
#: 1 Hz poll re-reads, or the two paints disagree (review round 2, D6).
PROGRESS = "Applying the review fixes"

#: Seconds of elapsed time the header reports. Fixed so the frame is stable.
_ELAPSED_S = 72.0

#: The child's role, which the header prints beside its label. Set on the job
#: as well as passed to ``show()`` for the same reason as PROGRESS.
ROLE = "coder"

#: The delegated brief the page prints above the child's work.
PROMPT = "Fix the duplicate notice rows in the subagent view."

#: How many 1 Hz refreshes to simulate. Each one appends a child event and
#: evicts the oldest, which is what re-spells the notice's positional key.
TICKS = 10


def _filler(index: int) -> dict[str, Any]:
    """An event that occupies a trajectory slot but paints no row.

    A ``message_start`` with no delta folds to an empty stream, and the fold
    spends no row on one — so the frame shows the notice defect rather than a
    wall of scaffolding.
    """
    return {"type": "message_start", "message": {"role": "assistant", "id": f"f{index}"}}


def _seed() -> list[dict[str, Any]]:
    # Exactly CAP events, because the defect only exists AT the cap: below it
    # nothing is evicted, every event keeps its offset, and the positional key
    # is accidentally stable. The four real events below occupy four slots.
    events: list[dict[str, Any]] = [_filler(i) for i in range(CAP - 4)]
    events += [
        {"type": "message_start", "message": {"role": "assistant", "id": "m-visible"}},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "id": "m-visible",
                "content": [{"type": "text", "text": "Applying the review fixes to the widget."}],
            },
        },
        {
            "type": "tool_execution_start",
            "tool_call_id": "call-edit",
            "tool_name": "edit",
            "intent": "Applying the review fixes",
            "args": {"path": "local_operator/tui/widgets/subagent_view.py"},
        },
        {"type": "notice", "kind": "error", "text": ERR},
    ]
    return events


def _pin_view_tail(view: SubagentView) -> None:
    """Anchor the body at its TAIL, deterministically.

    The before-frame overflows its viewport: eleven duplicate rows do not fit,
    which is the whole point of the artifact. The page opens at the tail and
    settles there over several refreshes, so a capture could land at one of two
    stable offsets and show either 7 or 11 of the duplicates — a difference in
    the headline number that has nothing to do with the fix (round 3, D6).

    Anchoring at the TAIL is both stable and the right frame to publish: it is
    where the page actually opens for a reader watching a live child, and it is
    the state the bug report showed — the newest rows, with the duplicates
    filling the viewport above the working line. The after-frame does not
    overflow, so this is a no-op for it and the pair stays comparable.
    """
    body = view._body
    body.scroll_end(animate=False)
    body.scroll_y = body.max_scroll_y
    body.scroll_target_y = body.max_scroll_y


def _pin_live_clocks(view: SubagentView) -> None:
    """Freeze the still-running tool card's own elapsed counter at 0s.

    A live ToolCard renders a counter driven by its OWN ``time.monotonic()``
    anchors, independent of the header clock this harness already pins. The
    settle loop is long enough that on a slower run the counter ticks 0s -> 1s,
    which made a fresh capture differ from the committed artifact by one glyph
    for a reason unrelated to the fix under review (round 3, D6). Re-anchoring
    to NOW makes the counter a function of the captured state rather than of
    how long the harness happened to take.
    """
    # The working line carries the same kind of clock and paints on the same
    # frame, so it is pinned from the same place — pinning it earlier let it
    # tick between the settle loop and the shot (round 3, D6).
    for line in view.query(WorkingBlock):
        line._phase_started = time.monotonic()
    for card in view.query(ToolCard):
        if getattr(card, "_started", None) is not None:
            card._started = time.monotonic()
        if getattr(card, "_compose_started", None) is not None:
            card._compose_started = time.monotonic()


async def main() -> None:
    job = _Job("child-1", "coder", status="running")
    # EVERY field the frame shows must live on the JOB, not only in the
    # ``show()`` call below. The app polls subagents once a second
    # (``_poll_subagents``) and re-``show()``s the page from the job itself, so
    # a value passed only as an argument here survives exactly until the next
    # tick and the captured frame then depends on which paint landed last.
    # That raced: two captures of this same script disagreed on the header and
    # the tail row (review round 2, D6).
    #
    # The clock is anchored to now for the same reason it is set at all: the
    # poll recomputes elapsed from ``start_time`` against the wall clock, so a
    # fixed epoch renders as "100d+".
    job.start_time = time.time() - _ELAPSED_S
    job.started_at = job.start_time
    # What the poll reads for the working line; without it the tail falls back
    # to the generic "thinking".
    job.latest_details = {"progress": PROGRESS}
    # The poll reads the ROLE from the job too. The fixture defaults to
    # ``task``, so leaving it made the header alternate between
    # ``coder · coder`` and ``coder`` depending on which paint landed last.
    job.agent_role = ROLE
    job.trajectory = _seed()
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    comms = SubagentComms(cast(Session, session))
    comms.record_launch(job.id, job.label)
    session._subagent_comms = comms

    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 32)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._open_subagent_view(job.id)
        await pilot.pause()
        view = app.query_one(SubagentView)

        # The 1 Hz poll: every refresh re-folds the SAME retained window after
        # the engine has evicted from its front. Nothing new happened; only
        # the notice's offset moved.
        for tick in range(TICKS):
            assert job.trajectory is not None
            # Re-anchored every tick, not once at launch: ``job_seconds``
            # measures a RUNNING job against ``time.time()``, so a single
            # anchor lets the header advance a second mid-capture and two runs
            # of this script disagree. Pinning it here fixes the rendered
            # clock at exactly _ELAPSED_S.
            job.start_time = time.time() - _ELAPSED_S
            job.started_at = job.start_time
            job.trajectory.append(_filler(10_000 + tick))
            overflow = len(job.trajectory) - CAP
            if overflow > 0:
                del job.trajectory[:overflow]
            view.show(
                job_id=job.id,
                label=job.label,
                status="running",
                queued=False,
                elapsed=job_elapsed(job),
                events=job.trajectory,
                prompt=PROMPT,
                progress=PROGRESS,
                agent_role=ROLE,
            )
            await pilot.pause()

        # The spinner glyph and the tool card's own elapsed are the only two
        # things left that advance with real time. Both are animation, not the
        # subject of this capture, so they are pinned to their first frame —
        # otherwise the committed artifact differs from a fresh run for a
        # reason the diff cannot explain (review round 2, D6).
        view._spinner_index = 0
        view._stop_spinner()
        view.show(
            job_id=job.id,
            label=job.label,
            status="running",
            queued=False,
            elapsed=job_elapsed(job),
            events=job.trajectory,
            prompt=PROMPT,
            progress=PROGRESS,
            agent_role=ROLE,
        )
        view._spinner_index = 0
        view._stop_spinner()
        # The working line keeps its own monotonic clock, so re-anchor it for
        # the same reason as the header's. Its 30 fps shimmer is handled by
        # LOCAL_OPERATOR_NO_SHIMMER, set at import above.
        for block in view.query(WorkingBlock):
            block._phase_started = time.monotonic()
        await pilot.pause()
        for block in view.query(WorkingBlock):
            block._phase_started = time.monotonic()
        # Let the layout SETTLE before the shot. The body mounts its rows and
        # applies its tail scroll over several refreshes, so a capture taken
        # too early lands mid-reflow and the frame is offset by a row or two
        # against an otherwise identical run — which is what made the
        # committed artifact unreproducible (review round 2, D6). Pausing to a
        # fixed point is what makes the capture a function of the state rather
        # than of timing.
        for _ in range(12):
            await pilot.pause()
        _pin_view_tail(view)
        for _ in range(6):
            await pilot.pause()

        # LAST, because every `show()` restarts the spinner for a running job:
        # stopping it earlier only holds until the next paint. Pinned to frame
        # 0 so the header glyph is a constant rather than whatever the 80 ms
        # timer had reached when the screenshot was taken.
        view._stop_spinner()
        view._spinner_index = 0
        view._chrome_state = None
        view._paint_chrome()
        # The still-running tool card keeps its OWN monotonic clock and renders
        # it as a live counter, so the settle loop above is long enough for it
        # to tick 0s -> 1s on a slower run. That one glyph was the last thing
        # making a fresh capture differ from the committed artifact for a
        # reason unrelated to the fix (review round 3, D6). Re-anchor both the
        # row's clock and its compose-time clock to NOW, immediately before the
        # shot, so the counter is a function of the state and not of how long
        # the harness happened to take.
        await pilot.pause()
        rows = [row for row in view.rendered_rows() if ERR in row]
        print(f"error rows painted: {len(rows)}")
        # LAST, with nothing awaited afterwards: `refresh()` marks the widget
        # dirty but `save_screenshot` renders from current state, so pinning
        # here sets the counter without giving the layout a chance to reflow.
        # Awaiting after this point re-runs the tail scroll and shifts every
        # row (round 3, D6).
        _pin_view_tail(view)
        _pin_live_clocks(view)
        app.save_screenshot(sys.argv[1])


asyncio.run(main())
