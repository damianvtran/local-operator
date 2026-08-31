"""Render the subagent page in the three states issues #405/#406/#407 describe.

Drives the REAL ``OperatorApp`` (so ``local_operator.tcss`` applies), following
the determinism discipline documented in ``docs/assets/pr-notice-key/README.md``:
every displayed field lives on the JOB (the 1 Hz poll re-``show()``s from it),
the elapsed clock is re-anchored on every tick, shimmer is off, the spinner is
stopped last, the layout is settled to a fixed point, and the live clocks are
pinned with nothing awaited afterwards.

Usage:
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python savf_shot.py \
        <scenario> <out.svg> [WxH]

Scenarios: dupe | failed | narrow
Run from the repo root (or a worktree of it) — ``tests.`` must import.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.getcwd())

os.environ["LOCAL_OPERATOR_NO_SHIMMER"] = "1"

from local_operator.harness.jobs import TRAJECTORY_SEQ_KEY  # noqa: E402
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
PROGRESS = "Applying the review fixes"
ROLE = "coder"
PROMPT = "Fix the duplicate notice rows in the subagent view."
_ELAPSED_S = 77.0

_seq = 0


def _stamp(event: dict[str, Any]) -> dict[str, Any]:
    """Give the event the writer's monotonic stamp the real relay assigns.

    Every relayed event carries one, so the fixtures have to as well or the
    fold takes the legacy fingerprint path instead of the one production uses.
    """
    global _seq
    _seq += 1
    return {TRAJECTORY_SEQ_KEY: _seq, **event}


def _filler(index: int) -> dict[str, Any]:
    """An event that occupies a trajectory slot but paints no row."""
    return _stamp({"type": "message_start", "message": {"role": "assistant", "id": f"f{index}"}})


def _errored_edit(call: str, path: str) -> list[dict[str, Any]]:
    """One `edit` call that starts and then fails with the arguments error."""
    return [
        _stamp(
            {
                "type": "tool_execution_start",
                "tool_call_id": call,
                "tool_name": "edit",
                "intent": "Applying the review fixes",
                "args": {"path": path},
            }
        ),
        _stamp(
            {
                "type": "tool_execution_end",
                "tool_call_id": call,
                "is_error": True,
                "duration_s": 0.02,
                "result": {"content": [{"type": "text", "text": ERR}], "is_error": True},
            }
        ),
    ]


def _events(scenario: str) -> list[dict[str, Any]]:
    if scenario == "dupe":
        # A child that genuinely emits the SAME notice twice in a row: the
        # frame issue #405 is about. Distinct sequence stamps, so the fold
        # keeps two rows — the data model is right and the rows are the ones
        # a reader cannot tell from the old duplicate-render bug.
        return [
            _stamp(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "id": "m-visible",
                        "content": [
                            {"type": "text", "text": "Applying the review fixes to the widget."}
                        ],
                    },
                }
            ),
            _stamp(
                {
                    "type": "tool_execution_start",
                    "tool_call_id": "call-edit",
                    "tool_name": "edit",
                    "intent": "Applying the review fixes",
                    "args": {"path": "local_operator/tui/widgets/subagent_view.py"},
                }
            ),
            _stamp({"type": "notice", "kind": "error", "text": ERR}),
            _stamp({"type": "notice", "kind": "error", "text": ERR}),
        ]
    if scenario == "failed":
        # The header case (#406): two `edit` calls, both errored, on a page
        # that still reads `running · … · 2 tools`.
        events: list[dict[str, Any]] = [
            _stamp(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "id": "m-visible",
                        "content": [
                            {"type": "text", "text": "Applying the review fixes to the widget."}
                        ],
                    },
                }
            )
        ]
        events += _errored_edit("call-edit-1", "local_operator/tui/widgets/subagent_view.py")
        events += _errored_edit("call-edit-2", "tests/unit/tui/test_subagent_view.py")
        return events
    if scenario == "narrow":
        # A TRUNCATED child whose last error wraps several times at 62
        # columns, with enough PAINTED rows above it that scroll-to-tail
        # lands inside the wrap — the #407 frame. Fillers that paint no
        # row cannot overflow the viewport, so the overflow is real
        # assistant lines, not cap-padding.
        long_err = (
            f"{ERR} while calling edit on "
            "local_operator/tui/widgets/subagent_view.py "
            "with a payload that also failed validation on every subsequent "
            "field of the same call: path, old_text, new_text, replace_all, "
            "and the trailing context the child included to justify the edit"
        )
        # Fillers occupy the cap so TRUNCATION_NOTE is mounted; the painted
        # overflow is a handful of real rows plus a notice taller than the
        # 62x24 body, which is what puts a continuation line at the top of
        # the viewport before the landing snap.
        events = [_filler(i) for i in range(CAP - 5)]
        for index in range(2):
            events.append(
                _stamp(
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "id": f"m-{index}",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Step {index}: applying the review " "fixes to the widget."
                                    ),
                                }
                            ],
                        },
                    }
                )
            )
        events += _errored_edit("call-edit-1", "local_operator/tui/widgets/subagent_view.py")
        events.append(_stamp({"type": "notice", "kind": "error", "text": long_err}))
        return events
    if scenario == "trunc":
        # Same overflow as `narrow`, captured after scrolling HOME so the
        # truncation note's presence (and, after the fix, its pin) is in
        # frame rather than off the top of a tail-following viewport.
        return _events("narrow")
    raise SystemExit(f"unknown scenario {scenario!r}")


def _pin_view_tail(view: SubagentView) -> None:
    body = view._body
    body.scroll_end(animate=False)
    body.scroll_y = body.max_scroll_y
    body.scroll_target_y = body.max_scroll_y


def _pin_live_clocks(view: SubagentView) -> None:
    for line in view.query(WorkingBlock):
        line._phase_started = time.monotonic()
    for card in view.query(ToolCard):
        if getattr(card, "_started", None) is not None:
            card._started = time.monotonic()
        if getattr(card, "_compose_started", None) is not None:
            card._compose_started = time.monotonic()


def _geometry(view: SubagentView) -> dict[str, Any]:
    """The numbers behind the pixels: where the body sits and what heads it.

    ``scroll_offset.y`` is the first CONTENT row on screen. A block's region
    ``y`` is in the same coordinate space, so a first visible row that is a row
    HEAD is one where some block starts exactly at that offset.
    """
    body = view._body
    offset = int(body.scroll_offset.y)
    blocks = body.blocks()
    starts = []
    for block in blocks:
        try:
            starts.append((int(block.region.y + offset - body.region.y), block.__class__.__name__))
        except Exception:  # pragma: no cover - unmounted block
            continue
    at_head = any(start == offset for start, _ in starts)
    owner = None
    for start, name in starts:
        if start <= offset:
            owner = (start, name)
    return {
        "scroll_y": offset,
        "max_scroll_y": int(body.max_scroll_y),
        "viewport_h": int(body.size.height),
        "virtual_h": int(body.virtual_size.height),
        "first_visible_is_row_head": at_head,
        "owner_block_start": owner[0] if owner else None,
        "owner_block": owner[1] if owner else None,
        "rows_into_owner": (offset - owner[0]) if owner else None,
        "block_starts": starts[:40],
    }


async def main() -> None:
    scenario = sys.argv[1]
    out = sys.argv[2]
    size = sys.argv[3] if len(sys.argv) > 3 else "100x30"
    width, height = (int(part) for part in size.lower().split("x"))

    job = _Job("child-1", "child", status="running")
    job.start_time = time.time() - _ELAPSED_S
    job.started_at = job.start_time
    job.latest_details = {"progress": PROGRESS}
    job.agent_role = ROLE
    job.prompt = PROMPT
    job.trajectory = _events(scenario)

    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(width, height)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._refresh_band()
        await pilot.pause()
        app._open_subagent_view(str(job.id))
        for _ in range(8):
            await pilot.pause()
        view = app.query_one(SubagentView)
        for _ in range(4):
            job.start_time = time.time() - _ELAPSED_S
            job.started_at = job.start_time
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
        view._spinner_index = 0
        view._stop_spinner()
        for _ in range(12):
            await pilot.pause()
        # The landing position is the SUBJECT of the narrow scenario, so it is
        # left exactly where the page put it. `trunc` is the same overflow
        # scrolled HOME so the truncation note is in frame. The other two
        # overflow for reasons unrelated to their subject, so they are pinned
        # at the tail the way the PR #404 artifact is.
        if scenario == "trunc":
            view._body.scroll_home(animate=False)
            view._body.scroll_y = 0
            view._body.scroll_target_y = 0
        elif scenario != "narrow":
            _pin_view_tail(view)
        for _ in range(6):
            await pilot.pause()
        view._stop_spinner()
        view._spinner_index = 0
        view._chrome_state = None
        view._paint_chrome()
        await pilot.pause()
        rows = view.rendered_rows()
        report = {
            "scenario": scenario,
            "size": [width, height],
            "title": rows[0],
            "error_rows": [row for row in rows if ERR in row],
            "geometry": _geometry(view),
            "visible_rows": [row for row in rows if row.strip()][:40],
        }
        print(json.dumps(report, indent=2))
        _pin_live_clocks(view)
        app.save_screenshot(out)


asyncio.run(main())
