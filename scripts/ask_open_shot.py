"""Capture the queued-ask surface's OPEN-BY-DEFAULT states over a populated transcript.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/ask_open_shot.py OUT.svg [COLSxROWS] [MODE]

MODE is one of (one frame per run, so a still is the state its filename claims):

    no-asks          a conversation with nothing waiting             (clause 1)
    pending          opened on ONE pending ask                       (clause 2)
    pending-list     opened on TWO pending asks                      (clause 2)
    pending-typed    the same, after the user typed "3 more" at once (clause 5)
    addressed        opened on a queue of only settled rows          (clause 3)
    dismissed-open   the surface the policy opened, before the close (clause 4, step 1)
    dismissed-closed ...after the user pressed f4 to close it        (clause 4, step 2)
    dismissed-back   ...after a re-render, a new ask, and a switch away and back
                     (clause 4, step 3: it must STILL be closed)
    list-tab         the auto-opened list after Tab handed it the caret

THE SAME SCRIPT RUNS ON BOTH BUILDS IT CAPTURES, which is why it never imports the
policy module: the before-frames come from a detached worktree of ``origin/main``,
where nothing opens a surface on its own, and the after-frames from the branch. A
script that needed the new code would have no before. Everything it touches
(``SidebarRemote``, ``_switch``) is older than the feature.

FRAMES TRAVEL THE PRODUCTION PATH: asks are published into a real
``FrontendStateStore``, the app folds the frame as it does for any session, and a
switch goes through the sidebar's real prepare/commit pair. ``_sync_ask_surface`` is
never called by hand, because the decision to open is taken inside that fold and a
hand-fed call would skip exactly the edge (the first frame landing inside a switch)
the feature has to get right.

WHY THE CONVERSATION CARRIES A HISTORY: every frame has to answer "can the user still read
the conversation behind this surface, and is the caret still in the composer?". An empty
app makes both unanswerable. The history rides on the TARGET conversation (``Convo``) and is
replayed by the real switch, not appended to the app first: a sidebar switch mounts a NEW
transcript view for the target, so blocks added to the home view beforehand vanish with it
and every frame photographs the splash (the first cut of this script did exactly that, and
its frames looked fine while answering neither question).
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.asks import policy  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from tests.unit.tui.test_app_pilot import _factory  # noqa: E402
from tests.unit.tui.test_sidebar_swap_reset import (  # noqa: E402
    SidebarRemote,
    _message,
    _switch,
)

#: The flag is an env seam; these frames are the flag-ON surface.
policy.NONBLOCKING_ASK = True


def _row(
    ask_id: str,
    question: str,
    *,
    status: str = "open",
    age_ms: int = 60_000,
    **extra: Any,
) -> dict[str, Any]:
    """One wire row. ``age_ms`` is how long ago it was asked, relative to NOW.

    A minute by default, which is "already waiting when the conversation opened"; the
    arrival case (a question the agent raises while the user is looking) passes a
    negative age. Dates are relative because the list paints a countdown from
    ``expires_at`` against the client clock, and a frozen epoch renders every row as
    expired — a frame that is evidence of nothing.
    """
    created = int(time.time() * 1000) - age_ms
    return {
        "ask_id": ask_id,
        "created_at": created,
        "expires_at": created + 3_600_000,
        "timeout_s": 3600,
        "urgent": False,
        "status": status,
        "delivered": status == "answered",
        "questions": [
            {
                "id": "q1",
                "question": question,
                "options": [
                    {"label": "Drop the rows", "description": "nothing reads the column"},
                    {
                        "label": "Backfill from the audit log",
                        "description": "slower, keeps history",
                    },
                ],
                "multi": False,
                "recommended": None,
                "secret": False,
                "persist": False,
            }
        ],
        **extra,
    }


def _wire(rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    """The two frame fields a runtime publishes (``frontend_state.ask_wire``'s shapes)."""
    if rows is None:
        return {"asks": None, "asks_open": None}
    outstanding = sum(1 for row in rows if row["status"] in ("open", "timed_out"))
    return {"asks": list(rows) or None, "asks_open": outstanding}


class Convo(SidebarRemote):
    """An owner-backed conversation on a real frontend store, carrying a queue."""

    def __init__(self, session_id: str, rows: list[dict[str, Any]] | None = None) -> None:
        super().__init__(session_id, history=_turns())
        if rows is not None:
            self.publish(rows)

    def publish(self, rows: list[dict[str, Any]] | None) -> None:
        self._store.mutate(**_wire(rows))

    def respond_ask(self, ask_id, answers, *, by="unknown"):
        return {"ok": True}

    def decline_ask(self, ask_id, *, by="unknown"):
        return {"ok": True}

    def dismiss_ask(self, ask_id, *, by="unknown"):
        return {"ok": True}


QUESTION_ONE = "Which rollout should the stale-row migration take?"
TWO = [
    _row("a1", "Which rollout should the stale-row migration take?"),
    _row("a2", "Rotate the deploy key before the cutover?"),
]
SETTLED = [
    _row("s1", "Backfill from the audit log or drop the column?", status="answered"),
    _row("s2", "Should the retry budget double?", status="declined"),
]


def _turns() -> list[Any]:
    """Four turns of conversation, so a surface is judged against something to read."""
    out: list[Any] = []
    for turn in range(1, 5):
        out.append(_message("user", f"Turn {turn}: what should we do about the stale rows?"))
        out.append(
            _message(
                "assistant",
                f"Answer {turn}: the audit log still has every row, so a backfill is possible. "
                "Nothing else reads that column today.",
            )
        )
    return out


async def _pump(pilot, turns: int = 30) -> None:
    for _ in range(turns):
        await pilot.pause(0.03)


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    mode = "pending"
    for argument in sys.argv[2:]:
        if "x" in argument and argument.replace("x", "").isdigit():
            columns, rows_ = argument.split("x")
            size = (int(columns), int(rows_))
        else:
            mode = argument

    home = Convo("conv-home")
    target_rows = {
        "no-asks": [],
        "pending": [_row("a1", QUESTION_ONE)],
        "pending-typed": [_row("a1", QUESTION_ONE)],
        "pending-list": TWO,
        "list-tab": TWO,
        "addressed": SETTLED,
        "dismissed-open": TWO,
        "dismissed-closed": TWO,
        "dismissed-back": TWO,
    }
    if mode not in target_rows:
        raise SystemExit(f"unknown mode {mode!r}; one of {sorted(target_rows)}")
    target = Convo("conv-a", target_rows[mode])
    elsewhere = Convo("conv-b")

    app = OperatorApp(lambda: _factory(home))
    with patch("local_operator.session.attached.AttachedSession", Convo):
        async with app.run_test(size=size) as pilot:
            await _pump(pilot, 10)
            # Opened by a switch, like a user clicking the conversation in the sidebar:
            # the path where the first frame lands inside the adopt.
            await _switch(app, pilot, target)
            await _pump(pilot)

            if mode == "pending-typed":
                # The user starts typing the moment the conversation is up. Every
                # character must land in the composer, none of them in the question.
                await pilot.press("3", "space", "m", "o", "r", "e")
                await _pump(pilot, 10)
            elif mode == "list-tab":
                await pilot.press("tab")
                await _pump(pilot, 10)
            elif mode.startswith("dismissed"):
                # Close it the way a user does (the toggle) — but only if something
                # opened: on a build with no open-by-default the same press would OPEN
                # it, and the frame would be evidence of the script rather than the app.
                if mode != "dismissed-open" and app._ask_mode:
                    await pilot.press("f4")
                    await _pump(pilot, 10)
                if mode == "dismissed-back":
                    # A re-render, a queue refresh, a brand-new ask, then away and back.
                    target._store.mutate(cumulative_parent_cost=1.5)
                    await _pump(pilot, 6)
                    target.publish([*TWO, _row("a3", "Roll back on failure?", age_ms=-60_000)])
                    await _pump(pilot, 6)
                    await _switch(app, pilot, elsewhere)
                    await _switch(app, pilot, target)
                    await _pump(pilot)

            await pilot.pause()
            save_capture(app, out)
            print(
                f"{mode}: ask_mode={app._ask_mode} focus={type(app.focused).__name__} "
                f"conversation={app._conversation_id()}"
            )


asyncio.run(main())
