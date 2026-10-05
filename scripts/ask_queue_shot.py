"""Capture the queued-ask surfaces over a populated transcript (§5.1's B recipes).

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/ask_queue_shot.py OUT.svg [COLSxROWS] [MODE] [ROW]

MODE is one of (the design note's frame list, one frame per run so a still is
always the state its filename claims):

    bar          MINIMIZED, one open ask — the default presentation
    bar3         MINIMIZED, three open asks (the count and the head question)
    bar-timeout  MINIMIZED with nothing open but one timed-out ask still
                 answerable — the case where the count is not what is drawn
    list         EXPANDED onto the list of the three, with a timed-out row
    card         EXPANDED onto one ask's picker
    card-timeout the picker of an ask that timed out UNDER it (the title flips)
    list-late    the list with one ask answered after its deadline: it is absent
                 from the rows AND from the count
    list-refreshed the list after a snapshot drops two of its three asks (it
                 follows the wire; it used to keep the stale rows)
    list-long    the list with questions too wide for one row — the fixture the
                 wrap defect needed (round 3)
    list-next    the list after an ask was answered OUT of it: the wire's next
                 snapshot has dropped the answered row and the highlight has
                 ADVANCED to the next outstanding ask
    response     the ask_response card COLLAPSED
    response-open the same card with its Q&A open (ROW is ignored)
    late         the ask_response card for an answer that landed AFTER its own
                 deadline — the receipt that must read as warning, not dim
    timeout      the ask_timeout card collapsed, then opened with ROW=1

THE FLEET-SCOPE MODES (design §4/§11: settled rows, the three-way filter):

    list-settled       the list with the SETTLED half showing — newest-first
                 chips in place of the countdown, no answerable row at all
    filter-all         the list, All selected, a mixed queue
    filter-outstanding the middle half selected: only what is still owed
    filter-settled     the third half selected
    empty-outstanding  the middle half selected over a queue with nothing in it
    empty-settled      the third half selected over a queue with nothing in it
    list-fleet         the ONE list reading All conversations (rows from two
                 sessions, the scope subject in the header)
    list-fleet-empty   the fleet scope over an index with nothing outstanding
    truncated          the header stating the BACKEND tally and withholding the
                 waiting/moved-on split (a capped wire frame)

WHY THE TRANSCRIPT IS SEEDED FIRST. Every frame here has to answer "can the
user still read the conversation behind this surface?" — the bar is one row in
a dock that also carries the composer, and the list/card are panels above it.
An empty app would make that question unanswerable.

The same reason `ask_shot.py` gives for its own seeding: the seeded turns are
what make a still a picture of a STATE rather than of an empty screen.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.asks import policy  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.ask_queue import (  # noqa: E402
    FILTER_ALL,
    FILTER_OUTSTANDING,
    FILTER_SETTLED,
    SCOPE_FLEET,
    AskQueueList,
    ask_rows,
)
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.transcript import (  # noqa: E402
    AskResponseBlock,
    UserBlock,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: The flag is an env seam (D9). Set for the CAPTURE only: these frames are the
#: flag-ON surface, and the before-frames on `origin/main` are the flag-off app.
policy.NONBLOCKING_ASK = True

#: When the fixture's asks were created: a minute ago, so `created_at` precedes
#: `expires_at` by roughly the deadline the row carries.
_CREATED_AT = int(time.time() * 1000) - 60_000


def _row(
    ask_id: str,
    question: str,
    *,
    status: str = "open",
    urgent: bool = False,
    expires_at: int | None = None,
    delivered: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "ask_id": ask_id,
        # CREATED BEFORE IT EXPIRES BY ABOUT ITS OWN TIMEOUT, because the list
        # reserves the widest countdown a row can reach — derived from
        # ``expires_at - created_at`` — and a fixture with a 2023 creation and
        # a live deadline claims a three-year countdown, which reserves five
        # digits of hours and steals question cells no real ask would lose.
        "created_at": _CREATED_AT,
        # Overridable so a frame can show a deadline the client clock can still
        # measure; the fixed value is the "expiring" case, which is honest too.
        "expires_at": 1_700_003_600_000 if expires_at is None else expires_at,
        "timeout_s": 3600,
        "urgent": urgent,
        "status": status,
        "delivered": delivered,
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
    }


#: The queue the frames show, with deadlines relative to NOW rather than to a
#: fixed epoch: the list paints a countdown from ``expires_at`` against the
#: client's clock, so a frozen constant renders "expiring" for every row on
#: every machine and the frame would be evidence of nothing (design D8's fix is
#: only visible with a deadline that is still in the future).
_DEADLINE_MIN = 42
_URGENT_MIN = 4


def _deadline(minutes: int) -> int:
    return int(time.time() * 1000) + minutes * 60_000


class _AnswerableSession(FakeSession):
    """``FakeSession`` plus the ONE queued-ask op a submit needs.

    ``respond_ask`` is what tells the app that a card's settle was the LAST
    question answered (a submit) rather than an Escape (a partial map), so a
    frame that claims an ask was answered has to stand in front of a session
    that can take the answer. Its verdict is the owner's shape — ``{"ok": ...}``
    — because that is what the surfaces read.
    """

    def respond_ask(self, ask_id, answers, *, by="unknown"):
        return {"ok": True}


#: Longer than any width this capture set uses, so it MUST wrap if the row is
#: allowed to (round 3).
LONG_QUESTION = (
    "Which rollout should the stale-row migration take tonight, and which shard "
    "should take the read traffic while the backfill runs?"
)

THREE = [
    _row(
        "a1",
        "Which rollout should the stale-row migration take?",
        expires_at=_deadline(_DEADLINE_MIN),
    ),
    _row(
        "a2",
        "Rotate the deploy key before the cutover?",
        urgent=True,
        expires_at=_deadline(_URGENT_MIN),
    ),
    _row("a3", "Which region do we fail over to?", status="timed_out"),
]

#: The MIXED queue the filter frames draw: two halves with something in each,
#: so `All` / `Waiting or moved on` / `Settled` are three different views of one
#: list rather than three names for the same rows.
MIXED = [
    THREE[0],
    THREE[2],
    _row(
        "a4",
        "Backfill from the audit log or drop the column?",
        status="answered",
        delivered=True,
    ),
    _row("a5", "Should the retry budget double?", status="declined"),
]

#: The two single-half queues the empty-state frames need: each shows what the
#: OTHER half's sentence looks like over a queue that plainly has rows.
OPEN_ONLY = [
    _row(
        "a1",
        "Which rollout should the stale-row migration take?",
        expires_at=_deadline(_DEADLINE_MIN),
    ),
    _row(
        "a2",
        "Rotate the deploy key before the cutover?",
        urgent=True,
        expires_at=_deadline(_URGENT_MIN),
    ),
]
SETTLED_ONLY = [
    _row(
        "a4",
        "Backfill from the audit log or drop the column?",
        status="answered",
        delivered=True,
    ),
    _row("a5", "Should the retry budget double?", status="declined"),
]

#: The FLEET rows: two conversations' queues, each row carrying its own session
#: (which is what an answer is addressed by — never the session on screen).
FLEET = [
    dict(THREE[0], session_id="s-aida", cwd="/Users/damian/aida"),
    dict(THREE[2], ask_id="a9", session_id="s-pergamon", cwd="/Users/damian/pergamon"),
]

RESPONSE = {
    "ask_id": "a1",
    "status": "answered",
    "questions": [
        {
            "id": "q1",
            "question": "Which rollout should the stale-row migration take?",
            "options": [],
            "multi": False,
            "recommended": None,
            "secret": False,
            "persist": False,
        },
        {
            "id": "deploy_key",
            "question": "Paste the deploy key",
            "options": [],
            "multi": False,
            "recommended": None,
            "secret": True,
            "persist": False,
        },
    ],
    "answers": {"q1": ["Backfill from the audit log"], "deploy_key": ["[DEPLOY_KEY]"]},
    "text": "Answered — the whole ask was answered in one write.",
}

#: The late twin of RESPONSE: the SAME ``response`` kind, one status apart
#: (``late``), which is the whole point of the frame — the ink must follow the
#: severity the shared row computes, not the kind, or this receipt reads dim
#: beside an amber timeout that reports the same missed deadline (audit C).
LATE = {
    **RESPONSE,
    "ask_id": "a2",
    "status": "late",
    "text": "Answered late — the agent was told, one deadline too late.",
}


TIMEOUT = {
    "ask_id": "a3",
    "status": "timed_out",
    "waited_s": 3600,
    "urgent": False,
    "lapsed_while_stopped": False,
    "questions": [
        {
            "id": "q1",
            "question": "Which region do we fail over to?",
            "options": [],
            "multi": False,
            "recommended": None,
            "secret": False,
            "persist": False,
        }
    ],
    "text": "The deadline passed while the agent kept working.",
}


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    mode = "bar"
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    if len(sys.argv) > 3:
        mode = sys.argv[3]
    reveal = False
    if len(sys.argv) > 4:
        reveal = sys.argv[4].strip() in {"1", "reveal", "open", "true"}

    session = _AnswerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        # The queued-ask OPS are the session's and the surfaces resolve them
        # through the ADOPTED session, so the submit path has to be reachable
        # from the app (the surface tests adopt the same way before answering).
        app._session = session
        await pilot.pause()
        # `ASK_QUEUE_SHOT_THEME=light` renders the same state on the paper ramp.
        # An env seam rather than an argument, because the frame's FILENAME is
        # what a reader compares against: threading a theme through every mode's
        # name is how two runs of "the same" frame end up labelled alike.
        wanted = os.environ.get("ASK_QUEUE_SHOT_THEME", "").strip()
        if wanted:
            # The APP's own switch rather than ``theme_mod.set_theme`` +
            # ``refresh_css``. The theme is spread across four systems and the
            # markdown console theme is one of them (``_apply_theme``'s own
            # docstring) — and it is pushed at MOUNT, so the short spelling
            # re-inked the TCSS variables while every markdown block kept the
            # ramp it was BUILT with. That is how a light frame came out with
            # near-white prose on paper: half the frame, honestly unreadable
            # (design D16).
            app._apply_theme(wanted)
            await pilot.pause()
        for turn in range(1, 5):
            app._append_block(UserBlock(f"Turn {turn}: what should we do about the stale rows?"))
            prose = AssistantBlock()
            prose.update_text(
                f"Answer {turn}: the audit log still has every row, so a backfill is possible. "
                "Nothing else reads that column today."
            )
            app._append_block(prose)
        await pilot.pause()

        if mode == "bar":
            app._sync_ask_surface(ask_rows([THREE[0]]))
        elif mode == "bar3":
            app._sync_ask_surface(ask_rows(THREE))
        elif mode == "bar-timeout":
            app._sync_ask_surface(ask_rows([THREE[2]]))
        elif mode == "list":
            app._sync_ask_surface(ask_rows(THREE))
            await pilot.pause()
            app._expand_asks()
        elif mode == "list-late":
            # Round 1's late-answer state (round 1: UX U3 / design D6): the
            # first ask was answered AFTER its deadline, so the surfaces must
            # not offer it an answer box. Captured to show it is gone from the
            # list and from the bar's count while the open ask remains.
            # TWO asks stay open, so the frame shows the LIST — the surface
            # whose count and rows the late answer must not appear in — rather
            # than collapsing to a single ask's card.
            app._sync_ask_surface(ask_rows([dict(THREE[0], status="late"), THREE[1], THREE[2]]))
            await pilot.pause()
            app._expand_asks()
        elif mode == "list-long":
            # Round 3's wrap defect: questions that cannot fit one line. The
            # frame is the evidence that each ask spends exactly ONE painted
            # row, which is the assumption the pointer hit test rests on.
            app._sync_ask_surface(
                ask_rows(
                    [
                        _row("a1", LONG_QUESTION),
                        _row("a2", LONG_QUESTION, urgent=True),
                        _row("a3", LONG_QUESTION),
                    ]
                )
            )
            await pilot.pause()
            app._expand_asks()
        elif mode == "list-refreshed":
            # Round 1's stale-list state (QA Q3 / design D4): the panel used to
            # keep a row the wire had dropped. The frame is the AFTER half — the
            # list is fed a shorter snapshot while it is up, which is exactly
            # what used to leave "3 open asks" over a bar saying "1 question
            # waiting".
            app._sync_ask_surface(ask_rows(THREE))
            await pilot.pause()
            app._expand_asks()
            await pilot.pause()
            app._sync_ask_surface(ask_rows(THREE[:1]))
        elif mode == "list-next":
            # The re-entry frame (audit B): an ask answered OUT of the list must
            # hand the list back at the NEXT outstanding ask, not collapse to
            # the bar and make the user re-expand for every remaining ask. The
            # sequence is the real one — expand, pick a2, settle it, then the
            # wire's next snapshot drops the answered row — and the frame is the
            # state the user settles on: two rows with the highlight advanced to
            # a3. On the pre-fix tree the same sequence paints the minimized bar.
            app._sync_ask_surface(ask_rows(THREE))
            await pilot.pause()
            app._expand_asks()
            await pilot.pause()
            app.on_ask_queue_list_picked(AskQueueList.Picked("a2"))
            await pilot.pause()
            app._on_queue_ask_settle("a2", {"q1": ["Rotate after the cutover"]})
            await pilot.pause()
            # The next snapshot: the answered ask is gone, so this is also what
            # proves the highlight survives the wire dropping the row under it.
            app._sync_ask_surface(ask_rows([THREE[0], THREE[2]]))
        elif mode in (
            "list-settled",
            "filter-all",
            "filter-outstanding",
            "filter-settled",
            "empty-outstanding",
            "empty-settled",
        ):
            # The filter frames (design §4 / §11). One queue each, chosen so the
            # still is the state its filename claims: a MIXED queue for the three
            # half views, and a single-half queue for each empty-state sentence
            # ("No asks are waiting or moved on..." is only honest over a queue
            # that visibly has settled rows in it).
            queue = {
                "list-settled": SETTLED_ONLY,
                "filter-all": MIXED,
                "filter-outstanding": MIXED,
                "filter-settled": MIXED,
                "empty-outstanding": SETTLED_ONLY,
                "empty-settled": OPEN_ONLY,
            }[mode]
            app._sync_ask_surface(ask_rows(queue))
            await pilot.pause()
            app._expand_asks()
            await pilot.pause()
            picked = {
                "filter-all": FILTER_ALL,
                "filter-outstanding": FILTER_OUTSTANDING,
                "filter-settled": FILTER_SETTLED,
                "empty-outstanding": FILTER_OUTSTANDING,
                "empty-settled": FILTER_SETTLED,
            }.get(mode)
            if picked is not None:
                app.query_one(AskQueueList).set_filter(picked)
        elif mode in {"list-fleet", "list-fleet-empty"}:
            # THE FLEET SCOPE. Set the way the door sets it (`_ask_scope` +
            # `_ask_fleet_rows`, then the mount) rather than through a click:
            # the frame is about what the LIST paints for a scope, and the
            # sidebar's own door is captured by `scripts/ask_fleet_shot.py`.
            app._ask_scope = SCOPE_FLEET
            app._ask_fleet_rows = ask_rows(FLEET if mode == "list-fleet" else [])
            app._mount_ask_list(scope=SCOPE_FLEET)
        elif mode == "truncated":
            # A CAPPED wire frame: the backend says seven are outstanding while
            # the rows are a prefix. The header states the tally and WITHHOLDS
            # the waiting/moved-on split (A6) — a prefix must not pass for the
            # whole queue.
            app._sync_ask_surface(ask_rows(THREE), open_count=7, truncated=True)
            await pilot.pause()
            app._expand_asks()
        elif mode == "card-timeout":
            # Round 1 (UX U9): the card kept saying the agent was waiting after
            # the ask's own deadline had fired. The status changes UNDER the
            # mounted card here, which is the case the title hook exists for.
            app._sync_ask_surface(ask_rows([THREE[0]]))
            await pilot.pause()
            app._expand_asks()
            await pilot.pause()
            app._sync_ask_surface(ask_rows([dict(THREE[0], status="timed_out")]))
        elif mode == "card":
            app._sync_ask_surface(ask_rows([THREE[0]]))
            await pilot.pause()
            app._expand_asks()
        elif mode in {"response", "response-open", "timeout", "late"}:
            details = {"timeout": TIMEOUT, "late": LATE}.get(mode, RESPONSE)
            kind = "timeout" if mode == "timeout" else "response"
            block = AskResponseBlock(details, kind=kind)
            app._append_block(block)
            await pilot.pause()
            if reveal:
                block.action_activate()
                # The expansion opens BELOW the last line of a transcript that is
                # already at the viewport's end, so without this the still shows
                # the collapsed row and the Q&A sits off-screen — two frames
                # that differ only in the widget's state, not in the pixels.
                # Explicit rather than left to the tail anchor: the anchor
                # follows content that GROWS while the reader is at the bottom,
                # and this is the capture confirming it did.
                try:
                    app._transcript_view().scroll_end(animate=False)
                except Exception:  # pragma: no cover - harness shape only
                    pass
        else:  # pragma: no cover - a typo must not produce a silent empty frame
            raise SystemExit(f"unknown mode {mode!r}")
        for _ in range(12):
            await pilot.pause()
        save_capture(app, out)


asyncio.run(main())
