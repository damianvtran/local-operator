"""Serve the REAL mobile bundle with synthetic projections that provoke both
defects under review, for before/after capture at a phone viewport.

Run:  PYTHONPATH=. .venv/bin/python scripts/mobile_overflow_fixture.py [PORT]
Login at http://127.0.0.1:<port>. The password is NOT fixed and never printed:
it does NOT default and does NOT print one: export `LOP_MOBILE_FIXTURE_PASSWORD`
(or pass it as the second argument) and give the capture scripts the same variable,
or the fixture refuses to start.

Three sessions, each one shaped to isolate one surface:

* ``roster``   — 22 subagent rows, exactly one running. This is the operator's
  reported screen: ``defaultOpen={running > 0}`` opens the roster on arrival.
* ``ask-long`` — a long question plus 10 options carrying paragraph-length
  descriptions, the shape that overruns the card.
* ``approval`` — the approval variant of the same card (approve/deny +
  remember), which overflows identically.
* ``ask-free`` — the free-text/secret variant, to prove the cap does not strand
  the input or its send button.
* ``stacked`` / ``stacked-approval`` — todos panel AND subagent roster AND a
  pending request in ONE column. This is the screen the operator actually has,
  and the one the per-panel caps could not bound: each region was individually
  capped while their SUM overran the column, which is `overflow-hidden`, so the
  decision controls were clipped rather than scrolled (D1).
* ``stale`` — an ask whose answers are refused as moved-on, so the error line's
  position can be measured on arrival rather than inferred (U3).
* ``failures`` — a fan-out with failed agents, for the collapsed header's
  failure count (U5).
* ``failures-pending`` — the same fan-out WITH a request pending, so the
  failure count and the held-shut panel state appear at once. The compounding
  case: `forceClosed` dims the header, and a dim above the count's level takes
  the count with it (design D4). No other fixture puts both on screen, which is
  why the dim's cost to that glyph went unmeasured until round 2.
* ``asks`` — the QUEUED-ASK surface (design
  ``docs/design/ask-nonblocking.md`` §5.3): five queued asks covering open (one
  multi-question head), addressed, and TIMED-OUT-and-still-answerable, plus two
  settled ones; the transcript carries an ``ask_response`` and an
  ``ask_timeout`` row; and the legacy single-slot mirror is published beside
  them so a client drawing the ask twice would show it here. The asks are also
  seeded into the daemon's INDEX, which is what ``GET /api/asks`` reads.
* ``asks-settled`` — the same surface at ZERO outstanding asks (the bar must be
  absent), with only the settling rows left in the transcript.

No runtime scanner and no registrant sockets (``dial_registrants=False``), so
this never touches the operator's live daemon or their sessions. HOME and
LOCAL_OPERATOR_CONFIG_DIR are re-homed by ``scripts.probe_isolation`` on import.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any

import uvicorn

import scripts.probe_isolation  # noqa: F401  -- must be the first local import
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, _dial, build_app
from local_operator.mobile.types import (
    AskOptionWire,
    PendingAskWire,
    PendingRequest,
    SessionProjection,
    SessionRecord,
    SubagentRow,
    TodoItem,
    TodoPhase,
    TranscriptEntry,
)
from local_operator.session.runtime.server import RuntimeServer  # noqa: E402

#: The daemon's password for THIS run: the caller's second argument, or a value
#: generated here.
#:
#: WHY THERE IS NO LITERAL. A fixed password is REUSABLE: it outlives the fixture,
#: it lives in the repo, and every script that imports it shares one value — so it
#: ended up in transcripts that merely READ this file, and it had to be rotated.
#: The replacement is per-run and loopback-only, and NOTHING IN THIS FILE PRINTS IT
#: (the startup banner names the port only). A caller that wants to log in by hand,
#: or a capture script that fills the login form, supplies the value explicitly —
#: which is also the only way it is ever shared.
#: The variable NAME (never a value) a caller may use instead of the second argument.
#: (``scripts/mobile_overflow_capture.py``). Written out here rather than imported
#: from there ON PURPOSE: these are sibling capture scripts, and importing one from
#: the other would drag Chrome/CDP code into a fixture that only serves a daemon.
FIXTURE_PASSWORD_ENV = "LOP_MOBILE_FIXTURE_PASSWORD"


def required_password(argv_rest: list[str]) -> str:
    """The per-run password, or a refusal that names how to supply one.

    THERE IS NO DEFAULT AND NO GENERATED FALLBACK, and that is the point: a fixed
    fallback is a reusable credential the moment two runs share it, which is exactly
    how the previous value became one that had to be rotated; and a SILENTLY generated
    one leaves a human unable to log in, which invites a fixed one back. The caller
    supplies it — the capture scripts read the same variable — and nothing here prints
    it, so neither the value nor a placeholder for it appears in any source or log.
    """
    value = argv_rest[0] if argv_rest else os.environ.get(FIXTURE_PASSWORD_ENV, "")
    if not value:
        raise SystemExit(
            "this fixture needs a per-run password: pass it as the second argument, or "
            f"set {FIXTURE_PASSWORD_ENV}. Generate one with "
            "python -c 'import secrets;print(secrets.token_urlsafe(16))' and export it. "
            "It is never defaulted and never printed by this script."
        )
    return value


LONG_QUESTION = (
    "The remediation touches the mobile projection wire, the phone's ask card, "
    "the subagent roster and the session layout column, and each of those has a "
    "different blast radius on the terminal viewer. Which sequencing do you want "
    "for the rollout, given that the phone bundle ships inside the wheel and the "
    "terminal viewer reads the same projection fold?"
)

OPTION_DESCRIPTIONS = [
    "Land the layout cap first and leave the roster default alone, so the "
    "smallest possible diff reaches the released wheel and the roster change "
    "can be judged against a card that already cannot overflow.",
    "Land the roster default first, because it is the change the operator "
    "actually reported, and accept that a long ask still clips until the "
    "second patch lands in the following release window.",
    "Land both together in one patch, which is the shape under review here: "
    "the two defects share the same layout column and reviewing them apart "
    "means reading the same contract twice.",
    "Hold both behind a feature flag read off the projection, so a phone on an "
    "older bundle keeps today's behaviour and the rollout can be reverted "
    "without cutting a release.",
    "Split the work by surface rather than by defect: one patch for every "
    "collapsible panel in the column, another for every card pinned above the "
    "composer, which is a larger diff but a cleaner contract.",
    "Rewrite the column as a single scroll container with sticky header and "
    "composer, which removes the whole class of defect and is far too large a "
    "change to review inside this window.",
    "Defer everything to the next minor and ship only a documentation note in "
    "AGENTS.md describing the overflow, which leaves the operator's phone "
    "broken but costs nothing to review.",
    "Cap the card at a fixed pixel height instead of a viewport fraction, "
    "which is simpler to reason about and wrong on every device whose screen "
    "is not the one it was measured on.",
    "Move the ask card into a modal sheet over the transcript, reusing the "
    "existing sheet idiom and its scroller, at the cost of hiding the "
    "conversation the question is about.",
    "Escalate to the operator with the measured frames and let him pick the "
    "sequencing, which is what this fixture's evidence is for and the option "
    "that has to stay reachable at the bottom of a ten-option list.",
]

APPROVAL_DETAIL = (
    'bash — export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH" && cd '
    "~/workspace/repos/lo-mobile-ask-collapse/local_operator/mobile/web && "
    "pnpm install --frozen-lockfile && pnpm build && pnpm test && "
    "cd ../../.. && .venv/bin/python -m flake8 local_operator && "
    ".venv/bin/python -m black --check local_operator && "
    ".venv/bin/python -m pytest tests/unit/mobile -q\n\n"
    "The command rebuilds the phone bundle in place and runs the vitest suite. "
    "It writes into the worktree's dist/ directory, which is gitignored, and it "
    "reads no credentials. Approving it authorises the whole pipeline, including "
    "the prebuild step that regenerates themes.generated.css and src/lib/mark.ts "
    "from their sources, so a stale generated file becomes a real diff in the "
    "working tree rather than a silent mismatch at review time.\n\n"
    "It then runs the Python gates against the same worktree. Those read the "
    "worktree's own .venv, which is installed editable from this checkout, so "
    "nothing here resolves the parent repository's tree — verified by importing "
    "local_operator and printing its __file__ before the run. The pytest "
    "selection is scoped to tests/unit/mobile because the diff under review is "
    "confined to the phone bundle's TypeScript, and the root conftest caps xdist "
    "worker count so a concurrent sibling worktree is not starved of memory.\n\n"
    "Nothing in this pipeline reaches the network beyond the pnpm store, which "
    "is already populated, and nothing touches the operator's live daemon, "
    "their real tunnel, or any registrant socket. Denying it leaves the tree "
    "exactly as it is; the only cost is that the evidence for the review round "
    "has to be regenerated by hand afterwards, which is slower but not lossy."
)


def _roster_projection() -> SessionProjection:
    """22 rows, one running — the reported ``subagents 1/22 running`` screen.

    The transcript is deliberately non-empty: the defect is the roster pushing a
    real conversation off the top, and an empty transcript would hide it behind
    the "no messages yet" placeholder instead.
    """
    projection = SessionProjection(
        session_id="roster",
        pid=900001,
        kind="tui",
        conversation_name="Roster overflow",
        streaming=True,
        activity="coordinating remediation",
        activity_started_s=612,
        transcript=[
            TranscriptEntry(
                id=f"r-{i}",
                kind="user" if i % 2 == 0 else "assistant",
                text=(
                    "Audit every mobile surface against the phone viewport."
                    if i % 2 == 0
                    else "Fanning the audit out across one subagent per surface; "
                    "this line is the conversation the roster must not cover."
                ),
            )
            for i in range(6)
        ],
        version=7,
    )
    projection.subagents = [
        SubagentRow(
            job_id="row-00",
            label="viewport-audit-running",
            agent="reviewer",
            status="running",
            progress="measuring the session column",
            elapsed_s=612,
            parent_job_id=None,
            session_id="roster-child-00",
            activity="measuring the session column",
        )
    ] + [
        SubagentRow(
            job_id=f"row-{i:02d}",
            label=f"surface-audit-{i:02d}",
            agent="coder" if i % 2 else "designer",
            status="completed",
            elapsed_s=30 + i,
            parent_job_id=None,
            session_id=f"roster-child-{i:02d}",
            result_text=f"Surface {i} measured.",
        )
        for i in range(1, 22)
    ]
    return projection


def _ask_projection() -> SessionProjection:
    projection = SessionProjection(
        session_id="ask-long",
        pid=900002,
        kind="tui",
        conversation_name="Ask overflow",
        streaming=False,
        transcript=[
            TranscriptEntry(
                id="a-1",
                kind="user",
                text="Decide the rollout sequencing for the mobile layout fixes.",
            ),
            TranscriptEntry(
                id="a-2",
                kind="assistant",
                text="Both defects share the session column, so the sequencing matters.",
            ),
        ],
        version=3,
    )
    projection.pending = PendingRequest(
        request_id="req-ask-long",
        kind="ask",
        title=LONG_QUESTION,
        options=[
            AskOptionWire(label=f"option-{i + 1:02d}", description=text)
            for i, text in enumerate(OPTION_DESCRIPTIONS)
        ],
    )
    projection.pending_count = 1
    return projection


def _approval_projection() -> SessionProjection:
    projection = SessionProjection(
        session_id="approval",
        pid=900003,
        kind="tui",
        conversation_name="Approval overflow",
        streaming=False,
        transcript=[
            TranscriptEntry(id="p-1", kind="user", text="Rebuild the bundle and run the suite."),
        ],
        version=2,
    )
    projection.pending = PendingRequest(
        request_id="req-approval",
        kind="approval",
        title="bash",
        detail=APPROVAL_DETAIL,
    )
    projection.pending_count = 2
    return projection


def _free_text_projection() -> SessionProjection:
    projection = SessionProjection(
        session_id="ask-free",
        pid=900004,
        kind="tui",
        conversation_name="Secret ask",
        streaming=False,
        transcript=[
            TranscriptEntry(id="f-1", kind="user", text="Load the staging token."),
        ],
        version=2,
    )
    projection.pending = PendingRequest(
        request_id="req-ask-free",
        kind="ask",
        title=LONG_QUESTION,
        detail="\n\n".join(OPTION_DESCRIPTIONS[:4]),
        secret=True,
        persist=True,
    )
    projection.pending_count = 1
    return projection


def _stacked_projection(session_id: str, approval: bool) -> SessionProjection:
    """Todos + roster + a pending request in one column (D1).

    The panels are what make this distinct from the plain ask/approval
    scenarios: each region carries its own cap, but caps do not compose — two
    panels at 40% plus a card at 60% demand 140% of a column that cannot
    scroll, so whatever lands past the foot is CLIPPED. Measured before the
    fix: approve/deny 120px below the fold at 390x844 with the card's own
    scroller already at its end, and at 360x780 the card's top at y=781 in a
    780px viewport.
    """
    projection = SessionProjection(
        session_id=session_id,
        pid=900005 if approval else 900006,
        kind="tui",
        conversation_name="Stacked panels",
        streaming=True,
        activity="coordinating remediation",
        activity_started_s=612,
        transcript=[
            TranscriptEntry(
                id=f"s-{i}",
                kind="user" if i % 2 == 0 else "assistant",
                text="The conversation the panels and the card compete with.",
            )
            for i in range(6)
        ],
        version=5,
    )
    projection.subagents = _roster_projection().subagents
    projection.todos = [
        TodoPhase(
            name="Remediation",
            items=[
                TodoItem(text=f"Land finding {i:02d} and re-measure it", status="pending")
                for i in range(1, 13)
            ],
        )
    ]
    if approval:
        projection.pending = PendingRequest(
            request_id="req-stacked-approval",
            kind="approval",
            title="bash",
            detail=APPROVAL_DETAIL,
        )
    else:
        projection.pending = PendingRequest(
            request_id="req-stacked-ask",
            kind="ask",
            title=LONG_QUESTION,
            options=[
                AskOptionWire(label=f"option-{i + 1:02d}", description=text)
                for i, text in enumerate(OPTION_DESCRIPTIONS)
            ],
        )
    projection.pending_count = 1
    return projection


def _failures_projection() -> SessionProjection:
    """A fan-out with failures (U5).

    Collapsing the roster by default is right, but it took the status glyphs
    off screen with it — and `1/22 running` is exactly what a healthy session
    shows, so three failed agents were indistinguishable from none.
    """
    projection = _roster_projection()
    projection.session_id = "failures"
    projection.pid = 900007
    projection.conversation_name = "Failing fan-out"
    for rowobj in projection.subagents[1:4]:
        rowobj.status = "failed"
        rowobj.error_text = "surface probe exited 1"
    return projection


def _failures_pending_projection() -> SessionProjection:
    """Failed agents AND a pending request in one column (D4).

    The state where a failed fan-out matters most is the one that dimmed it:
    the panels are held shut precisely while the user is being asked for a
    decision, so the `· 3 failed` count is dimmed at the moment it is most
    worth reading. Needs both halves at once — `failures` has no pending
    request and `stacked` has no failures — so neither could show it.
    """
    projection = _failures_projection()
    projection.session_id = "failures-pending"
    projection.pid = 900009
    projection.conversation_name = "Failing fan-out, decision waiting"
    projection.todos = [
        TodoPhase(
            name="Remediation",
            items=[
                TodoItem(text=f"Land finding {i:02d} and re-measure it", status="pending")
                for i in range(1, 13)
            ],
        )
    ]
    projection.pending = PendingRequest(
        request_id="req-failures-pending",
        kind="approval",
        title="bash",
        detail=APPROVAL_DETAIL,
    )
    projection.pending_count = 1
    return projection


def _stale_projection() -> SessionProjection:
    """An ask whose answers the daemon refuses as moved-on (U3).

    `control_port=1` means every answer fails, which is what this scenario
    wants: the point is WHERE the resulting error line renders, and before the
    fix it was appended as the scroller's last child — i.e. ~26px below
    wherever the user was standing, which is the foot of the option list,
    because that is where they had just tapped.
    """
    projection = _ask_projection()
    projection.session_id = "stale"
    projection.pid = 900008
    projection.conversation_name = "Stale tap"
    projection.pending = PendingRequest(
        request_id="req-stale",
        kind="ask",
        title=LONG_QUESTION,
        options=[
            AskOptionWire(label=f"option-{i + 1:02d}", description=text)
            for i, text in enumerate(OPTION_DESCRIPTIONS)
        ],
    )
    projection.pending_count = 1
    return projection


def _question(
    qid: str,
    text: str,
    *,
    options: list[tuple[str, str]] | None = None,
    multi: bool = False,
    secret: bool = False,
    persist: bool = False,
) -> dict[str, Any]:
    """One question in the shape ``asks/queue._question_shape`` stores, which is
    the shape the wire carries verbatim (design §4: the FULL question rides, so
    a surface draws a picker without re-deriving the ask)."""
    return {
        "id": qid,
        "question": text,
        "options": [
            {"label": label, "description": description} for label, description in (options or [])
        ],
        "multi": multi,
        "recommended": 0 if options else None,
        "secret": secret,
        "persist": persist,
    }


def _ask(
    ask_id: str,
    *,
    status: str = "open",
    created_s: int,
    expires_in_s: int,
    timeout_s: int = 900,
    questions: list[dict[str, Any]],
    urgent: bool = False,
    delivered: bool = False,
    answers: dict[str, list[str]] | None = None,
    answered_by: str = "",
) -> PendingAskWire:
    """One queued ask, in epoch MILLISECONDS (``created_at``/``expires_at`` are
    ``now_ms()`` on the wire, unlike the seconds every other timestamp uses)."""
    now = int(time.time() * 1000)
    return PendingAskWire(
        ask_id=ask_id,
        created_at=now + created_s * 1000,
        expires_at=now + expires_in_s * 1000,
        timeout_s=timeout_s,
        urgent=urgent,
        status=status,
        delivered=delivered,
        questions=questions,
        answers=answers,
        answered_by={"surface": answered_by} if answered_by else None,
    )


def _queued_ask_projection() -> SessionProjection:
    """The phone's queued-ask surface with every state that matters on screen.

    Three LIVE asks and two settled ones, in the wire's own order (open first,
    newest first), because the states are what a reader has to be able to tell
    apart:

    * ``qa-head`` — the OLDEST open ask (two questions: a picker with a
      recommended option, and a multi-select), so the bar's head rule and the
      card's multi-question form are both visible;
    * ``qa-second`` — a newer open ask, so the count and the list have more than
      one row;
    * ``qa-deadline`` — TIMED OUT and still answerable, which is the state the
      fixed copy exists for ("the agent moved on; you can still answer");
    * ``qa-settled`` / ``qa-declined`` — terminal, so the card's receipt shape
      and the honest "no reply was sent" line are in the same frame.

    The transcript also carries the two SETTLING rows (an ``ask_response`` row
    for a late answer and an ``ask_timeout`` row), because the response card is
    the other half of this surface and a capture that only ever shows the queue
    would not look at it.
    """
    now = int(time.time() * 1000)
    asks = [
        _ask(
            "qa-second",
            created_s=5,
            expires_in_s=600,
            questions=[
                _question(
                    "s1",
                    "Answer the second ask first?",
                    options=[
                        ("yes", "answers the newest row first"),
                        ("no", "leaves it for the head ask"),
                    ],
                )
            ],
        ),
        _ask(
            "qa-head",
            created_s=-120,
            expires_in_s=780,
            questions=[
                _question(
                    "h1",
                    "Which sequencing should the rollout use?",
                    options=[
                        ("layout first", "the smallest diff reaches the wheel"),
                        ("roster first", "the change the operator reported"),
                        ("both together", "one patch, one contract to read"),
                    ],
                ),
                _question(
                    "h2",
                    "Which surfaces must the verification cover?",
                    options=[
                        ("phone", "the bundle that ships in the wheel"),
                        ("terminal", "the viewer reading the same fold"),
                        ("desktop", "the third surface on the same wire"),
                    ],
                    multi=True,
                ),
            ],
        ),
        _ask(
            "qa-deadline",
            created_s=-900,
            expires_in_s=-240,
            status="timed_out",
            questions=[
                _question(
                    "d1",
                    "The deadline passed — does the answer still hold?",
                    options=[
                        ("it holds", "the agent will be told late"),
                        ("changed", "say what changed"),
                    ],
                )
            ],
        ),
        _ask(
            "qa-settled",
            created_s=-1800,
            expires_in_s=600,
            status="answered",
            delivered=True,
            questions=[
                _question(
                    "t1",
                    "Ship the phone surface behind the flag?",
                    options=[("yes", "dark until the flip"), ("no", "hold the surface too")],
                )
            ],
            answers={"t1": ["yes"]},
            answered_by="tui",
        ),
        _ask(
            "qa-declined",
            created_s=-2400,
            expires_in_s=600,
            status="declined",
            questions=[_question("x1", "Take the long route through the relay?")],
            answered_by="desktop",
        ),
    ]
    projection = SessionProjection(
        session_id="asks",
        pid=900010,
        kind="tui",
        conversation_name="Queued asks",
        streaming=False,
        transcript=[
            TranscriptEntry(
                id="q-1",
                kind="user",
                text="Ask me before you land the phone surface.",
            ),
            TranscriptEntry(
                id="q-2",
                kind="assistant",
                text=(
                    "Queued three questions; continuing with the parts that do "
                    "not depend on them."
                ),
            ),
            TranscriptEntry(
                id="q-3",
                kind="ask_timeout",
                text=(
                    "Timed out after 4m — the agent moved on; you can still answer "
                    "(ask qa-deadline)"
                ),
                details={
                    "ask_id": "qa-deadline",
                    "status": "timed_out",
                    "waited_s": 240,
                    "urgent": False,
                    "severity": "warning",
                    "text": (
                        "[Ask timed out] No reply to ask qa-deadline arrived within 4m "
                        "(asked 15m ago). The deadline passed — does the answer still "
                        "hold?\nProceed without it: use your recommended option or best "
                        "judgment and state the assumption in your report. The ask stays "
                        "open for the user; if they answer later you will be told."
                    ),
                },
            ),
            TranscriptEntry(
                id="q-4",
                kind="ask_response",
                text="Answered late — the agent was told (ask qa-settled)",
                details={
                    "ask_id": "qa-settled",
                    "status": "late",
                    "severity": "warning",
                    "questions": [
                        _question(
                            "t1",
                            "Ship the phone surface behind the flag?",
                            options=[
                                ("yes", "dark until the flip"),
                                ("no", "hold the surface too"),
                            ],
                        )
                    ],
                    "answers": {"t1": ["yes"]},
                    "at": now - 30_000,
                    "text": (
                        "You already proceeded when this ask timed out; reconsider "
                        "only if the answer changes your work.\n\n"
                        "The user answered yes."
                    ),
                },
            ),
        ],
        version=11,
    )
    projection.asks = asks
    projection.asks_open = 2
    # THE LEGACY MIRROR IS DELIBERATELY ALSO SET. A real runtime publishes it for
    # one release so old clients can still answer, and the phone must NOT draw it
    # a second time (design §4, client rule N3) — so the state that would show the
    # defect twice is the state the fixture serves.
    projection.pending = PendingRequest(
        request_id="qa-head.0",
        kind="ask",
        title="Which sequencing should the rollout use?",
        options=[
            AskOptionWire(label="layout first", description="the smallest diff reaches the wheel"),
            AskOptionWire(label="roster first", description="the change the operator reported"),
            AskOptionWire(label="both together", description="one patch, one contract to read"),
        ],
    )
    projection.pending_count = 0
    return projection


def _settled_ask_projection() -> SessionProjection:
    """The zero-outstanding state with a HISTORY: a conversation whose asks have
    all settled. The minimized bar must be absent here (zero asks), while the
    transcript still carries the settling rows a reader scrolls back to."""
    projection = SessionProjection(
        session_id="asks-settled",
        pid=900011,
        kind="tui",
        conversation_name="Asks settled",
        streaming=False,
        transcript=[
            TranscriptEntry(id="s-1", kind="user", text="Answer when you can."),
            TranscriptEntry(
                id="s-2",
                kind="ask_response",
                text="Answered — delivering (ask sa-answered)",
                details={
                    "ask_id": "sa-answered",
                    "status": "answered",
                    "severity": "info",
                    "questions": [
                        _question(
                            "a1",
                            "Ship behind the flag?",
                            options=[
                                ("yes", "dark until the flip"),
                                ("no", "hold the surface too"),
                            ],
                        )
                    ],
                    "answers": {"a1": ["yes"]},
                    "at": int(time.time() * 1000) - 60_000,
                    "text": "The user answered yes.",
                },
            ),
            TranscriptEntry(
                id="s-3",
                kind="ask_response",
                text="Ask sa-declined declined — the agent was told",
                details={
                    "ask_id": "sa-declined",
                    "status": "declined",
                    "severity": "info",
                    "questions": [_question("d1", "Take the long route through the relay?")],
                    "answers": {},
                    "at": int(time.time() * 1000) - 30_000,
                    "text": "The user declined; decide yourself.",
                },
            ),
        ],
        version=2,
    )
    # Presence with an EMPTY list is not expressible on this wire (§4's A2
    # addendum: absence is the empty case), so a session with nothing waiting
    # publishes NO asks field at all — which is also what an old runtime sends,
    # and the reason the bar hides on both.
    projection.asks = None
    projection.asks_open = None
    return projection


class QueuedAskHarness:
    """A REAL runtime handle serving the queued-ask surface.

    WHY A LIVE RUNTIME AND NOT ANOTHER SYNTHETIC PROJECTION. The other fixture
    sessions are static pictures of a state, which is all an overflow capture
    needs. This one has to be ANSWERABLE: the phone's answer flow (fill the
    form, send, watch the card become a receipt) only means anything if the
    frame actually reaches a runtime and the runtime's own fold comes back —
    a synthetic entry would refuse every answer, and the refusal would be the
    rig's, not the route's.

    So the answers are recorded here, the projection is re-derived from them
    (the same ``PendingAsk`` the wire carries, with its status flipped), and the
    runtime pushes the new projection to the relay — which is also how the
    capture sees the phone update with no reload: the live-update half of the
    requirement, through the same path a real session uses.
    """

    def __init__(self, projection: SessionProjection) -> None:
        self.projection = projection
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.answers: dict[str, dict[str, list[str]]] = {}
        self.declined: list[str] = []
        self.dismissed: list[str] = []
        #: Called after every settlement, so the runtime repaints its attached
        #: readers. A REAL handle has its own event stream driving that push
        #: (``subscribe_events``); this one has no events, so the fixture wires
        #: the runtime's own coalesced repaint instead. Without it the answer
        #: lands on the wire and the phone keeps its pre-answer frame — the
        #: difference between a capture of the flow and a capture of a stall.
        self.on_change: Any = None

    # -- the SessionHandle surface the runtime probes ----------------------
    @property
    def session_projection_seed(self) -> SessionProjection:
        return self.projection

    def subscribe(self, on_projection: Any) -> Any:
        return lambda: None

    async def refresh(self) -> None:
        return None

    # -- the ORDINARY SessionHandle ops, present so the class satisfies the
    # -- protocol ``RuntimeServer`` is typed against (pyright checks the
    # -- argument structurally, and 11 missing methods is a type error even
    # -- though nothing in this fixture sends one). They RECORD rather than
    # -- raise, the same shape ``test_daemon.FakeHandle`` uses: a fixture that
    # -- dies because a future runtime probes an op it does not use would be
    # -- trading a type error for a worse one.
    def _record(self, name: str, *args: Any, **kwargs: Any) -> str:
        self.calls.append((name, args, kwargs))
        return f"{name} ok"

    async def prompt(self, text: Any, images: Any = None, command_id: Any = None) -> str:
        return self._record("prompt", text)

    async def steer(self, text: Any, images: Any = None, command_id: Any = None) -> str:
        return self._record("steer", text)

    async def abort(self) -> str:
        return self._record("abort")

    async def set_model(self, provider: Any, model_id: Any) -> str:
        return self._record("set_model", provider, model_id)

    async def set_effort(self, effort: Any) -> str:
        return self._record("set_effort", effort)

    async def slash(self, command: Any, args: Any) -> str:
        return self._record("slash", command, args)

    async def new_conversation(self) -> str:
        return self._record("new_conversation")

    async def resume_session(self, session_id: Any) -> str:
        return self._record("resume_session", session_id)

    async def approval_answer(self, request_id: Any, approved: Any, remember: Any) -> str:
        return self._record("approval_answer", request_id, approved, remember)

    async def ask_answer(
        self, request_id: Any, value: Any, question_index: Any = None
    ) -> str:
        return self._record("ask_answer", request_id, value)

    #: The ask whose answer this runtime REFUSES with the queue's own sentence,
    #: standing in for the single-winner rule (design §2.4): another surface
    #: settled it first, the phone's screen is one repaint behind, and the copy
    #: the loser reads is the queue's rather than the client's. It is the one
    #: refusal a capture can reach deterministically.
    already_answered_elsewhere = "qa-deadline"

    async def ask_respond(self, ask_id: str, answers: dict[str, list[str]], by: str = "") -> str:
        if str(ask_id) == self.already_answered_elsewhere:
            raise ValueError("already answered by desktop.")
        self.answers[str(ask_id)] = {
            str(key): [str(item) for item in (values or [])] for key, values in answers.items()
        }
        self._settle(str(ask_id), "answered", self.answers[str(ask_id)])
        return "answered"

    async def ask_decline(self, ask_id: str, by: str = "") -> str:
        self.declined.append(str(ask_id))
        self._settle(str(ask_id), "declined", None)
        return "declined"

    async def ask_dismiss(self, ask_id: str, by: str = "") -> str:
        self.dismissed.append(str(ask_id))
        return "dismissed"

    def _publish_index(self) -> None:
        """Write the derived index the AGGREGATE reads (design §4).

        What a real session's reconcile tick does, and for the same reason: an
        ask outlives the runtime that asked it, so ``GET /api/asks`` answers from
        this file rather than from a live fold. Without it the asks sheet would
        keep showing the pre-answer rows while the session view showed the
        receipt — two surfaces disagreeing about one ask.
        """
        from local_operator.asks.store import write_entry
        from local_operator.paths import config_dir

        root = config_dir()
        session_id = self.projection.session_id
        (root / "sessions" / session_id).mkdir(parents=True, exist_ok=True)
        write_entry(
            root,
            session_id,
            cwd="/synthetic",
            asks=[row.to_json() for row in (self.projection.asks or [])],
        )

    def _settle(self, ask_id: str, status: str, answers: dict[str, list[str]] | None) -> None:
        """Flip the row the way the queue's fold does, so the pushed projection
        is the state a real runtime would publish (a settled row, and the open
        count falling with it)."""
        for row in self.projection.asks or []:
            if row.ask_id != ask_id:
                continue
            row.status = status
            row.delivered = True
            row.answers = answers
            row.answered_by = {"surface": "phone"}
        open_rows = [row for row in (self.projection.asks or []) if row.status == "open"]
        self.projection.asks_open = len(open_rows)
        self._publish_index()
        # Mirror the reset the fold's own publisher does: with nothing left to
        # mirror, the legacy card goes away rather than showing a settled ask.
        self.projection.pending = None if not open_rows else self.projection.pending
        self.projection.version += 1
        if self.on_change is not None:
            self.on_change()


def seed_ask_index(projections: list[SessionProjection]) -> None:
    """Publish the asks into the daemon's INDEX, the way a live runtime does.

    The aggregate (``GET /api/asks``) is index-backed on purpose (design §4):
    an ask outlives the runtime that queued it, so the route reads a derived
    file rather than dialling anything. A fixture that only set the projection
    would therefore leave the asks sheet empty while the session view showed a
    full queue — the exact skew the index exists to avoid.

    The session DIRECTORIES are created too, and that is load-bearing rather
    than cosmetic: ``store.entry_is_stale`` sweeps an entry whose session is
    gone (nothing could ever answer it), so an index seeded beside no
    conversation is swept on first read.
    """
    from local_operator.asks.store import write_entry
    from local_operator.paths import config_dir

    root = config_dir()
    sessions = root / "sessions"
    for projection in projections:
        (sessions / projection.session_id).mkdir(parents=True, exist_ok=True)
        rows = [row.to_json() for row in (projection.asks or [])]
        write_entry(root, projection.session_id, cwd="/synthetic", asks=rows)


async def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4187
    password = required_password(sys.argv[2:])
    daemon = MobileDaemon(port=port, password=password, dial_registrants=False)
    projections = [
        _roster_projection(),
        _ask_projection(),
        _approval_projection(),
        _free_text_projection(),
        _stacked_projection("stacked", approval=False),
        _stacked_projection("stacked-approval", approval=True),
        _failures_projection(),
        _failures_pending_projection(),
        _stale_projection(),
        # THE ZERO-ASKS STATE of the same surface (the bar must be absent, and
        # the settling rows are still in the transcript).
        _settled_ask_projection(),
    ]
    # The ANSWERABLE queued-ask session is not in that list: it is served by a
    # real runtime below, so its projection arrives over the relay's own dial
    # rather than being injected into the daemon's table.
    queued = _queued_ask_projection()
    for projection in projections:
        record = SessionRecord(
            pid=projection.pid,
            kind="tui",
            session_id=projection.session_id,
            conversation_name=projection.conversation_name,
            cwd="/synthetic",
            model_label="fixture",
            control_port=1,
            control_key="fixture",
        )
        entry = SessionEntry(record)
        entry.projection = projection
        daemon.session_projections[projection.session_id] = projection
        daemon.table.entries[record.pid] = entry
    # The aggregate route reads the derived INDEX, not the projections (an ask
    # outlives the runtime that asked it), so the fixture has to seed it or the
    # asks sheet would open empty beside a session view showing a full queue.
    # ``LOP_ASK_FIXTURE_EMPTY_INDEX=1`` serves the OTHER side of the aggregate:
    # the session directories are still created, but no ask is published into the
    # index, so ``GET /api/asks`` answers empty while the live runtime still
    # publishes its own asks to the session view. That skew is real rather than
    # invented — the runtime writes the index on its reconcile tick, so an ask
    # can be on the projection a moment before the aggregate carries it — and it
    # is the only way to photograph the sheet's empty state at all, because at
    # zero asks a session offers neither the bar nor the header entry to open it.
    seed_ask_index([] if os.environ.get("LOP_ASK_FIXTURE_EMPTY_INDEX") else [*projections, queued])
    # THE ANSWERABLE SESSION IS A REAL RUNTIME, not another still picture: the
    # phone's answer flow is only meaningful if the frame reaches a runtime and
    # the runtime's own fold comes back (see ``QueuedAskHarness``). The relay
    # dials it exactly as it dials a terminal session — its record is the real
    # registry one, and the projection the phone renders is pushed over that
    # socket rather than injected into the daemon's table.
    harness = QueuedAskHarness(queued)
    registrant = RuntimeServer(harness, kind="tui")
    registrant.start()
    harness.on_change = registrant._schedule_push
    dial = None
    from local_operator.session.runtime import registry

    deadline = asyncio.get_running_loop().time() + 10
    record = None
    while asyncio.get_running_loop().time() < deadline:
        found = [pair for pair in registry.scan() if pair[1] == "live"]
        if found:
            record = found[0][0]
            break
        await asyncio.sleep(0.05)
    if record is None:
        raise SystemExit("the fixture's runtime never registered")
    entry = SessionEntry(record)
    daemon.table.entries[record.pid] = entry
    dial = asyncio.ensure_future(_dial(daemon, entry))
    app = build_app(daemon)
    print(f"Fixture mobile: http://127.0.0.1:{port}", flush=True)
    try:
        await uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        ).serve()
    finally:
        if dial is not None:
            dial.cancel()
        registrant.close()


if __name__ == "__main__":
    asyncio.run(main())
