"""Serve the REAL mobile bundle with synthetic projections that provoke both
defects under review, for before/after capture at a phone viewport.

Run:  PYTHONPATH=. .venv/bin/python scripts/mobile_overflow_fixture.py [PORT]
Login at http://127.0.0.1:<port> with password `overflow-demo`.

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

No runtime scanner and no registrant sockets (``dial_registrants=False``), so
this never touches the operator's live daemon or their sessions. HOME and
LOCAL_OPERATOR_CONFIG_DIR are re-homed by ``scripts.probe_isolation`` on import.
"""

from __future__ import annotations

import asyncio
import sys

import uvicorn

import scripts.probe_isolation  # noqa: F401  -- must be the first local import
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
from local_operator.mobile.types import (
    AskOptionWire,
    PendingRequest,
    SessionProjection,
    SessionRecord,
    SubagentRow,
    TodoItem,
    TodoPhase,
    TranscriptEntry,
)

PASSWORD = "overflow-demo"

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


async def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4187
    daemon = MobileDaemon(port=port, password=PASSWORD, dial_registrants=False)
    for projection in (
        _roster_projection(),
        _ask_projection(),
        _approval_projection(),
        _free_text_projection(),
        _stacked_projection("stacked", approval=False),
        _stacked_projection("stacked-approval", approval=True),
        _failures_projection(),
        _failures_pending_projection(),
        _stale_projection(),
    ):
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
    app = build_app(daemon)
    print(f"Fixture mobile: http://127.0.0.1:{port} password {PASSWORD}", flush=True)
    await uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())
