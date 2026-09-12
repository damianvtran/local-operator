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


async def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4187
    daemon = MobileDaemon(port=port, password=PASSWORD, dial_registrants=False)
    for projection in (
        _roster_projection(),
        _ask_projection(),
        _approval_projection(),
        _free_text_projection(),
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
