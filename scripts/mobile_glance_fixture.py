"""Serve synthetic sessions that carry the spend/context block, for capture.

Run:  PYTHONPATH=. .venv/bin/python scripts/mobile_glance_fixture.py [PORT] <password>

Login at http://127.0.0.1:<port>. The password is NOT fixed and never printed:
pass it as the second argument (or set ``LOP_MOBILE_FIXTURE_PASSWORD``) — see
the sibling fixtures' note on why a literal here would be a reusable credential.

Three sessions, one per spelling state the session-state row ships in phase 1
of the mobile parity program, plus the estimate marker's own row — four since
the design round ruled the marker to be the desktop's WORD:

* ``glance``         — the typical reading: an exact $1.25 spent and a measured
  ``6.2%/200k`` context. The row's two cells, both unmarked.
* ``glance-floor``   — a FLOOR spend (``≥$2.00``: part of the conversation ran
  before this process was tracking, or a child is not fully known) beside an
  ESTIMATE context (``6.2%/200k estimate``: the app counted, the provider has
  not reported yet).
* ``glance-unknown`` — money we cannot state: tokens were billed at a price
  nobody could resolve, so the cell is ``$—``, and a token count with no known
  window, so the context cell is ``12.4k/—`` rather than an invented percentage.
* ``glance-estimate`` — the estimate marker alone on an otherwise exact row
  (``$0.0042`` + ``6.2%/200k estimate``), so the marker is judged on its own
  rather than only beside a floor's ``≥``.
* ``glance-label``    — the warm rung the band calls ``label`` (``60.0%/200k``,
  past 0.55 of the window): must carry the weighted reading as well as the
  accent, the colour-vision carrier the first design round measured absent.
* ``glance-danger``   — the top rung (``99.9%/1M``) AND the layout stress at
  once: the estimate word on the widest reading beside the longest ledger
  figure (``$123456.78``), so "the word costs width" is measured on the
  worst state rather than the friendliest.
* ``glance-spend``    — spend alone (no context reading): the single-cell state
  whose edge the order swap moves, so the "mirror the same way" half of the
  ruling is captured rather than argued.
* ``glance-context``  — context alone (nothing billed yet): the other single.

No runtime scanner and no registrant sockets (``dial_registrants=False``), so
this never touches the operator's live daemon or their sessions. HOME and
LOCAL_OPERATOR_CONFIG_DIR are re-homed by ``scripts.probe_isolation`` on import.
"""

from __future__ import annotations

import asyncio
import os
import sys

import uvicorn

import scripts.probe_isolation  # noqa: F401  -- must be the first local import
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
from local_operator.mobile.types import (
    SessionProjection,
    SessionRecord,
    TranscriptEntry,
)

#: The variable NAME (never a value) a caller may use instead of the second
#: argument — the same contract the sibling fixtures document.
FIXTURE_PASSWORD_ENV = "LOP_MOBILE_FIXTURE_PASSWORD"


def required_password(args: list[str]) -> str:
    """The password this run serves with, or a refusal naming the contract."""
    value = args[0] if args else os.environ.get(FIXTURE_PASSWORD_ENV, "")
    if not value:
        raise SystemExit(
            "this fixture needs a per-run password: pass it as the second argument, or "
            f"set {FIXTURE_PASSWORD_ENV}. Generate one with "
            "python -c 'import secrets;print(secrets.token_urlsafe(16))' and export it. "
            "It is never defaulted and never printed by this script."
        )
    return value


def _conversation(lead: str, followups: list[str]) -> list[TranscriptEntry]:
    """A short exchange, so the frames show the row above a REAL conversation."""
    entries = [TranscriptEntry(id="g-1", kind="user", text=lead)]
    for index, text in enumerate(followups, start=2):
        entries.append(TranscriptEntry(id=f"g-{index}", kind="assistant", text=text))
    return entries


def _typical_projection() -> SessionProjection:
    projection = SessionProjection(
        session_id="glance",
        pid=900101,
        kind="tui",
        conversation_name="Spend + context glance",
        streaming=False,
        transcript=_conversation(
            "How much has this session cost, and how full is the context?",
            [
                "The phone now answers both off the session's own ledger.",
                "No slash command needed to read either number.",
            ],
        ),
        version=4,
    )
    projection.cumulative_parent_cost = 1.25
    projection.child_costs = {}
    projection.subagent_cost = None
    projection.subagent_cost_knowledge = None
    projection.cost_knowledge = "exact"
    projection.context_tokens = 12_400
    projection.context_window = 200_000
    projection.context_is_estimate = False
    projection.usage = {"input_tokens": 12_400, "output_tokens": 320}
    return projection


def _floor_projection() -> SessionProjection:
    projection = SessionProjection(
        session_id="glance-floor",
        pid=900102,
        kind="tui",
        conversation_name="Floor and estimate",
        streaming=False,
        transcript=_conversation(
            "Resume this session and keep an eye on the window.",
            ["Restored spend is a lower bound until a turn settles in this process."],
        ),
        version=3,
    )
    projection.cumulative_parent_cost = 2.0
    projection.child_costs = {}
    projection.subagent_cost = None
    projection.subagent_cost_knowledge = None
    projection.cost_knowledge = "floor"
    projection.context_tokens = 12_400
    projection.context_window = 200_000
    projection.context_is_estimate = True
    projection.usage = {"input_tokens": 12_400, "output_tokens": 80}
    return projection


def _unknown_projection() -> SessionProjection:
    projection = SessionProjection(
        session_id="glance-unknown",
        pid=900103,
        kind="tui",
        conversation_name="Unpriceable reading",
        streaming=False,
        transcript=_conversation(
            "Run the turn against the provider with no published price.",
            ["Tokens were billed; the app cannot state the money."],
        ),
        version=2,
    )
    projection.cumulative_parent_cost = None
    projection.child_costs = {}
    projection.subagent_cost = None
    projection.subagent_cost_knowledge = None
    projection.cost_knowledge = "unknown"
    projection.context_tokens = 12_400
    projection.context_window = 0
    projection.context_is_estimate = False
    projection.usage = {"input_tokens": 9_000, "output_tokens": 100}
    return projection


def _estimate_projection() -> SessionProjection:
    projection = SessionProjection(
        session_id="glance-estimate",
        pid=900104,
        kind="tui",
        conversation_name="Estimate marker",
        streaming=False,
        transcript=_conversation(
            "The provider has not reported yet; the app counted this one.",
            ["An estimate is a word, not a glyph: `estimate`, dim, beside the reading."],
        ),
        version=5,
    )
    projection.cumulative_parent_cost = 0.0042
    projection.child_costs = {}
    projection.subagent_cost = None
    projection.subagent_cost_knowledge = None
    projection.cost_knowledge = "exact"
    projection.context_tokens = 12_400
    projection.context_window = 200_000
    projection.context_is_estimate = True
    projection.usage = {"input_tokens": 12_400, "output_tokens": 320}
    return projection


def _label_projection() -> SessionProjection:
    projection = SessionProjection(
        session_id="glance-label",
        pid=900105,
        kind="tui",
        conversation_name="Context pressure",
        streaming=False,
        transcript=_conversation(
            "How close is this window to compaction?",
            ["Past the label rung the reading also carries weight — hue alone cannot."],
        ),
        version=6,
    )
    projection.cumulative_parent_cost = 0.61
    projection.child_costs = {}
    projection.subagent_cost = None
    projection.subagent_cost_knowledge = None
    projection.cost_knowledge = "exact"
    projection.context_tokens = 120_000
    projection.context_window = 200_000
    projection.context_is_estimate = False
    projection.usage = {"input_tokens": 120_000, "output_tokens": 900}
    return projection


def _danger_projection() -> SessionProjection:
    projection = SessionProjection(
        session_id="glance-danger",
        pid=900106,
        kind="tui",
        conversation_name="Context alarm",
        streaming=False,
        transcript=_conversation(
            "Compaction is imminent and this ledger has run long.",
            ["Both the widest reading and the longest figure, with the marker on top."],
        ),
        version=7,
    )
    projection.cumulative_parent_cost = 123_456.78
    projection.child_costs = {}
    projection.subagent_cost = None
    projection.subagent_cost_knowledge = None
    projection.cost_knowledge = "exact"
    projection.context_tokens = 999_000
    projection.context_window = 1_000_000
    projection.context_is_estimate = True
    projection.usage = {"input_tokens": 999_000, "output_tokens": 1_200}
    return projection


def _spend_only_projection() -> SessionProjection:
    projection = SessionProjection(
        session_id="glance-spend",
        pid=900107,
        kind="tui",
        conversation_name="Spend only",
        streaming=False,
        transcript=_conversation(
            "Money billed, no context reading yet.",
            ["A lone spend hugs the right edge after the swap."],
        ),
        version=8,
    )
    projection.cumulative_parent_cost = 0.75
    projection.child_costs = {}
    projection.subagent_cost = None
    projection.subagent_cost_knowledge = None
    projection.cost_knowledge = "exact"
    projection.context_tokens = None
    projection.context_window = None
    projection.context_is_estimate = None
    projection.usage = {"input_tokens": 12_400, "output_tokens": 320}
    return projection


def _context_only_projection() -> SessionProjection:
    projection = SessionProjection(
        session_id="glance-context",
        pid=900108,
        kind="tui",
        conversation_name="Context only",
        streaming=False,
        transcript=_conversation(
            "A reading before anything has been billed.",
            ["A lone context hugs the left edge after the swap."],
        ),
        version=9,
    )
    projection.cumulative_parent_cost = None
    projection.child_costs = {}
    projection.subagent_cost = None
    projection.subagent_cost_knowledge = None
    projection.cost_knowledge = "unknown"
    projection.context_tokens = 12_400
    projection.context_window = 200_000
    projection.context_is_estimate = False
    projection.usage = {}
    return projection


async def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4188
    password = required_password(sys.argv[2:])
    daemon = MobileDaemon(port=port, password=password, dial_registrants=False)
    for projection in (
        _typical_projection(),
        _floor_projection(),
        _unknown_projection(),
        _estimate_projection(),
        _label_projection(),
        _danger_projection(),
        _spend_only_projection(),
        _context_only_projection(),
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
    print(f"Fixture mobile: http://127.0.0.1:{port}", flush=True)
    await uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())
