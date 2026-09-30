"""Serve the REAL phone bundle with the sessions-tool visuals, for capture.

Run:  PYTHONPATH=. .venv/bin/python scripts/sessions_tool_mobile_fixture.py PORT PASSWORD
The CAPTURE script passes the password, so this file holds no literal.

FOUR ROWS, and each answers one question the two changes are about:

* ``Release cutter`` — a WORKSTREAM opened by a named agent (``manager``), the
  row the phone-agent-opened chip exists for. The marker is written through the
  real ``resume.mark_session_origin`` writer with the same ``opened_by`` object
  the stamp writes, so the row is the shape the product produces, not a
  hand-rolled JSON a reader would have to translate.
* ``Anonymous workstream`` — opener present but every member null (a top-level
  requester whose role/label could not be read). Presence is the fact the chip
  is drawn from, so this row must carry the mark all the same.
* ``Operator session`` — an ordinary conversation with no marker at all: the
  control that must NOT carry the mark.
* ``Sessions tool run`` — a live entry whose projection carries ``sessions``
  tool rows. Summaries come from ``mobile.projection._summarize_args`` — the
  SAME function the fold uses — computed here at seed time, so this fixture run
  on the base revision and on the change differ in exactly the strings the
  change edits.
"""

from __future__ import annotations

import asyncio
import sys

import uvicorn

import scripts.probe_isolation  # noqa: F401  -- must be the first local import
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
from local_operator.mobile.types import SessionProjection, TranscriptEntry
from local_operator.session.runtime.types import SessionRecord

#: (session_id, title, opener-or-None). Ids are dir-shaped; the ids themselves
#: are opaque to everything this fixture exercises.
ROWS: list[tuple[str, str, dict[str, str | None] | None]] = [
    (
        "aaaa00000001",
        "Release cutter",
        {"agent": "manager", "label": "sessions-tool-0b39", "session": "e84228882245"},
    ),
    (
        "aaaa00000002",
        "Anonymous workstream",
        {"agent": None, "label": None, "session": None},
    ),
    ("aaaa00000003", "Operator session", None),
]

#: The ``sessions`` tool calls the session view draws — every op whose summary
#: this change gives a discriminator, in the argument shapes the tool's schema
#: carries (``peek`` painted ahead of its merge, like the TUI shot).
TOOL_CALLS: list[tuple[str, str, dict[str, object]]] = [
    (
        "c1",
        "spawn",
        {
            "op": "spawn",
            "name": "night-audit",
            "prompt": "audit the release window",
            "visibility": "workstream",
        },
    ),
    (
        "c2",
        "spawn",
        {"op": "spawn", "prompt": "fix the flaky shard in isolation", "visibility": "ephemeral"},
    ),
    ("c3", "stop", {"op": "stop", "target": "release-crew"}),
    ("c4", "peek", {"op": "peek", "target": "release-crew", "steps": 12}),
    ("c5", "list", {"op": "list", "include_stored": True, "query": "flaky shard"}),
]


def _seed_durable(session_id: str, title: str, opener: dict[str, str | None] | None) -> None:
    """One listable conversation, seeded through the real writers.

    ``transcript.jsonl`` is what makes a directory a conversation to the scan;
    the title sidecar and the marker go through ``resume``'s own writers so the
    fixture cannot drift from what a session actually looks like on disk.
    """
    from local_operator.paths import config_dir
    from local_operator.resume import (
        ORIGIN_AGENT_WORKSTREAM,
        mark_session_origin,
        write_session_title,
    )

    directory = config_dir() / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    write_session_title(directory, title, user_set=False, past_names=[])
    if opener is not None:
        mark_session_origin(directory, ORIGIN_AGENT_WORKSTREAM, opened_by=opener)


def _seed_tools_session(daemon: MobileDaemon) -> None:
    """A live session whose projection is the sessions tool ledger.

    The summaries are computed by the projection's OWN function — the fold calls
    exactly this with the call's (name, args) — so a run of this fixture on the
    base revision and on the change produces the before/after strings by
    construction rather than by a copy.
    """
    from local_operator.mobile.projection import _summarize_args

    session_id = "aaaa00000004"
    pid = 910001
    record = SessionRecord(
        pid=pid,
        kind="tui",
        session_id=session_id,
        conversation_name="Sessions tool run",
        cwd="/synthetic/worktree",
        model_label="fixture/model",
        control_port=1,
        control_key="fixture",
        detached=True,
        busy=True,
    )
    entries: list[TranscriptEntry] = [
        TranscriptEntry(
            id="m1",
            kind="assistant",
            text="Working through the sessions-tool steps below.",
        )
    ]
    for call_id, _op, args in TOOL_CALLS:
        entries.append(
            TranscriptEntry(
                id=f"m1:{call_id}",
                kind="tool",
                tool_call_id=call_id,
                tool_name="sessions",
                tool_state="done",
                summary=_summarize_args("sessions", args),
                details={"args": args},
            )
        )
    entry = SessionEntry(record)
    entry.projection = SessionProjection(
        session_id=session_id,
        pid=pid,
        kind="tui",
        conversation_name="Sessions tool run",
        cwd="/synthetic/worktree",
        model_label="fixture/model",
        transcript=entries,
    )
    daemon.table.entries[pid] = entry


#: The variable name a caller may use instead of the second argument (a NAME,
#: never a value; shared spelling with the capture module on purpose).
FIXTURE_PASSWORD_ENV = "LOP_MOBILE_FIXTURE_PASSWORD"


def required_password(argv_rest: list[str]) -> str:
    """The per-run password, or a refusal that names how to supply one."""
    value = argv_rest[0] if argv_rest else ""
    if not value:
        raise SystemExit(
            "this fixture needs a per-run password: pass it as the second argument. "
            "Generate one with python -c 'import secrets;print(secrets.token_urlsafe(16))'. "
            "It is never defaulted and never printed by this script."
        )
    return value


async def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4189
    password = required_password(sys.argv[2:])
    daemon = MobileDaemon(port=port, password=password, dial_registrants=False)
    for session_id, title, opener in ROWS:
        _seed_durable(session_id, title, opener)
    _seed_tools_session(daemon)
    app = build_app(daemon)
    print(f"Fixture phone list on http://127.0.0.1:{port}", flush=True)
    await uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())
