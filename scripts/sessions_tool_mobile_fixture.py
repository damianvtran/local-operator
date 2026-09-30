"""Serve the REAL phone bundle with the sessions-tool visuals, for capture.

Run:  PYTHONPATH=. .venv/bin/python scripts/sessions_tool_mobile_fixture.py PORT PASSWORD
The CAPTURE script passes the password, so this file holds no literal.

FIVE ROWS, and each answers one question the two changes are about:

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
* ``Release crew reshuffle`` — THE STACKED WORST CASE (design round 1, mobile
  D2): agent-opened AND pinned AND unread AND delegating with todos, so the
  chip sits directly beside ``★``, ``new``, ``N subagents`` and ``N todo`` in
  one row. Every cluster member is seeded through the product's own writer
  (pin store, attention receipt, origin marker, projection todos), so the
  frame proves the words parse side by side, not merely that they can exist
  in isolation.
* ``Sessions tool run`` — a live entry whose projection carries ``sessions``
  tool rows. Summaries come from ``mobile.projection._summarize_args`` — the
  SAME function the fold uses — computed here at seed time, so this fixture run
  on the base revision and on the change differ in exactly the strings the
  change edits.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

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
    """The per-run password, or a refusal that names how to supply one.

    The env spelling mirrors both sibling fixtures (``mobile_glance_fixture``,
    ``mobile_delegating_fixture``): a caller may pass it as the second argument
    OR export it under ``FIXTURE_PASSWORD_ENV`` — the constant above documents a
    path this function has to actually implement, or the documented path is a
    refusal (round-1 review, R3).
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


def _seed_stacked_neighbour(daemon: MobileDaemon) -> None:
    """The one row carrying the WHOLE right cluster (design round 1, D2).

    Every member comes from the product's own path: the origin marker +
    listing via ``_seed_durable``, the pin through ``SessionTable.set_pins``
    (the same mutation the pin route performs), the unread completion through
    ``AttentionStore.publish`` (what a runtime's receipt goes through), and
    the subagent counts / todos as fields an ordinary live record and
    projection carry. The frame's question is whether ``agent`` parses beside
    ``★`` / ``new`` / ``2 subagents`` / ``2 todo`` at 360 — a hand-built row
    would not prove it of a row the product can produce.
    """
    from local_operator.mobile.types import TodoItem, TodoPhase
    from local_operator.session.attention import AttentionStore

    session_id = "aaaa00000005"
    _seed_durable(
        session_id,
        "Release crew reshuffle",
        {"agent": "manager", "label": "sessions-tool-0b39", "session": "e84228882245"},
    )
    daemon.table.set_pins(session_id, True)
    AttentionStore().publish(
        f"session/{session_id}", str(uuid.uuid4()), "reruns finished", "complete"
    )
    pid = 910002
    record = SessionRecord(
        pid=pid,
        kind="tui",
        session_id=session_id,
        conversation_name="Release crew reshuffle",
        cwd="/synthetic/worktree",
        model_label="fixture/model",
        control_port=1,
        control_key="fixture",
        detached=True,
        busy=False,
        # Two children of its own: the `N subagents` chip's half of the stack.
        subagents_running=2,
    )
    entry = SessionEntry(record)
    entry.projection = SessionProjection(
        session_id=session_id,
        pid=pid,
        kind="tui",
        conversation_name="Release crew reshuffle",
        cwd="/synthetic/worktree",
        model_label="fixture/model",
        # Two open (pending + blocked) of three: the `N todo` chip's half.
        todos=[
            TodoPhase(
                name="Release",
                items=[
                    TodoItem(text="rerun the flaky shard", status="pending"),
                    TodoItem(
                        text="fold the docs pass", status="blocked", reason="waiting on review"
                    ),
                    TodoItem(text="push the bump", status="completed"),
                ],
            )
        ],
    )
    daemon.table.entries[pid] = entry


async def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4189
    password = required_password(sys.argv[2:])
    daemon = MobileDaemon(port=port, password=password, dial_registrants=False)
    for session_id, title, opener in ROWS:
        _seed_durable(session_id, title, opener)
    _seed_tools_session(daemon)
    _seed_stacked_neighbour(daemon)
    app = build_app(daemon)
    print(f"Fixture phone list on http://127.0.0.1:{port}", flush=True)
    await uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())
