"""Serve the REAL phone bundle with Team Bridge tool rows, for capture.

Run:  PYTHONPATH=. .venv/bin/python scripts/bridge_tool_mobile_fixture.py PORT PASSWORD
The CAPTURE script (``bridge_tool_mobile_shot.py``) passes the password, so
this file holds no literal.

WHY IT EXISTS: the bridge row's outcome is the part a reader must not misread
(an ``inbox_only`` send is NOT a conversation post), and the generic row it
replaces led with ``bcv_`` ids. The frames have to show the product's own
facts, so every row here is built by the projection's OWN path:
``_summarize_args`` for the summary and ``_tool_row_details`` (which calls the
shared ``harness/rows.py::bridge_row``) for the expand payload. Run on a base
revision, the same fixture paints whatever that tree paints by construction -
the before frame - because a tree without ``bridge_row`` simply emits no facts.

ONE SESSION, five beats, each a question the change answers:

* a ``channel_post`` (the ordinary landed post: no chip, glyph + summary);
* a ``dm`` with a markdown body awaiting acceptance (expanded in the frame);
* a refused ``channel_post`` (``refused · not_member``);
* a mailbox ``send`` accepted into a peer inbox (``inbox_only``);
* a settled earlier turn mixing bridge calls with lop's own ``send``, which the
  condense model folds into ONE group.

Every id is synthetic (``bcv_FIXTURE…``) and nothing dials a real hub.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import uvicorn

import scripts.probe_isolation  # noqa: F401  -- must be the first local import
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
from local_operator.mobile.types import SessionProjection, TranscriptEntry
from local_operator.session.runtime.types import SessionRecord

SESSION_ID = "bbbb00000001"
TITLE = "Bridge tool run"

#: (call id, tool name, args, result details, state, output) - shapes taken
#: from ``team_bridge/tool.py``'s real results (peer_inbox receipt, DM content
#: line, typed refusal).
LIVE_CALLS: list[tuple[str, str, dict[str, object], dict[str, object], str, str]] = [
    (
        "b1",
        "bridge",
        {
            "op": "channel_post",
            "endpoint_id": "ep_FIXTURE01",
            "conversation_id": "bcv_FIXTURE0RELEASE",
            "conversation_label": "#release",
            "body": "0.93 is cut - notes in the thread.",
        },
        {"ok": True, "content": "posted to #release"},
        "done",
        "posted",
    ),
    (
        "b2",
        "bridge",
        {
            "op": "dm",
            "endpoint_id": "ep_FIXTURE01",
            "recipient": "acc_FIXTURE0PRIYA",
            "body": (
                "**Heads up** - the `0.93` cut lands *tonight*.\n\n"
                "- relay row ships with it\n- desktop row follows\n\n"
                "Ping me if the window moves: "
                "https://example.invalid/a/very/long/link/that/must/wrap/on/a/phone"
            ),
        },
        {
            "ok": True,
            "content": "direct message to Priya Shah in bcv_FIXTURE0DM",
            "recipient_state": "invited",
        },
        "done",
        "sent",
    ),
    (
        "b3",
        "bridge",
        {
            "op": "channel_post",
            "endpoint_id": "ep_FIXTURE01",
            "conversation_id": "bcv_FIXTURE0OPS",
            "body": "rolling back qa",
        },
        {"ok": False, "code": "not_member", "error": "not a member of bcv_FIXTURE0OPS"},
        "failed",
        "",
    ),
    (
        "b4",
        "bridge",
        {"op": "send", "recipient": "acc_FIXTURE0PRIYA", "body": "Can you take the cut?"},
        {"ok": True, "message_id": "bcmsg_FIXTURE01", "destination_type": "peer_inbox"},
        "done",
        "accepted",
    ),
]

#: The earlier, settled turn: bridge posts and reads beside lop's own ``send``,
#: so the condensed group's count line shows the split nouns.
GROUP_CALLS: list[tuple[str, str, dict[str, object], dict[str, object], str, str]] = [
    (
        "g1",
        "bridge",
        {"op": "directory"},
        {"ok": True, "content": '{"count": 4}'},
        "done",
        "",
    ),
    (
        "g2",
        "bridge",
        {"op": "channel_post", "conversation_id": "bcv_FIXTURE0RELEASE", "body": "starting"},
        {"ok": True},
        "done",
        "",
    ),
    ("g3", "send", {"target": "release-crew", "message": "window open"}, {}, "done", ""),
    (
        "g4",
        "bridge",
        {"op": "channel_history", "conversation_id": "bcv_FIXTURE0RELEASE"},
        {"ok": True},
        "done",
        "",
    ),
    (
        "g5",
        "bridge",
        {"op": "reply", "body": "on it"},
        {"ok": True, "destination_type": "peer_inbox"},
        "done",
        "",
    ),
]


def _settled_summary(name: str, args: dict, details: dict, state: str, start: Any) -> str:
    """The SETTLED summary, as the live/replay settle arms paint it.

    ``_summarize_args`` is the call-START paint (a running verb for bridge);
    the projection repaints a bridge row from its result at settle. A base
    tree has no settle repaint, so its start summary IS its settled one.
    """
    try:
        from local_operator.mobile.projection import _settled_bridge_summary
    except ImportError:
        return start(name, args)
    settled = _settled_bridge_summary(name, args, details, failed=state == "failed")
    return settled if settled is not None else start(name, args)


def _rows(
    prefix: str, calls: list[tuple[str, str, dict[str, object], dict[str, object], str, str]]
) -> list[TranscriptEntry]:
    from local_operator.mobile.projection import _summarize_args, _tool_row_details

    rows: list[TranscriptEntry] = []
    for call_id, name, args, details, state, output in calls:
        try:
            # The change's signature; a base tree has no ``tool_name`` keyword
            # and paints the generic row, which IS its before frame.
            payload = _tool_row_details(args, output, details, tool_name=name)
        except TypeError:
            payload = _tool_row_details(args, output, details)
        rows.append(
            TranscriptEntry(
                id=f"{prefix}:{call_id}",
                kind="tool",
                tool_call_id=call_id,
                tool_name=name,
                tool_state=state,  # type: ignore[arg-type]
                summary=_settled_summary(name, args, details, state, _summarize_args),
                elapsed_s=1.4,
                error=str(details.get("error") or "") if state == "failed" else "",
                details=payload,
                final=True,
            )
        )
    return rows


def _seed(daemon: MobileDaemon) -> None:
    from local_operator.paths import config_dir
    from local_operator.resume import write_session_title

    directory = config_dir() / "sessions" / SESSION_ID
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    write_session_title(directory, TITLE, user_set=False, past_names=[])

    entries: list[TranscriptEntry] = [
        TranscriptEntry(id="u0", kind="user", text="Open the release window", final=True),
        *_rows("t0", GROUP_CALLS),
        TranscriptEntry(id="a0", kind="assistant", text="Window is open.", final=True),
        TranscriptEntry(id="u1", kind="user", text="Tell the team and Priya", final=True),
        TranscriptEntry(id="a1", kind="assistant", text="Posting to the team now."),
        *_rows("t1", LIVE_CALLS),
    ]
    pid = 920001
    record = SessionRecord(
        pid=pid,
        kind="tui",
        session_id=SESSION_ID,
        conversation_name=TITLE,
        cwd="/synthetic/worktree",
        model_label="fixture/model",
        control_port=1,
        control_key="fixture",
        detached=True,
        busy=True,
    )
    entry = SessionEntry(record)
    entry.projection = SessionProjection(
        session_id=SESSION_ID,
        pid=pid,
        kind="tui",
        conversation_name=TITLE,
        cwd="/synthetic/worktree",
        model_label="fixture/model",
        transcript=entries,
    )
    daemon.table.entries[pid] = entry


FIXTURE_PASSWORD_ENV = "LOP_MOBILE_FIXTURE_PASSWORD"


def _password(argv_rest: list[str]) -> str:
    value = argv_rest[0] if argv_rest else os.environ.get(FIXTURE_PASSWORD_ENV, "")
    if not value:
        raise SystemExit(f"pass a per-run password as argv[2] or set {FIXTURE_PASSWORD_ENV}")
    return value


async def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4190
    daemon = MobileDaemon(port=port, password=_password(sys.argv[2:]), dial_registrants=False)
    _seed(daemon)
    app = build_app(daemon)
    print(f"Fixture phone list on http://127.0.0.1:{port}", flush=True)
    await uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())
