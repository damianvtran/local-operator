"""Serve the REAL phone bundle in the conversations the open-by-default capture needs.

Run (normally started for you by ``scripts/mobile_asks_open_capture.py``)::

    LOP_MOBILE_FIXTURE_PASSWORD=<per-run value> \
        .venv/bin/python scripts/mobile_asks_open_fixture.py [PORT]

WHAT THIS IS FOR. The phone's asks sheet now opens by ITSELF, once, when a conversation
with pending asks is opened (``lib/ask-open-policy.ts``; six clauses, the same ones the TUI
keeps). Evidence for that is a state matrix, and each state needs a conversation in exactly
that state plus a way to change it mid-run, because two of the states are about what the
phone does when the queue changes UNDER an open screen:

* ``open-none``    -- a live queue with nothing in it (``asks`` absent, ``asks_open: 0``).
  State 1: closed.
* ``open-settled`` -- every ask already answered or declined, ``asks_open: 0``.
  State 3: closed.
* ``open-pending`` -- TWO asks waiting (three questions), served by a REAL runtime so the
  cards carry no "this conversation is not running" strip and read as they do in use.
  State 2: opens on arrival. State 4: the user closes it; a re-render, a new ask and a
  trip to the session list and back must leave it closed.
* ``open-typing``  -- starts with NO ask fields at all (a runtime that has not said yet),
  so the view is still undecided when the user starts typing; the asks arrive afterwards.
  State 5: the sheet must not open over the keyboard.

THE CONTROL CHANNEL IS STDIN, not an HTTP route: a route would be a second way to mutate a
daemon that the capture would have to authenticate to, and this fixture never listens on
anything but the daemon's own port. One command per line, one ``ack <command>`` line back
once the frame has been pushed to every phone watching (an ``error <command>: ...`` line if
it could not):

* ``pending-refresh``  -- re-publish ``open-pending`` unchanged but for its version: a
  re-render with no queue change.
* ``pending-new-ask``  -- add a third open ask to ``open-pending``: an ask ARRIVING while
  the phone watches.
* ``typing-arrive``    -- publish two open asks into ``open-typing``.

No runtime scanner and no registrant sockets beyond the one live session (``dial_registrants
=False``), so this never touches the operator's live daemon or sessions. HOME and
LOCAL_OPERATOR_CONFIG_DIR are re-homed by ``scripts.probe_isolation`` on import. The tree
that is SERVED is the one this file sits in, and the startup line names it: the same script
runs against a detached worktree of the pre-change build for the BEFORE frames, so the
provenance of each set is a printed fact and not an assumption.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# THE DOCUMENTED RUN LINE HAS TO WORK without PYTHONPATH, and the tree that CONTAINS this
# file has to win over any other checkout the interpreter's site-packages points at: the
# BEFORE frames come from a detached worktree of the pre-change build, driven by another
# tree's venv, and without this the venv's editable install would silently serve the wrong
# tree's bundle. Same two-line bootstrap as ``scripts/mobile_asks_capture.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn  # noqa: E402

import scripts.probe_isolation  # noqa: E402,F401  -- must be the first local import
from local_operator.mobile.daemon import (  # noqa: E402
    MobileDaemon,
    SessionEntry,
    _fan_out,
    build_app,
)
from local_operator.mobile.types import (  # noqa: E402
    PendingAskWire,
    SessionProjection,
    SessionRecord,
    TranscriptEntry,
)
from scripts.mobile_overflow_fixture import (  # noqa: E402
    QueuedAskHarness,
    _ask,
    _question,
    required_password,
    seed_ask_index,
    serve_queued_ask_runtime,
)


def _exchange(prefix: str, lead: str, reply: str) -> list[TranscriptEntry]:
    """A short conversation, so every frame is photographed over a REAL transcript: the
    question each frame answers is "can the reader still see what the sheet covers"."""
    return [
        TranscriptEntry(id=f"{prefix}-1", kind="user", text=lead),
        TranscriptEntry(id=f"{prefix}-2", kind="assistant", text=reply),
    ]


def _static_projection(
    session_id: str,
    pid: int,
    name: str,
    transcript: list[TranscriptEntry],
    *,
    asks: list[PendingAskWire] | None,
    asks_open: int | None,
) -> SessionProjection:
    projection = SessionProjection(
        session_id=session_id,
        pid=pid,
        kind="tui",
        conversation_name=name,
        streaming=False,
        transcript=transcript,
        version=2,
    )
    # ABSENCE IS THE CAPABILITY PROXY (design section 4): ``asks=None`` is "no rows on the
    # wire", and ``asks_open`` says whether the runtime can answer at all. ``(None, 0)`` is a
    # live queue with nothing in it; ``(None, None)`` is a runtime that has not said.
    projection.asks = asks
    projection.asks_open = asks_open
    return projection


def _none_projection() -> SessionProjection:
    return _static_projection(
        "open-none",
        900301,
        "Nothing waiting",
        _exchange(
            "n",
            "Tidy the release notes.",
            "Done. Nothing needs your input, so I carried on with the changelog.",
        ),
        asks=None,
        asks_open=0,
    )


def _settled_projection() -> SessionProjection:
    return _static_projection(
        "open-settled",
        900302,
        "Everything answered",
        _exchange(
            "s",
            "Ask me before you touch the schema.",
            "Both of your answers are in, so I went ahead with the migration.",
        ),
        asks=[
            _ask(
                "os-1",
                status="answered",
                created_s=-1800,
                expires_in_s=600,
                delivered=True,
                questions=[
                    _question(
                        "a1",
                        "Which migration order should I use?",
                        options=[("additive first", "no downtime"), ("drop first", "smaller diff")],
                    )
                ],
                answers={"a1": ["additive first"]},
                answered_by="tui",
            ),
            _ask(
                "os-2",
                status="declined",
                created_s=-1200,
                expires_in_s=600,
                questions=[_question("d1", "Backfill from the audit log?")],
                answered_by="desktop",
            ),
        ],
        asks_open=0,
    )


def _pending_asks() -> list[PendingAskWire]:
    """Two asks, three questions: the dock reads ``3 questions waiting``, and the sheet's
    head card is the multi-question form (a picker and a multi-select)."""
    return [
        _ask(
            "op-2",
            created_s=-60,
            expires_in_s=840,
            questions=[
                _question(
                    "q3",
                    "Ship it behind the flag?",
                    options=[("yes", "dark until the flip"), ("no", "hold the surface too")],
                )
            ],
        ),
        _ask(
            "op-1",
            created_s=-300,
            expires_in_s=600,
            questions=[
                _question(
                    "q1",
                    "Which rollout order should I use?",
                    options=[
                        ("layout first", "the smallest diff reaches the wheel"),
                        ("roster first", "the change the operator reported"),
                    ],
                ),
                _question(
                    "q2",
                    "Which surfaces must the verification cover?",
                    options=[
                        ("phone", "the bundle that ships in the wheel"),
                        ("terminal", "the viewer reading the same fold"),
                    ],
                    multi=True,
                ),
            ],
        ),
    ]


def _pending_projection() -> SessionProjection:
    projection = _static_projection(
        "open-pending",
        900303,
        "Two questions waiting",
        _exchange(
            "p",
            "Plan the rollout and ask me what you need.",
            "I queued three questions and kept going on the parts that do not depend on them.",
        ),
        asks=_pending_asks(),
        asks_open=2,
    )
    projection.version = 3
    return projection


def _typing_projection() -> SessionProjection:
    # NO ask fields: the runtime has not said whether it has a queue, so the phone's view of
    # this conversation is undecided. That is the only state in which a sheet could still
    # open while the user is already typing, which is what the frame exists to rule out.
    return _static_projection(
        "open-typing",
        900304,
        "Typing when it lands",
        _exchange(
            "t",
            "Draft the migration plan.",
            "Working on it. I may need to ask you a couple of things.",
        ),
        asks=None,
        asks_open=None,
    )


def _typing_asks() -> list[PendingAskWire]:
    return [
        _ask(
            "ot-1",
            created_s=-30,
            expires_in_s=870,
            questions=[
                _question(
                    "t1",
                    "Which environment should the dry run target?",
                    options=[("staging", "safe to break"), ("production", "read-only")],
                )
            ],
        ),
        _ask(
            "ot-2",
            created_s=-20,
            expires_in_s=880,
            questions=[_question("t2", "Should I also rotate the key?")],
        ),
    ]


def _push_live(harness: QueuedAskHarness) -> None:
    """Publish the live session's current queue, the way a real runtime's fold does:
    the derived index first (so the aggregate the sheet reads agrees), then the frame."""
    harness.projection.version += 1
    harness._publish_index()
    if harness.on_change is not None:
        harness.on_change()


async def _control_loop(
    daemon: MobileDaemon,
    harness: QueuedAskHarness,
    static_entries: dict[str, SessionEntry],
) -> None:
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:  # EOF: the capture closed our stdin; the daemon keeps serving until killed
            return
        command = line.strip()
        if not command:
            continue
        try:
            if command == "pending-refresh":
                _push_live(harness)
            elif command == "pending-new-ask":
                rows = list(harness.projection.asks or [])
                rows.insert(
                    0,
                    _ask(
                        "op-3",
                        created_s=0,
                        expires_in_s=900,
                        questions=[
                            _question(
                                "q4",
                                "Also rotate the staging key?",
                                options=[("yes", "after the rollout"), ("no", "leave it")],
                            )
                        ],
                    ),
                )
                harness.projection.asks = rows
                harness.projection.asks_open = (harness.projection.asks_open or 0) + 1
                _push_live(harness)
            elif command == "typing-arrive":
                entry = static_entries["open-typing"]
                projection = entry.projection
                assert projection is not None
                projection.asks = _typing_asks()
                projection.asks_open = 2
                projection.version += 1
                _fan_out(entry, daemon)
            else:
                print(f"error {command}: unknown command", flush=True)
                continue
        except Exception as exc:  # noqa: BLE001 -- a control failure must be reported, not die
            print(f"error {command}: {exc}", flush=True)
            continue
        print(f"ack {command}", flush=True)


async def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4189
    password = required_password(sys.argv[2:])
    daemon = MobileDaemon(port=port, password=password, dial_registrants=False)

    static_entries: dict[str, SessionEntry] = {}
    static = (_none_projection(), _settled_projection(), _typing_projection())
    for projection in static:
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
        static_entries[projection.session_id] = entry

    # The aggregate (``GET /api/asks``) is index-backed, and the sheet reads it when it
    # opens: seed it with the live session's rows or the sheet would open onto "nothing
    # waiting" beside a dock that says three questions are. EVERY session is passed, the ones
    # with no asks too, because ``seed_ask_index`` is also what creates a session's directory,
    # and the session LIST only paints a conversation whose directory exists (the daemon's
    # ``_live_generation_is_user_facing``): without it the three static conversations were
    # served and routable but absent from the list the capture taps them from.
    live = _pending_projection()
    seed_ask_index([*static, live])
    harness, registrant, dial = await serve_queued_ask_runtime(daemon, live)
    app = build_app(daemon)

    import local_operator

    # A PRINTED FACT, so the provenance of a frame set is not an assumption: the capture
    # records this line beside the frames, and a BEFORE set that served the wrong tree
    # would say so here.
    print(f"Fixture serves tree: {Path(local_operator.__file__).resolve().parents[1]}", flush=True)
    print(f"Fixture mobile: http://127.0.0.1:{port}", flush=True)
    control = asyncio.ensure_future(_control_loop(daemon, harness, static_entries))
    try:
        await uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        ).serve()
    finally:
        control.cancel()
        dial.cancel()
        registrant.close()


if __name__ == "__main__":
    asyncio.run(main())
