"""Serve the REAL mobile bundle with two synthetic conversations (idle, working)
for composer action-cluster capture.

Run:  PYTHONPATH=. python scripts/mobile_composer_slots_fixture.py PORT [stt]

``stt`` is ``available`` (default) or ``unavailable``: the daemon's dictation
capability answer is forced, because the real answer depends on the host's
provider keys and a frame must not. The password is NOT defaulted and never
printed: export ``LOP_MOBILE_FIXTURE_PASSWORD`` (the same variable
``scripts/mobile_overflow_fixture.py`` and the capture scripts use).

No registrant sockets and no runtime scanner (``dial_registrants=False``); HOME
and LOCAL_OPERATOR_CONFIG_DIR are re-homed by ``scripts.probe_isolation``.
"""

from __future__ import annotations

import asyncio
import sys

import uvicorn

import scripts.probe_isolation  # noqa: F401  -- must be the first local import
from local_operator.mobile import stt
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
from local_operator.mobile.types import (
    SessionProjection,
    SessionRecord,
    TranscriptEntry,
)
from scripts.mobile_overflow_fixture import required_password

LONG_ACTIVITY = (
    "running the full mobile web unit suite and the bundle guard against the worktree build"
)


def _projection(
    session_id: str, pid: int, *, streaming: bool, rows: int = 0, activity: str = ""
) -> SessionProjection:
    projection = SessionProjection(
        session_id=session_id,
        pid=pid,
        kind="tui",
        conversation_name=f"Composer {session_id}",
        streaming=streaming,
        activity=(activity or "reading the composer") if streaming else "",
        activity_started_s=42 if streaming else None,
        model_label="Claude Opus 5.5",
        effort="high",
        cwd="/synthetic/project",
        transcript=[
            TranscriptEntry(id="u-1", kind="user", text="Move the dictation button next to send."),
            TranscriptEntry(
                id="a-1",
                kind="assistant",
                text="Both controls now live in one right-hand cluster.",
            ),
        ],
        version=3,
    )
    # A transcript long enough to scroll, so "scrolled up" is a real position.
    for i in range(rows):
        projection.transcript.append(
            TranscriptEntry(
                id=f"long-{i}",
                kind="user" if i % 2 == 0 else "assistant",
                text=f"Row {i}: "
                + "the dictation slot and the stop slot never share a place. " * 3,
            )
        )
    return projection


async def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4188
    mode = sys.argv[2] if len(sys.argv) > 2 else "available"
    password = required_password([])

    async def answer(**_: object) -> dict[str, object]:
        if mode == "unavailable":
            return {"available": False, "path": None, "reason": "fixture"}
        return {"available": True, "path": "provider_stt_radient", "reason": ""}

    # Forced: the daemon imports the function lazily at request time, so
    # patching the module attribute is what the capabilities frame reads.
    stt.stt_availability = answer  # type: ignore[assignment]

    daemon = MobileDaemon(port=port, password=password, dial_registrants=False)
    for projection in (
        _projection("slots-idle", 910001, streaming=False),
        _projection("slots-working", 910002, streaming=True),
        _projection("slots-long", 910003, streaming=True, rows=40, activity=LONG_ACTIVITY),
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
    print(f"Fixture mobile: http://127.0.0.1:{port}", flush=True)
    await uvicorn.Server(
        uvicorn.Config(build_app(daemon), host="127.0.0.1", port=port, log_level="warning")
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())
