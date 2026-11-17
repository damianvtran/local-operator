"""Serve the REAL mobile bundle with synthetic conversations that provoke the two
reading defects, for before/after capture at a phone viewport.

Run:  PYTHONPATH=. .venv/bin/python scripts/mobile_reading_fixture.py [PORT] <password>

Open ``/showcase`` in the harness browser (phone size and theme controls live
there) and log in at ``/``. The password is NOT fixed and never printed: pass
it as the second argument or set ``LOP_MOBILE_FIXTURE_PASSWORD`` (see the
sibling fixtures' note on why a literal would be a reusable credential).

Sessions, each shaped to isolate one reading surface:

* ``reading-long``  -- three completed turns. The first two carry a long, wrapped
  user message plus tool work, so they fold into the condensed turn bar and the
  user's words collapse to ONE truncated line; the last turn ends on a very long
  assistant answer, taller than the transcript, to show the foot of a long reply.
* ``reading-short`` -- one short user message and one short answer: the control,
  which must look the same before and after except for the user bubble's edge.

No runtime scanner and no registrant sockets (``dial_registrants=False``), so
this never touches a live daemon or any real session. HOME and
LOCAL_OPERATOR_CONFIG_DIR are re-homed by ``scripts.probe_isolation`` on import.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import uvicorn
from starlette.responses import HTMLResponse
from starlette.routing import Route

import scripts.probe_isolation  # noqa: F401  -- must be the first local import
from local_operator.mobile import daemon as daemon_module
from local_operator.mobile.daemon import MobileDaemon, SessionEntry
from local_operator.mobile.types import (
    SessionProjection,
    SessionRecord,
    SubagentRow,
    TranscriptEntry,
)
from scripts.mobile_showcase_fixture import (
    _conversation_dir,
    fixture_cookie_name,
    required_password,
    showcase_app,
)

LONG_USER_1 = (
    "Two things on the phone relay. First, some replies look cut off at the "
    "bottom and I can't tell whether there is more to read, because nothing "
    "lets me open or close them. Second, my own messages blend into the "
    "background, so when I scroll back I lose track of which lines I wrote "
    "and which the agent wrote. Please look at both and tell me what you would "
    "change before touching anything."
)

LONG_USER_2 = (
    "Good. Go ahead with the smaller of the two fixes first, keep the change "
    "inside the relay bundle, and show me a before and after at phone size in "
    "both the light and the dark theme so I can judge it with my own eyes "
    "rather than from your description."
)

ANSWER_1 = (
    "I read both surfaces in the relay source and reproduced them in a phone "
    "sized frame.\n\n"
    "**The cut-off message.** A long reply is not clipped by the message "
    "itself. The transcript scrolls under the composer and fades its last "
    "24px, so a tall reply at the tail looks sliced off, and nothing on the "
    "reply says it can be opened or closed.\n\n"
    "**The blended message.** Your message sits on the `surface` ground, one "
    "step from the page ground, so in light mode the two are nearly the same "
    "colour and nothing marks the edge."
)

ANSWER_3 = "\n\n".join(
    [
        "Here is the full write-up. It is deliberately long so the reply is "
        "taller than the visible transcript on a phone.",
        "**1. What changed.** Your own messages now carry a visible outline, and a "
        "long one inside a finished turn shows three lines with a Show more button.",
        "**2. Why a control and not a fade.** A fade says there might be more but "
        "gives the reader no way to get it. A labelled control says there is more "
        "and what pressing it does.",
        "**3. What does not change.** Replies from the agent are not folded. They "
        "scroll, and the edge of the transcript fades where more text exists.",
        "**4. Where it stops.** The preview is a line count, not a pixel height, so "
        "large text settings and the 320, 360 and 390 widths all behave.",
        "**5. Checks.** Typecheck, the unit suite for the transcript, and a "
        "phone-size capture in light and dark. The capture shows the folded state, "
        "the opened state and the short control case.",
        "**6. Risks.** Opening a message changes the height of a row, so the scroll "
        "position must hold. Opening keeps the start of the message where it was; "
        "closing keeps the button under the thumb.",
        "**7. Not covered.** Tablet and desktop widths, right-to-left layout beyond "
        "the logical-property sweep, and the live streaming transition.",
        "**8. Next.** Review rounds, then the same treatment for the steer row if " "you want it.",
        "**9. Notes for the record.** The numbered list above is only here to give "
        "this reply real height. Every paragraph is a full sentence, so the wrapped "
        "height at 390px is close to what a genuine answer takes.",
        "**10. Last line.** If you can read this line without scrolling past the "
        "composer, the reply is not tall enough for the check.",
    ]
)


def _tool(entry_id: str, name: str, summary: str) -> TranscriptEntry:
    return TranscriptEntry(
        id=entry_id,
        kind="tool",
        tool_call_id=f"call-{entry_id}",
        tool_name=name,
        tool_state="done",
        summary=summary,
        elapsed_s=1.2,
    )


def _tools(prefix: str, count: int) -> list[TranscriptEntry]:
    names = ["read", "grep", "bash", "edit", "read", "grep", "bash"]
    return [
        _tool(f"{prefix}-t{i}", names[i % len(names)], f"relay/src/file-{i}.tsx")
        for i in range(count)
    ]


def _long_projection() -> SessionProjection:
    transcript = [
        TranscriptEntry(id="rl-u1", kind="user", text=LONG_USER_1),
        *_tools("rl1", 7),
        TranscriptEntry(id="rl-a1", kind="assistant", text=ANSWER_1),
        TranscriptEntry(id="rl-u2", kind="user", text=LONG_USER_2),
        *_tools("rl2", 5),
        TranscriptEntry(id="rl-a2", kind="assistant", text=ANSWER_3),
    ]
    projection = SessionProjection(
        session_id="5c0a1e000001",
        pid=940001,
        kind="tui",
        conversation_name="Reading: long turns",
        cwd="/synthetic/home/projects/relay",
        model_label="Claude Sonnet 5",
        streaming=False,
        transcript=transcript,
        version=3,
    )
    # The operator's screen carried the roster pill ("Agents 0/90"): 90 finished
    # agents, none running. It sits in the tray directly under the transcript.
    projection.subagents = [
        SubagentRow(job_id=f"rl-job-{i}", label=f"worker-{i}", status="completed", elapsed_s=30)
        for i in range(90)
    ]
    return projection


def _short_projection() -> SessionProjection:
    return SessionProjection(
        session_id="5c0a1e000002",
        pid=940002,
        kind="tui",
        conversation_name="Reading: short control",
        cwd="/synthetic/home/projects/relay",
        model_label="Claude Sonnet 5",
        streaming=False,
        transcript=[
            TranscriptEntry(id="rs-u1", kind="user", text="What time is it in UTC?"),
            TranscriptEntry(id="rs-a1", kind="assistant", text="It is 14:05 UTC."),
        ],
        version=2,
    )


async def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4189
    password = required_password(sys.argv[2:])
    daemon_module.COOKIE_NAME = fixture_cookie_name(port)
    daemon = MobileDaemon(port=port, password=password, dial_registrants=False)
    for projection in (_long_projection(), _short_projection()):
        # The listing admits a session by its durable directory, not by the live
        # projection alone.
        _conversation_dir(projection.session_id, projection.conversation_name)
        record = SessionRecord(
            pid=projection.pid,
            kind="tui",
            session_id=projection.session_id,
            conversation_name=projection.conversation_name,
            cwd=projection.cwd,
            model_label=projection.model_label,
            control_port=1,
            control_key="fixture",
        )
        entry = SessionEntry(record)
        entry.projection = projection
        daemon.session_projections[projection.session_id] = projection
        daemon.table.entries[record.pid] = entry
    app = showcase_app(daemon)

    async def reading_phone(_request):
        html = Path(__file__).with_name("mobile_reading_phone.html").read_text()
        return HTMLResponse(html)

    # Prepend before the daemon's SPA catch-all, like the showcase's own routes.
    app.router.routes.insert(0, Route("/reading", reading_phone))
    print(json.dumps({"fixture": "reading", "port": port}), flush=True)
    await uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())
