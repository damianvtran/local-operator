"""Serve the REAL phone bundle with synthetic conversations that carry mermaid fences.

Run:  PYTHONPATH=. .venv/bin/python scripts/mobile_pele_fixture.py PORT [--dist DIR] [--host HOST]

One conversation per case (``pele-<name>``), so a person or the harness browser can
open exactly one: flowchart, a wide left-to-right flowchart, sequence, state, pie,
a malformed source, an UNCLOSED fence (a reply still streaming), a hostile source
(HTML in labels, ``javascript:`` and ``click`` callbacks) and a mixed answer.

``/phone?s=<name>&w=390&h=844&theme=<id>`` wraps the real app in a phone-sized
iframe, because the harness browser cannot resize its own window: media queries
and widths inside the frame are the phone's, and the app is the unmodified bundle.
``--dist`` serves another build of the bundle (the BEFORE frame is the same fixture
against ``origin/main``'s ``dist``).

The password is NOT defaulted and never printed: export ``LOP_MOBILE_FIXTURE_PASSWORD``
(the variable the sibling mobile fixtures use). No registrant sockets and no runtime
scanner; HOME and LOCAL_OPERATOR_CONFIG_DIR are re-homed by ``scripts.probe_isolation``,
so this never touches the operator's live daemon or any real session.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import uvicorn
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

import scripts.probe_isolation  # noqa: F401  -- must be the first local import
from local_operator.mobile import daemon as daemon_module
from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
from local_operator.mobile.types import (
    SessionProjection,
    SessionRecord,
    TranscriptEntry,
)
from scripts.mobile_overflow_fixture import required_password

FLOWCHART = """```mermaid
flowchart TD
    A[Operator sends a prompt] --> B{Plane reachable?}
    B -->|yes| C[Stream the turn]
    B -->|no| D[Show the offline banner]
    C --> E[Render the transcript]
    D --> F[Retry with backoff]
    F --> B
    E --> G((Done))
```"""

WIDE = """```mermaid
flowchart LR
    A[Receive request] --> B[Validate input] --> C[Look up the session]
    C --> D[Check permissions] --> E[Run the tool] --> F[Collect the output]
    F --> G[Format the reply] --> H[Stream it to the phone]
```"""

SEQUENCE = """```mermaid
sequenceDiagram
    participant U as Operator
    participant UI as Phone
    participant P as Relay
    U->>UI: Send prompt
    UI->>P: POST /turns
    P-->>UI: Stream events
    loop Every event
        UI->>UI: Reduce into transcript
    end
    UI-->>U: Settled answer
```"""

STATE = """```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Streaming: send
    Streaming --> Settled: done
    Streaming --> Failed: error
    Failed --> Idle: retry
    Settled --> Idle: next prompt
    Settled --> [*]
```"""

PIE = """```mermaid
pie title Turn time
    "Model" : 62
    "Tools" : 25
    "Render" : 8
    "Other" : 5
```"""

INVALID = """This one is broken on purpose:

```mermaid
flowchart TD
    A[Start --> B
```"""

# The fence has not closed yet: a reply still streaming in.
STREAMING = """Streaming state, the fence is still open:

```mermaid
flowchart LR
    A --> B
    B --> C"""

HOSTILE = """```mermaid
flowchart LR
    A["<img src=x onerror=alert(1)>"] --> B["<script>alert(2)</script>"]
    B --> C["javascript:alert(3)"]
    click A "javascript:alert(4)" "tip"
    click B callback "alert(5)"
    click C href "https://example.com/phish" _blank
```"""

MIXED = (
    "Here is the plan. First the **overview**, then the detail:\n\n"
    + FLOWCHART
    + "\n\nAnd the call order:\n\n"
    + SEQUENCE
    + "\n\nAn ordinary fence stays code:\n\n```python\nprint('hi')\n```\n"
)


def conversations() -> dict[str, str]:
    """Session name -> the assistant text, in the order they are registered."""
    return {
        "flowchart": "Here is the flow:\n\n" + FLOWCHART,
        "wide": "A wide one:\n\n" + WIDE,
        "sequence": "The call order:\n\n" + SEQUENCE,
        "state": "The turn's states:\n\n" + STATE,
        "pie": "Where the time goes:\n\n" + PIE,
        "invalid": INVALID,
        "streaming": STREAMING,
        "hostile": "Hostile labels:\n\n" + HOSTILE,
        "mixed": MIXED,
    }


def _projection(session_id: str, pid: int, text: str, user: str) -> SessionProjection:
    return SessionProjection(
        session_id=session_id,
        pid=pid,
        kind="tui",
        conversation_name=f"Diagram {session_id}",
        streaming=False,
        activity="",
        activity_started_s=None,
        model_label="Claude Opus 5.5",
        effort="high",
        cwd="/synthetic/project",
        transcript=[
            TranscriptEntry(id="u-1", kind="user", text=user),
            TranscriptEntry(id="a-1", kind="assistant", text=text),
        ],
        version=3,
    )


PHONE_PAGE = Path(__file__).with_name("mobile_pele_phone.html")


def with_phone_frame(app):
    """Prepend ``/phone`` before the daemon's SPA catch-all. Fixture-only."""

    async def phone(request: Request) -> HTMLResponse:
        query = request.query_params
        name = query.get("s", "flowchart")
        width = max(240, min(int(query.get("w", "390")), 1200))
        height = max(240, min(int(query.get("h", "844")), 2400))
        theme = query.get("theme", "localOperatorDark")
        page = PHONE_PAGE.read_text(encoding="utf-8")
        for token, value in (
            ("__W__", str(width)),
            ("__H__", str(height)),
            ("__PRE_LEFT__", str(width + 30)),
            ("__THEME__", _js(theme)),
            ("__SESSION__", _js(f"pele-{name}")),
        ):
            page = page.replace(token, value)
        return HTMLResponse(page)

    app.router.routes.insert(0, Route("/phone", phone))
    return app


def _js(value: str) -> str:
    return json.dumps(value).replace("</", "<\\/")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int)
    parser.add_argument("--dist", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    if args.dist is not None:
        # The module reads its dist directory at app build time; this fixture is the
        # one place that points it somewhere else (the BEFORE frame's bundle).
        daemon_module._DIST_DIR = args.dist.resolve()
    password = required_password([])

    daemon = MobileDaemon(port=args.port, password=password, dial_registrants=False)
    for index, (name, text) in enumerate(conversations().items()):
        projection = _projection(
            f"pele-{name}", 930001 + index, text, f"Show me the {name} diagram."
        )
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
    app = with_phone_frame(build_app(daemon))
    print(f"Fixture mobile: http://{args.host}:{args.port}", flush=True)
    await uvicorn.Server(
        uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    ).serve()


if __name__ == "__main__":
    asyncio.run(main())
