import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
PORT = int(os.environ.get("LO_MOBILE_FIXTURE_PORT", "4179"))
DISCONNECT_SENTINEL = Path(f"/tmp/lop-fixture-disconnect-{PORT}")
DETAIL_FAILURE_SENTINEL = Path(f"/tmp/lop-fixture-detail-failure-{PORT}")

from local_operator.mobile.daemon import (  # noqa: E402
    MobileDaemon,
    SessionEntry,
    build_app,
)
from local_operator.mobile.types import (  # noqa: E402
    SessionProjection,
    SessionRecord,
    SubagentRow,
    TodoItem,
    TodoPhase,
    TranscriptEntry,
)

SESSION = "fixture-root"


def entry(id, kind, text="", **kwargs):
    return TranscriptEntry(id=id, kind=kind, text=text, **kwargs)


projection = SessionProjection(
    session_id=SESSION,
    pid=4242,
    conversation_name="Mobile agent review fixture",
    streaming=True,
    activity="coordinating remediation evidence",
    activity_started_s=128,
    version=41,
    transcript=[entry("root-1", "user", "Review the full-screen mobile agent flow.")],
)
long_result = """## Completed result

**PR #298** now preserves selected detail through rapid projection updates and
exposes reconnecting state without discarding cached content.

- Verified child-only pagination and expandable tool diffs.
- Preserved native browser Back and Forward for in-app visits.
- Made direct-link Header Back resolve to the hierarchy parent.

See [the pull request](https://github.com/damianvtran/local-operator/pull/298)
for request logs and exact evidence. This deliberately long result continues in
normal document flow so every line remains discoverable by scrolling the
conversation once, rather than finding a tiny nested scroller.
"""
child_transcript = [
    entry(
        "c-user", "parent_message", "Implement all round-one findings with real mobile evidence."
    ),
    entry(
        "c-assistant",
        "assistant",
        "I am updating the detail loader and preserving cached state while SSE reconnects.",
    ),
    entry(
        "c-tool",
        "tool",
        "",
        tool_call_id="tool-1",
        tool_name="edit",
        tool_state="done",
        summary="agent-view.tsx",
        intent="Fixing detail routing",
        diff_added=18,
        diff_removed=7,
        elapsed_s=1.4,
        details={"diff": ["@@ -1,2 +1,3 @@", "-history.back()", "+navigateUp(parentPath)"]},
    ),
]
projection.subagents = [
    SubagentRow(
        job_id="child",
        label="mobile-fullscreen-remediation",
        agent="coder",
        status="running",
        progress="capturing populated 360px evidence",
        elapsed_s=128,
        parent_job_id=None,
        session_id="fixture-child",
        prompt="Remediate code, design, and UX findings.",
        effort="high",
        child_ids=["grandchild"],
        peer_ids=["peer-complete", "peer-failed"],
        transcript=child_transcript,
        todos=[
            TodoPhase(
                name="Remediation",
                items=[
                    TodoItem(text="Fix coalesced fetching", status="done"),
                    TodoItem(text="Capture browser evidence", status="pending"),
                ],
            )
        ],
        activity="capturing populated 360px evidence",
    ),
    SubagentRow(
        job_id="grandchild",
        label="fixture-builder",
        agent="coder",
        status="completed",
        elapsed_s=44,
        parent_job_id="child",
        session_id="fixture-grandchild",
        prompt="Build nested mobile fixture.",
        ancestor_ids=["child"],
        ancestors=["mobile-fullscreen-remediation"],
        peer_ids=["grandchild-peer"],
        child_ids=["great-grandchild"],
        result_text=long_result,
        transcript=[
            entry(
                "g-assistant",
                "assistant",
                "The nested fixture is populated and ready for route testing.",
            )
        ],
    ),
    SubagentRow(
        job_id="grandchild-peer",
        label="responsive-hierarchy-audit-with-a-deliberately-long-label",
        agent="reviewer",
        status="running",
        elapsed_s=19,
        parent_job_id="child",
        session_id="fixture-grandchild-peer",
        peer_ids=["grandchild"],
        progress="checking deep hierarchy navigation",
        activity="checking deep hierarchy navigation",
    ),
    SubagentRow(
        job_id="great-grandchild",
        label="true-pixel-capture",
        agent="designer",
        status="completed",
        elapsed_s=31,
        parent_job_id="grandchild",
        session_id="fixture-great-grandchild",
        ancestor_ids=["child", "grandchild"],
        ancestors=["mobile-fullscreen-remediation", "fixture-builder"],
        result_text="Captured nested evidence without horizontal overflow.",
    ),
    SubagentRow(
        job_id="peer-complete",
        label="design-evidence",
        agent="designer",
        status="completed",
        elapsed_s=73,
        parent_job_id=None,
        session_id="fixture-peer-complete",
        peer_ids=["child", "peer-failed"],
        result_text=long_result,
        transcript=[],
    ),
    SubagentRow(
        job_id="peer-failed",
        label="disconnect-probe",
        agent="reviewer",
        status="failed",
        elapsed_s=15,
        parent_job_id=None,
        session_id="fixture-peer-failed",
        peer_ids=["child", "peer-complete"],
        error_text=(
            "The first relay connection was intentionally interrupted. Cached detail "
            "remained visible and the stream recovered."
        ),
        transcript=[],
    ),
]

daemon = MobileDaemon(port=PORT, password="fixture-review")
daemon.capture_subagent_details(projection)
daemon.session_projections[SESSION] = projection
record = SessionRecord(
    pid=4242,
    kind="tui",
    session_id=SESSION,
    conversation_name=projection.conversation_name,
    cwd=str(Path.cwd()),
    model_label="fixture",
    control_port=1,
    control_key="fixture",
)
entry_state = SessionEntry(record)
entry_state.projection = projection
daemon.table.entries[record.pid] = entry_state
base_app = build_app(daemon)


class FixtureRecovery:
    """Expose deterministic relay and detail failures without changing fixture data."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] == "http"
            and scope["path"] == f"/api/sessions/{SESSION}/agents/child"
            and DETAIL_FAILURE_SENTINEL.exists()
        ):
            body = b'{"error":"temporary detail failure"}'
            await send(
                {
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        if scope["type"] == "http" and scope["path"] == f"/api/sessions/{SESSION}/events":
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"text/event-stream"),
                        (b"cache-control", b"no-cache"),
                    ],
                }
            )
            body = f"event: projection\ndata: {json.dumps(projection.to_json())}\n\n".encode()
            await send({"type": "http.response.body", "body": body, "more_body": True})
            while not DISCONNECT_SENTINEL.exists():
                await asyncio.sleep(0.1)
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return
        await self.app(scope, receive, send)


app = FixtureRecovery(base_app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
