"""A REAL ``_run_turn`` carrying a reserved body into the provider request.

WHY this file exists: every other test of the body vertical stops at a module
boundary. This one drives the actual ``Session`` turn — the same ``_run_turn``,
``LoopConfig`` and ``transform_context`` the operator's agent runs — with a
scripted provider that CAPTURES the request instead of answering over a socket.

Hermetic by construction: no hub, no network, no credentials. The bridge store
is a real sqlite file in ``tmp_path`` and the "hub" is a fake client whose only
job is to answer ``conversation_body`` with the bytes the reservation hashes.

What it proves (and the operator's question — did the body reach a request):
1. the body rides the request EXACTLY ONCE, byte-identical, inside the
   untrusted object, under an intact trusted header;
2. it is on every provider call of the turn (including after a tool result) and
   never twice within one call;
3. settlement completes the row and the transcript holds no body text;
4. a MID-TURN aside carried no frame;
5. a turn whose consent is revoked mid-flight sends no frame afterwards.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.team_bridge.store import (
    OwnerReceipt,
    OwnerSnapshot,
    Store,
    state_db_path,
)

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)
CONVERSATION_ID = "bcv_abcdefgh"
MESSAGE_ID = "bcmsg_abcdef01"
TENANT = "3f2a1111-2222-4333-8444-555566667777"
SESSION_ID = "sess_cos"
BODY = "OPERATOR-CANARY-4417 the admitted message body"
DIGEST = __import__("hashlib").sha256(BODY.encode("utf-8")).hexdigest()
FINGERPRINT = "aa11" * 16
FRAME_MARKER = "body_representation:"
HEADER_MARKER = "VERIFIED SOURCE"


class CapturingStream:
    """Answers each request from a script and keeps the request itself.

    ``scripted[i]`` answers call i (the last entry repeats). The aside made by a
    tool during call 1 is its own call, so the scripts are ordered by PHYSICAL
    request, which is exactly the unit the invariant is written in.
    """

    def __init__(self, scripted: list[list[StreamEvent]]) -> None:
        self.scripted = scripted
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        events = self.scripted[min(len(self.requests) - 1, len(self.scripted) - 1)]

        async def gen():
            for event in events:
                yield event

        return gen()

    def texts(self, index: int) -> list[str]:
        return _request_texts(self.requests[index])


def _request_texts(request: ChatRequest) -> list[str]:
    """Every text block the provider would have received, in order."""

    out: list[str] = []
    for message in request.messages:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            out.append(content)
            continue
        for block in content or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                out.append(text)
    return out


class FakeBridgeClient:
    """The hub's one route this path uses, with no socket in sight."""

    def __init__(self, *, body: str = BODY, sender_kind: str = "agent") -> None:
        self.body = body
        self.sender_kind = sender_kind
        self.fetches: list[str] = []

    async def current_access(self) -> Any:
        return SimpleNamespace(credential_id=7, account_id="acc_1")

    async def conversation_body(
        self,
        conversation_id: str,
        message_id: str,
        *,
        tenant_id: str | None = None,
        endpoint_id: str | None = None,
        access: Any = None,
    ) -> dict[str, Any]:
        self.fetches.append(message_id)
        return {
            "message_id": message_id,
            "body": self.body,
            "body_hash": __import__("hashlib").sha256(self.body.encode("utf-8")).hexdigest(),
            "conversation_name": "#qa",
            "sender_label": "Ada",
            "sender_kind": self.sender_kind,
        }


def _owner() -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=7,
        account_id="acc_1",
        tenant_id=TENANT,
        endpoint_id="ep_1",
        canonical_base_url="https://hub.example",
        enrollment_generation=1,
    )


def _store(config_dir: Path) -> Store:
    store = Store(config_dir)
    store.commit_enrollment(
        _owner(), OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id=SESSION_ID
    )
    store.publish_screening_policy(
        _owner(),
        version="v1",
        template_sha256="ab" * 32,
        default_model="qa-default",
        revision="rev-1",
    )
    return store


def _epoch(store: Store) -> int:
    policy = store.current_screening_policy(_owner())
    assert policy is not None
    return int(policy["epoch"])


def _reserve(store: Store) -> None:
    assert store.record_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        conversation_seq=3,
        mention=True,
        body_sha256=DIGEST,
        screening_fingerprint=FINGERPRINT,
        policy_epoch=_epoch(store),
        consent_version="bridge_consent_v1",
        designated_session=SESSION_ID,
        now=time.time(),
    )


def _runtime(config_dir: Path, client: FakeBridgeClient) -> Any:
    return SimpleNamespace(
        client=client,
        config_dir=config_dir,
        tenant_id=TENANT,
        owner=_owner(),
        session_id=SESSION_ID,
    )


def _session(tmp_path: Path, stream: CapturingStream, runtime: Any, tools=None) -> Session:
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=tools or [],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable", "env"],
        bridge_runtime=runtime,
    )


def _tool_call(call_id: str, name: str) -> list[StreamEvent]:
    events: list[StreamEvent] = [
        StreamToolCallDelta(index=0, id=call_id, name=name, argument_delta="{}"),
        StreamEndEvent(stop_reason="toolUse"),
    ]
    return events


def _row(config_dir: Path) -> dict[str, Any]:
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM body_reservations WHERE message_id = ?", (MESSAGE_ID,)
        ).fetchone()
    assert row is not None
    return dict(row)


def _transcript_text(directory: Path) -> str:
    return "\n".join(path.read_text() for path in sorted(directory.rglob("*")) if path.is_file())


@pytest.mark.asyncio
async def test_the_admitted_body_reaches_every_provider_call_of_the_turn(tmp_path) -> None:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store)
    client = FakeBridgeClient()
    stream = CapturingStream(
        [
            _tool_call("call-1", "noop"),
            [StreamTextDelta(delta="the answer"), StreamEndEvent(stop_reason="stop")],
        ]
    )

    async def execute(tool_call_id, args, signal, on_update, context):
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="noop", content=[TextContent(text="done")]
        )

    session = _session(
        tmp_path,
        stream,
        _runtime(config_dir, client),
        tools=[AgentTool(name="noop", description="a tool", execute=execute)],
    )
    try:
        await session.prompt("hello")
    finally:
        await session.dispose()

    assert len(stream.requests) == 2, "one tool call, then the final answer"
    for index in (0, 1):
        texts = stream.texts(index)
        carrier = [text for text in texts if BODY in text]
        assert len(carrier) == 1, f"call {index}: the body must ride exactly once"
        assert carrier[0].count(BODY) == 1
        assert HEADER_MARKER in carrier[0]
        assert f"tenant_id: {TENANT}" in carrier[0]

    row = _row(config_dir)
    assert row["state"] == "completed", row["last_error"]
    assert row["completed_entry_id"]
    assert _transcript_text(tmp_path / "sess").count(BODY) == 0, "no body in the transcript"


@pytest.mark.asyncio
async def test_a_mid_turn_aside_never_carries_the_frame(tmp_path) -> None:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store)
    client = FakeBridgeClient()
    stream = CapturingStream(
        [
            _tool_call("call-1", "ask_aside"),
            [StreamTextDelta(delta="the side answer"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="the answer"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    holder: dict[str, Any] = {}

    async def execute(tool_call_id, args, signal, on_update, context):
        await holder["session"].complete_aside([Message.user("why?")])
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="ask_aside",
            content=[TextContent(text="asked")],
        )

    session = _session(
        tmp_path,
        stream,
        _runtime(config_dir, client),
        tools=[AgentTool(name="ask_aside", description="a tool", execute=execute)],
    )
    holder["session"] = session
    try:
        await session.prompt("hello")
    finally:
        await session.dispose()

    aside_requests = [
        index
        for index, request in enumerate(stream.requests)
        if any("why?" in text for text in _request_texts(request))
    ]
    assert aside_requests, "the aside must have produced a real provider request"
    for index in aside_requests:
        joined = "\n".join(stream.texts(index))
        assert FRAME_MARKER not in joined, "an aside request must never carry a frame"
        assert BODY not in joined


@pytest.mark.asyncio
async def test_a_withdrawn_consent_stops_the_frame_on_later_calls(tmp_path) -> None:
    """S-10/S-21 in a live turn: the first call carries, the second does not."""

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store)
    client = FakeBridgeClient()
    stream = CapturingStream(
        [
            _tool_call("call-1", "revoke"),
            [StreamTextDelta(delta="the answer"), StreamEndEvent(stop_reason="stop")],
        ]
    )

    async def execute(tool_call_id, args, signal, on_update, context):
        # The real withdrawal path writes a new consent version; the local
        # read then disagrees with the version recorded at admission.
        with sqlite3.connect(state_db_path(config_dir)) as conn:
            conn.execute("UPDATE enrollments SET consent_version = 'bridge_consent_v2'")
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="revoke", content=[TextContent(text="revoked")]
        )

    session = _session(
        tmp_path,
        stream,
        _runtime(config_dir, client),
        tools=[AgentTool(name="revoke", description="a tool", execute=execute)],
    )
    try:
        await session.prompt("hello")
    finally:
        await session.dispose()

    assert len(stream.requests) == 2
    assert BODY in "\n".join(stream.texts(0)), "the first call is still authorised"
    assert BODY not in "\n".join(stream.texts(1)), "a withdrawn consent must stop the frame"

    row = _row(config_dir)
    assert row["state"] == "pending"
    assert row["last_error"] == "admission_drift"
