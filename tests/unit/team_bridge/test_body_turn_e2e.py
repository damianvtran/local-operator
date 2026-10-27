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
5. a turn whose consent is revoked mid-flight sends no frame afterwards;
6. no provider request carries both a person's words from this run and a
   body frame: a person's prompt opens no body, a body turn that has not
   started yields to a waiting person, and a person who steers into a running
   body turn ends it at the next boundary instead of being folded in after the
   frame. A pending body is served by its own turn once the person's turn
   releases the lock, and is re-offered after an abort, a failure or a restart.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.analytics import recorder as analytics_recorder
from local_operator.analytics.recorder import AnalyticsRecorder
from local_operator.analytics.store import AnalyticsStore
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


@pytest.fixture(autouse=True)
def telemetry_off(tmp_path, monkeypatch):
    # Local analytics is a process singleton, not session-owned. Close an
    # isolated recorder BEFORE any turns: its documented closed fence refuses
    # writes and thread starts, unlike suppressing the leak warning afterwards.
    owned = AnalyticsRecorder(store=AnalyticsStore(tmp_path / "analytics.sqlite3"))
    owned.close()
    monkeypatch.setattr(analytics_recorder, "_recorder", owned)
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "false")
    try:
        yield
    finally:
        owned.close()
        assert owned._thread is None, "fixture analytics writer must not survive"


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

    return _message_texts(request.messages)


def _message_texts(messages: Iterable[Message]) -> list[str]:
    """Every text block in ``messages``, in order -- a slice of one request.

    Split from :func:`_request_texts` so the layout helpers can read one
    message or a tail of a request without dressing it up as a request.
    """

    out: list[str] = []
    for message in messages:
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
    # The follow-up waiter re-checks the full authority fence, which reads the
    # feature flag from disk (the one a connector-less unit fixture must write).
    (config_dir / "config.yml").write_text(
        json.dumps({"values": {"team_bridge": {"enabled": True}}}), encoding="utf-8"
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


def _reserve(
    store: Store, message_id: str = MESSAGE_ID, seq: int = 3, config_dir: Path | None = None
) -> None:
    assert store.record_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=message_id,
        conversation_seq=seq,
        mention=True,
        body_sha256=DIGEST,
        screening_fingerprint=FINGERPRINT,
        policy_epoch=_epoch(store),
        consent_version="bridge_consent_v1",
        designated_session=SESSION_ID,
        now=time.time(),
    )
    if config_dir is not None:
        _allow_receipt(config_dir, message_id)


def _allow_receipt(config_dir: Path, message_id: str) -> None:
    """The trusted screening receipt the follow-up waiter's eligibility reads.

    A body served by its OWN turn (the follow-up waiter) is admitted only when
    this machine holds an ``allow`` receipt whose hash matches the reservation:
    ``_pending_body_followup_row`` is the authority, not the reservation alone.
    The real screener writes this row before a body is ever reserved for a wake.
    """

    with sqlite3.connect(state_db_path(config_dir)) as conn:
        owner_key, generation = conn.execute(
            "SELECT owner_key, enrollment_generation FROM enrollments"
        ).fetchone()
        policy_epoch = conn.execute(
            "SELECT policy_epoch FROM body_reservations WHERE message_id = ?", (message_id,)
        ).fetchone()[0]
        conn.execute(
            "INSERT OR REPLACE INTO screening_receipts (owner_key, message_id, body_sha256, "
            "template_sha256, prompt_version, model, verdict, categories, envelope_digest, "
            "admission_revision, policy_epoch, creator_generation, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                owner_key,
                message_id,
                DIGEST,
                "ab" * 32,
                "bridge-screen-v2",
                "qa-default",
                "allow",
                "[]",
                "e" * 64,
                "rev-1",
                policy_epoch,
                generation,
                time.time(),
            ),
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


async def _serve_body_turn(session: Session) -> None:
    """Drive the body through its OWN turn: the follow-up waiter the session arms.

    This is the production path for a body that is not riding a user's prompt.
    The user's turn queues the waiter on the FIFO turn lock; it runs the moment
    that lock is released, so the test awaits the tracked task instead of
    calling a private turn runner.
    """

    session._ensure_pending_body_followup()
    waiter = session._pending_body_followup_task
    assert waiter is not None, "a pending body must have a follow-up waiter"
    await waiter


@pytest.mark.asyncio
async def test_the_admitted_body_reaches_every_provider_call_of_the_turn(tmp_path) -> None:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store, config_dir=config_dir)
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
        await _serve_body_turn(session)
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
    _reserve(store, config_dir=config_dir)
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
        await _serve_body_turn(session)
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
    _reserve(store, config_dir=config_dir)
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
        await _serve_body_turn(session)
    finally:
        await session.dispose()

    assert len(stream.requests) == 2
    assert BODY in "\n".join(stream.texts(0)), "the first call is still authorised"
    assert BODY not in "\n".join(stream.texts(1)), "a withdrawn consent must stop the frame"

    row = _row(config_dir)
    assert row["state"] == "pending"
    assert row["last_error"] == "admission_drift"


def _carrying(stream: CapturingStream) -> list[int]:
    """Indices of the provider requests that carried the body."""

    return [i for i in range(len(stream.requests)) if BODY in "\n".join(stream.texts(i))]


@pytest.mark.asyncio
async def test_a_typed_prompt_never_carries_the_body_and_the_next_turn_does(tmp_path) -> None:
    """The operator's rule: a direct user message is about the user.

    Before this fix the user's own turn claimed the oldest pending reservation
    and appended the frame AFTER their words. Now the prompt's request is
    body-free, the reservation stays pending through it, and the body is served
    by its own follow-up turn that the user's turn queued on the turn lock.
    """

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store, config_dir=config_dir)
    client = FakeBridgeClient()
    stream = CapturingStream([[StreamTextDelta(delta="hi"), StreamEndEvent(stop_reason="stop")]])
    session = _session(tmp_path, stream, _runtime(config_dir, client))
    try:
        await session.prompt("hello")
        assert len(stream.requests) == 1
        assert any("hello" in text for text in stream.texts(0))
        assert _carrying(stream) == [], "the user's own request must not carry a bridge body"
        assert client.fetches == [], "a user turn must not even hydrate the body"
        assert _row(config_dir)["state"] == "pending", "the body is still owed"

        waiter = session._pending_body_followup_task
        assert waiter is not None, "the user's turn must queue the body's own turn"
        await waiter
    finally:
        await session.dispose()

    assert _carrying(stream) == [1], "the following bridge turn carries the body, once"
    assert _row(config_dir)["state"] == "completed"


@pytest.mark.asyncio
async def test_sequential_user_prompts_each_leave_the_body_to_its_own_turn(tmp_path) -> None:
    """Two prompts with the body's turn run in between: user, body 1, user, body 2.

    Sequential, not back to back: the waiter is awaited between the prompts.
    The back-to-back cells (a prompt racing a queued or running body turn) are
    ``test_a_prompt_waiting_on_the_lock_runs_before_a_queued_body_turn`` and
    ``test_a_person_steering_into_a_body_turn_gets_their_own_turn``.
    """

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store, config_dir=config_dir)
    client = FakeBridgeClient()
    stream = CapturingStream([[StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]])
    session = _session(tmp_path, stream, _runtime(config_dir, client))
    try:
        await session.prompt("first")
        waiter = session._pending_body_followup_task
        assert waiter is not None
        await waiter
        assert _carrying(stream) == [1]
        # A second body arrives while the user keeps talking.
        second = "bcmsg_abcdef02"
        _reserve(store, second, 4, config_dir)
        await session.prompt("second")
        assert _carrying(stream) == [1], "the second user prompt must not carry the body"
        second_waiter = session._pending_body_followup_task
        assert second_waiter is not None
        await second_waiter
    finally:
        await session.dispose()

    # user, body 1, user, body 2: each body had its own turn, no user request has one.
    assert len(stream.requests) == 4
    assert _carrying(stream) == [1, 3]
    assert client.fetches == [MESSAGE_ID, second]


@pytest.mark.asyncio
async def test_a_non_user_turn_still_serves_the_pending_body(tmp_path) -> None:
    """The exclusion is only for a person's turn: a peer/wake-class run serves it."""

    from local_operator.harness.types import CustomMessage

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store, config_dir=config_dir)
    client = FakeBridgeClient()
    stream = CapturingStream([[StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]])
    session = _session(tmp_path, stream, _runtime(config_dir, client))
    try:
        await session._prompt_messages(
            [
                CustomMessage(
                    custom_type="peer_message", attribution="user", details={"text": "ping"}
                )
            ],
            carried_prompt=True,
        )
    finally:
        await session.dispose()

    assert _carrying(stream) == [0], "a peer-triggered turn is not a user turn"
    assert _row(config_dir)["state"] == "completed"


@pytest.mark.parametrize(
    ("label", "directed"),
    [
        ("typed_prompt", True),
        ("queued_typed_steer", True),
        ("spooled_source_user_row", True),
        ("ask_answer", True),
        ("queued_ask_answer", True),
        ("spooled_owner_chrome", False),
        ("injected_steer", False),
        ("harness_injected_continuation", False),
        ("wake", False),
        ("monitor", False),
        ("peer", False),
        ("job_result", False),
        ("bridge_followup", False),
    ],
)
@pytest.mark.asyncio
async def test_only_a_persons_own_message_makes_a_turn_user_directed(
    tmp_path, label, directed
) -> None:
    """The exact gate: which run inputs may NOT serve a pending body.

    A typed TUI/desktop/phone prompt arrives as a plain user ``Message``
    (``_run_typed_user``), and so does a person's answer to an ask. A message
    typed while a turn runs, or spooled as ``SOURCE_USER`` by a draining
    runtime (``_run_spooled_owner_prompt``, driven for real here), waits on the
    steering queue. ``lop send`` is NOT in that list: it is a ``peer_message``
    into ``receive_peer_message``, another session speaking, and a peer-opened
    run may carry a body. Everything the harness mints -- spooled owner chrome,
    an injected steer, a goal continuation, wake, monitor, job and the bridge's
    own follow-up -- stays eligible, so non-person work never starves a body.
    """

    from local_operator.harness.message_types import (
        ASK_RESPONSE_MESSAGE_TYPE,
        TEAM_BRIDGE_MESSAGE_TYPE,
    )
    from local_operator.harness.types import CustomMessage
    from local_operator.harness.wake import WAKE_PROMPT_MESSAGE_TYPE
    from local_operator.monitors.spec import MONITOR_PROMPT_MESSAGE_TYPE
    from local_operator.session.runtime.inbox import SOURCE_USER, InboxLine

    def answered() -> CustomMessage:
        return CustomMessage(
            custom_type=ASK_RESPONSE_MESSAGE_TYPE,
            attribution="user",
            details={"ask_id": "a-1", "status": "answered", "answers": {"q": ["yes"]}},
        )

    session = _session(
        tmp_path,
        CapturingStream([[StreamEndEvent(stop_reason="stop")]]),
        _runtime(tmp_path, FakeBridgeClient()),
    )
    session._run_triggers = set()
    session._run_typed_user = False
    # A live turn is "streaming": the steer arms queue rather than spawning
    # their own idle wake, which is the mid-turn shape this table is about.
    session._is_streaming = True
    if label == "typed_prompt":
        session._note_run_input(Message.user("hello"))
    elif label == "queued_typed_steer":
        session.steer("hello", [])
    elif label in ("spooled_source_user_row", "spooled_owner_chrome"):
        session._run_spooled_owner_prompt(
            InboxLine(
                text="hello",
                sender={},
                source=SOURCE_USER,
                command_id=f"cmd-{label}",
                harness_injected=label == "spooled_owner_chrome",
            ),
            seen=set(),
        )
    elif label == "ask_answer":
        session._note_run_input(answered())
    elif label == "queued_ask_answer":
        session._steering_queue.put_nowait(answered())
    elif label == "injected_steer":
        session.steer("continue", [], harness_injected=True)
    elif label == "harness_injected_continuation":
        injected = Message.user("continue")
        injected.provider_payload = {"harness_injected": True}
        session._note_run_input(injected)
    else:
        custom_type = {
            "wake": WAKE_PROMPT_MESSAGE_TYPE,
            "monitor": MONITOR_PROMPT_MESSAGE_TYPE,
            "peer": "peer_message",
            "job_result": "job_result",
            "bridge_followup": TEAM_BRIDGE_MESSAGE_TYPE,
        }[label]
        session._note_run_input(
            CustomMessage(custom_type=custom_type, attribution="user", details={"text": "x"})
        )
    try:
        assert session._turn_is_user_directed() is directed
    finally:
        session._is_streaming = False
        await session.dispose()


# -- MAJOR-1 / D-Q1 / D-F2 cells (agent review round 1 remediation) ----------


def _hook_tool(on_run) -> AgentTool:
    """A tool whose execution runs ``on_run`` -- the moment a person acts mid-turn."""

    async def execute(tool_call_id, args, signal, on_update, context):
        result = on_run()
        if hasattr(result, "__await__"):
            await result
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="hook", content=[TextContent(text="done")]
        )

    return AgentTool(name="hook", description="a tool", execute=execute)


# -- history-prefixed variants (MAJOR-R4-1) ------------------------------------
#
# A body-only turn opens with no new messages and the body rides the request
# transform, not the context, so the live tail at its first boundaries is
# whatever the session already held. Real sessions almost always hold history;
# a fresh context hid a latch that fired on the PREVIOUS turn's answer.

HISTORY = ["fresh", "after_answer", "after_no_reply"]


def _seed_history(session: Session, history: str) -> None:
    """Give the session a prior exchange ending the way ``history`` names."""

    from local_operator.harness.types import QUIET_TURN_KEY, ToolCall

    if history == "fresh":
        return
    prior: list[Any] = [Message.user("PRIOR-HELLO")]
    if history == "after_answer":
        prior.append(Message(role="assistant", content=[TextContent(text="PRIOR-HI")]))
    else:
        prior.append(
            Message(
                role="assistant",
                content=[],
                tool_calls=[ToolCall(id="prior-q", name="no_reply", arguments={})],
            )
        )
        prior.append(
            Message.tool_result(
                ToolResult(
                    tool_call_id="prior-q",
                    tool_name="no_reply",
                    content=[TextContent(text="Quiet.")],
                    details={QUIET_TURN_KEY: True},
                )
            )
        )
    session._context.messages.extend(prior)
    assert session._context_tail_is_terminal(), "the seeded history must end terminal"


def _layout(stream: CapturingStream) -> str:
    """One line per request: role and a short tag per text block, the body masked."""

    rows = []
    for i, request in enumerate(stream.requests):
        cells = []
        for message in request.messages:
            role = getattr(message, "role", "?")
            for text in _message_texts([message]):
                tag = "BODY" if BODY in text else text[:28].replace("\n", " ")
                cells.append(f"{role}:{tag}")
        rows.append(f"{i}: " + " | ".join(cells))
    return "\n".join(rows)


def _carries_both(stream: CapturingStream, words: str) -> list[int]:
    """Requests where the person's words are NEW input beside a body: must be empty.

    "New" means after the request's last assistant message -- not yet answered.
    That is exactly the fold-in shape (``tool result | person | frame``) and the
    old prompt shape (``person | frame``). A later body turn that holds the
    person's ANSWERED words as history is legitimate and not counted.
    """

    hits = []
    for i, request in enumerate(stream.requests):
        if BODY not in "\n".join(stream.texts(i)):
            continue
        messages = list(request.messages)
        last_assistant = max(
            (j for j, m in enumerate(messages) if getattr(m, "role", None) == "assistant"),
            default=-1,
        )
        fresh = _message_texts(messages[last_assistant + 1 :])
        if any(words in text for text in fresh):
            hits.append(i)
    return hits


async def _settle(session: Session) -> None:
    """Await every turn the session queued until it is idle and nothing re-arms."""

    for _ in range(50):
        pending = [task for task in session._background_tasks if not task.done()]
        waiter = session._pending_body_followup_task
        if waiter is not None and not waiter.done():
            pending.append(waiter)
        if not pending and not session._turn_lock.locked():
            return
        await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.sleep(0)
    raise AssertionError("session never settled")


def _steer_typed(session: Session) -> None:
    session.steer("typed mid-body words", [])


def _steer_spooled(session: Session) -> None:
    from local_operator.session.runtime.inbox import SOURCE_USER, InboxLine

    session._run_spooled_owner_prompt(
        InboxLine(
            text="typed mid-body words", sender={}, source=SOURCE_USER, command_id="cmd-spool-1"
        ),
        seen=set(),
    )


@pytest.mark.parametrize("history", HISTORY)
@pytest.mark.parametrize("steer", [_steer_typed, _steer_spooled], ids=["tui_steer", "spooled"])
@pytest.mark.asyncio
async def test_a_person_steering_into_a_body_turn_gets_their_own_turn(
    tmp_path, steer, history
) -> None:
    """MAJOR-1(b): a person's steer is never folded in after the frame.

    The body turn ends at its next boundary (aborted, so ``settle`` releases the
    row); the person's row opens its own turn with no body; the body is then
    re-served by a fresh pass armed at that turn's release.
    """

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store, config_dir=config_dir)
    client = FakeBridgeClient()
    stream = CapturingStream(
        [
            _tool_call("call-1", "hook"),
            [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    holder: dict[str, Session] = {}
    fired: list[bool] = []

    def on_run() -> None:
        if not fired:
            fired.append(True)
            steer(holder["session"])

    session = _session(tmp_path, stream, _runtime(config_dir, client), tools=[_hook_tool(on_run)])
    holder["session"] = session
    _seed_history(session, history)
    try:
        await _serve_body_turn(session)
        await _settle(session)
    finally:
        await session.dispose()

    words = "typed mid-body words"
    assert _carries_both(stream, words) == [], _layout(stream)
    person = [i for i in range(len(stream.requests)) if words in "\n".join(stream.texts(i))]
    assert person and person[0] == 1, "the person's row opens the very next turn"
    assert _carrying(stream)[0] == 0 and 1 not in _carrying(stream)
    assert _row(config_dir)["state"] == "completed", "the yielded body is re-served"
    assert client.fetches == [MESSAGE_ID, MESSAGE_ID]


@pytest.mark.asyncio
async def test_a_harness_injected_steer_still_rides_the_body_turn(tmp_path) -> None:
    """MINOR-1: spooled owner chrome is no person; the body turn is not cut."""

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store, config_dir=config_dir)
    client = FakeBridgeClient()
    stream = CapturingStream(
        [
            _tool_call("call-1", "hook"),
            [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    holder: dict[str, Session] = {}
    session = _session(
        tmp_path,
        stream,
        _runtime(config_dir, client),
        tools=[
            _hook_tool(lambda: holder["session"].steer("chrome row", [], harness_injected=True))
        ],
    )
    holder["session"] = session
    try:
        await _serve_body_turn(session)
        await _settle(session)
    finally:
        await session.dispose()

    assert _carrying(stream) == [0, 1], "the body turn ran to its end"
    assert any("chrome row" in text for text in stream.texts(1))
    assert client.fetches == [MESSAGE_ID]
    assert _row(config_dir)["state"] == "completed"


@pytest.mark.asyncio
async def test_a_prompt_waiting_on_the_lock_runs_before_a_queued_body_turn(tmp_path) -> None:
    """MAJOR-1(a) / D-F2: the follow-up waiter is QUEUED when a desktop prompt arrives.

    The real serving handle waits out a foreign turn, so the body waiter sits
    ahead of the person on the FIFO lock. The waiter yields to the waiting
    person without hydrating; the person's turn runs first, then the body.
    """

    from local_operator.session.runtime.serving import ServingSessionHandle

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store, config_dir=config_dir)
    client = FakeBridgeClient()
    stream = CapturingStream([[StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]])
    session = _session(tmp_path, stream, _runtime(config_dir, client))
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), cwd=str(tmp_path), install_gates=False
    )
    # The handle titles a new conversation with a side request through the
    # same stream; only turn requests are under test here.
    handle._name_requested = True
    try:
        # A foreign turn (a job delivery, a wake) holds the lock.
        await session._turn_lock.acquire()
        session._ensure_pending_body_followup()
        await asyncio.sleep(0)
        person = asyncio.ensure_future(handle.prompt("desktop words", wait_complete=True))
        for _ in range(5):
            await asyncio.sleep(0)
        assert session._person_is_waiting(), "the waiting desktop prompt is visible"
        session._turn_lock.release()
        await person
        await _settle(session)
    finally:
        await handle.dispose()
        await session.dispose()

    assert len(stream.requests) == 2, _layout(stream)
    assert any("desktop words" in text for text in stream.texts(0)), "the person goes first"
    assert _carrying(stream) == [1], "then the body, in its own turn"
    assert _carries_both(stream, "desktop words") == []
    assert _row(config_dir)["state"] == "completed"


@pytest.mark.asyncio
async def test_a_prompt_during_a_running_body_turn_waits_for_that_turn_only(tmp_path) -> None:
    """The bound: current body turn, then the person, then the rest (D-F2).

    Two bodies pending, the desktop prompt arrives while body 1's turn is in a
    tool. Body 1's turn is NOT cut (a waiting prompt is not a steer), but body 2
    no longer starts ahead of the person: body1, USER, body2 -- not body1,
    body2, USER.
    """

    from local_operator.session.runtime.serving import ServingSessionHandle

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store, config_dir=config_dir)
    second = "bcmsg_abcdef02"
    _reserve(store, second, 4, config_dir)
    client = FakeBridgeClient()
    stream = CapturingStream(
        [
            _tool_call("call-1", "hook"),
            [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    started: list[asyncio.Future[str]] = []
    holder: dict[str, Any] = {}

    def on_run() -> None:
        if not started:
            started.append(
                asyncio.ensure_future(holder["handle"].prompt("desktop words", wait_complete=True))
            )

    session = _session(tmp_path, stream, _runtime(config_dir, client), tools=[_hook_tool(on_run)])
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), cwd=str(tmp_path), install_gates=False
    )
    handle._name_requested = True  # no title side request through the stream
    holder["handle"] = handle
    try:
        await _serve_body_turn(session)
        assert started, "the prompt arrived mid-turn"
        await started[0]
        await _settle(session)
    finally:
        await handle.dispose()
        await session.dispose()

    person = [
        i for i in range(len(stream.requests)) if "desktop words" in "\n".join(stream.texts(i))
    ]
    assert person[0] == 2, _layout(stream)
    assert _carrying(stream)[:2] == [0, 1]
    assert _carrying(stream)[2:] == [3], "body 2 runs after the person"
    assert client.fetches == [MESSAGE_ID, second]
    assert _carries_both(stream, "desktop words") == []


@pytest.mark.asyncio
async def test_an_idle_user_steer_opens_a_body_free_turn(tmp_path) -> None:
    """D-Q1: the idle steer wake has an empty ``initial`` and still excludes the body."""

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store, config_dir=config_dir)
    client = FakeBridgeClient()
    stream = CapturingStream([[StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]])
    session = _session(tmp_path, stream, _runtime(config_dir, client))
    try:
        session.steer("idle words", [])
        await _settle(session)
    finally:
        await session.dispose()

    assert any("idle words" in text for text in stream.texts(0))
    assert _carrying(stream) == [1]
    assert _row(config_dir)["state"] == "completed"


@pytest.mark.asyncio
async def test_a_post_compaction_continuation_inherits_the_users_verdict(tmp_path) -> None:
    """D-Q1: ``_drain_continuation`` runs inside the person's pipeline, body-free."""

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store, config_dir=config_dir)
    client = FakeBridgeClient()
    stream = CapturingStream(
        [
            _tool_call("call-1", "hook"),
            [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    holder: dict[str, Session] = {}
    session = _session(
        tmp_path,
        stream,
        _runtime(config_dir, client),
        tools=[
            _hook_tool(
                lambda: holder["session"]._continuation_queue.append(
                    Message.user("CONTINUE-AFTER-COMPACTION")
                )
            )
        ],
    )
    holder["session"] = session
    try:
        await session.prompt("person words")
        await _settle(session)
    finally:
        await session.dispose()

    cont = [
        i
        for i in range(len(stream.requests))
        if "CONTINUE-AFTER-COMPACTION" in "\n".join(stream.texts(i))
    ]
    assert cont, "the continuation ran"
    assert cont[0] not in _carrying(stream), "the continuation's own request is body-free"
    assert _carrying(stream) == [len(stream.requests) - 1], "the body gets its own last turn"
    assert _carries_both(stream, "CONTINUE-AFTER-COMPACTION") == []
    assert _row(config_dir)["state"] == "completed"


@pytest.mark.asyncio
async def test_a_goal_loop_continuation_may_serve_the_body(tmp_path) -> None:
    """Harness-minted ``'user'`` rows are no person: deliberately NOT excluded."""

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store, config_dir=config_dir)
    client = FakeBridgeClient()
    stream = CapturingStream([[StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]])
    session = _session(tmp_path, stream, _runtime(config_dir, client))
    try:
        await session.prompt("goal: keep going", harness_injected=True)
    finally:
        await session.dispose()

    assert _carrying(stream) == [0]
    assert _row(config_dir)["state"] == "completed"


@pytest.mark.asyncio
async def test_an_aborted_body_waits_while_idle_and_the_next_turn_rearms_it(tmp_path) -> None:
    """D-F2: Esc on a body turn spends nothing while idle; the next turn's end re-arms."""

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store, config_dir=config_dir)
    client = FakeBridgeClient()
    stream = CapturingStream(
        [
            _tool_call("call-1", "hook"),
            [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    holder: dict[str, Session] = {}
    fired: list[bool] = []

    def on_run() -> None:
        if not fired:
            fired.append(True)
            holder["session"].abort()

    session = _session(tmp_path, stream, _runtime(config_dir, client), tools=[_hook_tool(on_run)])
    holder["session"] = session
    try:
        await _serve_body_turn(session)
        await _settle(session)
        assert len(stream.requests) == 1, "an Esc'd body is not re-served while idle"
        assert _row(config_dir)["state"] == "pending"
        await session.prompt("next words")
        await _settle(session)
    finally:
        await session.dispose()

    assert any("next words" in text for text in stream.texts(1))
    assert 1 not in _carrying(stream)
    assert _carrying(stream)[-1] == len(stream.requests) - 1
    assert _row(config_dir)["state"] == "completed"


@pytest.mark.asyncio
async def test_a_body_left_pending_by_a_restart_is_offered_at_bind(tmp_path) -> None:
    """D-F2: no arrival is coming after a restart; ``async_init`` arms the pass."""

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store, config_dir=config_dir)
    client = FakeBridgeClient()
    stream = CapturingStream([[StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]])
    session = _session(tmp_path, stream, _runtime(config_dir, client))
    try:
        await session.async_init()
        assert session._pending_body_followup_task is not None
        await _settle(session)
    finally:
        await session.dispose()

    assert _carrying(stream) == [0]
    assert _row(config_dir)["state"] == "completed"


class FlakyBridgeClient(FakeBridgeClient):
    """Fails the first body fetch, then serves it -- a hub hiccup."""

    async def conversation_body(self, conversation_id, message_id, **kwargs):
        if not self.fetches:
            self.fetches.append(message_id)
            raise RuntimeError("hub unavailable")
        return await super().conversation_body(conversation_id, message_id, **kwargs)


@pytest.mark.asyncio
async def test_a_once_failed_body_is_retried_after_the_next_turn(tmp_path) -> None:
    """D-F2: a failed attempt is excluded for its pass; the next turn's end retries it."""

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store, config_dir=config_dir)
    client = FlakyBridgeClient()
    stream = CapturingStream([[StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]])
    session = _session(tmp_path, stream, _runtime(config_dir, client))
    try:
        await _serve_body_turn(session)
        assert stream.requests == [], "a refused hydration costs no request"
        assert _row(config_dir)["state"] == "pending"
        await session.prompt("next words")
        await _settle(session)
    finally:
        await session.dispose()

    assert _carrying(stream) == [1]
    assert client.fetches == [MESSAGE_ID, MESSAGE_ID]
    assert _row(config_dir)["state"] == "completed"


@pytest.mark.asyncio
async def test_a_persons_ask_answer_turn_is_body_free(tmp_path) -> None:
    """MINOR-2: the real idle ask-answer delivery opens a person's turn, no body."""

    from local_operator.harness.message_types import ASK_RESPONSE_MESSAGE_TYPE
    from local_operator.harness.types import CustomMessage

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store, config_dir=config_dir)
    client = FakeBridgeClient()
    stream = CapturingStream([[StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]])
    session = _session(tmp_path, stream, _runtime(config_dir, client))
    try:
        await session.deliver_ask_messages(
            [
                CustomMessage(
                    custom_type=ASK_RESPONSE_MESSAGE_TYPE,
                    attribution="user",
                    details={
                        "ask_id": "a-1",
                        "status": "answered",
                        "questions": [{"id": "q", "question": "Ship it?"}],
                        "answers": {"q": ["ANSWER-YES-SHIP"]},
                    },
                )
            ]
        )
        await _settle(session)
    finally:
        await session.dispose()

    assert len(stream.requests) == 2, _layout(stream)
    assert BODY not in "\n".join(stream.texts(0)), "the person's answer turn carried a body"
    assert _carrying(stream) == [1]
    assert _row(config_dir)["state"] == "completed"


# -- MAJOR-R2-1 cells (agent review round 2 remediation) ----------------------


class SteeringStream(CapturingStream):
    """A ``CapturingStream`` that runs ``during`` INSIDE the answer of request ``at``.

    Probe C's shape: the model's events for that request are being streamed when
    the person acts, so the steer lands after the provider has already answered
    and before the loop reaches its post-response boundary.
    """

    def __init__(self, scripted: list[list[StreamEvent]], *, at: int, during) -> None:
        super().__init__(scripted)
        self.at = at
        self.during = during

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        index = len(self.requests) - 1
        events = self.scripted[min(index, len(self.scripted) - 1)]

        async def gen():
            for position, event in enumerate(events):
                if index == self.at and position == len(events) - 1:
                    self.during()
                yield event

        return gen()


def _record_settles(monkeypatch) -> list[tuple[dict[str, Any], str]]:
    """Every ``PendingTurn.settle`` call: its verdict and the outcome it returned."""

    from local_operator.team_bridge.pending_turn import PendingTurn

    seen: list[tuple[dict[str, Any], str]] = []
    original = PendingTurn.settle

    def spy(self, **kwargs: Any) -> str:
        outcome = original(self, **kwargs)
        seen.append((kwargs, outcome))
        return outcome

    monkeypatch.setattr(PendingTurn, "settle", spy)
    return seen


@pytest.mark.parametrize("history", HISTORY)
@pytest.mark.asyncio
async def test_a_steer_during_the_final_answer_does_not_re_serve_the_body(
    tmp_path, monkeypatch, history
) -> None:
    """MAJOR-R2-1, probe C: the body was answered, so the turn ends normally.

    A person steering while the body turn's FINAL answer streams must not cut a
    turn that already finished: no further request would carry the frame, so
    there is nothing left to protect. Cutting it released the row and re-served
    the body (answered twice). The person's row is held off the body turn and
    gets its own turn from the turn-end flush.
    """

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store, config_dir=config_dir)
    client = FakeBridgeClient()
    settles = _record_settles(monkeypatch)
    holder: dict[str, Session] = {}
    stream = SteeringStream(
        [[StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]],
        at=0,
        during=lambda: holder["session"].steer("PERSON-C", []),
    )
    session = _session(tmp_path, stream, _runtime(config_dir, client))
    holder["session"] = session
    _seed_history(session, history)
    try:
        await _serve_body_turn(session)
        await _settle(session)
    finally:
        await session.dispose()

    assert client.fetches == [MESSAGE_ID], _layout(stream)
    assert settles, "the body turn settled"
    verdict, outcome = settles[0]
    assert verdict["aborted"] is False and outcome == "completed", settles
    assert _row(config_dir)["state"] == "completed"
    assert _carries_both(stream, "PERSON-C") == [], _layout(stream)
    assert _carrying(stream)[0] == 0
    person = [i for i in range(len(stream.requests)) if "PERSON-C" in "\n".join(stream.texts(i))]
    assert person == [1], "the person's held row opens its own turn next"
    assert BODY not in "\n".join(stream.texts(1)), "the person's turn carried the body"


@pytest.mark.asyncio
async def test_a_steer_during_the_final_answer_never_repeats_an_earlier_side_effect(
    tmp_path, monkeypatch
) -> None:
    """MAJOR-R2-1: the tool a body turn already ran is not run a second time.

    The turn ran a side-effecting tool (a bridge ``reply`` in production), then
    the person steered while the final answer streamed. Re-serving the body
    would make a fresh model decision and repeat the effect.
    """

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store, config_dir=config_dir)
    client = FakeBridgeClient()
    settles = _record_settles(monkeypatch)
    effects: list[str] = []
    holder: dict[str, Session] = {}
    # Scripted so a RE-SERVED body turn (request 3) would call the tool again:
    # the fix must leave ``effects`` at one, not merely the fetch count.
    answer: list[StreamEvent] = [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]
    stream = SteeringStream(
        [_tool_call("call-1", "hook"), answer, answer, _tool_call("call-2", "hook"), answer],
        at=1,
        during=lambda: holder["session"].steer("PERSON-C", []),
    )
    session = _session(
        tmp_path,
        stream,
        _runtime(config_dir, client),
        tools=[_hook_tool(lambda: effects.append("reply sent"))],
    )
    holder["session"] = session
    try:
        await _serve_body_turn(session)
        await _settle(session)
    finally:
        await session.dispose()

    assert effects == ["reply sent"], _layout(stream)
    assert client.fetches == [MESSAGE_ID]
    assert [outcome for _, outcome in settles][:1] == ["completed"], settles
    assert settles[0][0]["aborted"] is False
    assert _carrying(stream) == [0, 1], "the body turn ran to its answer"
    assert _carries_both(stream, "PERSON-C") == [], _layout(stream)
    assert BODY not in "\n".join(stream.texts(2)), "the person's turn carried the body"
    assert "PERSON-C" in "\n".join(stream.texts(2))


# -- Q-R3-1 cells (QA round 3 remediation) ------------------------------------
#
# The other affirmative way a body turn ends: the model calls ``no_reply``. A
# person row queued while it does is HELD for its own turn (``_drain_steering``),
# so it must not make ``no_reply`` refuse: a refused quiet end is not terminal,
# the cut then fired on a finished turn, and the body was served twice.

REFUSAL = "A person asked this turn"


def _no_reply_call(call_id: str) -> list[StreamEvent]:
    return _tool_call(call_id, "no_reply")


def _assert_no_reply_mounted(session: Session) -> None:
    # The REAL builtin, through the session's own quiet-end door: a spliced
    # stand-in would prove nothing about the refusal seam under test.
    assert any(tool.name == "no_reply" for tool in session._tools)


def _steer_spooled_c(session: Session) -> None:
    from local_operator.session.runtime.inbox import SOURCE_USER, InboxLine

    session._run_spooled_owner_prompt(
        InboxLine(text="PERSON-C", sender={}, source=SOURCE_USER, command_id="cmd-spool-r3"),
        seen=set(),
    )


@pytest.mark.parametrize(
    "steer",
    [lambda session: session.steer("PERSON-C", []), _steer_spooled_c],
    ids=["R3b_tui_steer", "R3d_spooled"],
)
@pytest.mark.parametrize("history", HISTORY)
@pytest.mark.asyncio
async def test_a_body_turn_ending_on_no_reply_is_not_re_served_for_a_queued_person(
    tmp_path, monkeypatch, steer, history
) -> None:
    """Q-R3-1 (R3b, R3d): ``no_reply`` ends the body turn; the person goes next."""

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store, config_dir=config_dir)
    client = FakeBridgeClient()
    settles = _record_settles(monkeypatch)
    holder: dict[str, Session] = {}
    answer: list[StreamEvent] = [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]
    # A re-served body turn (request 2) would call ``no_reply`` again.
    stream = SteeringStream(
        [_no_reply_call("q-1"), answer, _no_reply_call("q-2"), answer],
        at=0,
        during=lambda: steer(holder["session"]),
    )
    session = _session(tmp_path, stream, _runtime(config_dir, client))
    holder["session"] = session
    _seed_history(session, history)
    _assert_no_reply_mounted(session)
    try:
        await _serve_body_turn(session)
        await _settle(session)
    finally:
        await session.dispose()

    assert client.fetches == [MESSAGE_ID], _layout(stream)
    verdict, outcome = settles[0]
    assert verdict["aborted"] is False and outcome == "completed", settles
    assert _row(config_dir)["state"] == "completed"
    assert _carrying(stream) == [0], _layout(stream)
    assert _carries_both(stream, "PERSON-C") == [], _layout(stream)
    assert "PERSON-C" in "\n".join(stream.texts(1)), "the person's held row opens its own turn"
    for index in range(len(stream.requests)):
        assert REFUSAL not in "\n".join(stream.texts(index)), _layout(stream)


@pytest.mark.parametrize("history", HISTORY)
@pytest.mark.asyncio
async def test_a_body_turn_ending_on_no_reply_never_repeats_an_earlier_side_effect(
    tmp_path, monkeypatch, history
) -> None:
    """Q-R3-1 (R3e): a side-effecting tool, then ``no_reply`` with a typed steer queued."""

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store, config_dir=config_dir)
    client = FakeBridgeClient()
    settles = _record_settles(monkeypatch)
    effects: list[str] = []
    holder: dict[str, Session] = {}
    answer: list[StreamEvent] = [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]
    # Scripted so a re-served body turn (request 3) runs the tool again.
    stream = SteeringStream(
        [
            _tool_call("call-1", "hook"),
            _no_reply_call("q-1"),
            answer,
            _tool_call("call-2", "hook"),
            _no_reply_call("q-2"),
            answer,
        ],
        at=1,
        during=lambda: holder["session"].steer("PERSON-C", []),
    )
    session = _session(
        tmp_path,
        stream,
        _runtime(config_dir, client),
        tools=[_hook_tool(lambda: effects.append("reply sent"))],
    )
    holder["session"] = session
    _seed_history(session, history)
    _assert_no_reply_mounted(session)
    try:
        await _serve_body_turn(session)
        await _settle(session)
    finally:
        await session.dispose()

    assert effects == ["reply sent"], _layout(stream)
    assert client.fetches == [MESSAGE_ID]
    assert settles[0][0]["aborted"] is False and settles[0][1] == "completed", settles
    assert _carrying(stream) == [0, 1], _layout(stream)
    assert _carries_both(stream, "PERSON-C") == [], _layout(stream)
    assert "PERSON-C" in "\n".join(stream.texts(2))
    for index in range(len(stream.requests)):
        assert REFUSAL not in "\n".join(stream.texts(index)), _layout(stream)


@pytest.mark.asyncio
async def test_a_non_body_turn_still_refuses_no_reply_for_a_queued_person(tmp_path) -> None:
    """The Q-R3-1 exemption is body-turn-only: elsewhere a queued person still refuses.

    With no body pending nothing holds the person's row off the turn: it is
    folded into this turn's next request, so this turn IS the one they wait
    on and the quiet end must stay refused (docs/design/quiet-turns.md §4).
    """

    from local_operator.harness.types import CustomMessage

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    _store(config_dir)
    client = FakeBridgeClient()
    holder: dict[str, Session] = {}
    stream = SteeringStream(
        [
            _no_reply_call("q-1"),
            [StreamTextDelta(delta="answered"), StreamEndEvent(stop_reason="stop")],
        ],
        at=0,
        during=lambda: holder["session"].steer("PERSON-C", []),
    )
    session = _session(tmp_path, stream, _runtime(config_dir, client))
    holder["session"] = session
    _assert_no_reply_mounted(session)
    try:
        await session._prompt_messages(
            [
                CustomMessage(
                    custom_type="peer_message", attribution="user", details={"text": "ping"}
                )
            ],
            carried_prompt=True,
        )
        await _settle(session)
    finally:
        await session.dispose()

    assert client.fetches == []
    assert len(stream.requests) == 2, _layout(stream)
    second = "\n".join(stream.texts(1))
    assert REFUSAL in second and "PERSON-C" in second, _layout(stream)


@pytest.mark.asyncio
async def test_an_answered_body_is_not_cut_by_a_later_yield_edge_re_entry(
    tmp_path, monkeypatch
) -> None:
    """MINOR-R3-1: a hub aside re-entering after the answer must not cut the turn.

    The steer and the aside both land while the final answer streams; the
    yield edge re-enters for the aside, and that later call's boundary saw a
    non-terminal tail and cut a turn whose body was already answered.
    """

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    _reserve(store, config_dir=config_dir)
    client = FakeBridgeClient()
    settles = _record_settles(monkeypatch)
    effects: list[str] = []
    holder: dict[str, Session] = {}

    def both() -> None:
        session = holder["session"]
        session.steer("PERSON-C", [])
        session.queue_aside(lambda: Message.user("HUB-ASIDE"))

    answer: list[StreamEvent] = [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]
    stream = SteeringStream(
        [
            _tool_call("call-1", "hook"),
            answer,
            answer,
            answer,
            _tool_call("call-2", "hook"),
            answer,
        ],
        at=1,
        during=both,
    )
    session = _session(
        tmp_path,
        stream,
        _runtime(config_dir, client),
        tools=[_hook_tool(lambda: effects.append("reply sent"))],
    )
    holder["session"] = session
    try:
        await _serve_body_turn(session)
        await _settle(session)
    finally:
        await session.dispose()

    assert effects == ["reply sent"], _layout(stream)
    assert client.fetches == [MESSAGE_ID], _layout(stream)
    assert settles[0][0]["aborted"] is False and settles[0][1] == "completed", settles
    assert any("HUB-ASIDE" in "\n".join(stream.texts(i)) for i in range(len(stream.requests)))
    assert _carries_both(stream, "PERSON-C") == [], _layout(stream)


@pytest.mark.asyncio
async def test_a_body_turn_after_a_real_prior_turn_still_yields_mid_work(tmp_path) -> None:
    """MAJOR-R4-1, the reviewer's probe: real history, then a mid-work steer.

    A real ``prompt`` turn answers first (its answer is the context's tail
    when the body turn opens), then the body turn runs two tool calls and the
    person steers in the first. The body turn must be cut after that first
    tool, not run its whole chain with the person held.
    """

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    store = _store(config_dir)
    client = FakeBridgeClient()
    answer: list[StreamEvent] = [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]
    stream = CapturingStream(
        [
            [StreamTextDelta(delta="PRIOR-HI"), StreamEndEvent(stop_reason="stop")],
            _tool_call("call-1", "hook"),
            _tool_call("call-2", "hook"),
            answer,
        ]
    )
    holder: dict[str, Session] = {}
    runs: list[int] = []

    def on_run() -> None:
        runs.append(1)
        if len(runs) == 1:
            _steer_typed(holder["session"])

    session = _session(tmp_path, stream, _runtime(config_dir, client), tools=[_hook_tool(on_run)])
    holder["session"] = session
    try:
        await session.prompt("PRIOR-HELLO")
        await _settle(session)
        assert len(stream.requests) == 1
        _reserve(store, config_dir=config_dir)
        await _serve_body_turn(session)
        await _settle(session)
    finally:
        await session.dispose()

    words = "typed mid-body words"
    assert _carries_both(stream, words) == [], _layout(stream)
    person = [i for i in range(len(stream.requests)) if words in "\n".join(stream.texts(i))]
    # Request 1 is the body turn's first call; it runs call-1, the person
    # steers there, and the cut makes request 2 the person's own turn.
    assert person and person[0] == 2, _layout(stream)
    assert 2 not in _carrying(stream), _layout(stream)
    assert _row(config_dir)["state"] == "completed", "the yielded body is re-served"
