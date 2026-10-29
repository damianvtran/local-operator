"""Anthropic cache diagnostics: body shaping, stream parse, chain ownership,
fail-closed latch, and the ledger columns that keep the verdict.

The API compares a request with the one that produced ``previous_message_id``
and says on ``message_start`` WHERE the prompt prefix diverged
(https://platform.claude.com/docs/en/build-with-claude/cache-diagnostics).
These tests pin the four properties that make that telemetry trustworthy:
the chain is per-conversation, only turns join it, a refusal costs one retry
for the whole process, and the verdict reaches ``calls`` unchanged.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sqlite3
import time
from typing import Any

import httpx
import pytest

from local_operator.analytics.model import CallSnapshot
from local_operator.analytics.store import AnalyticsStore
from local_operator.harness.types import (
    ChatRequest,
    Message,
    ModelSpec,
    StreamStartEvent,
    Usage,
)
from local_operator.providers import clients
from local_operator.providers.clients import AnthropicClient


@pytest.fixture(autouse=True)
def _reset_latch():
    clients._ANTHROPIC_DIAGNOSTICS_REFUSED[0] = False
    yield
    clients._ANTHROPIC_DIAGNOSTICS_REFUSED[0] = False


def _sse(message: dict[str, Any], *, cache_write: int = 0) -> bytes:
    events = [
        (
            "message_start",
            {
                "message": {
                    "usage": {
                        "input_tokens": 5,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": cache_write,
                    },
                    **message,
                }
            },
        ),
        ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
        ("message_stop", {}),
    ]
    return "".join(
        f"event: {name}\ndata: {json.dumps({'type': name, **payload})}\n\n"
        for name, payload in events
    ).encode()


def _request(**overrides: Any) -> ChatRequest:
    return ChatRequest(
        model=ModelSpec(provider="anthropic", model_id="claude-haiku-4-5"),
        system_blocks=["stable"],
        messages=[Message.user("hi")],
        **overrides,
    )


def _client(handler) -> AnthropicClient:
    return AnthropicClient(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def _collect(stream: Any) -> list[Any]:
    return [event async for event in stream]


def _start(events: list[Any]) -> StreamStartEvent:
    return next(event for event in events if isinstance(event, StreamStartEvent))


# -- body shaping ------------------------------------------------------------


def test_an_undiagnosed_request_sends_no_diagnostics_key() -> None:
    """Default OFF: helpers, scripts and other conversations' bodies are unchanged."""
    assert "diagnostics" not in AnthropicClient()._build_body(_request())


def test_the_first_turn_opts_in_with_an_explicit_null() -> None:
    body = AnthropicClient()._build_body(_request(cache_diagnostics=True))
    assert body["diagnostics"] == {"previous_message_id": None}
    assert json.dumps(body["diagnostics"]) == '{"previous_message_id": null}'


def test_a_later_turn_names_the_previous_response() -> None:
    body = AnthropicClient()._build_body(
        _request(cache_diagnostics=True, cache_diagnostics_previous_id="msg_prev")
    )
    assert body["diagnostics"] == {"previous_message_id": "msg_prev"}


def test_a_non_first_party_endpoint_never_sends_it() -> None:
    """Bedrock/Vertex/relays do not offer the feature; an unknown key 400s there."""
    client = AnthropicClient(base_url="https://relay.example.com")
    assert "diagnostics" not in client._build_body(_request(cache_diagnostics=True))


def test_the_latch_suppresses_the_key() -> None:
    clients._ANTHROPIC_DIAGNOSTICS_REFUSED[0] = True
    assert "diagnostics" not in AnthropicClient()._build_body(_request(cache_diagnostics=True))


# -- stream parse --------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("diagnostics", "previous", "expected"),
    [
        (None, None, ("baseline", 0)),
        (None, "msg_prev", ("unchanged", 0)),
        ({"cache_miss_reason": None}, "msg_prev", ("pending", 0)),
        (
            {
                "cache_miss_reason": {
                    "type": "system_changed",
                    "cache_missed_input_tokens": 4321,
                }
            },
            "msg_prev",
            ("system_changed", 4321),
        ),
        (
            {"cache_miss_reason": {"type": "previous_message_not_found"}},
            "msg_prev",
            ("previous_message_not_found", 0),
        ),
        ("garbage", "msg_prev", ("unavailable", 0)),
        ({"cache_miss_reason": {"type": ""}}, "msg_prev", ("unavailable", 0)),
    ],
)
async def test_message_start_diagnostics_reach_the_start_event(
    diagnostics: Any, previous: str | None, expected: tuple[str, int]
) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse({"id": "msg_new", "diagnostics": diagnostics}),
            headers={"content-type": "text/event-stream"},
        )

    events = await _collect(
        _client(handler).stream(
            _request(cache_diagnostics=True, cache_diagnostics_previous_id=previous), "sk-test"
        )
    )
    start = _start(events)
    assert start.response_id == "msg_new"
    assert (start.cache_miss_type, start.cache_missed_tokens) == expected


@pytest.mark.asyncio
async def test_an_undiagnosed_request_reports_no_verdict() -> None:
    """``None``, never a fabricated ``baseline``, when we did not opt in."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=_sse({"id": "msg_x"}), headers={"content-type": "text/event-stream"}
        )

    start = _start(await _collect(_client(handler).stream(_request(), "sk-test")))
    assert start.cache_miss_type is None


# -- fail closed ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_refused_field_is_retried_once_without_it_and_latched_off(caplog) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "diagnostics" in body:
            return httpx.Response(
                400,
                json={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "diagnostics: Extra inputs are not permitted",
                    },
                },
            )
        return httpx.Response(
            200, content=_sse({"id": "msg_ok"}), headers={"content-type": "text/event-stream"}
        )

    client = _client(handler)
    with caplog.at_level(logging.INFO, logger="local_operator.providers.clients"):
        events = await _collect(client.stream(_request(cache_diagnostics=True), "sk-test"))
    assert [("diagnostics" in body) for body in bodies] == [True, False]
    assert _start(events).cache_miss_type is None, "the retried call was not diagnosed"
    assert clients._ANTHROPIC_DIAGNOSTICS_REFUSED[0] is True
    assert sum("cache diagnostics refused" in r.message for r in caplog.records) == 1

    # Latched: the next request never carries the key, so no second round trip.
    await _collect(client.stream(_request(cache_diagnostics=True), "sk-test"))
    assert [("diagnostics" in body) for body in bodies] == [True, False, False]


@pytest.mark.asyncio
async def test_an_unrelated_400_is_not_retried() -> None:
    """Status alone never triggers the retry: an ordinary 400 is a request defect."""
    from local_operator.providers.failover import ProviderError

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            400,
            json={"error": {"type": "invalid_request_error", "message": "max_tokens: too big"}},
        )

    with pytest.raises(ProviderError):
        await _collect(_client(handler).stream(_request(cache_diagnostics=True), "sk-test"))
    assert len(calls) == 1
    assert clients._ANTHROPIC_DIAGNOSTICS_REFUSED[0] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "tools.0.name: run_diagnostics is not a valid tool name",
        "tools.0.input_schema.properties.diagnostics_level: invalid type",
        "messages.1.content.0.tool_use.input: diagnostics_report too long",
        "diagnostics_level: Extra inputs are not permitted",
        "tools.0.input_schema.properties.diagnostics: invalid type",
        # PR #2984 review round 2, r2-n1: a 400 QUOTING request text that
        # contains ``diagnostics:`` after a space latched the old matcher.
        "system.0.text: run diagnostics: please",
        # PR #3018 review round 1, m1: prose ending a comma clause with
        # ``diagnostics.`` latched while the tail accepted any period.
        "tools.0.description: checks health, runs diagnostics. returns status",
        "messages.0.content.0.text: hello, please run diagnostics. thanks",
    ],
)
async def test_a_400_that_merely_mentions_diagnostics_does_not_latch(message: str) -> None:
    """Only the field's own refusal shape latches; a tool/schema path containing
    the word is an ordinary request defect and must surface unretried."""
    from local_operator.providers.failover import ProviderError

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            400, json={"error": {"type": "invalid_request_error", "message": message}}
        )

    with pytest.raises(ProviderError):
        await _collect(_client(handler).stream(_request(cache_diagnostics=True), "sk-test"))
    assert len(calls) == 1
    assert clients._ANTHROPIC_DIAGNOSTICS_REFUSED[0] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "diagnostics: Extra inputs are not permitted",
        "diagnostics.previous_message_id: Input should be a valid string",
        "messages: ok, but diagnostics: Extra inputs are not permitted",
        # The rest of the round-2 reviewer's latch set, pinned here so the
        # clause rule that closed r2-n1 is proved not to have narrowed it.
        "Diagnostics: Extra inputs are not permitted",
        "   diagnostics: Field required",
        "diagnostics.previous_message_id: String should match pattern '^msg_'",
        "messages.0.content: bad ; diagnostics: Extra inputs are not permitted",
        "diagnostics is not supported for this model",
        # A real refusal (anthropics/claude-code#68553): the ``\.\w`` tail
        # must still take a path segment after the period.
        "diagnostics.previous_message_id: must be the `id` from a prior /v1/messages "
        "response (starts with `msg_`)",
    ],
)
async def test_the_field_refusal_shapes_latch(message: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "diagnostics" in json.loads(request.content):
            return httpx.Response(
                400, json={"error": {"type": "invalid_request_error", "message": message}}
            )
        return httpx.Response(
            200, content=_sse({"id": "msg_ok"}), headers={"content-type": "text/event-stream"}
        )

    await _collect(_client(handler).stream(_request(cache_diagnostics=True), "sk-test"))
    assert clients._ANTHROPIC_DIAGNOSTICS_REFUSED[0] is True


# -- the conversation chain (SessionStreamFn) ------------------------------------


def _anthropic_stream_fn(
    tmp_path,
    monkeypatch,
    session_id: str,
    bodies: list[dict[str, Any]],
    snapshots: list[CallSnapshot] | None = None,
    diagnostics: Any = None,
):
    """A real ``create_stream_fn`` over a mock Anthropic wire.

    ``snapshots`` collects what ``_record_usage`` hands the recorder, so the
    ledger wiring is asserted end to end; ``diagnostics`` is the
    ``message_start.diagnostics`` every response carries.
    """
    from local_operator import analytics
    from local_operator.model.configure import create_stream_fn
    from local_operator.providers.auth_store import AuthStore

    sink = snapshots if snapshots is not None else []
    monkeypatch.setattr(analytics, "record_call", sink.append)
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        counter["n"] += 1
        return httpx.Response(
            200,
            content=_sse(
                {"id": f"msg_{session_id}_{counter['n']}", "diagnostics": diagnostics},
                cache_write=30_000,
            ),
            headers={"content-type": "text/event-stream"},
        )

    store = AuthStore(tmp_path / f"auth-{session_id}.db")
    store.upsert_credential("anthropic", {"key": "sk-ant-test", "source": "login"})
    stream = create_stream_fn(store, {}, session_id=session_id)
    return stream, store, handler


@pytest.mark.asyncio
async def test_turns_chain_per_conversation_and_helpers_stay_out(tmp_path, monkeypatch) -> None:
    """Turn N names turn N-1's id; a fork starts its own chain; an aside is
    never diagnosed and never moves the chain."""
    parent_bodies: list[dict[str, Any]] = []
    parent, store, handler = _anthropic_stream_fn(tmp_path, monkeypatch, "p", parent_bodies)
    await parent._http.aclose()
    parent._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    parent._transport.http = parent._http
    spec = ModelSpec(provider="anthropic", model_id="claude-haiku-4-5")

    def turn(**extra: Any) -> ChatRequest:
        return ChatRequest(model=spec, messages=[Message.user("hi")], **extra)

    try:
        await _collect(parent(turn(), None))
        await _collect(parent(turn(purpose="aside"), None))
        await _collect(parent(turn(), None))

        child = parent.fork("c")
        await _collect(child(turn(), None))
        await _collect(parent(turn(), None))
        await child.close()
    finally:
        await parent.close()
        store.close()

    diag = [body.get("diagnostics", "absent") for body in parent_bodies]
    assert diag == [
        {"previous_message_id": None},  # parent turn 1 opts in
        "absent",  # the aside is never diagnosed
        {"previous_message_id": "msg_p_1"},  # turn 2 skips the aside's response
        {"previous_message_id": None},  # the fork starts its OWN chain
        {"previous_message_id": "msg_p_3"},  # parent unaffected by the child's id
    ]


async def _mounted(tmp_path, monkeypatch, bodies, **kwargs):
    stream, store, handler = _anthropic_stream_fn(tmp_path, monkeypatch, "p", bodies, **kwargs)
    await stream._http.aclose()
    stream._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    stream._transport.http = stream._http
    return stream, store


def _turn() -> ChatRequest:
    spec = ModelSpec(provider="anthropic", model_id="claude-haiku-4-5")
    return ChatRequest(model=spec, messages=[Message.user("hi")])


@pytest.mark.asyncio
async def test_an_undiagnosed_start_does_not_advance_the_chain(tmp_path, monkeypatch) -> None:
    """A turn whose response carries no verdict (here: the latch is on, so the
    key was not sent and the API kept no fingerprint) must not become the next
    turn's ``previous_message_id``: the comparison could never find it."""
    bodies: list[dict[str, Any]] = []
    stream, store = await _mounted(tmp_path, monkeypatch, bodies)
    try:
        await _collect(stream(_turn(), None))  # msg_p_1, diagnosed
        clients._ANTHROPIC_DIAGNOSTICS_REFUSED[0] = True
        await _collect(stream(_turn(), None))  # msg_p_2, NOT diagnosed
        clients._ANTHROPIC_DIAGNOSTICS_REFUSED[0] = False
        await _collect(stream(_turn(), None))
    finally:
        await stream.close()
        store.close()
    diag = [body.get("diagnostics", "absent") for body in bodies]
    assert diag == [
        {"previous_message_id": None},
        "absent",
        {"previous_message_id": "msg_p_1"},  # skips the undiagnosed msg_p_2
    ]


@pytest.mark.asyncio
async def test_an_undiagnosed_turn_stamps_the_chain_value_not_its_own_id(
    tmp_path, monkeypatch
) -> None:
    """The usage a turn persists carries the CHAIN value — what the next turn
    would compare against — never the response's own id. On an undiagnosed
    turn (here: the latch is on, so the API kept no fingerprint for msg_p_2)
    the two differ, and persisting msg_p_2 would make a resume name an id the
    comparison can never find (PR #3018 review round 1, m2)."""
    from local_operator.harness.types import StreamEndEvent, cache_diagnostics_id_of

    bodies: list[dict[str, Any]] = []
    stream, store = await _mounted(tmp_path, monkeypatch, bodies)
    try:
        first = await _collect(stream(_turn(), None))  # msg_p_1, diagnosed
        clients._ANTHROPIC_DIAGNOSTICS_REFUSED[0] = True
        second = await _collect(stream(_turn(), None))  # msg_p_2, NOT diagnosed
    finally:
        await stream.close()
        store.close()

    def stamped(events: list[Any]) -> str | None:
        end = next(e for e in events if isinstance(e, StreamEndEvent))
        return cache_diagnostics_id_of(end.usage)

    assert _start(second).response_id == "msg_p_2"
    assert _start(second).cache_miss_type is None, "the second turn must be undiagnosed"
    assert stamped(first) == "msg_p_1"
    assert stamped(second) == "msg_p_1", "stamped its own id instead of the chain value"


def test_a_resume_seed_never_overwrites_a_live_chain() -> None:
    """``seed_cache_diagnostics`` fills an EMPTY chain only: a stream that has
    already diagnosed a turn in this process holds a fresher id than any
    transcript row (PR #3018 review round 1, m2)."""
    from unittest.mock import MagicMock

    from local_operator.model.configure import SessionStreamFn

    stream = SessionStreamFn(MagicMock(), {}, "sess-1")
    stream.seed_cache_diagnostics("msg_persisted")
    assert stream._cache_diagnostics_previous_id == "msg_persisted"

    stream._cache_diagnostics_previous_id = "msg_live"
    stream.seed_cache_diagnostics("msg_persisted")
    assert stream._cache_diagnostics_previous_id == "msg_live"


@pytest.mark.asyncio
async def test_the_verdict_reaches_the_call_snapshot(tmp_path, monkeypatch) -> None:
    """``StreamStartEvent`` -> ``_record_stream`` -> ``_record_usage`` ->
    ``CallSnapshot``: the ledger columns are the deliverable, so the wiring is
    asserted on what the recorder actually receives, not on a hand-built row."""
    bodies: list[dict[str, Any]] = []
    snapshots: list[CallSnapshot] = []
    reason = {"cache_miss_reason": {"type": "tools_changed", "cache_missed_input_tokens": 777}}
    stream, store = await _mounted(
        tmp_path, monkeypatch, bodies, snapshots=snapshots, diagnostics=reason
    )
    try:
        await _collect(stream(_turn(), None))
        await _collect(stream(_turn(), None))
        await _collect(stream(_turn().model_copy(update={"purpose": "aside"}), None))
    finally:
        await stream.close()
        store.close()
    assert [(s.purpose, s.cache_miss_type, s.cache_missed_tokens) for s in snapshots] == [
        ("turn", "tools_changed", 777),
        ("turn", "tools_changed", 777),
        ("aside", "", 0),  # undiagnosed: the column reads '' rather than a guess
    ]


# -- ledger ------------------------------------------------------------------------


def _snap(**overrides: Any) -> CallSnapshot:
    return CallSnapshot(
        ts_ms=int(time.time() * 1000),
        session_id="s1",
        provider="anthropic",
        model_id="claude",
        input_tokens=10,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=30_000,
        reasoning_tokens=0,
        context_tokens=30_010,
        component_chars={"conversation": 10},
        ok=True,
        cost_micro=0,
        cost_known=True,
        priced=True,
        **overrides,
    )


def test_the_verdict_is_recorded_and_an_older_ledger_gains_the_columns(tmp_path) -> None:
    db = tmp_path / "pre-diag.db"
    store = AnalyticsStore(db)
    assert store.record_batch([_snap()]) == 1
    conn = store._connect()
    assert conn is not None
    conn.execute("ALTER TABLE calls DROP COLUMN cache_miss_type")
    conn.execute("ALTER TABLE calls DROP COLUMN cache_missed_tokens")
    store.close()

    # Re-open: the ALTER path, not the CREATE, is what must put them back.
    store = AnalyticsStore(db)
    snap = dataclasses.replace(
        _snap(), cache_miss_type="system_changed", cache_missed_tokens=29_000
    )
    assert store.record_batch([snap]) == 1
    store.close()
    rows = (
        sqlite3.connect(str(db))
        .execute("SELECT cache_miss_type, cache_missed_tokens FROM calls ORDER BY id")
        .fetchall()
    )
    assert rows == [("", 0), ("system_changed", 29_000)], "old rows read as undiagnosed"


# -- the break log ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("miss_type", "write", "logged"),
    [
        ("system_changed", 30_000, True),
        ("system_changed", 5_000, False),  # small write: the "rare, low impact" case
        ("unchanged", 30_000, False),  # TTL expiry, not a break
        ("previous_message_not_found", 30_000, False),
        ("", 30_000, False),
    ],
)
def test_a_costly_named_break_logs_one_line(caplog, miss_type, write, logged) -> None:
    from unittest.mock import MagicMock

    from local_operator.model.configure import SessionStreamFn

    stream = SessionStreamFn(MagicMock(), {}, "sess-1")
    usage = Usage(input_tokens=10, cache_read_tokens=0, cache_write_tokens=write)
    with caplog.at_level(logging.INFO, logger="local_operator.model.configure"):
        stream._log_cache_break(usage, miss_type, 1234)
    lines = [r.getMessage() for r in caplog.records if "[prompt cache break]" in r.getMessage()]
    assert len(lines) == (1 if logged else 0)
    if logged:
        assert "session=sess-1" in lines[0] and "type=system_changed" in lines[0]
        assert "missed_tokens=1234" in lines[0]


# -- the chain across a resume (PR #2984 QA Q1) --------------------------------


async def _session_turn(tmp_path, monkeypatch, directory, stream_id: str, bodies):
    """One real ``Session`` turn over a real ``create_stream_fn`` on the mock wire.

    A fresh stream per call is the point: each call is a NEW process's view
    of the conversation, so the only way a chain id can carry over is the
    transcript.
    """
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    stream, store = await _mounted_as(tmp_path, monkeypatch, stream_id, bodies)
    session = Session(
        model=ModelSpec(provider="anthropic", model_id="claude-haiku-4-5"),
        stream_fn=stream,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable"],
    )
    try:
        await session.prompt("hi")
    finally:
        await session.dispose()
        await stream.close()
        store.close()


async def _mounted_as(tmp_path, monkeypatch, stream_id, bodies):
    stream, store, handler = _anthropic_stream_fn(tmp_path, monkeypatch, stream_id, bodies)
    await stream._http.aclose()
    stream._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    stream._transport.http = stream._http
    return stream, store


@pytest.mark.asyncio
async def test_a_resume_continues_the_chain_and_a_fork_starts_its_own(
    tmp_path, monkeypatch
) -> None:
    """The diagnosed id is persisted with the turn's usage, so a NEW process
    resuming the conversation names it as ``previous_message_id`` instead of
    restarting at null (which recorded an instruction-edit break as
    ``baseline``). A transcript fork copies those rows, but they name its
    PARENT's requests, so the fork still opens its own chain at null."""
    from local_operator.fork import fork_session
    from local_operator.session.transcript import Transcript

    config_dir = tmp_path / "cfg"
    directory = config_dir / "sessions" / "parent01"
    bodies: list[dict[str, Any]] = []

    await _session_turn(tmp_path, monkeypatch, directory, "a", bodies)
    # The row carries the id; nothing else about the row shape moved.
    persisted = [
        e.payload["usage"].get("cache_diagnostics_id")
        for e in Transcript(directory).entries()
        if e.type == "message" and isinstance(e.payload.get("usage"), dict)
    ]
    assert persisted == ["msg_a_1"]

    await _session_turn(tmp_path, monkeypatch, directory, "b", bodies)  # the resume
    fork_id = fork_session(config_dir, "parent01")
    await _session_turn(tmp_path, monkeypatch, config_dir / "sessions" / fork_id, "c", bodies)
    await _session_turn(tmp_path, monkeypatch, directory, "d", bodies)  # parent again

    assert [body.get("diagnostics", "absent") for body in bodies] == [
        {"previous_message_id": None},  # first ever turn
        {"previous_message_id": "msg_a_1"},  # resumed: the previous PROCESS's id
        {"previous_message_id": None},  # the fork opens its own chain
        {"previous_message_id": "msg_b_1"},  # the parent is unaffected by the fork
    ]


def test_a_fork_resumes_its_own_chain_once_it_has_one(tmp_path) -> None:
    """The fork exclusion is by TIME, not "never seed a fork": rows the fork
    journalled after ``forked_at`` are its own and must seed its next resume."""
    from local_operator.resume import ORIGIN_FORK, ORIGIN_NAME
    from local_operator.session.session import _resumed_cache_diagnostics_id
    from local_operator.session.transcript import (
        ENTRY_MESSAGE,
        TranscriptEntry,
        newest_cache_diagnostics_id,
    )

    copied = TranscriptEntry(
        id="e1",
        ts=100.0,
        type=ENTRY_MESSAGE,
        payload={"role": "assistant", "usage": {"cache_diagnostics_id": "msg_parent"}},
    )
    own = TranscriptEntry(
        id="e2",
        ts=300.0,
        type=ENTRY_MESSAGE,
        payload={"role": "assistant", "usage": {"cache_diagnostics_id": "msg_fork"}},
    )
    assert newest_cache_diagnostics_id([copied, own]) == "msg_fork"
    assert newest_cache_diagnostics_id([copied]) == "msg_parent"
    assert newest_cache_diagnostics_id([]) is None

    class _Journal:
        """Just the two members the seed reads: entries and the directory
        whose origin marker says when the fork was taken."""

        def __init__(self, entries):
            self._entries = entries
            self.directory = tmp_path

        def entries(self):
            return list(self._entries)

    (tmp_path / ORIGIN_NAME).write_text(
        json.dumps({"origin": ORIGIN_FORK, "parent": "p", "forked_at": 200.0})
    )
    assert _resumed_cache_diagnostics_id(_Journal([copied])) is None  # type: ignore[arg-type]
    assert (
        _resumed_cache_diagnostics_id(_Journal([copied, own]))  # type: ignore[arg-type]
        == "msg_fork"
    )


@pytest.mark.asyncio
async def test_a_mid_session_account_switch_restarts_the_chain(tmp_path, monkeypatch) -> None:
    """Fingerprints live per account: after the session moves to another
    credential the previous id cannot be found, so the call must open a fresh
    chain (``baseline``) instead of naming it (``previous_message_not_found``)."""
    bodies: list[dict[str, Any]] = []
    stream, store = await _mounted(tmp_path, monkeypatch, bodies)
    try:
        await _collect(stream(_turn(), None))  # msg_p_1 on the first credential
        first = store.session_credential_id("anthropic", "p")
        assert first is not None, "the resolve pins the serving credential"
        await _collect(stream(_turn(), None))  # same account: chain continues
        other = store.upsert_credential("anthropic", {"key": "sk-ant-other", "source": "login"})
        store.pin_session_credential("anthropic", "p", other.id)
        await _collect(stream(_turn(), None))  # switched account: chain restarts
        await _collect(stream(_turn(), None))  # and continues on the new one
    finally:
        await stream.close()
        store.close()
    diag = [body.get("diagnostics", "absent") for body in bodies]
    assert diag == [
        {"previous_message_id": None},
        {"previous_message_id": "msg_p_1"},
        {"previous_message_id": None},
        {"previous_message_id": "msg_p_3"},
    ]


@pytest.mark.asyncio
async def test_a_released_sticky_on_the_same_account_keeps_the_chain(tmp_path, monkeypatch) -> None:
    """An UNSET sticky is unknown, not switched: after the pin is released the
    next resolve lands on the same credential, so the chain must continue
    rather than open a spurious ``baseline`` (review #3072 m2)."""
    bodies: list[dict[str, Any]] = []
    stream, store = await _mounted(tmp_path, monkeypatch, bodies)
    try:
        await _collect(stream(_turn(), None))  # msg_p_1
        first = store.session_credential_id("anthropic", "p")
        store.release_session_credential("anthropic", "p")
        await _collect(stream(_turn(), None))
        assert store.session_credential_id("anthropic", "p") == first
    finally:
        await stream.close()
        store.close()
    diag = [body.get("diagnostics", "absent") for body in bodies]
    assert diag == [
        {"previous_message_id": None},
        {"previous_message_id": "msg_p_1"},
    ]
