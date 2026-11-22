"""The tool-result seam, fail-closed policy, and SDK durable receipts."""

import json
import socket
import sqlite3
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
import pytest_asyncio

from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.types import (
    AbortSignal,
    Message,
    StreamEndEvent,
    StreamTextDelta,
    StreamUsageEvent,
    TextContent,
    ToolResult,
    Usage,
)
from local_operator.research_evidence import (
    CANADA,
    COMPANION,
    BedrockWireIdentity,
    PublicEvidenceConditioner,
    _Evidence,
)
from local_operator.sdk import (
    PublicResearchPolicy,
    SessionSpec,
    auxiliary_usage,
    raw_public_evidence,
)
from local_operator.session.transcript import Transcript

FETCH = "mcp__fixture__fetch"
SEARCH = "mcp__fixture__search"
MAIN = BedrockWireIdentity("ca.anthropic.claude-sonnet-5", CANADA, "test-binding")
NEEDLE = "The source explicitly says the committee did not approve the plan."
OMITTED = "The later vote had 27 abstentions."


def page():
    content = "\n".join(
        [NEEDLE, OMITTED]
        + [
            f"Archive paragraph {i}: " + "Background about the legislative agenda. " * 12
            for i in range(22)
        ]
    )
    return json.dumps(
        {
            "url": "https://example.org/page",
            "offset": 14,
            "next_offset": 14 + len(content),
            "total_chars": 14 + len(content) + 35,
            "source_partial": True,
            "content": content,
        }
    )


def text_of(result: ToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def fake_usage():
    return Usage(
        input_tokens=2500,
        output_tokens=4,
        context_tokens=2504,
        provider="bedrock",
        model_id=COMPANION,
        estimated_usd_cost=0.0003,
    )


@pytest_asyncio.fixture
async def wire(tmp_path):
    transcript = Transcript(tmp_path / "synthetic-evidence-session")
    await transcript.append_custom("test_fixture", {})
    return SimpleNamespace(
        transcript=transcript, requests=[], response="[[1]]", stop="stop", abort=None
    )


async def append(
    wire,
    *,
    text=None,
    name=FETCH,
    is_error=False,
    secret=False,
    raw=False,
    mode: Literal["extract", "raw"] = "extract",
    masker=lambda value: value,
    extra_content=(),
    signal=None,
):
    async def stream(request, abort):
        wire.requests.append(request)
        request._evidence_binding.selected = request._evidence_binding.expected
        if wire.abort == "unknown":
            abort.abort()
            return
        yield StreamUsageEvent(usage=fake_usage())
        if wire.abort == "known":
            abort.abort()
            return
        if wire.abort == "exception":
            raise RuntimeError("connection interrupted after usage")
        yield StreamTextDelta(delta=wire.response)
        yield StreamEndEvent(stop_reason=wire.stop, usage=fake_usage())

    condition = PublicEvidenceConditioner(
        PublicResearchPolicy(
            "Find the vote's public outcome",
            {FETCH: "paginated_web_text", SEARCH: "search_results"},
            mode=mode,
            attest_public_source=lambda *_: True,
        ),
        stream,
        wire.transcript,
    )
    condition.main = MAIN
    wire._public_evidence = condition
    raw_result = ToolResult(
        tool_name=name,
        tool_call_id="call-1",
        is_error=is_error,
        duration_s=1.25,
        details={"status": "original"},
        content=[TextContent(text=page() if text is None else text), *extra_content],
    )
    context = LoopContext(
        condition_public_evidence=condition,
        one_turn_secret_run_active=secret,
        original_call_args={"call-1": {"raw": raw, "token": "not-for-the-companion"}},
    )
    messages = []
    await AgentLoop()._append_results(
        context,
        [raw_result],
        messages,
        redact=masker,
        evidence_signal=signal,
        condition_evidence=True,
    )
    return raw_result, messages[0]


@pytest.mark.asyncio
async def test_partial_verbatim_omission_recoverable_without_refetch(wire):
    result, last = await append(wire)
    assert len(wire.requests) == 1
    request = wire.requests[0]
    assert request.isolated and request.same_route_retry and request.cache_markers == "none"
    assert request.model.model_id == COMPANION and not request.tools
    assert "not-for-the-companion" not in request.messages[0].text
    assert "https://example.org/page" not in request.messages[0].text
    header, encoded = last.text.split("\n", 1)
    selected = json.loads(encoded)
    raw = json.loads(page())
    assert "not complete" in header and OMITTED not in selected["content"]
    assert NEEDLE in selected.pop("content")
    assert selected == {key: value for key, value in raw.items() if key != "content"}
    assert text_of(result) == page()
    assert await raw_public_evidence(wire, result.tool_call_id) == page()
    [receipt] = await auxiliary_usage(wire)
    assert receipt.usage == fake_usage() and receipt.tool_call_id == result.tool_call_id
    assert len(wire.requests) == 1


@pytest.mark.parametrize(
    "case",
    [
        "unauthorized",
        "code",
        "secret",
        "raw-policy",
        "raw-tool",
        "error",
        "redacted",
        "wrong-kind",
        "signed-url",
        "truncated",
        "mixed-content",
    ],
)
@pytest.mark.asyncio
async def test_fail_closed_without_dispatch(wire, case):
    args = {
        "unauthorized": {"name": "mcp__elsewhere__fetch"},
        "code": {"name": "read"},
        "secret": {"secret": True},
        "raw-policy": {"mode": "raw"},
        "raw-tool": {"raw": True},
        "error": {"is_error": True},
        "redacted": {"masker": lambda value: value.replace("abstentions", "MASKED")},
        "wrong-kind": {"name": SEARCH},
        "signed-url": {"text": page().replace("/page", "/page?signature=example")},
        "truncated": {"text": page()[:-3]},
        "mixed-content": {"extra_content": [TextContent(text="other block")]},
    }[case]
    raw, result = await append(wire, **args)
    assert not wire.requests and not await auxiliary_usage(wire)
    assert await raw_public_evidence(wire, raw.tool_call_id) is None
    assert ("MASKED" if case == "redacted" else text_of(raw)) in result.text


@pytest.mark.parametrize(
    "reply,stop",
    [
        ("[[999]]", "stop"),
        ("[[1],[2]]", "stop"),
        ("[[]]", "stop"),
        ("[[true]]", "stop"),
        ("[[1]]", "length"),
        ("null", "stop"),
        (json.dumps([list(range(1, 25))]), "stop"),
    ],
)
@pytest.mark.asyncio
async def test_invalid_capped_omitted_or_uneconomical_response_keeps_paid_usage(wire, reply, stop):
    wire.response, wire.stop = reply, stop
    original, result = await append(wire)
    assert result.text == text_of(original)
    assert await raw_public_evidence(wire, original.tool_call_id) is None
    [receipt] = await auxiliary_usage(wire)
    assert receipt.usage == fake_usage()


@pytest.mark.parametrize("case", ["known", "unknown", "exception"])
@pytest.mark.asyncio
async def test_abort_or_transport_failure_retains_known_or_unknown_billing(wire, case):
    wire.abort = case
    original, result = await append(wire, signal=AbortSignal())
    assert result.text == text_of(original)
    [receipt] = await auxiliary_usage(wire)
    assert receipt.usage == (None if case == "unknown" else fake_usage())
    assert receipt.outcome == ("error" if case == "exception" else "aborted")
    reopened = SimpleNamespace(transcript=Transcript(wire.transcript.directory, read_only=True))
    assert (await auxiliary_usage(reopened)) == [receipt]


def test_search_preserves_rank_titles_urls_dates_flags_and_every_row():
    rows = [
        {
            "title": 'A "quoted" title',
            "url": "https://example.org/a",
            "id": 7,
            "date": None,
            "snippet": "line 1\nline 2\nline 3",
        },
        {
            "title": "B",
            "url": "https://example.org/b",
            "id": 1,
            "date": "2026-04-07",
            "snippet": "line 4\nline 5",
        },
    ]
    value = _Evidence.parse(
        json.dumps({"results": rows, "total": 9, "capped": False}), "search_results"
    )
    assert value is not None
    assert value.render("[[2],[]]") is None
    output = value.render("[[2],[1]]")
    assert output is not None
    decoded = json.loads(output.split("\n", 1)[1])
    assert decoded["total"] == 9 and decoded["capped"] is False
    for row, observed, line in zip(rows, decoded["results"], ["line 2", "line 4"], strict=True):
        assert line in observed.pop("snippet")
        assert observed == {key: data for key, data in row.items() if key != "snippet"}


def test_default_off_and_untrustable_policy_rejected():
    assert SessionSpec().public_research is None
    with pytest.raises(ValueError):
        PublicResearchPolicy("public", {"bash": "paginated_web_text"})
    with pytest.raises(ValueError):
        PublicResearchPolicy("public", {FETCH: cast(Any, "unexpected_kind")})


@pytest.mark.parametrize("on", ["started", "stop"])
@pytest.mark.asyncio
async def test_failed_billing_append_never_disappears_as_zero(wire, monkeypatch, on):
    real_append = wire.transcript.append_custom

    async def append_with_failure(kind, details, **kwargs):
        if details["outcome"] == on:
            raise OSError("fixture journal error")
        return await real_append(kind, details, **kwargs)

    monkeypatch.setattr(wire.transcript, "append_custom", append_with_failure)
    raw, result = await append(wire)
    assert result.text == text_of(raw)
    if on == "started":
        assert wire.requests == [] and await auxiliary_usage(wire) == []
    else:
        [live] = await auxiliary_usage(wire)
        assert live.outcome == "journal_error" and live.usage == fake_usage()
        reopened = SimpleNamespace(transcript=Transcript(wire.transcript.directory, read_only=True))
        [durable] = await auxiliary_usage(reopened)
        assert durable.request_id == live.request_id
        assert durable.outcome == "started" and durable.usage is None


@pytest.mark.asyncio
async def test_unreadable_journal_raises_instead_of_returning_empty(wire, monkeypatch):
    from local_operator.session import transcript as journal

    def unreadable(*args, **kwargs):
        raise PermissionError("fixture cannot open journal")

    monkeypatch.setattr(journal, "_resolved_transcript_path", unreadable)
    with pytest.raises(PermissionError):
        await auxiliary_usage(wire)


@pytest.fixture
def analytics_ledger(tmp_path):
    from local_operator.analytics.recorder import reset_recorder_for_test
    from local_operator.analytics.store import AnalyticsStore

    path = tmp_path / "test-analytics.sqlite"
    recorder = reset_recorder_for_test(AnalyticsStore(path))
    try:
        yield recorder, path
    finally:
        reset_recorder_for_test()


@pytest.mark.parametrize(
    "case",
    [
        "complete",
        "identity",
        "default",
        "missing_provenance",
        "unverified_search_row",
        "local_origin",
        "capped",
        "truncated",
        "producer_error",
        "missing_message_stop",
        "unexpected_eof",
        "missing_usage",
        "no_checkpoint",
        "stale_checkpoint",
        "abort_no_checkpoint",
    ],
)
@pytest.mark.asyncio
async def test_sdk_prompt_uses_real_bedrock_wire_and_single_ledger_id(
    tmp_path, monkeypatch, case, analytics_ledger
):
    import httpx

    from local_operator.harness.types import AgentTool
    from local_operator.model.configure import SessionStreamFn
    from local_operator.providers.bedrock.client import BedrockAnthropicClient
    from local_operator.sdk import SessionRoots, open_session
    from local_operator.session.session import Session
    from local_operator.session.spend import SESSION_SPEND_CUSTOM_TYPE
    from tests.unit.providers.bedrock.eventstream import anthropic_events, stream_bytes

    home = tmp_path / "isolated-home"
    config, agent_home, work = (home / part for part in ("config", "agent", "work"))
    for directory in (config, agent_home, work):
        directory.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("AWS_REGION", CANADA)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "synthetic-first-bearer")
    monkeypatch.chdir(work)

    def deny_network(*args, **kwargs):
        raise AssertionError("the isolated wire fixture must not use DNS or network sockets")

    monkeypatch.setattr(socket, "getaddrinfo", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    with pytest.raises(AssertionError, match="must not use DNS"):
        socket.getaddrinfo("example.org", 443)

    search = case == "unverified_search_row"
    tool_name = SEARCH if search else FETCH
    page_record = json.loads(page())
    fixture_content = page_record["content"]
    fixture_url = page_record["url"]
    assertions = []
    if case == "missing_provenance":
        page_record["public_source"] = True  # Self-assertion is not provenance.
    elif case == "local_origin":
        page_record["url"] = "http://127.0.0.1/example"
    elif case == "capped":
        page_record["capped"] = True
    elif case == "truncated":
        page_record["truncated"] = True
    elif case == "producer_error":
        page_record["status"] = "error"
    if search:
        page_record = {
            "results": [
                {"url": fixture_url, "snippet": fixture_content, "title": "Verified"},
                {"url": "https://example.org/unverified", "snippet": NEEDLE, "title": "Unverified"},
            ]
        }
    source_key = (tool_name, "toolu_fixture", fixture_url, fixture_content)
    original = json.dumps(page_record)

    def attest(name, call_id, url, text):
        assertions.append((name, call_id, url))
        if case == "local_origin":
            pytest.fail("non-public origins must be excluded before the adapter assertion")
        return (name, call_id, url, text) == source_key

    no_checkpoint = case in {"no_checkpoint", "abort_no_checkpoint"}
    incomplete = case in {"missing_message_stop", "unexpected_eof", "missing_usage"}
    accepted = case in {"complete", "no_checkpoint", "stale_checkpoint"}
    no_attempt = case in {
        "default",
        "missing_provenance",
        "unverified_search_row",
        "local_origin",
        "capped",
        "truncated",
        "producer_error",
    }
    calls = []

    async def handler(request):
        body = json.loads(request.content)
        calls.append((request.url.host, request.url.path, body))
        if COMPANION in request.url.path:
            events = anthropic_events(
                text="[[1]]",
                usage={"input_tokens": 2500},
                output_tokens=None if case == "missing_usage" else 4,
            )
            if case == "missing_message_stop":
                events = events[:-1]
            elif case == "unexpected_eof":
                events = events[:-2]
            elif case == "abort_no_checkpoint":
                session.abort()
                events = events[:-2]
        elif len(calls) == 1:
            block = [
                {
                    "type": "content_block_start",
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_fixture",
                        "name": tool_name,
                        "input": {},
                    },
                },
                {
                    "type": "content_block_delta",
                    "delta": {"type": "input_json_delta", "partial_json": "{}"},
                },
                {"type": "content_block_stop"},
            ]
            events = anthropic_events(
                text="", usage={"input_tokens": 800}, extra_blocks=[block], stop_reason="tool_use"
            )
        else:
            events = anthropic_events(text="Fixture answer", usage={"input_tokens": 1000})
        return httpx.Response(200, content=stream_bytes(events))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:

        def client_for(stream, spec):
            if spec.provider == "bedrock":
                return BedrockAnthropicClient(http_client=http)
            raise AssertionError(f"Unexpected provider: {spec.provider}")

        monkeypatch.setattr(SessionStreamFn, "_client_for", client_for)

        async def execute(call_id, args, signal, on_update, context):
            if case == "identity":
                monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "synthetic-changed-bearer")
            if case == "stale_checkpoint":
                assert isinstance(session, Session)
                await session._write_spend_record()
                assert session.transcript.latest_custom(SESSION_SPEND_CUSTOM_TYPE)
            return ToolResult(
                tool_call_id=call_id,
                tool_name=tool_name,
                content=[TextContent(text=original)],
            )

        append_custom = Transcript.append_custom

        async def append_with_spend_fault(transcript, custom_type, details, **kwargs):
            stale = case == "stale_checkpoint" and details.get("auxiliary_request_ids")
            if custom_type == SESSION_SPEND_CUSTOM_TYPE and (no_checkpoint or stale):
                raise OSError("synthetic spend writer fault")
            return await append_custom(transcript, custom_type, details, **kwargs)

        roots = SessionRoots(
            config_dir=config, agent_home=agent_home, cwd=work, allow_volatile=True
        )
        policy = PublicResearchPolicy(
            "Find the public outcome",
            {FETCH: "paginated_web_text", SEARCH: "search_results"},
            attest_public_source=None if case == "missing_provenance" else attest,
        )
        spec = SessionSpec(
            hosting="bedrock",
            model=MAIN.model_id,
            world_root=work,
            public_research=None if case == "default" else policy,
        )
        observed = []
        monkeypatch.setattr(Transcript, "append_custom", append_with_spend_fault)
        async with open_session(spec, roots=roots) as session:
            assert isinstance(session, Session)
            session.refresh_tools(
                [
                    AgentTool(
                        name=tool_name,
                        parameters={"type": "object", "properties": {}},
                        execute=execute,
                        approval_tier="read",
                    )
                ]
            )
            unbind = session.subscribe_requests(observed.append)
            try:
                await session.prompt("Look up the public outcome and report it.")
            finally:
                unbind()
            session_id = session.session_id
            [tool_message] = [
                msg for msg in session._context.messages
                if isinstance(msg, Message) and msg.tool_call_id == "toolu_fixture"
            ]
            if not accepted:
                assert tool_message.text == original
            if session._spend_persist_task is not None:
                await session._spend_persist_task
            checkpoint = session.transcript.latest_custom(SESSION_SPEND_CUSTOM_TYPE)
            if no_checkpoint:
                assert checkpoint is None
            elif case == "stale_checkpoint":
                assert checkpoint and "auxiliary_request_ids" not in checkpoint
            recorder, db_path = analytics_ledger
            recorder.flush_for_test()
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT purpose, request_id, cost_known, usage_reported, outcome, "
                    "output_tokens FROM calls WHERE session_id = ?",
                    (session_id,),
                ).fetchall()
            receipts = await auxiliary_usage(session)
            aux_rows = [row for row in rows if row[0] == "tool_conditioning"]
            aux_observed = [record for record in observed if record.purpose == "tool_conditioning"]
            assert all(host == "bedrock-runtime.ca-central-1.amazonaws.com" for host, *_ in calls)
            if no_attempt:
                assert not receipts and not aux_rows and not aux_observed
                assert len(calls) == 2
                assert OMITTED in json.dumps(calls[-1][2])
                if search:
                    assert assertions == [
                        (SEARCH, "toolu_fixture", fixture_url),
                        (SEARCH, "toolu_fixture", "https://example.org/unverified"),
                    ]
                return

            [receipt] = receipts
            [auxiliary] = aux_observed
            assert auxiliary.request_id == receipt.request_id
            if case == "abort_no_checkpoint":
                assert len(calls) == 2 and COMPANION in calls[-1][1]
                assert receipt.usage is None and receipt.outcome == "aborted"
                assert auxiliary.outcome == "aborted" and not auxiliary.usage_reported
                assert aux_rows == [("tool_conditioning", receipt.request_id, 0, 0, "aborted", 0)]
            else:
                assert len(rows) == 3
                assert len([record for record in observed if record.purpose == "turn"]) == 2
                turns = [
                    m for m in session._context.messages
                    if isinstance(m, Message) and m.role == "assistant"
                ]
                assert len(turns) == 2 and turns[-1].text == "Fixture answer"
                assert session._last_usage is not None
                assert session._last_usage.context_tokens == 1000
                assert session.spend.calls == 3
                final_prompt = json.dumps(calls[-1][2])
                if accepted:
                    assert len(calls) == 3 and COMPANION in calls[1][1]
                    assert receipt.usage is not None and receipt.usage.model_id == COMPANION
                    assert receipt.usage.estimated_usd_cost is not None
                    assert auxiliary.usage_reported and auxiliary.outcome == "stop" and auxiliary.ok
                    assert not calls[1][2].get("tools")
                    raw = await raw_public_evidence(session, "toolu_fixture")
                    assert raw == original
                    assert NEEDLE in final_prompt and OMITTED not in final_prompt
                    assert aux_rows == [("tool_conditioning", receipt.request_id, 1, 1, "stop", 4)]
                else:
                    outcome = "incomplete" if incomplete else "error"
                    assert len(calls) == (3 if incomplete else 2)
                    assert receipt.usage is None and receipt.outcome == outcome
                    observed_outcome = outcome if incomplete else "ProviderError"
                    assert auxiliary.outcome == observed_outcome and not auxiliary.usage_reported
                    assert auxiliary.usage is None and not auxiliary.ok
                    assert OMITTED in final_prompt
                    assert aux_rows == [
                        ("tool_conditioning", receipt.request_id, 0, 0, observed_outcome, 0)
                    ]
            if accepted and case == "complete":
                assert checkpoint and checkpoint["auxiliary_request_ids"] == [receipt.request_id]

        if case in {"complete", "no_checkpoint", "stale_checkpoint", "abort_no_checkpoint"}:
            monkeypatch.setattr(Transcript, "append_custom", append_custom)
            resume_spec = SessionSpec(
                hosting="bedrock", model=MAIN.model_id, resume=session_id, world_root=work
            )
            async with open_session(resume_spec, roots=roots) as reopened:
                assert isinstance(reopened, Session)
                assert reopened._public_evidence is None
                [recovered] = await auxiliary_usage(reopened)
                assert recovered.request_id == receipt.request_id
                if no_checkpoint:
                    reopened.refresh_frontend_usage()
                    for task in tuple(reopened._spend_tasks):
                        await task
                if reopened._spend_persist_task is not None:
                    await reopened._spend_persist_task
                recovered_checkpoint = reopened.transcript.latest_custom(SESSION_SPEND_CUSTOM_TYPE)
                assert recovered_checkpoint is not None
                assert recovered_checkpoint["auxiliary_request_ids"] == [receipt.request_id]
                if case in {"complete", "no_checkpoint"}:
                    assert reopened.spend.calls == 3  # Both main replies plus the companion.
                if case == "stale_checkpoint":
                    assert checkpoint is not None
                    assert reopened.spend.calls == checkpoint["calls"] + 1
                if accepted:
                    entry = reopened.spend.by_identity[f"bedrock/{COMPANION}"]
                    assert receipt.usage and receipt.usage.estimated_usd_cost is not None
                    assert entry["calls"] == 1
                    assert entry["micro"] == round(receipt.usage.estimated_usd_cost * 1_000_000)
                    raw = await raw_public_evidence(reopened, "toolu_fixture")
                    assert raw == original
                else:
                    assert reopened.spend.unpriced_calls == 1
            async with open_session(resume_spec, roots=roots) as reopened_again:
                assert isinstance(reopened_again, Session)
                assert reopened_again.spend.calls == recovered_checkpoint["calls"]
                assert reopened_again.spend.micro == recovered_checkpoint["micro"]
                assert reopened_again.spend.unpriced_calls == recovered_checkpoint["unpriced_calls"]


@pytest.mark.asyncio
async def test_unreadable_attempt_row_is_not_an_empty_rollup(wire):
    with wire.transcript.path.open("ab") as journal:
        journal.write(b'{"type":"custom","payload":{"custom_type":"auxiliary_usage_v1"\n')
    with pytest.raises(ValueError, match="candidate session journal record"):
        await auxiliary_usage(wire)
