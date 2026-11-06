"""Wire contract for the openai-chat recorded-replay gate.

INCIDENT: a subagent's turn 40 confabulated a stranger's project. The streamed
thought block was stored (``providers/replay.py``) and then re-attached to that
message on every later request, so ONE derailed turn became durable context. The
route was ``radient/auto`` -- served by Together as
``deepseek-ai/DeepSeek-V4.1-Flash-1017`` -- where ``requires_reasoning_echo`` is False
(the client only ever sees the router's id ``auto``, so no family resolves), and
the thought was replayed anyway: the body builder copied any native block whose
provenance matched, with no gate at all.

What is pinned HERE is the wire -- what turn N+1 actually carries -- driven
through the real ``OpenAICompatClient`` with scripted SSE. The rule itself
(per-route answers, the exception table with its citations, and the recovery
composition) is pinned beside it in ``tests/unit/model/test_configure.py``.

SCOPE: the ``openai-chat`` wire only. The responses wire re-attaches provider-side
ENCRYPTED state, anthropic requires thinking blocks back complete and unmodified,
and google needs its ``thoughtSignature``; none of them is gated by this rule, and
applying it to them by analogy would break all three.
"""

import json
from typing import Any

import httpx
import pytest

from local_operator.harness.types import (
    ChatRequest,
    Message,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolCall,
)
from local_operator.model import configure
from local_operator.model.configure import build_model_spec
from local_operator.providers.clients import OpenAICompatClient
from local_operator.providers.context import (
    ContextBinding,
    ContextTokenTracker,
    measure_request,
)
from local_operator.providers.replay import (
    REASONING_ECHO_PLACEHOLDER,
    native_payload,
)

DEEPSEEK_BASE = "https://api.deepseek.com/v1"
RADIENT_BASE = "https://api.radienthq.com/v1"
XAI_BASE = "https://api.x.ai/v1"

#: A side-effect-free tool. Nothing here runs it: the harness answers the call
#: itself with a fixed string, so this file cannot touch the machine.
CLOCK = "clock"

#: The one credential scope every request in this file is built and captured
#: under, so a provenance mismatch cannot quietly drop the replay under test.
SCOPE = "fixture"


@pytest.fixture(autouse=True)
def isolated_metadata(monkeypatch, tmp_path):
    """Keep ``build_model_spec`` on the static registry, as the deepseek suite does."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    configure.invalidate_model_info_cache()
    yield
    configure.invalidate_model_info_cache()


def _sse(events: list[dict[str, Any]]) -> bytes:
    """An SSE body ending in the ``[DONE]`` terminator a direct route demands.

    DeepSeek's chat protocol rejects a stream that ends without it (natural EOF
    reads as a truncated response), and every compatible route here accepts it,
    so one scripted body serves all three transports under test.
    """
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return (body + "data: [DONE]\n\n").encode()


def _chat_reply(call_id: str, *, thought: str, text: str) -> list[dict[str, Any]]:
    """One scripted assistant reply: a thought, visible text, and a tool call."""
    return [
        {"id": "chatcmpl-gate", "choices": [{"delta": {"reasoning_content": thought}}]},
        {
            "choices": [
                {
                    "delta": {
                        "content": text,
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call_id,
                                "type": "function",
                                "function": {"name": CLOCK, "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {"usage": {"prompt_tokens": 11, "completion_tokens": 5}},
    ]


def _assistant_entries(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in body["messages"] if entry.get("role") == "assistant"]


def _replayed_thoughts(body: dict[str, Any]) -> list[str]:
    """The native thoughts the body puts back on the wire, in message order.

    Reads BOTH keys the compat wire uses for a recorded thought: routes that emit
    the OpenAI spelling and the ones that emit the legacy ``reasoning`` key end up in the
    same place, and a gate that only suppressed one of them would be a hole.
    """
    replayed = []
    for entry in _assistant_entries(body):
        for key in ("reasoning_content", "reasoning"):
            value = entry.get(key)
            if isinstance(value, str) and value:
                replayed.append(value)
                break
    return replayed


async def _drive(
    base_url: str, model: Any, *, thought: str = "private plan", turns: int = 3
) -> list[dict[str, Any]]:
    """Run a multi-turn tool conversation; return the bodies the client SENT.

    Every turn is answered with a thought plus a call to a side-effect-free tool
    that the harness answers itself, so every request after the first carries
    assistant turns with recorded native state -- which is exactly what the gate
    decides about. Bodies are captured from the TRANSPORT, so nothing can pass
    here that the client would not have put on the wire.
    """
    bodies: list[dict[str, Any]] = []
    history: list[Message] = [Message.user("read the clock once per turn, three turns")]
    served = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal served
        bodies.append(json.loads(request.content))
        served += 1
        return httpx.Response(
            200,
            content=_sse(
                _chat_reply(
                    f"call_{served}", thought=f"{thought} {served}", text=f"checking {served}"
                )
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = OpenAICompatClient(base_url, http_client=http)
        for _ in range(turns):
            request = ChatRequest(model=model, messages=list(history), system_blocks=["stable"])
            events = [event async for event in client.stream(request, SCOPE)]
            terminal = next(e for e in reversed(events) if isinstance(e, StreamEndEvent))
            # Assemble the call the way the client itself did when it wrote the
            # payload: the id and name arrive on one delta and the argument
            # fragments on later ones, so a reconstruction that took only the
            # delta carrying a name loses the id -- and the recorded ``visible``
            # fingerprint covers the id, so the replay would be refused for a
            # reason that has nothing to do with this gate.
            parts: dict[int, dict[str, Any]] = {}
            for event in events:
                if not isinstance(event, StreamToolCallDelta):
                    continue
                slot = parts.setdefault(event.index, {"id": "", "name": "", "args": []})
                if event.id:
                    slot["id"] = event.id
                if event.name:
                    slot["name"] += event.name
                if event.argument_delta:
                    slot["args"].append(event.argument_delta)
            calls = [
                ToolCall(
                    id=slot["id"],
                    name=slot["name"],
                    arguments=json.loads("".join(slot["args"]) or "{}"),
                )
                for slot in parts.values()
                if slot["name"]
            ]
            # The capture is only evidence if the reply really was recorded: a
            # fingerprint or provenance mismatch drops the payload silently, and
            # an assertion about "no replay" would then pass for the wrong reason.
            assert terminal.provider_payload is not None, "the parsed reply carried no payload"
            assert calls, "the scripted reply produced no tool call"
            history.append(
                Message.assistant(
                    "".join(e.delta for e in events if isinstance(e, StreamTextDelta)),
                    tool_calls=calls,
                    stop_reason=terminal.stop_reason,
                    provider_payload=terminal.provider_payload,
                    usage=terminal.usage,
                )
            )
            history.append(
                Message(
                    role="tool",
                    tool_call_id=calls[0].id,
                    tool_name=CLOCK,
                    content=[TextContent(text="5")],
                )
            )
    return bodies


def _recorded_turn(model: Any, base_url: str, *, thought: str, text: str = "done") -> Message:
    """An assistant turn carrying real recorded native state, as resume loads it.

    The payload is built by the same helper the wire parser uses, with the
    endpoint and scope the client itself will derive -- a spelling that differs by
    one character drops the replay for real rather than failing an assertion.
    """
    return Message.assistant(
        text,
        provider_payload=native_payload(
            model,
            f"{base_url}/chat/completions",
            "openai-chat",
            [{"reasoning_content": thought}],
            text,
            [],
            SCOPE,
        ),
    )


@pytest.mark.asyncio
async def test_direct_deepseek_continuity_is_unchanged_across_turns():
    """The family the echo contract was MEASURED on still replays every thought.

    WHAT THIS PROTECTS: continuity on the family-known route. Turn N+1 must carry
    each earlier assistant turn's recorded thought BYTE-FOR-BYTE (the exact text
    the wire produced), because omitting it is the refusal
    ``requires_reasoning_echo`` exists to prevent -- asking for the echo while dropping
    the real thoughts would be the worst of both. The gate must be a NO-OP here,
    which is what "before and after are identical" means for this route.
    """
    model = build_model_spec("deepseek", "deepseek-flash")
    assert model.requires_reasoning_echo is True, "the direct thinking route must keep the echo"

    bodies = await _drive(DEEPSEEK_BASE, model)

    assert _replayed_thoughts(bodies[0]) == []
    assert _replayed_thoughts(bodies[1]) == ["private plan 1"]
    assert _replayed_thoughts(bodies[2]) == ["private plan 1", "private plan 2"]
    # The exact bytes, not merely "something": the stored thought is what the
    # earlier turn's stream produced.
    assert _assistant_entries(bodies[2])[0]["reasoning_content"] == "private plan 1"


@pytest.mark.asyncio
async def test_the_incident_route_carries_no_thought_at_all():
    """``radient/auto`` -- the incident route -- no longer re-attaches the thought.

    The router's id is not a model: ``auto`` resolves to no family, so the echo
    requirement is False and nothing requires the replay. This is the case that
    made one confabulated turn durable context across 40+ turns of a subagent run;
    turn 2 here is the smallest reproduction of it.
    """
    model = build_model_spec("radient", "auto")
    assert model.requires_reasoning_echo is False, "the router id must resolve no family"

    bodies = await _drive(RADIENT_BASE, model)

    assert _replayed_thoughts(bodies[1]) == []
    assert _replayed_thoughts(bodies[2]) == []
    # Neither spelling of the key may appear on any assistant turn.
    for body in bodies:
        for entry in _assistant_entries(body):
            assert "reasoning_content" not in entry
            assert "reasoning" not in entry


@pytest.mark.asyncio
async def test_the_recorded_exception_keeps_the_replay_on_the_grok_wire():
    """xAI keeps its replay by RECORDED EXCEPTION, not by the echo requirement.

    ``docs/XAI_CACHING.md:155-165`` records the live bench and the vendor's own
    wording: an omitted previous thought is the top documented cause of cache
    misses there, so suppressing this replay buys correctness nothing and costs a
    measured cache hit rate. Nothing about grok refuses a request without it --
    that is why it is an exception rather than a family rule -- and the citation
    lives beside the gate in ``model.configure.NATIVE_REPLAY_EXCEPTIONS``.
    """
    model = build_model_spec("xai", "grok-4.6")
    assert model.requires_reasoning_echo is False, "grok does not REQUIRE the echo"

    bodies = await _drive(XAI_BASE, model)

    assert _replayed_thoughts(bodies[1]) == ["private plan 1"]
    assert _replayed_thoughts(bodies[2]) == ["private plan 1", "private plan 2"]


def test_the_fill_recovery_re_enables_both_halves_it_arms():
    """A spec the loop ARMED keeps the fill AND the replay it just started sending.

    ``harness/loop.py`` sets ``requires_reasoning_echo`` on the RESOLVED spec when a
    provider refuses a request for a missing echo. That is direct evidence about a
    route no family rule could know, and the gate takes the same flag, so the
    recovery's own retry must carry a placeholder on the blank turn AND the real
    thought on the recorded one. A gate that ignored the flag would fill an echo
    and then deny the thoughts the same validator asked for.
    """
    model = build_model_spec("radient", "auto").model_copy(
        update={"requires_reasoning_echo": True}
    )
    blank = Message.assistant("considered it")
    request = ChatRequest(
        model=model,
        messages=[
            Message.user("go"),
            _recorded_turn(model, RADIENT_BASE, thought="recorded thought", text="first"),
            blank,
        ],
        system_blocks=["stable"],
    )

    body = OpenAICompatClient(RADIENT_BASE)._build_body(request, scope=SCOPE)

    entries = _assistant_entries(body)
    assert entries[0]["reasoning_content"] == "recorded thought"
    assert entries[1]["reasoning_content"] == REASONING_ECHO_PLACEHOLDER


def test_accounting_counts_replayed_bytes_only_where_the_body_carries_them():
    """``bind_native_context`` must agree with the gate, not restate it.

    The count is an ESTIMATE OF INPUT the provider is about to read. On a route
    whose body suppresses the replay, counting those bytes over-reports the call's
    own input and sends the calibration drifting -- so the two read one predicate:
    the body carries the thought exactly when the accounting counts it.
    """
    thought = "a recorded thought of exactly forty bytes.."  # 43 chars -> 10 tokens
    expected = len(thought) // 4

    for provider, base_url, replays in (
        ("deepseek", DEEPSEEK_BASE, True),
        ("radient", RADIENT_BASE, False),
    ):
        model = build_model_spec(provider, "deepseek-flash" if replays else "auto")
        request = ChatRequest(
            model=model,
            messages=[
                Message.user("go"),
                _recorded_turn(model, base_url, thought=thought),
            ],
            system_blocks=["stable"],
        )
        request.context_binding = ContextBinding(ContextTokenTracker(), measure_request(request))

        body = OpenAICompatClient(base_url)._build_body(request, scope=SCOPE)

        assert bool(_replayed_thoughts(body)) is replays, (provider, body["messages"])
        counted = request.context_binding.measured.native_tokens
        assert counted == (expected if replays else 0), (provider, counted)
