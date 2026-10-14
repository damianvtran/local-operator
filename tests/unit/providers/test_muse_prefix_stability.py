"""Client churn guards, not a benchmark or a claim about backend token packing.

Identical requests must encode identically. Growing conversations preserve logical
instructions/tools/input items, NOT the JSON envelope's byte prefix (``input`` is
before instructions and its closing bracket moves). Exercise the shipped Muse
stream/factory with an in-memory HTTP transport, not copied serializer functions.
"""

from __future__ import annotations

import copy
import json
from contextlib import asynccontextmanager

import httpx
import pytest

from local_operator.harness.types import (
    AgentTool,
    ChatRequest,
    Message,
    TextContent,
    ToolCall,
    ToolContext,
)
from local_operator.mcp.manager import McpManager
from local_operator.model.configure import build_model_spec
from local_operator.providers.clients import client_for_spec
from local_operator.providers.meta_muse_client import MetaMuseClient, tool_aliases
from local_operator.providers.replay import credential_scope, native_payload
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools import registry

KEY = "LLM|1234|fixture_prefix_subscription"
ENDPOINT = "https://api.meta.ai/v1/responses"
MODEL = build_model_spec("meta-muse", "muse-wonder-1.0")
# Deliberately independent of DEFAULT_TOOL_NAMES: deriving the expectation from
# that list would let an accidental default reorder pass undetected.
BUILTINS = (
    "bash read write edit glob grep eval lsp todo web_search web_read web_fetch "
    "wake patience task wait jobs hub send ask secret list_variables read_variable "
    "browser console agent team team_delete network project project_delete monitor "
    "sessions ask_withdraw generate_image code_requests no_reply generate_video "
    "video_status setting"
).split()


async def never_execute(*_args, **_kwargs):
    raise AssertionError("Serialization fixtures never execute tools")


def tool(name, properties=None):
    return AgentTool(
        name=name,
        execute=never_execute,
        parameters={"type": "object", "properties": properties or {}},
    )


def request(**updates):
    return ChatRequest(
        model=MODEL,
        max_tokens=16,
        messages=[Message.user("Inspect only.")],
        system_blocks=["Standing authority.", "Tools.", "Environment.", "Goal."],
        tools=[tool("a-b"), tool("a_b")],
    ).model_copy(update=updates)


def encoded(value):
    # HTTPX's shipped JSON encoder, with no canonical sort that could hide churn.
    return httpx.Request("POST", ENDPOINT, json=value).content


def equal(left, right):
    assert left == right


def detects(label, expected, broken):
    """Prove the very same equality guard rejects the changed-input control."""
    with pytest.raises(AssertionError):
        equal(expected, broken)
    print(f"NEGATIVE CONTROL {label}: equality guard rejected changed bytes")


class WireCapture:
    def __init__(self, http):
        self.client = client_for_spec(MODEL, http_client=http)
        assert isinstance(self.client, MetaMuseClient)
        self.wires = []

    async def stream(self, req, signal=None):
        async for event in self.client.stream(req, KEY):
            yield event

    async def send(self, req):
        events = [event async for event in self.stream(req)]
        assert events
        return json.loads(self.wires[-1])


@asynccontextmanager
async def capture():
    recorder = None

    def handle(wire):
        assert str(wire.url) == ENDPOINT
        assert wire.headers["authorization"] == f"Bearer {KEY}"
        assert recorder is not None
        recorder.wires.append(wire.content)
        frames = [
            {"type": "response.output_text.delta", "delta": "OK"},
            {
                "type": "response.completed",
                "response": {
                    "id": "fixture-response",
                    "output": [],
                    "usage": {"input_tokens": 4, "output_tokens": 1},
                },
            },
            {
                "type": "response.subscription_usage",
                "subscription": {
                    "window": {"used_percent": 7, "resets_at": 1788431188},
                },
            },
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join("data: " + json.dumps(frame) + "\n\n" for frame in frames),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        recorder = WireCapture(http)
        yield recorder


@pytest.mark.asyncio
async def test_equivalent_requests_repeat_wire_bytes_without_mutation():
    properties = {"first": {"type": "string"}, "second": {"type": "integer"}}
    req = request(tools=[tool("a-b", properties), tool("a_b")])
    before = req.model_dump()
    async with capture() as wire:
        await wire.send(req)
        first = wire.wires[-1]
        await wire.send(req.model_copy(deep=True))
        equal(first, wire.wires[-1])
        equal(before, req.model_dump())
        await wire.send(req.model_copy(update={"system_blocks": ["Changed authority."]}))
        detects("system prompt", first, wire.wires[-1])
        await wire.send(
            req.model_copy(
                update={"tools": [tool("a-b", dict(reversed(properties.items()))), tool("a_b")]}
            )
        )
        # Schema insertion order is not normalized by production; do not paper
        # over it with sort_keys in a guard supposedly measuring actual bytes.
        detects("schema property order", first, wire.wires[-1])


@pytest.mark.asyncio
async def test_registry_mcp_order_and_turn_latch_reach_stable_wire(tmp_path, monkeypatch):
    # Replace capability-dependent builders, not the ordering implementation.
    # Reversing table registration must NOT reverse the default builtin list.
    monkeypatch.setattr(
        registry,
        "TOOL_BUILDERS",
        {name: (lambda _context, name=name: tool(name)) for name in reversed(BUILTINS)},
    )
    context = ToolContext(cwd=str(tmp_path))
    builtins = registry.create_tools(context)
    assert [entry.name for entry in builtins] == BUILTINS
    manager = McpManager(tmp_path)
    manager._tools_by_server = {
        "z": [tool("mcp__z-last"), tool("mcp__z_first")],
        "a": [tool("mcp__a")],
    }
    inventory = builtins + manager.get_tools()
    assert [entry.name for entry in inventory[len(BUILTINS) :]] == [
        "mcp__a",
        "mcp__z-last",
        "mcp__z_first",
    ]
    async with capture() as wire:
        session = Session(
            model=MODEL,
            stream_fn=wire.stream,
            tools=inventory,
            transcript=Transcript(tmp_path / "ordered"),
            system_blocks_provider=lambda: ["Standing authority."],
        )
        try:
            session.set_tool_deferral(pins=tuple(entry.name for entry in inventory))
            inventory = list(session._tools)
            builtins = inventory[: len(BUILTINS)]
            first = await wire.send(request(tools=session._wire_tools()))
            manager._tools_by_server = {
                name: list(reversed(tools))
                for name, tools in reversed(list(manager._tools_by_server.items()))
            }
            equal(
                encoded(first["tools"]),
                encoded((await wire.send(request(tools=builtins + manager.get_tools())))["tools"]),
            )
            equal(
                tool_aliases(request(tools=inventory)),
                tool_aliases(request(tools=list(reversed(inventory)))),
            )
            session._tools = list(reversed(inventory))
            equal(
                encoded(first["tools"]),
                encoded((await wire.send(request(tools=session._wire_tools())))["tools"]),
            )
            # Next-turn publication is intentionally allowed to change. Reset
            # the real latch rather than freezing a stale session inventory.
            session._published_tools = None
            changed = await wire.send(request(tools=session._wire_tools()))
            detects(
                "builtin/tool order after turn boundary",
                encoded(first["tools"]),
                encoded(changed["tools"]),
            )
            unsorted = builtins + [
                entry for tools in manager._tools_by_server.values() for entry in tools
            ]
            changed = await wire.send(request(tools=unsorted))
            detects("MCP sorting bypass", encoded(first["tools"]), encoded(changed["tools"]))
        finally:
            await session.dispose()
            await manager.disconnect_all()


@pytest.mark.asyncio
async def test_persisted_prompt_delta_resume_and_authority_epoch_reach_wire(tmp_path):
    state = {"head": "Standing authority.", "date": "2026-10-14", "goal": "Inspect."}

    def provider():
        return [state["head"], "Tools.", state["date"], state["goal"]]

    setattr(provider, "append_only_state", True)
    path = tmp_path / "prompt"
    async with capture() as wire:
        session = Session(
            model=MODEL,
            stream_fn=wire.stream,
            tools=[],
            transcript=Transcript(path),
            system_blocks_provider=provider,
        )
        try:
            await session.prompt("Start.")
            assert len(wire.wires) == 1
            first = json.loads(wire.wires[-1])
            assert (
                first["instructions"] == "Standing authority.\n\nTools.\n\n2026-10-14\n\nInspect."
            )
            epoch = session._transcript.latest_custom_entry("system_prefix")
            state.update(date="2026-10-15", goal="Inspect src only.")
            await session.prompt("Continue.")
            later = json.loads(wire.wires[-1])
            equal(encoded(first["instructions"]), encoded(later["instructions"]))
            equal(encoded(first["input"]), encoded(later["input"][: len(first["input"])]))
            assert "2026-10-15" in json.dumps(later["input"])
            assert "Inspect src only." in json.dumps(later["input"])
            assert session._transcript.latest_custom_entry("system_prefix") is epoch
            count = len(session._context.messages)
            await session._prepare_system_blocks()
            assert len(session._context.messages) == count
            await session.dispose()
            session = Session(
                model=MODEL,
                stream_fn=wire.stream,
                tools=[],
                transcript=Transcript(path),
                system_blocks_provider=provider,
            )
            await session.prompt("Resume.")
            resumed = json.loads(wire.wires[-1])
            equal(encoded(later["instructions"]), encoded(resumed["instructions"]))
            equal(encoded(later["input"]), encoded(resumed["input"][: len(later["input"])]))
            state["head"] = "NEW authority: never write."
            await session.prompt("Use new authority.")
            changed = json.loads(wire.wires[-1])
            assert changed["instructions"].startswith(state["head"])
            new_epoch = session._transcript.latest_custom_entry("system_prefix")
            assert epoch is not None and new_epoch is not None and epoch.id != new_epoch.id
            detects(
                "authoritative prompt epoch",
                encoded(first["instructions"]),
                encoded(changed["instructions"]),
            )
        finally:
            await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mismatch", ["credential_scope", "endpoint", "model", "provider", "protocol", "visible"]
)
async def test_native_continuation_intact_aliases_and_scope(mismatch):
    opaque = {
        "type": "reasoning",
        "id": "rs-fixture",
        "summary": [],
        "encrypted_content": "opaque-fixture",
    }
    call = {
        "type": "function_call",
        "id": "fc-fixture",
        "call_id": "call-fixture",
        "name": "a-b",
        "arguments": '{"path":"a"}',
        "status": "completed",
    }
    calls = [ToolCall(id="call-fixture", name="a-b", arguments={"path": "a"})]
    payload = native_payload(
        MODEL,
        ENDPOINT,
        "openai-responses",
        [opaque, call],
        "",
        [{"id": "call-fixture", "name": "a-b", "args": {"path": "a"}}],
        credential_scope(KEY),
    )
    assistant = Message.assistant(
        "", tool_calls=calls, stop_reason="tool_use", provider_payload=payload
    )
    # Round-trip the actual durable representation, not an in-memory response.
    assistant = Message.model_validate_json(assistant.model_dump_json())
    result = Message(
        role="tool", tool_call_id="call-fixture", content=[TextContent(text="original result")]
    )
    req = request(messages=[Message.user("Inspect."), assistant, result])
    assert assistant.provider_payload is not None
    before = copy.deepcopy(assistant.provider_payload)
    async with capture() as wire:
        first = await wire.send(req)
        expected_call = {**call, "name": tool_aliases(req)["a-b"]}
        assert first["input"][1:3] == [opaque, expected_call]
        await wire.send(req.model_copy(deep=True))
        equal(wire.wires[0], wire.wires[-1])
        later = await wire.send(
            req.model_copy(update={"messages": [*req.messages, Message.user("Continue.")]})
        )
        equal(encoded(first["input"]), encoded(later["input"][: len(first["input"])]))
        # The adapter aliases native call names in-place through the parent
        # input list. Pin the opaque item and all other call fields intact,
        # without claiming whole-payload immutability the shipped path lacks.
        assert assistant.provider_payload is not None
        expected_payload = copy.deepcopy(before)
        expected_payload["native_replay"]["items"][1]["name"] = expected_call["name"]
        equal(expected_payload, assistant.provider_payload)
        altered = req.model_copy(deep=True)
        altered.messages[2].content = [TextContent(text="edited early result")]
        changed = await wire.send(altered)
        detects("early tool result", encoded(first["input"]), encoded(changed["input"]))
        invalid = req.model_copy(deep=True)
        assert invalid.messages[1].provider_payload is not None
        invalid.messages[1].provider_payload["native_replay"][mismatch] = "different provenance"
        changed = await wire.send(invalid)
        assert not any(
            item.get("encrypted_content") == "opaque-fixture" for item in changed["input"]
        )
        assert any(
            item.get("type") == "function_call" and item.get("name") == expected_call["name"]
            for item in changed["input"]
        )
        detects(f"native replay {mismatch}", encoded(first["input"]), encoded(changed["input"]))


@pytest.mark.asyncio
async def test_none_safety_dialect_omits_tools_on_actual_wire():
    async with capture() as wire:
        req = request(prompt_cache_key="synthetic-lineage")
        normal = await wire.send(req)
        assert normal["tools"]
        none = await wire.send(req.model_copy(update={"tool_choice": "none"}))
        assert "tools" not in none and "tool_choice" not in none
        assert none["store"] is False
        assert none["include"] == ["reasoning.encrypted_content"]
        assert "prompt_cache_key" not in none and "prompt_cache_retention" not in none
        equal(encoded(normal["instructions"]), encoded(none["instructions"]))
        equal(encoded(normal["input"]), encoded(none["input"]))
        # Safety intentionally wins over tools-prefix reuse on a side channel.
        detects("none versus automatic tools", wire.wires[0], wire.wires[1])
        with pytest.raises(AssertionError):
            assert "tools" not in normal
        print("NEGATIVE CONTROL none omission: omission guard rejected automatic tools")
