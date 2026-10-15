"""Real host-built system + actual provider encoders; no remote inference/cache claim."""

import asyncio
import copy
import hashlib
import json

import pytest

from local_operator.harness.types import ModelSpec, StreamEndEvent, StreamTextDelta
from local_operator.memory import MemoryIdentity, MemoryStore
from local_operator.memory_capture import finish
from local_operator.providers.clients import OpenAICompatClient
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.session_factory import _KnowledgeHooks, _make_system_blocks_provider


def encoded_message_bytes(body):
    return [
        json.dumps(row, separators=(",", ":"), ensure_ascii=False).encode()
        for row in body["messages"]
    ]


def assert_preserved(previous, following):
    old = encoded_message_bytes(previous)
    new = encoded_message_bytes(following)
    assert new[: len(old)] == old, "Previously committed provider messages changed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base,provider,model_id",
    [
        ("https://api.deepseek.com/v1", "deepseek", "deepseek-chat"),
        ("https://openrouter.ai/api/v1", "openrouter", "deepseek/deepseek-chat"),
    ],
)
async def test_memory_is_tail_only_in_real_host_wire_with_negative_control(
    tmp_path, base, provider, model_id
):
    root = tmp_path / "config"
    root.mkdir()
    (root / "config.yml").write_text(
        json.dumps({"values": {"memory": {"enabled": True, "capture": True, "suggest": True}}})
    )
    transcript = Transcript(root / "sessions" / "wire-session")
    built = []
    requests = []
    encoder = OpenAICompatClient(base)

    def stream(request, signal):
        # Snapshot AT DISPATCH, not a reference that later assistant mutations
        # could silently change under the prefix comparator.
        built.append(copy.deepcopy(encoder._build_body(request)))
        requests.append(request.model_copy(deep=True))

        async def events():
            yield StreamTextDelta(delta="Recorded reply")
            yield StreamEndEvent(stop_reason="stop")

        return events()

    provider_fn = _make_system_blocks_provider(
        [],
        transcript,
        _KnowledgeHooks(),
        cwd=str(tmp_path),
        user_instructions="Keep the user's prefix stable.",
        repo_guidance="Synthetic repository guidance.",
    )
    session = Session(
        model=ModelSpec(provider=provider, model_id=model_id, context_window=100000),
        stream_fn=stream,
        tools=[],
        transcript=transcript,
        system_blocks_provider=provider_fn,
        session_id="wire-session",
    )
    session._config_dir = root
    store = MemoryStore(
        root, MemoryIdentity("wire-session", "", "main"), transcript.directory, origin="wire-device"
    )
    await session.prompt("Initial ordinary conversation")
    committed_first = copy.deepcopy(
        encoder._build_body(
            requests[-1].model_copy(
                update={"messages": session._convert_to_llm(session._context.messages)},
            )
        )
    )
    item = store.put(
        "session",
        {
            "title": "Mercury deployment",
            "body": "Mercury deployment proof docs/proof.md",
            "tags": ["mercury", "deployment", "proof"],
        },
    )
    await session.prompt("Mercury deployment proof")
    assert_preserved(built[0], built[1])
    assert_preserved(committed_first, built[1])
    committed_second = copy.deepcopy(
        encoder._build_body(
            requests[-1].model_copy(
                update={"messages": session._convert_to_llm(session._context.messages)},
            )
        )
    )
    assert "memory_suggestions" in json.dumps(built[1])
    # Every committed prior message remains the prefix, not only block zero or
    # a convenient substring at the start of the old user's text.
    store.put(
        "session",
        {
            "title": "Mercury deployment corrected",
            "body": "Mercury proof docs/corrected.md",
            "tags": ["mercury", "proof"],
        },
        entity=item["id"],
        expected_version=item["version"],
    )
    await session.prompt("Mercury proof again")
    assert_preserved(built[1], built[2])
    assert_preserved(committed_second, built[2])
    mutated = copy.deepcopy(built[2])
    mutated["messages"].insert(
        0, {"role": "system", "content": "Memory regenerated ahead of old history"}
    )
    with pytest.raises(AssertionError):
        assert_preserved(built[1], mutated)
    print(
        "WIRE",
        provider,
        "old prefix messages",
        len(built[1]["messages"]),
        "sha256",
        hashlib.sha256(b"".join(encoded_message_bytes(built[1]))).hexdigest(),
        "append-only PASS; inserted-prefix negative rejected",
    )
    await session.dispose()


@pytest.mark.asyncio
async def test_idle_capture_completion_emits_no_agent_start_or_request(tmp_path):
    requests = []

    def stream(request, signal):
        requests.append(request)

        async def events():
            yield StreamTextDelta(delta="Deliberate turn control")
            yield StreamEndEvent(stop_reason="stop")

        return events()

    transcript = Transcript(tmp_path / "sessions" / "idle-session")
    session = Session(
        model=ModelSpec(provider="test", model_id="deterministic", context_window=100000),
        stream_fn=stream,
        tools=[],
        transcript=transcript,
        system_blocks_provider=lambda: ["stable"],
    )
    events = []
    session.subscribe(events.append)
    store = MemoryStore(
        tmp_path / "config",
        MemoryIdentity("idle-session", "", "main"),
        transcript.directory,
        origin="idle-device",
    )
    gate = asyncio.Event()

    async def capture():
        await gate.wait()
        return [{"title": "Idle locator", "body": "docs/idle.md", "tags": ["idle"]}]

    candidate = asyncio.create_task(capture())
    completed = asyncio.create_task(finish(store, "committed-idle-marker", candidate, ""))
    before = (len(requests), sum(event.type == "agent_start" for event in events))
    gate.set()
    await completed
    after = (len(requests), sum(event.type == "agent_start" for event in events))
    assert before == after == (0, 0)
    # Positive instrument control: a deliberate turn changes both counters.
    await session.prompt("Deliberate wake control")
    with pytest.raises(AssertionError):
        assert before == (len(requests), sum(event.type == "agent_start" for event in events))
    print("IDLE capture request/agent_start delta=0/0; deliberate-turn control rejected")
    await session.dispose()
