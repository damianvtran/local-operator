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


@pytest.mark.asyncio
async def test_a_recall_set_already_visible_is_not_re_shipped_on_its_return(tmp_path):
    """The A -> empty -> A duplication trap, measured end-to-end on the wire.

    A suggestion set frozen into the first system block stays readable in every
    later request. When the next message does not match, the delta correctly
    ships the removal (``(empty)``); when a later message matches again, the set
    is byte-identical to what the model can already read, so it must NOT be
    appended a second time. The old baseline (the previous turn's DESIRED state)
    re-appended it, and a fixture measured 4 verbatim copies / 9.68% of the
    request by turn 8 for one distinct fact.

    The new-fact leg is the positive control: a suggestion set the model has NOT
    seen still ships, so the rule suppresses duplicates only.
    """
    root = tmp_path / "config"
    root.mkdir()
    (root / "config.yml").write_text(
        json.dumps({"values": {"memory": {"enabled": True, "capture": True, "suggest": True}}})
    )
    transcript = Transcript(root / "sessions" / "dedupe-session")
    built = []
    encoder = OpenAICompatClient("https://api.deepseek.com/v1")

    def stream(request, signal):
        built.append(copy.deepcopy(encoder._build_body(request)))

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
        model=ModelSpec(provider="deepseek", model_id="deepseek-chat", context_window=100000),
        stream_fn=stream,
        tools=[],
        transcript=transcript,
        system_blocks_provider=provider_fn,
        session_id="dedupe-session",
    )
    session._config_dir = root
    store = MemoryStore(root, MemoryIdentity("dedupe-session", "", "main"), transcript.directory)
    store.put(
        "session",
        {
            "title": "Mercury deployment",
            "body": "Mercury deployment proof docs/proof.md",
            "tags": ["mercury", "deployment", "proof"],
        },
    )

    def suggestions(body):
        blob = json.dumps(body["messages"], ensure_ascii=False)
        return blob.count("<memory_suggestions>")

    await session.prompt("Mercury deployment proof")
    assert suggestions(built[0]) == 1
    await session.prompt("Please order two pizzas for lunch")
    # The removal stays authoritative: it must still reach the model.
    state_texts = [m.get("content") for m in built[1]["messages"] if m.get("role") == "user"]
    assert any(
        isinstance(text, str) and "## Memory suggestions (untrusted)\n(empty)" in text
        for text in state_texts
    ), state_texts
    await session.prompt("Mercury deployment proof")
    # Same bytes as the frozen prefix copy -> exactly one copy in the request.
    assert suggestions(built[2]) == 1
    assert_preserved(built[0], built[1])
    assert_preserved(built[1], built[2])

    store.put(
        "session",
        {
            "title": "Mercury deployment rollback runbook",
            "body": "Rollback runbook docs/rollback.md",
            "tags": ["mercury", "deployment", "rollback"],
        },
    )
    await session.prompt("Mercury deployment rollback")
    # Positive control: a set the model has not seen still ships, so the count
    # grows to two and the earlier one is not repeated.
    assert suggestions(built[3]) == 2
    assert_preserved(built[2], built[3])
    await session.dispose()


def state_texts(body):
    """Every rendered ``[session-state]`` record in one request body."""
    return [
        message["content"]
        for message in body["messages"]
        if isinstance(message.get("content"), str) and "[session-state]" in message["content"]
    ]


@pytest.mark.asyncio
async def test_a_suggestion_a_compaction_cut_away_ships_again(tmp_path):
    """A text only a compacted-away record carried is NOT visible: it must ship.

    ``_visible_advisory_texts`` answers "can the model still read this?" from the
    frozen prefix and ``_context.messages`` — the LIVE context — and deliberately
    NOT from the transcript, which keeps every record forever. If it read the
    transcript the rule would silently become "never re-ship an identical set",
    and the two tests above would still pass because they never compact. This
    one does: a compaction cuts the record carrying the set, which also trips the
    ``lost_state`` re-anchor, and the set must reach the model again on the first
    turn that re-produces it.
    """
    from local_operator.compaction.api import CompactionSettings

    root = tmp_path / "config"
    root.mkdir()
    # capture off: the suggestion set under test must stay byte-identical, and
    # capture would append locator facts that can enter the recall ranking.
    (root / "config.yml").write_text(
        json.dumps({"values": {"memory": {"enabled": True, "capture": False, "suggest": True}}})
    )
    transcript = Transcript(root / "sessions" / "cut-away")
    built = []
    encoder = OpenAICompatClient("https://api.deepseek.com/v1")

    def stream(request, signal):
        built.append(copy.deepcopy(encoder._build_body(request)))

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
        model=ModelSpec(provider="deepseek", model_id="deepseek-chat", context_window=100000),
        stream_fn=stream,
        tools=[],
        transcript=transcript,
        system_blocks_provider=provider_fn,
        session_id="cut-away",
        compaction_settings=CompactionSettings(keep_recent_tokens=40, strategy="context-full"),
    )
    session._config_dir = root
    store = MemoryStore(root, MemoryIdentity("cut-away", "", "main"), transcript.directory)
    store.put(
        "session",
        {
            "title": "Mercury deployment",
            "body": "Mercury deployment proof docs/proof.md",
            "tags": ["mercury", "deployment", "proof"],
        },
    )

    def copies(body):
        return json.dumps(body["messages"], ensure_ascii=False).count("<memory_suggestions>")

    # Turn 0 freezes a prefix with NO suggestions: the set must reach the model
    # only through its own record, never through the prefix.
    await session.prompt("Please order two pizzas for lunch")
    assert copies(built[0]) == 0
    await session.prompt("Mercury deployment proof")
    carrier = session._system_state_message_id
    assert carrier is not None and copies(built[1]) == 1
    await session.prompt("Please order two more pizzas")
    await session.prompt("Filler turn " + "detail " * 40)
    assert (await session.compact_now()).ran
    # Premise, asserted rather than assumed: the compaction really cut the record
    # that carried the set, while the transcript (not consulted) still holds it.
    assert all(message.id != carrier for message in session._context.messages)

    await session.prompt("Mercury deployment proof")
    # One copy, and it is a fresh state record: the set shipped again.
    assert copies(built[-1]) == 1
    assert any("<memory_suggestions>" in text for text in state_texts(built[-1]))
    await session.dispose()


@pytest.mark.asyncio
async def test_a_goal_return_ships_beside_a_visible_copy(tmp_path):
    """The suppression stops at ``ADVISORY_TAIL_SECTIONS``: authoritative state ships.

    Goal A -> goal B -> goal A is where a wider rule would go wrong. A is still
    in the frozen prefix, so "the model can already read it" is true — but the
    last record the model read says B, and suppressing A would leave its most
    recent statement about the goal stale. That staleness is tolerable only for
    the declared untrusted hints, which is why the rule is bounded by the section
    set rather than by the scenario. The provider is scripted so A and B are the
    only variables.
    """
    root = tmp_path / "config"
    root.mkdir()
    (root / "config.yml").write_text(json.dumps({"values": {"memory": {"enabled": False}}}))
    transcript = Transcript(root / "sessions" / "goal-boundary")
    built = []
    encoder = OpenAICompatClient("https://api.deepseek.com/v1")

    def stream(request, signal):
        built.append(copy.deepcopy(encoder._build_body(request)))

        async def events():
            yield StreamTextDelta(delta="Recorded reply")
            yield StreamEndEvent(stop_reason="stop")

        return events()

    goal_a = "<goal>\nShip the memory hint fix\n</goal>"
    goal_b = "<goal>\nShip the release notes\n</goal>"
    script = {"tail": goal_a}

    def provider():
        return ["stable", "", "", script["tail"]]

    # The epoch/delta machinery is opted into by this attribute; ``setattr`` is the
    # repo's spelling for it because a plain function has no declared attribute.
    setattr(provider, "append_only_state", True)

    session = Session(
        model=ModelSpec(provider="deepseek", model_id="deepseek-chat", context_window=100000),
        stream_fn=stream,
        tools=[],
        transcript=transcript,
        system_blocks_provider=provider,
        session_id="goal-boundary",
    )
    session._config_dir = root

    def copies(body):
        return json.dumps(body["messages"], ensure_ascii=False).count("Ship the memory hint fix")

    await session.prompt("first turn")
    assert copies(built[0]) == 1  # the frozen prefix copy
    script["tail"] = goal_b
    await session.prompt("second turn")
    assert any("Ship the release notes" in text for text in state_texts(built[1]))
    script["tail"] = goal_a
    await session.prompt("third turn")
    # The older identical copy is visible in the prefix and must NOT suppress the
    # change: the intervening record said B, so A is new information again.
    assert copies(built[2]) == 2
    assert any("Ship the memory hint fix" in text for text in state_texts(built[2]))
    await session.dispose()
