"""First-save guidance must work without a bank or a paid classifier.

A real same-prompt pilot hallucinated `lop memory remember/recall` when the guide
was only in the catalogue. These tests exercise the actual knowledge-tail renderer,
not a duplicate matcher or a forced provider tool choice.
"""

import pytest

from local_operator.config import ConfigManager
from local_operator.session_factory import (
    _KnowledgeHooks,
    _memory_recommendation,
    _select_knowledge_block,
)

SAVE_PROMPT = (
    "Remember these synthetic work facts for future sessions: the Kestrel rollout goal is "
    "to ship Friday; its runbook is docs/kestrel.md; updates go to Slack channel #kestrel-ops "
    "in thread https://example.invalid/threads/kestrel; contact Mira Vale at "
    "mira@example.invalid is the rollout owner. Keep them available after context "
    "compaction, and report what you retained."
)
RECALL_PROMPT = (
    "What is the Kestrel rollout goal, where is its runbook, which channel and thread "
    "carry updates, and who owns it? Look up retained knowledge rather than inventing it."
)


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [SAVE_PROMPT, RECALL_PROMPT, "Save the key decisions and files"])
async def test_empty_bank_no_classifier_gets_readable_guide_in_actual_tail(query, tmp_path):
    hooks = _KnowledgeHooks()  # classifier/index absent, no store or paid backend
    block = await _select_knowledge_block(hooks, query, task_id="synthetic-user")
    assert "<resource_recommendations>" in block and "- guide://memory" in block
    assert block.count("- guide://memory") == 1
    assert list(tmp_path.iterdir()) == [], "routing does not open/create a bank"
    assert await _select_knowledge_block(hooks, query, task_id="synthetic-user") == block


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "How much RAM does this process use?",
        "Fix the SQLite memory store implementation",
        "Save the SQL query plan to a file",
        "Look up the agent profile in the registry",
        "What is the database memory profile?",
        "Remember to take a break",
        "Write a parser for these log files",
        "What is a memory-mapped file?",
    ],
)
async def test_no_memory_intent_stays_quiet(query):
    assert _memory_recommendation(query) is None
    assert "guide://memory" not in await _select_knowledge_block(
        _KnowledgeHooks(), query, task_id="negative-user"
    )


PERSONAL_SAVE_PROMPT = (
    "Please remember a few things about me so you can help me later: my name is Elara Venn, "
    "I live in Dunedin, New Zealand, and my email address is elara.venn@example.invalid."
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        PERSONAL_SAVE_PROMPT,
        "Remember my email is user@example.invalid",
        "Store where I live: a coastal town",
        "Retain things about me for later",
    ],
)
async def test_personal_first_save_routes_without_bank_or_classifier(query):
    assert "guide://memory" in await _select_knowledge_block(
        _KnowledgeHooks(), query, task_id="personal"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "Remember the SQL memory profile implementation",
        "Store my email parser code",
        "Save the address validation schema",
        "Remember my name field validation implementation",
        "How do I profile RAM usage?",
        "Fix the user profile implementation",
    ],
)
async def test_technical_personal_word_neighbors_do_not_route(query):
    assert "guide://memory" not in await _select_knowledge_block(
        _KnowledgeHooks(), query, task_id="technical"
    )


@pytest.mark.asyncio
async def test_personal_message_reaches_native_provider_recording_with_classification_off(
    tmp_path,
):
    """No paid provider: record the actual factory-built Session request, not a hint helper.

    Shipped recommendation delivery is allowed at the tail; the frozen system prefix
    must remain byte-identical and no first-save bank can be needed to recommend it.
    """
    import argparse
    import json

    from local_operator.agents import AgentRegistry
    from local_operator.harness.types import StreamEndEvent, StreamTextDelta
    from local_operator.session.runtime.publication import PublicationGate
    from local_operator.session.session import Session
    from local_operator.session_factory import create_session

    root = tmp_path / "config"
    root.mkdir()
    (root / "config.yml").write_text(json.dumps({"values": {"classification": {"auto": False}}}))
    requests = []
    session = await create_session(
        argparse.Namespace(
            hosting="test",
            model="test-model",
            agent_name=None,
            agent_id=None,
            yolo=True,
            train=False,
        ),
        ConfigManager(root),
        AgentRegistry(root),
        cwd=str(tmp_path),
        world_root=tmp_path,
        has_ui=False,
        defer_mcp_wiring=True,
        mcp_publication_gate=PublicationGate(),
    )

    def recording(request, signal):
        requests.append(request.model_copy(deep=True))

        async def events():
            yield StreamTextDelta(delta="Recorded")
            yield StreamEndEvent(stop_reason="stop")

        return events()

    assert isinstance(session, Session)
    session._stream_fn = recording
    try:
        await session.prompt("Hello")
        assert not (root / "memory").exists()
        await session.prompt(PERSONAL_SAVE_PROMPT)
        assert len(requests) == 2
        assert "guide://memory" in requests[-1].model_dump_json()
        assert requests[0].system_blocks == requests[1].system_blocks
        assert not (
            root / "memory"
        ).exists(), "zero-provider first-save routing never creates a store"
    finally:
        await session.dispose()
