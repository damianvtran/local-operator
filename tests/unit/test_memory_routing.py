"""First-save guidance must work without a bank or a paid classifier.

A real same-prompt pilot hallucinated `lop memory remember/recall` when the guide
was only in the catalogue. These tests exercise the actual knowledge-tail renderer,
not a duplicate matcher or a forced provider tool choice.
"""

import pytest

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
