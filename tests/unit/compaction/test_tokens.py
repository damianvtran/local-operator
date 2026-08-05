"""Token estimation: tiktoken path, fallback, memoization, invalidation."""

import pytest

from local_operator.compaction import tokens as tokens_mod
from local_operator.compaction.tokens import (
    IMAGE_TOKEN_ESTIMATE,
    clear_estimate_cache,
    estimate_messages_tokens,
    estimate_tokens,
    invalidate_message_cache,
    register_invalidator,
)
from local_operator.harness.types import ImageContent, Message, TextContent, ToolCall, Usage


@pytest.fixture(autouse=True)
def _clean_cache():
    """Every test starts with an empty estimate cache (memoization is global)."""
    clear_estimate_cache()
    yield
    clear_estimate_cache()


def test_text_tokens_positive_and_stable():
    """Same text yields the same positive estimate (cl100k or fallback)."""
    message = Message.user("The quick brown fox jumps over the lazy dog. " * 10)
    first = estimate_tokens(message)
    assert first > 0
    assert estimate_tokens(message) == first


def test_image_blocks_add_flat_estimate():
    """Each ImageContent costs IMAGE_TOKEN_ESTIMATE regardless of payload."""
    text_only = Message.user("hello")
    with_images = Message(
        role="user",
        content=[
            TextContent(text="hello"),
            ImageContent(data="A" * 10),
            ImageContent(data="B" * 100000),
        ],
    )
    assert estimate_tokens(with_images) - estimate_tokens(text_only) == 2 * IMAGE_TOKEN_ESTIMATE


def test_messages_tokens_is_sum():
    a = Message.user("alpha beta gamma")
    b = Message.assistant("delta epsilon zeta")
    assert estimate_messages_tokens([a, b]) == estimate_tokens(a) + estimate_tokens(b)


def test_tool_calls_contribute_tokens():
    """Tool-call names + arguments ride along on the wire and must count."""
    plain = Message.assistant("")
    with_call = Message.assistant("")
    with_call.tool_calls = [ToolCall(name="bash", arguments={"command": "ls -la /tmp"})]
    assert estimate_tokens(with_call) > estimate_tokens(plain)


def test_memoization_hits_cache_and_invalidation_recomputes():
    """Estimate is keyed on message id; invalidate forces recompute.

    Documents the settle rule: mutating in place WITHOUT invalidating serves
    the stale cached value; invalidation (which every mutator must call)
    refreshes it.
    """
    message = Message.user("word " * 50)
    before = estimate_tokens(message)
    assert message.id in tokens_mod._ESTIMATE_CACHE

    # In-place mutation without invalidation -> stale value (the rule).
    message.content = [TextContent(text="word " * 500)]
    assert estimate_tokens(message) == before

    invalidate_message_cache(message)
    after = estimate_tokens(message)
    assert after > before


def test_invalidation_notifies_subscribers_and_unsubscribe():
    """register_invalidator subscribers fire on invalidation; unsubscribe stops them."""
    seen: list[str] = []
    message = Message.user("abc")
    estimate_tokens(message)

    unsubscribe = register_invalidator(lambda m: seen.append(m.id))
    invalidate_message_cache(message)
    assert seen == [message.id]

    unsubscribe()
    invalidate_message_cache(message)
    assert seen == [message.id]  # no second notification


def test_fallback_estimator_on_tiktoken_failure(monkeypatch):
    """When tiktoken cannot load, degrade to len(text)//4 — never raise."""
    clear_estimate_cache()
    monkeypatch.setattr(tokens_mod, "_ENCODING", None)
    monkeypatch.setattr(tokens_mod, "_ENCODING_FAILED", True)
    message = Message.user("x" * 400)
    assert estimate_tokens(message) == 100


def test_settle_gate_unsettled_assistants_never_cached():
    """RC-20: assistants with provisional usage/stop_reason are computed but
    never cached, so a growing stream never serves a stale frozen count."""
    provisional = Message.assistant("partial " * 10)
    provisional.usage = None
    provisional.stop_reason = None
    before = estimate_tokens(provisional)
    assert before > 0
    assert provisional.id not in tokens_mod._ESTIMATE_CACHE  # never cached

    provisional.content = [TextContent(text="partial " * 30)]
    assert estimate_tokens(provisional) > before  # recomputed, not stale
    assert provisional.id not in tokens_mod._ESTIMATE_CACHE


def test_settle_gate_terminal_states():
    """Settled = usage set AND terminal stop_reason (not aborted/error)."""
    aborted = Message.assistant("x " * 10)
    aborted.usage = Usage(input_tokens=5, output_tokens=5)
    aborted.stop_reason = "aborted"
    estimate_tokens(aborted)
    assert aborted.id not in tokens_mod._ESTIMATE_CACHE

    errored = Message.assistant("x " * 10)
    errored.usage = Usage(input_tokens=5, output_tokens=5)
    errored.stop_reason = "error"
    estimate_tokens(errored)
    assert errored.id not in tokens_mod._ESTIMATE_CACHE

    settled = Message.assistant("x " * 10)
    settled.usage = Usage(input_tokens=5, output_tokens=5)
    settled.stop_reason = "stop"
    estimate_tokens(settled)
    assert settled.id in tokens_mod._ESTIMATE_CACHE


def test_user_messages_always_cache():
    """The settle gate is assistant-only: user messages cache at once."""
    user = Message.user("hello")
    estimate_tokens(user)
    assert user.id in tokens_mod._ESTIMATE_CACHE
