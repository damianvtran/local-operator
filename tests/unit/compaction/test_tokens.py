"""Token estimation: tiktoken path, fallback, memoization, invalidation."""

import pytest

from local_operator.compaction import tokens as tokens_mod
from local_operator.compaction.tokens import (
    IMAGE_TOKEN_ESTIMATE,
    approx_text_tokens,
    clear_estimate_cache,
    count_text_tokens,
    estimate_messages_tokens,
    estimate_tokens,
    invalidate_message_cache,
    messages_tokens_upper_bound,
    register_invalidator,
)
from local_operator.harness.types import (
    Content,
    ImageContent,
    Message,
    TextContent,
    ToolCall,
    Usage,
)


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


# --- Upper bound (the cheap pre-check the compaction trigger relies on) ------


def test_upper_bound_never_undercounts_across_scripts():
    """``messages_tokens_upper_bound`` must NEVER fall below the exact estimate.

    ``Session._maybe_compact`` substitutes this bound into the monotonic
    ``context_tokens > threshold`` test to avoid loading tiktoken's 43.6 MB BPE
    table on sessions nowhere near their threshold. That substitution is only
    sound while the bound genuinely dominates, so this sweeps the character
    classes where a naive ``len(text)`` bound breaks: a single CJK glyph or
    emoji is one code point but several cl100k tokens.

    Both argument-serialization branches are swept. ``_compute_tokens`` and the
    bound each read ``call.raw_arguments or json.dumps(call.arguments)``, and
    the streaming path is the one that populates ``raw_arguments`` — leaving it
    unset on every trial would test only the branch production does not take,
    and a bound that read ``arguments`` while the estimator read the raw string
    would pass.
    """
    import random

    alphabets = [
        "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJ0123456789 \n\t.,;:!?/\\-_=+*&^%$#@()[]{}",
        "日本語のテキストです。漢字とひらがな、カタカナ。",
        "🙂🚀🧠🔥💥🎉👩‍👩‍👧‍👦",
        "Ωμέγα ΑΒΓΔ ЖЗИЙ عربى עברית",
        "\u0001\u0002\ufffd\u200b\u2028",
    ]
    rng = random.Random(1234)

    def text() -> str:
        alphabet = rng.choice(alphabets)
        return "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 200)))

    for trial in range(400):
        blocks: list[Content] = [TextContent(text=text()) for _ in range(rng.randint(0, 4))]
        calls = []
        if rng.random() < 0.3:
            raw = text()[:60] if rng.random() < 0.5 else None
            calls = [
                ToolCall(
                    id=f"c{trial}",
                    name=text()[:20] or "t",
                    arguments={"k": text()[:50]},
                    raw_arguments=raw,
                )
            ]
        message = Message(
            role=rng.choice(["user", "assistant", "tool"]), content=blocks, tool_calls=calls
        )
        assert messages_tokens_upper_bound([message]) >= estimate_tokens(message)


def test_upper_bound_dominates_after_a_mutate_then_invalidate_round_trip():
    """The bound dominates ``estimate_messages_tokens`` only while the module's
    invalidation contract is honoured, so exercise the contract rather than
    assuming it.

    ``estimate_tokens`` memoizes settled messages on ``message.id``. Pruning
    blanks a message IN PLACE, keeping its id, so the cache would keep serving
    the pre-blank estimate — a value the bound on the new, tiny content has no
    reason to exceed, which would make the compaction pre-check claim a session
    is nowhere near its threshold when it is over it. The other bound tests all
    estimate fresh messages, so none of them touches the cached path at all.
    """
    message = Message.user("The quick brown fox jumps over the lazy dog. " * 200)
    before = estimate_messages_tokens([message])
    assert before > 0
    assert messages_tokens_upper_bound([message]) >= before

    # Blank it the way pruning._blank does: same object, same id, new content.
    message.content = [TextContent(text="[pruned]")]
    invalidate_message_cache(message)

    after = estimate_messages_tokens([message])
    assert after < before, "the cache still served the pre-blank estimate"
    assert messages_tokens_upper_bound([message]) >= after


def test_upper_bound_counts_images_and_is_additive():
    """Images are charged at the same flat rate as the exact estimator, and the
    bound sums over messages — an image-heavy history must not slip under."""
    blocks: list[Content] = [ImageContent(data="data:image/png;base64,AA==")]
    image = Message(role="user", content=blocks)
    assert messages_tokens_upper_bound([image]) == IMAGE_TOKEN_ESTIMATE
    assert messages_tokens_upper_bound([image, image]) == 2 * IMAGE_TOKEN_ESTIMATE
    assert messages_tokens_upper_bound([image]) >= estimate_tokens(image)


def test_upper_bound_does_not_load_tiktoken(monkeypatch):
    """The bound must reach its answer without the tokenizer.

    Pinned by making the encoding loader explode: any accidental call into the
    estimator from the bound turns into a hard failure instead of a silent
    43.6 MB regression.
    """

    def _boom():
        raise AssertionError("upper bound must not touch the tiktoken encoding")

    monkeypatch.setattr(tokens_mod, "_get_encoding", _boom)
    messages = [Message.user("hello " * 500), Message.assistant("world " * 500)]
    assert messages_tokens_upper_bound(messages) > 0


class TestApproxTextTokens:
    """The estimate for callers who cannot afford the exact ruler."""

    def test_it_never_reaches_the_encoding(self, monkeypatch) -> None:
        """The whole point. Loading cl100k_base costs ~43.6 MB RSS and, on a
        cold cache, a network fetch — so a status readout that wanted a number
        immediately was buying the most expensive object in the process."""

        def _boom():
            raise AssertionError("the approximation must not touch tiktoken")

        monkeypatch.setattr(tokens_mod, "_get_encoding", _boom)
        monkeypatch.setattr(tokens_mod, "_get_model_encoding", lambda _m: _boom())
        assert approx_text_tokens("hello world " * 200) > 0

    def test_empty_text_is_zero(self) -> None:
        assert approx_text_tokens("") == 0

    def test_it_tracks_the_exact_count_closely_enough_for_a_percentage(self) -> None:
        """A reading rendered to one decimal of a context window tolerates a few
        percent; it does not tolerate being absent. Measured at +7.0% against
        cl100k_base on a real system prompt plus tool inventory."""
        text = (
            "You are a careful assistant. Read the repository before editing. "
            '{"type":"object","properties":{"path":{"type":"string"}}}'
        ) * 40
        exact = count_text_tokens(text)
        approx = approx_text_tokens(text)
        assert exact > 0
        assert abs(approx - exact) / exact < 0.25
