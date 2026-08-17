"""Token estimation: tiktoken path, fallback, memoization, invalidation."""

import time

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


def test_estimate_cache_evicts_least_recent_entry_at_its_hard_bound():
    """Long-lived servers must retain one turn's working set, not every session."""
    messages = [
        Message.user(f"message {index}") for index in range(tokens_mod._ESTIMATE_CACHE_MAX + 1)
    ]
    for message in messages:
        estimate_tokens(message)

    assert len(tokens_mod._ESTIMATE_CACHE) == tokens_mod._ESTIMATE_CACHE_MAX
    assert messages[0].id not in tokens_mod._ESTIMATE_CACHE
    assert messages[-1].id in tokens_mod._ESTIMATE_CACHE


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
        """The claim is narrow on purpose: this payload, this direction.

        Skipped without a working ENCODING, which is NOT paranoia: whenever
        ``_get_encoding()`` returns None, ``count_text_tokens`` degrades to the
        identical ``len(text) // 4`` expression, so both assertions below hold
        by identity and this test measures nothing while looking green.

        Guarding the import alone was not enough. ``_get_encoding`` also
        swallows a failure from ``tiktoken.get_encoding("cl100k_base")``, which
        on a cold cache is a network fetch and offline is a connection timeout
        — tiktoken imports fine, ``importorskip`` does not fire, and the
        comparison is an identity again. The same offline-cold-cache case the
        estimator's own docstring calls out.

        The bound is 20% because the payload below measures +17.3% — not
        because 20% is a general property of ``chars // 4``. On other content
        the same function is off by -66% (CJK) to +40% (English prose), which
        is why the docstring tabulates rather than advertises a single figure,
        and why this test names the payload it is calibrated against.

        The direction matters as much as the size. Over-counting is the safe
        error for a context readout: it reports less headroom than there really
        is, so the failure mode is compacting slightly early rather than
        promising room that does not exist.
        """
        pytest.importorskip("tiktoken")
        if tokens_mod._get_encoding() is None:
            pytest.skip("cl100k_base unavailable — the comparison would be an identity")
        text = (
            "You are a careful assistant. Read the repository before editing. "
            '{"type":"object","properties":{"path":{"type":"string"}}}'
        ) * 40
        exact = count_text_tokens(text)
        approx = approx_text_tokens(text)
        assert exact > 0
        assert approx >= exact, "a context readout must not under-report a Latin+JSON payload"
        assert (approx - exact) / exact < 0.20


class TestThreadSafetyForOffloadedEstimation:
    """The estimators run in worker threads now; the shared cache must hold.

    Compaction's rulers are the largest synchronous stretch in a turn, and one
    event loop serves the parent session, every subagent and the TUI repaint —
    so counting inline made one agent's threshold check stall all of them
    (measured: 2.5 s of loop stall with eight children running, 116 of 121
    stall samples inside the encoder). ``Session._offloaded`` moves large
    histories to ``asyncio.to_thread``, which makes this module's module-level
    LRU cache and lazy encoding singleton reachable from several threads at
    once. These tests pin the properties that makes safe.
    """

    def test_history_chars_sizes_both_message_kinds(self):
        """The offload decision is made on this probe, so it must not raise on
        a CustomMessage — the cut-point walker sees those, and an exception
        here would abort a compaction rather than merely mis-size it."""
        from local_operator.compaction.tokens import history_chars
        from local_operator.harness.types import CustomMessage

        messages = [
            Message.user("abcde"),
            CustomMessage(custom_type="compaction_summary", details={"summary": "x" * 100}),
        ]
        # Counts the text block, tolerates the content-less custom entry.
        assert history_chars(messages) == 5

    def test_concurrent_estimation_agrees_with_serial_estimation(self):
        """Same messages, many threads, one answer.

        Guards the LRU sequences (``get`` + ``move_to_end``, and insert +
        ``popitem`` eviction) that are individually atomic but not atomic
        together: an unguarded interleaving can evict an entry another thread
        is mid-promotion on, which shows up as a wrong total rather than a
        crash.
        """
        import concurrent.futures

        messages = [Message.user(f"message {i} " + "lorem ipsum dolor " * 50) for i in range(40)]
        expected = estimate_messages_tokens(messages)
        clear_estimate_cache()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: estimate_messages_tokens(messages), range(24)))

        assert set(results) == {expected}

    def test_cache_stays_within_its_bound_under_concurrent_writers(self):
        """Eviction must not be defeated (or overrun) by concurrent inserts:
        the bound is what keeps a long-running process from leaking an entry
        per message ever estimated."""
        import concurrent.futures

        def work(batch: int) -> None:
            estimate_messages_tokens(
                [Message.user(f"b{batch} m{i} text text text") for i in range(200)]
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(work, range(8)))

        assert len(tokens_mod._ESTIMATE_CACHE) <= tokens_mod._ESTIMATE_CACHE_MAX

    def test_the_encoding_singleton_loads_once_under_a_thread_race(self):
        """Two workers reaching a cold singleton together must not both pay the
        table load (~60 ms and ~44 MB RSS) nor both log the failure warning."""
        import concurrent.futures

        loads: list[int] = []
        real_import = tokens_mod._get_encoding

        tokens_mod._ENCODING = None
        tokens_mod._ENCODING_FAILED = False

        class _Sentinel:
            def encode(self, text, **_kwargs):
                return [0] * (len(text) // 4)

        def fake_get_encoding(_name: str):
            loads.append(1)
            return _Sentinel()

        import sys
        import types

        fake_tiktoken = types.ModuleType("tiktoken")
        fake_tiktoken.get_encoding = fake_get_encoding  # type: ignore[attr-defined]
        original = sys.modules.get("tiktoken")
        sys.modules["tiktoken"] = fake_tiktoken
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                encodings = list(pool.map(lambda _: real_import(), range(16)))
            assert len(loads) == 1, f"loaded the encoding {len(loads)} times"
            assert len({id(e) for e in encodings}) == 1, "workers saw different encodings"
        finally:
            if original is not None:
                sys.modules["tiktoken"] = original
            else:
                sys.modules.pop("tiktoken", None)
            tokens_mod._ENCODING = None
            tokens_mod._ENCODING_FAILED = False

    def test_an_invalidation_during_a_computation_is_not_lost(self):
        """A mutation that lands mid-encode must not leave a stale count.

        `_compute_tokens` runs OUTSIDE `_CACHE_LOCK` on purpose (holding the
        lock across the encode would re-serialize what the offload exists to
        parallelize). That opens a window: a thread reads a cache miss, starts
        encoding, and while it encodes the loop mutates the message and
        invalidates it — the `pop` finds nothing, and the thread then inserts a
        count describing the PRE-mutation message. Nothing ever clears it, so
        the stale value survives for the life of the process.

        Reproduced at 295/300 before `_ESTIMATE_GENERATION` existed. This is
        the exact silent-staleness the module contract forbids, and the
        compaction gate's cheap upper-bound early return depends on it not
        happening.
        """
        import threading

        from local_operator.compaction import tokens as mod

        real_compute = mod._compute_tokens
        stale = 0
        try:
            for _ in range(25):
                message = Message.user("lorem ipsum dolor " * 400)
                clear_estimate_cache()
                computing = threading.Event()

                def slow_compute(msg, _real=real_compute):
                    value = _real(msg)
                    computing.set()
                    time.sleep(0.002)  # hold the compute->insert window open
                    return value

                mod._compute_tokens = slow_compute
                worker = threading.Thread(target=lambda: estimate_tokens(message))
                worker.start()
                computing.wait(timeout=5)
                # Mutate + invalidate inside the window, exactly as pruning does.
                message.content = [TextContent(text="blanked")]
                invalidate_message_cache(message)
                worker.join(timeout=5)
                mod._compute_tokens = real_compute

                cached = mod._ESTIMATE_CACHE.get(message.id)
                if cached is not None and cached != real_compute(message):
                    stale += 1
        finally:
            mod._compute_tokens = real_compute

        assert stale == 0, f"{stale}/25 invalidations were lost, leaving a stale estimate"

    def test_race_bookkeeping_cannot_grow_without_bound(self):
        """The race fix must not become the leak the cache bound exists to stop.

        Both paths, because the INSERT path alone hides the leak that matters.
        An earlier per-message-id counter was pruned only on eviction, and
        eviction only happens on an insert — so the once-per-turn path that
        invalidates without ever estimating (a prune below the compaction
        threshold, where the cheap upper bound returns early and the exact
        estimator never runs) grew one permanent entry per turn, on a
        module-global dict shared by every session. Measured 20,000 entries
        after 20,000 such turns with the cache still empty.
        """
        from local_operator.compaction import tokens as mod

        clear_estimate_cache()
        for batch in range(20):
            estimate_messages_tokens(
                [Message.user(f"b{batch} m{i} hello world") for i in range(400)]
            )
        assert len(mod._ESTIMATE_CACHE) <= mod._ESTIMATE_CACHE_MAX

        # The path that leaked: invalidate, never estimate.
        for turn in range(5_000):
            invalidate_message_cache(Message.user(f"turn {turn} tool output"))
        assert not mod._INFLIGHT_ESTIMATES, (
            f"{len(mod._INFLIGHT_ESTIMATES)} entries survived invalidations that "
            "never ran an estimate; the race bookkeeping is leaking"
        )

    def test_a_stalled_computation_cannot_be_revived_by_cache_turnover(self):
        """Race state is keyed by a unique ticket, never by message id.

        With a per-id counter, evicting the id dropped its counter, so a later
        read returned 0 — the same value a stalled computation had recorded
        before it started encoding. That resurrects the exact lost-invalidation
        bug the counter was added to fix, just behind a narrower window.
        """
        import threading

        from local_operator.compaction import tokens as mod

        real_compute = mod._compute_tokens
        try:
            clear_estimate_cache()
            message = Message.user("lorem ipsum " * 400)
            computing = threading.Event()

            def slow_compute(msg, _real=real_compute):
                value = _real(msg)
                computing.set()
                time.sleep(0.15)
                return value

            mod._compute_tokens = slow_compute
            worker = threading.Thread(target=lambda: estimate_tokens(message))
            worker.start()
            computing.wait(timeout=5)

            message.content = [TextContent(text="blanked")]
            invalidate_message_cache(message)

            # Churn the cache past its bound so any per-id state would recycle.
            mod._compute_tokens = real_compute
            for i in range(mod._ESTIMATE_CACHE_MAX + 500):
                estimate_tokens(Message.user(f"churn {i}"))
            worker.join(timeout=5)

            cached = mod._ESTIMATE_CACHE.get(message.id)
            assert cached is None or cached == real_compute(message), (
                "a stalled computation's pre-mutation count was cached after "
                "cache turnover recycled its race state"
            )
        finally:
            mod._compute_tokens = real_compute
