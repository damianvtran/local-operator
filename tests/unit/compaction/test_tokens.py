"""Token estimation: tiktoken path, fallback, memoization, invalidation."""

import threading
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
    """When tiktoken cannot load, degrade to len(text)//4 — never raise.

    The failed attempt's backoff deadline is pushed out of reach so the
    assertion measures the fallback itself; with the deadline elapsed,
    ``_get_encoding`` is allowed to retry and would restore the real ruler.
    """
    clear_estimate_cache()
    monkeypatch.setattr(tokens_mod, "_ENCODING", None)
    monkeypatch.setattr(tokens_mod, "_ENCODING_FAILED", True)
    monkeypatch.setattr(tokens_mod, "_ENCODING_RETRY_AT", float("inf"))
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

        # The supported reset: it also clears the epoch-keyed load claim, so a
        # stale claim from another test cannot make these workers observe the
        # fallback instead of the singleton.
        tokens_mod.reset_tokenizer_state()

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
            tokens_mod.reset_tokenizer_state()

    def test_an_invalidation_during_a_computation_is_not_lost(self):
        """A mutation that lands mid-encode must not leave a stale count.

        `_compute_tokens` runs OUTSIDE `_CACHE_LOCK` on purpose (holding the
        lock across the encode would re-serialize what the offload exists to
        parallelize). That opens a window: a thread reads a cache miss, starts
        encoding, and while it encodes the loop mutates the message and
        invalidates it — the `pop` finds nothing, and the thread then inserts a
        count describing the PRE-mutation message. Nothing ever clears it, so
        the stale value survives for the life of the process.

        Reproduced at 295/300 before the in-flight ticket bookkeeping existed.
        This is
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

            # Re-cache at the CURRENT content BEFORE churning. This step is
            # what makes the test able to fail: with per-id race state, this
            # insert is what created the entry whose later eviction recycled
            # the id's counter back to its initial value — the value the
            # stalled worker recorded — reviving the stale insert. Without it
            # there is nothing for the churn to evict, and the buggy
            # implementation passes too.
            mod._compute_tokens = real_compute
            estimate_tokens(message)
            # Churn the cache past its bound so any per-id state would recycle.
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


# --- first-use warm (the attach-latency seam) --------------------------------


def test_warm_tokenizer_loads_the_encoding(monkeypatch):
    """``warm_tokenizer`` is the same load first use would do, brought forward.

    Building cl100k_base is 122 ms cold against 0.004 ms warm on this machine,
    and its first use is ``measure_request`` — which the stream function awaits
    before it opens the provider request. Bring it forward to boot and the
    first turn does not pay it.
    """
    monkeypatch.setattr(tokens_mod, "_ENCODING", None)
    monkeypatch.setattr(tokens_mod, "_ENCODING_FAILED", False)

    tokens_mod.warm_tokenizer()

    assert tokens_mod._ENCODING is not None or tokens_mod._ENCODING_FAILED


def test_warm_tokenizer_is_idempotent(monkeypatch):
    """The second call must not rebuild the table — the singleton is the point."""
    monkeypatch.setattr(tokens_mod, "_ENCODING", None)
    monkeypatch.setattr(tokens_mod, "_ENCODING_FAILED", False)
    tokens_mod.warm_tokenizer()
    loaded = tokens_mod._ENCODING

    tokens_mod.warm_tokenizer()

    assert tokens_mod._ENCODING is loaded


def test_warm_tokenizer_never_raises(monkeypatch):
    """A warm-up must not be the thing that reports a missing extra."""

    def explode():
        raise RuntimeError("tiktoken is having a day")

    monkeypatch.setattr(tokens_mod, "_get_encoding", explode)
    tokens_mod.warm_tokenizer()  # must not raise


def test_background_warm_returns_none_when_already_loaded(monkeypatch):
    """Nothing to wait for means no thread and no scheduling cost."""
    monkeypatch.setattr(tokens_mod, "_ENCODING", object())
    monkeypatch.setattr(tokens_mod, "_ENCODING_FAILED", False)
    assert tokens_mod.warm_tokenizer_in_background() is None


def test_background_warm_refuses_to_retry_a_failed_load(monkeypatch):
    """A host without the tokenizer extra must not spawn a thread per boot."""
    monkeypatch.setattr(tokens_mod, "_ENCODING", None)
    monkeypatch.setattr(tokens_mod, "_ENCODING_FAILED", True)
    assert tokens_mod.warm_tokenizer_in_background() is None


def test_background_warm_loads_on_a_thread(monkeypatch):
    """The cold path returns a started thread, and the load lands on it."""
    monkeypatch.setattr(tokens_mod, "_ENCODING", None)
    monkeypatch.setattr(tokens_mod, "_ENCODING_FAILED", False)

    thread = tokens_mod.warm_tokenizer_in_background()

    assert thread is not None
    thread.join(60)
    assert not thread.is_alive()
    assert tokens_mod._ENCODING is not None or tokens_mod._ENCODING_FAILED


# --- the failure latch must not be permanent ---------------------------------


def _require_real_encoding() -> None:
    """Skip when this process cannot load cl100k_base at all.

    The recovery assertions below observe the REAL ruler returning; with no
    ruler available (an offline box with a cold BPE cache) they would pass or
    fail by identity, which measures nothing.
    """
    pytest.importorskip("tiktoken")
    tokens_mod.reset_tokenizer_state()
    if tokens_mod._get_encoding() is None:
        pytest.skip("cl100k_base unavailable — the recovery cannot be observed")


def _wait_until(predicate, timeout: float = 30.0) -> bool:
    """Poll ``predicate`` until it holds; ``False`` on timeout.

    Used for the states that are now reached on the loader thread — the retry
    is deliberately not run inline, so a test cannot observe its outcome by
    return value.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def test_a_failed_load_is_retried_once_its_backoff_elapses():
    """The latch is a cooldown, not a verdict for the life of the process.

    Repro of the measured defect: a first-use fetch that fails once used to
    make every later count ``len(text) // 4`` — in CI that turned the
    calibration-margin test's pair from (97502, 90002) into (106876, 97501)
    and made a 100k window refuse a request it could serve. A latch whose
    deadline has passed must recover the real encoder.

    The recovery is observed on the loader thread, because the retry is
    dispatched rather than run inline: the call that notices the elapsed
    deadline may degrade for that one estimate (the fast, table-cached load can
    publish before it returns), and the encoder is back for the next. Running
    it inline is the round-1 defect (a caller blocked on the fetch for its
    whole timeout); the non-blocking half is pinned separately by
    ``test_a_due_retry_does_not_block_an_unrelated_estimate``.
    """
    _require_real_encoding()
    try:
        tokens_mod._ENCODING = None
        tokens_mod._ENCODING_FAILED = True
        tokens_mod._ENCODING_FAILURES = 1
        tokens_mod._ENCODING_RETRY_AT = 0.0  # the backoff has elapsed
        tokens_mod._get_encoding()  # dispatches the retry
        assert _wait_until(lambda: tokens_mod._ENCODING is not None)
        assert tokens_mod._ENCODING_FAILED is False
        assert tokens_mod._ENCODING_FAILURES == 0
    finally:
        tokens_mod.reset_tokenizer_state()


def test_a_due_retry_does_not_block_an_unrelated_estimate():
    """A retry runs on its own thread; no caller waits on its fetch.

    Round-1 defect: the elapsed-backoff retry ran inline holding the shared
    lock, so an unrelated thread's ``estimate_tokens`` blocked for the whole
    fetch — reproduced at 2.01 s with a 2 s fake, and 75.7 s offline, every
    backoff interval, forever. The retry must be dispatched and the estimate
    must answer immediately from the fallback.
    """
    import sys
    import types

    entered = threading.Event()
    release = threading.Event()

    def slow_load(_name: str):
        entered.set()
        release.wait(20)
        return object()

    fake_tiktoken = types.ModuleType("tiktoken")
    fake_tiktoken.get_encoding = slow_load  # type: ignore[attr-defined]
    original = sys.modules.get("tiktoken")

    tokens_mod.reset_tokenizer_state()
    try:
        tokens_mod._ENCODING_FAILED = True
        tokens_mod._ENCODING_FAILURES = 1
        tokens_mod._ENCODING_RETRY_AT = 0.0
        sys.modules["tiktoken"] = fake_tiktoken

        assert tokens_mod._get_encoding() is None  # dispatches, does not load
        assert entered.wait(5), "the retry never reached its fetch"

        started = time.monotonic()
        assert estimate_tokens(Message.user("x" * 400)) == 100
        elapsed = time.monotonic() - started
        assert elapsed < 0.5, f"an unrelated estimate waited {elapsed:.2f}s on the retry"
    finally:
        release.set()
        _wait_until(lambda: tokens_mod._LOAD_EPOCH is None)
        if original is not None:
            sys.modules["tiktoken"] = original
        else:
            sys.modules.pop("tiktoken", None)
        tokens_mod.reset_tokenizer_state()


def test_a_retry_failing_after_a_reset_cannot_rearm_the_latch():
    """QA's interleaving: a reset landing while a retry is in flight.

    Reproduced before this round as ``failed=True, retry_in=30s`` on the next
    count — the loader compared the epoch, the reset landed, and the loader
    published its failure into the fresh state. The compare and the store are
    now ONE critical section, and the reset takes the same lock, so the reset
    is either entirely before the publication (discarded) or entirely after it
    (the reset clears what was published).
    """
    import sys
    import types

    entered = threading.Event()
    release = threading.Event()

    def slow_failure(_name: str):
        entered.set()
        release.wait(20)
        raise RuntimeError("cold BPE fetch failed")

    fake_tiktoken = types.ModuleType("tiktoken")
    fake_tiktoken.get_encoding = slow_failure  # type: ignore[attr-defined]
    original = sys.modules.get("tiktoken")

    tokens_mod.reset_tokenizer_state()
    try:
        tokens_mod._ENCODING_FAILED = True
        tokens_mod._ENCODING_FAILURES = 1
        tokens_mod._ENCODING_RETRY_AT = 0.0
        sys.modules["tiktoken"] = fake_tiktoken

        text = "word " * 100
        assert count_text_tokens(text) == len(text) // 4  # the retry is now in flight
        assert entered.wait(5), "the retry never reached its fetch"

        tokens_mod.reset_tokenizer_state()  # lands inside the publication window
        release.set()
        assert _wait_until(lambda: tokens_mod._LOAD_EPOCH is None)

        assert tokens_mod._ENCODING_FAILED is False, "the late retry re-armed the latch"
        assert tokens_mod._ENCODING_FAILURES == 0
        assert tokens_mod._ENCODING_RETRY_AT == 0.0
        assert tokens_mod._ENCODING is None
    finally:
        release.set()
        if original is not None:
            sys.modules["tiktoken"] = original
        else:
            sys.modules.pop("tiktoken", None)
        tokens_mod.reset_tokenizer_state()

    # The fresh state still loads: no cooldown was smuggled into it.
    _require_real_encoding()
    tokens_mod.reset_tokenizer_state()


def test_a_failed_load_is_not_retried_inside_its_backoff():
    """The retry is gated, because an offline load costs 75.7 s to time out.

    Retrying on every estimate would put that stall on the turn path; the
    cooldown is what makes recovery safe to attempt at all.
    """
    try:
        tokens_mod.reset_tokenizer_state()
        tokens_mod._ENCODING_FAILED = True
        tokens_mod._ENCODING_RETRY_AT = time.monotonic() + 60.0
        assert tokens_mod._get_encoding() is None
        assert tokens_mod._LOAD_EPOCH is None, "a gated attempt was dispatched anyway"
    finally:
        tokens_mod.reset_tokenizer_state()


def test_a_cooldown_does_not_poison_the_per_model_cache():
    """A transient cooldown must not be cached as "tiktoken is gone" for a model.

    ``count_text_tokens(text, model=...)`` is the HTTP chat response's
    ``stats.total_tokens`` path. When the model resolution has to fall back to
    the shared singleton and that singleton is inside a retry cooldown, the
    answer for THIS call is legitimately the chars/4 fallback — but caching
    that ``None`` under the model name would re-create the round-1 defect one
    layer down: a 30-300 s transient published as a permanent property of the
    model. Both directions are pinned: no poisoning during the cooldown, and
    the model's real encoder used (and cached) once the tokenizer recovers.
    """
    import sys
    import types

    def unavailable(_name: str):
        raise RuntimeError("no BPE table while offline")

    fake_tiktoken = types.ModuleType("tiktoken")
    fake_tiktoken.encoding_for_model = unavailable  # type: ignore[attr-defined]
    fake_tiktoken.get_encoding = unavailable  # type: ignore[attr-defined]
    original = sys.modules.get("tiktoken")

    tokens_mod.reset_tokenizer_state()
    try:
        tokens_mod._ENCODING_FAILED = True
        tokens_mod._ENCODING_FAILURES = 1
        tokens_mod._ENCODING_RETRY_AT = time.monotonic() + 300.0
        sys.modules["tiktoken"] = fake_tiktoken

        text = "hello world, this is a token count"
        assert count_text_tokens(text, model="gpt-4o") == len(text) // 4
        assert (
            "gpt-4o" not in tokens_mod._MODEL_ENCODINGS
        ), "a cooldown result was cached as the model's encoder"
    finally:
        if original is not None:
            sys.modules["tiktoken"] = original
        else:
            sys.modules.pop("tiktoken", None)
        tokens_mod.reset_tokenizer_state()

    # Recovery direction: the model's own encoder must be resolvable again.
    pytest.importorskip("tiktoken")
    cjk = "大语言模型会显著改变分词计数方式" * 4
    try:
        tokens_mod.reset_tokenizer_state()
        real = count_text_tokens(cjk, model="gpt-4o")
        assert real != len(cjk) // 4, "the recovered model encoder was not used"
        assert "gpt-4o" in tokens_mod._MODEL_ENCODINGS
        assert count_text_tokens(cjk, model="gpt-4o") == real  # served from cache
    finally:
        tokens_mod.reset_tokenizer_state()


def test_a_cooldown_bounds_the_model_path_to_one_fetch_per_window():
    """The cooldown latches ``count_text_tokens(model=...)`` as well.

    Round-2 regression this pins: not caching ``None`` (the round-1 fix) also
    removed an accidental short-circuit, and model resolution never consulted
    the latch — so during a cooldown every per-message count paid two
    synchronous BPE fetches (``encoding_for_model`` fails, then
    ``get_encoding("cl100k_base")``), i.e. minutes per HTTP chat request at
    the module's 75.7 s offline timeout.

    The fake fetch is deliberately SLOW (0.4 s): with an instant raiser the
    pre-fix code still passes this test, because the cost only shows up as
    elapsed fetch time. Do not "simplify" it back to ``raise``.
    """
    import sys
    import types

    attempts: list[str] = []
    attempts_lock = threading.Lock()
    # Armed only for the boundary phase; see ``slow_base_fetch``.
    hold_loader = threading.Event()
    loader_entered = threading.Event()
    release_loader = threading.Event()

    def slow_model_fetch(_model: str):
        with attempts_lock:
            attempts.append("encoding_for_model")
        time.sleep(0.4)
        raise RuntimeError("offline: model BPE fetch timed out")

    def slow_base_fetch(_name: str):
        with attempts_lock:
            attempts.append("get_encoding")
        # The boundary phase below holds the RETRY LOADER open, so "a loader is
        # in flight" is a fact for the whole loop. Offline that fetch runs for
        # 75.7 s, so in production the loader outlives every caller; a fake that
        # merely sleeps would finish mid-loop and re-arm the cooldown, hiding
        # the boundary defect behind the window it replaces.
        if hold_loader.is_set() and threading.current_thread().name == "lop-tokenizer-load":
            loader_entered.set()
            release_loader.wait(10.0)
        time.sleep(0.4)
        raise RuntimeError("offline: BPE fetch timed out")

    fake_tiktoken = types.ModuleType("tiktoken")
    fake_tiktoken.encoding_for_model = slow_model_fetch  # type: ignore[attr-defined]
    fake_tiktoken.get_encoding = slow_base_fetch  # type: ignore[attr-defined]
    original = sys.modules.get("tiktoken")

    tokens_mod.reset_tokenizer_state()
    try:
        sys.modules["tiktoken"] = fake_tiktoken
        text = "hello world, this is a token count"

        # First call: no latch yet, so it legitimately pays its resolution
        # attempts and the first-use load, and its failure arms the cooldown.
        assert count_text_tokens(text, model="gpt-4o") == len(text) // 4
        assert tokens_mod._ENCODING_FAILED, "the first failing call did not arm the cooldown"
        with attempts_lock:
            before = len(attempts)
        assert before <= 3, (
            f"the first failing call made {before} fetches; it should be at most "
            "one model resolution plus the first-use load"
        )

        # Every call INSIDE the cooldown window must pay nothing. Pre-fix this
        # was two synchronous fetches per call (four over the next two).
        for _ in range(2):
            assert count_text_tokens(text, model="gpt-4o") == len(text) // 4
        with attempts_lock:
            during_cooldown = len(attempts) - before
        assert during_cooldown == 0, (
            f"{during_cooldown} fetch attempts inside the cooldown window; the "
            "model path must consult the latch like the base path does "
            "(pre-fix: two per call, four here)"
        )

        # Boundary case (round 3): once the retry deadline elapses the BASE
        # path dispatches its retry to a loader thread — but this path has no
        # loader, so a windowed gate let it fetch INLINE again, two synchronous
        # fetches per caller for as long as the attempt runs (75.7 s offline).
        # Pre-fix: these 4 calls made 8 caller-thread fetches. Distinct texts
        # because nothing memoises counts; the same string would hide repeats.
        tokens_mod._ENCODING_RETRY_AT = 0.0  # the backoff has elapsed
        hold_loader.set()
        assert count_text_tokens(text) == len(text) // 4  # retry now in flight
        assert loader_entered.wait(5.0), "the retry loader never started"
        with attempts_lock:
            before_expiry = len(attempts)
        for i in range(4):
            call_text = f"{text} {i}"
            assert count_text_tokens(call_text, model="gpt-4o") == len(call_text) // 4
        with attempts_lock:
            during_loader = len(attempts) - before_expiry
        assert during_loader == 0, (
            f"{during_loader} fetch attempts after the retry deadline expired "
            "while a loader was in flight (pre-fix: two synchronous fetches "
            "per call, eight here); the model path must be gated for the "
            "whole failure episode, not just the cooldown window"
        )
        # Let the loader publish before the reset below, so its outcome cannot
        # race teardown (the epoch guard would discard it, but waiting keeps
        # the test deterministic).
        release_loader.set()
        deadline = time.monotonic() + 5.0
        while tokens_mod._LOAD_EPOCH is not None and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        release_loader.set()
        if original is not None:
            sys.modules["tiktoken"] = original
        else:
            sys.modules.pop("tiktoken", None)
        tokens_mod.reset_tokenizer_state()


def test_a_model_encoder_resolved_across_a_reset_is_not_republished():
    """A reset landing mid-resolution must not leave its encoder in the cache.

    ``reset_tokenizer_state`` promises a cached per-model encoder cannot
    outlive it. The base singleton's publication is epoch-guarded, but the
    ``_MODEL_ENCODINGS`` write was not: a reset landing during a fetch let the
    resolution republish a pre-reset encoder into the cache the reset had just
    cleared (a test-isolation hole — no latch re-arm, no wrong count, just a
    stale encoder reported as current).
    """
    import sys
    import types

    entered = threading.Event()
    release = threading.Event()

    def unknown_model(_model: str):
        # Instant: the fetch under test is the cl100k_base fallback.
        raise KeyError(_model)

    def slow_base_fetch(_name: str):
        entered.set()
        release.wait(20)
        return object()

    fake_tiktoken = types.ModuleType("tiktoken")
    fake_tiktoken.encoding_for_model = unknown_model  # type: ignore[attr-defined]
    fake_tiktoken.get_encoding = slow_base_fetch  # type: ignore[attr-defined]
    original = sys.modules.get("tiktoken")

    tokens_mod.reset_tokenizer_state()
    sys.modules["tiktoken"] = fake_tiktoken
    resolved: list[object] = []
    try:
        worker = threading.Thread(
            target=lambda: resolved.append(tokens_mod._get_model_encoding("gpt-4o")),
            daemon=True,
        )
        worker.start()
        try:
            assert entered.wait(5), "the model resolution never reached its fetch"
            # The owning test tears down while the resolution is in flight.
            tokens_mod.reset_tokenizer_state()
            release.set()
            worker.join(10)
        finally:
            release.set()
    finally:
        if original is not None:
            sys.modules["tiktoken"] = original
        else:
            sys.modules.pop("tiktoken", None)

    assert not worker.is_alive()
    assert (
        tokens_mod._MODEL_ENCODINGS == {}
    ), "an encoder resolved before the reset was republished into the cleared cache"
    assert resolved and resolved[0] is not None, "the caller lost its resolved encoder"
    tokens_mod.reset_tokenizer_state()


def test_the_retry_backoff_doubles_and_caps():
    """30 s, then doubling per failure, pinned at the 5-minute cap.

    Pinned because the schedule is the whole reason a permanently offline
    process is bounded: a wrong base or a missing cap is either a retry storm
    (a 75.7 s fetch on nearly every estimate) or a never-retried process.
    """
    import sys
    import types

    def always_fails(_name: str):
        raise RuntimeError("no BPE table")

    fake_tiktoken = types.ModuleType("tiktoken")
    fake_tiktoken.get_encoding = always_fails  # type: ignore[attr-defined]
    original = sys.modules.get("tiktoken")

    tokens_mod.reset_tokenizer_state()
    try:
        sys.modules["tiktoken"] = fake_tiktoken
        epoch = tokens_mod._ENCODING_EPOCH
        delays: list[float] = []
        for _ in range(6):
            with tokens_mod._STATE_LOCK:
                tokens_mod._LOAD_EPOCH = epoch  # claim, as _get_encoding would
            started = time.monotonic()
            assert tokens_mod._load_encoding(epoch) is None
            delays.append(tokens_mod._ENCODING_RETRY_AT - started)

        assert delays == pytest.approx([30.0, 60.0, 120.0, 240.0, 300.0, 300.0], abs=1.0)
        assert tokens_mod._ENCODING_FAILURES == 6
    finally:
        if original is not None:
            sys.modules["tiktoken"] = original
        else:
            sys.modules.pop("tiktoken", None)
        tokens_mod.reset_tokenizer_state()


def test_reset_tokenizer_state_clears_the_latch_and_reloads():
    """The test-visible reset returns a poisoned process to a cold state.

    Including the per-model cache: an encoder resolved from the pre-reset
    tiktoken state must not outlive the reset (that cache is what
    ``count_text_tokens(model=...)`` answers from).
    """
    _require_real_encoding()
    try:
        tokens_mod._ENCODING = None
        tokens_mod._ENCODING_FAILED = True
        tokens_mod._ENCODING_FAILURES = 3
        tokens_mod._ENCODING_RETRY_AT = time.monotonic() + 300.0
        tokens_mod._MODEL_ENCODINGS["gpt-4o"] = object()

        tokens_mod.reset_tokenizer_state()

        assert tokens_mod._ENCODING_FAILED is False
        assert tokens_mod._ENCODING_FAILURES == 0
        assert tokens_mod._ENCODING_RETRY_AT == 0.0
        assert tokens_mod._MODEL_ENCODINGS == {}
        assert tokens_mod._get_encoding() is not None
    finally:
        tokens_mod.reset_tokenizer_state()


def test_a_warm_thread_finishing_after_a_reset_cannot_poison_counts(monkeypatch):
    """A late daemon warm must not publish its failure into the reset state.

    This is the CI mechanism: a warm thread completed after its test had torn
    down, set the process-wide latch, and every count in that shard ran on the
    chars/4 ruler. The reset's epoch bump discards an in-flight attempt's
    outcome, so the thread that started before it can no longer re-arm it.
    """
    import sys
    import threading
    import types

    entered = threading.Event()
    release = threading.Event()

    fake_tiktoken = types.ModuleType("tiktoken")

    def slow_failure(_name: str):
        entered.set()
        release.wait(10)
        raise RuntimeError("cold BPE fetch failed")

    fake_tiktoken.get_encoding = slow_failure  # type: ignore[attr-defined]
    original = sys.modules.get("tiktoken")

    tokens_mod.reset_tokenizer_state()
    sys.modules["tiktoken"] = fake_tiktoken
    try:
        worker = threading.Thread(target=tokens_mod._get_encoding, daemon=True)
        worker.start()
        try:
            assert entered.wait(5), "the worker never started its load"
            # The owning test tears down (and thus resets) while the load runs.
            tokens_mod.reset_tokenizer_state()
            release.set()
            worker.join(10)
        finally:
            release.set()
    finally:
        if original is not None:
            sys.modules["tiktoken"] = original
        else:
            sys.modules.pop("tiktoken", None)

    assert not worker.is_alive()
    assert tokens_mod._ENCODING_FAILED is False, "a late warm re-armed the failure latch"
    _require_real_encoding()
    tokens_mod.reset_tokenizer_state()
