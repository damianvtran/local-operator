"""Token estimation with per-message memoization.

Why this module exists
-----------------------
Every compaction decision (threshold checks, cut-point search, pruning guards)
is driven by token counts. Providers only report real usage after a request,
so between requests we estimate locally with tiktoken's ``cl100k_base``.
Estimation runs on hot paths (per-index suffix sums, backwards cut-point
walks), so results are memoized.

tiktoken lives behind the ``tokenizer`` extra: it is a compiled wheel that
also fetches its BPE ranks over the network on first use, neither of which
the default install should require. Without it every estimate falls back to
the ``len(text) // 4`` heuristic below, which is coarser but keeps compaction
working — real usage from the provider still corrects the accounting after
each request.

This estimator is a RULER OF ITS OWN, not a prediction of the bill
------------------------------------------------------------------
``cl100k_base`` is OpenAI's tokenizer, and Anthropic's differs measurably on
the code/JSON-dense content an agent session is made of. Fitting
``provider = a * local + b`` over a real 10-pass session
(``docs/evidence/compaction-ruler/slope_fit.py``) puts **``claude-opus-5`` at
slope 1.685 and ``claude-opus-4-8`` at 1.622**, against **1.036 for an OpenAI
control and 1.019 for GLM in the same session with the same tool schemas**.
Per-request, an opus-5 context bills 1.75-1.90x this module's estimate.

The slope is a per-MODEL property, not a session constant: fitted per epoch,
the three single-model stretches fit tightly (mean error 419 / 653 / 1,415
tokens) while every model-switching stretch does not (9,913 to 71,026), since
one line cannot describe two tokenizers.

So a number from this module may be compared against ANOTHER number from this
module, never against a provider figure. Mixing the two is a real bug that has
shipped twice — the compaction receipt subtracted a local saving from a
provider total, and ``Session._advisor_floor_cap`` capped a locally-summed
span with a provider-scale threshold. Both are fixed by keeping each
comparison on one ruler; neither is fixed by counting more here.

Per-provider calibration was deliberately NOT added. The intercept is not
identifiable across a model-heterogeneous session (the fit only holds inside
contiguous single-model runs), and the compaction TRIGGER does not need it:
``compaction_context_tokens`` already takes ``max(provider, local)`` on any
request that carries a usage record, so the provider's own count drives when a
pass fires. A multiplier table here would add a constant that rots on the next
retokenization while fixing nothing the trigger depends on.

Settle rule / cache contract
----------------------------
The cache is a module-level dict keyed on the message's stable ``id``. Two
invariants keep it honest:

1. **Settle gate.** A streaming assistant is mutated under one identity while
   its ``usage``/``stop_reason`` are provisional, so assistants are cached
   only when settled — real ``usage`` (not ``None``) and a terminal
   ``stop_reason`` that is not ``None`` / ``"aborted"`` / ``"error"``.
   Unsettled assistants never read or insert. Non-assistant messages are
   immutable once appended and always cache. Loop finalization
   (``harness/loop.py``) sets ``usage``/``stop_reason`` BEFORE any estimate
   of the finished message runs and drops a provisional entry at that point.
2. **Owner invalidation.** Any in-place mutation of a message MUST call
   :func:`invalidate_message_cache`. Owners that mutate messages in place are
   the pruning passes (``local_operator.compaction.pruning``), streaming
   finalize, and any future image/reasoning stripper.

Cross-package consumers that keep derived caches can subscribe via
:func:`register_invalidator` and are notified on every invalidation.

Thread safety / why the async variants exist
--------------------------------------------
Encoding is the single most expensive synchronous stretch on the event loop
(measured: ~90 ms for eight ~40 KB messages, and 116 of 121 samples in a
stall trace of eight concurrent subagents landed inside ``_encode_len``).
Because every session shares ONE event loop, that stretch does not merely
slow the session doing the counting — it stops every sibling subagent, the
parent's own stream, and the TUI repaint for its whole duration.

tiktoken's ``encode`` is a Rust extension that RELEASES the GIL, so moving it
to a worker thread genuinely buys parallelism rather than just moving the
stall (measured: 90 ms → 0.7 ms of loop stall, 3x faster wall clock for the
same work). ``Session._offloaded`` is the caller that does that hop, for
histories past :data:`OFFLOAD_MIN_CHARS`; the functions here stay synchronous
so they remain usable from a worker thread, from pure-CPU callers, and from
tests.

That makes the module-level cache reachable from worker threads, so the
mutations that are NOT individually atomic — the LRU ``move_to_end`` +
``popitem`` eviction sequence — are guarded by :data:`_CACHE_LOCK`. The lock
deliberately covers only the dict bookkeeping (sub-microsecond) and never
``encode`` itself, so concurrent estimates still tokenize in parallel. Because
the encode runs unlocked, an invalidation can land between a computation and
its insert; that race is closed by :data:`_INFLIGHT_ESTIMATES` rather than by
widening the lock.

The lazy tokenizer singleton has its own, separate state lock
(:data:`_STATE_LOCK`). It has to be separate from :data:`_CACHE_LOCK` for one
measured reason: a load can be a network fetch that takes 75 s to time out, and
holding the *estimate cache's* lock across it blocked unrelated threads'
``estimate_tokens`` calls for the whole timeout (measured at 2.01 s with an
artificial delay, recurring every backoff interval while offline). No lock in
this module is held across a fetch.
"""

from __future__ import annotations

import itertools
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Sequence

from local_operator.harness.types import (
    AttachmentContent,
    DocumentContent,
    Message,
    TextContent,
)
from local_operator.optional import missing_extra_error

logger = logging.getLogger(__name__)

#: Guards the estimate cache's non-atomic LRU sequences, which are reachable
#: from ``asyncio.to_thread`` workers. Held for dict bookkeeping only — never
#: across ``encode``, and never across a tokenizer fetch (see
#: :data:`_STATE_LOCK`).
_CACHE_LOCK = threading.Lock()

#: Below this many characters of history, estimating inline is cheaper than a
#: thread hop (~50-100 us round trip). Above it the encode dominates and the
#: hop pays for itself many times over. Applied by ``Session._offloaded`` so
#: callers do not each re-derive the trade-off.
OFFLOAD_MIN_CHARS = 20_000

#: Flat token cost charged per image block. Matches
#: ``IMAGE_TOKEN_ESTIMATE``: vision providers bill images in fixed visual-token
#: chunks, and a flat estimate keeps the estimator deterministic and
#: independent of base64 payload length.
IMAGE_TOKEN_ESTIMATE = 1200

#: Ceiling for ONE document block's local token charge. A native PDF is priced
#: by the provider's own page/pixel rules (no local ruler can reproduce them),
#: and the fallback is page-marked TEXT bounded at 400k characters; charging
#: either as an image would be wrong in both directions. The block's inline
#: base64 is used as a cheap upper bound (1 char ~ 1 token is safe for ASCII),
#: capped here so one 32 MB PDF cannot report an eight-million-token history
#: and drive compaction into a loop. The cap is deliberately generous: a
#: six-hundred-page PDF really can cost six figures of tokens.
MAX_DOCUMENT_TOKEN_ESTIMATE = 100_000


def _document_token_estimate(block: DocumentContent) -> int:
    """Tokens to charge for one document block (see the constant's note).

    Deterministic and allocation-free: ``len(data)`` is already on the block,
    and both ``_compute_tokens`` and ``messages_tokens_upper_bound`` must use
    THIS rule or the upper bound stops being an upper bound.
    """
    return min(len(block.data), MAX_DOCUMENT_TOKEN_ESTIMATE)


#: Fallback ratio when tiktoken is unavailable: ~4 chars per token.
_CHARS_PER_TOKEN_FALLBACK = 4

# ---------------------------------------------------------------------------
# Lazy tiktoken singleton
# ---------------------------------------------------------------------------

#: The lazy tokenizer singleton's state lock. EVERY read and write of the
#: fields below, and of :data:`_MODEL_ENCODINGS`, happens under it, so a load's
#: publication is one atomic step against a concurrent
#: :func:`reset_tokenizer_state`. It is held for bookkeeping only — a fetch is
#: always performed with it released, because with tiktoken installed and the
#: BPE table uncached a load is a network fetch measured at 75.7 s to time out,
#: and a load under a shared lock stalls every other caller for that long.
_STATE_LOCK = threading.Lock()

#: Signalled after every load attempt releases its claim, so a caller waiting
#: on a first-load (:func:`_get_encoding`) re-checks the state instead of
#: polling. Shares :data:`_STATE_LOCK`.
_STATE_CHANGED = threading.Condition(_STATE_LOCK)

#: The loaded cl100k_base encoding, or ``None`` when there is none.
_ENCODING: object | None = None

#: ``True`` while the most recent load attempt failed. NOT a permanent verdict:
#: it suppresses attempts only until :data:`_ENCODING_RETRY_AT`, then the next
#: call dispatches a retry. A sticky version of this flag was a measured defect
#: — a transient first-use failure (a cold BPE fetch that times out, a tiktoken
#: import that races a test's ``sys.modules`` patch) silently switched EVERY
#: later token count in the process to ``len(text) // 4``, for the life of the
#: process. Compaction budgets decide what gets summarised, so that degradation
#: is a correctness bug, not a rounding difference.
_ENCODING_FAILED = False

#: Consecutive failed load attempts, for the retry backoff. Reset on success and
#: on :func:`reset_tokenizer_state`.
_ENCODING_FAILURES = 0

#: ``time.monotonic()`` deadline before which a failed load is not retried.
#: The backoff exists because the failure can be EXPENSIVE, not just wrong to
#: repeat: with tiktoken installed and the BPE table uncached, a load is a
#: network fetch that measured 75.7 s to time out on an unreachable network.
#: Retrying on every estimate would put that stall on the turn path repeatedly;
#: a bounded number of retries recovers a transient failure without it.
_ENCODING_RETRY_AT = 0.0
_ENCODING_RETRY_BASE_SECONDS = 30.0
_ENCODING_RETRY_MAX_SECONDS = 300.0

#: Bumped by :func:`reset_tokenizer_state`. A load attempt snapshots it and
#: discards its outcome if the epoch moved while it ran, so a background warm
#: that finishes after a reset cannot publish its stale result — the exact way
#: a daemon thread used to poison the next test's counts. The compare and the
#: publication are ONE critical section under :data:`_STATE_LOCK`, which the
#: reset also takes: a reset either lands entirely before the publication (the
#: attempt's outcome is discarded) or entirely after it (it clears what was
#: published). Check-then-act was a reproduced race — a reset landing in that
#: window re-armed the latch with a fresh 30 s cooldown.
_ENCODING_EPOCH = 0

#: Epoch of the load attempt currently in flight, ``None`` when idle. Keyed by
#: epoch rather than a bare flag so a claim left over from before a reset cannot
#: block a load for the fresh state: a reset bumps the epoch, which makes the
#: old claim stale, and the old attempt can no longer clear the new one (its
#: release only matches its own epoch).
_LOAD_EPOCH: int | None = None


def _get_encoding() -> object | None:
    """Lazily load the cl100k_base encoding; never raise.

    A missing or broken tiktoken install must degrade to the ``len(text) //
    4`` fallback, not crash compaction (or startup — loading is deferred to
    first use). The warning fires on the FIRST failure of a failure episode
    (later attempts log at debug), so it names the extra without turning into
    per-turn noise.

    A FAILURE IS RETRIED, WITH BACKOFF. The old code latched failure for the
    whole process, so one transient failure — a network fetch that timed out
    on the first turn, a ``sys.modules`` patch a test installed while a warm
    thread was loading — silently made every later count an approximation.
    The first retry waits 30 s, then the delay doubles per attempt to a
    5-minute cap (pinned by
    ``test_the_retry_backoff_doubles_and_caps``), so a permanently offline
    process attempts a bounded number of fetches, while a transient failure
    heals on the next attempt after its window.

    WHO PAYS FOR A LOAD. The FIRST-USE load runs on the calling thread: it
    needs the encoding to answer, and that is the once-per-process cost
    :func:`warm_tokenizer` exists to move off the boot path. A RETRY NEVER
    BLOCKS A CALLER: once a latch exists the fetch is known to be expensive
    (75.7 s to time out offline), so the retry is dispatched to a daemon
    loader thread and this call — and every call until the thread publishes
    — returns ``None`` and counts with the fallback. That is the difference
    between a bounded approximation across a recovering window and a stall
    on the turn path every backoff interval, which is what running the retry
    inline under a lock measured (2.01 s of another thread's
    ``estimate_tokens``).

    CONCURRENCY. Every state transition is taken under :data:`_STATE_LOCK`,
    which is never held across the fetch, so publication is atomic against
    :func:`reset_tokenizer_state` and no unrelated estimate waits on a load.
    A caller that arrives while the FIRST load is in flight waits on
    :data:`_STATE_CHANGED` instead of starting a second load: without that,
    one pass over a message list could count half its messages with each
    ruler. It waits only for a first load, never for a retry.
    """
    global _LOAD_EPOCH
    while True:
        with _STATE_LOCK:
            if _ENCODING is not None:
                return _ENCODING
            if _ENCODING_FAILED and time.monotonic() < _ENCODING_RETRY_AT:
                return None
            epoch = _ENCODING_EPOCH
            if _LOAD_EPOCH == epoch:
                # A load for this state is in flight. A retry is the expensive
                # case: never make this caller wait for it. A first load is
                # cheap and consistency-critical: wait for it to publish.
                if _ENCODING_FAILED:
                    return None
                _STATE_CHANGED.wait(5.0)  # timeout is a safety net, not a wait
                continue
            _LOAD_EPOCH = epoch
            retrying = _ENCODING_FAILED
        if retrying:
            if not _spawn_loader(epoch):
                # No thread to be had (an exhausted process): pay the fetch
                # here rather than leave the retry permanently unscheduled.
                return _load_encoding(epoch)
            with _STATE_LOCK:
                return _ENCODING
        return _load_encoding(epoch)


def _spawn_loader(epoch: int) -> bool:
    """Run one load attempt on a daemon thread; ``False`` when none started."""
    try:
        thread = threading.Thread(
            target=_load_encoding,
            args=(epoch,),
            name="lop-tokenizer-load",
            daemon=True,
        )
        thread.start()
    except RuntimeError:
        logger.debug("tokenizer retry could not start a thread", exc_info=True)
        return False
    return True


def _load_encoding(epoch: int) -> object | None:
    """One load attempt; the caller must own the claim for ``epoch``.

    Runs the fetch with :data:`_STATE_LOCK` released. The outcome is published
    only if the epoch still matches, and the compare-then-store happens inside
    a single critical section — the reset takes the same lock, so it cannot
    land between them and leave a stale latch armed.
    """
    global _ENCODING, _ENCODING_FAILED, _ENCODING_FAILURES, _ENCODING_RETRY_AT
    global _LOAD_EPOCH
    try:
        try:
            import tiktoken

            encoding: object | None = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # noqa: BLE001 - degrade, never raise
            announced = False
            delay = 0.0
            failures = 0
            with _STATE_LOCK:
                # A reset during the load means this outcome describes a state
                # the caller no longer wants; publishing the failure here would
                # re-arm the latch the reset just cleared.
                if epoch == _ENCODING_EPOCH:
                    _ENCODING_FAILED = True
                    _ENCODING_FAILURES += 1
                    delay = min(
                        _ENCODING_RETRY_BASE_SECONDS * 2 ** (_ENCODING_FAILURES - 1),
                        _ENCODING_RETRY_MAX_SECONDS,
                    )
                    _ENCODING_RETRY_AT = time.monotonic() + delay
                    announced = _ENCODING_FAILURES == 1
                    failures = _ENCODING_FAILURES
            # Logged outside the lock: a handler writing to a slow sink must not
            # hold the state lock against every estimator.
            if delay:
                if announced:
                    logger.warning(
                        "%s; falling back to the chars/4 token estimate, "
                        "retrying after %.0fs (%s)",
                        missing_extra_error("tokenizer", "Exact token counting"),
                        delay,
                        exc,
                    )
                else:
                    logger.debug(
                        "tokenizer load failed again (attempt %d); next retry in %.0fs: %s",
                        failures,
                        delay,
                        exc,
                    )
            with _STATE_LOCK:
                return _ENCODING
        with _STATE_LOCK:
            if epoch == _ENCODING_EPOCH:
                _ENCODING = encoding
                _ENCODING_FAILED = False
                _ENCODING_FAILURES = 0
                _ENCODING_RETRY_AT = 0.0
            return _ENCODING
    finally:
        with _STATE_LOCK:
            if _LOAD_EPOCH == epoch:
                _LOAD_EPOCH = None
            _STATE_CHANGED.notify_all()


def reset_tokenizer_state() -> None:
    """Return the tokenizer to a cold state: no encoding, no latch, no backoff.

    Clears the loaded singleton, the failure latch, the failure count and the
    retry deadline, bumps the epoch, AND drops the per-model encoding cache
    (:data:`_MODEL_ENCODINGS`) — a cached per-model encoder was resolved from
    the same tiktoken state and must not outlive it.

    For tests and diagnostics: a process that has already failed a load can be
    returned to a cold, retry-immediately state. Production code never needs
    it — the backoff in :func:`_get_encoding` retries on its own.

    SAFE AGAINST AN IN-FLIGHT LOAD. It takes :data:`_STATE_LOCK`, but that lock
    is never held across a fetch, so this cannot block teardown behind a 75 s
    network load. The epoch bump makes the stale attempt's claim stale (so a
    fresh load can start immediately) and its outcome un-publishable — and
    because the compare and the publication share this lock, a reset cannot
    land *between* them, which is the window that used to re-arm the latch.
    """
    global _ENCODING, _ENCODING_FAILED, _ENCODING_FAILURES, _ENCODING_RETRY_AT
    global _ENCODING_EPOCH
    with _STATE_LOCK:
        _ENCODING = None
        _ENCODING_FAILED = False
        _ENCODING_FAILURES = 0
        _ENCODING_RETRY_AT = 0.0
        _ENCODING_EPOCH += 1
        _MODEL_ENCODINGS.clear()
        _STATE_CHANGED.notify_all()


def warm_tokenizer() -> None:
    """Pay the encoding's first-use cost now, off the first turn's path.

    WHY THIS EXISTS. :func:`_get_encoding` builds cl100k_base on first use:
    tiktoken reads (or fetches) ``cl100k_base.tiktoken`` and base64-decodes
    100,256 BPE ranks into a mergeable-ranks table. Measured on an M-series
    box: **122 ms cold, 0.004 ms warm**. The first use is
    ``providers.context.measure_request``, which
    ``model.configure.SessionStreamFn.__call__`` AWAITS *before* it opens the
    provider request — so the whole cost lands inside time-to-first-token,
    once per process.

    That "once per process" is what makes it worth moving: a TUI is one
    process for hours, but the desktop plane spawns a fresh runtime child for
    every session, so every attach pays it again. Warming it at boot turns a
    first-token cost into a startup cost that overlaps construction.

    Never raises. A missing ``tokenizer`` extra is a supported configuration
    (:func:`_get_encoding` degrades to the chars/4 fallback) and a warm-up
    must not be the thing that reports it.
    """
    try:
        _get_encoding()
    except Exception:  # noqa: BLE001 — a warm-up must never be the failure
        logger.debug("tokenizer prewarm skipped", exc_info=True)


def warm_tokenizer_in_background() -> threading.Thread | None:
    """Run :func:`warm_tokenizer` on a daemon thread; ``None`` when already warm.

    NEVER RAISES — and that is a contract, not a hope: the runtime child calls
    it as the FIRST statement of ``main()``, before the log file is configured,
    so anything escaping here kills a session before it can write down why, and
    the daemon calls it inside its FastAPI ``lifespan``, where escaping fails
    startup. The conditions, the thread and the warm itself are each covered.

    WHY A THREAD RATHER THAN A CALL. Both callers are processes whose first act
    is to spend 600-2,400 ms constructing a session (imports, tool registry,
    transcript). The tokenizer is not part of that work — nothing reads it
    until the first provider request — so overlapping the two removes the
    cost instead of relocating it. The load releases the GIL for its bulk (a
    file read plus the base64 decode of those 100,256 ranks), so the overlap
    is genuine rather than two threads taking turns.

    The unguarded read of ``_ENCODING`` is deliberate and benign: the worst
    case race is starting a thread that joins an in-flight first load, or finds
    the encoding warm and returns. Taking :data:`_CACHE_LOCK` here would be the one thing that
    could serialize this against a concurrent first estimate.

    Returns the thread so a caller — or a test — can join it; ``None`` means
    there is nothing to wait for.

    A previously FAILED load returns ``None`` too: both callers run this once
    per boot, so spawning a thread per boot to re-attempt a load the estimator
    already retries on its own schedule would only add threads to a process
    that most likely cannot load the table at all.
    """
    try:
        return _warm_tokenizer_in_background()
    except Exception:  # noqa: BLE001 — a warm-up must never be the failure
        logger.debug("tokenizer prewarm skipped", exc_info=True)
        return None


def _warm_tokenizer_in_background() -> threading.Thread | None:
    if _ENCODING is not None or _ENCODING_FAILED:
        return None
    try:
        thread = threading.Thread(target=warm_tokenizer, name="lop-tokenizer-warm", daemon=True)
        thread.start()
    except RuntimeError:
        # `can't start new thread`. See the contract above: the cost simply
        # moves back to where it was.
        logger.debug("tokenizer prewarm could not start a thread", exc_info=True)
        return None
    return thread


#: Passed to every ``encode`` call. tiktoken REFUSES by default to encode text
#: containing a special-token literal such as ``<|endoftext|>`` and raises
#: ``ValueError``. Everything counted here is untrusted content — tool output,
#: file contents, model text — so that default turns any transcript that
#: merely MENTIONS the literal into a crash in the estimator, and the
#: estimator runs on every turn from pruning and the compaction threshold
#: check. Reproduced: a bash result containing ``<|endoftext|>`` raised out of
#: ``estimate_tokens``. An empty set means "treat them as ordinary text",
#: which is both non-raising and the correct count for content that is being
#: measured rather than sent as a control token.
_DISALLOWED_SPECIAL: tuple[str, ...] = ()


def _encode_len(text: str) -> int:
    """Token count of ``text`` via cl100k_base, or the chars/4 fallback."""
    if not text:
        return 0
    encoding = _get_encoding()
    if encoding is not None:
        encode = encoding.encode  # type: ignore[attr-defined]
        return len(encode(text, disallowed_special=_DISALLOWED_SPECIAL))
    return len(text) // _CHARS_PER_TOKEN_FALLBACK


#: Per-model encodings, resolved lazily and cached. Separate from the compaction
#: singleton on purpose: compaction wants ONE stable ruler for its budget
#: arithmetic, while an API reporting token stats should answer for the model the
#: caller actually named.
#:
#: ONLY REAL ENCODERS ARE CACHED — never ``None``. ``None`` here means "tiktoken
#: is unavailable right now", which includes a tokenizer cooldown after a
#: transient load failure; caching that verdict would make a 30-300 s cooldown
#: permanent for the model, which is the same
#: silent-degradation-from-a-transient-failure class this module exists to
#: remove. The cache is cleared by :func:`reset_tokenizer_state` along with the
#: rest of the tokenizer state, and a resolution that was in flight across a
#: reset is not republished into the cleared cache (epoch-guarded below).
_MODEL_ENCODINGS: dict[str, object] = {}


def _get_model_encoding(model: str) -> object | None:
    """Encoding for ``model``, or None when tiktoken is unavailable.

    Falls back to the shared cl100k_base encoding for an unknown model name,
    which is what the previous inline implementation did.

    A ``None`` result is NOT cached (see :data:`_MODEL_ENCODINGS`): during a
    failure episode this call returns the fallback for the current count, and
    the next call after recovery resolves and caches the model's real encoder
    instead of inheriting the episode forever.

    A FAILURE EPISODE GATES THIS PATH FOR ITS WHOLE LENGTH. Model resolution
    does not go through :func:`_get_encoding`'s load claim — it calls tiktoken
    directly — so without an explicit latch check every ``count_text_tokens(...
    model=...)`` pays its own synchronous fetches, TWICE per call: a failed
    ``encoding_for_model`` falls through to ``get_encoding("cl100k_base")``.
    Measured with a 0.4 s fake fetch: 7 attempts over 3 calls. The call site is
    per message, per HTTP request (``server/routes/chat.py``), and an offline
    fetch is 75.7 s to time out, so that is minutes per request for as long as
    the process is offline — the unbounded-fetch behaviour the cooldown exists
    to prevent.

    THE GATE IS :data:`_ENCODING_FAILED` ALONE, NOT THE RETRY WINDOW. Gating on
    ``now < _ENCODING_RETRY_AT`` bounded the window and left the boundary
    broken: once the deadline elapses this path has no loader thread to hand
    the retry to, so every caller fetched INLINE again, for the 75.7 s each
    attempt takes. The deadline says when the BASE path may try again; it is
    not the model path's cue to fetch. Recovery arrives only through the base
    load clearing the flag — one recovery path, deliberately (do not add a
    second one here).

    THE TRADE, and it is intentional: an encoding tiktoken has LOCALLY cached
    (model file present, base table missing) is not served during an episode.
    Serving it would mean retrying ``encoding_for_model`` per call, which is
    the fetch storm this gate exists to stop. Keep the gate episode-wide; do
    not "improve" it back into a per-call fetch.

    Cached encoders are still served first: they cost no fetch and outlive a
    cooldown, the same order :func:`_get_encoding` uses (encoding, then latch).
    """
    with _STATE_LOCK:
        cached = _MODEL_ENCODINGS.get(model)
        if cached is not None:
            return cached
        if _ENCODING_FAILED:
            return None
        # Snapshot for the publish guard below: a reset that lands while this
        # resolution is fetching invalidates the result (see the store).
        epoch = _ENCODING_EPOCH
    encoding: object | None = None
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except Exception:  # noqa: BLE001 - unknown model name, not a failure
            encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001 - degrade, never raise
        # May be None while the tokenizer is in cooldown; that is a transient
        # answer, not a property of the model, so it is not cached.
        encoding = _get_encoding()
    if encoding is not None:
        # Epoch-guarded like every other publication of tokenizer state, so
        # :func:`reset_tokenizer_state`'s promise holds for THIS cache too: an
        # encoder resolved from the pre-reset tiktoken state must not be
        # republished after the reset cleared the cache (a reset landing
        # mid-fetch otherwise leaves a stale encoder reported as current — no
        # latch re-arm and no wrong count, but a test-isolation hole). The
        # caller still counts with the object it resolved; only the cache
        # write is discarded.
        with _STATE_LOCK:
            if epoch == _ENCODING_EPOCH:
                _MODEL_ENCODINGS[model] = encoding
    return encoding


def count_text_tokens(text: str, model: str | None = None) -> int:
    """Public token count for a plain string.

    Callers outside compaction (the HTTP API's token stats, for example) need a
    count without reaching for tiktoken themselves. Going through here means
    they inherit the degradation ladder — exact counts when the ``tokenizer``
    extra is installed, the chars/4 estimate otherwise — instead of raising
    ImportError on an install that never asked for it.

    ``model`` selects the model's own encoding. It matters more than it looks:
    cl100k_base and o200k_base agree on ASCII prose and code, but differ by
    ~23% on CJK, so hardcoding one ruler made a reported ``total_tokens``
    over-report for non-Latin content on every o200k model (gpt-4o and later).
    Omit it only when any consistent ruler will do.
    """
    if not text:
        return 0
    encoding = _get_model_encoding(model) if model else _get_encoding()
    if encoding is not None:
        encode = encoding.encode  # type: ignore[attr-defined]
        return len(encode(text, disallowed_special=_DISALLOWED_SPECIAL))
    return len(text) // _CHARS_PER_TOKEN_FALLBACK


def approx_text_tokens(text: str) -> int:
    """Token count for ``text`` that NEVER loads the tokenizer.

    :func:`count_text_tokens` is exact when the ``tokenizer`` extra is present,
    and reaching for that exactness has a price paid once per process: loading
    cl100k_base costs ~84 ms and ~43.6 MB RSS (``scripts/bench_base_overhead``),
    and on a cold cache tiktoken fetches the ranks over the NETWORK — which
    offline is a connection timeout rather than a slow answer. Both
    :func:`local_operator.compaction.pruning.prune_transcript` and the session's
    compaction gate go out of their way to defer that load, precisely so a short
    run does not buy a 43.6 MB table to be told there is nothing to do.

    This is for callers who want a number NOW and would rather be somewhat out
    than spend that: a status readout, a progress hint, a size warning.

    **The error is a property of the content, not a bound.** ``chars // 4`` is
    exactly right only where the text averages four characters per token, and
    real text does not. Measured against cl100k_base:

    ==========================  ============  ========
    payload                     chars/token   error
    ==========================  ============  ========
    English prose                      5.59    +39.8%
    Python source                      3.61     -9.8%
    system prompt + tool JSON          4.30     +7.0%
    minified JSON                      1.82    -54.4%
    CJK                                1.36    -65.9%
    ==========================  ============  ========

    So the honest claim is narrow: on the mix this exists to measure — a Latin
    system prompt plus JSON tool schemas — it runs roughly +7% to +17% high,
    which a percentage rendered to one decimal can carry. It is NOT a
    general-purpose estimator, and a caller measuring CJK or minified payloads
    would be told they had spent a third of what they actually had. Anything
    load-bearing wants :func:`count_text_tokens` and should pay for it.

    Deliberately NOT "use the encoder when it is already resident". That would
    make the same session report different numbers depending on whether
    compaction happened to have run, and an estimate that moves for reasons the
    user cannot see is worse than one that is consistently approximate.
    """
    return len(text) // _CHARS_PER_TOKEN_FALLBACK if text else 0


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Prefix of ``text`` that fits within ``max_tokens`` tokens.

    Token-aligned truncation (decode of the first ``max_tokens`` ids) when
    tiktoken is available; otherwise a conservative ``max_tokens * 4`` chars
    prefix to match the estimator's fallback ratio.
    """
    if max_tokens <= 0 or not text:
        return ""
    encoding = _get_encoding()
    if encoding is not None:
        encode = encoding.encode  # type: ignore[attr-defined]
        ids = encode(text, disallowed_special=_DISALLOWED_SPECIAL)
        if len(ids) <= max_tokens:
            return text
        return encoding.decode(ids[:max_tokens])  # type: ignore[attr-defined]
    limit = max_tokens * _CHARS_PER_TOKEN_FALLBACK
    return text if len(text) <= limit else text[:limit]


# ---------------------------------------------------------------------------
# Estimate cache
# ---------------------------------------------------------------------------

#: Memoized estimates keyed by ``message.id``, LRU-bounded. The cache used to
#: be an unbounded dict that no session ever cleared — the server facade
#: builds and disposes a session per request, and converters mint fresh
#: message ids per call, so every message ever estimated leaked an entry for
#: the life of the process. The bound keeps the memoization useful for a
#: turn's working set without unbounded growth.
_ESTIMATE_CACHE: OrderedDict[str, int] = OrderedDict()
_ESTIMATE_CACHE_MAX = 4096

#: Cross-package subscribers notified on every invalidation. Insertion-ordered
#: dict keyed by ``id(callable)`` so registration is idempotent per callable
#: and unsubscribe is O(1).
_INVALIDATORS: dict[int, Callable[[Message], None]] = {}


@dataclass
class _Inflight:
    """One token computation that is running right now.

    ``message_id`` is what an invalidation matches on; ``invalidated`` is the
    flag it sets. A tiny mutable record rather than a tuple because the flag
    is written by a different thread than the one that created it.
    """

    message_id: str
    invalidated: bool = False


#: In-flight computations, keyed by a unique ticket. Closes a
#: lost-invalidation race that exists only because ``_compute_tokens``
#: deliberately runs OUTSIDE :data:`_CACHE_LOCK`:
#:
#:   thread A reads the cache (miss) -> starts encoding
#:   the loop mutates the message and invalidates it (``pop`` finds nothing)
#:   thread A inserts the count it computed from the PRE-mutation message
#:
#: The stale value then survives forever, because the invalidation that should
#: have removed it already ran. A computation registers a ticket before it
#: encodes; an invalidation flags every ticket for that message id; the
#: computation declines to cache a result whose ticket was flagged. The
#: invalidation therefore wins without pulling the encode under the lock.
#:
#: Keyed by TICKET, not by message id, which is what makes the state bounded
#: and ABA-free. Bounded: an entry exists only while a computation is
#: running, so the dict is sized by concurrency (a handful), not by the
#: number of messages ever seen — an earlier per-id counter leaked one entry
#: per invalidation on the once-per-turn prune path, which never inserts and
#: so never evicted (20k turns -> 20k entries, cache still empty). ABA-free:
#: tickets are unique and never reused, so a stalled computation cannot
#: observe a recycled value and mistake it for its own.
_INFLIGHT_ESTIMATES: dict[int, "_Inflight"] = {}

#: Source of ticket ids. Guarded by :data:`_CACHE_LOCK`; monotonic so a
#: ticket is never reused within a process.
_TICKET_SEQUENCE = itertools.count()


def estimate_tokens(message: Message) -> int:
    """Estimated token cost of ``message`` as sent on the wire.

    Memoized per ``message.id``. Any in-place mutation of the message MUST be
    followed by :func:`invalidate_message_cache` (see module docstring) or a
    stale value is served. Images count as :data:`IMAGE_TOKEN_ESTIMATE` each;
    text is tokenized with cl100k_base or ``len(text) // 4`` when tiktoken is
    unavailable. Tool calls contribute their name + serialized arguments.
    """
    if not _is_settled(message):
        # Streaming assistant, still provisional: compute but never cache.
        return _compute_tokens(message)
    # Lock the LRU bookkeeping, not the encode. ``get`` + ``move_to_end`` and
    # the insert + ``popitem`` eviction are each multi-step sequences on a
    # shared OrderedDict, and the estimators now run in worker threads (see
    # the module docstring), so an unguarded interleaving can evict an entry
    # another thread is mid-promotion on. ``_compute_tokens`` stays OUTSIDE
    # the lock: it is the expensive part, it releases the GIL, and holding the
    # lock across it would re-serialize exactly what the offload exists to
    # parallelize. Two threads racing the same uncached id therefore both
    # compute it — wasteful but correct, and far cheaper than serializing.
    with _CACHE_LOCK:
        cached = _ESTIMATE_CACHE.get(message.id)
        if cached is not None:
            _ESTIMATE_CACHE.move_to_end(message.id)
            return cached
        # Register this computation BEFORE releasing the lock, so any
        # invalidation that lands while it encodes can find and flag it.
        ticket = next(_TICKET_SEQUENCE)
        _INFLIGHT_ESTIMATES[ticket] = _Inflight(message_id=message.id)

    try:
        tokens = _compute_tokens(message)
    except BaseException:
        # The ticket must not outlive its computation, or the bound above
        # ceases to hold the moment an encode raises.
        with _CACHE_LOCK:
            _INFLIGHT_ESTIMATES.pop(ticket, None)
        raise

    with _CACHE_LOCK:
        inflight = _INFLIGHT_ESTIMATES.pop(ticket, None)
        if inflight is None or inflight.invalidated:
            # Invalidated mid-flight: return the value to THIS caller (it is
            # the honest answer for the message as it was read) but leave the
            # cache empty so the next reader recomputes from the current one.
            return tokens
        _ESTIMATE_CACHE[message.id] = tokens
        while len(_ESTIMATE_CACHE) > _ESTIMATE_CACHE_MAX:
            _ESTIMATE_CACHE.popitem(last=False)
    return tokens


def _is_settled(message: Message) -> bool:
    """Cacheability gate (``isEstimateCacheable``): non-assistant messages
    are immutable once appended; assistants only count as settled once the
    loop finalization has stored real ``usage`` and a terminal ``stop_reason``
    (not ``None`` / ``"aborted"`` / ``"error"``)."""
    if message.role != "assistant":
        return True
    return message.usage is not None and message.stop_reason not in (None, "aborted", "error")


def _compute_tokens(message: Message) -> int:
    """Uncached token computation for a single message."""
    total = 0
    text_parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextContent):
            text_parts.append(block.text)
        elif isinstance(block, AttachmentContent):
            # An output artifact NEVER reaches a provider (dispatch renders
            # text/image/audio and stops), so charging it an image estimate
            # would over-count a ruler whose job is "as sent on the wire"
            # (agent review round 1, F4). Zero, explicitly, beside the wire
            # reader's own zero below.
            continue
        elif isinstance(block, DocumentContent):
            # NOT an image: see ``_document_token_estimate``. The previous
            # fall-through charged a flat image estimate, which is wrong for
            # both delivery paths and hid the block from any document-aware
            # accounting (QA round 1 sweep).
            total += _document_token_estimate(block)
        else:
            total += IMAGE_TOKEN_ESTIMATE

    if text_parts:
        total += _encode_len("\n".join(text_parts))

    if message.tool_calls:
        # Tool-call scaffolding + arguments ride along in the wire payload.
        for call in message.tool_calls:
            args = call.raw_arguments or json.dumps(call.arguments, sort_keys=True)
            total += _encode_len(call.name + args)
    return total


def estimate_messages_tokens(messages: Sequence[Message]) -> int:
    """Sum of per-message estimates — the local context-size estimator."""
    return sum(estimate_tokens(m) for m in messages)


def estimate_wire_bytes(messages: Sequence[Message]) -> int:
    """Serialized request size in BYTES — deliberately NOT a token count.

    A third ruler, answering the one question neither token estimator asks:
    "will this HTTP request fit?". Keeping it separate is the whole point.
    :data:`IMAGE_TOKEN_ESTIMATE` is a flat per-image charge because vision
    providers bill by pixel area, so it is *correct* for billing and useless
    as a size proxy — the session this function was written for carried 42
    screenshots worth 50,400 accounted tokens against **33.9 MB** actually
    sent, read as 15.5% of a 1M window while the wire was over Anthropic's
    32 MB cap. Making ``IMAGE_TOKEN_ESTIMATE`` size-aware instead would
    corrupt the billing ruler to fix a transport problem and silently
    re-scale every calibrated threshold in this package.

    So: a number from THIS function may only be compared against a byte
    budget, never against a token figure. That is the same one-ruler-per-
    comparison discipline this module's header establishes, and the mixing
    bug it describes has already shipped twice.

    Unlike the token estimators this is **exact and cheap**: no tokenizer to
    load, no slope to calibrate, ``len(block.data)`` *is* the answer. It is
    cheap enough that the render seam runs it unconditionally and the trigger
    passes an exact figure where the token path needs an upper bound first —
    but "free" would overstate it, and the first revision of this docstring
    did. An image-heavy history scans in ~0.06 ms; a long TOOL-heavy one costs
    a few milliseconds, dominated entirely by sizing tool-call arguments, and
    that is why :func:`_argument_bytes` measures them structurally instead of
    re-serializing (agent review round 1, R3, which measured the naive form at
    6.5-7.9 ms per scan on real transcripts).

    Counts base64 payloads, text, and tool-call name+arguments — the three
    things that carry real length.

    **Accuracy, stated with its sign and its SCOPE.** The never-under property
    holds for TOOL-CALL ARGUMENTS specifically, not for this function's total,
    and the difference matters enough to state plainly rather than let a reader
    generalise it:

    - **Arguments** (:func:`_argument_bytes`, and ``raw_arguments`` beside it)
      are sized exactly for the encoding the provider clients actually use,
      with a small POSITIVE bias — +0.189% measured over 488,138 real tool
      calls, zero under-counts.
    - **The total this function returns UNDER-counts a real history**, by a
      measured mean of -2.42% and a worst case of -17.39% across 4,686 real
      sessions. Two known causes, both deliberate: message text is counted as
      characters rather than UTF-8 bytes, and the request's own JSON envelope
      (roles, keys, block wrappers) is not modelled at all.

    That residual is absorbed by the budget's headroom rather than modelled —
    24 MB against a 32 MB cap is a ~28% margin, several times the worst
    observed shortfall plus the envelope — because a per-provider
    serialization model is a constant that rots. It is a DISCLOSED gap, not a
    claim of exactness: anyone tightening the budget toward the cap has to
    close it first, and the fast path for it already exists (the
    ``isascii()`` shape :func:`_string_bytes` uses).

    The one property that must not regress: the ARGUMENT sizing must never
    come in under the wire. An under-count means believing an oversize request
    fits and sending it, which is the failure this guard exists to prevent —
    and it is exactly what the first remediation of :func:`_argument_bytes`
    introduced (-1.9% aggregate, -75% on CJK) before this revision corrected
    it (agent review round 2 R7 / QA Q5).
    """
    total = 0
    for message in messages:
        for block in message.content:
            if isinstance(block, TextContent):
                total += len(block.text)
            elif isinstance(block, AttachmentContent):
                # Never serialized to a provider, so it contributes nothing to
                # the wire — the same rule as `_compute_tokens` above. Without
                # this arm ANY history holding an artifact raised here (agent
                # review round 1, F1): the block has no `data`, and it never
                # will, because its bytes live in the store.
                continue
            else:
                # Documents included: ``block.data`` is the base64 payload,
                # which IS the wire size of a native part; a fallback renders
                # less than that, so this stays the safe (over-)estimate the
                # byte budget wants.
                total += len(block.data)
        for call in message.tool_calls or ():
            # Prefer the provider's own rendering when we kept it: that string
            # is literally what goes on the wire. Otherwise MEASURE the
            # arguments structurally rather than re-serializing them — see
            # :func:`_argument_bytes` for why that distinction is load-bearing.
            total += _utf8_len(call.name)
            total += (
                # ``raw_arguments`` is ALREADY serialized JSON, so it needs the
                # UTF-8 length and NOT the escape expansion ``_argument_bytes``
                # applies — its quotes and backslashes are the escaping, not
                # content to be escaped again. ``len`` was wrong here for the
                # same reason it was wrong inside the walk: it counts
                # characters, and this is the branch carrying essentially all
                # LIVE traffic (``harness/loop.py`` always populates it), where
                # 6.42% of real calls under-counted, worst -29.79% (agent
                # review round 3, R7 residual).
                _utf8_len(call.raw_arguments)
                if call.raw_arguments
                else _argument_bytes(call.arguments)
            )
    return total


def _utf8_len(text: str) -> int:
    """UTF-8 byte length of ``text``, for a string that is already wire-shaped.

    Shares the surrogate tolerance :func:`_string_bytes` documents at length:
    a lone surrogate is legal in a Python ``str`` and reaches here from a
    transcript, so a plain ``encode`` would raise and kill the render rather
    than return a number.
    """
    if text.isascii():
        return len(text)
    return len(text.encode("utf-8", "surrogatepass"))


def _argument_bytes(arguments: object) -> int:
    """Serialized size of tool-call arguments, WITHOUT serializing them.

    A ``json.dumps`` here would be the same 60x trap
    :func:`messages_tokens_upper_bound` documents, and it would bite far
    harder than it looks: ``raw_arguments`` is deliberately dropped on the way
    to disk (``session/transcript.py`` pops it for a documented space win), so
    **every resumed session has none** — measured at 0 of 39,250 tool calls
    across 400 real sessions. The fallback is the steady state, not the edge
    case.

    That matters because :func:`estimate_wire_bytes` runs at the render seam,
    which is synchronous and on the event loop shared by the parent session,
    every subagent and the TUI repaint. Re-serializing measured 6.5-7.9 ms per
    scan on real transcripts at roughly 3 scans per turn — about 20 ms of
    event-loop stall per turn, the same class of shared-loop cost
    ``Session._offloaded`` exists to prevent (agent review round 1, R3).

    So this walks the structure and adds up what each node will occupy in
    :data:`_WIRE_JSON_ENCODING` — the encoding the provider clients actually
    use, not ``json.dumps``'s defaults. That distinction is the whole point of
    the second revision of this function: sizing strings with ``len(s)``
    UNDER-counts the wire in every direction that matters, and under-counting
    is the one error this guard must never make, because it means believing an
    oversize request fits and sending it. Measured across 487,652 tool calls
    in 4,774 real transcripts, the character walk came out at **-1.9%** (under)
    where the ``json.dumps`` it replaced was **+1.2%** (over); worst real cases
    were ordinary ASCII ``write`` calls at -3% to -12%, and CJK/emoji reached
    -75% (agent review round 1 R7 / QA Q5).

    The three corrections that make it exact rather than approximate:

    - **UTF-8 length, not character count.** ``ensure_ascii=False`` means a
      CJK codepoint rides as 3 bytes and an emoji as 4, where ``len`` said 1.
    - **Short escapes.** ``"``, ``\\`` and the C0 characters JSON has
      two-character forms for each cost one byte MORE than their source.
    - **Long escapes.** The remaining C0 controls render as ``\\uXXXX`` — six
      bytes for one.

    Numbers are sized with ``repr``, which matches ``json.dumps`` exactly for
    every value CPython's encoder emits (it uses ``float.__repr__`` too), so
    the old flat charge of 8 — which under-counted every float and long int —
    is gone.

    The result never under-counts the real wire, and the R3 performance win is
    kept rather than traded back for it. Measured against the honest baseline —
    ``json.dumps`` in the SAME encoding the provider uses, which is what being
    exact the naive way would cost:

    ======================  =========  ===============  =======
    shape                   this       exact-by-dumps   speedup
    ======================  =========  ===============  =======
    200 calls x 4 KB          1.0 ms          6.0 ms     5.8x
    400 calls x 8 KB         11.9 ms         15.1 ms     1.3x
    600 calls x 64 KB        91.5 ms        230.1 ms     2.5x
    ======================  =========  ===============  =======

    The 400x8 KB row is the weakest case and is stated rather than hidden: many
    small strings amortise the per-string overhead worst. It is still faster
    AND correct, where the character walk it replaces was faster and wrong.

    Residual error is a small POSITIVE bias — a trailing separator is charged
    for the last element of every container — measured at **+0.189%** across
    488,138 tool calls in the real session store. The sign is the point: the
    guard may shed marginally early, never send an oversize request believing
    it fits.

    Iterative rather than recursive: arguments come from a model and their
    nesting depth is not bounded by anything we control, so recursion could
    hit the interpreter's stack limit on a pathological payload. A cycle is
    not reachable (these are freshly parsed JSON) but the flat walk would
    terminate on the size accumulator anyway.

    The "sized for the wire" contract holds for WIRE-LEGAL input, which is
    what ``arguments`` can hold: it is always the output of a JSON parse. This
    returns a number for ``NaN``/``inf``, ``bytes``, ``set`` and arbitrary
    objects where the real encoder would raise (``allow_nan=False``, or
    ``TypeError``) — deliberately, because a sizer that raises is the R10
    failure, and by the time such a value could appear the request is already
    unsendable for reasons this function does not exist to detect (agent
    review round 3, R11).
    """
    total = 0
    stack: list[object] = [arguments]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            total += _string_bytes(node) + 2  # the quotes
        elif isinstance(node, dict):
            total += 2  # the braces
            for key, value in node.items():
                # A non-string key is coerced to a string by the encoder, so
                # size what will actually be written rather than the object.
                total += _string_bytes(key if isinstance(key, str) else str(key))
                total += 4  # its quotes, the colon and the separating comma
                stack.append(value)
        elif isinstance(node, (list, tuple)):
            total += 2  # the brackets
            stack.extend(node)
            total += max(0, len(node) - 1)  # the separating commas
        elif node is None or isinstance(node, bool):
            # "null" / "true" / "false" — sized exactly rather than averaged,
            # since the difference is free to compute.
            total += 4 if node is None or node is True else 5
        else:
            # ``repr`` and the JSON encoder agree on every int and float the
            # encoder will emit; a flat charge here under-counted both.
            total += len(repr(node))
    return total


#: What ``httpx`` puts on the wire for a ``json=`` body, which is how every
#: provider client sends a request (``providers/clients.py``): UTF-8 with
#: ``ensure_ascii=False``. Recorded as prose rather than as parameters we pass
#: anywhere, because nothing here serializes — it is the specification
#: :func:`_string_bytes` is written against, and the reason character counts
#: are wrong for this job.
_WIRE_JSON_ENCODING = "utf-8, ensure_ascii=False"

#: The characters JSON renders as a two-character escape (``\\n``, ``\\"``,
#: ``\\\\``, …): one byte MORE than the source character.
_JSON_SHORT_ESCAPES = '"\\\n\t\r\b\f'

#: The C0 controls with no two-character form, which JSON renders as the
#: six-byte ``\\uXXXX`` — five bytes more than the source character. Rare in
#: real arguments, which is why they are counted only when the cheap
#: whole-string check below says at least one is present.
_JSON_LONG_ESCAPES = tuple(
    chr(code) for code in range(0x20) if chr(code) not in _JSON_SHORT_ESCAPES
)

#: Every character that expands at all, as one string, for a single membership
#: sweep before any per-character counting happens.
_JSON_ESCAPED_CHARS = _JSON_SHORT_ESCAPES + "".join(_JSON_LONG_ESCAPES)


def _string_bytes(text: str) -> int:
    """Bytes ``text`` occupies inside a JSON string, excluding its quotes.

    Exact for :data:`_WIRE_JSON_ENCODING`, and shaped so the common case pays
    almost nothing. Two fast paths carry that:

    - ``str.isascii`` is a cached flag rather than a scan, so an ASCII payload
      never pays for the UTF-8 encode.
    - Escapes are counted with ``str.count`` per escape character: each is a
      C-level scan, and the seven short escapes cover essentially all real
      occurrences.
    - The twenty-five ``\\uXXXX`` controls are gated behind ``str.isprintable``,
      which is C-level and never reports a C0 control as printable — so a
      printable string provably contains none and skips all twenty-five scans.
      That gate is what keeps the common case cheap. Two tempting alternatives
      are both far worse and should not be re-introduced: a generator testing
      each control in turn ran 896,000 times on a 400-call history and
      dominated the function, and ``min(text)`` iterates the string in Python
      at ~9 us per KB against ~0.3 us for the whole short-escape loop.

    ``str.translate`` over the same characters reads more neatly than the
    counts but is ~7x slower — it does a dict lookup per character, where
    ``count`` stays in C.

    **``surrogatepass`` is load-bearing, not defensive.** A lone surrogate
    (``\\ud800``) is legal in a Python ``str`` AND legal JSON, so a model that
    emits one round-trips it through ``json.dumps``/``json.loads`` and it
    lands in ``transcript.jsonl`` verbatim. A plain ``encode("utf-8")`` raises
    ``UnicodeEncodeError`` on it — and this function runs at the render seam,
    inside ``_render_history``, which every wire path and ``/compact`` go
    through. So a bare encode turns one stray codepoint into a session that
    raises on every turn forever, including the escape hatch: the exact
    wedge this whole change exists to delete, reintroduced through the
    sizer meant to prevent it (agent review round 3, R10).

    Sizing with ``surrogatepass`` is also the RIGHT number, not merely a
    non-raising one: it is what a lenient encoder puts on the wire for that
    codepoint, so the estimate stays honest instead of being a fallback guess.
    This function must always return a number for any legal Python ``str`` —
    it is a size estimate, and there is no input for which failing to size is
    better than sizing.
    """
    total = len(text) if text.isascii() else len(text.encode("utf-8", "surrogatepass"))
    for char in _JSON_SHORT_ESCAPES:
        if char in text:
            total += text.count(char)
    # The short escapes above (\n, \t, …) also make a string non-printable, so
    # this gate is checked AFTER them and only filters the rare remainder.
    if not text.isprintable():
        for char in _JSON_LONG_ESCAPES:
            if char in text:
                total += 5 * text.count(char)
    return total


def history_chars(messages: Sequence[object]) -> int:
    """Rough size of a history in characters, for the offload decision.

    Deliberately shallow: it walks text blocks only, skipping tool arguments,
    because it exists to answer "is this big enough to be worth a thread hop?"
    — a question a cheap lower bound answers correctly. Anything expensive here
    would defeat its own purpose.

    Consequence worth knowing, since it is a deliberate under-count and not an
    oversight: a history whose bulk lives in tool-call ARGUMENTS (a ``write``
    with a large ``content``) reads as smaller than it is and takes the inline
    path. That is bounded and acceptable — tool RESULTS, the usual bulky item,
    are text blocks and are counted; estimates are memoized per message, so
    arguments are encoded once per message rather than once per turn; and the
    exposure is therefore a one-shot cold pass, measured at 136 ms on a
    pathological 400-turn history against the 860 ms stall the offload exists
    to remove. Sizing arguments here would mean serializing every tool call on
    a probe whose whole value is being cheaper than the work it gates.

    Accepts any message-shaped object so the one definition serves both the
    plain-``Message`` estimator and the cut-point walker, which also sees
    :class:`~local_operator.harness.types.CustomMessage` (no ``content``).
    """
    total = 0
    for message in messages:
        content = getattr(message, "content", None)
        if not content:
            continue
        for block in content:
            if isinstance(block, TextContent):
                total += len(block.text)
    return total


def messages_tokens_upper_bound(messages: Sequence[Message]) -> int:
    """A cheap value that is NEVER below :func:`_compute_tokens`.

    Exists so a caller can prove "nowhere near the compaction threshold"
    without touching tiktoken. The first real estimate in a process loads the
    cl100k_base BPE table, which costs ~84 ms and ~43.6 MB RSS (measured with
    scripts/bench_base_overhead.py) — the single largest item in the peak RSS
    of a short session, spent so a threshold check on a few thousand tokens can
    return False.

    The bound is rigorous, not heuristic, which is what lets a caller
    substitute it into a monotonic ``context_tokens > threshold`` test:

    * cl100k_base is a byte-level BPE, so every token consumes at least one
      UTF-8 byte and ``tokens <= utf8_bytes`` always holds. The chars/4
      fallback used when tiktoken is missing is smaller still.
    * ``str.isascii()`` is an O(1) flag read on CPython, so the common
      all-ASCII block gets the tight bound ``len(text)`` for free; anything
      else uses ``4 * len(text)``, the maximum UTF-8 width per code point.
      Neither path encodes the string, so nothing is allocated.
    * Images and tool calls are counted exactly as :func:`_compute_tokens`
      counts them, so the bound cannot slip under the real estimate when a
      history is mostly images.

    PRECONDITION for the claim about :func:`estimate_messages_tokens`. The
    bound dominates :func:`_compute_tokens` for ANY input, unconditionally.
    It therefore dominates :func:`estimate_messages_tokens` only while this
    module's invalidation contract holds, because that path serves
    ``_ESTIMATE_CACHE`` entries keyed on ``message.id``: a message mutated in
    place without :func:`invalidate_message_cache` keeps the estimate of its
    OLD content, which the bound on the NEW content has no reason to exceed.
    Blanking a settled 1501-token message to ``"[pruned]"`` without
    invalidating yields ``est=1501`` against ``bound=8``; invalidating first
    yields ``est=4`` against the same bound. The failure is
    silent and one-directional: a stale LARGER cached estimate makes the
    bound's early return claim "nowhere near the threshold" for a session that
    is over it, so compaction never fires. All three in-place mutation owners
    (``pruning._blank``, loop finalize, and streaming deltas — never cached
    while unsettled) invalidate today; that is what this bound rests on.

    Deliberately NOT memoized, but not because it is cheap in relative terms —
    on a 400-message tool-heavy history it is SLOWER than the cached estimate,
    measured on this repo (Python 3.13, M3 Max): 0.107 ms vs 0.050 ms when
    every call carries ``raw_arguments`` (~2x), and 3.2 ms vs 0.050 ms when it
    does not (~60x), because the ``json.dumps`` fallback re-serializes every
    tool call in the history on every turn. The reason it stays unmemoized is
    that single-digit milliseconds per turn is nothing against the ~84 ms and
    ~43.6 MB the tokenizer costs once, while a second cache keyed on message
    id would be one more structure every mutation owner must invalidate — a
    new way to produce exactly the stale-value failure described above.

    Scale note: the bound is loose, so the early return only covers the low
    end of the range. Measured ratios of bound to exact estimate: ~3.5-4.5x on
    ASCII prose, chat text and source code (cl100k averages ~4 bytes/token and
    the bound charges one token per byte), ~3.8x on pure CJK, but ~18x on
    ASCII text sprinkled with non-ASCII, where a single non-ASCII character
    makes the whole block take the 4x-per-code-point branch. So the tokenizer
    is skipped only while real context is roughly under ``threshold / 4`` and
    possibly much less; past that the caller falls through to the real
    estimator and loads tiktoken anyway. The substitution buys the cheap
    common case, not the whole range.
    """
    total = 0
    for message in messages:
        text_blocks = 0
        for block in message.content:
            if isinstance(block, TextContent):
                text = block.text
                total += len(text) if text.isascii() else 4 * len(text)
                text_blocks += 1
            elif isinstance(block, DocumentContent):
                # MUST mirror ``_compute_tokens`` exactly: this function's one
                # contract is never being below it, and a document charged
                # differently here would break that invariant silently.
                total += _document_token_estimate(block)
            else:
                total += IMAGE_TOKEN_ESTIMATE
        # _compute_tokens encodes the text blocks as ONE "\n"-joined string, so
        # the separators are inside what it tokenizes. Charging them here keeps
        # the bound above the estimate for a many-block message whose parts are
        # each a single token.
        total += max(0, text_blocks - 1)
        for call in message.tool_calls or ():
            args = call.raw_arguments or json.dumps(call.arguments, sort_keys=True)
            text = call.name + args
            total += len(text) if text.isascii() else 4 * len(text)
    return total


def invalidate_message_cache(message: Message) -> None:
    """Drop the cached estimate for ``message`` and notify subscribers.

    MUST be called after any in-place mutation of a message (pruning blanks,
    streaming finalize). Idempotent; safe for never-cached messages.

    The pop is locked for the same reason the LRU sequences in
    :func:`estimate_tokens` are: a worker thread may be mid-eviction on the
    same OrderedDict. Subscriber callbacks run OUTSIDE the lock — they are
    arbitrary cross-package code and must never be able to deadlock the
    estimator by re-entering it.
    """
    with _CACHE_LOCK:
        _ESTIMATE_CACHE.pop(message.id, None)
        # Flag even when nothing was cached: the entry this invalidation is
        # racing may not have been INSERTED yet (see _INFLIGHT_ESTIMATES).
        # Popping a miss and returning is exactly how the stale value used to
        # survive. Costs one pass over the in-flight computations, which are
        # bounded by concurrency rather than by history size.
        for inflight in _INFLIGHT_ESTIMATES.values():
            if inflight.message_id == message.id:
                inflight.invalidated = True
    for invalidator in list(_INVALIDATORS.values()):
        try:
            invalidator(message)
        except Exception:  # noqa: BLE001 - a subscriber must not break pruning
            logger.exception("message-cache invalidator failed")


def register_invalidator(invalidator: Callable[[Message], None]) -> Callable[[], None]:
    """Subscribe ``invalidator(message)`` to every cache invalidation.

    Returns an unsubscribe callable. For cross-package consumers that keep
    their own derived caches (e.g. provider payload caches) coherent with the
    token estimates.
    """
    _INVALIDATORS[id(invalidator)] = invalidator

    def _unsubscribe() -> None:
        _INVALIDATORS.pop(id(invalidator), None)

    return _unsubscribe


def clear_estimate_cache() -> None:
    """Wipe the whole estimate cache (tests, memory pressure)."""
    with _CACHE_LOCK:
        _ESTIMATE_CACHE.clear()
        # In-flight computations are flagged rather than dropped: a caller
        # that wipes the cache must not have a computation started before the
        # wipe land in the fresh one. Their tickets are still removed by the
        # computations themselves, so this leaks nothing.
        for inflight in _INFLIGHT_ESTIMATES.values():
            inflight.invalidated = True
