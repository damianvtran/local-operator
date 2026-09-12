#!/usr/bin/env python3
"""Measure OpenRouter prompt-cache hit rate with and without cache affinity.

OpenRouter's default route is price-weighted load balancing across many
upstream hosts (~11 endpoints for ``deepseek/deepseek-v4.1-flash``), and each
switch is a COLD prompt cache: DeepSeek needs a full prefix match from token 0,
so a re-routed turn re-bills the whole conversation. Server-side sticky routing
fed by ``prompt_cache_key`` does not hold under load. This benchmark runs the
SAME integration path twice over a long multi-turn loop -- once with
``providers.openrouter.provider_affinity`` off (baseline: today's shipped
behaviour) and once with it on (the pin) -- so the only difference between arms
is whether the harness names the host that served the previous turn.

WHY THE REAL PATH, not a transport A/B: the pin is produced by
``SessionStreamFn`` (capture on ``StreamEndEvent.served_provider``, gate in
``_affinity_enabled``, stamp onto ``ChatRequest.provider_affinity``) and only
then rendered by the wire client. Stripping a field in a transport wrapper
would measure the wire half while leaving the half that decides WHEN to pin
untested -- and that half is where the feature can silently no-op. So each arm
constructs a real ``SessionStreamFn`` over the real ``AuthStore`` with a real
settings mapping, and the transport is a passive OBSERVER that fails the run if
the arms are not actually different on the wire.

ISOLATION (mandatory, and checked rather than assumed):
  * ``LOCAL_OPERATOR_CONFIG_DIR`` and ``HOME`` are redirected to a temp dir for
    the whole run, so nothing touches the operator's live sessions, analytics
    ledger, or model catalogue cache. A redirected config dir alone is NOT
    enough -- the catalogue cache derives its root from the home directory
    independently (see AGENTS.md, "Isolating a run").
  * ``auth.db`` is COPIED into that temp dir from a read-only URI handle; the
    live store is never opened for writing and the key is never printed.
  * The run refuses to start if the copy carries no openrouter row, rather than
    silently measuring nothing.
  * A synthetic session id and a per-arm unique prefix namespace AND cache
    lineage, so arms can neither share nor poison each other's cache groups.

COST: each turn sends the whole grown context (~60k tokens) and asks for a
short answer. At deepseek-v4.1-flash list pricing the default run (2 arms x 30
turns) estimates well under $1; the hard ceiling is ``--budget`` (default
$5.00), checked after every turn against the provider's OWN reported cost, and
the run aborts the moment the estimate would cross it.

THE COMPACTION BOUNDARY (``--compact-at``): the cache-quality guard retires a
host after two consecutive warm-but-uncached turns, and a COMPACTION is the
awkward place for that heuristic — a cold-by-design provider call next to a
rebuilt prefix the host has never been given. ``--compact-at N`` performs a real
compaction at turn N (a ``purpose="compaction"`` call, then a rebuilt prefix)
and FAILS the lane if a host serving the remainder was retired without being
backed by ``PROVIDER_STRIKES_TO_RETIRE`` cold turns of its own.

BE PRECISE ABOUT WHAT THIS PROVES, because the obvious reading is wrong (QA
round 2, Q1). It is a SMOKE assertion that the fixed sequence runs clean end to
end — no unbacked retirement, caching resumed — and NOT a regression test for
the ``purpose == "turn"`` gate. The reason is arithmetic: the large synthetic
prefix lives in ``system_blocks``, and a compaction request replaces those with
``COMPACTION_SYSTEM`` plus a flattened short history, so its call carries
~170-180 prompt tokens against ``PROVIDER_STRIKE_MIN_PROMPT_TOKENS = 8192``.
That floor is checked BEFORE any strike is recorded and on every revision of
this code, so the compaction returns early regardless — this check cannot
distinguish the fixed head from the unfixed one. QA demonstrated it by
relabelling the bench's compaction as ``purpose="turn"``, which is exactly what
deleting the gate would do: identical outcome at 171 tokens, divergent at 74k.

So the gate's regression coverage is the UNIT SUITE (14 tests under
``tests/unit/model/test_configure.py -k "compaction or retire or strike or
purpose"``, each mutation-verified to fail when its guard is removed). What the
live check is genuinely worth is the post-boundary retirement PATTERN, which
unit tests cannot observe because it depends on how real upstreams behave: it
holds the observed sequence to "a retirement must be backed by at least two
cold turns of that host's own". Giving ``--compact-at`` a large-prompt mode
(putting the transcript in ``messages`` so the compaction crosses 8192) would
make it cover the gate for real — deliberately not done here, since the shipped
behaviour is correct and only this claim needed fixing.

Examples:
    .venv/bin/python scripts/bench_openrouter_cache_rate.py --dry-run
    .venv/bin/python scripts/bench_openrouter_cache_rate.py --live
    .venv/bin/python scripts/bench_openrouter_cache_rate.py --live --arm on
    .venv/bin/python scripts/bench_openrouter_cache_rate.py --live \
        --runs 3 --output evidence.jsonl   # alternates arm order per run
    .venv/bin/python scripts/bench_openrouter_cache_rate.py --live --arm on \
        --turns 8 --compact-at 4          # compaction-boundary smoke check
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    # Annotation-only: every `local_operator` import in this script is deferred
    # into the function that needs it so `--dry-run` prints a plan without
    # loading the harness, and `from __future__ import annotations` above makes
    # the signature below cost nothing at runtime.
    from local_operator.harness.types import Message

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MODEL_ID = "deepseek/deepseek-v4.1-flash"
TERSE = " Reply in 10 words or fewer. No code, no lists."
TURN_PROMPTS = [
    "Name one item from the log above.",
    "How many revisions are mentioned?",
    "Name the approved status word.",
    "Which record number did you cite last?",
    "Name any record in the log.",
    "State the namespace word only.",
]


@dataclass
class TurnRecord:
    """One call's provider counters. ``cached`` is a SUBSET of ``prompt``."""

    turn: int
    arm: str
    run: int
    prompt_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    provider: str | None = None
    generation_id: str | None = None
    stop_reason: str | None = None
    usd_cost: float = 0.0
    pin_on_wire: str | None = None
    #: Hosts this conversation had retired as of this turn (cache-quality
    #: guard), and the `provider.ignore` list actually sent.
    avoid_on_wire: list[str] = field(default_factory=list)
    retired: list[str] = field(default_factory=list)
    #: Which concurrent conversation within the arm produced this turn.
    lane: int = 0
    #: ``"compaction"`` for the deliberately cold summary call `--compact-at`
    #: injects, ``"turn"`` otherwise. A compaction's counters must be read
    #: separately: it is a fresh write-once prefix, so its ~0% reuse is the
    #: design and pooling it into the lane's cache share would understate the
    #: feature. Also the marker for "this turn is the first on a REBUILT
    #: prefix", which is cold for the same honest reason.
    purpose: str = "turn"
    #: Wall seconds since this lane's previous turn STARTED. Recorded because
    #: a host-side cache entry expires on a timer (~10 min documented), so a
    #: bench whose natural spacing is far wider than a real session's cannot
    #: tell an eviction apart from a re-route.
    gap_s: float = 0.0

    @property
    def cache_share(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


@dataclass
class ArmResult:
    name: str
    run: int
    lane: int = 0
    turns: list[TurnRecord] = field(default_factory=list)
    error: str | None = None

    @property
    def total_prompt(self) -> int:
        return sum(t.prompt_tokens for t in self.turns)

    @property
    def total_cached(self) -> int:
        return sum(t.cached_tokens for t in self.turns)

    @property
    def cache_rate(self) -> float:
        return self.total_cached / self.total_prompt if self.total_prompt else 0.0

    @property
    def avoidable_loss(self) -> float:
        """Share of prompt tokens that a warm cache SHOULD have covered.

        The previous turn's prompt is a lower bound on what this turn's prefix
        repeats (the transcript only ever grows), so anything below that which
        was not served from cache is loss the pin is meant to remove. Turn 0
        has no predecessor and contributes nothing.
        """
        loss = 0
        previous: int | None = None
        for turn in self.turns:
            if previous is not None:
                loss += max(0, previous - turn.cached_tokens)
            previous = turn.prompt_tokens
        return loss / self.total_prompt if self.total_prompt else 0.0

    @property
    def switches(self) -> int:
        """Host changes between consecutive turns; each one is a cold prefix."""
        names = [t.provider for t in self.turns if t.provider]
        return sum(1 for a, b in zip(names, names[1:]) if a != b)

    @property
    def compaction_turns(self) -> list[TurnRecord]:
        return [t for t in self.turns if t.purpose == "compaction"]

    @property
    def transitions(self) -> int:
        """Turn-to-turn pairs, the denominator switch frequency is read against.

        Reported alongside ``switches`` because a run with NO switches is a
        no-pressure sample, not evidence that the pin is unnecessary: the pin
        can only pay where re-routing happens, so an arm that never re-routed
        had nothing to lose and must be read that way rather than as a null
        result for the mechanism.
        """
        return max(0, len([t for t in self.turns if t.provider]) - 1)

    def conditional(self) -> dict[str, dict[str, int]]:
        """Cache outcome split by whether the host CHANGED on that transition.

        This is the measurement that isolates the mechanism. Aggregate cache
        share confounds two populations — turns that stayed on their host
        (where a warm cache exists to hit) and turns that moved (where it does
        not) — so a window that happened not to re-route reports a null
        difference between arms while the underlying effect is unchanged.
        """
        out = {
            "same": {"n": 0, "hits": 0, "cached": 0, "prompt": 0},
            "changed": {"n": 0, "hits": 0, "cached": 0, "prompt": 0},
        }
        for previous, current in zip(self.turns, self.turns[1:]):
            if not (previous.provider and current.provider):
                continue
            bucket = out["changed" if previous.provider != current.provider else "same"]
            bucket["n"] += 1
            bucket["hits"] += int(current.cached_tokens > 0)
            bucket["cached"] += current.cached_tokens
            bucket["prompt"] += current.prompt_tokens
        return out

    @property
    def pin_honored(self) -> tuple[int, int]:
        """(served by the pinned host, turns that carried a pin).

        ``order`` is a PREFERENCE, not a constraint — OpenRouter falls through
        to another host when the pinned one is loaded or erroring — so this is
        the honest measure of how often the pin actually placed the call.
        """
        pinned = [t for t in self.turns if t.pin_on_wire]
        return sum(1 for t in pinned if t.provider == t.pin_on_wire), len(pinned)

    @property
    def gaps(self) -> list[float]:
        """Observed inter-turn spacing, which the cache TTL is read against."""
        return [t.gap_s for t in self.turns if t.gap_s > 0]

    @property
    def mix(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for turn in self.turns:
            if turn.provider:
                counts[turn.provider] = counts.get(turn.provider, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    @property
    def usd_cost(self) -> float:
        return sum(t.usd_cost for t in self.turns)


class _ObserverTransport(httpx.AsyncBaseTransport):
    """Passive: records whether a request carried ``provider.order``.

    It never edits a request. The arms differ because the SETTINGS differ, so
    this exists purely to prove the gate actually fired -- an ON arm that never
    sends ``provider.order``, or an OFF arm that ever does, means the harness
    half is broken and the two arms are silently measuring the same thing.
    """

    def __init__(self) -> None:
        self.inner = httpx.AsyncHTTPTransport(retries=0)
        self.pins_sent: list[str | None] = []
        self.last_pin: str | None = None
        self.last_avoid: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        pin = None
        try:
            body = json.loads(request.content)
            provider_obj = body.get("provider") or {}
            order = provider_obj.get("order")
            if isinstance(order, list) and order:
                pin = str(order[0])
            ignore = provider_obj.get("ignore")
            self.last_avoid = [str(x) for x in ignore] if isinstance(ignore, list) else []
        except (ValueError, UnicodeDecodeError, AttributeError):
            pass
        self.last_pin = pin
        self.pins_sent.append(pin)
        return await self.inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self.inner.aclose()


def _guard_openrouter_credential(auth_db: Path) -> None:
    """Refuse to run against a copy with no openrouter row.

    Read-only URI handle, and the value is never read into a printable place:
    this asserts EXISTENCE only. Without the guard a miscopied store produces a
    run of auth failures that looks like a measurement.
    """
    with sqlite3.connect(auth_db.resolve().as_uri() + "?mode=ro", uri=True) as db:
        count = db.execute(
            "SELECT COUNT(*) FROM auth_credentials "
            "WHERE provider='openrouter' AND disabled_cause IS NULL"
        ).fetchone()[0]
    if not count:
        raise SystemExit(
            "No usable openrouter credential in the benchmark's auth.db copy; "
            "log in outside this benchmark and re-run."
        )


#: System prompt for the injected compaction call. Short and DELIBERATELY
#: unrelated to `_system_blocks`: a real compaction sends a summariser prompt,
#: not the session's frozen prefix, which is a large part of why its request
#: cannot match anything the host has cached.
COMPACTION_SYSTEM = (
    "You compact conversations. Summarise the exchange below in two sentences. "
    "No preamble, no lists."
)


def _compaction_prompt(history: Sequence[Message]) -> str:
    """Flatten the transcript into the single user prompt a compaction sends."""
    # Imported here, not at module scope, for the reason the rest of this script
    # defers its `local_operator` imports: `--dry-run` must print a plan without
    # loading the harness.
    from local_operator.harness.types import TextContent

    lines: list[str] = []
    for message in history:
        text = " ".join(
            part.text for part in message.content if isinstance(part, TextContent)
        ).strip()
        if text:
            lines.append(f"{message.role}: {text}")
    return "\n".join(lines) or "user: (nothing yet)"


def _system_blocks(namespace: str, rows: int) -> list[str]:
    """A stable synthetic prefix per arm; never private context.

    The unique namespace must FILL the first cache block rather than merely
    name itself: an aggregator matches content prefixes at block granularity,
    so a short namespace line followed by shared synthetic records lets
    unrelated invocations hit each other's cache and report a warm start that
    this run did not earn (measured on the xAI bench, which this mirrors).
    """
    padding = " ".join(f"namespace-{namespace}-{i}" for i in range(rows * 4))
    return [
        f"Synthetic OpenRouter cache benchmark, namespace {namespace}. {padding}",
        " ".join(
            f"Synthetic record {i}: approved status is GREEN, revision {i % 7}, "
            f"owner team-{i % 11}, region r{i % 5}."
            for i in range(rows)
        ),
    ]


def _source_identity() -> dict[str, Any]:
    from local_operator.model import configure
    from local_operator.providers import clients

    try:
        revision = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "source_revision": revision,
        "adapter_sha256": hashlib.sha256(Path(clients.__file__).read_bytes()).hexdigest(),
        "configure_sha256": hashlib.sha256(Path(configure.__file__).read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _settings(arm: str) -> dict[str, Any]:
    """The ONLY difference between arms: the harness-side affinity switch.

    Every other openrouter key stays at its "no opinion" default, so neither
    arm sends a `provider` routing object of its own and the OFF arm is
    byte-for-byte today's shipped request path.
    """
    return {"providers": {"openrouter": {"provider_affinity": arm == "on"}}}


async def _run_lane(
    args: argparse.Namespace,
    arm: str,
    run: int,
    lane: int,
    auth_db: Path,
    output: Any,
    lock: asyncio.Lock,
) -> ArmResult:
    """One conversation: its own session, prefix namespace and cache lineage.

    A lane is the unit the pin operates on, so concurrency is expressed by
    running several of these at once rather than by interleaving turns of one
    conversation — the latter would measure a workload nobody runs.
    """
    from local_operator.harness.types import (
        ChatRequest,
        Message,
        StreamEndEvent,
        StreamTextDelta,
        StreamUsageEvent,
        TextContent,
    )
    from local_operator.model.configure import SessionStreamFn, build_model_spec
    from local_operator.providers.auth_store import AuthStore

    result = ArmResult(arm, run, lane)
    namespace = f"{args.seed}:{arm}:{run}:{lane}"
    blocks = _system_blocks(namespace, args.prefix_rows)
    spec = build_model_spec("openrouter", args.model)
    if not spec.supports_prompt_cache:
        # The gate refuses to pin a model with no server-side cache, so a
        # registry that does not know this model would make both arms identical
        # and the run meaningless. Say so instead of reporting a null result.
        result.error = f"{args.model} does not advertise prompt cache support; nothing to measure"
        return result

    store = AuthStore(auth_db)
    transport = _ObserverTransport()
    # A synthetic session id per arm, run AND lane: the cache lineage derives
    # from it, so no two conversations share a sticky-routing group.
    stream = SessionStreamFn(store, _settings(arm), f"bench-{namespace}")
    await stream._http.aclose()
    stream._http = httpx.AsyncClient(
        transport=transport, timeout=httpx.Timeout(600.0, connect=30.0)
    )
    stream._transport.http = stream._http

    history: list[Message] = []
    last_start: float | None = None
    # Turn indices that send a `purpose="compaction"` request instead of an
    # ordinary turn. Nothing is injected unless --compact-at asks for it, so
    # the default run is byte-identical to the measured A/B.
    compaction_turns = {args.compact_at} if args.compact_at is not None else set()
    try:
        for index in range(args.turns):
            compacting = index in compaction_turns
            prompt = TURN_PROMPTS[index % len(TURN_PROMPTS)] + TERSE
            if compacting:
                # Shaped like the real thing (`ChatCompaction._one_shot_
                # complete`): the transcript arrives as ONE user prompt to be
                # summarised, no tools, `context_tokens_hint=0` because this is
                # a fresh write-once prefix rather than the turn's cached one.
                # That shape is exactly why it cannot cache, and why scoring it
                # as cache evidence retired hosts that cache fine.
                #
                # It also makes the call SHORT here (~170-180 tokens), because
                # this bench's bulk lives in `system_blocks` and those are what
                # a compaction replaces. Short enough to sit under the strike
                # floor, which is why this lane cannot exercise the purpose gate
                # — see the module docstring (QA round 2, Q1).
                request = ChatRequest(
                    model=spec,
                    system_blocks=[COMPACTION_SYSTEM],
                    messages=[Message.user(_compaction_prompt(history))],
                    tools=[],
                    tool_choice="none",
                    replayable=True,
                    purpose="compaction",
                    context_tokens_hint=0,
                )
            else:
                request = ChatRequest(
                    model=spec,
                    system_blocks=blocks,
                    messages=[*history, Message.user(prompt)],
                )
            started = time.monotonic()
            record = TurnRecord(
                turn=index,
                arm=arm,
                run=run,
                lane=lane,
                purpose="compaction" if compacting else "turn",
                gap_s=round(started - last_start, 2) if last_start is not None else 0.0,
            )
            last_start = started
            text = ""
            usage = None
            terminal: Any = None
            try:
                async with asyncio.timeout(300):
                    async for event in stream(request, None):
                        if isinstance(event, StreamTextDelta):
                            text += event.delta
                        elif isinstance(event, StreamUsageEvent):
                            usage = event.usage
                        elif isinstance(event, StreamEndEvent):
                            terminal = event
                            usage = event.usage or usage
                if terminal is None or terminal.stop_reason not in ("stop", "toolUse"):
                    raise ValueError(f"Incomplete response: {terminal and terminal.stop_reason}")
                if usage is None:
                    raise ValueError("Provider completed without usage; cache rate is unknown.")
                record.prompt_tokens = usage.input_tokens
                record.cached_tokens = usage.cache_read_tokens
                record.output_tokens = usage.output_tokens
                record.usd_cost = usage.usd_cost or 0.0
                record.provider = terminal.served_provider
                record.generation_id = (terminal.provider_payload or {}).get("id")
                record.stop_reason = terminal.stop_reason
                record.pin_on_wire = transport.last_pin
                record.avoid_on_wire = transport.last_avoid
                # Read off the session rather than the wire: a retirement that
                # happens on THIS turn is only visible on the next request, and
                # the run has to be able to show that the guard fired at all.
                record.retired = sorted(stream._provider_retired.get(spec.model_id, ()))
                if compacting:
                    # REPLACE the transcript with the summary, which is what a
                    # real compaction does to the prefix and the reason the next
                    # turn cannot hit the host's cache: from token 0 onward the
                    # request no longer matches anything the host holds. The
                    # guard must survive that cold turn without retiring a host
                    # whose only failing is that the prefix changed underneath
                    # it.
                    history = [
                        Message.user("Summary of the conversation so far:"),
                        Message(
                            role="assistant",
                            content=[TextContent(text=text or "(empty)")],
                        ),
                    ]
                else:
                    history = [
                        *history,
                        Message.user(prompt),
                        Message(role="assistant", content=[TextContent(text=text or "(empty)")]),
                    ]
                result.turns.append(record)
                line = json.dumps(
                    {
                        "turn": record.turn,
                        "arm": record.arm,
                        "run": record.run,
                        "lane": record.lane,
                        "prompt_tokens": record.prompt_tokens,
                        "cached_tokens": record.cached_tokens,
                        "cache_share": round(record.cache_share, 4),
                        "output_tokens": record.output_tokens,
                        "provider": record.provider,
                        "generation_id": record.generation_id,
                        "stop_reason": record.stop_reason,
                        "usd_cost": record.usd_cost,
                        "pin_on_wire": record.pin_on_wire,
                        "avoid_on_wire": record.avoid_on_wire,
                        "retired": record.retired,
                        "purpose": record.purpose,
                        "gap_s": record.gap_s,
                    }
                )
                # Serialised: concurrent lanes share one output stream, and
                # interleaved partial writes would corrupt the JSONL evidence.
                async with lock:
                    print(line, file=output, flush=True)
            except Exception as exc:  # noqa: BLE001 - one failed turn fails the lane, loudly
                # The TYPE, always: several provider/timeout errors carry an
                # empty str(), and a bare "turn 1: " tells a reader nothing
                # about whether the run hit a rate limit or a bug.
                result.error = f"turn {index}: {type(exc).__name__}: {exc}".rstrip(": ")
                break
            if index + 1 < args.turns:
                await asyncio.sleep(args.gap)
    finally:
        await stream.close()
        store.close()

    # The wire check. An ON lane that never pinned, or an OFF lane that ever
    # did, means the arms were not actually different and the numbers would be
    # a comparison of one condition with itself.
    pinned = [p for p in transport.pins_sent if p]
    if arm == "on" and result.turns and not pinned:
        result.error = result.error or (
            "ON arm never sent provider.order: the affinity gate never fired, "
            "so this run compares the baseline against itself"
        )
    if arm == "off" and pinned:
        result.error = result.error or (
            f"OFF arm sent provider.order {len(pinned)}x: the switch is not honoured"
        )
    if args.compact_at is not None:
        result.error = result.error or _compaction_guard_verdict(result, args.compact_at)
    return result


def _compaction_guard_verdict(result: ArmResult, compact_at: int) -> str | None:
    """Hold a live compaction boundary to the retirement pattern it must show.

    SCOPE, stated first because the name invites a stronger reading (QA round 2,
    Q1): this is a SMOKE assertion that the sequence runs clean, not a
    regression test for the ``purpose == "turn"`` gate. The compaction call this
    bench sends carries ~170-180 prompt tokens — the synthetic prefix lives in
    ``system_blocks``, which a compaction replaces — and
    ``PROVIDER_STRIKE_MIN_PROMPT_TOKENS = 8192`` is checked before any strike is
    recorded, on every revision of this code. So the call short-circuits either
    way and this function cannot tell a fixed head from an unfixed one. The
    gate's regression coverage is the unit suite (14 tests, mutation-verified);
    see the module docstring for the measurement.

    What it IS worth is the post-boundary retirement pattern, which unit tests
    cannot observe because it depends on how real upstreams behave: across live
    lanes, a retirement must be earned by that host's own cold turns rather than
    by the boundary itself.

    What this does NOT assert is that no host is ever retired after a
    compaction. That would be a stricter claim than the fix makes, and it would
    fail on the guard doing its job: a host that misses the rebuilt prefix AND
    then misses its own write is a host that does not cache, and leaving it is
    the whole point (observed live: a lane retired such a host and its
    replacement immediately cached at 99.8%). So the invariants are:

    1. every post-boundary retirement is backed by at least
       ``PROVIDER_STRIKES_TO_RETIRE`` cold TURNS served by that host after the
       boundary — the compaction call cannot be one of them, and neither can a
       lone cold turn on the rebuilt prefix. (That turn is not reliably cold
       either way: its prefix is the system blocks the host still holds plus the
       new summary, so above a ~1-2k system prefix it clears the
       ``max(1024, 2%)`` floor and CLEARS the record instead — review round 2,
       NIT-1. Observed live both ways, 99.8% in one lane and 0.0% in others.);
       and
    2. the boundary CLEARED whatever was retired before it, since those
       retirements were recorded against a prefix that no longer exists.
    """
    from local_operator.model.configure import SessionStreamFn

    after = [t for t in result.turns if t.turn > compact_at and t.purpose == "turn"]
    if not after:
        return f"--compact-at {compact_at} left no post-boundary turns to judge"

    def cold(turn: TurnRecord) -> bool:
        # The guard's own "no meaningful reuse" floor, read off the class rather
        # than restated, so the bench cannot drift from the code it checks.
        floor = max(
            SessionStreamFn.PROVIDER_STRIKE_MIN_CACHED_TOKENS,
            int(turn.prompt_tokens * SessionStreamFn.PROVIDER_STRIKE_MIN_CACHED_FRACTION),
        )
        return turn.cached_tokens < floor

    for host in sorted(set(after[-1].retired)):
        backing = [t for t in after if t.provider == host and cold(t)]
        if len(backing) < SessionStreamFn.PROVIDER_STRIKES_TO_RETIRE:
            return (
                f"compaction guard: {host} was retired after the turn-{compact_at} "
                f"compaction on only {len(backing)} cold turn(s) of its own "
                f"({SessionStreamFn.PROVIDER_STRIKES_TO_RETIRE} required) — the "
                "compaction call is not cache evidence, and one cold turn on a "
                "rebuilt prefix is not enough to convict a host"
            )
    before = [t for t in result.turns if t.turn < compact_at]
    if before and before[-1].retired and after[0].retired:
        return (
            f"compaction guard: retirements {before[-1].retired} survived the "
            f"turn-{compact_at} compaction; the prefix they were recorded against "
            "no longer exists, and `provider.ignore` has no other removal path"
        )
    return None


async def _run_arm(
    args: argparse.Namespace, arm: str, run: int, auth_db: Path, output: Any
) -> list[ArmResult]:
    """Run ``--concurrency`` conversations of one arm at once.

    Concurrency is the regime the pin is FOR. A single sequential conversation
    is a no-pressure sample: OpenRouter's load balancer has no reason to move
    it, so an unpinned baseline parks on one host by itself and both arms
    measure the same thing. The operator's real workload is many simultaneous
    sessions and subagents, which is what drives the re-routing (and the 429
    fallbacks behind it) that a cold prefix is paid for.

    Starts are STAGGERED so the lanes do not all present an identical cold
    prefix in the same instant, which would be a thundering herd rather than
    the steady overlapping load a working day produces.
    """
    lock = asyncio.Lock()

    async def lane(index: int) -> ArmResult:
        await asyncio.sleep(index * args.stagger)
        return await _run_lane(args, arm, run, index, auth_db, output, lock)

    return list(await asyncio.gather(*(lane(i) for i in range(args.concurrency))))


def _pool(results: list[ArmResult], arm: str) -> dict[str, Any]:
    """Pool an arm's lanes/runs into the numbers the evidence table reports."""
    arms = [r for r in results if r.name == arm and r.total_prompt]
    if not arms:
        return {}
    prompt = sum(r.total_prompt for r in arms)
    cached = sum(r.total_cached for r in arms)
    loss = sum(r.avoidable_loss * r.total_prompt for r in arms)
    cond = {k: {"n": 0, "hits": 0, "cached": 0, "prompt": 0} for k in ("same", "changed")}
    for r in arms:
        for key, bucket in r.conditional().items():
            for field_name, value in bucket.items():
                cond[key][field_name] += value
    honored = sum(r.pin_honored[0] for r in arms)
    pinned = sum(r.pin_honored[1] for r in arms)
    gaps = sorted(g for r in arms for g in r.gaps)
    mix: dict[str, int] = {}
    for r in arms:
        for host, count in r.mix.items():
            mix[host] = mix.get(host, 0) + count
    return {
        "conversations": len(arms),
        "turns": sum(len(r.turns) for r in arms),
        "cache_share": cached / prompt,
        "avoidable_loss": loss / prompt,
        "switches": sum(r.switches for r in arms),
        "transitions": sum(r.transitions for r in arms),
        "conditional": cond,
        "pin_honored": (honored, pinned),
        "median_gap_s": gaps[len(gaps) // 2] if gaps else 0.0,
        "usd": sum(r.usd_cost for r in arms),
        "mix": dict(sorted(mix.items(), key=lambda kv: -kv[1])),
    }


def _print_summary(results: list[ArmResult]) -> None:
    print("", file=sys.stderr)
    print(
        f"{'arm':<4} {'run':>3} {'lane':>4} {'turns':>5} {'prompt':>9} {'cached':>9} "
        f"{'cache':>6} {'avoid':>6} {'sw/tr':>7}  {'cost':>7}",
        file=sys.stderr,
    )
    for r in results:
        rate = f"{r.cache_rate:.1%}" if r.total_prompt else "n/a"
        print(
            f"{r.name:<4} {r.run:>3} {r.lane:>4} {len(r.turns):>5} {r.total_prompt:>9} "
            f"{r.total_cached:>9} {rate:>6} {r.avoidable_loss:>5.1%} "
            f"{r.switches:>3}/{r.transitions:<3}  ${r.usd_cost:>6.3f}",
            file=sys.stderr,
        )
        if r.error:
            print(f"     ERROR: {r.error}", file=sys.stderr)

    for arm in ("off", "on"):
        pooled = _pool(results, arm)
        if not pooled:
            continue
        same = pooled["conditional"]["same"]
        changed = pooled["conditional"]["changed"]
        print("", file=sys.stderr)
        print(
            f"POOLED {arm}: {pooled['conversations']} conversations, {pooled['turns']} turns, "
            f"cache={pooled['cache_share']:.1%}, avoidable={pooled['avoidable_loss']:.1%}, "
            f"cost=${pooled['usd']:.3f}, median gap={pooled['median_gap_s']:.1f}s",
            file=sys.stderr,
        )
        # Switch FREQUENCY, with the denominator, so a no-pressure window is
        # visible as such instead of reading as "the pin did not help".
        print(
            f"  host switches: {pooled['switches']}/{pooled['transitions']} transitions"
            + (
                f" ({pooled['switches'] / pooled['transitions']:.1%})"
                if pooled["transitions"]
                else ""
            )
            + ("   <- NO-PRESSURE SAMPLE: nothing to re-route" if not pooled["switches"] else ""),
            file=sys.stderr,
        )
        for label, bucket in (("host SAME", same), ("host CHANGED", changed)):
            if bucket["n"]:
                hit_rate = bucket["hits"] / bucket["n"]
                share = bucket["cached"] / max(bucket["prompt"], 1)
                print(
                    f"  {label:<13} n={bucket['n']:>3}  any-hit={hit_rate:>5.0%}"
                    f"  cache share={share:>5.1%}",
                    file=sys.stderr,
                )
        if pooled["pin_honored"][1]:
            honored, pinned = pooled["pin_honored"]
            print(
                f"  pin honored:   {honored}/{pinned} ({honored / pinned:.1%}) "
                f"— `order` is a preference, not a constraint",
                file=sys.stderr,
            )
        print(f"  mix: {pooled['mix']}", file=sys.stderr)


async def _run(args: argparse.Namespace, auth_db: Path, output: Any) -> int:
    results: list[ArmResult] = []
    spent = 0.0
    for run in range(args.runs):
        # Alternate arm order across runs: the first arm of a pair pays any
        # provider-side warming the second then inherits, so a fixed order
        # would hand one arm a systematic advantage.
        order = ["off", "on"] if run % 2 == 0 else ["on", "off"]
        if args.arm != "both":
            order = [args.arm]
        for arm in order:
            lanes = await _run_arm(args, arm, run, auth_db, output)
            results.extend(lanes)
            spent += sum(lane.usd_cost for lane in lanes)
            print(f"[budget] spent ${spent:.3f} of ${args.budget:.2f}", file=sys.stderr)
            failed = [lane for lane in lanes if lane.error]
            if failed:
                _print_summary(results)
                return 1
            if spent > args.budget:
                print("[budget] ceiling reached; stopping early", file=sys.stderr)
                _print_summary(results)
                return 1
    _print_summary(results)
    return int(any(r.error for r in results))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Print plan; no requests, no key.")
    mode.add_argument("--live", action="store_true", help="Spend the bounded budget.")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--arm", choices=("off", "on", "both"), default="both")
    parser.add_argument("--turns", type=int, default=30)
    parser.add_argument("--runs", type=int, default=1, help="Arm-order alternates across runs.")
    parser.add_argument(
        "--prefix-rows",
        type=int,
        default=950,
        help="Synthetic prefix rows; ~950 grows the context past 60k tokens "
        "(calibrated live: 400 rows measured 27,471 prompt tokens, ~68.7/row).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Conversations run at once per arm. The pin only pays where "
        "re-routing happens, and a single sequential conversation does not "
        "generate the route pressure that causes it.",
    )
    parser.add_argument(
        "--stagger",
        type=float,
        default=3.0,
        help="Seconds between concurrent lane starts (avoids a thundering herd).",
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=1.0,
        help="Seconds between turns WITHIN a conversation. Real sessions think "
        "for longer than this; observed gaps are reported so an eviction can "
        "be told apart from a re-route.",
    )
    parser.add_argument(
        "--compact-at",
        type=int,
        default=None,
        metavar="N",
        help='Send a real purpose="compaction" call at turn N and REPLACE the '
        "transcript with its summary, then keep going. A smoke check on the "
        "compaction boundary: the lane FAILS if a host serving the turns after "
        "it was retired without two cold turns of its own. It does NOT cover the "
        "purpose gate (its call is below the strike floor) \u2014 see the module "
        "docstring.",
    )
    parser.add_argument("--budget", type=float, default=5.0, help="Hard USD ceiling.")
    parser.add_argument("--seed", default=uuid.uuid4().hex[:8])
    parser.add_argument(
        "--source-auth-db",
        type=Path,
        default=Path.home() / ".local-operator" / "auth.db",
        help="Copied read-only into the isolated config dir; never written.",
    )
    parser.add_argument(
        "--output", type=Path, help="New JSONL file; existing evidence is never overwritten."
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "plan": (
                        f"{args.runs} run(s) x {args.arm} x {args.concurrency} "
                        f"concurrent conversations x {args.turns} turns"
                        + (
                            f", compaction injected at turn {args.compact_at}"
                            if args.compact_at is not None
                            else ""
                        )
                    ),
                    "model": args.model,
                    "seed": args.seed,
                    "budget_usd": args.budget,
                    **_source_identity(),
                },
                indent=2,
            )
        )
        return 0

    output = args.output.open("x") if args.output else sys.stdout
    saved = {key: os.environ.get(key) for key in ("HOME", "LOCAL_OPERATOR_CONFIG_DIR")}
    try:
        # BOTH are redirected, for the whole run: the config dir alone leaves
        # the catalogue cache resolving out of the operator's real home (see
        # the module docstring and AGENTS.md).
        with tempfile.TemporaryDirectory(prefix="lop-or-cache-") as home:
            config_dir = Path(home) / ".local-operator"
            config_dir.mkdir(parents=True)
            source = args.source_auth_db
            if not source.exists():
                raise SystemExit(f"No auth store at {source}")
            auth_db = config_dir / "auth.db"
            shutil.copyfile(source, auth_db)
            _guard_openrouter_credential(auth_db)
            os.environ["HOME"] = home
            os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(config_dir)
            print(json.dumps({**_source_identity(), "seed": args.seed}), file=output, flush=True)
            return asyncio.run(_run(args, auth_db, output))
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        output.flush()
        if output is not sys.stdout:
            output.close()


if __name__ == "__main__":
    raise SystemExit(main())
