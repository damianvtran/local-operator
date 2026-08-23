"""The analytics data model: what a call contributes and how it is broken down.

Two kinds of number live here and they must never be conflated:

1. **Authoritative provider counts.** ``input_tokens``, ``output_tokens``,
   ``cache_read_tokens``, ``cache_write_tokens`` and ``reasoning_tokens`` come
   straight off the wire (:class:`local_operator.harness.types.Usage`). Cache
   rates and the thinking/generation split are computed from these and are
   exact.

2. **Estimated component attribution.** The provider reports ONE input total,
   not a per-component breakdown, so the split across the system prompt, the
   operator's custom instructions, the tool inventory, tool schemas, the
   environment block, loaded knowledge, the conversation, and tool results is
   an *estimate*: each component's share of the authoritative context-token
   total, proportional to its character length. It is labelled as an estimate
   everywhere it surfaces. Char length rather than a re-tokenisation because
   the ratio is what matters and a proportional split of the real total is
   both cheaper and more honest than tokenising each fragment and hoping the
   sum matches what the provider billed.

The apportionment runs on a *snapshot* of character lengths taken on the event
loop (the transcript mutates after the call), but the arithmetic — and every
tokeniser touch — happens on the recorder's background thread.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Component taxonomy
# ---------------------------------------------------------------------------

#: The context components analytics attributes input tokens to, in the order a
#: report should present them. Stored as explicit columns in the rollup table,
#: so ADDING one is a schema migration (see ``store._SCHEMA``); the key strings
#: are therefore a stable contract, not a display concern.
#:
#: Why these boundaries and not finer ones: they are exactly the seams that are
#: RECOVERABLE from the wire ``ChatRequest`` without the session having to hand
#: the recorder a parallel structure. ``custom_instructions`` is the operator's
#: ``system_prompt.md`` PLUS any selected agent/team profile prompt — the
#: harness appends the profile inside the ``<user_instructions>`` span
#: (``prompts_api.build_system_blocks``), so on the wire they are one tagged
#: region and splitting them further would be a guess. ``system_prompt`` is the
#: packaged persona plus repo guidance (AGENTS.md/CLAUDE.md), which share block
#: 0's stable head with the persona and carry no wire delimiter of their own.
#: ``images`` is APPENDED, never inserted, so the CREATE TABLE / docs /
#: ``_COMPONENT_COLUMNS`` order stays a stable contract. Inserts and the
#: aggregate SELECT address named ``c_*`` columns, so a mid-list insert would
#: not silently remap an existing DB — appending is still right, just not
#: for the ordinal-remap reason an earlier comment claimed.
COMPONENT_KEYS: tuple[str, ...] = (
    "system_prompt",
    "custom_instructions",
    "tool_inventory",
    "tool_schemas",
    "environment",
    "knowledge",
    "conversation",
    "tool_results",
    "images",
)

#: Human labels for each component key, for the ``/analytics /usage`` screen.
COMPONENT_LABELS: dict[str, str] = {
    "system_prompt": "System prompt",
    "custom_instructions": "Custom instructions (agents/teams)",
    "tool_inventory": "Tool inventory",
    "tool_schemas": "Tool schemas",
    "environment": "Environment",
    "knowledge": "Knowledge / skills",
    "conversation": "Conversation",
    "tool_results": "Tool results",
    # "(est.)" because image tokens are a flat char proxy, not a measured
    # per-image count — the honesty label the rest of the split already carries.
    "images": "Images (est.)",
}

#: Marks the operator's custom-instructions section inside system block 0. Kept
#: in sync with ``prompts_api.build_system_blocks``: the header text is the
#: seam analytics uses to separate the operator's standing instructions (and
#: the agent/team profile appended to them) from the packaged persona above.
_CUSTOM_INSTRUCTIONS_HEADER = "## User's custom instructions"


def split_system_prompt(block0: str) -> tuple[int, int]:
    """Split system block 0 into ``(system_prompt_chars, custom_instr_chars)``.

    Block 0 is the byte-stable head: packaged persona, then optional repo
    guidance, then — when the operator has any — a ``## User's custom
    instructions`` section wrapping ``<user_instructions>…</user_instructions>``
    (see ``prompts_api.build_system_blocks``). The header is the only seam that
    survives onto the wire, so everything from it onward is attributed to
    ``custom_instructions`` (the operator prompt plus any agent/team profile),
    and everything before it to ``system_prompt`` (persona + repo guidance).

    Best-effort: a block with no custom section attributes the whole thing to
    ``system_prompt``, which is exactly right — there were no custom
    instructions to separate.
    """
    if not block0:
        return 0, 0
    idx = block0.find(_CUSTOM_INSTRUCTIONS_HEADER)
    if idx < 0:
        return len(block0), 0
    return idx, len(block0) - idx


#: Flat per-image char proxy (~4 chars/token * IMAGE_TOKEN_ESTIMATE). Kept as a
#: module constant, not an import, so the tokeniser module stays off this hot
#: snapshot path; it matches ``compaction.tokens.IMAGE_TOKEN_ESTIMATE`` scaled
#: back to chars so the two estimators tell the same story.
_IMAGE_CHAR_PROXY = 4 * 1200


def _content_chars(message: Any) -> tuple[int, int]:
    """A message's content length as ``(text_chars, image_chars)``.

    The two are returned SEPARATELY so image consumption can be attributed to
    its own ``images`` component instead of inflating whichever text bucket the
    message happens to sit in (conversation or tool_results). Images still cost
    real context tokens the provider billed, and leaving them at zero would push
    their share onto text; but folding them INTO the text total would over-weight
    that text bucket in the proportional apportionment (see
    ``snapshot_component_chars``). Splitting here keeps both honest.
    """
    text_chars = 0
    image_chars = 0
    content = getattr(message, "content", None) or []
    for part in content:
        text = getattr(part, "text", None)
        if isinstance(text, str):
            text_chars += len(text)
            continue
        if getattr(part, "type", "") == "image":
            image_chars += _IMAGE_CHAR_PROXY
    return text_chars, image_chars


@dataclass(frozen=True)
class CallSnapshot:
    """One provider call's contribution, captured on the event loop.

    Everything here is a scalar or a small dict of scalars: the snapshot is
    handed to a background thread, so it must not reference the live
    ``ChatRequest`` (whose messages the transcript mutates after the call) or
    any object with a lock. ``component_chars`` is the per-component character
    length measured from the request at call time; the recorder turns it into
    an estimated token split against the authoritative context total.
    """

    ts_ms: int
    session_id: str
    provider: str
    model_id: str
    # Authoritative provider counts (0 when the provider reported none).
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    context_tokens: int
    # Per-component character lengths (COMPONENT_KEYS ⊆ keys). Estimated token
    # attribution is derived from these against ``context_tokens``.
    component_chars: dict[str, int] = field(default_factory=dict)
    # Whether the model actually replied (a failed/aborted call still cost
    # input tokens and is worth recording, but is counted separately so a
    # provider-outage session does not look like normal spend).
    ok: bool = True
    # This call's dollar cost in MICRO-USD (USD × 1_000_000) and whether it is
    # known. Populated by the STORE on its background thread
    # (``price_snapshot``) unless ``priced`` is already True: pricing calls
    # ``resolve_model_info``, which can block for seconds on a cold memo, and
    # that must never run on the event loop the turn is unwinding on (review
    # C1). Micro-USD integers rather than floats so the store's ``SUM`` is exact.
    # A model with no published price prices to ``cost_known=False`` /
    # ``cost_micro=0``, rendered as an unknown share rather than a confident
    # ``$0.00`` — the "unknown ≠ free" distinction the status band's ``$—``
    # already makes (see ``tui/costs.turn_cost``).
    cost_micro: int = 0
    cost_known: bool = False
    # ``True`` when ``cost_micro``/``cost_known`` are already final and the store
    # must NOT re-price (a test recording a known cost without a price table, or
    # a future caller that priced off-thread itself). The normal recording path
    # leaves this False so the store prices from the model + token counts.
    priced: bool = False


def price_snapshot(snapshot: "CallSnapshot") -> tuple[int, bool]:
    """``(cost in micro-USD, price known)`` for a snapshot, never raising.

    Called by the store on its BACKGROUND thread (never the event loop), where
    the potentially-blocking ``resolve_model_info`` cold-memo lookup is free —
    the same reason the SQLite write lives there. Prices through
    ``cost_for_usage``, the one money computation the whole app shares, so the
    analytics dollar total cannot diverge from the status band. Returns
    ``(0, False)`` for a model with no published price (rendered ``$—``, not a
    misleading ``$0.00``) and for any failure — pricing is best-effort, and an
    unpriceable call is still worth recording token-wise.

    The lazy import keeps the model layer off this module's import graph (the
    analytics package must stay importable without the provider/pricing stack)
    and matches how the recorder already defers its own imports.
    """
    if snapshot.priced:
        return int(snapshot.cost_micro), bool(snapshot.cost_known)
    try:
        from local_operator.model.configure import cost_for_usage, resolve_model_info

        info = resolve_model_info(snapshot.provider, snapshot.model_id)
        if not (info.input_price or info.output_price):
            return 0, False
        # ``cost_for_usage`` duck-types its usage arg, so a plain object with the
        # token fields it reads is enough — no need to rebuild a ``Usage``.
        dollars = cost_for_usage(snapshot.provider, info, snapshot)
        return int(round(dollars * 1_000_000)), True
    except Exception:  # noqa: BLE001 — an unpriceable call is not an error
        return 0, False


def snapshot_component_chars(request: Any) -> dict[str, int]:
    """Measure each context component's character length from a ChatRequest.

    Runs on the event loop, so it does the least work that still measures the
    right thing: string-length reads for the prompt blocks and messages (no
    tokenising, no copying message bodies), plus one compact ``json.dumps`` per
    tool for the schema size (review A3 — this is real work that grows with
    tool count, not a bare length read, but it is the only honest proxy for
    what a provider serialises for a tool and it is bounded and one-shot).
    Benchmarked at 0.02 ms (small) to 0.34 ms (a 291k-token context, 60 tools,
    200 messages) — and it runs AFTER the response stream is consumed, never on
    the path the user is waiting on, so even the worst case is imperceptible.

    ``tool_schemas`` mirrors what a provider serialises beside the prompt: the
    tool name, description, and compact JSON of its parameters. It is counted
    separately from ``tool_inventory`` (the prose list in block 1) because the
    two are billed as different regions and a 126-tool MCP server's schema cost
    is exactly the kind of invisible regression this feature exists to expose.
    """
    blocks: list[str] = list(getattr(request, "system_blocks", []) or [])
    while len(blocks) < 4:
        blocks.append("")
    system_prompt_chars, custom_instr_chars = split_system_prompt(blocks[0])

    tool_schema_chars = 0
    for tool in getattr(request, "tools", []) or []:
        name = getattr(tool, "name", "") or ""
        description = getattr(tool, "description", "") or ""
        params = getattr(tool, "parameters", {}) or {}
        try:
            params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            params_json = ""
        tool_schema_chars += len(name) + len(description) + len(params_json)

    conversation_chars = 0
    tool_result_chars = 0
    image_chars = 0
    for message in getattr(request, "messages", []) or []:
        role = getattr(message, "role", "")
        text_chars, msg_image_chars = _content_chars(message)
        # CRITICAL: image chars are pulled OUT of the text buckets and summed
        # into ``images``, NEVER added alongside. Apportionment splits the fixed
        # ``context_tokens`` proportionally to total chars, so counting an
        # image's chars in both its text bucket AND ``images`` would double its
        # weight and over-attribute the shared total to conversation/tool_results.
        # Images from BOTH conversation and tool messages accumulate here.
        image_chars += msg_image_chars
        if role == "tool":
            tool_result_chars += text_chars
        else:
            conversation_chars += text_chars

    return {
        "system_prompt": system_prompt_chars,
        "custom_instructions": custom_instr_chars,
        "tool_inventory": len(blocks[1]),
        "tool_schemas": tool_schema_chars,
        "environment": len(blocks[2]),
        "knowledge": len(blocks[3]),
        "conversation": conversation_chars,
        "tool_results": tool_result_chars,
        "images": image_chars,
    }


def apportion_components(component_chars: Mapping[str, int], context_tokens: int) -> dict[str, int]:
    """Split ``context_tokens`` across components proportionally to char length.

    The provider bills one input total; this hands each component its share of
    that authoritative number, so the estimates SUM to the real context total
    (largest-remainder rounding, no drift). When the provider reported no
    context total (a failed call, or a provider that omits usage), every
    component is 0 — an honest "unknown" rather than a fabricated split.
    """
    keys = list(COMPONENT_KEYS)
    total_chars = sum(max(0, int(component_chars.get(k, 0))) for k in keys)
    if context_tokens <= 0 or total_chars <= 0:
        return {k: 0 for k in keys}

    # Largest-remainder apportionment: floor each share, then hand the leftover
    # tokens to the components with the largest fractional parts, so the split
    # is exact (sums to context_tokens) and stable rather than double-counting
    # or losing a few tokens to rounding on every call.
    exact: list[tuple[str, float]] = []
    for key in keys:
        share = max(0, int(component_chars.get(key, 0))) / total_chars * context_tokens
        exact.append((key, share))
    floored = {key: int(value) for key, value in exact}
    remainder = context_tokens - sum(floored.values())
    if remainder > 0:
        # Order by descending fractional part; ties broken by key for
        # determinism (two runs of the same snapshot must produce the same
        # split).
        by_frac = sorted(exact, key=lambda kv: (-(kv[1] - int(kv[1])), kv[0]))
        for key, _ in by_frac[:remainder]:
            floored[key] += 1
    return floored


@dataclass
class UsageAggregate:
    """A summed view over a set of rollup rows, for one report scope.

    Produced by :meth:`AnalyticsStore.aggregate`. ``by_provider`` and
    ``by_session`` map a name to its own :class:`UsageAggregate` (without the
    nested maps, one level deep) so the ``/analytics /usage`` screen can show
    totals, a per-provider table, and a per-session table from one query.
    """

    calls: int = 0
    ok_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    context_tokens: int = 0
    # Summed dollar cost in MICRO-USD, and how many of ``calls`` had a known
    # price. ``cost_known_calls < calls`` means the dollar figure is a LOWER
    # BOUND — some calls used models with no published price — which the report
    # marks (``$12.30+``) rather than presenting a partial sum as complete.
    cost_micro: int = 0
    cost_known_calls: int = 0
    components: dict[str, int] = field(default_factory=lambda: {k: 0 for k in COMPONENT_KEYS})
    by_provider: dict[str, "UsageAggregate"] = field(default_factory=dict)
    by_session: dict[str, "UsageAggregate"] = field(default_factory=dict)

    @property
    def generation_tokens(self) -> int:
        """Output tokens that were visible generation, not hidden thinking."""
        return max(0, self.output_tokens - self.reasoning_tokens)

    @property
    def cost_usd(self) -> float:
        """The summed cost in whole dollars (from the micro-USD accumulator)."""
        return self.cost_micro / 1_000_000.0

    @property
    def cost_is_partial(self) -> bool:
        """True when some priced-in calls had no published price.

        The dollar total is then a lower bound: it counts every call we could
        price and silently omits the ones we could not. A report must SAY so
        rather than imply the omitted calls were free.
        """
        return self.cost_known_calls < self.calls

    @property
    def cost_is_known(self) -> bool:
        """True when at least one call in scope had a known price.

        ``False`` means nothing here is priceable (e.g. a local Ollama-only
        run), which the report renders as ``$—`` rather than ``$0.00``.
        """
        return self.cost_known_calls > 0

    @property
    def total_tokens(self) -> int:
        """Every token the providers billed: full input context + output.

        ``context_tokens`` — not ``input_tokens`` — is the input half, because
        providers disagree on whether ``input_tokens`` already counts cache.
        Anthropic (this agent's primary provider) reports ``input_tokens``
        EXCLUDING cache reads/writes, so ``input_tokens + output_tokens`` would
        undercount a cached turn by the entire cache volume — usually the bulk
        of it (review A1). ``context_tokens`` is normalised upstream to the full
        input actually read (``input + cache_read + cache_write`` on Anthropic,
        ``== input`` on OpenAI where input already includes cache), so it is the
        one field that means "everything the provider read" on every provider.

        A call the provider gave no context total for still reports its output;
        its input simply reads as 0 here, which is honest — an unknown input is
        not a zero-token turn, but it is the only number we can stand behind.
        """
        return self.context_tokens + self.output_tokens

    @property
    def fresh_tokens(self) -> int:
        """Uncached input tokens, independent of how the provider reports input.

        Providers disagree on whether ``input_tokens`` already includes cache:
        Anthropic reports input EXCLUDING cache (so ``input == context - read
        - write``), while OpenAI-shaped usage folds cache into input (so
        ``context == input`` and a naive Fresh=input would show the FULL
        context). Subtracting cache from the normalised ``context_tokens`` is
        the one definition that is the uncached slice on every provider, and
        it is linear so computing it on the aggregate equals summing per-call.
        """
        return max(0, self.context_tokens - self.cache_read_tokens - self.cache_write_tokens)

    @property
    def cache_hit_rate(self) -> float | None:
        """Fraction of read context served from cache, or None when unknowable.

        Denominator is ``cache_read + context`` normalised: for providers whose
        ``input_tokens`` excludes cache (Anthropic) ``context_tokens`` already
        sums the three, and for providers whose input includes cache (OpenAI)
        ``context_tokens`` equals input, so ``cache_read / context_tokens`` is
        the honest hit rate in both worlds.
        """
        if self.context_tokens <= 0:
            return None
        return min(1.0, self.cache_read_tokens / self.context_tokens)


@dataclass(frozen=True)
class UsagePeriod:
    """One row of the daily/monthly rollup: a calendar bucket's summed spend.

    Produced by :meth:`AnalyticsStore.daily_series` / ``monthly_series`` /
    ``series_totals`` from the ``usage_daily`` / ``usage_monthly`` rollup
    tables (which the ledger's own ``record_batch`` maintains — see the store
    module docstring). ``period`` is the bucket key: a local ``YYYY-MM-DD`` for
    a daily row, ``YYYY-MM`` for a monthly row, or ``""`` for a totals row that
    sums a whole window. ``model`` is ``provider/model_id`` for a per-model row
    and ``""`` for the across-models aggregate a ``by_model=False`` query
    returns (summed in SQL, not Python, so the view stays cheap).

    The cost fields mirror :class:`UsageAggregate` exactly so the time-series
    view reuses the screen's existing money vocabulary: micro-USD integers for
    an exact ``SUM``, plus a count of the priced calls so a bucket that
    included an unpriceable model renders as a lower bound (``$X+`` / the ``≥``
    floor) rather than a confident understatement. ``cost_is_floor`` is the
    name the chart reads; it is the same condition as ``cost_is_partial`` but
    stated positively for the bar renderer.
    """

    period: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    context_tokens: int = 0
    cost_micro: int = 0
    cost_known_calls: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        """Every token the providers billed in this bucket: input + output.

        ``context_tokens`` (the normalised, cache-inclusive input) plus
        ``output_tokens`` — the same definition :meth:`UsageAggregate.total_tokens`
        uses, so a daily bar and the headline totals cannot tell two different
        stories about "how many tokens".
        """
        return self.context_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        """The bucket's summed cost in whole dollars (from micro-USD)."""
        return self.cost_micro / 1_000_000.0

    @property
    def cost_is_known(self) -> bool:
        """True when at least one call in this bucket had a published price."""
        return self.cost_known_calls > 0

    @property
    def cost_is_partial(self) -> bool:
        """True when some calls in this bucket could not be priced.

        The dollar figure is then a LOWER BOUND: it counts every priceable
        call and silently omits the rest, so the chart marks it (``$X+`` / the
        ``≥`` floor) rather than presenting a partial sum as complete — the
        same "unknown ≠ free" honesty the aggregate carries.
        """
        return self.cost_known_calls < self.calls

    #: Positive-sense alias for :attr:`cost_is_partial`, named for the bar
    #: renderer that draws a ``≥`` floor mark when a bucket's cost is a lower
    #: bound. One concept, two names, so the store and the view each read the
    #: one that states their intent.
    @property
    def cost_is_floor(self) -> bool:
        return self.cost_is_partial


def usage_component_chars_json(chars: Mapping[str, int]) -> str:
    """Serialise a component-char map compactly (for a per-call debug row)."""
    return json.dumps({k: int(chars.get(k, 0)) for k in COMPONENT_KEYS}, separators=(",", ":"))


_WS_RE = re.compile(r"\s+")


def short_session_label(session_id: str, name: str = "") -> str:
    """A compact label for a session in the per-session table.

    Prefers a human name when one exists, falling back to the 12-hex id. Kept
    here so the store and the TUI agree on how a session is named without the
    store importing a widget.
    """
    name = _WS_RE.sub(" ", (name or "").strip())
    if name:
        return name[:32]
    return (session_id or "unknown")[:12]
