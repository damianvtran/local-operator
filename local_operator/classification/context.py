"""The state builder and its truncation ladder (§5).

WHAT GOES OUT, AND WHAT NEVER DOES
==================================

The transcript is never sent. Not the history, not the compaction summary, not
a tool result. The state is exactly three things: the newest user message, an
OPTIONAL short representative context string the caller already redacted, and
the candidate roster as ``name: description`` lines. Every one of those is
bounded here, and the bounding is the reason this module exists separately from
the client — the ceiling is not a transport limit (the model takes 64k), it is a
decision about cost: a recommendation that costs more than the resources it
saves is a net loss.

THE LADDER, IN ORDER (§5)
=========================

Applied one rung at a time, cumulatively, stopping at the first rung whose
serialized state fits ``values.classification.maxStateChars``:

1. drop ``context`` entirely;
2. trim each candidate line to 120 chars, then to 60;
3. drop the lowest-priority candidate kind (``mcp`` → ``guide`` → ``skill``);
4. truncate ``request`` to the remaining budget, appending a truncation marker.

The order is the contract and it is also the right order: the context line is
the least load-bearing part of the state (it is a convenience for the model, and
its absence only makes the answer less grounded), a candidate description is
what the model actually judges against, and the user's own request is what the
whole call is about — so the request is truncated last and only when nothing
else is left to give up. A message longer than the whole budget is classified on
its head, which is acceptable because every answer here is advisory and the head
of a request is where the intent is.

WHAT "SERIALIZED" MEANS HERE
============================

``json.dumps(state)`` with its DEFAULT separators, which is slightly longer than
the compact form httpx puts on the wire. That direction is deliberate: the
budget is a cap, so measuring with the more pessimistic serializer can only
make us send less, never more.

A kind with no candidates is omitted from the mapping rather than sent as an
empty list: the ladder drops kinds entirely at rung 3, and keeping the key with
``[]`` would pay tokens to say nothing.

WHO SUPPLIES ``context`` TODAY: NOBODY
======================================

Recorded here because it is the one §5 field no caller fills. The contract
describes ``context`` as "in practice the newest compaction summary line or the
last assistant message's first line", and §7's wiring passes ``None``
(`session_factory.py`'s request builder argues the case: the transcript is
deliberately not read, and the newest user message is already the state's
request field). The builder keeps it because it is the contract's field and
because it is the only lever that grounds a very short request — "continue",
"yes", a bare path — with something to judge against, so the caller-side half is
a wiring decision rather than something to delete here. The cost while it is
unused is exactly zero: an absent context is simply absent from the state, and
the ladder's rung 1 is then a no-op. What the package owes in the meantime is
proof that the field WORKS when a caller does supply one, which
``tests/unit/classification/test_context.py`` provides rung by rung.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar, cast, runtime_checkable

from local_operator.classification.cascade import classification_section

ResourceKind = Literal["skill", "guide", "mcp"]


@runtime_checkable
class CandidateLike(Protocol):
    """The four attributes this module reads off a candidate row.

    STRUCTURAL, and that is the point: the wiring owns its own row type
    (``session_factory``'s ``_ClassificationCandidate``) precisely so the turn path
    never imports this package, and both sides are read by ATTRIBUTE. Annotating
    these helpers with the concrete dataclass made the wiring's rows a type error at
    a seam that already works at runtime — the fix is to state the interface the
    helpers actually use, not to make either side import the other.

    Read-only ``property`` declarations rather than plain attributes: the rows on
    both sides are frozen dataclasses, and a writable attribute in the protocol would
    refuse them while describing something neither side does.
    """

    @property
    def kind(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def resource_url(self) -> str: ...


#: ``values.classification.maxStateChars`` — the whole state's hard cap.
DEFAULT_MAX_STATE_CHARS = 6000

#: ``values.classification.maxCandidates`` — per kind, applied before any rung.
DEFAULT_MAX_CANDIDATES = 12

#: Appended to a truncated request. The leading space is part of the marker: the
#: head is cut mid-word far more often than at a word boundary, and a run-in
#: marker reads as part of the sentence ("…staging envir…[truncated]").
TRUNCATION_MARKER = " …[truncated]"

#: Rung 2's two line limits, in the order they are tried.
CANDIDATE_LINE_LIMITS: tuple[int, ...] = (120, 60)

#: Rung 3's drop order — lowest priority first.
KIND_DROP_ORDER: tuple[ResourceKind, ...] = ("mcp", "guide", "skill")

#: How each kind is keyed on the wire. The plural/`mcp_servers` spelling is the
#: contract's (§5), matched rather than "improved": the model's rubric comes
#: from the harness, but the field names are the vendor's.
KIND_TO_STATE_KEY: dict[ResourceKind, str] = {
    "skill": "skills",
    "guide": "guides",
    "mcp": "mcp_servers",
}

#: The inverse, for filtering a cached roster by dropped kind.
KIND_FROM_STATE_KEY: dict[str, ResourceKind] = {
    state_key: kind for kind, state_key in KIND_TO_STATE_KEY.items()
}


@dataclass(frozen=True)
class Candidate:
    """One resource the model may pick.

    ``description`` must be HARNESS-OWNED text (§6): the skill/guide description
    as discovered from the local filesystem, or an MCP server's own name plus
    harness-written capability hints. Config-authored or remote-authored prose
    in here re-opens the prompt-injection surface that ``mcp/resources.py``
    deliberately excludes — the option text is the one part of the request the
    model treats as a rubric rather than as data.
    """

    kind: ResourceKind
    name: str
    description: str
    resource_url: str


#: The row type ``shortlist`` preserves: it decides WHICH rows travel and hands back
#: the CALLER'S OWN rows, so the caller keeps its concrete type (the wiring's own
#: ``_ClassificationCandidate``, which must not be replaced by this module's).
CandidateT = TypeVar("CandidateT", bound=CandidateLike)


def setting_int(settings: Mapping[str, Any] | None, key: str, default: int) -> int:
    """A positive integer from the ``classification`` section, else ``default``.

    One reader for every integer key in §8, so ``maxStateChars``,
    ``maxCandidates``, ``maxRecommendations`` and ``timeoutMs`` cannot end up
    parsed by four slightly different rules. A bool is refused explicitly —
    ``True`` is an ``int`` in Python, and a hand-edited ``timeoutMs: true``
    would otherwise configure a 1 ms deadline.

    Reads the section through
    :func:`~local_operator.classification.cascade.classification_section` rather
    than re-deriving it: two readers of the same mapping is how a key ends up
    honoured on one path and ignored on another.
    """
    raw = classification_section(settings).get(key)
    if isinstance(raw, bool):
        return default
    if isinstance(raw, int):
        return raw if raw > 0 else default
    if isinstance(raw, str):
        # A hand-edited YAML quoting a number is a typo we can read, not a
        # reason to fall back to a default the operator did not ask for.
        text = raw.strip()
        if text.isdigit() and int(text) > 0:
            return int(text)
    return default


def max_state_chars(settings: Mapping[str, Any] | None) -> int:
    """``values.classification.maxStateChars``, defensively read.

    A negative or non-integer value reads as the default rather than as "no
    budget": a hand-edited config must not be able to wedge this layer into a
    state that carries nothing, and a cap of ``0`` would do exactly that.
    """
    return setting_int(settings, "maxStateChars", DEFAULT_MAX_STATE_CHARS)


def max_candidates(settings: Mapping[str, Any] | None) -> int:
    """``values.classification.maxCandidates`` — per kind."""
    return setting_int(settings, "maxCandidates", DEFAULT_MAX_CANDIDATES)


def select_candidates(
    candidates: Sequence[Candidate], limit: int | None = None
) -> tuple[Candidate, ...]:
    """Cap the roster per kind, preserving the caller's order.

    Order is preserved rather than sorted: the caller's order carries meaning
    (discovery rank), and it is also the option order the model sees, so a
    re-ordered roster is a different question and legitimately gets its own
    answer. The cache key reflects that — see :func:`candidates_digest`.
    """
    per_kind: dict[ResourceKind, int] = {}
    kept: list[Candidate] = []
    cap = limit if limit is not None else DEFAULT_MAX_CANDIDATES
    for candidate in candidates:
        seen = per_kind.get(candidate.kind, 0)
        if seen >= cap:
            continue
        per_kind[candidate.kind] = seen + 1
        kept.append(candidate)
    return tuple(kept)


def _tokens(text: str) -> set[str]:
    """The word set a score is computed over.

    Same shape as ``skills/index.py``'s tokenizer (``[a-z0-9]+`` on lowercased
    text) on purpose: the roster here and the embedder there are judging the same
    strings, and two tokenizers would make the two surfaces disagree about what a
    word is for no benefit.
    """
    return set(_WORD_RE.findall(text.lower()))


#: Local, and module level so the pattern is compiled once: this runs per message.
_WORD_RE = re.compile(r"[a-z0-9]+")


def shortlist(
    candidates: Sequence[CandidateT],
    query: str,
    limit: int | None = None,
) -> tuple[CandidateT, ...]:
    """The candidates to spend budget on, chosen by LOCAL relevance to ``query``.

    WHY A CHOICE HAS TO BE MADE AT ALL
    ---------------------------------
    ``maxCandidates`` is a cost bound (every candidate travels TWICE: one line in
    the state and one option description in its kind's question), and the roster
    is the whole discovered catalogue. An operator with hundreds of installed
    skills therefore cannot send all of them, and the first ``limit`` of them in
    discovery order is an arbitrary permanent shortlist: the same dozen favourites
    would be offered on every message for the life of the session while the rest
    of the catalogue was unreachable — precisely when scaling to hundreds is the
    thing that matters.

    WHY THE SCORING IS LOCAL, AND LEXICAL
    -------------------------------------
    Three constraints pushed it here rather than into the embedder:

    * **No extra round trip.** The layer's latency budget is one network call
      (§5a rule 1); a second embedding request per message is not affordable, and
      the embedder's own selection is a different question (it picks what the
      PROMPT should carry, this picks what the classifier should judge).
    * **Deterministic and inspectable.** A Jaccard overlap over lowercase word
      sets, plus a literal-name bonus, answers "is this resource's text about what
      the user just asked?" with numbers a reviewer can recompute by hand.
    * **Cheap at catalogue scale.** Word sets are built once per (roster, query);
      hundreds of candidates cost well under a millisecond of set arithmetic,
      against a 50 ms turn budget.

    The order of the RETURNED rows is the caller's (discovery) order, not the
    score order — the state's line order and the question's option order are part
    of the request the vendor sees, and this function's job is to decide WHICH
    rows travel, not to reshuffle them. Ties keep discovery order for the same
    reason: two equally relevant candidates must not swap places between two
    identical messages, or the session cache key would churn for nothing.

    A roster that already fits per kind is returned BY IDENTITY, so a small
    catalogue behaves exactly as it did before this existed — the wiring's warm
    path stays allocation-free (asserted in the wiring's own tests).
    """
    cap = limit if limit is not None else DEFAULT_MAX_CANDIDATES
    if cap <= 0:
        if isinstance(candidates, tuple):
            return cast(tuple[CandidateT, ...], candidates)
        return tuple(candidates)
    per_kind: dict[str, list[int]] = {}
    for index, candidate in enumerate(candidates):
        per_kind.setdefault(candidate.kind, []).append(index)
    if all(len(rows) <= cap for rows in per_kind.values()):
        # IDENTITY, not a copy: the wiring asserts that a warm message carries the
        # cached roster object, and a fresh tuple per message would churn that for
        # nothing. A list is normalised because the annotation promises a tuple.
        return candidates if isinstance(candidates, tuple) else tuple(candidates)

    query_tokens = _tokens(query)
    lowered_query = query.lower()
    keep: list[int] = []
    for rows in per_kind.values():
        if len(rows) <= cap:
            keep.extend(rows)
            continue
        scored: list[tuple[float, int]] = []
        for position, roster_index in enumerate(rows):
            item = candidates[roster_index]
            line_tokens = _tokens(candidate_line(item))
            union = query_tokens | line_tokens
            overlap = len(query_tokens & line_tokens) / len(union) if union else 0.0
            # A resource whose NAME the user typed is the strongest signal this
            # scorer has, and Jaccard alone can bury it under a long shared
            # description ("deploy the service" against a deploy skill's blurb).
            named = 1.0 if item.name and item.name.lower() in lowered_query else 0.0
            scored.append((overlap + named, position))
        # Highest score first, discovery order as the tie-break, then the survivors
        # go back in the caller's order — this decides WHICH rows travel, not how
        # they are laid out for the vendor.
        scored.sort(key=lambda item: (-item[0], item[1]))
        keep.extend(rows[position] for _, position in scored[:cap])
    kept = set(keep)
    return tuple(item for index, item in enumerate(candidates) if index in kept)


def candidate_line(candidate: CandidateLike, limit: int | None = None) -> str:
    """``"name: description"``, trimmed to ``limit`` chars when one is given."""
    line = f"{candidate.name}: {candidate.description}" if candidate.description else candidate.name
    if limit is not None and limit > 0 and len(line) > limit:
        # The ellipsis is inside the limit, not after it: "trimmed to 60 chars"
        # has to mean the LINE is 60 chars, or the cap is a suggestion. Marking
        # the trim matters — a description cut mid-word without a marker reads
        # as a shorter, complete description.
        return line[: limit - 1] + "…"
    return line


class RosterCache:
    """The roster's candidate lines, built once per roster digest and line limit.

    WHY THIS EXISTS (the latency budget)
    ------------------------------------

    This layer runs once per user message, on the critical path, before the
    turn's first token — so every millisecond it spends building strings is a
    millisecond the user waits. The ``candidates`` half of the state changes only
    when the discovered skill/guide/MCP roster changes, while the per-message
    half is one user message. Re-serializing a dozen candidate lines per turn
    buys nothing: the lines for a given ``(roster, line limit)`` pair are
    identical every time.

    So they are memoized here and the per-message work becomes a join. The key is
    the roster digest (which covers names, URLs and descriptions) plus the line
    limit, so a changed roster or a different rung of the ladder misses rather
    than serving stale text — and a miss is exactly the work that would have
    happened without the cache. The mapping returned is immutable by convention
    (tuples), so a caller cannot mutate a cached entry and poison the next
    message.

    Bounded on purpose: a session's roster changes a handful of times, so a
    handful of entries is the whole working set, and an unbounded cache would
    hold every roster the machine has ever seen for no benefit.
    """

    def __init__(self, capacity: int = 4) -> None:
        self._capacity = max(1, capacity)
        self._entries: OrderedDict[tuple[str, int | None], dict[str, tuple[str, ...]]] = (
            OrderedDict()
        )

    def lines(
        self,
        digest: str,
        candidates: Sequence[Candidate],
        line_limit: int | None,
    ) -> dict[str, tuple[str, ...]]:
        """The candidate lines per state key, memoized by digest and limit."""
        key = (digest, line_limit)
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            return cached
        roster: dict[str, list[str]] = {}
        for candidate in candidates:
            roster.setdefault(KIND_TO_STATE_KEY[candidate.kind], []).append(
                candidate_line(candidate, line_limit)
            )
        frozen = {state_key: tuple(lines) for state_key, lines in roster.items()}
        self._entries[key] = frozen
        self._entries.move_to_end(key)
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)
        return frozen


def candidates_digest(candidates: Sequence[Candidate]) -> str:
    """A stable digest of the roster as sent — order-sensitive, description included.

    The description is in the digest because the questions are built from it
    (§6: option descriptions ARE name plus harness-owned description), so a
    roster whose descriptions changed is a different question set and must not
    hit the previous answer in the session cache.
    """
    payload = "\n".join(
        f"{candidate.kind}:{candidate.name}:{candidate.resource_url}:{candidate.description}"
        for candidate in candidates
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def serialized_size(state: Mapping[str, Any]) -> int:
    """Character length of the state as it would be serialized, conservatively."""
    return len(json.dumps(state, ensure_ascii=False))


def build_state(
    *,
    user_message: str,
    context: str | None,
    candidates: Sequence[Candidate],
    settings: Mapping[str, Any] | None = None,
    max_chars: int | None = None,
    candidate_limit: int | None = None,
    roster_cache: RosterCache | None = None,
) -> dict[str, Any]:
    """The bounded ``state`` object for a decision request.

    ``max_chars`` / ``candidate_limit`` override the settings-derived values, so
    the ladder can be exercised rung by rung in a test without fabricating a
    config file. Both are optional and default to the §8 keys.

    ``roster_cache`` is the per-service :class:`RosterCache`. With one, the
    candidate lines are built once per roster and reused for every later message,
    which is what keeps the per-message cost down to the user message plus a
    join; without one the lines are built here on every call.
    """
    cap = max_chars if max_chars is not None else max_state_chars(settings)
    roster = select_candidates(
        candidates, candidate_limit if candidate_limit is not None else max_candidates(settings)
    )
    digest = candidates_digest(roster)
    for context_text, line_limit, dropped in _ladder(context, roster):
        # The cache holds the WHOLE roster at this line limit; dropping a kind is
        # then a dict filter, not a rebuild. The line limit is part of the cache
        # key, so rung 1 and rung 2 never share an entry, and the filter runs
        # identically whether the lines came from the cache or were just built.
        full = (
            _lines_for(roster, line_limit)
            if roster_cache is None
            else roster_cache.lines(digest, roster, line_limit)
        )
        lines = {
            state_key: full[state_key]
            for state_key in full
            if KIND_FROM_STATE_KEY[state_key] not in dropped
        }
        state = _state(user_message, context_text, lines)
        if serialized_size(state) <= cap:
            return state
    # Every rung overflowed, so the roster and the context are already gone and
    # the only thing left to shrink is the request itself.
    return _truncate_request(user_message, cap)


def _ladder(
    context: str | None,
    roster: Sequence[Candidate],
) -> list[tuple[str | None, int | None, frozenset[ResourceKind]]]:
    """The cumulative rungs, in the contract's order (§5 steps 1-3).

    Built eagerly (it is at most seven small tuples) so the ladder is one
    readable list a reviewer can diff against the contract instead of a nest of
    loops whose order is implicit in the control flow.
    """
    rungs: list[tuple[str | None, int | None, frozenset[ResourceKind]]] = [
        # Rung 0 is not a rung in the contract — it is "nothing had to give".
        (context or None, None, frozenset()),
    ]
    if context:
        rungs.append((None, None, frozenset()))
    for limit in CANDIDATE_LINE_LIMITS:
        rungs.append((None, limit, frozenset()))
    dropped: set[ResourceKind] = set()
    for kind in KIND_DROP_ORDER:
        dropped.add(kind)
        rungs.append((None, CANDIDATE_LINE_LIMITS[-1], frozenset(dropped)))
    return rungs


def _lines_for(candidates: Sequence[Candidate], line_limit: int | None) -> dict[str, list[str]]:
    """The state's candidate lines, keyed by wire name, in the roster's order."""
    lines: dict[str, list[str]] = {}
    for candidate in candidates:
        lines.setdefault(KIND_TO_STATE_KEY[candidate.kind], []).append(
            candidate_line(candidate, line_limit)
        )
    return lines


def _state(
    user_message: str,
    context: str | None,
    lines: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """One candidate state at one rung."""
    state: dict[str, Any] = {"request": user_message}
    if context:
        state["context"] = context
    if lines:
        state["candidates"] = {state_key: list(values) for state_key, values in lines.items()}
    return state


def _truncate_request(user_message: str, cap: int) -> dict[str, Any]:
    """Rung 4: keep the head of the request, mark the cut, never exceed the cap.

    The marker is counted BEFORE the head is sized — a truncation that pushes the
    state over the cap would defeat the point of the ladder — and the size is
    re-measured after every cut, because JSON escaping means one input character
    can cost up to six (`"` → ``\\u0022`` is beyond even ``\\"``). The loop
    therefore subtracts the measured overshoot rather than trusting arithmetic
    on character counts, and it terminates because the head strictly shrinks.
    """
    state: dict[str, Any] = {"request": TRUNCATION_MARKER}
    keep = max(0, cap - serialized_size(state))
    while True:
        state["request"] = user_message[:keep] + TRUNCATION_MARKER
        overflow = serialized_size(state) - cap
        if overflow <= 0 or keep == 0:
            # ``keep == 0`` is the pathological-cap case (a cap smaller than the
            # JSON scaffolding itself): the marker alone is the smallest valid
            # request this module can build, and it is still a valid one.
            return state
        keep = max(0, keep - max(1, overflow))
