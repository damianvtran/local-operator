"""Model ranking — the order ``/model`` offers a catalogue in, for every surface.

Lifted out of ``tui/widgets/model_picker.py`` verbatim so that a NON-TUI caller
can rank the same catalogue the picker does. The phone's model sheet is the one
that forced it: its ``GET /api/models`` handed the client 962 rows in registry
order, so ~445 aggregated Radient rows rendered before the first direct
provider — roughly 45 phone screens of scrolling to reach ``anthropic/``. It
could not simply import the widget: doing so costs ~0.48 s and pulls textual and
rich onto the mobile daemon's import path, for a daemon that renders no
terminal at all.

So the split is by CONCERN, not by convenience. What lives here is pure ordering
over plain data — stdlib only, no rich, no textual — and what stays in the widget
is presentation (``format_window``, ``format_price_pair``), which is written in
``rich.Text`` and means nothing off a terminal. The widget re-exports these names
so every existing importer keeps working unchanged.

The docstrings below are not decoration: each one records a measured regression
that a plausible-looking rewrite reintroduces (the ``kimi-k2`` lookbehind, the
substring-before-subsequence ordering, first-number-not-largest). Read them
before changing a pattern or a sort rung.
"""

from __future__ import annotations

import dataclasses
import functools
import re

#: Version-shaped numbers inside a model id: `4`, `4.1`, `2.5`, the `2` in `k2`,
#: the `3` in `qwen3:8b`.
#:
#: The lookbehind excludes digits and dots ONLY. Excluding word characters as well
#: looked tidier and silently broke every id that glues the version to a letter —
#: `kimi-k2` matched nothing at all, so its version came from the `0905` serial and
#: `kimi-k2-0905` outranked `kimi-k3`. The dot is what stops a decimal's own
#: fraction being counted a second time as a standalone number.
_VERSION_PATTERN = re.compile(r"(?<![\d.])(\d+(?:[.]\d+)?)")

#: A dash followed by a SHORT run of digits is a minor version (`opus-4-1` = 4.1),
#: rewritten to a decimal before the version scan. Capped at two digits on purpose:
#: `sonnet-4-20250514` is a dated snapshot, and reading it as 4.20250514 would put
#: it above every real version in the catalogue.
_MINOR_VERSION_PATTERN = re.compile(r"(?<![\d.])(\d+)-(\d{1,2})(?![\d])")

#: Everything that is not a lowercase letter or a digit, as a word separator.
#: Every run of punctuation, whitespace and symbols collapses to ONE space; a
#: letter next to a digit is separated too, so compact model queries converge.
_MATCH_SEPARATOR_PATTERN = re.compile(r"[^0-9a-z]+")
_MATCH_ALPHANUMERIC_BOUNDARY = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")


@dataclasses.dataclass(frozen=True)
class ModelRow:
    """One offerable model.

    ``connected`` is the provider's credential state, not the model's. The app
    filters unreachable rows out before they get here — a picker is a list of
    choices — so a False row is one of the two the filter deliberately keeps: the
    session's CURRENT model when its provider stopped being usable, or every row
    at once when the credential store could not be read. Both need to look
    different from a model that will run, which is what this flag drives (dim id,
    `login required` where the numbers go, and last place in the ranking).
    Choosing one starts a login instead of a switch — see
    :meth:`ModelPicker.highlighted`.
    """

    provider: str
    model_id: str
    #: The model's display name, already through ``model/naming.py``'s honesty
    #: rule upstream — so it is either a name that identifies this model alone or
    #: the selector itself, never a name two models answer to. Empty means the
    #: caller had none; the row then shows its selector and nothing more.
    label: str = ""
    #: The source listing's own human name, carried past ``label``'s honesty
    #: rule for a consumer that disambiguates the route some other way — see
    #: ``CatalogueEntry.listing_name``. It IS a match input (see
    #: :func:`_match_key`): the human name is the string a user types, so
    #: `SpaceXAI: Grok 4.7` has to resolve `grok 4.7` even though the selector
    #: spells it `x-ai/grok-4.7`. It enters the match QUALITY rung, not the sort
    #: keys: a hit in the SELECTOR still outranks a hit only in this name, so a
    #: row is never promoted past one whose id the user actually typed (R1-3).
    #: Membership is unaffected too — every row that matched before this field
    #: was consulted still matches, because the pool is no longer re-decided from
    #: the target set (R1-1).
    #:
    #: Keyword-only so it cannot disturb the POSITIONAL argument order this row
    #: is widely constructed with (the TUI's tests build it positionally); a new
    #: field in the middle of that sequence silently re-binds every caller's
    #: arguments.
    listing_name: str = dataclasses.field(default="", kw_only=True)
    context_window: int = 0
    default_context_window: int | None = dataclasses.field(default=None, kw_only=True)
    max_context_window: int | None = dataclasses.field(default=None, kw_only=True)
    input_price: float = 0.0
    output_price: float = 0.0
    connected: bool = True
    #: True when this row comes from a RESELLER rather than the model's own
    #: provider. Set by the caller, which is the only layer that knows the
    #: registry; the picker only needs it as a sort rung.
    aggregated: bool = False
    #: This row is a META-ROUTE — a router whose price is the price of whichever
    #: model it dispatches to. Set by the caller from the listing that said so
    #: (``CatalogueEntry.routed``), never inferred here: the renderer cannot
    #: tell a router's unknown price from any other unknown one, and deciding it
    #: from the id in this layer would be the second, divergent statement of the
    #: rule that :func:`format_price_pair`'s docstring exists to warn against.
    routed: bool = False
    #: The time-of-use schedule NAME this row's prices are quoted at, or ``None``
    #: when they do not vary by time of day (``CatalogueEntry.time_of_use``).
    #: Keyword-only for ``listing_name``'s reason: rows are built POSITIONALLY all
    #: over the TUI's tests, and a new field in the middle of that sequence would
    #: silently re-bind their arguments.
    time_of_use: str | None = dataclasses.field(default=None, kw_only=True)

    @property
    def selector(self) -> str:
        """``provider/id`` — what ``/model`` takes and what the user types."""
        return f"{self.provider}/{self.model_id}"


#: `(tier, preferred_router, rung, -score, version_key, row)` — the shape
#: `rank_rows` sorts.
#:
#: ``preferred_router`` sits ABOVE ``-score``, and the reason is the SCORED branch
#: alone — it is the only one carrying a score to outrank. On the ``auto`` query
#: ``openrouter/auto`` scores 7 to ``radient/auto``'s 6, so a rung below the score
#: would leave that branch's order unchanged. The empty-query branch has no score
#: at all and this rung is simply the first non-tier term there; see
#: :func:`_preferred_router_rank`.
#:
#: ``rung`` is the OTHER non-tier term and it is this branch's own: how the row
#: matched (0 selector substring, 1 selector subsequence, 2 name substring, 3 name
#: subsequence), used only to order rows INSIDE one pool. It sits below
#: ``preferred_router`` rather than above because the product preference is
#: QUERY-INDEPENDENT — it holds for any query the Radient route matches — while
#: ``rung`` is meaningful only within a query.
_RankEntry = tuple[tuple[int, int], int, int, int, tuple[float, float, str], "ModelRow"]


#: The ROUTE the preference elevates: ``(provider, model_id)``, the Radient
#: auto-router appended to the aggregator's own namespace. Deliberately NOT the id
#: alone: any provider can serve a model called ``auto``, and
#: :func:`local_operator.model.discovery.is_meta_route_id` is the predicate over
#: the ID set this route is a member of — see :func:`_preferred_router_rank` for
#: why that predicate is not reused here.
_PREFERRED_ROUTER = ("radient", "auto")


def _preferred_router_rank(row: ModelRow) -> int:
    """0 for the Radient auto-router, 1 for everything else.

    A DELIBERATE PRODUCT PREFERENCE, not a heuristic, and it says so plainly
    because the code cannot invent a rationale it does not have. Measured on the
    operator's catalogue with the query ``auto`` typed, BEFORE this rung:

        openrouter/openrouter/auto
        openrouter/openrouter/auto-beta
        radient/auto                       <-- the row the operator wanted first
        radient/openrouter/auto-beta

    The two OpenRouter rows led because the ``-score`` rung placed them there:
    ``openrouter/auto`` scores 7 to ``radient/auto``'s 6 (the OpenRouter id carries
    the query twice as a contiguous run), and the version keys collide at
    ``(-0.0, -0.0, …)`` because neither id holds a number. So the ordering was a
    property of the ids, not a decision.

    QUERY-INDEPENDENT, and that is the intent rather than an accident of where it
    was inserted: the rung lifts the row for EVERY query the route matches, not
    only the literal ``auto``. A partial query shows it plainly: for ``aut``
    ``openrouter/openrouter/auto`` scores 5 to ``radient/auto``'s 4, yet the
    operator's own reading of the preference — "Radient Auto comes first at the
    top" — is that it lead whenever it is OFFERED. Since the dogfood catalogue
    always offers it, ``auto`` is the only query the rule is experienced with, but
    scoping the rung to ``needle == "auto"`` would make every partial spelling
    (``aut``, ``au``) rank it behind the OpenRouter rows it leads for ``auto``,
    which is the incoherent half-measure.

    It leads for every user who CAN RUN it, not every user unconditionally: the
    connected tier outranks this rung, so a user signed in only to OpenRouter
    still leads with the OpenRouter row and never sees this one at all — the
    picker drops an unconnected row before ranking
    (``providers.catalogue.picker_rows``). Presenting a row the account cannot run
    above one it can was never the ask; the ask was precedence among the rows on
    offer.

    Only the Radient route in :data:`_PREFERRED_ROUTER` is elevated, and NARROWLY:
    OpenRouter is an aggregator the operator also uses, so demoting OpenRouter
    rows wholesale would be a much larger behavioural change deserving its own
    design review. A caller that builds rows from a different catalogue still gets
    the same precedence, which is the point of ranking owning it rather than the
    picker.

    WHY THE ROUTE LITERAL RATHER THAN ``is_meta_route_id``. The predicate owns
    "which ids name a ROUTER, per provider" (``discovery._META_ROUTE_IDS``, which
    also names OpenRouter's ``openrouter/auto``), and the reviewer's suggestion to
    reuse it was tested rather than assumed: importing it costs ~15 ms of marginal
    import (``discovery`` is otherwise resident — ``providers.registry`` already
    pulls the same ``httpx``/``pydantic`` stack ``model.registry`` does — but under
    the module's own budget no import is free), and with the provider gate it does
    NOT do this job: ``radient`` is an aggregator, so the predicate is ALSO true of
    ``radient/openrouter/auto``, which this PR deliberately leaves BELOW the
    OpenRouter rows. Reusing it would silently lift that row too — a second
    behavioural change this finding did not ask for. So the id is stated here as a
    ROUTE, and the drift the finding feared (a new router id needing two updates)
    is named in the comment above rather than traded for a wider rung.

    The gate on the PROVIDER matters for the same reason ``is_meta_route_id`` is
    provider-scoped: a local ``ollama/auto`` is a model a user can simply have,
    and it must not be lifted by a bare id test.
    """
    return 0 if (row.provider, row.model_id) == _PREFERRED_ROUTER else 1


def rank_rows(rows: list[ModelRow], query: str) -> list[ModelRow]:
    """Rows matching ``query``, best first, matched on the DISPLAYED string.

    Matching what the user can see (``provider/id``) rather than the id alone is
    what lets bare names, provider prefixes and scoped queries all flow through
    one matcher.

    THE POOL IS NEVER SHRUNK BY ADDING A MATCH TARGET. Every row that matches
    ``query`` under ANY target stays in the answer; only its order moves. An
    earlier revision bucketed the rows (`pool = exact or fuzzy`) and re-decided
    membership from "is the needle a substring of ANY target now", so widening the
    target set to include ``listing_name`` EVICTED rows the old selector-only test
    had matched — measurably on the live listing: query `banana` dropped five
    direct-provider rows, `older` dropped eight of nine (R1-1). The membership
    decision and the ordering decision are therefore separated: the row enters on
    "matched anything at all", and how WELL it matched is a RUNG inside one pool.

    The rungs, in order: connected rows, then direct providers over aggregators,
    then the PREFERRED-ROUTER preference, then match QUALITY, then score, then
    version. Quality is ranked above density
    so a literal substring always leads a mere subsequence, and ABOVE a name-only
    hit: a needle found in the SELECTOR (what the user typed as an id) outranks
    one found only in the human ``listing_name``, which is what keeps
    ``xai/grok-4.7`` ahead of ``xai/grok-4.7-fast`` for the query `grok 4.7`
    (R1-3/B4) — the shorter sibling scores denser on the NAME, but its SELECTOR
    is not the substring the user typed. Connected/aggregated still lead, so a
    runnable row of either kind beats a locked one.

    The subsequence matcher is a real tier, not a fallback pool, so it never
    displaces a stronger match: ``opus`` is a subsequence of
    ``anthropic/claude-sonnet-4`` (o and p from "anthropic", u from "claude", s
    from "sonnet"), and scoring it *below* every substring rung is what stops a
    different model leading the list. It is still what resolves ``anthopus`` and
    ``sonnet4``, the typo and elision cases fuzzy matching exists for.

    A PREFERRED-ROUTER rung sits ABOVE the score and the version key, in BOTH
    branches, so ``/model`` with no query and ``/model auto`` agree. It is the one
    non-lexical thing here and it is a product preference rather than a heuristic:
    it lifts ``radient/auto`` above every other row — QUERY-INDEPENDENTLY, for any
    query the route matches, which is the intent and not merely the ``auto`` case
    — see :func:`_preferred_router_rank` for the measured order that motivated it,
    the partial-query case that shows the width, and how narrowly it is scoped. It
    has to outrank the SCORE rather than merely ``_version_key``, because on the
    ``auto`` query the OpenRouter rows score higher (7 to 6) and a rung below the
    score would change nothing.

    DECISION-ONLY PROVIDERS ARE DROPPED, in both branches, before any scoring.
    This is the surface ``/model`` offers the catalogue THROUGH, and ranking is
    the last gate before a row becomes a choice: a decision model (TypeSafe's Jev)
    rejects ``chat/completions`` on every host, so selecting one would open a
    session that cannot answer a turn — the failure the user has no way to read
    as "that model was never offerable". The controller's catalogue already
    excludes them (``_chat_providers``); this keeps the promise for a caller that
    builds rows from somewhere else, which is exactly how the phone's sheet and
    the desktop picker came to rank the same list two ways.
    """
    # Imported at CALL time, like the sibling helpers in ``providers.failover``:
    # ``providers.registry`` is a heavier module and this one is stdlib-only by
    # design (it is imported by the mobile daemon, which renders no terminal).
    from local_operator.providers.registry import is_decision_only

    rows = [row for row in rows if not is_decision_only(row.provider)]
    # TWO EMPTY-ISH CASES, and they are not the same. `query.strip() == ""` is the
    # user having typed nothing, which lists the catalogue in its natural order.
    # A query that is non-empty but NORMALISES to "" (`.`, `!`, `...`, `-`) is the
    # user having typed something that carries no words, and falling into the
    # empty branch there would replace the list with the WHOLE catalogue on one
    # keystroke (`'.'` measured 257 -> 574 rows, `'...'` 0 -> 1816; R1-2). It
    # matches nothing instead, which is what the pre-normaliser code did.
    needle = _match_key(query)
    if not needle:
        if query.strip():
            return []
        return sorted(
            rows,
            key=lambda row: (
                not row.connected,
                row.aggregated,
                _preferred_router_rank(row),
                row.provider,
                _version_key(row),
            ),
        )
    ranked_strong: list[_RankEntry] = []
    ranked_fuzzy: list[_RankEntry] = []
    for row in rows:
        # Only treat a number as a model-version constraint when the immediately
        # preceding query word belongs to this row's model ID. This keeps an
        # OpenRouter/provider number from filtering out an Opus row, while still
        # constraining `openrouter opus 5.5` on the `opus` term and selector.
        query_version = _query_version(query, row.model_id)
        if query_version is not None and not _matches_query_version(row, query_version):
            continue
        # SCORED against every string a user can SEE, not the selector alone.
        # The selector is `openrouter/x-ai/grok-4.7`; the row also carries
        # `SpaceXAI: Grok 4.7` as ``listing_name``, and the human name is what
        # someone actually types — so scoring the selector only is how
        # `grok 4.7` returned an empty list while the row sat in the catalogue
        # (D2). `label` joins them because ``label`` is the picker's own
        # resolved display form; a reseller's ``label`` degrades to the
        # selector, which is already covered.
        selector = _match_key(row.selector)
        selector_score = _score(selector, needle)
        tiers = (0 if row.connected else 1, 1 if row.aggregated else 0)
        if selector_score is not None:
            # A match on the SELECTOR — what the user typed as an id — keeps the
            # established two-pool rule: substring hits (`strong`) replace
            # subsequence ones (`fuzzy`), which is the `/model opus` behaviour the
            # settings view pins. ``rung`` 0/1 only orders the two within one
            # pool and never removes a row.
            rung = 0 if needle in selector else 1
            entry = (
                tiers,
                _preferred_router_rank(row),
                rung,
                -selector_score,
                _version_key(row),
                row,
            )
            (ranked_strong if needle in selector else ranked_fuzzy).append(entry)
            continue
        # MATCHED ONLY a non-selector target. These rows are APPENDED to the fuzzy
        # pool, never allowed to replace it: an earlier revision widened the
        # "exact" pool with name matches and, because `exact` REPLACED `fuzzy`, a
        # row that matched the selector only as a subsequence was dropped from the
        # answer entirely — measured on the live listing: query `banana` lost five
        # direct-provider rows, `older` eight of nine (R1-1). The pool is now
        # chosen from SELECTOR matches alone (unchanged from the base), and a
        # name-only match can only ADD to it. `rung` 2/3 keeps every name hit
        # below every selector hit, so a sibling whose NAME scores denser cannot
        # lead the model the user actually named (R1-3/B4).
        name_score: int | None = None
        name_substring = False
        for target in (_match_key(row.listing_name), _match_key(row.label)):
            if not target:
                continue
            score = _score(target, needle)
            if score is None:
                continue
            if needle in target:
                name_substring = True
            if name_score is None or score > name_score:
                name_score = score
        if name_score is None:
            continue
        ranked_fuzzy.append(
            (
                tiers,
                _preferred_router_rank(row),
                2 if name_substring else 3,
                -name_score,
                _version_key(row),
                row,
            )
        )
    pool = ranked_strong or ranked_fuzzy
    pool.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    return [item[5] for item in pool]


@functools.lru_cache(maxsize=8192)
def _match_key(text: str) -> str:
    """A string's spelling as the matcher sees it: lowercase, space-separated words.

    WHY this exists rather than a bare ``.lower()``. A model's human name is
    published by the listing with spaces and a colon (``SpaceXAI: Grok 4.7``)
    while its id glues the same words with hyphens and a slash
    (``x-ai/grok-4.7``), so a plain substring test can only ever match one of
    them: a user typing `grok 4.7` matched NEITHER, because the query's space
    appears in neither the name's colon-adjacent spelling nor the id at all. The
    rule is one normalisation applied to both sides, never a special case per
    spelling: every run of non-alphanumerics becomes a single space, so
    ``x-ai/grok-4.7`` and ``grok 4.7`` both read ``x ai grok 4 7`` /
    ``grok 4 7`` and the substring test succeeds on the shared words.

    MEMOISED because the widget calls :func:`rank_rows` on every keystroke and
    normalises three strings per row each time; the function is pure, so an
    ``lru_cache`` removes the repeated work without changing an answer (R1-7
    measured the uncached path at ~9-10x the base cost, 0.9 -> 8.7 ms on a
    1,816-row catalogue). The cache is per-process and unbounded in content but
    bounded in entries: a real catalogue holds a few thousand distinct strings.

    Version components are deliberately kept as separate words: ``4.7`` reads
    ``4 7``. The exact-substring test therefore does NOT match a query of
    ``grok 47`` — ``47`` is a different token from ``4`` then ``7`` — but the
    SUBSEQUENCE pass still does, because it finds a ``4`` then a ``7`` in order
    in ``grok 4 7`` (R1-6: measured, `grok 47` DOES resolve `x-ai/grok-4.7`).
    That is the intended behaviour, and a future change that "restores" a
    non-match by fusing ``47`` back into one token would regress it.
    """
    separated = _MATCH_SEPARATOR_PATTERN.sub(" ", text.lower()).strip()
    return _MATCH_ALPHANUMERIC_BOUNDARY.sub(" ", separated)


def _query_version(query: str, model_id: str) -> float | None:
    """Read a numeric constraint only when its preceding family term is in this ID.

    Decimal punctuation is meaningful here, even though `_match_key` turns it
    into separate words: `opus 5.5`, `opus-5.5`, and `opus5.5` name minor 5.5,
    while `grok 47` is not a version spelling and remains a fuzzy subsequence
    query. A provider or unrelated metadata number has no model-family token
    immediately before it and therefore cannot hide otherwise matching rows.
    """
    # Aggregator IDs can include a leading owner namespace such as
    # `anthropic/claude-opus-5.5`; that namespace is not the model-family term
    # whose number this filter constrains.
    model_part = model_id.partition("/")[2] or model_id
    model_words = [
        word for word in _match_key(model_part).split() if not word.isdigit()
    ]
    decimals = list(re.finditer(r"(?<!\d)(\d+)[.-](\d{1,2})(?!\d)", query))
    for match in reversed(decimals):
        preceding = _match_key(query[: match.start()]).split()
        if preceding and _term_names_model(preceding[-1], model_words):
            return float(f"{match.group(1)}.{match.group(2)}")

    key = _match_key(query)
    numeric = list(re.finditer(r"(?<!\d)(\d+)(?!\d)", key))
    if not numeric or len(numeric[-1].group(1)) != 1:
        return None
    preceding = key[: numeric[-1].start()].split()
    if not preceding or not _term_names_model(preceding[-1], model_words):
        return None
    return float(numeric[-1].group(1))


def _term_names_model(term: str, model_words: list[str]) -> bool:
    """Accept a typed family prefix/subsequence found in the row's model ID."""
    return any(word.startswith(term) or term in word for word in model_words)


def _matches_query_version(row: ModelRow, query_version: float) -> bool:
    """Match the row's family version, not incidental digits in its label.

    ``_match_key`` deliberately removes punctuation, so the selector's original
    model ID is the authoritative source for distinguishing a decimal version
    from two unrelated numbers. The established version helper already handles
    short hyphen minors and dated suffixes consistently with picker ordering.
    """
    version = -_version_key(row)[0]
    if version <= 0:
        return False
    if query_version.is_integer():
        return int(version) == int(query_version)
    return version == query_version


def _version_key(row: ModelRow) -> tuple[float, float, str]:
    """Sort key placing the NEWEST-looking model first within its tier.

    Alphabetical order on model ids is actively wrong for this catalogue:
    `claude-opus-4-1` sorts before `claude-opus-5` and `gpt-4o` before `gpt-5.4`,
    so a plain sort leads every family with its oldest member — the one a user is
    least likely to be reaching for.

    The version is the FIRST number in the id, not the largest. Taking the largest
    looked equivalent and was not: `kimi-k2-0905` carries 2 and 905, so it scored
    905 and led a list in which `kimi-k3` came ninth. Every id in this catalogue
    puts the family version first and its serials, dates and parameter counts
    after, so position is the reliable signal and magnitude is not.

    Three rungs:

    1. **version**, descending — the first number, with a SHORT run after a dash
       folded in as a minor (`claude-opus-4-1` reads 4.1 and beats
       `claude-opus-4`). The run has to be short, or `claude-sonnet-4-20250514`
       would read as 4.20250514 and outrank every real version in the list.
    2. **remaining numbers**, descending — the dates and serials rung 1 ignores.
       Two snapshots of one model (`-20250514` vs `-20260101`, `-0905` vs bare)
       differ only here, and the later one is what a user wants.
    3. **id**, ascending, so ids with no numbers at all stay in a stable,
       predictable order rather than an arbitrary one.
    """
    normalized = _MINOR_VERSION_PATTERN.sub(r"\1.\2", row.model_id)
    numbers = [float(match) for match in _VERSION_PATTERN.findall(normalized)]
    version = numbers[0] if numbers else 0.0
    return (-version, -max(numbers[1:], default=0.0), row.model_id)


def _score(target: str, needle: str) -> int | None:
    """Subsequence score, or None when ``needle`` is not a subsequence.

    Density is what the score measures: consecutive matched characters are worth
    double, so ``opus`` scores ``claude-opus-5`` far above a model that merely
    happens to contain o, p, u and s in order. An exact substring therefore always
    wins, without needing a separate substring pass.
    """
    if not needle:
        return 0
    score = 0
    previous = -2
    index = 0
    for char in needle:
        found = target.find(char, index)
        if found < 0:
            return None
        score += 2 if found == previous + 1 else 1
        previous = found
        index = found + 1
    # A match that starts at the beginning is a prefix, which is the strongest
    # signal a short query can carry.
    if target.startswith(needle):
        score += len(needle)
    return score
