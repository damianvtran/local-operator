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
#: This is the match-normaliser's whole rule (see :func:`_match_key`): every run
#: of punctuation, whitespace and symbols collapses to ONE space, so a query and
#: a row that spell the same words with different separators compare equal.
_MATCH_SEPARATOR_PATTERN = re.compile(r"[^0-9a-z]+")


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
    #: spells it `x-ai/grok-4.7`. Ordering is unaffected — this joins the
    #: substring/subsequence test only, never a sort rung.
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


#: `(tier, preferred_router, -score, version_key, row)` — the shape `rank_rows` sorts.
#:
#: ``preferred_router`` sits ABOVE ``-score``, and the reason is the SCORED branch
#: alone — it is the only one carrying a score to outrank. On the ``auto`` query
#: ``openrouter/auto`` scores 7 to ``radient/auto``'s 6, so a rung below the score
#: would leave that branch's order unchanged. The empty-query branch has no score
#: at all and this rung is simply the first non-tier term there; see
#: :func:`_preferred_router_rank`.
_RankEntry = tuple[tuple[int, int], int, int, tuple[float, float, str], "ModelRow"]


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

    SUBSTRING matches win outright when there are any; the subsequence matcher is
    the fallback. Ordering it the other way round is technically a superset and
    practically much worse: ``opus`` is a subsequence of
    ``anthropic/claude-sonnet-4`` (o and p from "anthropic", u from "claude", s
    from "sonnet"), so a user typing the name of one model got a list led by a
    different one. Keeping the fallback is what still resolves ``anthopus`` and
    ``sonnet4``, which are the typo and elision cases fuzzy matching exists for.

    Two tiers come before the score. Connected rows outrank unconnected ones,
    because a model you can use right now beats one that needs a login and
    interleaving them scatters the usable rows through a list of locked ones. Then
    DIRECT providers outrank aggregators: `openrouter/anthropic/claude-opus-5` and
    `anthropic/claude-opus-5` are the same model, and after logging in to Anthropic
    the direct route is the one the user meant.

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
    needle = _match_key(query)
    if not needle:
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
    exact: list[_RankEntry] = []
    fuzzy: list[_RankEntry] = []
    for row in rows:
        # SCORED against every string a user can SEE, not the selector alone.
        # The selector is `openrouter/x-ai/grok-4.7`; the row also carries
        # `SpaceXAI: Grok 4.7` as ``listing_name``, and the human name is what
        # someone actually types — so scoring the selector only is how
        # `grok 4.7` returned an empty list while the row sat in the catalogue
        # (D2). `label` joins them because ``label`` is the picker's own
        # resolved display form; a reseller's ``label`` degrades to the
        # selector, which is already covered.
        targets = tuple(
            _match_key(candidate)
            for candidate in (row.selector, row.listing_name, row.label)
            if candidate
        )
        scores = [(score, target) for target in targets if (score := _score(target, needle))]
        if not scores:
            continue
        # Densest match wins; ties fall through to the tier/version rungs below.
        # ``target.startswith`` is folded into the score for a PREFIX match, so
        # this is the same "best-looking" choice the single-target version made.
        score = max(scores)[0]
        entry = (
            (0 if row.connected else 1, 1 if row.aggregated else 0),
            _preferred_router_rank(row),
            -score,
            _version_key(row),
            row,
        )
        (exact if any(needle in target for target in targets) else fuzzy).append(entry)
    pool = exact or fuzzy
    pool.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return [item[4] for item in pool]


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

    Version components are deliberately kept as separate words: ``4.7`` reads
    ``4 7``, so a query of ``grok 47`` would NOT match — that is the right call,
    since ``47`` is a different token from ``4`` then ``7`` and a fuzzy
    subsequence pass already covers the typo case.
    """
    return _MATCH_SEPARATOR_PATTERN.sub(" ", text.lower()).strip()


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
