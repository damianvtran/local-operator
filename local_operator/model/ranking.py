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
    #: ``CatalogueEntry.listing_name``. Nothing in the RANKING reads it: scoring
    #: and ordering are on ``label`` and the selector exactly as before, so this
    #: is display payload travelling through, not a new sort input.
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

    @property
    def selector(self) -> str:
        """``provider/id`` — what ``/model`` takes and what the user types."""
        return f"{self.provider}/{self.model_id}"


#: `(tier, -score, version_key, row)` — the shape `rank_rows` sorts.
_RankEntry = tuple[tuple[int, int], int, tuple[float, float, str], "ModelRow"]


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
    """
    needle = query.strip().lower()
    if not needle:
        return sorted(
            rows,
            key=lambda row: (not row.connected, row.aggregated, row.provider, _version_key(row)),
        )
    exact: list[_RankEntry] = []
    fuzzy: list[_RankEntry] = []
    for row in rows:
        target = row.selector.lower()
        score = _score(target, needle)
        if score is None:
            continue
        entry = (
            (0 if row.connected else 1, 1 if row.aggregated else 0),
            -score,
            _version_key(row),
            row,
        )
        (exact if needle in target else fuzzy).append(entry)
    pool = exact or fuzzy
    pool.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in pool]


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
