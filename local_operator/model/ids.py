"""Model-id spelling rules shared by every lookup that matches ids across sources.

WHY A MODULE OF ITS OWN
-----------------------
Three modules compare ids that different parties spelled: ``configure`` matches
what the user typed against a listing, ``discovery`` decides whether a requested
id is missing from a cached document (the ``want_id`` refetch trigger), and
``prices`` looks a provider's id up in a third-party price catalogue. They
cannot share the helpers through any one of themselves: ``configure`` is on the
CLI import path and must not import ``discovery`` (httpx) at module level, and
``discovery`` must not import ``configure`` back. Pure string functions with no
imports are the only thing all three can depend on without a cycle.

Nothing here touches the network, a file, or the registry.
"""

from __future__ import annotations

import re

#: A trailing Anthropic-style release stamp: `claude-opus-4-5-20251101`.
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")

#: A version separated by a dash between two digits: the `4-5` of `claude-opus-4-5`.
#: Anchored on digits BOTH sides so `qwen2.5-coder-1.5b` and `gpt-4o-mini` are left
#: alone — only a dash that is standing in for a decimal point is rewritten.
_DOTTED_VERSION_RE = re.compile(r"(?<=\d)-(?=\d)")


def normalised_id(model_id: str) -> str:
    """A model id in the one spelling both sides of a match can agree on.

    Discovery NORMALISES ids on ingest — ``_row_from_gemini_entry`` strips
    Google's ``models/`` resource prefix so the rest of the system sees a bare id
    — while the user types whatever the provider's own documentation shows, which
    for Gemini is ``models/gemini-2.5-pro``. An exact-match-only lookup therefore
    missed the spelling Google itself publishes and handed that session the 128k
    unknown default. Case is folded for the same reason: an id is a wire
    identifier, not prose, and no provider ships two models differing only in case.
    """
    trimmed = model_id.strip()
    prefix = "models/"
    if trimmed.startswith(prefix):
        trimmed = trimmed[len(prefix) :]
    return trimmed.casefold()


def id_spellings(model_id: str) -> list[str]:
    """``model_id`` as a second-hand catalogue might spell it, most literal first.

    Providers and catalogues disagree about punctuation for the SAME model, and
    the disagreement is systematic rather than per-model: Anthropic ships
    `claude-opus-4-5-20251101` while OpenRouter lists `anthropic/claude-opus-4.5`
    and models.dev keys `openai/gpt-5.4` where the user may type `gpt-5-4`.
    Trying only the literal id would leave every dated Claude snapshot unpriced,
    which is precisely the population a price lookup exists for.

    Both rewrites are conservative — a date stamp is eight digits at the end, and a
    dash becomes a dot only between two digits — and every candidate must still be
    found in the catalogue before it is believed. A miss costs one dict lookup.

    Order is most-literal-first, and the DOTTED-WITH-DATE form comes before either
    date-stripped one on purpose: catalogues publish dated snapshots under their
    own dated ids alongside the undated alias (`anthropic/claude-3.5-sonnet-20240620`
    as well as `anthropic/claude-3.5-sonnet`), so stripping the date first would
    answer a question about one snapshot with the alias's price. Harmless while
    snapshots of a family share a rate, wrong the day one does not.
    """
    stripped = _DATE_SUFFIX_RE.sub("", model_id)
    candidates = [model_id]
    for candidate in (
        _DOTTED_VERSION_RE.sub(".", model_id),
        stripped,
        _DOTTED_VERSION_RE.sub(".", stripped),
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates
