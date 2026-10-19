"""Canonical ids for the JEV decision-model family.

WHY THIS MODULE EXISTS
======================

One billable decision model reaches a client under several spellings:

* ``jev-1.13`` — the canonical id, and the id agent-server prices the call
  from: it is what the decision response's top-level ``cost.model`` states and
  what ``GET /v1/prices`` publishes under ``rates.decision_models``;
* ``jev-1.13.0`` — the pinned TypeSafe vendor id, echoed in the response's
  top-level ``model``;
* ``typesafe/jev-1.13`` — the OpenRouter catalogue id;
* ``jev-1.13-YYYYMMDD`` and ``typesafe/jev-1.13-YYYYMMDD`` — dated vendor
  echoes of the SAME generation.

Those are four wire spellings of one model, not four models. A ledger that
keys spend by the raw spelling grows a second row for the same money (and a
failed attempt under another spelling reads as an unexplained unknown), which
is exactly the defect this module removes: every row carries the canonical id
and every surface groups by it.

THE ALIAS LIST MIRRORS THE SERVER, IT IS NOT A GUESS
====================================================

The aliases mirror the rate ids agent-server publishes in ``GET /v1/prices``
(``rates.decision_models``); the canonical id is the ``cost.model`` the server
says it priced the amount from. A new server generation therefore requires a
NEW entry here — never a broader pattern. An unknown id is returned unchanged
on purpose: silently rewriting a future model into this generation's identity
would put its money on the wrong price row.

This module is a LEAF (imports the standard library only) so the accounting
layer, the session ledger and the TUI can all use it without a cycle.
"""

from __future__ import annotations

import re

#: The id agent-server prices a decision call from (``cost.model`` /
#: ``GET /v1/prices`` ``rates.decision_models``). Every ledger row for this
#: model carries this spelling.
CANONICAL_DECISIONS_MODEL = "jev-1.13"

#: The pinned vendor id TypeSafe echoes in a response's top-level ``model``.
PINNED_DECISIONS_MODEL = "jev-1.13.0"

#: The OpenRouter catalogue id for the same model.
OPENROUTER_DECISIONS_MODEL = "typesafe/jev-1.13"

#: Undated aliases of :data:`CANONICAL_DECISIONS_MODEL`. Dated echoes are
#: matched by :data:`_DATED_ECHO_RE` below, with the same family restriction.
DECISION_MODEL_ALIASES = frozenset(
    {
        CANONICAL_DECISIONS_MODEL,
        PINNED_DECISIONS_MODEL,
        OPENROUTER_DECISIONS_MODEL,
    }
)

#: ``jev-1.13-YYYYMMDD`` / ``typesafe/jev-1.13-YYYYMMDD``. The generation is
#: literal in the pattern: a new generation is a new entry, not a wider regex.
_DATED_ECHO_RE = re.compile(r"^(?:typesafe/)?jev-1\.13-\d{8}$")


def canonical_model_id(model: str | None) -> str:
    """The canonical id for a known spelling; an unknown id is returned unchanged.

    ``None`` and ``""`` stay as they are: "the caller recorded no model" is a
    different fact from a model this module does not recognise, and neither is
    a reason to invent the canonical id.
    """
    if not model:
        return model or ""
    spelling = model.strip()
    if spelling in DECISION_MODEL_ALIASES or _DATED_ECHO_RE.fullmatch(spelling):
        return CANONICAL_DECISIONS_MODEL
    # Deliberately the ORIGINAL string, not ``spelling``: an unknown id that
    # arrived with incidental whitespace is still not ours to rewrite.
    return model


__all__ = [
    "CANONICAL_DECISIONS_MODEL",
    "DECISION_MODEL_ALIASES",
    "OPENROUTER_DECISIONS_MODEL",
    "PINNED_DECISIONS_MODEL",
    "canonical_model_id",
]
