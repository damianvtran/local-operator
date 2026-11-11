"""Id suffixes of the derived COPIES of a message the request path builds.

The token-estimate memo (``compaction.tokens._ESTIMATE_CACHE``) is keyed on
``Message.id``, and each copy below carries different content from its source,
so each gets a deterministic suffixed id of its own. That is also why
invalidating a source message must evict these ids: the request is built from
the copy, so a stale copy estimate survives the source being mutated and the
client keeps refusing for content that is no longer there.

One place, so the writers (``harness.replay_bound``, ``session.attachment_hints``)
and the eviction in ``compaction.tokens`` cannot drift. Lives in ``harness``
because ``compaction.tokens`` must not import ``session``.
"""

from __future__ import annotations

import itertools

#: ``harness.replay_bound``: a copy with the middle of a long tool result elided.
ELIDED_ID_SUFFIX = "~rb"

#: ``session.attachment_hints``: a copy carrying the attachment hint note.
HINT_ID_SUFFIX = "~ah"

#: ``harness.tool_names``: a copy whose garbled tool names were rewritten for
#: the wire only (the durable transcript still holds the original name).
NAME_ID_SUFFIX = "~nm"

#: Every derived-copy suffix. Add a new derived-copy kind here, and the
#: invalidation covers it.
DERIVED_ID_SUFFIXES: tuple[str, ...] = (ELIDED_ID_SUFFIX, HINT_ID_SUFFIX, NAME_ID_SUFFIX)


def derived_copy_ids(base_id: str) -> frozenset[str]:
    """Every id a derived copy of ``base_id`` could carry.

    The writers append their suffix to whatever id their input already
    carries, so a message transformed twice is keyed with the composed id
    (``<id>~rb~nm`` for a bounded, then named, call). Eviction must cover any
    ordering or combination of the suffixes, not just the single ones, so
    this enumerates every subset in every order: 15 derived ids for the
    three current suffixes, and a new suffix adds O(n) ones without touching the
    eviction. Excludes ``base_id`` itself, which the caller pops separately.
    """
    out: set[str] = set()
    for length in range(1, len(DERIVED_ID_SUFFIXES) + 1):
        for combination in itertools.permutations(DERIVED_ID_SUFFIXES, length):
            out.add(f"{base_id}{''.join(combination)}")
    return frozenset(out)
