"""Community read taint: a session that has read community posts asks every time.

WHY THIS EXISTS (RFC v2.4 §5.9, review rows S1 and R3-1). Community posts are
written by strangers. When the owner asks their chief of staff to read a
channel (``bridge op=community_read``), that text lands in the model's context,
and an injected instruction in it could try to make the chief post on the
owner's behalf. The RFC's rule: "A standing grant is not usable in any session
whose model context contains community content: once the chief has read
community posts, every later turn in that session falls back to Ask me each
time — including after compaction, because a compaction summary can carry
injected text forward — until a new session starts. The harness records this
as a flag in the session journal, so a resumed session keeps it and a forked
session inherits it."

WHAT IT IS. A monotonic per-session flag, the Team Bridge taint latch's shape
(``harness/bridge_egress.py`` ``BridgeTaintLatch``) narrowed to one source:

* SOURCE: a ``bridge`` tool result whose ``details`` carry
  :data:`COMMUNITY_CONTENT_DETAILS_KEY` (``community/agent_ops.py`` stamps
  every ``community_read`` result that returned at least one message), or the
  flag row itself;
* ROW: one :data:`COMMUNITY_TAINT_ROW_TYPE` custom entry in the append-only
  transcript, written once by the session's commit observer. Custom entries
  never render to the model. The row is what survives a prune that blanks the
  tool result's payload, and — because ``Transcript.fork`` excludes only
  conversation rows after a cut (custom rows are kept) and compaction appends a
  marker rather than rewriting rows — a fork inherits it and compaction never
  lifts it;
* BACKSTOP: a full scan of the journal from row 0 the first time a session
  object is asked, so a resume, a fork (it copies ``transcript.jsonl``) or a
  transcript whose row write failed still reads tainted.

FAIL-CLOSED. An error while deriving the flag means tainted: the only thing
the flag can do is take a shortcut away, so a wrong "tainted" costs one extra
question and a wrong "clean" costs an unapproved post.

WHAT IT GATES TODAY. The community server does not implement standing agent
post grants yet (``createAgentPostGrant`` is ``x-status: planned`` in
``docs/community-api.yaml``), so every ``community_post`` already asks its
owner. The flag is still recorded now, and :func:`standing_grant_usable` is the
one predicate a grant path must consult, so that the day grants ship a session
that already read community content cannot use one — the flag has to predate
the grant, or a session tainted before the upgrade would read clean.

DISTINCT FROM THE BRIDGE LATCH on purpose. A ``community_read`` result also
latches the Team Bridge egress latch (its shape is inbound text, which that
latch counts fail-closed), which is the right egress protection. This flag
answers a narrower question — "did this session read COMMUNITY content" — that
the approval card states to the owner in words, and a private-team message must
not make the card claim the chief read community posts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

#: The custom transcript row that records the flag. Version 1 carries no
#: fields a reader needs; its presence is the fact.
COMMUNITY_TAINT_ROW_TYPE = "community_read_taint"

#: The ``details`` key a ``community_read`` result carries when it returned
#: community text. Only ever written as ``True``.
COMMUNITY_CONTENT_DETAILS_KEY = "community_content"

#: The tool whose results can carry community content (the bridge tool's
#: ``community_*`` ops). Restated, not imported, so this leaf stays importable
#: from the session without pulling the team-bridge package in.
BRIDGE_TOOL_NAME = "bridge"


def details_carry_community(details: Any) -> bool:
    """Whether a tool result's ``details`` mark community content."""

    return isinstance(details, Mapping) and details.get(COMMUNITY_CONTENT_DETAILS_KEY) is True


def entry_is_community_source(entry_type: str, payload: Mapping[str, Any]) -> bool:
    """Whether one durable transcript row is a source, or the flag row itself."""

    if entry_type == "custom":
        return payload.get("custom_type") == COMMUNITY_TAINT_ROW_TYPE
    if entry_type != "message" or payload.get("role") != "tool":
        return False
    if payload.get("tool_name") != BRIDGE_TOOL_NAME:
        return False
    provider_payload = payload.get("provider_payload")
    details = provider_payload.get("details") if isinstance(provider_payload, Mapping) else None
    return details_carry_community(details)


class CommunityTaint:
    """The session's monotonic "read community content" flag.

    ``entries`` returns the transcript's durable rows as ``(type, payload)``
    pairs for the backstop scan. ``row_present`` tells the session's writer the
    durable row already exists, so it is written once.
    """

    def __init__(
        self,
        *,
        inherited: bool = False,
        entries: Callable[[], Iterable[tuple[str, Mapping[str, Any]]]] | None = None,
    ) -> None:
        #: A child spawned from a tainted session is tainted for life (the
        #: bridge latch's tighten-only inheritance).
        self.inherited = bool(inherited)
        self._tainted = self.inherited
        self._entries = entries
        self._scanned = False
        self.row_present = False

    def mark(self) -> None:
        """Set the flag in memory (the session's observer persists the row)."""

        self._tainted = True

    def observe_entries(self, rows: Iterable[tuple[str, Mapping[str, Any]]]) -> bool:
        """Set on any source among ``rows``; notes when the flag row is among them."""

        for entry_type, payload in rows:
            if not entry_is_community_source(entry_type, payload):
                continue
            self._tainted = True
            if entry_type == "custom":
                self.row_present = True
        return self._tainted

    @property
    def tainted(self) -> bool:
        """Whether this session has read community content. Never fails open."""

        if self._tainted:
            return True
        if self._scanned or self._entries is None:
            return self._tainted
        try:
            self.observe_entries(self._entries())
            self._scanned = True
        except Exception:  # noqa: BLE001 - an error deriving the flag means tainted
            logger.warning("community taint scan failed; treating as tainted", exc_info=True)
            return True
        return self._tainted

    def row_details(self) -> dict[str, Any]:
        return {"version": 1, "inherited": self.inherited}


def taint_of(tool_context: Any) -> CommunityTaint | None:
    return getattr(tool_context, "community_taint", None) if tool_context is not None else None


def session_tainted(tool_context: Any) -> bool:
    """Whether the session behind ``tool_context`` read community content.

    A host with no flag object at all (a bare test host, an SDK embedder) has
    no journal to prove it clean, so it reads tainted: fail closed.
    """

    taint = taint_of(tool_context)
    if taint is None:
        return True
    try:
        return taint.tainted
    except Exception:  # noqa: BLE001 - fail closed
        return True


def standing_grant_usable(tool_context: Any) -> bool:
    """The ONE predicate a standing agent-post grant path must consult (§5.9)."""

    return not session_tainted(tool_context)


__all__ = [
    "BRIDGE_TOOL_NAME",
    "COMMUNITY_CONTENT_DETAILS_KEY",
    "COMMUNITY_TAINT_ROW_TYPE",
    "CommunityTaint",
    "details_carry_community",
    "entry_is_community_source",
    "session_tainted",
    "standing_grant_usable",
    "taint_of",
]
