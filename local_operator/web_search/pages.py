"""Session-scoped custody of the page context a DeepSeek search leaves behind.

DeepSeek's native search hands back `web_search_result` items whose
`encrypted_content` is the retrieved page, recoverable ONLY by replaying the
assistant's blocks verbatim to DeepSeek (Anthropic's `encrypted_content`
contract). That makes "read the pages you already found" possible with no
network fetch and no new search -- but it needs somewhere to keep the blocks
between the search and the read, which is what this store is.

Three properties are deliberate, and each one is a correctness requirement
rather than a convenience:

* **Verbatim or useless.** The blocks are stored exactly as DeepSeek returned
  them. Rewriting, filtering or "cleaning" an item invalidates the content, so
  nothing here touches them.
* **Session-scoped, never global.** A read must only ever see pages that THIS
  session's own search retrieved. Pages are private to the session that fetched
  them (and to the operator's machine): a shared cache would let one session
  answer from another's pages, which is both a correctness bug and a
  data-boundary violation.
* **Bounded and expiring.** The blocks are opaque, sizable, and of unknown
  server-side lifetime, so the store keeps a small number of recent contexts per
  process and drops them after a TTL. An expired context degrades to a fetch;
  a growing dict of page blobs would not degrade, it would leak.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

__all__ = ["PAGE_CONTEXTS", "PageContext", "PageContextStore"]

#: How long a captured page context stays readable. The server side may keep the
#: decryption material longer, but a read is only useful soon after the search
#: that produced it, and a stale context answers from pages the session has
#: already moved past.
DEFAULT_TTL_SECONDS = 1_800.0

#: Contexts retained per process. Each one is roughly the ten result blocks of
#: one search, so this is a memory bound rather than a policy knob.
DEFAULT_MAX_CONTEXTS = 8


@dataclass(frozen=True, slots=True)
class PageContext:
    """One search's retrievable page set, plus what produced it."""

    context_id: str
    provider: str
    query: str
    created_at: float
    #: The assistant content blocks, verbatim. Never mutated after construction.
    blocks: list[dict[str, Any]] = field(default_factory=list)
    #: Normalized sources (url/title/relevance) for naming pages in a request.
    sources: list[dict[str, Any]] = field(default_factory=list)
    #: True when the capture came from a search that ran the evidence pass, which
    #: is informational only: a read works either way.
    enriched: bool = False

    @property
    def urls(self) -> list[str]:
        return [str(source.get("url") or "") for source in self.sources if source.get("url")]

    def ages_out(self, ttl_seconds: float, now: float | None = None) -> bool:
        return (now or time.monotonic()) - self.created_at > ttl_seconds


class PageContextStore:
    """Bounded, session-scoped custody of captured page contexts.

    Process-wide and lock-guarded for the same reason the search-spend ledger is:
    the tool that captures a context and the tool that reads it are built
    independently, and the session id is what keeps the two honest.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_contexts: int = DEFAULT_MAX_CONTEXTS,
    ) -> None:
        self._contexts: dict[str, PageContext] = {}
        #: session id -> context ids, most recent last.
        self._sessions: dict[str, list[str]] = {}
        self._ttl = ttl_seconds
        self._max = max_contexts
        self._lock = threading.Lock()

    def store(
        self,
        *,
        provider: str,
        query: str,
        blocks: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        enriched: bool = False,
    ) -> PageContext | None:
        """Capture a search's blocks. Returns None when there is nothing to keep.

        A search with no blocks (a provider with no page payload, or a transport
        that failed before producing results) is not an error here: it simply
        has no readable pages, and the caller reports that at read time.
        """
        usable = [
            block
            for block in blocks
            if isinstance(block, dict)
            and block.get("type") == "web_search_tool_result"
            and isinstance(block.get("content"), list)
            and block["content"]
        ]
        if not usable:
            return None
        context = PageContext(
            context_id=uuid.uuid4().hex,
            provider=provider,
            query=query,
            created_at=time.monotonic(),
            # Every block is kept, not just the result blocks: the replay must
            # reproduce the assistant turn as it was, and dropping the rest
            # changes the turn the tool results belong to.
            blocks=list(blocks),
            sources=list(sources),
            enriched=enriched,
        )
        with self._lock:
            self._contexts[context.context_id] = context
            while len(self._contexts) > self._max:
                oldest = min(self._contexts.values(), key=lambda item: item.created_at)
                self._drop_locked(oldest.context_id)
        return context

    def attach(self, session_id: str, context_id: str | None) -> None:
        """Make a captured context the one ``session_id`` will read next."""
        if not context_id:
            return
        key = session_id or ""
        with self._lock:
            if context_id not in self._contexts:
                return
            ids = self._sessions.setdefault(key, [])
            ids.append(context_id)
            # Keep the per-session list bounded too, or a long session grows a
            # list of ids pointing at contexts the store already evicted.
            del ids[: max(len(ids) - self._max, 0)]

    def for_session(self, session_id: str) -> PageContext | None:
        """The newest unexpired context this session captured, if any."""
        key = session_id or ""
        now = time.monotonic()
        with self._lock:
            ids = self._sessions.get(key, [])
            for context_id in reversed(ids):
                context = self._contexts.get(context_id)
                if context is None:
                    continue
                if context.ages_out(self._ttl, now):
                    self._drop_locked(context_id)
                    continue
                return context
        return None

    def latest(self) -> PageContext | None:
        """The newest unexpired context in the process, whoever captured it.

        For diagnostics and tests only. A read path must use
        :meth:`for_session`, which is the one that enforces the session boundary.
        """
        now = time.monotonic()
        with self._lock:
            live = [c for c in self._contexts.values() if not c.ages_out(self._ttl, now)]
            return max(live, key=lambda item: item.created_at) if live else None

    def get(self, context_id: str) -> PageContext | None:
        with self._lock:
            return self._contexts.get(context_id)

    def forget_session(self, session_id: str) -> None:
        key = session_id or ""
        with self._lock:
            for context_id in self._sessions.pop(key, []):
                self._drop_locked(context_id)

    def reset(self) -> None:
        with self._lock:
            self._contexts.clear()
            self._sessions.clear()

    def _drop_locked(self, context_id: str) -> None:
        """Drop one context and every reference to it. Caller holds the lock."""
        self._contexts.pop(context_id, None)
        for key, ids in list(self._sessions.items()):
            self._sessions[key] = [item for item in ids if item != context_id]


#: The process-wide store. Sessions read their own contexts through it.
PAGE_CONTEXTS = PageContextStore()
