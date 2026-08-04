"""Append-only context management for provider prefix-cache friendliness.

Port of omp ``packages/agent/src/append-only-context.ts``. Two cooperating
mechanisms keep the bytes sent to a provider maximally cacheable:

- :class:`StablePrefix` snapshots the system prompt blocks and tool inventory
  once per fingerprint change, so providers can place cache breakpoints ahead
  of them.
- :class:`AppendOnlyLog` records the normalized message list and only ever
  grows at the tail — except for the three controlled mutations in
  :meth:`AppendOnlyContextManager.sync_messages`.

``sync_messages`` distinguishes three cases (the third is a real bug fix over
naive clear-and-replay, see omp issue #3406): append, shrink-clear
(compaction), and longest-stable-prefix rewrite (in-place pruning / re-render).
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from local_operator.harness.types import Message, TextContent

if TYPE_CHECKING:
    from collections.abc import Sequence

    from local_operator.harness.loop import LoopContext


def message_digest(message: Message) -> str:
    """Byte-stable digest of every field a provider may serialize.

    Missing any of these fields makes an in-place rewrite invisible and
    silently corrupts the cached prefix, so the digest deliberately covers:
    role, text content, tool-call ids/names/raw arguments, ``tool_call_id``,
    ``tool_name``, ``is_error`` and the opaque ``provider_payload``.
    """
    payload: dict[str, object] = {
        "role": message.role,
        "content": [
            block.text for block in message.content if isinstance(block, TextContent)
        ],
        "tool_calls": [
            {"id": call.id, "name": call.name, "raw_arguments": call.raw_arguments}
            for call in message.tool_calls
        ],
        "tool_call_id": message.tool_call_id,
        "tool_name": message.tool_name,
        "is_error": message.is_error,
        "provider_payload": message.provider_payload,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class StablePrefix:
    """Snapshot of the cache-stable head of a request (system + tools).

    ``build`` re-snapshots only when the fingerprint changes and reports
    whether it changed, so callers can reset provider-side cache state exactly
    when needed and never otherwise.
    """

    def __init__(self) -> None:
        self._system_blocks: list[str] = []
        self._tool_keys: list[tuple[str, str]] = []
        self._fingerprint: str = ""
        self._built: bool = False
        self._version: int = 0

    @staticmethod
    def _compute_fingerprint(context: "LoopContext") -> tuple[str, list[str], list[tuple[str, str]]]:
        blocks = list(context.system_blocks)
        tool_keys = [(tool.name, tool.description) for tool in context.tools]
        canonical = json.dumps(
            {"system": blocks, "tools": tool_keys},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), blocks, tool_keys

    def build(self, context: "LoopContext") -> bool:
        """Snapshot the prefix from ``context``. Returns True when changed."""
        fingerprint, blocks, tool_keys = self._compute_fingerprint(context)
        if self._built and fingerprint == self._fingerprint:
            return False
        self._fingerprint = fingerprint
        self._system_blocks = blocks
        self._tool_keys = tool_keys
        self._built = True
        self._version += 1
        return True

    def invalidate(self) -> None:
        """Force a rebuild on the next ``build`` (e.g. model switch)."""
        self._built = False
        self._fingerprint = ""

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def version(self) -> int:
        return self._version

    @property
    def built(self) -> bool:
        return self._built

    def to_context(self) -> dict[str, object]:
        """Return the snapshotted stable head as plain data."""
        return {"system_blocks": list(self._system_blocks), "tools": list(self._tool_keys)}


class AppendOnlyLog:
    """The normalized message log. Grows at the tail only.

    ``replace_tail`` is legal for compaction only — every other rewrite path
    must go through :meth:`AppendOnlyContextManager.sync_messages` so the
    longest-stable-prefix invariant is preserved.
    """

    def __init__(self) -> None:
        self._entries: list[Message] = []

    def append(self, message: Message) -> None:
        self._entries.append(message)

    def extend(self, messages: Sequence[Message]) -> None:
        self._entries.extend(messages)

    def replace_tail(self, message: Message) -> None:
        """Replace the last entry. COMPACTION ONLY — see class docstring."""
        if not self._entries:
            raise ValueError("replace_tail on an empty log")
        self._entries[-1] = message

    def truncate(self, count: int) -> None:
        """Keep the first ``count`` entries."""
        del self._entries[count:]

    def clear(self) -> None:
        self._entries.clear()

    def to_messages(self) -> list[Message]:
        return list(self._entries)

    def entries(self) -> tuple[Message, ...]:
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


class AppendOnlyContextManager:
    """Keeps the provider-bound message list append-only across turns.

    The host converts its transcript to normalized :class:`Message` objects
    each turn and hands them to :meth:`sync_messages`; the manager reconciles
    them against the log using per-message digests so the log (and therefore
    the provider prefix cache) is disturbed as little as possible.
    """

    def __init__(self) -> None:
        self.prefix = StablePrefix()
        self.log = AppendOnlyLog()
        self._digests: list[str] = []

    def build(self, context: "LoopContext") -> bool:
        """(Re)build the stable prefix. Returns True when it changed."""
        return self.prefix.build(context)

    def sync_messages(self, normalized: list[Message]) -> None:
        """Reconcile ``normalized`` against the log.

        Three cases:

        1. **Append** — the log is a prefix of ``normalized``: append the tail.
        2. **Shrink-clear** — ``normalized`` shrank (compaction): clear and
           replay the whole list.
        3. **Rewrite** — same or greater length but a digest diverged
           (pruning, re-render): truncate to the longest stable prefix and
           append the diverged tail.
        """
        new_digests = [message_digest(message) for message in normalized]
        stored = self._digests

        # Longest common prefix between stored digests and the new list.
        stable = 0
        limit = min(len(stored), len(new_digests))
        while stable < limit and stored[stable] == new_digests[stable]:
            stable += 1

        if len(normalized) < len(stored):
            # Case 2: compaction or reset — the array shrank. Clear and replay.
            self.log.clear()
            self.log.extend(normalized)
            self._digests = new_digests
            return

        if stable == len(stored):
            # Case 1: pure append.
            if len(normalized) > len(stored):
                self.log.extend(normalized[len(stored) :])
            self._digests = new_digests
            return

        # Case 3: in-place rewrite. Keep the byte-stable head, replay the tail.
        self.log.truncate(stable)
        self.log.extend(normalized[stable:])
        self._digests = new_digests

    def invalidate_for_model_change(self) -> None:
        """The prefix cache cannot survive a model switch."""
        self.prefix.invalidate()

    def reset(self, context: "LoopContext") -> None:
        """Full reset (session reset, stale-replay recovery)."""
        self.prefix.invalidate()
        self.log.clear()
        self._digests = []
        self.prefix.build(context)
