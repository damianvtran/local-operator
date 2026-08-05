"""Cache-aware pruning of tool outputs.

In-place blanking, never deletion: provider conversations must keep every
tool-call/tool-result pair intact or the next request is rejected, so pruning
replaces a victim's content with a short notice and marks it ``pruned`` in its
details. The flags never reach provider wire formats.

Two passes (``pruneSupersededToolResults``):

(a) **Superseded reads** — a later tool result for the same path makes the
    earlier output dead weight (the model re-read precisely because the old
    copy was stale). The supersede key is ``details['path']``; results without
    a path group by tool name.
(b) **Useless-flagged results** — tools self-flag contextually worthless
    output (zero-match search, timed-out wait); blanked once consumed.

Prompt-cache guards (the whole point of doing this carefully):

- Warm-suffix guard: ``suffix_tokens[i]`` is the estimated size of everything
  after index ``i`` — the content the provider must re-write at cacheWrite
  premium if ``i`` is mutated. Candidates whose suffix exceeds
  ``cache_warm_suffix_tokens`` sit inside the warm cache prefix and are left
  for compaction (which rebuilds the cache anyway) — *unless* the idle flush
  applies.
- Idle flush: once ``now - last_activity >= idle_flush_ms`` the provider cache
  is provably cold (the default 90 min exceeds Anthropic's 1 h long-retention
  TTL), so re-writing the sent region is free and every candidate flushes.
- ``MIN_PRUNE_TOKENS`` floor: blanking a result smaller than the floor costs
  more than it saves (the placeholder itself has a token cost) and only churns
  the cache — skip it.
- Never blank ``is_error`` results (errors are signal), and never blank skill
  reads (``skill://`` paths or the ``skill`` tool) — a pruned skill forces the
  model into a re-read loop.
"""

from __future__ import annotations

from typing import Any, Sequence

from local_operator.harness.types import Message, TextContent

from .tokens import estimate_tokens, invalidate_message_cache

__all__ = [
    "MIN_PRUNE_TOKENS",
    "SUPERSEDED_NOTICE",
    "USELESS_NOTICE",
    "compute_suffix_tokens",
    "prune_tool_outputs",
]

#: Generic pruning floor. Below this, blanking recovers nothing — the
#: placeholder costs tokens too — so a sub-floor prune only grows the context
#: and churns the prompt cache.
MIN_PRUNE_TOKENS = 50

#: Exact placeholder written over a superseded tool result.
SUPERSEDED_NOTICE = "[Superseded by a newer read of this file]"

#: Exact placeholder written over an elided useless tool result.
USELESS_NOTICE = "[Uneventful result elided]"


def compute_suffix_tokens(messages: Sequence[Message]) -> list[int]:
    """Per-index sum of estimated tokens of all messages strictly AFTER it.

    ``suffix[i]`` is exactly how much prompt-cache content the provider must
    re-write if ``messages[i]`` is mutated in place. Computed with a reversed
    accumulation in O(n).
    """
    suffix = [0] * len(messages)
    accumulated = 0
    for i in range(len(messages) - 1, -1, -1):
        suffix[i] = accumulated
        accumulated += estimate_tokens(messages[i])
    return suffix


def _payload_of(message: Message) -> dict[str, Any]:
    """The message's provider payload when it is a plain dict, else ``{}``."""
    payload = message.provider_payload
    return payload if isinstance(payload, dict) else {}


def _details_of(message: Message) -> dict[str, Any]:
    """Structured details carried with a tool-result message.

    ``Message`` forbids extra fields, so tool metadata rides in
    ``provider_payload`` (set by the loop when it converts a ``ToolResult``):
    ``details = provider_payload.get('details')`` — e.g. ``{'path': ...}``
    for read/glob/grep results.
    """
    details = _payload_of(message).get("details")
    return details if isinstance(details, dict) else {}


def _is_useless(message: Message) -> bool:
    """Whether the tool flagged this result contextually worthless.

    The flag lands at ``provider_payload['useless']`` (details as a lenient
    fallback). Errors are never useless (errors win) — enforced here even
    though tools must not set both.
    """
    if message.is_error:
        return False
    payload = _payload_of(message)
    if payload.get("useless"):
        return True
    details = payload.get("details")
    return isinstance(details, dict) and bool(details.get("useless"))


def _is_pruned(message: Message) -> bool:
    """Whether a prior prune pass already blanked this result."""
    return bool(_payload_of(message).get("pruned")) or bool(_details_of(message).get("pruned"))


def _is_prunable(message: Message) -> bool:
    """One gate for BOTH prune passes (``protectedTools`` applies to the
    superseded AND useless passes alike): never errors, never skill reads.

    Skill reads are exempt because a pruned skill gets re-read in a loop.
    Protection matches ``["skill", isSkillReadToolResult]``: the
    ``skill`` tool itself, or a ``read`` result whose recorded target is a
    ``skill://`` URL — the loop stores internal-URL reads under
    ``details['url']`` (tools/builtin.py) and filesystem reads under
    ``details['path']``; both are checked.
    """
    if message.is_error:
        return False
    if message.tool_name == "skill":
        return False
    if message.tool_name == "read":
        details = _details_of(message)
        for key in ("path", "url"):
            value = details.get(key)
            if isinstance(value, str) and value.startswith("skill://"):
                return False
    return True


def _supersede_key(message: Message) -> str | None:
    """Group key for supersede detection.

    ``(tool_name, details['path'], details['range'])`` when a path is present:
    a later result for the same tool reading the same file supersedes the
    earlier one. Results without a path are exempt — grouping them by bare
    tool name would blank legitimately distinct outputs (two different bash
    runs, two searches). Mirrors the established behavior, where only path-carrying read results
    supersede.

    The range rides in the key: a ranged read must NOT supersede a different
    range of the same file (they share not one line), while a full-file read
    supersedes every earlier range — handled in the pass below.
    """
    details = _details_of(message)
    path = details.get("path")
    if isinstance(path, str) and path and message.tool_name:
        return f"{message.tool_name}:{path}:{details.get('range') or 'full'}"
    return None


def _supersede_path_range(message: Message) -> tuple[str, str | None] | None:
    """(path, range-or-None) for the supersede pass."""
    details = _details_of(message)
    path = details.get("path")
    if isinstance(path, str) and path and message.tool_name:
        return path, details.get("range")
    return None


def _blank(message: Message, notice: str) -> None:
    """Replace ``message`` content with ``notice`` and mark it pruned.

    Pairing survives (the message object stays at its index with its
    ``tool_call_id``). The prior payload is preserved and gains the marker:
    ``provider_payload = {**old_payload, 'pruned': True}`` — the pruned flag
    must not mask any previously stored keys. The estimate cache is
    invalidated because the content changed in place.
    """
    old_payload = _payload_of(message)
    message.content = [TextContent(text=notice)]
    message.provider_payload = {**old_payload, "pruned": True}
    invalidate_message_cache(message)


def prune_tool_outputs(
    messages: Sequence[Message],
    now_ms: int,
    last_activity_ms: int,
    cache_warm_suffix_tokens: int = 8000,
    idle_flush_ms: int = 5400000,
) -> tuple[list[Message], bool]:
    """Blank superseded and useless tool outputs in place.

    ``now_ms`` and ``last_activity_ms`` are MILLISECONDS epoch values — the
    idle-flush gap is compared against ``idle_flush_ms`` directly, so passing
    seconds here silently disables the idle flush (see RC-8 regression test).

    Returns ``(messages, changed)`` — the same messages, mutated in place;
    ``changed`` is True when at least one result was blanked. Messages are
    NEVER removed so tool-call/result pairing survives for every provider.

    Guards, in order per candidate: never errors, never skill reads, never
    already-pruned, never below :data:`MIN_PRUNE_TOKENS`, and never inside
    the warm cache suffix — unless the idle flush window has elapsed.
    """
    if not messages:
        return list(messages), False

    idle = (now_ms - last_activity_ms) >= idle_flush_ms
    suffix_tokens = compute_suffix_tokens(messages)

    # Pass (a): superseded reads. Walk newest-to-oldest; a seen key marks any
    # earlier result with the same key superseded, and a seen FULL read marks
    # any earlier RANGED read of the same path superseded (never the reverse).
    seen_keys: set[str] = set()
    seen_full_paths: set[str] = set()
    superseded: list[int] = []
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if message.role != "tool":
            continue
        if not _is_prunable(message):
            continue
        if _is_pruned(message):
            continue
        key = _supersede_key(message)
        if key is None:
            continue
        path_range = _supersede_path_range(message)
        is_superseded = key in seen_keys or (
            path_range is not None
            and path_range[1] is not None
            and path_range[0] in seen_full_paths
        )
        if is_superseded:
            superseded.append(i)
        seen_keys.add(key)
        if path_range is not None and path_range[1] is None:
            seen_full_paths.add(path_range[0])

    # Pass (b): useless-flagged results (excluding supersede victims).
    superseded_set = set(superseded)
    useless = [
        i
        for i, message in enumerate(messages)
        if message.role == "tool"
        and i not in superseded_set
        and _is_prunable(message)
        and not _is_pruned(message)
        and _is_useless(message)
    ]

    changed = False
    for indices, notice in ((superseded, SUPERSEDED_NOTICE), (useless, USELESS_NOTICE)):
        for i in indices:
            message = messages[i]
            if estimate_tokens(message) < MIN_PRUNE_TOKENS:
                continue
            if not idle and suffix_tokens[i] > cache_warm_suffix_tokens:
                # Inside the warm cache prefix: re-writing it costs the
                # cacheWrite premium on the whole suffix. Leave it for
                # compaction, which rebuilds the cache anyway.
                continue
            _blank(message, notice)
            changed = True

    return list(messages), changed
