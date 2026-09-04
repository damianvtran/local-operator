"""Cache-aware pruning of tool outputs.

In-place blanking, never deletion: provider conversations must keep every
tool-call/tool-result pair intact or the next request is rejected, so pruning
replaces a victim's content with a short notice and marks it ``pruned`` in its
details. The flags never reach provider wire formats.

Two passes (``pruneSupersededToolResults``):

(a) **Superseded results** — a later tool result for the same path, or with
    the same explicitly declared ``details['supersede_key']``, makes the
    earlier output dead weight. Results with neither key are exempt: grouping
    pathless results by tool name would collapse legitimately distinct output.
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

from local_operator.harness.types import Content, ImageContent, Message, TextContent

from .marker import marker_exists
from .tokens import estimate_tokens, estimate_wire_bytes, invalidate_message_cache

__all__ = [
    "MIN_PRUNE_TOKENS",
    "STALE_FRAME_NOTICE",
    "SUPERSEDED_NOTICE",
    "USELESS_NOTICE",
    "compute_suffix_tokens",
    "count_frame_messages",
    "count_stale_observations",
    "prune_stale_frames",
    "prune_tool_outputs",
    "shed_frames_to_wire_budget",
    "shed_stale_frames",
]

#: Generic pruning floor. Below this, blanking recovers nothing — the
#: placeholder costs tokens too — so a sub-floor prune only grows the context
#: and churns the prompt cache.
MIN_PRUNE_TOKENS = 50

#: Exact placeholder written over a superseded tool result.
SUPERSEDED_NOTICE = "[Superseded by a newer result for the same resource]"

#: Exact placeholder written over an elided useless tool result.
USELESS_NOTICE = "[Uneventful result elided]"

#: Exact placeholder written over a screenshot that a newer view of the same
#: surface has superseded (see :func:`prune_stale_frames`).
STALE_FRAME_NOTICE = "[screenshot omitted: superseded by a more recent view of the same surface]"


def count_frame_messages(messages: Sequence[Message]) -> int:
    """How many messages still carry at least one image block.

    This is the unit :func:`prune_stale_frames` keeps and the unit a
    screen-driving caller budgets its rebuild cadence in (one observation =
    one message = one frame in the runner's history), so both count the same
    thing.
    """
    return sum(1 for message in messages if _has_frame(message))


def _has_frame(message: Message) -> bool:
    return any(isinstance(block, ImageContent) for block in message.content)


def _without_frames(content: Sequence[Content]) -> list[Content]:
    """``content`` with every image replaced by ONE notice.

    Consecutive images collapse into a single notice rather than one each: a
    message that carried three views of a surface reads as "a screenshot was
    here", not as three lines of the same boilerplate, and the point of the
    prune is to spend fewer tokens on the past, not to spend them on notices.
    """
    out: list[Content] = []
    for block in content:
        if isinstance(block, ImageContent):
            if out and isinstance(out[-1], TextContent) and out[-1].text == STALE_FRAME_NOTICE:
                continue
            out.append(TextContent(text=STALE_FRAME_NOTICE))
        else:
            out.append(block)
    return out


def prune_stale_frames(
    messages: Sequence[Message], *, keep_recent_frames: int
) -> tuple[list[Message], int]:
    """Replace every image except those in the newest ``keep_recent_frames``
    frame-bearing messages with :data:`STALE_FRAME_NOTICE`.

    Returns ``(messages, frames_dropped)`` where ``messages`` is a NEW list:
    messages that lose a frame are COPIED (``model_copy``) with the replaced
    content, and every other message is reused by identity. That split is the
    contract a prompt-cache-aware caller relies on. The session's tool-output
    prune blanks in place because the session owns its messages outright; the
    episode runner's history is a sent prefix whose identity IS the proof that
    a message was not rewritten, so it must be able to tell "untouched" from
    "rewritten" by ``is``. Copying the victims and reusing the rest gives it
    that, and gives the session (which does not care) the same result.

    No message is ever removed, so the user/assistant alternation and every
    tool pairing survive. ``frames_dropped`` counts image blocks replaced, not
    messages touched. ``keep_recent_frames`` counts frame-bearing MESSAGES
    from the end, so a message with two frames costs one slot.

    There is deliberately no cadence knob here: WHEN to prune is the caller's
    decision (the runner rebuilds its whole prefix on a frame budget, the
    session would do it inside a compaction pass) and encoding a frontier
    here would make two callers disagree about it.
    """
    if keep_recent_frames < 0:
        raise ValueError("keep_recent_frames must be non-negative")
    # Walk from the end: the newest ``keep_recent_frames`` frame-bearing
    # messages are kept verbatim, everything older loses its images.
    remaining = keep_recent_frames
    out: list[Message] = []
    dropped = 0
    for message in reversed(messages):
        if not _has_frame(message):
            out.append(message)
            continue
        if remaining > 0:
            remaining -= 1
            out.append(message)
            continue
        dropped += sum(1 for block in message.content if isinstance(block, ImageContent))
        # A copy rather than in-place mutation, so the caller's identity test
        # (see the docstring) holds; ``model_copy`` keeps the id, which is
        # what the token-estimate cache is keyed on, so invalidate it.
        replaced = message.model_copy(update={"content": _without_frames(message.content)})
        invalidate_message_cache(replaced)
        out.append(replaced)
    out.reverse()
    return out, dropped


def shed_frames_to_wire_budget(
    messages: Sequence[Message], *, budget: int
) -> tuple[list[Message], int]:
    """Replace the OLDEST frames with notices until the request fits ``budget``
    bytes, keeping as many recent frames as possible.

    The last-resort transport guard. Ordinary compaction decides what to keep
    by *context* value; this decides by whether the provider will accept the
    request at all, which is a different and strictly narrower question — so
    it engages only when the payload is already over a ceiling set below the
    provider's cap, and does nothing at all below it. The early return is not
    an optimisation detail: it is the guarantee that every session under
    budget behaves byte-identically to one without this function.

    Built on :func:`prune_stale_frames` rather than as a second
    frame-dropping primitive, so there is ONE definition of "which frames are
    recent" and one set of copy/identity semantics. That also means no message
    is ever removed — user/assistant alternation and tool pairings survive
    untouched, which a transport guard running on arbitrary history must
    guarantee, since it cannot know what is mid-tool-call.

    The loop mirrors ``_shed_stale_turns`` in the evaluation runner's provider
    client: walk the keep count DOWN and stop when a step frees nothing.
    Termination is structural — ``keep`` strictly decreases toward 0, and
    ``keep=0`` drops every image there is — so a payload still over budget
    with no images left exits rather than spinning. That residual case is
    real: a text-only history cannot be shed here, and the caller must be
    prepared for "still over" rather than assuming success (this is what the
    byte-side auto-continue band checks).

    Sheds the FEWEST frames that fit, not a fixed count, because every frame
    dropped is evidence the user may still need. Returns ``(messages,
    frames_dropped)``; ``frames_dropped`` is 0 and the input list is returned
    unchanged when nothing was over budget.
    """
    if budget <= 0:
        return list(messages), 0
    if estimate_wire_bytes(messages) <= budget:
        return list(messages), 0

    keep = count_frame_messages(messages)
    working = list(messages)
    dropped = 0
    while keep > 0:
        keep -= 1
        candidate, candidate_dropped = prune_stale_frames(messages, keep_recent_frames=keep)
        if candidate_dropped <= dropped:
            # This step freed nothing new; another turn of the loop cannot
            # either, since the keep count only shrinks.
            break
        working, dropped = candidate, candidate_dropped
        if estimate_wire_bytes(working) <= budget:
            break
    return working, dropped


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

    ``details['supersede_key']`` lets a tool opt a result in explicitly, for
    resources a path does not name: a URL, a page, a live surface. It is an
    OPT-IN the emitting tool states, deliberately not inferred from ambient
    fields like a url or a surface handle. Inference is unsafe here because a
    single handle covers many DIFFERENT results: the browser tool stamps its
    ``surface_id`` on every action, so keying on it made a 40-character
    ``click`` blank the accessibility snapshot and console errors the agent had
    just gathered to decide what to click. Only the tool knows which of its
    results are re-reads of one thing and which are distinct observations of
    it, so only the tool may say so.

    The value must therefore identify the CONTENT, including any variant that
    changes what comes back (a raw versus rendered fetch of one URL is two
    different answers, not one superseding the other).
    """
    details = _details_of(message)
    if not message.tool_name:
        return None
    path = details.get("path")
    if isinstance(path, str) and path:
        # The namespace marker prevents a path crafted like a declared key
        # from colliding with a tool's explicit opt-in contract.
        return f"{message.tool_name}:path:{path}:{details.get('range') or 'full'}"
    declared = details.get("supersede_key")
    if isinstance(declared, str) and declared:
        return f"{message.tool_name}:declared:{declared}"
    return None


def _supersede_path_range(message: Message) -> tuple[str, str | None] | None:
    """(path, range-or-None) for the supersede pass."""
    details = _details_of(message)
    path = details.get("path")
    if isinstance(path, str) and path and message.tool_name:
        return path, details.get("range")
    return None


def _span_of(
    path_range: tuple[str, str | None],
) -> tuple[str, tuple[int, int]] | None:
    """``(path, (start, end))`` for a ranged read, or None for full/non-file.

    ``range`` is a 1-based inclusive ``"start-end"`` (or ``"start-"``). A
    ``None`` or ``"full"`` range is a full read (handled by the full-path
    supersede, not here). Any unparsable range degrades to None: an
    unreadable spec must not blank a real read on a guess.
    """
    path, spec = path_range
    if not spec or spec == "full":
        return None
    try:
        start_s, sep, end_s = spec.partition("-")
        start = int(start_s)
        if not sep:
            return None
        end = int(end_s) if end_s.strip() else None
    except (TypeError, ValueError):
        return None
    if start < 1 or (end is not None and end < start):
        return None
    return path, (start, end if end is not None else start)


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

    # Pass (a): superseded reads. Walk newest-to-oldest; a seen key marks any
    # earlier result with the same key superseded, and a seen FULL read marks
    # any earlier RANGED read of the same path superseded (never the reverse).
    # A seen ranged read ALSO supersedes an earlier ranged read of the same
    # path whose span it fully covers — the model pages through a large file
    # in overlapping chunks (read 162-500, then 540-900, ...), and without
    # coverage supersede every chunk stays live, burning the whole file into
    # context N times. Coverage, not mere adjacency: a later read only blanks
    # an earlier one it genuinely includes.
    seen_keys: set[str] = set()
    seen_full_paths: set[str] = set()
    seen_covered_spans: dict[str, list[tuple[int, int]]] = {}
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
        span = _span_of(path_range) if path_range is not None else None
        is_superseded = key in seen_keys or (
            path_range is not None
            and path_range[1] is not None
            and path_range[0] in seen_full_paths
        )
        if not is_superseded and span is not None:
            path, (start, end) = span
            for later_start, later_end in seen_covered_spans.get(path, []):
                if start >= later_start and end <= later_end:
                    is_superseded = True
                    break
        if is_superseded:
            superseded.append(i)
        seen_keys.add(key)
        if path_range is not None and path_range[1] is None:
            seen_full_paths.add(path_range[0])
        elif span is not None:
            path, (start, end) = span
            seen_covered_spans.setdefault(path, []).append((start, end))

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

    # Suffix sums are computed HERE, after the candidate passes, and only when
    # there is a candidate to judge. They are read by exactly one guard below
    # (``suffix_tokens[i] > cache_warm_suffix_tokens``), and both passes above
    # are pure structural walks that never look at a token count — so deferring
    # is behaviour-identical: nothing has been mutated yet, and the sums are a
    # pure function of ``messages``.
    #
    # It matters because the estimator is not cheap to reach. The first
    # ``estimate_tokens`` call in a process loads tiktoken's cl100k_base BPE
    # table, which costs ~84 ms and ~43.6 MB RSS (measured with
    # scripts/bench_base_overhead.py) — 43% of the peak RSS of a no-op ``exec``
    # run. Pruning runs every turn, while the common turn has NO prunable tool
    # output at all (and a no-op run has no tool results whatsoever), so the
    # old unconditional call made every session pay 43.6 MB to decide there was
    # nothing to prune. Candidates present -> the guard needs real numbers and
    # we pay honestly.
    suffix_tokens = compute_suffix_tokens(messages) if (superseded or useless) else []

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


def _is_stale_observation(message: Message) -> bool:
    """An observation turn the shed may remove: it carries a frame, or the
    notice a frame prune left in a frame's place.

    The pruned form matters more than the framed one. ``run_compaction_pass``
    prunes frames to ``keep_recent_frames`` BEFORE a caller can shed, so the
    front of the kept tail is always pruned notices, never images; a shed that
    stopped at the first frameless message stopped before it began (review
    round 3, M3). A plain text-only turn (no frame, no notice) is not stale in
    this sense — a text-only benchmark has nothing here to shed, and its
    window is the summary's problem, not deletion's.
    """
    if _has_frame(message):
        return True
    return any(
        isinstance(block, TextContent) and block.text == STALE_FRAME_NOTICE
        for block in message.content
    )


def count_stale_observations(messages: Sequence[Message]) -> int:
    """How many observation turns :func:`shed_stale_frames` could shed
    (framed or pruned-to-notice), the unit its ``limit`` is counted in.

    A compaction marker is excluded even when it carries frames (a snapcompact
    replay does): the shed never removes one, so counting it would hand a
    caller a ``limit`` one too high — the first shed request would then be
    for the count the tail already has, remove nothing, and read as "nothing
    left to shed" (the exact miscount that made the client's shed a no-op).
    """
    return sum(
        1 for message in messages if _is_stale_observation(message) and not marker_exists([message])
    )


def shed_stale_frames(messages: Sequence[Message], *, limit: int) -> tuple[list[Message], int]:
    """Remove whole stale observation turns, oldest first, from the FRONT of
    the kept tail so at most ``limit`` stale observations remain.

    A stale observation is one that carries a frame OR the notice a frame
    prune left behind (:func:`_is_stale_observation`) — the shed operates on
    TURNS, not on the presence of an image, because by the time a caller
    sheds, the oldest turns have already been pruned to notices and are the
    stalest thing in the tail. "Oldest first" means those, never the recent
    frames a screen-driving surface actually acts on. A turn is its
    observation plus EVERYTHING up to the next observation: the reply, and
    any rejected-reply / correction pair that preceded the accepted reply. All
    of it goes with the observation, because a reply that answers an
    observation no longer visible is a decision made about a screen the model
    cannot see.

    The same rule covers a tail whose FRONT is not an observation. The
    compaction cut point may legally land on an assistant reply or on a
    correction message, leaving the end of a turn whose observation is
    already behind the marker; those orphans are shed first. Stopping on them
    instead (the earlier behaviour) stalled the shed for the rest of the
    episode: every pass found a non-observation at the front, removed
    nothing, and the client refused the request as unrecoverable with a dozen
    sheddable turns still in the prefix.

    Never removes the current observation (the last message — without it the
    request has no question), and stops short of a compaction marker in the
    kept tail: content already summarised must not be deleted twice. Returns
    ``(messages, removed)``; the survivors are reused by identity, same as
    :func:`prune_stale_frames`.
    """
    if limit < 0:
        raise ValueError("limit must be non-negative")
    head = 0
    for index, message in enumerate(messages):
        if marker_exists([message]):
            head = index + 1
    tail = list(messages[head:])
    while tail and count_stale_observations(tail) > limit:
        # Everything before the NEXT stale observation is the oldest turn (or
        # the orphaned end of one); never the last message, which is the
        # current observation and the request's only question.
        end = 1
        while end < len(tail) and not _is_stale_observation(tail[end]):
            end += 1
        if end >= len(tail):
            break
        del tail[:end]
    if head == 0 and len(tail) == len(messages):
        return list(messages), 0
    return [*messages[:head], *tail], len(messages) - head - len(tail)
