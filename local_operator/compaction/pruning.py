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
  applies, or the candidate is a superseded victim at least as large as that
  suffix (see :data:`OVERSIZED_SUPERSEDE_TOKENS`). The useless pass has no
  such exception: a worthless-once-consumed result does not repay a warm
  re-write on any horizon the supersede arithmetic promises.
- Idle flush: once ``now - last_activity >= idle_flush_ms`` the provider cache
  is provably cold (the default 90 min exceeds Anthropic's 1 h long-retention
  TTL), so re-writing the sent region is free and every candidate flushes.
- ``MIN_PRUNE_TOKENS`` floor: blanking a result smaller than the floor costs
  more than it saves (the placeholder itself has a token cost) and only churns
  the cache — skip it.
- Never blank ``is_error`` results (errors are signal), and never blank skill
  or guide reads (``skill://``/``guide://`` paths or the ``skill`` tool) — a
  pruned skill or guide forces the model into a re-read loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Sequence

from local_operator.harness.types import (
    CapturedFrame,
    Content,
    ImageContent,
    Message,
    TextContent,
)

from .marker import marker_exists
from .tokens import estimate_tokens, estimate_wire_bytes, invalidate_message_cache

__all__ = [
    "CaptureElision",
    "CapturePartition",
    "MIN_PRUNE_TOKENS",
    "SESSION_KEEP_RECENT_FRAMES",
    "STALE_FRAME_NOTICE",
    "SUPERSEDED_NOTICE",
    "USELESS_NOTICE",
    "apply_capture_elisions",
    "compute_suffix_tokens",
    "count_frame_messages",
    "count_stale_observations",
    "downscale_stale_frames",
    "fit_frames_to_wire_budget",
    "partition_captures",
    "prune_stale_frames",
    "prune_stale_frames_in_place",
    "prune_tool_outputs",
    "reclaim_captures_to_wire_budget",
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

#: Minimum size for the Fix C oversized-supersede exception (see
#: :func:`prune_tool_outputs`). The warm-suffix guard exists because blanking
#: forces the provider to re-write everything after the victim at the
#: cacheWrite premium, so a small victim costs more than it saves. That
#: arithmetic inverts only when the victim itself outweighs the suffix it
#: invalidates: the suffix is re-written once, while the victim would be
#: re-billed on every request until compaction, so a victim at least as
#: large as its suffix is repaid on the NEXT request alone. The exception
#: therefore compares the victim estimate against the ACTUAL suffix
#: (``suffix_tokens[i]``), never against a fixed number alone. This constant
#: is only a floor beneath that comparison, mirroring
#: ``cache_warm_suffix_tokens``' default (8,000) in the same units the guard
#: already uses, so the exception can never fire for a victim the guard
#: itself would call small. It deliberately does NOT apply to the useless
#: pass (see the module docstring). This is NOT an earlier trigger: pruning
#: already runs post-turn (session._maybe_compact -> _plan_compaction);
#: the victims that accumulated did so because this guard deferred them,
#: so the guard is where the fix belongs.
OVERSIZED_SUPERSEDE_TOKENS = 8000

#: How many of the newest frame-bearing messages the session-side stale-frame
#: sweep always keeps (see :func:`prune_stale_frames_in_place`). Two, not one:
#: a before/after capture pair — the flow the sweep must not break — is exactly
#: "the newest two frames" while the comparison is in flight, and a newest-2
#: window still leaves the measured majority droppable on this fleet (83.9%
#: of resident frames sat outside it, 91.7% outside newest-1). The evaluation
#: runner keeps more (3) because a screen-driving loop re-frames every step.
SESSION_KEEP_RECENT_FRAMES = 2


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


def _without_frames(
    content: Sequence[Content],
    *,
    index: int = 0,
    reclaimable: Collection[tuple[int, int]] | None = None,
) -> list[Content]:
    """``content`` with every image replaced by ONE notice.

    Consecutive images collapse into a single notice rather than one each: a
    message that carried three views of a surface reads as "a screenshot was
    here", not as three lines of the same boilerplate, and the point of the
    prune is to spend fewer tokens on the past, not to spend them on notices.

    ``reclaimable`` narrows which image blocks may be replaced at all: when
    given, a block is a candidate only when its ``(index, content position)``
    is listed -- the coordinates :func:`partition_captures` reports, so the
    prune and the epoch elision agree block for block about what is
    superseded. ``None`` keeps the legacy every-image contract for callers
    that manage protection themselves.
    """
    out: list[Content] = []
    for position, block in enumerate(content):
        if isinstance(block, ImageContent):
            if reclaimable is not None and (index, position) not in reclaimable:
                out.append(block)
                continue
            if out and isinstance(out[-1], TextContent) and out[-1].text == STALE_FRAME_NOTICE:
                continue
            out.append(TextContent(text=STALE_FRAME_NOTICE))
        else:
            out.append(block)
    return out


def prune_stale_frames(
    messages: Sequence[Message],
    *,
    keep_recent_frames: int,
    reclaimable: Collection[tuple[int, int]] | None = None,
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

    ``reclaimable`` gates WHICH blocks the prune may replace, in
    :func:`partition_captures` coordinates: the render fallback passes the
    superseded-capture set, so user uploads, unknown images and every
    stream's latest capture survive the hard-wire shed exactly as they
    survive the epoch elision. ``None`` keeps the legacy every-image
    contract for callers that own protection themselves.
    """
    if keep_recent_frames < 0:
        raise ValueError("keep_recent_frames must be non-negative")
    # Walk from the end: the newest ``keep_recent_frames`` frame-bearing
    # messages are kept verbatim; any older message loses only the image
    # blocks ``reclaimable`` admits (all of them under the legacy default).
    remaining = keep_recent_frames
    out: list[Message] = []
    dropped = 0
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not _has_frame(message):
            out.append(message)
            continue
        if remaining > 0:
            remaining -= 1
            out.append(message)
            continue
        replaced_count = sum(
            1
            for position, block in enumerate(message.content)
            if isinstance(block, ImageContent)
            and (reclaimable is None or (index, position) in reclaimable)
        )
        if replaced_count == 0:
            out.append(message)
            continue
        dropped += replaced_count
        # A copy rather than in-place mutation, so the caller's identity test
        # (see the docstring) holds; ``model_copy`` keeps the id, which is
        # what the token-estimate cache is keyed on, so invalidate it.
        replaced = message.model_copy(
            update={
                "content": _without_frames(message.content, index=index, reclaimable=reclaimable)
            }
        )
        invalidate_message_cache(replaced)
        out.append(replaced)
    out.reverse()
    return out, dropped


def downscale_stale_frames(
    messages: Sequence[Message],
    *,
    keep_recent_frames: int,
    max_edge: int,
    reclaimable: Collection[tuple[int, int]] | None = None,
) -> tuple[list[Message], int]:
    """Replace every image except those in the newest ``keep_recent_frames``
    frame-bearing messages with a smaller re-rendering bounded to ``max_edge``.

    The fidelity-preserving sibling of :func:`prune_stale_frames`: same walk,
    same copy/identity contract (messages that lose bytes are COPIED so a
    caller holding the original can still tell untouched from rewritten by
    ``is``; every other message is reused by identity), same guarantee that no
    message is ever removed. The difference is what happens to the old frame —
    the prune replaces it with :data:`STALE_FRAME_NOTICE`, this replaces it
    with a smaller copy of itself. A notice is information destroyed; a
    smaller frame is information concentrated, and keeping every block in
    place preserves the history's SHAPE, which is what the model's ability to
    reference earlier observations (and a prompt cache's prefix) both rest on.

    A block the transform cannot shrink (undecodable, or the rung cannot beat
    its current bytes) is left EXACTLY as it was: shrinking history must never
    be able to destroy context on its own, so the fallback is always the
    original bytes. ``max_edge`` is a concrete rung, not a policy — WHEN to
    downscale and how far to walk is :func:`fit_frames_to_wire_budget`'s
    decision, so the two callers of a downscale cannot disagree about the
    frontier.

    Returns ``(messages, frames_downscaled)``; ``frames_downscaled`` counts
    image blocks whose bytes were replaced, not messages touched.
    ``keep_recent_frames`` counts frame-bearing MESSAGES from the end, so a
    message with two frames costs one slot (the same unit the prune uses).

    ``reclaimable`` gates WHICH blocks are re-rendered, in
    :func:`partition_captures` coordinates -- the same protection contract
    :func:`prune_stale_frames` documents. ``None`` keeps the legacy
    every-image contract; the render fallback passes the superseded-capture
    set so protected pixels are never re-rendered by this walk.
    """
    # Imported HERE, not at module scope, and the placement is load-bearing:
    # this module sits on the shared renderer's import closure
    # (``harness.render`` -> ``compaction.api`` -> this file), and a benchmark
    # episode importing that renderer must not drag the application in —
    # ``local_operator.imaging`` is explicitly on the episode denylist
    # (tests/unit/evaluation/runner/test_isolation.py). The downscale itself
    # only ever runs on the session side, where the decode is wanted.
    from local_operator.imaging import downscale_context_frame

    if keep_recent_frames < 0:
        raise ValueError("keep_recent_frames must be non-negative")
    remaining = keep_recent_frames
    out: list[Message] = []
    downscaled = 0
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not _has_frame(message):
            out.append(message)
            continue
        if remaining > 0:
            remaining -= 1
            out.append(message)
            continue
        content: list[Content] = []
        changed = False
        for position, block in enumerate(message.content):
            if isinstance(block, ImageContent):
                if reclaimable is not None and (index, position) not in reclaimable:
                    content.append(block)
                    continue
                shrunk = downscale_context_frame(block.data, block.mime_type, max_edge=max_edge)
                if shrunk is not None:
                    data, mime_type = shrunk
                    block = block.model_copy(update={"data": data, "mime_type": mime_type})
                    changed = True
                    downscaled += 1
            content.append(block)
        if changed:
            # A copy rather than in-place mutation, so the caller's identity
            # test (see the docstring) holds; ``model_copy`` keeps the id,
            # which is what the token-estimate cache is keyed on, so
            # invalidate it — the same dance ``prune_stale_frames`` does.
            replaced = message.model_copy(update={"content": content})
            invalidate_message_cache(replaced)
            out.append(replaced)
        else:
            out.append(message)
    out.reverse()
    return out, downscaled


def shed_frames_to_wire_budget(
    messages: Sequence[Message],
    *,
    budget: int,
    reclaimable: Collection[tuple[int, int]] | None = None,
) -> tuple[list[Message], int]:
    """Replace the OLDEST frames with notices until the request fits ``budget``
    bytes, keeping as many recent frames as possible.

    The last-resort transport guard, and the DROP step of
    :func:`fit_frames_to_wire_budget` (which downscales context frames first
    and calls this only when none of its rungs fit). Ordinary compaction
    decides what to keep by *context* value; this decides by whether the
    provider will accept the request at all, which is a different and strictly
    narrower question — so it engages only when the payload is already over a
    ceiling set below the provider's cap, and does nothing at all below it.
    The early return is not an optimisation detail: it is the guarantee that
    every session under budget behaves byte-identically to one without this
    function.

    Built on :func:`prune_stale_frames` rather than as a second
    frame-dropping primitive, so there is ONE definition of "which frames are
    recent" and one set of copy/identity semantics. That also means no message
    is ever removed — user/assistant alternation and tool pairings survive
    untouched, which a transport guard running on arbitrary history must
    guarantee, since it cannot know what is mid-tool-call.

    The loop mirrors ``_shed_stale_turns`` in the evaluation runner's provider
    client: walk the keep count DOWN and stop when a step frees nothing.
    Termination is structural — ``keep`` strictly decreases toward 0, and
    ``keep=0`` drops every image there is that ``reclaimable`` admits — so a
    payload still over budget with no eligible images left exits rather than
    spinning. That residual case is real: a text-only history, or one whose
    remaining images are all protected, cannot be shed here, and the caller
    must be prepared for "still over" rather than assuming success (this is
    what the byte-side auto-continue band checks).

    ``reclaimable`` carries the same protection gate
    :func:`prune_stale_frames` documents; the render fallback passes the
    superseded-capture set so the shed can never drop user uploads, unknown
    images or a stream's latest capture.

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
        candidate, candidate_dropped = prune_stale_frames(
            messages, keep_recent_frames=keep, reclaimable=reclaimable
        )
        if candidate_dropped <= dropped:
            # This step freed nothing new; another turn of the loop cannot
            # either, since the keep count only shrinks.
            break
        working, dropped = candidate, candidate_dropped
        if estimate_wire_bytes(working) <= budget:
            break
    return working, dropped


def fit_frames_to_wire_budget(
    messages: Sequence[Message], *, budget: int, keep_recent_frames: int = 1
) -> tuple[list[Message], int, int, int | None]:
    """Fit the request under ``budget`` by DOWNSCALING context frames first
    and shedding the oldest only as the last resort.

    The byte ladder every wire-history path runs, ordered by how much visual
    history each step costs the user:

    1. Replace every frame older than the newest ``keep_recent_frames``
       frame-bearing messages with a downscaled re-rendering
       (:func:`downscale_stale_frames`), walking
       :data:`~local_operator.imaging.IMAGE_CONTEXT_EDGES` from its widest
       rung and stopping at the first rung that fits. The newest frame keeps
       full fidelity — it is the one the model is reasoning about (and, for
       screen-driving callers, clicking on) — while every older frame keeps
       its place in the conversation as reference. This is deliberately NOT
       ``SESSION_KEEP_RECENT_FRAMES`` (2): that constant governs the
       browser-sweep's DROP rule, where a blanked frame cannot be consulted at
       all and the window therefore has to be wider; a downscale keeps every
       frame, so it protects exactly the one that is a live target.
    2. When no rung fits with the newest protected, walk the SAME ladder again
       with nothing exempt (``keep_recent_frames=0``): the newest frame is
       allowed to descend rather than being exempt from it, so the session
       keeps a degraded view instead of losing the view entirely. It is the
       LAST thing sacrificed — every older frame already sits at the tightest
       rung step 1 reached before this step renders it — and it descends only
       as far as the fit requires: first fit from the widest rung, rendered
       from the original bytes exactly as step 1 does. Without this step the
       protection was absolute at the boundary and the drop path engaged
       first: one 670 KB frame against a 250 KB budget dropped every frame —
       the model saw 0 image blocks — although the frame's own ≤768 rung is
       165 KB and fits, and a six-frame context dropped five frames where
       rendering all six at ≤512 fits in 344 KB. A screen-driving turn went
       blind while a degraded view of the screen was one rung away.
    3. Only when even the tightest unprotected rung is over budget, hand the
       tightest candidate to the existing
       :func:`shed_frames_to_wire_budget`, whose minimal-drop loop stays the
       backstop.

    PROTECTION, the shared-image-compaction contract: steps 1 and 3 degrade
    only SUPERSEDED captures -- the same set the epoch elision consumes
    (:func:`partition_captures`) -- so user uploads, unknown images and every
    stream's latest capture are never resized or dropped by them; a still-over
    return with nothing eligible is the honest disposition rather than silent
    loss. Step 2's descent of the newest frame is the remaining OPEN boundary
    (see the in-body note).

    WHY DOWNSCALE FIRST, stated as the failure it repairs: the shed's steady
    state on a long screenshot session is one MORE frame dropped per turn
    (each new frame costs bytes, so the minimum shed to fit grows by one)
    with every request still pinned at the ceiling — the model loses visual
    context frame by frame while the wire never gets cheaper. Downscaling is
    the fix that failure could not reach: a context frame re-renders to a
    fraction of its bytes at the first rung (measured on a real 1280x720
    code-UI capture: 304 KB → 106 KB), so the whole history fits where only a
    handful of full-size frames did. And because each frame's replacement is
    deterministic and memoized per ``(bytes, rung)``, a frame's bytes change
    at most once per rung it renders at — the turn it stops being the newest
    frame, again only on a rung step in either direction, and under step 2 the turn the
    newest itself first has to descend (at most three changes across the
    ladder) — so the provider's prompt-cache prefix is rewritten at most once
    per rung transition rather than churned on every render. A growing session
    normally steps frames down; the descent entry (step 2) can also step older
    frames one rung UP when it lands on a rung wider than the one step 1 had
    reached — a bounded one-off, not per-turn churn. An episode whose step 1
    fits renders every frame's bytes exactly as it did before step 2 existed:
    step 2 only widens the set the same walk applies to, below that boundary.

    Driven by the SAME ``budget`` the shed was driven by (the number
    ``resolve_wire_bytes_budget`` resolves; the render seam and the
    compaction trigger already agree on it) — deliberately not a second knob,
    and not a second resolver.

    The under-budget guarantee is inherited unchanged: when the request
    already fits, or ``budget`` is non-positive, every element of the input
    is returned by identity and both counts are 0, so a session under budget
    behaves byte-identically to one predating this function.

    Cost, stated because this runs at the render seam: the rung walk decodes
    each stale frame once per ``(frame, rung)`` per PROCESS (memoized in
    ``imaging._CONTEXT_FRAME_CACHE``, exactly like the repair path); later
    renders pay a base64 decode and a digest lookup per block. A session over
    budget is already paying multiples of that in request bytes.

    Returns ``(messages, frames_downscaled, frames_dropped, rung)``, where
    ``rung`` is the long-edge rung every re-rendered frame settled at —
    ``None`` when nothing was downscaled — so the render seam's diagnostic can
    name how aggressive the re-render was. ``frames_downscaled`` counts the
    re-rendered blocks of whichever walk settled the request, so it includes
    the newest frame when step 2 settled it.
    """
    # Imported HERE for the reason ``downscale_stale_frames`` documents: a
    # module-scope import would put ``local_operator.imaging`` on the import
    # closure of the shared renderer an episode imports.
    from local_operator.imaging import IMAGE_CONTEXT_EDGES

    if budget <= 0:
        return list(messages), 0, 0, None
    if estimate_wire_bytes(messages) <= budget:
        return list(messages), 0, 0, None
    # The render fallback obeys the SAME protection rules as the epoch
    # elision: steps 1 and 3 may degrade only SUPERSEDED captures -- user
    # uploads, unknown images and every stream's latest capture are
    # unreachable, so the hard-wire path cannot quietly lose what the epoch
    # guaranteed. The terminal descent (step 2) deliberately keeps the
    # pre-existing last-resort for the newest frame: a provider refusal is
    # worse than a degraded view it alone can prevent, and replacing that
    # with the absolute "latest captures are never resized" contract (plus
    # its typed pre-dispatch transport-limit disposition) is an OPEN
    # boundary with the architect rather than something improvised here.
    partition = partition_captures(messages)
    reclaimable = frozenset(
        (capture.message_index, capture.content_index) for capture in partition.reclaimable
    )
    working: list[Message] = list(messages)
    downscaled_total = 0
    for edge in IMAGE_CONTEXT_EDGES:
        candidate, downscaled = downscale_stale_frames(
            messages,
            keep_recent_frames=keep_recent_frames,
            max_edge=edge,
            reclaimable=reclaimable,
        )
        working, downscaled_total = candidate, downscaled
        if estimate_wire_bytes(working) <= budget:
            return working, downscaled_total, 0, edge
    # Step 2 (see the docstring): the same ladder with the newest frame's
    # exemption removed, reached only when no protected rung fits so the
    # common path never pays for it. Deliberately NOT gated by
    # ``reclaimable`` -- see the open-boundary note above. Run from the
    # ORIGINAL ``messages`` — like
    # every rung above — so a replacement stays the deterministic, memoized
    # function of (bytes, rung) the prompt-cache argument rests on.
    for edge in IMAGE_CONTEXT_EDGES:
        candidate, downscaled = downscale_stale_frames(
            messages, keep_recent_frames=0, max_edge=edge
        )
        working, downscaled_total = candidate, downscaled
        if estimate_wire_bytes(working) <= budget:
            return working, downscaled_total, 0, edge
    working, dropped = shed_frames_to_wire_budget(working, budget=budget, reclaimable=reclaimable)
    return working, downscaled_total, dropped, IMAGE_CONTEXT_EDGES[-1]


@dataclass(frozen=True)
class CaptureElision:
    """One reclaimable image block, addressed in the SOURCE message list.

    Coordinates are source coordinates -- message index in the sequence and
    content index inside that message's ``content`` -- so a commit or a
    replay applies the same projection to the same source without carrying
    copies of the image anywhere.
    """

    message_index: int
    content_index: int
    stream_id: str
    revision_id: str
    producer_scope: str


@dataclass(frozen=True)
class CapturePartition:
    """Which image blocks an epoch may reclaim, and which are untouchable.

    ``reclaimable`` is ordered OLDEST FIRST by canonical arrival, so a
    consumer that must clear a byte band consumes a prefix of it -- the same
    prefix on every run over the same source, which is what makes the
    projection reproducible without a second registry. ``protected`` is
    informational (evidence counts and tests): it lists absent-metadata
    images first, then each stream's latest capture.
    """

    reclaimable: tuple[CaptureElision, ...]
    protected: tuple[tuple[int, int], ...]


def partition_captures(messages: Sequence[Message]) -> CapturePartition:
    """Split image blocks into latest captures vs SUPERSEDED captures.

    WHY. Image bytes dominate transport pressure, but until capture
    provenance existed no image was distinguishable from any other, so the
    only safe move was nothing -- and every image a session ever saw stayed
    on the wire. This is the selection half of the fix: a pure, metadata-only
    reading of the source that names exactly which blocks a later step MAY
    reclaim and leaves everything else PROTECTED.

    Rules, all deliberate:

    - Every image WITHOUT capture provenance is protected: user uploads,
      unknown images and legacy rows. Absence is not "probably a frame";
      the safe reading of an unannotated image is retention.
    - The LATEST capture for each ``(producer_scope, stream_id)`` is
      protected -- the current pixels of every stream survive pressure,
      including streams that have not been updated recently.
    - An older capture is reclaimable IFF a newer capture of the SAME
      ``(producer_scope, stream_id)`` exists. Supersession is strict: another
      producer's claim never supersedes this producer's captures, even with
      identical stream/revision strings.
    - Ordering is canonical arrival (message order), never a revision sort:
      the ids are bounded opaque strings with no ordered meaning, and a
      producer's invented revision numbers must not reorder what history
      says arrived first.

    Pure metadata: no mutation, no copies, no decoding -- the caller decides
    HOW MUCH of ``reclaimable`` to consume against the byte band, and the
    result is deterministic for a given source.
    """
    protected: list[tuple[int, int]] = []
    annotated: list[tuple[tuple[int, int], CapturedFrame]] = []
    for message_index, message in enumerate(messages):
        for content_index, block in enumerate(message.content):
            if not isinstance(block, ImageContent):
                continue
            frame = block.captured_frame
            address = (message_index, content_index)
            if frame is None:
                protected.append(address)
            else:
                annotated.append((address, frame))

    # Latest = highest arrival index per (scope, stream); computed over the
    # annotated subset only, so an unannotated image between two captures
    # cannot become anyone's "latest".
    latest: dict[tuple[str, str], int] = {}
    for index, (_address, frame) in enumerate(annotated):
        latest[(frame.producer_scope, frame.stream_id)] = index

    reclaimable: list[CaptureElision] = []
    for index, (address, frame) in enumerate(annotated):
        if latest[(frame.producer_scope, frame.stream_id)] == index:
            protected.append(address)
            continue
        reclaimable.append(
            CaptureElision(
                message_index=address[0],
                content_index=address[1],
                stream_id=frame.stream_id,
                revision_id=frame.revision_id,
                producer_scope=frame.producer_scope,
            )
        )
    return CapturePartition(reclaimable=tuple(reclaimable), protected=tuple(protected))


def _elide_capture_prefix(
    messages: Sequence[Message],
    candidates: Sequence[CaptureElision],
    *,
    fit_budget: int | None = None,
    limit: int | None = None,
) -> tuple[list[Message], int]:
    """Apply a deterministic prefix of ``candidates``; ONE elision algorithm.

    Shared by the byte-band rung (:func:`reclaim_captures_to_wire_budget`)
    and the replay watermark (:func:`apply_capture_elisions`) so projection
    and replay can never drift into two opinions about what an epoch elided.

    - ``fit_budget``: stop as soon as the estimate is at or below it.
    - ``limit``: apply exactly this many eligible elisions (deterministic
      skips do not count toward it), for replaying a persisted frontier.
    - Neither: apply every eligible elision.

    ``candidates`` must come from :func:`partition_captures` over the SAME
    ``messages``; addresses resolve through the source block objects by
    identity, and the never-empty guard skips rather than ever producing an
    empty content list. Sources are never mutated: only elided messages are
    rebuilt as copies.
    """
    working = list(messages)
    contents: dict[int, list[Content]] = {}
    elided = 0
    for candidate in candidates:
        if limit is not None and elided >= limit:
            break
        message_index = candidate.message_index
        content = contents.get(message_index)
        if content is None:
            content = list(messages[message_index].content)
            contents[message_index] = content
        # Identity, not equality: two captures may compare equal by VALUE, and
        # earlier elisions from the same message shift positions -- so the
        # SOURCE address is resolved through the original block object.
        block = messages[message_index].content[candidate.content_index]
        if not any(item is block for item in content):
            continue
        if len(content) <= 1:
            # Never-empty guard (see :func:`reclaim_captures_to_wire_budget`);
            # deterministic skip.
            continue
        content[:] = [item for item in content if item is not block]
        working[message_index] = messages[message_index].model_copy(
            update={"content": list(content)}
        )
        elided += 1
        if fit_budget is not None and estimate_wire_bytes(working) <= fit_budget:
            break
    return working, elided


def apply_capture_elisions(messages: Sequence[Message], *, limit: int) -> tuple[list[Message], int]:
    """Reapply a persisted elision frontier to ``messages`` (replay side).

    The compaction record persists HOW MANY eligible superseded captures an
    epoch elided; replay re-derives the partition from the same source and
    applies the same number -- same candidates, same never-empty skips,
    same identity resolution -- so the projected history a resumed session
    replays is the one the live session sent, without storing image copies
    or a second registry. ``limit`` beyond the eligible count applies all of
    them; ``limit <= 0`` returns the input list untouched.
    """
    if limit <= 0:
        return list(messages), 0
    candidates = partition_captures(messages).reclaimable
    if not candidates:
        return list(messages), 0
    return _elide_capture_prefix(messages, candidates, limit=limit)


def reclaim_captures_to_wire_budget(
    messages: Sequence[Message], *, budget: int
) -> tuple[list[Message], int]:
    """Elide SUPERSEDED captures, oldest first, until the request fits ``budget``.

    The FIRST recovery rung for image byte pressure, and deliberately the
    narrowest one: it consumes only a prefix of :func:`partition_captures`'
    ``reclaimable`` list -- captures a producer superseded, carrying harness
    provenance -- and never touches unannotated images, any stream's LATEST
    capture, text blocks, or tool pairing. General frame shedding
    (:func:`shed_frames_to_wire_budget`) stays a separate, later rung for the
    pressure this cannot clear. The caller derives ``budget`` (the byte
    ceiling to land at-or-below) from the same wire trigger/recovery band the
    auto-continue gate uses (``cleared_wire_headroom`` semantics), so both
    sides measure with ONE ruler.

    Behaviour mirrors the shed rung in the ways that matter:

    - Early return: at or under ``budget`` nothing is recomputed and the
      input list is returned, so every session under budget behaves
      byte-identically to one without this function.
    - Oldest first, deterministic: elisions follow the partition's
      canonical-arrival order, so a replay of the same source consumes the
      same prefix -- the property the replay watermark depends on.
    - A block is never elided if it is its message's ONLY remaining block:
      an empty content list breaks wire and transcript invariants, and the
      skip is itself deterministic (the next-oldest eligible elision is
      tried). If the eligible elisions cannot clear the budget the function
      returns without pretending: the caller re-measures and records "no
      eligible relief" rather than a fabricated success.

    Elision REMOVES the block from the projected copy; the source message
    objects stay untouched, which is what lets a later replay rebuild the
    exact same projection from the source plus the persisted frontier.

    Returns ``(messages, elided)``; ``elided`` is 0 and the input list is
    returned when nothing was over budget or nothing eligible could run.
    """
    if budget <= 0:
        return list(messages), 0
    if estimate_wire_bytes(messages) <= budget:
        return list(messages), 0

    candidates = partition_captures(messages).reclaimable
    if not candidates:
        return list(messages), 0
    return _elide_capture_prefix(messages, candidates, fit_budget=budget)


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
    superseded AND useless passes alike): never errors, never skill/guide reads.

    Skill and guide reads are exempt because a pruned one gets re-read in a
    loop — guides carry MUST-read rules whose loss is the same failure as a
    lost skill, and Fix A now stamps ``guide://`` on filesystem guide reads
    too, so exempting only ``skill://`` would leave exactly those new shaped
    results blankable. Protection matches ``["skill", isSkillReadToolResult]``:
    the ``skill`` tool itself, or a ``read`` result whose recorded target is a
    ``skill://``/``guide://`` URL — the loop stores internal-URL reads under
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
            if isinstance(value, str) and value.startswith(("skill://", "guide://")):
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

    Guards, in order per candidate: never errors, never skill/guide reads,
    never already-pruned, never below :data:`MIN_PRUNE_TOKENS`, and never
    inside the warm cache suffix — unless the idle flush window has elapsed,
    or (superseded pass only) the victim is at least as large as the suffix
    it invalidates and at least :data:`OVERSIZED_SUPERSEDE_TOKENS`.
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
    # WHY the two passes do not share the oversized exception (R2): the
    # exception's arithmetic is supersede-specific — a superseded victim is
    # dead weight that would otherwise be re-billed verbatim on every
    # request, so a victim outweighing its suffix repays the one re-write
    # immediately. A useless-flagged result (zero-match search, timed-out
    # wait) is worthless once consumed by definition; blanking it inside the
    # warm suffix buys no such repayment, so it keeps the plain guard.
    for indices, notice, is_superseded_pass in (
        (superseded, SUPERSEDED_NOTICE, True),
        (useless, USELESS_NOTICE, False),
    ):
        for i in indices:
            message = messages[i]
            victim_tokens = estimate_tokens(message)
            if victim_tokens < MIN_PRUNE_TOKENS:
                continue
            if not idle and suffix_tokens[i] > cache_warm_suffix_tokens:
                # Inside the warm cache prefix: re-writing it costs the
                # cacheWrite premium on the whole suffix. Leave it for
                # compaction, which rebuilds the cache anyway — UNLESS this
                # is a superseded victim that alone outweighs that suffix
                # (see OVERSIZED_SUPERSEDE_TOKENS): then the re-write is
                # repaid by not re-billing the victim on the very next
                # request. Both halves are required: the floor keeps the
                # exception off victims the guard calls small, and the
                # suffix comparison is the repayment invariant itself.
                oversized_supersede = (
                    is_superseded_pass
                    and victim_tokens >= OVERSIZED_SUPERSEDE_TOKENS
                    and victim_tokens >= suffix_tokens[i]
                )
                if not oversized_supersede:
                    continue
            _blank(message, notice)
            changed = True

    return list(messages), changed


def _blank_frames(message: Message) -> int:
    """Fold ``message``'s image blocks into :data:`STALE_FRAME_NOTICE`.

    In place, like :func:`prune_tool_outputs` blanks tool outputs — the session
    owns its messages, and its prune journal persists the blanking by the
    ``pruned`` marker afterwards. Text survives (a frame's caption names the
    file it came from, so the model can re-read pixels that are gone), and
    images fold to ONE notice: a message that carried three views of a surface
    reads as "a screenshot was here", not as three lines of the same
    boilerplate. The whole content collapses to a single text block because
    the journal replays ``message.text`` as one block (``append_prune`` →
    ``_pruned_entry``); keeping the original text blocks and appending a notice
    would make a resumed context differ in block structure from the live one.
    Returns the number of image blocks replaced.
    """
    dropped = sum(1 for block in message.content if isinstance(block, ImageContent))
    text = "".join(
        block.text for block in message.content if isinstance(block, TextContent)
    ).strip()
    folded = f"{text}\n{STALE_FRAME_NOTICE}" if text else STALE_FRAME_NOTICE
    message.content = [TextContent(text=folded)]
    message.provider_payload = {**_payload_of(message), "pruned": True}
    invalidate_message_cache(message)
    return dropped


def prune_stale_frames_in_place(
    messages: Sequence[Message],
    *,
    now_ms: int,
    last_activity_ms: int,
    keep_recent_frames: int = SESSION_KEEP_RECENT_FRAMES,
    cache_warm_suffix_tokens: int = 8000,
    idle_flush_ms: int = 5400000,
) -> int:
    """Blank stale frames in place — the session-side sibling of
    :func:`prune_stale_frames`.

    Reuses that function's retention rule — everything outside the newest
    ``keep_recent_frames`` frame-bearing messages is superseded by a newer
    view — and applies the same caller guards :func:`prune_tool_outputs`
    applies, so the session's two in-place sweeps behave alike:

    - never errors and never skill/guide reads (:func:`_is_prunable`), never an
      already-pruned result, and never below :data:`MIN_PRUNE_TOKENS`;
    - never a candidate whose suffix lies inside the warm prompt cache
      (``cache_warm_suffix_tokens``) — mutating there re-writes everything
      after it at cacheWrite price — *unless* the idle flush
      (``idle_flush_ms`` since the last provider request) proves the cache
      cold.

    Two deliberate narrowings vs the runner's primitive, each the reason this
    sibling exists. Only TOOL-result frames are candidates: the runner's
    history has no user-authored frames, while a session does, and blanking a
    frame a human pasted would be a visible loss of the user's own message —
    markers are not candidates either (user-role by construction; their
    frames ARE a snapcompact archive, and mutating one would corrupt the
    replay it exists to carry). And the blanking is in place with a
    ``pruned`` marker, exactly like :func:`prune_tool_outputs`, because the
    session owns its messages and journals the blanking; the runner instead
    relies on :func:`prune_stale_frames`' copy/identity contract.

    ``now_ms`` and ``last_activity_ms`` are MILLISECONDS epoch values,
    matching :func:`prune_tool_outputs` (passing seconds silently disables
    the idle flush). Returns the number of image blocks replaced.
    """
    if not messages:
        return 0
    # Victim selection IS the module's retention rule, read off the primitive
    # so "recent" has one definition: the copy walk returns victims as the
    # entries whose identity changed.
    after, _ = prune_stale_frames(messages, keep_recent_frames=keep_recent_frames)
    victims = [index for index in range(len(messages)) if after[index] is not messages[index]]
    if not victims:
        return 0
    idle = (now_ms - last_activity_ms) >= idle_flush_ms
    suffix_tokens = compute_suffix_tokens(messages)
    dropped = 0
    for index in victims:
        message = messages[index]
        # User frames and markers are never candidates (see the docstring);
        # the marker check is belt-and-braces — markers are user-role by
        # construction — and is what keeps a future renderer that moves them
        # out of that class from letting an archive be corrupted here.
        if message.role != "tool" or marker_exists([message]):
            continue
        if not _is_prunable(message) or _is_pruned(message):
            continue
        if estimate_tokens(message) < MIN_PRUNE_TOKENS:
            continue
        if not idle and suffix_tokens[index] > cache_warm_suffix_tokens:
            # Inside the warm cache prefix: re-writing it costs the
            # cacheWrite premium on everything after it. Leave the frame for
            # the idle flush, or for compaction, which cuts it away.
            continue
        dropped += _blank_frames(message)
    return dropped


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
