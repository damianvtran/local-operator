"""One host-independent compaction pass: prune → trigger → cut → summarize → rebuild.

The session's pass (``Session._plan_compaction`` / ``_finish_compaction``) is
entangled with its transcript, event bus and UI, so a second host that wants
the same engine — the evaluation runner, whose history is a sent message
prefix rather than a persisted transcript — cannot call it. What it CAN share
is every decision the pass makes, and those are what this module composes,
in the same order and through the same single resolvers the session uses:

1. :func:`prune_tool_outputs` — blank superseded/useless tool results.
2. :func:`prune_stale_frames` — only when ``settings.keep_recent_frames`` is
   set; ``None`` (the ordinary-session default) skips this step entirely, so
   a pass with default settings is byte-identical to one that predates it.
3. :func:`compaction_context_tokens` + :func:`should_compact` — the trigger,
   through the one resolver in ``thresholds`` (never a mirrored formula).
4. :func:`find_cut_point` — the same pairing-safe cut the session takes.
5. :func:`resolve_strategy` — snapcompact archives locally with NO provider
   call; context-full asks the injected ``summarize`` callable.
6. Rebuild as ``[marker, *kept]`` where the marker is the same
   ``compaction_summary`` entry the session commits (:mod:`.marker`).

Import isolation: this module must not import ``session``, ``model``,
``providers`` or ``config``. Its callers include the evaluation runner, which
asserts that boundary in a subprocess probe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Sequence

from local_operator.harness.types import AgentMessage, CustomMessage, Message, ModelSpec

from .api import summarize_messages
from .cutpoint import find_cut_point
from .marker import (
    COMPACTION_MARKER_TYPE,
    build_compaction_marker,
    render_compaction_marker,
)
from .pruning import prune_stale_frames, prune_tool_outputs
from .thresholds import (
    CompactionSettings,
    compaction_context_tokens,
    resolve_strategy,
    should_compact,
)
from .tokens import estimate_messages_tokens

logger = logging.getLogger(__name__)

__all__ = [
    "CompactionPassResult",
    "RefusalReason",
    "Summarizer",
    "run_compaction_pass",
]

#: Why a pass did not rebuild. ``below-threshold`` and
#: ``nothing-to-summarize`` are the everyday answers; the other two are
#: configuration or provider outcomes the caller may want to surface.
RefusalReason = Literal[
    "disabled",
    "below-threshold",
    "nothing-to-summarize",
    "summarizer-failed",
]

#: ``summarize(prompt) -> summary_text``. The host owns the provider call
#: (credentials, route, billing), which is why the pass takes a callable
#: rather than a stream function: it must not know how a summary is bought.
Summarizer = Callable[[str], Awaitable[str]]


@dataclass(frozen=True)
class CompactionPassResult:
    """What one pass did.

    ``messages`` is ALWAYS the list to use next: when the pass refused, it is
    the pruned input (pruning is a side benefit the session also keeps on a
    refused pass); when it ran, it is the rebuilt ``[marker, *kept]``.
    ``ran`` distinguishes the two; ``reason`` names the refusal.

    ``pruned`` reports whether the tool-output prune blanked anything and
    ``frames_dropped`` how many image blocks the frame prune replaced — both
    are non-zero on a refused pass too, so a caller that rebuilds its prefix
    on a frame budget learns the frames went even when no summary was bought.
    """

    messages: list[Message]
    ran: bool
    reason: RefusalReason | None
    strategy: Literal["context-full", "snapcompact"] | None
    tokens_before: int
    tokens_after: int
    pruned: bool
    frames_dropped: int
    preserve_data: dict[str, Any] | None = None
    summary_text: str | None = None


async def run_compaction_pass(
    messages: Sequence[Message],
    *,
    model: ModelSpec,
    settings: CompactionSettings,
    summarize: Summarizer | None,
    now_ms: int,
    last_activity_ms: int,
    provider_context_tokens: int | None = None,
    respect_threshold: bool = True,
) -> CompactionPassResult:
    """Run one pass over ``messages`` and return the list to send next.

    ``messages`` are LLM-visible messages (the session passes its rendered
    history; the runner its sent prefix). A previous pass's marker is already
    a rendered user message here, and it is recognised by
    :data:`.marker.COMPACTION_MARKER_TYPE` on ``provider_payload`` so the cut
    never re-summarises a summary.

    ``provider_context_tokens`` is the provider's last reported context size
    when the host has one; the trigger takes the larger of it and the local
    estimate (:func:`compaction_context_tokens`). ``respect_threshold=False``
    is the manual ``/compact`` posture: prune, then summarise whatever the cut
    finds regardless of size.

    ``summarize`` may be ``None`` only when the resolved strategy is
    snapcompact (which archives locally); a context-full pass without a
    summarizer refuses as ``summarizer-failed`` rather than inventing one.
    """
    working: list[Message] = list(messages)
    if not settings.enabled or settings.strategy == "off":
        return _refused(working, "disabled", pruned=False, frames_dropped=0)

    # (1) Tool-output prune, in place, exactly as the session does before it
    # decides anything: a context the prune alone brought under the line
    # never buys a summary.
    working, pruned = prune_tool_outputs(working, now_ms, last_activity_ms)

    # (2) Frame prune — opt-in per surface. ``None`` skips the call itself so
    # the default path performs no work here at all (see the setting).
    frames_dropped = 0
    if settings.keep_recent_frames is not None:
        working, frames_dropped = prune_stale_frames(
            working, keep_recent_frames=settings.keep_recent_frames
        )

    # (3) Trigger through the single resolver.
    local_estimate = estimate_messages_tokens(working)
    tokens_before = compaction_context_tokens(provider_context_tokens, local_estimate)
    if respect_threshold and not should_compact(tokens_before, model.context_window, settings):
        return _refused(
            working,
            "below-threshold",
            pruned=pruned,
            frames_dropped=frames_dropped,
            tokens=tokens_before,
        )

    # (4) Cut. The cut-point walker understands the marker as a
    # ``CustomMessage``; a rendered marker is a plain user message here, so
    # lift it back for the walk and index the same positions.
    cut = find_cut_point(_lift_markers(working), settings.keep_recent_tokens)
    if cut is None:
        return _refused(
            working,
            "nothing-to-summarize",
            pruned=pruned,
            frames_dropped=frames_dropped,
            tokens=tokens_before,
        )
    to_summarize = working[:cut]
    kept = working[cut:]

    # (5) Summarise per strategy.
    strategy = resolve_strategy(settings, model)
    preserve_data: dict[str, Any] | None = None
    summary: str
    if strategy == "snapcompact":
        try:
            from . import snapcompact

            archive = snapcompact.compact_to_archive(
                to_summarize,
                model.provider,
                model.model_id,
                _previous_archive_text(to_summarize),
                context_window=model.context_window,
            )
            summary = snapcompact.archive_summary(archive, model.provider, model.model_id) or " "
            preserve_data = {"snapcompact": archive.model_dump(mode="json")}
        except Exception:
            # Same degrade the session takes: a snapcompact failure is not a
            # reason to leave the context full when a text summary can be had.
            logger.warning("snapcompact failed; falling back to context-full", exc_info=True)
            strategy = "context-full"
    if strategy == "context-full":
        if summarize is None:
            return _refused(
                working,
                "summarizer-failed",
                pruned=pruned,
                frames_dropped=frames_dropped,
                tokens=tokens_before,
            )
        try:
            # The system half of the call (``SUMMARIZATION_SYSTEM_PROMPT``) is
            # the host's to send; the pass hands over only the rendered
            # prompt. A prior marker inside ``to_summarize`` is lifted so it
            # serializes as a labelled previous summary rather than as a user
            # turn, which is what the session's summarizer also sees.
            summary = await summarize_messages(
                _lift_markers(to_summarize),
                lambda _system, prompt: summarize(prompt),
            )
        except Exception:
            logger.warning("compaction summarizer failed", exc_info=True)
            return _refused(
                working,
                "summarizer-failed",
                pruned=pruned,
                frames_dropped=frames_dropped,
                tokens=tokens_before,
            )

    # (6) Rebuild. The marker is rendered the way the session renders it on
    # every request, and it is tagged on ``provider_payload`` (harness
    # bookkeeping the wire builders never ship) so the next pass can lift it
    # back into the ``CustomMessage`` the cut-point walker understands.
    marker = build_compaction_marker(summary, preserve_data)
    rendered = render_compaction_marker(marker, entry_id=marker.id)
    rendered.provider_payload = {
        COMPACTION_MARKER_TYPE: {"summary": summary, "preserve_data": preserve_data}
    }
    rebuilt: list[Message] = [rendered, *kept]
    tokens_after = estimate_messages_tokens(rebuilt)
    return CompactionPassResult(
        messages=rebuilt,
        ran=True,
        reason=None,
        strategy=strategy,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        pruned=pruned,
        frames_dropped=frames_dropped,
        preserve_data=preserve_data,
        summary_text=summary,
    )


def _refused(
    messages: list[Message],
    reason: RefusalReason,
    *,
    pruned: bool,
    frames_dropped: int,
    tokens: int | None = None,
) -> CompactionPassResult:
    # ``tokens`` is measured lazily: a disabled pass must not pay for a
    # tokenizer load to report a number nobody asked for.
    measured = estimate_messages_tokens(messages) if tokens is None else tokens
    return CompactionPassResult(
        messages=messages,
        ran=False,
        reason=reason,
        strategy=None,
        tokens_before=measured,
        tokens_after=measured,
        pruned=pruned,
        frames_dropped=frames_dropped,
    )


def _marker_details(message: Message) -> dict[str, Any] | None:
    payload = message.provider_payload
    if not isinstance(payload, dict):
        return None
    details = payload.get(COMPACTION_MARKER_TYPE)
    return details if isinstance(details, dict) else None


def _lift_markers(messages: Sequence[Message]) -> list[AgentMessage]:
    """The same positions, with rendered markers lifted back to ``CustomMessage``.

    Only the cut-point walker and the summarizer's serializer need the lifted
    form; the sent list keeps the rendered messages. Positions are preserved
    one-to-one so an index into the lifted list indexes the working list.
    """
    lifted: list[AgentMessage] = []
    for message in messages:
        details = _marker_details(message)
        if details is None:
            lifted.append(message)
            continue
        lifted.append(
            CustomMessage(
                custom_type=COMPACTION_MARKER_TYPE,
                attribution="system",
                details={
                    "summary": details.get("summary", ""),
                    **(
                        {"preserve_data": details["preserve_data"]}
                        if details.get("preserve_data") is not None
                        else {}
                    ),
                },
                id=message.id,
            )
        )
    return lifted


def _previous_archive_text(to_summarize: Sequence[Message]) -> str | None:
    """The newest archive's accumulated text, so snapcompact re-renders from
    accumulated history instead of carrying old PNGs forward (mirrors
    ``Session._previous_archive_text``)."""
    for message in reversed(to_summarize):
        details = _marker_details(message)
        if details is None:
            continue
        preserve = details.get("preserve_data") or {}
        snap = preserve.get("snapcompact") if isinstance(preserve, dict) else None
        if isinstance(snap, dict) and snap.get("text"):
            return str(snap["text"])
        summary = details.get("summary")
        return str(summary) if summary else None
    return None
