"""Cut-point selection for history compaction.

The hardest correctness rule in compaction: **never cut at a tool result, and
never cut at an assistant message whose tool-call results would follow the
cut.** Either mistake orphans a tool call/result pair and every provider
rejects (or silently corrupts) the conversation. The rule is enforced both by
the candidate predicate and by a final assertion — as an assertion, not a
comment, so the invariant is enforced in code rather than documented.

Algorithm (``findCutPoint``): walk **backwards** from the newest message
accumulating estimated tokens until the kept region reaches
``keep_recent_tokens``, then snap **forward** to the first valid cut message
at or after that index. Valid cut messages are ``user`` messages, assistant
messages with no pending tool calls, or compaction-summary markers.
"""

from __future__ import annotations

from typing import Sequence

from local_operator.harness.types import AgentMessage, CustomMessage, Message

from .tokens import _encode_len, estimate_tokens

__all__ = ["find_cut_point", "prepare_partitions"]


def _is_compaction_marker(message: AgentMessage) -> bool:
    """True for the custom entry that replays a prior compaction summary."""
    return isinstance(message, CustomMessage) and message.custom_type == "compaction_summary"


def _message_tokens(message: AgentMessage) -> int:
    """Token estimate for either message kind.

    ``CustomMessage`` payloads vary by ``custom_type``; compaction markers are
    counted straight from their summary text (the replayed content) via the
    raw encoder — no throwaway ``Message`` wrapper. Anything else is
    conservatively free — custom entries are small by design.
    """
    if isinstance(message, Message):
        return estimate_tokens(message)
    if message.custom_type == "compaction_summary":
        summary = message.details.get("summary", "")
        if isinstance(summary, str) and summary:
            return _encode_len(summary)
    return 0


def _is_valid_cut(messages: Sequence[AgentMessage], index: int) -> bool:
    """Whether cutting *before* ``messages[index]`` keeps pairing legal.

    Valid: user messages, assistant messages with no pending tool calls,
    compaction-summary markers. Tool results and assistants whose results
    follow are never valid — see module docstring.
    """
    message = messages[index]
    if _is_compaction_marker(message):
        return True
    if not isinstance(message, Message):
        return False
    if message.role == "user":
        return True
    if message.role != "assistant" or message.tool_calls:
        return False
    return True


def find_cut_point(messages: Sequence[AgentMessage], keep_recent_tokens: int) -> int | None:
    """Index of the first KEPT message, or ``None`` when nothing is worth
    summarizing.

    Walks backwards accumulating :func:`estimate_tokens` until the kept region
    reaches ``keep_recent_tokens``, then snaps forward to the next valid cut
    message (role ``user``, or ``assistant`` without pending tool calls, or a
    compaction-summary marker). Returns ``None`` when the accumulated region
    already covers everything (all tokens in the kept region) or when fewer
    than two REAL messages fall before the cut (nothing worth summarizing —
    a previous compaction's marker does not count, it is already a summary).

    That ``None`` is also the answer an on-demand compaction gets when there is
    nothing to do, so it is a decision a host reports, not just an internal
    short-circuit: see ``Session.compact_now``.

    HARD RULE: the returned index never points at a tool-role message or at an
    assistant message whose tool-call results follow it — orphaned tool calls
    break every provider.
    """
    total = len(messages)
    if total == 0 or keep_recent_tokens <= 0:
        return None

    # Backwards walk: accumulate from the newest message until the kept
    # region is large enough. The index where the walk stops is the first
    # cut candidate.
    def backwards_walk(start: int) -> int | None:
        accumulated = 0
        for i in range(start, -1, -1):
            accumulated += _message_tokens(messages[i])
            if accumulated >= keep_recent_tokens:
                return i
        return None

    index = backwards_walk(total - 1)
    if index is None:
        # Never reached the keep budget: everything is "recent".
        return None

    # Snap forward to the first valid cut message. Skipping tool results and
    # pending-tool-call assistants keeps call/result pairing intact: the pair
    # always moves to the kept side together.
    while index < total and not _is_valid_cut(messages, index):
        index += 1
    if index >= total:
        return None

    # The snap can collapse the kept region far below the budget — one user
    # message followed by a long tool chain walks into the chain and snaps to
    # the next user message, keeping a few hundred tokens instead of
    # keep_recent_tokens. When that happens, retry the walk from BEFORE the
    # chain so the cut lands ahead of it and the recent working context the
    # setting protects survives.
    kept_tokens = sum(_message_tokens(m) for m in messages[index:])
    if kept_tokens < keep_recent_tokens // 2 and index > 0:
        retry = backwards_walk(index - 1)
        if retry is not None:
            index = retry
            while index < total and not _is_valid_cut(messages, index):
                index += 1
            if index >= total:
                return None

    # Defensive invariant (the predicate already excludes violations; the
    # assertion makes a future regression loud instead of silent). GENERAL
    # form (RC-17): collect the tool_call_ids answered AFTER the cut, and
    # assert none is issued by a SUMMARIZED message — covers assistant cuts
    # and user cuts alike (a cut inside a call/result cluster would strand the
    # result on the kept side with its call summarized away).
    answered_after = {
        m.tool_call_id
        for m in messages[index:]
        if isinstance(m, Message) and m.role == "tool" and m.tool_call_id
    }
    for summarized in messages[:index]:
        if isinstance(summarized, Message) and summarized.tool_calls:
            orphaned = [c.id for c in summarized.tool_calls if c.id in answered_after]
            assert not orphaned, (
                f"cut point {index} orphans tool calls {orphaned}: their results "
                "stay kept while the issuing calls are summarized away, and every "
                "provider rejects that"
            )

    # Summarizing zero or one message is not worth the cache rewrite. The
    # ORIGINAL rule, unchanged and absolute: one message before the cut buys a
    # provider call and a full prompt-cache rewrite for one message's worth of
    # headroom.
    if index <= 1:
        return None

    # Past that, count over REAL messages: a previous compaction's marker is
    # not history to summarize, it is a summary already, so ``[marker, older,
    # recent…]`` would otherwise re-compress what was just compressed.
    #
    # That exclusion is the only rule here STRICTER than ``index <= 1``, and it
    # answers in messages a question that is really about TOKENS. ``[marker, X,
    # …]`` cut at index 2 leaves one summarizable message and the old rule ran
    # it; when X is a 50k tool result it is the whole reason the window is
    # full, and refusing leaves the AUTOMATIC trigger nothing to do while the
    # context keeps growing — a pass blocked forever by arithmetic meant to
    # skip trivial ones. So a lone message still counts when it outweighs the
    # entire recency budget the caller asked to protect: at that size it is not
    # a trivial rewrite, it is the problem.
    summarizable = [m for m in messages[:index] if not _is_compaction_marker(m)]
    if not summarizable:
        return None
    if len(summarizable) == 1 and _message_tokens(summarizable[0]) < keep_recent_tokens:
        return None
    return index


def prepare_partitions(
    messages: Sequence[AgentMessage], cut: int
) -> tuple[list[AgentMessage], list[AgentMessage]]:
    """Split ``messages`` at ``cut`` into ``(to_summarize, kept)``.

    ``cut`` must be a value returned by :func:`find_cut_point` (an index into
    ``messages`` pointing at the first kept message).
    """
    if not 0 < cut <= len(messages):
        raise ValueError(f"invalid cut point {cut} for {len(messages)} messages")
    return list(messages[:cut]), list(messages[cut:])
