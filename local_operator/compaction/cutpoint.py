"""Cut-point selection for history compaction.

The hardest correctness rule in compaction: **never cut at a tool result, and
never cut at an assistant message whose tool-call results would follow the
cut.** Either mistake orphans a tool call/result pair and every provider
rejects (or silently corrupts) the conversation. The rule is enforced both by
the candidate predicate and by a final assertion — as an assertion, not a
comment, so the invariant is enforced in code rather than documented.

Algorithm (``findCutPoint``): walk **backwards** from the newest message
accumulating estimated tokens until the kept region reaches
``keep_recent_tokens``, then snap to the nearest valid cut message — forward
from that index when one exists (a later cut keeps more recent history), and
otherwise backwards. Valid cut messages are ``user`` messages, assistant
messages with no pending tool calls, or compaction-summary markers.

The backwards fallback exists because a history captured MID-RUN ends inside
an unfinished tool chain, where every trailing position is an illegal cut; a
forward-only snap ran off the end and reported "nothing to compact" at exactly
the moment the context was growing fastest. See :func:`_snap_to_valid_cut`.
"""

from __future__ import annotations

from typing import Sequence

from local_operator.harness.types import AgentMessage, CustomMessage, Message

from .tokens import _encode_len, estimate_tokens

__all__ = [
    "find_cut_point",
    "prepare_partitions",
    "extract_preserved_user_turns",
    "task_boundary_floor",
]


def _is_compaction_marker(message: AgentMessage) -> bool:
    """True for the custom entry that replays a prior compaction summary."""
    return isinstance(message, CustomMessage) and message.custom_type == "compaction_summary"


#: Marks a user turn re-injected VERBATIM by a prior compaction pass (see
#: ``Session._run_compaction``). It rides ``provider_payload`` — harness
#: bookkeeping the wire builders never ship as content — and persists through
#: the transcript, so it survives a resume the same way the ``pruned`` flag
#: does.
PRESERVED_USER_TURN_KEY = "compaction_preserved"


def _is_preserved_user_turn(message: AgentMessage) -> bool:
    """True for a user turn a prior pass already preserved verbatim.

    Such a turn is carried-forward, already-compacted context — not new
    history — so :func:`find_cut_point` must not count it as "worth
    summarizing", exactly as it excludes the marker. Without this, pressing
    /compact twice in a row re-fires on nothing but the marker and the
    preserved turns, spending a pass for zero headroom (and the preserved
    turns' own bytes make the second pass look like it has new work).
    """
    return (
        isinstance(message, Message)
        and bool(message.provider_payload)
        and bool(message.provider_payload.get(PRESERVED_USER_TURN_KEY))
    )


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


def _result_indices(messages: Sequence[AgentMessage]) -> dict[str, int]:
    """``tool_call_id`` -> index of the tool result answering it.

    Precomputed once per :func:`find_cut_point` so the validity predicate is
    O(1) per candidate instead of rescanning the suffix; the snap can inspect
    every index, which made the naive form quadratic on exactly the long tool
    runs this code exists to relieve.
    """
    return {
        message.tool_call_id: index
        for index, message in enumerate(messages)
        if isinstance(message, Message) and message.role == "tool" and message.tool_call_id
    }


def _is_valid_cut(
    messages: Sequence[AgentMessage], index: int, result_at: dict[str, int] | None = None
) -> bool:
    """Whether cutting *before* ``messages[index]`` keeps pairing legal.

    Valid: user messages, compaction-summary markers, assistant messages with
    no tool calls, and an assistant message whose OWN calls are all answered at
    or after ``index`` — that last case keeps the assistant and its results
    together on the kept side, which is the actual pairing rule.

    Treating every tool-calling assistant as invalid (the earlier rule) was
    stricter than the invariant and starved compaction: inside a long tool run
    every message is either a tool result or a tool-calling assistant, so NO
    index qualified and the whole run was uncompactable no matter how large it
    grew. The partition sweep already encodes the looser rule — it asserts an
    assistant candidate's pending calls are answered in ``kept`` rather than
    that no such candidate exists.

    An UNANSWERED call is still invalid: its result has not been produced yet,
    so keeping the assistant would hand the provider a dangling call.
    """
    message = messages[index]
    if _is_compaction_marker(message):
        return True
    if not isinstance(message, Message):
        return False
    if message.role == "user":
        return True
    if message.role != "assistant":
        return False
    if not message.tool_calls:
        return True
    if result_at is None:
        result_at = _result_indices(messages)
    return all(result_at.get(call.id, -1) >= index for call in message.tool_calls)


def _snap_to_valid_cut(
    messages: Sequence[AgentMessage], index: int, result_at: dict[str, int]
) -> int | None:
    """Nearest index at or after ``index`` that is a legal cut, else the
    nearest one BEFORE it; ``None`` when the history has no legal cut at all.

    Forward is preferred because a later cut keeps more recent history, which
    is what ``keep_recent_tokens`` is asking for. The backwards fallback is
    what makes the gate work MID-RUN, and it is not symmetric politeness — it
    fixes a real starvation:

    An unfinished tool run ends in an assistant message with pending tool
    calls, usually followed by its tool results. Every one of those trailing
    positions fails ``_is_valid_cut``, so a forward-only snap walks off the
    end of the list and the caller reads that as "nothing to compact". A long
    tool run is precisely when the context is growing fastest and no user turn
    is coming to relieve it, so compaction was refused at every mid-turn
    boundary and the session sailed past its configured threshold — relief
    arrived only once the run finally ended and a terminal assistant message
    made a forward cut legal again.

    Cutting slightly EARLIER than the token walk asked for is the right trade:
    it keeps a little more history than requested (never less), preserves the
    call/result pairing rule exactly as the forward snap does, and lets a pass
    actually run. The alternative — refusing — is what let the window fill.
    """
    total = len(messages)
    forward = index
    while forward < total and not _is_valid_cut(messages, forward, result_at):
        forward += 1
    if forward < total:
        return forward
    backward = min(index, total - 1)
    while backward >= 0 and not _is_valid_cut(messages, backward, result_at):
        backward -= 1
    return backward if backward >= 0 else None


def find_cut_point(messages: Sequence[AgentMessage], keep_recent_tokens: int) -> int | None:
    """Index of the first KEPT message, or ``None`` when nothing is worth
    summarizing.

    Walks backwards accumulating :func:`estimate_tokens` until the kept region
    reaches ``keep_recent_tokens``, then snaps to the nearest valid cut
    message (role ``user``, or ``assistant`` without pending tool calls, or a
    compaction-summary marker) — forward when possible, else backwards, see
    :func:`_snap_to_valid_cut`. Returns ``None`` when the accumulated region
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

    result_at = _result_indices(messages)

    # Snap to the nearest valid cut message. Skipping tool results and
    # pending-tool-call assistants keeps call/result pairing intact: the pair
    # always moves to the kept side together.
    snapped = _snap_to_valid_cut(messages, index, result_at)
    if snapped is None:
        return None
    index = snapped

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
            snapped = _snap_to_valid_cut(messages, retry, result_at)
            if snapped is None:
                return None
            index = snapped

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
    summarizable = [
        m
        for m in messages[:index]
        if not _is_compaction_marker(m) and not _is_preserved_user_turn(m)
    ]
    if not summarizable:
        return None
    if len(summarizable) == 1 and _message_tokens(summarizable[0]) < keep_recent_tokens:
        return None
    return index


def task_boundary_floor(
    messages: Sequence[AgentMessage],
    genuine_user_ids: set[str] | None = None,
    *,
    cap: int,
) -> int:
    """Estimated tokens from the last GENUINE user turn to the end, capped.

    ``find_cut_point`` is recency-shaped ("keep the last N tokens") while a
    session is task-shaped ("keep what this request has been working on").
    Measured on a real 8102-record session, the active-task span at the seven
    compaction passes was 0.3k / 46.9k / 48.8k / 30.0k / 19.8k / 123.4k /
    49.1k tokens (p50 46.9k; p90 78.8k interpolated, 123.4k nearest-rank)
    against a ``keep_recent_tokens`` of 20k — so five of the seven passes
    summarized away the first half of the task the agent was still executing.
    A later 10-pass session measured spans of the same shape
    (``docs/evidence/compaction-ruler/span_percentiles.txt``); pooled, the 17
    are bimodal, thirteen under 54k and four between 113k and 132k.

    (This docstring previously read "p50 32k, p90 99k". Neither follows from
    the seven spans listed beside them, and the 99k figure was load-bearing in
    a later argument about the preserve-window cap, so it is corrected here
    rather than left to be re-derived — pre-existing, found while sizing
    ``_TASK_FLOOR_KEEP_MULTIPLE``.) That severance, not the token spend, is what
    makes an EARLIER trigger dangerous; widening the preserve window to the
    task boundary is what makes it safe.

    Callers use this as a FLOOR under ``keep_recent_tokens``
    (``max(keep_recent_tokens, task_boundary_floor(...))``), never as a
    replacement: it can only keep MORE history, so it can never introduce a
    cut that the recency rule would not already have allowed.

    ``cap`` is mandatory and load-bearing. A session whose last genuine user
    turn is 500k tokens back would otherwise demand a preserve window larger
    than the context itself, and ``find_cut_point`` would answer ``None`` —
    turning "protect the task" into "never compact", which is the failure the
    trigger exists to prevent. At the cap the pass reverts to plain recency
    behaviour, which is exactly the pre-existing behaviour.

    ``genuine_user_ids`` is the same discriminator
    :func:`extract_preserved_user_turns` takes and for the same reason: in the
    RENDERED history a compaction marker and every injected user-role delivery
    (wake, hub, session-incident, todo reminder) is structurally a
    ``Message(role="user")``, so counting from the last of THOSE would measure
    from an injection rather than from the request the user actually made.
    Omitted (unit tests over raw lists), every user ``Message`` qualifies.

    Returns ``0`` when there is no genuine user turn at all, which leaves the
    caller's ``keep_recent_tokens`` untouched.
    """
    if cap <= 0:
        return 0
    total = len(messages)
    last_user = -1
    for index in range(total - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, Message) or message.role != "user":
            continue
        # A turn a PRIOR pass preserved verbatim is carried-forward context,
        # not the start of the live task: measuring from it would re-anchor
        # every subsequent pass to the same ancient request and grow the floor
        # without bound.
        if _is_preserved_user_turn(message):
            continue
        if genuine_user_ids is not None and message.id not in genuine_user_ids:
            continue
        last_user = index
        break
    if last_user < 0:
        return 0
    span = sum(_message_tokens(m) for m in messages[last_user:])
    return min(span, cap)


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


def extract_preserved_user_turns(
    to_summarize: Sequence[AgentMessage],
    genuine_user_ids: set[str] | None = None,
) -> list[dict[str, str]]:
    """Verbatim ``{"id", "text"}`` for every USER turn in the summarized block.

    The structural half of "never summarize a user turn": a summarizer
    paraphrases assistant/tool content, and a paraphrased user constraint
    ("use the existing helper, don't add a new one" / "NEVER touch billing.py")
    is exactly how an agent later does the forbidden thing. So user-authored
    text is lifted out of what the summarizer sees and re-injected verbatim on
    both the live and the replay path (see ``Session._run_compaction`` and
    ``Transcript.build_llm_history``).

    ``genuine_user_ids`` is the crucial discriminator in production. The block
    passed here is the RENDERED history, where a prior compaction marker and
    every injected user-role delivery (wake, hub, session-incident, todo
    reminder) have ALREADY been rendered from a ``CustomMessage`` into a plain
    ``Message(role="user")`` — structurally indistinguishable from a real
    prompt. Preserving those would carry a previous summary forward verbatim
    and nest summaries across generations. The caller therefore passes the ids
    of the genuine user turns (the ``Message(role="user")`` entries in the LIVE
    context, which injected content is a ``CustomMessage`` in), and only those
    ids are lifted. When it is omitted (unit tests over raw message lists with
    no rendered markers), every user ``Message`` qualifies.

    Empty-text turns (a bare pasted screenshot) carry no constraint to protect
    and are skipped so the preserved block does not accrue blank messages
    every pass.
    """
    preserved: list[dict[str, str]] = []
    for message in to_summarize:
        if not isinstance(message, Message) or message.role != "user":
            continue
        if genuine_user_ids is not None and message.id not in genuine_user_ids:
            continue
        text = message.text
        if not text:
            continue
        preserved.append({"id": message.id, "text": text})
    return preserved
