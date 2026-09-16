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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Container, Sequence

from local_operator.harness.rows import is_harness_injection, is_harness_notice_text
from local_operator.harness.types import AgentMessage, CustomMessage, Message

from .tokens import _encode_len, estimate_tokens

__all__ = [
    "find_cut_point",
    "prepare_partitions",
    "extract_preserved_user_turns",
    "task_boundary_floor",
    "cap_preserved_user_turns",
    "CappedPreservedTurns",
    "elision_notice_text",
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

#: One entry of a preserved block: ``{"id", "text"}`` for an operator turn, plus
#: the two int counts on an elision notice. ``int`` is in the value type because
#: those counts must travel as NUMBERS beside the text rather than as digits
#: inside it — a count embedded in content is a count something re-parses and
#: re-adds, which is how an earlier revision turned 6 into 7,167.
PreservedTurn = dict[str, "str | int"]

#: Marks a user-role ``Message`` that the RENDERER minted from a harness
#: injection (``session_state``, ``hub_message``, ``peer_message``,
#: ``session_incident``, a wake/job delivery, …) rather than from something the
#: operator typed. Set at render time by ``_default_convert_to_llm``; rides
#: ``provider_payload`` exactly as :data:`PRESERVED_USER_TURN_KEY` does, which
#: the wire builders never ship as content.
#:
#: This exists because provenance CANNOT be recovered from the message's shape.
#: The old discriminator asked "is this id one of the user Messages in the live
#: context?", justified by the claim that an injected delivery is a
#: ``CustomMessage`` there. That claim holds only for the FIRST pass: a pass
#: commits ``[marker, *preserved, *kept]`` where ``kept`` comes from the
#: RENDERED history, so from the second pass onward every injection inside the
#: kept window is already a plain ``Message(role="user")`` in the live context
#: — it therefore passed the very test written to exclude it. Measured on the
#: session that motivated this (2f95e374dd22): 160 of 171 preserved turns were
#: injections, 79 of them ``session_state`` repeats of the same system/team
#: brief, and the block reached ~362,780 tokens against a ~640k trigger.
#: A positive marker set at the point of minting is the only test that stays
#: true across generations, because it records where the message CAME FROM
#: instead of inferring it from what it currently looks like.
RENDERED_INJECTION_KEY = "harness_injected"

#: Id of the synthetic turn that reports an elision. Stable rather than
#: count-suffixed: the count is carried STRUCTURALLY beside the block (see
#: :class:`CappedPreservedTurns` and the ``preserved_turns_dropped`` payload
#: field), never parsed back out of a string.
#:
#: **This turn must never be JOURNALLED.** It does live in
#: ``Session._context.messages`` — that is how the model reads it, and how a
#: mid-turn pass carries its counts forward — but it is synthesized on both
#: paths and so a stored copy is a duplicate by construction. An earlier
#: revision asserted here that "no entry was ever written for it". That was
#: false in the assembled system: ``_is_persistable_message`` returned True for
#: every plain ``Message``, so the next turn boundary journalled it AFTER
#: ``first_kept_entry_id`` — inside the replayed suffix — while
#: ``replay_entries`` re-injected it from the marker payload as well. Live and
#: resumed contexts diverged (2 rows vs 3) whenever the cap bound, and because
#: the journalled copy carried ``compaction_preserved`` it was skipped by
#: ``find_cut_point`` and folded into the NEXT count: measured doubling every
#: resume, 6 -> 7,167 over ten cycles, a 231x overstatement in the one message
#: whose job is to tell the model the truth about what it lost.
#:
#: Enforced by ``_is_persistable_message`` (which excludes this id) and pinned
#: by a test that greps the raw journal, rather than by this comment.
#:
#: So the notice is synthesized at RENDER time on both paths from the counts
#: below, and is excluded from persistence by id (see
#: ``_is_persistable_message``). Two independent guards, because one of them
#: failing silently is what produced the defect.
PRESERVED_TURN_ELISION_ID = "compaction-elision"

#: Retained so a transcript written by the previous revision still parses: its
#: notices were journalled with ``compaction-elision-<count>`` ids, and the read
#: path has to recognise and drop them rather than replay them as user turns.
PRESERVED_TURN_ELISION_ID_PREFIX = "compaction-elision"

#: The elision counts a RENDERED notice carries on ``provider_payload``, so a
#: READ-path heal's drops survive into the next write pass.
#:
#: They exist because the heal is the one producer whose numbers are in no
#: payload: it acts on records written before the fields existed, computes the
#: drops at render time, and the next pass seeds its carry-forward from the
#: PREVIOUS marker — which has nothing. Measured before these keys: a resume
#: reporting 160 shed injections was followed by a pass reporting none.
#:
#: Ints on the bookkeeping channel, never digits inside the notice text. The
#: distinction is load-bearing rather than stylistic: a count that lives in
#: content is a count something re-reads and re-adds, which is how an earlier
#: revision turned 6 into 7,167 across ten resumes.
ELISION_GENUINE_COUNT_KEY = "compaction_elision_genuine"
ELISION_INJECTION_COUNT_KEY = "compaction_elision_injections"

#: Preserved-block cap applied on the READ path to a record written BEFORE the
#: cap existed. Records written since carry the cap the pass actually used
#: (``preserved_turns_cap`` in the compaction payload), and replay reuses that
#: so a resumed context is byte-identical to the live one it resumed from.
#: Only a legacy record has no such figure, and it needs SOME bound or the
#: sessions already poisoned on disk stay poisoned across a restart.
#:
#: 100,000 is what ``_advisor_floor_cap`` resolves to on the shipped defaults
#: (``keep_recent_tokens`` 20,000 x ``_TASK_FLOOR_KEEP_MULTIPLE`` 5) above the
#: ~250k window crossover. Deliberately the LOOSER of the two terms rather than
#: a guess at the session's model: replay does not know the model's context
#: window, and healing a legacy record too aggressively would drop constraints
#: the live session still holds. Erring wide costs headroom on one resume; the
#: next pass then re-caps with the real figure.
DEFAULT_PRESERVED_TURN_CAP = 100_000


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


def is_rendered_injection(message: AgentMessage) -> bool:
    """True for a user-role message the RENDERER minted from a harness aside.

    The provenance test that replaced "is its id in the live context?" — see
    :data:`RENDERED_INJECTION_KEY` for why that older test was not merely weak
    but actively self-defeating (compaction's own output defeated it).

    Used on both the write path (``Session._finish_compaction``) and the read
    path (``Transcript.build_llm_history``), so a live and a resumed context
    filter identically and stay byte-for-byte equal.

    A thin alias for ``harness.rows.is_harness_injection``, which is where the
    stamp and its key live: two implementations of one provenance decision is
    exactly how the folds and compaction drift apart about a row.
    """
    return is_harness_injection(message)


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
    (session-state, hub, peer, incident, wake, todo reminder) is structurally a
    ``Message(role="user")``, so counting from the last of THOSE would measure
    from an injection rather than from the request the user actually made.
    Omitted (unit tests over raw lists), every user ``Message`` qualifies.

    That id set is NOT sufficient on its own, and this docstring previously
    claimed it was (\"injected content is a ``CustomMessage`` in the live
    context\"). From the second pass onward it is not: compaction rebuilds the
    context from the RENDERED history, so injections inside the kept window are
    plain user ``Message`` entries there and pass the test. The floor therefore
    also skips anything :func:`is_rendered_injection` recognises, which is the
    provenance marker the renderer stamps at mint time. Without it a repeated
    ``session_state`` delivery re-anchors the floor to itself on every pass and
    the preserve window grows for the same reason the preserved block did.

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
        # A harness injection is not the start of the live task. The id set
        # below cannot catch it after the first pass (see the docstring), so
        # provenance is checked here too — otherwise a session_state delivery
        # arriving every turn re-anchors the floor to itself forever.
        if is_rendered_injection(message):
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
) -> list[Mapping[str, object]]:
    """Verbatim ``{"id", "text"}`` for every USER turn in the summarized block.

    The structural half of "never summarize a user turn": a summarizer
    paraphrases assistant/tool content, and a paraphrased user constraint
    ("use the existing helper, don't add a new one" / "NEVER touch billing.py")
    is exactly how an agent later does the forbidden thing. So user-authored
    text is lifted out of what the summarizer sees and re-injected verbatim on
    both the live and the replay path (see ``Session._run_compaction`` and
    ``Transcript.build_llm_history``).

    Provenance, not shape, decides what counts as user-authored. The block
    passed here is the RENDERED history, where a prior compaction marker and
    every injected user-role delivery (session-state, hub, peer, incident,
    wake, job result, todo reminder) has ALREADY been rendered from a
    ``CustomMessage`` into a plain ``Message(role="user")`` — structurally
    indistinguishable from a real prompt. :func:`is_rendered_injection` reads
    the marker the renderer stamps at mint time and excludes them.

    **The id-set test alone is not sufficient, and this is the bug that took
    a session down.** ``genuine_user_ids`` was previously documented as
    complete because injected content "is a ``CustomMessage`` in the LIVE
    context". That is true only before the first pass: a pass commits
    ``[marker, *preserved, *kept]`` from the RENDERED history, so on pass two
    every injection in the kept window is a plain user ``Message`` in the live
    context and is therefore IN the id set. See :data:`RENDERED_INJECTION_KEY`
    for the measurements. ``genuine_user_ids`` is retained as a second,
    narrowing filter (it still excludes anything not in the live context at
    all), but the provenance marker is what actually carries the guarantee.

    Hub, peer and job-result deliveries are excluded along with the rest, and
    deliberately: they are messages from SUBAGENTS and PEER SESSIONS, not
    constraints the operator authored. The guarantee this function exists to
    provide is "the user's own words are never paraphrased away" — a subagent's
    status report is ordinary history the summarizer is entitled to compress,
    and on the motivating session those three types alone accounted for 80 of
    171 preserved turns. Preserving them buys no protection and costs the
    headroom that makes a pass useful.

    Empty-text turns (a bare pasted screenshot) carry no constraint to protect
    and are skipped so the preserved block does not accrue blank messages
    every pass.
    """
    preserved: list[Mapping[str, object]] = []
    for message in to_summarize:
        if not isinstance(message, Message) or message.role != "user":
            continue
        if is_rendered_injection(message):
            continue
        # The elision NOTICE is compaction's own output, not user text. It has
        # to be excluded here rather than only stripped later by
        # ``cap_preserved_user_turns``: extracting it and stripping it discards
        # the counts it carries, which is what silently retracted a READ-path
        # heal's figure (a resume reporting 160 shed injections followed by a
        # pass reporting none). The caller reads those counts off the message
        # via ``elision_counts_of`` and seeds the next block with them.
        if message.id.startswith(PRESERVED_TURN_ELISION_ID_PREFIX):
            continue
        # A notice the harness minted, carried in a block from before the stamp
        # existed (see :func:`is_harness_notice_text`): a preserved copy of one
        # is re-seated as a ``role="user"`` row on every replay, so it titles
        # the conversation after itself and paints the harness's words as the
        # operator's. Excluded at HARVEST so no new marker bakes one in; the
        # blocks an older build already wrote are healed on the read path by
        # :func:`cap_preserved_user_turns`.
        #
        # The cost of the text test here is bounded and worth naming: a genuine
        # prompt that happens to OPEN with one of these heads is not preserved
        # verbatim. It is still summarized like any other turn and never hidden
        # (this function only decides what rides the marker), whereas a notice
        # carried forward is re-painted as the user's own words on every
        # surface — the asymmetry is what makes the trade right.
        if is_harness_notice_text(message.text):
            continue
        if genuine_user_ids is not None and message.id not in genuine_user_ids:
            continue
        text = message.text
        if not text:
            continue
        preserved.append({"id": message.id, "text": text})
    return preserved


@dataclass(frozen=True)
class CappedPreservedTurns:
    """The preserved block after bounding, with the elision counted separately.

    The counts are STRUCTURAL rather than encoded into a synthetic turn's id.
    An earlier revision carried them in the id string, which the caller then
    minted as a live ``Message``; the persist path journalled that message and
    replay re-injected it from the payload as well, so the count was parsed
    back out of a turn that existed twice and each pass added it to the next.
    Measured: the reported figure doubled on every resume (6 -> 7,167 over ten
    cycles, 231x overstatement). A number that must stay true across a
    round trip belongs in a field, not in a string that is also content.

    ``genuine_dropped`` is tracked apart from ``injections_dropped`` because
    the two mean opposite things to the operator. Shedding a harness injection
    reclaims context and loses nothing they wrote. Evicting a genuine turn
    drops something they DID write, and the notice says so in as many words —
    they should never have to infer that from a total.
    """

    turns: list[Mapping[str, object]]
    genuine_dropped: int = 0
    injections_dropped: int = 0

    @property
    def total_dropped(self) -> int:
        return self.genuine_dropped + self.injections_dropped


def elision_notice_text(genuine_dropped: int, injections_dropped: int) -> str | None:
    """The notice rendered in place of an elided prefix, or ``None``.

    Synthesized at RENDER time on both the live and the replay path, from the
    counts, so it is never itself a stored turn that can be re-read as history.

    The two causes are phrased differently on purpose. A shed injection is
    bookkeeping the model does not need to act on; an evicted GENUINE turn is
    the operator's own words leaving the context, and the whole point of a
    visible elision is that the model asks rather than concluding the
    instruction was never given.

    **The counts must be CUMULATIVE, and once told the fact may not be
    retracted.** Every producer of these numbers has to carry forward what
    earlier passes reported: the block is re-capped on every pass AND on every
    replay, and a notice that vanished the first time a block happened to fit
    would silently retract a fact the model had already been told. This
    paragraph was deleted in the same commit that broke the invariant it
    describes — a resume reporting 160 shed injections was followed by a pass
    reporting nothing — so it is restored here, next to the function every
    producer calls, rather than beside any one of them. The carriers are the
    ``preserved_turns_dropped``/``preserved_turns_shed`` payload fields for a
    stored marker and :data:`ELISION_GENUINE_COUNT_KEY` /
    :data:`ELISION_INJECTION_COUNT_KEY` for a rendered notice whose counts are
    in no payload yet; :func:`elision_counts_of` reads the latter back.
    """
    if genuine_dropped <= 0 and injections_dropped <= 0:
        return None
    parts: list[str] = []
    if genuine_dropped > 0:
        parts.append(
            f"{genuine_dropped} older user message(s) you wrote were dropped here to "
            "bound the context. They are still in the session transcript but are no "
            "longer in the model's context. If an earlier instruction seems to be "
            "missing, ask rather than assuming it was never given."
        )
    if injections_dropped > 0:
        parts.append(
            f"{injections_dropped} harness-injected message(s) (session state, "
            "subagent and peer deliveries) were also dropped; these were not "
            "authored by the user."
        )
    return "[" + " ".join(parts) + "]"


def replay_preserved_turns(
    payload: dict[str, object],
    injection_ids: Container[str] | None = None,
) -> list[Mapping[str, object]]:
    """The preserved block a REPLAY should inject, from a compaction payload.

    One function for what were three hand-kept copies (``build_llm_history``,
    ``mobile.durable._compaction_prefix``, and the fold's rebuild). The review
    called that out: logic that "must agree message for message" and is
    duplicated is where a fix lands in two places out of three. The notice is
    synthesized here too, so no caller can accidentally reintroduce it as a
    stored turn.

    ``injection_ids`` is resolved by the caller against the JOURNAL — a stored
    turn has no rendered message to read a provenance stamp off, but it does
    have an id, and the entry that id names records its ``custom_type``. That
    is a provenance lookup, not the shape-based guess the write path rejects.

    Records written before the cap existed carry no ``preserved_turns_cap``;
    they are healed under :data:`DEFAULT_PRESERVED_TURN_CAP`. Records written
    since replay under the figure their own pass used, which is what keeps a
    resumed context byte-identical to the live one it resumed from.

    **The notice this returns carries its own counts.** A READ-path heal
    computes drops that exist in no payload — the record it healed predates the
    fields — so the numbers have to travel with the rendered notice or the next
    write pass, which seeds its carry-forward from the previous marker, silently
    RETRACTS them. Measured before this was carried: a resume reporting 160 shed
    injections was followed by a pass reporting nothing. That is the same defect
    as the notice-doubling one with the sign flipped to under-reporting, and it
    is why :func:`elision_counts_of` exists rather than the count being
    re-parsed out of the notice text (parsing a number back out of content is
    exactly what produced the doubling).
    """
    stored = payload.get("preserved_user_turns") or ()
    if not isinstance(stored, (list, tuple)):
        stored = ()
    turns = [turn for turn in stored if isinstance(turn, dict)]
    raw_cap = payload.get("preserved_turns_cap")
    cap = int(raw_cap) if isinstance(raw_cap, int) else DEFAULT_PRESERVED_TURN_CAP
    capped = cap_preserved_user_turns(
        turns,
        cap=cap,
        already_dropped_genuine=_payload_int(payload, "preserved_turns_dropped"),
        already_dropped_injections=_payload_int(payload, "preserved_turns_shed"),
        injection_ids=injection_ids,
    )
    notice = elision_notice_text(capped.genuine_dropped, capped.injections_dropped)
    if notice is None:
        return list(capped.turns)
    return [
        {
            "id": PRESERVED_TURN_ELISION_ID,
            "text": notice,
            # Ints, deliberately, beside the text rather than inside it. The
            # caller stamps these onto the rendered message's
            # ``provider_payload`` so the write path can seed from the block it
            # RECEIVED instead of only from the previous marker.
            ELISION_GENUINE_COUNT_KEY: capped.genuine_dropped,
            ELISION_INJECTION_COUNT_KEY: capped.injections_dropped,
        },
        *capped.turns,
    ]


def preserved_turn_payload(turn: Mapping[str, object]) -> dict[str, object]:
    """The ``provider_payload`` a replayed preserved turn must carry.

    One builder for the three render sites (``build_llm_history``, the mobile
    fold, and the live commit) so the bookkeeping cannot drift between them —
    the same argument that collapsed the read-path cap into one helper. Every
    turn is flagged already-compacted; a notice additionally carries its counts,
    which is what lets a READ-path heal's drops reach the next write pass.
    """
    payload: dict[str, object] = {PRESERVED_USER_TURN_KEY: True}
    for key in (ELISION_GENUINE_COUNT_KEY, ELISION_INJECTION_COUNT_KEY):
        value = turn.get(key)
        if isinstance(value, int) and value > 0:
            payload[key] = value
    return payload


def elision_counts_of(message: AgentMessage) -> tuple[int, int]:
    """``(genuine_dropped, injections_dropped)`` a rendered notice carries.

    ``(0, 0)`` for anything that is not an elision notice, so a caller can scan
    a history unconditionally.

    Read off ``provider_payload`` — the same harness-bookkeeping channel
    :data:`PRESERVED_USER_TURN_KEY` uses, which the wire builders never ship as
    content. Never parsed out of the notice text: a count that lives in content
    is a count that gets re-counted, which is precisely how an earlier revision
    inflated 6 to 7,167 across ten resumes.
    """
    if not isinstance(message, Message) or not message.provider_payload:
        return 0, 0
    payload = message.provider_payload
    genuine = payload.get(ELISION_GENUINE_COUNT_KEY)
    injections = payload.get(ELISION_INJECTION_COUNT_KEY)
    return (
        genuine if isinstance(genuine, int) and genuine > 0 else 0,
        injections if isinstance(injections, int) and injections > 0 else 0,
    )


def _payload_int(payload: dict[str, object], key: str) -> int:
    """A non-negative int from a compaction payload, or 0.

    Lenient because the field is absent on every record written before it
    existed, and a malformed one must degrade to "nothing carried forward"
    rather than break a resume.
    """
    value = payload.get(key)
    return value if isinstance(value, int) and value > 0 else 0


def cap_preserved_user_turns(
    turns: Sequence[Mapping[str, object]],
    *,
    cap: int,
    already_dropped_genuine: int = 0,
    already_dropped_injections: int = 0,
    injection_ids: Container[str] | None = None,
) -> CappedPreservedTurns:
    """Bound the preserved block to ``cap`` tokens, dropping OLDEST first.

    Without a bound the block is a monotonic ratchet.
    :func:`_is_preserved_user_turn` makes :func:`find_cut_point` skip turns a
    prior pass preserved, so they are never re-summarized and never dropped:
    each pass carries the whole previous block forward and appends to it.
    Measured over the last eight passes of session 2f95e374dd22, carried-forward
    turns went 166→167→168→168→169→170→171 with **zero** removed, and the block
    reached 362,780 tokens against a ~640k trigger — 57% of the context
    permanently unshrinkable, so every pass produced no usable headroom and
    immediately re-fired (25 passes in seven minutes).

    Filtering injections out (the fix above) shrinks the block but does not
    bound it: N genuine user turns in a long session still grow without limit.
    This is the same argument :func:`task_boundary_floor` makes for its own
    ``cap`` being "mandatory and load-bearing" — an unbounded preserve window
    turns "protect the task" into "never compact" — and it is answered the same
    way, with the same cap vocabulary (``Session._preserved_turns_cap``) rather
    than a parallel knob nobody would keep in step.

    ``injection_ids`` names turns the caller has identified as harness
    injections by PROVENANCE. They are shed FIRST, before any age-based
    eviction, and the reason is a correctness one rather than a preference: an
    earlier revision relied on oldest-first eviction to remove them, asserting
    they were "the oldest". They are typically the NEWEST — ``session_state``
    arrives every turn — so age-based eviction preferentially dropped the
    operator's genuine constraints and kept the injections. Measured on the
    real fleet, only 12% of stored injections were reached that way.

    **The contract this establishes, stated plainly because it is a trade.**
    Once the cap binds, the guarantee is "the NEWEST genuine constraints
    survive verbatim", not "every constraint survives forever". An old enough
    operator turn CAN be evicted — after the injections are gone, oldest-first
    among genuine turns is what remains. That is deliberate: unbounded
    preservation is the defect this function exists to fix, and a rule stated
    ten tasks ago is likelier to be spent than the one stated on the current
    task. The eviction is never silent (see :func:`elision_notice_text`, which
    names genuine drops separately from shed injections).

    Capping by TOKENS rather than turn count is what makes the bound mean
    anything — one 300k-token paste and 300 short turns are the same problem,
    and only the token budget sees both.

    The newest turn always survives, even alone and even when it exceeds the
    cap on its own: returning an empty block for one oversized turn would
    discard the live constraint, which is the exact failure the whole
    preservation mechanism exists to prevent.

    ``already_dropped_*`` carry forward what earlier passes reported so the
    figure keeps describing everything ever dropped from this block. They are
    passed in from the payload rather than recovered from the turns, which is
    what makes the count survive a round trip without existing as content.
    """
    real_turns = [
        turn
        for turn in turns
        # A notice journalled by the previous revision is not history. Dropping
        # it here is what stops a resumed session replaying it as a user turn.
        if not str(turn.get("id", "")).startswith(PRESERVED_TURN_ELISION_ID_PREFIX)
    ]
    genuine_dropped = already_dropped_genuine
    injections_dropped = already_dropped_injections

    # Provenance first: shedding an injection costs the operator nothing, so it
    # must never compete with a genuine turn for the budget.
    #
    # Two shapes count as one. ``injection_ids`` is the journal lookup, which is
    # how every delivery a modern build wrote is identified. A stored turn whose
    # TEXT is a harness notice head is the LEGACY shape — the copy exists
    # because an older build harvested the notice as a user turn, before there
    # was a stamp to refuse it — and it is shed for the same reason: replaying
    # it re-seats the harness's words as a user row. Text is sound evidence at
    # THIS site in a way it is not on a live fold: these are stored copies
    # inside a compaction block, never a row the operator typed in this process.
    def _is_injection_turn(turn: Mapping[str, object]) -> bool:
        if injection_ids is not None and str(turn.get("id", "")) in injection_ids:
            return True
        return is_harness_notice_text(str(turn.get("text", "")))

    kept_turns = [turn for turn in real_turns if not _is_injection_turn(turn)]
    injections_dropped += len(real_turns) - len(kept_turns)
    real_turns = kept_turns

    if cap <= 0:
        # Fails CLOSED. Returning the block unbounded here would restore the
        # exact ratchet this function exists to prevent, on the one input a
        # caller would plausibly pass to mean "nothing survives". The shipped
        # default is the floor instead, so the invariant holds for every
        # caller rather than only the careful one.
        cap = DEFAULT_PRESERVED_TURN_CAP

    kept: list[Mapping[str, object]] = []
    used = 0
    # Walk NEWEST first and stop at the budget; the surviving prefix is then
    # re-reversed so the block keeps chronological order.
    for turn in reversed(real_turns):
        cost = _encode_len(str(turn.get("text", "")))
        if kept and used + cost > cap:
            break
        kept.append(turn)
        used += cost
    kept.reverse()
    genuine_dropped += len(real_turns) - len(kept)
    return CappedPreservedTurns(
        turns=kept,
        genuine_dropped=genuine_dropped,
        injections_dropped=injections_dropped,
    )
