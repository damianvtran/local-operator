"""Canonical, transport-neutral state consumed by every full terminal UI.

Raw agent events remain the animation stream. This module owns everything a
newly attached ``OperatorApp`` needs before the next event arrives, so local and
remote terminals hydrate from the same typed facts instead of reconstructing
session semantics from the phone's deliberately capped projection.
"""

from __future__ import annotations

import copy
import json
import os
import time
import uuid
from collections import deque
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
)
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic_core import PydanticSerializationError, to_jsonable_python

from local_operator.harness.subagent import TRAJECTORY_CAP as _TRAJECTORY_CAP
from local_operator.harness.types import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    CompactionEndEvent,
    MessageEndEvent,
    ModelSpec,
    Usage,
)
from local_operator.mcp.grants import GRANT_SUBCOMMANDS as _GRANT_SUBCOMMANDS
from local_operator.session.history_window import DisplayHistoryWindow
from local_operator.tui.costs import cost_summary, job_cost, turn_cost

FRONTEND_STATE_VERSION = 1
FRONTEND_CAPABILITY = "tui_state_v1"
FRONTEND_CHECKPOINT_CUSTOM_TYPE = "frontend_state_checkpoint_v1"

#: How many per-call billing receipts ``usage_components`` retains.
#:
#: The list grows by one receipt per model call for the LIFE of a conversation
#: and is re-serialized in full on two paths that both have hard budgets: the
#: attach frame (``server._MAX_LINE_BYTES``, 1 MiB) and the turn-end checkpoint
#: appended to the transcript. Uncapped it broke both on the reference machine:
#: a 2,685-receipt session serialized a 1,052,296-byte ``frontend_sync`` — over
#: the socket's line limit, so `AttachClient` could not read it, every attach to
#: that session timed out after 15 s and silently degraded to a runtime-less
#: cold viewer. The same field was ALSO 48.2% of that session's 103 MB
#: transcript (49.8 MB across 119 checkpoint rows, each re-writing the whole
#: accumulated list) because the checkpoint stripped ``live_events`` and job
#: trajectories for exactly this reason but not this list.
#:
#: Capped HERE, at accumulation, rather than only at the wire boundary where
#: trajectories are stripped: a wire-only bound leaves the transcript growth in
#: place, and the transcript is the more expensive of the two (it is durable,
#: and it is re-parsed on every resume). The tail is what a mixed-provider
#: aggregate needs — the receipts state which call was served by which model at
#: which price — and the lifetime figures the UI actually paints
#: (``cumulative_parent_cost``, ``child_costs``, ``last_usage``) are running
#: totals maintained independently, so dropping an old receipt cannot move a
#: number on screen. 200 covers several turns of a busy multi-provider session
#: at roughly 55 KB, two orders of magnitude inside both budgets.
USAGE_COMPONENT_CAP = 200

#: Per-job free-text bounds for the ATTACH FRAME (not for the in-process store).
#:
#: ``jobs`` is the third instance of the same shape and the next one that would
#: have overflowed: the rows are unbounded in BYTES even once trajectories and
#: receipts are handled, because ``result_text`` and ``prompt`` are whole child
#: outputs. Measured on the post-fix wire path, a roster of settled children
#: carrying 4 KB in each field crosses the 1 MiB line limit at ~130 rows with no
#: receipt involved (review round 1, C4).
#:
#: Truncating on the wire loses nothing a reader cannot recover: both fields
#: live verbatim in the CHILD's own transcript, which the subagent page pages in
#: lazily — the same argument ``_load_subagent_roster``'s docstring already makes
#: for dropping them from the roster sidecar, and the same one the mobile
#: projection makes for ``SUBAGENT_OUTCOME_CHARS`` / ``SUBAGENT_PROMPT_PREVIEW_CHARS``.
#: The bounds mirror that surface so two frontends do not disagree about how much
#: of a child's output is "the preview".
#:
#: ``error_text`` is deliberately far more generous and matches
#: ``session._ROSTER_ERROR_CAP``: it is NOT in the child transcript (it is
#: ``str(exc)`` from the parent's runner), so the wire value is the only copy the
#: reader will ever see of why a child failed.
JOB_RESULT_WIRE_CHARS = 2_000
JOB_PROMPT_WIRE_CHARS = 1_000
JOB_ERROR_WIRE_CHARS = 2_000

#: Aggregate budget for all per-job free text in one frame, and the floor a
#: single row's share may not go below.
#:
#: A PER-ROW cap alone does not close the class, it only moves the threshold: the
#: frame still grows linearly with roster depth, so "bounded per row, unbounded
#: in total" is the same defect one level up. The budget is therefore shared —
#: each row gets ``BUDGET // len(jobs)``, clamped to the per-field caps above —
#: so a deep roster spends the same total bytes on text as a shallow one, with
#: each child described more briefly.
#:
#: Rows are never DROPPED to make space, at any depth. A missing child reads as
#: "this never ran", which is a lie the reader cannot detect; a short preview is
#: visibly short and the full text is one page-open away in the child's own
#: transcript. The floor keeps every row's preview legible for that reason.
#:
#: This does not make the frame unconditionally bounded — each row still carries
#: irreducible identity (id, label, status, folded usage: ~600 B), so the frame
#: is bounded by roster COUNT alone at roughly 1,400 children. That is stated
#: rather than engineered away: the deepest roster observed across every session
#: on the reference machine is 19.
#:
#: Reduced from 120,000 by the 128 bytes that `history_generation` and its
#: sibling window fields add to every frame. The socket line limit is fixed at
#: 1 MiB, so a new session-level field has to be PAID FOR out of an elastic
#: budget rather than shrinking the guard's margin: at the worst case the size
#: guard asserts, the frame had 13 bytes of headroom and the new field costs
#: 25. Per-row text is the right place to take it from — it is already shared,
#: already floored at a legible preview, and 128 chars spread across a roster
#: is invisible, whereas an over-limit frame cannot be sent at all.
JOB_TEXT_FRAME_BUDGET_CHARS = 119_872
JOB_TEXT_FLOOR_CHARS = 200

#: Fields :meth:`FrontendStateStore.read_field` may serve without the
#: whole-state deep copy — see that method for the measurement that motivates
#: it.
#:
#: The membership test is the SAFETY ARGUMENT, not a convenience list, and the
#: bar is DEEP IMMUTABILITY OF THE HANDED-OUT VALUE — not "the reducer replaces
#: it".
#:
#: An earlier version of this set admitted `selected_model`, `effective_model`
#: and `last_usage` on the reducer-replacement argument. That argument is
#: wrong, and review round 2 (Q6/F4) demonstrated it: `ModelSpec` and `Usage`
#: are ordinary NON-FROZEN pydantic models, so `read_field` handed out the
#: store's own instance and a caller writing `spec.model_id = ...` or
#: `usage.input_tokens += n` rewrote canonical session state in place. Nothing
#: shipped did that — but this codebase accumulates `Usage` with `+=` in
#: `harness/jobs.py` and `harness/subagent.py`, so it was one ordinary edit
#: from silent corruption, and `state` DOES protect those fields by returning a
#: fresh object per read. Admitting them removed an existing invariant.
#:
#: They are therefore back on the copying path. Freezing the two models was the
#: alternative and was rejected here: 13 in-place mutation sites across the
#: harness rely on them being writable, which is a refactor well outside the
#: change that introduced this accessor. The measured saving is nearly all in
#: the scalars regardless (a per-read `model_copy` costs ~0.0075 ms against
#: ~0.0001 ms shared, versus ~0.15 ms for the whole-state clone at a 19-job
#: roster), so the fast path keeps the reads that are genuinely free.
#:
#: What remains admitted satisfies both requirements:
#:
#: 1. DEEPLY IMMUTABLE. Every entry is a `str`, `bool`, `int`, `float` or
#:    `None`. A caller cannot mutate any of them, so sharing the value cannot
#:    reach the store at all.
#: 2. NOT A MUTABLE CONTAINER. `jobs`, `usage_components`, `child_costs`,
#:    `attention`, `context_breakdown` and `todos` are excluded even though
#:    single-field readers exist for some: a `dict`/`list` handed out unguarded
#:    can be mutated by its caller, which is exactly what `state`'s deep copy
#:    exists to prevent. Those reads keep paying for it.
#:
#: `test_shareable_state_fields_are_real_and_immutable` asserts both properties
#: against `model_fields`, so neither a renamed field nor a newly mutable type
#: can silently re-enter this set.
_SHAREABLE_STATE_FIELDS = frozenset(
    {
        "streaming",
        "session_id",
        "epoch",
        "sequence",
        "history_generation",
        "conversation_title",
        "goal",
        "active_agent",
        "active_team",
        "context_tokens",
        "context_window",
        "context_is_estimate",
        "cumulative_parent_cost",
    }
)
#: Wire budget for the in-flight seed's retained tool results.
#:
#: ``live_events`` used to bound itself: every ``tool_execution_end`` ERASED its
#: call's row, so a turn of completed calls left zero rows behind and the field
#: could not grow. Retaining the end (so a viewer that reconnects mid-turn can
#: settle a card for work that finished while it was away) removed that bound
#: and made the field accumulate one whole tool result per completed call —
#: measured at 2,028,680 B over 100 calls, putting the ``frontend_sync`` frame
#: at 2,029,839 B against the socket's 1,048,576-byte line limit. Overflow
#: began at ~18 completed calls returning 60 KB each, which is an ordinary
#: heavy turn rather than a pathological one.
#:
#: This is the third instance of the shape :func:`sync_wire_payload` documents
#: after trajectories and ``usage_components``, and the consequence is the one
#: ``server.py`` records: an oversized sync is UNREADABLE rather than merely
#: large, so the viewer waits out its sync timeout and degrades to a cold
#: session with no roster and no todos.
#:
#: Bounded by TRUNCATING result text rather than by dropping rows, because the
#: seed's job is to say WHICH calls ended and HOW — ``on_tool_ended`` needs
#: ``tool_call_id``, ``is_error`` and a first line to settle the card, and
#: dropping a row would strand the card live and re-open the very
#: ``⊘ interrupted`` artefact the retention exists to close. The full result is
#: never lost: it is in the durable transcript, and the live relay delivers it
#: untouched to a viewer that stayed connected. Only the RECONNECT seed is
#: clipped, and only its text.
#:
#: A shared frame budget rather than a per-row cap, for the reason
#: :data:`JOB_TEXT_FRAME_BUDGET_CHARS` gives: "bounded per row, unbounded in
#: total" is the same defect one level up, and call count per turn has no cap.
#: The floor keeps every retained end legible — a card that says how it ended
#: is what stops the retirement pass marking it interrupted.
LIVE_EVENT_TEXT_FRAME_BUDGET_CHARS = 60_000
LIVE_EVENT_TEXT_FLOOR_CHARS = 200

#: Cap on retained ``tool_execution_end`` rows in the seed, newest kept.
#:
#: The text budget above bounds what a row COSTS; it does not bound how many
#: rows there are, and "bounded per row, unbounded in total" is exactly the
#: defect the budget exists to answer. The floor makes that gap reachable: at
#: :data:`LIVE_EVENT_TEXT_FLOOR_CHARS` every row still costs ~500 B, and even
#: stripped to bare identity a row is ~300 B, so 5,000 completed calls in one
#: turn measured 2,528,943 B — back over the line limit with the text budget
#: fully applied. A cap is the only thing that closes it.
#:
#: 100 holds the field's worst case to ~71 KB — the same order as
#: ``usage_components`` (69 KB), and a fair share of a frame whose roster alone
#: can run to 620 KB. The all-year fixture is what set the number, and it had
#: to be set twice: at 400 the seed took 255 KB and at 150 it took 107 KB, both
#: of which put the frame back over the limit once stacked on that roster. With
#: every other field at its own maximum the fixture leaves ~99 KB here, so
#: "bounded" is not sufficient on its own — the bound has to be small enough to
#: COEXIST. A viewer needs the seed only for the CURRENT turn's unsettled
#: cards, and 100 completed calls in one turn is already far beyond that.
#:
#: Dropping OLDEST-first is what makes the cap safe: the newest calls are the
#: ones most likely to still be on screen unsettled, and a dropped row costs
#: nothing durable — the transcript replay repaints those cards regardless.
LIVE_EVENT_END_ROWS_MAX = 100

#: Smallest catalogue the wire will clip to, however little the frame has left.
#:
#: ``model_catalogue`` is the same shape as the job text above — one row per
#: offerable model, bounded by "whatever the provider lists" rather than by
#: anything in this process — but it CANNOT take the same fixed budget, and the
#: arithmetic is worth recording because the obvious fix is the wrong one.
#:
#: A production row is 11 keys and ~267 B (see :meth:`refresh_model_catalogue`).
#: Real providers already list ~600 models and one QA backend published 1,410,
#: so honouring the real catalogue needs ~392 KB. But the frame guard's
#: pathological session (a 200-child roster, every collection field maxed) is
#: already ~981 KB with an EMPTY catalogue, leaving only ~67 KB of the 1 MiB
#: socket line — about 241 rows. No constant satisfies both: a budget big
#: enough for a real provider overflows that frame, and one small enough to fit
#: it hides two thirds of OpenRouter from the picker to satisfy a fixture.
#:
#: So the budget is RESIDUAL rather than constant: the catalogue is measured
#: against what the socket line actually has left once every other field has
#: been bounded, which is the only quantity that answers both cases. An
#: ordinary session — the deepest roster ever observed on the reference machine
#: is 19, and 1,410 models there serialize to ~442 KB against a 1 MiB cap —
#: has hundreds of kilobytes spare and is never touched. Only the pathological
#: combination clips, which is the case the socket cannot carry anyway.
#:
#: Clipping keeps the FIRST rows in the owner's existing sort order (best/most
#: relevant first, the order the picker already renders) rather than dropping
#: from the middle, and sets ``model_catalogue_truncated`` so the reader can
#: say the list is partial instead of presenting it as the whole set.
#:
#: The floor is what stops a frame that is over budget for OTHER reasons from
#: emptying the picker: an empty catalogue reads as "this session can switch to
#: nothing", which is a lie the reader cannot detect, exactly the reasoning
#: :data:`JOB_TEXT_FRAME_BUDGET_CHARS` gives for never dropping a job row.
MODEL_CATALOGUE_FLOOR_ROWS = 50

#: The control socket's line limit, mirrored rather than imported.
#:
#: ``runtime.server`` imports THIS module (``FRONTEND_CAPABILITY``), so
#: importing ``_MAX_LINE_BYTES`` back would close a cycle. The value is pinned
#: to the reader's by ``test_the_catalogue_budget_tracks_the_socket_line_limit``
#: — the same "mirror it and pin it" discipline ``settings_io`` uses for
#: ``tools.builtin.BASH_SHELL_PATH``, and for the same reason: the consumer
#: must stay cheap to import.
_MODEL_CATALOGUE_LINE_LIMIT = 1 << 20

#: Wire bounds for the launch-row reconciliation identities (see
#: :func:`_wire_launch_prompts`).
#:
#: ``launch_prompts`` is a dict of PROMPTS — free text, one entry per resume
#: attempt #314 collapsed into a record — so putting it on the wire verbatim
#: would be the fourth instance of the unbounded-field shape the three caps
#: above exist to close, and the one that grows with RESUME DEPTH rather than
#: with roster depth. It is bounded at CONSTRUCTION (``_with_lineage``) rather
#: than at the wire boundary like ``prompt``/``result_text``, because the delta
#: stream serializes job rows through :func:`_job_summary`, which never passes
#: through :func:`sync_wire_payload` — a wire-boundary-only bound would hold for
#: the attach snapshot and leak on every subsequent frame.
#:
#: The entry cap keeps the NEWEST attempts: durable history pages from the tail,
#: so the most recently collapsed launch rows are the ones a reader loads first
#: and the oldest are the least likely to be on screen at all.
#:
#: The emitted value is at most ``cap + 1`` characters: the ellipsis is appended
#: after the slice, exactly as the neighbouring text bounds do it (round-1
#: finding 4). The budget arithmetic below is computed against the cap itself,
#: which is why the one extra marker character is called out rather than folded
#: into the constant.
#:
#: 200 chars is the same "legible preview" floor :data:`JOB_TEXT_FLOOR_CHARS`
#: sets, and it is sufficient for what this field is FOR: the viewer replaces
#: the durable row wholesale, so a truncated concise prompt still prevents the
#: preamble leak completely and only costs informational richness. Worst case
#: is ~1.9 KB per job before the roster budget below, and the common case is
#: zero entries — a child that was never resumed has no collapsed attempts.
JOB_LAUNCH_PROMPT_WIRE_CHARS = 200
JOB_LAUNCH_PROMPTS_MAX = 8

#: Aggregate budget for collapsed-attempt prompt text across ONE frame.
#:
#: The per-row caps above bound a row; they do not bound a FRAME, and "bounded
#: per row, unbounded in total" is the same defect one level up — the argument
#: :data:`JOB_TEXT_FRAME_BUDGET_CHARS` already makes for ``prompt``/
#: ``result_text``. Measured without this: a 200-row roster of resumed children
#: at the per-row cap adds 392 KB to a 1 MiB frame, and 1,000 rows adds 1.9 MB,
#: so roster depth alone would put the frame back over the socket's line limit.
#:
#: Shared across the rows that actually carry collapsed attempts rather than
#: across the whole roster, because the common row contributes nothing and
#: dividing by it would starve the few rows that do. A starved row keeps its
#: KEYS and loses only their text (see
#: :data:`LAUNCH_PROMPT_ELIDED_PLACEHOLDER`), so reconciliation still fires for
#: every collapsed attempt at any roster depth. 20,000 chars is ~2% of the line
#: limit and covers 100 collapsed attempts at the per-entry cap — two orders of
#: 2,000 chars is deliberately SMALL, and it can be: the budget governs only how
#: much concise prompt TEXT rides, never whether a row reconciles. A starved row
#: keeps its keys (tier 2) or has them rebuilt from ``attempt_aliases`` (tier 3),
#: so the duplicate-preamble fix holds at every depth and the budget trades only
#: informational richness. It is sized to the frame's REAL residual headroom:
#: the ``ran all year`` guard passes with 3,316 bytes spare over 200 rows, i.e.
#: ~16 B/row for anything new, and a budget that ignored that would reintroduce
#: the round-1 blocker in a different field.
JOB_LAUNCH_PROMPTS_FRAME_BUDGET_CHARS = 2_000

#: Aggregate budget for launch IDENTITIES that cannot be elided, per frame.
#:
#: Round-1 blocker, second half. ``launch_message_id`` is normally omitted
#: because it derives from the job id, but a row whose recorded identity belongs
#: to an EARLIER attempt (rebuilt from a persisted comms row) has to carry the
#: literal string, and nothing bounded how many such rows a frame could hold.
#: The guard's worst case — 200 rows of non-derivable ids — measured 5,600 B
#: against a frame with 3,316 B of headroom.
#:
#: A row past this budget drops the string and keeps NOTHING, because unlike the
#: prompts map there is no alias to rebuild from: the whole point of a literal
#: id is that it is not reconstructible. Such a row falls back to exactly the
#: pre-#681 behaviour — the synthetic head plus an unreconciled durable row —
#: which is a bounded, visible degradation rather than an unopenable session.
#:
#: 600 chars covers ~16 non-derivable rows, against a realistic roster where the
#: count is ZERO: every producer verified (launch, #314 resume-fold, and the
#: persist/restore round trip) emits the derivable form, which costs nothing.
#: Sized to the same residual headroom as the prompts budget above.
JOB_LAUNCH_IDS_FRAME_BUDGET_CHARS = 600

#: What a starved row's collapsed-attempt prompts are replaced WITH.
#:
#: Round-1 review MAJOR, reproduced independently by QA on a realistic 45-child
#: roster: emptying a starved row's map re-introduced the exact duplicate this
#: change exists to remove. The original reasoning ("the view re-derives the
#: current launch from ``prompt``") held only for the CURRENT attempt — a
#: COLLAPSED PRIOR attempt has no other source, so its durable row fell back to
#: a plain user row carrying the full role/team/system preamble.
#:
#: Keeping the key with a placeholder reconciles that row to a bare ellipsis
#: instead. It MUST be a visible character: ``SubagentView.show`` drops an entry
#: whose text is falsy after ``strip_control_sequences(...).strip()``, so an
#: empty string would be discarded and leak exactly as before.
#:
#: It also cuts a starved entry from ~249 to ~49 bytes, which is why this and
#: the identity bound are one change: the cheaper entry buys back part of the
#: frame headroom the blocker needed.
LAUNCH_PROMPT_ELIDED_PLACEHOLDER = "…"


def _folded_components(components: Sequence[Any]) -> list[Any]:
    """Collapse receipts that share a serving identity, without moving money.

    The per-JOB twin of :data:`USAGE_COMPONENT_CAP`, and it may not use the
    same mechanism. A tail cap is safe for the session-level list because the
    figures it feeds are running totals maintained elsewhere — but a job's
    ``usage.cost_components`` IS the input to :func:`job_cost`, so dropping a
    receipt there would silently undercount a child's spend. Money must not be
    truncated to fit a socket.

    Folding is the alternative, and it only holds if summing before pricing
    equals pricing before summing. ``cost_for_usage`` prices each receipt
    independently and the caller sums, so the fold has to reproduce that
    exactly. Two things in the pricing path do NOT commute with summation and
    both are handled here rather than assumed away (review round 1, C1):

    * **The reported/estimated split.** ``cost_for_usage`` returns
      ``usd_cost`` verbatim when ``_usage_cost`` ACCEPTS it, and falls back to
      the token estimate when it does not — and ``_usage_cost`` rejects
      negative and non-finite values. Keying the bucket on
      ``usd_cost is not None`` therefore put a rejected receipt in the
      reported bucket, where its poison value was summed into a total that
      ``_usage_cost`` then accepted: two independently-estimated receipts
      became one wrongly-reported one. The key is the ACCEPTED price, so a
      receipt lands in the bucket matching the branch it would really take.
    * **The per-receipt floors.** ``_usage_field`` floors each count at zero
      (a provider spelling "unknown" as ``-1``), and for OpenAI-shaped wires
      ``cost_for_usage`` computes ``max(0, input - read - write)`` per receipt.
      ``max(0, a) + max(0, b) != max(0, a + b)``, so the counts are normalised
      through the same floors BEFORE they are summed. The folded receipt then
      carries counts the pricing path will not floor again, and the two orders
      of operation agree.

    The result is bounded by the number of distinct serving identities a child
    actually used (one or two in practice) rather than by how many calls it
    made, which is what makes it scale with a deep roster.

    Ordering is preserved by first appearance so the newest identity does not
    jump the list, and a component that carries no identity is passed through
    untouched rather than folded into a bucket it may not belong in — pricing
    would resolve it against the caller's default label, which differs per
    call site, so it is not foldable in the first place.
    """
    from local_operator.model.configure import (
        _cache_tokens_are_inside_input,
        _usage_cost,
        _usage_field,
    )

    folded: dict[tuple[str, str, str], Any] = {}
    order: list[tuple[str, str, str]] = []
    passthrough: list[Any] = []
    for raw_component in components:
        provider = str(getattr(raw_component, "provider", "") or "")
        model_id = str(getattr(raw_component, "model_id", "") or "")
        if not provider and not model_id:
            # No identity to fold on: pricing would resolve it against the
            # caller's default label, which differs per call site.
            passthrough.append(raw_component)
            continue
        # The price this receipt would ACTUALLY be billed at, which is the
        # branch ``cost_for_usage`` takes — not merely whether a field is set.
        accepted = _usage_cost(raw_component)
        estimated = _usage_cost({"usd_cost": getattr(raw_component, "estimated_usd_cost", None)})
        # Normalise the counts through the same floors the pricing path
        # applies, so summing them afterwards cannot disagree with pricing
        # them individually. ``input_tokens`` absorbs the OpenAI-shaped
        # subtraction here; the folded receipt is then already disjoint and
        # the second application is a no-op.
        plain = _usage_field(raw_component, "input_tokens")
        read = _usage_field(raw_component, "cache_read_tokens")
        written = _usage_field(raw_component, "cache_write_tokens")
        if _cache_tokens_are_inside_input(provider):
            # Floor the subtraction HERE, per receipt, then re-add the cache
            # buckets so the stored count keeps this wire's "cached tokens are
            # a SUBSET of input" convention. Pricing the folded receipt runs
            # the same subtraction a second time and recovers exactly
            # ``sum(max(0, input_i - read_i - write_i))`` — the per-receipt
            # floors are already baked in, so the second pass cannot floor a
            # sum that a single malformed row drove negative.
            plain = max(0, plain - read - written) + read + written
        component = raw_component.model_copy(
            update={
                "input_tokens": plain,
                "cache_read_tokens": read,
                "cache_write_tokens": written,
                "cache_write_5m_tokens": _usage_field(raw_component, "cache_write_5m_tokens"),
                "cache_write_1h_tokens": _usage_field(raw_component, "cache_write_1h_tokens"),
                "output_tokens": _usage_field(raw_component, "output_tokens"),
                "reasoning_tokens": _usage_field(raw_component, "reasoning_tokens"),
                # A rejected price must not survive into the folded receipt:
                # it would be re-summed and could become acceptable.
                "usd_cost": accepted,
                "estimated_usd_cost": estimated,
            }
        )
        mode = (
            "reported"
            if accepted is not None
            else "estimated" if estimated is not None else "unknown"
        )
        key = (provider, model_id, mode)
        existing = folded.get(key)
        if existing is None:
            folded[key] = component.model_copy(deep=True)
            order.append(key)
            continue
        folded[key] = existing.model_copy(
            update={
                "input_tokens": existing.input_tokens + component.input_tokens,
                "output_tokens": existing.output_tokens + component.output_tokens,
                "cache_read_tokens": existing.cache_read_tokens + component.cache_read_tokens,
                "cache_write_tokens": existing.cache_write_tokens + component.cache_write_tokens,
                "cache_write_5m_tokens": (
                    existing.cache_write_5m_tokens + component.cache_write_5m_tokens
                ),
                "cache_write_1h_tokens": (
                    existing.cache_write_1h_tokens + component.cache_write_1h_tokens
                ),
                "reasoning_tokens": existing.reasoning_tokens + component.reasoning_tokens,
                "usd_cost": (
                    None
                    if existing.usd_cost is None
                    else existing.usd_cost + (component.usd_cost or 0.0)
                ),
                "estimated_usd_cost": (
                    None if estimated is None else (existing.estimated_usd_cost or 0.0) + estimated
                ),
                # Occupancy is a level, not a sum: the newest reading wins, the
                # same rule ``_aggregate_usage`` applies.
                "context_tokens": (
                    component.context_tokens
                    if component.context_tokens is not None
                    else existing.context_tokens
                ),
                # Nested components are already folded into the buckets above;
                # keeping them would double-count on the next fold.
                "cost_components": [],
            }
        )
    return [folded[key] for key in order] + passthrough


def _capped_components(components: Sequence[Any]) -> list[Any]:
    """The newest :data:`USAGE_COMPONENT_CAP` receipts, oldest evicted first.

    Mirrors ``AsyncJob.trajectory``'s eviction discipline (newest-wins, drop
    from the front) so the two unbounded-per-turn lists in this module behave
    the same way. Returns a new list; callers rebuild rather than mutate,
    because ``FrontendSessionState`` is replaced wholesale by ``mutate``.
    """
    values = list(components)
    if len(values) <= USAGE_COMPONENT_CAP:
        return values
    return values[len(values) - USAGE_COMPONENT_CAP :]


def _inherited_identity_fixups(state: Any, session_id: str) -> dict[str, Any]:
    """Fields a restored checkpoint must NOT be allowed to carry across a fork.

    A fork copies ``transcript.jsonl`` verbatim (``fork.py``), and every
    ``frontend_state_checkpoint_v1`` row in it was written by the PARENT — so
    the newest checkpoint a fork restores is stamped with the parent's
    identity, and down a fork chain with whatever the previous hop was
    already serving (the grandparent's, observed in #573). The directory a
    session runs in is authoritative for who it is; a status row it inherited
    is not. Without this the fork's runtime served the parent's ``session_id``
    in every ``frontend_sync``, ``RemoteSession._install_frontend`` refused
    the frame ("frontend state belongs to another session"), and the fork was
    permanently un-attachable: the switched-to fork's own viewer never got a
    state install, so ``/model`` appeared to do nothing and the band never
    painted its context (#573).

    ``session_id`` is always re-stamped. The other two are corrected only when
    the checkpoint is provably someone else's — a same-session resume keeps
    them, because they are then this session's own history:

    * ``checkpoint_id`` names the parent's last status row. It self-heals on
      the fork's first turn end, but until then a reconciling reader would
      match it against a row this transcript did write for the parent.
    * ``jobs`` are the parent's children. ``subagent-roster.v1.json`` is on
      ``fork.EXCLUDED_SIDECARS`` precisely so a fork does not list children it
      cannot address (their comms registry belongs to the parent's process),
      and the checkpoint smuggled equivalent rows past that exclusion.
    """
    fixups: dict[str, Any] = {"session_id": session_id}
    inherited = str(getattr(state, "session_id", "") or "")
    if inherited and inherited != session_id:
        fixups["checkpoint_id"] = None
        fixups["jobs"] = []
    return fixups


# Commands whose effect belongs to the process drawing the widgets. Every other
# advertised slash is routed to the authoritative session owner; keeping this a
# complement means adding a command without classification fails the test rather
# than silently acquiring follower-local behavior.
_FRONTEND_LOCAL_SLASHES = {
    "help",
    "exit",
    "clear",
    # The terminal schedules source-bound iterations and owns sidebar focus;
    # each iteration still submits through that conversation's real owner.
    "loop",
    "sidebar",
    # The clipboard is the machine the USER IS SITTING AT, and every part of
    # this command is already here: the transcript it reads is painted by this
    # frontend, the picker it opens is painted by this frontend, and the OSC 52
    # write goes out this terminal. Routed to the owner it would draw a chooser
    # on a screen nobody is looking at and copy onto a host nobody is at, which
    # is the same argument `/theme` and `/settings` make about config.yml.
    "copy",
    "new",
    "reload",
    "update",
    "resume",
    # A fork opens a window on THIS machine and reads THIS machine's config.yml
    # for where to put it — the same argument `/settings` and `/theme` make. On a
    # follower attached to a remote owner, forking must open a window here, not
    # on the owner's host where nobody is sitting.
    "fork",
    "theme",
    # The settings page reads and writes THIS machine's config.yml, exactly
    # like `/theme` and `/search` above it. Routed to the owner it would open
    # against the owner's config and persist a default governing a machine the
    # user is not sitting at — the same rule `/model default` states explicitly
    # when it refuses to run on a follower.
    "settings",
    "provider",
    "search",
    "accounts",
    # Reads config.yml and the local credential COUNTS, and compares the
    # frontend's own effective model — all of it available on a follower, so
    # routing it to the owner would only add a hop.
    "failovers",
    "usage",
    "analytics",
    # Like analytics, reads the shared local ledger for the mirrored current
    # session ID. No owner RPC or new transport payload is needed.
    "session",
    "skills",
    "login",
    "logout",
    # Remote access is enrolled using this frontend computer's OAuth store and
    # service manager; routing to the session owner would expose another host.
    "mobile",
    # FRONTEND-LOCAL because the command hosts a MASKED PASTE, and the user is
    # sitting at this terminal — routing the whole command would raise the
    # paste prompt on the owner's screen, which nobody is looking at. This is
    # the same split bare ``/model`` makes: the interaction is hosted here, the
    # effect lands on the owner.
    #
    # The STORE is emphatically NOT local. ``VariableStore._credentials`` is an
    # in-memory per-process dict, and the reader is ``credential_env()`` inside
    # the `bash` tool, which runs in the OWNER's process. A viewer that stored
    # locally would hold a secret no tool could ever read while telling the
    # model the key exists — a silent failure whose workaround is pasting the
    # secret into the chat, which is the exact leak this feature prevents. So
    # ``_cmd_credential`` keeps the prompt here and routes every store/forget
    # over the dedicated ``credential`` op to the owner's store.
    "credential",
    # The overlay is local UI; its provider request crosses the authoritative
    # complete_aside operation on RemoteSession.
    "btw",
    # The kill switch is the one session command that must act from THIS
    # process even on a follower: bare /stop ends the session the viewer is
    # looking at (the owner's runtime via the direct stop op, not a routed
    # slash), `/stop <target>` and `/stop all` enumerate THIS machine's
    # registry — a different machine's registry than the owner's is exactly
    # the point of stopping from here. Routing it to the owner would stop the
    # OWNER's neighbours, not the viewer's.
    "stop",
}
# Bare ``/mcp`` renders the canonical server list locally, but its grant
# subcommands mutate OAuth state that lives on the authoritative owner — the
# follower's MCP facade is a read-only snapshot with no config accessor, so
# routing the mutation (not faking it locally) is the only non-crashing,
# non-divergent answer. The dispatch splits the two shapes by argument.
#
# Aliased from the module that IMPLEMENTS these verbs rather than restated:
# two literal copies of the set is how a fourth verb ends up routed by one
# half of the dispatch and refused by the other.
_MCP_GRANT_SUBCOMMANDS = frozenset(_GRANT_SUBCOMMANDS)
#: Every verb ``/mcp`` accepts, in the order the refusal offers them. Canonical
#: HERE rather than on `OperatorApp` because both the terminal and the detached
#: runtime validate against it, and a runtime that knew a shorter list silently
#: swallowed `add`/`remove` as a server listing (round 5, U15).
MCP_SUBCOMMANDS = ("list", "add", "remove", "login", "logout", "reauth")
_IMAGE_SLASHES = {"agent", "team"}


class CommandScope(StrEnum):
    """Where one advertised slash command executes."""

    FRONTEND_LOCAL = "frontend_local"
    AUTHORITATIVE_SESSION = "authoritative_session"
    UNAVAILABLE = "unavailable"


class CostKnowledge(StrEnum):
    """How confidently the cumulative dollar amount is known."""

    UNKNOWN = "unknown"
    EXACT = "exact"
    PARTIAL = "partial"
    FLOOR = "floor"


class SlashCapability(BaseModel):
    model_config = ConfigDict(extra="allow")

    command: str
    scope: CommandScope
    operation: str | None = None
    reason: str | None = None
    supports_images: bool = False


class SlashResult(BaseModel):
    """The typed outcome of one slash command run on the authoritative owner.

    The v5 replacement for the synthetic ``ran /…`` receipt: the owner runs a
    shared slash command and returns WHAT happened as data, so the terminal
    that asked renders it locally instead of the answer painting in another
    process's transcript. Every product-facing string is produced by the
    standard handlers, so a follower's ``/goal``, ``/rename``, ``/mcp login``
    or ``/context`` says exactly what a local session would — there is no
    attach-specific vocabulary anywhere in the fields.

    ``kind`` is one of ``notice`` (the invoker prints ``text`` through the
    normal notice path), ``block`` (``data`` is a renderable payload the
    follower builds its standard block from), or ``noop`` (nothing to print —
    e.g. a picker the invoker opens itself).
    """

    model_config = ConfigDict(extra="allow")

    kind: str = "notice"
    text: str = ""
    style: str = "info"
    data: dict[str, Any] = Field(default_factory=dict)


class TodoItemState(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str
    status: str = "pending"
    reason: str | None = None


class TodoPhaseState(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = "Todos"
    items: list[TodoItemState] = Field(default_factory=list)


class WakeState(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    message: str
    next_due_at: int
    created_at: int = 0
    every_ms: int | None = None
    remaining: int | None = None


class McpServerState(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    status: str
    error: str | None = None
    tool_count: int | None = None


class PendingGateState(BaseModel):
    model_config = ConfigDict(extra="allow")

    request_id: str
    kind: str
    title: str
    detail: str = ""
    options: list[dict[str, Any]] = Field(default_factory=list)
    secret: bool = False
    question_index: int = 0
    question_total: int = 1


class _FrozenSequence(tuple[Any, ...]):
    """Tuple-backed sequence with list-compatible equality and wire shape."""

    __slots__ = ()

    def __new__(cls, values: Iterable[Any]) -> "_FrozenSequence":
        return tuple.__new__(cls, tuple(values))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sequence):
            return tuple(self) == tuple(other)
        return False

    def __hash__(self) -> int:
        return tuple.__hash__(self)

    def __copy__(self) -> "_FrozenSequence":
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> "_FrozenSequence":
        return self

    def __reduce__(self) -> tuple[type["_FrozenSequence"], tuple[tuple[Any, ...]]]:
        return type(self), (tuple(self),)


class _FrozenMapping(tuple[tuple[str, Any], ...]):
    """Tuple-of-pairs mapping with no writable or rebindable backing storage."""

    __slots__ = ()

    def __new__(cls, values: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> "_FrozenMapping":
        items = values.items() if isinstance(values, Mapping) else values
        return tuple.__new__(cls, tuple(items))

    def __getitem__(self, key: str) -> Any:  # type: ignore[override]
        for candidate, value in tuple.__iter__(self):
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        return (key for key, _value in tuple.__iter__(self))

    def __len__(self) -> int:
        return tuple.__len__(self)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> Iterator[str]:
        return self.__iter__()

    def values(self) -> Iterator[Any]:
        return (value for _key, value in tuple.__iter__(self))

    def items(self) -> Iterator[tuple[str, Any]]:
        return tuple.__iter__(self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _FrozenMapping):
            return tuple(tuple.__iter__(self)) == tuple(tuple.__iter__(other))
        if isinstance(other, Mapping):
            return dict(self.items()) == {key: other[key] for key in other}
        return False

    def __hash__(self) -> int:
        return tuple.__hash__(self)

    def __copy__(self) -> "_FrozenMapping":
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> "_FrozenMapping":
        return self

    def __reduce__(self) -> tuple[type["_FrozenMapping"], tuple[tuple[tuple[str, Any], ...]]]:
        return type(self), (tuple(tuple.__iter__(self)),)


# Virtual registration keeps the runtime mapping contract without inheriting a
# second, incompatible Collection generic beside tuple's pair iteration.
Mapping.register(_FrozenMapping)  # type: ignore[attr-defined]


def _freeze_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    if isinstance(value, (_FrozenSequence, _FrozenMapping)):
        return value
    if isinstance(value, FrontendUsage):
        return _freeze_frontend_usage(value)
    if isinstance(value, Usage):
        return _freeze_usage(value)
    if isinstance(value, BaseModel):
        # Unknown future job extras have no stable attribute contract. Preserve
        # all their data as a recursively immutable mapping rather than sharing
        # a mutable model the older follower cannot know how to freeze safely.
        return _freeze_value(value.model_dump(mode="python"))
    if isinstance(value, MutableMapping):
        return _FrozenMapping({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, bytearray):
        # bytearray is the remaining built-in mutable container accepted by an
        # unconstrained extra; bytes preserves its Pydantic JSON string shape.
        return bytes(value)
    if isinstance(value, memoryview):
        return bytes(value)
    if isinstance(value, (MutableSequence, deque)):
        return _FrozenSequence(_freeze_value(item) for item in value)
    if isinstance(value, MutableSet):
        # ``frozenset`` preserves set equality/membership and Pydantic's JSON
        # serializer emits the same array shape as a mutable set. Nested values
        # are frozen first so a hashable mutable wrapper cannot retain an alias.
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_freeze_value(item) for item in value)
    # Closed boundary: Pydantic is the wire serializer, so anything beyond the
    # known fast paths is admitted only if that serializer can reduce it to JSON
    # values. Dataclasses/namedtuples become mappings/lists; datetime/path/enums
    # become scalars. Arbitrary __dict__/slots objects are rejected rather than
    # shared into canonical state with an unknowable mutation surface.
    try:
        serialized = to_jsonable_python(value)
    except PydanticSerializationError as exc:
        raise TypeError(
            f"unsupported canonical frontend value: {type(value).__qualname__}"
        ) from exc
    if serialized is value:
        raise TypeError(f"unsupported canonical frontend value: {type(value).__qualname__}")
    return _freeze_value(serialized)


def _is_frozen_value(value: Any) -> bool:
    if isinstance(value, (_FrozenSequence, _FrozenMapping, _FrozenUsage, _FrozenFrontendUsage)):
        return True
    if isinstance(value, (str, int, float, bool, bytes, type(None), frozenset)):
        return True
    if isinstance(value, tuple):
        return all(_is_frozen_value(item) for item in value)
    return False


def _wire_value(value: Any) -> Any:
    """Return ordinary JSON containers at the serialization boundary only."""
    if isinstance(value, _FrozenSequence):
        return [_wire_value(item) for item in value]
    if isinstance(value, _FrozenMapping):
        return {key: _wire_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_wire_value(item) for item in value]
    if isinstance(value, frozenset):
        return [_wire_value(item) for item in value]
    if isinstance(value, BaseModel):
        return _thaw_model(value)
    return value


def _job_summary(job: "JobState") -> dict[str, Any]:
    """Serialize one roster row without visiting its retained trajectory."""
    values = {
        name: _wire_value(value)
        for name, value in job.__dict__.items()
        if name not in {"trajectory", "todos"}
    }
    values.update({name: _wire_value(value) for name, value in (job.model_extra or {}).items()})
    return to_jsonable_python(values)


def _bound_launch_prompts_across_jobs(jobs: list[Any]) -> None:
    """Spend one frame's collapsed-attempt text budget across the roster.

    See :data:`JOB_LAUNCH_PROMPTS_FRAME_BUDGET_CHARS`. Rows are served in roster
    order until the budget is gone, then later rows keep every KEY and replace
    its text with :data:`LAUNCH_PROMPT_ELIDED_PLACEHOLDER`.

    Keeping the keys is load-bearing, not tidiness. Emptying the map was the
    round-1 MAJOR: reconciliation matches a durable row BY KEY, so a starved
    row's collapsed attempts fell back to plain user rows carrying the full
    preamble — the very duplicate this change exists to remove. The CURRENT
    launch survived either way, because the view re-derives it from the job's
    own ``prompt``, which is exactly why the original reasoning looked sound and
    the defect only appeared on a resumed child.

    A whole row's map is elided rather than part of it: a half-served map would
    reconcile some of a child's collapsed attempts and leak the others' preamble,
    which reads as a bug rather than as a bounded preview.
    """
    spent = 0
    # Degradation is MONOTONIC: once the budget cannot afford a row's full text,
    # no later row gets full text either. Deciding each row independently let a
    # cheap row after an expensive one pass at tier 1 while its predecessors
    # were elided, so two children with the same history rendered differently
    # depending on roster position — and, because a tier-3 row frees no budget,
    # a starved row could be followed by an unstarved one. Order-independence
    # matters more than squeezing the last row in.
    degraded = False
    for job in jobs:
        if not isinstance(job, dict):
            continue
        prompts = job.get("launch_prompts")
        if not isinstance(prompts, dict) or not prompts:
            continue
        # Re-bounded HERE as well as in ``_with_lineage``. A ``JobState`` can be
        # built directly — a restored reader row, a test, an owner that never
        # went through the comms path — so the wire boundary cannot assume the
        # construction-time bound ran. Cheap: the common row has no entries.
        prompts = _wire_launch_prompts(prompts)
        if not prompts:
            job["launch_prompts"] = {}
            continue
        job["launch_prompts"] = prompts
        cost = sum(len(str(key)) + len(str(value)) for key, value in prompts.items())
        if not degraded and spent + cost <= JOB_LAUNCH_PROMPTS_FRAME_BUDGET_CHARS:
            spent += cost
            continue
        degraded = True
        # Tier 2: keep the KEYS, elide the text. This is what preserves
        # reconciliation for collapsed attempts (the round-1 MAJOR), and it is
        # ~49 bytes an entry rather than ~249.
        elided = dict.fromkeys(prompts, LAUNCH_PROMPT_ELIDED_PLACEHOLDER)
        elided_cost = sum(len(str(key)) + len(LAUNCH_PROMPT_ELIDED_PLACEHOLDER) for key in elided)
        if spent + elided_cost <= JOB_LAUNCH_PROMPTS_FRAME_BUDGET_CHARS:
            job["launch_prompts"] = elided
            # Charged, because keys are not free: leaving an elided map
            # unbilled would let a long tail of starved rows spend unbounded key
            # bytes after the budget is nominally gone — the same "bounded per
            # row, unbounded in total" defect one level down.
            spent += elided_cost
            continue
        # Tier 3: not even the keys fit, so the map goes — but WITHOUT losing
        # reconciliation, because the keys are themselves derivable. Every
        # collapsed attempt's identity is ``subagent-launch:<alias>`` over
        # ``attempt_aliases``, which already rides this row and is already
        # bounded (one short id per attempt). The follower rebuilds the elided
        # entries from it in ``SnapshotSubagentComms._node_for``, so a tier-3
        # row still folds every attempt to a placeholder rather than leaking the
        # preamble. Dropping the map here therefore costs presentation detail,
        # never the fix.
        job["launch_prompts"] = {}


#: The deterministic launch-row identity ``run_subagent`` mints for a child.
#:
#: ``harness/subagent.py`` builds BOTH the recorded ``job.launch_message_id``
#: and the ``message_id`` it prompts the child under from this one f-string, so
#: for every child launched by this runtime the identity is a pure function of
#: the job id.
LAUNCH_MESSAGE_ID_PREFIX = "subagent-launch:"


def _derived_launch_message_id(job_id: str) -> str:
    """The launch identity a job id implies, or ``""`` without one."""
    return f"{LAUNCH_MESSAGE_ID_PREFIX}{job_id}" if job_id else ""


def _restore_elided_launch_prompts(job: JobState) -> dict[str, str]:
    """A follower's launch map, with budget-elided attempts restored.

    The wire map is authoritative for everything it carries. What it may be
    missing is whole ENTRIES for collapsed attempts, dropped when the roster's
    shared text budget could not even afford their keys
    (:func:`_bound_launch_prompts_across_jobs`, tier 3).

    Those keys are recoverable rather than lost: an attempt's launch identity is
    ``subagent-launch:<its job id>``, and ``attempt_aliases`` — already on this
    row, already bounded at one short id per attempt — lists exactly the ids
    #314 collapsed into this record. Rebuilding them keeps reconciliation
    matching every durable launch row, which is the whole point of the mapping:
    a missing key renders that attempt's full role/team/system preamble.

    The value is the same placeholder tier 2 sends, because the text really is
    gone; only the identity is reconstructible. The current launch is not
    handled here — ``SubagentView.show`` re-derives it from the job's own
    ``prompt``, which is richer than a placeholder.
    """
    restored = dict(job.launch_prompts or {})
    for alias in job.attempt_aliases:
        identity = _derived_launch_message_id(str(alias or ""))
        # Never overwrite: a wire entry carries real text, this only fills gaps.
        if identity and identity not in restored:
            restored[identity] = LAUNCH_PROMPT_ELIDED_PLACEHOLDER
    return restored


def _bound_launch_ids_across_jobs(jobs: list[Any]) -> None:
    """Spend one frame's non-derivable-identity budget across the roster.

    Run AFTER :func:`_elide_derivable_launch_id_in_place`, so it only ever sees
    the ids that genuinely have to ride the wire. See
    :data:`JOB_LAUNCH_IDS_FRAME_BUDGET_CHARS`: the derivable case is free, but
    the literal case was unbounded, and a per-row scalar times roster depth is
    what cost 43 rows of attach headroom in round 1.

    Truncation is not an option — a clipped identity matches no durable row and
    silently stops reconciling — so a row past the budget loses the field
    entirely and degrades to pre-#681 behaviour for that one child.
    """
    spent = 0
    for job in jobs:
        if not isinstance(job, dict):
            continue
        value = job.get("launch_message_id")
        if not isinstance(value, str) or not value:
            continue
        if spent + len(value) > JOB_LAUNCH_IDS_FRAME_BUDGET_CHARS:
            del job["launch_message_id"]
            continue
        spent += len(value)


def _elide_derivable_launch_id_in_place(job: dict[str, Any]) -> None:
    """Drop ``launch_message_id`` when the job id already implies it.

    Round-1 BLOCKER. Unlike the prompts, this field is a per-row SCALAR that
    every task child carries, so nothing bounded it and nothing could: an
    identity must never be truncated — a clipped key matches no durable row and
    would silently stop reconciling. QA measured the cost at 46.7 B/row, which
    dropped the maximum attachable roster from 812 rows on v0.49.2 to 769 on the
    first version of this change: a session that attaches on the shipped release
    and does not attach here, which is the exact failure the wire stripping
    exists to prevent.

    Omission rather than truncation is what makes it safe. The value is
    reconstructible (:func:`_derived_launch_message_id`), so the follower
    rebuilds the identical string in :meth:`SnapshotSubagentComms._node_for` and
    reconciliation is byte-for-byte unchanged — this trades ~47 bytes a row for
    one f-string per node rebuild.

    Only an EQUAL value is dropped. A row whose id is not the derived one — a
    resumed child rebuilt from a persisted comms row, where the recorded
    identity belongs to an earlier attempt — keeps its literal value, because
    there the id genuinely carries information the job id does not.
    """
    value = job.get("launch_message_id")
    if isinstance(value, str) and value == _derived_launch_message_id(str(job.get("id") or "")):
        # Deleted outright, with NO replacement marker. An earlier revision
        # stamped a ``launch_id_derived`` bool here so the follower could tell
        # "elided" from "never had one" — but that bool was itself 28 B on every
        # row, which is the same per-row cost the elision exists to remove, and
        # it cost 17 rows of attach ceiling.
        #
        # The distinction is already free: only ``type == "task"`` jobs ever get
        # a launch turn (``run_subagent`` is the sole minting site; bash and
        # eval jobs register as ``"bash"``), so the follower re-derives for task
        # rows only and a bash row keeps the empty string it always had.
        del job["launch_message_id"]


def _drop_absent_launch_fields_in_place(job: dict[str, Any]) -> None:
    """Omit the launch-reconciliation keys from a row that has no launch.

    An absent fact must not buy wire bytes. These two keys are empty on every
    bash job and on every child that was never resumed, and at roster scale the
    empty values alone were measured at ~9 KB of the 1 MiB line budget — enough
    to push the ``ran all year`` class guard over the limit on their own, with
    no information in them at all.

    Omission is exactly equivalent to sending the empty values: ``JobState``
    defaults both, a delta rebuilds each row by revalidating the raw dict rather
    than merging it onto the prior row (``apply_update``), and a follower
    reading neither key takes the same degrade path as one attached to an owner
    that predates the fields. So this is a pure byte saving, not a semantic one.

    Applied at BOTH wire boundaries — the delta assembly in ``mutate`` and the
    attach snapshot in :func:`sync_wire_payload` — because the two serialize job
    rows by different routes and a saving in one does not reach the other.
    """
    if not job.get("launch_message_id"):
        job.pop("launch_message_id", None)
    if not job.get("launch_prompts"):
        job.pop("launch_prompts", None)


def _jobs_equal(current: Sequence["JobState"], candidate: Sequence["JobState"]) -> bool:
    """Compare changed rows only; stable refreshes reuse frozen job identities."""
    return len(current) == len(candidate) and all(
        old is new or old == new for old, new in zip(current, candidate)
    )


def _freeze_job(job: "JobState") -> "JobState":
    """Detach the owning model and freeze every nested canonical value."""
    values = {
        name: _freeze_value(value)
        for name, value in job.__dict__.items()
        if name not in {"usage", "descendant_usage"}
    }
    values["usage"] = _freeze_usage(job.usage) if job.usage is not None else None
    values["descendant_usage"] = _FrozenSequence(
        _freeze_frontend_usage(component) for component in job.descendant_usage
    )
    for name, value in (job.model_extra or {}).items():
        values[name] = _freeze_value(value)
    return job.model_copy(update=values)


class JobState(BaseModel):
    """Read-only job shape used by the existing job widgets.

    ``extra='allow'`` is intentional: retained trajectory fields can grow without
    forcing an older follower to reject a newer owner. Unknowns stay attached to
    the DTO and survive a round trip rather than being discarded.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    id: str
    type: str
    status: str = "running"
    queued: bool = False
    label: str = ""
    agent: str = ""
    intent: str = ""
    latest_details: dict[str, Any] | str | None = None
    error_text: str = ""
    result_text: str = ""
    model_label: str | None = None
    context_window: int | None = None
    usage: Usage | None = None
    # None knowledge marks old owners, which still need the legacy pricing path.
    # Explicit UNKNOWN forbids viewers from discovering/pricing independently.
    direct_cost: float | None = None
    direct_cost_knowledge: CostKnowledge | None = None
    start_time: float = 0.0
    started_at: float | None = None
    settled_at: float | None = None
    trajectory: list[dict[str, Any]] = Field(default_factory=list)
    #: How many events the OWNER retains for this job, independent of how many
    #: ride this particular frame. The attach snapshot omits trajectories
    #: entirely (see :func:`_job_roster_row`) because a busy session's retained
    #: events exceed the socket's ``_MAX_LINE_BYTES`` and made the session
    #: unopenable; a viewer therefore needs the COUNT before it has the rows,
    #: to say "loading 500 events" rather than "no activity". Owner-side this
    #: is always ``len(trajectory)``; follower-side it stays the owner's number
    #: even while the local list is empty or a partial page.
    trajectory_length: int = 0
    # Nested spend (#297): a finished grandchild's usage folds into its root's
    # row here. Without carrying it, follower-side child-cost pricing counted
    # only the direct child while the owner priced the whole subtree.
    descendant_usage: list[FrontendUsage] = Field(default_factory=list)
    prompt: str | None = None
    agent_role: str | None = None
    effort: str | None = None
    output_tail: str = ""
    output_seq: int = 0
    restored: bool = False
    # Canonical lineage (U5): the owner's subagent-comms tree is not itself
    # serializable, but its one fact — who launched whom — is. Stamping the
    # parent's job id (and the child's session/role for the page header) lets
    # a follower rebuild the full parent/peer/child graph from ``state.jobs``
    # alone, so the hierarchy keys navigate the authoritative structure rather
    # than silently doing nothing.
    parent_job_id: str | None = None
    session_id: str | None = None
    #: The child's durable session directory, as a string (``Path`` is not
    #: JSON-native). Projected here for the same reason ``session_id`` is:
    #: the owner's ``SubagentComms`` registry never crosses the socket, and
    #: the full-page view lazy-loads the child's ``transcript.jsonl`` through
    #: ``comms.session_dir_of(job_id)``. A follower with no directory treated
    #: every child as "no saved transcript" and could show only the 500-event
    #: in-memory trajectory window of a child that had hours of history on
    #: disk. ``None`` from an owner that predates this field is tolerated —
    #: ``SnapshotSubagentComms`` derives the path from ``session_id`` then.
    session_dir: str | None = None
    #: The deterministic ``subagent-launch:<job_id>`` id of the CURRENT launch
    #: turn, and every launch identity in this lineage mapped to its concise
    #: authored prompt.
    #:
    #: These ride for the same reason ``session_dir`` does, and they are only
    #: load-bearing BECAUSE it does. The full-page view folds the durable launch
    #: turn into the synthetic prompt head by matching a durable row's key
    #: against these (``SubagentView._chronological_entries``); with them empty
    #: it cannot correlate the two and renders BOTH — and the durable copy is
    #: the full role/team/system preamble, not a short line. That was inert
    #: until a follower loaded durable history at all, which is what #669
    #: shipped, so the visible duplicate arrived with it (its review Q1 /
    #: round-1 item 6, deferred there and fixed here).
    #:
    #: ``launch_prompts`` is bounded before it is stamped — see
    #: :data:`JOB_LAUNCH_PROMPT_WIRE_CHARS` for why the bound cannot live at the
    #: wire boundary. It stays a MAPPING rather than collapsing to the current
    #: id: a resumed child's transcript holds one launch turn per collapsed
    #: attempt, and reconciling only the newest leaks every earlier attempt's
    #: preamble (the exact defect owner-side review round 4 R4-1 fixed).
    #:
    #: Both are absent from an owner that predates them (``extra='allow'``), and
    #: a follower missing them degrades to rendering the synthetic head plus the
    #: unreconciled durable row — today's behaviour, not a crash.
    launch_message_id: str = ""
    launch_prompts: dict[str, str] = Field(default_factory=dict)
    attempt_aliases: list[str] = Field(default_factory=list)
    # Child plans are detail payloads, not roster metadata. None is unavailable;
    # an empty list is an authoritative clear and must never restore a stale plan.
    todos: list[dict[str, Any]] | None = None

    @model_validator(mode="before")
    @classmethod
    def _derive_trajectory_length(cls, data: Any) -> Any:
        """Default the retained-event COUNT to the rows the job was built with.

        Only when the caller supplied none. An explicit value is always kept,
        because on a FOLLOWER the two legitimately disagree: the wire snapshot
        carries the owner's count with an empty list (the rows do not fit the
        frame), and a watched job accumulates only the appends seen since its
        page opened. Deriving unconditionally would replace the owner's real
        number with the length of whatever partial window happens to be local.
        """
        if isinstance(data, dict) and "trajectory_length" not in data:
            trajectory = data.get("trajectory")
            if isinstance(trajectory, Sequence) and not isinstance(trajectory, (str, bytes)):
                data = {**data, "trajectory_length": len(trajectory)}
        return data

    @model_serializer(mode="wrap")
    def _serialize_frozen_values(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # Pydantic's field serializers expect concrete lists/dicts. Thaw only
        # the ephemeral wire copy; canonical state keeps the immutable wrappers.
        mutable = self.model_copy(
            update={name: _wire_value(value) for name, value in self.__dict__.items()}
        )
        if self.model_extra:
            mutable.__pydantic_extra__ = {
                name: _wire_value(value) for name, value in self.model_extra.items()
            }
        return handler(mutable)

    @classmethod
    def from_job(cls, job: Any) -> "JobState":
        trajectory = []
        for event in list(getattr(job, "trajectory", None) or []):
            if hasattr(event, "model_dump"):
                trajectory.append(event.model_dump(mode="json"))
            elif isinstance(event, dict):
                trajectory.append(copy.deepcopy(event))
        details = getattr(job, "latest_details", None)
        if isinstance(details, dict):
            details = copy.deepcopy(details)
        elif details is not None and not isinstance(details, str):
            details = {"progress": str(details)}
        usage = getattr(job, "usage", None)
        if isinstance(usage, dict):
            usage = Usage.model_validate(usage)
        direct_cost, unknown = cost_summary(
            (usage.cost_components or [usage]) if usage is not None else [],
            model_label=getattr(job, "model_label", None) or "",
        )
        descendants = []
        for component in list(getattr(job, "descendant_usage", None) or []):
            if isinstance(component, dict):
                descendants.append(FrontendUsage.model_validate(component))
            elif isinstance(component, Usage):
                descendants.append(FrontendUsage.model_validate(component.model_dump(mode="json")))
        return cls(
            id=str(getattr(job, "id", "") or ""),
            type=str(getattr(job, "type", "") or ""),
            status=str(getattr(job, "status", "running") or "running"),
            queued=bool(getattr(job, "queued", False)),
            label=str(getattr(job, "label", "") or getattr(job, "agent", "") or ""),
            agent=str(getattr(job, "agent", "") or ""),
            intent=str(getattr(job, "intent", "") or ""),
            latest_details=details,
            error_text=str(getattr(job, "error_text", "") or getattr(job, "error", "") or ""),
            result_text=str(getattr(job, "result_text", "") or getattr(job, "result", "") or ""),
            model_label=getattr(job, "model_label", None),
            context_window=getattr(job, "context_window", None),
            usage=usage,
            direct_cost=direct_cost,
            direct_cost_knowledge=_cost_knowledge(direct_cost, unknown),
            start_time=float(
                getattr(job, "start_time", 0.0)
                or getattr(job, "started_at", 0.0)
                or getattr(job, "created_at", 0.0)
                or 0.0
            ),
            started_at=getattr(job, "started_at", None),
            settled_at=getattr(job, "settled_at", None) or getattr(job, "finished_at", None),
            trajectory=trajectory,
            trajectory_length=len(trajectory),
            descendant_usage=descendants,
            prompt=getattr(job, "prompt", None),
            # Recorded on the job at registration beside ``prompt``
            # (``harness/subagent.py``), so it survives for a child whose comms
            # record has been swept: ``_with_lineage`` can then stamp nothing,
            # and this is the only copy of the launch identity left. That is
            # enough on its own for a never-resumed child — the view derives
            # ``{launch_message_id: prompt}`` itself in ``SubagentView.show``.
            launch_message_id=str(getattr(job, "launch_message_id", "") or ""),
            agent_role=getattr(job, "agent_role", None),
            effort=getattr(job, "effort", None),
            output_tail=str(getattr(job, "output_tail", "") or ""),
            output_seq=int(getattr(job, "output_seq", 0) or 0),
            restored=bool(getattr(job, "restored", False)),
        )


class FrontendModelSpec(ModelSpec):
    """Wire model spec that preserves fields introduced by newer owners."""

    model_config = ConfigDict(extra="allow")


class FrontendUsage(Usage):
    """Lossless wire usage, including future cost component metadata."""

    model_config = ConfigDict(extra="allow")


class _FrozenUsage(Usage):
    """Usage-compatible immutable value retained inside a shared job snapshot."""

    model_config = ConfigDict(extra="allow", frozen=True)

    @model_serializer(mode="wrap")
    def _serialize_frozen_values(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return handler(_thaw_model(self))


class _FrozenFrontendUsage(FrontendUsage):
    """FrontendUsage-compatible immutable descendant accounting value."""

    model_config = ConfigDict(extra="allow", frozen=True)

    @model_serializer(mode="wrap")
    def _serialize_frozen_values(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return handler(_thaw_model(self))


def _thaw_model(value: BaseModel) -> BaseModel:
    mutable = value.model_copy(
        update={name: _wire_value(item) for name, item in value.__dict__.items()}
    )
    if value.model_extra:
        mutable.__pydantic_extra__ = {
            name: _wire_value(item) for name, item in value.model_extra.items()
        }
    return mutable


def _freeze_usage(value: Usage) -> _FrozenUsage:
    frozen = _FrozenUsage.model_validate(value.model_dump(mode="python"))
    updates = {name: _freeze_value(item) for name, item in frozen.__dict__.items()}
    for name, item in (frozen.model_extra or {}).items():
        updates[name] = _freeze_value(item)
    return frozen.model_copy(update=updates)


def _freeze_frontend_usage(value: FrontendUsage) -> _FrozenFrontendUsage:
    frozen = _FrozenFrontendUsage.model_validate(value.model_dump(mode="python"))
    updates = {name: _freeze_value(item) for name, item in frozen.__dict__.items()}
    for name, item in (frozen.model_extra or {}).items():
        updates[name] = _freeze_value(item)
    return frozen.model_copy(update=updates)


class FrontendSessionState(BaseModel):
    """Versioned JSON-safe source of truth for one standard terminal UI."""

    model_config = ConfigDict(extra="allow")

    state_version: int = FRONTEND_STATE_VERSION
    session_id: str
    epoch: str
    sequence: int = 0
    checkpoint_id: str | None = None
    cwd: str = ""
    conversation_title: str = ""
    conversation_title_user_set: bool = False
    #: True while the session is a FORK still wearing its parent's title. The
    #: band/tab tag rides on the SAME refresh as the name so a rename that
    #: clears the session flag reaches every frontend in the same frame —
    #: without it a follower's snapshot could not heal a stale `[fork]` tab
    #: (review round 1, R1). `extra="allow"` keeps older readers tolerant of
    #: the new field, and `getattr` at the source keeps a reduced facade
    #: (embedded SDK, test double) from failing the whole refresh.
    conversation_title_forked: bool = False
    goal: str = ""
    active_agent: str = ""
    active_team: str = ""
    selected_model: FrontendModelSpec | None = None
    effective_model: FrontendModelSpec | None = None
    last_usage: FrontendUsage | None = None
    usage_components: list[FrontendUsage] = Field(default_factory=list)
    context_tokens: int | None = None
    context_is_estimate: bool | None = None
    context_window: int | None = None
    context_breakdown: dict[str, int] | None = None
    cumulative_parent_cost: float | None = None
    child_costs: dict[str, float] = Field(default_factory=dict)
    # Whole manager ledger, not a sum of visible rows: live descendants, folded
    # attempts and retained-away jobs all belong here exactly once. Optional
    # knowledge distinguishes legacy checkpoints from an explicitly empty ledger.
    subagent_cost: float | None = None
    subagent_cost_knowledge: CostKnowledge | None = None
    cost_knowledge: CostKnowledge = CostKnowledge.UNKNOWN
    streaming: bool = False
    generation: int = 0
    #: How the last logical turn ended, for a viewer that dropped mid-turn and
    #: rebinds after it settled. ``live_events`` is emptied at ``agent_end``, so
    #: the real end is not in the snapshot; without this a rebind cannot tell
    #: aborted from completed and would synthesise ``aborted=True`` (today's
    #: false "interrupted"). ``""`` is the wire default and the old-runtime
    #: value — treat as aborted. Additive; extra="allow" keeps older readers
    #: tolerant. One value per user prompt, not per compaction continuation.
    last_turn_outcome: Literal["completed", "aborted", "error", ""] = ""
    activity_started_at: float | None = None
    active_duration_s: float = 0.0
    current_turn_accrued_cost: float = 0.0
    queued_steering: list[dict[str, Any]] = Field(default_factory=list)
    # Bounded transient seed for a frontend that joins mid-turn. Existing
    # frontends consume raw events; only the atomic snapshot needs this fold.
    live_events: list[dict[str, Any]] = Field(default_factory=list)
    jobs: list[JobState] = Field(default_factory=list)
    todos: list[TodoPhaseState] = Field(default_factory=list)
    wakes: list[WakeState] = Field(default_factory=list)
    mcp_servers: list[McpServerState] = Field(default_factory=list)
    mcp_startup: dict[str, Any] | None = None
    loop: dict[str, Any] | None = None
    pending_gate: PendingGateState | None = None
    slash_capabilities: list[SlashCapability] = Field(default_factory=list)
    # The owner's provider-catalogue rows, so an attached terminal's bare
    # ``/model`` picker lists the models the SESSION can actually switch to
    # (owner credentials/aggregators), never the follower's own possibly-
    # credential-less registry (D3, review round 2). Bounded to the direct
    # (non-aggregator) rows; a follower's current model and its own live
    # refresh stay authoritative for their own rows.
    model_catalogue: list[dict[str, Any]] = Field(default_factory=list)
    #: True when the frame could not carry every catalogue row and
    #: ``model_catalogue`` is a prefix of the owner's list rather than all of
    #: it (see :data:`MODEL_CATALOGUE_FLOOR_ROWS`).
    #:
    #: Set at the WIRE boundary, not at accumulation: the owner's state holds
    #: the whole catalogue and only a frame that cannot fit clips, so this
    #: describes the copy the reader was actually sent. A reader that shows a
    #: clipped list as if it were complete is the failure this exists to
    #: prevent — the picker would say "these are your models" while silently
    #: omitting hundreds of them.
    model_catalogue_truncated: bool = False
    history_cursor: str | None = None
    history_generation: int = 0
    attachment_root: str | None = None
    # Durable receipts are independent of owner epoch and pending action gates.
    attention: dict[str, Any] = Field(default_factory=dict)

    @model_serializer(mode="wrap")
    def _serialize_frozen_jobs(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        mutable = self.model_copy(update={"jobs": list(self.jobs)})
        return handler(mutable)

    @field_validator("selected_model", "effective_model", mode="before")
    @classmethod
    def _model_wire(cls, value: Any) -> Any:
        return value.model_dump(mode="json") if isinstance(value, ModelSpec) else value

    @field_validator("last_usage", mode="before")
    @classmethod
    def _usage_wire(cls, value: Any) -> Any:
        return value.model_dump(mode="json") if isinstance(value, Usage) else value

    @field_validator("usage_components", mode="before")
    @classmethod
    def _usage_components_wire(cls, value: Any) -> Any:
        return [
            item.model_dump(mode="json") if isinstance(item, Usage) else item
            for item in value or []
        ]

    @property
    def cumulative_cost(self) -> float | None:
        children = (
            self.subagent_cost
            if self.subagent_cost_knowledge is not None
            else sum(self.child_costs.values()) if self.child_costs else None
        )
        if self.cumulative_parent_cost is None and children is None:
            return None
        return float(self.cumulative_parent_cost or 0.0) + (children or 0.0)

    @property
    def cumulative_cost_knowledge(self) -> CostKnowledge:
        if self.cost_knowledge in {CostKnowledge.PARTIAL, CostKnowledge.FLOOR}:
            return self.cost_knowledge
        if self.subagent_cost_knowledge == CostKnowledge.PARTIAL:
            return CostKnowledge.PARTIAL
        if self.subagent_cost_knowledge == CostKnowledge.UNKNOWN and self.child_costs:
            return CostKnowledge.PARTIAL
        return self.cost_knowledge

    @property
    def model_label(self) -> str:
        spec = self.selected_model
        return f"{spec.provider}/{spec.model_id}" if spec is not None else ""

    @property
    def effective_model_label(self) -> str:
        spec = self.effective_model or self.selected_model
        return f"{spec.provider}/{spec.model_id}" if spec is not None else ""


def _freeze_state_jobs(
    state: FrontendSessionState, *, jobs_are_canonical: bool = False
) -> FrontendSessionState:
    """Detach incoming owners, or preserve already-owned jobs on scalar updates."""
    jobs = state.jobs if jobs_are_canonical else (_freeze_job(job) for job in state.jobs)
    return state.model_copy(update={"jobs": _FrozenSequence(jobs)})


def _public_job(job: JobState) -> JobState:
    """Detach every Pydantic owner while sharing immutable retained payloads."""
    values = {
        name: copy.deepcopy(value) if isinstance(value, BaseModel) else value
        for name, value in job.__dict__.items()
        if name != "descendant_usage"
    }
    values["descendant_usage"] = _FrozenSequence(
        copy.deepcopy(component) for component in job.descendant_usage
    )
    detached = job.model_copy(update=values)
    if job.model_extra:
        detached.__pydantic_extra__ = {
            name: copy.deepcopy(value) if isinstance(value, BaseModel) else value
            for name, value in job.model_extra.items()
        }
    return detached


class FrontendSync(BaseModel):
    model_config = ConfigDict(extra="allow")

    state_version: int = FRONTEND_STATE_VERSION
    epoch: str
    sequence: int
    snapshot: FrontendSessionState
    live_cursor: str | None = None
    display_history: DisplayHistoryWindow | None = None


class FrontendUpdate(BaseModel):
    """One typed field delta in the canonical stream.

    Deltas and raw events share one ordered transport queue. A sequence is
    consumed only when canonical fields actually change, so any missing number
    is a real transport gap and forces a fresh snapshot rather than being
    mistaken for intentional coalescing.
    """

    model_config = ConfigDict(extra="allow")

    epoch: str
    sequence: int
    changes: dict[str, Any]
    job_trajectory_appends: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    # Jobs whose appended events are a REPLACEMENT, not a suffix. The owner's
    # ``AsyncJob.trajectory`` evicts oldest past ``subagent.TRAJECTORY_CAP``, so
    # once a child crosses the cap the prefix check can never hold again;
    # without this marker a follower would extend forever (500 → 1000 → 1500…)
    # while duplicating rows in its click-through view.
    job_trajectory_replacements: list[str] = Field(default_factory=list)
    job_todo_updates: dict[str, list[dict[str, Any]] | None] = Field(default_factory=dict)


# One watched plan must not take down the 1 MiB control stream. Larger plans
# remain authoritative on the owner and are explicitly unavailable on a follower,
# never silently truncated into what looks like a complete task list.
JOB_TODOS_WIRE_BYTES = 128 * 1024


def job_todos_wire_value(todos: Any) -> list[dict[str, Any]] | None:
    if todos is None:
        return None
    value = _wire_value(todos)
    if len(json.dumps(value).encode("utf-8")) > JOB_TODOS_WIRE_BYTES:
        return None
    return value


def sync_wire_payload(sync: FrontendSync) -> dict[str, Any]:
    """Serialize one attach snapshot with job trajectories left OUT.

    The runtime's socket refuses a line over ``server._MAX_LINE_BYTES`` (1 MiB),
    and a retained trajectory is unbounded in bytes while bounded only in COUNT
    (``TRAJECTORY_CAP`` = 500 events, each holding a whole tool result). Ten
    children at the cap serialize to ~3.1 MB, so before this the first frame of
    a busy session could not be sent at all and the session simply could not be
    attached to — 12 of 17 sessions on the reference machine.

    Stripping happens HERE, at the wire boundary, rather than in
    :class:`FrontendStateStore`: an in-process owner subscribes to the same
    store and still wants its rows, and the follower re-acquires them per job
    through the ``job_trajectory`` op once a reader actually opens that child's
    page. ``trajectory_length`` survives so the viewer can say how many events
    it is about to load instead of rendering the child as empty.

    Two more per-turn lists are bounded here for the same reason, because the
    trajectory fix addressed one instance of the shape rather than the shape
    itself and the next unbounded field grew past the same cap on its own:

    * ``snapshot.usage_components`` — capped at accumulation
      (:data:`USAGE_COMPONENT_CAP`), and capped AGAIN here because an owner
      that restored a pre-cap checkpoint holds the uncapped list in memory for
      the life of that process.
    * each job's ``usage.cost_components`` and ``descendant_usage`` — the
      per-job twins of the same list, measured at 29 KB for ONE job on the
      reference machine, which is what made 18 stripped-trajectory jobs still
      serialize to 196 KB. FOLDED by serving identity rather than capped:
      these price the child's spend (``job_cost``), so dropping rows would
      undercount money, while folding is lossless. See
      :func:`_folded_components`.

    * ``snapshot.live_events`` — the in-flight seed. It bounded itself while a
      ``tool_execution_end`` erased its call's row; once the end is RETAINED so
      a reconnecting viewer can settle the card, one whole tool result per
      completed call accumulates for the turn. See
      :data:`LIVE_EVENT_TEXT_FRAME_BUDGET_CHARS`.

    :func:`assert_frame_fits` is the guard that fails CI when a THIRD such
    field appears.
    """
    payload = sync.model_dump(mode="json")
    snapshot = payload.get("snapshot")
    if isinstance(snapshot, dict):
        components = snapshot.get("usage_components")
        if isinstance(components, list) and len(components) > USAGE_COMPONENT_CAP:
            snapshot["usage_components"] = _capped_components(components)
        _bound_live_events_in_place(snapshot)
        jobs = snapshot.get("jobs")
        if isinstance(jobs, list):
            # Share one text budget across the roster so the frame does not grow
            # linearly with depth (see JOB_TEXT_FRAME_BUDGET_CHARS). Floored so
            # every child keeps a legible preview however deep the roster is.
            text_share = max(JOB_TEXT_FLOOR_CHARS, JOB_TEXT_FRAME_BUDGET_CHARS // max(1, len(jobs)))
            for job in jobs:
                if not isinstance(job, dict):
                    continue
                if job.get("trajectory"):
                    # ``trajectory_length`` already states the count (derived at
                    # construction, see JobState), so dropping the rows here
                    # loses nothing the viewer needs to describe the job.
                    job["trajectory"] = []
                job["todos"] = None
                _fold_job_usage_in_place(job)
                _bound_job_text_in_place(job, share=text_share)
            # After the per-row pass: this budget is spent ACROSS rows, so it
            # needs the whole roster rather than one row at a time.
            _bound_launch_prompts_across_jobs(jobs)
            for job in jobs:
                if isinstance(job, dict):
                    _elide_derivable_launch_id_in_place(job)
            # After elision, so only ids that must ride the wire are charged.
            _bound_launch_ids_across_jobs(jobs)
            for job in jobs:
                if isinstance(job, dict):
                    _drop_absent_launch_fields_in_place(job)
        # LAST, after every other field has been bounded: this budget is what
        # the socket line has LEFT, so it can only be measured once nothing
        # else will shrink. See MODEL_CATALOGUE_FLOOR_ROWS for why the
        # catalogue takes a residual budget where jobs take a fixed one.
        _bound_model_catalogue_in_place(payload, snapshot)
    return payload


def _bound_model_catalogue_in_place(payload: dict[str, Any], snapshot: dict[str, Any]) -> None:
    """Clip ``model_catalogue`` to whatever the frame has left, honestly.

    Measures the frame as it will actually be sent and spends the remainder on
    catalogue rows, so an ordinary session keeps every model and only a frame
    that could not be transmitted at all is clipped. See
    :data:`MODEL_CATALOGUE_FLOOR_ROWS` for the arithmetic and for why this
    budget is residual rather than a constant like the job-text one.

    The flag is set BEFORE the search so every measurement pays for the key
    that actually ships, then CLEARED again when the search kept every row --
    clearing can only shrink the line, so a frame that fit with the flag still
    fits without it. What the reader sees is therefore true in both
    directions: ``model_catalogue_truncated`` is set exactly when rows were
    dropped. A silently short catalogue is the failure mode this closes, and a
    complete list that claims to be partial is the same lie inverted -- it
    sends the user hunting for a model that was never omitted.
    """
    rows = snapshot.get("model_catalogue")
    if not isinstance(rows, list) or not rows:
        return
    # The whole frame is measured as it will actually be written, rather than
    # summing per-row estimates against a budget. Estimating was tried and was
    # quietly wrong by 22 bytes on a full frame: the envelope, the flag's own
    # key and JSON's separators are all paid for by the LINE, not by the rows,
    # so any arithmetic that reconstructs the total from its parts is a second
    # model of the serializer that drifts from the real one. The size of the
    # thing being sent is the only quantity the socket cares about.
    if _frame_line_bytes(payload) <= _MODEL_CATALOGUE_LINE_LIMIT:
        return
    # Binary search for the longest prefix that fits. The owner's order is the
    # picker's order, so a prefix keeps the most relevant rows rather than an
    # arbitrary slice, and the flag below stops the short list from reading as
    # a complete one (MODEL_CATALOGUE_FLOOR_ROWS).
    #
    # O(log n) serializations of a frame that is already ~1 MB, on the ONLY
    # path that overflows: an ordinary session returns above without ever
    # serializing twice.
    snapshot["model_catalogue_truncated"] = True
    low, high = MODEL_CATALOGUE_FLOOR_ROWS, len(rows)
    while low < high:
        middle = (low + high + 1) // 2
        snapshot["model_catalogue"] = rows[:middle]
        if _frame_line_bytes(payload) <= _MODEL_CATALOGUE_LINE_LIMIT:
            low = middle
        else:
            high = middle - 1
    # The floor wins even when it does not fit: an empty picker claims the
    # session can switch to nothing, which is a lie the reader cannot detect,
    # and a frame this large is unsendable for reasons the catalogue cannot fix.
    snapshot["model_catalogue"] = rows[:low]
    # An oversized frame does not imply a clipped catalogue. When the overflow
    # comes from other fields and the catalogue is already at or below the
    # floor, the search keeps every row -- and the flag set above would then
    # tell the reader models are missing when none are. Withdraw it.
    if low >= len(rows):
        snapshot.pop("model_catalogue_truncated", None)


def _frame_line_bytes(payload: dict[str, Any]) -> int:
    """Bytes the writer will put on the wire for this payload, delimiter included.

    Mirrors the transport exactly — default ``json.dumps`` separators plus the
    newline the writer appends — because a budget measured any other way is a
    budget for a line nobody sends.
    """
    return len(json.dumps({"op": "frontend_sync", "data": payload}).encode()) + 1


def _bound_live_events_in_place(snapshot: dict[str, Any]) -> None:
    """Clip the in-flight seed's retained tool results to their wire bound.

    See :data:`LIVE_EVENT_TEXT_FRAME_BUDGET_CHARS` for why the seed grew a
    bound at all and why it clips TEXT instead of dropping rows: every retained
    end must survive as a row, because the row is what settles the card.

    The budget is shared only across the rows that actually carry result text,
    for the reason :func:`_bound_job_text_in_place` gives — the common row
    contributes nothing, and dividing by it would starve the few that do. The
    floor wins over an arithmetically smaller share so a turn with very many
    calls still hands each card a legible line rather than an empty one; that
    trades an exactly-proportional budget for a guarantee the card can settle,
    which is the property this field exists to provide.

    Truncation is marked with an ellipsis, as the neighbouring text bounds do
    it, so a reader can tell a clipped preview from a tool that really did
    return that little.
    """
    events = snapshot.get("live_events")
    if not isinstance(events, list):
        return
    # Row COUNT first, then per-row text: see :data:`LIVE_EVENT_END_ROWS_MAX`
    # for why the text budget alone leaves the frame reachable.
    #
    # An evicted end takes its START with it. Evicting the end alone would
    # leave the start behind as a card the viewer paints LIVE and can never
    # settle — a stranded spinner, and then an `⊘ interrupted` at retirement on
    # a call that succeeded. That is the precise artefact this retention was
    # added to remove, so the cap must not reintroduce it by the back door.
    # A start with no end is a call still RUNNING and is always kept: that card
    # is the one the viewer most needs.
    end_positions = [
        index
        for index, item in enumerate(events)
        if isinstance(item, dict) and item.get("type") == "tool_execution_end"
    ]
    if len(end_positions) > LIVE_EVENT_END_ROWS_MAX:
        evicted_ends = end_positions[: len(end_positions) - LIVE_EVENT_END_ROWS_MAX]
        evicted = set(evicted_ends)
        settled = {
            str(events[index].get("tool_call_id") or "")
            for index in evicted_ends
            if events[index].get("tool_call_id")
        }
        for index, item in enumerate(events):
            if (
                isinstance(item, dict)
                and item.get("type") == "tool_execution_start"
                and str(item.get("tool_call_id") or "") in settled
            ):
                evicted.add(index)
        events = [item for index, item in enumerate(events) if index not in evicted]
        snapshot["live_events"] = events
    # Only ``tool_execution_end`` retains a payload; starts and message rows
    # are already small and are folded by phase, so they are not counted into
    # the share they would otherwise dilute.
    texts: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for item in events:
        if not isinstance(item, dict) or item.get("type") != "tool_execution_end":
            continue
        result = item.get("result")
        if not isinstance(result, dict):
            continue
        blocks = [
            block
            for block in (result.get("content") or [])
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        if blocks:
            texts.append((result, blocks))
    if not texts:
        return
    share = max(LIVE_EVENT_TEXT_FLOOR_CHARS, LIVE_EVENT_TEXT_FRAME_BUDGET_CHARS // len(texts))
    for _result, blocks in texts:
        # Per ROW, not per block: a result of many blocks would otherwise
        # multiply its own share and reinstate the unbounded shape one level
        # further down.
        remaining = share
        for block in blocks:
            text = block["text"]
            if len(text) > remaining:
                block["text"] = text[:remaining] + "…" if remaining > 0 else "…"
            remaining = max(0, remaining - len(text))


def _bound_job_text_in_place(job: dict[str, Any], *, share: int) -> None:
    """Clip one serialized job row's free text to its wire bounds.

    See :data:`JOB_RESULT_WIRE_CHARS` for why these are safe to truncate and
    why ``error_text`` is treated differently, and
    :data:`JOB_TEXT_FRAME_BUDGET_CHARS` for why ``share`` exists: the per-field
    caps bound one ROW, and the share is this row's slice of the whole frame's
    text budget, so a deep roster cannot spend linearly more than a shallow one.
    The effective cap is the tighter of the two.

    Truncation is marked with an ellipsis so a reader can tell a clipped preview
    from a child that really did return that little — an unmarked cut reads as
    the whole answer.
    """
    for key, cap in (
        ("result_text", JOB_RESULT_WIRE_CHARS),
        ("prompt", JOB_PROMPT_WIRE_CHARS),
        ("error_text", JOB_ERROR_WIRE_CHARS),
    ):
        value = job.get(key)
        limit = min(cap, share)
        if isinstance(value, str) and len(value) > limit:
            job[key] = value[:limit] + "…"


def _fold_job_usage_in_place(job: dict[str, Any]) -> None:
    """Fold one serialized job row's receipt lists, losslessly (see above).

    Operates on the already-serialized dict rather than on the model, because
    this runs at the wire boundary where the payload is plain JSON. The rows
    are revalidated as ``Usage`` to fold them and dumped straight back, so a
    malformed row simply stays as it is rather than failing the whole frame —
    an attach must not be refused because one job's accounting is odd.
    """
    # ``usage`` may be absent (a bash job, a child that has not reported a turn)
    # — the two lists live on different objects, so each is checked in turn.
    for container, key in ((job.get("usage"), "cost_components"), (job, "descendant_usage")):
        if not isinstance(container, dict):
            continue
        rows = container.get(key)
        # One row cannot fold into anything, so the common case costs a length
        # check and no validation.
        if not isinstance(rows, list) or len(rows) <= 1:
            continue
        try:
            folded = _folded_components([Usage.model_validate(row) for row in rows])
        except Exception:  # noqa: BLE001 — odd accounting must not refuse an attach
            continue
        container[key] = [item.model_dump(mode="json") for item in folded]


def oversized_frame_report(frame: dict[str, Any], cap_bytes: int) -> str | None:
    """Diagnose a frame that will not fit ``cap_bytes``, or ``None`` if it fits.

    The failure this describes is silent by construction: an oversized line
    makes the reader's ``readline`` raise ``LimitOverrunError``, which killed
    the client's pump task, which left the viewer waiting out its full 15 s
    sync timeout and then degrading to a runtime-less cold session. A hard bug
    therefore wore the costume of a slow/absent owner, and diagnosing it took a
    profiling session rather than a log line.

    So the report names the size, the cap, and the biggest contributors by
    field — because the actionable question is always "which unbounded list
    grew this time", and that is exactly what a bare "frame too large" does not
    answer. Cheap: it serializes only when the frame is already known not to
    fit, so the common path pays one length check.
    """
    encoded = len(json.dumps(frame).encode()) + 1  # the socket writes a "\n" too
    if encoded <= cap_bytes:
        return None
    data = frame.get("data")
    snapshot = data.get("snapshot") if isinstance(data, dict) else None
    parts: list[str] = []
    if isinstance(snapshot, dict):
        sizes = sorted(
            ((len(json.dumps(value).encode()), key, value) for key, value in snapshot.items()),
            reverse=True,
            key=lambda row: row[0],
        )[:3]
        for size, key, value in sizes:
            count = f", n={len(value)}" if isinstance(value, (list, dict)) else ""
            parts.append(f"{key}={size:,}B{count}")
    detail = f" largest fields: {', '.join(parts)}" if parts else ""
    return (
        f"{frame.get('op', 'frame')} is {encoded:,} bytes, over the "
        f"{cap_bytes:,}-byte socket line limit; it cannot be sent and the "
        f"client cannot read it.{detail}"
    )


def filter_update_trajectories(
    payload: dict[str, Any], watched: Callable[[str], bool]
) -> dict[str, Any]:
    """Drop trajectory deltas for jobs this connection has not subscribed to.

    Same budget as :func:`sync_wire_payload` applied to the delta stream: a
    viewer that never opens a child's page must not pay for its events, and a
    500-event burst on an unwatched job is exactly the frame that overflows the
    line limit mid-turn. The row COUNT still rides along (``trajectory_length``
    on the job summary), so an unwatched page opened later fetches the whole
    window on demand rather than resuming from a hole.

    Returns the input unchanged when nothing needs dropping, so the common
    no-trajectory delta costs one dict lookup and no copy.
    """
    appends = payload.get("job_trajectory_appends")
    replacements = payload.get("job_trajectory_replacements")
    todos = payload.get("job_todo_updates")
    has_todos = isinstance(todos, dict) and todos
    has_appends = isinstance(appends, dict) and appends
    has_replacements = isinstance(replacements, list) and replacements
    if not has_appends and not has_replacements and not has_todos:
        return payload
    kept_appends = (
        {job_id: rows for job_id, rows in appends.items() if watched(str(job_id))}
        if isinstance(appends, dict)
        else {}
    )
    kept_replacements = (
        [job_id for job_id in replacements if watched(str(job_id))]
        if isinstance(replacements, list)
        else []
    )
    kept_todos = (
        {job_id: rows for job_id, rows in todos.items() if watched(str(job_id))}
        if isinstance(todos, dict)
        else {}
    )
    if (
        len(kept_appends) == (len(appends) if isinstance(appends, dict) else 0)
        and len(kept_replacements) == (len(replacements) if isinstance(replacements, list) else 0)
        and len(kept_todos) == (len(todos) if isinstance(todos, dict) else 0)
    ):
        return payload
    filtered = dict(payload)
    filtered["job_trajectory_appends"] = kept_appends
    filtered["job_trajectory_replacements"] = kept_replacements
    filtered["job_todo_updates"] = kept_todos
    return filtered


@dataclass(frozen=True, slots=True)
class FrontendSubscription:
    sync: FrontendSync
    unsubscribe: Callable[[], None]


class SnapshotJobs:
    """Small manager facade preserving the existing widgets' one renderer."""

    def __init__(self, values: Iterable[JobState] = ()) -> None:
        self.replace(values)

    def replace(self, values: Iterable[JobState]) -> None:
        # Canonical jobs already own recursively immutable retained payloads.
        # Route copies through the public detacher instead of asking Pydantic to
        # deep-copy tuple-backed Mapping/Sequence wrappers: the wrappers must
        # stay immutable while consumers retain their abstract container API.
        self._values = [_public_job(value) for value in values]

    def list(self) -> list[JobState]:
        return [_public_job(value) for value in self._values]

    def get(self, job_id: str) -> JobState | None:
        return next((_public_job(value) for value in self._values if value.id == job_id), None)


class SnapshotWakeScheduler:
    def __init__(self, values: Iterable[WakeState] = ()) -> None:
        self.replace(values)

    def replace(self, values: Iterable[WakeState]) -> None:
        self.schedules = [SimpleNamespace(**value.model_dump()) for value in values]


class SnapshotSubagentComms:
    """A follower's read-only job-graph facade, rebuilt from canonical jobs.

    The owner's ``SubagentComms`` is a live registry of running children and
    cannot cross the socket, but every navigation the full-page view needs —
    parent/peer/child, ancestors, the node's session/role — is pure graph over
    ``(job_id, parent_job_id, label, session_id, prompt, agent_role, effort)``,
    all of which ``JobState`` now carries. This facade answers the SAME methods
    the app calls on ``_subagent_comms`` from ``state.jobs``, so the hierarchy
    keys work identically on a follower (U5) with no attach-specific code path.

    ``session_dir_of`` is the one method whose answer is not pure graph: it is
    a filesystem path the view reads durable history from. It is answered
    from the projected ``session_dir``/``session_id`` (see
    :func:`_snapshot_session_dir`) because the alternative — fetching the
    child's history over the socket — would put an hour-long transcript on
    the wire that a same-machine follower can page straight off disk.
    """

    def __init__(self, jobs: Iterable[JobState] = ()) -> None:
        self.replace(jobs)

    def replace(self, jobs: Iterable[JobState]) -> None:
        rows = list(jobs)
        self._nodes = {job.id: self._node_for(job) for job in rows}
        self._aliases = {alias: job.id for job in rows for alias in job.attempt_aliases}

    @staticmethod
    def _node_for(job: JobState) -> Any:
        return SimpleNamespace(
            job_id=job.id,
            label=job.label or job.agent or job.id,
            parent_job_id=job.parent_job_id,
            session_id=job.session_id,
            session_dir=_snapshot_session_dir(job),
            prompt=job.prompt or "",
            agent_role=job.agent_role or "",
            effort=job.effort or "",
            # Launch-row reconciliation, read off the node by
            # ``_refresh_subagent_view`` exactly as it is on an owner. Defaulted
            # rather than conditional: an older owner sends neither field, and
            # the empty values are what the view already tolerates (it keeps the
            # synthetic head and leaves the durable row unreconciled).
            #
            # RE-DERIVED only when the owner SAID it elided a derivable value
            # (``launch_id_derived``). At 46.7 B on every task row the literal
            # string cost 43 rows of attach headroom, and an identity cannot be
            # truncated, so it is omitted where it is reconstructible instead.
            #
            # Gated on the job TYPE rather than on a marker field: only a
            # ``task`` row can have a launch turn at all (``run_subagent`` is
            # the sole minting site; bash and eval jobs register as ``"bash"``),
            # so a bash row keeps the empty string it always had and no marker
            # has to ride the wire to say so. A v0.49.2 owner's task rows are
            # derived here too — the id is right whenever the child was launched
            # by this runtime, and a wrong-shaped guess simply matches no
            # durable row, which is the pre-#681 behaviour.
            launch_message_id=(
                job.launch_message_id
                or (_derived_launch_message_id(job.id) if job.type == "task" else "")
            ),
            # Copied out of the (possibly frozen) wire value into a plain dict:
            # the view iterates and indexes it, and a follower must never be
            # handed a container whose identity is shared with canonical state.
            #
            # Backfilled from ``attempt_aliases`` for any collapsed attempt the
            # frame budget elided (tier 3). Reconciliation matches BY KEY, so a
            # missing key means that attempt's durable row renders its full
            # preamble; the alias list names exactly those attempts and is
            # already on the row, so the entry is rebuilt with the same
            # placeholder tier 2 would have sent.
            launch_prompts=_restore_elided_launch_prompts(job),
        )

    def node(self, job_id: str) -> Any | None:
        return self._nodes.get(self._aliases.get(job_id, job_id))

    def job(self, job_id: str) -> Any | None:
        # The page reads live job fields from the jobs facade, not here; the
        # comms lookup exists on the owner for a manager cross-reference the
        # follower resolves through its own SnapshotJobs instead.
        return None

    def parent(self, job_id: str) -> Any | None:
        node = self.node(job_id)
        if node is None or not node.parent_job_id:
            return None
        return self.node(node.parent_job_id)

    def children(self, job_id: str | None) -> list[Any]:
        parent_id = self._aliases.get(job_id, job_id) if job_id else None
        return [node for node in self._nodes.values() if node.parent_job_id == parent_id]

    def peers(self, job_id: str) -> list[Any]:
        node = self._nodes.get(job_id)
        if node is None:
            return []
        return [peer for peer in self.children(node.parent_job_id) if peer.job_id != job_id]

    def ancestors(self, job_id: str) -> list[Any]:
        rows: list[Any] = []
        seen = {job_id}
        current = self.parent(job_id)
        while current is not None and current.job_id not in seen:
            seen.add(current.job_id)
            rows.append(current)
            current = self.parent(current.job_id)
        rows.reverse()
        return rows

    def session_dir_of(self, job_id: str) -> Path | None:
        """Where the child's durable transcript lives, for lazy history paging.

        The owner answers this from its live registry; a follower answers it
        from the projected job (see :func:`_snapshot_session_dir`). Existence
        is deliberately NOT checked here: the view already re-probes a
        directory whose ``transcript.jsonl`` is missing on every refresh
        (``SubagentView._reconsider_missing_history``), and a path that never
        materialises on this machine degrades to the same "no saved
        transcript" note a missing file does.
        """
        node = self.node(job_id)
        return getattr(node, "session_dir", None) if node is not None else None


#: Memoised ownership verdicts for DERIVED child directories, keyed by
#: ``(path, label, agent_role)`` — the exact question asked, so a second job
#: cannot reuse a first job's answer. The value is the verdict alone.
#:
#: A cache rather than a per-call read because ``session_dir_of`` sits on the
#: page's 1 Hz refresh: QA measured this path at 1.08 stats/s and 0 page reads
#: over 5.6 s idle, and an ``origin.json`` read per tick would regress exactly
#: that. Safe to memoise because the marker is written ONCE by whoever creates
#: the directory (``mark_session_origin``) and never rewritten — unlike the
#: transcript's presence, which the view deliberately re-probes because it can
#: appear later.
#:
#: Only facts about well-formed CONTENT are cached. A read that failed
#: (``OSError`` — a transient EMFILE, a volume blip) or a marker that did not
#: parse (the truncate-then-write window of ``mark_session_origin``, reachable
#: while a child is still starting) is a fact about the MOMENT and yields no
#: entry, so it is re-asked on the next frame instead of pinning a wrong
#: verdict for the life of the process — the same distinction
#: ``resume._session_origin_read`` draws with its ``readable`` flag.
#:
#: A node that never resolves therefore re-reads once per frame. That is the
#: honest cost of not deciding early, it is bounded by one stat, and it is not
#: a regression: before this check the same node simply resolved to ``None``
#: without reading at all.
_DERIVED_OWNERSHIP: dict[tuple[str, str, str], bool] = {}

#: Cap on the memo above. Generous beside the 256 live records a roster may
#: hold, because the point is only to stop unbounded growth across a long
#: session, never to keep the working set small.
_DERIVED_OWNERSHIP_MAX = 2048


def _derived_dir_belongs_to(directory: Path, label: str, agent_role: str) -> bool:
    """Whether a DERIVED directory really is this child's session.

    The derivation below turns a ``session_id`` into a path by construction,
    and a 48-bit truncated uuid is not an ownership proof: any local session
    with that id answers to it. Worse, the id space is per-config-root while
    the derivation reads the PROCESS-GLOBAL ``config_dir()`` — so a follower
    attached with a different ``config_dir`` still resolves against this
    process's root, and an unrelated local session's transcript could render
    under a remote child's page (review round 1, M2; QA reproduced the
    config-root half of it directly).

    ``origin.json`` is the proof already on disk: ``run_subagent`` stamps it
    with ``{"origin": "subagent", "label": ..., "agent": ...}`` at creation.
    The predicate is "this is a subagent session AND the marker POSITIVELY
    identifies the child the node describes": origin must be ``subagent``, and
    the marker's label and agent must both be present and both equal the
    node's.

    **An absent identity field means NOT PROVEN, never proven** (review round
    2, R2-1). The earlier "match only the fields present" rule degenerated to
    ``True`` for a marker carrying neither, and such markers are real rather
    than hypothetical: ``resume.backfill_session_origins`` stamps
    ``{"origin": "subagent", "backfilled": true}`` — no label, no agent — over
    the operator's existing store at startup, so one of those would have
    authorised ANY child's derived path. The same hole let a label-only marker
    ignore an agent mismatch, and let a node with no identity of its own match
    every marker.

    That refuses genuinely backfilled OLD sessions, and refusing them is the
    right trade: a marker that cannot say WHICH child it belongs to cannot
    discharge the only question being asked. Those sessions degrade to the
    existing "no saved transcript" note — exactly what a follower saw before
    this derivation existed — whereas trusting them renders somebody else's
    conversation under this child's name. Nothing is lost that the wire path
    does not already cover: an owner that stamps ``session_dir`` never reaches
    this check at all.

    Neither is a missing, unreadable, or malformed marker ownership, and the
    two failure kinds are cached differently — see the call below.
    """
    from local_operator.resume import ORIGIN_NAME, ORIGIN_SUBAGENT

    key = (str(directory), label, agent_role)
    cached = _DERIVED_OWNERSHIP.get(key)
    if cached is not None:
        return cached
    try:
        raw = (directory / ORIGIN_NAME).read_text(encoding="utf-8", errors="replace")
    except OSError:
        # A fact about this MOMENT, not about the file: do not memoise it.
        return False
    try:
        payload = json.loads(raw)
    except ValueError:
        # Also a fact about the moment, for a reason that is easy to miss:
        # ``mark_session_origin`` writes with ``write_text``, which TRUNCATES
        # and then writes, so a reader that lands between the two sees an
        # empty or half-written file. That window is reachable for a child
        # that is still starting — between ``claim_session`` and the stamp —
        # which is precisely the case the memo must not decide early. Caching
        # this ``False`` pinned a starting child as unusable for the life of
        # the process (review round 2, R2-2). Same distinction the ``OSError``
        # arm above draws, and the one ``resume._session_origin_read`` draws
        # with its ``readable`` flag: only facts about CONTENT are memoised.
        return False
    verdict = False
    if isinstance(payload, dict) and payload.get("origin") == ORIGIN_SUBAGENT:
        marked_label = payload.get("label")
        marked_agent = payload.get("agent")
        # Both sides must be non-empty and equal. A node missing its own
        # identity cannot be proven to own anything either, so it is refused
        # by the same conjunction rather than by a separate branch.
        verdict = bool(
            label
            and agent_role
            and isinstance(marked_label, str)
            and isinstance(marked_agent, str)
            and marked_label == label
            and marked_agent == agent_role
        )
    # Only a decided verdict about well-formed CONTENT reaches the memo.
    #
    # Bounded explicitly rather than leaning on the roster's own cap. The live
    # roster is capped (``SubagentComms.MAX_RECORDS`` = 256) but that bounds it
    # at an INSTANT: eviction there does not clear entries here, so over a long
    # session the key space is the number of distinct (directory, label, agent)
    # triples ever SEEN, which keeps growing. The cap is generous next to 256
    # live records — a session would have to churn through eight full rosters
    # to evict anything — and re-deciding an evicted entry costs one stat.
    if len(_DERIVED_OWNERSHIP) >= _DERIVED_OWNERSHIP_MAX:
        # First-seen insertion order, so the oldest verdict goes first.
        _DERIVED_OWNERSHIP.pop(next(iter(_DERIVED_OWNERSHIP)))
    _DERIVED_OWNERSHIP[key] = verdict
    return verdict


def _snapshot_session_dir(job: JobState) -> Path | None:
    """Resolve a projected job's child session directory on a follower.

    Two sources, in order of authority:

    1. The wire ``session_dir`` the owner stamped (``_with_lineage``). This is
       the owner's own ``Path`` and is right by construction, so it is trusted
       as-is — the owner knows where it put the child.
    2. ``config_dir() / "sessions" / session_id`` when the owner sent only a
       ``session_id``. Children are always created there
       (``harness/subagent.py``: ``session_id`` IS ``session_dir.name``), and a
       TUI attached to a daemon on the same machine reads the same config
       root. The derivation exists so an already-running daemon from before
       ``session_dir`` rode the wire — the exact situation an operator is in
       when they upgrade the viewer under a long-lived owner — gets history
       without a restart, rather than only after the owner is relaunched. It
       is also what rescues a child rebuilt from the comms graph after a
       restart, which carries a ``session_id`` and never had a wire directory.

    Only the GUESS is verified (:func:`_derived_dir_belongs_to`): a path this
    function invented must prove it is the right child's session before the
    page reads it, because the id alone is not an ownership proof.

    A follower on ANOTHER machine (the mobile daemon relaying a remote
    session) derives a path that does not exist locally, and now also one that
    cannot prove ownership. Both degrade to ``None``: the view treats that as
    "no saved transcript" and keeps painting the live trajectory, which is
    exactly the behaviour every follower had before this field existed, so the
    fallback can only add history, never take a page away or show the wrong
    one.
    """
    wire = job.session_dir
    if wire:
        return Path(str(wire))
    if job.session_id:
        from local_operator.paths import config_dir

        derived = config_dir() / "sessions" / str(job.session_id)
        if _derived_dir_belongs_to(derived, job.label or job.agent or "", job.agent_role or ""):
            return derived
    return None


class SnapshotMcpManager:
    """Read-only manager API used by status and ``/mcp`` reporting."""

    def __init__(self, values: Iterable[McpServerState] = ()) -> None:
        self.replace(values)
        self._callback: Callable[..., Any] | None = None

    def replace(self, values: Iterable[McpServerState]) -> None:
        self._values = [value.model_copy(deep=True) for value in values]

    def get_all_server_names(self) -> list[str]:
        return sorted(value.name for value in self._values)

    def get_connected_servers(self) -> list[str]:
        return sorted(value.name for value in self._values if value.status == "connected")

    def get_connection_status(self, name: str) -> str:
        match = next((value for value in self._values if value.name == name), None)
        return match.status if match is not None else "disconnected"

    def set_on_tools_changed(self, callback: Callable[..., Any]) -> None:
        self._callback = callback

    @property
    def on_tools_changed(self) -> Callable[..., Any] | None:
        return self._callback


class FrontendStateStore:
    """Atomic snapshot/update store shared by local and remote sessions.

    Initial joins receive one immutable snapshot. Later mutations publish only
    typed field deltas, keeping high-frequency transport bounded while preserving
    one reducer and a strict sequence suitable for gap detection.
    """

    def __init__(self, state: FrontendSessionState) -> None:
        self._state = _freeze_state_jobs(state.model_copy(deep=True))
        self._subscribers: list[Callable[[FrontendUpdate], None]] = []
        self._todo_sequences: dict[str, int] = {}
        self._todo_seed_floor = state.sequence

    @property
    def state(self) -> FrontendSessionState:
        # Never share an owning Pydantic instance. Clone the small state/job/usage
        # shells while immutable retained payload wrappers remain shared.
        snapshot = self._state.model_copy(deep=True)
        return snapshot.model_copy(
            update={"jobs": _FrozenSequence(_public_job(job) for job in self._state.jobs)}
        )

    def read_field(self, name: str) -> Any:
        """One field of the state, WITHOUT cloning the whole state.

        The general sibling of :attr:`pending_gate`, and it exists for the same
        measured reason: `state` deep-copies every job, usage component and
        trajectory row so no caller can mutate the store's instance, and the
        per-frame accessors that read a SINGLE field were paying that clone
        each time. Profiling one cold sidebar navigation measured 37 `state`
        reads and ~25,000 `deepcopy` calls, ~30 ms of a 135 ms frame, almost
        all of it from single-field reads like `model_label` and
        `effective_model`.

        RESTRICTED TO DEEPLY IMMUTABLE FIELDS, and the restriction is enforced
        rather than documented. The value is the store's OWN object, so the bar
        is that a caller cannot mutate it at all — every admitted field is a
        `str`/`bool`/`int`/`float`/`None`. Model-valued fields and mutable
        collections are deliberately absent: reading those still goes through
        `state` and still pays for the deep copy that protects them. See
        :data:`_SHAREABLE_STATE_FIELDS` for why "the reducer replaces it" was
        NOT a sufficient bar.
        """
        if name not in _SHAREABLE_STATE_FIELDS:
            raise KeyError(
                f"{name!r} is not a copy-free state field; read it through `state` "
                "so the caller cannot mutate the store's own instance"
            )
        return getattr(self._state, name)

    @property
    def pending_gate(self) -> "PendingGateState | None":
        """The pending gate alone, WITHOUT cloning the whole state.

        ``state`` deep-copies every job, usage and trajectory row so no caller
        can mutate the store's instance. A per-frame readiness check only needs
        this one immutable field, and paying that clone on every display cost
        ~23 ms of the cold sidebar frame — the single largest item in its
        profile. ``PendingGateState`` is never mutated in place by the reducer
        (it is replaced), so sharing the instance is safe here.
        """
        return self._state.pending_gate

    @property
    def has_subscribers(self) -> bool:
        return bool(self._subscribers)

    def replace(self, state: FrontendSessionState) -> None:
        self._state = _freeze_state_jobs(state.model_copy(deep=True))
        self._todo_sequences.clear()
        self._todo_seed_floor = state.sequence

    def replace_and_notify(self, state: FrontendSessionState) -> None:
        """Install a proven wire snapshot without reaching into subscribers."""
        self.replace(state)
        update = FrontendUpdate(
            epoch=state.epoch,
            sequence=state.sequence,
            changes=state.model_dump(mode="json"),
        )
        for subscriber in list(self._subscribers):
            subscriber(update.model_copy(deep=True))

    def apply_update(self, update: FrontendUpdate) -> FrontendSessionState:
        """Apply one already-validated ordered delta from an owner."""
        if update.epoch != self._state.epoch or update.sequence != self._state.sequence + 1:
            raise ValueError("frontend update is not the next state sequence")
        changes = copy.deepcopy(update.changes)
        if "jobs" in changes:
            previous = {job.id: job for job in self._state.jobs}
            replacements = set(update.job_trajectory_replacements)
            rebuilt = []
            for raw in changes["jobs"]:
                job_id = str(raw.get("id", ""))
                prior = previous.get(job_id)
                if job_id in replacements:
                    trajectory = []
                else:
                    trajectory = list(prior.trajectory if prior is not None else [])
                trajectory.extend(update.job_trajectory_appends.get(job_id, []))
                # Defensive mirror of the owner-side eviction: even a
                # misbehaving owner cannot grow a follower without bound.
                if len(trajectory) > _TRAJECTORY_CAP:
                    del trajectory[: len(trajectory) - _TRAJECTORY_CAP]
                raw["trajectory"] = trajectory
                raw["todos"] = _wire_value(prior.todos) if prior is not None else None
                if (
                    job_id in update.job_todo_updates
                    and update.sequence > self._todo_sequences.get(job_id, -1)
                ):
                    raw["todos"] = update.job_todo_updates[job_id]
                    self._todo_sequences[job_id] = update.sequence
                rebuilt.append(raw)
            changes["jobs"] = rebuilt
            retained = {str(row["id"]) for row in rebuilt}
            self._todo_sequences = {
                key: seq for key, seq in self._todo_sequences.items() if key in retained
            }
        payload = self._state.model_dump()
        payload.update(changes)
        payload["epoch"] = update.epoch
        payload["sequence"] = update.sequence
        self._state = _freeze_state_jobs(FrontendSessionState.model_validate(payload))
        for subscriber in list(self._subscribers):
            subscriber(update.model_copy(deep=True))
        return self.state

    def seed_job_trajectory(self, job_id: str, rows: Sequence[dict[str, Any]]) -> bool:
        """Install rows a FOLLOWER fetched on demand for one child's page.

        The attach snapshot ships job rows with empty trajectories — a busy
        session's would exceed the socket's line limit — so the viewer pulls
        one child's window over ``job_trajectory`` and seeds it here.

        It goes into the canonical state rather than a cache beside it because
        ``apply_update`` extends each job's trajectory from the PREVIOUS
        canonical value: a second store would make the delta stream and the
        fetched window two accumulators of the same list, and the page would
        show whichever won. Seeding is deliberately NOT a sequence-bearing
        mutation — no delta is published and the sequence does not move, since
        this changes what this follower has locally, not what the owner said.

        Returns False when the job is no longer on the roster (it settled and
        was swept while the fetch was in flight), so the caller can leave the
        page's "no longer on the ledger" state alone.
        """
        jobs = list(self._state.jobs)
        for index, job in enumerate(jobs):
            if job.id != job_id:
                continue
            jobs[index] = job.model_copy(
                update={"trajectory": list(rows), "trajectory_length": max(len(rows), 0)}
            )
            self._state = _freeze_state_jobs(
                self._state.model_copy(update={"jobs": jobs}), jobs_are_canonical=False
            )
            return True
        return False

    def seed_job_todos(
        self,
        job_id: str,
        todos: list[dict[str, Any]] | None,
        *,
        epoch: str,
        sequence: int,
        session_id: str | None,
    ) -> bool:
        """Join an on-demand snapshot with deltas without rolling a plan back.

        The request reply and the ordered stream race in both directions. Keep a
        per-job watermark: a newer fetched snapshot also fences older deltas
        still queued on the wire, without consuming their global sequence.
        """
        if epoch != self._state.epoch or sequence < max(
            self._todo_seed_floor, self._todo_sequences.get(job_id, -1)
        ):
            return False
        jobs = list(self._state.jobs)
        for index, job in enumerate(jobs):
            if job.id != job_id or job.session_id != session_id:
                continue
            jobs[index] = _freeze_job(job.model_copy(update={"todos": todos}))
            self._state = self._state.model_copy(update={"jobs": _FrozenSequence(jobs)})
            self._todo_sequences[job_id] = sequence
            return True
        return False

    def mutate(self, **changes: Any) -> FrontendUpdate | None:
        normalized: dict[str, Any] = {}
        wire_changes: dict[str, Any] = {}
        trajectory_appends: dict[str, list[dict[str, Any]]] = {}
        trajectory_replacements: list[str] = []
        todo_updates: dict[str, list[dict[str, Any]] | None] = {}
        for key, value in changes.items():
            if key == "jobs":
                # JobState equality walks the bounded trajectories without first
                # cloning them into JSON. On a 100-child roster at the 500-event
                # cap this is ~20x cheaper for the common unchanged refresh.
                candidate_jobs = [
                    _freeze_job(
                        item if isinstance(item, JobState) else JobState.model_validate(item)
                    )
                    for item in value
                ]
                if not _jobs_equal(self._state.jobs, candidate_jobs):
                    normalized[key] = candidate_jobs
                    wire_changes[key] = candidate_jobs
                continue
            candidate = _json_value(value)
            # Serialize only fields the caller proposes changing. Dumping the
            # complete state here cloned every retained trajectory even for a
            # one-bit streaming update, turning unrelated UI reads into stalls.
            current_value = _json_value(getattr(self._state, key))
            if current_value != candidate:
                normalized[key] = _validate_state_field(key, candidate)
                wire_changes[key] = candidate
        if "jobs" in wire_changes:
            previous = {job.id: job for job in self._state.jobs}
            summaries = []
            for job in wire_changes["jobs"]:
                job_id = job.id
                trajectory = job.trajectory
                prior = previous.get(job_id)
                if prior is None or prior.todos != job.todos:
                    todo_updates[job_id] = job_todos_wire_value(job.todos)
                old = prior.trajectory if prior is not None else []
                if trajectory[: len(old)] == old:
                    appended = trajectory[len(old) :]
                else:
                    # The owner list rotated past its cap (or was rebuilt):
                    # a suffix no longer exists, so ship a replacement once
                    # rather than the whole list disguised as appends forever.
                    appended = trajectory
                    trajectory_replacements.append(job_id)
                if appended:
                    # Unlike the job snapshot, this delta does not pass through
                    # JobState's serializer. Pydantic validates the outer event
                    # dict but leaves its Any-valued args/message/result frozen;
                    # JSON then turns their tuple-backed mappings into arrays of
                    # pairs. The viewer loses tool details until a fresh history
                    # fetch. Thaw at this wire boundary, just as snapshots do.
                    trajectory_appends[job_id] = [_wire_value(event) for event in appended]
                summaries.append(_job_summary(job))
            # Same roster-shared budget the attach snapshot applies, and applied
            # here too because a delta re-serializes job rows by its own route:
            # a bound placed only at the snapshot boundary holds for the first
            # frame and leaks on every one after it.
            _bound_launch_prompts_across_jobs(summaries)
            for summary in summaries:
                _elide_derivable_launch_id_in_place(summary)
            _bound_launch_ids_across_jobs(summaries)
            for summary in summaries:
                _drop_absent_launch_fields_in_place(summary)
            wire_changes["jobs"] = summaries
        if not normalized:
            return None
        # Unchanged fields are immutable snapshot components and can be shared.
        # Re-validating a full model here deep-copied all job trajectories for
        # every small delta; each changed field was validated above instead.
        jobs_changed = "jobs" in normalized
        self._state = _freeze_state_jobs(
            self._state.model_copy(update={**normalized, "sequence": self._state.sequence + 1}),
            jobs_are_canonical=not jobs_changed,
        )
        update = FrontendUpdate(
            epoch=self._state.epoch,
            sequence=self._state.sequence,
            changes=wire_changes,
            job_trajectory_appends=trajectory_appends,
            job_trajectory_replacements=trajectory_replacements,
            job_todo_updates=todo_updates,
        )
        for subscriber in list(self._subscribers):
            subscriber(update.model_copy(deep=True))
        return update

    def subscribe(self, callback: Callable[[FrontendUpdate], None]) -> FrontendSubscription:
        # Capture and register without yielding. Session mutations happen on one
        # event loop, so this is the atomic boundary: no update can fit between
        # the snapshot's sequence and the subscriber becoming visible.
        self._subscribers.append(callback)
        state = self.state
        sync = FrontendSync(
            epoch=state.epoch,
            sequence=state.sequence,
            snapshot=state,
            live_cursor=state.history_cursor,
        )

        def unsubscribe() -> None:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

        return FrontendSubscription(sync=sync, unsubscribe=unsubscribe)

    @classmethod
    def from_session(cls, session: Any) -> "FrontendStateStore":
        store = cls(cls._restored_state(session))
        store.refresh_from_session(session, initial=True)
        return store

    @classmethod
    def from_checkpoint(cls, session: Any) -> "FrontendStateStore":
        """Headless construction: durable restore only, no live source scan.

        A headless host (scheduler, owned session, exec CLI) must stay cheap —
        ``refresh_from_session`` walks jobs/todos/MCP and imports the TUI
        registry — but its turn-end checkpoint is unconditional, so the store
        MUST begin from the richest durable state or a single headless turn
        would persist a bare checkpoint over the TUI's spend/duration/title.
        """
        return cls(cls._restored_state(session))

    @staticmethod
    def _restored_state(session: Any) -> FrontendSessionState:
        transcript = getattr(session, "_transcript", None)
        checkpoint = (
            transcript.latest_custom(FRONTEND_CHECKPOINT_CUSTOM_TYPE) if transcript else None
        )
        restored = None
        if isinstance(checkpoint, dict):
            raw = checkpoint.get("state")
            try:
                restored = (
                    FrontendSessionState.model_validate(raw) if isinstance(raw, dict) else None
                )
            except Exception:
                restored = None
        epoch = uuid.uuid4().hex
        session_id = str(session.session_id)
        state = restored or FrontendSessionState(session_id=session_id, epoch=epoch)
        # A new owner epoch invalidates stale wire updates while preserving the
        # durable checkpoint identity used to reconcile takeover without addition.
        #
        # The receipt cap is applied to the RESTORED value, not just to fresh
        # accumulation: every transcript written before the cap carries the full
        # uncapped list (958 KB in the largest observed row), so an owner that
        # resumed one would emit an oversized attach frame and re-persist the fat
        # list forever despite the cap above. Capping on the way in is what makes
        # an existing session heal on its first resume rather than staying broken.
        return state.model_copy(
            update={
                "epoch": epoch,
                "sequence": 0,
                "usage_components": _capped_components(state.usage_components),
                **_inherited_identity_fixups(state, session_id),
            }
        )

    def refresh_from_session(self, session: Any, *, initial: bool = False) -> FrontendSessionState:
        current = self._state
        selected = getattr(session, "model", None)
        effective = getattr(session, "effective_model", None) or selected
        last_usage = None
        restore = getattr(session, "restored_usage", None)
        if callable(restore):
            try:
                last_usage = restore()
            except Exception:
                last_usage = None
        jobs = self._jobs(session)
        child_costs: dict[str, float] = dict(current.child_costs)
        for job in jobs:
            cost = _job_subtree_cost(job, default_model_label=_label(selected))
            if cost is not None:
                child_costs[job.id] = cost
        parent_cost = current.cumulative_parent_cost
        knowledge = current.cost_knowledge
        if parent_cost is None and last_usage is not None:
            cost = turn_cost(_label(effective), last_usage)
            if cost is not None:
                parent_cost = cost
                knowledge = CostKnowledge.FLOOR
        title_state = getattr(session, "conversation_name_state", None)
        title = str(getattr(session, "conversation_name", "") or "")
        todos = _todo_state(str(getattr(session, "session_id", current.session_id)))
        wakes = _wake_state(getattr(session, "wake_scheduler", None))
        mcp_servers = _mcp_state(
            getattr(session, "mcp_manager", None), getattr(session, "mcp_startup", None)
        )
        mcp_startup = _json_value(getattr(session, "mcp_startup", None))
        queued = []
        try:
            for message in session.queued_steering():
                content = list(getattr(message, "content", ()) or ())
                text = str(getattr(message, "text", "") or "")
                if not text:
                    # Not everything on the steering queue is a plain user
                    # Message: a busy-path wake and a busy-path peer steer
                    # queue their CustomMessage, which carries no ``text``
                    # property or ``content`` blocks — only ``details``.
                    # Without this a follower saw those entries as
                    # ``{"text": ""}`` and a renderer would paint a blank
                    # queued row. ``body`` is the raw human-facing text a
                    # peer row keeps for the UIs, preferred over ``text``
                    # (the model-facing provenance envelope); a wake only has
                    # ``text``, which IS its human-facing message.
                    details = getattr(message, "details", None)
                    if isinstance(details, dict):
                        text = str(details.get("body") or details.get("text") or "")
                queued.append(
                    {
                        "id": str(getattr(message, "id", "") or ""),
                        "text": text,
                        "image_count": sum(
                            1 for block in content if block.__class__.__name__ == "ImageContent"
                        ),
                        "status": "queued",
                    }
                )
        except Exception:
            pass
        history_cursor = None
        transcript = getattr(session, "_transcript", None)
        if transcript is not None:
            try:
                entries = transcript.entries()
                history_cursor = entries[-1].id if entries else None
            except Exception:
                pass
        # `/context` remains an on-demand operation. Computing its schema
        # breakdown on every unrelated mutation would serialize the unbounded
        # tool inventory on the session loop.
        context_breakdown = current.context_breakdown
        # A compaction creates a newer occupancy estimate than the last provider
        # receipt while that receipt remains authoritative for billing. Generic
        # source refreshes run after `agent_end`; replaying the unchanged receipt
        # there caused every frontend to paint `tokens_after` and then rebound to
        # the pre-pass level. Only a genuinely newer receipt may supersede it.
        receipt_context = getattr(last_usage, "context_tokens", None) if last_usage else None
        current_receipt_context = (
            current.last_usage.context_tokens if current.last_usage is not None else None
        )
        preserve_settled_context = bool(
            current.context_is_estimate
            and current.context_tokens
            and receipt_context
            and receipt_context == current_receipt_context
        )
        changes = dict(
            attention=dict(getattr(session, "_attention", {}) or {}),
            cwd=str(getattr(session, "cwd", "") or getattr(session, "_cwd", "") or os.getcwd()),
            conversation_title=title,
            conversation_title_user_set=bool(getattr(title_state, "user_set", False)),
            conversation_title_forked=bool(getattr(session, "wears_inherited_title", False)),
            goal=str(getattr(session, "goal", "") or ""),
            active_agent=str(getattr(session, "active_agent", "") or ""),
            active_team=str(getattr(session, "active_team_name", "") or ""),
            selected_model=(
                selected.model_dump(mode="json") if isinstance(selected, ModelSpec) else selected
            ),
            effective_model=(
                effective.model_dump(mode="json") if isinstance(effective, ModelSpec) else effective
            ),
            last_usage=(
                last_usage.model_dump(mode="json") if isinstance(last_usage, Usage) else last_usage
            ),
            context_tokens=(current.context_tokens if preserve_settled_context else receipt_context)
            or current.context_tokens,
            context_is_estimate=(
                current.context_is_estimate
                if preserve_settled_context
                else False if receipt_context else current.context_is_estimate
            ),
            context_window=(
                getattr(effective, "context_window", None) if effective is not None else None
            ),
            context_breakdown=context_breakdown,
            cumulative_parent_cost=parent_cost,
            child_costs=child_costs,
            **_ledger_cost(session),
            cost_knowledge=knowledge,
            streaming=bool(getattr(session, "is_streaming", False)),
            generation=int(getattr(session, "_generation", current.generation) or 0),
            last_turn_outcome=_last_turn_outcome_from(session, current.last_turn_outcome),
            activity_started_at=(
                current.activity_started_at
                if bool(getattr(session, "is_streaming", False))
                else None
            ),
            queued_steering=queued,
            jobs=jobs,
            todos=todos,
            wakes=wakes,
            mcp_servers=mcp_servers,
            mcp_startup=mcp_startup,
            history_cursor=history_cursor,
            history_generation=int(getattr(transcript, "_history_generation", 0)),
            attachment_root=str(getattr(transcript, "directory", "") or "") or None,
            slash_capabilities=_slash_capabilities(),
        )
        if initial:
            payload = current.model_dump()
            payload.update(changes)
            self._state = _freeze_state_jobs(FrontendSessionState.model_validate(payload))
        else:
            self.mutate(**changes)
        return self.state

    def refresh_restored_usage(self, session: Any) -> FrontendUpdate | None:
        """Price the restored point-in-time reading without rescanning state."""
        restore = getattr(session, "restored_usage", None)
        usage = restore() if callable(restore) else None
        if not isinstance(usage, Usage):
            return None
        state = self._state
        changes: dict[str, Any] = {
            "last_usage": usage.model_dump(mode="json"),
            "context_tokens": usage.context_tokens,
            "context_is_estimate": False if usage.context_tokens else state.context_is_estimate,
        }
        if state.cumulative_parent_cost is None:
            cost = turn_cost(_label(getattr(session, "effective_model", None)), usage)
            if cost is not None:
                changes.update(
                    cumulative_parent_cost=cost,
                    cost_knowledge=CostKnowledge.FLOOR,
                )
        return self.mutate(**changes)

    def accrue_usage(self, session: Any, usage: Usage) -> FrontendUpdate | None:
        """Accrue a provider call outside the ordinary agent event stream."""
        state = self._state
        cost = turn_cost(_label(getattr(session, "effective_model", None)), usage)
        changes: dict[str, Any] = {
            "last_usage": usage.model_dump(mode="json"),
            "context_tokens": usage.context_tokens or usage.input_tokens or state.context_tokens,
            "context_is_estimate": False,
        }
        if cost is not None:
            changes.update(
                cumulative_parent_cost=(state.cumulative_parent_cost or 0.0) + cost,
                cost_knowledge=(
                    CostKnowledge.EXACT
                    if state.cost_knowledge in {CostKnowledge.UNKNOWN, CostKnowledge.EXACT}
                    else state.cost_knowledge
                ),
                usage_components=_capped_components(
                    list(state.usage_components) + list(usage.cost_components or [usage])
                ),
            )
        elif usage.input_tokens or usage.output_tokens:
            changes["cost_knowledge"] = CostKnowledge.PARTIAL
        return self.mutate(**changes)

    def refresh_jobs(self, session: Any) -> FrontendUpdate | None:
        """Publish the job roster without rescanning unrelated session state.

        Canonical lineage (parent/child identity) is folded in HERE rather than
        in ``refresh_from_session``: the comms lookup is per-job, and the
        per-event refresh runs on the session loop for every streaming edge —
        folding there stalled concurrent children by over a second. The jobs
        roster is republished on a 50 ms coalesce, so lineage lands on the same
        cadence the page that needs it already refreshes on, at a fraction of
        the cost.
        """
        jobs = self._jobs(session)
        child_costs = dict(self._state.child_costs)
        selected = getattr(session, "model", None)
        for job in jobs:
            cost = _job_subtree_cost(job, default_model_label=_label(selected))
            if cost is not None:
                child_costs[job.id] = cost
        return self.mutate(jobs=jobs, child_costs=child_costs, **_ledger_cost(session))

    def refresh_model_catalogue(self, entries: Iterable[Any]) -> FrontendUpdate | None:
        """Publish the owner's offerable model rows as canonical state.

        The catalogue answers one question — which models may this SESSION
        switch to — and only the owner's provider controller knows it (the
        owner's credentials, its aggregators, its registry). Kept out of
        ``refresh_from_session`` because that runs on the session loop for
        every streaming edge; the catalogue changes on credential/registry
        timescales, so the TUI publishes it on adoption and after login-style
        mutations instead.
        """
        rows: list[dict[str, Any]] = []
        for entry in entries:
            rows.append(
                {
                    "provider": str(getattr(entry, "provider", "") or ""),
                    "model_id": str(getattr(entry, "model_id", "") or ""),
                    "label": str(getattr(entry, "label", "") or ""),
                    "context_window": int(getattr(entry, "context_window", 0) or 0),
                    "default_context_window": getattr(entry, "default_context_window", None),
                    "max_context_window": getattr(entry, "max_context_window", None),
                    "input_price": float(getattr(entry, "input_price", 0.0) or 0.0),
                    "output_price": float(getattr(entry, "output_price", 0.0) or 0.0),
                    "connected": bool(getattr(entry, "connected", False)),
                    "aggregated": bool(getattr(entry, "aggregated", False)),
                    # Carried so the round-trip stays faithful to
                    # ``CatalogueEntry`` rather than silently defaulting a
                    # field the reader knows about. ``getattr`` with a default
                    # for the same reason every key above uses one: this takes
                    # ``Any``, and a duck-typed entry from an embedding host
                    # need not have the attribute.
                    "routed": bool(getattr(entry, "routed", False)),
                }
            )
        return self.mutate(model_catalogue=rows)

    def observe_event(self, session: Any, event: AgentEvent[Any]) -> FrontendUpdate | None:
        now = time.time()
        state = self._state
        changes: dict[str, Any] = {}
        self._fold_live_event(event)
        if isinstance(event, AgentStartEvent):
            changes.update(
                streaming=True,
                generation=int(event.generation or state.generation + 1),
                activity_started_at=now,
            )
        elif isinstance(event, AgentEndEvent):
            duration = state.active_duration_s
            if state.activity_started_at is not None:
                duration += max(0.0, now - state.activity_started_at)
            if event.error:
                outcome: Literal["completed", "aborted", "error", ""] = "error"
            elif event.aborted:
                outcome = "aborted"
            else:
                outcome = "completed"
            changes.update(
                streaming=False,
                activity_started_at=None,
                active_duration_s=duration,
                last_turn_outcome=outcome,
            )
            # Reconcile the whole turn once. Per-call receipts are retained so a
            # mixed-provider aggregate never loses which call owned which price.
            usages = [
                usage
                for message in (event.messages or [])
                if isinstance((usage := getattr(message, "usage", None)), Usage)
            ]
            if usages:
                aggregate = _aggregate_usage(usages)
                total = turn_cost(_label(getattr(session, "effective_model", None)), aggregate)
                if total is not None:
                    remainder = max(0.0, total - state.current_turn_accrued_cost)
                    previous = state.cumulative_parent_cost or 0.0
                    changes.update(
                        cumulative_parent_cost=previous + remainder,
                        current_turn_accrued_cost=0.0,
                        usage_components=_capped_components(
                            list(state.usage_components)
                            if state.current_turn_accrued_cost > 0
                            else list(state.usage_components) + list(aggregate.cost_components)
                        ),
                        cost_knowledge=(
                            CostKnowledge.EXACT
                            if state.cost_knowledge in {CostKnowledge.UNKNOWN, CostKnowledge.EXACT}
                            else state.cost_knowledge
                        ),
                    )
                elif any(u.input_tokens or u.output_tokens for u in usages):
                    changes["cost_knowledge"] = CostKnowledge.PARTIAL
                    changes["current_turn_accrued_cost"] = 0.0
                # `messages` remain the billing authority even when a post-turn
                # compaction invalidates their occupancy. The session stamps the
                # settled level separately so this late boundary cannot rebound a
                # frontend to the provider reading from before the rewrite.
                settled_context = event.context_tokens or aggregate.context_tokens
                changes.update(
                    last_usage=aggregate.model_dump(mode="json"),
                    context_tokens=settled_context,
                    context_is_estimate=(
                        True
                        if event.context_tokens is not None
                        else False if aggregate.context_tokens else state.context_is_estimate
                    ),
                )
        elif (
            isinstance(event, MessageEndEvent) and getattr(event.message, "usage", None) is not None
        ):
            usage = getattr(event.message, "usage", None)
            if not isinstance(usage, Usage):
                return
            # Occupancy is a level. Cost accrues per call so arbitrary joins see
            # the same lifetime figure; AgentEnd adds only the final remainder.
            call_cost = turn_cost(_label(getattr(session, "effective_model", None)), usage)
            changes.update(
                last_usage=usage.model_dump(mode="json"),
                context_tokens=usage.context_tokens or usage.input_tokens or state.context_tokens,
                context_is_estimate=False,
            )
            if call_cost is not None:
                changes.update(
                    cumulative_parent_cost=(state.cumulative_parent_cost or 0.0) + call_cost,
                    current_turn_accrued_cost=state.current_turn_accrued_cost + call_cost,
                    cost_knowledge=(
                        CostKnowledge.EXACT
                        if state.cost_knowledge in {CostKnowledge.UNKNOWN, CostKnowledge.EXACT}
                        else state.cost_knowledge
                    ),
                    usage_components=_capped_components(
                        list(state.usage_components) + list(usage.cost_components or [usage])
                    ),
                )
            elif usage.input_tokens or usage.output_tokens:
                changes["cost_knowledge"] = CostKnowledge.PARTIAL
        elif isinstance(event, CompactionEndEvent) and event.success:
            changes.update(
                context_tokens=event.tokens_after or None,
                context_is_estimate=True,
            )
        update = self.mutate(**changes) if changes else None
        # Expensive source snapshots are explicit mutation hooks. Turn edges and
        # tool/result boundaries are the defensive fallback; token deltas never
        # rescan transcript/jobs or publish replacement state. Subagent progress
        # is excluded too: the job manager already coalesces that live row onto
        # ``refresh_jobs`` within 50 ms. Re-scanning the entire session here
        # published a duplicate frame for every child boundary and made remote
        # followers pay a second wire update for the same activity string.
        kind = str(getattr(event, "type", ""))
        if isinstance(event, (AgentEndEvent, MessageEndEvent)) or kind in {
            "tool_execution_end",
            "subagent_end",
            "model_change",
        }:
            before = self._state.sequence
            self.refresh_from_session(session)
            if self._state.sequence != before:
                update = None
        return update

    def _fold_live_event(self, event: AgentEvent[Any]) -> None:
        """Maintain only the bounded in-flight seed, without publishing deltas.

        Connected frontends already receive the raw event. Updating this local
        snapshot before event fan-out makes a join at that exact boundary see
        the accumulated assistant/tool state without flooding every peer with a
        second frame for every token.
        """
        data = event.model_dump(mode="json")
        kind = str(data.get("type", ""))
        live = list(self._state.live_events)
        if kind in {"agent_start", "agent_end"}:
            live = []
        elif kind == "message_start":
            message = data.get("message") or {}
            if message.get("role") != "user":
                live = [
                    item
                    for item in live
                    if item.get("type") not in {"message_start", "message_update"}
                ]
                live.append(data)
        elif kind == "message_update":
            live = [item for item in live if item.get("type") != "message_update"]
            live.append(data)
        elif kind == "message_end":
            live = [
                item for item in live if item.get("type") not in {"message_start", "message_update"}
            ]
        elif kind in {"tool_call_compose", "tool_execution_start"}:
            call_id = str(data.get("tool_call_id") or "")
            live = [item for item in live if str(item.get("tool_call_id") or "") != call_id]
            live.append(data)
        elif kind == "tool_execution_end":
            call_id = str(data.get("tool_call_id") or "")
            # The END REPLACES the start rather than erasing the call. A
            # frontend that joins mid-turn needs to learn the outcome of a
            # call that finished while it was away: dropping the row left the
            # viewer holding a card it had painted live with no way to settle
            # it, so ``_retire_live_tool_cards`` marked it ``⊘ interrupted``
            # at turn end — on a call that had SUCCEEDED (QA round 1, Q2).
            # A joiner that never saw the start still renders correctly: the
            # app buffers an unmatched end in ``_pending_tool_ends``.
            live = [item for item in live if str(item.get("tool_call_id") or "") != call_id]
            live.append(data)
        # Shallow copy on purpose: this runs per streaming delta on the session
        # loop, and a deep copy re-clones a 500-event trajectory each token.
        # ``live`` is freshly built above, and every other field is replaced
        # (never mutated in place) by ``mutate``/``apply_update``.
        self._state = self._state.model_copy(update={"live_events": live})

    async def checkpoint(self, transcript: Any) -> None:
        state = self.state
        checkpoint_id = uuid.uuid4().hex
        state.checkpoint_id = checkpoint_id
        self.replace(state)
        # Trajectories are reconstructable from durable child transcripts and
        # live_events are transient by definition; persisting them appended
        # ~71 KiB per busy child to the transcript at EVERY turn end.
        #
        # The per-job receipt lists are folded for the same reason and by the
        # same measurement: this row is rewritten IN FULL on every turn end, so
        # anything that grows with the conversation is paid for once per turn
        # forever. On the reference machine the checkpoint rows were 64.3% of a
        # 103 MB transcript. The fold is lossless for cost (see
        # ``_folded_components``), so the restored spend is unchanged — which
        # is the only property the durable copy owes anyone.
        durable = state.model_copy(
            update={
                "live_events": [],
                "jobs": [
                    job.model_copy(
                        update={
                            "trajectory": [],
                            "todos": None,
                            "usage": (
                                job.usage.model_copy(
                                    update={
                                        "cost_components": _folded_components(
                                            job.usage.cost_components
                                        )
                                    }
                                )
                                if job.usage is not None
                                else None
                            ),
                            "descendant_usage": _folded_components(job.descendant_usage),
                        }
                    )
                    for job in state.jobs
                ],
            }
        )
        await transcript.append_custom(
            FRONTEND_CHECKPOINT_CUSTOM_TYPE,
            {"checkpoint_id": checkpoint_id, "state": durable.model_dump(mode="json")},
        )

    @staticmethod
    def _jobs(session: Any) -> list[JobState]:
        manager = getattr(session, "jobs", None)
        try:
            rows = manager.list() if manager else []
        except Exception:
            return []
        # Execution stays local to each manager; presentation follows the shared
        # graph. Snapshot its ledgers once, without per-node scans or disk reads.
        comms = getattr(session, "_subagent_comms", None)
        graph_jobs = getattr(comms, "job_rows", None)
        if callable(graph_jobs):
            graph_rows = graph_jobs()
            if isinstance(graph_rows, Sequence):
                rows = graph_rows
        values: list[JobState] = []
        for job in rows:
            try:
                value = JobState.from_job(job)
                values.append(_with_lineage(value, comms) if comms is not None else value)
            except Exception:
                # One malformed extension row cannot erase unrelated jobs.
                continue
        nodes = getattr(comms, "nodes", None)
        if callable(nodes):
            known = {job.id for job in values}
            graph_nodes = nodes()
            for node in graph_nodes if isinstance(graph_nodes, Sequence) else []:
                if node.job_id in known:
                    continue
                # After a cold restart nested execution ledgers no longer exist,
                # but the shared durable graph still does. These are reader rows
                # only: never register a fake runnable job or invent a trajectory.
                row = JobState(
                    id=node.job_id,
                    type="task",
                    label=node.label,
                    status=getattr(node, "status", "gone"),
                    restored=True,
                    prompt=getattr(node, "prompt", ""),
                    agent_role=getattr(node, "agent_role", ""),
                    effort=getattr(node, "effort", ""),
                    result_text=getattr(node, "result_text", ""),
                    error_text=getattr(node, "error_text", ""),
                )
                values.append(_with_lineage(row, comms))
        return values


def format_context_tokens(tokens: int) -> str:
    """Compact context estimate: ``12.4k`` / ``1.2m`` style, plain under 1k.

    Lives HERE rather than in `tui/widgets/status_line.py` because both a TUI
    and a headless runtime render `/context` rows now, and importing the
    widget module to reach it costs 478 ms and drags all of Textual into a
    detached runtime that has no terminal (round 4, R2/U13). `status_line`
    re-exports it, so the two surfaces cannot drift into two different
    roundings of the same number.
    """
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}m"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.1f}k"
    return str(tokens)


def format_window(window: int) -> str:
    """Abbreviate a context window for the denominator: ``1M``, ``200k``.

    Capital ``M`` and lower-case ``k`` are the conventional units for model
    windows, and a whole window renders without a decimal (``1M``, not
    ``1.0M``) — the denominator is a label, not a measurement.
    """
    if window >= 1_000_000:
        scaled = window / 1_000_000
        return f"{scaled:.0f}M" if scaled == int(scaled) else f"{scaled:.1f}M"
    if window >= 1_000:
        scaled = window / 1_000
        return f"{scaled:.0f}k" if scaled == int(scaled) else f"{scaled:.1f}k"
    return str(window)


def _slash_capabilities() -> list[SlashCapability]:
    # Imported lazily so module import remains headless-safe; a full frontend
    # store needs the authoritative registry rather than a duplicated name list.
    from local_operator.slash_commands import SLASH_COMMANDS

    values = []
    for command in SLASH_COMMANDS:
        scope = (
            CommandScope.FRONTEND_LOCAL
            if command.name in _FRONTEND_LOCAL_SLASHES
            else CommandScope.AUTHORITATIVE_SESSION
        )
        values.append(
            SlashCapability(
                command=command.name,
                scope=scope,
                operation=(None if scope is CommandScope.FRONTEND_LOCAL else "slash"),
                supports_images=command.name in _IMAGE_SLASHES,
            )
        )
    # ``/mcp`` is advertised once but its scope is argument-dependent: bare it
    # is a local listing, with a grant subcommand it is authoritative. The
    # capability records the authoritative shape (the one that needs routing);
    # the follower's dispatch keeps the bare listing local by inspecting args.
    for capability in values:
        if capability.command == "mcp":
            capability.scope = CommandScope.AUTHORITATIVE_SESSION
            capability.operation = "slash"
    return values


def _wire_launch_prompts(value: Any) -> dict[str, str]:
    """Bound the launch-identity map before it is stamped onto a job row.

    See :data:`JOB_LAUNCH_PROMPT_WIRE_CHARS` for the budget and for why the
    bound is applied HERE rather than in :func:`sync_wire_payload` beside the
    other text caps: :func:`_job_summary` serializes the delta stream directly
    off the model, so anything left unbounded on ``JobState`` rides every
    subsequent frame however well the attach snapshot is trimmed.

    Truncation is marked with an ellipsis for the same reason the other text
    bounds mark theirs — a silent cut reads as the whole instruction — and the
    newest entries are kept because durable history pages from the tail.
    """
    if not isinstance(value, Mapping):
        return {}
    bounded: dict[str, str] = {}
    # Insertion order is launch order (``prior_launch_prompts`` merges oldest
    # first, then the current attempt), so the tail is the newest attempts.
    items = list(value.items())
    for key, prompt in items[-JOB_LAUNCH_PROMPTS_MAX:]:
        identity = str(key or "").strip()
        # Stripped before the falsy check, like ``identity`` above (round-1
        # finding 3): a whitespace-only prompt otherwise rode the wire verbatim
        # and was then discarded by the view's own strip, spending bytes for an
        # entry that could never reconcile. The drop decision belongs here,
        # owner-side, where it is made once.
        text = str(prompt or "").strip()
        if not identity or not text:
            continue
        if len(text) > JOB_LAUNCH_PROMPT_WIRE_CHARS:
            text = text[:JOB_LAUNCH_PROMPT_WIRE_CHARS] + "\u2026"
        bounded[identity] = text
    return bounded


def _with_lineage(job: JobState, comms: Any) -> JobState:
    """Stamp one job's canonical parent/child identity from the comms tree.

    The comms registry — not the job manager — knows who launched whom, so the
    lineage is merged here at snapshot time rather than carried on the job
    itself. Read defensively: a job with no comms record (a bash job, a swept
    child) keeps ``parent_job_id=None`` and simply has no navigation targets.
    """
    try:
        node = comms.node(job.id)
    except Exception:
        node = None
    if node is None:
        return job
    from local_operator.tools.builtin import TODO_STORE

    child_id = getattr(node, "session_id", None)
    has_plan = bool(child_id) and (bool(getattr(node, "live", False)) or child_id in TODO_STORE)
    # Stringified: the wire is JSON and ``Path`` is not. A follower's
    # ``SnapshotSubagentComms`` turns it back into a ``Path``; the owner-side
    # view never reads this field, it asks the live registry directly.
    session_dir = getattr(node, "session_dir", None)
    return job.model_copy(
        update={
            "parent_job_id": getattr(node, "parent_job_id", None),
            "session_id": getattr(node, "session_id", None),
            "session_dir": str(session_dir) if session_dir is not None else None,
            # Launch-row reconciliation identity. From the NODE, not the job
            # row: a live job knows only its own attempt, while the node
            # carries every attempt #314 collapsed into this record. The job's
            # own ``launch_message_id`` is the fallback for a node that has
            # none, mirroring how the owner-side view reads the two.
            "launch_message_id": str(
                getattr(node, "launch_message_id", "")
                or getattr(job, "launch_message_id", "")
                or ""
            ),
            "launch_prompts": _wire_launch_prompts(getattr(node, "launch_prompts", None)),
            "attempt_aliases": list(getattr(node, "attempt_aliases", ())),
            "todos": (
                [phase.model_dump(mode="json") for phase in _todo_state(node.session_id)]
                if has_plan
                else None
            ),
        }
    )


def _cost_knowledge(cost: float | None, unknown: bool) -> CostKnowledge:
    if cost is None:
        return CostKnowledge.UNKNOWN
    return CostKnowledge.PARTIAL if unknown else CostKnowledge.EXACT


def _ledger_cost(session: Any) -> dict[str, Any]:
    manager = getattr(session, "jobs", None)
    accounting = getattr(manager, "accounting_components", None)
    if not callable(accounting):
        return {}
    components = accounting()
    cost, unknown = cost_summary(components)
    return {
        "subagent_cost": cost,
        # An empty ledger has no missing bills; an unknown-only nonempty ledger
        # does. PARTIAL keeps that distinction when parent spend is known.
        "subagent_cost_knowledge": (CostKnowledge.PARTIAL if unknown else CostKnowledge.EXACT),
    }


def _job_subtree_cost(job: Any, *, default_model_label: str) -> float | None:
    """Direct plus nested descendant spend for one root job row.

    Mirrors the harness accounting (`jobs.py`): each descendant component is
    priced at ITS OWN serving identity, never the parent's rate. Any
    unpriceable component returns ``None`` so the prior figure is retained
    rather than silently undercounted — the same honesty rule the legacy
    harvest applied.
    """
    direct = job_cost(job, default_model_label=default_model_label)
    components = list(getattr(job, "descendant_usage", None) or [])
    if direct is None and not components:
        return None
    descendant = 0.0
    for component in components:
        provider = getattr(component, "provider", None) or ""
        model_id = getattr(component, "model_id", None) or ""
        cost = turn_cost(f"{provider}/{model_id}" if provider else model_id, component)
        if cost is None:
            return None
        descendant += cost
    return (direct or 0.0) + descendant


def _last_turn_outcome_from(session: Any, current: str) -> str:
    """Prefer the session's published outcome; keep the store's if it has none.

    ``observe_event`` writes ``last_turn_outcome`` onto the store from the
    emitted ``AgentEndEvent``. ``refresh_from_session`` then copies the
    session's fields over the store — and a reduced test double (or a
    runtime that has not yet grown the attribute) would wipe a just-written
    value back to ``""``, which a rebinding viewer treats as aborted.
    """
    if not hasattr(session, "_last_turn_outcome"):
        return current if current in ("completed", "aborted", "error") else ""
    raw = str(getattr(session, "_last_turn_outcome", "") or "")
    return raw if raw in ("completed", "aborted", "error") else ""


def _label(spec: Any) -> str:
    if spec is None:
        return ""
    return f"{getattr(spec, 'provider', '')}/{getattr(spec, 'model_id', '')}".strip("/")


_STATE_FIELD_ADAPTERS: dict[str, TypeAdapter[Any]] = {}


def _validate_state_field(key: str, value: Any) -> Any:
    """Validate one changed field without rebuilding the unrelated state."""
    field = FrontendSessionState.model_fields.get(key)
    if field is None:
        raise ValueError(f"unknown frontend state field: {key}")
    adapter = _STATE_FIELD_ADAPTERS.get(key)
    if adapter is None:
        # The key space is the model's finite field set, so this cache is
        # intrinsically bounded and avoids rebuilding pydantic schemas per delta.
        adapter = _STATE_FIELD_ADAPTERS[key] = TypeAdapter(field.annotation)
    return adapter.validate_python(value)


def _json_value(value: Any) -> Any:
    """Fully JSON-shape a candidate so equality against dumped state is real.

    ``mutate`` decides "changed" by comparing this against the state's
    ``model_dump(mode="json")``. A list of pydantic models (jobs, capabilities)
    left as model instances can NEVER equal its dict form, which made change
    detection constant-true and published a sequence-consuming frame on every
    no-op refresh — so recurse into containers, not just the top level.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return copy.deepcopy(value.__dict__)
    return copy.deepcopy(value)


def _todo_state(session_id: str) -> list[TodoPhaseState]:
    try:
        from local_operator.tools.builtin import todo_snapshot

        raw = todo_snapshot(session_id)
    except Exception:
        return []
    phases: list[TodoPhaseState] = []
    for phase in raw or []:
        if "items" in phase:
            phases.append(TodoPhaseState.model_validate(phase))
        else:
            phases.append(TodoPhaseState(name="Todos", items=[TodoItemState.model_validate(phase)]))
    return phases


def _wake_state(scheduler: Any) -> list[WakeState]:
    try:
        return [WakeState.model_validate(schedule.model_dump()) for schedule in scheduler.schedules]
    except Exception:
        return []


def _mcp_state(manager: Any, startup: Any) -> list[McpServerState]:
    names: set[str] = set()
    failures = dict(getattr(startup, "failures", {}) or {})
    if manager is not None:
        try:
            names.update(manager.get_all_server_names())
        except Exception:
            pass
    names.update(getattr(startup, "configured", ()) or ())
    values = []
    for name in sorted(names):
        status = "failed" if name in failures else "disconnected"
        if manager is not None:
            try:
                status = str(manager.get_connection_status(name) or status)
            except Exception:
                pass
        values.append(McpServerState(name=name, status=status, error=failures.get(name)))
    return values


def _aggregate_usage(usages: list[Usage]) -> Usage:
    last_context = next(
        (u.context_tokens for u in reversed(usages) if u.context_tokens is not None), None
    )
    components: list[Usage] = []
    for usage in usages:
        components.extend(
            component.model_copy(deep=True) for component in (usage.cost_components or [usage])
        )
    return Usage(
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        cache_read_tokens=sum(u.cache_read_tokens for u in usages),
        cache_write_tokens=sum(u.cache_write_tokens for u in usages),
        # Same contract as ``cache_write_tokens`` above: the TTL split folds
        # wherever the write count does, or the frontend's split would read
        # as the first call's value only (see ``Usage.cache_write_1h_tokens``).
        cache_write_5m_tokens=sum(u.cache_write_5m_tokens for u in usages),
        cache_write_1h_tokens=sum(u.cache_write_1h_tokens for u in usages),
        reasoning_tokens=sum(u.reasoning_tokens for u in usages),
        context_tokens=last_context,
        cost_components=components,
    )
