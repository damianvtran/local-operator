"""Row semantics shared by every human-facing surface.

WHY THIS MODULE EXISTS
----------------------
The TUI (``tui/session_presentation.py``) and the phone
(``mobile/projection.py``) each fold the same ``AgentMessage`` history into
rows. They are two *renderers* of one *contract*, and for a long time that
contract was asserted only by comments on each side — comments which were
wrong. An architecture review (``docs/design/history-fold-convergence.md``,
§3) enumerated twelve divergences that accumulated under that regime; eight
were user-visible, and five of them were rows the phone silently DROPPED
because its ``if/elif`` chain ended with no ``else``.

The decisions below are the ones both surfaces must make identically:
whether a message is harness chrome that no surface may attribute to the
user, whether a row was minted by the harness from a ``CustomMessage`` at
all (:func:`is_harness_injection`), what a timed-out gate says, what ink a
refused compaction gets, and which assistant turns produce a notice instead
of prose. Each is a pure function of a message, with **no host dependency**
— no Textual, no wire type, no session import at module scope — so both
hosts can call it and neither can own it.

The injection predicate reads a raw payload MAPPING as readily as a message,
because the subagents panel (``tui/widgets/subagent_view.py``) folds raw
transcript entry payloads rather than ``AgentMessage``s. It is the same
decision on a third surface, so it belongs here rather than in that fold.

CONSTRAINT — this module must stay host-free. It sits below both renderers
the same way ``compaction/marker.py`` sits below the hosts that must not
import the session. The chrome prompts live in modules that would be
the wrong import direction from here (``session.goal_loop``,
``session.session``, ``harness.loop``), so they are gathered behind a
function that imports
them lazily. That keeps ONE list rather than a copy per host — the partial
copy is precisely the drift signature the review named: the phone had
suppressed one of the three prompts and rendered the other two as the
user's own words.

Adding a new row decision belongs HERE, not in a host. A decision made in
one renderer is a decision the other will not make.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

#: The severity vocabulary shared by the surfaces. A superset is deliberately
#: NOT used: these are the tiers a *replayed* row can carry, and the TUI's
#: wider ``NoticeKind`` (which adds ``note``/``success`` for live receipts)
#: accepts every one of them, so a value produced here is always renderable
#: on both sides.
NoticeSeverity = Literal["info", "warning", "error"]

#: Message heads the harness mints for its own notices, each with the producer
#: that writes it, and the ONE enumeration a display surface uses to recognise a
#: notice row (see :func:`is_harness_notice_row`).
#:
#: Why an enumeration is acceptable HERE, when the guard it feeds replaced one:
#: these are the lines the harness itself writes, not a policy about content, and
#: a missed head degrades to the pre-fix behaviour — a notice painted as the
#: user's words, which the operator can see and report — rather than to losing
#: anything of theirs. The stamp (:func:`is_harness_injection`) stays the primary
#: test for everything minted since it existed; this list is what covers the rows
#: written before it did, and the ones a compaction marker carried from that era.
#:
#: ``[`` + the elision notice is NOT here: that row has a fixed id
#: (``PRESERVED_TURN_ELISION_ID``, ``compaction-elision``), which is provenance
#: rather than wording, so the id prefix is checked instead.
#:
#: The DELIVERY ENVELOPES (``<parent-message>``, ``<subagent-message …>``,
#: ``<peer-session-message …>``) are deliberately NOT here either, and the reason is
#: what already covers a copy of one rather than a parser: a carried envelope copy
#: is shed by PROVENANCE — ``cap_preserved_user_turns``' ``injection_ids``, resolved
#: against the journal, where the stored turn's id names the entry whose
#: ``custom_type`` is the delivery — so it never needs a wording rule. Measured
#: fleet-wide (14 sessions): 229 of 233 carried envelope copies are shed that way on
#: BOTH heads. The residue is NOT shed, and what a surface then shows depends on the
#: surface: the phone fold and the subagents panel PARSE the envelope
#: (``mobile.projection`` calls :func:`extract_parent_message`; the panel has its own
#: call) and paint a parent receipt, while the TUI fold has NO envelope parser and
#: paints the model-facing envelope as the operator's own words. That difference is
#: pre-existing on ``main`` and is not changed here: the PR records the measurements
#: and leaves the fix — a display product call for hub steers — to a follow-up. It is
#: also a shape a person quotes verbatim when asking about it, which a wording rule
#: would then eat
#: (`test_a_human_quoting_the_envelope_keeps_their_own_words` pins that). A notice
#: has no such provenance to fall back on: nothing about ``[model switch] …`` is
#: addressable, which is why text is the only test for one.
_HARNESS_NOTICE_HEADS: tuple[str, ...] = (
    "[model switch] ",  # incidents.format_model_switch_message
    "[session incident",  # incidents.Incident.render
    "[session credential] ",  # incidents.format_credential_message
    # ``[credential redaction] `` is retained deliberately after the notice it heads
    # stopped being written (2026-09-27). Nothing armed writes that head any more —
    # ``Session.journal_shape_incident`` is silent by the operator's ruling — but a
    # transcript written before that date still contains rows carrying it, and the
    # audit replay serves those STORED rows verbatim. Removing the head would make
    # an OLD session's persisted notice paint as the operator's own words on replay
    # on every surface that consults this tuple, which is exactly the defect the
    # head exists to prevent; the rule costs a live session nothing, because the
    # words no longer occur in one.
    "[credential redaction] ",  # incidents.format_shape_incident_message (legacy rows)
    "[mcp recovery] ",  # incidents.format_mcp_recovery_message
    # The MCP-unavailable WARNING, and the one head here that is not an
    # incident: a server whose tools are gone is a missing capability, not a
    # failed turn. It is listed for the reason every head is — a transcript
    # written before the ``harness_injected`` stamp existed can only prove its
    # own provenance by its opening words, and this row must never paint as the
    # operator's own sentence.
    "[session warning] ",  # incidents.format_mcp_unavailable_message
    "[session-state]\n",  # Session._system_state_message
    # The unattended-gate timeouts, in _default_convert_to_llm. Two heads rather
    # than the shared ``[system] `` prefix: that prefix is also minted by
    # ``harness.loop.CONNECTIVITY_CONTINUATION_PROMPT``, so it does not select the
    # gates. Both gate producers still match here, the connectivity prompt is hidden
    # by :func:`is_harness_chrome` on every surface regardless, and the narrowing is
    # therefore display-neutral — with one consequence recorded rather than hidden:
    # a CARRIED copy of the connectivity prompt is no longer shed by text, so it
    # returns to the model's context. No receipt changes visibly.
    "[system] The question for ",
    "[system] The approval request for ",
    "<system-reminder>",  # Session._todo_reminder_text
    "(alarm) Scheduled wake ",  # harness.wake.format_wake_delivery_text
)


def is_harness_notice_text(text: str) -> bool:
    """Whether ``text`` is one of the notice shapes the harness itself mints.

    The only evidence a row written before the ``harness_injected`` stamp
    existed can offer about its own provenance: a 2026-09-08-era transcript
    carries switch notices as plain ``role="user"`` rows with no
    ``provider_payload`` at all, and they are still there — the audit phase of
    the attached viewer replays them straight from the journal (a stored row is
    not a copy, so there is no id to look up and no payload to read).
    """
    return text.lstrip().startswith(_HARNESS_NOTICE_HEADS)


def harness_chrome_prompts() -> tuple[str, ...]:
    """Every prompt the harness injects as a user turn that NO surface paints.

    Each of these is persisted as a ``role="user"`` message because the
    TRANSCRIPT must record why the conversation continued — but the user did
    not type any of them, and painting one attributes the harness's words to
    them. The live path never shows them, so replay must not either.

    The four, and why each is persisted:

    - ``LOOP_PROMPT`` — the goal loop's self-continuation.
    - ``_CONTINUATION_PROMPT`` — the session's auto-continuation after a
      turn hit its step budget.
    - ``CONNECTIVITY_CONTINUATION_PROMPT`` — records why ONE answer arrived
      in two pieces across a network interruption; the live run showed a
      notice instead.
    - ``NAMELESS_CALL_RECOVERY_PROMPT`` — the re-ask after a tool call arrived
      with no name and was dropped; the live run showed a notice instead.

    This is the EXACT-match half of the decision, and it is no longer the whole
    of it: two fixed prompts that look like the user's words do not, because the
    GOAL the harness interpolates into them is called for. The goal-mode loop's
    working turn was the one that had neither leg (see
    :func:`~local_operator.session.goal_loop.is_loop_goal_instruction`).

    Imported lazily and gathered here rather than listed per host: this list
    having been copied INCOMPLETELY into the phone fold (it suppressed the
    third and rendered the first two) is the drift this function exists to
    make impossible. A fourth prompt is added in one place.

    This tuple is the EXACT-match half of the decision only. Chrome prompts
    with a family of shapes rather than a single string live in the producer's
    own recogniser instead, and there is one per family: the connectivity
    instruction, which ``harness.loop._continuation_instruction`` composes per
    cut (prose alone, an aborted tool call alone, or the two joined) — those
    cannot be enumerated here because the tool-call half interpolates the
    aborted calls' names — and the goal judge's continuation
    (``session.goal_judge``), which interpolates the goal. :func:`is_harness_chrome`
    therefore adds each producer's own recogniser; there is still exactly ONE
    decision, and it is this function plus those recognisers, not a list per
    surface.
    """
    from local_operator.harness.loop import (
        CONNECTIVITY_CONTINUATION_PROMPT,
        NAMELESS_CALL_RECOVERY_PROMPT,
    )
    from local_operator.session.goal_loop import LOOP_PROMPT
    from local_operator.session.session import _CONTINUATION_PROMPT

    return (
        LOOP_PROMPT,
        _CONTINUATION_PROMPT,
        CONNECTIVITY_CONTINUATION_PROMPT,
        NAMELESS_CALL_RECOVERY_PROMPT,
    )


def is_harness_chrome(text: str) -> bool:
    """Whether this user-role text is harness chrome rather than the user's words.

    Normalises before comparing because THE HOSTS DO NOT AGREE on what they
    hand in: the TUI strips a message's text at the top of its replay loop,
    the phone fold passes ``message.text`` verbatim. Leaving the strip to the
    callers is the exact substrate this module exists to remove — one surface
    would suppress a chrome prompt that the other painted as the user's own
    words the moment a persisted prompt gained a trailing newline.

    Two legs, one decision. Exact membership covers the three prompts that are
    single fixed strings; a producer-side recogniser covers each family the
    producer composes and therefore cannot enumerate — the loop's
    :func:`~local_operator.harness.loop.is_connectivity_continuation_instruction`
    for the "prose then an aborted tool call" instruction, the goal judge's
    :func:`~local_operator.session.goal_judge.is_goal_continuation_instruction`
    for the continuation that interpolates the standing goal, and the goal loop's
    :func:`~local_operator.session.goal_loop.is_loop_goal_instruction` for the
    working turn that interpolates its own. Equality alone used
    to be the whole test, and the composed connectivity instruction — the
    incident's own shape — was a member of neither, so a resumed session painted
    it (and the tool-call-only shape) as the operator's own words; the goal
    continuation would have done the same on every surface, since its text
    changes with the goal. The goal-mode loop's turn was the third instance of
    exactly that, and it had no recogniser at all.
    """
    from local_operator.harness.loop import is_connectivity_continuation_instruction
    from local_operator.session.goal_judge import is_goal_continuation_instruction
    from local_operator.session.goal_loop import is_loop_goal_instruction

    stripped = text.strip()
    return (
        stripped in harness_chrome_prompts()
        or is_connectivity_continuation_instruction(stripped)
        or is_goal_continuation_instruction(stripped)
        or is_loop_goal_instruction(stripped)
    )


def is_harness_injection(row: Any) -> bool:
    """Whether this row was minted BY the harness from a ``CustomMessage``.

    :func:`~local_operator.harness.render._default_convert_to_llm` renders a
    harness aside — a model-switch notice, a session incident, a wake
    delivery, a gate timeout — into a plain ``Message(role="user")`` and
    stamps ``provider_payload["harness_injected"]`` on it, so the row is
    structurally indistinguishable from an operator prompt once it exists.
    The stamp is compaction's provenance signal, and it is also the only
    signal a fold has: a row carrying it was never typed by a person, so no
    human-facing surface may paint it as their words. The live path never
    paints one either (the failover moment has its own receipt), which makes
    dropping it live/replay parity rather than a second opinion — the same
    doctrine :func:`is_harness_chrome` follows for the continuation and
    recovery prompts.

    Accepts EITHER a message-like object or a raw payload mapping, because
    the surfaces do not all read the same shape: the TUI and phone folds hold
    ``AgentMessage``s, while the subagents panel folds raw transcript entry
    payloads (``{"role": ..., "provider_payload": {...}}``). Making the
    caller adapt is how the second copy of a decision gets written.

    The marker constant is imported LAZILY, like :func:`harness_chrome_prompts`
    imports its prompts: ``compaction.cutpoint`` pulls ``compaction.tokens``,
    and this module sits on both folds' import path. Measured here, importing
    this module costs 2.3 ms while importing it together with
    ``compaction.cutpoint`` costs 308 ms — a module-scope import would pay
    that on every TUI and mobile start for one string constant.
    """
    from local_operator.compaction.cutpoint import RENDERED_INJECTION_KEY

    payload = _row_provider_payload(row)
    if not payload:
        # A replayed row can carry a malformed payload (an older writer, a
        # hand-edited journal). Absent proof of injection is not proof of it.
        return False
    return bool(payload.get(RENDERED_INJECTION_KEY))


def _row_provider_payload(row: Any) -> Mapping[str, Any]:
    """The ``provider_payload`` of a message or of a raw transcript payload."""
    payload = (
        row.get("provider_payload")
        if isinstance(row, Mapping)
        else getattr(row, "provider_payload", None)
    )
    # A malformed payload (an older writer, a hand-edited journal) reads as
    # empty: absent proof of provenance is not proof of it.
    return payload if isinstance(payload, Mapping) else {}


def _row_text(row: Any) -> str:
    """The text of a message, or of a raw transcript payload's content blocks.

    The payload arm is what the subagents panel holds; its ``content`` is the
    same list of ``{"type": "text", "text": ...}`` blocks the message type
    flattens into ``.text`` on the wire, so reading it here keeps the decision
    in one place rather than asking each raw-payload caller to pre-flatten.
    """
    if isinstance(row, Mapping):
        blocks = row.get("content") or ()
        if not isinstance(blocks, Sequence):
            return ""
        return "".join(
            str(block.get("text") or "") for block in blocks if isinstance(block, Mapping)
        )
    text = getattr(row, "text", "")
    return text if isinstance(text, str) else ""


def _row_id(row: Any) -> str:
    value = row.get("id") if isinstance(row, Mapping) else getattr(row, "id", None)
    return value if isinstance(value, str) else ""


def _row_custom_type(row: Any) -> str:
    value = (
        row.get("custom_type") if isinstance(row, Mapping) else getattr(row, "custom_type", None)
    )
    return value if isinstance(value, str) else ""


def _row_details(row: Any) -> Mapping[str, Any]:
    value = row.get("details") if isinstance(row, Mapping) else getattr(row, "details", None)
    return value if isinstance(value, Mapping) else {}


def is_hidden_wake_delivery(row: Any) -> bool:
    """Whether this row is a HIDDEN wake delivery (a patience fire).

    Hidden deliveries must be invisible on every human surface while their text
    stays in the model's context — the requirement is "no wake line, no card, no
    badge, no timer notification". The marker is ``details.hidden`` on the
    ``wake_prompt`` custom message (written by the delivery path), and it is
    read here so the fold, the tail-snap walk, the replay receipt and the
    desktop window all make ONE decision rather than four.
    """
    # Lazy import, the pattern :func:`is_harness_injection` documents: this
    # module sits on both folds' import path, and the constant's owner
    # (``harness/wake.py``) drags the scheduler's asyncio import with it.
    from local_operator.harness.wake import WAKE_PROMPT_MESSAGE_TYPE

    return _row_custom_type(row) == WAKE_PROMPT_MESSAGE_TYPE and bool(
        _row_details(row).get("hidden")
    )


#: Tool names whose rows must NEVER paint on a human surface.
#:
#: ``patience`` arms a hidden internal timer: R30 promises "no wake line, no
#: timer notification", and the tool's own copy tells the model the wait and
#: its fire "are invisible to the user" — the tool ledger row (``▸ patience
#: arm 90s ✓``) contradicted both (UX round 1, U2). The rows STAY in the
#: model's context, which is what lets it read what it armed; every
#: presentation skips them the way a hidden wake delivery is skipped.
HIDDEN_TOOL_NAMES: frozenset[str] = frozenset({"patience"})


def is_hidden_tool_name(name: Any) -> bool:
    """Whether a tool NAME is one whose rows must not render."""
    return str(name or "") in HIDDEN_TOOL_NAMES


def is_hidden_tool_call(call: Any) -> bool:
    """Whether a ``ToolCall``-shaped object names a hidden tool."""
    return is_hidden_tool_name(getattr(call, "name", None))


def is_hidden_tool_row(row: Any) -> bool:
    """Whether a stored row is a hidden tool's RESULT or CALL row.

    Two stored shapes carry one call: the ``role: "tool"`` row carries its own
    ``tool_name``, and the assistant row carries ``tool_calls[].name``. Only a
    row that carries NOTHING visible is hidden by the calls it also happens to
    mention — an assistant row with prose keeps rendering (its patience entry
    is skipped by :func:`is_hidden_tool_call` at the paint sites that mount
    per-call rows), and a result row is judged by its own ``tool_name`` rather
    than by remembering a call that may sit on a page this row's does not.
    """
    payload = row.get("payload") if isinstance(row, Mapping) else getattr(row, "payload", None)
    if not isinstance(payload, Mapping):
        return False
    if str(payload.get("tool_name") or "") in HIDDEN_TOOL_NAMES:
        return True
    calls = payload.get("tool_calls") or ()
    if not calls:
        return False
    if str(payload.get("role") or "") != "assistant" or _row_text(payload).strip():
        return False
    return all(
        str(call.get("name") if isinstance(call, Mapping) else getattr(call, "name", "") or "")
        in HIDDEN_TOOL_NAMES
        for call in calls
    )


def is_hidden_tool_message(message: Any) -> bool:
    """Whether a rendered ``Message`` is a hidden tool's RESULT or CALL row.

    The replay-side twin of :func:`is_hidden_tool_row`, and deliberately a
    SEPARATE function rather than a wider one: they read different shapes —
    that one a stored row wrapping a ``payload``, this one the ``build_llm_
    history`` output the history window serves — and merging them would make a
    stored-row caller silently depend on ``Message``'s attributes. Same
    two-shape rule, because the window builder must subtract exactly what the
    paint seams would have skipped: a ``role: "tool"`` row by its own
    ``tool_name``; an assistant row only when it carries no visible text and
    every call on it is hidden (prose keeps rendering; its hidden call is
    skipped per-call at the paint sites).
    """
    role = str(getattr(message, "role", "") or "")
    if role == "tool":
        return is_hidden_tool_name(getattr(message, "tool_name", None))
    if role != "assistant":
        return False
    calls = getattr(message, "tool_calls", None) or ()
    if not calls:
        return False
    if str(getattr(message, "text", "") or "").strip():
        return False
    return all(is_hidden_tool_call(call) for call in calls)


def is_ask_gate_divert_details(details: Any) -> bool:
    """Whether a tool result's ``details`` mapping carries the divert marker.

    THE ASK GATE'S MARKER (design ``docs/design/ask-gate.md`` §3). A diverted
    ask's tool result carries ``{"ask_gate": {"hidden": True, "verdict": …,
    "reason": …}}`` — the ONE fact every human surface reads to decide "this
    call/result pair never happened for the user". The marker is read through
    this predicate set rather than re-derived per host, so a surface cannot
    disagree with another about what a divert looks like.
    """
    if not isinstance(details, Mapping):
        return False
    gate = details.get("ask_gate")
    if not isinstance(gate, Mapping):
        return False
    return bool(gate.get("hidden"))


def is_ask_gate_divert_row(row: Any) -> bool:
    """Whether a stored ``{type, payload}`` row is a diverted ask's RESULT row.

    The desktop rows path's shape: ``Message.tool_result`` writes the marker
    into ``provider_payload.details`` and ``encode_message_payload``
    serializes it, so a stored row carries it at
    ``payload.provider_payload.details`` — the same key the transcript replay
    reads off the rendered message (:func:`is_ask_gate_divert_message`), seen
    one serialization earlier. Filtering SERVER-side is the point (design §3
    row 5): the client reducer has no filter of its own, so a build that
    predates the marker would otherwise paint it.
    """
    payload = row.get("payload") if isinstance(row, Mapping) else None
    if not isinstance(payload, Mapping):
        return False
    provider_payload = payload.get("provider_payload")
    if not isinstance(provider_payload, Mapping):
        return False
    return is_ask_gate_divert_details(provider_payload.get("details"))


def is_ask_gate_divert_message(message: Any) -> bool:
    """Whether a rendered ``Message`` is a diverted ask's RESULT row.

    The replay-side twin of :func:`is_ask_gate_divert_row`, and deliberately a
    SEPARATE function for the reason :func:`is_hidden_tool_message` states —
    the two read different shapes (a stored row wrapping a ``payload``, this
    one the ``build_llm_history`` output every fold walks), and merging them
    would make a stored-row caller silently depend on ``Message`` attributes.
    """
    payload = getattr(message, "provider_payload", None)
    if not isinstance(payload, Mapping):
        return False
    return is_ask_gate_divert_details(payload.get("details"))


def ask_gate_diverted_call_ids(messages: Iterable[Any]) -> set[str]:
    """The ``tool_call_id``s whose RESULT carries the divert marker.

    ONE derivation for every fold's call-chip skip (design §3: "one helper, no
    per-fold drift"): a diverted ask leaves TWO rows — its result (marked) and
    the assistant call that made it — and the only place a fold can learn the
    call id is the marker on its result. Folds that hold a message set compute
    this once and consult it for the call chip, the settle skip and the
    up-front indexes alike.
    """
    call_ids: set[str] = set()
    for message in messages:
        if not is_ask_gate_divert_message(message):
            continue
        call_id = getattr(message, "tool_call_id", None)
        if isinstance(call_id, str) and call_id:
            call_ids.add(call_id)
    return call_ids


def without_ask_gate_divert(message: Any, call_ids: set[str]) -> Any | None:
    """``message`` with any ask-gate divert subtracted; ``None`` when it goes.

    A diverted ask contributes two rows — the marked result and the assistant
    row that made the call — and an owner-side display seam must subtract
    BOTH so every viewer build, an older one that cannot read the marker
    included, receives clean rows (design §3 row 4, the wake-fire id-set
    helper's shape). An assistant row keeps its prose and any other calls and
    loses only the diverted ones; a row left with no prose and no calls is
    dropped, exactly the rule :func:`is_hidden_tool_message` applies to a
    prose-free hidden call. ``call_ids`` comes from
    :func:`ask_gate_diverted_call_ids` over the whole message set — the marker
    can sit several messages away from the call it names.
    """
    if is_ask_gate_divert_message(message):
        return None
    if not call_ids:
        return message
    calls = list(getattr(message, "tool_calls", None) or ())
    if not calls:
        return message
    kept = [call for call in calls if str(getattr(call, "id", "") or "") not in call_ids]
    if len(kept) == len(calls):
        return message
    if kept:
        return message.model_copy(update={"tool_calls": kept})
    if str(getattr(message, "text", "") or "").strip():
        return message.model_copy(update={"tool_calls": []})
    return None


#: The ``ask`` tool's name, spelled once for the settle-only predicate. The
#: literal lives with the tool's builder (``tools/builtin.py``); this module
#: only ever compares against it, so a rename shows up here as a predicate
#: that stops matching rather than as a silent pass.
_ASK_TOOL_NAME = "ask"


def is_settle_only_ask(tool_name: Any, *, queued_engine: bool) -> bool:
    """Whether a call's rows are SETTLE-ONLY on a human surface (design §3).

    While the queued engine is live an ``ask`` call gets NO live row — the
    forked clearance check may divert it, and a row that flashed for the
    gate's whole duration on every surface is what the design rejects — so
    its one row is created at SETTLE: the receipt for a raise, nothing for a
    divert (the marker is read there). ``queued_engine`` is the caller's mode
    read;
    ``False`` means today's mounting plus the settle-marker drop, which is the
    fallback for the blocking arm and for an un-negotiated mixed build alike.
    """
    return bool(queued_engine) and str(tool_name or "") == _ASK_TOOL_NAME


def queued_ask_engine_live(session: Any) -> bool:
    """Whether the session behind a surface runs the queued-ask engine.

    THE modal read for the ask-gate seams (design §3), one function so the
    owner probe and the viewer probe cannot drift into two answers about one
    session:

    * a session that OWNS its queue answers ``ask_queue()`` — ``None``
      exactly on the blocking arm (the flag off, or no host that can show an
      ask), and constructing it is side-effect-free and already happens at
      turn binding;
    * a VIEWER — a facade with no queue behind it — reads the presence of the
      ``asks`` OR ``asks_open`` wire field off its frontend state, which the
      owner publishes only while the queued engine is live in its process
      (``FrontendSessionState.asks`` documents the capability-proxy rule; the
      WIRE FIX added ``asks_open`` to the read because a live-but-empty queue
      publishes the tally with the rows absent, and a viewer that read only
      ``asks`` would misclassify it as the blocking arm — painting a live ask
      row that then vanished, the flash the settle-only rule exists to stop).

    An un-negotiated mixed build answers ``False``; callers keep today's
    mount and the settle marker still drops the trace (the flash residual the
    design records, §5). Never raises: a probe that cannot answer is a
    ``False`` — the settle-only read only decides WHEN a row is created, and
    "cannot say" must land on today's behaviour, not on an exception in a
    paint path.
    """
    probe = getattr(session, "ask_queue", None)
    if callable(probe):
        try:
            return probe() is not None
        except Exception:  # noqa: BLE001 — an unreadable queue reads as the blocking arm
            return False
    try:
        state = getattr(session, "frontend_state", None)
    except Exception:  # noqa: BLE001 — a facade that cannot say
        return False
    if state is None:
        return False
    return getattr(state, "asks", None) is not None or getattr(state, "asks_open", None) is not None


def is_harness_notice_row(row: Any) -> bool:
    """Whether this row is harness-authored and must not paint as the user's words.

    The decision every human-facing surface and every title/query/tail scan
    makes, in one place, and it holds in BOTH phases a display can serve: the
    context replay (where the row may be a stamped render or a copy a compaction
    marker carried) and the audit replay (where it is the stored row itself).

    THREE shapes answer True now, and they need different evidence:

    * **a HIDDEN wake delivery** (:func:`is_hidden_wake_delivery`) — a patience
      fire is harness-internal by requirement; it must not paint as a wake
      receipt, a user row, or a fold anchor. Checked FIRST because it is the
      one shape the stamp below cannot recognise on the raw payload (the
      stamp lands on the rendered user message, while the fold sometimes holds
      the CustomMessage itself).
    * **the stamp** (:func:`is_harness_injection`) — the renderer minted the row
      from a ``CustomMessage`` in this process. The primary test, and the only
      provenance a live or freshly rendered row carries.
    * **a NOTICE head** (:func:`is_harness_notice_text`) — what remains when the
      row predates the stamp: the plain stored notices a pre-stamp build wrote
      into the journal, which the audit phase serves verbatim, and the copies a
      compaction marker re-seats from that era.

    **The cost, stated exactly because it is a trade rather than a free win.** A
    person who pastes a harness notice verbatim loses their display row: the text
    is hidden on every human surface. It is NOT lost anywhere else — the row is
    still in the journal, and still in the model's context; only the renderer drops
    it, exactly as :func:`is_harness_chrome` does for the three loop/continuation
    prompts, which a person can equally paste verbatim. Those two are the verified
    retention surfaces, and they are named alone because a claim about anywhere
    else would be one nobody measured. That precedent is the reason this is
    acceptable at all: a display that must decide from text will occasionally
    hide something a person wrote, and the alternative — leaving the harness's own
    words behind the user gutter on a surface that has no other provenance to read
    — is the defect being fixed. The elision notice is identified by ID rather
    than wording: it has a fixed one (``compaction-elision``).

    Accepts a message-like object or a raw payload mapping, for the reason
    :func:`is_harness_injection` documents.
    """
    from local_operator.compaction.cutpoint import PRESERVED_TURN_ELISION_ID_PREFIX

    if is_hidden_wake_delivery(row):
        return True
    if is_harness_injection(row):
        return True
    if _row_id(row).startswith(PRESERVED_TURN_ELISION_ID_PREFIX):
        return True
    return is_harness_notice_text(_row_text(row))


def typed_line_of(text: str) -> str | None:
    """The ``$skill`` line behind a persisted payload, or ``None``.

    A ``$skill`` invocation persists as its EXPANDED payload, because the
    payload is what the model was sent. Painting it verbatim shows the whole
    SKILL.md body as the user's row — and, because a session is titled from
    its first user turn, titles the thread after the skill body too.

    Lazy-imported and failure-swallowing by contract: a broken or absent
    skills package must never stop a transcript replaying, on either surface.
    Every failure degrades to "not an invocation", so the caller falls
    through to painting the text verbatim.
    """
    try:
        from local_operator.skills.invoke import typed_line_of as _typed_line_of

        return _typed_line_of(text)
    except Exception:  # noqa: BLE001 — replay must never fail on this
        return None


def reference_block_stripped(text: str) -> str:
    """``text`` without the ``<operator-references>`` blocks appended to it.

    An ``@path`` reference is expanded ONCE, at submit, and the expansion is
    appended to the message as a block. The model needs that block; the
    transcript must not show it, or a one-line question about a file paints as
    the whole file. Exactly the failure :func:`typed_line_of` exists to prevent
    for a skill payload.

    Stripping HERE rather than in a host is the point (design R5). Both surfaces
    paint through :func:`user_row_text`, so a strip written in the TUI would
    leave the phone showing the block — which is precisely how one surface got
    the skill rule and the other did not.

    EVERY COMPLETE BLOCK goes, and the spans come from
    ``references.reference_block_spans`` rather than being found here. Both
    halves are load-bearing:

    - COMPLETE is what keeps the marker quotable. The tag is part of the
      product's visible vocabulary now, so asking about this feature, pasting a
      log line, or quoting a prompt is ordinary — and an unanchored ``find``
      truncated every one of those messages at the word they quoted: "why does
      my message contain <operator-references> in it?" painted as "why does my
      message contain", silently, with no notice, on both surfaces. The
      transcript showing strictly less than what was typed is the failure R4
      names. A marker the operator quoted is not followed by the block's own
      preamble line, so it is not a span; an UNCLOSED block is not provably one
      and reports nothing either (truncated history, a message cut mid-write —
      showing too much is recoverable, showing less than the operator typed, with
      no notice, is not).
    - PASTED, not only appended. The anchor is the preamble line plus the
      closer, so a COMPLETE block the operator pasted, quoted or forwarded is
      stripped too, and that is intended rather than overlooked (review round 2,
      MINOR-2). An appended block and a pasted one are the same bytes and there
      is no sound way to tell them apart — which is exactly why base anchored
      to the END of the message, and why that anchor broke the ordinary
      forwarding shape. The accepted cost is that asking about a complete
      payload of this feature paints less than was typed; the property bought
      for it is that the ordinary case cannot leak a file body into the row.
    - EVERY is because a message can carry more than one. Expansion only ever
      APPENDS, so a second pass over text that already held a block leaves the
      FIRST one mid-message: pass 1's block, then prose carrying a new token,
      then pass 2's block. Stripping only the trailing block painted 6,496
      characters of file body — the whole of the first block — into the row,
      which is the failure this function exists to prevent, reached through the
      ordinary forwarding shape (a subagent launch re-prompting a manager's own
      text).

    The block's grammar lives with the code that writes it, because recognising
    a real block means knowing the open marker AND the preamble line; a second
    copy of that rule here is how the TUI and the phone came to disagree in the
    first place.

    Lazy-imported and failure-swallowing by contract, like :func:`typed_line_of`
    above and for the same reason: this module stays host-free, and a broken or
    absent resolver must never stop a transcript replaying. Every failure
    degrades to "no block here", so the caller paints the text verbatim.
    """
    try:
        from local_operator.references import reference_block_spans
    except Exception:  # noqa: BLE001 — replay must never fail on this
        return text
    spans = reference_block_spans(text)
    if not spans:
        return text
    kept: list[str] = []
    cursor = 0
    for start, end in spans:
        # The resolver appends a block as ``JOIN + block``, so the separator
        # before one belongs to the block rather than to the operator's prose:
        # keeping it would paint a blank line where the block stood, turning
        # "first @a.txt and now @b.txt" into a two-paragraph row.
        kept.append(text[cursor:start].rstrip())
        cursor = end
    kept.append(text[cursor:].rstrip())
    # A SEPARATOR where two segments would otherwise fuse. Each segment is
    # right-stripped, because the separator BEFORE a block belongs to the block
    # rather than to the operator's prose, so a segment can end at a word — and
    # a caller that appends text to an expanded message with no space of its own
    # then painted `first @a.txtand next` (review round 2, NIT-1). A single
    # space only when the next segment starts on a non-space character: the
    # ordinary shapes already carry their own leading whitespace, so they are
    # joined byte-identically to before.
    joined = ""
    for segment in kept:
        if not segment:
            continue
        if joined and not segment[0].isspace():
            joined += " "
        joined += segment
    return joined


def user_row_text(text: str) -> str:
    """What a surface paints for a ``role="user"`` message.

    Collapses a skill payload to the line the user actually typed, strips an
    expanded reference block, and leaves every other message alone. Kept beside
    :func:`is_harness_chrome` because the two are the whole of the "what did the
    user really say" decision, and splitting them across hosts is how one
    surface got the skill rule and the other did not.

    The reference block is removed FIRST, before the skill fallback: a
    `$skill` invocation whose request cited a file carries both, and
    :func:`typed_line_of` parses the payload envelope, which a trailing block
    would sit outside of and survive.

    Strips for the same reason :func:`is_harness_chrome` does: the two hosts
    normalise differently, so an envelope with surrounding whitespace would
    resolve to a typed line on the TUI and to the whole SKILL.md body on the
    phone. The fallback returns the stripped text so both surfaces paint one
    row, not one padded and one not.
    """
    stripped = reference_block_stripped(text).strip()
    return typed_line_of(stripped) or stripped


def assistant_row_text(text: str) -> str:
    """What a surface paints for a ``role="assistant"`` message, or ``""``.

    The counterpart to :func:`user_row_text`, and it exists for the same
    reason: THE HOSTS DO NOT AGREE on what they hand in. The TUI strips a
    message's text at the top of its replay loop and tests the stripped
    value, so a whitespace-only turn produces NO block there. The phone
    tested ``message.text`` verbatim, and ``"   "`` is truthy — so the same
    history produced an extra EMPTY assistant row on the phone, ~8px of
    blank space and, more importantly, a row SEQUENCE the TUI never emits.

    That is the weaker version of this module's whole claim. "The two folds
    produce the same rows" is worth something; "the same rows plus one blank
    one" is a divergence with a smaller pixel budget, not an agreement.

    Returning the stripped text rather than a bare truthiness verdict also
    settles the content half: both surfaces paint the same bytes, so a
    padded message cannot render one row indented and the other not.
    """
    return text.strip()


def gate_waited_text(details: Mapping[str, Any] | None) -> str:
    """The gate's own wait, as a short span — ONE implementation, two audiences.

    Shared by :func:`gate_timeout_notice` (the transcript row the HUMAN reads) and
    ``harness/render.py`` (the row the MODEL reads, ``GATE_TIMEOUT_CUSTOM_TYPE``):
    both report the same expiry, and a second formatter for one number is how the
    wrong one survived — this used to floor at one hour (``max(1, waited //
    3600)``), so a 30-second expiry rendered as "waited 1h" (round 3, D12).
    Whatever a reader is told, it must be the wait that actually happened.

    An absent or unreadable value says so rather than rounding up to an hour.
    """
    waited = (details or {}).get("waited_s")
    try:
        seconds = float(waited) if waited is not None else 0.0
    except (TypeError, ValueError):
        seconds = 0.0
    if seconds <= 0:
        return "a while"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def ask_response_notice(details: Mapping[str, Any] | None) -> tuple[str, NoticeSeverity]:
    """The human one-liner for an ``ask_response`` row: ``(text, kind)``.

    ONE implementation, two surfaces (the TUI fold and the phone fold), for the
    reason every other row in this module is shared: a decision one host makes
    and the other misses is how a row renders on one surface and nowhere on the
    other (docs/design/history-fold-convergence.md §3).

    The three statuses are deliberately distinguishable to a HUMAN even though
    they ride one message type for the model — a person reading back a
    conversation needs to know whether their answer landed in time, landed late,
    or was a decline. ``kind`` is the ink: an answer is a receipt, a late answer
    is a receipt the user should notice, and a decline is a plain state.
    """
    details = details or {}
    status = str(details.get("status") or "answered")
    ask_id = str(details.get("ask_id") or "")
    if status == "declined":
        return (f"Ask {ask_id} declined — the agent was told", "info")
    if status == "late":
        return (f"Answered late — the agent was told (ask {ask_id})", "warning")
    return (f"Answered — delivering (ask {ask_id})", "info")


def ask_timeout_notice(details: Mapping[str, Any] | None) -> tuple[str, NoticeSeverity]:
    """The human one-liner for an ``ask_timeout`` row: ``(text, kind)``.

    It says the agent MOVED ON and that the ask is still answerable, because
    both halves are true and each is useless without the other: "timed out"
    alone reads as finished, and "you can still answer" alone hides that the
    agent stopped waiting. ``warning`` ink: this is the moment the user learns
    their silence had a cost.
    """
    details = details or {}
    ask_id = str(details.get("ask_id") or "")
    waited = gate_waited_text(details)
    return (
        f"Timed out after {waited} — the agent moved on; you can still answer (ask {ask_id})",
        "warning",
    )


def gate_timeout_notice(details: dict[str, Any]) -> str:
    """Say what expired, what it wanted, and that nobody chose it.

    The distinction this line has to carry is denial-by-expiry versus
    denial-by-decision: nobody said no. Naming the tool matters for the same
    reason the picker's parked row wants it — "a tool was denied" and "`bash rm
    -rf build/` was denied" are different amounts of help when you are
    reconstructing what happened overnight.

    An unattended gate timeout is the most expensive event in the detached
    feature (up to a day of held residency ends here), which is why it is
    worth one shared implementation rather than a per-surface paraphrase.

    It says NOTHING about who was attached (round 1, D7). It used to read "with
    nobody attached", which this PR's own change made reachable as a false
    statement: attachment-first parking holds a gate for
    ``runtime.unattended_gate_timeout`` hours when a pane IS attached, so a
    question the operator never got round to answering expires with a pane on it.
    The measurable fact is the wait, so the row reports that.
    """
    tool = str(details.get("tool") or "a tool").strip()
    description = str(details.get("description") or "").strip()
    waited_text = gate_waited_text(details)
    # An `ask` is a QUESTION, and an unanswered question was not "denied":
    # describing it in the approval gate's vocabulary told the user something
    # that did not happen (D12's copy note).
    kind = str(details.get("kind") or "approval").strip().lower()
    subject = f"{tool} · {description}" if description else tool
    if kind == "ask":
        return f"waited {waited_text} for an answer, then moved on — {subject}"
    return f"waited {waited_text} for approval, then expired unanswered — {subject}"


def wake_receipt_headline(text: str) -> str:
    """The human-readable headline of a wake delivery.

    A wake's persisted text is ``<envelope>\\n\\n<message>``, and the envelope
    is MODEL-FACING markup: ``(alarm) Scheduled wake w-9 (1, every 6h) —
    cancel with wake({op:"cancel",id:"w-9"})``. The cancel how-to is an
    instruction for the model, and the ``(alarm)``/``Scheduled wake`` prefixes
    restate what the row's own affordance already says — so what the user
    wants from this line is WHICH wake fired, not how to stop it.

    Extracted from ``WakeBlock._summary`` (which owned the only copy) because
    the phone renders the same receipt: while this logic lived in a TUI widget
    the phone had no way to reach it and showed the raw envelope verbatim —
    model-facing markup on a human surface, the same class of defect as the
    leaked ``<parent-message>`` rows this module exists to close.

    Catch-up deliveries are deliberately NOT handled here: the phone fold
    skips them entirely (they are user-attributed), so the folding summary
    stays in the TUI widget where it has the block's ``catchup`` flag.
    """
    head, _, _ = text.partition("\n\n")
    head = " ".join(head.split())  # collapse any envelope whitespace
    head = head.split(" — cancel with wake(", 1)[0]
    # The scratchpad clause needs its OWN strip rather than riding the cancel
    # hint's, and the difference is why this line exists: a FINAL delivery carries
    # no cancel hint at all, so there the clause survived the split above and
    # landed in the headline — model-facing markup on a human surface, the defect
    # this function exists to prevent. Imported lazily to keep this module
    # host-free (see the module docstring), and imported rather than re-spelled so
    # the stripper and the formatter cannot drift apart.
    from local_operator.harness.wake import WAKE_SCRATCH_CLAUSE

    head = head.split(WAKE_SCRATCH_CLAUSE, 1)[0]
    # Strip EVERY leading marker, not one. A single strip leaves a doubled
    # prefix ("(alarm) (alarm) …") leaking model-facing markup onto a human
    # surface — the exact defect this function exists to prevent, surviving
    # in the function that prevents it. No producer emits a doubled prefix
    # today, so this is closing the shape rather than a live bug; a partial
    # strip is the same "handles the case it happens to have seen" reasoning
    # that produced the incomplete chrome-prompt copy in this module's
    # docstring.
    alarm = "(alarm) "
    while head.startswith(alarm):
        head = head[len(alarm) :]
    # The surface's own wake affordance already says "wake"; repeating
    # "Scheduled wake" in the headline is a caption where a label belongs.
    prefix = "Scheduled wake "
    if head.startswith(prefix):
        head = head[len(prefix) :]
    # Engine-armed rows carry ids a person never chose (`aida-greeting`), and a
    # raw schedule id where a greeting goes reads as instrumentation on the
    # first frame of a fresh install's first conversation (design review round
    # 1, D2). The lookup is SHARED with the catch-up composition rather than
    # re-spelled here, so both receipt shapes name a wake the same way.
    ident, sep, rest = head.partition(" ")
    if ident:
        label = wake_display_name(ident)
        if label != ident:
            head = f"{label}{sep}{rest}"
    return head.strip()


#: The model-facing envelope prefix ``monitors.delivery`` opens a delta with.
#: BOTH receipt surfaces strip it — the collapsed line and the expanded body
#: (design review round 1, D2) — so it lives here once rather than in the
#: widget.
MONITOR_ENVELOPE_PREFIX = "(monitor) "

#: How the monitor formatter opens its cancel instruction (a whole LINE,
#: capital C — ``monitors/delivery.py``).
_MONITOR_CANCEL_LINE_PREFIX = "cancel with monitor("


#: The human's half of a lifecycle notice's remedy (UX review round 1, U1).
#:
#: ``monitors.delivery`` writes the notice for the MODEL: its "what next" is
#: ``monitor({op:"create",…})``, an agent tool call a person cannot type. The
#: card is the surface a person reads, so the same remedy is stated there in
#: their words, and "what next" was in the model's. Two routes, both real: the
#: CLI's own external-writer path (``lop monitor cancel``) and asking the agent
#: in the conversation the monitor belongs to.
MONITOR_HUMAN_REMEDY = (
    "For you: `lop monitor cancel <session> <id>` stops it (`lop monitor status` "
    "lists both), or ask the agent in that conversation to re-arm the same call."
)


def monitor_receipt_body(text: str) -> str:
    """A monitor delivery's text minus the model-facing envelope prefix.

    The expansion's half of the receipt polish, symmetric with
    :func:`monitor_receipt_headline` — the collapse has always stripped
    ``(monitor) ``; the expansion kept it until design review round 1 (D2),
    which is the inconsistency this closes. Nothing else is rewritten: the
    expansion is what the model was handed (design §12), cancel hint and all.

    A BROKEN WATCH (``disabled`` / ``stalled``) gains
    :data:`MONITOR_HUMAN_REMEDY` below its own text (UX review round 1, U1).
    Added HERE rather than in the wire text because the card body is the surface
    a person reads while the model-facing text must keep the shape the model was
    handed.

    REACH: this function's one production caller is the TUI's receipt builder
    (``tui/widgets/transcript.MonitorDeltaBlock``), so the sentence reaches the
    expanded card there. The phone does NOT show it today: ``mobile/projection``
    has custom-message branches for hub/peer/wake/ask/gate/compaction/job and
    none for ``monitor_prompt``, so a monitor notice falls to the generic notice
    row carrying the wire text. Wiring that fold is deferred (QA round 3, Q2);
    the claim is narrowed here rather than the mobile path widened in a
    remediation commit.

    ``restored`` is EXEMPT (design round 3, D14 / UX round 1, U9): a recovery
    notice reports good news, and a "stop it" remedy on it offers an action the
    reader did not ask for.
    """
    body = (
        text[len(MONITOR_ENVELOPE_PREFIX) :] if text.startswith(MONITOR_ENVELOPE_PREFIX) else text
    )
    if monitor_notice_kind(text) in ("disabled", "stalled"):
        return f"{body}\n{MONITOR_HUMAN_REMEDY}"
    return body


#: The phrases ``monitors.delivery`` opens each lifecycle notice with, ON ITS
#: FIRST LINE. Read from the text rather than from a field because both paths
#: that build a receipt block — the live ``MonitorDeltaEvent`` and the replayed
#: ``monitor_prompt`` row — hand the widget the formatted text and nothing else,
#: so a kind carried on the event would be lost on one of them and the two
#: would render differently for the same notice.
_NOTICE_MARKERS: tuple[tuple[str, str], ...] = (
    (" was DISABLED ", "disabled"),
    (" could not run its check ", "stalled"),
    (" is running again ", "restored"),
)


def monitor_notice_kind(text: str) -> str | None:
    """The lifecycle kind of a monitor delivery, or ``None`` for a delta.

    ONE definition for every reader (the receipt card's ink and its bounded
    headline today): a monitor delta's first line is its envelope
    (``… : N changes at …``), which carries none of these phrases, so a delta
    can never be mistaken for a notice.
    """
    if not text.startswith(MONITOR_ENVELOPE_PREFIX):
        return None
    first = text.split("\n", 1)[0]
    for marker, kind in _NOTICE_MARKERS:
        if marker in first:
            return kind
    return None


def _strip_source_notes(line: str) -> str:
    """A notice's first line without its ``(via <tool>)`` notes.

    Every notice names its source right after the clock — ``was DISABLED at
    10:46 (via mcp__datadog_search_datadog_hosts)`` — which is where the
    expansion should read it, but not what a one-line collapsed card has room
    for: 36 cells of MCP tool name is part of what pushed the consequence out of
    the row (design round 2, D13; UX round 1, U5). The note is dropped WHOLE,
    parentheses and the spacing they owned included, so the clause around it
    keeps its own punctuation.
    """
    out = line
    while True:
        start = out.find("(via ")
        if start == -1:
            return out
        end = out.find(")", start)
        if end == -1:
            return out
        out = " ".join((out[:start].rstrip() + out[end + 1 :]).split())


def monitor_receipt_headline(text: str) -> str:
    """The human-readable headline of a monitor delivery.

    The monitor twin of :func:`wake_receipt_headline`. A delta's persisted
    text starts ``(monitor) '<name>' m1: <n> change(s) at <clock> — check
    <k> …``, and the ``(monitor)`` prefix is model-facing markup the row's own
    affordance already says — so what the reader wants from the collapsed line
    is WHICH monitor fired and how much moved.

    The cancel instruction is dropped BY LINE as well as by inline clause.
    The formatter emits it as its own capital-C line, and the collapsed
    paragraph is every line up to the blank one — so at fold widths wide
    enough to fit the whole envelope the sentence reached the user (round 1
    review, F1, reproduced at width 160). The inline-clause strip stays as
    shape-closing for a producer that ever moves it onto the identity line.
    """
    if monitor_notice_kind(text) is not None:
        # A NOTICE's headline is its FIRST LINE, and only that line: the kinds
        # are written one sentence per line with no blank separator, so the
        # paragraph rule below would collapse the whole multi-sentence block
        # into a row that cannot show it — the collapsed card became the entire
        # notice, 529 characters of it, with the news cut off (design review
        # round 1, D4). The source note is dropped WHEREVER it sits in that
        # line — since D13 it rides behind the clock, not at the end — because
        # it would spend the row's first cells on the tool name; the expansion
        # keeps it.
        head = _strip_source_notes(text.split("\n", 1)[0])
        if head.startswith(MONITOR_ENVELOPE_PREFIX):
            head = head[len(MONITOR_ENVELOPE_PREFIX) :]
        return head.strip()

    head, _, _ = text.partition("\n\n")
    lines = [
        line
        for line in head.splitlines()
        if not line.strip().lower().startswith(_MONITOR_CANCEL_LINE_PREFIX)
    ]
    head = " ".join(" ".join(lines).split())  # collapse any envelope whitespace
    head = head.split(" — cancel with monitor(", 1)[0]
    while head.startswith(MONITOR_ENVELOPE_PREFIX):
        head = head[len(MONITOR_ENVELOPE_PREFIX) :]
    return head.strip()


def wake_display_name(wake_id: str) -> str:
    """A wake row's human name: its engine label, or the id itself.

    ONE lookup for every human surface that names a wake row — the
    single-delivery headline above and the catch-up composition in
    ``tui/widgets/transcript.WakeBlock._summary``, which builds its own
    headline from the per-schedule bullet ids and so never went through the
    fold: a fresh install's first receipt read ``catch-up — 1 missed wake
    (aida-greeting)`` while the live seat beside it already said ``Aida's
    introduction`` (UX review round 2, U5 / QA Q3). The labels live WITH the
    ids in ``aida/proactive.py`` and are read lazily — the same reason the
    scratch clause above is imported rather than re-spelled: ONE copy, and
    this module stays host-free. Every user-created wake answers with its own
    id.
    """
    try:
        from local_operator.aida import proactive as aida_proactive

        return aida_proactive.wake_display_label(wake_id)
    except Exception:  # noqa: BLE001 — a label is never worth a receipt
        return wake_id


#: Painted under a child's report that was HELD when it arrived — the delivery
#: turn it would have opened either fell on the leaving latch or was dropped
#: before it could run — so the row is durable and no turn ran for it at the
#: time. The report above it is the child's own words (the same text a delivered
#: row carries); this line is what says how it arrived, because without it a held
#: row and a delivered one read identically (UX round 1, U6).
#:
#: A FACT ABOUT THE ARRIVAL, NEVER A PRESENT-TENSE CLAIM ABOUT NOW (design review
#: round 2, D10; UX round 2, U7). The first wording said "held for your next turn …
#: no turn has read it yet", which the durable row cannot ever retract: it is
#: written once and painted on EVERY later replay, so after the successor's turn
#: had read and answered the reports, a resumed session still told the operator a
#: delegated report was owed — a permanent false alarm on the primary surface, in
#: exactly the family this change exists to remove. "no turn ran for it at that
#: point" stays true forever, and still separates the two rows.
#:
#: The bracketed lead-in marks where the harness's own words begin (design review
#: round 2, D11): the row is one block on one ink carrying the child's report and
#: then this sentence, and the sibling rows in the same frame state their
#: provenance with a ``[session …]`` head. It says **warning**, not "note" (design
#: round 3, D14): the row is painted on the warning tier with the ``!`` glyph, the
#: sibling directly above it in the same frame is ``[session warning]`` on the same
#: ink, and "note" is this product's word for the muted ``·`` tier — so a
#: tier-free word would have contradicted the row's own paint. It is NOT listed in
#: ``_HARNESS_NOTICE_HEADS`` and needs no entry: that list exists for texts a
#: persisted row can prove provenance by (the pre-stamp era), while this sentence
#: is composed at fold time and never persisted.
#: CAUSE-NEUTRAL AFTER THE DROP ARM JOINED (review round 3, R3-1). The sentence
#: used to say "this report reached the session while its runtime was leaving",
#: which the drop arm makes false: that arm is only reachable when the runtime is
#: NOT leaving (with the latch armed, ``_deliver_job_results`` takes the hold arm
#: and opens no turn, so the drop gate fires only for the abort-stopped case),
#: and reading a cause that never happened is worse than reading no cause. Both
#: arriving causes — the leaving latch and the dropped turn — share the one fact
#: that is always true, and it is what a reader needs: the report was held, and
#: no turn ran for it then.
HELD_DELIVERY_NOTICE = "[session warning] held when it arrived — no turn ran for it at that point"


def held_delivery_notice(
    details: dict[str, Any], *, report_budget: int | None = None
) -> tuple[str, NoticeSeverity] | None:
    """A child's report that was HELD for the next turn, and its ink — or None.

    ``None`` for every ``job_result`` row that was DELIVERED, and that is the
    whole point (UX round 1, U6): the two are the same message in every other
    respect, so the ONLY thing that can tell a reader "this one is still waiting
    for you" from "this one was already answered" is the flag the arm that held
    it writes (``Session._job_result_message(..., held=True)``; the leaving
    latch and the pre-abort drop arm both use it). A row without it is the
    ordinary delivery, whose own turn is what acknowledged it.

    The severity is derived HERE rather than at a renderer, by this module's rule
    for the fold decisions the two surfaces share (see
    ``docs/design/history-fold-convergence.md`` §3): the phone and the TUI must
    agree on the words and the tier, and a tier decided inside one renderer is a
    tier the other does not have. **Both folds call this**, which is what makes
    that rule true of this row rather than aspirational: Round 1 added the TUI
    branch alone and the rationale was measured false on the phone, where a held
    row and a delivered one still read identically (review round 2 MINOR-B,
    design D9, UX U8). A single caller would be a single opinion wearing a shared
    home's name.

    ``warning``, because the row is a state the operator has to know about and
    cannot otherwise see: nothing ran, nothing acknowledged it, and the only
    other trace is the model's own next turn — which has not happened yet. That
    is the same tier the MCP-unavailable row takes for the same reason, and one
    above the ``note`` tier of a receipt that answers something the user just
    did.
    """
    if not details.get("held"):
        return None
    report = str(details.get("text") or "").strip()
    if report_budget is None:
        # No budget: the caller bounds its own surface, so the sentence simply
        # follows the report.
        return (f"{report}\n{HELD_DELIVERY_NOTICE}" if report else HELD_DELIVERY_NOTICE), "warning"
    # A BUDGETED CALLER TRUNCATES THE REPORT, NEVER THE MARKER (design review
    # round 3, D15 = reviewer MAJOR-1 = UX U8-R). The phone composes the row and
    # then caps the whole string, so with the marker appended last it was the part
    # that got cut: intact to ~263 characters of report, cut mid-sentence from
    # ~264, gone entirely beyond ~363, and at 400+ the held entry and its delivered
    # twin were BYTE-IDENTICAL (only the severity differed). The job summary cap is
    # 2000 characters, so the realistic case was the losing one. Reserving the
    # marker's room first makes the sentence survive at every length, which is the
    # only allocation that keeps the row distinguishable — the report is already
    # truncated for the model at 2000, and this row's whole purpose is the marker.
    room = max(0, report_budget - len(HELD_DELIVERY_NOTICE) - 1)
    body = report[:room] if room else ""
    return (f"{body}\n{HELD_DELIVERY_NOTICE}" if body else HELD_DELIVERY_NOTICE), "warning"


def compaction_refused_notice(details: dict[str, Any]) -> tuple[str, NoticeSeverity]:
    """A compaction that did NOT run, and the ink it deserves.

    This row exists to CORRECT the optimistic "compacting context…" receipt
    the routed command already showed (round 5, U17). ``warning`` ink because
    the context the user asked to reclaim is still there; ``error`` when the
    attempt actually FAILED rather than being declined, because those are
    different things to the person deciding what to do next — a refusal is
    the system saying "not worth it", a failure is the system saying "I
    could not".

    The severity is derived here rather than at each call site so the phone
    cannot flatten a failure into the same ink as a decline.
    """
    text = str(details.get("detail") or "compaction did not run").strip()
    severity: NoticeSeverity = "error" if text.startswith("compaction failed") else "warning"
    return text, severity


#: THE NEUTRAL CLOSURE'S sentence (v2, operator directive 2026-09-29, session
#: 23fc556c3799): a disposal caught a run that had spent no provider
#: round-trip, so it renders as a receipt on both surfaces rather than the
#: ``Stopped with an error`` the operator re-reported. Defined HERE — the module
#: that owns this row on both surfaces — and reused verbatim by the
#: notification vocabulary (``tui/notify.py`` BODY_CLOSED) so the banner cannot
#: drift from the row.
CLOSED_NOTICE_TEXT = "Completed — runtime retired/disposed"

#: The row a turn cut by a build drain paints — TRUTHFUL but DISTINGUISHED
#: FROM A FAILURE (retire-for-build arm, 2026-09-29; seed 7e797aaaf6e7): the
#: turn really was cut, but the update was routine and the operator asked for
#: these transitions to stop reading as errors. One sentence shared by the TUI
#: poller, the phone projection and the desktop notice, so no two surfaces can
#: disagree about the same record.
#:
#: THE KEPT-OUTPUT CLAUSE (design round 2, D1) is the one fact a user who lost
#: work needs — the partial work survives — which the live cut sentence carries
#: ("the transcript holds what it wrote before that") and the terse receipt had
#: dropped. It rides byte-identical in the desktop's ``RETIRED_OUTCOME_TEXT``.
RETIRED_NOTICE_TEXT = (
    "Retired for an update — a turn was in flight and was cut; its earlier output is kept"
)


def completion_notice(kind: str, reason: str = "") -> tuple[str, NoticeSeverity]:
    """A RETURNED-TO turn's outcome row: its sentence and the ink it deserves.

    The row a session paints when you come back to it, as opposed to the one it
    paints while you are watching it die. Both surfaces have one
    (``tui/app.py``'s attention poller and the mobile daemon's projection
    frame), and they must agree on the words AND on the tier — the daemon's
    frame is a dict that a phone renders from, so a tier decided in a renderer
    is a tier the other renderer does not have.

    THE TIER IS ``error`` FOR A FAILURE, and this is the whole point of the
    helper. The poller's error branch passed no kind at all, so a cut-off — an
    alarm while you watch it die — came back as a dim ``·`` whisper, in the ink
    of the routine ``Interrupted`` receipt beside it (design review round 1,
    D1; UX U4). ``interrupted`` stays ``info``: that row is a receipt for the
    user's own act, and the live surface's louder ``warning`` for it is a
    turn-scoped statement this replay is not making.

    ``reason`` is optional because a pre-taxonomy record carries none, and an
    empty one must read exactly as it always has rather than leaving a dangling
    em-dash. The INTERRUPTED arm reads it too, and only for the attribution an
    escalated stop carries (design round 1, D1): a rung-3 kill and a rung-1
    request used to render identically because this row was kind-gated and threw
    the reason away, so the one surface the phone and the TUI poller share said
    nothing about who killed what.
    """
    if kind == "closed":
        # A RECEIPT, not a failure: the row STATES the closure and keeps the
        # info tier, so a disposal no longer paints danger ink over a turn
        # whose output had already been delivered (v2 directive). The reason
        # is ignored on purpose — the closure has no sentence to explain.
        return CLOSED_NOTICE_TEXT, "info"
    if kind == "retired":
        # WARNING, NEVER DANGER (architect addendum, 2026-09-29): a latched
        # build drain bounded by a signal sweep cut a live turn, and the row
        # must say so in warning ink — the update was routine; the lost work
        # is real but it is not a failure. The reason is ignored on purpose:
        # the cause token survives in the store for readers that want it, and
        # this row's one-line budget goes to the fact.
        return RETIRED_NOTICE_TEXT, "warning"
    if kind == "error":
        text = f"Stopped with an error — {reason}" if reason else "Stopped with an error"
        return text, "error"
    if kind == "interrupted":
        # WHAT THE DELIBERATE ROW WAS MISSING (design round 1, D1). ``reason``
        # reaches this helper as one composed sentence, and for a stop the
        # operator's own case that sentence is "the session was stopped by the
        # user" plus the attribution — the bit this row dropped, which made a
        # rung-3 kill and a rung-1 request render the same 11 cells. Only the
        # ESCALATED rung appends it (see
        # ``incidents.stop_rung_phrase``): a plain request is what the word
        # ``Interrupted`` already means, and appending its own sentence would
        # say "Interrupted — the session was stopped by the user".
        from local_operator.incidents import stop_rung_phrase

        phrase = stop_rung_phrase(reason or "")
        return (f"Interrupted — {phrase}" if phrase else "Interrupted"), "info"
    return "Interrupted", "info"


def assistant_stop_notice(
    *,
    text: str,
    has_tool_calls: bool,
    stop_reason: str | None,
    provider_payload: dict[str, Any] | None,
    cut_tool_call: bool | None = None,
) -> tuple[str, NoticeSeverity] | None:
    """The notice an assistant turn's ``stop_reason`` demands, or ``None``.

    Four turns end in a way the prose alone does not explain, and a surface
    that omits the notice tells the user something false:

    - **refusal** — the provider cut the answer off. Fires EVEN WHEN the
      model streamed prose first (Gemini safety stops often cut a partial
      answer): the prose alone reads as a complete, oddly short reply, and
      the user re-reading the session needs to know why it ends there.
    - **length** — the provider cut the answer off at the OUTPUT limit. Fires
      for the same reason as a refusal and more urgently: a reply stopped by
      the generation bound is the one case where the visible text is not merely
      short but INCOMPLETE BY CONSTRUCTION, and it used to reach no surface at
      all. The live loop's own notice covered only the silent case (a turn that
      thought away its whole budget and produced nothing), so a partial answer
      replayed as a whole one on every surface — which is the failure a lower
      generation bound makes reachable more often, not less (review round 1,
      B1; QA round 1, Q1 measured it on a live provider).
    - **error** with nothing produced — the turn FAILED. Without the notice
      the user sees their prompt followed by silence and reads it as the
      agent having ignored them.
    - **aborted** with nothing produced — the turn was interrupted.

    The error/aborted arms are guarded on there being neither prose nor a
    call because a turn that produced either already shows the user what
    happened; the notice exists for the turn that shows nothing at all. The
    length arm is deliberately NOT guarded that way — it is the one whose
    prose misleads most, and the empty variant of it still needs a line, since
    "spent the whole budget and said nothing" has no text to explain it. It is
    split THREE ways rather than two, because a cut tool call is neither: there
    is no answer on that turn to cut off, and saying there is puts a false row
    directly beneath the failed call card that already says the arguments were
    cut (design round 1, D3).

    The third arm — a call in flight, no prose — is ARM-AWARE, through
    ``cut_tool_call``, and has to be: the limit has two arms (a call whose
    arguments it CUT mid-dictation, and a call that arrived complete in a turn
    it ended before it could run), and the row a resume paints for the call
    already distinguishes them (``output_limit_call_receipt``). A single
    notice line covering both therefore names a cause the row right above it
    contradicts: measured on this branch, a length-stopped turn whose every
    call arrived complete read "tool call cut off at the output limit" under a
    card that said "turn cut off at the output limit before this call ran" —
    the false-cause class this PR exists to remove, surviving on the reader's
    two surfaces at once (design round 1, D1; QA Q-R2-1; review round 2,
    MINOR-2).

    ``True`` means "at least one call in this turn had its arguments cut",
    which is what that CALL's own row then says too; ``False`` and ``None``
    both take the arm-neutral line. ``None`` is "this host cannot tell" — a
    legacy transcript whose results predate ``OUTPUT_LIMIT_KEY``, or a call
    whose result never reached the transcript — and an arm-neutral sentence is
    the only honest one there, because a notice may not claim a cut nobody
    established. Every in-tree caller reads the arm off the turn's own results;
    the default exists so a host with no results to read still gets a true line
    rather than a false one.

    The notice states the TURN and the row states the CALL. That division is
    what keeps the sentence the operator reads painted once (design round 1,
    D2): the per-call receipt is owned by the call's own row
    (``output_limit_call_receipt``) and nothing here repeats it.

    Returning ``None`` means "this turn needs no notice", which is the
    ordinary case. Every surface must call this — the phone had no
    ``stop_reason`` branch whatsoever, so refusals, failed turns and
    interrupted turns were invisible on it.

    ``text`` is stripped HERE rather than trusted from the caller. The TUI
    strips before calling and the phone fold does not, so a whitespace-only
    assistant turn with ``stop_reason="error"`` produced "turn failed" on the
    TUI and SILENCE on the phone — D3 reopening inside the module built to
    close it. The emptiness test below is the whole decision for the
    error/aborted arms, so whose definition of "empty" wins cannot be a
    per-host choice.

    The same strip decides the length arm's silent variant, which is why the
    live loop's ``length`` branch strips too: a turn whose whole answer was
    ``"   "`` is silent to the reader, and a bare truthiness test on the
    loop's side had the live notice announce a CUT ANSWER over a fold that
    said no answer existed at all — the two-voices failure this module exists
    to prevent, on the one event class it cannot see both halves of (review
    R2-n4).
    """
    text = text.strip()
    if stop_reason == "refusal":
        payload = provider_payload or {}
        # The fallback keeps the marker grammar (D3): every other refusal
        # line ends in a parenthetical, and a user who has learned that shape
        # would read its absence as meaningful.
        refusal = str(payload.get("refusal") or "") or (
            "model refused the request (no details recorded)"
        )
        return refusal, "error"
    if stop_reason == "length":
        # Tier `warning`, not `error`: the turn produced what it produced and
        # stopped where the limit sits, which is a statement about the reply's
        # completeness rather than a diagnosis of anything failing — the same
        # tier the live loop's own truncation notices take.
        #
        # The three arms are the three shapes the loop itself distinguishes
        # (prose, a call in flight, neither), and this is the phone's only view
        # of a cut turn on a late-attaching client, so a fold that collapses
        # them cannot be fixed downstream.
        if text:
            return "answer cut off at the output limit", "warning"
        if has_tool_calls:
            # The arm comes from the caller, which read it off the turn's own
            # results; see the docstring. ``mid tool call`` is the cut claim,
            # and it is made only where a call really was cut.
            return (_CUT_CALL_NOTICE if cut_tool_call else _LIMIT_ENDED_TURN_NOTICE), "warning"
        return "no answer: the model spent its whole output budget", "warning"
    if not text and not has_tool_calls and stop_reason in ("error", "aborted"):
        return ("turn failed" if stop_reason == "error" else "interrupted"), "error"
    return None


#: The receipt for a call the OUTPUT LIMIT cut mid-arguments. OWNED by the
#: call's own row (``output_limit_call_receipt`` below): the card body on the
#: TUI, the row's error line on the phone. It is deliberately NOT the turn
#: notice's line too — see ``_CUT_CALL_NOTICE`` below for why one event painting
#: one sentence twice (the notice sits two rows under the card) is the thing to
#: avoid (design round 1, D2).
_CUT_CALL_RECEIPT = "tool call cut off at the output limit (nothing ran)"

#: The receipt for the OTHER limit arm: a call whose arguments arrived COMPLETE
#: in a turn the limit ended before it could run. Deliberately not the line
#: above — nothing about this call was cut, so reusing it would put a cause the
#: loop has not established on the operator's row (and, before this arm had its
#: own model-facing text, on the model's too: review round 1, F1 == QA Q1).
#:
#: Operator voice, not model voice: it is a RECEIPT for someone reconstructing
#: what happened, so it says what did not happen and stops there. No imperative,
#: no "reply with the call", no instruction the loop means for the model.
#:
#: ``cut off``, not ``ended``, is this family's verb for an involuntary end at
#: the generation bound (``answer cut off at the output limit`` above, the
#: stranded card's own "cut off", the live notice's "turn cut off"): the TURN is
#: the subject and the limit did cut it off, so the family word is the true one
#: and this arm's precision rides in the clause that follows it, not in the verb
#: (design round 1, D4).
_LIMIT_ENDED_TURN_RECEIPT = "turn cut off at the output limit before this call ran"

#: The TURN's own line for a length stop with a call in flight — the notice's,
#: never a row's. Two variants, because a turn can genuinely dictate one call to
#: completion and die writing a second, and the current one then says so:
#: ``mid tool call`` is the cut claim and is made only where a call really was
#: cut, while the sentence WITHOUT it is true of both arms, which is what makes
#: it the safe one for a mixed turn.
#:
#: These are the notice's own sentences rather than the receipts' (design round
#: 1, D1 and D2 are one decision): the receipt the operator reads for a call is
#: on the call's own row above, and a turn-level notice that repeated it verbatim
#: painted one sentence twice on one screen. What a notice adds that no row can
#: is the turn ITSELF — that the conversation stopped here — and that is what
#: these say.
_CUT_CALL_NOTICE = "turn cut off at the output limit mid tool call — nothing ran"
_LIMIT_ENDED_TURN_NOTICE = "turn cut off at the output limit — nothing ran"


def output_limit_cut_call(details: Mapping[str, Any] | None) -> bool:
    """Whether this result belongs to a call the OUTPUT LIMIT cut MID-ARGUMENTS.

    The arm question, asked where the other half of it lives: the marker
    (``harness.types.OUTPUT_LIMIT_KEY``) the loop stamps on the synthetic result
    it appends for a call the length arm kept from running.

    A host asks this about each call of a length-stopped turn to answer
    ``assistant_stop_notice``'s ``cut_tool_call`` — the folds have the turn's
    messages in hand and the notice has none, which is why the question is a
    function here rather than a branch inside the notice.

    ``False`` is the answer for everything that is not the cut arm, including a
    result carrying no marker at all (a transcript written before the marker
    existed) and a result this fold never saw. The caller is asking "may I claim
    the limit cut this?", and the honest answer for a result that does not say
    so is no: the notice's arm-neutral line is true either way, while a claim
    the record does not support is the defect this family exists to remove.
    """
    from local_operator.harness.types import OUTPUT_LIMIT_ARGUMENTS, OUTPUT_LIMIT_KEY

    if not isinstance(details, Mapping):
        return False
    return details.get(OUTPUT_LIMIT_KEY) == OUTPUT_LIMIT_ARGUMENTS


def turn_cut_tool_call(calls: Iterable[Any], results: Mapping[str, Any]) -> bool:
    """Whether any call of a turn was cut mid-arguments by the OUTPUT LIMIT.

    The question ``assistant_stop_notice(cut_tool_call=...)`` asks, answered
    from the turn's OWN results so neither host has to guess it from the turn's
    shape. ``results`` maps a call id to whatever settled it — a tool message in
    the phone's history fold, the session's result object in the TUI's replay
    fold — and both carry ``provider_payload``, which is where the marker rides.

    A call with NO entry in ``results`` is UNKNOWN, not cut: the fold may be
    looking at a page of history whose tail was capped, or at a conversation
    that predates the marker. Unknown takes the notice's arm-neutral line, which
    is true either way; see :func:`output_limit_cut_call` for why the answer
    errs toward "no" rather than toward the more dramatic claim.
    """
    for call in calls:
        settled = results.get(getattr(call, "id", "") or "")
        if settled is None:
            continue
        payload = getattr(settled, "provider_payload", None)
        details = payload.get("details") if isinstance(payload, Mapping) else None
        if output_limit_cut_call(details):
            return True
    return False


def output_limit_call_receipt(details: Mapping[str, Any] | None) -> str | None:
    """The row's own words for a call the OUTPUT LIMIT kept from running.

    ``None`` means "this result says nothing about an output limit", which is
    every result a tool actually produced — the ordinary case, and the one that
    must keep painting its own text.

    WHY A ROW NEEDS ITS OWN WORDS AT ALL. A call the limit stopped never runs,
    so its result is SYNTHETIC: the loop appends a placeholder the model is
    meant to read, and the message it persists is that same text. So the row a
    resume paints for the call is the text ADDRESSED TO THE MODEL — measured on
    this branch, the operator's screen carried "Reply with the call itself, not
    with an explanation of why it cannot be sent." (review round 1, F2). The
    marker (``harness.types.OUTPUT_LIMIT_KEY``) says which arm wrote it; the
    words come from here, where every other operator-facing receipt lives.

    Read the MARKER, never the model-facing wording: keying a row on a string
    the loop is free to reword is a row whose text changes for a copy edit, and
    it cannot tell the two arms apart at all.

    The arm is not decoration. A complete-arguments call in a limit-ended turn
    is a call that fits and only needs re-issuing, and telling its reader the
    call was cut is the false cause F1/Q1 measured; the two arms therefore
    carry two different lines rather than one hedged one.
    """
    # Imported HERE rather than at module scope: this module's contract is that
    # it imports no harness type at module scope (see the header), and the keys
    # live beside ``FAULT_KEY`` in ``harness.types`` because that is where the
    # harness declares the bookkeeping it writes into ``ToolResult.details``.
    # Cheap enough for the fold path: after the first call it is a `sys.modules`
    # lookup. The same shape as ``harness_chrome_prompts`` above.
    from local_operator.harness.types import (
        OUTPUT_LIMIT_ARGUMENTS,
        OUTPUT_LIMIT_KEY,
        OUTPUT_LIMIT_TURN,
    )

    if not isinstance(details, Mapping):
        return None
    arm = details.get(OUTPUT_LIMIT_KEY)
    if arm == OUTPUT_LIMIT_ARGUMENTS:
        return _CUT_CALL_RECEIPT
    if arm == OUTPUT_LIMIT_TURN:
        return _LIMIT_ENDED_TURN_RECEIPT
    return None


def _sessions_scalar(value: object) -> str:
    """One ``sessions`` argument as a one-line string ("" when unusable).

    Whitespace collapsed so a multi-line prompt cannot break a row, numbers
    stringified because ``pid``/``steps`` ride the row, and booleans refused
    deliberately — a bool is not an address, and ``True`` on the row would be
    a field name dressed as a value.
    """
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _sessions_address(args: Mapping[str, object]) -> str:
    """The one thing an addressed ``sessions`` op acts on, in the resolver's order.

    ``pid``, then the session id, then the substring — the same precedence the
    tool's own resolver uses, so the row cannot disagree with the call (or the
    approval prompt) about WHICH session it names.
    """
    pid = args.get("pid")
    if isinstance(pid, int) and not isinstance(pid, bool):
        return f"pid {pid}"
    session = _sessions_scalar(args.get("session"))
    if session:
        return f"session {session}"
    return _sessions_scalar(args.get("target")) or "?"


def _sessions_peek_window(args: Mapping[str, object]) -> str:
    """The peek row's window, in the op's own words (``last 12`` / ``first 20``).

    Empty when the call names no window: the tool's default applies, and the
    row must not claim a value nobody read.

    ``query`` is tested FIRST because it is the discriminator that survives
    beside ``steps``: the tool's own validation keeps ``steps`` as the SIZE of
    the match window (``query`` + ``steps=6`` reads six steps around the match,
    not the tail), so asking for ``steps`` first painted ``last 6`` — a read
    this call never makes — and dropped the search term (round-1 review, R1).
    The search rides out first and a call-sized window rides after it.
    """
    query = _sessions_scalar(args.get("query"))
    if query:
        steps = args.get("steps")
        if isinstance(steps, int) and not isinstance(steps, bool):
            return f"search {query} · {steps} around"
        return f"search {query}"
    steps = args.get("steps")
    if isinstance(steps, int) and not isinstance(steps, bool):
        return f"last {steps}"
    head = args.get("head")
    if isinstance(head, int) and not isinstance(head, bool):
        return f"first {head}"
    if args.get("digest") is True:
        return "digest"
    before = _sessions_scalar(args.get("before_id"))
    if before:
        return f"before {before}"
    around = _sessions_scalar(args.get("around_id"))
    return f"around {around}" if around else ""


def sessions_row_summary(args: Mapping[str, object]) -> str:
    """One ``sessions`` tool row's summary, for every host to call.

    WHY IT LIVES HERE rather than beside either renderer: the row is drawn on
    two surfaces (the TUI tool card's collapsed row and the phone's transcript
    row), the two are required to say the same thing about one call, and this
    module is where the shared row decisions live (see the header — "a decision
    made in one renderer is a decision the other will not make"). Pure by
    contract: a Mapping of the call's arguments in, one line out, no host
    imports.

    THE OP LEADS because both rows shed from the RIGHT, and the op is the
    discriminator: `stop` and `peek` on one session painted byte-identical
    rows under the generic first-scalars scan. ``spawn`` carries its VISIBILITY
    next — both values — because the invisible disposition is the incident this
    tool exists to prevent: the one field that must survive a narrow row is the
    one that says whether the operator will see the run.

    ``?`` stands in when an addressed op names no address: the row is painted
    before the call settles, and a blank slot reads as though the next field
    were the target (the send row's rule). ``workstream`` stands in when a
    spawn omits the flag because that is the tool schema's own default —
    spelled here rather than imported so this module keeps its no-imports
    contract, and pinned by tests on both hosts so a schema change cannot
    leave the row claiming a value no call produced. An unknown op returns its
    own word and nothing else, never a guess; "" when even that is unreadable,
    and the caller substitutes the tool name.
    """
    op = _sessions_scalar(args.get("op"))
    if op == "spawn":
        visibility = _sessions_scalar(args.get("visibility")) or "workstream"
        name = _sessions_scalar(args.get("name")) or _sessions_scalar(args.get("prompt"))
        parts = [op, visibility] + ([name] if name else [])
        return " · ".join(parts)
    if op in ("stop", "resume", "info", "peek"):
        parts = [op, _sessions_address(args)]
        if op == "peek":
            window = _sessions_peek_window(args)
            if window:
                parts.append(window)
        return " · ".join(parts)
    if op == "list":
        parts = [op]
        if args.get("include_stored") is True:
            parts.append("stored")
        query = _sessions_scalar(args.get("query"))
        if query:
            parts.append(query)
        return " · ".join(parts)
    return op
