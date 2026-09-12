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
import the session. The three chrome prompts live in modules that would be
the wrong import direction from here (``session.goal_loop``,
``session.session``), so they are gathered behind a function that imports
them lazily. That keeps ONE list rather than a copy per host — the partial
copy is precisely the drift signature the review named: the phone had
suppressed one of the three prompts and rendered the other two as the
user's own words.

Adding a new row decision belongs HERE, not in a host. A decision made in
one renderer is a decision the other will not make.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    "[mcp recovery] ",  # incidents.format_mcp_recovery_message
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

    The three, and why each is persisted:

    - ``LOOP_PROMPT`` — the goal loop's self-continuation.
    - ``_CONTINUATION_PROMPT`` — the session's auto-continuation after a
      turn hit its step budget.
    - ``CONNECTIVITY_CONTINUATION_PROMPT`` — records why ONE answer arrived
      in two pieces across a network interruption; the live run showed a
      notice instead.

    Imported lazily and gathered here rather than listed per host: this list
    having been copied INCOMPLETELY into the phone fold (it suppressed the
    third and rendered the first two) is the drift this function exists to
    make impossible. A fourth prompt is added in one place.
    """
    from local_operator.harness.loop import CONNECTIVITY_CONTINUATION_PROMPT
    from local_operator.session.goal_loop import LOOP_PROMPT
    from local_operator.session.session import _CONTINUATION_PROMPT

    return (LOOP_PROMPT, _CONTINUATION_PROMPT, CONNECTIVITY_CONTINUATION_PROMPT)


def is_harness_chrome(text: str) -> bool:
    """Whether this user-role text is harness chrome rather than the user's words.

    Normalises before comparing because THE HOSTS DO NOT AGREE on what they
    hand in: the TUI strips a message's text at the top of its replay loop,
    the phone fold passes ``message.text`` verbatim. Leaving the strip to the
    callers is the exact substrate this module exists to remove — one surface
    would suppress a chrome prompt that the other painted as the user's own
    words the moment a persisted prompt gained a trailing newline.
    """
    return text.strip() in harness_chrome_prompts()


def is_harness_injection(row: Any) -> bool:
    """Whether this row was minted BY the harness from a ``CustomMessage``.

    :func:`~local_operator.session.session._default_convert_to_llm` renders a
    harness aside — a model-switch notice, a session incident, a wake
    delivery, a gate timeout — into a plain ``Message(role="user")`` and
    stamps ``provider_payload["harness_injected"]`` on it, so the row is
    structurally indistinguishable from an operator prompt once it exists.
    The stamp is compaction's provenance signal, and it is also the only
    signal a fold has: a row carrying it was never typed by a person, so no
    human-facing surface may paint it as their words. The live path never
    paints one either (the failover moment has its own receipt), which makes
    dropping it live/replay parity rather than a second opinion — the same
    doctrine :func:`is_harness_chrome` follows for the three continuation
    prompts.

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


def is_harness_notice_row(row: Any) -> bool:
    """Whether this row is harness-authored and must not paint as the user's words.

    The decision every human-facing surface and every title/query/tail scan
    makes, in one place, and it holds in BOTH phases a display can serve: the
    context replay (where the row may be a stamped render or a copy a compaction
    marker carried) and the audit replay (where it is the stored row itself).

    Two shapes answer True, and they need different evidence:

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


def user_row_text(text: str) -> str:
    """What a surface paints for a ``role="user"`` message.

    Collapses a skill payload to the line the user actually typed and leaves
    every other message alone. Kept beside :func:`is_harness_chrome` because
    the two are the whole of the "what did the user really say" decision, and
    splitting them across hosts is how one surface got the skill rule and the
    other did not.

    Strips for the same reason :func:`is_harness_chrome` does: the two hosts
    normalise differently, so an envelope with surrounding whitespace would
    resolve to a typed line on the TUI and to the whole SKILL.md body on the
    phone. The fallback returns the stripped text so both surfaces paint one
    row, not one padded and one not.
    """
    stripped = text.strip()
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


def gate_timeout_notice(details: dict[str, Any]) -> str:
    """Say what expired, what it wanted, and that nobody chose it.

    The distinction this line has to carry is denial-by-expiry versus
    denial-by-decision: the user did not say no, they were not there. Naming
    the tool matters for the same reason the picker's parked row wants it —
    "a tool was denied" and "`bash rm -rf build/` was denied" are different
    amounts of help when you are reconstructing what happened overnight.

    An unattended gate timeout is the most expensive event in the detached
    feature (up to a day of held residency ends here), which is why it is
    worth one shared implementation rather than a per-surface paraphrase.
    """
    tool = str(details.get("tool") or "a tool").strip()
    description = str(details.get("description") or "").strip()
    waited = details.get("waited_s")
    # REPORT THE WAIT THAT HAPPENED. This used to floor at one hour
    # (`max(1, waited // 3600)`), so a 30-second expiry — a live, reachable
    # path when there is no registrant, or notifications are off and nothing
    # is watching — rendered as "waited 1h" (round 3, D12). This row exists
    # to preserve the difference between denied-by-decision and
    # denied-by-absence, and a fabricated duration undermines the one number
    # that has to be trustworthy. An absent or unreadable value says so
    # rather than rounding up to an hour.
    try:
        seconds = float(waited) if waited is not None else 0.0
    except (TypeError, ValueError):
        seconds = 0.0
    if seconds <= 0:
        waited_text = "a while"
    elif seconds < 60:
        waited_text = f"{int(seconds)}s"
    elif seconds < 3600:
        waited_text = f"{int(seconds // 60)}m"
    elif seconds < 86400:
        waited_text = f"{int(seconds // 3600)}h"
    else:
        waited_text = f"{int(seconds // 86400)}d"
    # An `ask` is a QUESTION, and an unanswered question was not "denied":
    # describing it in the approval gate's vocabulary told the user something
    # that did not happen (D12's copy note).
    kind = str(details.get("kind") or "approval").strip().lower()
    subject = f"{tool} · {description}" if description else tool
    if kind == "ask":
        return (
            f"waited {waited_text} for an answer with nobody attached, "
            f"then moved on — {subject}"
        )
    return f"waited {waited_text} for approval with nobody attached, then denied it — {subject}"


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
    return head.strip()


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


def assistant_stop_notice(
    *,
    text: str,
    has_tool_calls: bool,
    stop_reason: str | None,
    provider_payload: dict[str, Any] | None,
) -> tuple[str, NoticeSeverity] | None:
    """The notice an assistant turn's ``stop_reason`` demands, or ``None``.

    Three turns end in a way the prose alone does not explain, and a surface
    that omits the notice tells the user something false:

    - **refusal** — the provider cut the answer off. Fires EVEN WHEN the
      model streamed prose first (Gemini safety stops often cut a partial
      answer): the prose alone reads as a complete, oddly short reply, and
      the user re-reading the session needs to know why it ends there.
    - **error** with nothing produced — the turn FAILED. Without the notice
      the user sees their prompt followed by silence and reads it as the
      agent having ignored them.
    - **aborted** with nothing produced — the turn was interrupted.

    The error/aborted arms are guarded on there being neither prose nor a
    call because a turn that produced either already shows the user what
    happened; the notice exists for the turn that shows nothing at all.

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
    if not text and not has_tool_calls and stop_reason in ("error", "aborted"):
        return ("turn failed" if stop_reason == "error" else "interrupted"), "error"
    return None
