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

    This tuple is the EXACT-match half of the decision only. One chrome prompt
    has a family of shapes rather than a single string — the connectivity
    instruction, which ``harness.loop._continuation_instruction`` composes per
    cut (prose alone, an aborted tool call alone, or the two joined) — and those
    cannot be enumerated here because the tool-call half interpolates the
    aborted calls' names. :func:`is_harness_chrome` therefore adds the producer's
    own recogniser for that family; there is still exactly ONE decision, and it
    is this function plus that one recogniser, not a list per surface.
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

    Two legs, one decision. Exact membership covers the three prompts that are
    single fixed strings; the loop's
    :func:`~local_operator.harness.loop.is_connectivity_continuation_instruction`
    covers the shapes of the fourth, which the producer composes per cut and
    which therefore cannot live in a tuple. Equality alone used to be the whole
    test, and the composed "prose then an aborted tool call" instruction — the
    incident's own shape — was a member of neither, so a resumed session
    painted it (and the tool-call-only shape) as the operator's own words.
    """
    from local_operator.harness.loop import is_connectivity_continuation_instruction

    stripped = text.strip()
    return stripped in harness_chrome_prompts() or is_connectivity_continuation_instruction(
        stripped
    )


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
