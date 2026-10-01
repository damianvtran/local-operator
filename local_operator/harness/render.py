"""The transcript→LLM renderer, shared by every surface that drives the loop.

Why it lives here rather than in ``session/session.py``, where it was written:
the evaluation runner must not import session code. An episode is reproducible
from its pinned inputs, so nothing on its import path may reach the operator's
own configuration (``tests/unit/evaluation/runner/test_isolation.py``), which is
what keeps ``session.py`` off it. The renderer needs none of that — it is a pure
function of the transcript — so hoisting it lets an episode render through the
*same* function the TUI renders through, instead of growing a second
implementation that drifts from it.

The vocabulary this module matches on has ONE neutral home,
``harness/message_types.py``, and that is what makes the hoist reachable.
Seven markers used to be defined beside their owning subsystem, and the four of
those homes a runner may not import — ``local_operator.incidents`` (the four
``SESSION_*_MESSAGE_TYPE`` records), ``local_operator.session.peer``
(``PEER_MESSAGE_MESSAGE_TYPE``), ``local_operator.tools.builtin``
(``TODO_REMINDER_MESSAGE_TYPE``) and ``local_operator.harness.comms``
(``HUB_MESSAGE_TYPE``) — leaked between them **17** denied modules into this
module's import closure: 1, 2, 11 and 8 respectively, and the union is the 17
(``tests/unit/evaluation/runner/test_isolation.py`` is the denylist that
measures it). ``harness.comms`` is the home that reads as clean and is not — it
needs ``PEER_MESSAGE_MESSAGE_TYPE`` and ``TRANSCRIPT_FILENAME`` for its own
replay work, so it imports ``session.peer`` and ``session.transcript`` at module
level; moving the six obvious constants and leaving ``HUB_MESSAGE_TYPE`` there
would still have leaked those same 8 through this one import. Moving all seven
took the closure to zero, and that test now probes this module, so a fresh
import here cannot put the leak back unnoticed.
"""

from __future__ import annotations

from typing import TypeGuard

from local_operator.compaction.cutpoint import RENDERED_INJECTION_KEY

# Imported under the name the renderer body used at its old home, so the move
# stays byte-identical rather than renaming a call while relocating it.
from local_operator.compaction.marker import (
    render_compaction_marker as _render_compaction_marker,
)
from local_operator.harness.approval import GATE_TIMEOUT_CUSTOM_TYPE
from local_operator.harness.jobs import JOB_RESULT_MESSAGE_TYPE

# The vocabulary this renderer matches on, from its one neutral home rather
# than from the four subsystems that own each marker: three of those are barred
# for a runner and the fourth (``harness.comms``) drags the session package in
# behind them, which is precisely what made this module unreachable from an
# episode (see the module docstring). Import nothing heavy into that module.
from local_operator.harness.message_types import (
    ASK_RESPONSE_MESSAGE_TYPE as ASK_RESPONSE_CUSTOM_TYPE,
)
from local_operator.harness.message_types import (
    ASK_TIMEOUT_MESSAGE_TYPE as ASK_TIMEOUT_CUSTOM_TYPE,
)
from local_operator.harness.message_types import (
    HUB_MESSAGE_TYPE,
    PEER_MESSAGE_MESSAGE_TYPE,
    PROJECT_REMINDER_MESSAGE_TYPE,
    SESSION_CREDENTIAL_MESSAGE_TYPE,
    SESSION_INCIDENT_MESSAGE_TYPE,
    SESSION_MCP_RECOVERY_MESSAGE_TYPE,
    SESSION_MCP_UNAVAILABLE_MESSAGE_TYPE,
    SESSION_MODEL_SWITCH_MESSAGE_TYPE,
    SESSION_SEND_NOTICE_MESSAGE_TYPE,
    TODO_REMINDER_MESSAGE_TYPE,
)
from local_operator.harness.rows import gate_waited_text
from local_operator.harness.types import (
    AgentMessage,
    CustomMessage,
    Message,
    TextContent,
)
from local_operator.harness.wake import WAKE_PROMPT_MESSAGE_TYPE


def _injected_user_message(text: str, entry_id: str) -> Message:
    """A user-role message minted from a harness aside, stamped as such.

    The stamp is compaction's provenance signal. Once this function has run,
    an injected delivery and an operator prompt are both a plain
    ``Message(role="user")`` and no structural test can separate them — which
    is precisely how a preserved-turn block on a real session came to be 160
    injections against 11 genuine turns (see
    :data:`~local_operator.compaction.cutpoint.RENDERED_INJECTION_KEY`).

    It rides ``provider_payload``, which the wire builders never ship as
    content, so this is invisible to the model and to every provider.
    """
    message = Message(role="user", content=[TextContent(text=text)], id=entry_id)
    message.provider_payload = {RENDERED_INJECTION_KEY: True}
    return message


def _default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """Default transcript→LLM rendering.

    ``compaction_summary`` markers become a user message carrying the summary;
    a snapcompact archive in ``preserve_data`` is rendered back into
    text_head → imaged middle → text_tail blocks (base64 ``ImageContent``
    between ``TextContent`` edges). ``fork_boundary`` and ``wake_prompt``
    deliveries become user messages of their formatted text, and the newest of
    EACH reminder type — ``todo_reminder`` and ``project_reminder`` (each only
    its own newest; the two are independent claims) — becomes one too; other
    custom entries are dropped (bookkeeping never enters LLM context).
    ``provider_payload`` rides along untouched.

    ``gate_timed_out_unattended`` is rendered from its STRUCTURED payload
    rather than a ``text`` field, because the same fact is phrased differently
    for the three audiences that need it (the model here, the transcript
    notice, the picker's parked row). It must never be dropped: an expiry that
    reads as a plain denial makes the next turn re-plan around a decision
    nobody made.

    Every user-role message minted HERE from a ``CustomMessage`` is stamped
    with :data:`RENDERED_INJECTION_KEY` (see :func:`_injected_user_message`).
    That stamp is compaction's only reliable way to tell a harness injection
    from an operator prompt once both are plain user messages, which is what
    they both are the moment this function has run.
    """
    out: list[Message] = []
    # Only the NEWEST todo reminder and the NEWEST project reminder survive
    # the render — one scan per type, because the two are independent claims
    # about different stores and the latest of each is the one that can still
    # be true. An earlier one asserts state that has since changed, so
    # replaying it would hand the model a stale — and by then actively false —
    # claim and re-argue a nudge it has already answered. The pruning belongs
    # here because the renderer is a pure function of the whole list and
    # reminders are never persisted, so nothing downstream could do it. Older
    # ones simply fall through to the allow-list's drop.
    newest_reminder = -1
    newest_project_reminder = -1
    for index in range(len(messages) - 1, -1, -1):
        if newest_reminder < 0 and _is_todo_reminder(messages[index]):
            newest_reminder = index
        if newest_project_reminder < 0 and _is_project_reminder(messages[index]):
            newest_project_reminder = index
        if newest_reminder >= 0 and newest_project_reminder >= 0:
            break
    for index, message in enumerate(messages):
        if isinstance(message, Message):
            out.append(message)
        elif message.custom_type == "compaction_summary":
            # Pass the ORIGINAL entry id through the render: the transcript
            # persists custom entries with their CustomMessage.id, so a
            # compaction cut landing on a rendered marker can still locate
            # ``first_kept_entry_id`` on replay.
            out.append(_render_compaction_marker(message, entry_id=message.id))
        elif message.custom_type in (
            SESSION_INCIDENT_MESSAGE_TYPE,
            SESSION_MODEL_SWITCH_MESSAGE_TYPE,
            SESSION_CREDENTIAL_MESSAGE_TYPE,
            SESSION_MCP_RECOVERY_MESSAGE_TYPE,
            SESSION_MCP_UNAVAILABLE_MESSAGE_TYPE,
            SESSION_SEND_NOTICE_MESSAGE_TYPE,
            "session_state",
        ):
            # An incident rides the sender's preformatted text (the classifier
            # already wrote category + suggested action), exactly like a wake
            # delivery: it must reach the model as a user turn or the session
            # stays blind to why its last run died. A model-switch record uses
            # the same path so the model becomes aware it is now answering as a
            # different model (a deliberate switch or a failover fallback),
            # rather than only seeing a changed static "Model:" system line.
            # A credential record rides the same path so a mid-session
            # ``/credential`` is ANNOUNCED to the model rather than only
            # changing the prompt tail, which the model has no reason to
            # re-read (the failure behind session 835fbcafdc27).
            # An MCP record rides it as a pair: the FAILURE reaches the model
            # as a ``session_mcp_unavailable`` WARNING (a preformatted row, so
            # it never touches the classifier), and the recovery that
            # supersedes it has to arrive on the same surface or the model
            # keeps believing the older, more emphatic claim that its tools are
            # gone.
            #
            # A SEND NOTICE rides it for the opposite reason to the two
            # deliberate absences below: the reader who must not re-send a
            # duplicate is the MODEL, which is the one holding the ``send`` tool.
            # The row is a plain sentence naming the target, the message id and
            # what to do (peek, do not resend), and it is the durable half of a
            # notice whose live half is the tool result of a turn the model may
            # have long since compacted away.
            #
            # ``SESSION_CREDENTIAL_REDACTION_MESSAGE_TYPE`` IS DELIBERATELY
            # ABSENT FROM THIS TUPLE, AND THAT ABSENCE IS THE FEATURE. Do not
            # "fix" it by adding the type. That record reports a
            # credential-shape guard hit to the OPERATOR, who is the only reader
            # who can act on it; it rode ``session_incident`` until
            # 2026-09-24 and put a detection notice in front of the model —
            # measured at 1,493 unnamed notices across 1,080 sessions on this
            # machine, plus the named ones. The model never held the value (the
            # guard masked it out of the text the model got), so the notice told
            # it nothing it could act on, and the guard's own false positives
            # meant agents ended turns investigating leaks that had not
            # happened. Unlisted custom types are dropped here as bookkeeping,
            # which is exactly the intended treatment; the transcript row and
            # the live operator receipt were asserted by
            # ``tests/unit/secrets/test_credential_shapes.py`` at the time.
            #
            # THE NOTICE ITSELF IS GONE NOW (2026-09-27) and THIS EXCLUSION STAYS,
            # deliberately: no in-tree path writes a new row of this type, so on
            # the live path the exclusion is belt-and-braces — and it is
            # load-bearing on the REPLAY path, where a transcript written before
            # that date rehydrates a stored row through ``build_llm_history`` and
            # this is what keeps it out of the model's context. A guard whose
            # symptom disappears is exactly the one not to delete while stored
            # rows still exist; it becomes dead only once no transcript anywhere
            # carries the type, which is a migration decision and not this one.
            #
            # ``SESSION_BINDING_NOTICE_MESSAGE_TYPE`` IS THE SECOND SUCH
            # DELIBERATE ABSENCE (mesh credential binding, slice B): the
            # account-change notice is OPERATOR-facing — it exists so a session
            # does not silently change which account serves it — and its
            # sentence may carry the account label, which §2.1 keeps off
            # anything the model reads. Same treatment, same reason: unlisted
            # means bookkeeping-dropped here, and the transcript row, the TUI
            # fold and replay carry it (see `harness/message_types.py`).
            out.append(_injected_user_message(message.details.get("text", ""), message.id))
        elif message.custom_type == ASK_TIMEOUT_CUSTOM_TYPE:
            # A queued ask's DEADLINE (design docs/design/ask-nonblocking.md
            # §2.5). Unlike the approval row below this one DOES carry its own
            # preformatted text: the notice quotes the questions (clipped) or, for
            # a secret ask, names the key and never the prompt — and the model and
            # the card must see the same words, so ``asks/render.py`` writes them
            # once and this branch carries them through. It reports the wait and
            # says the ask stays answerable; it never says the user denied
            # anything, because nobody did.
            out.append(_injected_user_message(message.details.get("text", ""), message.id))
        elif message.custom_type == ASK_RESPONSE_CUSTOM_TYPE:
            # A queued ask's ANSWER, late answer or DECLINE — one type, one row,
            # because all three settle the same ask and a second registration is
            # a row that can silently drop (design §2.3). The text again comes
            # from ``asks/render.py``: for an answer it IS ``_ask_report``, the
            # same report the blocking tool returned, so the model cannot tell
            # the two paths apart in what it is told.
            out.append(_injected_user_message(message.details.get("text", ""), message.id))
        elif message.custom_type == GATE_TIMEOUT_CUSTOM_TYPE:
            # An unattended gate that expired is NOT a user decision, and the
            # difference is the whole reason the row exists: without it the
            # next turn reads a plain denial and re-plans around a choice
            # nobody made. Rendered here rather than carrying a `text` field
            # like the branches below because the payload is structured (tool,
            # description, waited_s) — the picker and the transcript notice
            # each phrase it for their own audience, and this is the model's.
            #
            # IT REPORTS THE WAIT AND NOTHING ABOUT WHO WAS ATTACHED (round 1,
            # D7). It used to read "nobody was attached to this session and it
            # expired", and this PR's own change makes that reachable as a FALSE
            # claim: attachment-first parking holds a gate for the configured
            # ``unattended_gate_timeout`` (24h by default) when a pane IS
            # attached, so a question the operator never got round to answering
            # expires with a pane on it — and the model is then told nobody was
            # there, which is the sentence class this whole change exists to
            # stop. What was measured is the wait; ``gate_waited_text`` is the
            # ONE formatter for it, shared with the human row so the two cannot
            # disagree about the number.
            details = message.details or {}
            tool = str(details.get("tool") or "a tool")
            description = str(details.get("description") or "").strip()
            subject = f"{tool} ({description})" if description else tool
            waited = gate_waited_text(details)
            # An `ask` is a QUESTION, and an unanswered question was not
            # "denied" — the approval gate's vocabulary describes a refusal
            # nobody issued, and a model told its question was denied re-plans
            # around that phantom decision. `tui/app.py`'s parked-gate summary
            # already branches here for the HUMAN (D12's copy note); this is
            # the same row rendered for the model, and until #868 made the ask
            # gate reachable it could only ever carry an approval.
            #
            # The ask arm ends the way ``ASK_UNANSWERED_TEXT`` does, on
            # purpose: an expiry and a user pressing `esc` are both "no answer
            # came back", so the two must leave the model in the same place
            # rather than one nudging it to decide and the other implying it
            # was refused.
            kind = str(details.get("kind") or "approval").strip().lower()
            if kind == "ask":
                # "never answered: it expired unanswered after 1d" said the same
                # thing twice, and "a choice by the user" named the person one way
                # in a row this PR had already moved to "the operator" everywhere
                # else (round 2, D8r).
                text = (
                    f"[system] The question for {subject} expired unanswered after "
                    f"{waited}. No decision was made — this was a timeout, not a "
                    "choice by the operator. Decide yourself (take your recommended "
                    "option where you gave one), then say in one line what you "
                    "assumed and carry on."
                )
            else:
                text = (
                    f"[system] The approval request for {subject} expired unanswered "
                    f"after {waited} and was denied automatically. This was a "
                    "timeout, not a decision by the operator."
                )
            out.append(_injected_user_message(text, message.id))
        elif message.custom_type in (
            "fork_boundary",
            WAKE_PROMPT_MESSAGE_TYPE,
            HUB_MESSAGE_TYPE,
            JOB_RESULT_MESSAGE_TYPE,
            PEER_MESSAGE_MESSAGE_TYPE,
        ):
            # A hub message renders exactly like a wake delivery: the sender
            # already formatted ``details["text"]``, and it must reach the
            # model as a user turn or the agent it was addressed to never
            # sees it. A peer message (`lop send` from another local session)
            # rides the same path: it MUST be listed here or the human sees the
            # cross-session transcript row but the model never does. Unlisted
            # custom types are dropped (bookkeeping), which is precisely the
            # trap a new aside type falls into.
            out.append(_injected_user_message(message.details.get("text", ""), message.id))
        elif message.custom_type == TODO_REMINDER_MESSAGE_TYPE and index == newest_reminder:
            # The continuation guardrail's nudge (``Session._todo_continuation``)
            # reaches the model as a user turn or it does nothing at all: this
            # allow-list is the trap a new aside type falls into, and a dropped
            # reminder would make the loop re-enter with nothing to react to.
            out.append(
                Message(
                    role="user",
                    content=[TextContent(text=message.details.get("text", ""))],
                    id=message.id,
                )
            )
        elif (
            message.custom_type == PROJECT_REMINDER_MESSAGE_TYPE
            and index == newest_project_reminder
        ):
            # The completion-time project check's nudge
            # (``Session._project_continuation``) rides the same path as the
            # todo reminder, and for the same reason: without this arm the
            # allow-list would drop it as bookkeeping and the loop would
            # re-enter with nothing to react to. It keeps its own newest
            # (above), so a todo nudge and a project nudge in one batch both
            # reach the model.
            out.append(
                Message(
                    role="user",
                    content=[TextContent(text=message.details.get("text", ""))],
                    id=message.id,
                )
            )
    return out


def _is_todo_reminder(message: AgentMessage) -> TypeGuard[CustomMessage]:
    """Is ``message`` a live continuation nudge (``_todo_continuation``)?

    One predicate for the three places that have to agree about it — the
    renderer's newest-only rule, the expiry scan
    (:meth:`Session._live_todo_reminders`) and the compaction render
    (:meth:`Session._render_for_compaction`). The ``isinstance`` half is
    load-bearing rather than defensive: a RENDERED reminder is a plain
    ``Message`` carrying the same text, and a predicate that matched that too
    would read compaction's own output back as a fresh nudge.
    """
    return isinstance(message, CustomMessage) and message.custom_type == TODO_REMINDER_MESSAGE_TYPE


def _is_project_reminder(message: AgentMessage) -> TypeGuard[CustomMessage]:
    """Is ``message`` a live project-progress nudge (``_project_continuation``)?

    The todo predicate's twin, for the same three agreements — the newest-only
    rule here, the expiry scan (:meth:`Session._live_project_reminders`) and
    the compaction render — and the same load-bearing ``isinstance`` half: a
    rendered reminder is a plain ``Message`` carrying the same text.
    """
    return (
        isinstance(message, CustomMessage) and message.custom_type == PROJECT_REMINDER_MESSAGE_TYPE
    )
