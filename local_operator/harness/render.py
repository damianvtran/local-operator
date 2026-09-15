"""The transcript→LLM renderer, shared by every surface that drives the loop.

Why it lives here rather than in ``session/session.py``, where it was written:
the evaluation runner must not import session code. An episode is reproducible
from its pinned inputs, so nothing on its import path may reach the operator's
own configuration (``tests/unit/evaluation/runner/test_isolation.py``), which is
what keeps ``session.py`` off it. The renderer needs none of that — it is a pure
function of the transcript — so hoisting it lets an episode render through the
*same* function the TUI renders through, instead of growing a second
implementation that drifts from it.

KNOWN INCOMPLETENESS, deliberately not resolved by the move: the vocabulary the
renderer matches on is spread across four modules. Three of them the runner may
not import at all — ``local_operator.incidents`` (the four
``SESSION_*_MESSAGE_TYPE`` records), ``local_operator.session.peer``
(``PEER_MESSAGE_MESSAGE_TYPE``) and ``local_operator.tools.builtin``
(``TODO_REMINDER_MESSAGE_TYPE``) — and the fourth, ``local_operator.harness.comms``
(``HUB_MESSAGE_TYPE``), is the only one of the four that sits in a package the
runner may otherwise import: the module itself imports cleanly and its closure
does not, which is the next paragraph. Seven
constants, and importing this module leaks **17** modules that
``tests/unit/evaluation/runner/test_isolation.py`` denies, so the hoist does not
by itself make the renderer reachable from an episode: giving those constants a
neutral home is that stage's first move, not this one's.

``harness.comms`` is the home that reads as clean and is not. It imports
``local_operator.session.peer`` and ``local_operator.session.transcript`` at
module level for ``PEER_MESSAGE_MESSAGE_TYPE`` and ``TRANSCRIPT_FILENAME``, so
moving the six constants in ``incidents``, ``session.peer`` and ``tools.builtin``
still leaks **8**
(``incidents``, ``paths``, ``session``, ``session.attachments``,
``session.creation``, ``session.peer``, ``session.spend``,
``session.transcript``) — all of it through this one import. ``HUB_MESSAGE_TYPE``
must leave ``harness.comms`` for the neutral module too: that seventh constant
takes the residual to zero, which is why a plan built on three homes would do
the move, re-probe, and still find the module unimportable.
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
from local_operator.harness.comms import HUB_MESSAGE_TYPE
from local_operator.harness.jobs import JOB_RESULT_MESSAGE_TYPE
from local_operator.harness.types import (
    AgentMessage,
    CustomMessage,
    Message,
    TextContent,
)
from local_operator.harness.wake import WAKE_PROMPT_MESSAGE_TYPE
from local_operator.incidents import (
    SESSION_CREDENTIAL_MESSAGE_TYPE,
    SESSION_INCIDENT_MESSAGE_TYPE,
    SESSION_MCP_RECOVERY_MESSAGE_TYPE,
    SESSION_MODEL_SWITCH_MESSAGE_TYPE,
)
from local_operator.session.peer import PEER_MESSAGE_MESSAGE_TYPE
from local_operator.tools.builtin import TODO_REMINDER_MESSAGE_TYPE


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
    deliveries become user messages of their formatted text, and the newest
    ``todo_reminder`` (only the newest) becomes one too; other custom entries
    are dropped (bookkeeping never enters LLM context). ``provider_payload``
    rides along untouched.

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
    # Only the NEWEST todo reminder survives the render. An earlier one asserts
    # a todo list that has since changed, so replaying it would hand the model a
    # stale — and by then actively false — claim about its own state, and
    # re-argue a nudge it has already answered. The pruning belongs here because
    # the renderer is a pure function of the whole list and reminders are never
    # persisted, so nothing downstream could do it. Older ones simply fall
    # through to the allow-list's drop.
    newest_reminder = -1
    for index in range(len(messages) - 1, -1, -1):
        if _is_todo_reminder(messages[index]):
            newest_reminder = index
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
            # An MCP-recovery record rides it for the symmetric reason: the
            # FAILURE reaches the model as a ``session_incident`` user turn, so
            # the recovery that supersedes it has to arrive on the same surface
            # or the model keeps believing the older, more emphatic claim.
            out.append(_injected_user_message(message.details.get("text", ""), message.id))
        elif message.custom_type == GATE_TIMEOUT_CUSTOM_TYPE:
            # An unattended gate that expired is NOT a user decision, and the
            # difference is the whole reason the row exists: without it the
            # next turn reads a plain denial and re-plans around a choice
            # nobody made. Rendered here rather than carrying a `text` field
            # like the branches below because the payload is structured (tool,
            # description, waited_s) — the picker and the transcript notice
            # each phrase it for their own audience, and this is the model's.
            details = message.details or {}
            tool = str(details.get("tool") or "a tool")
            description = str(details.get("description") or "").strip()
            subject = f"{tool} ({description})" if description else tool
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
                text = (
                    f"[system] The question for {subject} was never answered: nobody "
                    "was attached to this session and it expired. No decision was "
                    "made — this was a timeout, not a choice by the user. Decide "
                    "yourself (take your recommended option where you gave one), then "
                    "say in one line what you assumed and carry on."
                )
            else:
                text = (
                    f"[system] The approval request for {subject} expired with "
                    "nobody attached to this session and was denied automatically. "
                    "This was a timeout, not a decision by the user."
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
