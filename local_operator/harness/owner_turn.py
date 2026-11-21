"""Owner-turn witness: did the operator's OWN typed turn, in this session, ask for this?

WHY THIS EXISTS. Some agent tool ops act AS THE HUMAN OWNER rather than as the
session's agent — the first is the Team Bridge ``bridge`` tool's
``channel_rename``, which goes through the same human client path as the
desktop rename route because the hub refuses agent renames in v1 (operator
decision, 2026-11-21). Such an op must run only when the operator personally
asked for it in the turn being served. Peer ``send`` messages, Team Bridge
deliveries, subagent output, scheduled wakes, monitors, tool results and a
delegating agent's brief are DATA: anyone who can write into them could
otherwise make the agent act as its owner.

WHAT IT REUSES — no new heuristic. Every fact here is one the session already
records for the account-profile consent seam (``Session._profile_owner_consent``)
and the dangerous-delete guard:

* ``Session._note_run_input`` classifies each input the run consumed with
  ``_profile_input_class``: only a typed, non-injected, non-remote user row with
  no ``custom_type`` is an ``owner`` row (its text lands in
  ``_run_owner_texts``); peer, Team Bridge, hub, remote-prompt and unknown
  internal inputs set ``_run_untrusted_input``; wake and monitor deliveries set
  ``_run_last_trigger`` to their own type.
* ``guardrails.provenance.session_lineage`` answers "did a person open this
  session?" (no parent, no opener, no lineage-registry row), which is how the
  guard refuses to treat an agent-opened workstream's first prompt as the
  operator's words.

THE DECISION (``decide``) is pure so it can be tested without a Session; the
session gathers the facts at call time (``Session.owner_turn_witness``) and the
tool receives only that callable through ``ToolContext.owner_turn_witness``.
It fails CLOSED: any missing fact is a refusal.

THE BINDING. Provenance proves WHO spoke, not WHAT they asked. ``required_text``
is the exact command ``rename <bcv_id> to <new name>``. A whole operator row must
match it (case-insensitive command words, whitespace-normalized, with an
optional leading ``please``); the target id and name keep their exact case.
This deliberately narrow grammar is not a semantic guess: a name
merely mentioned, a negated request, a question or an instruction for a different
room cannot authorize the call. Other wording refuses and tells the operator
the accepted command. The witness cites the matching row's stable entry id.

"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

#: Refusal reasons, closed. Machine words for the audit line and the tool's
#: refusal detail; never shown as product copy.
REASON_NOT_OWNER_HOST = "not_owner_host"  # no person in front of this host (SDK)
REASON_SUBAGENT = "subagent"  # a child's prompt is its parent's text
REASON_AGENT_OPENED = "agent_opened"  # a workstream another agent opened
REASON_NO_OWNER_TURN = "no_owner_turn"  # this run consumed no operator-typed row
REASON_UNTRUSTED_INPUT = "untrusted_input"  # peer/bridge/remote/unknown input in this run
REASON_LATER_TRIGGER = "later_trigger"  # a wake/monitor/injection arrived after the operator
REASON_NOT_NAMED = "not_named"  # no operator text of this run names the required text
REASON_UNAVAILABLE = "unavailable"  # the host gave the tool no witness at all

REASONS = frozenset(
    {
        REASON_NOT_OWNER_HOST,
        REASON_SUBAGENT,
        REASON_AGENT_OPENED,
        REASON_NO_OWNER_TURN,
        REASON_UNTRUSTED_INPUT,
        REASON_LATER_TRIGGER,
        REASON_NOT_NAMED,
        REASON_UNAVAILABLE,
    }
)


@dataclass(frozen=True)
class OwnerTurnWitness:
    """The verdict. ``entry_id`` is the authorizing operator row's id when ``ok``."""

    ok: bool
    entry_id: str = ""
    reason: str = ""


# Only the command words ignore case. A name's capitalization IS part of the
# requested edit; lowercasing the whole row would authorize a different name.
_COMMAND = re.compile(r"(?i:(?:please )?rename) (bcv_[0-9a-z]{8,64}) (?i:to) (.+)")


def _command(text: str) -> tuple[str, str] | None:
    match = _COMMAND.fullmatch(" ".join(text.split()))
    return (match[1], match[2]) if match is not None else None


def decide(
    *,
    owner_host: bool,
    is_subagent: bool,
    operator_opened: bool,
    owner_rows: Sequence[tuple[str, str]],
    untrusted_input: bool,
    last_trigger_typed: bool,
    required_text: str,
) -> OwnerTurnWitness:
    """The witness for one call. ``owner_rows`` is ``(entry_id, text)``, oldest first."""

    if not owner_host:
        return OwnerTurnWitness(False, reason=REASON_NOT_OWNER_HOST)
    if is_subagent:
        return OwnerTurnWitness(False, reason=REASON_SUBAGENT)
    if not operator_opened:
        return OwnerTurnWitness(False, reason=REASON_AGENT_OPENED)
    if not owner_rows:
        return OwnerTurnWitness(False, reason=REASON_NO_OWNER_TURN)
    if untrusted_input:
        return OwnerTurnWitness(False, reason=REASON_UNTRUSTED_INPUT)
    if not last_trigger_typed:
        return OwnerTurnWitness(False, reason=REASON_LATER_TRIGGER)
    requested = _command(required_text)
    if requested is None:
        return OwnerTurnWitness(False, reason=REASON_NOT_NAMED)
    # Newest first: the audit should cite the most recent message that asked.
    for entry_id, text in reversed(owner_rows):
        if entry_id and _command(text) == requested:
            return OwnerTurnWitness(True, entry_id=entry_id)
    return OwnerTurnWitness(False, reason=REASON_NOT_NAMED)


__all__ = ["REASONS", "OwnerTurnWitness", "decide"]
