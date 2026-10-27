"""What a person reads about a credential approval card: its outcome, and what to do next.

WHY ONE MODULE. Design and UX review round 1 on #3002 found the same card telling the
operator three different things on three surfaces: ``approvals show`` printed ``failed``
with no reason, the desktop row's remedy pointed at a terminal, and the typed route said
the card "can only be approved with Touch ID" on a machine where the terminal approves it.
Every sentence below is spelled once, here, and both the CLI renderers and the desktop row
read it, so the two surfaces cannot disagree about one card again.

THE RETRY RULE (UX U2). After a card is signed, an apply that refuses is either TRANSIENT
(another writer held the lock, a write could not be confirmed) -- the SAME signed card can
still succeed -- or PERMANENT for that card: the policy it signed moved, the recipient's
key changed, the request window ended. Offering "Try again" on a permanent refusal sends a
non-technical operator round a loop that cannot end. :func:`retryable` is the one
classification every surface reads, and anything it does not name is treated as permanent
(the safe answer: a "new request needed" remedy can be wrong only by being one extra step).

COPY RULES, same as :mod:`local_operator.network.credentials.messages`: plain sentence
case (design N1, no capitals for emphasis), a next step a person can actually take, and no
terminal syntax on a desktop remedy (operator a-b815: "connections can be done without
running any CLI commands"). The CLI renderers may add the command beside the sentence.
"""

from __future__ import annotations

from typing import Any

#: Apply refusals after which the SAME signed card can still succeed. Everything else is
#: permanent for the card (see the module docstring). ``busy``: another writer held the
#: placement lock. The ``*_unconfirmed`` codes: a write whose result could not be read
#: back, which the apply re-checks idempotently on the next attempt.
RETRYABLE_CODES = frozenset(
    {
        "busy",
        "finite_write_unconfirmed",
        "consent_write_unconfirmed",
        "finite_capability_unconfirmed",
        "finite_compensation_unconfirmed",
        "consent_prepared_pending",
    }
)

#: Why a card cannot be applied any more, in words, by refusal code. A code missing here
#: still gets :data:`_UNKNOWN_REASON` -- never silence (UX U1: a ``failed`` card with no
#: reason was a dead end).
_REASONS: dict[str, str] = {
    "busy": "Another change to who may use a credential was in progress at the same time.",
    "finite_write_unconfirmed": "The change could not be confirmed as saved.",
    "consent_write_unconfirmed": "The change could not be confirmed as saved.",
    "finite_capability_unconfirmed": "The change could not be confirmed as saved.",
    "finite_compensation_unconfirmed": "The change could not be confirmed as saved.",
    "consent_prepared_pending": "An earlier change is still being finished.",
    "consent_conflict": (
        "Who may use this credential changed after the card was approved, so it no longer "
        "matches what was approved."
    ),
    "finite_denial_mismatch": (
        "The earlier withdrawals this card would undo changed after it was approved."
    ),
    "finite_journal_conflict": "The record this card would update changed after it was approved.",
    "finite_subject_mismatch": (
        "The device, its key or the credential no longer match what was approved."
    ),
    "inherit_subject_mismatch": (
        "The device, its key or the account no longer match what was approved."
    ),
    "consent_subject_mismatch": (
        "The device, its key or the account no longer match what was approved."
    ),
    "finite_scope_mismatch": "The scope no longer matches what was approved.",
    "approval_expired": "The time to answer this card ran out.",
    "approval_denied": "This card was denied.",
    "approval_already_connected": "This card was already used.",
    "consent_recipient_inactive": "The device is no longer an active member of the network.",
}

_UNKNOWN_REASON = "The change was refused."

#: Every surface's sentence for a permanent refusal (UX U1/U2). It names the rule, not a
#: command: the requester files a new card, which the operator then approves.
NEW_REQUEST_NEEDED = (
    "This card can't be used again; a new request is needed. Ask the agent or person who "
    "asked for this access to request it again, then approve the new card."
)

#: Every surface's sentence for a transient refusal.
TRY_AGAIN = "Nothing was changed. Apply the same approved card again in a moment."


def retryable(code: str) -> bool:
    """Whether the SAME signed card can still be applied after refusal ``code``."""
    return code in RETRYABLE_CODES


def reason_for(code: str) -> str:
    """The plain-words reason for an apply refusal ``code`` (never empty)."""
    return _REASONS.get(code, _UNKNOWN_REASON)


def remedy(code: str) -> str:
    """The next step after refusal ``code``, without terminal syntax (desktop-safe)."""
    return TRY_AGAIN if retryable(code) else NEW_REQUEST_NEEDED


def outcome(code: str) -> dict[str, Any]:
    """The machine-plus-words block a surface renders for one apply refusal.

    ``retryable`` is the ONLY signal a renderer may use to offer "Try again" (UX U2); the
    two sentences are what it shows. ``code`` rides verbatim for clients that switch on it.
    """
    return {
        "code": code,
        "retryable": retryable(code),
        "reason": reason_for(code),
        "remedy": remedy(code),
    }


def refusal_sentence(code: str) -> str:
    """One refusal's words for EVERY surface: the reason, then the next step (UX U8, D9).

    No terminal syntax: the desktop's 409 ``text`` is this sentence verbatim, and the CLI
    prints its runnable commands on their own lines beside it, never inside it.
    """
    return reason_for(code) + " " + remedy(code)


def last_failure(record: dict[str, Any]) -> dict[str, Any] | None:
    """The outcome of a ``failed`` card's LAST failed run, read from its durable receipts.

    PERSISTED, not remembered (UX U2): the receipt ``mark_failed`` wrote is on the record,
    so a list read, a detail read and ``approvals show`` all carry the same reason and
    remedy after a reload or a daemon restart. ``None`` for a card that is not ``failed``.
    """
    if record.get("state") != "failed":
        return None
    for receipt in reversed(record.get("receipts") or []):
        if isinstance(receipt, dict) and receipt.get("ok") is False:
            return outcome(str(receipt.get("detail") or ""))
    return outcome("")


# ---------------------------------------------------------------------------
# The file-only disclosure and the standing-inheritance rule (UX U3)
# ---------------------------------------------------------------------------

#: Shown wherever a credential card is approved with a FILE-ONLY operator key -- the
#: desktop sheet (``typed_approval.unproven_human``) and, since this round, the CLI
#: ``approvals approve`` verb, which proceeds exactly as before (operator a-4ba6: the CLI
#: path stays allowed; a-b492 attached "with the warning" to file-only approval).
FILE_ONLY_DISCLOSURE = (
    "This Mac's operator key is a file, not Touch ID, so any program running as you could "
    "approve with it. This machine cannot prove that a person approved this card."
)


def inheritance_desktop_refusal(upgrade_available: bool) -> str:
    """The desktop's refusal of standing inheritance on a file-only key (UX U3).

    States the REAL rule: it is the DESKTOP that needs Touch ID for a permission that lasts
    until revoked (operator a-e3a5, desktop surface per a-b815); the terminal path is
    unchanged (a-4ba6). An unavailable upgrade is never offered as the fix: when this
    install cannot set up Touch ID approvals, the sentence says so and names what works.
    """
    rule = (
        "This permission lasts until you revoke it, so the desktop app approves it only with "
        "an operator key that asks for Touch ID each time. This Mac's operator key is a file. "
        "Nothing was approved."
    )
    if upgrade_available:
        return rule + " Set up Touch ID approvals on this Mac to approve it here."
    return (
        rule + " Touch ID approvals can't be set up on this install of Local Operator. To "
        "allow it anyway, approve it in a terminal on this Mac, where the same warning is "
        "shown; or ask for a share of just the current sign-in instead."
    )


# ---------------------------------------------------------------------------
# The signer (UX U4)
# ---------------------------------------------------------------------------

#: The key exists but did not produce a signature: a dismissed prompt, a locked or
#: unreadable key file. Distinct from "authority not set up" so an operator who has set it
#: up is not sent into setup again. No exception text: it can carry paths and OS jargon.
KEY_DID_NOT_SIGN = (
    "The operator key did not sign, so nothing was approved. If a prompt appeared and was "
    "dismissed, review the card and approve again; if not, check that this Mac's operator key "
    "is still in place."
)
