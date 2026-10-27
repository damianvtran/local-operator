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
classification every surface reads, and a refusal CODE it does not name is treated as
permanent (the safe answer: a "new request needed" remedy can be wrong only by being one
extra step). A receipt detail that is not a code at all -- prose, an exception class name --
is the exception, and :func:`outcome` says why.

COPY RULES, same as :mod:`local_operator.network.credentials.messages`: plain sentence
case (design N1, no capitals for emphasis), a next step a person can actually take, and no
terminal syntax on a desktop remedy (operator a-b815: "connections can be done without
running any CLI commands"). The CLI renderers may add the command beside the sentence.
"""

from __future__ import annotations

import re
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

#: Refusal codes the credential apply can store that have no sentence of their own yet. They
#: ARE codes -- permanent for the card, worded as :data:`_UNKNOWN_REASON` -- and are listed
#: so :data:`KNOWN_CODES` can tell a code from prose. ``test_every_refusal_code_is_known``
#: walks the credential modules' refusal literals and fails when one is missing here.
_UNWORDED_CODES = frozenset(
    {
        "approval_anchor_mismatch",
        "approval_anchor_missing",
        "approval_anchor_unavailable",
        "approval_decision_conflict",
        "approval_invalid",
        "approval_not_approved",
        "approval_not_runnable",
        "approval_receipt_refused",
        "approval_record_tampered",
        "approval_request_conflict",
        "approval_requester_mismatch",
        "approval_run_in_flight",
        "approval_signature_invalid",
        "approval_signing_unavailable",
        "approval_withdraw_conflict",
        "bad_request",
        "consent_action_refused",
        "consent_execution_missing",
        "consent_identity_invalid",
        "consent_invalid",
        "consent_not_enabled",
        "consent_origin_refused",
        "consent_path_invalid",
        "consent_revision_invalid",
        "consent_run_stale",
        "consent_source_mismatch",
        "consent_subset_refused",
        "consent_unsupported",
        "credential_approval_unsupported",
        "credential_authority_unavailable",
        "credential_disclosure_changed",
        "credential_operation_mismatch",
        "credential_path_invalid",
        "credential_presence_required",
        "credential_prompt_cancelled",
        "credential_request_changed",
        "credential_sign_in_flight",
        "credential_sign_rate_limited",
        "device_bound",
        "finite_disclosure_invalid",
        "finite_future_scope_conflict",
        "finite_intent_missing",
        "finite_journal_unknown",
        "finite_policy_unknown",
        "finite_projection_withheld",
        "finite_recovery_failed",
        "finite_recovery_pending",
        "finite_supersede_unneeded",
        "inherit_account_required",
        "inherit_account_unknown",
        "inherit_inventory_unreadable",
        "inherit_login_ambiguous",
        "inherit_no_native_login",
        "inherit_not_applicable",
        "inherit_org_unknown",
        "inherit_permission_exists",
        "inherit_record_unreadable",
        "inherit_recovery_pending",
        "inherit_request_open",
        "inherit_scope_unrepresentable",
        "inherit_transport_not_ready",
        "inherit_withdrawn",
        "lifecycle_counter_exhausted",
        "lifecycle_invalid",
        "lifecycle_path_invalid",
        "local_only",
        "network_enrollment_snapshot_changed",
        "no_local_credential",
        "not_a_holder",
        "not_owner",
        "profile_parent_missing",
        "protocol_error",
        "scope_mismatch",
        "scope_unknown",
        "unknown_approval",
    }
)

#: Every value of a failed receipt's ``detail`` that IS a refusal code (#3002 review r3,
#: M1). Anything else is not a code at all: ``approvals.py`` stores the whole "the run was
#: superseded: ..." sentence there, and ``finite_share._release_run`` falls back to an
#: exception's class name (``OSError``, ``KeyError``) for a failure that was not a refusal.
#: Reading those as unmapped CODES labelled them "The change was refused." and permanent,
#: so the operator lost the real reason and ``_refuse_spent`` blocked a retry that could
#: well succeed (an fs error, a superseded run). See :func:`outcome`.
KNOWN_CODES = frozenset(_REASONS) | RETRYABLE_CODES | _UNWORDED_CODES

#: The SHAPE of a refusal code (``word_word``), the backstop beside :data:`KNOWN_CODES`: a
#: code raised by a module outside the credential package (``enrollment``'s, the store's)
#: keeps the permanent "refused" reading instead of being shown raw as if it were prose.
#: Neither failure detail the whitelist exists for has this shape -- the supersede receipt
#: is a sentence, and an exception class name is CamelCase.
_CODE_SHAPE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+")


def is_refusal_code(value: str) -> bool:
    """Whether a stored failure ``detail`` is a refusal code rather than prose (review r3 M1)."""
    return value in KNOWN_CODES or _CODE_SHAPE.fullmatch(value) is not None


#: The longest raw detail shown as a reason. The record's receipts are device-local, but
#: an exception class name or a supersede sentence is the most a reader needs.
_RAW_REASON_MAX_CHARS = 300

#: A filesystem path inside a raw detail (POSIX, ``~``-relative or a Windows drive), the
#: same shape ``server.utils.desktop_onboard.detail`` scrubs from receipts: an OS error can
#: name a profile path or a temporary key file, and the card needs the fact, not the path.
_PATH_RE = re.compile(r"(?<![\w:])(?:~?/[^\s\"']+|[A-Za-z]:\\[^\s\"']+)")

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


def _raw_reason(detail: str) -> str:
    """A non-code failure detail as a readable reason: printable, path-free, bounded."""
    # A control character becomes a SPACE, not nothing: dropping it would glue the words
    # either side together, and a path glued to a word escapes the scrub's boundary.
    text = " ".join("".join(ch if ch.isprintable() else " " for ch in detail).split())
    text = _PATH_RE.sub("[path]", text)
    if len(text) > _RAW_REASON_MAX_CHARS:
        text = text[: _RAW_REASON_MAX_CHARS - 1].rstrip() + "\u2026"
    return text


def outcome(code: str) -> dict[str, Any]:
    """The machine-plus-words block a surface renders for one apply refusal.

    ``retryable`` is the ONLY signal a renderer may use to offer "Try again" (UX U2); the
    two sentences are what it shows. ``code`` rides verbatim for clients that switch on it.

    A value that is not a refusal code (:func:`is_refusal_code`) is a failure DETAIL (review
    r3, M1): its own words become the reason, sanitized, and it is classified RETRYABLE. That
    is the conservative direction here, unlike for a code: the apply re-verifies the
    signature, decision, window and policy on every attempt, so a retry against a card that
    really is spent stores the real refusal code and is then refused as permanent -- while
    marking an ``OSError`` or a superseded run permanent locked the operator out of a card
    the same click could still apply. ``""`` (no receipt detail at all) keeps the generic
    permanent reading it always had.
    """
    if code and not is_refusal_code(code):
        return {
            "code": code,
            "retryable": True,
            "reason": _raw_reason(code) or _UNKNOWN_REASON,
            "remedy": TRY_AGAIN,
        }
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
