"""Borrower-side healing: re-asking the ONE brokering channel, and naming why it failed.

WHY THIS MODULE EXISTS, and what it deliberately is not. A session running on a peer
borrows its credentials from the owner device. When the borrowed bearer stops working
the owner may ALREADY hold a working login — it re-signed-in, or the share was
re-established — and the correct move is to ask again, once, through the grant path
that already exists (``MeshCredentialClient.request_grant_sync`` /
``grant_async``). This module composes that ask and the sentence its answer deserves;
it opens no second channel, mints nothing, stores nothing and never sees a credential
value. The one state it removes is THIS device's own cached refusal
(``client.forget_refusal``), because that memo cannot tell "the owner still has
nothing" from "the owner signed in a moment ago" — and the second is the case the
operator's repair creates.

THE DISTINCTION THAT MATTERS, stated once here so no surface has to re-derive it:

* ``owner_login`` — the owner has nothing to offer until a PERSON signs in THERE.
  Recoverable by the owner signing in; the borrower needs a fresh ask afterwards,
  and the owning device may also need to re-run the share so the new login is the
  one it serves.
* ``grant`` — this device is not (or no longer) permitted to ask for that key. No
  amount of signing in on either side changes it; a grant/share must be re-made on
  the owner. Healing re-asks and is REFUSED again, which is the correct outcome:
  healing never widens who may borrow.
* ``unreachable`` — the owner did not answer. That proves nothing about the
  credential either way, so nothing is repaired and nothing is invalidated.
* ``unknown`` — this build cannot classify the answer. Said plainly rather than
  guessed into one of the three above.

PROVENANCE. The codes are the broker's own (``types.BROKER_ERROR_CODES``) and the
evidence states are ``mcp_state``'s durable vocabulary; the mapping lives here so the
session path and the status surface cannot classify one refusal two ways. Every
operator-facing sentence is composed in ``credentials/messages.py``, which is the one
home for that copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The causes themselves live in ``types.py`` (the vocabulary home) because the
#: COPY composer cannot import this module back — see the comment there. They are
#: re-exported here so a caller reads one name for the classifier and its labels.
from local_operator.network.credentials.types import (
    CAUSE_GRANT,
    CAUSE_OWNER_LOGIN,
    CAUSE_UNKNOWN,
    CAUSE_UNREACHABLE,
)

#: Broker code -> cause. Every code in ``types.BROKER_ERROR_CODES`` is either here
#: or intentionally absent (an unclassified answer reads as ``unknown``), so a new
#: code cannot silently join one of the three named causes: it has to be stated.
CAUSE_BY_CODE: dict[str, str] = {
    # The owner has no usable login of its own: nothing to lend until a person
    # signs in there, and a fresh ask afterwards is what picks the new login up.
    "no_local_credential": CAUSE_OWNER_LOGIN,
    "interactive_required": CAUSE_OWNER_LOGIN,
    "refresh_failed": CAUSE_OWNER_LOGIN,
    "scope_mismatch": CAUSE_OWNER_LOGIN,
    "scope_unknown": CAUSE_OWNER_LOGIN,
    # Permission, not login: a person on the owner must re-share (or re-approve).
    "not_a_holder": CAUSE_GRANT,
    "revoked": CAUSE_GRANT,
    "member_not_active": CAUSE_GRANT,
    "consent_withdrawn": CAUSE_GRANT,
    "not_authorised": CAUSE_GRANT,
    "device_bound": CAUSE_GRANT,
    "device_scope_required": CAUSE_GRANT,
    "identity_mismatch": CAUSE_GRANT,
    # The grant itself is no longer the one the owner honours (a re-mint is the
    # repair), and a provider only this device could serve is a placement question
    # for the owner. Both are permission, not login.
    "grant_invalid": CAUSE_GRANT,
    "epoch_stale": CAUSE_GRANT,
    "local_only": CAUSE_GRANT,
    # No answer from the owner: unproven, and repairable only by reconnecting it.
    "owner_offline": CAUSE_UNREACHABLE,
    "not_owner": CAUSE_UNREACHABLE,
    "unavailable": CAUSE_UNREACHABLE,
    "consent_unverifiable": CAUSE_UNREACHABLE,
}

#: Durable evidence state -> cause. ``expired`` and ``unauthorized`` both mean the
#: owner's own sign-in is what stopped working (a token past expiry, or a server that
#: refused the bearer the owner served) — the two are one repair for a person.
CAUSE_BY_EVIDENCE: dict[str, str] = {
    "expired": CAUSE_OWNER_LOGIN,
    "unauthorized": CAUSE_OWNER_LOGIN,
    "revoked": CAUSE_GRANT,
    "unavailable": CAUSE_UNREACHABLE,
    "synced": "",
    # A ``held`` row (the offline hold, 2026-11-14) is a working borrow, not a
    # failure, so it names no cause — even though its code is ``owner_offline``.
    "held": "",
}


def cause_for_code(code: str) -> str:
    """The cause a broker refusal ``code`` names, or ``CAUSE_UNKNOWN``."""
    return CAUSE_BY_CODE.get(str(code or ""), CAUSE_UNKNOWN)


def cause_for_evidence(state: str, code: str = "") -> str:
    """The cause a durable evidence row names, or ``""`` when it names none.

    ``code`` refines the two states that both mean "the owner's sign-in stopped
    working": ``refresh_failed`` is a refresh the owner could not perform, and that
    is the same repair for a person. Kept for symmetry with the broker mapping so a
    reader sees one classification, not two.
    """
    if str(state or "") == "held":
        # Before the code lookup: a held row carries ``owner_offline`` as its code,
        # which alone would classify a working hold as unreachable (#3947 N).
        return ""
    by_code = CAUSE_BY_CODE.get(str(code or ""), "")
    if by_code:
        return by_code
    return CAUSE_BY_EVIDENCE.get(str(state or ""), "")


@dataclass(frozen=True)
class Repair:
    """What is wrong with one borrowed key, who fixes it, and the command they run.

    ``commands`` are the exact command lines that repair the key, and they run on
    ``owner_device`` (never here). Two honest qualifiers: they are empty for
    ``unreachable``, whose remedy is reconnecting a device rather than a command;
    and a caller that had no member name to give composes them with the same
    ``<this device>`` placeholder the rest of this copy uses, so they are pasteable
    as they stand only when ``.peer_name`` was known — otherwise the reader
    substitutes their device, exactly as ``render_broker_error``'s arms ask (review
    round 1, NIT).
    """

    key: str
    cause: str
    owner_device: str = ""
    owner_name: str = ""
    commands: tuple[str, ...] = ()
    sentence: str = ""


def repair_for(
    key: str,
    *,
    cause: str,
    code: str = "",
    owner_device: str = "",
    owner_name: str = "",
    peer_name: str = "",
) -> Repair:
    """The ``Repair`` for one key: the sentence, and the command that runs THERE.

    One composer for both kinds of key so the session path, the status row and the
    CLI cannot print three different remedies for one fact. MCP keys take the share
    verb (the grant is per holder, so the command must name the holder); provider
    keys take the owner's own login verb for a login cause, and the share verb when
    permission is what is missing.
    """
    from local_operator.network.credentials.messages import (
        render_broker_error,
        render_mcp_disconnect,
    )
    from local_operator.network.credentials.types import BrokerError, is_mcp_key

    peer = peer_name or "<this device>"
    owner = owner_name or owner_device or "the owning device"
    if is_mcp_key(key):
        sentence = render_mcp_disconnect(
            key, cause=cause, code=code, owner_name=owner, peer_name=peer
        )
        commands: tuple[str, ...] = ()
        if cause in (CAUSE_OWNER_LOGIN, CAUSE_GRANT):
            commands = (f"lop network credential share {key} --with {peer}",)
        return Repair(
            key=key,
            cause=cause,
            owner_device=owner_device,
            owner_name=owner_name,
            commands=commands,
            sentence=sentence,
        )
    # A provider key: the broker's own sentence is already the precise one, and it
    # is rendered from the CODE rather than echoed from the owner (messages.py's
    # rule), so it is reused here instead of being spelled again.
    error = BrokerError(
        code=cause_to_code(cause), key=key, owner_device=owner_device, owner_device_name=owner_name
    )
    sentence = render_broker_error(error, key=key, provider=key, owner_name=owner, accounts=())
    if cause == CAUSE_OWNER_LOGIN:
        commands = (f"lop login {key}",)
    elif cause == CAUSE_GRANT:
        commands = (f"lop network credential share {key} --with {peer}",)
    else:
        commands = ()
    return Repair(
        key=key,
        cause=cause,
        owner_device=owner_device,
        owner_name=owner_name,
        commands=commands,
        sentence=sentence,
    )


#: The one code each cause is rendered back from for a provider key, so a caller
#: holding only a CAUSE (a status row, an evidence file) still gets the broker's own
#: sentence for it rather than a second spelling composed beside it.
_CODE_FOR_CAUSE: dict[str, str] = {
    CAUSE_OWNER_LOGIN: "interactive_required",
    CAUSE_GRANT: "not_a_holder",
    CAUSE_UNREACHABLE: "owner_offline",
    CAUSE_UNKNOWN: "internal",
}


def cause_to_code(cause: str) -> str:
    """The broker code a cause is rendered from ``messages.render_broker_error`` by."""
    return _CODE_FOR_CAUSE.get(str(cause or ""), "internal")


@dataclass(frozen=True)
class HealOutcome:
    """The result of ONE healing re-ask. ``recovered`` means a bearer came back."""

    key: str
    recovered: bool
    code: str = ""
    cause: str = ""
    sentence: str = ""

    def as_dict(self) -> dict[str, Any]:
        """A JSON-safe view. There is no token here by construction, not by filter."""
        return {
            "credential_name": self.key,
            "recovered": self.recovered,
            "code": self.code,
            "cause": self.cause,
            "reason": self.sentence,
        }


async def heal_borrow(
    mesh: Any,
    key: str,
    *,
    provider: str = "",
    session_id: str = "",
    model_id: str = "",
    peer_name: str = "",
    owner_name: str = "",
) -> HealOutcome:
    """Re-ask the owner ONCE, past this device's own memo; say what it means.

    THE ASK IS ORDINARY — NOT ``force_refresh``. The owner refuses a forced refresh
    from a device that is not its admin (``owner.py``'s arms;
    ``tests/unit/network/test_credentials_owner.py``'s
    ``test_a_forced_refresh_from_a_non_admin_is_refused``), and a repair that needed
    admin rights would be no repair at all for the ordinary borrower this exists for.
    It does not need them either: the owner re-resolves its own credential on EVERY
    request ("a grant TTL is not an upstream credential generation"), so an ordinary
    ask is answered from its CURRENT login and its current grants.

    What healing changes is therefore exactly one thing — THIS device's cached
    refusal, which is what stands between a repaired owner and a running session. The
    drop happens through ``forget_refusal`` and nothing else: nothing is minted, no
    grant is extended, and a device that is still not permitted to ask is refused
    again on the dial (that refusal is the outcome, and its code is preserved).

    Never raises: a healing attempt sits on a path that is already reporting a
    failure, and replacing one error with another would lose the first.
    """
    from local_operator.network.credentials.types import Grant

    owner_device = ""
    try:
        owner_device = str(mesh.owner_of(key) or "")
    except Exception:  # noqa: BLE001 — a label read must not fail the repair
        owner_device = ""
    if not owner_name:
        try:
            owner_name = str(mesh.owner_label(key) or "")
        except Exception:  # noqa: BLE001 — see above
            owner_name = ""
    forget = getattr(mesh, "forget_refusal", None)
    if callable(forget):
        try:
            forget(key)
        except Exception:  # noqa: BLE001 — a cache repair must not fail the attempt
            pass
    try:
        answer = await mesh.grant_async(
            key,
            session_id=session_id,
            model_id=model_id,
            force_refresh=False,
            provider=provider or key,
        )
    except Exception as exc:  # noqa: BLE001 — reported, never raised
        # The class NAME only: an exception's message can carry a URL, a header or a
        # provider's echo of the credential, and none of that may reach a surface.
        return HealOutcome(
            key=key,
            recovered=False,
            code="internal",
            cause=CAUSE_UNKNOWN,
            sentence=(
                f"the request to {owner_name or owner_device or 'the owning device'} "
                f"failed ({exc.__class__.__name__}); nothing was changed"
            ),
        )
    if isinstance(answer, Grant):
        # The bearer itself is NOT carried out of here: it is already in the client's
        # grant cache, which is where every reader of a borrow gets it. This object
        # is a receipt, and a receipt is not a credential.
        return HealOutcome(key=key, recovered=True)
    code = str(getattr(answer, "code", "") or "")
    cause = cause_for_code(code)
    if not cause or cause == CAUSE_UNKNOWN:
        # An unclassified code still names itself: the code is the honest detail,
        # and inventing a cause for it would send the operator at the wrong repair.
        sentence = (
            f"the owner refused with code '{code or 'unknown'}', which this build "
            "cannot classify; run 'lop network doctor'"
        )
    else:
        sentence = repair_for(
            key,
            cause=cause,
            code=code,
            owner_device=owner_device,
            owner_name=owner_name,
            peer_name=peer_name,
        ).sentence
    return HealOutcome(key=key, recovered=False, code=code, cause=cause, sentence=sentence)
