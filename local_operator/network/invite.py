"""Invites: mint, redeem, burn. The one single-use credential in the mesh.

    lop1.<b64url(payload)>.<b64url(tag)>

ONE LINE, SELF-CONTAINED, AND PASTEABLE — and it is a BEARER CREDENTIAL, so it
is handled like one: ``lop network invite`` writes it to a 0600 file and prints
the PATH, never the token. A token on stdout ends up in the agent's transcript,
and the transcript is replayed to the provider on every later turn.

The envelope carries a ``ttl_s`` DURATION rather than an absolute expiry, which
takes cross-host clock skew out of pairing entirely: the inviter enforces
freshness against its own clock (it minted the token, so there is no skew to
speak of) and the joiner enforces nothing. ``expires_at`` exists for the human and
is computed locally.

SINGLE USE IS STATE ON DISK, not a protocol convention. ``minted`` → ``redeemed``
is written the instant a valid redemption arrives, BEFORE any human sees
anything; ``consumed`` is written when the ceremony ends in EITHER outcome, and
before the frame that announces it. So a relay restart mid-pairing cannot be used
to replay an invite, and a mismatched SAS burns the token — which is what makes
an attacker's next attempt need a fresh human action.

THE OPTIONAL ``device_id`` BINDING. An invite may name the one device id allowed
to redeem it; a token stolen from the transport of an unbound invite is a token
anyone can present, while a bound one is worthless to a different device. Checked
at redemption here, so the rule has one home.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from secrets import token_bytes
from typing import Any

from local_operator.network.handshake import Credential
from local_operator.network.types import (
    INVITE_ROLES,
    InviteRecord,
    NetworkRecord,
    PairingRefusal,
    capabilities_for_role,
)
from local_operator.network.wire import b64u, crockford, invite_key, invite_mac, unb64u

INVITE_PREFIX = "lop1"
INVITE_ENVELOPE_VERSION = 1
INVITE_KIND = "lop-invite"

#: Ten minutes: long enough to walk a token to another machine and read a code
#: off two screens, short enough that a leaked token is usually dead already.
INVITE_TTL_DEFAULT_S = 600.0

#: The largest token this will parse, checked before decoding: the envelope is a
#: few hundred bytes, and a "token" of a megabyte is not one.
MAX_TOKEN_BYTES = 8 * 1024

REASON_INVALID = "invite_invalid"
REASON_EXPIRED = "invite_expired"
REASON_IN_USE = "invite_in_use"
REASON_USED = "invite_already_used"
REASON_EPOCH = "invite_epoch_stale"
REASON_INVITER = "invite_wrong_inviter"
REASON_DEVICE = "invite_device_mismatch"
REASON_TAG = "invite_bad_tag"


def new_invite_id() -> str:
    """A short Crockford base32 id: short because a human reads it in a listing."""
    return crockford(token_bytes(8))


@dataclass(frozen=True)
class InviteEnvelope:
    """The decoded payload of a token. Untrusted until the tag verifies."""

    network_id: str
    network_name: str
    epoch: int
    material: str  # base64url(32) — this is the transfer of the network secret
    inviter_device_id: str
    inviter_name: str
    invite_id: str
    issued_at: float
    ttl_s: float
    role: str
    capabilities: list[str] = field(default_factory=list)
    hosts: list[str] = field(default_factory=list)
    device_id: str = ""
    v: int = INVITE_ENVELOPE_VERSION
    kind: str = INVITE_KIND

    def payload(self) -> dict[str, Any]:
        return {
            "v": self.v,
            "kind": self.kind,
            "network_id": self.network_id,
            "network_name": self.network_name,
            "epoch": self.epoch,
            "secret": self.material,
            "inviter_device_id": self.inviter_device_id,
            "inviter_name": self.inviter_name,
            "invite_id": self.invite_id,
            "issued_at": self.issued_at,
            "ttl_s": self.ttl_s,
            "role": self.role,
            "capabilities": list(self.capabilities),
            "hosts": list(self.hosts),
            "device_id": self.device_id,
        }

    @property
    def expires_at(self) -> float:
        return self.issued_at + self.ttl_s

    def to_record(self, epoch: int = 0) -> InviteRecord:
        """The on-disk invite entry for THIS device's own minted invites.

        ``epoch`` is the minting epoch, stored rather than re-derived: the inviter
        must be able to tell a stale token from a fresh one after its own rotation,
        and the token is not in its hands any more.
        """
        return InviteRecord(
            invite_id=self.invite_id,
            minted_at=self.issued_at,
            epoch=epoch,
            ttl_s=self.ttl_s,
            role=self.role,
            capabilities=list(self.capabilities),
            hosts=list(self.hosts),
            device_id=self.device_id,
        )


@dataclass
class MintedInvite:
    """What ``lop network invite`` produces: the token, and the record to store."""

    token: str
    envelope: InviteEnvelope
    record: InviteRecord


# ---------------------------------------------------------------------------
# Mint
# ---------------------------------------------------------------------------


def mint(
    record: NetworkRecord,
    material: str,
    *,
    role: str,
    ttl_s: float = INVITE_TTL_DEFAULT_S,
    hosts: list[str] | None = None,
    device_id: str = "",
    invite_id: str | None = None,
    now: float | None = None,
) -> MintedInvite:
    """Mint one invite token for ``record``.

    ``material`` is the CURRENT epoch's secret. The capabilities are resolved from
    the role HERE and carried in the envelope and the record, so a later change to
    ``ROLE_CAPABILITIES`` cannot retroactively widen an invite that has already
    been handed to someone.
    """
    if role not in INVITE_ROLES:
        raise PairingRefusal(
            "bad_role", f"unknown invite role {role!r}; use one of {', '.join(INVITE_ROLES)}"
        )
    if ttl_s <= 0:
        raise PairingRefusal("bad_ttl", "an invite must expire; give a positive lifetime")
    resolved = sorted(capabilities_for_role(role))
    envelope = InviteEnvelope(
        network_id=record.network_id,
        network_name=record.name,
        epoch=record.epoch,
        material=material,
        inviter_device_id=record.self_device_id,
        inviter_name=record.listen.get("name") or record.self_device_id,
        invite_id=invite_id or new_invite_id(),
        issued_at=time.time() if now is None else now,
        ttl_s=float(ttl_s),
        role=role,
        capabilities=resolved,
        # The hosts default to what this device advertises: a token that told the
        # joiner nothing about where to dial would need the operator to remember
        # an endpoint, which is exactly the kind of step a pairing flow must not
        # have.
        hosts=list(hosts or record.listen.get("advertised") or []),
        device_id=device_id,
    )
    return MintedInvite(
        token=encode(envelope, material), envelope=envelope, record=envelope.to_record(record.epoch)
    )


def encode(envelope: InviteEnvelope, material: str) -> str:
    """``lop1.<b64url(payload)>.<b64url(tag)>`` with a canonical payload.

    Canonical (sorted keys, no whitespace) because the tag is over the payload's
    BYTES: two encoders that disagreed about key order would produce tokens one
    side could not verify, and the disagreement would look like tampering.
    """
    payload = json.dumps(
        envelope.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    key = invite_key(material, envelope.network_id, envelope.invite_id)
    tag = invite_mac(key, payload)
    return f"{INVITE_PREFIX}.{b64u(payload)}.{b64u(tag)}"


# ---------------------------------------------------------------------------
# Decode and verify
# ---------------------------------------------------------------------------


def decode(token: str) -> InviteEnvelope:
    """Parse a token's structure. Says nothing about authenticity (see check_tag)."""
    text = token.strip()
    if len(text) > MAX_TOKEN_BYTES:
        raise PairingRefusal(REASON_INVALID, "that invite token is too long to be one")
    parts = text.split(".")
    if len(parts) != 3 or parts[0] != INVITE_PREFIX:
        raise PairingRefusal(
            REASON_INVALID,
            "that does not look like an invite token (expected lop1.<payload>.<tag>)",
        )
    try:
        payload = json.loads(unb64u(parts[1]).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise PairingRefusal(REASON_INVALID, "that invite token's payload is not readable") from exc
    if not isinstance(payload, dict) or payload.get("kind") != INVITE_KIND:
        raise PairingRefusal(REASON_INVALID, "that invite token is not a lop invite")
    if int(payload.get("v") or 0) != INVITE_ENVELOPE_VERSION:
        raise PairingRefusal(
            "invite_version",
            f"that invite was minted by a different version of lop "
            f"(token v{payload.get('v')}, this build v{INVITE_ENVELOPE_VERSION})",
        )
    missing = [
        field_name
        for field_name in ("network_id", "invite_id", "secret", "inviter_device_id", "ttl_s")
        if not payload.get(field_name)
    ]
    if missing:
        raise PairingRefusal(REASON_INVALID, f"that invite token is missing {', '.join(missing)}")
    return InviteEnvelope(
        network_id=str(payload["network_id"]),
        network_name=str(payload.get("network_name") or ""),
        epoch=int(payload.get("epoch") or 0),
        material=str(payload["secret"]),
        inviter_device_id=str(payload["inviter_device_id"]),
        inviter_name=str(payload.get("inviter_name") or ""),
        invite_id=str(payload["invite_id"]),
        issued_at=float(payload.get("issued_at") or 0.0),
        ttl_s=float(payload["ttl_s"]),
        role=str(payload.get("role") or "read"),
        capabilities=[str(cap) for cap in payload.get("capabilities") or []],
        hosts=[str(host) for host in payload.get("hosts") or []],
        device_id=str(payload.get("device_id") or ""),
        v=int(payload.get("v") or INVITE_ENVELOPE_VERSION),
        kind=str(payload.get("kind") or INVITE_KIND),
    )


def check_tag(token: str, envelope: InviteEnvelope, key: bytes) -> None:
    """Verify the MAC over the token's payload bytes.

    The bytes are re-derived from the token rather than from the decoded envelope,
    because re-serialising the envelope is only canonical if every field survived
    the round trip — and the day one does not, this check would start passing for
    the wrong reason.
    """
    import hmac

    parts = token.strip().split(".")
    if len(parts) != 3:
        raise PairingRefusal(REASON_INVALID, "that invite token is malformed")
    payload = unb64u(parts[1])
    expected = invite_mac(key, payload)
    if not hmac.compare_digest(expected, unb64u(parts[2])):
        # The reason is deliberately coarse: a wrong tag means the token was not
        # minted from this network's material, which is also what a tampered
        # token looks like, and the distinction is not the peer's business.
        raise PairingRefusal(
            REASON_TAG,
            "that invite token was not minted by this network (its signature does not "
            "verify against this network's key)",
        )


def open_token(
    token: str, *, material_for: Callable[[InviteEnvelope], str | None]
) -> InviteEnvelope:
    """Decode a token and verify its tag against the material the caller can find.

    ``material_for`` answers "which secret should this token be checked against",
    given the envelope's own (untrusted) claim about which network it is for. The
    caller supplies the current secret for that network id, or ``None`` when it has
    no such network — in which case the token cannot be verified and is refused
    with the same coarse sentence.
    """
    envelope = decode(token)
    material = material_for(envelope)
    if not material:
        raise PairingRefusal(
            REASON_TAG,
            f"this device has no network {envelope.network_id}, so it cannot verify that "
            "invite token",
        )
    check_tag(token, envelope, invite_key(material, envelope.network_id, envelope.invite_id))
    return envelope


def invite_credential_for(record: NetworkRecord, material: str, invite_id: str) -> Credential:
    """The MAC key a JOIN handshake uses: one token, one key, one handshake.

    Lives here rather than in the handshake because the inviter derives it from
    its own stored secret while the joiner derives the SAME key from the material
    the token carried — two callers, one function, so the two derivations cannot
    drift.
    """
    invite = record.invite(invite_id)
    return Credential(
        "invite",
        invite.epoch if invite else record.epoch,
        invite_key(material, record.network_id, invite_id),
    )


# ---------------------------------------------------------------------------
# Redemption
# ---------------------------------------------------------------------------


@dataclass
class Redemption:
    """A validated redemption, and the record state change it implies."""

    envelope: InviteEnvelope
    record: InviteRecord
    #: The capabilities the joiner will be admitted with — from the RECORD (minted
    #: here), never from the token's copy of them, because the token is untrusted
    #: input and a forged envelope must not be able to upgrade its own grant.
    capabilities: frozenset[str]


def claim(
    record: NetworkRecord,
    invite_id: str,
    *,
    device_id: str,
    epoch: int,
    now: float | None = None,
) -> Redemption:
    """The INVITER's path: validate a redemption from the record it minted.

    Same checks and the same order as :func:`redeem`, but built from the stored
    invite rather than from a token: the inviter never sees the token again (the
    joiner holds it, and its possession is proven by the MAC), so re-deriving the
    checks from the envelope would mean trusting the joiner to resend what it says
    the token contained. The stored record is the authority here.
    """
    moment = time.time() if now is None else now
    stored = record.invite(invite_id)
    if stored is None:
        raise PairingRefusal(REASON_INVALID, "that invite was never minted by this device")
    if stored.state == "redeemed":
        raise PairingRefusal(REASON_IN_USE, "that invite is already being redeemed")
    if stored.state == "consumed":
        raise PairingRefusal(REASON_USED, "that invite has already been used")
    if not stored.is_fresh(moment):
        raise PairingRefusal(
            REASON_EXPIRED,
            f"that invite expired {int(moment - stored.expires_at)}s ago; mint a new one",
        )
    if stored.epoch and epoch != record.epoch:
        raise PairingRefusal(
            REASON_EPOCH,
            f"the network rotated to epoch {record.epoch} after that invite was minted at "
            f"epoch {stored.epoch}, so the invite is stale; mint a new one",
        )
    if stored.device_id and stored.device_id != device_id:
        raise PairingRefusal(
            REASON_DEVICE,
            f"that invite is bound to device {stored.device_id} and is being presented by "
            f"{device_id}",
        )
    return Redemption(
        envelope=InviteEnvelope(
            network_id=record.network_id,
            network_name=record.name,
            epoch=stored.epoch or record.epoch,
            material="",
            inviter_device_id=record.self_device_id,
            inviter_name=record.self_device_id,
            invite_id=stored.invite_id,
            issued_at=stored.minted_at,
            ttl_s=stored.ttl_s,
            role=stored.role,
            capabilities=list(stored.capabilities),
            hosts=list(stored.hosts),
            device_id=stored.device_id,
        ),
        record=stored,
        capabilities=frozenset(stored.capabilities),
    )


def redeem(
    record: NetworkRecord,
    envelope: InviteEnvelope,
    *,
    device_id: str,
    now: float | None = None,
) -> Redemption:
    """Validate a redemption on the INVITER and move the invite to ``redeemed``.

    Order matters and each step has its own sentence, because "the invite did not
    work" is the least useful thing a human can be told:

    1. this device actually minted that invite id (a token stolen from another
       network's inviter must not redeem here);
    2. its state is ``minted`` — ``redeemed`` means someone else is mid-ceremony,
       ``consumed`` means it burned;
    3. it is fresh by the MINTING clock;
    4. the network is still at the epoch the token names (a rotation mid-invite
       invalidates it, and re-minting is one command);
    5. the token names THIS device as the inviter — a token stolen from a
       permissive peer must not be redeemable against this one;
    6. if the invite is BOUND to a device id, this is that device.

    The state change is made by the caller via :func:`mark_redeemed`, AFTER
    validation, so a refused redemption does not burn an invite whose owner is
    simply mid-typing a code somewhere else.
    """
    moment = time.time() if now is None else now
    stored = record.invite(envelope.invite_id)
    if stored is None:
        raise PairingRefusal(REASON_INVALID, "that invite was never minted by this device")
    if stored.state == "redeemed":
        raise PairingRefusal(REASON_IN_USE, "that invite is already being redeemed")
    if stored.state == "consumed":
        raise PairingRefusal(REASON_USED, "that invite has already been used")
    if not stored.is_fresh(moment):
        raise PairingRefusal(
            REASON_EXPIRED,
            f"that invite expired {int(moment - stored.expires_at)}s ago; mint a new one",
        )
    if envelope.epoch != record.epoch:
        raise PairingRefusal(
            REASON_EPOCH,
            f"the network rotated from epoch {envelope.epoch} to {record.epoch} after that "
            "invite was minted, so the invite is stale; mint a new one",
        )
    if envelope.inviter_device_id != record.self_device_id:
        raise PairingRefusal(
            REASON_INVITER,
            "that invite names a different device as its inviter, so it cannot be redeemed " "here",
        )
    if stored.device_id and stored.device_id != device_id:
        raise PairingRefusal(
            REASON_DEVICE,
            f"that invite is bound to device {stored.device_id} and is being presented by "
            f"{device_id}",
        )
    return Redemption(
        envelope=envelope,
        record=stored,
        capabilities=frozenset(stored.capabilities),
    )


def mark_redeemed(
    record: NetworkRecord, invite_id: str, *, device_id: str, now: float | None = None
) -> InviteRecord:
    """Write ``redeemed`` — BEFORE any human is shown anything.

    The order is the point: the state is on disk before the SAS prompt appears, so
    a crash between "a valid redemption arrived" and "a person looked at a code"
    cannot leave a replayable token behind.
    """
    invite = record.invite(invite_id)
    if invite is None:
        raise PairingRefusal(REASON_INVALID, "that invite was never minted by this device")
    invite.state = "redeemed"
    invite.redeemed_by = device_id
    invite.redeemed_at = time.time() if now is None else now
    return invite


def consume(
    record: NetworkRecord,
    invite_id: str,
    *,
    outcome: str,
    now: float | None = None,
) -> InviteRecord:
    """Write ``consumed`` — the terminal state, whatever the outcome was.

    Admitted, aborted, SAS mismatch, timeout: all four consume. A mismatched SAS
    that left the invite usable would let an attacker try again with the same
    token, which is exactly the ~2²⁰ grind the design accepts only once per human
    action.
    """
    invite = record.invite(invite_id)
    if invite is None:
        raise PairingRefusal(REASON_INVALID, "that invite was never minted by this device")
    invite.state = "consumed"
    invite.outcome = outcome
    if invite.redeemed_at is None:
        invite.redeemed_at = time.time() if now is None else now
    return invite


def claim_or_consume(
    record: NetworkRecord,
    invite_id: str,
    *,
    device_id: str,
    epoch: int,
    now: float | None = None,
) -> Redemption:
    """``claim``, with design §5.1/§5.4's one TERMINAL refusal applied.

    Every other refusal in ``claim`` is about THIS attempt — a stale epoch, an
    expired or unknown invite, another redemption already in flight — and leaving
    the invite standing is what lets the operator fix the cause and retry. A
    ``bound_device`` mismatch is not: ``--device`` exists so that a token which
    leaked to a second device is worthless to it, and a bound invite presented by
    another device is therefore the leak itself, which the design consumes rather
    than leaves standing for a second try. The cost is one more
    ``lop network invite``, and it is the whole point of the binding.

    The state change happens HERE rather than at the call site because the rule is
    a property of the state machine, not of one caller: the caller's only extra
    duty is to persist ``record`` (the same duty it has after
    :func:`mark_redeemed`).
    """
    try:
        return claim(record, invite_id, device_id=device_id, epoch=epoch, now=now)
    except PairingRefusal as refusal:
        if refusal.code == REASON_DEVICE:
            consume(record, invite_id, outcome="wrong_device", now=now)
        raise


# ---------------------------------------------------------------------------
# The human step's words
# ---------------------------------------------------------------------------


def host_candidates(envelope: InviteEnvelope, override: str | None) -> list[str]:
    """Where the joiner dials, in order: an explicit ``--host`` first, then the token.

    The token's host list is untrusted input used only as a dial target — every
    frame it would let an attacker intercept still has to pass the handshake and
    the MAC under the invite key, so a hostile host list buys a denial of service
    at worst.
    """
    if override:
        return [override]
    return list(envelope.hosts)


def joiner_prompt(envelope: InviteEnvelope, sas: str, fingerprint: str) -> str:
    """What the JOINING device shows its human before anything is admitted.

    ``code 481 926`` is a TRANSCRIPTION request, not a yes/no, and the fingerprint
    is printed in the same panel: the digits are worth ~20 bits and the fingerprint
    160, so a pairing over a public path has a real check available without a
    second command.
    """
    from local_operator.network.wire import sas_display

    inviter = envelope.inviter_name or envelope.inviter_device_id
    network = envelope.network_name or envelope.network_id
    return (
        f"{inviter} ({network}) offers role {envelope.role} — code {sas_display(sas)}\n"
        f"  fingerprint {fingerprint}\n"
        "type the code shown there:"
    )


def inviter_prompt(
    envelope: InviteEnvelope, device_id: str, name: str, transcribed: str, *, derived: str
) -> str:
    """The inviter's question, from the envelope this device minted."""
    return inviter_prompt_for(
        network_name=envelope.network_name,
        role=envelope.role,
        device_id=device_id,
        name=name,
        transcribed=transcribed,
        derived=derived,
    )


def inviter_prompt_for(
    *,
    network_name: str,
    role: str,
    device_id: str,
    name: str,
    transcribed: str,
    derived: str,
) -> str:
    """What the INVITING device shows its human — BOTH codes.

    §5.3 is "B transcribes, A compares", and the comparison is made by a PERSON on
    A: their own derived code is what the other screen must show. Printing only the
    transcribed value (which is what this prompt first did, and what the guide agent
    found) leaves the inviter's human with nothing to compare against, so the check
    is decorative — the joiner's value was already verified by the relay, and the
    inviter's own screen said nothing.

    One renderer, used by the relay's foreground prompt, the pending record (whose
    stored ``prompt`` is this string), ``lop network confirm`` and ``--json``: two
    renderings of one question is how two prompts drift apart.
    """
    from local_operator.network.wire import sas_display

    label = name or device_id
    return (
        f'{device_id} ("{label}", new device) transcribed {sas_display(transcribed)} to join '
        f"{network_name} as {role}.\n"
        f"YOUR screen shows {sas_display(derived)}.\n"
        "Do they match? Confirm only if the other device shows the same code."
    )
