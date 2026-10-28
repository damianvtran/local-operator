"""The ``operator_device_pair`` approval: a phone asking to become a signer of this computer.

WHY A RECORD AND NOT A PAIRING CODE. ``lop pair`` today mints a short-lived code and
signs whatever key presents it (``operator/devices.py``'s handshake), which is one
gesture and no statement: the person never sees WHICH phone they are letting in, and on
a host whose operator key is a file the signature silently proves nothing about a person
(design §4 F1). This op replaces that for the phone-driven flow with the ordinary approval
storage: the phone files a record naming its key, the person reads a card that says what
would be granted, types the six digits the phone shows, and the certificate is issued on
that decision. The store already gives the record a window, a write-once decision, an
audit trail, a run lease and a withdrawal path, so this module adds the pairing's payload
and its apply -- not a second authority mechanism.

TWO SIGNATURES, NOT ONE (design §6 / the split's §5, operator decision for v1). Signing
the approval and issuing the certificate are two calls into the operator key, therefore
two OS prompts on a presence-gated host and none on a file-only host, where signing is
silent. The one-gesture cross-domain artifact (a DeviceCert that doubles as the approval
decision) was rejected for v1: it makes the approval record itself a live credential and
inverts the ordering. Nothing here assumes the decision and the certificate are the same
signature.

THE TYPED CODE IS THE FIRST LOCK; IT IS NOT A BOUNDARY. ``check_ack`` derives the six
digits from ``HKDF(np || nd || device_spki || operator_spki || approval_id)`` and compares
them in constant time with what the person typed. The desktop must compute that value to
check it, so anything running as the operator can compute it too (RESIDUAL, accepted by
the operator, disclosed on the card: ask a-2a8b, "Allow, with the warning and a 'Set up
Touch ID' offer"). What the code DOES buy, and the reason it is required rather than
optional: a reflex approval is impossible without transcribing a value that exists only on
the phone's screen, and a guess costs a strike (three strikes closes the card as a deny).
What it does NOT buy: proof that a person, rather than the operator's uid, approved. Only
the presence-gated signature says that, which is why the row and the audit event carry the
assurance level and the card warns when it is ``file_only``.

WHAT IS NEVER IN THIS RECORD, and therefore never in ``request_digest``: the six digits
(only a commitment to the phone's nonce and the two nonces, which are useless without the
keys), the relay password, the Radient OAuth token or its edge assertion, the desktop
bearer, and any address (IP, hostname, tunnel id or custom URL). Identity is the KEY: a
retry from a new network is digest-equal, so ``create_request`` returns the same record
(operator decision a-2a8b, "network changes must not break pairing").

WHY THE PAYLOAD'S SHAPE IS CHECKED THIS EXACTLY. ``request_digest`` covers the whole
``what`` block, so a key the disclosure does not show is a byte the person signed unseen.
:func:`exact_shape` is the ONE predicate behind :func:`disclosure`, :func:`apply`, the
filing guard and the typed route's operation lookup -- #3087's discipline, applied to a
record whose ``sas`` block is the tempting place to hide one.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from pathlib import Path
from typing import Any, Mapping

from local_operator.network import approval_ops, approvals
from local_operator.network.types import MeshRefusal

#: The ``what`` key this op claims. Presence is the boundary (``approval_ops``), so a
#: record carrying it can never fall through to a generic signer, whatever its payload.
OP = "pair_phone"

#: The receipt step of the typed-code check. Named because three places read it: the
#: strike count, the decision guard below, and the card's own progress lines.
STEP_ACK = "pair_code"

#: The receipt step of the certificate install.
STEP_INSTALL = "pair_cert"

#: The HKDF ``info`` label of the SAS. Its own label, like ``wire.sas_code``'s
#: ``lop-mesh-sas-v1``, so a value derived for one ceremony can never be replayed as
#: another one's digits.
SAS_LABEL = b"lop-pair-sas-v1"

#: Nonce width. Fixed, so ``np || nd`` is unambiguous without length prefixes and the
#: derivation stays the design's literal expression.
NONCE_BYTES = 32

#: How many wrong codes close the card. Small on purpose: a person transcribing six digits
#: from their own screen gets there first try, and the alternative to a bound is an
#: unlimited oracle over a ~20-bit value.
MAX_STRIKES = 3

#: The roles and platforms v1 can state. A role this build cannot describe is refused at
#: filing rather than signed with a limit nothing enforces (design §4: the runtime does not
#: read ``scope``, so a narrower request would disclose a promise this build cannot keep).
_ROLES = frozenset({"approver"})
_PLATFORMS = frozenset({"ios", "android", "web"})
_VIA_ROUTES = frozenset({"radient", "custom"})

#: The longest certificate this build issues. The disclosure states the number, so the
#: bound is about refusing absurd statements, not about hiding one.
MAX_CERT_LIFETIME_S = 365 * 24 * 3600

#: The EXACT ``what`` keys, payload keys and envelope blocks this build mints. ONE set per
#: depth, read by the shape predicate AND by apply, so the two cannot disagree about which
#: records are describable.
_WHAT = frozenset({OP, "anchor"})
_FIELDS = frozenset(
    {"role", "scopes", "cert_lifetime_s", "via", "sas", "computer_id_pub"}
)
_DEVICE_KEYS = frozenset({"device_id", "spki", "key_fp", "name", "platform"})
_SAS_KEYS = frozenset({"commit", "nd", "np"})
_REQUESTED_BY = frozenset({"surface", "device_id", "session_id"})

DISCLOSURE_VERSION = 1

#: Every surface's sentence for "this card is not the exact shape this build can state".
#: The desktop card, the CLI's card lines and the generic approvers all answer with it, so
#: a malformed pairing request reads the same wherever it is met.
PAYLOAD_INVALID_SENTENCE = (
    "This phone pairing request is not the exact shape this build can check and describe, "
    "so nothing was approved and nothing was installed. Ask the phone to start pairing "
    "again."
)


def _refuse(code: str, message: str) -> MeshRefusal:
    return MeshRefusal(code, message)


def payload_invalid() -> MeshRefusal:
    return _refuse("pair_payload_invalid", PAYLOAD_INVALID_SENTENCE)


# ---------------------------------------------------------------------------
# The salt the phone shows (the first lock)
# ---------------------------------------------------------------------------


def sas_commit(nonce: bytes) -> str:
    """The phone's commitment to its nonce, in the wire form the record carries.

    A COMMITMENT, not a copy: the phone picks ``np`` and sends ``sha256(np)`` BEFORE this
    computer picks ``nd``, so the phone cannot grind a pairing code against a desktop nonce
    it has already seen. That is exactly the property an ordinary ``lop pair`` code lacks.
    """
    return _b64(hashlib.sha256(nonce).digest())


def sas_code(
    *,
    np: bytes,
    nd: bytes,
    device_spki: bytes,
    operator_spki: bytes,
    approval_id: str,
) -> str:
    """The six digits both screens show: ``HKDF(np||nd||device||operator||id) % 10^6``.

    THE ONE derivation (design §4). The desktop derives it from the RECORD's nonces and
    the LOCAL anchor's public point -- never from a point the record chose -- so a filer
    cannot pick which operator key the code is checked against. The phone derives the same
    value from what it holds. Its strength is ~20 bits per card and that is stated rather
    than implied: what makes it usable is everything around it (a window of minutes, three
    strikes, a card that shows what is granted, and a signature behind it).
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return _sas_from(
        HKDF(algorithm=hashes.SHA256(), length=8, salt=None, info=SAS_LABEL).derive(
            np + nd + device_spki + operator_spki + approval_id.encode("utf-8")
        )
    )


def _sas_from(okm: bytes) -> str:
    return f"{int.from_bytes(okm, 'big') % 1_000_000:06d}"


# ---------------------------------------------------------------------------
# The record's shape
# ---------------------------------------------------------------------------


def intent(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The parsed ``pair_phone`` payload, or ``None`` when this record carries none."""
    what = record.get("what")
    return what.get(OP) if isinstance(what, dict) and isinstance(what.get(OP), dict) else None


def exact_shape(record: Mapping[str, Any]) -> bool:
    """True only for the EXACT record this build mints, at EVERY depth the digest binds.

    ``request_digest`` covers ``what``, the ``device`` block, ``requested_by``,
    ``credential_ref`` and the window, so a key set checked only at the top level would let
    a nested extra -- in the anchor, the ``sas`` block, the ``via`` route -- be signed
    unseen (the hole #3087's ``exact_shape`` closed for finite shares). Checked here:

    * ``what`` is exactly the op and its anchor; the anchor is exactly
      ``approvals.ANCHOR_KEYS``;
    * the payload is exactly :data:`_FIELDS`; ``role``/``platform``/``via`` name a value
      this build can state; ``scopes`` is exactly ``devices.DEVICE_SCOPES``;
    * ``sas`` is exactly the three wire values, each a 32-byte nonce (or the commitment),
      and ``commit`` really is ``sha256(np)``;
    * ``computer_id_pub`` is a P-256 point;
    * ``device`` repeats the payload's key: the id IS ``new_device_id(spki)`` (derived,
      never chosen -- an allow-list keyed on a claimed id would not survive F-A), the
      fingerprint is the short display form of the same point, and the name is already its
      own display form;
    * ``requested_by`` names the SAME device that would be paired;
    * ``credential_ref`` is ``None``.

    Values that are DISPLAYED (name, platform, via) are checked by :func:`disclosure`; this
    predicate only decides which records are structurally describable at all.
    """
    from local_operator.operator import devices
    from local_operator.operator.verify import decode_point, spki_fp

    what = record.get("what")
    data = intent(record)
    if data is None or not isinstance(what, dict) or set(what) != _WHAT or set(data) != _FIELDS:
        return False
    anchor = what["anchor"]
    if not isinstance(anchor, dict) or set(anchor) != approvals.ANCHOR_KEYS:
        return False
    if data["role"] not in _ROLES:
        return False
    if data["scopes"] != list(devices.DEVICE_SCOPES):
        return False
    lifetime = data["cert_lifetime_s"]
    if not isinstance(lifetime, int) or isinstance(lifetime, bool):
        return False
    if not 0 < lifetime <= MAX_CERT_LIFETIME_S:
        return False
    if not _valid_via(data["via"]):
        return False
    if _valid_sas(data["sas"]) is None:
        return False
    computer = devices.decode_spki(data["computer_id_pub"])
    if computer is None or decode_point(computer) is None:
        return False
    device = record.get("device")
    if not isinstance(device, dict) or set(device) != _DEVICE_KEYS:
        return False
    spki = devices.decode_spki(device["spki"])
    if spki is None:
        return False
    if device["device_id"] != devices.new_device_id(spki):
        return False
    if device["key_fp"] != spki_fp(spki):
        # The SHORT display form of the phone's own point -- what the card shows a person
        # to compare with the phone's screen. Never the computer's point: that one rides
        # ``computer_id_pub`` and is disclosed as its own value.
        return False
    if device["platform"] not in _PLATFORMS:
        return False
    if not _is_plain(device["name"]):
        return False
    asked_by = record.get("requested_by")
    if (
        not isinstance(asked_by, dict)
        or set(asked_by) != _REQUESTED_BY
        or asked_by.get("device_id") != device["device_id"]
        or not isinstance(asked_by.get("session_id"), str)
        or not isinstance(asked_by.get("surface"), str)
    ):
        return False
    if record.get("credential_ref") is not None:
        return False
    if str(record.get("kind") or "") != approvals.KIND_OPERATOR_DEVICE_PAIR:
        return False
    return True


def _valid_via(value: Any) -> bool:
    """``{route, account}`` or ``{route}`` -- the two shapes v1 can state, nothing else."""
    if not isinstance(value, dict) or value.get("route") not in _VIA_ROUTES:
        return False
    if value["route"] == "custom":
        return set(value) == {"route"}
    account = value.get("account")
    return (
        set(value) == {"route", "account"}
        and isinstance(account, str)
        and 0 < len(account) <= 64
        and account.isprintable()
    )


def _valid_sas(value: Any) -> tuple[bytes, bytes] | None:
    """``(np, nd)`` from the record's ``sas`` block, or ``None`` if it is not the exact shape.

    The commitment is verified here rather than in the derivation: a record whose reveal
    does not open its commitment is not a record this build wrote, and refusing it at shape
    time means no surface can be asked to show or apply it.
    """
    if not isinstance(value, dict) or set(value) != _SAS_KEYS:
        return None
    np = _unb64(value["np"])
    nd = _unb64(value["nd"])
    commit = _unb64(value["commit"])
    if np is None or nd is None or commit is None:
        return None
    if len(np) != NONCE_BYTES or len(nd) != NONCE_BYTES:
        return None
    if hashlib.sha256(np).digest() != commit:
        return None
    return np, nd


def _is_plain(value: Any) -> bool:
    """True when ``value`` is a str :func:`_plain` would show byte for byte."""
    return isinstance(value, str) and value == _plain(value)


def _plain(text: Any, limit: int = 64) -> str:
    """A display-only label: no control characters, bounded. Never used as an identifier."""
    clean = "".join(ch for ch in str(text or "") if ord(ch) >= 32 and not 127 <= ord(ch) <= 159)
    return clean[:limit]


def _b64(raw: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: Any) -> bytes | None:
    import base64
    import binascii

    if not isinstance(text, str) or not text or len(text) > 512:
        return None
    try:
        return base64.urlsafe_b64decode((text + "=" * (-len(text) % 4)).encode("ascii"))
    except (binascii.Error, ValueError):
        return None


# ---------------------------------------------------------------------------
# What the person is shown
# ---------------------------------------------------------------------------


def disclosure(record: dict[str, Any]) -> dict[str, Any]:
    """Every fact the signer must see BEFORE signing, as DATA, from the signed record alone.

    Same contract as the credential ops' disclosure (one source for the desktop card and
    the CLI's lines), with the design's §4 fact list. ``via_words``/``grants`` are the
    product words the card shows; the machine tokens beside them (``via.route``,
    ``scopes``) are what a localized renderer keys on. Assurance is deliberately NOT here:
    it is a property of the host, re-checked at sign time and carried on the row and in the
    audit event, so a record does not claim a level the machine does not have.
    """
    data = intent(record)
    if data is None or not exact_shape(record):
        raise payload_invalid()
    from local_operator.operator.verify import spki_fp

    device = record["device"]
    computer = _unb64(data["computer_id_pub"]) or b""
    anchor = record["what"]["anchor"]
    lifetime_days = max(1, round(int(data["cert_lifetime_s"]) / 86400))
    try:
        return {
            "version": DISCLOSURE_VERSION,
            "operation": OP,
            "approval_id": str(record.get("approval_id") or ""),
            "op": OP,
            "phone_name": _plain(device["name"]),
            "device_id": str(device["device_id"]),
            "platform": str(device["platform"]),
            "key_fingerprint": str(device["key_fp"]),
            "computer_key_fingerprint": spki_fp(computer),
            "signing_principal": str(anchor["key_id"]),
            "via": dict(data["via"]),
            "via_words": _via_words(data["via"]),
            "scopes": list(data["scopes"]),
            "grants": [_grant_words(scope) for scope in data["scopes"]],
            "lifetime_days": lifetime_days,
            "requested_from": _surface_words(str(record["requested_by"]["surface"])),
            "request_window": {
                "created_at": record.get("created_at"),
                "expires_at": record.get("expires_at"),
            },
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise payload_invalid() from exc


def _via_words(via: Mapping[str, Any]) -> str:
    if via.get("route") == "radient":
        return "through your Radient account"
    return "through your own tunnel address"


def _grant_words(scope: str) -> str:
    """The words for one device scope. Both are authority-INCREASING, which the card says."""
    return {
        "approve": "approve requests on this computer",
        "loosen": "loosen limits on this computer",
    }.get(scope, scope)


def _surface_words(surface: str) -> str:
    return {
        "phone": "your phone",
        "radient-relay": "your phone, through your Radient account",
        "relay": "your phone, through this computer's relay",
    }.get(surface, "another surface on this computer")


def expected(record: Mapping[str, Any]) -> dict[str, Any]:
    """The apply ``expected`` block, derived from the SIGNED record (the card, not a command).

    A signed card is the only thing that can authorise an install here, so the block is a
    projection of the record rather than a second set of instructions: apply compares it
    against the record it re-read and refuses a caller that assembled its own.
    """
    data = intent(record)
    if data is None:
        raise payload_invalid()
    from local_operator.operator import devices
    from local_operator.operator.verify import key_id_for

    device = record.get("device") if isinstance(record.get("device"), Mapping) else {}
    computer = devices.decode_spki(data["computer_id_pub"])
    return {
        "device_id": str(device.get("device_id") or ""),
        "spki": str(device.get("spki") or ""),
        "key_fp": str(device.get("key_fp") or ""),
        "name": str(device.get("name") or ""),
        "platform": str(device.get("platform") or ""),
        "cert_lifetime_s": int(data["cert_lifetime_s"]),
        "scopes": list(data["scopes"]),
        "computer_key_id": key_id_for(computer) if computer is not None else "",
    }


# ---------------------------------------------------------------------------
# Filing: the guards create_request runs (approvals.create_request)
# ---------------------------------------------------------------------------


def require_filing_shape(record: Mapping[str, Any], root: Path | None = None) -> None:
    """Refuse a pairing record at FILING that no surface could ever sign or apply.

    ``create_request`` is not authority-increasing, but a record it wrote is a card a person
    will be asked about, and filing one this build cannot describe would spend their
    attention on a refusal. Two refusals: the shape (:func:`exact_shape`) and a REVOKED key
    -- a revoked device cannot re-pair by approval, because the id it would install is
    derived from the key that was revoked; bringing it back stays
    ``lop operator devices --authorise``.
    """
    from local_operator.operator import devices

    if not exact_shape(record):
        raise payload_invalid()
    device = record["device"]
    device_id = str(device["device_id"])
    if devices.is_revoked(_config_root(root), device_id):
        raise _refuse(
            "pair_device_revoked",
            "This phone's key was revoked on this computer, so it cannot be paired again by "
            "request. Nothing was filed; the operator can authorise the device again on the "
            "computer first.",
        )


def require_acked(record: Mapping[str, Any]) -> None:
    """The per-kind decision guard: no approval of this kind without a typed code first.

    WHY THIS IS IN THE STORE AND NOT ONLY IN THE CARD. ``approval_ops`` already refuses this
    kind on both generic approvers (``cli_approvable=False``), so the terminal and the
    bodiless route cannot reach a signature. This is the second lock for the same rule: a
    caller that reaches ``approvals.approve`` some other way -- a future route, a script, a
    surface that grows an approve button -- still cannot sign a pairing card whose six
    digits nobody typed. A receipt is MUTABLE and outside ``request_digest``, so this is
    discipline, not a boundary; on a file-only key the boundary is absent by the operator's
    own decision (a-2a8b), and on a presence key it is the OS prompt.
    """
    if not _ack_ok(record):
        raise _refuse(
            "pair_code_required",
            "A phone pairing is approved by typing the code the phone shows, so the computer "
            "can check it is the same phone. Nothing was approved; open the pairing card and "
            "type the code.",
        )


def _ack_ok(record: Mapping[str, Any]) -> bool:
    for receipt in record.get("receipts") or []:
        if isinstance(receipt, Mapping) and receipt.get("step") == STEP_ACK and receipt.get("ok"):
            return True
    return False


def _strikes(record: Mapping[str, Any]) -> int:
    return sum(
        1
        for receipt in record.get("receipts") or []
        if isinstance(receipt, Mapping)
        and receipt.get("step") == STEP_ACK
        and receipt.get("ok") is False
    )


# ---------------------------------------------------------------------------
# The typed code (before the signer)
# ---------------------------------------------------------------------------


def check_ack(approval_id: str, *, code: str, root: Path | None = None) -> None:
    """Check the six digits the person typed, BEFORE any signer runs. This writes a receipt.

    THE ORDER MATTERS: a wrong code must never spend a gesture, and the receipt must land
    whether the code matched or not, because the strike count lives in the receipts (they
    are mutable and outside ``request_digest``, which is what lets an attempt be recorded
    without changing what was signed). Three wrong codes close the card as an ordinary
    DENY -- the safe direction, and the answer that stops an unlimited oracle over a
    ~20-bit value.

    A transcription that is not six digits is refused WITHOUT a strike: a paste the
    normalization cannot read is not a guess at the value, and burning a strike for it
    would close a card because someone's clipboard had a space in it.

    The deny is written AFTER this function's record lock is dropped: ``approvals.deny``
    takes the same cross-process flock on a second descriptor, which this process would
    wait on forever (the reason ``finite_share._release_run`` runs where it does).
    """
    from local_operator.network.wire import normalize_sas

    typed = normalize_sas(str(code or ""))
    if not typed:
        raise _refuse(
            "pair_code_invalid",
            "That is not six digits. Type the six digits the phone shows, or deny this "
            "request; nothing was approved.",
        )
    with approvals._record_lock(approval_id, root):
        record = approvals._load_raw(approval_id, root)
        if str(record.get("kind") or "") != approvals.KIND_OPERATOR_DEVICE_PAIR:
            raise _refuse(
                "pair_operation_mismatch",
                "This card is not a phone pairing request. Nothing was approved.",
            )
        _require_checkable(record)
        matched = _code_matches(record, typed, root)
        strikes = _strikes(record) + (0 if matched else 1)
        record.setdefault("receipts", []).append(
            {
                "run_id": "",
                "step": STEP_ACK,
                "at": time.time(),
                "ok": matched,
                "detail": "" if matched else "pair_code_mismatch",
                "digest": "",
            }
        )
        _write(record, root)
    _audit(
        "device_pair.desktop_ack",
        approval_id=approval_id,
        root=root,
        detail={
            "surface": "desktop",
            "ok": matched,
            "strikes": strikes,
            "device_id": str((record.get("device") or {}).get("device_id") or ""),
        },
    )
    if matched:
        return
    if strikes >= MAX_STRIKES:
        _deny_after_strikes(approval_id, root)
        raise _refuse(
            "pair_code_mismatch",
            "The code did not match three times, so this pairing request is now closed. "
            "Nothing was installed; ask the phone to start pairing again.",
        )
    raise _refuse(
        "pair_code_mismatch",
        f"Those digits are not the code the phone is showing. Nothing was approved; check "
        f"the phone's screen and type it again ({MAX_STRIKES - strikes} "
        f"{'try' if MAX_STRIKES - strikes == 1 else 'tries'} left).",
    )


def _require_checkable(record: Mapping[str, Any]) -> None:
    """Refuse a card in a state where a code check would mean nothing."""
    state = str(record.get("state") or "")
    expires_at = record.get("expires_at")
    if state == approvals.STATE_EXPIRED or (
        isinstance(expires_at, (int, float)) and expires_at and time.time() >= float(expires_at)
    ):
        raise _refuse(
            "approval_expired",
            "This pairing request's window has passed, so there is nothing to confirm. Ask "
            "the phone to start pairing again.",
        )
    if state != approvals.STATE_REQUESTED:
        raise _refuse(
            "approval_decision_conflict",
            f"This card is {state!r}, not waiting for a decision. Nothing was approved.",
        )
    if approvals.request_digest(record) != str(record.get("request_digest") or ""):
        raise _refuse(
            "approval_record_tampered",
            "This pairing request no longer matches the digest it was filed with. Nothing was "
            "approved; ask the phone to start pairing again.",
        )


def _code_matches(record: Mapping[str, Any], typed: str, root: Path | None) -> bool:
    """Derive the six digits from THIS record and compare them in constant time."""
    data = intent(record)
    device = record.get("device") if isinstance(record.get("device"), Mapping) else {}
    if data is None or not exact_shape(record):
        raise payload_invalid()
    trio = approvals.local_anchor_trio(_config_root(root))
    if trio is None:
        raise _refuse(
            "approval_anchor_unavailable",
            "This computer has no operator key to check a pairing code against; ask Local "
            "Operator to set up operator authority here first. Nothing was approved.",
        )
    from local_operator.operator import devices

    spki = devices.decode_spki(device.get("spki"))
    nonces = _valid_sas(data["sas"])
    if spki is None or nonces is None:  # pragma: no cover - exact_shape already refused
        raise payload_invalid()
    np_value, nd_value = nonces
    expected_code = sas_code(
        np=np_value,
        nd=nd_value,
        device_spki=spki,
        operator_spki=trio["spki"],
        approval_id=str(record.get("approval_id") or ""),
    )
    return hmac.compare_digest(expected_code, typed)


def _deny_after_strikes(approval_id: str, root: Path | None) -> None:
    """Write the ordinary deny the third strike earns. A card already settled is left alone."""
    try:
        approvals.deny(approval_id, root=root)
    except MeshRefusal:
        # A concurrent deny/withdraw/expiry won the write-once race; the record is in the
        # state that race chose, which is still "this card cannot be approved".
        pass


# ---------------------------------------------------------------------------
# Apply: the second signature, the install, the connect
# ---------------------------------------------------------------------------


def apply(approval_id: str, expected_block: dict[str, Any], root: Path) -> dict[str, Any]:
    """Issue the certificate for a signed pairing card, install it, and connect the record.

    THE ORDER, and why each step is where it is:

    1. the signed record is re-read and its exact shape re-checked -- apply never accepts a
       record the person was not shown;
    2. the operator key is compared on ``key_id``/``spki_fp`` ONLY. The statement digest
       moves whenever ANY anchor write lands (it covers the device lists), so a revoke that
       landed while the person was reading the card would otherwise refuse a pairing with
       ``approval_anchor_mismatch`` -- and the S2 admit helper writes the anchor INSIDE this
       step in the next slice. The key itself is what the certificate is signed with and
       what the decision verified against, so the check that matters stays;
    3. admission is confirmed through the privileged writer BEFORE the signer, so a certificate
       is never issued for a phone the root-owned anchor does not list;
    4. the run lease is taken (``begin_run``) and re-verified (``verify_for_run``), so a
       deny that landed mid-run stops here and the certificate is never issued;
    5. the certificate is signed (the SECOND prompt on a presence host), written to the
       device store, and the pending pairing request is dropped;
    6. the record moves to ``connected`` with a receipt;
    7. every failure after the lease releases the run as a retry-eligible ``failed`` record,
       so a cancelled prompt does not wedge the card in ``connecting`` for its window.

    The privileged write precedes certificate issuance because a first-time phone has no
    certificate for ``devices.is_admitted`` to parse yet. ``admit_device`` validates the
    pending pairing request, refuses revocation lifts through ``install_anchor_from``, and
    requires the in-person admission purpose rather than a stored administrator password.
    """
    return _apply(approval_id, expected_block, root)


def _apply(
    approval_id: str, expected_block: dict[str, Any], root: Path | None
) -> dict[str, Any]:
    from local_operator.operator import devices

    config_root = _config_root(root)
    record = approvals.load_record(approval_id, root=config_root)
    if str(record.get("state") or "") == approvals.STATE_CONNECTED:
        raise _refuse(
            "approval_already_connected",
            "This pairing has already finished, so nothing else was installed. Pair the phone "
            "again if it needs a new certificate.",
        )
    data = intent(record)
    if data is None or not exact_shape(record):
        raise payload_invalid()
    device = record["device"]
    device_id = str(device["device_id"])
    spki = devices.decode_spki(device["spki"])
    if spki is None or devices.new_device_id(spki) != device_id:
        # F-A: a certificate's ``device_id`` is not bound to its key by the verifier, so the
        # ID this record names is DERIVED here rather than trusted.
        raise _refuse(
            "pair_subject_mismatch",
            "This pairing request's device id does not belong to the key it carries, so it was "
            "not installed. Ask the phone to start pairing again.",
        )
    for field_name in ("device_id", "spki", "key_fp", "name", "platform"):
        if expected_block.get(field_name) != device.get(field_name):
            raise _refuse(
                "pair_subject_mismatch",
                "What this pairing request would install is not what was signed, so nothing was "
                "installed. Ask the phone to start pairing again.",
            )
    if expected_block.get("cert_lifetime_s") != int(data["cert_lifetime_s"]):
        raise _refuse(
            "pair_subject_mismatch",
            "What this pairing request would install is not what was signed, so nothing was "
            "installed. Ask the phone to start pairing again.",
        )
    trio = approvals.local_anchor_trio(config_root)
    if trio is None:
        raise _refuse(
            "approval_anchor_unavailable",
            "This computer has no operator key to issue a certificate with; ask Local Operator "
            "to set up operator authority here first. Nothing was installed.",
        )
    anchor = record["what"]["anchor"]
    mismatched = [
        field_name
        for field_name in ("key_id", "spki_fp")
        if str(anchor.get(field_name) or "") != str(trio[field_name])
    ]
    if mismatched:
        from local_operator.operator.trust import TRIO_LABELS

        labels = ", ".join(TRIO_LABELS.get(name, name) for name in mismatched)
        raise _refuse(
            "approval_anchor_mismatch",
            "This pairing request was filed for an operator key that does not match the one "
            f"this computer holds ({labels}) — ask the phone to start pairing again. Nothing "
            "was installed.",
        )
    computer_spki = devices.decode_spki(data["computer_id_pub"])
    if computer_spki is None or devices.new_device_id(computer_spki) != str(trio["key_id"]):
        # The phone derives the same six digits from ``computer_id_pub``; a record naming a
        # different point is one whose code the phone and this computer would never agree
        # on, and whose certificate would then be signed by a key the phone did not expect.
        raise _refuse(
            "pair_subject_mismatch",
            "This pairing request names a different computer key than the one installed here, "
            "so nothing was installed. Ask the phone to start pairing again.",
        )
    from local_operator.operator.handlers import admit_device

    admitted = admit_device(config_root, device_id)
    if admitted != 0:
        raise _refuse(
            "pair_device_not_admitted",
            "This phone was not confirmed on this computer, so no certificate was issued. "
            "Confirm the phone here and try the pairing again; nothing was installed.",
        )
    run_id = approvals.new_run_id()
    approvals.begin_run(approval_id, run_id=run_id, runner_pid=os.getpid(), root=config_root)
    try:
        approvals.verify_for_run(approval_id, root=config_root)
        certificate = _issue_certificate(
            device_spki=spki,
            device_id=device_id,
            label=_plain(device["name"]),
            lifetime_s=int(data["cert_lifetime_s"]),
            root=config_root,
        )
        from local_operator.operator.verify import read_device_cert

        parsed = read_device_cert(certificate)
        if parsed is None:  # pragma: no cover - issue_device_cert builds this exact format
            raise _refuse(
                "pair_cert_invalid",
                "The certificate this computer produced could not be read back, so nothing was "
                "installed. Try again.",
            )
        devices.write_device_cert(
            config_root,
            certificate=certificate,
            parsed=parsed,
            operator_key_id=str(trio["key_id"]),
            name=_plain(device["name"]),
            scope=tuple(devices.DEVICE_SCOPES),
        )
        # The relay's pairing request is settled by the install: leaving it would offer the
        # same phone a second, unchecked install path (``lop pair``'s watcher).
        devices.drop_pending(config_root, device_id)
        approvals.mark_connected(
            approval_id,
            run_id=run_id,
            step=STEP_INSTALL,
            detail="the certificate is installed on this computer",
            root=config_root,
        )
    except Exception as exc:
        _release_run(
            approval_id,
            run_id,
            getattr(exc, "code", type(exc).__name__),
            config_root,
        )
        raise
    _audit(
        "device_pair.desktop_applied",
        approval_id=approval_id,
        root=config_root,
        detail={
            "surface": "desktop",
            "device_id": device_id,
            "run_id": run_id,
            "lifetime_days": max(1, round(int(data["cert_lifetime_s"]) / 86400)),
        },
    )
    return {
        "approval_id": approval_id,
        "device_id": device_id,
        "key_fp": str(device["key_fp"]),
        "state": approvals.STATE_CONNECTED,
        "run_id": run_id,
    }


def _issue_certificate(
    *, device_spki: bytes, device_id: str, label: str, lifetime_s: int, root: Path
) -> str:
    """The operator's SECOND signature for this card: the device certificate itself.

    The same signer resolution ``approvals.sign_decision`` uses (the anchor names the
    backend), and the same open/close discipline, because a pairing that signed with a
    different key than the one the card named would install a certificate the runtime
    refuses at the next frame.
    """
    from local_operator.operator.keychain import KeyBackendError
    from local_operator.operator.sign import (
        issue_device_cert,
        load_signer,
        resolve_backend_name,
    )

    backend = resolve_backend_name(root)
    signer = load_signer(config_root=root, backend_name=backend)
    if signer is None:
        raise _refuse(
            "approval_signing_unavailable",
            "The operator key on this computer could not be loaded, so the certificate was not "
            "issued. Nothing was installed; check that operator authority is set up here and "
            "try again.",
        )
    try:
        return issue_device_cert(
            device_spki=device_spki,
            device_id=device_id,
            label=label,
            signer=signer,
            lifetime_s=lifetime_s,
        )
    except KeyBackendError as exc:
        raise _refuse(
            "approval_signing_unavailable",
            "The certificate was not signed, so nothing was installed. Try again, or set up "
            "operator authority on this computer if it is not set up yet.",
        ) from exc
    finally:
        signer.close()


def _release_run(approval_id: str, run_id: str, detail: str, root: Path | None) -> None:
    """``connecting → failed`` for THIS run only, so the card stays retryable.

    Mirrors ``finite_share._release_run``: a refusal after ``begin_run`` otherwise leaves the
    record ``connecting`` under a live pid, and ``run_is_stale`` never fires in the daemon,
    so every retry reads "run in flight" until the window closes. A record another run
    superseded, or one already terminal, is that writer's.
    """
    try:
        record = approvals.load_record(approval_id, root=root)
        raw_run = record.get("run")
        run: Mapping[str, Any] = raw_run if isinstance(raw_run, Mapping) else {}
        if str(run.get("run_id") or "") != run_id:
            return
        approvals.mark_failed(
            approval_id, run_id=run_id, step=STEP_INSTALL, detail=detail, root=root
        )
    except MeshRefusal:
        pass


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def _bounded_subject(approval_id: str) -> str:
    """The id as an audit subject: control characters dropped, length bounded.

    The audit writer bounds its DETAIL values but writes ``subject`` verbatim; every subject
    this module writes comes through here (the same rule ``desktop_approval`` follows).
    """
    cleaned = "".join(ch for ch in approval_id if ch.isprintable())
    return cleaned[:64]


def _audit(
    event: str,
    *,
    approval_id: str,
    root: Path | None,
    detail: Mapping[str, Any],
) -> None:
    approvals._audit(
        event,
        actor="self",
        subject=_bounded_subject(approval_id),
        root=root,
        detail=dict(detail),
    )


def audit_signed(record: Mapping[str, Any], approval_id: str, root: Path | None) -> None:
    """The decision's own row: WHICH card was signed, and whether a person was proven.

    Beside the store's generic ``onboard_approved`` because a pairing decision is the one
    approval that issues a standing signer, so an incident review asks of it in particular:
    which device, under which operator key, and at what assurance -- ``presence`` (an OS
    prompt per signature) or ``file_only`` (a 0600 file signed; no person is proven).
    """
    detail: dict[str, Any] = {
        "surface": "desktop",
        "operation": OP,
        "device_id": str((record.get("device") or {}).get("device_id") or ""),
        "key_id": (
            str((record.get("signature") or {}).get("key_id") or "")
            if isinstance(record.get("signature"), Mapping)
            else ""
        ),
    }
    try:
        from local_operator.network.credentials import desktop_approval

        detail["assurance"] = desktop_approval.assurance()[0] or ""
    except Exception:  # noqa: BLE001 - the row is worth writing without the level
        detail["assurance"] = ""
    _audit("device_pair.desktop_signed", approval_id=approval_id, root=root, detail=detail)


def audit_apply_refused(
    record: Mapping[str, Any] | None, approval_id: str, code: str, root: Path | None
) -> None:
    """The apply's refusal row, written for a card that exists (never for a stray id)."""
    if record is None:
        return
    device = record.get("device")
    _audit(
        "device_pair.desktop_apply_refused",
        approval_id=approval_id,
        root=root,
        detail={
            "surface": "desktop",
            "device_id": str((device or {}).get("device_id") or "")
            if isinstance(device, Mapping)
            else "",
            "code": code,
        },
    )


# ---------------------------------------------------------------------------
# The CLI's human half (``approvals list``/``show`` route through this)
# ---------------------------------------------------------------------------


def card_lines(record: Mapping[str, Any]) -> list[str]:
    """The card a terminal reader gets, in the same order the desktop sheet shows it.

    WHY IT CANNOT FALL THROUGH TO THE ONBOARDING CARD: the fallback calls this an anchor
    install and lists "definitions, MCP servers, git identity and logins", which is a
    different action entirely. Nothing here names the approve command: this kind is refused
    on the terminal, so a command would be an instruction that cannot work.
    """
    facts = disclosure(dict(record))
    lines = [
        f"A phone is asking to be paired with this computer: {facts['phone_name']} "
        f"({facts['platform']}), key {facts['key_fingerprint']}.",
        f"It would be able to {facts['grants'][0]}"
        + (f", and to {facts['grants'][1]}" if len(facts["grants"]) > 1 else "")
        + ".",
        f"It reaches this computer {facts['via_words']}, and its certificate would last "
        f"{facts['lifetime_days']} days.",
        f"Requested from {facts['requested_from']}; signed by key {facts['signing_principal']}.",
        "Type the six digits the phone shows on this computer's approval sheet to finish. "
        "The terminal and the TUI cannot approve a pairing.",
        "Anything that runs programs as you on this computer can work out those digits, so "
        "the sheet warns when this computer cannot prove a person approved.",
    ]
    return lines


# ---------------------------------------------------------------------------
# Registration (the typed-approval registry, ``network/approval_ops.py``)
# ---------------------------------------------------------------------------


class _PairOp:
    """``pair_phone`` as the registry's :class:`approval_ops.ApprovalOp`.

    ``file_only_signing = True`` is the operator's decision, not a default (a-2a8b: "Allow,
    with the warning and a 'Set up Touch ID' offer"), which is why the row carries
    ``unproven_human`` and the sheet offers the upgrade. ``cli_approvable = False`` is the
    first lock: the six digits have to be typed, and neither the terminal verb nor the TUI
    can show the code check, so both refuse this kind (deny stays available).
    """

    op = OP
    kind = approvals.KIND_OPERATOR_DEVICE_PAIR
    file_only_signing = True
    cli_approvable = False

    def intent(self, record: dict[str, Any]) -> dict[str, Any] | None:
        return intent(record)

    def disclosure(self, record: dict[str, Any]) -> dict[str, Any]:
        return disclosure(record)

    def expected(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return expected(record)

    def apply(self, approval_id: str, expected: dict[str, Any], root: Path) -> Any:
        try:
            return apply(approval_id, expected, root)
        except MeshRefusal as exc:
            audit_apply_refused(_safe_record(approval_id, root), approval_id, exc.code, root)
            raise

    def generic_refusal(self, record: Mapping[str, Any], approval_id: str) -> MeshRefusal | None:
        """Both generic approvers refuse this kind, in the same words.

        Key PRESENCE is the boundary, so a malformed payload answers here too rather than
        falling through: a card this build cannot describe must never be signable by a
        button labelled Approve.
        """
        if intent(record) is None or not exact_shape(record):
            return payload_invalid()
        return _refuse(
            "pair_typed_approval_required",
            "This is a phone pairing request: it is approved on this computer by typing the "
            "six digits the phone shows, so the computer can check it is the same phone. "
            "Nothing was approved here; open the pairing card on this computer, or deny this "
            "request.",
        )


def _safe_record(approval_id: str, root: Path | None) -> dict[str, Any] | None:
    try:
        return approvals.load_record(approval_id, root=_config_root(root))
    except MeshRefusal:
        return None


def _config_root(root: Path | None) -> Path:
    from local_operator.paths import config_dir

    return Path(root) if root is not None else config_dir()


def _write(record: Mapping[str, Any], root: Path | None) -> Path:
    return approvals._write_record(record, root)


approval_ops.register(_PairOp())


#: ``what`` key prefixes this module owns. ``pair_`` is registered as a FAMILY so an
#: unknown pairing operation beside the shape this build knows (``pair_phone_v2`` and the
#: like) is refused as undescribable instead of being signed blind -- the same
#: fail-closed rule the ``credential_`` family follows.
_PAIR_PREFIX = "pair_"


def _pair_unsupported(record: Mapping[str, Any], approval_id: str) -> MeshRefusal:
    return _refuse(
        "approval_invalid",
        "This pairing request uses an operation this build cannot check and describe, so it "
        "cannot be approved here. Nothing was approved.",
    )


approval_ops.register_family(_PAIR_PREFIX, _pair_unsupported)
