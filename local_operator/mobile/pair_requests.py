"""Phone pairing REQUEST filing. This module deliberately has no approval signer.

A phone proves control of a fresh key later, at use time; filing only publishes its
public point and a SAS ceremony. The desktop-signed typed engine is the sole path
that can turn the resulting record into a device certificate. Keeping this module
free of signer imports makes the phone-to-desktop authority boundary inspectable.
"""

from __future__ import annotations

import asyncio
import base64
import secrets
import time
from pathlib import Path
from typing import Any, Mapping

from local_operator.network import approvals
from local_operator.network.types import MeshRefusal

#: A bounded daemon-wide budget, not a per-phone budget: before a device is paired
#: its key is not an identity worth quota. Five attempts per ten minutes allows a
#: mistyped/restarted ceremony without turning basic relay access into a card flood.
FILING_LIMIT = 5
FILING_WINDOW_S = 600.0
CLAIM_TTL_S = 300.0
MAX_CLAIMS = 64
CERT_LIFETIME_S = 90 * 24 * 3600

_CLAIMS: dict[str, dict[str, Any]] = {}
# All filing routes in this process share one lock. In particular the live-record
# check and create_request cannot be interleaved by two phones claiming together.
_FILING_LOCK = asyncio.Lock()


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _refuse(code: str, message: str) -> MeshRefusal:
    return MeshRefusal(code, message)


def _claim(body: Mapping[str, Any], *, now: float) -> dict[str, Any]:
    if set(body) != {"spki", "commit"}:
        raise _refuse(
            "pair_claim_invalid",
            "A pairing claim needs exactly the phone's public key and SAS commitment.",
        )
    from local_operator.operator import devices, pair_op

    spki = devices.decode_spki(body["spki"])
    commit = body["commit"]
    if spki is None or not isinstance(commit, str) or not commit or len(commit) > 64:
        raise pair_op.payload_invalid()
    # Expired claims are never revived. Repeating the same commitment and key inside
    # its short window is idempotent, so a reconnect does not change the desktop nonce.
    for key in [key for key, row in _CLAIMS.items() if float(row["expires_at"]) <= now]:
        _CLAIMS.pop(key, None)
    claim = _CLAIMS.get(commit)
    if claim is not None and claim["spki"] != body["spki"]:
        raise _refuse(
            "pair_claim_conflict",
            "That SAS commitment was already claimed with a different phone key.",
        )
    if claim is None:
        if len(_CLAIMS) >= MAX_CLAIMS:
            raise _refuse(
                "pair_claim_busy",
                "Too many pairing claims are waiting on this computer. Try again shortly.",
            )
        claim = {
            "spki": body["spki"],
            "nd": _b64(secrets.token_bytes(32)),
            "expires_at": now + CLAIM_TTL_S,
        }
        _CLAIMS[commit] = claim
    trio = approvals.local_anchor_trio()
    if trio is None:
        raise _refuse(
            "approval_anchor_unavailable",
            "This computer has no operator key for phone pairing. Set up operator authority first.",
        )
    return {
        "nd": claim["nd"],
        "computer_id_pub": devices.encode_spki(trio["spki"]),
        "anchor": {key: trio[key] for key in approvals.ANCHOR_KEYS},
        "expires_at": claim["expires_at"],
    }


#: The gateway-owned header carrying the Radient edge's verified assertion for THIS
#: filing (pairing design v2 §1 item 1). The gateway sets it only on
#: ``POST /api/pair/request`` to the local-operator relay, from the proof it has just
#: verified; an inbound copy from a phone never crosses (``gateway.headers`` copies an
#: allowlist that does not name it). The daemon RE-VERIFIES it rather than trusting the
#: hop, so a forged or replayed value can only fall back to the custom route.
ASSERTION_HEADER = "x-lop-radient-assertion"
#: The exact request the assertion must authorise: the reveal post itself.
FILING_TARGET = "/api/pair/request"
_RELAY_HARNESS_ID = "local-operator"


async def attested_route(
    assertion: str | None, *, host: str, body: bytes
) -> tuple[dict[str, str], str]:
    """``(via, surface)`` for one filing, from the edge's assertion or nothing.

    Success re-runs the gateway's own verifier (``OriginVerifier``) against the pinned
    ``origin_auth`` in ``tunnels/config.json``: RS256 under a pinned key, ``iss``,
    ``aud`` = the public host, ``sub`` = the configured owner account, ``tunnel_id``,
    ``version``, harness, method ``POST``, target :data:`FILING_TARGET`,
    ``body_sha256`` of THESE bytes, a lifetime of at most thirty seconds, and ``exp``.
    The ``jti`` was already consumed by the gateway, so a fresh verifier instance is
    used and its replay cache is irrelevant here. Then the route is
    ``{route: radient, account, attested: edge-v1}`` and the card may be confirmed.

    ANY failure -- no header, no tunnel config, a forged or stale token, a body that
    does not hash to the claim -- answers ``{route: custom}``, never a refusal: the
    filing still lands and the desktop simply asks for the typed code (design v2 §1's
    table). The assertion itself is never stored (``pair_op``'s "never in this record").
    """
    if assertion:
        try:
            from local_operator.tunnels import config
            from local_operator.tunnels.gateway import OriginVerifier

            connection = config.load()
            access = connection["origin_auth"]
            claims = await OriginVerifier(access, None).verify(  # type: ignore[arg-type]
                assertion,
                host=host.lower(),
                harness_id=_RELAY_HARNESS_ID,
                method="POST",
                target=FILING_TARGET,
                body=body,
            )
            account = claims.get("sub")
            if isinstance(account, str) and account and account == access.get("owner_account_id"):
                from local_operator.operator import pair_op

                return (
                    {"route": "radient", "account": account, "attested": pair_op.ATTESTED_EDGE_V1},
                    "radient-relay",
                )
        except Exception:  # noqa: BLE001 - every verification failure is "not attested"
            pass
    return {"route": "custom"}, "phone"


def _existing_retry(
    request_id: str, spki: str, np: str, commit: str, root: Path
) -> dict[str, Any] | None:
    existing = approvals._find_by_request_id(request_id, root)
    if existing is None or existing.get("kind") != approvals.KIND_OPERATOR_DEVICE_PAIR:
        return None
    data = existing.get("what", {}).get("pair_phone", {})
    sas = data.get("sas", {}) if isinstance(data, Mapping) else {}
    device = existing.get("device", {})
    if (
        isinstance(device, Mapping)
        and device.get("spki") == spki
        and sas.get("np") == np
        and sas.get("commit") == commit
    ):
        return existing
    return None


class PairRefusal(MeshRefusal):
    """A filing refusal with keyless extras the phone keys on (design v2 §2.2, N2).

    ``extras`` are additive JSON fields beside ``{error, code}``: ``same_key`` (and,
    only then, the pending ``request_id``) on ``pair_request_pending``, and
    ``retry_after_s`` on ``pair_filing_rate_limited``. Nothing in them names another
    phone's request.
    """

    def __init__(self, code: str, sentence: str, **extras: Any) -> None:
        super().__init__(code, sentence)
        self.extras = extras


def _budget(root: Path, now: float, *, spki: str = "") -> None:
    records = approvals.list_records(root=root, now=now)
    live = [
        row
        for row in records
        if row.get("kind") == approvals.KIND_OPERATOR_DEVICE_PAIR
        and row.get("state") not in approvals.TERMINAL_STATES
        and row.get("state") != approvals.STATE_FAILED
    ]
    if live:
        # U6: tell the phone whether the pending request is ITS OWN (same key), so it can
        # resume or withdraw it; another phone's request id is never disclosed.
        pending = live[0]
        device = pending.get("device")
        same_key = bool(spki) and isinstance(device, Mapping) and device.get("spki") == spki
        extras: dict[str, Any] = {"same_key": same_key}
        if same_key:
            extras["request_id"] = str(pending.get("request_id") or "")
        raise PairRefusal(
            "pair_request_pending",
            "A phone pairing request is already waiting on this computer. "
            "Finish or withdraw it first.",
            **extras,
        )
    recent = sorted(
        float(row.get("created_at") or 0.0)
        for row in records
        if row.get("kind") == approvals.KIND_OPERATOR_DEVICE_PAIR
        and 0 <= now - float(row.get("created_at") or 0.0) < FILING_WINDOW_S
    )
    if len(recent) >= FILING_LIMIT:
        # When the OLDEST filing in the window ages out, one slot frees (N2: the phone
        # shows m:ss). Derived from the existing window, not a new number.
        retry_after = max(1, int(recent[-FILING_LIMIT] + FILING_WINDOW_S - now + 0.999))
        raise PairRefusal(
            "pair_filing_rate_limited",
            "This computer accepted five pairing requests in ten minutes. "
            "Wait before filing another.",
            retry_after_s=retry_after,
        )


async def file_request(
    body: Mapping[str, Any],
    root: Path,
    *,
    route: tuple[dict[str, str], str] | None = None,
) -> dict[str, Any]:
    """Claim a SAS nonce, or reveal a commitment and file ONE typed approval record."""
    if "np" not in body:
        return _claim(body, now=time.time())
    if set(body) != {"request_id", "spki", "commit", "np", "name", "platform"}:
        raise _refuse(
            "pair_filing_invalid",
            "A pairing filing needs exactly its request id, phone key, SAS reveal, "
            "name and platform.",
        )
    from local_operator.operator import devices, pair_op
    from local_operator.operator.verify import spki_fp

    spki = devices.decode_spki(body["spki"])
    if spki is None:
        raise pair_op.payload_invalid()
    request_id, commit, np = body["request_id"], body["commit"], body["np"]
    if not all(isinstance(value, str) for value in (request_id, commit, np)):
        raise pair_op.payload_invalid()
    # A retry with an existing record does not need the in-memory claim: daemon
    # restarts and reconnects must not consume budget or change the filed request.
    async with _FILING_LOCK:
        now = time.time()
        existing = _existing_retry(request_id, body["spki"], np, commit, root)
        claim = _CLAIMS.get(commit) if existing is None else None
        if existing is None:
            if claim is None or claim["spki"] != body["spki"] or float(claim["expires_at"]) <= now:
                raise _refuse(
                    "pair_claim_expired",
                    "Claim a fresh SAS nonce before revealing this pairing.",
                )
            nd = claim["nd"]
        else:
            nd = existing["what"][pair_op.OP]["sas"]["nd"]
        try:
            revealed = base64.b64decode(np + "=" * (-len(np) % 4), altchars=b"-_", validate=True)
        except (ValueError, UnicodeEncodeError) as exc:
            raise pair_op.payload_invalid() from exc
        if pair_op.sas_commit(revealed) != commit:
            raise pair_op.payload_invalid()
        trio = approvals.local_anchor_trio(root)
        if trio is None:
            raise _refuse(
                "approval_anchor_unavailable",
                "This computer has no operator key for phone pairing.",
            )
        # A retry keeps the route its record was filed with: the record is digest-bound,
        # and a phone that retries over a different network (attested the first time,
        # not the second) must get its own request back, not a conflict.
        if existing is not None:
            via = dict(existing["what"][pair_op.OP]["via"])
            surface = str(existing["requested_by"]["surface"])
        else:
            via, surface = route if route is not None else ({"route": "custom"}, "phone")
        device_id = devices.new_device_id(spki)
        request = {
            "kind": approvals.KIND_OPERATOR_DEVICE_PAIR,
            "request_id": request_id,
            "requested_by": {"surface": surface, "device_id": device_id, "session_id": ""},
            "device": {
                "device_id": device_id,
                "spki": devices.encode_spki(spki),
                "key_fp": spki_fp(spki),
                "name": body["name"],
                "platform": body["platform"],
            },
            "what": {
                pair_op.OP: {
                    "role": "approver",
                    "scopes": list(devices.DEVICE_SCOPES),
                    "cert_lifetime_s": CERT_LIFETIME_S,
                    "via": via,
                    "sas": {"commit": commit, "nd": nd, "np": np},
                    "computer_id_pub": devices.encode_spki(trio["spki"]),
                },
                "anchor": {key: trio[key] for key in approvals.ANCHOR_KEYS},
            },
            "credential_ref": None,
            "created_at": now,
            "expires_at": now + CLAIM_TTL_S,
        }
        if existing is None:
            _budget(root, now, spki=body["spki"])
        # THE ONLY WRITE THIS ROUTE CAN MAKE. create_request supplies the envelope,
        # immutable digest and pair_op.require_filing_shape (which re-derives the
        # device id/fingerprint and validates every nested SAS/anchor/route key).
        record = approvals.create_request(**request, root=root)
        _CLAIMS.pop(commit, None)
        return record


# ---------------------------------------------------------------------------
# The phone's other three questions (design v2 §2.2). None of them decides anything.
# ---------------------------------------------------------------------------

#: The domain label of the phone's withdraw signature. Its own label, so a signature
#: made for any other purpose (an operator proof, a future pairing step) can never be
#: replayed as a withdrawal -- and a replayed withdrawal names its own request id.
WITHDRAW_LABEL = b"lop-pair-withdraw-v1"


def availability() -> dict[str, Any]:
    """``{available, reason}``: can this computer issue a phone certificate at all?

    ``no_operator_authority`` when no usable operator anchor is installed (the same
    host predicate the relay and the pairing receipt use), ``unsupported`` when this
    build does not advertise ``features.phone_pairing``. The phone offers pairing only
    on ``ok``, so no claim is made against a computer that could never finish it.
    """
    try:
        from local_operator.server.features import static_feature_flags

        supported = bool(static_feature_flags().get("phone_pairing"))
    except Exception:  # noqa: BLE001 - an unreadable feature map is "not supported"
        supported = False
    if not supported:
        return {"available": False, "reason": "unsupported"}
    from local_operator.operator import operator_authority_unusable

    if operator_authority_unusable():
        return {"available": False, "reason": "no_operator_authority"}
    return {"available": True, "reason": "ok"}


def _pair_record(request_id: str, root: Path) -> dict[str, Any] | None:
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
        return None
    record = approvals._find_by_request_id(request_id, root)
    if record is None or record.get("kind") != approvals.KIND_OPERATOR_DEVICE_PAIR:
        return None
    return record


def request_status(request_id: str, root: Path) -> dict[str, Any] | None:
    """``{state, reason, mode, device_id, expires_at}`` for the phone's own request."""
    from local_operator.operator import pair_op

    record = _pair_record(request_id, root)
    if record is None:
        return None
    now = time.time()
    view = approvals.presented(record, now)
    device = view.get("device")
    return {
        "state": str(view.get("state") or ""),
        "reason": pair_op.progress(record, now),
        "mode": pair_op.mode(record),
        "device_id": str(device.get("device_id") or "") if isinstance(device, Mapping) else "",
        "expires_at": view.get("expires_at"),
    }


def withdraw_request(request_id: str, body: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """The phone's Cancel: verify its key's signature, then withdraw as the requester.

    THE KEY IS THE REQUESTER (U6). The relay cookie is shared by every phone, so it
    cannot say which phone filed a card; the record's ``device.spki`` can. The body is
    exactly ``{sig}``: hex DER ECDSA-P256-SHA256 over ``WITHDRAW_LABEL || request_id``,
    verified against the record's own key BEFORE anything is written. Then the store's
    write-once ``withdraw`` runs with the record's own identity keys.

    Idempotent: ``withdrawn`` or ``expired`` answers 200 with that state. Once the
    person has decided (code matched or codes confirmed, a signature or a run under
    way) it is ``approval_in_progress`` -- the phone stays on "Confirming" -- and a
    wrong key is ``pair_withdraw_forbidden`` with nothing written.
    """
    from local_operator.operator import devices, pair_op
    from local_operator.operator.verify import verify_signature

    record = _pair_record(request_id, root)
    if record is None:
        raise _refuse("unknown_request", pair_op._copy("refusal_unknown_request"))
    if set(body) != {"sig"} or not isinstance(body.get("sig"), str) or len(body["sig"]) > 512:
        raise _refuse("pair_withdraw_invalid", pair_op._copy("refusal_withdraw_invalid"))
    try:
        signature = bytes.fromhex(body["sig"])
    except ValueError:
        signature = b""
    device = record.get("device")
    spki = devices.decode_spki(device.get("spki")) if isinstance(device, Mapping) else None
    message = WITHDRAW_LABEL + request_id.encode("utf-8")
    if spki is None or not verify_signature(spki=spki, message=message, signature=signature):
        raise _refuse("pair_withdraw_forbidden", pair_op._copy("refusal_withdraw_forbidden"))
    view = approvals.presented(record, time.time())
    state = str(view.get("state") or "")
    if state in (approvals.STATE_WITHDRAWN, approvals.STATE_EXPIRED):
        return {"state": state}
    if state != approvals.STATE_REQUESTED or pair_op.progress(record) != pair_op.REASON_WAITING:
        raise _refuse("approval_in_progress", pair_op._copy("refusal_in_progress"))
    try:
        settled = approvals.withdraw(
            str(record["approval_id"]), requested_by=dict(record["requested_by"]), root=root
        )
    except MeshRefusal as exc:
        if exc.code == "approval_expired":
            return {"state": approvals.STATE_EXPIRED}
        if exc.code == "approval_withdraw_conflict":
            # A decision landed between our read and the store's lock: same answer as above.
            raise _refuse("approval_in_progress", pair_op._copy("refusal_in_progress")) from exc
        raise
    return {"state": str(settled.get("state") or approvals.STATE_WITHDRAWN)}


def _clear_for_tests() -> None:
    _CLAIMS.clear()
