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


def _route() -> tuple[dict[str, str], str]:
    """The configured route is server evidence; an ``account`` body field never is.

    Radient's gateway authenticates the owner before injecting the ordinary relay
    cookie. The relay cannot distinguish that cookie from a custom-password cookie,
    so this is an ATTRIBUTION HINT, not an authentication claim or authority grant.
    """
    try:
        from local_operator.tunnels import config

        connection = config.load()
        account = connection.get("origin_auth", {}).get("owner_account_id")
        if isinstance(account, str) and account:
            return {"route": "radient", "account": account}, "radient-relay"
    except (OSError, ValueError, TypeError, KeyError):
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


def _budget(root: Path, now: float) -> None:
    records = approvals.list_records(root=root, now=now)
    live = [
        row for row in records
        if row.get("kind") == approvals.KIND_OPERATOR_DEVICE_PAIR
        and row.get("state") == approvals.STATE_REQUESTED
    ]
    if live:
        raise _refuse(
            "pair_request_pending",
            "A phone pairing request is already waiting on this computer. "
            "Finish or withdraw it first.",
        )
    recent = sum(
        1 for row in records
        if row.get("kind") == approvals.KIND_OPERATOR_DEVICE_PAIR
        and 0 <= now - float(row.get("created_at") or 0.0) < FILING_WINDOW_S
    )
    if recent >= FILING_LIMIT:
        raise _refuse(
            "pair_filing_rate_limited",
            "This computer accepted five pairing requests in ten minutes. "
            "Wait before filing another.",
        )


async def file_request(body: Mapping[str, Any], root: Path) -> dict[str, Any]:
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
        via, surface = _route()
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
            _budget(root, now)
        # THE ONLY WRITE THIS ROUTE CAN MAKE. create_request supplies the envelope,
        # immutable digest and pair_op.require_filing_shape (which re-derives the
        # device id/fingerprint and validates every nested SAS/anchor/route key).
        record = approvals.create_request(**request, root=root)
        _CLAIMS.pop(commit, None)
        return record


def _clear_for_tests() -> None:
    _CLAIMS.clear()
