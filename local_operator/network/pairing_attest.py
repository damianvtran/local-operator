"""App attestation for answering a mesh pairing from the desktop's Mesh approvals.

WHAT THIS PROTECTS, AND WHAT IT DOES NOT. The desktop API is reachable by anything
that holds the desktop bearer — including an agent the operator is running. Without
this, an agent with that bearer could POST the six digits it already knows (the code
is not a secret: it is in the park JSON and the pending files) and complete a pairing
the person never saw. With this, ``POST /v1/desktop/mesh/pairings/{side}/{invite_id}
/answer`` acts only after the SIGNED, unmodified desktop app's main process attested
that exact answer over a kernel-authenticated local socket (#4132's
``asks.app_attest`` verifier: audit token + Security.framework designated
requirement; nothing the dialler says counts).

It is DESKTOP-API PROTECTION ONLY — not relay-level proof that a human answered:
- the relay and the parked joiner cannot tell an attested decision from one written
  by the same-user terminal fallback (``lop network confirm``, ``lop network join
  --confirm``), which stays available by design;
- the same uid can write the 0600 decision/answer files directly.
So the honest claim is: "an agent holding only the desktop bearer cannot answer a
pairing through the app". Closing the CLI/file paths would need the deciders
themselves to verify (design §3.4-B, deferred) and app-required invites (out of scope).

WHAT AN ATTESTATION BINDS (security review 60334a0b4c12, A4). The table key is
``(SCOPE, "<side>:<invite_id>:<nonce>", act)`` and the content is ``{"code": [code],
"shares": sorted(shares), "deadline": [<int seconds>]}``, where an inviter admit
MUST carry an explicit ``shares`` list (``[]`` shares nothing; absent is refused on the
frame, on the HTTP body and by the relay — S-R1-1) and every other answer carries none:
- ``side`` + ``invite_id``: which card;
- ``nonce``: the per-ceremony, per-stage tag on the card row
  (``types.new_pairing_nonce``) — a retry of the invite, or the inviter's stage-1 →
  stage-2 flip, mints a new one, so an attestation can never answer another ceremony
  and an expired ceremony's attestation can never be spent on its successor;
- ``act`` (``respond`` | ``decline``) and the exact answer;
- ``deadline``: wall-clock seconds the app chose, which the route requires to be in
  the future and no later than the ceremony's own ``expires_at``.
Single use (``AttestationTable.consume``), and consumed BEFORE any code comparison, so
a submission with no attestation spends no attempt and writes no decision.

MACOS ONLY. ``app_attest.supported()`` is false elsewhere; there is no listener, every
card row says ``can_answer_here: false`` / ``answer_unavailable: "unsupported_host"``
and the route answers 403 ``pairing_needs_app``. There is deliberately no downgrade to
a bearer-only or "unverified" answer — that would be exactly the hole this closes.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

from local_operator.asks import app_attest

logger = logging.getLogger(__name__)

#: The attestation table scope (first key element) for pairing answers.
SCOPE = "mesh-pairing"
#: The listener's name: ``<config>/run/attest/mesh-pairing.sock`` — the app's dial
#: contract for this feature. One listener per ``lop serve``, for both sides.
LISTENER_NAME = "mesh-pairing"
#: The frame op the app sends on that socket.
ATTEST_OP = "mesh_pairing_attest"
SIDES = frozenset({"inviter", "joiner"})
#: ``respond`` = admit with the typed code; ``decline`` = refuse (no code compared).
ACTS = frozenset({"respond", "decline"})
#: Why ``answer_unavailable`` is set on a card row; ``""`` when this host can answer.
UNSUPPORTED_HOST = "unsupported_host"
NO_APP_ATTESTATION = "no_app_attestation"

_CODE = re.compile(r"^[0-9]{6}$")
_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def act_for(decision: str) -> str:
    """The attestation act for a route ``decision`` (``admit`` | ``decline``)."""
    return "respond" if decision == "admit" else "decline"


def target(side: str, invite_id: str, nonce: str) -> str:
    return f"{side}:{invite_id}:{nonce}"


def content(code: str, shares: list[str] | None, deadline: int) -> dict[str, list[str]]:
    """The exact answer content both the app's frame and the route's consume bind.

    ``shares`` is sorted (order carries no meaning) and ``deadline`` is an integer of
    seconds so a float round-trip through JSON cannot make two honest sides disagree.
    A decline binds ``code: [""]`` — nothing is compared for it.
    """
    return {
        "code": [str(code or "")],
        "shares": sorted(str(key) for key in (shares or [])),
        "deadline": [str(int(deadline))],
    }


def shares_shape_ok(side: str, act: str, shares: Any) -> bool:
    """Whether ``shares`` has the one shape this answer may carry (S-R1-1).

    An INVITER's admit (``respond``) must carry an explicit list — possibly empty,
    which shares nothing — because that list is exactly what the relay will serve:
    ``content`` binds ``[]`` for both "empty" and "absent", so if absence were allowed
    here an attestation of "share nothing" could be spent by a body that omitted the
    key and fell through to a default. Every other answer (a decline, any joiner
    answer) shares nothing, so it carries no list or an empty one. The route applies
    this same function to the body, so the app's frame and the HTTP body agree.
    """
    if side == "inviter" and act == "respond":
        return isinstance(shares, list) and all(isinstance(key, str) for key in shares)
    return shares is None or shares == []


def validate_frame(frame: dict[str, Any]) -> app_attest.AttestedRow | str:
    """``app_attest.FrameValidator`` for ``{"op": "mesh_pairing_attest", ...}``.

    Frame: ``{op, side, invite_id, nonce, act, code, shares, deadline}``. Shape only
    (pure, no I/O): whether the ceremony is live and the deadline fits it is the
    route's check at consume time, against the row it read.
    """
    if frame.get("op") != ATTEST_OP:
        return "bad-op"
    side, invite_id = str(frame.get("side") or ""), str(frame.get("invite_id") or "")
    nonce, act = str(frame.get("nonce") or ""), str(frame.get("act") or "")
    if side not in SIDES or act not in ACTS or not _ID.match(invite_id) or not _ID.match(nonce):
        return "bad-frame"
    code = frame.get("code")
    if act == "respond":
        if not (isinstance(code, str) and _CODE.match(code)):
            return "bad-frame"
    elif code not in (None, ""):
        return "bad-frame"
    shares = frame.get("shares")
    if not shares_shape_ok(side, act, shares):
        return "bad-frame"
    shares = list(shares or [])
    deadline = frame.get("deadline")
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
        return "bad-frame"
    if int(deadline) <= time.time():
        return "expired-deadline"
    return app_attest.AttestedRow(
        SCOPE,
        target(side, invite_id, nonce),
        act,
        content(str(code or ""), shares, int(deadline)),
    )


def make_service(**kwargs: Any) -> app_attest.AttestService:
    """The pairing listener: #4132's service with this module's validator.

    ``kwargs`` pass through (``verifier``, ``table``, ``token_of``) so tests inject a
    ``PeerVerifier`` exactly as ``app_attest``'s own tests do.
    """
    return app_attest.AttestService(LISTENER_NAME, validator=validate_frame, **kwargs)


async def start_service(config_dir: Path) -> app_attest.AttestService | None:
    """Bind ``<config>/run/attest/mesh-pairing.sock`` on macOS; ``None`` elsewhere.

    Additive and never a boot gate: any failure is logged and leaves no listener, and
    then every card answer is refused (``no_app_attestation``) — fail closed.
    """
    if not app_attest.supported():
        return None
    try:
        from local_operator.session.runtime.registry import run_dir

        service = make_service()
        attest_dir = run_dir(Path(config_dir), app_attest.ATTEST_RUN_DIRNAME)
        if await service.start(attest_dir) is not None:
            return service
    except Exception:  # noqa: BLE001 — additive, never a boot gate
        logger.warning("mesh pairing attest: listener not started", exc_info=True)
    return None


def answer_unavailable(service: app_attest.AttestService | None) -> str:
    """Why this host cannot answer a pairing card (``""`` when it can)."""
    if not app_attest.supported():
        return UNSUPPORTED_HOST
    if service is None:
        return NO_APP_ATTESTATION
    return ""


def consume(
    service: app_attest.AttestService | None,
    *,
    side: str,
    invite_id: str,
    nonce: str,
    decision: str,
    code: str,
    shares: list[str] | None,
    deadline: int,
) -> bool:
    """Spend the matching attestation, if any. A mismatch consumes nothing."""
    if service is None or not app_attest.supported() or not nonce:
        return False
    return service.table.consume(
        SCOPE,
        target(side, invite_id, nonce),
        act_for(decision),
        content(code if decision == "admit" else "", shares, deadline),
        [],
    )


def discard(service: app_attest.AttestService | None, *, side: str, invite_id: str) -> None:
    """Drop every pending attestation for a card whose ceremony is gone (expiry).

    The nonce binding already makes such a row unspendable; dropping it as well means
    an expired ceremony leaves nothing behind in the table at all.
    """
    if service is None:
        return
    service.table.discard(SCOPE, f"{side}:{invite_id}:")
