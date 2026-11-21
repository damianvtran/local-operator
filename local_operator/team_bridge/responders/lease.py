"""The ``lease_epoch`` claim client (RFC §7.3 A3/S6; agent-server #451).

One alert root has at most one claim row on the hub. The first responder to
``POST …/claim`` gets ``201`` and a monotonic ``lease_epoch``; every other caller
gets ``409 already_claimed`` and stands down having spent zero triage tokens. The
holder renews every 60 s (the hub's lease is 3 minutes) carrying that epoch; ANY
mismatch — a human preempted, the sweeper handed the alert to someone else after
this laptop slept — answers ``409 lease_fenced`` and the run must abort at once.

The hub re-reads the autonomy kill switch (``autonomy_disabled``) and the
responder registration (``responder_not_enabled``) UNCACHED on claim and on every
heartbeat, so the heartbeat is also how an organization-wide "switch responders
off" reaches a run that is already going: the next renew is refused and the run
is cancelled. Release is never refused by a kill switch on the hub (stopping must
always be possible), and a release that answers ``lease_fenced`` means the claim
is already someone else's — nothing to undo.

LOCAL EXPIRY. If renewals stop succeeding (hub unreachable, 5xx), this side cannot
prove it still holds the lease. Once the last confirmed ``lease_expires_at`` has
passed the lease is treated as LOST and the run aborts: the sweeper will have
handed the alert on, and two responders working one alert is the failure the
lease exists to prevent.
"""

from __future__ import annotations

import datetime as _dt
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

__all__ = [
    "HEARTBEAT_INTERVAL_S",
    "ClaimResult",
    "Lease",
    "LeaseClient",
    "LeaseLost",
    "claim",
    "heartbeat",
    "release",
    "sanitize_status_line",
]

#: RFC §7.3 / agent-server ``IntegrationClaimLease``: a 3-minute lease renewed every
#: 60 s, so two consecutive renewals may fail transiently before the lease lapses.
HEARTBEAT_INTERVAL_S = 60.0

#: ``already_claimed`` / ``alert_resolved``: someone else has it, or it is over.
#: Standing down is the correct, silent outcome — never an error to surface.
_STAND_DOWN = frozenset({"already_claimed", "alert_resolved"})

#: Codes on claim/heartbeat that mean "this endpoint may not work this alert now".
#: Grouped only for the caller's outcome label; every one of them stops the run.
_REFUSED = frozenset(
    {
        "autonomy_disabled",
        "responder_not_enabled",
        "not_conversation_member",
        "not_endpoint_owner",
        "conversation_message_not_found",
        "conversation_not_found",
        "lease_fenced",
    }
)

#: §9.3: the public status line is one line of ≤140 plain-text runes, no links.
_STATUS_LINE_MAX = 140


class LeaseClient(Protocol):
    """The four ``TeamBridgeClient`` methods this module drives (a fake in tests)."""

    async def alert_claim(
        self, conversation_id: str, root_message_id: str, *, tenant_id: str, actor_endpoint_id: str
    ) -> dict[str, Any]: ...

    async def alert_claim_heartbeat(
        self,
        conversation_id: str,
        root_message_id: str,
        *,
        tenant_id: str,
        actor_endpoint_id: str,
        lease_epoch: int,
        status_line: str | None = None,
        waiting_approval_for: Mapping[str, str] | None = None,
    ) -> dict[str, Any]: ...

    async def alert_claim_release(
        self,
        conversation_id: str,
        root_message_id: str,
        *,
        tenant_id: str,
        actor_endpoint_id: str,
        lease_epoch: int,
    ) -> None: ...


@dataclass(frozen=True)
class Lease:
    """One held claim. ``epoch`` is the fencing token every renew/release carries."""

    tenant_id: str
    conversation_id: str
    root_message_id: str
    endpoint_id: str
    epoch: int
    expires_at: _dt.datetime


@dataclass(frozen=True)
class ClaimResult:
    """``outcome`` is ``claimed`` | ``stand_down`` | ``refused`` | ``unreachable``."""

    outcome: str
    code: str
    lease: Lease | None = None


class LeaseLost(Exception):
    """The run no longer holds the alert; ``code`` says why (hub code or ``local_expiry``)."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _refusal_code(exc: BaseException) -> str | None:
    """The hub's machine code for a ``BridgeRefusal``; ``None`` for anything else."""

    from local_operator.team_bridge.client import BridgeRefusal

    if isinstance(exc, BridgeRefusal):
        return str(exc.code or "bridge_error")
    return None


def _parse_expiry(value: Any) -> _dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _lease_from(result: Mapping[str, Any], base: Lease | None, **ids: str) -> Lease | None:
    epoch = result.get("lease_epoch")
    expires = _parse_expiry(result.get("lease_expires_at"))
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0 or expires is None:
        return None
    if base is not None:
        return Lease(
            tenant_id=base.tenant_id,
            conversation_id=base.conversation_id,
            root_message_id=base.root_message_id,
            endpoint_id=base.endpoint_id,
            epoch=epoch,
            expires_at=expires,
        )
    return Lease(epoch=epoch, expires_at=expires, **ids)


async def claim(
    client: LeaseClient,
    *,
    tenant_id: str,
    conversation_id: str,
    root_message_id: str,
    endpoint_id: str,
) -> ClaimResult:
    """Claim the alert root as this agent endpoint. Never raises for a hub answer."""

    from local_operator.team_bridge.client import BridgeUnreachable

    try:
        result = await client.alert_claim(
            conversation_id, root_message_id, tenant_id=tenant_id, actor_endpoint_id=endpoint_id
        )
    except BridgeUnreachable:
        return ClaimResult("unreachable", "bridge_unreachable")
    except Exception as exc:  # noqa: BLE001 — only coded refusals are answers; re-raise the rest
        code = _refusal_code(exc)
        if code is None:
            raise
        if code in _STAND_DOWN:
            return ClaimResult("stand_down", code)
        # Every other coded refusal (including a dark hub's uncoded 404,
        # ``bridge_error``) is a refusal: fail closed, zero tokens.
        return ClaimResult("refused", code)
    lease = _lease_from(
        result,
        None,
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        root_message_id=root_message_id,
        endpoint_id=endpoint_id,
    )
    if lease is None or str(result.get("holder_endpoint_id") or "") != endpoint_id:
        # A 2xx we cannot read as OUR lease is not a lease: release nothing we
        # cannot name, and do no work.
        return ClaimResult("refused", "invalid_claim_response")
    return ClaimResult("claimed", "claimed", lease)


async def heartbeat(
    client: LeaseClient,
    lease: Lease,
    *,
    now: _dt.datetime,
    status_line: str | None = None,
    waiting_approval_for: Mapping[str, str] | None = None,
) -> Lease:
    """Renew at ``lease.epoch``; return the renewed lease or raise :class:`LeaseLost`.

    A transient failure (unreachable, 5xx) keeps the CURRENT lease while it is still
    unexpired locally; past ``expires_at`` it is lost (module docstring).
    """

    from local_operator.team_bridge.client import BridgeUnreachable

    try:
        result = await client.alert_claim_heartbeat(
            lease.conversation_id,
            lease.root_message_id,
            tenant_id=lease.tenant_id,
            actor_endpoint_id=lease.endpoint_id,
            lease_epoch=lease.epoch,
            status_line=status_line,
            waiting_approval_for=waiting_approval_for,
        )
    except BridgeUnreachable:
        return _keep_or_lose(lease, now)
    except Exception as exc:  # noqa: BLE001 — classify coded refusals; re-raise the rest
        code = _refusal_code(exc)
        if code is None:
            raise
        if code in _REFUSED or code in _STAND_DOWN:
            raise LeaseLost(code) from None
        http_status = getattr(exc, "http_status", None)
        if isinstance(http_status, int) and http_status >= 500:
            return _keep_or_lose(lease, now)
        # Any other 4xx is a refusal we do not understand: stop rather than guess.
        raise LeaseLost(code) from None
    renewed = _lease_from(result, lease)
    if renewed is None or renewed.epoch != lease.epoch:
        # The hub renews IN PLACE at the same epoch; a different one means the row
        # is not the one we hold.
        raise LeaseLost("lease_fenced")
    return renewed


def _keep_or_lose(lease: Lease, now: _dt.datetime) -> Lease:
    if now >= lease.expires_at:
        raise LeaseLost("local_expiry")
    return lease


async def release(client: LeaseClient, lease: Lease) -> str:
    """Release at ``lease.epoch``. Returns ``released`` | ``already_gone`` | ``failed:<code>``.

    Best effort by design: a release that fails leaves the hub's 3-minute lease to
    lapse and the sweeper to hand off, which is the same end state, later.
    """

    try:
        await client.alert_claim_release(
            lease.conversation_id,
            lease.root_message_id,
            tenant_id=lease.tenant_id,
            actor_endpoint_id=lease.endpoint_id,
            lease_epoch=lease.epoch,
        )
    except Exception as exc:  # noqa: BLE001 — release must never raise into the run's teardown
        code = _refusal_code(exc)
        if code is None:
            return "failed:exception"
        if code == "lease_fenced":
            return "already_gone"
        return f"failed:{code}"
    return "released"


def sanitize_status_line(text: str) -> str | None:
    """Shape a run's status into §9.3's public status line, or ``None`` if nothing is left.

    One line, ≤140 runes, no links or markdown, and passed through the credential
    shape scrubber (S-37): this line is rendered on the alert card for every member
    of the channel, so it is outbound text in the strictest sense. The hub rejects
    a line that breaks its shape; trimming here keeps a renew from being refused for
    formatting.
    """

    from local_operator.redaction_shapes import scrub_shapes

    first = " ".join(str(text or "").split())
    first = scrub_shapes(first)
    # Links and markup are the hub's refusal class; drop rather than render them.
    # Mirrors agent-server ``validStatusLine`` exactly (case-insensitive ``://`` and
    # ``www.`` anywhere, control runes, and ``<>[]`*_#|``) so a renew is never refused.
    words = [w for w in first.split(" ") if "://" not in w.lower() and "www." not in w.lower()]
    line = "".join(
        ch for ch in " ".join(words) if ch not in "<>[]`*_#|" and unicodedata.category(ch) != "Cc"
    )
    line = " ".join(line.split())
    if not line:
        return None
    if len(line) > _STATUS_LINE_MAX:
        line = line[: _STATUS_LINE_MAX - 1].rstrip() + "…"
    return line
