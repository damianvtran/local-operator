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
prove it still holds the lease. Once the last confirmed lease has run out the lease
is treated as LOST and the run aborts: the sweeper will have handed the alert on,
and two responders working one alert is the failure the lease exists to prevent.
"Run out" is measured on THIS process's monotonic clock from the moment the
confirming request was SENT (review m5): comparing the hub's ``lease_expires_at``
with a laptop wall clock that runs behind would keep a partitioned run working
after the sweeper handed off. The remaining duration is the hub's lease
(:data:`LEASE_S`), shortened when the wall-clock reading says less is left —
whichever is SOONER, so neither clock error can lengthen it.

RESOLVED. ``alert_resolved`` on a heartbeat is not a loss to cancel on: the hub
woke this holder to write a closing summary (agent-server ``wakeForResolve``), it
refuses every further renew of a resolved root, and its release still succeeds.
The caller stops renewing and lets the run finish inside the lease it already
holds (``ResponderRunner``), then releases.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

__all__ = [
    "ALERT_RESOLVED",
    "HEARTBEAT_INTERVAL_S",
    "LEASE_S",
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
#: agent-server ``models.IntegrationClaimLease`` (3 minutes). The claim response
#: carries only the absolute ``lease_expires_at``, never a server "now", so this is
#: the only clock-free reading of how long a confirmed lease lasts.
LEASE_S = 180.0
#: The heartbeat refusal that means "over, write your closing summary".
ALERT_RESOLVED = "alert_resolved"

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
    #: ``time.monotonic()`` reading past which this side no longer holds the lease
    #: (module docstring, LOCAL EXPIRY). ``inf`` only for a lease built by hand.
    deadline: float = math.inf


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


def _epoch(result: Mapping[str, Any]) -> int | None:
    epoch = result.get("lease_epoch") if isinstance(result, Mapping) else None
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0:
        return None
    return epoch


def _lease_from(
    result: Mapping[str, Any],
    base: Lease | None,
    *,
    sent: float,
    now: _dt.datetime,
    **ids: str,
) -> Lease | None:
    epoch = _epoch(result)
    expires = _parse_expiry(result.get("lease_expires_at")) if isinstance(result, Mapping) else None
    if epoch is None or expires is None:
        return None
    # Whichever is sooner: the hub's lease length from when we SENT the request
    # (the hub stamped its expiry after that), or what the wall clock says is left.
    remaining = min(LEASE_S, max(0.0, (expires - now).total_seconds()))
    deadline = sent + remaining
    if base is not None:
        return Lease(
            tenant_id=base.tenant_id,
            conversation_id=base.conversation_id,
            root_message_id=base.root_message_id,
            endpoint_id=base.endpoint_id,
            epoch=epoch,
            expires_at=expires,
            deadline=deadline,
        )
    return Lease(epoch=epoch, expires_at=expires, deadline=deadline, **ids)


async def claim(
    client: LeaseClient,
    *,
    tenant_id: str,
    conversation_id: str,
    root_message_id: str,
    endpoint_id: str,
    now: _dt.datetime | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> ClaimResult:
    """Claim the alert root as this agent endpoint. Never raises for a hub answer."""

    from local_operator.team_bridge.client import BridgeUnreachable

    sent = monotonic()
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
    ids = dict(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        root_message_id=root_message_id,
        endpoint_id=endpoint_id,
    )
    lease = _lease_from(result, None, sent=sent, now=now or _dt.datetime.now(_dt.UTC), **ids)
    holder = str(result.get("holder_endpoint_id") or "") if isinstance(result, Mapping) else ""
    if lease is None or holder != endpoint_id:
        # A 2xx we cannot read as OUR lease is not a lease: do no work. If it
        # still names an epoch, the hub may have granted it — give it back now
        # rather than stall the alert for a whole lease before the sweeper
        # hands off (review S-7). A release at an epoch that is not ours is
        # fenced by the hub, so this can never take someone else's claim.
        epoch = _epoch(result)
        if epoch is not None:
            await release(
                client,
                Lease(
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    root_message_id=root_message_id,
                    endpoint_id=endpoint_id,
                    epoch=epoch,
                    expires_at=_dt.datetime.now(_dt.UTC),
                ),
            )
        return ClaimResult("refused", "invalid_claim_response")
    return ClaimResult("claimed", "claimed", lease)


async def heartbeat(
    client: LeaseClient,
    lease: Lease,
    *,
    now: _dt.datetime,
    status_line: str | None = None,
    waiting_approval_for: Mapping[str, str] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> Lease:
    """Renew at ``lease.epoch``; return the renewed lease or raise :class:`LeaseLost`.

    A transient failure (unreachable, 5xx) keeps the CURRENT lease while it is still
    unexpired locally; past ``lease.deadline`` it is lost (module docstring).
    ``LeaseLost("alert_resolved")`` is the caller's cue to stop renewing, not to
    cancel (module docstring, RESOLVED).
    """

    from local_operator.team_bridge.client import BridgeUnreachable

    sent = monotonic()
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
        return _keep_or_lose(lease, monotonic())
    except Exception as exc:  # noqa: BLE001 — classify coded refusals; re-raise the rest
        code = _refusal_code(exc)
        if code is None:
            raise
        if code in _REFUSED or code in _STAND_DOWN:
            raise LeaseLost(code) from None
        http_status = getattr(exc, "http_status", None)
        if isinstance(http_status, int) and http_status >= 500:
            return _keep_or_lose(lease, monotonic())
        # Any other 4xx is a refusal we do not understand: stop rather than guess.
        raise LeaseLost(code) from None
    renewed = _lease_from(result, lease, sent=sent, now=now)
    if renewed is None or renewed.epoch != lease.epoch:
        # The hub renews IN PLACE at the same epoch; a different one means the row
        # is not the one we hold.
        raise LeaseLost("lease_fenced")
    return renewed


def _keep_or_lose(lease: Lease, mono_now: float) -> Lease:
    if mono_now >= lease.deadline:
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


#: Link-shaped words are dropped by SHAPE, not by a list of URL forms (review
#: S-10 then S-13: every denylist round missed a shape — a bare subdomain carrying
#: data, ``1.2.3.4/x``, a trailing-dot FQDN, punycode and IDN TLDs, ideographic
#: full stops, ``\`` paths, ``mailto:``). The status line is plain prose on a card
#: every channel member sees, so a word survives only if it cannot be a host, a
#: path or a scheme:
#:
#: * no ``@`` ``/`` ``\`` anywhere;
#: * no ``:`` followed by more of the word (a scheme, ``host:port``, IPv6) — a
#:   trailing ``:`` (``status:``) and a clock time (``18:05``) are prose;
#: * no dot-like character between two label characters (``a.b``, ``a。b``,
#:   ``a｡b``…), checked AFTER NFKC so compatibility forms (``．`` ``․`` ``﹒``)
#:   fold to ``.`` first — unless the whole word is a short version or number
#:   (``v1.2.3``, ``99.5%``), which has at most three numeric parts, so a
#:   dotted-quad IP never qualifies.
_DOTS = frozenset(".\u3002\uff61\uff0e\u2024\ufe52")
_DOTTED_LABEL = re.compile(r"[\w-][" + "".join(sorted(_DOTS)) + r"][\w-]")
_SCHEMEISH = re.compile(r":(?=\S)")
_NUMBERISH = re.compile(r"[(]?v?\d+(?:\.\d+){0,2}%?[).,;:!?]*")
_CLOCK = re.compile(r"[(]?\d{1,2}(?::\d{2}){1,2}[).,;!?]*")


def _link_shaped(word: str) -> bool:
    if any(ch in word for ch in "@/\\"):
        return True
    if _SCHEMEISH.search(word) and not _CLOCK.fullmatch(word):
        return True
    return bool(_DOTTED_LABEL.search(word)) and not _NUMBERISH.fullmatch(word)


_MARKUP = frozenset("<>[]`*_#|")


def _valid_status_line(line: str) -> bool:
    """agent-server ``validStatusLine``, rune for rune (the hub's 400 floor)."""

    lower = line.lower()
    if len(line) > _STATUS_LINE_MAX or "://" in lower or "www." in lower:
        return False
    return not any(ch in _MARKUP or unicodedata.category(ch) == "Cc" for ch in line)


def sanitize_status_line(text: str) -> str | None:
    """Shape a run's status into §9.3's public status line, or ``None`` if nothing is left.

    One line, ≤140 runes, no links or markdown, and passed through the credential
    shape scrubber (S-37): this line is rendered on the alert card for every member
    of the channel, so it is outbound text in the strictest sense. The hub rejects
    a line that breaks its shape, and a refused renew is a lost lease, so the line
    is reshaped to a FIXPOINT and then checked against the hub's own rule: removing
    a character can CREATE a link (``w_ww.`` → ``www.``, review m3), so one pass of
    each step is not enough. Anything that still fails the check is not sent.
    """

    from local_operator.redaction_shapes import scrub_shapes

    # NFKC first so the shape test below judges the characters a renderer will
    # show (fullwidth/compatibility dots fold to ``.``), and send that same form.
    line = scrub_shapes(" ".join(unicodedata.normalize("NFKC", str(text or "")).split()))
    for _ in range(8):
        before = line
        # Characters first (the hub's markup and control runes), THEN links, so
        # a stripped character can never splice a link back together unseen.
        line = "".join(ch for ch in line if ch not in _MARKUP and unicodedata.category(ch) != "Cc")
        words = [w for w in line.split() if not _link_shaped(w) and "www." not in w.lower()]
        line = " ".join(words)
        if line == before:
            break
    if not line:
        return None
    if len(line) > _STATUS_LINE_MAX:
        line = line[: _STATUS_LINE_MAX - 1].rstrip() + "…"
    return line if _valid_status_line(line) else None
