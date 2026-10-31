"""Publish the chief of staff's picture to the hub — the one writer there is.

WHY THIS EXISTS. Team Bridge shows a person's agent to everybody in the
organization. The person's own machine holds the agent definition, and the
picture on it is the one they chose; the hub is a projection of it so that a
teammate's console can render the agent without reaching this machine. The
direction is fixed and single: LOCAL -> HUB. Nothing here writes local state,
and no hub answer can change what the local definition says.

WHERE IT RUNS. On the status boundary only — the enrollment write and the fresh
status read (a rename, a sign-in, an explicit save) that
``client.StatusDiscipline._sync_agent_avatar`` already runs beside the agent
name. Never on a timer, never from a cached answer, never while a message is
being sent: the hub's status answer already carries the digest it holds, so a
boundary is the only moment this module needs.

THE THREE ARMS, and why "nothing" is a real answer.

* the local file's digest differs from the hub's ``source_sha256`` -> PUT;
* they agree -> keep (a PUT is harmless, but pointless: the hub answers 204 for
  bytes it already holds);
* the agent definition exists and carries NO ``avatar`` block -> the local
  writer stores that block together with the file, so its absence IS the user's
  decision to have no picture -> DELETE, which is idempotent.

Everything else publishes NOTHING, never a delete. A block that is present but
whose file cannot be resolved (a hand-edited ``filename``, a traversal attempt,
a vanished or oversized file) is a DOUBT, and a transient filesystem fault must
never be able to wipe the copy a teammate is looking at. ``published_agent``
answers ``None`` for the same reason when there is no designated chief of staff,
no agent definition behind her, or no config root.

FAILURES NEVER ESCAPE :func:`reconcile`. Its call sites are the status read and
the enrollment write — surfaces answering a user's question about their own
organization — so a cosmetic field must never turn either into an error. Every
failure is caught there and reported as ONE log line carrying the exception
TYPE and the hub's own error ``code`` only: never the bytes, never a path,
never the agent's name. A log line is the easiest place for private data to
leak, and none of it is needed to know that the picture did not publish.

The explicit paths (:func:`publish_agent_avatar`, :func:`clear_agent_avatar`)
deliberately PROPAGATE refusals: an explicit save or a CLI invocation has a
caller who can read the answer, and swallowing it there would report success
for a picture that never left the machine.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from local_operator.team_bridge.cos_name import (
    SHA256_HEX,
    published_agent,
)

logger = logging.getLogger(__name__)

#: Actions a plan can carry. ``none`` = there is nothing to answer for (no
#: endpoint, no designated chief of staff, or an older hub without the field),
#: and it is the ONLY action a doubt may produce.
NONE = "none"
KEEP = "keep"
PUBLISH = "publish"
DELETE = "delete"


class AvatarClient(Protocol):
    """The client calls this module needs, and nothing else.

    A Protocol rather than the concrete client so the reconciler is exercisable
    against a fake with no auth store, no transport and no status discipline.
    """

    async def status(self, *, access: Any = None) -> dict[str, Any]: ...

    async def put_endpoint_avatar(
        self,
        endpoint_id: str,
        data: bytes,
        source_sha256: str,
        *,
        expected_account_id: str | None = None,
    ) -> dict[str, Any]: ...

    async def delete_endpoint_avatar(
        self, endpoint_id: str, *, expected_account_id: str | None = None
    ) -> dict[str, Any]: ...

    async def current_access(self) -> Any: ...


@dataclass(frozen=True)
class AvatarPlan:
    """What a reconcile would do, resolved WITHOUT doing it.

    Holds the bytes it would send so one plan is one file read: the digest and
    the body must describe the same read, or the hub would store bytes that do
    not match the digest it records for them.
    """

    action: str = NONE
    endpoint_id: str | None = None
    agent_id: str | None = None
    name: str | None = None
    #: The picture itself. ``repr=False`` because an exception, a traceback or a
    #: test dump would otherwise carry somebody's image bytes into a log — the
    #: one thing this module's doctrine says never escapes (review R1-S5).
    data: bytes | None = field(default=None, repr=False)
    digest: str | None = None
    #: What the hub holds, for the CLI's ``status`` answer only.
    hub_hash: str | None = None
    hub_sha256: str | None = None

    def replaced(self, action: str) -> AvatarPlan:
        """This plan with a forced action (the explicit publish/clear paths)."""

        return AvatarPlan(
            action=action,
            endpoint_id=self.endpoint_id,
            agent_id=self.agent_id,
            name=self.name,
            data=self.data,
            digest=self.digest,
            hub_hash=self.hub_hash,
            hub_sha256=self.hub_sha256,
        )


def _payload(status: Any) -> Mapping[str, Any] | None:
    """The served status payload, whether a ``StatusCheck`` or the body itself."""

    payload = getattr(status, "payload", status)
    return payload if isinstance(payload, Mapping) else None


def _endpoint(
    payload: Mapping[str, Any], tenant_id: str, endpoint_id: str | None = None
) -> tuple[str, Mapping[str, Any]] | None:
    """``(endpoint_id, endpoint_block)`` for THIS tenant — or ``None``.

    Two callers, one rule: AMBIGUITY IS A DOUBT.

    * ``endpoint_id`` given (the status boundary): that exact endpoint must BE
      the row's endpoint. The boundary's ownership guard validated that id, and
      re-deriving the address by tenant alone let a payload with two rows for
      one tenant write to an endpoint the guard never checked (review R1-S3);
    * ``endpoint_id`` omitted (the explicit CLI paths): the tenant must appear
      EXACTLY once. A duplicate row is a hub bug or a hostile hub, and this
      module's rule everywhere else is that a doubt publishes nothing.
    """

    tenants = payload.get("tenants")
    if not isinstance(tenants, list):
        return None
    rows = [
        row for row in tenants if isinstance(row, Mapping) and row.get("tenant_id") == tenant_id
    ]
    if len(rows) != 1:
        return None
    endpoint = rows[0].get("endpoint")
    if not isinstance(endpoint, Mapping):
        return None
    found_id = endpoint.get("id")
    if not isinstance(found_id, str) or not found_id:
        return None
    if endpoint_id is not None and found_id != endpoint_id:
        return None
    return found_id, endpoint


def _hub_avatar(endpoint: Mapping[str, Any]) -> tuple[bool, str | None, str | None]:
    """``(hub_holds_one, hub_hash, hub_source_sha256)`` from the endpoint block.

    "Holds one" is answered by the PRESENCE of the block, not by a digest that
    parses: the local side may DELETE a copy it cannot interpret (DELETE is
    idempotent), while only a parsable ``source_sha256`` may decide "no change
    needed" — comparing against a value we had to guess would re-PUT every
    boundary.
    """

    block = endpoint.get("avatar")
    if not isinstance(block, Mapping):
        return False, None, None
    digest = block.get("source_sha256")
    key = block.get("hash")
    return (
        True,
        key if isinstance(key, str) and re.fullmatch(r"[0-9a-f]{16}", key) else None,
        digest if isinstance(digest, str) and SHA256_HEX.fullmatch(digest) else None,
    )


def plan_avatar(
    status: Any,
    tenant_id: str,
    config_dir: Path | str | None,
    endpoint_id: str | None = None,
) -> AvatarPlan:
    """Resolve the one action this status answer calls for.

    ``config_dir`` is ``None`` on the clear path, where the local definition is
    deliberately not consulted: clearing the hub copy is the caller's decision
    and must work on a machine whose local files are unreachable.

    ``endpoint_id`` is the endpoint the caller's ownership guard VALIDATED. The
    status boundary always passes it, so the plan can only ever address the row
    that guard approved; the CLI omits it, and then a duplicated tenant row is
    refused rather than resolved by first match (review R1-S3).
    """

    payload = _payload(status)
    if payload is None:
        return AvatarPlan()
    found = _endpoint(payload, tenant_id, endpoint_id)
    if found is None:
        return AvatarPlan()
    endpoint_id, endpoint = found
    # ABSENT is the ordinary answer, not an old-hub signal: the contract's status
    # block carries ``avatar`` only once something is published, so absence means
    # "the hub holds nothing" and the local picture must be sent. (The
    # ``agent_name`` field is the opposite — always served, ``null`` when unset —
    # which is why the name re-sync may gate on its presence and this may not.)
    # Against a hub without the route the PUT answers 404 and is caught below.
    holds_one, hub_hash, hub_sha = _hub_avatar(endpoint)
    base = AvatarPlan(endpoint_id=endpoint_id, hub_hash=hub_hash, hub_sha256=hub_sha)
    if config_dir is None:
        return base
    agent = published_agent(config_dir)
    if agent is None or agent.avatar_broken:
        # A doubt: neither publish nor delete (see the module docstring).
        return base
    agent_facts = AvatarPlan(
        action=NONE,
        endpoint_id=endpoint_id,
        agent_id=agent.local_agent_id,
        name=agent.name,
        hub_hash=hub_hash,
        hub_sha256=hub_sha,
    )
    if agent.avatar_data is None:
        return agent_facts.replaced(DELETE if holds_one else NONE)
    # The digest describes the bytes being sent, never ``agent.yml``'s claim: the
    # store already refused any file that does not hash to its block (R3-F2), and
    # the hub dedupes on exactly this header, so it is recomputed here from the
    # same bytes rather than trusted across a hop (R1-F3).
    data = agent.avatar_data
    digest = hashlib.sha256(data).hexdigest()
    planned = AvatarPlan(
        action=NONE,
        endpoint_id=endpoint_id,
        agent_id=agent.local_agent_id,
        name=agent.name,
        data=data,
        digest=digest,
        hub_hash=hub_hash,
        hub_sha256=hub_sha,
    )
    return planned.replaced(KEEP if hub_sha == digest else PUBLISH)


async def _apply(client: AvatarClient, plan: AvatarPlan, account_id: str | None) -> None:
    """Carry out one plan. Refusals and transport failures PROPAGATE by design."""

    if plan.endpoint_id is None:
        return
    if plan.action == DELETE:
        await client.delete_endpoint_avatar(plan.endpoint_id, expected_account_id=account_id)
        return
    if plan.action != PUBLISH or plan.data is None or plan.digest is None:
        return
    await client.put_endpoint_avatar(
        plan.endpoint_id, plan.data, plan.digest, expected_account_id=account_id
    )


def _account_id(status: Any) -> str | None:
    account_id = getattr(status, "account_id", None)
    return account_id if isinstance(account_id, str) and account_id else None


async def reconcile(
    client: AvatarClient,
    status: Any,
    tenant_id: str,
    config_dir: Path | str | None,
    endpoint_id: str | None = None,
) -> None:
    """Publish, keep or clear this endpoint's picture. NEVER raises.

    Called beside ``StatusDiscipline._sync_agent_name``: same boundary, same
    per-account lock, same "a cosmetic field never fails the read" posture. A
    failure leaves the status answer exactly as it was and is reported as one
    line naming the exception TYPE and the hub's error code only.
    """

    try:
        plan = plan_avatar(status, tenant_id, config_dir, endpoint_id)
        await _apply(client, plan, _account_id(status))
    except Exception as exc:  # noqa: BLE001 - a picture never fails a status read
        code = getattr(exc, "code", None)
        logger.warning(
            "team_bridge: avatar reconciliation deferred (%s%s)",
            type(exc).__name__,
            f", {code}" if isinstance(code, str) and code else "",
        )


async def publish_agent_avatar(
    client: AvatarClient,
    tenant_id: str,
    config_dir: Path | str | None,
    *,
    access: Any = None,
    status: Any = None,
) -> AvatarPlan:
    """Publish the local picture NOW, whatever the hub already holds.

    The explicit-save and CLI path: an explicit save means "this file is the
    picture", so the digest comparison is deliberately skipped. Refusals and
    transport failures propagate — there is a caller here who can read them.

    ONE identity, one read. ``status`` and ``access`` are the caller's already
    resolved records; when they are not supplied both are resolved here, in this
    order, and the status is fetched WITH that access. Resolving the access
    afterwards and pinning the write to it compared a value with itself, so the
    advertised account pin could never fail (review R1-S4).
    """

    access = access if access is not None else await client.current_access()
    status = status if status is not None else await client.status(access=access)
    plan = plan_avatar(status, tenant_id, config_dir)
    if plan.endpoint_id is None or plan.digest is None or plan.data is None:
        # Nothing to publish IS the answer: an explicit publish never turns into
        # a delete behind the caller's back.
        return plan.replaced(NONE)
    forced = plan.replaced(PUBLISH)
    await _apply(client, forced, _account_id(access))
    return forced


async def clear_agent_avatar(
    client: AvatarClient, tenant_id: str, *, access: Any = None, status: Any = None
) -> AvatarPlan:
    """Remove the HUB copy. The local file is the local writer's to delete.

    Deliberately does not touch ``<config>/agents``: clearing locally is one
    write through the agent store, and a second writer for the same block is
    exactly how the two would end up disagreeing about what the file says.
    Same one-identity rule as :func:`publish_agent_avatar`.
    """

    access = access if access is not None else await client.current_access()
    status = status if status is not None else await client.status(access=access)
    plan = plan_avatar(status, tenant_id, None)
    if plan.endpoint_id is None:
        return plan
    forced = plan.replaced(DELETE)
    await _apply(client, forced, _account_id(access))
    return forced


__all__ = [
    "AvatarClient",
    "AvatarPlan",
    "clear_agent_avatar",
    "plan_avatar",
    "publish_agent_avatar",
    "reconcile",
]
