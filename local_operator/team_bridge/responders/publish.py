"""Mirror the local policy's non-sensitive projection to the hub (RFC §7.1, §9.3).

The hub wakes an agent for an alert only if that agent's member row carries an
enabled, consented, unpaused responder record (agent-server ``wakeableResponder``).
This module writes that record FROM the owner's local policy: ``PUT
…/responders/:endpointid`` for every channel the file names, ``DELETE`` (Stop) for a
channel the file disables.

CONSENT IS THE OWNER'S, BY HASH. The hub records which immutable consent text the
owner accepted (``responder_consent_v1`` + its SHA-256). The owner states that hash
in their own file (``consent_sha256``); this module fetches the text the hub serves
NOW and publishes only when the two agree. A hub that has changed its consent text
is therefore never silently re-accepted on the owner's behalf: the channel is
skipped with ``consent_mismatch`` until the owner reads the new text and updates
the file.

The local runner re-checks the file on every wake regardless (``runner``), so a
failed or stale publication can only make the hub wake this agent LESS or MORE than
the file says — and a wake the file does not admit is refused locally at no cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .policy import CONSENT_VERSION, ResponderPolicy

__all__ = ["PublishClient", "PublishResult", "publish_policy"]


class PublishClient(Protocol):
    async def responder_consent(self) -> dict[str, Any]: ...

    async def responder_publish(
        self,
        conversation_id: str,
        endpoint_id: str,
        *,
        tenant_id: str,
        enabled: bool,
        autonomy_tier: str,
        consent_version: str,
        consent_sha256: str,
        policy_sha256: str,
        paused_until: str | None = None,
    ) -> dict[str, Any]: ...

    async def responder_stop(
        self, conversation_id: str, endpoint_id: str, *, tenant_id: str
    ) -> None: ...


@dataclass(frozen=True)
class PublishResult:
    """Per-channel developer codes.

    ``published`` | ``stopped`` | ``consent_mismatch`` | ``failed:<code>``.
    """

    channels: Mapping[str, str]
    problem: str | None = None


def _code(exc: BaseException) -> str:
    return str(getattr(exc, "code", "") or exc.__class__.__name__)


async def publish_policy(
    client: PublishClient, policy: ResponderPolicy, *, tenant_id: str, endpoint_id: str
) -> PublishResult:
    """Publish every channel in ``policy``. Never raises for a hub answer."""

    if policy.problem is not None:
        # An invalid file publishes NOTHING (not even Stops): the previous hub state
        # stays, and the local runner refuses every wake because the file reads as off.
        return PublishResult({}, problem=policy.problem)
    try:
        served = await client.responder_consent()
    except Exception as exc:  # noqa: BLE001 — a dark hub (404) or outage: nothing to publish
        return PublishResult({}, problem=f"consent_unavailable:{_code(exc)}")
    served_version = str(served.get("version") or "")
    served_sha = str(served.get("sha256") or "")
    out: dict[str, str] = {}
    for conversation_id, channel in sorted(policy.channels.items()):
        try:
            if not channel.enabled:
                await client.responder_stop(conversation_id, endpoint_id, tenant_id=tenant_id)
                out[conversation_id] = "stopped"
                continue
            if served_version != CONSENT_VERSION or channel.consent_sha256 != served_sha:
                out[conversation_id] = "consent_mismatch"
                continue
            await client.responder_publish(
                conversation_id,
                endpoint_id,
                tenant_id=tenant_id,
                enabled=True,
                autonomy_tier=channel.autonomy_tier,
                consent_version=CONSENT_VERSION,
                consent_sha256=served_sha,
                policy_sha256=str(policy.sha256),
                paused_until=(
                    channel.paused_until.isoformat().replace("+00:00", "Z")
                    if channel.paused_until
                    else None
                ),
            )
            out[conversation_id] = "published"
        except Exception as exc:  # noqa: BLE001 — one channel's refusal never blocks the next
            out[conversation_id] = f"failed:{_code(exc)}"
    return PublishResult(out)
