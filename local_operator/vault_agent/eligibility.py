"""Whether this machine's agent may be OFFERED Vault access, and the cached grant.

THE OPERATOR'S CONSTRAINT. The vault capability is available only to a person
signed in to Radient AND on a Team plan, and it is advertised only when they
have access. So the ``secret`` tool's ``vault_resolve`` verb, its two
properties, the guide paragraph and the ``lop vault`` pointers all ask
:func:`gate` and nothing else. One answer, one rule.

THE GATE (zero network; every conjunct must hold; any failure reads CLOSED):

1. ``hub_org.eligibility.eligible_orgs`` — the existing Radient Team record
   (signed in, active membership, entitled plan).
2. A committed Team Bridge owner (``team_bridge.store.Store.active_owner_readonly``)
   whose tenant is among (1)'s orgs. Agent-server accepts an agent call only from
   the caller's OWN active endpoint in the vault's tenant (VAULT.md §6.20 item 5,
   step 6a, ``422 not_owned``), so no other endpoint could ever work.
3. This module's Vault record for that tenant and endpoint, written by
   :func:`refresh` from route 1: ``delegations_available`` (the deployment's dark
   gate, VAULT.md §6.12), ``eligible``, ``tenant_state == "enabled"`` and
   ``vault_principal.state == "approved"``. Route 1's ``eligible`` is the Vault
   service's own paid-Team decision, the same field the console reads, so the
   harness and the console cannot disagree about one organization.

While production's delegation port is unwired, route 1 answers
``delegations_available: false`` (or is not registered at all), so the gate stays
closed and nothing is advertised — the dark state, by construction.

REFRESH only at boundaries (sign-in and daemon start through
``hub_org.enrol.run_boundary``, TUI start, a 6 h TTL, ``lop vault status|connect``
forced, and a ``403 vault_not_available``/``plan_required`` that closes it at
once). Fail-closed reads mirror the hub record: no record, a record for another
Radient base, another tenant/endpoint, older than :data:`MAX_AGE_SECONDS`, or
corrupt.

THE GRANT CACHE (``agent_access.json``) holds the route-29 pickup — the
``vdl_`` id, its expiry and the (tenant, endpoint, fingerprint) it was selected
for. None of it is secret; the authority's verify of the agent proof is the only
check, so a stale or wrong id costs one ``403 delegation_required``, which clears
it. Two files rather than one so the refresh and the pickup never race a
read-modify-write over each other's keys.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

SCHEMA = 1
ELIGIBILITY_NAME = "eligibility.json"
GRANT_NAME = "agent_access.json"

#: Same horizon as the hub record: a plan lapse cannot keep the verb offered
#: indefinitely on a machine that never reaches a refresh boundary.
MAX_AGE_SECONDS = 7 * 24 * 3600
REFRESH_TTL_SECONDS = 6 * 3600
ERROR_BACKOFF_SECONDS = 30 * 60


@dataclass(frozen=True)
class Gate:
    """An OPEN gate: the owner endpoint the agent proves as. Not secret."""

    tenant_id: str
    endpoint_id: str
    tenant_name: str


@dataclass(frozen=True)
class Grant:
    """The selected route-29 row, cached. Not secret."""

    delegation_id: str
    expires_at: str
    tenant_id: str
    endpoint_id: str
    key_fingerprint: str


def vault_dir(config_dir: Path) -> Path:
    return Path(config_dir) / "vault"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return None
    return data


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    # The hub record's writer (0700 dir, 0600 temp file, fsync, os.replace), not a
    # second spelling of it.
    from local_operator.hub_org.eligibility import _atomic_write

    _atomic_write(path, dict(payload, schema=SCHEMA))


def _configured_base_url(config_dir: Path) -> str:
    from local_operator.hub_org.eligibility import _configured_base_url as hub_base

    return hub_base(Path(config_dir))


def _owner(config_dir: Path) -> Any | None:
    from local_operator.team_bridge.store import Store

    return Store(Path(config_dir)).active_owner_readonly()


# -- the gate -------------------------------------------------------------------


def _record_open(
    data: Mapping[str, Any] | None, *, tenant_id: str, endpoint_id: str, base: str
) -> bool:
    if data is None:
        return False
    checked = data.get("checked_at")
    if not isinstance(checked, (int, float)) or time.time() - checked > MAX_AGE_SECONDS:
        return False
    return (
        data.get("base_url") == base
        and data.get("tenant_id") == tenant_id
        and data.get("endpoint_id") == endpoint_id
        and data.get("delegations_available") is True
        and data.get("eligible") is True
        and data.get("tenant_state") == "enabled"
        and data.get("principal_state") == "approved"
    )


def gate(config_dir: Path | None) -> Gate | None:
    """The open gate, or ``None`` (closed). Zero network; never raises."""

    if config_dir is None:
        return None
    try:
        return _gate(Path(config_dir))
    except Exception:  # noqa: BLE001 - a gate read must never break a session build
        logger.debug("vault agent gate read failed", exc_info=True)
        return None


def _gate(config_dir: Path) -> Gate | None:
    from local_operator.hub_org.eligibility import eligible_orgs

    orgs = {org.tenant_id: org for org in eligible_orgs(config_dir)}
    if not orgs:
        return None
    # The record before the owner: a JSON read is cheaper than opening SQLite,
    # and the record is absent on every machine that never ran a refresh.
    data = _read_json(vault_dir(config_dir) / ELIGIBILITY_NAME)
    if data is None:
        return None
    owner = _owner(config_dir)
    if owner is None or owner.tenant_id not in orgs:
        return None
    if not _record_open(
        data,
        tenant_id=owner.tenant_id,
        endpoint_id=owner.endpoint_id,
        base=_configured_base_url(config_dir),
    ):
        return None
    return Gate(
        tenant_id=owner.tenant_id,
        endpoint_id=owner.endpoint_id,
        tenant_name=orgs[owner.tenant_id].tenant_name,
    )


def is_open(config_dir: Path | None) -> bool:
    return gate(config_dir) is not None


# -- refresh (network) ------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """What a refresh learned, for ``lop vault status``/``connect`` to explain."""

    #: open | cached | signed_out | not_team | no_endpoint | unavailable |
    #: not_eligible | frozen | disabled | not_approved | error
    state: str
    tenant_id: str | None = None
    endpoint_id: str | None = None
    tenant_name: str | None = None


def _write_closed(config_dir: Path, **fields: Any) -> None:
    _write_json(
        vault_dir(config_dir) / ELIGIBILITY_NAME,
        {
            "base_url": _configured_base_url(config_dir),
            "checked_at": time.time(),
            "delegations_available": False,
            "eligible": False,
            **fields,
        },
    )


def close(config_dir: Path, *, reason: str) -> None:
    """Close the gate now (a server ``403 vault_not_available``/``plan_required``)."""

    with contextlib.suppress(Exception):
        _write_closed(Path(config_dir), closed_by=reason)


def refresh(config_dir: Path, *, boundary: str, force: bool = False) -> Observation:
    """Re-read route 1 for the owner tenant and rewrite the record. Never raises."""

    config_dir = Path(config_dir)
    try:
        return _refresh(config_dir, boundary=boundary, force=force)
    except Exception:  # noqa: BLE001 - boundaries must never fail on the gate
        logger.debug("vault agent gate refresh failed (%s)", boundary, exc_info=True)
        return Observation(state="error")


def _refresh(config_dir: Path, *, boundary: str, force: bool) -> Observation:
    from local_operator.hub_org import eligibility as hub

    path = vault_dir(config_dir) / ELIGIBILITY_NAME
    if not force:
        data = _read_json(path)
        if data is not None:
            now = time.time()
            checked = data.get("checked_at")
            failed = data.get("last_error_at")
            fresh = isinstance(checked, (int, float)) and now - checked < REFRESH_TTL_SECONDS
            backing_off = isinstance(failed, (int, float)) and now - failed < ERROR_BACKOFF_SECONDS
            if fresh or backing_off:
                current = gate(config_dir)
                return Observation(state="open" if current else "cached")

    base = _configured_base_url(config_dir)
    resolved = hub.person_client(config_dir, base)
    if resolved is None:
        _write_closed(config_dir)
        return Observation(state="signed_out")
    orgs = {org.tenant_id: org for org in hub.eligible_orgs(config_dir)}
    if not orgs:
        _write_closed(config_dir)
        return Observation(state="not_team")
    owner = _owner(config_dir)
    if owner is None or owner.tenant_id not in orgs:
        _write_closed(config_dir)
        return Observation(state="no_endpoint")

    from local_operator.vault_agent.client import VaultClient

    client = VaultClient.for_person(resolved[0])
    tenant = owner.tenant_id
    try:
        body = client.status(tenant)
    except Exception as exc:  # noqa: BLE001 - classified below
        status = getattr(exc, "status_code", None)
        if status == 404:
            # VAULT_ENABLED off: route 1 is not registered at all (VAULT.md
            # §6.12), which is the dark gate, not an outage.
            _write_closed(config_dir, tenant_id=tenant, endpoint_id=owner.endpoint_id)
            return Observation(state="unavailable", tenant_id=tenant)
        if status in (401, 403):
            _write_closed(config_dir, tenant_id=tenant, endpoint_id=owner.endpoint_id)
            return Observation(state="signed_out" if status == 401 else "not_eligible")
        data = _read_json(path) or {}
        data["last_error_at"] = time.time()
        _write_json(path, data)
        logger.debug("vault status read failed (%s)", boundary, exc_info=True)
        return Observation(state="error", tenant_id=tenant)

    org = next(
        (
            entry
            for entry in body.get("organizations") or []
            if isinstance(entry, Mapping) and entry.get("tenant_id") == tenant
        ),
        None,
    )
    principal = (org or {}).get("vault_principal")
    record = {
        "base_url": base,
        "checked_at": time.time(),
        "tenant_id": tenant,
        "endpoint_id": owner.endpoint_id,
        "delegations_available": body.get("delegations_available") is True,
        "eligible": bool(org) and org.get("eligible") is True,
        "tenant_state": (org or {}).get("tenant_state"),
        "principal_state": principal.get("state") if isinstance(principal, Mapping) else None,
        "last_error_at": None,
    }
    _write_json(path, record)
    name = orgs[tenant].tenant_name
    base_obs = {"tenant_id": tenant, "endpoint_id": owner.endpoint_id, "tenant_name": name}
    # Frozen and disabled first: route 1 answers a frozen tenant with
    # ``eligible:false`` too (VAULT.md §14), and "frozen" is the true reason.
    if record["tenant_state"] in ("frozen", "disabled"):
        return Observation(state=str(record["tenant_state"]), **base_obs)
    if not record["eligible"] or record["tenant_state"] == "not_available":
        return Observation(state="not_eligible", **base_obs)
    if record["tenant_state"] != "enabled" or record["principal_state"] != "approved":
        return Observation(state="not_approved", **base_obs)
    if not record["delegations_available"]:
        return Observation(state="unavailable", **base_obs)
    return Observation(state="open", **base_obs)


# -- the grant cache ------------------------------------------------------------


def read_grant(
    config_dir: Path, *, tenant_id: str, endpoint_id: str, fingerprint: str
) -> Grant | None:
    """The cached grant for EXACTLY this (tenant, endpoint, key), else ``None``."""

    data = _read_json(vault_dir(config_dir) / GRANT_NAME)
    if data is None:
        return None
    try:
        grant = Grant(
            delegation_id=str(data["delegation_id"]),
            expires_at=str(data["expires_at"]),
            tenant_id=str(data["tenant_id"]),
            endpoint_id=str(data["endpoint_id"]),
            key_fingerprint=str(data["key_fingerprint"]),
        )
    except KeyError:
        return None
    if (grant.tenant_id, grant.endpoint_id, grant.key_fingerprint) != (
        tenant_id,
        endpoint_id,
        fingerprint,
    ):
        return None
    if grant_expired(grant):
        return None
    return grant


def grant_expired(grant: Grant, now: float | None = None) -> bool:
    """Whether the cached expiry has passed. A display aid only: the authority's
    clock decides, and an unparseable value reads as expired (fail closed)."""

    try:
        when = datetime.fromisoformat(grant.expires_at.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return True
    return (time.time() if now is None else now) >= when


def write_grant(config_dir: Path, grant: Grant) -> None:
    _write_json(
        vault_dir(config_dir) / GRANT_NAME,
        {
            "delegation_id": grant.delegation_id,
            "expires_at": grant.expires_at,
            "tenant_id": grant.tenant_id,
            "endpoint_id": grant.endpoint_id,
            "key_fingerprint": grant.key_fingerprint,
        },
    )


def clear_grant(config_dir: Path) -> None:
    with contextlib.suppress(OSError):
        (vault_dir(config_dir) / GRANT_NAME).unlink()


def select_grant(
    rows: Sequence[Mapping[str, Any]], *, tenant_id: str, endpoint_id: str, fingerprint: str
) -> Grant | None:
    """VAULT.md §6.20's pickup: the ``live`` row whose ``endpoint_id`` AND
    ``key_fingerprint`` equal ours byte for byte; with several, the latest
    ``expires_at``. Selection, not verification (the authority verifies)."""

    best: tuple[float, Grant] | None = None
    for row in rows:
        if row.get("state") != "live":
            continue
        if row.get("endpoint_id") != endpoint_id or row.get("key_fingerprint") != fingerprint:
            continue
        delegation_id = row.get("delegation_id")
        expires_at = row.get("expires_at")
        if not isinstance(delegation_id, str) or not isinstance(expires_at, str):
            continue
        try:
            stamp = datetime.fromisoformat(expires_at.replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
        grant = Grant(
            delegation_id=delegation_id,
            expires_at=expires_at,
            tenant_id=str(row.get("tenant_id") or tenant_id),
            endpoint_id=endpoint_id,
            key_fingerprint=fingerprint,
        )
        if best is None or stamp > best[0]:
            best = (stamp, grant)
    return best[1] if best else None
