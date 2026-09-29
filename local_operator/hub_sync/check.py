"""Classification and the two check arms (design B1.3, B1.4).

One classification for agents and teams: :func:`classify` compares the three
texts (or their fingerprints) and says which side moved. The arms do the I/O
(fetch the remote, read the local row, read the baseline) and hand a
:class:`ItemCheck` to the service; neither arm writes anything.

CREDENTIALS DIFFER BY FAMILY (design B0.6). A public agent downloads anonymously;
an organization agent and EVERY team call needs the signed-in person's bearer.
The arms therefore take ``client_for_tenant(tenant_id)`` and pick per item, and
the tenant a row was pulled from comes from its baseline record — never from the
remote. A 404 is "unavailable / hub-item-missing" and is NEVER read as "the hub
removed it, delete locally": removal and lost membership are indistinguishable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync.segment import norm

logger = logging.getLogger(__name__)

State = Literal["up-to-date", "remote-only", "local-only", "both-changed", "baseline-unknown"]
Verdict = Literal["up-to-date", "available", "unavailable"]

ClientFor = Callable[[str | None], Any]

#: Bound on ONE hub request (design B3.3, "per-request timeout 20 s"). Every check
#: runs in a worker thread that cannot be cancelled; without a socket timeout a
#: stalled connection would hold the runner tick, its lock and the update lease,
#: and daemon shutdown could hang at interpreter exit.
REQUEST_TIMEOUT_S = 20.0


@dataclass(frozen=True)
class Classification:
    state: State
    baseline: Literal["known", "unknown"]


def classify(
    base_fp: str | None, local_fp: str, remote_fp: str, *, baseline_known: bool = True
) -> Classification:
    """Which side moved, from fingerprints (design B1.4).

    ``local-only`` (L != B, R == B) is up-to-date for a PULL: there is nothing to
    fetch and the push side owns it.
    """

    if local_fp == remote_fp:
        return Classification("up-to-date", "known" if baseline_known else "unknown")
    if not baseline_known or base_fp is None:
        return Classification("baseline-unknown", "unknown")
    if remote_fp == base_fp:
        return Classification("local-only", "known")
    if local_fp == base_fp:
        return Classification("remote-only", "known")
    return Classification("both-changed", "known")


@dataclass(frozen=True)
class ItemCheck:
    """One linked item, compared with the hub. ``remote`` is not persisted."""

    kind: Literal["agent", "team"]
    local_id: str
    name: str
    hub_id: str
    tenant_id: str | None
    verdict: Verdict
    classification: Classification | None = None
    local_fp: str | None = None
    remote_fp: str | None = None
    #: Fields of the row as read for this check — the ``L`` a merge is computed
    #: from, and what the apply re-verifies under the lock (B2.8.3).
    local: Mapping[str, Any] | None = None
    remote: Mapping[str, Any] | None = None
    base: Mapping[str, Any] | None = None
    #: B4.3 failure class when ``verdict == "unavailable"``.
    reason: str = ""
    detail: str = ""


def _status_of(error: BaseException) -> int | None:
    return getattr(error, "status_code", None)


def _failure_reason(error: BaseException) -> tuple[str, str]:
    """``(error_class, sentence)`` for a failed fetch (credentials are scrubbed upstream)."""

    status = _status_of(error)
    text = str(error)
    if status == 404 or "404" in text[:200] and "Not Found" in text:
        return (
            "hub-item-missing",
            "the hub no longer lists this item, or this account cannot see it",
        )
    return "hub-error", f"could not reach the hub: {text[:200]}"


# -- agent arm -----------------------------------------------------------------------------


def _agent_fields(instructions: str, description: str) -> dict[str, str]:
    return {"instructions": norm(instructions), "description": norm(description)}


def check_hub_agents(
    registry: Any,
    *,
    client_for_tenant: ClientFor,
    names: Sequence[str] | None = None,
    only_ids: "set[str] | None" = None,
    fetch: Callable[..., tuple[str, str]] | None = None,
) -> list[ItemCheck]:
    """Compare every hub-pulled agent (or the named ones) with its listing.

    Baseline resolution (A2.3): the record if present; else, when the row still
    equals its pulled fingerprint (``hub_sha256:`` tag, legacy or canonical form),
    B := L exactly and the record is written lazily; else the baseline is unknown
    and the item is check-only.
    """

    from local_operator.agents import (
        HUB_SHA256_PREFIX,
        _fetch_hub_profile,
        hub_origin,
        marker_value,
    )

    fetcher = fetch or _fetch_hub_profile
    wanted = {str(n).strip().lower() for n in names} if names else None
    config_dir = Path(registry.config_dir)
    rows = [
        a
        for a in registry.list_agents()
        if hub_origin(a) is not None
        and (wanted is None or str(a.name or "").strip().lower() in wanted)
        and (only_ids is None or a.id in only_ids)
    ]
    rows.sort(key=lambda a: str(a.name or "").lower())
    checks: list[ItemCheck] = []
    for agent in rows:
        hub_id = hub_origin(agent) or ""
        name = str(agent.name or "")
        record = prov.read_baseline(config_dir, "agent", agent.id)
        tenant = record.tenant_id if record else None
        try:
            local_text = registry.get_agent_system_prompt(agent.id)
        except Exception as error:  # noqa: BLE001 - one row never ends the run
            checks.append(
                ItemCheck(
                    "agent",
                    agent.id,
                    name,
                    hub_id,
                    tenant,
                    "unavailable",
                    reason="hub-error",
                    detail=f"could not read the local row: {error}",
                )
            )
            continue
        local = _agent_fields(local_text, str(agent.description or ""))
        local_fp = prov.fingerprint_agent(local["instructions"], local["description"])

        client = client_for_tenant(tenant)
        if client is None:
            checks.append(
                ItemCheck(
                    "agent",
                    agent.id,
                    name,
                    hub_id,
                    tenant,
                    "unavailable",
                    local_fp=local_fp,
                    local=local,
                    reason="no-credential",
                    detail="no Radient credential is available for this item",
                )
            )
            continue
        try:
            if tenant:
                text, desc = fetcher(
                    client, hub_id, with_credential=True, timeout=REQUEST_TIMEOUT_S
                )
            else:
                text, desc = fetcher(client, hub_id, timeout=REQUEST_TIMEOUT_S)
        except Exception as error:  # noqa: BLE001 - classified below
            reason, detail = _failure_reason(error)
            checks.append(
                ItemCheck(
                    "agent",
                    agent.id,
                    name,
                    hub_id,
                    tenant,
                    "unavailable",
                    local_fp=local_fp,
                    local=local,
                    reason=reason,
                    detail=detail,
                )
            )
            continue
        remote = _agent_fields(text, desc)
        remote_fp = prov.fingerprint_agent(remote["instructions"], remote["description"])

        base_fields: Mapping[str, Any] | None = None
        base_fp: str | None = None
        if record is not None:
            base_fields, base_fp = record.fields, prov.fingerprint_fields("agent", record.fields)
        else:
            tag = marker_value(agent, HUB_SHA256_PREFIX)
            if tag and tag in (
                local_fp,
                prov.legacy_agent_fingerprint(local_text, str(agent.description or "")),
            ):
                # Unedited since the pull: B == L exactly, not a guess. Adopt lazily.
                base_fields, base_fp = local, local_fp
                prov.record_agent_baseline(
                    config_dir,
                    local_id=agent.id,
                    hub_id=hub_id,
                    instructions=local["instructions"],
                    description=local["description"],
                    tenant_id=None,
                    recorded_by="adopt",
                )
        classification = classify(
            base_fp, local_fp, remote_fp, baseline_known=base_fields is not None
        )
        checks.append(
            ItemCheck(
                "agent",
                agent.id,
                name,
                hub_id,
                tenant,
                (
                    "up-to-date"
                    if classification.state in ("up-to-date", "local-only")
                    else "available"
                ),
                classification=classification,
                local_fp=local_fp,
                remote_fp=remote_fp,
                local=local,
                remote=remote,
                base=base_fields,
            )
        )
    return checks


# -- team arm -----------------------------------------------------------------------------------


def team_fields_of(team: Any) -> dict[str, Any]:
    return prov.team_fields(
        description=team.description,
        manager=team.manager,
        members=team.members,
        instructions=team.instructions,
        project=team.project,
    )


def _team_confirmed_gone(registry: Any, team_id: str) -> bool:
    """True only when ``teams/<id>`` is absent AND no swap-in-progress sibling exists.

    Any doubt (an unreadable directory, a hidden ``.<id>.*`` staging/backup entry
    from a save in flight, an id that cannot be a path segment) answers False, i.e.
    "keep the baseline".
    """

    teams_dir = getattr(registry, "teams_dir", None)
    if teams_dir is None or not team_id or "/" in team_id or team_id.startswith("."):
        return False
    try:
        if not teams_dir.is_dir() or (teams_dir / team_id).exists():
            return False
        return not any(child.name.startswith(f".{team_id}.") for child in teams_dir.iterdir())
    except OSError:
        return False


def check_hub_teams(
    registry: Any,
    *,
    client_for_tenant: ClientFor,
    names: Sequence[str] | None = None,
    only_ids: "set[str] | None" = None,
) -> list[ItemCheck]:
    """Compare every LINKED team (one with a baseline record) with the hub.

    A team with no record was never linked (pulled before this feature, or made
    locally) and is not touched. The whole document — briefs included — is one
    small JSON GET, so a team check is cheap. ``name`` is never compared: a hub
    rename must never rename the local row (A7).
    """

    config_dir = Path(registry.config_dir)
    wanted = {str(n).strip().lower() for n in names} if names else None
    teams = sorted(registry.list_teams(), key=lambda t: t.name.lower())
    # A baseline is what keeps a deleted brief section from being re-added, so it is
    # pruned only when the team is CONFIRMED gone on disk right now. The listing alone
    # is not authoritative: ``TeamRegistry._load`` skips a row whose directory is
    # mid-swap (the documented publish gap) and reads as EMPTY on an ``iterdir``
    # error, so a check racing a user's save would otherwise unlink the baseline and
    # silently unlink the team.
    prov.prune(
        config_dir,
        "team",
        {t.id for t in teams},
        confirmed_absent=lambda team_id: _team_confirmed_gone(registry, team_id),
    )
    checks: list[ItemCheck] = []
    for meta in teams:
        if wanted is not None and meta.name.strip().lower() not in wanted:
            continue
        if only_ids is not None and meta.id not in only_ids:
            continue
        record = prov.read_baseline(config_dir, "team", meta.id)
        if record is None:
            continue
        tenant = record.tenant_id
        try:
            team = registry.get_team(meta.id)
        except Exception as error:  # noqa: BLE001
            checks.append(
                ItemCheck(
                    "team",
                    meta.id,
                    meta.name,
                    record.hub_id,
                    tenant,
                    "unavailable",
                    reason="hub-error",
                    detail=f"could not read the local team: {error}",
                )
            )
            continue
        local = team_fields_of(team)
        local_fp = prov.fingerprint_team(local)
        client = client_for_tenant(tenant)
        if client is None:
            checks.append(
                ItemCheck(
                    "team",
                    meta.id,
                    meta.name,
                    record.hub_id,
                    tenant,
                    "unavailable",
                    local_fp=local_fp,
                    local=local,
                    reason="no-credential",
                    detail="teams need `lop login radient` (organization sign-in)",
                )
            )
            continue
        try:
            document = client.get_team(record.hub_id, timeout=REQUEST_TIMEOUT_S)
        except Exception as error:  # noqa: BLE001
            reason, detail = _failure_reason(error)
            checks.append(
                ItemCheck(
                    "team",
                    meta.id,
                    meta.name,
                    record.hub_id,
                    tenant,
                    "unavailable",
                    local_fp=local_fp,
                    local=local,
                    reason=reason,
                    detail=detail,
                )
            )
            continue
        owner = str(document.get("tenant_id") or "")
        if tenant and owner and owner != tenant:
            # Same rule as the pull route's 409: never store under the wrong tenant.
            checks.append(
                ItemCheck(
                    "team",
                    meta.id,
                    meta.name,
                    record.hub_id,
                    tenant,
                    "unavailable",
                    local_fp=local_fp,
                    local=local,
                    reason="hub-error",
                    detail=f"the hub now reports this team under organization {owner}",
                )
            )
            continue
        remote = prov.team_fields(
            description=str(document.get("description") or ""),
            manager=str(document.get("manager") or ""),
            members=document.get("members"),
            instructions=str(document.get("instructions") or ""),
            project=str(document.get("project") or ""),
        )
        remote_fp = prov.fingerprint_team(remote)
        known = record.recorded_by != prov.UNKNOWN_BASELINE
        base_fp = prov.fingerprint_fields("team", record.fields) if known else None
        classification = classify(base_fp, local_fp, remote_fp, baseline_known=known)
        note = ""
        remote_name = str(document.get("name") or "")
        if remote_name and remote_name != meta.name:
            note = "remote-renamed"
        checks.append(
            ItemCheck(
                "team",
                meta.id,
                meta.name,
                record.hub_id,
                tenant,
                (
                    "up-to-date"
                    if classification.state in ("up-to-date", "local-only")
                    else "available"
                ),
                classification=classification,
                local_fp=local_fp,
                remote_fp=remote_fp,
                local=local,
                remote=remote,
                base=record.fields if known else None,
                detail=note,
            )
        )
    return checks


@dataclass(frozen=True)
class Link:
    """A linked item, without any network: enough to order and bound a tick."""

    kind: Literal["agent", "team"]
    local_id: str
    name: str
    hub_id: str
    tenant_id: str | None


def list_links(agent_registry: Any, team_registry: Any) -> list[Link]:
    """Every linked agent (``hub:`` tag) and team (baseline record). Local reads only."""

    from local_operator.agents import hub_origin

    links: list[Link] = []
    config_dir = Path((agent_registry or team_registry).config_dir)
    if agent_registry is not None:
        for agent in agent_registry.list_agents():
            hub_id = hub_origin(agent)
            if hub_id is None:
                continue
            record = prov.read_baseline(config_dir, "agent", agent.id)
            links.append(
                Link(
                    "agent",
                    agent.id,
                    str(agent.name or ""),
                    hub_id,
                    record.tenant_id if record else None,
                )
            )
    if team_registry is not None:
        for team in team_registry.list_teams():
            record = prov.read_baseline(config_dir, "team", team.id)
            if record is not None:
                links.append(Link("team", team.id, team.name, record.hub_id, record.tenant_id))
    return links
