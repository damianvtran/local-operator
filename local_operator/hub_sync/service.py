"""Check, merge and apply: the one orchestration every surface calls (design B5).

The runner, the desktop routes, the CLI and the ``agent`` tool all go through
:func:`check_items`, :func:`apply_items` and :func:`status_snapshot`; none of them
re-implements classification or wording. Everything here is SYNCHRONOUS and
blocking (registry files, ``requests`` clients, a model call in a worker loop),
so async callers reach it through ``asyncio.to_thread`` — the shape every desktop
route uses.

THE NEVER-BLINDLY-OVERWRITE ENFORCEMENT POINTS (B2.8), where they live here:
(1) the only write path takes a :class:`~local_operator.hub_sync.merge.MergeResult`
produced by ``merge_field``/``replace_field`` — there is no ``write(remote)``;
(2) proposals are validated inside ``merge_field``; (3) the apply re-verifies,
under the team registry's writer lock (agents: immediately before the write —
there is no agent registry lock, design Q4), that the row still equals the
snapshot the merge was computed from; (4) a backup of the local fields is written
BEFORE any change; (5) a large shrink forces ``needs-review`` under auto-update.

ONE WRITER AT A TIME, ACROSS PROCESSES. :func:`apply_items` takes the
:class:`~local_operator.hub_sync.store.RunnerLease` (heartbeat-renewed) for the
whole run unless it is a dry run, so the runner tick, the desktop routes, ``lop
agents|teams sync`` and the ``agent`` tool cannot compute and write the same item
concurrently (double model spend, double write). The daemon's routes additionally
serialise on the runner's ``asyncio.Lock`` (``HubSyncRunner.run_exclusive``) so a
button press queues behind a running tick instead of waiting out the lease. A
CHECK writes only ``status.json``, which has its own file lock; it does not take
the lease.

PARTIAL APPLY. Deliberately not done: an item is applied whole or not at all. The
design lets an agent's ``description`` apply while an ``instructions`` region is
unresolved; that needs per-field baselines and is listed as a follow-up rather
than half-built here. It is the conservative direction (nothing lands until a
human resolves the item).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence

from local_operator.hub_sync import check as chk
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync import store as st
from local_operator.hub_sync.identity import (
    PUBLICATION_CONTRACT,
    Kind,
    LinkError,
    LinkIdentity,
    LinkOrigin,
    Visibility,
    link_origin,
)
from local_operator.hub_sync.merge import (
    FieldInput,
    MergeOptions,
    MergeResult,
    Resolver,
    merge_field,
    replace_field,
)
from local_operator.hub_sync.report import ApplyReport, ItemMergeReport, ItemOutcome
from local_operator.hub_sync.settings import HubSyncSettings

logger = logging.getLogger(__name__)

AGENT_INSTRUCTIONS_MAX = 8_000
TEAM_BRIEF_MAX = 32_768
TEAM_DESCRIPTION_MAX = 2_000
TEAM_MANAGER_MAX = 128
TEAM_ROSTER_MAX = 64

Prefer = Literal["none", "local", "remote"]

#: The one batch/listing order: agents first, then teams, then skills. Sorted by
#: this rank instead of the kind string so a skill never lands between the two
#: kinds it was added after (the TUI's ``section_order()`` is the same list).
KIND_ORDER: dict[str, int] = {"agent": 0, "team": 1, "skill": 2}


#: How long an interactive caller waits for another process's update run before
#: refusing. Long enough to ride out a short tick, short enough that a click never
#: appears hung (the lease TTL is the bound for a CRASHED holder, not for this).
LEASE_WAIT_S = 30.0


class ConcurrentEdit(Exception):
    """The row changed between the snapshot and the write; nothing was written."""


class HubBusy(Exception):
    """Another process holds the update lease; nothing was checked or written."""


@contextlib.contextmanager
def exclusive(ctx: "HubSyncContext", wait_s: float | None = None) -> Iterator[None]:
    """Hold the cross-process update lease, or raise :class:`HubBusy` after ``wait_s``.

    Blocking (it polls); async callers reach it through ``asyncio.to_thread``.
    """

    lease = st.RunnerLease(ctx.config_dir)
    # Resolved at call time so a test (or an operator override) can shorten it.
    if not lease.acquire(LEASE_WAIT_S if wait_s is None else wait_s):
        # A live holder heartbeats, so the honest bound is the lease's own expiry:
        # it is only reached if that process died without releasing.
        left = lease.remaining_s()
        hint = (
            f"; if it is gone, its lease expires in about {int(left) + 1}s and this works again"
            if left is not None
            else "; try again shortly"
        )
        raise HubBusy(
            "another hub update is running (the update runner or a second `lop` process)" + hint
        )
    try:
        yield
    finally:
        lease.release()


@dataclass
class HubSyncContext:
    """Everything a check/apply needs, injectable for tests.

    ``client_for_tenant`` is resolved BEFORE the run (async credential
    resolution belongs to the caller); ``None`` for a tenant means "no client".
    Registries default to fresh instances per call: an apply must see the rows
    as they are on disk, not through a snapshot cache.
    """

    config_dir: Path
    config_manager: Any
    client_for_tenant: chk.ClientFor
    credential: Literal["ok", "none"] = "ok"
    resolver: Resolver | None = None
    agent_registry: Any = None
    team_registry: Any = None

    def agents(self) -> Any:
        if self.agent_registry is not None:
            return self.agent_registry
        from local_operator.agents import AgentRegistry

        return AgentRegistry(self.config_dir)

    def teams(self) -> Any:
        if self.team_registry is not None:
            return self.team_registry
        from local_operator.teams import TeamRegistry

        return TeamRegistry(self.config_dir)

    def settings(self) -> HubSyncSettings:
        return HubSyncSettings.read_fresh(self.config_manager)

    def store(self) -> st.StatusStore:
        return st.StatusStore(self.config_dir)


# -- credentials -------------------------------------------------------------------------


def _clients(base_url: str, access: Any) -> tuple[chk.ClientFor, str]:
    from pydantic import SecretStr

    from local_operator.clients.radient import RadientClient

    public = RadientClient(api_key=None, base_url=base_url)
    org = (
        RadientClient(api_key=SecretStr(access.access_token), base_url=base_url) if access else None
    )

    def for_tenant(tenant_id: str | None) -> Any:
        # ``None`` = a public agent (anonymous download, design Q3). Any string,
        # even "", is an organization item and needs the signed-in person.
        return public if tenant_id is None else org

    return for_tenant, "ok" if access else "none"


async def build_clients(
    config_manager: Any, auth_store: Any | None = None
) -> tuple[chk.ClientFor, str]:
    """Async host: the per-item client picker and ``"ok"|"none"`` for the login."""

    from local_operator.providers.radient_credentials import (
        configured_radient_base_url,
        resolve_radient_oauth_access,
    )

    base_url = configured_radient_base_url(config_manager)
    access = await resolve_radient_oauth_access(
        config_manager.config_dir, base_url, store=auth_store
    )
    return _clients(base_url, access)


def build_clients_sync(config_manager: Any) -> tuple[chk.ClientFor, str]:
    """CLI host (no running loop)."""

    from local_operator.providers.radient_credentials import (
        configured_radient_base_url,
        resolve_radient_oauth_access_sync,
    )

    base_url = configured_radient_base_url(config_manager)
    access = resolve_radient_oauth_access_sync(config_manager.config_dir, base_url)
    return _clients(base_url, access)


# -- check -------------------------------------------------------------------------------


def _state_of(check: chk.ItemCheck) -> str | None:
    if check.classification is None or check.verdict != "available":
        return None
    return check.classification.state


def record_checks(
    ctx: HubSyncContext, checks: Sequence[chk.ItemCheck], *, live: set[str] | None = None
) -> None:
    """Fold check results into the status store (one locked write)."""

    def fold(doc: dict[str, Any]) -> None:
        for c in checks:
            st.apply_check(
                doc,
                kind=c.kind,
                local_id=c.local_id,
                name=c.name,
                hub_id=c.hub_id,
                tenant_id=c.tenant_id,
                verdict=c.verdict,
                classification=_state_of(c),
                baseline=c.classification.baseline if c.classification else "known",
                local_fp=c.local_fp,
                remote_fp=c.remote_fp,
                reason=c.reason,
                detail=c.detail,
            )
            # A response that carried the server's sync_contract is an OBSERVATION
            # of the deployed source -- the only honest way the per-source
            # capability attestation becomes true (see record_server_capabilities).
            # It is credited to the base that ACTUALLY ANSWERED the check
            # (round-2 R14), never to the link's recorded source: a client pointed
            # elsewhere cannot launder its contract onto a source it never spoke
            # to (eligibility still requires served_by == identity.hub_source, so
            # a mismatch simply earns no credit).
            if c.sync_contract is not None and c.served_by:
                _fold_capability(doc, c.sync_contract, c.served_by)
        if live is not None:
            st.prune_items(doc, live)

    ctx.store().mutate(fold)


def _canonical_for_compare(kind: Kind, fields: Mapping[str, Any]) -> dict[str, Any]:
    """The shape two sides are compared in for AGREEMENT, normalization included.

    The merge engine normalizes text (trailing whitespace and blank edges) when
    it writes, so a fast-forwarded skill's on-disk content can differ from the
    server's canonical spelling by exactly that normalization -- and comparing
    the RAW fingerprints would then read a clean fast-forward as a real
    difference and never seed. Normalizing both sides the same way keeps the
    agreement criterion about CONTENT: a local edit that changes anything beyond
    trailing whitespace still differs and still refuses to seed.
    """

    from local_operator.hub_sync import segment as seg

    out = dict(fields)
    # Text normalizes the same way the engine writes it, for every kind.
    for key, value in list(out.items()):
        if isinstance(value, str):
            out[key] = seg.norm(value)
    if kind != "skill":
        return out
    files = out.get("files")
    if isinstance(files, list):
        out["files"] = [
            (
                {**dict(row), "content": seg.norm(str(row.get("content") or ""))}
                if isinstance(row, Mapping)
                else row
            )
            for row in files
        ]
    if "description" in out:
        out["description"] = seg.norm(str(out.get("description") or ""))
    return out


def _seed_receipt_if_agreed(
    ctx: HubSyncContext, c: chk.ItemCheck, local_fields: Mapping[str, Any] | None
) -> bool:
    """Seed the outbound receipt (A/P) at a KNOWN-AGREED point (L == R).

    THE FIRST PUBLICATION PROBLEM: an unattended push projects from an (A, P)
    receipt, and until one exists a link can never publish -- a permanently
    dormant auto-push. The honest seeding points are exactly where local and hub
    are KNOWN to agree (a pull/adopt integrates the remote content, or a manual
    publish is accepted), because there A = P's canonical text and no delta can
    be invented. Safety properties, in order:

    * the two sides must actually AGREE (fingerprint equality) -- nothing is
      seeded from a merely similar state, so a later projection never starts
      from a lie;
    * A is the LOCAL canonical text (the bytes on disk after the write), P the
      REMOTE canonical document, and the revision is the SERVER's opaque ETag --
      client-side facts only where they ARE the server's own;
    * the inbound integrated baseline ``B`` is NOT touched (seeding is not an
      integration);
    * a public origin is not special-cased here: when L == R for a public link
      the local text IS the published text, and the egress GRANT still gates the
      push itself.
    """

    if c.remote is None or not c.remote_revision or local_fields is None:
        return False
    try:
        local_fp = prov.fingerprint_snapshot(c.kind, _canonical_for_compare(c.kind, local_fields))
        remote_fp = prov.fingerprint_snapshot(c.kind, _canonical_for_compare(c.kind, c.remote))
    except Exception:  # noqa: BLE001 - an unfingerprintable state simply earns no seed
        return False
    if local_fp != remote_fp:
        return False
    record = prov.read_baseline(ctx.config_dir, c.kind, c.local_id)
    if record is None or record.identity is None:
        return False
    # LINEAGE GATE (round-3 QN9): a receipt is a durable claim that "this text
    # was accepted for this origin", so it must not be minted from content whose
    # provenance is unestablished or crosses origins. Mixed-lineage never seeds
    # (any scope); a PUBLIC origin seeds only from clean lineage, because a
    # receipt is what a later unattended push would project -- an unknown-lineage
    # public seed could be weaponised by a later flip. A non-public origin may
    # seed from unknown lineage: egress there never leaves the organization, and
    # the trusted-origin integration that produced the agreement is the same
    # evidence the lifecycle records.
    if record.lineage == "mixed-lineage":
        return False
    if record.identity.visibility == "public" and record.lineage != "clean":
        return False
    from dataclasses import replace as dc_replace

    from local_operator.hub_sync.push import receipt_for

    receipt = receipt_for(c.kind, c.remote, dict(local_fields), revision=c.remote_revision)
    try:
        prov.write_baseline(ctx.config_dir, dc_replace(record, outbound=receipt))
    except (prov.BaselineError, OSError):
        logger.debug("could not seed outbound receipt for %s %s", c.kind, c.local_id, exc_info=True)
        return False
    return True


def record_publish_origin(
    config_dir: Any,
    kind: Kind,
    local_id: str,
    *,
    hub_id: str,
    origin: LinkOrigin | None,
    accepted: Mapping[str, Any] | None = None,
) -> bool:
    """Attach the origin an ACCEPTED publication proved.

    A publication is the strongest proof of origin there is: the user named the
    destination and the hub accepted the row. The SCOPE comes from the TARGET
    the user chose, never from the accepted document -- a publicly published
    row carries its publisher's ``tenant_id`` (measured on the live hub on
    2026-10-19), so a scope read from the document would label a public
    publication as an organization link and skip the public egress grant in
    ``_eligibility``.

    A scope that conflicts with the record is REFUSED (``origin-conflict``),
    never silently relinked: the record's own tenant has to be the one the
    target names, and the record's hub id has to be the row that was
    published. An already matching identity returns True without rewriting it;
    False refuses all downstream receipt mutation. A first publication creates
    a baseline from the accepted canonical comparable fields, never local text.

    ``False`` does not undo the hub publication: it refuses local bookkeeping
    so the original link and its outbound ancestry remain untouched.
    """

    if origin is None or not hub_id:
        return False
    record = prov.read_baseline(config_dir, kind, local_id)
    if record is None:
        if not accepted:
            return False
        # The accepted canonical document, not the local authored copy, is the
        # inbound ancestor. Generalization may have changed it; using L as B
        # would allow a future pull to overwrite unintegrated local edits.
        fields = (
            prov.team_fields(
                description=str(accepted.get("description") or ""),
                manager=str(accepted.get("manager") or ""),
                members=accepted.get("members"),
                instructions=str(accepted.get("instructions") or ""),
                project=str(accepted.get("project") or ""),
            )
            if kind == "team"
            else {
                "instructions": prov.norm(accepted.get("instructions")),
                "description": prov.norm(accepted.get("description")),
            }
        )
        record = prov.make_record(kind, local_id, hub_id, origin.tenant_id, fields, "publish")
    if record.hub_id != hub_id or record.tenant_id != origin.tenant_id:
        logger.warning(
            "the published scope for %s %s does not match the linked row; leaving it untouched",
            kind,
            local_id,
        )
        return False
    identity = prov.link_identity(kind, local_id, hub_id, origin=origin)
    if identity is None:
        return False
    if record.identity is not None:
        if record.identity.link_id != identity.link_id:
            logger.warning(
                "the published scope for %s %s conflicts with its confirmed origin; "
                "leaving it untouched",
                kind,
                local_id,
            )
            return False
        # An already matching origin is compatible, not a refusal: republish
        # may advance its receipt. A different origin must never touch ancestry.
        return True
    try:
        prov.write_baseline(config_dir, replace(record, identity=identity))
    except (prov.BaselineError, OSError, ValueError):
        logger.warning(
            "could not record the published origin for %s %s; the link stays unresolved",
            kind,
            local_id,
            exc_info=True,
        )
        return False
    return True


def published_hub_id(kind: Kind, result: Mapping[str, Any]) -> str | None:
    """The hub id an accepted publication created or updated.

    The hub's own answer: the publication result names the row it wrote
    (``agent_id`` for an agent, ``team.id`` for a team), and the accepted
    document repeats it. ``None`` when the result names neither -- the caller
    then records the row without an origin rather than guessing an id.
    """

    if not isinstance(result, Mapping):
        return None
    if kind == "team":
        team = result.get("team")
        if isinstance(team, Mapping):
            value = str(team.get("id") or "").strip()
            if value:
                return value
    else:
        value = str(result.get("agent_id") or "").strip()
        if value:
            return value
    accepted = result.get("accepted_document")
    if isinstance(accepted, Mapping):
        value = str(accepted.get("id") or "").strip()
        if value:
            return value
    return None


def seed_receipt_after_publish(
    config_dir: Any,
    kind: Kind,
    local_id: str,
    *,
    accepted: Mapping[str, Any],
    local_fields: Mapping[str, Any],
    revision: str | None,
    origin: LinkOrigin | None = None,
    hub_id: str | None = None,
) -> bool:
    """Seed A/P after an ACCEPTED manual publish/republish (parent ruling (A)b).

    The server returned the canonical document and its ETag; A is the local input
    that was published and P the accepted document. Nothing client-side is
    invented: without a revision (an older server) nothing is seeded and the link
    stays honestly ``no-receipt``. An existing inbound baseline B is untouched;
    a first publication creates B from the accepted canonical fields.

    The ORIGIN the acceptance proved is recorded as well -- the base and the
    scope the user chose -- so a published row is a CONFIRMED link instead of
    one the inventory reports as unresolved. It is recorded independently of the
    receipt: a server that answers no ETag still proved where the row went.
    """

    if origin is not None:
        # Compatibility is a fence for BOTH mutations. A rejected destination's
        # accepted bytes/ETag must never become this link's outbound ancestry.
        if not hub_id or not record_publish_origin(
            config_dir, kind, local_id, hub_id=hub_id, origin=origin, accepted=accepted
        ):
            return False
    if not revision:
        return False
    from dataclasses import replace as dc_replace

    from local_operator.hub_sync.push import receipt_for

    record = prov.read_baseline(config_dir, kind, local_id)
    if record is None:
        return False
    receipt = receipt_for(kind, accepted, dict(local_fields), revision=revision)
    try:
        prov.write_baseline(config_dir, dc_replace(record, outbound=receipt))
    except (prov.BaselineError, OSError):
        return False
    return True


def _adopt_if_equal(ctx: HubSyncContext, c: chk.ItemCheck) -> None:
    """L == R with no baseline: the texts agree, so B := L (A2.3.2)."""

    if c.verdict != "up-to-date" or c.base is not None or c.local is None:
        return
    record = prov.make_record(c.kind, c.local_id, c.hub_id, c.tenant_id, c.local, "adopt")
    identity = _carried_identity(ctx, c)
    if identity is not None:
        record = replace(record, identity=identity)
    try:
        prov.write_baseline(ctx.config_dir, record)
    except (prov.BaselineError, OSError):
        logger.debug("could not adopt baseline for %s %s", c.kind, c.local_id, exc_info=True)
        return
    # The adopted baseline IS an agreed point (L == R by construction here), so
    # this is where a freshly discovered/never-pulled link first becomes able to
    # auto-push. Seeding never gates the adopt itself.
    _seed_receipt_if_agreed(ctx, c, c.local)


def resolve_source(
    ctx: HubSyncContext,
    *,
    kind: Kind,
    name: str,
    hub_source: str,
    visibility: Visibility,
    tenant_id: str | None,
) -> prov.BaselineRecord:
    """Confirm SOURCE+SCOPE for an unknown-origin (legacy) linked item.

    The adopt paths elsewhere materialize a baseline only when the texts agree;
    a legacy row that carries a hub id but no recorded source/scope is otherwise
    stuck at "unknown baseline" forever. This arm is the explicit act that
    resolves it: it names the destination source+scope, validates it against the
    configured credential-free bases the caller already vouches for (the route
    only supplies a registered source), and materializes the immutable identity
    plus a guarded baseline.

    Refusals are the point: a known scope that differs (Public vs another Org),
    or a differing hub id, is refused with ``origin-conflict`` and the prior
    bytes are left untouched -- this route can never silently relink an artifact
    or invent a ``link_id``. The real ``link_id`` comes only from the canonical
    identity tuple (``LinkIdentity.link_id``).

    A record a real pull/merge already wrote is preserved byte-for-byte apart
    from identity/scope: ownership, lineage, the remote revision, the outbound
    receipt AND the merge-base payload/fingerprint all survive -- rebuilding
    the payload from current local content would launder unintegrated edits
    into the base (review R2). The unknown marker itself stays unknown,
    because a resolve is not an integration.
    """
    from local_operator.agents import hub_origin

    # Typed refusal for a kind this surface does not resolve: resolving a NAME to
    # a source+scope is implemented for agent/team rows only; a silent
    # fallthrough routed skills through the teams registry.
    if kind not in ("agent", "team"):
        raise LinkError("unsupported-kind", f"{kind} links cannot be resolved by name")

    registry = ctx.agents() if kind == "agent" else ctx.teams()
    row = registry.get_agent_by_name(name) if kind == "agent" else registry.get_team_by_name(name)
    if row is None:
        # BEFORE any field read: a missing row must be a 404-able LookupError,
        # never an AttributeError from reading fields off None.
        raise LookupError(f"no linked {kind} named {name!r}")
    local_id = str(row.id)
    if kind == "agent":
        hub_id = hub_origin(row)
    else:
        # A team's remote id lives in its baseline record (a linked team HAS one),
        # never on the row itself.
        linked = prov.read_baseline(ctx.config_dir, kind, local_id)
        hub_id = linked.hub_id if linked is not None else None
    if hub_id is None:
        raise LookupError(f"no linked {kind} named {name!r}")
    fields = _local_fields_for(ctx, kind, row)

    identity = LinkIdentity(kind, local_id, hub_source, visibility, tenant_id, hub_id)
    existing = prov.read_baseline(ctx.config_dir, kind, local_id)
    if existing is not None and existing.identity is not None:
        # One writable origin per artifact: a differing source/scope/remote id
        # refuses here rather than relinking, and the old record is untouched.
        existing.identity.require_same_origin(identity)
    if (
        existing is not None
        and existing.recorded_by != prov.UNKNOWN_BASELINE
        and existing.hub_id == hub_id
    ):
        # A known baseline: REPLACE identity/scope on the ORIGINAL record.
        # Rebuilding the payload from current local content would make the
        # user's unintegrated edit the merge base -- the next unchanged-remote
        # apply then classifies as remote-only and erases it (review R2,
        # reproduced against the identical-code no-resolve control).
        record = replace(existing, identity=identity, scope_known=True, tenant_id=tenant_id or None)
    else:
        record = prov.make_record(kind, local_id, hub_id, tenant_id, fields, "adopt-unknown")
        record = replace(record, identity=identity, scope_known=True, lineage="unknown")
    prov.write_baseline(ctx.config_dir, record)
    return record


def _local_fields_for(ctx: HubSyncContext, kind: str, row: Any) -> Mapping[str, Any]:
    """The LOCAL fields a baseline records, matching the check/merge field maps."""
    if kind == "agent":
        return {
            "instructions": ctx.agents().get_agent_system_prompt(row.id) or "",
            "description": str(row.description or ""),
        }
    return chk.team_fields_of(row)


def check_items(
    ctx: HubSyncContext,
    *,
    kinds: Sequence[str] = ("agent", "team", "skill"),
    names: Sequence[str] | None = None,
    only_ids: set[str] | None = None,
    record: bool = True,
) -> list[chk.ItemCheck]:
    """Network check of linked items; folds the outcome into the store."""

    out: list[chk.ItemCheck] = []
    if "agent" in kinds:
        out.extend(
            chk.check_hub_agents(
                ctx.agents(),
                client_for_tenant=ctx.client_for_tenant,
                names=names,
                only_ids=only_ids,
            )
        )
    if "team" in kinds:
        out.extend(
            chk.check_hub_teams(
                ctx.teams(), client_for_tenant=ctx.client_for_tenant, names=names, only_ids=only_ids
            )
        )
    if "skill" in kinds:
        out.extend(
            chk.check_hub_skills(
                config_dir=ctx.config_dir,
                client_for_tenant=ctx.client_for_tenant,
                names=names,
                only_ids=only_ids,
            )
        )
    for c in out:
        _adopt_if_equal(ctx, c)
    # THE ORIGIN REPAIR: a baseline written before links carried an origin has
    # no identity, and the check that just fetched the row is the proof of where
    # it came from. It runs after the adopts (which can mint the record the
    # repair then confirms) and before the store fold, so the inventory the
    # caller reads already carries the confirmed link.
    source = prov.canonical_hub_source(ctx.config_manager)
    if source:
        for c in out:
            attach_served_origin(ctx, c, source=source)
    if record:
        live = None
        if names is None and only_ids is None and set(kinds) == {"agent", "team"}:
            live = {st.item_key(c.kind, c.local_id) for c in out}
        record_checks(ctx, out, live=live)
    return out


# -- merge -------------------------------------------------------------------------------


def attach_served_origin(ctx: HubSyncContext, c: chk.ItemCheck, *, source: str | None) -> bool:
    """Attach the origin a hub-served row actually came from (the repair).

    A baseline written before links carried an origin has no identity, and an
    identity-less row is a dead end: the inventory lists it as "source
    unresolved" and the desktop draws a "?" badge for a row this machine
    pulled from a source the operator trusts. The HUB proves the serving base;
    only an anonymous fetch of a tenant-free known-scope record proves public
    visibility. An authenticated detail fetch does not prove org visibility.

    Guards -- every one of them a REFUSAL, never a relink:

    * only a record that has NO identity (one attachment per row);
    * only a row the hub ACTUALLY served (an unavailable check proves
      nothing, so a row the hub no longer serves stays unresolved);
    * only when the serving base is the CONFIGURED one (a row answered by
      some other destination says nothing about this hub);
    * only a tenant-free record fetched anonymously proves public scope;
      historical tenant metadata is ambiguous (public pulls stored publisher
      tenants) and must remain unresolved until explicitly confirmed.
    """

    if not source or c.verdict == "unavailable" or not c.served_by:
        return False
    if c.served_by != source:
        return False
    record = prov.read_baseline(ctx.config_dir, c.kind, c.local_id)
    if record is None or record.identity is not None:
        return False
    if record.hub_id != c.hub_id or not record.scope_known:
        return False
    # Pre-fix public pulls stored the publisher's tenant here. An optional-auth
    # fetch cannot distinguish that from an organization row, so certifying it
    # would bypass public consent. Preserve the immutable scope and require an
    # explicit operator confirmation; only an anonymous read proves public scope.
    if record.tenant_id:
        logger.info("leaving ambiguous legacy hub scope unresolved for %s %s", c.kind, c.local_id)
        return False
    origin = link_origin(source, "public", None)
    identity = prov.link_identity(c.kind, c.local_id, record.hub_id, origin=origin)
    if identity is None:
        return False
    try:
        prov.write_baseline(ctx.config_dir, replace(record, identity=identity))
    except (prov.BaselineError, OSError, ValueError):
        logger.warning(
            "could not attach the hub origin to %s %s; the link stays unresolved",
            c.kind,
            c.local_id,
            exc_info=True,
        )
        return False
    return True


def _specs(kind: str) -> list[tuple[str, str, int | None, int | None]]:
    """``(field, merge kind, max_chars, max_items)`` per comparable field."""

    if kind == "team":
        return [
            ("description", "text", TEAM_DESCRIPTION_MAX, None),
            ("manager", "scalar", TEAM_MANAGER_MAX, None),
            ("members", "roster", None, TEAM_ROSTER_MAX),
            ("instructions", "markdown", TEAM_BRIEF_MAX, None),
            ("project", "markdown", TEAM_BRIEF_MAX, None),
        ]
    return [
        ("instructions", "markdown", AGENT_INSTRUCTIONS_MAX, None),
        ("description", "text", None, None),
    ]


def _resolver_for(ctx: HubSyncContext, allow_llm: bool) -> Resolver | None:
    if not allow_llm:
        return None
    if ctx.resolver is not None:
        return ctx.resolver
    from local_operator.hub_sync.resolver import LlmResolver

    return LlmResolver(ctx.config_manager)


def _empty_of(kind: str) -> Any:
    return [] if kind == "roster" else ""


def _field_inputs(c: chk.ItemCheck) -> list[tuple[FieldInput, int | None, int | None]]:
    out = []
    for name, kind, max_chars, max_items in _specs(c.kind):
        base = None
        if c.base is not None:
            base = c.base.get(name, _empty_of(kind))
        out.append(
            (
                FieldInput(
                    field=name,
                    kind=kind,  # type: ignore[arg-type]
                    base=base,
                    local=(c.local or {}).get(name, _empty_of(kind)),
                    remote=(c.remote or {}).get(name, _empty_of(kind)),
                ),
                max_chars,
                max_items,
            )
        )
    return out


def _summary(results: Sequence[MergeResult]) -> dict[str, int]:
    total: dict[str, int] = {}
    for r in results:
        for k, v in r.counts.items():
            total[k] = total.get(k, 0) + v
    return {k: v for k, v in total.items() if k != "unchanged"} | {
        "unresolved": total.get("unresolved", 0)
    }


def _failure_class(results: Sequence[MergeResult]) -> str | None:
    for r in results:
        if r.engine.failure_class:
            return r.engine.failure_class
    return None


def merge_item(
    ctx: HubSyncContext,
    c: chk.ItemCheck,
    *,
    prefer: Prefer = "none",
    acknowledge_unknown_baseline: bool = False,
    replace: Literal[None, "remote", "local"] = None,
    allow_llm: bool = True,
) -> ItemMergeReport:
    """Compute (never write) the merge of one checked item."""

    resolver = _resolver_for(ctx, allow_llm)
    results: list[MergeResult] = []
    for inp, max_chars, max_items in _field_inputs(c):
        if replace is not None:
            results.append(replace_field(inp, take=replace))
        else:
            results.append(
                merge_field(
                    inp,
                    MergeOptions(
                        prefer=prefer,
                        allow_llm=allow_llm,
                        acknowledge_unknown_baseline=acknowledge_unknown_baseline,
                        max_chars=max_chars,
                        max_items=max_items,
                        resolver=resolver,
                    ),
                )
            )
    warnings = tuple(dict.fromkeys(w for r in results for w in r.warnings))
    outcomes = {r.outcome for r in results}
    replaced: dict[str, Any] = {}
    message = ""
    error_class: str | None = None
    if "refused" in outcomes:
        outcome = "refused"
        first = next(r for r in results if r.outcome == "refused")
        message = f"{first.field} {first.refusal}"
        error_class = "merge-refused"
    elif "needs-review" in outcomes:
        outcome = "needs-review"
        failure = _failure_class(results)
        if failure and failure not in ("invalid-output",):
            error_class = failure
        else:
            error_class = "merge-refused"
        if c.base is None and not acknowledge_unknown_baseline:
            message = (
                "the baseline is unknown (this copy was edited before hub tracking "
                "began), so a deliberate deletion cannot be told from a hub addition — "
                "not applied. Re-run with --accept-unknown-baseline to take hub additions "
                "(nothing of yours is deleted)."
            )
    elif "merged" in outcomes:
        outcome = "merged"
        if replace == "remote":
            replaced = {
                r.field: (c.local or {}).get(r.field) for r in results if r.outcome == "merged"
            }
    else:
        outcome = "unchanged"
    if outcome == "merged" and c.kind == "team":
        warnings += _missing_roles(ctx, results)
    return ItemMergeReport(
        kind=c.kind,
        local_id=c.local_id,
        name=c.name,
        hub_id=c.hub_id,
        outcome=outcome,
        fields=tuple(results),
        applied=False,
        error_class=error_class,
        message=message,
        warnings=warnings,
        replaced=replaced,
        classification=c.classification.state if c.classification else None,
    )


def _deletion_covered(report: ItemMergeReport) -> bool:
    """True when every field's shrink is explained by deterministic removals.

    The item-level mirror of :func:`merge.deletion_covered`: an item is covered
    only if EVERY merged field is (a single unexplained field keeps the refusal).
    """
    from local_operator.hub_sync.merge import deletion_covered

    merged = [r for r in report.fields if r.outcome == "merged"]
    return bool(merged) and all(deletion_covered(r) for r in merged)


def _missing_roles(ctx: HubSyncContext, results: Sequence[MergeResult]) -> tuple[str, ...]:
    """A roster naming a role nobody has installed is a warning, never a refusal (A7)."""

    roster = next((r for r in results if r.field == "members" and r.outcome == "merged"), None)
    if roster is None or not isinstance(roster.merged, list):
        return ()  # a roster is always a list; the other arms only narrow the type
    slots = roster.merged
    try:
        from local_operator.agent_profiles import list_seeds

        known = {a.name.casefold() for a in ctx.agents().list_agents()} | {
            s.casefold() for s in list_seeds()
        }
    except Exception:  # noqa: BLE001 - a warning must never block an apply
        return ()
    return tuple(
        f"missing-role:{slot['role']}"
        for slot in slots
        if slot["kind"] == "agent" and slot["role"].casefold() not in known
    )


# -- apply -------------------------------------------------------------------------------


def _merged_fields(c: chk.ItemCheck, report: ItemMergeReport) -> dict[str, Any]:
    fields = dict(c.local or {})
    for r in report.fields:
        if r.outcome == "merged":
            fields[r.field] = r.merged
    return fields


def _would_empty_a_text_field(c: chk.ItemCheck, report: ItemMergeReport) -> bool:
    """True when a merged TEXT field is empty although the baseline still has text.

    The complement of the survivor proof: removals the engine EXPLAINED are still
    removals, and auto-applying one that empties a field the hub published would
    discard text nobody signed off on. Roster/scalar fields are not text loss and
    keep their own rules.
    """

    if not isinstance(c.base, Mapping):
        return False
    merged = _merged_fields(c, report)
    for name, merge_kind, _max_chars, _max_items in _specs(c.kind):
        if merge_kind not in ("text", "markdown"):
            continue
        base_value = c.base.get(name)
        if not (isinstance(base_value, str) and base_value.strip()):
            continue
        if not str(merged.get(name) or "").strip():
            return True
    return False


def _write_agent(ctx: HubSyncContext, c: chk.ItemCheck, merged: Mapping[str, Any]) -> None:
    from local_operator.agents import _apply_hub_update

    registry = ctx.agents()
    agent = registry.get_agent(c.local_id)
    # Design Q4: agents have no persistence lock, so this fingerprint re-check is
    # a narrow-window guard, not an airtight one. Accepted for this PR.
    current = chk._agent_fields(
        registry.get_agent_system_prompt(c.local_id), str(agent.description or "")
    )
    if prov.fingerprint_agent(current["instructions"], current["description"]) != c.local_fp:
        raise ConcurrentEdit("the agent changed while the merge was being computed")
    remote = c.remote or {}
    _apply_hub_update(
        registry,
        agent,
        c.hub_id,
        merged["instructions"],
        merged["description"],
        baseline=(remote.get("instructions", ""), remote.get("description", "")),
    )


def _carried_members(members: Any, current: Any) -> list[Any]:
    """``members`` with the stored row's overrides copied onto its surviving slots.

    The merged roster is the HUB's shape -- ``{role, kind, count}``
    (:func:`hub_sync.provenance.normalize_roster`) -- so rebuilding the members
    from it would silently WIPE every local ``tier``/``effort`` the operator had
    pinned: the merge touched the roster, the overrides never ride the wire, and
    nothing else would put them back. Copied by ``(kind, role)``, first match,
    the same lookup a launch resolves with.

    ``current`` is the row this happens against, and the caller re-runs this with
    the row RE-READ UNDER THE WRITER LOCK: the merge's own fingerprint describes
    the roster and ``normalize_roster`` drops overrides, so an override edited
    inside the merge window would otherwise be overwritten by the older value
    with nothing to notice. That second pass receives this function's own OUTPUT,
    so it accepts a slot either as the hub's mapping or as a built member -- the
    two passes must be idempotent.
    """
    from local_operator.teams import TeamMember, find_member

    rebuilt: list[Any] = []
    for slot in members or []:
        member = slot if isinstance(slot, TeamMember) else TeamMember(**slot)
        stored = find_member(getattr(current, "members", None), member.role, kind=member.kind)
        if stored is not None and not stored.is_default():
            member = member.model_copy(update={"tier": stored.tier, "effort": stored.effort})
        rebuilt.append(member)
    return rebuilt


def _write_team(ctx: HubSyncContext, c: chk.ItemCheck, report: ItemMergeReport) -> None:
    from local_operator.teams import TeamEditFields

    # The merged values verbatim: for ``members`` that is the HUB's shape
    # (``{role, kind, count}``), which ``TeamEditFields`` coerces into members
    # with no overrides. The carry-forward happens in ``precondition`` below,
    # where the row has been re-read under the writer lock.
    payload: dict[str, Any] = {r.field: r.merged for r in report.fields if r.outcome == "merged"}
    fields = TeamEditFields(**payload)

    def precondition(current: Any) -> None:
        now_fp = prov.fingerprint_team(chk.team_fields_of(current))
        if now_fp != c.local_fp:
            raise ConcurrentEdit("the team changed while the merge was being computed")
        # ``precondition`` is the only hook that sees the row RE-READ UNDER THE
        # WRITER LOCK and still runs before anything is staged, so the roster
        # the overrides carry forward FROM is read here rather than outside: a
        # read taken while computing the merge leaves a window in which an
        # override edited since is silently replaced by the older value, and the
        # fingerprint above cannot catch it -- it describes the roster, and
        # ``normalize_roster`` drops overrides.
        if "members" in payload:
            fields.members = _carried_members(payload["members"], current)

    ctx.teams().update_team(c.local_id, fields, precondition=precondition)


def _carried_identity(ctx: HubSyncContext, c: chk.ItemCheck) -> LinkIdentity | None:
    """The identity a baseline rewrite has to carry forward.

    A rewrite that dropped it would turn a CONFIRMED link back into an
    identity-less row: the links inventory would list it as "source
    unresolved" again and the "?" badge would come back for a row the
    operator already confirmed. The record on disk is the authority -- the
    same shape the skill merge carries forward.
    """

    try:
        current = prov.read_baseline(ctx.config_dir, c.kind, c.local_id)
    except (prov.BaselineError, OSError):
        return None
    return current.identity if current is not None else None


def _advance_baseline(ctx: HubSyncContext, c: chk.ItemCheck, recorded_by: str) -> None:
    """B := the remote text just integrated (see ``merge`` docstring).

    The record also carries the OPAQUE revision the remote was read at
    (``c.remote_revision``) when the check exposed one: it is the fallback
    expected token a later conditional write sends, and ``write_baseline``
    preserves the last known one when a caller cannot supply a fresh value.
    """

    if c.remote is None:
        return
    record = prov.make_record(c.kind, c.local_id, c.hub_id, c.tenant_id, c.remote, recorded_by)
    if c.remote_revision:
        record = replace(record, remote_revision=c.remote_revision)
    identity = _carried_identity(ctx, c)
    if identity is not None:
        record = replace(record, identity=identity)
    prov.write_baseline(ctx.config_dir, record)
    _mark_integrated(ctx, c, integrated=True)


def _agreement_is_genuine(c: chk.ItemCheck) -> bool:
    """Whether the CHECK itself proves genuine canonical L == R (round-3 delta R21).

    The verdict alone is NOT that proof: the check maps classification
    ``local-only`` (L != R, R == B -- a locally edited row) to the SAME
    ``up-to-date`` verdict an equal pair gets, and an equality read that could
    not expose a revision (the zip fallback) gives a conditional write nothing
    to stand on. A genuine agreement is exactly: the classification says
    ``up-to-date``, the fingerprints are equal, and a live revision exists.
    """

    state = c.classification.state if c.classification is not None else None
    return bool(
        state == "up-to-date"
        and c.local_fp is not None
        and c.local_fp == c.remote_fp
        and c.remote_revision
    )


def _mark_integrated(
    ctx: HubSyncContext,
    c: chk.ItemCheck,
    *,
    integrated: bool = False,
    agreement: bool = False,
) -> None:
    """Record that remote content from the link's OWN origin integrated here.

    THE LINEAGE LIFECYCLE, in one place (round-3 QN7): ``resolve_source`` can
    only ever say ``unknown`` -- it proves who the origin is, not that content
    from it ever landed. A pull/apply (or a genuine L == R agreement with a live
    revision) that read through the link's OWN recorded base IS that landing, so
    the record advances to ``clean``. The CALLER must state which proof it has;
    no claim is refused, because an unqualified ``up-to-date`` verdict is not
    evidence (round-3 delta R21: ``local-only`` rides it too):

    * ``integrated=True`` -- this call WROTE remote-derived content (the
      baseline advance / the skill write). A real integration event.
    * ``agreement=True`` -- the caller proved genuine canonical L == R; the
      gate re-checks ``_agreement_is_genuine`` anyway (defense in depth), so a
      local-only verdict can never mark, lineage or revision, no matter which
      path arrives here.

    Guards, all refusals:

    * no identity, or no corroborated ``served_by`` -> no write;
    * a base that is not the recorded ``hub_source`` -> no write (an integration
      answered by another source says nothing about this origin);
    * ``mixed-lineage`` never advances (that is the state the guard exists for)
      and ``clean`` is left alone; only the unestablished states flip.

    This must NEVER launder: a local edit, a mixed-origin state, or an
    unlinked/local-only row has no path through here at all -- only the two
    trusted-origin events reach it, and both were read from the origin the
    identity names.
    """

    from dataclasses import replace as dc_replace

    if not integrated and not (agreement and _agreement_is_genuine(c)):
        return
    if not c.served_by:
        return
    record = prov.read_baseline(ctx.config_dir, c.kind, c.local_id)
    if record is None or record.identity is None:
        return
    if c.hub_id != record.hub_id or c.served_by != record.identity.hub_source:
        return
    updated = record
    if record.lineage not in ("clean", "mixed-lineage"):
        updated = dc_replace(updated, lineage="clean")
    if record.recorded_by == prov.UNKNOWN_BASELINE:
        # The resolve only LINKED the item to a source ("adopt-unknown" does not
        # vouch for its content); this agreement with the origin (content equal
        # to a live-revision read that came from the recorded base) IS the
        # verification A2.3 asks for, so the record advances to a vouching one.
        updated = dc_replace(updated, recorded_by="adopt")
    if c.remote_revision and c.remote_revision != updated.remote_revision:
        # The revision is the ancestry fact this agreement observed, and the
        # fallback expected token a later conditional write uses.
        updated = dc_replace(updated, remote_revision=c.remote_revision)
    if updated == record:
        return
    try:
        prov.write_baseline(ctx.config_dir, updated)
    except (prov.BaselineError, OSError, LinkError):
        logger.debug("could not record integration lineage for %s %s", c.kind, c.local_id)


def _local_fields_for_backup(c: chk.ItemCheck) -> dict[str, Any]:
    return dict(c.local or {})


def _apply_skill(
    ctx: HubSyncContext,
    c: chk.ItemCheck,
    *,
    prefer: Prefer = "none",
    replace: Literal[None, "remote", "local"] = None,
    dry_run: bool = False,
    auto: bool = False,
    allow_llm: bool = True,
) -> ItemMergeReport:
    """Merge and WRITE one skill through the journaled managed-skill storage.

    A skill's local copy is a managed alias directory, so the write path is the
    crash-safe staged swap (journal + backup + concurrent-edit re-verification +
    policy CAS) that ``ManagedSkillStorage`` owns -- NEVER a bare filesystem
    write and never a generic-field write. The baseline advances ONLY inside that
    same transaction (the storage finalizes it), which is what makes the old
    "baseline moved, bytes did not" failure impossible.

    Merge model (frozen): local == baseline fast-forwards to the remote document;
    both sides changed -> per-FILE three-way merge through the shared engine;
    any unresolved file -> a conflict report with NOTHING written and the
    baseline NOT advanced.
    """
    from local_operator.hub_sync.outbound import merge_skill_document
    from local_operator.hub_sync.skill_document import (
        SkillDocument,
        SkillValidationError,
    )
    from local_operator.hub_sync.skill_storage import ManagedSkillStorage
    from local_operator.hub_sync.transactions import TransactionError

    key = st.item_key(c.kind, c.local_id)

    def fold(fn: Callable[[dict[str, Any]], None]) -> None:
        def inner(doc: dict[str, Any]) -> None:
            item = doc.setdefault("items", {}).get(key)
            if item is not None:
                fn(item)

        ctx.store().mutate(inner)

    def fail(cls: str, message: str, outcome: ItemOutcome = "failed") -> ItemMergeReport:
        fold(lambda item: st.record_failure(item, cls, message))
        return ItemMergeReport(
            c.kind,
            c.local_id,
            c.name,
            c.hub_id,
            outcome,
            error_class=cls,
            message=message,
        )

    record = prov.read_baseline(ctx.config_dir, "skill", c.local_id)
    if record is None or record.identity is None or record.local_alias is None:
        # A legacy/unknown-origin skill link has no writer identity or alias:
        # refusing here is exactly what keeps a baseline from advancing without
        # a write (the invariant the QA repro tripped over).
        return fail("baseline-unknown", "this skill link has no writer identity yet", "refused")
    alias = record.local_alias
    # ABSENT CONTENT IS NOT A DIFF. The recorded state (not a missing directory)
    # is the authority: an explicitly absent link has nothing to compare or
    # merge, so an apply installs the remote document WHOLESALE -- the same
    # shape ``install_skill`` writes for a fresh link -- and never reports
    # ``unchanged`` or computes a three-way merge against bytes that are not
    # there.
    #
    # The CHECK's fact is carried too: if a pull landed between the check and
    # this apply, the record is present but this row still has no local side.
    # Installing wholesale is safe there because the storage re-verifies under
    # its lock (``expected_fingerprint`` is None, so existing bytes refuse with
    # ``local-concurrent-edit``) -- never a bogus "invalid skill document".
    absent = c.content_state == "absent" or record.content_state == "absent"

    try:
        remote_doc = SkillDocument.from_json(dict(c.remote or {}))
        local_doc = None if absent else SkillDocument.from_json(dict(c.local or {}))
        base_doc = None if absent else (SkillDocument.from_json(dict(c.base)) if c.base else None)
    except SkillValidationError as exc:
        return fail("merge-refused", f"the skill document is invalid: {exc}", "refused")

    results: list[MergeResult] = []
    if absent:
        if replace == "local":
            # "Keep my side" has no side to keep: pretending otherwise would
            # install the hub's bytes under a request that asked for the
            # opposite. Refuse by name and write nothing.
            return fail(
                "merge-refused",
                "there is no local copy to keep: this link is subscribed without content",
                "refused",
            )
        merged: SkillDocument | None = remote_doc
    elif replace == "remote":
        merged = remote_doc
    elif replace == "local":
        merged = local_doc
    else:
        if local_doc is None:
            # Unreachable: an absent row took the wholesale branch above. The
            # guard keeps a three-way merge from ever running against nothing.
            return fail("merge-refused", "the local skill document is missing", "refused")
        merged_json, results = merge_skill_document(
            base_doc.to_json() if base_doc is not None else None,
            local_doc.to_json(),
            remote_doc.to_json(),
            opts=MergeOptions(
                prefer=prefer,
                allow_llm=allow_llm,
                resolver=_resolver_for(ctx, allow_llm),
            ),
        )
        merged = None
        if merged_json is not None:
            merged = SkillDocument.from_json(dict(merged_json))
    if merged is None:
        message = (
            "the merge left files neither side can keep (an edit against a removal)"
            if results
            else "the merged skill document is not valid"
        )
        return fail("merge-refused", message, "needs-review")

    # Source-name space comparison: nothing to write when the merge reproduced the
    # local document exactly (the remote and local already agree on content).
    # ``local_doc`` is None exactly for an absent row, which has nothing to
    # reproduce.
    if local_doc is not None and merged.to_json() == local_doc.to_json() and replace is None:
        if not dry_run:
            # This site compared the FULL documents (stronger than the check's
            # fingerprints), but the mark still goes through the agreement gate:
            # a classification that is not a genuine equal pair refuses.
            _mark_integrated(ctx, c, agreement=True)
            _seed_receipt_if_agreed(ctx, c, merged.to_json())
        fold(st.settle_unchanged)
        return ItemMergeReport(c.kind, c.local_id, c.name, c.hub_id, "unchanged")
    if dry_run:
        return ItemMergeReport(
            c.kind, c.local_id, c.name, c.hub_id, "would-merge", fields=tuple(results)
        )

    try:
        local_write = merged.with_name(alias)
        # The disk copy is alias-rendered; its fingerprint is what the storage
        # re-verifies under its lock (a local edit since this check refuses).
        # An absent link has no bytes, so the precondition is exactly None.
        before_fingerprint = None if local_doc is None else local_doc.with_name(alias).fingerprint
    except SkillValidationError as exc:
        return fail(
            "merge-refused", f"the merged skill cannot be rendered locally: {exc}", "refused"
        )

    baseline = prov.make_record(
        "skill", c.local_id, c.hub_id, c.tenant_id, dict(c.remote or {}), "merge-pull"
    )
    from dataclasses import replace as dc_replace

    baseline = dc_replace(
        baseline,
        identity=record.identity,
        local_alias=alias,
        remote_revision=c.remote_revision or record.remote_revision,
        # This write makes the copy exist: the state must say so atomically with
        # the content transaction's baseline finalize.
        content_state="present",
    )
    storage = ManagedSkillStorage(ctx.config_dir)
    try:
        storage.apply(
            record.identity,
            alias,
            local_write,
            baseline,
            expected_fingerprint=before_fingerprint,
            automatic=auto,
            basis="merge-pull",
        )
    except (LinkError, TransactionError) as exc:
        code = getattr(exc, "code", "hub-error")
        return fail(str(code), str(exc))
    except (ValueError, KeyError) as exc:
        return fail("merge-refused", f"the result was refused: {exc}", "refused")
    except Exception as exc:  # noqa: BLE001 - one item's failure never ends the run
        logger.warning("hub skill apply failed for %s", c.name, exc_info=True)
        return fail("hub-error", f"could not write the update: {exc}")

    # The merged document IS the remote's canonical text when the pull integrated
    # it (fingerprint equality decides); that is the skill's agreed point.
    _mark_integrated(ctx, c, integrated=True)
    _seed_receipt_if_agreed(ctx, c, merged.to_json())
    summary = _summary(results)
    fold(lambda item: st.settle_applied(item, summary, None))
    return ItemMergeReport(
        c.kind,
        c.local_id,
        c.name,
        c.hub_id,
        "merged",
        applied=True,
        fields=tuple(results),
    )


def apply_one(
    ctx: HubSyncContext,
    c: chk.ItemCheck,
    *,
    prefer: Prefer = "none",
    acknowledge_unknown_baseline: bool = False,
    replace: Literal[None, "remote", "local"] = None,
    dry_run: bool = False,
    auto: bool = False,
    allow_llm: bool = True,
) -> ItemMergeReport:
    """Merge one available item and, when allowed, write it. Folds the result into the store."""

    if c.verdict == "unavailable":
        return ItemMergeReport(
            c.kind,
            c.local_id,
            c.name,
            c.hub_id,
            "unavailable",
            error_class=c.reason,
            message=c.detail,
        )
    if c.kind == "skill" and c.content_state == "absent" and auto:
        # UNATTENDED NEVER INSTALLS. ``auto`` is the runner tick and the retry
        # composition; installing a subscribed-only skill is an explicit pull
        # (or a per-item apply), never a side effect of those paths. Writing a
        # copy the operator never asked for is the same class of surprise as
        # clobbering local edits.
        return ItemMergeReport(
            c.kind,
            c.local_id,
            c.name,
            c.hub_id,
            "skipped",
            skipped_reason="install-requires-pull",
        )
    # An EXPLICIT replace acts on a copy that differs from the hub even when the hub
    # has not moved (classification ``local-only``, which a plain pull reads as
    # up-to-date): "discard my edits, take the hub's" is the main reason anyone
    # asks for it, and the old ``--force`` did exactly that. Truly equal texts
    # (L == R) still have nothing to replace.
    replace_differs = (
        replace is not None
        and c.remote is not None
        and c.local is not None
        and c.local_fp != c.remote_fp
    )
    if c.verdict == "up-to-date" and not replace_differs:
        if not dry_run and _agreement_is_genuine(c):
            # GENUINE L == R with a live revision: the agreed point (round-3
            # QN7). A pull that finds nothing to change still IS the integration
            # event that establishes the origin's lineage and the (A, P) receipt
            # the first unattended push derives from. A ``local-only``
            # classification lands on the same verdict and is NOT this: it has
            # its own explicit-replace semantics and must never mark or seed
            # (round-3 delta R21).
            _mark_integrated(ctx, c, agreement=True)
            _seed_receipt_if_agreed(ctx, c, c.local)
        return ItemMergeReport(c.kind, c.local_id, c.name, c.hub_id, "up-to-date")

    if c.kind == "skill":
        # Skills write through the managed-skill transaction, not through the
        # generic-field path: see _apply_skill for the frozen merge model and the
        # baseline-advances-only-with-a-write invariant.
        return _apply_skill(
            ctx,
            c,
            prefer=prefer,
            replace=replace,
            dry_run=dry_run,
            auto=auto,
            allow_llm=allow_llm,
        )

    report = merge_item(
        ctx,
        c,
        prefer=prefer,
        acknowledge_unknown_baseline=acknowledge_unknown_baseline,
        replace=replace,
        allow_llm=allow_llm,
    )
    summary = _summary(report.fields)
    key = st.item_key(c.kind, c.local_id)

    def fold(fn: Callable[[dict[str, Any]], None]) -> None:
        def inner(doc: dict[str, Any]) -> None:
            item = doc.setdefault("items", {}).get(key)
            if item is not None:
                fn(item)

        ctx.store().mutate(inner)

    def fail(cls: str, message: str, outcome: ItemOutcome = "failed") -> ItemMergeReport:
        # Bind the input/config fingerprint on a prompt-too-long failure so the
        # cheap scheduled eligibility retry can refuse to re-send an unchanged
        # (still-too-long) request and only re-arm when it actually changed.
        fp = prov.fingerprint_snapshot(c.kind, c.local or {})

        def record(item: dict[str, Any]) -> None:
            st.record_failure(item, cls, message, summary=summary)
            if cls == "prompt-too-long":
                item["prompt_fingerprint"] = fp

        fold(record)
        return _with(report, outcome=outcome, error_class=cls, message=message)

    if report.outcome in ("needs-review", "refused"):
        return fail(
            report.error_class or "merge-refused",
            report.message or "the hub and your copy both changed the same text",
            outcome=report.outcome,
        )
    if auto and report.outcome == "merged":
        # Auto-update only ever applies a clean merge (B3.1): an UNEXPLAINED large
        # shrink or an unknown baseline waits for a human. A shrink fully explained
        # by deterministic ancestry removals is NOT unexplained loss and applies.
        if "large-shrink" in report.warnings and not _deletion_covered(report):
            return fail(
                "merge-refused", "the merged text is much shorter than both sides", "needs-review"
            )
        if _would_empty_a_text_field(c, report):
            # A two-sided complementary removal can legitimately reach '' while the
            # hub still holds text (each side removed what the other kept). That is
            # not the ratio-guard's "unexplained" loss, but auto-applying it would
            # silently discard hub-held text, so it is HELD for review; a manual
            # apply (auto=False) stays available.
            return fail(
                "merge-refused",
                "the merge would empty text the hub still has; held for review",
                "needs-review",
            )
        if c.base is None:
            return fail(
                "merge-refused", "baseline unknown; not applied automatically", "needs-review"
            )
    if report.outcome == "unchanged":
        if not dry_run:
            try:
                _advance_baseline(ctx, c, "merge-pull")
            except (prov.BaselineError, OSError):
                logger.debug("could not advance baseline", exc_info=True)
            _seed_receipt_if_agreed(ctx, c, c.local)
            fold(lambda item: st.settle_unchanged(item))
        return report
    if dry_run:
        return _with(report, outcome="would-merge")

    fold(st.mark_updating)
    try:
        backup = prov.write_backup(
            ctx.config_dir,
            c.kind,
            c.local_id,
            _local_fields_for_backup(c),
            reason="replace" if replace else "merge-pull",
        )
        if c.kind == "agent":
            _write_agent(ctx, c, _merged_fields(c, report))
        else:
            _write_team(ctx, c, report)
        _advance_baseline(ctx, c, "merge-pull")
        written = dict(c.local or {})
        for field_result in report.fields:
            if field_result.outcome == "merged" and field_result.merged is not None:
                written[field_result.field] = field_result.merged
        _seed_receipt_if_agreed(ctx, c, written)
    except ConcurrentEdit as exc:
        return fail("concurrent-edit", str(exc))
    except (ValueError, KeyError) as exc:
        return fail("merge-refused", f"the result was refused: {exc}", "refused")
    except Exception as exc:  # noqa: BLE001 - one item's failure never ends the run
        logger.warning("hub apply failed for %s %s", c.kind, c.name, exc_info=True)
        return fail("hub-error", f"could not write the update: {exc}")
    fold(lambda item: st.settle_applied(item, summary, backup))
    return _with(report, outcome="merged", applied=True, backup=backup)


def _with(report: ItemMergeReport, **changes: Any) -> ItemMergeReport:
    from dataclasses import replace as dc_replace

    return dc_replace(report, **changes)


def apply_items(
    ctx: HubSyncContext,
    *,
    kind: str | None = None,
    names: Sequence[str] | None = None,
    all_available: bool = False,
    prefer: Prefer = "none",
    acknowledge_unknown_baseline: bool = False,
    replace: Literal[None, "remote", "local"] = None,
    dry_run: bool = False,
    auto: bool = False,
    checks: Sequence[chk.ItemCheck] | None = None,
    allow_llm: bool = True,
) -> ApplyReport:
    """Check (unless ``checks`` is given) and apply, in B4.4 order.

    Agents first, then teams, each alphabetical. A per-item failure never stops
    the run EXCEPT a systemic class, after which the rest are ``skipped``. Update-all
    (``all_available``) cannot carry a conflict decision: ``prefer`` and ``replace``
    are ignored there so a batch can never discard anyone's work.
    """

    kwargs: dict[str, Any] = dict(
        kind=kind,
        names=names,
        all_available=all_available,
        prefer=prefer,
        acknowledge_unknown_baseline=acknowledge_unknown_baseline,
        replace=replace,
        dry_run=dry_run,
        auto=auto,
        checks=checks,
        allow_llm=allow_llm,
    )
    if dry_run:
        return _apply_items_locked(ctx, **kwargs)  # writes nothing, so no lease
    with exclusive(ctx):
        return _apply_items_locked(ctx, **kwargs)


def _apply_items_locked(
    ctx: HubSyncContext,
    *,
    kind: str | None,
    names: Sequence[str] | None,
    all_available: bool,
    prefer: Prefer,
    acknowledge_unknown_baseline: bool,
    replace: Literal[None, "remote", "local"],
    dry_run: bool,
    auto: bool,
    checks: Sequence[chk.ItemCheck] | None,
    allow_llm: bool,
) -> ApplyReport:
    if all_available:
        prefer, replace = "none", None
    kinds = (kind,) if kind else ("agent", "team", "skill")
    todo = list(checks) if checks is not None else check_items(ctx, kinds=kinds, names=names)
    # Agents first, then teams, then skills: the frozen batch order, extended by
    # one rank rather than replaced, so a client's existing expectations hold.
    todo.sort(key=lambda c: (KIND_ORDER.get(c.kind, len(KIND_ORDER)), c.name.casefold()))
    reports: list[ItemMergeReport] = []
    stopped: str | None = None
    for c in todo:
        if kind and c.kind != kind:
            continue
        if all_available and c.verdict != "available":
            continue
        if all_available and c.kind == "skill" and c.content_state == "absent":
            # UPDATE-ALL NEVER INSTALLS A SUBSCRIBED-ONLY SKILL. The row is
            # visible here because it is subscribed and the remote is readable,
            # but the batch action must not write a copy the operator never
            # asked for; installing one is an explicit per-item pull.
            reports.append(
                ItemMergeReport(
                    c.kind,
                    c.local_id,
                    c.name,
                    c.hub_id,
                    "skipped",
                    skipped_reason="content-absent",
                )
            )
            continue
        if stopped is not None and c.verdict == "available":
            reports.append(
                ItemMergeReport(
                    c.kind, c.local_id, c.name, c.hub_id, "skipped", skipped_reason=stopped
                )
            )
            continue
        report = apply_one(
            ctx,
            c,
            prefer=prefer,
            acknowledge_unknown_baseline=acknowledge_unknown_baseline,
            replace=replace,
            dry_run=dry_run,
            auto=auto,
            allow_llm=allow_llm,
        )
        reports.append(report)
        cls = report.error_class
        if cls and cls != "merge-refused":
            # ``error_class`` already carries the subclass ("provider-error/quota",
            # from ``failure_class``): appending it again produced
            # "provider-error/quota/quota", which is in no set, so a quota error
            # never stopped the run.
            if cls in st.SYSTEMIC:
                stopped = cls
    return ApplyReport(tuple(reports))


# -- status snapshot (store read, no network) --------------------------------------------


def status_snapshot(ctx: HubSyncContext) -> dict[str, Any]:
    """The ``GET /v1/desktop/hub/updates`` payload (B5.1): O(items), no network."""

    doc = ctx.store().load()
    settings = ctx.settings()
    now = datetime.now(timezone.utc)
    counts = {"available": 0, "failed": 0, "updating": 0, "up-to-date": 0, "applied": 0}
    items: list[dict[str, Any]] = []
    for item in doc.get("items", {}).values():
        kind = item.get("kind", "agent")
        if kind not in ("agent", "team", "skill"):
            # Skill failures are part of the updates surface now (the operator's
            # brief: a skill that cannot update must be visible and retryable
            # through the SAME channel as agents and teams), so this legacy
            # projection lists them beside the other two kinds. Unknown kinds
            # still never reach a typed consumer.
            continue
        state = st.effective_state(item, now)
        # A ROW THAT CARRIES AN ERROR IS FAILED IN THIS PROJECTION. The store
        # deliberately keeps a retryable row ``available`` so the timer can still
        # act on it, and the desktop sidebar compensates by also reading
        # ``error_class`` -- but every other consumer of this projection sees
        # only ``state`` and ``counts``. Deriving the failure HERE, for every
        # kind, means one rule rather than per-kind compensation: a carried
        # error is counted in ``counts.failed`` and a consumer can act on
        # ``state`` alone.
        #
        # ``no-credential`` is NOT a failure -- it is the sign-in fact an
        # up-to-date row is listed FOR (UX round 2, U11) -- and a row a writer
        # still holds in ``updating`` has not failed yet.
        error_class = item.get("error_class")
        if error_class and error_class != "no-credential" and state != "updating":
            state = "failed"
        counts[state] = counts.get(state, 0) + 1
        # An up-to-date item is listed only when it carries the sign-in fact
        # (``no-credential``): the user has no login for an item they have linked, and
        # this list is the only place the UI can learn it (UX round 2, U11).
        if state == "up-to-date" and item.get("error_class") != "no-credential":
            continue
        auto = settings.auto_for(kind)
        items.append(
            {
                "kind": kind,
                "name": item.get("name"),
                "local_id": item.get("local_id"),
                "hub_id": item.get("hub_id"),
                "tenant_id": item.get("tenant_id"),
                "state": state,
                "classification": item.get("classification"),
                "auto_will_apply": bool(
                    auto
                    and state == "available"
                    and item.get("classification") == "remote-only"
                    and item.get("baseline") == "known"
                ),
                "remote_fingerprint": item.get("remote_fingerprint"),
                "first_seen_available_at": item.get("first_seen_available_at"),
                "last_checked_at": item.get("last_checked_at"),
                "last_applied_at": item.get("last_applied_at"),
                "error_class": item.get("error_class"),
                "error_subclass": item.get("error_subclass"),
                "last_error": item.get("last_error"),
                "next_retry_at": item.get("next_retry_at"),
                "summary": item.get("summary") or {},
            }
        )
    items.sort(
        key=lambda i: (KIND_ORDER.get(i["kind"], len(KIND_ORDER)), str(i["name"]).casefold())
    )
    last = doc.get("last_tick") or {}
    return {
        "generated_at": prov.now_iso(),
        "credential": last.get("credential", ctx.credential),
        "settings": {
            "auto_agents": settings.auto_agents,
            "auto_teams": settings.auto_teams,
            "interval_min": settings.interval_min,
        },
        "counts": counts,
        "items": items,
    }


def find_check(checks: Sequence[chk.ItemCheck], kind: str, name: str) -> chk.ItemCheck | None:
    for c in checks:
        if c.kind == kind and c.name.casefold() == name.strip().casefold():
            return c
    return None


@dataclass
class TickOutcome:
    checked: int = 0
    available: int = 0
    applied: int = 0
    failed: int = 0
    reports: list[ItemMergeReport] = field(default_factory=list)


def check_report(c: chk.ItemCheck) -> ItemMergeReport:
    """A check-only view of one item (no merge computed)."""

    if c.verdict == "unavailable":
        return ItemMergeReport(
            c.kind,
            c.local_id,
            c.name,
            c.hub_id,
            "unavailable",
            error_class=c.reason,
            message=c.detail,
        )
    if c.verdict == "up-to-date":
        return ItemMergeReport(
            c.kind,
            c.local_id,
            c.name,
            c.hub_id,
            "up-to-date",
            classification=c.classification.state if c.classification else None,
            message=c.detail,
        )
    message = c.detail
    if c.classification and c.classification.state == "baseline-unknown":
        message = message or (
            "baseline unknown (edited before hub tracking began); apply with "
            "--accept-unknown-baseline to take hub additions without deleting anything of yours"
        )
    return ItemMergeReport(
        c.kind,
        c.local_id,
        c.name,
        c.hub_id,
        "available",
        classification=c.classification.state if c.classification else None,
        message=message,
    )


def clear_failure(ctx: HubSyncContext, kind: str, name: str) -> None:
    """Retire an item's stale failure after a manual retry proved it computes (U10)."""

    wanted = name.strip().casefold()

    def fold(doc: dict[str, Any]) -> None:
        for item in doc.get("items", {}).values():
            if item.get("kind") == kind and str(item.get("name", "")).casefold() == wanted:
                st.clear_failure(item)

    ctx.store().mutate(fold)


def clear_retry(ctx: HubSyncContext, kind: str, name: str) -> None:
    """Manual Retry: user intent outranks the schedule (B4.3). Clears attempts and the timer."""

    wanted = name.strip().casefold()

    def fold(doc: dict[str, Any]) -> None:
        for item in doc.get("items", {}).values():
            if item.get("kind") == kind and str(item.get("name", "")).casefold() == wanted:
                st.clear_retry(item)

    ctx.store().mutate(fold)


def sync_context(config_manager: Any, *, resolver: Resolver | None = None) -> HubSyncContext:
    """A context for a SYNC host (the CLI): credentials resolved with the blocking bridge."""

    for_tenant, credential = build_clients_sync(config_manager)
    return HubSyncContext(
        config_dir=config_manager.config_dir,
        config_manager=config_manager,
        client_for_tenant=for_tenant,
        credential=credential,  # type: ignore[arg-type]
        resolver=resolver,
    )


def render_status(snapshot: Mapping[str, Any]) -> str:
    """Prose for ``lop hub status``: the same payload the sidebar polls, as lines."""

    counts = snapshot.get("counts", {})
    settings = snapshot.get("settings", {})
    head = (
        f"Agent Hub updates: {counts.get('available', 0)} available, "
        f"{counts.get('failed', 0)} failed, {counts.get('updating', 0)} updating "
        f"(auto: agents {'on' if settings.get('auto_agents') else 'off'}, "
        f"teams {'on' if settings.get('auto_teams') else 'off'}; "
        f"checked every {settings.get('interval_min')} min; login {snapshot.get('credential')})"
    )
    lines = [head]
    for item in snapshot.get("items", []):
        line = f"  {item['kind']} {item['name']}: {item['state']}"
        if item.get("classification"):
            line += f" ({item['classification']})"
        if item.get("auto_will_apply"):
            line += " — will update automatically"
        if item.get("last_error"):
            line += f" — {item['last_error']}"
        lines.append(line)
    if len(lines) == 1:
        lines.append("  nothing pending")
    return "\n".join(lines)


# -- links inventory (B5, new capability) ------------------------------------------------


def links_snapshot(ctx: HubSyncContext, *, kinds: Sequence[str] | None = None) -> dict[str, Any]:
    """The ``GET /v1/desktop/hub/links`` payload: EVERY link, healthy or not.

    The FULL frozen row shape (architecture "full link row"): identity
    (link_id/kind/local_id/name/hub_id/hub_source/visibility/tenant_id),
    policy{subscribe,auto_push,revision}, service-computed
    eligibility{can_auto_push,reason}, `auto_will_apply`/`auto_will_push`,
    baseline_state, remote_revision, `content_state` with `local_path` when the
    copy exists, and INDEPENDENT `pull`/`push` DirectionStatus objects. Distinct
    from ``status_snapshot`` (attention-only): this lists a healthy idle link too,
    so the sidebar can show its quiet mark and offer a source-scoped toggle with
    zero pending changes. Local reads only.

    ``kinds`` filters the enumeration (the route's ``?kind=``); skill rows are
    capability-gated by the route so old typed consumers never see them.
    Identity-less (legacy) links are returned separately under ``unresolved``
    with a named reason, never a synthetic link_id or policy.
    """
    from local_operator.hub_sync import policies as pol
    from local_operator.hub_sync import provenance as prov

    try:
        values = ctx.config_manager.get_config().values
    except Exception:  # noqa: BLE001 - no readable config means creation defaults
        values = {}
    doc = ctx.store().load()
    items = doc.get("items", {})
    now = datetime.now(timezone.utc)
    links: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for record in prov.iter_baselines(ctx.config_dir):
        if kinds is not None and record.kind not in kinds:
            continue
        item = items.get(st.item_key(record.kind, record.local_id), {})
        identity = record.identity
        name = record.local_alias or record.local_id
        if identity is None:
            # A legacy link with a hub id but no confirmed origin identity: it
            # cannot carry a policy or a source. Listed separately so a client
            # offers the resolve action instead of acting on a guessed origin.
            unresolved.append(
                {
                    "kind": record.kind,
                    "local_id": record.local_id,
                    "name": name,
                    "hub_id": record.hub_id,
                    "reason": "source-unresolved",
                }
            )
            continue
        # F1: serve the GRANT-AWARE revision — the same digest ``update_policy``
        # recomputes. ``policy.revision(identity)`` omitted the egress grant from
        # the digest (grant-unaware), so once a public link held a grant every
        # later write that used the served revision 409'd ("link policy changed")
        # on CLI, desktop and TUI alike. ``read_policy`` is that same grant-aware
        # path, so the served CAS token and the recomputed token cannot drift.
        snapshot = pol.read_policy(values, identity)
        policy = snapshot.policy
        eligibility = _eligibility(ctx, identity, policy, values, record, doc)
        capability = _capability_auto_push(doc, identity)
        content_present = record.content_state != "absent"
        links.append(
            {
                "link_id": identity.link_id,
                "kind": identity.kind,
                "local_id": record.local_id,
                "name": name,
                "hub_id": record.hub_id,
                "hub_source": identity.hub_source,
                "visibility": identity.visibility,
                "tenant_id": identity.tenant_id,
                "tenant_name": _tenant_name(values, identity),
                "policy": snapshot.to_json(),
                "eligibility": eligibility,
                # Where the managed copy lives, when it exists. A client needs this
                # to tell the operator where the bytes are without re-deriving the
                # managed root; an absent link has no local copy at all.
                "local_path": (
                    _managed_skill_path(ctx, name)
                    if identity.kind == "skill" and content_present
                    else None
                ),
                "content_state": record.content_state,
                "auto_will_apply": _auto_will_apply(
                    item, policy, ctx, now, content_present=content_present
                ),
                "auto_will_push": _auto_will_push(item, policy, ctx, now, eligibility),
                "baseline_state": record.recorded_by,
                "remote_revision": record.remote_revision,
                "pull": _direction_status(item, "pull", policy, now, capability),
                "push": _direction_status(item, "push", policy, now, capability),
            }
        )
    links.sort(
        key=lambda link: (
            KIND_ORDER.get(link["kind"], len(KIND_ORDER)),
            str(link["name"]).casefold(),
        )
    )
    unresolved.sort(
        key=lambda row: (KIND_ORDER.get(row["kind"], len(KIND_ORDER)), str(row["name"]).casefold())
    )
    return {
        "generated_at": prov.now_iso(),
        "capability": "hub_links:1",
        "links": links,
        "unresolved": unresolved,
    }


def _managed_skill_path(ctx: HubSyncContext, alias: str) -> str | None:
    """The managed directory for one skill alias, or None when it is not addressable.

    Read-only: ``ManagedSkillStorage.path`` only validates the alias and refuses a
    symlinked root/alias, so an inventory read can never aim a client at a path the
    writer would reject.
    """

    from local_operator.hub_sync.skill_storage import ManagedSkillStorage

    try:
        return str(ManagedSkillStorage(ctx.config_dir).path(alias))
    except Exception:  # noqa: BLE001 - an unaddressable alias simply has no path to show
        return None


def _tenant_name(values: Mapping[str, Any], identity: LinkIdentity) -> str | None:
    """Persisted display metadata for the org, or None; never a network lookup."""

    if identity.visibility != "org" or identity.tenant_id is None:
        return None
    hub = values.get("hub", {}) if isinstance(values, Mapping) else {}
    names = hub.get("tenant_names", {}) if isinstance(hub, Mapping) else {}
    if isinstance(names, Mapping):
        value = names.get(identity.tenant_id)
        if isinstance(value, str) and value:
            return value
    return None


def _attested(value: Any) -> int:
    """Normalize an attestation value to 0/1 (ints from the wire, or a legacy bool)."""

    if value is True:
        return 1
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return 1
    return 0


def _capability_auto_push(doc: Mapping[str, Any], identity: LinkIdentity) -> bool:
    """Whether THIS source's atomic revision-CAS is POSITIVELY attested.

    A positive attestation, not a constant: an actual response from the deployed
    server carried a ``sync_contract`` (recorded per SOURCE by
    :func:`record_server_capabilities`), and this reads that persisted fact
    locally -- the inventory is a local read and must never probe the network.

    PER-SOURCE, not a global flag: a link re-pointed at another origin cannot
    inherit an attestation observed elsewhere, and a SKILL link additionally
    requires the skill family (``skills >= 1``) on ITS source, because a server
    that speaks revision CAS for agents/teams but not the skill routes cannot
    safely accept an unattended skill publication. Absent the attestation the
    answer is honestly False; it never fakes the capability.
    """

    server = doc.get("server", {}) if isinstance(doc, Mapping) else {}
    sources = server.get("sources", {}) if isinstance(server, Mapping) else {}
    attest = sources.get(identity.hub_source) if isinstance(sources, Mapping) else None
    if not isinstance(attest, Mapping):
        return False
    if not _attested(attest.get("revision_cas")):
        return False
    if identity.kind == "skill" and not _attested(attest.get("skills")):
        return False
    return True


def _fold_capability(doc: dict[str, Any], capabilities: Mapping[str, Any], source: str) -> None:
    """Merge one observed ``sync_contract`` into the per-source attestation store.

    LAST OBSERVATION WINS per source and per field (round-2 R14): attestation is
    a fact about the DEPLOYED server, so a deployment that no longer advertises
    revision-CAS must be able to CLEAR it -- a monotone max() could only ever
    ratchet up and made a downgraded source permanently "proven". The legacy flat
    mirror is kept for any older reader of ``server.revision_cas``.
    """

    revision_cas = _attested(capabilities.get("revision_cas"))
    skills = _attested(capabilities.get("skills"))
    server = doc.setdefault("server", {})
    server["revision_cas"] = bool(revision_cas)
    server["attested_at"] = prov.now_iso()
    entries = server.setdefault("sources", {})
    entry = entries.setdefault(str(source), {})
    entry["revision_cas"] = revision_cas
    entry["skills"] = skills
    entry["attested_at"] = prov.now_iso()


def record_server_capabilities(
    ctx: HubSyncContext, capabilities: Mapping[str, Any], *, source: str | None = None
) -> None:
    """Persist a VERIFIED server capability attestation. PROBE-ONLY.

    Called by callers that have actually spoken to the deployed server and seen
    its ``sync_contract`` in a response: the outbound probe, and
    :func:`record_checks` (which folds the contract carried by every real check
    response, per link SOURCE).

    SECURITY INVARIANT: this must NEVER be reachable from a route, a UI action,
    a config key or a command. Attestation is a fact about the DEPLOYED server,
    so a local write is only honest when a request produced it; letting a client
    set ``revision_cas`` would let it SELF-GRANT publication eligibility, which
    is the exact thing ``_eligibility`` exists to refuse. The persisted facts are
    therefore read by the inventory and written solely by observation paths --
    there is no request field for them anywhere.

    ``source`` keys the attestation PER SOURCE (the normalized credential-free
    API base the response came from), so a link re-pointed at another origin
    cannot inherit it. An omitted ``source`` still records the legacy flat flag
    for older readers; new eligibility reads only the per-source entry.
    """

    def fold(doc: dict[str, Any]) -> None:
        _fold_capability(doc, capabilities, source or "")

    ctx.store().mutate(fold)


def _eligibility(
    ctx: HubSyncContext,
    identity: LinkIdentity,
    policy: Any,
    values: Mapping[str, Any],
    record: Any,
    doc: Mapping[str, Any],
) -> dict[str, Any]:
    """Service-computed publication eligibility: ``can_auto_push`` plus a reason.

    Computed ONLY from verified facts, never a UI guess or a request field:
    a POSITIVELY attested atomic revision-CAS on the deployed server (persisted
    by a probe), a KNOWN baseline (an unknown-lineage origin cannot be published
    unattended), clean lineage, and -- for public scope -- the persisted egress
    grant bound to THIS link at the current publication-contract version. A
    reason is always present when the answer is no.
    """
    # AN ABSENT COPY HAS NOTHING TO PUBLISH. A subscribed-without-pull link may be
    # checked and reported, but an unattended push would have to invent the
    # local side; the content state is a fact, not a momentary gap. It comes
    # FIRST because it is the actionable fact: no capability attestation would
    # make a link with no bytes publishable.
    if getattr(record, "content_state", "present") == "absent":
        return {"can_auto_push": False, "reason": "content-absent"}
    if not _capability_auto_push(doc, identity):
        return {"can_auto_push": False, "reason": "unsupported-precondition"}
    # SOURCE-SCOPED CAS ATTESTATION: eligibility is bound to this origin's
    # recorded revision, so a link re-pointed at another origin cannot inherit a
    # stale attestation. The recorded revision IS the source-scoped fact.
    if not record.remote_revision:
        return {"can_auto_push": False, "reason": "unsupported-precondition"}
    if not record.scope_known or record.recorded_by == "adopt-unknown":
        return {"can_auto_push": False, "reason": "baseline-unknown"}
    if record.lineage not in (None, "clean"):
        return {"can_auto_push": False, "reason": "mixed-lineage"}
    if identity.visibility == "public":
        # Public enable still requires a DELIBERATE egress consent bound into the
        # policy/preview digest (the grant); a scope change cannot reuse it.
        hub = values.get("hub", {}) if isinstance(values, Mapping) else {}
        grants = hub.get("public_egress_grants", {}) if isinstance(hub, Mapping) else {}
        grant = grants.get(identity.link_id) if isinstance(grants, Mapping) else None
        if (
            not isinstance(grant, Mapping)
            or grant.get("publication_contract") != PUBLICATION_CONTRACT
        ):
            return {"can_auto_push": False, "reason": "unsafe-publication"}
    if record.outbound is None:
        # NO AGREED POINT YET: an unattended push projects from an (A, P) receipt,
        # and nothing has ever recorded one -- first publication is a deliberate
        # action (manual push) or a pull that integrates the remote content
        # (both paths SEED the receipt at that point). Until then the honest
        # answer is a named ineligible, never a zero-delta guess. LAST on
        # purpose: a public link's missing EGRESS CONSENT is the actionable
        # refusal its confirm flow mints against, so that reason wins when both
        # are absent.
        return {"can_auto_push": False, "reason": "no-receipt"}
    return {"can_auto_push": True, "reason": "eligible"}


def _auto_will_apply(
    item: Mapping[str, Any],
    policy: Any,
    ctx: HubSyncContext,
    now: Any,
    *,
    content_present: bool = True,
) -> bool:
    """Whether the next run would apply an incoming change: policy+backoff+grace.

    ``content_present`` is False for an explicitly absent skill link: the runner
    never auto-INSTALLS one, so the row must not promise an automatic update it
    will not perform.
    """

    if not content_present:
        return False
    if not policy.subscribe:
        return False
    if st.effective_state(item, now) not in ("available", "updating"):
        return False
    return st.auto_apply_due(item)


def _auto_will_push(
    item: Mapping[str, Any],
    policy: Any,
    ctx: HubSyncContext,
    now: Any,
    eligibility: Mapping[str, Any],
) -> bool:
    """Whether the next run would PUSH, from actual service eligibility (not a UI guess)."""

    if not policy.auto_push or not eligibility.get("can_auto_push"):
        return False
    return st.push_due(item, ctx.settings().interval_min, now) and st.auto_apply_due(item)


def push_eligibility(ctx: HubSyncContext, record: prov.BaselineRecord | None) -> dict[str, Any]:
    """The SHARED eligibility verdict for one link, from current local facts.

    The runner recomputes this at PUBLISH time (never trusting a stale check or
    the policy bit alone), through the SAME ``_eligibility`` the inventory and
    the policy PATCH use, so a hand-edited ``auto_push: true``, a revoked
    attestation, a deleted egress grant or a bumped publication contract can
    never reach the wire: the refusal is a named, persisted status instead.
    """

    if record is None or record.identity is None:
        return {"can_auto_push": False, "reason": "baseline-unknown"}
    try:
        values = ctx.config_manager.get_config().values
    except Exception:  # noqa: BLE001 - an unreadable config is not an eligible link
        values = {}
    doc = ctx.store().load()
    return _eligibility(ctx, record.identity, None, values, record, doc)


def record_outbound_receipt(
    ctx: HubSyncContext,
    kind: Kind,
    local_id: str,
    *,
    accepted: Mapping[str, Any],
    current_local: Mapping[str, Any],
    revision: str | None,
    operation_id: str | None,
) -> None:
    """Advance the outbound receipt (A/P) after an ACCEPTED publication.

    P is the ACTUAL server-returned canonical document and ``revision`` the
    OPAQUE accepted ETag (``result['revision']``) -- the value the NEXT
    conditional write sends as If-Match. The integrated inbound baseline B is
    deliberately untouched: a push whose returned remote was never integrated
    locally must not look integrated (subscribe-off links rely on this).
    """

    from local_operator.hub_sync import push as push_mod

    record = prov.read_baseline(ctx.config_dir, kind, local_id)
    if record is None:
        return
    receipt = push_mod.receipt_for(
        kind, accepted, current_local, revision=revision, operation_id=operation_id
    )
    prov.write_baseline(ctx.config_dir, replace(record, outbound=receipt))


def _direction_status(
    item: Mapping[str, Any],
    direction: Literal["pull", "push"],
    policy: Any,
    now: Any,
    capability: bool,
) -> dict[str, Any]:
    """One direction's full status: pull and push are independent, never merged.

    A failure in one direction cannot overwrite the other's receipt or error;
    whole-row health is a derived view, not the source of truth.
    """

    pulled = direction == "pull"
    enabled = policy.subscribe if pulled else bool(policy.auto_push)
    last = item.get("last_pulled_at") if pulled else item.get("last_pushed_at")
    error = item.get("error_class") if pulled or not pulled else None
    state = st.effective_state(item, now)
    if pulled:
        if error:
            direction_state = "failed"
        elif state == "updating":
            direction_state = "merging"
        elif state == "available":
            direction_state = "available"
        elif last:
            direction_state = "idle"
        else:
            direction_state = "idle"
    else:
        # The push side has no separate store row yet; report the pushed timestamp
        # and the (currently unsupported) capability honestly.
        direction_state = "idle" if last else "idle"
        if not capability:
            direction_state = "idle"
    return {
        "state": direction_state,
        "classification": item.get("classification"),
        "enabled": enabled,
        "last_checked_at": item.get("last_checked_at"),
        "last_success_at": last,
        "attempts": int(item.get("attempts") or 0),
        "next_retry_at": item.get("next_retry_at"),
        "error_class": error if pulled else None,
        "error_subclass": item.get("error_subclass") if pulled else None,
        "safe_message": item.get("last_error") if pulled else None,
        "operation_id": None,
        "model_used": (
            item.get("summary", {}).get("model")
            if isinstance(item.get("summary"), Mapping)
            else None
        ),
        "summary": item.get("summary") or {},
        "backup_receipt": item.get("applied_backup"),
    }


# -- initial skill install (M1: authenticated read/install adapter) ----------------------


def skill_link_local_id(
    *, source: str, visibility: Visibility, tenant_id: str | None, hub_id: str
) -> str:
    """The deterministic local id a remote skill identity mints.

    Deterministic on purpose: a retry after a crash resumes the SAME link (its
    journal entry, its alias ownership, its policy CAS) instead of minting a
    second local id beside its own crashed transaction. Opaque and persisted --
    it is never a path and never a remote name (the architecture rule), and
    the alias remains display-only.
    """

    import json
    from hashlib import sha256

    digest = sha256(
        json.dumps(["skill", source, visibility, tenant_id, hub_id], separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return f"skl-{digest[:32]}"


def _resolve_remote_skill(
    ctx: HubSyncContext,
    *,
    hub_source: str,
    hub_id: str,
    visibility: Visibility,
    tenant_id: str | None,
    alias: str,
) -> tuple[str, str, LinkIdentity, dict[str, Any], str]:
    """The ONE remote-identity path a skill PULL or SUBSCRIBE runs before local state.

    Source approval, scope pairing, alias validation, the credentialed fetch and
    the answered-row proof live here so the two writers cannot diverge about
    which row they are allowed to trust: a subscribe records the same identity a
    pull does, and a row a pull would refuse is refused identically before any
    link exists. Returns ``(source, alias, identity, canonical, revision)`` and
    writes nothing.

    Every refusal keeps the pull route's vocabulary: ``skill-not-found`` for a
    missing OR foreign row (one flat 404, never distinguishable), ``no-credential``
    for a missing sign-in, ``origin-conflict`` for a mismatched source/scope, and
    ``hub-upstream`` for an unreachable hub.
    """

    from local_operator.hub_sync.adapters import SkillAdapter
    from local_operator.hub_sync.check import (
        REQUEST_TIMEOUT_S,
        _failure_reason,
        _fetch_skill,
        _remote_meta,
        _served_by,
    )
    from local_operator.hub_sync.identity import approve_source
    from local_operator.hub_sync.skill_document import (
        SkillValidationError,
        validate_name,
    )
    from local_operator.providers.radient_credentials import (
        ORG_LOGIN_REMEDY,
        configured_radient_base_url,
    )

    # 1. Source approval BEFORE anything could carry a credential: the ONE
    #    configured credential-free base, never a caller-supplied URL.
    approved = (configured_radient_base_url(ctx.config_manager),)
    source = approve_source(hub_source, approved)

    # 2. Scope pairing, server-side: the same rule the renderer asserts.
    if visibility == "public" and tenant_id is not None:
        raise LinkError("origin-conflict", "a public skill link cannot name an organization")
    if visibility == "org" and not tenant_id:
        raise LinkError("origin-conflict", "an organization skill link requires a tenant")

    # 3. The alias must be a valid skill name BEFORE any comparison or write.
    try:
        alias = validate_name(alias)
    except SkillValidationError as error:
        raise LinkError("skill-validation", str(error)) from None

    # 4. The credentialed client for this scope, and positive proof of which
    #    base answered: a client whose base is not the requested source is
    #    refused, never credited (R14's per-source attestation rule).
    client = ctx.client_for_tenant(tenant_id)
    if client is None:
        raise LinkError("no-credential", ORG_LOGIN_REMEDY)
    if _served_by(client) != source:
        raise LinkError(
            "origin-conflict", "the answering client's base is not the requested source"
        )

    try:
        document = _fetch_skill(client, hub_id, timeout=REQUEST_TIMEOUT_S)
    except Exception as error:  # noqa: BLE001 - one refusal class for the fetch
        reason, _detail = _failure_reason(error)
        if reason == "hub-item-missing":
            # ONE plain refusal for missing AND foreign rows: the hub answers
            # 404 to anyone who cannot see the item, and this side must not
            # re-derive which it was.
            raise LinkError("skill-not-found", "no such skill") from None
        raise LinkError("hub-upstream", "the hub could not serve this skill") from error

    # 5. The answered row must BE the requested row (R1/Q2). ``/skills/{id}``
    #    resolves access from the ROW itself, and the signed-in client can read
    #    public and org rows alike -- so the envelope, never the request, is the
    #    only evidence of which row actually answered. Accepting a relabeled row
    #    would attach an identity (and, on a pull, clean ancestry) to another
    #    scope's content. Refusals are the SAME flat 404 as a missing or foreign
    #    row: a probe must not be able to tell the checks apart.
    if not isinstance(document, dict):
        raise LinkError("skill-not-found", "no such skill")
    remote_id = document.get("id")
    remote_visibility = document.get("visibility")
    remote_tenant = document.get("tenant_id")
    if not (isinstance(remote_id, str) and remote_id == hub_id):
        raise LinkError("skill-not-found", "no such skill")
    if remote_visibility != visibility:
        raise LinkError("skill-not-found", "no such skill")
    if visibility == "org":
        if not (isinstance(remote_tenant, str) and remote_tenant == tenant_id):
            raise LinkError("skill-not-found", "no such skill")
    elif remote_tenant not in (None, ""):
        raise LinkError("skill-not-found", "no such skill")

    adapter = SkillAdapter()
    try:
        canonical = dict(adapter.canonical_fields(document))
    except SkillValidationError as error:
        raise LinkError(
            "invalid-remote-manifest", f"the remote skill manifest is invalid: {error}"
        ) from None
    revision, _contract = _remote_meta(document)
    if not revision:
        # A link records the ancestry fact; without a revision it could never
        # prove what it names. Refuse now, write nothing.
        raise LinkError("unsupported-precondition", "the remote skill carries no revision")

    # 6. Deterministic identity: a retry after a crash resumes the SAME link.
    local_id = skill_link_local_id(
        source=source, visibility=visibility, tenant_id=tenant_id, hub_id=hub_id
    )
    identity = LinkIdentity("skill", local_id, source, visibility, tenant_id, hub_id)
    return source, alias, identity, canonical, revision


def install_skill(
    ctx: HubSyncContext,
    *,
    hub_source: str,
    hub_id: str,
    visibility: Visibility,
    tenant_id: str | None,
    alias: str,
    subscribe: bool | None = None,
    auto_push: bool | None = None,
    confirm_public_egress: bool = False,
) -> dict[str, Any]:
    """Install ONE remote skill as a new local link (UI ``hub.skillPull``).

    This is the desktop plane's entry into the SAME engine the linked checks and
    applies use: the fetch goes through the check lane's own seam, the write
    through ``ManagedSkillStorage``'s journaled transaction, and the row the
    response carries is the ordinary ``links_snapshot`` inventory row.

    Identities are MINTED DETERMINISTICALLY from the remote identity tuple
    (kind, normalized source, visibility, tenant, hub id): the local id -- and
    so the link id -- is stable across retries, which is what makes a crashed
    install resume as the SAME link instead of minting a second one beside its
    own journal. The alias stays display-only (the architecture rule); one
    remote skill is one local link in this milestone, so a second alias for a
    linked skill is refused rather than silently duplicated.

    The answered row is validated against the REQUESTED identity before its
    manifest is trusted (R1): a row from another scope or identity refuses
    with the same nondisclosing 404 as a missing one. The initial policy is
    committed WITH the content transaction: an explicit ``subscribe`` rides
    the baseline's materialization hint, and only an ``auto_push: true``
    request -- eligible exactly like the link-policy PATCH, through the ONE
    ``policies`` writer with SERVICE-computed attestations -- is staged
    beforehand and reverted on a failed install while its revision is still
    ours. Every local-state step runs under the cross-process update lease
    (Q3), and a crashed attempt is recovered before fresh-vs-completed is
    decided, so its retry resumes the SAME link (Q4).
    """

    from local_operator.config import read_config_values
    from local_operator.hub_sync import policies as pol
    from local_operator.hub_sync.skill_document import (
        SkillDocument,
        SkillValidationError,
    )
    from local_operator.hub_sync.skill_storage import ManagedSkillStorage
    from local_operator.hub_sync.transactions import TransactionError

    _source, alias, identity, canonical, revision = _resolve_remote_skill(
        ctx,
        hub_source=hub_source,
        hub_id=hub_id,
        visibility=visibility,
        tenant_id=tenant_id,
        alias=alias,
    )
    local_id = identity.local_id

    storage = ManagedSkillStorage(ctx.config_dir)

    # 7. Every local-state step -- recovery, the linked scan, policy staging and
    #    the content transaction -- runs under the SAME cross-process update
    #    lease the apply path takes (Q3): a runner tick or another process
    #    mid-update must not interleave with an install. The wait is ZERO: this
    #    is an interactive surface, the refusal is the 409 whose hint names the
    #    wait, and a 30-second poll here would just hold the desktop request.
    with exclusive(ctx, wait_s=0.0):
        # 7a. Settle a crashed earlier attempt BEFORE deciding fresh-vs-completed
        #     (Q4): recovery finalizes its journal, so a content-committed crash
        #     becomes a visible completed baseline and the retry below returns
        #     its idempotent result instead of colliding with its own content
        #     (the unconditional expected_fingerprint=None used to compare
        #     against the just-recovered bytes and refuse local-concurrent-edit).
        try:
            storage.recover(local_id=local_id)
        except TransactionError as error:
            raise LinkError(
                getattr(error, "code", "") or "local-concurrent-edit", str(error)
            ) from None

        # 7b. Already-linked / replay arm BEFORE any write: a deterministic id
        #     that already exists is THIS install's own earlier attempt (or a
        #     completed one -- including one just finalized by the recovery
        #     above), never a second copy. EXCEPTION: a link whose content is
        #     explicitly ABSENT (subscribe-without-pull, or an unpulled copy) is
        #     not a completed install -- an explicit pull IS the install it was
        #     waiting for, so it falls through to the content transaction below
        #     and carries the existing identity, scope and policy into it.
        resuming_absent: prov.BaselineRecord | None = None
        for record in prov.iter_baselines(ctx.config_dir):
            if record.kind != "skill" or record.local_id != local_id:
                continue
            if record.identity is None or record.identity != identity:
                raise LinkError("origin-conflict", "a local skill already owns this identity")
            if (record.local_alias or record.local_id) != alias:
                raise LinkError(
                    "origin-conflict",
                    f"this skill is already linked as {record.local_alias!r}; one remote skill is "
                    "one local link here -- remove that link first",
                )
            if record.content_state == "absent":
                resuming_absent = record
                break
            snapshot = links_snapshot(ctx)
            row = next(
                (link for link in snapshot["links"] if link["link_id"] == identity.link_id), None
            )
            if row is None:
                raise LinkError("origin-conflict", "the linked snapshot has no row for this skill")
            report = ItemMergeReport("skill", local_id, alias, hub_id, "unchanged")
            payload = report.to_json()
            payload.update(
                {
                    "link_id": identity.link_id,
                    "direction": "pull",
                    "operation_id": None,
                    "remote_revision": record.remote_revision,
                    "model_used": None,
                }
            )
            return {"reports": [payload], "link": row, "links": snapshot}

        # 7c. The initial policy, committed WITH the content transaction
        #     (R2/Q1). A fresh link's creation default already is subscribe=true/
        #     auto_push=false, and an explicit ``subscribe`` rides the baseline's
        #     materialization hint so the transaction writes it atomically -- a
        #     refused install therefore leaves NO policy behind, and a retry at
        #     a free alias gets the default, never the dead request's value.
        #     Only ``auto_push: true`` is staged BEFORE the transaction: the
        #     public-consent carve-out is the one writer that can mint a first
        #     public grant, and the storage CASes the staged revision through
        #     its commit; if the transaction then refuses, the staged entry is
        #     reverted below -- but only while its revision is still ours (a
        #     newer writer's value is never rolled back).
        #
        #     THE FULL LOCAL PROPOSAL IS VALIDATED FIRST (R2/Q1b): rendering the
        #     alias can itself refuse -- e.g. the rename pushes SKILL.md past the
        #     200 KiB bound -- and nothing may be staged until every local
        #     refusal has had its chance. Everything that can fail locally runs
        #     before the staging below; the only operations after it are the
        #     content transaction (whose refusal paths revert the staged entry)
        #     and the success response.
        try:
            local_write = SkillDocument.from_json(canonical).with_name(alias)
        except SkillValidationError as error:
            raise LinkError("skill-validation", str(error)) from None
        baseline = prov.make_record("skill", local_id, hub_id, tenant_id, canonical, "pull")
        if resuming_absent is not None:
            # The prior link's identity/scope/policy/alias must survive an
            # install that was explicitly requested on top of it; only the
            # content facts this pull actually establishes move. The migration
            # hint is left untouched because that link's policy already exists.
            baseline = replace(
                resuming_absent,
                fields=dict(canonical),
                fingerprint=prov.fingerprint_fields("skill", canonical),
                recorded_by="pull",
                content_state="present",
                remote_revision=revision,
                lineage="clean",
            )
        else:
            baseline = replace(baseline, legacy_subscribe_hint=subscribe)
        baseline = replace(
            baseline,
            identity=identity,
            local_alias=alias,
            remote_revision=revision,
            lineage="clean",
            content_state="present",
        )

        values = read_config_values(ctx.config_dir)
        if values is None and (ctx.config_dir / "config.yml").exists():
            raise LinkError("invalid-link-policy", "configuration is unreadable")
        current = pol.read_policy(values or {}, identity)
        staged_revision: str | None = None
        if auto_push:
            # THE ATTESTATION DESCRIBES THE ROW THIS INSTALL WILL WRITE: the
            # very facts the storage transaction commits (identity, alias,
            # remote revision, clean pull lineage). Any earlier gate in
            # ``_eligibility`` -- capability, revision, known baseline, lineage,
            # egress grant, receipt -- still refuses with its own named code;
            # nothing here can fabricate a grant or a receipt to make auto-push
            # pass.
            intended = prov.make_record("skill", local_id, hub_id, tenant_id, canonical, "pull")
            intended = replace(
                intended,
                identity=identity,
                local_alias=alias,
                remote_revision=revision,
                lineage="clean",
            )
            eligibility = _eligibility(
                ctx, identity, current.policy, values or {}, intended, ctx.store().load()
            )
            current = pol.update_policy(
                ctx.config_manager,
                identity,
                expected_revision=current.revision,
                subscribe=subscribe,
                auto_push=True,
                confirm_public_egress=confirm_public_egress,
                can_enable_auto_push=bool(eligibility["can_auto_push"]),
                ineligible_reason=str(eligibility["reason"]),
            )
            staged_revision = current.revision
        elif resuming_absent is None and (
            pol.stored_link_policy(values or {}, identity.link_id) is not None
        ):
            # No baseline exists (the scan above would have returned), yet an
            # EXPLICIT entry does: a failed or crashed earlier attempt staged
            # it. This request's values are authoritative for the uninstalled
            # link, so the orphan is cleared through the same CAS-guarded
            # removal -- and the transaction's materialization then writes the
            # request's subscribe (or the creation default) fresh.
            pol.revert_policy_if_unchanged(
                ctx.config_manager, identity, expected_revision=current.revision
            )
            values = read_config_values(ctx.config_dir)
            current = pol.read_policy(values or {}, identity)

        # 7e. The content transaction: the alias-rendered bytes and the
        #     canonical remote baseline (carrying the subscribe hint) validated
        #     above, with the policy revision CASed through commit. The storage
        #     finalizes baseline + materialized policy in the SAME transaction,
        #     so "baseline moved, bytes did not" stays impossible.

        def _settle_staged_after_failure() -> None:
            # ONE uniform cleanup for EVERY non-success exit between staging and
            # the committed transaction (Q1c) -- a storage refusal, an
            # OSError(ENOSPC) from the staging writes, an unexpected crash, an
            # operator interrupt -- so no exception TYPE has to be enumerated
            # for the staged policy/grant to be cleaned up. Best-effort by
            # design: the original failure is the response, and the removal
            # refuses to touch a NEWER revision (CAS); a cleanup that cannot
            # run leaves an orphan the next attempt's orphan handling clears.
            #
            # LANDED-NESS NEEDS POSITIVE OWNERSHIP (Q1d). Content equality
            # alone misreads a pre-existing unknown authored folder that
            # happens to be byte-identical to the attempted manifest as "our
            # transaction landed" (that refusal leaves no journal at all), so
            # only an unfinished journal record OF THIS ATTEMPT -- tied by
            # local id, alias, the staged policy revision and the rendered
            # document fingerprint -- may keep the staged grant:
            #
            #   no own journal      -> nothing of ours ran; revert (the CAS
            #                          guard leaves newer writers alone).
            #   own journal, destination != rendered
            #                       -> the rename boundary was never crossed
            #                          (staging fault); recovery restores or
            #                          aborts; revert.
            #   own journal, destination == rendered
            #                       -> OUR rename put it there (the journal
            #                          stays 'prepared' through the rename
            #                          sequence, so the phase is deliberately
            #                          NOT the test); recovery finalizes and
            #                          the grant MUST survive, else the
            #                          recovered install would silently lose
            #                          it.
            #   baseline already present -> the install committed; keep.
            #
            # blocked/aborted records are attempts the storage already
            # declared dead and never count as ownership. The full state
            # table lives with the PR remediation receipt.
            if staged_revision is None:
                return
            if prov.read_baseline(ctx.config_dir, "skill", local_id) is not None:
                return
            owned = False
            for record in storage.journal.pending(kind="skill", local_id=local_id):
                if record.phase in ("blocked", "aborted"):
                    continue
                payload = record.payload
                if (
                    payload.get("local_alias") == alias
                    and payload.get("policy_revision") == staged_revision
                    and record.after_fingerprint == local_write.fingerprint
                ):
                    owned = True
                    break
            if owned:
                try:
                    landed = (
                        storage._fingerprint_path(storage.path(alias)) == local_write.fingerprint
                    )
                except Exception:  # noqa: BLE001 - an unreadable destination is not a landing
                    landed = False
                if landed:
                    return
            try:
                pol.revert_policy_if_unchanged(
                    ctx.config_manager, identity, expected_revision=staged_revision
                )
            except Exception:  # noqa: BLE001 - never mask the original failure
                pass

        installed = False
        try:
            transaction = storage.apply(
                identity,
                alias,
                local_write,
                baseline,
                expected_fingerprint=None,
                expected_policy_revision=current.revision,
                automatic=False,
                basis="initial-install",
            )
            installed = True
        except LinkError:
            # The refusal itself is the response; the guard below decides
            # whether the staged entry survives (committed content) or is
            # cleared.
            raise
        except (TransactionError, SkillValidationError) as error:
            # The storage's own validation refusals (an occupied alias, a name
            # that does not match its recorded alias) carry their code;
            # normalize them to the ONE refusal type the route maps.
            raise LinkError(
                getattr(error, "code", "") or "local-concurrent-edit", str(error)
            ) from None
        finally:
            # THE POST-STAGE FAILURE GUARD (Q1c): every exit that did not
            # return a committed transaction passes through one cleanup -- no
            # per-type catch can miss a future storage fault.
            if not installed:
                _settle_staged_after_failure()

        # 8. The response the desktop contract expects: the directional report
        #    the UI reads (applied + link_id + direction) and the ordinary
        #    inventory rows -- the same builders every other surface consumes.
        snapshot = links_snapshot(ctx)
        row = next(
            (link for link in snapshot["links"] if link["link_id"] == identity.link_id), None
        )
        if row is None:
            raise LinkError("origin-conflict", "the install did not materialize a link row")
        report = ItemMergeReport("skill", local_id, alias, hub_id, "merged", applied=True)
        payload = report.to_json()
        payload.update(
            {
                "link_id": identity.link_id,
                "direction": "pull",
                "operation_id": transaction.operation_id,
                "remote_revision": revision,
                "model_used": None,
            }
        )
        return {"reports": [payload], "link": row, "links": snapshot}


def subscribe_skill(
    ctx: HubSyncContext,
    *,
    hub_source: str,
    hub_id: str,
    visibility: Visibility,
    tenant_id: str | None,
    alias: str,
) -> dict[str, Any]:
    """Link one remote skill WITHOUT installing a local copy (``hub.skillSubscribe``).

    The link and its content are independent facts. Subscribing records the
    origin, the alias reservation, the baseline and an explicit
    ``subscribe: true / auto_push: false`` policy with ``content_state: absent``,
    so the updates surface can report the remote and a later explicit pull can
    install it -- while nothing is written into the managed skills root. This is
    the same identity a pull would mint (the shared ``_resolve_remote_skill``
    chain, including its nondisclosing refusals), so a repeat request resolves
    to the SAME link and is replayed rather than duplicated.

    An existing link is left exactly as it is: a subscribe is not a policy edit,
    and it must never silently turn off an auto_push the operator enabled.
    """

    from local_operator.config import read_config_values
    from local_operator.hub_sync import policies as pol

    _source, alias, identity, canonical, revision = _resolve_remote_skill(
        ctx,
        hub_source=hub_source,
        hub_id=hub_id,
        visibility=visibility,
        tenant_id=tenant_id,
        alias=alias,
    )
    local_id = identity.local_id

    # Same lease the install/apply paths take: creating the baseline is local
    # state a concurrent tick or install must not interleave with.
    with exclusive(ctx, wait_s=0.0):
        existing: prov.BaselineRecord | None = None
        for record in prov.iter_baselines(ctx.config_dir):
            if record.kind != "skill" or record.local_id != local_id:
                continue
            if record.identity is None or record.identity != identity:
                raise LinkError("origin-conflict", "a local skill already owns this identity")
            if (record.local_alias or record.local_id) != alias:
                raise LinkError(
                    "origin-conflict",
                    f"this skill is already linked as {record.local_alias!r}; one remote skill is "
                    "one local link here -- remove that link first",
                )
            existing = record
            break

        if existing is None:
            record = replace(
                prov.make_record("skill", local_id, hub_id, tenant_id, canonical, "subscribe"),
                identity=identity,
                local_alias=alias,
                remote_revision=revision,
                # NOT "clean": nothing from this origin has landed locally. The
                # baseline names what the hub served at subscribe time; the
                # lineage becomes clean only when an install integrates it, and
                # the absent state keeps the link ineligible for pushes until
                # then (``_eligibility`` refuses on content-absent).
                lineage="unknown",
                content_state="absent",
            )
            prov.write_baseline(ctx.config_dir, record)

        # The explicit subscription policy. An entry that already exists means
        # this request's link does too, and a replay never rewrites it; a link
        # with no entry (a hand-written baseline, or one whose policy was
        # removed) gets exactly the state this route promises.
        values = read_config_values(ctx.config_dir)
        if values is None and (ctx.config_dir / "config.yml").exists():
            raise LinkError("invalid-link-policy", "configuration is unreadable")
        if pol.stored_link_policy(values or {}, identity.link_id) is None:
            current = pol.read_policy(values or {}, identity)
            pol.update_policy(
                ctx.config_manager,
                identity,
                expected_revision=current.revision,
                subscribe=True,
                auto_push=False,
                can_enable_auto_push=False,
                ineligible_reason="content-absent",
            )

        snapshot = links_snapshot(ctx)
        row = next(
            (link for link in snapshot["links"] if link["link_id"] == identity.link_id), None
        )
        if row is None:
            raise LinkError("origin-conflict", "the subscription did not materialize a link row")
        return {"link": row, "links": snapshot}


def unpull_skill(ctx: HubSyncContext, *, link_id: str) -> dict[str, Any]:
    """Remove one skill link's managed copy, keeping baseline/policy/alias.

    Membership is the LINK; the bytes are one direction of it. Unpull is an
    explicit local action, so it is not a policy change and never unlinks: the
    baseline keeps owning the alias (no other origin can claim it), the policy
    keeps its subscribe/auto_push values, and the check later reports the remote
    as ``available`` for an explicit pull to install. Repeating it on an
    already-absent link is a success, not a "not found": the request's intent is
    already true.

    The content removal itself (atomic rename, intent marker, state write) lives
    in ``ManagedSkillStorage.unpull`` so its crash window is recoverable from
    durable evidence rather than inferred from a missing directory.
    """

    from local_operator.hub_sync.skill_storage import ManagedSkillStorage
    from local_operator.hub_sync.transactions import TransactionError

    with exclusive(ctx, wait_s=0.0):
        record = None
        for candidate in prov.iter_baselines(ctx.config_dir):
            if (
                candidate.kind == "skill"
                and candidate.identity is not None
                and candidate.identity.link_id == link_id
            ):
                record = candidate
                break
        if record is None:
            raise LinkError("skill-not-found", "no such skill link")
        try:
            ManagedSkillStorage(ctx.config_dir).unpull(record.local_id)
        except TransactionError as error:
            raise LinkError(
                getattr(error, "code", "") or "local-concurrent-edit", str(error)
            ) from None
        snapshot = links_snapshot(ctx)
        row = next((link for link in snapshot["links"] if link["link_id"] == link_id), None)
        if row is None:
            raise LinkError("origin-conflict", "the unpull left no link row")
        return {"link": row, "links": snapshot}


# -- initial skill publication (the complete adapter's push half) -------------------------
#
# `POST /v1/desktop/hub/skills/publish` is the desktop plane's initial-publication
# arm. It is deliberately NOT a republish path: the hub contract this adapter
# advertises creates a NEW remote row (`publication: "create"`), and a second
# publication of an already-linked skill is refused locally with the alias owner
# rather than answered by an upstream `name_taken`.
#
# TWO receipts are involved and they are different facts:
#   * the PREVIEW receipt is local and review-only. A dry run projects the exact
#     bytes a confirm would send, binds them to the draft tuple + policy, and
#     persists that under `hub/publication-previews`. Nothing is published, no
#     paid call is made, and no link/baseline/policy is written;
#   * the LOCAL LINK receipt is the answer to "did this machine finish turning
#     the accepted publication into a real `links_snapshot` row". It is
#     `complete` only after the managed baseline + policy exist, and
#     `pending-reconciliation` when the hub accepted the row but local
#     bookkeeping did not finish. The remote outcome is persisted before any
#     local write, so a retry resumes the SAME link instead of republishing.
#
# A REFUSED publication leaves every piece of skill state untouched: the hub call
# happens before the managed transaction, and the only bytes this path writes on
# a dry run are the preview receipt itself.

#: How long a publication preview stays confirmable. The hub's own preview pins
#: expire after 15 minutes and this receipt fronts the same reviewed-draft act, so
#: it uses the same bound: a longer window would let a reviewed payload drift from
#: the bytes a later confirm sends.
PUBLICATION_PREVIEW_TTL_S = 900

_PREVIEW_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")


def _publication_preview_dir(config_dir: Path) -> Path:
    return prov.hub_root(config_dir) / "publication-previews"


def _publication_preview_path(config_dir: Path, token: str) -> Path:
    if not isinstance(token, str) or not _PREVIEW_TOKEN_RE.fullmatch(token):
        # A token is an opaque local file name; anything path-shaped is refused
        # before it can name a file outside the preview directory.
        raise LinkError("preview_mismatch", "the publication preview receipt is not valid")
    return _publication_preview_dir(config_dir) / f"{token}.json"


def _read_publication_preview(config_dir: Path, token: str) -> dict[str, Any] | None:
    path = _publication_preview_path(config_dir, token)
    if path.is_symlink() or path.parent.is_symlink():
        raise LinkError("preview_mismatch", "the publication preview receipt is unsafe")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def _write_publication_preview(config_dir: Path, record: Mapping[str, Any]) -> None:
    token = str(record.get("preview_receipt") or "")
    path = _publication_preview_path(config_dir, token)
    directory = path.parent
    if directory.is_symlink():
        raise LinkError("preview_mismatch", "the publication preview store is unsafe")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(record), handle, ensure_ascii=False, indent=1, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _prune_publication_previews(directory, keep=path)


def _prune_publication_previews(directory: Path, *, keep: Path) -> None:
    """Drop expired receipts; a preview is single-review state, not a history.

    A receipt that carries an ACCEPTED remote outcome but has not completed local
    reconciliation is never pruned: its hub id exists nowhere else (no baseline,
    no link), so deleting it would orphan the publication. The 15-minute bound is
    the reviewed DRAFT's lifetime, not the accepted row's.
    """

    cutoff = datetime.now(timezone.utc)
    try:
        candidates = list(directory.glob("*.json"))
    except OSError:
        return
    for candidate in candidates:
        if candidate == keep:
            continue
        raw: Any = None
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            expires = _parse_iso(str(raw.get("expires_at") or ""))
        except (OSError, ValueError):
            expires = None
        if (
            expires is not None
            and expires < cutoff
            and not (isinstance(raw, Mapping) and _receipt_awaits_reconciliation(raw))
        ):
            try:
                candidate.unlink()
            except OSError:
                pass


def _receipt_awaits_reconciliation(record: Mapping[str, Any]) -> bool:
    """Does this receipt hold a hub row whose local link has not landed yet?"""

    accepted = isinstance(record.get("outcome"), Mapping) or isinstance(
        record.get("remote_result"), Mapping
    )
    return accepted and not record.get("completed")


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _resolve_authored_skill(config_dir: Path, name: str) -> Any:
    """The locally discovered skill a publication names, or a named refusal.

    The UI's source picker reads the SAME sessionless catalogue this resolves
    against (``GET /v1/desktop/skills`` with no folder: home roots), so a name the
    picker offered always resolves here and a hand-crafted name cannot reach an
    arbitrary path. Name collisions follow the catalogue's own rule (the earliest
    root wins).
    """

    from local_operator.skills import default_skill_roots, discover_skills

    skills, _warnings = discover_skills(default_skill_roots(Path.home()), config_dir=config_dir)
    match = next((skill for skill in skills if skill.name == name), None)
    if match is None:
        raise LinkError("skill-not-found", "no locally authored skill has that name")
    # A source that already owns a hub link is NOT authored content any more:
    # discovering it through the managed root would let a pulled private-org
    # skill be re-published publicly under a new alias (the baseline is what
    # proves the origin, and the destination-alias check cannot see a rename).
    # This is also why re-publishing a linked skill is refused rather than
    # answered upstream with `name_taken`: the adapter advertises CREATE only.
    if prov.alias_owner(config_dir, "skill", match.name) is not None:
        raise LinkError(
            "origin-conflict",
            "that skill is already linked to the hub; a new publication needs authored content",
        )
    return match


def _project_publication(
    config_dir: Path, *, source_skill: str, local_alias: str
) -> tuple[Any, Any, Any]:
    """Read + validate the authored source and render the exact publication draft.

    Returns ``(skill, source_document, draft_document)``. The source is read with
    :meth:`SkillDocument.from_directory` -- the SAME bounded, all-file validation
    the pull path applies to a remote manifest -- and the draft is its local-alias
    rendering (name/description are stored twice, so ``with_name`` rewrites the
    frontmatter scalar too). Every refusal is the adapter's own named code.
    """

    from local_operator.hub_sync.adapters import SkillAdapter
    from local_operator.hub_sync.skill_document import (
        SkillDocument,
        SkillValidationError,
        validate_name,
    )

    skill = _resolve_authored_skill(config_dir, source_skill)
    try:
        source_document = SkillDocument.from_directory(skill.base_dir)
        alias = validate_name(local_alias)
        draft = source_document.with_name(alias)
        SkillAdapter().validate(draft.to_json())
    except SkillValidationError as error:
        raise LinkError(getattr(error, "code", "") or "skill-validation", str(error)) from None
    return skill, source_document, draft


def _proposal_fingerprint(fields: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {"kind": "skill", "fields": dict(fields)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _publication_blocker(
    ctx: HubSyncContext, skill: Any, source_document: Any, alias: str
) -> str | None:
    """A LOCAL reason this draft can never confirm, or None.

    Checked in the preview so Confirm is disabled with a named reason instead of
    failing after the hub already created a row. Confirm re-checks the same facts
    under the update lease; this is not the authority, only the honest preview.
    """

    from local_operator.hub_sync.skill_storage import ManagedSkillStorage

    try:
        destination = ManagedSkillStorage(ctx.config_dir).path(alias)
    except LinkError:
        return "skill-validation"
    owner = prov.alias_owner(ctx.config_dir, "skill", alias)
    if owner is not None:
        return "origin-conflict"
    if destination.exists():
        try:
            same = destination.resolve() == Path(skill.base_dir).resolve()
        except OSError:
            same = False
        # An authored folder may be adopted only when it IS the published name:
        # adoption never rewrites bytes, so a differing alias would leave the
        # managed folder named differently from its link.
        if not same or alias != source_document.name:
            return "origin-conflict"
    return None


def _publication_preview_reason(
    ctx: HubSyncContext,
    *,
    hub_source: str,
    visibility: Visibility,
    tenant_id: str | None,
    auto_push: bool,
    confirm_public_egress: bool,
    blocker: str | None,
) -> str | None:
    if blocker is not None:
        return blocker
    if ctx.client_for_tenant(tenant_id or "") is None:
        return "no-credential"
    if auto_push:
        # The same capability attestation the policy PATCH and install paths
        # demand. It is a persisted fact about the DEPLOYED server, never a
        # boolean this request can supply.
        probe = LinkIdentity("skill", "preview", hub_source, visibility, tenant_id, "preview")
        if not _capability_auto_push(ctx.store().load(), probe):
            return "unsupported-precondition"
        if visibility == "public" and not confirm_public_egress:
            # The egress grant is minted by the confirmed policy write; the
            # preview may not claim confirmability without the explicit consent.
            return "unsafe-publication"
    return None


def preview_skill_publication(
    ctx: HubSyncContext,
    *,
    source_skill: str,
    local_alias: str,
    visibility: Visibility,
    tenant_id: str | None,
    subscribe: bool,
    auto_push: bool,
    confirm_public_egress: bool,
) -> dict[str, Any]:
    """Project a local skill into the exact initial-publication preview (no network)."""

    from local_operator.hub_sync.identity import approve_source
    from local_operator.providers.radient_credentials import configured_radient_base_url

    # Scope pairing first: a half pair must never reach a credential selector.
    if visibility == "public" and tenant_id is not None:
        raise LinkError("origin-conflict", "a public skill publication cannot name an organization")
    if visibility == "org" and not tenant_id:
        raise LinkError("origin-conflict", "an organization skill publication requires a tenant")

    source = approve_source(
        configured_radient_base_url(ctx.config_manager),
        (configured_radient_base_url(ctx.config_manager),),
    )
    skill, source_document, draft = _project_publication(
        ctx.config_dir, source_skill=source_skill, local_alias=local_alias
    )
    alias = draft.name
    fields = draft.to_json()
    blocker = _publication_blocker(ctx, skill, source_document, alias)
    reason = _publication_preview_reason(
        ctx,
        hub_source=source,
        visibility=visibility,
        tenant_id=tenant_id,
        auto_push=auto_push,
        confirm_public_egress=confirm_public_egress,
        blocker=blocker,
    )
    warnings: list[str] = []
    if alias != source_document.name:
        warnings.append(
            f"This publication renames the skill from {source_document.name!r} to {alias!r}."
        )
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(seconds=PUBLICATION_PREVIEW_TTL_S)
    preview: dict[str, Any] = {
        "preview_receipt": token,
        "kind": "skill",
        "direction": "push",
        "publication": "create",
        "source_skill": source_skill,
        "local_alias": alias,
        "hub_source": source,
        "visibility": visibility,
        "tenant_id": tenant_id,
        "local_fingerprint": source_document.fingerprint,
        "draft_fingerprint": draft.fingerprint,
        "proposal_fingerprint": _proposal_fingerprint(fields),
        "expires_at": expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy": {
            "subscribe": bool(subscribe),
            "auto_push": bool(auto_push),
            "confirm_public_egress": bool(confirm_public_egress),
        },
        "proposal": {"kind": "skill", "fields": fields},
        "warnings": warnings,
        "can_confirm": reason is None,
        "reason": reason,
    }
    record = dict(preview)
    # Private fields: never part of the wire contract, only the local receipt.
    record["source_dir"] = str(Path(skill.base_dir).resolve())
    _write_publication_preview(ctx.config_dir, record)
    return preview


def _publication_from_result(
    result: Mapping[str, Any], proposal: Mapping[str, Any], alias: str
) -> dict[str, Any]:
    """The `publication` half of the UI contract, from the hub's accepted answer.

    ``accepted_document`` is normalised through the adapter's tolerant-envelope /
    strict-manifest reader: the hub echoes metadata beside the manifest, and the
    managed baseline's own finalizer parses its fields as a bare skill document,
    so an envelope stored verbatim would be refused at commit time.
    """

    from local_operator.hub_sync.adapters import SkillAdapter
    from local_operator.hub_sync.skill_document import SkillValidationError

    accepted = result.get("accepted_document")
    raw_accepted_id = str(accepted.get("id") or "").strip() if isinstance(accepted, Mapping) else ""
    if isinstance(accepted, Mapping):
        try:
            accepted_fields = dict(SkillAdapter().canonical_fields(accepted))
        except SkillValidationError as error:
            raise LinkError(
                "hub-upstream", f"the hub's accepted skill document is invalid: {error}"
            ) from None
    else:
        accepted_fields = dict(proposal)
    row = result.get("skill")
    skill_id = ""
    skill_name = alias
    if isinstance(row, Mapping):
        skill_id = str(row.get("id") or "").strip()
        skill_name = str(row.get("name") or "").strip() or skill_name
    if not skill_id:
        skill_id = raw_accepted_id
    if not skill_id:
        raise LinkError("hub-upstream", "the hub accepted the publication but named no skill id")
    revision = result.get("revision")
    content_revision = result.get("content_revision")
    return {
        "skill": {"id": skill_id, "name": skill_name},
        "accepted_document": accepted_fields,
        "revision": str(revision) if isinstance(revision, str) and revision else "",
        "content_revision": (
            content_revision
            if isinstance(content_revision, int) and not isinstance(content_revision, bool)
            else 0
        ),
        "operation_id": str(result.get("operation_id") or ""),
        "sync_contract": (
            dict(result.get("sync_contract") or {})
            if isinstance(result.get("sync_contract"), Mapping)
            else {}
        ),
    }


def _published_baseline(
    identity: LinkIdentity,
    *,
    alias: str,
    accepted: Mapping[str, Any],
    local_fields: Mapping[str, Any],
    revision: str,
    operation_id: str,
    subscribe: bool,
) -> Any:
    """The durable published baseline: the hub's accepted canonical document, not L.

    A publication is the strongest origin proof there is, so the record is
    `recorded_by="publish"` with clean lineage, and the outbound (A, P) receipt is
    seeded in the SAME write: auto-push has no other honest starting point, and a
    first publication is exactly the known-agreed act that proves one.
    """

    from local_operator.hub_sync.push import receipt_for

    record = prov.make_record(
        "skill", identity.local_id, identity.hub_id, identity.tenant_id, accepted, "publish"
    )
    record = replace(
        record,
        identity=identity,
        local_alias=alias,
        remote_revision=revision or None,
        lineage="clean",
        legacy_subscribe_hint=subscribe,
        outbound=receipt_for(
            "skill",
            accepted,
            local_fields,
            revision=revision or None,
            operation_id=operation_id or None,
        ),
    )
    return record


def _materialize_published_skill(
    ctx: HubSyncContext,
    *,
    identity: LinkIdentity,
    alias: str,
    draft: Any,
    baseline: Any,
    source_dir: str,
    subscribe: bool,
    auto_push: bool,
    confirm_public_egress: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Turn an ACCEPTED publication into a real local link.

    Mirrors :func:`install_skill`'s local half under the same cross-process
    update lease: the alias-ownership check, the journaled managed transaction
    (or, when the authored folder already IS the destination, an in-place adopt
    that writes no bytes), and the policy materialization. The answered row is
    the ORDINARY ``links_snapshot`` row, never a synthesized projection.
    """

    from local_operator.config import read_config_values
    from local_operator.hub_sync import policies as pol
    from local_operator.hub_sync.skill_document import (
        SkillDocument,
        SkillValidationError,
    )
    from local_operator.hub_sync.skill_storage import ManagedSkillStorage
    from local_operator.hub_sync.transactions import TransactionError

    storage = ManagedSkillStorage(ctx.config_dir)
    destination = storage.path(alias)
    with exclusive(ctx, wait_s=0.0):
        values = read_config_values(ctx.config_dir)
        if values is None and (ctx.config_dir / "config.yml").exists():
            raise LinkError("invalid-link-policy", "configuration is unreadable")
        existing = prov.read_baseline(ctx.config_dir, "skill", identity.local_id)
        if existing is not None:
            if existing.identity != identity or existing.local_alias != alias:
                raise LinkError("origin-conflict", "a local skill already owns this identity")
            # REPLAY ARM: the remote outcome was persisted, but a crash may have
            # landed the baseline without the policy write (or the caller may be
            # repeating a completed confirm). Materialize is idempotent; only a
            # requested auto-push that is not ON yet needs the eligibility gate.
            current = pol.read_policy(values or {}, identity)
            if auto_push and not current.policy.auto_push:
                eligibility = _eligibility(
                    ctx, identity, current.policy, values or {}, baseline, ctx.store().load()
                )
                pol.update_policy(
                    ctx.config_manager,
                    identity,
                    expected_revision=current.revision,
                    subscribe=subscribe,
                    auto_push=True,
                    confirm_public_egress=confirm_public_egress,
                    can_enable_auto_push=bool(eligibility["can_auto_push"]),
                    ineligible_reason=str(eligibility["reason"]),
                )
            else:
                pol.materialize_policy(
                    ctx.config_manager, identity, legacy_subscribe_hint=subscribe
                )
            operation_id = f"link-{identity.local_id}"
        else:
            owner = prov.alias_owner(ctx.config_dir, "skill", alias)
            if owner not in (None, identity.local_id):
                raise LinkError(
                    "origin-conflict", "managed alias belongs to another local skill origin"
                )
            current = pol.read_policy(values or {}, identity)
            adopt = destination.exists()
            if adopt:
                # The authored source and the managed destination are one folder;
                # adoption writes no bytes, so the alias must already be the name
                # the draft carries. Confirm re-reads it under the lease: an edit
                # since the preview is a concurrent-edit refusal, not a silent
                # baseline over changed content.
                try:
                    same = destination.resolve() == Path(source_dir).resolve()
                except OSError:
                    same = False
                if not same:
                    raise LinkError(
                        "origin-conflict", "refusing to overwrite an unknown local skill"
                    )
                if SkillDocument.from_directory(destination).fingerprint != draft.fingerprint:
                    raise LinkError(
                        "local-concurrent-edit", "managed skill changed before publication"
                    )
            staged_revision: str | None = None
            if auto_push:
                eligibility = _eligibility(
                    ctx, identity, current.policy, values or {}, baseline, ctx.store().load()
                )
                current = pol.update_policy(
                    ctx.config_manager,
                    identity,
                    expected_revision=current.revision,
                    subscribe=subscribe,
                    auto_push=True,
                    confirm_public_egress=confirm_public_egress,
                    can_enable_auto_push=bool(eligibility["can_auto_push"]),
                    ineligible_reason=str(eligibility["reason"]),
                )
                staged_revision = current.revision
            try:
                if adopt:
                    prov.write_baseline(ctx.config_dir, baseline)
                    pol.materialize_policy(
                        ctx.config_manager, identity, legacy_subscribe_hint=subscribe
                    )
                    operation_id = f"link-{identity.local_id}"
                else:
                    transaction = storage.apply(
                        identity,
                        alias,
                        draft,
                        baseline,
                        expected_fingerprint=None,
                        expected_policy_revision=current.revision,
                        automatic=False,
                        basis="initial-publish",
                    )
                    operation_id = transaction.operation_id
            except Exception as error:
                # A staged auto-push entry must not orphan a link that never
                # landed -- but a grant whose content/baseline DID land is the
                # recovered install's policy and must survive (the same split
                # install_skill's settlement makes). Only revert when neither
                # durable fact is present.
                if staged_revision is not None:
                    landed = (
                        prov.read_baseline(ctx.config_dir, "skill", identity.local_id) is not None
                    )
                    if not landed and not adopt:
                        try:
                            landed = (
                                destination.exists()
                                and SkillDocument.from_directory(destination).fingerprint
                                == draft.fingerprint
                            )
                        except Exception:  # noqa: BLE001 - cannot prove it did not land
                            landed = True
                    if not landed:
                        try:
                            pol.revert_policy_if_unchanged(
                                ctx.config_manager, identity, expected_revision=staged_revision
                            )
                        except Exception:  # noqa: BLE001 - never mask the original failure
                            pass
                if isinstance(error, LinkError):
                    raise
                if isinstance(
                    error, (TransactionError, SkillValidationError, OSError, prov.BaselineError)
                ):
                    raise LinkError(getattr(error, "code", "") or "hub-error", str(error)) from None
                raise

    # The managed transaction writes policy through its OWN manager instance; the
    # request's manager still holds the pre-write snapshot, and reading the link
    # row from it would report a creation default instead of what was stored.
    # Reload before projecting the row so the answer is the persisted policy.
    ctx.config_manager.reload()
    snapshot = links_snapshot(ctx)
    row = next((link for link in snapshot["links"] if link["link_id"] == identity.link_id), None)
    if row is None:
        raise LinkError("origin-conflict", "the publication did not materialize a link row")
    receipt = {
        "state": "complete",
        "operation_id": operation_id,
        "link_id": identity.link_id,
        "safe_message": None,
    }
    return row, receipt


def _project_accepted_publication(
    result: Mapping[str, Any], proposal: Mapping[str, Any], alias: str
) -> dict[str, Any] | None:
    """Project an ACCEPTED hub result, or None when its shape cannot be read.

    A projection failure happens AFTER the hub created the row. It must reach the
    caller as `pending-reconciliation`, never as a 502: the row exists, and
    "nothing happened" would invite a duplicate publication.
    """

    try:
        return _publication_from_result(result, proposal, alias)
    except LinkError as error:
        logger.warning("could not project the hub's accepted skill publication: %s", error)
        return None


def _pending_publication(
    publication: Mapping[str, Any] | None, local_id: str | None, message: str
) -> dict[str, Any]:
    """The `pending-reconciliation` answer for an accepted row whose link is not done."""

    return {
        "preview": None,
        "link": None,
        "local_link_receipt": {
            "state": "pending-reconciliation",
            "operation_id": f"link-{local_id}" if local_id else "link-pending",
            "link_id": None,
            "safe_message": message,
        },
        "publication": dict(publication) if publication is not None else None,
    }


def _recorded_draft(record: Mapping[str, Any]) -> Any:
    """Rebuild the reviewed draft from a receipt, WITHOUT re-reading the source.

    Reconciliation depends on the bytes the hub accepted, not on whatever now
    sits in the authored folder: the source may have been edited or removed, and
    refusing the retry for that would orphan the remote row forever.
    """

    from local_operator.hub_sync.skill_document import (
        SkillDocument,
        SkillValidationError,
    )

    proposal = record.get("proposal")
    fields = proposal.get("fields") if isinstance(proposal, Mapping) else None
    if not isinstance(fields, Mapping):
        raise LinkError("preview_stale", "the publication receipt carries no reviewed draft")
    try:
        return SkillDocument.from_json(dict(fields))
    except SkillValidationError as error:
        raise LinkError("preview_stale", f"the recorded publication draft is invalid: {error}")


def publish_skill(
    ctx: HubSyncContext,
    *,
    source_skill: str,
    local_alias: str,
    visibility: Visibility,
    tenant_id: str | None,
    subscribe: bool,
    auto_push: bool,
    confirm_public_egress: bool,
    dry_run: bool = False,
    preview_receipt: str | None = None,
) -> dict[str, Any]:
    """Initial skill publication: preview (dry run) or confirm a reviewed draft.

    The confirm path is two-phase and crash-ordered:

    1. the preview receipt is re-read, expiry-checked and compared field-for-field
       with the request;
    2. the source is re-read from disk and ALL-file bytes are compared with the
       fingerprints the reviewed draft pinned (`local_fingerprint`,
       `draft_fingerprint`, `proposal_fingerprint`) -- an edit since the review is
       `preview_stale`, never a publication of unreviewed bytes;
    3. the hub call runs BEFORE any local write, and a refusal leaves every piece
       of local skill state untouched;
    4. the remote outcome is persisted into the receipt before the managed link
       transaction, so a retry with the same receipt resumes the SAME local link
       (the deterministic id is minted from the published hub id) instead of
       publishing twice;
    5. the local link is materialized exactly as ``install_skill`` does and the
       answer carries that real inventory row.

    A local failure AFTER an accepted publication is reported as
    ``pending-reconciliation`` -- the remote row exists and saying otherwise
    would invite a duplicate publication.
    """

    from local_operator.clients._http import APIError
    from local_operator.hub_sync.identity import approve_source
    from local_operator.providers.radient_credentials import (
        ORG_LOGIN_REMEDY,
        configured_radient_base_url,
    )

    if dry_run:
        return {
            "preview": preview_skill_publication(
                ctx,
                source_skill=source_skill,
                local_alias=local_alias,
                visibility=visibility,
                tenant_id=tenant_id,
                subscribe=subscribe,
                auto_push=auto_push,
                confirm_public_egress=confirm_public_egress,
            ),
            "link": None,
            "local_link_receipt": None,
            "publication": None,
        }

    if not preview_receipt:
        raise LinkError("preview_mismatch", "a publication confirm requires its preview receipt")
    record = _read_publication_preview(ctx.config_dir, preview_receipt)
    if record is None:
        raise LinkError("preview_expired", "the publication preview receipt is unknown or expired")
    durable = isinstance(record.get("outcome"), Mapping) or isinstance(
        record.get("remote_result"), Mapping
    )
    expires = _parse_iso(str(record.get("expires_at") or ""))
    if not durable and (expires is None or datetime.now(timezone.utc) > expires):
        raise LinkError("preview_expired", "the publication preview has expired; preview again")

    expected = {
        "source_skill": source_skill,
        "local_alias": local_alias,
        "visibility": visibility,
        "tenant_id": tenant_id,
        "policy": {
            "subscribe": bool(subscribe),
            "auto_push": bool(auto_push),
            "confirm_public_egress": bool(confirm_public_egress),
        },
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise LinkError("preview_mismatch", "the publication draft changed after its preview")

    hub_source = str(record.get("hub_source") or "")

    recorded_outcome = record.get("outcome")
    publication: dict[str, Any] | None = (
        dict(recorded_outcome) if isinstance(recorded_outcome, Mapping) else None
    )
    if publication is not None:
        # RECONCILIATION ARM: the hub accepted this row. Local bookkeeping is the
        # only work left, and it must not depend on the authored source -- the
        # source may have moved, been edited or been deleted since the review.
        draft = _recorded_draft(record)
        fields = draft.to_json()
        source_dir = str(record.get("source_dir") or "")
    elif isinstance(record.get("remote_result"), Mapping):
        # The hub accepted but its response could not be projected (or the
        # projected outcome could not be persisted). Re-read the RAW accepted
        # result, which was made durable at acceptance, so the retry never asks
        # for a fresh review nor republishes.
        draft = _recorded_draft(record)
        fields = draft.to_json()
        source_dir = str(record.get("source_dir") or "")
        publication = _project_accepted_publication(
            dict(record["remote_result"]), fields, draft.name
        )
        if publication is None:
            return _pending_publication(
                None,
                None,
                "the hub accepted the publication but its response could not be read",
            )
        record["outcome"] = publication
        try:
            _write_publication_preview(ctx.config_dir, record)
        except Exception:  # noqa: BLE001 - the raw result is already durable
            logger.warning("could not promote the publication outcome", exc_info=True)
    else:
        skill, source_document, draft = _project_publication(
            ctx.config_dir, source_skill=source_skill, local_alias=local_alias
        )
        try:
            recorded_dir = Path(str(record.get("source_dir") or "")).resolve()
        except OSError:
            recorded_dir = None
        if recorded_dir is None or recorded_dir != Path(skill.base_dir).resolve():
            # A different folder now answers this name (a new higher-priority root):
            # the reviewed bytes are not this draft, even if a file happens to agree.
            raise LinkError("preview_stale", "the authored skill moved since its preview")
        fields = draft.to_json()
        if (
            record.get("local_fingerprint") != source_document.fingerprint
            or record.get("draft_fingerprint") != draft.fingerprint
            or record.get("proposal_fingerprint") != _proposal_fingerprint(fields)
        ):
            raise LinkError("preview_stale", "the authored skill changed since its preview")

        configured = approve_source(
            configured_radient_base_url(ctx.config_manager),
            (configured_radient_base_url(ctx.config_manager),),
        )
        if hub_source != configured:
            # The configured destination moved since the review. Re-targeting a
            # reviewed publication silently would send the bytes somewhere the user
            # did not review. (Reconciliation does not re-target: it never calls the
            # hub, so it keeps the recorded destination.)
            raise LinkError("preview_stale", "the hub destination changed since the preview")

        # A preview whose Confirm control was disabled must not be confirmable by
        # replaying its receipt directly: the same local/capability/consent gates
        # are re-evaluated HERE, before the hub can create a row.
        preflight = _publication_preview_reason(
            ctx,
            hub_source=hub_source,
            visibility=visibility,
            tenant_id=tenant_id,
            auto_push=auto_push,
            confirm_public_egress=confirm_public_egress,
            blocker=_publication_blocker(ctx, skill, source_document, draft.name),
        )
        if preflight is not None:
            raise LinkError(preflight, "the publication preview is not confirmable")
        client = ctx.client_for_tenant(tenant_id or "")
        if client is None:
            raise LinkError("no-credential", ORG_LOGIN_REMEDY)
        if str(getattr(client, "base_url", "")).rstrip("/") != hub_source:
            raise LinkError(
                "origin-conflict", "the answering client's base is not the requested source"
            )
        try:
            result = client.publish_skill_document(
                dict(fields), visibility=visibility, tenant_id=tenant_id
            )
        except APIError as error:
            raise LinkError(getattr(error, "code", None) or "hub-error", str(error)) from None

        # THE HUB HAS ACCEPTED. Persist the RAW result before any projection: a
        # 2xx whose shape cannot be read still names a row that exists, and a 502
        # would invite a duplicate publication. This file is the only record of
        # the hub id until a baseline lands.
        raw_result = dict(result) if isinstance(result, Mapping) else None
        if raw_result is None:
            return _pending_publication(
                None,
                None,
                "the hub accepted the publication but returned an unreadable result",
            )
        record["remote_result"] = raw_result
        try:
            _write_publication_preview(ctx.config_dir, record)
        except Exception:  # noqa: BLE001 - report pending, never 502, after acceptance
            logger.warning("could not record the accepted publication", exc_info=True)
            return _pending_publication(
                None,
                None,
                "the hub accepted the publication but its outcome could not be recorded",
            )
        publication = _project_accepted_publication(raw_result, fields, draft.name)
        if publication is None:
            return _pending_publication(
                None,
                None,
                "the hub accepted the publication but its response could not be read",
            )
        record["outcome"] = publication
        try:
            _write_publication_preview(ctx.config_dir, record)
        except Exception:  # noqa: BLE001 - raw_result is durable; the retry re-projects
            logger.warning("could not persist the projected publication", exc_info=True)
        source_dir = str(Path(skill.base_dir).resolve())

    # Every arm above either returned a pending answer or set both facts; the
    # assertion makes that a runtime invariant as well as a typing one.
    assert publication is not None
    hub_id = str(publication["skill"]["id"])
    local_id = skill_link_local_id(
        source=hub_source, visibility=visibility, tenant_id=tenant_id, hub_id=hub_id
    )
    identity = LinkIdentity("skill", local_id, hub_source, visibility, tenant_id, hub_id)
    accepted = publication.get("accepted_document")
    accepted_fields = dict(accepted) if isinstance(accepted, Mapping) else fields
    baseline = _published_baseline(
        identity,
        alias=draft.name,
        accepted=accepted_fields,
        local_fields=fields,
        revision=str(publication.get("revision") or ""),
        operation_id=str(publication.get("operation_id") or ""),
        subscribe=subscribe,
    )
    try:
        row, local_receipt = _materialize_published_skill(
            ctx,
            identity=identity,
            alias=draft.name,
            draft=draft,
            baseline=baseline,
            source_dir=source_dir,
            subscribe=subscribe,
            auto_push=auto_push,
            confirm_public_egress=confirm_public_egress,
        )
    except Exception as error:
        # The hub row EXISTS. Reporting the failure as an error would read as
        # "nothing happened" and invite a duplicate publication; the contract's
        # pending receipt is the honest state, and the persisted outcome lets a
        # retry reconcile this same link.
        logger.warning("could not materialize the published skill link", exc_info=True)
        return _pending_publication(
            publication,
            local_id,
            (
                str(error)
                if isinstance(error, LinkError)
                else "the local link could not be materialized"
            ),
        )
    record["completed"] = True
    _write_publication_preview(ctx.config_dir, record)
    return {
        "preview": None,
        "link": row,
        "local_link_receipt": local_receipt,
        "publication": publication,
    }
