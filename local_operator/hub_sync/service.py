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
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
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
    Visibility,
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


def seed_receipt_after_publish(
    config_dir: Any,
    kind: Kind,
    local_id: str,
    *,
    accepted: Mapping[str, Any],
    local_fields: Mapping[str, Any],
    revision: str | None,
) -> bool:
    """Seed A/P after an ACCEPTED manual publish/republish (parent ruling (A)b).

    The server returned the canonical document and its ETag; A is the local input
    that was published and P the accepted document. Nothing client-side is
    invented: without a revision (an older server) nothing is seeded and the link
    stays honestly ``no-receipt``. The inbound baseline B is untouched.
    """

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
    if record:
        live = None
        if names is None and only_ids is None and set(kinds) == {"agent", "team"}:
            live = {st.item_key(c.kind, c.local_id) for c in out}
        record_checks(ctx, out, live=live)
    return out


# -- merge -------------------------------------------------------------------------------


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

    try:
        local_doc = SkillDocument.from_json(dict(c.local or {}))
        remote_doc = SkillDocument.from_json(dict(c.remote or {}))
        base_doc = SkillDocument.from_json(dict(c.base)) if c.base else None
    except SkillValidationError as exc:
        return fail("merge-refused", f"the skill document is invalid: {exc}", "refused")

    results: list[MergeResult] = []
    if replace == "remote":
        merged: SkillDocument | None = remote_doc
    elif replace == "local":
        merged = local_doc
    else:
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
    if merged.to_json() == local_doc.to_json() and replace is None:
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
        before_fingerprint = local_doc.with_name(alias).fingerprint
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
    kinds = (kind,) if kind else ("agent", "team")
    todo = list(checks) if checks is not None else check_items(ctx, kinds=kinds, names=names)
    todo.sort(key=lambda c: (0 if c.kind == "agent" else 1, c.name.casefold()))
    reports: list[ItemMergeReport] = []
    stopped: str | None = None
    for c in todo:
        if kind and c.kind != kind:
            continue
        if all_available and c.verdict != "available":
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
        if kind not in ("agent", "team"):
            # The legacy typed projection never learned a third kind: skill rows
            # (and their counts) travel on the capability-gated /hub/links route
            # only, so an older consumer can never receive an unexpected shape.
            continue
        state = st.effective_state(item, now)
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
    items.sort(key=lambda i: (i["kind"], str(i["name"]).casefold()))
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
    baseline_state, remote_revision, and INDEPENDENT `pull`/`push`
    DirectionStatus objects. Distinct from ``status_snapshot`` (attention-only):
    this lists a healthy idle link too, so the sidebar can show its quiet mark and
    offer a source-scoped toggle with zero pending changes. Local reads only.

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
                "auto_will_apply": _auto_will_apply(item, policy, ctx, now),
                "auto_will_push": _auto_will_push(item, policy, ctx, now, eligibility),
                "baseline_state": record.recorded_by,
                "remote_revision": record.remote_revision,
                "pull": _direction_status(item, "pull", policy, now, capability),
                "push": _direction_status(item, "push", policy, now, capability),
            }
        )
    links.sort(key=lambda link: (link["kind"], str(link["name"]).casefold()))
    unresolved.sort(key=lambda row: (row["kind"], str(row["name"]).casefold()))
    return {
        "generated_at": prov.now_iso(),
        "capability": "hub_links:1",
        "links": links,
        "unresolved": unresolved,
    }


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


def _auto_will_apply(item: Mapping[str, Any], policy: Any, ctx: HubSyncContext, now: Any) -> bool:
    """Whether the next run would apply an incoming change: policy+backoff+grace."""

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
        SkillDocument,
        SkillValidationError,
        validate_name,
    )
    from local_operator.hub_sync.skill_storage import ManagedSkillStorage
    from local_operator.hub_sync.transactions import TransactionError
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
    #    would attach clean, trusted ancestry and source-scoped policy to
    #    another scope's (or another identity's) content. Refusals are the SAME
    #    flat 404 as a missing or foreign row: a probe must not be able to tell
    #    the checks apart. Absent/unknown metadata fails closed -- the real
    #    detail envelope always carries id + visibility, and tenant_id for org
    #    rows.
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
        # An install records the ancestry fact; without a revision the link
        # could never prove what it integrated. Refuse now, write nothing.
        raise LinkError("unsupported-precondition", "the remote skill carries no revision")

    # 6. Deterministic identity: a retry after a crash resumes the SAME link.
    local_id = skill_link_local_id(
        source=source, visibility=visibility, tenant_id=tenant_id, hub_id=hub_id
    )
    identity = LinkIdentity("skill", local_id, source, visibility, tenant_id, hub_id)

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
        #     above), never a second copy.
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
        baseline = replace(
            baseline,
            identity=identity,
            local_alias=alias,
            remote_revision=revision,
            lineage="clean",
            legacy_subscribe_hint=subscribe,
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
        elif pol.stored_link_policy(values or {}, identity.link_id) is not None:
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
