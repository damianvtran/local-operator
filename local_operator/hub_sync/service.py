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
from local_operator.hub_sync.report import ApplyReport, ItemMergeReport
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
        if live is not None:
            st.prune_items(doc, live)

    ctx.store().mutate(fold)


def _adopt_if_equal(ctx: HubSyncContext, c: chk.ItemCheck) -> None:
    """L == R with no baseline: the texts agree, so B := L (A2.3.2)."""

    if c.verdict != "up-to-date" or c.base is not None or c.local is None:
        return
    record = prov.make_record(c.kind, c.local_id, c.hub_id, c.tenant_id, c.local, "adopt")
    try:
        prov.write_baseline(ctx.config_dir, record)
    except (prov.BaselineError, OSError):
        logger.debug("could not adopt baseline for %s %s", c.kind, c.local_id, exc_info=True)


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
    """
    from local_operator.agents import hub_origin

    registry = ctx.agents() if kind == "agent" else ctx.teams()
    if kind == "agent":
        row = registry.get_agent_by_name(name)
        hub_id = hub_origin(row) if row is not None else None
        local_id = str(row.id) if row is not None else None
        fields = _local_fields_for(ctx, kind, row)
    else:
        row = registry.get_team_by_name(name)
        local_id = str(row.id) if row is not None else None
        # A team's remote id lives in its baseline record (a linked team HAS one),
        # never on the row itself.
        linked = (
            prov.read_baseline(ctx.config_dir, kind, local_id) if local_id is not None else None
        )
        hub_id = linked.hub_id if linked is not None else None
        fields = _local_fields_for(ctx, kind, row)
    if row is None or hub_id is None or local_id is None:
        raise LookupError(f"no linked {kind} named {name!r}")

    identity = LinkIdentity(kind, local_id, hub_source, visibility, tenant_id, hub_id)
    existing = prov.read_baseline(ctx.config_dir, kind, local_id)
    if existing is not None and existing.identity is not None:
        # One writable origin per artifact: a differing source/scope/remote id
        # refuses here rather than relinking, and the old record is untouched.
        existing.identity.require_same_origin(identity)
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
    if roster is None or isinstance(roster.merged, str):
        return ()  # a roster is always a list; the str arm only narrows the type
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


def _write_team(ctx: HubSyncContext, c: chk.ItemCheck, report: ItemMergeReport) -> None:
    from local_operator.teams import TeamEditFields, TeamMember

    changed = {r.field: r.merged for r in report.fields if r.outcome == "merged"}
    payload: dict[str, Any] = {}
    for key, value in changed.items():
        if key == "members":
            payload[key] = [TeamMember(**slot) for slot in value]  # type: ignore[union-attr]
        else:
            payload[key] = value

    def precondition(current: Any) -> None:
        now_fp = prov.fingerprint_team(chk.team_fields_of(current))
        if now_fp != c.local_fp:
            raise ConcurrentEdit("the team changed while the merge was being computed")

    ctx.teams().update_team(c.local_id, TeamEditFields(**payload), precondition=precondition)


def _advance_baseline(ctx: HubSyncContext, c: chk.ItemCheck, recorded_by: str) -> None:
    """B := the remote text just integrated (see ``merge`` docstring)."""

    if c.remote is None:
        return
    record = prov.make_record(c.kind, c.local_id, c.hub_id, c.tenant_id, c.remote, recorded_by)
    prov.write_baseline(ctx.config_dir, record)


def _local_fields_for_backup(c: chk.ItemCheck) -> dict[str, Any]:
    return dict(c.local or {})


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
        return ItemMergeReport(c.kind, c.local_id, c.name, c.hub_id, "up-to-date")

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

    def fail(cls: str, message: str, outcome: str = "failed") -> ItemMergeReport:
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
        state = st.effective_state(item, now)
        counts[state] = counts.get(state, 0) + 1
        # An up-to-date item is listed only when it carries the sign-in fact
        # (``no-credential``): the user has no login for an item they have linked, and
        # this list is the only place the UI can learn it (UX round 2, U11).
        if state == "up-to-date" and item.get("error_class") != "no-credential":
            continue
        kind = item.get("kind", "agent")
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
                "eligibility": _eligibility(ctx, identity, policy, values, record, doc),
                "auto_will_apply": _auto_will_apply(item, policy, ctx, now),
                "auto_will_push": _auto_will_push(
                    item, policy, ctx, now, _eligibility(ctx, identity, policy, values, record, doc)
                ),
                "baseline_state": record.recorded_by,
                "remote_revision": record.remote_revision,
                "pull": _direction_status(item, "pull", policy, now, _capability_auto_push(doc)),
                "push": _direction_status(item, "push", policy, now, _capability_auto_push(doc)),
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


def _capability_auto_push(doc: Mapping[str, Any]) -> bool:
    """Whether the deployed server's atomic revision-CAS is POSITIVELY attested.

    A positive attestation, not a constant: a verification pass (run against the
    deployed server) records ``content_revision`` + ``If-Match`` CAS support in
    the status store, and this reads that persisted fact locally -- the inventory
    is a local read and must never probe the network. Absent the attestation the
    answer is honestly False, so an unattested server cannot be treated as if it
    supported CAS. Enabling auto-push is a decision this gates; it never fakes the
    capability.
    """

    server = doc.get("server", {}) if isinstance(doc, Mapping) else {}
    return bool(isinstance(server, Mapping) and server.get("revision_cas") is True)


def record_server_capabilities(ctx: HubSyncContext, capabilities: Mapping[str, Any]) -> None:
    """Persist a VERIFIED server capability attestation. PROBE-ONLY.

    Called by exactly one kind of caller: the outbound probe that has actually
    spoken to the deployed server and seen its atomic revision-CAS behaviour.

    SECURITY INVARIANT: this must NEVER be reachable from a route, a UI action,
    a config key or a command. Attestation is a fact about the DEPLOYED server,
    so a local write is only honest when a request produced it; letting a client
    set ``revision_cas`` would let it SELF-GRANT publication eligibility, which
    is the exact thing ``_eligibility`` exists to refuse. The persisted flag is
    therefore read by the inventory and written solely by the probe -- there is
    no request field for it anywhere. (Verified: this function has no route/CLI
    caller; the probe is wired with the outbound surface.)

    The revision-CAS flag is the ONLY fact eligibility needs here; recording it
    in the status store keeps the inventory a pure local read.
    """

    revision_cas = bool(capabilities.get("revision_cas") is True)

    def fold(doc: dict[str, Any]) -> None:
        server = doc.setdefault("server", {})
        server["revision_cas"] = revision_cas
        server["attested_at"] = prov.now_iso()

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
    if not _capability_auto_push(doc):
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
