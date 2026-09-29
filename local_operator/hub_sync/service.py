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

PARTIAL APPLY. Deliberately not done: an item is applied whole or not at all. The
design lets an agent's ``description`` apply while an ``instructions`` region is
unresolved; that needs per-field baselines and is listed as a follow-up rather
than half-built here. It is the conservative direction (nothing lands until a
human resolves the item).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from local_operator.hub_sync import check as chk
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync import store as st
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


class ConcurrentEdit(Exception):
    """The row changed between the snapshot and the write; nothing was written."""


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
        return HubSyncSettings.from_config(self.config_manager)

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


def check_items(
    ctx: HubSyncContext,
    *,
    kinds: Sequence[str] = ("agent", "team"),
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


def _missing_roles(ctx: HubSyncContext, results: Sequence[MergeResult]) -> tuple[str, ...]:
    """A roster naming a role nobody has installed is a warning, never a refusal (A7)."""

    roster = next((r for r in results if r.field == "members" and r.outcome == "merged"), None)
    if roster is None:
        return ()
    try:
        from local_operator.agent_profiles import list_seeds

        known = {a.name.casefold() for a in ctx.agents().list_agents()} | {
            s.casefold() for s in list_seeds()
        }
    except Exception:  # noqa: BLE001 - a warning must never block an apply
        return ()
    return tuple(
        f"missing-role:{slot['role']}"
        for slot in roster.merged  # type: ignore[union-attr]
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
    if c.verdict == "up-to-date":
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
        fold(lambda item: st.record_failure(item, cls, message, summary=summary))
        return _with(report, outcome=outcome, error_class=cls, message=message)

    if report.outcome in ("needs-review", "refused"):
        return fail(
            report.error_class or "merge-refused",
            report.message or "the hub and your copy both changed the same text",
            outcome=report.outcome,
        )
    if auto and report.outcome == "merged":
        # Auto-update only ever applies a clean merge (B3.1): a large shrink or an
        # unknown baseline waits for a human.
        if "large-shrink" in report.warnings:
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
            sub = _subclass_of(report)
            label = f"{cls}/{sub}" if sub else cls
            if cls in ("no-credential", "model-unavailable") or label in st.SYSTEMIC:
                stopped = label
    return ApplyReport(tuple(reports))


def _subclass_of(report: ItemMergeReport) -> str | None:
    for r in report.fields:
        fc = r.engine.failure_class
        if fc and "/" in fc:
            return fc.split("/", 1)[1]
    return None


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
        if state == "up-to-date":
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
