"""Bulk pull of an organization's catalogue: one call, every kind, subscribed.

WHY. "Pull the org's latest and stay subscribed" used to take one command per
item (about 20 for a typical org). Agents and teams were also NOT idempotent:
every import minted a new row, so a second run produced ``coder-2``. Skills
refused a same-named local folder outright. :func:`pull_all` is the one
idempotent entry. The CLI (``lop hub org pull``), the ``agent`` tool
(``org_pull``) and slice B's auto-enrol all call it, so none of them grows its
own rules.

IDEMPOTENCE. Every item is classified against the existing links first
(:func:`hub_org.catalogue.local_index`, re-read per item so earlier pulls in the
same batch are seen). A linked item is ``already-linked``: updates stay the
runner's/``sync``'s job, so there is no second update path.

CONFLICTS (an unlinked local copy with the org item's name), ``on_conflict``:

- ``adopt-link`` (default; automatic runs too): back the local copy up ONCE
  and link it subscribed WITHOUT writing a byte of its content. Identical
  content is ``identical``. Differing content is ``kept-local-linked``: the
  baseline is the HUB revision, so the check classifies the copy
  ``local-only`` and ``sync --check`` shows the drift instead of it going
  stale unseen (the Minerva-skills incident: unlinked copies drifted with
  nothing to see them). adopt-link NEVER merges and never calls a model: a
  baseline-less two-way merge concatenates differing lines (design review U1:
  ``description: Acme writing stylemine``), and authored work must not change
  without an explicit command (design note §5/§8).
- ``adopt``: hub wins. The local copy is backed up under ``hub/backups/`` first.
- ``reconcile``: the existing merge (``apply_items(acknowledge_unknown_baseline)``
  for agents/teams, the reviewed skill adoption ``merge`` strategy for skills).
  Only ever an explicit choice: ``--on-conflict reconcile`` on the CLI, or a
  ``confirm_replace=true`` tool call. Automatic runs never reach it.
- ``skip``: report ``conflict-skipped`` with the command that resolves it.

A local item with the org item's name that already follows a DIFFERENT hub
item is ``linked-elsewhere`` (:mod:`hub_org.catalogue`): reported, never
retagged, linked or merged, under every ``on_conflict``.

AUTO-PUSH is off unless the caller asks AND the member is an org owner/admin
AND the service's own ``push_eligibility`` agrees. An ineligible request is
reported in the item detail, never silently set. Per-item failures never stop
the batch.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from local_operator.hub_org.catalogue import (
    KINDS,
    CatalogueItem,
    item_from_row,
    local_index,
)

logger = logging.getLogger(__name__)

OnConflict = Literal["adopt-link", "adopt", "reconcile", "skip"]
ON_CONFLICT_CHOICES: tuple[str, ...] = ("adopt-link", "adopt", "reconcile", "skip")

#: Roles that may turn on unattended publication for an org link.
_PUSH_ROLES = frozenset({"owner", "admin"})

#: Outcomes, stable on the wire (slice B's enrol retries ``failed`` only).
#: already-linked | pulled | adopted | adopt-linked | reconciled |
#: conflict-skipped | linked-elsewhere | needs-review | failed


@dataclass(frozen=True)
class PullOutcome:
    kind: str
    hub_id: str
    name: str
    outcome: str
    detail: str = ""
    backup: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrgPullReport:
    items: tuple[PullOutcome, ...] = ()
    #: One entry per conflict that ended linked: {name, kind, hub_id, result,
    #: backup}. ``result`` is identical | reconciled | kept-local-linked |
    #: hub-wins (the last only for an explicit ``adopt``).
    adopted: tuple[dict[str, Any], ...] = ()
    errors: dict[str, str] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for item in self.items:
            out[item.outcome] = out.get(item.outcome, 0) + 1
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "items": [i.to_json() for i in self.items],
            "adopted": [dict(a) for a in self.adopted],
            "counts": self.counts(),
            "errors": dict(self.errors),
        }


class _Run:
    """Per-call state: the clients, registries and the member's role."""

    def __init__(
        self,
        config_manager: Any,
        tenant: str,
        *,
        client: Any,
        agent_registry: Any,
        team_registry: Any,
        allow_llm: bool,
    ) -> None:
        from local_operator.hub_sync import service as svc

        self.config_manager = config_manager
        self.config_dir = Path(config_manager.config_dir)
        self.tenant = tenant
        self.client = client
        self.allow_llm = allow_llm
        if agent_registry is None:
            from local_operator.agents import AgentRegistry

            agent_registry = AgentRegistry(self.config_dir)
        if team_registry is None:
            from local_operator.teams import TeamRegistry

            team_registry = TeamRegistry(self.config_dir)
        self.agents = agent_registry
        self.teams = team_registry
        self.ctx = svc.sync_context(config_manager)
        self.ctx.agent_registry, self.ctx.team_registry = agent_registry, team_registry

    def origin(self) -> Any:
        from local_operator.hub_sync import provenance as prov
        from local_operator.hub_sync.identity import link_origin

        base = prov.serving_base(self.client, self.config_dir) or ""
        return link_origin(base, "org", self.tenant)

    def hub_source(self) -> str:
        from local_operator.hub_sync.provenance import canonical_hub_source

        return canonical_hub_source(self.config_manager) or ""


# -- per-kind writers -------------------------------------------------------------------


def _pull_agent(run: _Run, item: CatalogueItem) -> PullOutcome:
    outcome = run.agents.download_agent_from_radient(
        run.client, item.hub_id, with_credential=True, tenant_id=run.tenant
    )
    return PullOutcome("agent", item.hub_id, item.name, "pulled", f"stored as {outcome.agent.name}")


def _fetch_team(run: _Run, hub_id: str) -> dict[str, Any]:
    document = run.client.get_team(hub_id)
    owner = str(document.get("tenant_id") or "")
    if owner and owner != run.tenant:
        # The list said this org; the row says another. Refuse rather than link
        # a foreign row under this organization's identity.
        raise ValueError(f"team belongs to organization {owner!r}, not {run.tenant!r}")
    return document


def _pull_team(run: _Run, item: CatalogueItem) -> PullOutcome:
    from local_operator.hub_sync.provenance import record_team_pull

    document = _fetch_team(run, item.hub_id)
    imported = run.teams.import_hub_team(document)
    record_team_pull(
        run.config_dir,
        run.teams.get_team(imported.team.id),
        document,
        tenant_id=run.tenant,
        origin=run.origin(),
    )
    return PullOutcome("team", item.hub_id, item.name, "pulled", f"stored as {imported.team.name}")


def _install_skill(run: _Run, item: CatalogueItem, **kwargs: Any) -> dict[str, Any]:
    from local_operator.hub_sync import service as svc

    return svc.install_skill(
        run.ctx,
        hub_source=run.hub_source(),
        hub_id=item.hub_id,
        visibility="org",
        tenant_id=run.tenant,
        alias=None,
        allow_llm=run.allow_llm,
        **kwargs,
    )


def _pull_skill(run: _Run, item: CatalogueItem, subscribe: bool) -> PullOutcome:
    # ``confirm`` is the org arm's authorization: an org bundle is authorized by
    # membership (no publisher pin), and a document install never asks.
    _install_skill(run, item, subscribe=subscribe, confirm=True)
    return PullOutcome("skill", item.hub_id, item.name, "pulled")


# -- conflicts: agents and teams ---------------------------------------------------------


def _local_agent_fields(run: _Run, local_id: str) -> dict[str, Any]:
    from local_operator.hub_sync.segment import norm

    agent = run.agents.get_agent(local_id)
    return {
        "instructions": norm(run.agents.get_agent_system_prompt(local_id)),
        "description": norm(agent.description or ""),
    }


def _local_team_fields(run: _Run, local_id: str) -> dict[str, Any]:
    from local_operator.hub_sync import provenance as prov

    team = run.teams.get_team(local_id)
    return prov.team_fields(
        description=team.description,
        manager=team.manager,
        members=team.members,
        instructions=team.instructions,
        project=team.project,
    )


def _remote_fields(run: _Run, item: CatalogueItem) -> dict[str, Any]:
    """The hub's fields for ``item``, normalised exactly as the sync check reads them.

    They become the BASELINE of a kept-local link, so they must fingerprint the
    way ``hub_sync.check`` fingerprints the remote side; otherwise an
    unchanged hub would read as moved and the link would never settle.
    """
    from local_operator.hub_sync import provenance as prov

    if item.kind == "agent":
        from local_operator.agents import _fetch_hub_profile
        from local_operator.hub_sync.segment import norm

        instructions, description = _fetch_hub_profile(
            run.client, item.hub_id, with_credential=True
        )
        return {"instructions": norm(instructions), "description": norm(description)}
    document = _fetch_team(run, item.hub_id)
    return prov.team_fields(
        description=str(document.get("description") or ""),
        manager=str(document.get("manager") or ""),
        members=document.get("members"),
        instructions=str(document.get("instructions") or ""),
        project=str(document.get("project") or ""),
    )


def _same(kind: str, local: dict[str, Any], remote: dict[str, Any]) -> bool:
    from local_operator.hub_sync import provenance as prov

    fingerprint = prov.fingerprint_fields
    return fingerprint(kind, local) == fingerprint(kind, remote)  # type: ignore[arg-type]


def _link_row(
    run: _Run,
    item: CatalogueItem,
    local_id: str,
    recorded_by: str,
    baseline: dict[str, Any] | None = None,
) -> None:
    """Link an existing local agent/team to ``item`` with the given baseline basis.

    ``baseline`` None records the LOCAL fields (``adopt``: local == remote was
    just proved, or the ``adopt-unknown`` link an explicit merge starts from).
    ``baseline`` = the hub's fields is the keep-local link: the ancestor is
    the hub revision, so the check classifies the untouched local copy
    ``local-only`` and the drift is visible rather than silently stale.
    ``adopt-unknown`` (``prov.UNKNOWN_BASELINE``) is the ``teams link`` shape:
    the link vouches for no ancestor, so every sync is the conservative
    two-way merge and a hub deletion can never read as a local one.

    Only tags and the baseline record are written; the row's content is not.
    """

    from local_operator.hub_sync import provenance as prov

    if item.kind == "agent":
        agent = run.agents.get_agent(local_id)
        # ``agents sync`` enumerates rows by their ``hub:`` tag, so the tag is
        # what makes the link visible to the check; the baseline record written
        # next is what the check reads first (it outranks the tag's hash).
        run.agents._stamp_hub_provenance(agent, item.hub_id)
        fields = baseline or {
            "instructions": run.agents.get_agent_system_prompt(local_id),
            "description": run.agents.get_agent(local_id).description or "",
        }
        prov.record_agent_baseline(
            run.config_dir,
            local_id=local_id,
            hub_id=item.hub_id,
            instructions=str(fields.get("instructions") or ""),
            description=str(fields.get("description") or ""),
            tenant_id=run.tenant,
            recorded_by=recorded_by,
            origin=run.origin(),
        )
        return
    if baseline is not None:
        record = prov.make_record("team", local_id, item.hub_id, run.tenant, baseline, recorded_by)
        prov.attach_origin(run.config_dir, record, run.origin())
        return
    prov.record_team_pull(
        run.config_dir,
        run.teams.get_team(local_id),
        {"id": item.hub_id},
        tenant_id=run.tenant,
        recorded_by=recorded_by,
        origin=run.origin(),
    )


def _apply(run: _Run, item: CatalogueItem, local_name: str, **kwargs: Any) -> Any:
    from local_operator.hub_sync import service as svc

    report = svc.apply_items(
        run.ctx, kind=item.kind, names=[local_name], allow_llm=run.allow_llm, **kwargs
    )
    return report.reports[0] if report.reports else None


def _conflict_row(
    run: _Run, item: CatalogueItem, on_conflict: OnConflict
) -> tuple[PullOutcome, dict[str, Any] | None]:
    from local_operator.hub_sync import provenance as prov

    local_id = str(item.local_id or "")
    local_name = str(item.local_name or item.name)
    local = (
        _local_agent_fields(run, local_id)
        if item.kind == "agent"
        else _local_team_fields(run, local_id)
    )
    if on_conflict == "adopt-link":
        # Never a merge, never a model (module docstring): one backup, then a
        # link whose baseline is the hub's revision. The local row is not
        # written, so identical content reads up-to-date and differing content
        # reads local-only to the check.
        remote = _remote_fields(run, item)
        identical = _same(item.kind, local, remote)
        backup = prov.write_backup(
            run.config_dir, item.kind, local_id, local, "org-adopt-link"  # type: ignore[arg-type]
        )
        _link_row(
            run,
            item,
            local_id,
            "adopt" if identical else "adopt-keep-local",
            baseline=None if identical else remote,
        )
        result = "identical" if identical else "kept-local-linked"
        return (
            PullOutcome(item.kind, item.hub_id, item.name, "adopt-linked", result, backup),
            _adopted(item, local_name, result, backup),
        )
    _link_row(run, item, local_id, prov.UNKNOWN_BASELINE)
    if on_conflict == "adopt":
        # ``apply_one`` writes the pre-apply backup itself (``prov.write_backup``).
        merged = _apply(run, item, local_name, replace="remote")
        backup = getattr(merged, "backup", None)
        outcome = "adopted" if merged is not None and merged.applied else "needs-review"
        return (
            PullOutcome(item.kind, item.hub_id, item.name, outcome, _merge_detail(merged), backup),
            _adopted(item, local_name, "hub-wins", backup) if outcome == "adopted" else None,
        )
    # ``reconcile``: the explicit merge. ``apply_one`` writes its own pre-apply
    # backup (``prov.write_backup``) before it changes anything, so this path
    # adds none of its own: one restore point per adopt (UX review U8).
    merged = _apply(run, item, local_name, acknowledge_unknown_baseline=True)
    settled = merged is not None and merged.applied
    backup = getattr(merged, "backup", None)
    outcome = "reconciled" if settled else "needs-review"
    return (
        PullOutcome(item.kind, item.hub_id, item.name, outcome, _merge_detail(merged), backup),
        _adopted(item, local_name, "reconciled", backup) if settled else None,
    )


def _merge_detail(report: Any) -> str:
    if report is None:
        return "no merge report"
    parts = [str(getattr(report, "outcome", ""))]
    if getattr(report, "error_class", None):
        parts.append(str(report.error_class))
    if getattr(report, "message", ""):
        parts.append(str(report.message)[:200])
    return ": ".join(p for p in parts if p)


def _adopted(
    item: CatalogueItem, local_name: str, result: str, backup: str | None
) -> dict[str, Any]:
    return {
        "name": local_name,
        "kind": item.kind,
        "hub_id": item.hub_id,
        "result": result,
        "backup": backup,
    }


# -- conflicts: skills -------------------------------------------------------------------


def _skill_backup(result: dict[str, Any]) -> str | None:
    snapshot = result.get("snapshot") or {}
    return str(snapshot.get("path") or "") or None


def _adopt_document(
    run: _Run, item: CatalogueItem, strategy: str, subscribe: bool
) -> dict[str, Any]:
    """Preview then confirm, in one process: the receipt still binds the snapshot.

    The preview is what takes the ``hub/backups/skill-adopt-*`` snapshot and
    builds the reviewed proposal. The confirm reuses those exact bytes (it
    never re-runs the model) and revalidates the local tree hash, so chaining
    the two here is safe in the same way ``lop skills pull --yes --strategy``
    is.
    """

    preview = _install_skill(
        run, item, subscribe=subscribe, dry_run=True, adoption_strategy=strategy
    )
    if not preview.get("adoption"):
        # The folder vanished between the catalogue read and now: a plain install.
        return _install_skill(run, item, subscribe=subscribe, confirm=True)
    state = (preview.get("adoption") or {}).get("state")
    if state == "needs-choice":
        preview["_needs_choice"] = True
        return preview
    result = _install_skill(
        run,
        item,
        subscribe=subscribe,
        confirm=True,
        adoption_receipt=preview["adoption_receipt"],
        adoption_strategy=strategy,
    )
    result.setdefault("snapshot", preview.get("snapshot"))
    result["_state"] = state
    return result


def _skill_conflict(
    run: _Run, item: CatalogueItem, on_conflict: OnConflict, subscribe: bool
) -> tuple[PullOutcome, dict[str, Any] | None]:
    name = str(item.local_name or item.name)

    def done(outcome: str, result: str | None, backup: str | None, detail: str = "") -> Any:
        return (
            PullOutcome("skill", item.hub_id, item.name, outcome, detail, backup),
            _adopted(item, name, result, backup) if result else None,
        )

    if item.form == "bundle":
        identical = False
        if on_conflict == "adopt-link" and item.tree_hash:
            from local_operator.hub_sync.skill_bundle import tree_hash_of

            try:
                identical = tree_hash_of(run.config_dir / "skills" / name) == item.tree_hash
            except Exception:  # noqa: BLE001 - an unreadable local tree is "differs"
                identical = False
        if on_conflict == "adopt" or identical:
            # Same bytes (identical) or hub-wins: the take-hub branch snapshots
            # the folder first, so either way a restore point exists.
            result = _install_skill(
                run, item, subscribe=subscribe, confirm=True, adoption_strategy="take-hub"
            )
            return done(
                "adopt-linked" if identical else "adopted",
                "identical" if identical else "hub-wins",
                _skill_backup(result),
            )
        # There is no archive merge engine, and a bundle cannot be linked
        # without installing its tree, so a differing bundle is never touched
        # automatically.
        return done(
            "needs-review",
            None,
            None,
            f"bundle differs from the local folder; `lop hub org pull --name {item.name} "
            "--on-conflict adopt --yes` takes the hub copy (your folder is backed up)",
        )

    if on_conflict == "adopt-link":
        # ``keep-local`` links WITHOUT touching authored bytes (the adoption
        # service writes no document for it) and records the HUB revision as
        # the baseline, so `skills sync --check` reports the drift as
        # local-only. One preview, so one ``skill-adopt-*`` snapshot (U8).
        # Whether the sides were identical is the classification the adoption
        # itself just recorded on the link: none (equal) or local-only.
        kept = _adopt_document(run, item, "keep-local", subscribe)
        drift = ((kept.get("link") or {}).get("pull") or {}).get("classification")
        result_name = "kept-local-linked" if drift else "identical"
        return done("adopt-linked", result_name, _skill_backup(kept), result_name)
    strategy = {"adopt": "take-hub", "reconcile": "merge"}[on_conflict]
    result = _adopt_document(run, item, strategy, subscribe)
    if result.get("_needs_choice"):
        return done(
            "needs-review",
            None,
            _skill_backup(result),
            "the merge needs a choice; `lop skills pull` shows the proposal",
        )
    state = result.get("_state")
    backup = _skill_backup(result)
    if on_conflict == "adopt":
        return done("adopted", "hub-wins", backup)
    if state == "equal":
        return done("adopt-linked", "identical", backup)
    return done("reconciled", "reconciled", backup)


# -- policy -------------------------------------------------------------------------------


def _apply_policy(
    run: _Run, item: CatalogueItem, *, subscribe: bool, auto_push: bool, role: str
) -> str:
    """Write the explicit link policy; return a note for the item detail ("" if none)."""

    from local_operator.hub_sync import provenance as prov
    from local_operator.hub_sync import service as svc
    from local_operator.hub_sync.policies import update_policy

    record = next(
        (
            r
            for r in prov.iter_baselines(run.config_dir)
            if r.kind == item.kind and r.hub_id == item.hub_id and r.identity is not None
        ),
        None,
    )
    if record is None or record.identity is None:
        return "linked without a confirmed origin; policy left at its default"
    note = ""
    push: bool | None = None
    if auto_push:
        verdict = svc.push_eligibility(run.ctx, record)
        if role not in _PUSH_ROLES:
            note = f"auto-push not enabled: your role is {role!r} (owner/admin only)"
        elif not verdict.get("can_auto_push"):
            note = f"auto-push not enabled: {verdict.get('reason')}"
        else:
            push = True
    snapshot = svc.links_snapshot(run.ctx, kinds=(item.kind,))
    row = next(
        (link for link in snapshot["links"] if link.get("link_id") == record.identity.link_id),
        None,
    )
    revision = ((row or {}).get("policy") or {}).get("revision")
    current = (row or {}).get("policy") or {}
    if not revision:
        return note or "policy revision unavailable; policy left at its default"
    if current.get("subscribe") == subscribe and push is None:
        return note
    eligibility = (row or {}).get("eligibility") or {}
    update_policy(
        run.config_manager,
        record.identity,
        expected_revision=revision,
        subscribe=subscribe,
        auto_push=push,
        can_enable_auto_push=bool(eligibility.get("can_auto_push")),
        ineligible_reason=str(eligibility.get("reason") or "unsupported-precondition"),
    )
    return note


# -- the entry ----------------------------------------------------------------------------


def _wanted(item: CatalogueItem, names: Sequence[str] | None) -> bool:
    """Exact hub id or exact name. Callers holding user input resolve it first
    (:func:`hub_org.catalogue.resolve_reference`: id, name, unique prefix) and
    pass the resolved ids, so a prefix never reaches this filter."""
    if not names:
        return True
    folded = {str(n).strip().casefold() for n in names}
    return item.hub_id.casefold() in folded or item.name.casefold() in folded


def pull_all(
    config_manager: Any,
    tenant: str,
    *,
    kinds: Sequence[str] = ("agent", "skill", "team"),
    names: Sequence[str] | None = None,
    subscribe: bool = True,
    auto_push: bool = False,
    on_conflict: OnConflict = "adopt-link",
    allow_llm: bool = True,
    progress: Callable[[PullOutcome], None] | None = None,
    client: Any = None,
    agent_registry: Any = None,
    team_registry: Any = None,
) -> OrgPullReport:
    """Pull every org item of ``kinds`` (or the ``names``: hub ids or names).

    ``client`` defaults to :func:`hub_org.client.build_org_client`; tests and
    the tool may pass one. Returns a report and never raises for an item's
    failure; a missing client or a failed list is an ``errors`` entry.
    """

    from local_operator.hub_org.catalogue import _remote_rows
    from local_operator.hub_org.eligibility import eligible_orgs

    if on_conflict not in ON_CONFLICT_CHOICES:
        raise ValueError(f"on_conflict must be one of {ON_CONFLICT_CHOICES}")
    if client is None:
        from local_operator.hub_org.client import build_org_client

        client = build_org_client(config_manager).client
        if client is None:
            return OrgPullReport(errors={"client": "not signed in to Radient"})
    config_dir = Path(config_manager.config_dir)
    role = next(
        (org.role for org in eligible_orgs(config_dir) if org.tenant_id == tenant), "member"
    )
    run = _Run(
        config_manager,
        tenant,
        client=client,
        agent_registry=agent_registry,
        team_registry=team_registry,
        allow_llm=allow_llm,
    )
    _settle_interrupted_adoptions(run)
    items: list[PullOutcome] = []
    adopted: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for kind in KINDS:
        if kind not in kinds:
            continue
        try:
            rows = _remote_rows(client, tenant, kind)
        except Exception as error:  # noqa: BLE001 - one kind's failure never stops the rest
            errors[kind] = str(error)[:300]
            continue
        for row in sorted(rows, key=lambda r: str(r.get("name") or "").casefold()):
            # Re-index per item: an agent pulled a moment ago must read as
            # linked/present to the team that names it, and a re-run must see
            # the links this run wrote (no ``-2`` rows).
            index = local_index(config_dir, agent_registry=run.agents, team_registry=run.teams)
            item = item_from_row(index, kind, row)  # type: ignore[arg-type]
            if item is None or not _wanted(item, names):
                continue
            outcome, adoption = _one(run, item, on_conflict=on_conflict, subscribe=subscribe)
            if outcome.outcome not in (
                "already-linked",
                "conflict-skipped",
                "linked-elsewhere",
                "failed",
                "needs-review",
            ):
                try:
                    note = _apply_policy(
                        run, item, subscribe=subscribe, auto_push=auto_push, role=role
                    )
                except Exception as error:  # noqa: BLE001 - the pull itself succeeded
                    note = f"policy not written: {error}"
                if note:
                    detail = f"{outcome.detail}; {note}" if outcome.detail else note
                    outcome = PullOutcome(**{**outcome.to_json(), "detail": detail})
            items.append(outcome)
            if adoption:
                adopted.append(adoption)
            if progress is not None:
                progress(outcome)
    return OrgPullReport(items=tuple(items), adopted=tuple(adopted), errors=errors)


def _settle_interrupted_adoptions(run: _Run) -> None:
    """Restore a skill an earlier, interrupted bundle adoption hid, before indexing.

    Without this the catalogue would read the hidden folder's alias as absent
    for one run. Zero wait on the lease: a busy lease means a live install,
    which settles its own move, and ``install_skill`` recovers again anyway.
    """
    from local_operator.hub_sync import service as svc

    try:
        with svc.exclusive(run.ctx, wait_s=0.0):
            svc.recover_bundle_adoption_asides(run.config_dir)
    except Exception:  # noqa: BLE001 - HubBusy or I/O: the per-item install retries it
        logger.debug("adoption recovery skipped", exc_info=True)


def _one(
    run: _Run, item: CatalogueItem, *, on_conflict: OnConflict, subscribe: bool
) -> tuple[PullOutcome, dict[str, Any] | None]:
    try:
        if item.local == "linked":
            return PullOutcome(item.kind, item.hub_id, item.name, "already-linked"), None
        if item.local == "linked-elsewhere":
            # Never adopted under any ``on_conflict`` (module docstring).
            return (
                PullOutcome(
                    item.kind,
                    item.hub_id,
                    item.name,
                    "linked-elsewhere",
                    f"the local {item.kind} {item.local_name!r} already follows another hub "
                    "item; left as it is",
                ),
                None,
            )
        if item.local == "absent":
            if item.kind == "agent":
                return _pull_agent(run, item), None
            if item.kind == "team":
                return _pull_team(run, item), None
            return _pull_skill(run, item, subscribe), None
        if on_conflict == "skip":
            return (
                PullOutcome(
                    item.kind,
                    item.hub_id,
                    item.name,
                    "conflict-skipped",
                    f"a local {item.kind} named {item.local_name!r} is not linked; "
                    f"`lop hub org pull --name {item.name} --yes` links it (backed up first)",
                ),
                None,
            )
        if item.kind == "skill":
            return _skill_conflict(run, item, on_conflict, subscribe)
        return _conflict_row(run, item, on_conflict)
    except Exception as error:  # noqa: BLE001 - per-item failures never stop the batch
        logger.warning("org pull of %s %s failed", item.kind, item.hub_id, exc_info=True)
        from local_operator.hub_sync.identity import refusal_text_en

        cause = getattr(error, "code", None)
        # PullOutcome.detail is read by the model through the org_pull tool
        # result, so a catalogued LinkError sentence is pinned to English.
        text = refusal_text_en(error)[:300]
        return (
            PullOutcome(
                item.kind, item.hub_id, item.name, "failed", f"{cause}: {text}" if cause else text
            ),
            None,
        )
