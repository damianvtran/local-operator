"""What an organization shares on its private Agent Hub, and how each item sits locally.

WHY. Before this module a member had no single place to see their org's
agents, teams and skills. Three list routes existed (``list_org_agents``,
``list_org_teams`` and, since this change, ``list_org_skills``), but nothing joined
them with the local state, so an agent had to make ~20 per-item calls to learn
what was already pulled. :func:`list_catalogue` is the join. Every org surface
reads it: ``lop hub org``, the ``agent`` tool's ``org_list``/``org_pull``, and
slice B's auto-enrol diff.

THE ``local`` COLUMN is a classification, never a write:

- ``linked``: a hub baseline already names this hub id. Updates are the
  existing runner's and ``sync``'s job, so there is no second update path.
- ``conflict``: nothing links it, but a local row/folder holds the same NAME.
  A naive pull would mint ``coder-2`` (agents/teams) or hit the skill
  ``alias-occupied`` refusal. :mod:`hub_org.pull` routes these through
  ``on_conflict``; ``lop hub org`` reports them as "unlinked local copies".
- ``absent``: a plain pull creates it.

Names are compared casefolded. That is the registries' own collision rule, so
a local ``Coder`` blocks an incoming ``coder`` exactly as ``create_*`` would.

Network: one GET per kind (agents paged). Never raises for one kind's failure
when the caller asks for several: the failed kind is reported in
``CatalogueResult.errors`` so the other kinds still render.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

Kind = Literal["agent", "team", "skill"]
LocalState = Literal["linked", "conflict", "absent"]

#: The pull order: rosters name agents, so agents land before teams.
#: ``pull_all`` reuses it.
KINDS: tuple[Kind, ...] = ("agent", "skill", "team")

#: Agent list page size. The route caps ``per_page``, and 50 keeps one org's
#: catalogue to a page or two without asking for more than the hub serves.
_AGENT_PAGE = 50
#: Bound on the agent page walk, so a misbehaving ``total_pages`` cannot loop forever.
_AGENT_MAX_PAGES = 40

#: Hub ids are UUIDs or (older rows) 24-hex ObjectIds; the same shapes
#: ``cli._looks_like_hub_id`` accepts.
_HUB_ID_RE = re.compile(
    r"(?:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"|[0-9a-fA-F]{24})\Z"
)


@dataclass(frozen=True)
class CatalogueItem:
    """One org item and its local standing (design §2 contract, fields 1-7)."""

    kind: Kind
    hub_id: str
    name: str
    description: str
    revision: str
    local: LocalState
    local_name: str | None
    #: Skills only: ``bundle`` or ``document``. Bundles have no merge engine, so
    #: the pull and the CLI must know the form before choosing a conflict path.
    form: str | None = None
    #: Skills only: the published tree hash of a bundle row (list rows carry it),
    #: which lets ``adopt-link`` prove "identical" without downloading.
    tree_hash: str | None = None
    #: The local row id for ``linked``/``conflict`` (agent/team uuid, skill alias).
    local_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CatalogueResult:
    items: tuple[CatalogueItem, ...] = ()
    #: kind -> one-line cause, for a kind whose list GET failed.
    errors: dict[str, str] = field(default_factory=dict)

    def unlinked_local_copies(self) -> list[CatalogueItem]:
        """Local items that share a name with an org item but carry no link."""
        return [item for item in self.items if item.local == "conflict"]

    def to_json(self) -> dict[str, Any]:
        return {"items": [i.to_json() for i in self.items], "errors": dict(self.errors)}


class ReferenceError_(LookupError):
    """A name/prefix that resolves to zero or several catalogue items."""

    def __init__(self, ref: str, candidates: Sequence[CatalogueItem]) -> None:
        self.ref = ref
        self.candidates = tuple(candidates)
        super().__init__(ref)


def looks_like_hub_id(value: str) -> bool:
    return bool(_HUB_ID_RE.match(str(value or "").strip()))


def resolve_reference(
    items: Iterable[CatalogueItem], ref: str, *, kind: Kind | None = None
) -> CatalogueItem:
    """The ONE catalogue item ``ref`` names: an id, an exact name, or a unique name prefix.

    WHY. The org pull routes take only a hub id, so ``lop skills pull --org T
    minerva-platform-deployments`` answered "no such skill" (evidence run). An
    operator, and especially an agent, knows the published NAME. Precedence:
    exact hub id, then exact (casefolded) name, then a unique name prefix. An
    ambiguous prefix REFUSES with the candidates rather than guessing, because
    pulling the wrong one of two similarly named skills is a silent mistake.

    Raises:
        ReferenceError_: no match (``candidates`` empty) or an ambiguous prefix.
    """

    needle = str(ref or "").strip()
    pool = [i for i in items if kind is None or i.kind == kind]
    for item in pool:
        if item.hub_id == needle:
            return item
    folded = needle.casefold()
    exact = [i for i in pool if i.name.casefold() == folded]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ReferenceError_(needle, exact)
    prefixed = [i for i in pool if folded and i.name.casefold().startswith(folded)]
    if len(prefixed) == 1:
        return prefixed[0]
    raise ReferenceError_(needle, prefixed)


# -- listing ---------------------------------------------------------------------------


def _list_agents(client: Any, tenant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page = 1
    while page <= _AGENT_MAX_PAGES:
        envelope = client.list_org_agents(tenant, page=page, per_page=_AGENT_PAGE)
        records = envelope.get("records") if isinstance(envelope, dict) else None
        rows.extend(r for r in (records or []) if isinstance(r, dict))
        try:
            total = int((envelope or {}).get("total_pages") or 1)
        except (TypeError, ValueError):
            total = 1
        if page >= total or not records:
            break
        page += 1
    return rows


def _remote_rows(client: Any, tenant: str, kind: Kind) -> list[dict[str, Any]]:
    if kind == "agent":
        return _list_agents(client, tenant)
    if kind == "team":
        return [r for r in client.list_org_teams(tenant) if isinstance(r, dict)]
    return [r for r in client.list_org_skills(tenant) if isinstance(r, dict)]


# -- the local index -------------------------------------------------------------------


@dataclass
class _LocalIndex:
    """Linked hub ids and local names per kind, read once per catalogue."""

    linked: dict[tuple[str, str], str]  # (kind, hub_id) -> local name/alias
    linked_ids: dict[tuple[str, str], str]  # (kind, hub_id) -> local id
    names: dict[tuple[str, str], tuple[str, str]]  # (kind, casefold name) -> (name, id)


def local_index(config_dir: Path, *, agent_registry: Any = None, team_registry: Any = None) -> _LocalIndex:
    """Index existing links (baselines) and local names, with no network.

    Exposed for :mod:`hub_org.pull`, which re-reads it per item so a pull earlier
    in the same batch (a team's roster agent, say) is seen by the next one.
    """

    from local_operator.hub_sync import provenance as prov

    config_dir = Path(config_dir)
    linked: dict[tuple[str, str], str] = {}
    linked_ids: dict[tuple[str, str], str] = {}
    for record in prov.iter_baselines(config_dir):
        if not record.hub_id:
            continue
        key = (record.kind, record.hub_id)
        linked[key] = record.local_alias or record.local_id
        linked_ids[key] = record.local_id
    names: dict[tuple[str, str], tuple[str, str]] = {}
    if agent_registry is None:
        from local_operator.agents import AgentRegistry

        agent_registry = AgentRegistry(config_dir)
    from local_operator.agents import hub_origin

    for agent in agent_registry.list_agents():
        name = str(agent.name or "")
        if name:
            names.setdefault(("agent", name.casefold()), (name, agent.id))
        # A row pulled before baselines existed still carries its ``hub:`` tag,
        # and ``agents sync`` already tracks it, so it is linked, not a conflict.
        origin = hub_origin(agent)
        if origin and ("agent", origin) not in linked_ids:
            linked_ids[("agent", origin)] = agent.id
    if team_registry is None:
        from local_operator.teams import TeamRegistry

        team_registry = TeamRegistry(config_dir)
    for team in team_registry.list_teams():
        names.setdefault(("team", team.name.casefold()), (team.name, team.id))
    skills_root = config_dir / "skills"
    try:
        entries = list(skills_root.iterdir()) if skills_root.is_dir() else []
    except OSError:
        entries = []
    for entry in entries:
        # Dot entries are staging/aside folders, never skills (discovery skips them too).
        if entry.name.startswith(".") or not entry.is_dir():
            continue
        names.setdefault(("skill", entry.name.casefold()), (entry.name, entry.name))
    # Linked local ids resolve to their display name for the agent/team arms.
    for (kind, hub_id), local_id in list(linked_ids.items()):
        if kind == "skill":
            continue
        for (k, _folded), (name, row_id) in names.items():
            if k == kind and row_id == local_id:
                linked[(kind, hub_id)] = name
                break
    return _LocalIndex(linked=linked, linked_ids=linked_ids, names=names)


def classify(index: _LocalIndex, kind: Kind, hub_id: str, name: str) -> tuple[LocalState, str | None, str | None]:
    """``(local, local_name, local_id)`` for one remote row, against ``index``."""

    key = (kind, hub_id)
    if key in index.linked:
        return "linked", index.linked[key], index.linked_ids.get(key)
    hit = index.names.get((kind, name.casefold())) if name else None
    if hit is not None:
        return "conflict", hit[0], hit[1]
    return "absent", None, None


def item_from_row(index: _LocalIndex, kind: Kind, row: dict[str, Any]) -> CatalogueItem | None:
    hub_id = str(row.get("id") or "").strip()
    if not hub_id:
        return None
    name = str(row.get("name") or "")
    local, local_name, local_id = classify(index, kind, hub_id, name)
    form = tree_hash = None
    if kind == "skill":
        bundle = row.get("bundle")
        form = "bundle" if isinstance(bundle, dict) else "document"
        if isinstance(bundle, dict):
            tree_hash = str(bundle.get("tree_hash") or "") or None
    return CatalogueItem(
        kind=kind,
        hub_id=hub_id,
        name=name,
        description=str(row.get("description") or ""),
        revision=str(row.get("revision") or ""),
        local=local,
        local_name=local_name,
        form=form,
        tree_hash=tree_hash,
        local_id=local_id,
    )


def list_catalogue(
    client: Any,
    tenant: str,
    kinds: Sequence[str] = KINDS,
    *,
    config_dir: Path | None = None,
    agent_registry: Any = None,
    team_registry: Any = None,
) -> list[CatalogueItem]:
    """The org's items of ``kinds`` with their local state (design §2 contract).

    ``config_dir`` defaults to the active config root. A kind whose list GET
    fails raises. Use :func:`catalogue_result` for the per-kind error report
    the multi-kind surfaces render.
    """

    result = catalogue_result(
        client,
        tenant,
        kinds,
        config_dir=config_dir,
        agent_registry=agent_registry,
        team_registry=team_registry,
        raise_errors=True,
    )
    return list(result.items)


def catalogue_result(
    client: Any,
    tenant: str,
    kinds: Sequence[str] = KINDS,
    *,
    config_dir: Path | None = None,
    agent_registry: Any = None,
    team_registry: Any = None,
    raise_errors: bool = False,
) -> CatalogueResult:
    if config_dir is None:
        from local_operator.paths import config_dir as _active

        config_dir = _active()
    index = local_index(Path(config_dir), agent_registry=agent_registry, team_registry=team_registry)
    items: list[CatalogueItem] = []
    errors: dict[str, str] = {}
    for kind in KINDS:
        if kind not in kinds:
            continue
        try:
            rows = _remote_rows(client, tenant, kind)
        except Exception as error:  # noqa: BLE001 - one kind's failure never hides the rest
            if raise_errors:
                raise
            errors[kind] = str(error)[:300]
            continue
        for row in sorted(rows, key=lambda r: str(r.get("name") or "").casefold()):
            item = item_from_row(index, kind, row)
            if item is not None:
                items.append(item)
    return CatalogueResult(items=tuple(items), errors=errors)
