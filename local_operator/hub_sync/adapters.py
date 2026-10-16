"""Thin per-kind adapter protocol over the ONE hub_sync engine.

The engine (check/merge/provenance/runner) is kind-agnostic today and that is
load-bearing: a second synchronizer per kind is the failure this module exists
to prevent. Each kind supplies only what genuinely differs -- its immutable
identity fields, a local snapshot, the canonical transferable projection, a
validator, and the local-write and remote-write boundaries -- and everything
else stays in the shared engine.

``commit_local`` and ``publish`` are BOUNDARY declarations, not implementations
of remote I/O: they name the exact precondition (expected snapshot / expected
remote revision) the engine must re-check under the writer lock before it calls
them. Keeping them here means the runner and the routes cannot invent a second
copy of "what counts as the same item".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from local_operator.hub_sync.identity import Kind, LinkError, LinkIdentity, Visibility
from local_operator.hub_sync.provenance import fingerprint_fields


@dataclass(frozen=True)
class Snapshot:
    """A canonical local reading of one artifact, plus its immutable local id."""

    kind: str
    local_id: str
    name: str
    fields: Mapping[str, Any]
    fingerprint: str


@runtime_checkable
class KindAdapter(Protocol):
    """What the engine may ask of ONE artifact kind. No network, no locks here."""

    kind: Kind

    def local_id(self, row: Any) -> str:
        """The IMMUTABLE local id (stable across rename), not a display name."""
        ...

    def display_name(self, row: Any) -> str: ...

    def hub_id(self, config_dir: Path, row: Any) -> str | None:
        """The recorded remote id for this row, or None when never linked."""

    def tenant_id(self, config_dir: Path, row: Any) -> str | None: ...

    def snapshot(self, config_dir: Path, row: Any) -> Snapshot:
        """Read the canonical authored fields now; the engine diffs these."""
        ...

    def canonical_fields(self, raw: Mapping[str, Any]) -> Mapping[str, Any]:
        """Normalize a remote document to the same field shape ``snapshot`` uses."""
        ...

    def validate(self, fields: Mapping[str, Any]) -> None:
        """Refuse a merged/remote document that cannot be this kind. Named error."""

    def identity(
        self,
        local_id: str,
        name: str,
        hub_source: str,
        visibility: Visibility,
        tenant_id: str | None,
        hub_id: str,
    ) -> LinkIdentity:
        """Build this adapter's immutable link identity (see ``_SharedIdentity``)."""
        ...


class _SharedIdentity:
    """The one ``identity()`` implementation the three concrete adapters share.

    A protocol's default body is not INHERITED by its implementers -- pyright
    requires the member on each class for conformance -- and the construction is
    kind-independent (``self.kind`` is the only kind-specific input), so it
    lives once here rather than three times beside the classes.
    """

    kind: Kind

    def identity(
        self,
        local_id: str,
        name: str,
        hub_source: str,
        visibility: Visibility,
        tenant_id: str | None,
        hub_id: str,
    ) -> LinkIdentity:
        return LinkIdentity(self.kind, local_id, hub_source, visibility, tenant_id, hub_id)


class AgentAdapter(_SharedIdentity):
    """Agent/profile rows: identity is the registry row id; fields are two texts."""

    kind: Kind = "agent"

    def local_id(self, row: Any) -> str:
        return str(row.id)

    def display_name(self, row: Any) -> str:
        return str(row.name or "")

    def hub_id(self, config_dir: Path, row: Any) -> str | None:
        from local_operator.agents import hub_origin

        return hub_origin(row)

    def tenant_id(self, config_dir: Path, row: Any) -> str | None:
        from local_operator.hub_sync import provenance as prov

        record = prov.read_baseline(config_dir, "agent", str(row.id))
        return record.tenant_id if record else None

    def snapshot(self, config_dir: Path, row: Any) -> Snapshot:
        from local_operator.agents import AgentRegistry

        registry = AgentRegistry(config_dir)
        fields = {
            "instructions": registry.get_agent_system_prompt(str(row.id)) or "",
            "description": str(row.description or ""),
        }
        return Snapshot(
            self.kind,
            str(row.id),
            str(row.name or ""),
            fields,
            fingerprint_fields("agent", fields),
        )

    def canonical_fields(self, raw: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "instructions": str(raw.get("instructions") or ""),
            "description": str(raw.get("description") or ""),
        }

    def validate(self, fields: Mapping[str, Any]) -> None:
        if not isinstance(fields.get("instructions"), str):
            raise LinkError("merge-refused", "agent instructions must be text")


class TeamAdapter(_SharedIdentity):
    """Team rows: identity is the registry row id; the remote id lives in the baseline."""

    kind: Kind = "team"

    def local_id(self, row: Any) -> str:
        return str(row.id)

    def display_name(self, row: Any) -> str:
        return str(row.name or "")

    def hub_id(self, config_dir: Path, row: Any) -> str | None:
        from local_operator.hub_sync import provenance as prov

        record = prov.read_baseline(config_dir, "team", str(row.id))
        return record.hub_id if record else None

    def tenant_id(self, config_dir: Path, row: Any) -> str | None:
        from local_operator.hub_sync import provenance as prov

        record = prov.read_baseline(config_dir, "team", str(row.id))
        return record.tenant_id if record else None

    def snapshot(self, config_dir: Path, row: Any) -> Snapshot:
        from local_operator.hub_sync.check import team_fields_of

        fields = team_fields_of(row)
        return Snapshot(
            self.kind,
            str(row.id),
            str(row.name or ""),
            fields,
            fingerprint_fields("team", fields),
        )

    def canonical_fields(self, raw: Mapping[str, Any]) -> Mapping[str, Any]:
        from local_operator.hub_sync.provenance import normalize_roster

        return {
            "description": str(raw.get("description") or ""),
            "manager": str(raw.get("manager") or "").strip(),
            "members": normalize_roster(raw.get("members")),
            "instructions": str(raw.get("instructions") or ""),
            "project": str(raw.get("project") or ""),
        }

    def validate(self, fields: Mapping[str, Any]) -> None:
        if not isinstance(fields.get("manager"), str):
            raise LinkError("merge-refused", "team manager must be a name")


class SkillAdapter(_SharedIdentity):
    """Managed skills: identity is the persisted opaque id; fields are the strict manifest."""

    kind: Kind = "skill"

    def local_id(self, row: Any) -> str:
        raise LinkError("origin-conflict", "a skill's local id is its persisted manifest id")

    def display_name(self, row: Any) -> str:
        return str(getattr(row, "name", ""))

    def hub_id(self, config_dir: Path, row: Any) -> str | None:
        from local_operator.hub_sync import provenance as prov

        record = prov.read_baseline(config_dir, "skill", str(getattr(row, "local_id", "")))
        return record.hub_id if record else None

    def tenant_id(self, config_dir: Path, row: Any) -> str | None:
        return None

    def snapshot(self, config_dir: Path, row: Any) -> Snapshot:
        """Read a managed skill through its storage (recovering a crashed transaction first).

        ``row`` is the managed skill's opaque local id; the alias and the content
        live in ``ManagedSkillStorage``/``SkillDocument``. Going through the
        storage (never a raw directory read) keeps the journal/recovery discipline
        the pull path relies on, so a snapshot is never taken mid-swap.
        """

        from local_operator.hub_sync.skill_storage import ManagedSkillStorage

        local_id = str(row)
        storage = ManagedSkillStorage(config_dir)
        snapshot = storage.snapshot(local_id)
        fields = snapshot.document.to_json()
        return Snapshot(
            self.kind,
            local_id,
            snapshot.alias,
            fields,
            fingerprint_fields("skill", fields),
        )

    def canonical_fields(self, raw: Mapping[str, Any]) -> Mapping[str, Any]:
        from local_operator.hub_sync.skill_document import SkillDocument

        # TOLERANT ENVELOPE, STRICT MANIFEST: a hub DETAIL response carries
        # metadata beside the manifest (revision/sync_contract/id/...), so the
        # manifest is built from the manifest subset only (round-2 QN1).
        return SkillDocument.from_remote(dict(raw)).to_json()

    def validate(self, fields: Mapping[str, Any]) -> None:
        from local_operator.hub_sync.skill_document import SkillDocument

        SkillDocument.from_json(dict(fields))


def publish_document(
    kind: Kind, config_dir: Path, local_id: str, fields: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Build the WIRE publication document for one kind from its merged fields.

    ONE place maps (local row + canonical merged fields) onto the server's frozen
    document family, so the runner, a route and the CLI cannot publish three
    slightly different documents for the same item:

    * agent -> the version-1 instruction-set document, built by the SAME
      ``instruction_set_fields`` + ``build_instruction_set_document`` the publish
      routes use, with the merged ``instructions``/``description`` supplied as
      validated overrides (name/kind/version/tags come from the row);
    * team -> the team document from ``hub_team_document`` with the merged five
      content keys applied as overrides, re-preflighted against the hub's rules;
    * skill -> the strict manifest itself (the canonical fields ARE the document).

    Raises the underlying refusal (``KeyError`` for a missing local row, the
    client's ``InstructionSetError``, ``TeamDocumentError``, or ``LinkError``)
    for the caller to map onto a named push refusal.
    """

    if kind == "agent":
        from local_operator.agents import AgentRegistry, instruction_set_fields
        from local_operator.clients.radient import (
            build_instruction_set_document,
            validate_document_overrides,
        )

        registry = AgentRegistry(config_dir)
        row = registry.get_agent(local_id)
        overrides = validate_document_overrides(
            {key: fields[key] for key in ("instructions", "description") if key in fields}
        )
        return build_instruction_set_document(**instruction_set_fields(registry, row, overrides))
    if kind == "team":
        from local_operator.agents import AgentRegistry, agents_store_present
        from local_operator.teams import TeamRegistry, hub_team_document

        registry = TeamRegistry(config_dir)
        row = registry.get_team(local_id)
        agent_registry = AgentRegistry(config_dir) if agents_store_present(config_dir) else None
        overrides = {
            key: fields[key]
            for key in ("description", "manager", "members", "instructions", "project")
            if key in fields
        }
        return hub_team_document(row, agent_registry=agent_registry, overrides=overrides)
    if kind == "skill":
        from local_operator.hub_sync.skill_document import SkillDocument

        # A skill's canonical merged fields ARE its wire manifest; validating it
        # here (again) is the last refusal point before a paid preview.
        return SkillDocument.from_json(dict(fields)).to_json()
    raise LinkError("unsupported-kind", f"{kind} publication is not implemented")


_ADAPTERS: dict[str, KindAdapter] = {
    "agent": AgentAdapter(),
    "team": TeamAdapter(),
    "skill": SkillAdapter(),
}


def adapter_for(kind: str) -> KindAdapter:
    try:
        return _ADAPTERS[kind]
    except KeyError:
        raise LinkError("origin-conflict", f"no adapter for kind {kind!r}") from None
