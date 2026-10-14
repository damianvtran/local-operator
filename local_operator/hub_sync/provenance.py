"""Baseline records: the TEXT a three-way merge diffs against (design A2, B1).

WHY A SIDE STORE. An agent row only ever recorded a fingerprint of what it
pulled, and a team row recorded nothing. A hash answers "is my copy still the
pulled one?" but not "what did the pulled one SAY?", so it cannot tell a section
the user deleted from a section the hub added. The text lives in
``<config_dir>/hub/baselines/<kind>-<local_id>.json`` instead of in the item,
because a team row is a directory the registry replaces wholesale on every edit
(anything extra inside it dies with the next save), and because a file outside
the item can never ride an export/publish bundle: a published archive cannot
plant a baseline (same trust rule as ``strip_provenance_tags``).

WHO MAY WRITE. Only pull/import, publish, and a successful merge apply. After
each, B is "the last remote text this machine integrated" (see the ``merge``
module docstring for how that differs from the design's literal wording when
local edits survive a merge).

``config_dir`` is always passed in from a ``ConfigManager``/registry, never
derived from ``Path.home()`` (AGENTS.md "Isolating a run": a store that defaults
its write path to the global root is the analytics-backfill bug).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from local_operator.config_writer import serialized_config_path_write
from local_operator.hub_sync.identity import Kind, LinkError, LinkIdentity
from local_operator.hub_sync.segment import norm

logger = logging.getLogger(__name__)

SCHEMA = 2
LEGACY_SCHEMA = 1
#: Last N pre-apply backups kept per item (A9.3).
BACKUPS_KEPT = 5

RecordedBy = Literal["pull", "publish", "merge-pull", "merge-push", "adopt", "adopt-unknown"]

#: ``recorded_by`` of a record that LINKS an item to a hub id without vouching for
#: what the two sides had in common (``lop teams link`` over a differing copy). The
#: checks read such a record as "baseline unknown".
UNKNOWN_BASELINE = "adopt-unknown"

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

AGENT_FIELDS = ("instructions", "description")
TEAM_FIELDS = ("description", "manager", "members", "instructions", "project")


class BaselineError(Exception):
    """A baseline path/record was refused (symlink, bad id, corrupt shape)."""


def hub_root(config_dir: Path) -> Path:
    return Path(config_dir) / "hub"


def baselines_dir(config_dir: Path) -> Path:
    return hub_root(config_dir) / "baselines"


def alias_owner(config_dir: Path, kind: Kind, alias: str) -> str | None:
    """The immutable local id a managed alias belongs to, durable across a crash gap.

    A crash between the two directory renames leaves the alias temporarily absent
    from the live root, but its baseline still names it. Reading only the live
    directory therefore let a DIFFERENT origin install the same alias and then
    return the other origin's text for the original id (reviewer R5). This reads
    the durable baselines so an alias that belongs to one origin is never
    claimable by another, present or not.
    """

    directory = baselines_dir(config_dir)
    try:
        entries = list(directory.glob(f"{kind}-*.json"))
    except OSError:
        return None
    for path in entries:
        local_id = path.name[len(kind) + 1 : -len(".json")]
        record = read_baseline(config_dir, kind, local_id)
        if record is not None and record.local_alias == alias:
            return record.local_id
    return None


def backups_dir(config_dir: Path) -> Path:
    return hub_root(config_dir) / "backups"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -- fingerprints (A2.4) ---------------------------------------------------------


def _sha(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_agent(instructions: str, description: str) -> str:
    """The canonical (``norm``) agent fingerprint used by baseline records."""

    return _sha(
        json.dumps(
            [norm(instructions), norm(description)], ensure_ascii=False, separators=(",", ":")
        )
    )


def legacy_agent_fingerprint(instructions: str, description: str) -> str:
    """Byte-identical to ``agents.hub_fingerprint`` (bare ``.strip()``).

    Kept here rather than imported so this module stays off the agents import
    graph; ``test_provenance`` pins the two equal. Existing ``hub_sha256:`` tags
    were written with this form, so comparisons accept either (one-way compat).
    """

    return _sha(
        json.dumps(
            [str(instructions or "").strip(), str(description or "").strip()],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def normalize_roster(members: Any) -> list[dict[str, Any]]:
    """Roster slots as ``{role, kind, count}``, unknown kinds coerced to ``agent``.

    The same coercion ``import_hub_team`` applies, so a hub roster and a local one
    compare on the vocabulary the local registry actually holds.
    """

    out: list[dict[str, Any]] = []
    for slot in members if isinstance(members, (list, tuple)) else []:
        if hasattr(slot, "model_dump"):
            slot = slot.model_dump()
        if not isinstance(slot, Mapping):
            continue
        role = str(slot.get("role") or "").strip()
        if not role:
            continue
        kind = "team" if str(slot.get("kind") or "").strip() == "team" else "agent"
        try:
            count = int(slot.get("count") or 1)
        except (TypeError, ValueError):
            count = 1
        out.append({"role": role, "kind": kind, "count": count})
    return out


def team_fields(
    *, description: str, manager: str, members: Any, instructions: str, project: str
) -> dict[str, Any]:
    return {
        "description": norm(description),
        "manager": str(manager or "").strip(),
        "members": normalize_roster(members),
        "instructions": norm(instructions),
        "project": norm(project),
    }


def fingerprint_team(fields: Mapping[str, Any]) -> str:
    """A2.4 team fingerprint. ``name`` and ``version`` are deliberately absent."""

    roster = sorted(
        normalize_roster(fields.get("members")),
        key=lambda s: (s["kind"], s["role"].casefold()),
    )
    payload = {
        "description": norm(str(fields.get("description") or "")),
        "manager": str(fields.get("manager") or "").strip(),
        "members": roster,
        "instructions": norm(str(fields.get("instructions") or "")),
        "project": norm(str(fields.get("project") or "")),
    }
    return _sha(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def fingerprint_snapshot(kind: Kind, fields: Mapping[str, Any]) -> str:
    """Hash every canonical transferable field for preview/CAS/outbound input.

    The original agent ancestry fingerprint intentionally covers only the two
    text fields and remains compatible with shipped tags. That legacy hash is
    NOT sufficient to bind a full publication/definition preview: changing a
    tool/flag/manifest attachment must invalidate its captured snapshot too.
    Adapters select transferable fields before this function, never runtime
    transcripts, credentials, or local provenance.
    """
    if kind == "skill":
        from local_operator.hub_sync.skill_document import SkillDocument

        return SkillDocument.from_json(dict(fields)).fingerprint
    return _sha(
        json.dumps(
            dict(fields), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    )


def fingerprint_fields(kind: Kind, fields: Mapping[str, Any]) -> str:
    if kind == "skill":
        from local_operator.hub_sync.skill_document import SkillDocument

        return SkillDocument.from_json(dict(fields)).fingerprint
    if kind == "team":
        return fingerprint_team(fields)
    return fingerprint_agent(
        str(fields.get("instructions") or ""), str(fields.get("description") or "")
    )


# -- records ----------------------------------------------------------------------


@dataclass(frozen=True)
class OutboundReceipt:
    """The same origin's paired projections, never a substitute inbound baseline.

    A captures the local input; P is the ORIGINAL server-accepted document.
    Generalization can change P, so input equality is measured against A rather
    than repeatedly comparing unmodified local bytes with generalized remote.
    """

    local_fields: Mapping[str, Any]
    accepted_fields: Mapping[str, Any]
    input_fingerprint: str
    remote_revision: str | None
    operation_id: str | None
    recorded_at: str = field(default_factory=now_iso)

    def __post_init__(self) -> None:
        # A captured projection must not share nested metadata/file arrays with
        # a registry snapshot that another writer can edit during network work.
        object.__setattr__(self, "local_fields", deepcopy(dict(self.local_fields)))
        object.__setattr__(self, "accepted_fields", deepcopy(dict(self.accepted_fields)))

    def to_json(self) -> dict[str, Any]:
        return {
            "local_fields": deepcopy(dict(self.local_fields)),
            "accepted_fields": deepcopy(dict(self.accepted_fields)),
            "input_fingerprint": self.input_fingerprint,
            "remote_revision": self.remote_revision,
            "operation_id": self.operation_id,
            "recorded_at": self.recorded_at,
        }


@dataclass(frozen=True)
class BaselineRecord:
    kind: Kind
    local_id: str
    hub_id: str
    tenant_id: str | None
    fingerprint: str
    fields: Mapping[str, Any]
    recorded_at: str = field(default_factory=now_iso)
    recorded_by: str = "pull"
    schema: int = SCHEMA
    identity: LinkIdentity | None = None
    remote_revision: str | None = None
    lineage: Literal["clean", "unknown", "mixed-lineage"] = "unknown"
    outbound: OutboundReceipt | None = None
    #: Managed skill aliases are directory/display names, not their immutable id.
    #: Keep this outside the transferred skill files and out of the policy digest.
    local_alias: str | None = None
    #: Capture legacy defaults once even if source resolution is deferred;
    #: this is per-artifact migration metadata, not another policy database.
    legacy_subscribe_hint: bool | None = None
    scope_known: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", deepcopy(dict(self.fields)))

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind,
            "local_id": self.local_id,
            "hub_id": self.hub_id,
            "tenant_id": self.tenant_id,
            "fingerprint": self.fingerprint,
            "fields": deepcopy(dict(self.fields)),
            "recorded_at": self.recorded_at,
            "recorded_by": self.recorded_by,
            "identity": self.identity.to_json() if self.identity is not None else None,
            "remote_revision": self.remote_revision,
            "lineage": self.lineage,
            "outbound": self.outbound.to_json() if self.outbound is not None else None,
            "local_alias": self.local_alias,
            "legacy_subscribe_hint": self.legacy_subscribe_hint,
            "scope_known": self.scope_known,
        }


def _check_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise BaselineError(f"{label} {value!r} is not a safe path segment")
    return value


def baseline_path(config_dir: Path, kind: Kind, local_id: str) -> Path:
    if kind not in ("agent", "team", "skill"):
        raise BaselineError(f"unknown kind {kind!r}")
    return baselines_dir(config_dir) / f"{kind}-{_check_id(local_id, 'local id')}.json"


def iter_baselines(config_dir: Path) -> list[BaselineRecord]:
    """Every readable baseline record, across kinds, sorted for a stable listing.

    The identity inventory the links endpoint enumerates: a baseline IS the
    per-origin record, so there is no second links database. An unreadable file
    is skipped rather than failing the whole read -- one corrupt record must not
    hide the rest, and the reader surfaces what it can.
    """

    directory = baselines_dir(config_dir)
    try:
        entries = sorted(directory.glob("*.json"))
    except OSError:
        return []
    records: list[BaselineRecord] = []
    for path in entries:
        stem = path.name[: -len(".json")]
        kind, _, local_id = stem.partition("-")
        if kind not in ("agent", "team", "skill") or not local_id:
            continue
        record = read_baseline(config_dir, kind, local_id)  # type: ignore[arg-type]
        if record is not None:
            records.append(record)
    return records


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Temp file in the same directory + ``os.replace`` (the monitors/state shape)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _refuse_links(path: Path) -> None:
    """A symlinked record or directory is refused: it could aim a write elsewhere."""

    for candidate in (path, path.parent):
        if candidate.is_symlink():
            raise BaselineError(f"{candidate} is a symlink; refusing to use it")


@serialized_config_path_write
def write_baseline(config_dir: Path, record: BaselineRecord) -> Path:
    path = baseline_path(config_dir, record.kind, record.local_id)
    _refuse_links(path)
    if record.identity is not None and (
        record.identity.kind,
        record.identity.local_id,
        record.identity.hub_id,
        record.identity.tenant_id,
    ) != (record.kind, record.local_id, record.hub_id, record.tenant_id):
        raise LinkError("origin-conflict", "baseline identity does not match the local origin")
    current = read_baseline(config_dir, record.kind, record.local_id)
    if current is not None:
        # A missing historical API URL is NOT permission to change the known
        # remote id or organization. Only a genuinely unknown scope may be
        # selected explicitly during source resolution.
        if current.hub_id != record.hub_id or (
            current.scope_known and current.tenant_id != record.tenant_id
        ):
            raise LinkError(
                "origin-conflict", "retain the existing remote identity and known scope"
            )
        if current.legacy_subscribe_hint is not None:
            record = replace(record, legacy_subscribe_hint=current.legacy_subscribe_hint)
    if current is not None and current.identity is not None:
        # Existing legacy record helpers update integrated B. They must not
        # erase immutable origin or the independently accepted outbound A/P
        # pair, nor turn another publish target into a silent relink.
        if (record.hub_id, record.tenant_id) != (current.hub_id, current.tenant_id):
            raise LinkError("origin-conflict", "create a separate source-scoped local copy")
        if record.identity is not None:
            current.identity.require_same_origin(record.identity)
        record = replace(
            record,
            identity=record.identity or current.identity,
            outbound=record.outbound or current.outbound,
            lineage=current.lineage if record.lineage == "unknown" else record.lineage,
            local_alias=record.local_alias or current.local_alias,
        )
    if type(record.scope_known) is not bool or (
        record.legacy_subscribe_hint is not None and type(record.legacy_subscribe_hint) is not bool
    ):
        raise LinkError("invalid-link-policy", "legacy policy hint must be a boolean")
    if record.identity is not None and not record.scope_known:
        raise LinkError("origin-conflict", "an established identity requires an explicit scope")
    if record.kind == "skill" and record.identity is not None:
        from local_operator.hub_sync.skill_document import validate_name

        validate_name(record.local_alias)
    _atomic_write(path, record.to_json())
    return path


def read_baseline(config_dir: Path, kind: Kind, local_id: str) -> BaselineRecord | None:
    """The record, or None when absent OR unusable (an unusable record is unknown, not trusted)."""

    try:
        path = baseline_path(config_dir, kind, local_id)
        _refuse_links(path)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (BaselineError, OSError, ValueError) as exc:
        logger.warning("ignoring unreadable hub baseline for %s %s: %s", kind, local_id, exc)
        return None
    return parse_baseline_record(raw, kind, local_id)


def parse_baseline_record(raw: Any, kind: Kind, local_id: str) -> BaselineRecord | None:
    """Validate a host journal candidate before replacing the live side record."""
    if not isinstance(raw, dict) or raw.get("schema") not in (LEGACY_SCHEMA, SCHEMA):
        return None
    fields = raw.get("fields")
    hub_id = raw.get("hub_id")
    if raw.get("kind") != kind or raw.get("local_id") != local_id:
        return None
    if not isinstance(fields, dict) or not isinstance(hub_id, str) or not hub_id:
        return None
    tenant = raw.get("tenant_id")
    tenant = tenant if isinstance(tenant, str) and tenant else None
    identity = None
    outbound = None
    try:
        if raw.get("identity") is not None:
            if not isinstance(raw["identity"], dict):
                return None
            identity = LinkIdentity.from_json(raw["identity"])
            if (identity.kind, identity.local_id, identity.hub_id, identity.tenant_id) != (
                kind,
                local_id,
                hub_id,
                tenant,
            ):
                return None
        if raw.get("outbound") is not None:
            value = raw["outbound"]
            if not isinstance(value, dict):
                return None
            for key in ("remote_revision", "operation_id"):
                if value.get(key) is not None and not isinstance(value[key], str):
                    return None
            local, accepted = value.get("local_fields"), value.get("accepted_fields")
            if not isinstance(local, dict) or not isinstance(accepted, dict):
                return None
            if value.get("input_fingerprint") != fingerprint_snapshot(kind, local):
                return None
            # Also validate the accepted projection, especially every skill
            # path/byte. An invalid persisted pair must not become publish input.
            fingerprint_snapshot(kind, accepted)
            outbound = OutboundReceipt(
                local_fields=local,
                accepted_fields=accepted,
                input_fingerprint=value["input_fingerprint"],
                remote_revision=value.get("remote_revision"),
                operation_id=value.get("operation_id"),
                recorded_at=str(value.get("recorded_at") or ""),
            )
    except (LinkError, ValueError, TypeError):
        return None
    if raw.get("remote_revision") is not None and not isinstance(raw["remote_revision"], str):
        return None
    hint = raw.get("legacy_subscribe_hint")
    scope_known = raw.get("scope_known", "tenant_id" in raw)
    if type(scope_known) is not bool or (hint is not None and type(hint) is not bool):
        return None
    if identity is not None and not scope_known:
        return None
    local_alias = raw.get("local_alias")
    if kind == "skill" and identity is not None:
        from local_operator.hub_sync.skill_document import (
            SkillValidationError,
            validate_name,
        )

        try:
            validate_name(local_alias)
        except SkillValidationError:
            return None
    elif local_alias is not None and not isinstance(local_alias, str):
        return None
    lineage = raw.get("lineage", "unknown")
    if lineage not in ("clean", "unknown", "mixed-lineage"):
        return None
    return BaselineRecord(
        kind=kind,
        local_id=local_id,
        hub_id=hub_id,
        tenant_id=tenant,
        fingerprint=str(raw.get("fingerprint") or ""),
        fields=fields,
        recorded_at=str(raw.get("recorded_at") or ""),
        recorded_by=str(raw.get("recorded_by") or ""),
        schema=raw["schema"],
        identity=identity,
        remote_revision=raw.get("remote_revision"),
        lineage=lineage,
        outbound=outbound,
        local_alias=local_alias,
        legacy_subscribe_hint=hint,
        scope_known=scope_known,
    )


def delete_baseline(config_dir: Path, kind: Kind, local_id: str) -> None:
    try:
        baseline_path(config_dir, kind, local_id).unlink(missing_ok=True)
    except (BaselineError, OSError):
        logger.debug("could not remove baseline for %s %s", kind, local_id, exc_info=True)


def prune(
    config_dir: Path,
    kind: Kind,
    live_ids: set[str],
    *,
    confirmed_absent: Callable[[str], bool] | None = None,
) -> list[str]:
    """Drop records whose local item is gone (deletes are not hooked; this is the sweep).

    A baseline is the guard against silently re-adding what the user deleted, so
    it must never be destroyed on a listing that can be transiently short (a
    registry read that raced a save, or failed and read as empty). When
    ``confirmed_absent`` is given, a record is dropped only if it ALSO answers
    True for that id — i.e. the item is confirmed gone on disk at prune time,
    not merely missing from ``live_ids``.
    """

    removed: list[str] = []
    directory = baselines_dir(config_dir)
    try:
        entries = list(directory.glob(f"{kind}-*.json"))
    except OSError:
        return removed
    for path in entries:
        local_id = path.name[len(kind) + 1 : -len(".json")]
        if local_id not in live_ids and (confirmed_absent is None or confirmed_absent(local_id)):
            try:
                path.unlink()
                removed.append(local_id)
            except OSError:
                pass
    return removed


def make_record(
    kind: Kind,
    local_id: str,
    hub_id: str,
    tenant_id: str | None,
    fields: Mapping[str, Any],
    recorded_by: str,
) -> BaselineRecord:
    return BaselineRecord(
        kind=kind,
        local_id=local_id,
        hub_id=hub_id,
        tenant_id=tenant_id,
        fingerprint=fingerprint_fields(kind, fields),
        fields=dict(fields),
        recorded_by=recorded_by,
    )


# -- stamping hooks (called by the pull paths) ----------------------------------------


def record_agent_baseline(
    config_dir: Path,
    *,
    local_id: str,
    hub_id: str,
    instructions: str,
    description: str,
    tenant_id: str | None = None,
    recorded_by: str = "pull",
) -> BaselineRecord | None:
    """Write B for an agent. Never raises: a pull must not fail over bookkeeping.

    A failure leaves the row without a record, which degrades it to
    baseline-unknown (check-only, never auto-applied) — the safe direction.
    """

    try:
        record = make_record(
            "agent",
            local_id,
            hub_id,
            tenant_id,
            {"instructions": norm(instructions), "description": norm(description)},
            recorded_by,
        )
        write_baseline(config_dir, record)
        return record
    except (BaselineError, OSError, ValueError):
        logger.warning("could not record hub baseline for agent %s", local_id, exc_info=True)
        return None


def record_team_pull(
    config_dir: Path,
    team: Any,
    document: Mapping[str, Any],
    *,
    tenant_id: str,
    recorded_by: str = "pull",
) -> BaselineRecord | None:
    """Link a freshly imported team to its hub document (B1.2).

    B is the fields of the STORED row (re-read after import so name/cap
    normalisation is reflected — the bytes a later re-fetch is compared with).
    Returns None (and logs) if the document names no id or the write fails; the
    pull itself has already succeeded and stays successful.
    """

    hub_id = str(document.get("id") or "").strip()
    if not hub_id:
        logger.warning("hub team document for %r carries no id; not linking it", team.name)
        return None
    try:
        record = make_record(
            "team",
            team.id,
            hub_id,
            tenant_id or None,
            team_fields(
                description=team.description,
                manager=team.manager,
                members=team.members,
                instructions=team.instructions,
                project=team.project,
            ),
            recorded_by,
        )
        write_baseline(config_dir, record)
        return record
    except (BaselineError, OSError, ValueError):
        logger.warning("could not record hub baseline for team %s", team.id, exc_info=True)
        return None


# -- backups (A9.3) --------------------------------------------------------------------


def write_backup(
    config_dir: Path, kind: Kind, local_id: str, fields: Mapping[str, Any], reason: str
) -> str:
    """Write the pre-apply local fields; keep the newest :data:`BACKUPS_KEPT`.

    Returns the path RELATIVE to ``config_dir`` (what the report and the store
    carry — an absolute path would leak the home directory into a payload).
    """

    _check_id(local_id, "local id")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = backups_dir(config_dir) / f"{kind}-{local_id}-{stamp}.json"
    _atomic_write(path, {"schema": SCHEMA, "reason": reason, "fields": dict(fields)})
    existing = sorted(backups_dir(config_dir).glob(f"{kind}-{local_id}-*.json"))
    for stale in existing[:-BACKUPS_KEPT]:
        try:
            stale.unlink()
        except OSError:
            pass
    return str(path.relative_to(config_dir))
