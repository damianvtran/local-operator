"""Agent and team DEFINITIONS over the mesh: what makes a name resolvable on a peer.

WHY THIS EXISTS
===============

R8 lets one device create a session on another (``net_session_create``), and the
create could name a working directory, a model, a title and a first prompt —
nothing about WHO the session is. So a session started on a peer could not name
an agent profile or a team, and even if the frame had carried the name there was
nothing on that device to resolve it against: agent rows live in
``<config>/agents/<id>/`` and team rows in ``<config>/teams/<id>/``, and neither
has ever crossed the wire. The user-visible consequence is the one the operator
reported: agent and team context is dropped the moment work is placed on a peer.

There are two halves to closing that, and they are separate on purpose:

1. **The create frame carries the identity** (``relay._op_session_create``): the
   agent/profile name, an optional team name, and the reasoning effort. That half
   is worth nothing alone, which is why the peer REFUSES BY NAME when the name
   does not resolve instead of falling back to the default agent — a silent
   fallback runs the wrong thing under the right name, which is worse than a
   refusal because the user cannot see it.
2. **A versioned, non-credential payload carries the definitions**
   (this module). It is what makes a name resolve on a device that has never seen
   it — including a CLEAN install that was paired a minute ago and has no config,
   no agent rows and no team rows. That case is the point rather than a corner:
   the forward direction this design must not need a refactor for is a pod that
   boots a bare ``lop``, pairs, runs workloads and spins down
   (``docs/design/mesh-compute-pool.md``, R21), and a pod has nobody at its end to
   set definitions up by hand.

WHY A NEW OP RATHER THAN A RIDE ON ``net_sync``
==============================================

``net_sync`` is the mobility primitive for SESSION replicas (its own docstring
says so, and §7.2's copy set is a spec of what a session directory may hold).
Definitions are install-wide configuration, not session content: they have no
owner device, no transcript and no replica directory. Reusing the session op
would put configuration rows inside a payload whose every reader assumes
"session", and the copy-set completeness guard would then have to reason about
agent rows. So this is ``net_definitions``, one op with two phases, registered
through the slice mechanism ``relay.SLICE_MODULES`` already has.

WHAT TRAVELS, AND WHAT DELIBERATELY DOES NOT
============================================

A bundle is ``{"kind": "lop.mesh.definitions.v1", "version": 1, ...}`` carrying
rows for agents and teams. Agents carry an ALLOW-LIST of fields
(:data:`AGENT_DEFINITION_FIELDS`) plus ``system_prompt.md``; teams carry
``team.yml``'s metadata plus both brief files. Everything that is HISTORY or
MACHINE-LOCAL is excluded by construction rather than by a deny-list:
``last_message`` / ``last_message_datetime`` (a private chat log), conversation
and execution history (which live in files this module never reads),
``current_working_directory`` (a path on the source device — the same argument
``desktop_mesh.create_on_peer`` makes for not forwarding ``cwd``), sessions and
their transcripts, and every credential store. Definitions are configuration.

``security_prompt`` is excluded with a reason of its own. It is a prompt, not a
secret, but it is the field most likely to describe a particular fleet — the
hosts, tenants or accounts an agent's code is allowed to touch — and a mirrored
copy of it would then apply on a device that is not in that fleet. A mirror
therefore keeps the local default, which is the safe direction for a field whose
whole job is to narrow what executes.

THE CREDENTIAL ASSERTION, AND WHY IT IS STRUCTURAL
==================================================

The requirement is that this payload can never carry a credential, and the
honest form of that is not a promise in a docstring: the builder reads agent and
team files and NOTHING ELSE — it never opens a config value, an environment
variable or a secret store, so there is no channel for one to arrive down. On top
of that structural fact there is a check, because a user CAN paste a token into
a role's instructions (that has happened: the incident ``redaction_shapes``
opens with is a DSN captured from a remote host into a transcript). Every free
text field of every row is passed through the shape table
(``redaction_shapes.match_shape_names``), and a row that trips it is WITHHELD
from the bundle and reported by name — never sent, and never installed if an
older or hostile peer sends it.
``tests/unit/network/test_definitions.py`` asserts both halves: that no value
from a session's credential store can appear in a built bundle, and that a row
carrying a credential-shaped value is withheld at the sender and refused at the
receiver.

THE CONFLICT POLICY
===================

Both devices may have a row by the same name, and the two may differ. The policy
is **the author owns the row, and a mirror follows its origin**:

* Every row this device installs from elsewhere is recorded in a provenance
  index (``<config>/network/definitions.json``) with the ORIGIN device, the row's
  id and the digest of what was installed.
* **Authored wins.** A row this device authored (no index entry) or one whose
  current digest differs from the recorded one (an operator edited a mirror — it
  is now theirs) is NEVER overwritten. That is a conflict, and it is REPORTED by
  name to the sender, which refuses the create rather than running a definition
  the user did not choose.
* **A mirror follows its own origin only.** A row installed from device A is
  updated by A and by nobody else; a row from device B for the same name is a
  conflict, because adopting it would silently transfer authorship of a name the
  user knows.
* **Idempotent.** An incoming row whose digest equals the recorded digest takes
  the ``unchanged`` path and touches no file, so the same bundle applied twice is
  a no-op and the cadence cannot churn the disk.
* **Deletions are not propagated** in v1, and that is a decision rather than a
  gap: a delete is indistinguishable from "the sender's config root moved", and
  mirroring a delete would delete configuration on the strength of an absence.
  Stated here so the next reader does not read it as an oversight.

Last-writer-wins is deliberately NOT the policy. It converges, and it is exactly
wrong for a name: two devices editing ``reviewer`` would each silently clobber the
other, and the loser would have no way to see it happened.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.network.relay import RelayServer

logger = logging.getLogger(__name__)

#: The bundle's own name and version. ``kind`` is checked on arrival so a payload
#: from a future or foreign producer is refused by name rather than partially
#: applied: an unrecognised bundle is not a bundle this build can be sure it
#: understands, and half-applying one is how a device ends up with a roster that
#: resolves on one machine and not the other.
BUNDLE_KIND = "lop.mesh.definitions.v1"
BUNDLE_VERSION = 1

#: The provenance index, inside the network directory. It is what makes the
#: conflict policy decidable: without a record of what this device mirrored, "this
#: row was authored here" and "this row is a copy nobody edited" are the same
#: bytes on disk.
INDEX_NAME = "definitions.json"

#: The agent fields a bundle carries. An ALLOW-LIST, so a field added to
#: ``AgentData`` later is excluded by default and has to be argued for here —
#: the reverse (a deny-list) leaks every future field to every peer. What is
#: absent and why is in the module docstring; ``security_prompt`` in particular.
AGENT_DEFINITION_FIELDS: tuple[str, ...] = (
    "name",
    "description",
    "tags",
    "categories",
    "hosting",
    "model",
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "stop",
    "frequency_penalty",
    "presence_penalty",
    "seed",
)

#: Free text of a row that the credential-shape check runs over. Named as its own
#: tuple so "what is scanned" cannot drift from "what is carried": a field added
#: above and forgotten here would be sent unchecked.
AGENT_TEXT_FIELDS: tuple[str, ...] = ("description", "system_prompt")
TEAM_TEXT_FIELDS: tuple[str, ...] = ("description", "instructions", "project")

#: Bounds, all of them the ones the product already applies locally where one
#: exists: a definition rides in front of a session's prompt on every turn, so an
#: unbounded body is an unbounded per-turn bill, and an unbounded ROW COUNT is an
#: unbounded payload. The brief caps are imported from the modules that own them
#: so a mirror can never store a body the local editor would refuse to write.
MAX_BUNDLE_BYTES = 2 * 1024 * 1024
MAX_DEFINITION_ROWS = 500

#: Owner-side deadline for ``net_definitions``. A bundle write is disk work over
#: up to :data:`MAX_DEFINITION_ROWS` rows, so the op runs OFF the link's reader
#: (``register_ops``'s ``slow``) with a budget that outlasts a slow disk — the
#: same reasoning ``net_sync`` gives for its own deadline.
DEFINITIONS_OP_DEADLINE_S = 60.0

#: How long one push may take from the requester's side. The create path waits on
#: it before sending its create frame, so it is bounded well under the create's
#: own 120 s budget: a definition push that hangs must not be what makes a create
#: look unreachable.
PUSH_TIMEOUT_S = 30.0


# ---------------------------------------------------------------------------
# Paths and the provenance index
# ---------------------------------------------------------------------------


def index_path(root: Path) -> Path:
    from local_operator.network import sync as sync_mod

    return sync_mod.network_dir(root) / INDEX_NAME


def read_index(root: Path) -> dict[str, Any]:
    """The provenance index, or an empty one when absent or unusable.

    Tolerant by contract, like every other sidecar reader in this package: an
    unreadable index costs a mirror the ability to be updated (the next row for
    that name is reported as a conflict), and it must never cost a device its
    ability to serve a peer at all.
    """
    try:
        payload = json.loads(index_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": BUNDLE_VERSION, "agents": {}, "teams": {}}
    if not isinstance(payload, dict):
        return {"version": BUNDLE_VERSION, "agents": {}, "teams": {}}
    for key in ("agents", "teams"):
        if not isinstance(payload.get(key), dict):
            payload[key] = {}
    return payload


def write_index(root: Path, payload: Mapping[str, Any]) -> None:
    """Publish the index atomically at 0600.

    Its own writer rather than ``store``'s private one: this file is not a
    network record, and reaching an underscore helper in a sibling module is a
    dependency nobody declared. The staging discipline is the package's own —
    unique temp beside the target, then ``os.replace`` — because a reader must
    never see a half-written index (a truncated one reads as "nothing was ever
    mirrored", which turns every mirror into a conflict).
    """
    target = index_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _index_rows(index: Mapping[str, Any], kind: str) -> dict[str, Any]:
    rows = index.get(kind if kind == "agents" else "teams")
    return rows if isinstance(rows, dict) else {}


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


def canonical(payload: Any) -> str:
    """One deterministic spelling of a payload, for hashing and for the wire.

    ``sort_keys`` and tight separators, so two devices that built the same row
    agree on its digest whatever order their dicts happened to be in. A digest
    that depended on insertion order would make every row look edited locally on
    one of the two devices, which is the conflict path — i.e. the whole feature
    would quietly become "always a conflict".
    """
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_of(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical(row).encode("utf-8")).hexdigest()


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value or "")


def _parse_iso(value: Any) -> datetime:
    """A row's creation date, or now.

    Never raises: a date this build cannot read is not a reason to refuse a
    definition, and ``created_date`` is record bookkeeping rather than behaviour.
    """
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# The credential guard
# ---------------------------------------------------------------------------


def credential_shape(text: Any) -> str:
    """The credential shape ``text`` is spelled like, or ``""``.

    Through the harness's ONE shape table (``redaction_shapes``), never a second
    pattern of this module's own: the table is what every export, tool result and
    transcript is scanned with, so a definition guarded by a different copy of it
    would eventually disagree with the rest of the product about what a credential
    looks like.
    """
    value = str(text or "")
    if not value.strip():
        return ""
    try:
        from local_operator.redaction_shapes import match_shape_names
    except Exception:  # noqa: BLE001 — an unimportable guard must not block a sync
        logger.debug("definitions: credential shape table unavailable", exc_info=True)
        return ""
    labels = match_shape_names(value)
    return labels[0] if labels else ""


def _bounded_text(value: Any, limit: int) -> str:
    return str(value or "")[:limit]


# ---------------------------------------------------------------------------
# Building the bundle
# ---------------------------------------------------------------------------


def _agent_row_from_agent(agent: Any, system_prompt: str) -> dict[str, Any]:
    """One agent's DEFINITION: the allow-listed fields plus its instructions."""
    fields = {name: getattr(agent, name, None) for name in AGENT_DEFINITION_FIELDS}
    fields["name"] = str(fields.get("name") or "")
    return {
        "kind": "agent",
        "name": fields["name"],
        "origin_id": str(getattr(agent, "id", "") or ""),
        "created_date": _iso(getattr(agent, "created_date", "")),
        "fields": fields,
        "system_prompt": system_prompt,
    }


def _agent_row_from_bundle(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The row a bundle will actually install/report, normalised.

    Rebuilt through this function on BOTH sides, so the digest the sender
    recorded and the digest the receiver computes are over the same shape even
    when one of them added or dropped an optional key. Without the single
    rebuild, the two ends disagree about the digest of an unchanged row and every
    push is a conflict.
    """
    fields_in = payload.get("fields")
    fields_in = fields_in if isinstance(fields_in, Mapping) else {}
    fields: dict[str, Any] = {}
    for name in AGENT_DEFINITION_FIELDS:
        value = fields_in.get(name)
        if name == "name":
            value = str(payload.get("name") or value or "")
        elif name == "description":
            value = str(value or "")
        elif name in ("tags", "categories"):
            value = [str(item) for item in (value or [])] if isinstance(value, list) else []
        fields[name] = value
    return {
        "kind": "agent",
        "name": str(fields["name"]),
        "origin_id": str(payload.get("origin_id") or ""),
        "created_date": _iso(payload.get("created_date") or ""),
        "fields": fields,
        "system_prompt": _bounded_text(payload.get("system_prompt"), _max_instructions()),
    }


def _max_instructions() -> int:
    from local_operator.agent_profiles import MAX_INSTRUCTIONS_CHARS

    return int(MAX_INSTRUCTIONS_CHARS)


def _max_team_instructions() -> int:
    from local_operator.teams import MAX_TEAM_INSTRUCTIONS_CHARS

    return int(MAX_TEAM_INSTRUCTIONS_CHARS)


def _team_row_from_team(team: Any) -> dict[str, Any]:
    members = [
        {
            "role": str(getattr(member, "role", "")),
            "count": int(getattr(member, "count", 1) or 1),
            "kind": str(getattr(member, "kind", "agent") or "agent"),
        }
        for member in (getattr(team, "members", None) or [])
    ]
    return {
        "kind": "team",
        "id": str(getattr(team, "id", "") or ""),
        "name": str(getattr(team, "name", "") or ""),
        "created_date": _iso(getattr(team, "created_date", "")),
        "description": str(getattr(team, "description", "") or ""),
        "manager": str(getattr(team, "manager", "") or "manager"),
        "members": members,
        "instructions": _bounded_text(getattr(team, "instructions", ""), _max_team_instructions()),
        "project": _bounded_text(getattr(team, "project", ""), _max_team_instructions()),
    }


def _team_row_from_bundle(payload: Mapping[str, Any]) -> dict[str, Any]:
    members_in = payload.get("members")
    members: list[dict[str, Any]] = []
    if isinstance(members_in, list):
        for raw in members_in[:64]:
            if not isinstance(raw, Mapping):
                continue
            members.append(
                {
                    "role": str(raw.get("role") or ""),
                    "count": int(raw.get("count") or 1),
                    "kind": "team" if str(raw.get("kind") or "agent") == "team" else "agent",
                }
            )
    limit = _max_team_instructions()
    return {
        "kind": "team",
        "id": str(payload.get("id") or ""),
        "name": str(payload.get("name") or ""),
        "created_date": _iso(payload.get("created_date") or ""),
        "description": str(payload.get("description") or ""),
        "manager": str(payload.get("manager") or "manager"),
        "members": members,
        "instructions": _bounded_text(payload.get("instructions"), limit),
        "project": _bounded_text(payload.get("project"), limit),
    }


def _row_texts(row: Mapping[str, Any]) -> list[str]:
    if row.get("kind") == "team":
        return [str(row.get(name) or "") for name in TEAM_TEXT_FIELDS]
    texts = [str(row.get("system_prompt") or "")]
    fields = row.get("fields")
    if isinstance(fields, Mapping):
        texts.extend(
            str(fields.get(name) or "") for name in AGENT_TEXT_FIELDS if name != "system_prompt"
        )
    return texts


def _withheld(row: Mapping[str, Any]) -> str:
    """The shape label that keeps this row off the wire, or ``""``."""
    for text in _row_texts(row):
        label = credential_shape(text)
        if label:
            return label
    return ""


def authored_locally(root: Path, *, kind: str, name: str, row: Mapping[str, Any]) -> bool:
    """Is this row the OPERATOR's, rather than a mirror of somebody else's?

    The question the conflict policy turns on. Two ways to be authored here:
    nothing was ever mirrored under this name, or what was mirrored no longer
    matches what is on disk (the operator edited it — it is theirs now).
    """
    index = read_index(root)
    recorded = _index_rows(index, kind).get(name)
    if not isinstance(recorded, Mapping):
        return True
    return str(recorded.get("digest") or "") != digest_of(row)


def local_rows(
    root: Path, *, kinds: Iterable[str] = ("agents", "teams")
) -> dict[str, list[dict[str, Any]]]:
    """This device's own definitions, as bundle rows, by index key.

    Returns ``{"agents": [...], "teams": [...]}`` where each row is canonical
    (:func:`_agent_row_from_bundle`/:func:`_team_row_from_bundle`), so a row built
    here and the same row read back from a peer hash identically.
    """
    wanted = set(kinds)
    out: dict[str, list[dict[str, Any]]] = {"agents": [], "teams": []}
    if "agents" in wanted:
        out["agents"] = [r for r in _read_agents(root)]
    if "teams" in wanted:
        out["teams"] = [r for r in _read_teams(root)]
    return out


def _read_agents(root: Path) -> list[dict[str, Any]]:
    from local_operator.agents import AgentRegistry

    registry = AgentRegistry(root)
    rows: list[dict[str, Any]] = []
    for agent in sorted(registry.list_agents(), key=lambda item: str(item.name).casefold()):
        try:
            prompt = registry.get_agent_system_prompt(str(agent.id)) or ""
        except Exception:  # noqa: BLE001 — one unreadable prompt is not all of them
            logger.debug("definitions: could not read the prompt for %s", agent.id, exc_info=True)
            prompt = ""
        raw = _agent_row_from_agent(agent, prompt)
        rows.append(_agent_row_from_bundle(raw))
    return rows


def _read_teams(root: Path) -> list[dict[str, Any]]:
    from local_operator.teams import TeamRegistry

    registry = TeamRegistry(root)
    rows: list[dict[str, Any]] = []
    # ``list_teams`` is metadata-only BY CONTRACT (briefs are deliberately not
    # hydrated), so each row is re-read through ``get_team`` — hashing a row whose
    # briefs read as "" would make every team look locally edited.
    for summary in sorted(registry.list_teams(), key=lambda item: str(item.name).casefold()):
        try:
            team = registry.get_team(str(summary.id))
        except Exception:  # noqa: BLE001 — a bad row must not hide the others
            logger.debug("definitions: could not read team %s", summary.id, exc_info=True)
            continue
        rows.append(_team_row_from_bundle(_team_row_from_team(team)))
    return rows


def local_bundle(
    root: Path,
    *,
    names: Mapping[str, Iterable[str]] | None = None,
    include_mirrors: bool = False,
) -> dict[str, Any]:
    """This device's definitions as one versioned, non-credential payload.

    ``names`` narrows the bundle to ``{"agents": {...}, "teams": {...}}``; the
    create path passes the handful of names its frame mentions so a create does
    not ship an install's whole configuration on every keystroke.
    ``include_mirrors`` is off by default and that default is load-bearing: a
    device that re-sent rows it had itself mirrored would hand them to a third
    device as if it had authored them, which both loses the authorship the
    conflict policy rests on and lets two peers echo one row back and forth.
    """
    from local_operator.network import identity as identity_mod

    device_id = ""
    try:
        device_id = identity_mod.load_or_mint(root).device_id
    except Exception:  # noqa: BLE001 — an unnamed origin is still a usable bundle
        logger.debug("definitions: no identity for the bundle origin", exc_info=True)

    agents = _read_agents(root)
    teams = _read_teams(root)
    withheld: list[dict[str, str]] = []
    bundle: dict[str, Any] = {
        "kind": BUNDLE_KIND,
        "version": BUNDLE_VERSION,
        "origin_device": device_id,
        "agents": [],
        "teams": [],
        "withheld": withheld,
    }
    for kind, rows in (("agents", agents), ("teams", teams)):
        wanted = names.get("agents" if kind == "agents" else "teams") if names else None
        wanted_set = {str(item).strip().casefold() for item in (wanted or ()) if str(item).strip()}
        for row in rows[:MAX_DEFINITION_ROWS]:
            if wanted_set and str(row.get("name") or "").casefold() not in wanted_set:
                continue
            if not include_mirrors and not authored_locally(
                root, kind=kind, name=str(row.get("name") or ""), row=row
            ):
                # A mirror: it belongs to the device that authored it, and that
                # device syncs it directly. See the docstring.
                continue
            label = _withheld(row)
            if label:
                withheld.append(
                    {"kind": row["kind"], "name": str(row.get("name") or ""), "shape": label}
                )
                continue
            bundle["agents" if kind == "agents" else "teams"].append(row)
    payload = canonical(bundle).encode("utf-8")
    if len(payload) > MAX_BUNDLE_BYTES:
        raise DefinitionsRefused(
            "bundle_too_large",
            f"this device's definitions are {len(payload)} bytes, over the "
            f"{MAX_BUNDLE_BYTES}-byte limit, so nothing was sent; sync a single "
            "definition by name instead",
        )
    bundle["digest"] = hashlib.sha256(
        canonical({k: v for k, v in bundle.items() if k != "digest"}).encode("utf-8")
    ).hexdigest()
    return bundle


def expect_from_push(result: Mapping[str, Any]) -> dict[str, Any]:
    """The ``expect`` block for a create frame, built from a push's own answer.

    Non-empty only when the push SUCCEEDED, and that is the interesting half: a
    peer whose definitions could not be reconciled must not be asked to prove a
    revision neither side agreed on. Such a create is better refused by the
    peer's own ``definition_missing`` sentence (which NAMES the missing name) than
    by a digest comparison against a row that is not there.
    """
    if not result.get("ok"):
        return {}
    out: dict[str, Any] = {}
    for key in ("agents", "teams"):
        rows = result.get(key)
        if isinstance(rows, Mapping) and rows:
            out[key] = {str(name): str(digest) for name, digest in rows.items()}
    return out


class DefinitionsRefused(Exception):
    """A refusal this module composed, carrying the code a surface branches on."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Applying a bundle
# ---------------------------------------------------------------------------


def bundle_rows(bundle: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(agents, teams)`` from a bundle, each row canonicalised and re-checked.

    The RECEIVER's own gate. Everything here is re-derived from the payload
    rather than trusted: the sender's digest is not used (it is recomputed), and
    every row is rebuilt through the same normaliser the sender used so the two
    ends agree. A row this build cannot normalise is dropped with a reason rather
    than installed half-formed.
    """
    if str(bundle.get("kind") or "") != BUNDLE_KIND:
        raise DefinitionsRefused(
            "unknown_bundle",
            f"that payload is {bundle.get('kind')!r}, not {BUNDLE_KIND}, so nothing "
            "was installed",
        )
    version = int(bundle.get("version") or 0)
    if version != BUNDLE_VERSION:
        raise DefinitionsRefused(
            "unknown_bundle_version",
            f"that payload is version {version} of {BUNDLE_KIND} and this build "
            f"understands version {BUNDLE_VERSION}, so nothing was installed",
        )
    agents_in = bundle.get("agents")
    teams_in = bundle.get("teams")
    agents = [
        _agent_row_from_bundle(raw)
        for raw in (agents_in if isinstance(agents_in, list) else [])[:MAX_DEFINITION_ROWS]
        if isinstance(raw, Mapping)
    ]
    teams = [
        _team_row_from_bundle(raw)
        for raw in (teams_in if isinstance(teams_in, list) else [])[:MAX_DEFINITION_ROWS]
        if isinstance(raw, Mapping)
    ]
    return agents, teams


def apply_bundle(root: Path, bundle: Mapping[str, Any], *, origin_device: str) -> dict[str, Any]:
    """Install what this device does not have; report what it will not touch.

    The whole conflict policy is here, and it is stated as a table in the module
    docstring. Every outcome is a list of named rows, because the caller (a
    create, or ``lop network definitions push``) has to be able to tell the user
    WHICH name did not land.
    """
    agents, teams = bundle_rows(bundle)
    origin = str(origin_device or bundle.get("origin_device") or "")
    index = read_index(root)
    summary: dict[str, Any] = {
        "installed": [],
        "updated": [],
        "unchanged": [],
        "conflicts": [],
        "refused": [],
        "digests": {"agents": {}, "teams": {}},
    }
    for kind, rows in (("agents", agents), ("teams", teams)):
        for row in rows:
            name = str(row.get("name") or "")
            if not name:
                summary["refused"].append(
                    {"kind": row.get("kind", kind), "name": "", "reason": "unnamed"}
                )
                continue
            label = _withheld(row)
            if label:
                # The receiver's half of the credential assertion. A sender this
                # build talks to withholds these, so a row arriving with one is a
                # foreign or older producer — refused by name, never installed.
                summary["refused"].append(
                    {
                        "kind": row["kind"],
                        "name": name,
                        "reason": (
                            f"its text is shaped like a credential ({label}), which is never "
                            "installed"
                        ),
                    }
                )
                continue
            try:
                outcome = (
                    _apply_agent(root, row, origin, index)
                    if kind == "agents"
                    else _apply_team(root, row, origin, index)
                )
            except Exception as exc:  # noqa: BLE001 — one bad row must not stop the rest
                logger.debug("definitions: could not apply %s %s", kind, name, exc_info=True)
                summary["refused"].append({"kind": row["kind"], "name": name, "reason": str(exc)})
                continue
            bucket = summary[outcome["outcome"]]
            bucket.append({"kind": row["kind"], "name": name, **outcome.get("extra", {})})
            if outcome["outcome"] in ("installed", "updated", "unchanged"):
                summary["digests"][kind][name] = digest_of(row)
    if any(summary[key] for key in ("installed", "updated")):
        write_index(root, index)
    return summary


def _recorded(index: Mapping[str, Any], kind: str, name: str) -> Mapping[str, Any] | None:
    rows = _index_rows(index, kind)
    entry = rows.get(name)
    return entry if isinstance(entry, Mapping) else None


def _conflict(
    kind: str, name: str, reason: str, extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return {"outcome": "conflicts", "extra": {"reason": reason, **(extra or {})}}


def _apply_agent(
    root: Path, row: Mapping[str, Any], origin: str, index: dict[str, Any]
) -> dict[str, Any]:
    """Install or update ONE mirrored agent, or report why it was left alone."""
    from local_operator.agents import AgentData, AgentEditFields, AgentRegistry

    name = str(row["name"])
    registry = AgentRegistry(root)
    local = registry.get_agent_by_name(name)
    current = None
    local_id = str(getattr(local, "id", "") or "") if local is not None else ""
    if local is not None:
        prompt = ""
        try:
            prompt = registry.get_agent_system_prompt(local_id) or ""
        except Exception:  # noqa: BLE001 — an unreadable prompt reads as empty
            prompt = ""
        current = _agent_row_from_bundle(_agent_row_from_agent(local, prompt))

    recorded = _recorded(index, "agents", name)
    incoming = digest_of(row)

    # THE IDEMPOTENCE SHORT-CIRCUIT, AND IT RE-HASHES THE ROW ON DISK. Comparing the
    # incoming digest with what this device RECORDED is not enough: if the operator
    # edited the mirrored row after it was installed, the recorded digest still equals
    # the sender's (nothing has pushed since) while the row here is somebody else's
    # work — and returning "unchanged" for that would report a sync that is not one
    # and leave the sender believing its definition is current here. Measured in the
    # two-relay rig: an edited mirror came back as ``unchanged`` instead of a conflict,
    # so the conflict the policy promises was reachable only in unit tests.
    if (
        recorded is not None
        and str(recorded.get("digest") or "") == incoming
        and current is not None
        and digest_of(current) == str(recorded.get("digest") or "")
    ):
        return {"outcome": "unchanged"}

    if current is not None:
        if recorded is None:
            return _conflict(
                "agent",
                name,
                "this device authored an agent by that name, so it was not overwritten",
                {"local_id": local_id, "origin": origin},
            )
        if str(recorded.get("origin") or "") != origin:
            return _conflict(
                "agent",
                name,
                f"that name here is a copy of {recorded.get('origin')}'s, so "
                f"{origin or 'the sender'}'s row was not installed",
                {"local_id": local_id, "origin": str(recorded.get("origin") or "")},
            )
        if str(recorded.get("digest") or "") != digest_of(current):
            return _conflict(
                "agent",
                name,
                "the copy of that name here has local edits, so it was not overwritten",
                {"local_id": local_id},
            )
        if local_id and str(recorded.get("id") or "") not in ("", local_id):
            return _conflict(
                "agent",
                name,
                "the row recorded for that name is no longer the one on disk, so nothing "
                "was overwritten",
                {"local_id": local_id},
            )

    # A DELETED MIRROR IS RE-INSTALLED, and the asymmetry with an edited one is
    # deliberate. An EDIT is refused because there is local work to protect (the
    # conflict above); a DELETION removed this device's copy of a definition whose
    # owner is the sending device, so there is nothing of the operator's to lose — and
    # the sync's whole job is that the name RESOLVES here, which is what the create
    # that triggered it asked for. A cleanup on the peer therefore does not leave a
    # name permanently unresolvable, which is the dead end a blanket refusal would
    # create (there is no verb that clears an index entry).
    fields = dict(row.get("fields") or {})
    agent_id = local_id or str(row.get("origin_id") or "")
    if agent_id and registry_has_id(registry, agent_id) and not local_id:
        # The origin's id is taken here by another row: install under a fresh id
        # rather than overwrite a stranger. Name resolution is by name, so the
        # definition still resolves — only the id differs, and the index records
        # which one this device holds.
        agent_id = ""
    if agent_id:
        registry.save_agent(
            AgentData(
                id=agent_id,
                name=name,
                created_date=_parse_iso(row.get("created_date")),
                version=str(getattr(local, "version", "") or _local_version()),
                hosting=str(fields.get("hosting") or ""),
                model=str(fields.get("model") or ""),
                description=str(fields.get("description") or ""),
                tags=[str(item) for item in (fields.get("tags") or [])],
                categories=[str(item) for item in (fields.get("categories") or [])],
                temperature=fields.get("temperature"),
                top_p=fields.get("top_p"),
                top_k=fields.get("top_k"),
                max_tokens=fields.get("max_tokens"),
                stop=fields.get("stop"),
                frequency_penalty=fields.get("frequency_penalty"),
                presence_penalty=fields.get("presence_penalty"),
                seed=fields.get("seed"),
                current_working_directory=str(
                    getattr(local, "current_working_directory", "") or _portable_cwd()
                ),
                # SPELLED OUT, AND EMPTY ON PURPOSE. ``AgentData`` declares these as
                # stored fields whose values belong to a ROW's own history: a mirror
                # must not inherit the origin's last message or its security context
                # (that field names the hosts and tenants an agent's code may touch —
                # see the module docstring's exclusion list).
                security_prompt="",
                last_message="",
                last_message_datetime=datetime.now(timezone.utc),
            )
        )
    else:
        created = registry.create_agent(
            AgentEditFields(
                name=name,
                hosting=str(fields.get("hosting") or ""),
                model=str(fields.get("model") or ""),
                description=str(fields.get("description") or ""),
                tags=[str(item) for item in (fields.get("tags") or [])],
                categories=[str(item) for item in (fields.get("categories") or [])],
                temperature=fields.get("temperature"),
                top_p=fields.get("top_p"),
                top_k=fields.get("top_k"),
                max_tokens=fields.get("max_tokens"),
                stop=fields.get("stop"),
                frequency_penalty=fields.get("frequency_penalty"),
                presence_penalty=fields.get("presence_penalty"),
                seed=fields.get("seed"),
                # Spelled out (as every other caller of this model does) rather than
                # left to the field defaults: a mirror's security context is its OWN,
                # and its row starts with no history.
                security_prompt=None,
                last_message=None,
                current_working_directory=None,
            )
        )
        agent_id = str(created.id)
    registry.set_agent_system_prompt(agent_id, str(row.get("system_prompt") or ""))
    _index_rows_mut(index, "agents")[name] = {
        "origin": origin,
        "id": agent_id,
        "digest": incoming,
        "name": name,
        "at": time.time(),
    }
    return {"outcome": "updated" if current is not None else "installed", "extra": {"id": agent_id}}


def registry_has_id(registry: Any, agent_id: str) -> bool:
    try:
        registry.get_agent(agent_id)
    except (KeyError, Exception):  # noqa: BLE001 — absent id is the answer we want
        return False
    return True


def _index_rows_mut(index: dict[str, Any], kind: str) -> dict[str, Any]:
    rows = index.get(kind)
    if not isinstance(rows, dict):
        rows = {}
        index[kind] = rows
    return rows


def _local_version() -> str:
    try:
        from importlib.metadata import version

        return str(version("local-operator"))
    except Exception:  # noqa: BLE001 — a missing build stamp is not a refusal
        return ""


def _portable_cwd() -> str:
    from local_operator.paths import default_agent_cwd

    return default_agent_cwd()


def _apply_team(
    root: Path, row: Mapping[str, Any], origin: str, index: dict[str, Any]
) -> dict[str, Any]:
    """Install or update ONE mirrored team, or report why it was left alone."""
    from local_operator.teams import Team, TeamEditFields, TeamMember, TeamRegistry

    name = str(row["name"])
    registry = TeamRegistry(root)
    local = registry.get_team_by_name(name)
    current = None
    local_id = str(getattr(local, "id", "") or "") if local is not None else ""
    if local is not None:
        current = _team_row_from_bundle(_team_row_from_team(local))

    recorded = _recorded(index, "teams", name)
    incoming = digest_of(row)

    # Re-hashes the row on disk, for the reason spelled out at the same short-circuit
    # in ``_apply_agent``: a locally edited mirror must be REPORTED, not called
    # unchanged.
    if (
        recorded is not None
        and str(recorded.get("digest") or "") == incoming
        and current is not None
        and digest_of(current) == str(recorded.get("digest") or "")
    ):
        return {"outcome": "unchanged"}

    if current is not None:
        if recorded is None:
            return _conflict(
                "team",
                name,
                "this device authored a team by that name, so it was not overwritten",
                {"local_id": local_id, "origin": origin},
            )
        if str(recorded.get("origin") or "") != origin:
            return _conflict(
                "team",
                name,
                f"that name here is a copy of {recorded.get('origin')}'s, so "
                f"{origin or 'the sender'}'s row was not installed",
                {"local_id": local_id, "origin": str(recorded.get("origin") or "")},
            )
        if str(recorded.get("digest") or "") != digest_of(current):
            return _conflict(
                "team",
                name,
                "the copy of that name here has local edits, so it was not overwritten",
                {"local_id": local_id},
            )

    # See the note at the same place in ``_apply_agent``: a deleted mirror is
    # re-installed on purpose (nothing of the operator's is lost), while an edited one
    # is refused.
    members = [
        TeamMember(
            role=str(item.get("role") or ""),
            count=int(item.get("count") or 1),
            kind=str(item.get("kind") or "agent"),  # type: ignore[arg-type]
        )
        for item in (row.get("members") or [])
        if str(item.get("role") or "")
    ]
    fields = TeamEditFields(
        name=name,
        description=str(row.get("description") or ""),
        manager=str(row.get("manager") or "manager"),
        members=members,
        instructions=str(row.get("instructions") or ""),
        project=str(row.get("project") or ""),
    )
    if local_id:
        # THE UPDATE PATH IS ``update_team``, and that is not a preference: the
        # briefs of a transported copy read as "" and ``save_team`` hydrates an
        # unloaded row FROM DISK under the R2-1 rule, which would overwrite the
        # incoming briefs with the ones already here. ``update_team`` hydrates
        # first and then applies explicit values, which is exactly "set both
        # briefs to what the author wrote".
        team = registry.update_team(local_id, fields)
    else:
        created = registry.create_team(fields)
        team_id = str(row.get("id") or "")
        if team_id and team_id != created.id:
            # Keep the AUTHOR's id so both devices address one row by the same id.
            # A create mints its own, so the row is re-saved under the origin's id
            # and the minted one removed — done through the registry's own verbs,
            # never by touching ``teams/`` here.
            registry.delete_team(created.id)
            team = registry.save_team(
                Team(
                    id=team_id,
                    name=name,
                    created_date=_parse_iso(row.get("created_date")),
                    description=str(row.get("description") or ""),
                    manager=str(row.get("manager") or "manager"),
                    members=members,
                    instructions=str(row.get("instructions") or ""),
                    project=str(row.get("project") or ""),
                )
            )
        else:
            team = created
    _index_rows_mut(index, "teams")[name] = {
        "origin": origin,
        "id": str(team.id),
        "digest": incoming,
        "name": name,
        "at": time.time(),
    }
    return {
        "outcome": "updated" if current is not None else "installed",
        "extra": {"id": str(team.id)},
    }


# ---------------------------------------------------------------------------
# The receiver's answer: what do you already hold?
# ---------------------------------------------------------------------------


def definition_state(
    root: Path, *, names: Mapping[str, Iterable[str]] | None = None
) -> dict[str, Any]:
    """A digest manifest of the definitions THIS device holds.

    The sender's input: it is cheaper and far more honest than pushing a bundle
    and reading a summary, because it lets the sender see that the peer already
    holds an equal revision (idempotence without a write) and that it holds a
    DIFFERENT one (a conflict to report rather than a row to clobber).
    """
    agents = {str(row["name"]): digest_of(row) for row in _read_agents(root)}
    teams = {str(row["name"]): digest_of(row) for row in _read_teams(root)}
    if names:
        wanted_agents = {str(item).casefold() for item in (names.get("agents") or ())}
        wanted_teams = {str(item).casefold() for item in (names.get("teams") or ())}
        if wanted_agents:
            agents = {
                name: value for name, value in agents.items() if name.casefold() in wanted_agents
            }
        if wanted_teams:
            teams = {
                name: value for name, value in teams.items() if name.casefold() in wanted_teams
            }
    return {"agents": agents, "teams": teams}


# ---------------------------------------------------------------------------
# Pushing
# ---------------------------------------------------------------------------


def push_to_peer(
    server: "RelayServer", device_id: str, *, names: Mapping[str, Iterable[str]] | None = None
) -> dict[str, Any]:
    """Reconcile this device's definitions onto ``device_id``.

    Idempotent by construction: the peer's ``state`` decides what is actually
    sent, so a peer that already holds every row costs one round trip and writes
    nothing. Called from three places — the create path (before the create frame,
    so a name resolves), the local ``definitions_sync`` verb, and the cadence.
    """
    link = server._ensure_link(device_id)  # noqa: SLF001 — the one dial seam
    if link is None:
        label = server._member_name(device_id) or device_id  # noqa: SLF001 — the mesh's own name
        return {
            "ok": False,
            "code": "unreachable",
            "message": f"{label} is not answering right now, so its definitions were not synced",
            "device_id": device_id,
        }
    try:
        state = link.request(
            {
                "op": "net_definitions",
                "req": server._next_relay_req(),
                "locality": "remote",
                "phase": "state",
                **({"names": {k: sorted(v) for k, v in names.items()}} if names else {}),
            },  # noqa: SLF001
            timeout=PUSH_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 — a probe failure is reported, not raised
        return {"ok": False, "code": "state_failed", "message": str(exc), "device_id": device_id}
    if not isinstance(state, dict) or state.get("op") == "error":
        detail = state if isinstance(state, dict) else {}
        return {
            "ok": False,
            "code": str(detail.get("code") or "state_failed"),
            "message": str(
                detail.get("message") or "that device did not answer with its definitions"
            ),
            "device_id": device_id,
        }
    detail_in = state.get("detail")
    remote: dict[str, Any] = detail_in if isinstance(detail_in, dict) else {}
    agents_in = remote.get("agents")
    remote_agents: dict[str, Any] = agents_in if isinstance(agents_in, dict) else {}
    teams_in = remote.get("teams")
    remote_teams: dict[str, Any] = teams_in if isinstance(teams_in, dict) else {}

    bundle = local_bundle(server.root, names=names)
    missing_agents = [
        row for row in bundle["agents"] if remote_agents.get(str(row["name"])) != digest_of(row)
    ]
    missing_teams = [
        row for row in bundle["teams"] if remote_teams.get(str(row["name"])) != digest_of(row)
    ]
    if not missing_agents and not missing_teams:
        return {
            "ok": True,
            "code": "in_sync",
            "message": self_sentence(server) + " and this device hold the same definitions",
            "device_id": device_id,
            "pushed": False,
            "agents": {str(row["name"]): digest_of(row) for row in bundle["agents"]},
            "teams": {str(row["name"]): digest_of(row) for row in bundle["teams"]},
            "withheld": bundle.get("withheld") or [],
        }
    outbound = {
        "kind": BUNDLE_KIND,
        "version": BUNDLE_VERSION,
        "origin_device": bundle.get("origin_device") or "",
        "agents": missing_agents,
        "teams": missing_teams,
    }
    try:
        reply = link.request(
            {
                "op": "net_definitions",
                "req": server._next_relay_req(),
                "locality": "remote",
                "phase": "apply",
                "bundle": outbound,
            },  # noqa: SLF001
            timeout=PUSH_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 — see above
        return {"ok": False, "code": "apply_failed", "message": str(exc), "device_id": device_id}
    if not isinstance(reply, dict) or reply.get("op") == "error":
        detail = reply if isinstance(reply, dict) else {}
        return {
            "ok": False,
            "code": str(detail.get("code") or "apply_failed"),
            "message": str(detail.get("message") or "that device refused the definitions"),
            "device_id": device_id,
        }
    summary_in = reply.get("detail")
    summary: dict[str, Any] = summary_in if isinstance(summary_in, dict) else {}
    conflicts = list(summary.get("conflicts") or [])
    refused = list(summary.get("refused") or [])
    ok = not conflicts and not refused
    return {
        "ok": ok,
        "code": "applied" if ok else "conflict",
        "message": (
            f"sent {len(missing_agents)} agent and {len(missing_teams)} team definition(s)"
            if ok
            else "that device would not take every definition: "
            + _describe_rows(conflicts + refused)
        ),
        "device_id": device_id,
        "pushed": True,
        "installed": summary.get("installed") or [],
        "updated": summary.get("updated") or [],
        "unchanged": summary.get("unchanged") or [],
        "conflicts": conflicts,
        "refused": refused,
        "withheld": bundle.get("withheld") or [],
        # The digests of what the peer now HOLDS, which the create frame pins as
        # ``expect``: a row that moves between this push and the create (a third
        # device) then refuses the create instead of running a revision the
        # requester did not reconcile.
        "agents": {str(row["name"]): digest_of(row) for row in bundle["agents"]},
        "teams": {str(row["name"]): digest_of(row) for row in bundle["teams"]},
    }


def self_sentence(server: "RelayServer") -> str:
    """The sender's own name, for a sentence a person reads."""
    try:
        return server._own_label()  # noqa: SLF001 — the mesh's own label
    except Exception:  # noqa: BLE001 — a sentence must not raise
        return "this device"


def _describe_rows(rows: Iterable[Mapping[str, Any]]) -> str:
    parts = []
    for row in rows:
        name = str(row.get("name") or "?")
        parts.append(f"{row.get('kind') or 'row'} {name!r} ({row.get('reason') or 'refused'})")
    return "; ".join(parts) or "no reason given"


# ---------------------------------------------------------------------------
# The create path's half
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BirthModel:
    """What the peer's runtime should be CONSTRUCTED on.

    A duck-typed stand-in for the desktop's birth sample: ``launch._spawn_runtime``
    reads exactly ``provider``, ``model_id`` and an optional ``reasoning_effort``
    and puts them in the child's environment, which is the only channel that
    reaches a runtime BEFORE its first provider call — and therefore the only one
    that can make the session's first turn run on the profile's model rather than
    switching to it after construction.
    """

    provider: str
    model_id: str
    reasoning_effort: str | None = None


@dataclass(frozen=True, slots=True)
class CreateIdentity:
    """The identity a create frame names, resolved on the device that will own it."""

    agent_name: str = ""
    agent_kind: str = ""
    agent_id: str = ""
    agent_digest: str = ""
    instructions_attachable: bool = False
    team_name: str = ""
    team_id: str = ""
    team_digest: str = ""
    birth: BirthModel | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "agent": (
                {
                    "name": self.agent_name,
                    "kind": self.agent_kind,
                    "id": self.agent_id,
                    "digest": self.agent_digest,
                    # NAMED RATHER THAN IMPLIED. A legacy agent row that is neither
                    # a role nor a specialist cannot be attached to a running
                    # session, so its instructions do not reach the runtime while
                    # its routing (hosting/model) does. That half-application is
                    # reported rather than hidden: the alternative is a session
                    # that silently runs the default persona under the agent's name.
                    "instructions_attached": self.instructions_attachable,
                }
                if self.agent_name
                else None
            ),
            "team": (
                {"name": self.team_name, "id": self.team_id, "digest": self.team_digest}
                if self.team_name
                else None
            ),
        }


def resolve_create_identity(
    root: Path,
    *,
    profile: str = "",
    agent_name: str = "",
    agent_id: str = "",
    team_name: str = "",
    effort: str = "",
) -> tuple[CreateIdentity | None, str]:
    """Resolve a create's identity against THIS device's registries.

    Returns ``(identity, "")`` or ``(None, sentence)``. The sentence NAMES what is
    missing — which is the whole point of the function. A create that named an
    agent this device cannot resolve used to be indistinguishable from one that
    named nothing, and the difference the user cares about is exactly that: a
    session that runs the default agent under a profile's name is the wrong thing
    running under the right name.

    ``profile`` and ``agent_name``/``agent_id`` are the two halves of "who the
    session is" the local product has: an attachable persona (a role, a
    specialist or a packaged seed — ``/agent``'s vocabulary) and a legacy named
    agent row (``--agent NAME``'s vocabulary). Both are resolved here because both
    are carried on the frame; see ``relay._op_session_create`` for what each half
    contributes.
    """
    from local_operator.agent_profiles import resolve_profile_or_specialist

    agent_kind = ""
    resolved_name = ""
    resolved_id = ""
    attachable = False
    digest = ""
    birth: BirthModel | None = None

    if profile:
        from local_operator.agents import AgentRegistry

        registry = AgentRegistry(root)
        complete = getattr(registry, "require_complete_metadata", None)
        if complete is not None:
            complete()
        kind, row, specialist_prompt, display = resolve_profile_or_specialist(
            profile, registry=registry
        )
        if kind is None:
            return None, (
                f"no role, specialist or packaged seed named {profile!r} on this device, so the "
                "session was not created; nothing was run under that name. Send its definition "
                f"first (from the device that has it: `lop network definitions push`), or name an "
                "agent this device holds."
            )
        agent_kind = kind
        resolved_name = display
        attachable = True
        if row is not None:
            resolved_id = str(getattr(row, "agent_id", "") or "")
            hosting = str(getattr(row, "hosting", "") or "")
            model = str(getattr(row, "model", "") or "")
            if hosting or model:
                birth = BirthModel(
                    provider=hosting, model_id=model, reasoning_effort=effort or None
                )
            if resolved_id:
                digest = _agent_digest(root, resolved_name)
        elif kind == "specialist" and specialist_prompt is not None:
            digest = _agent_digest(root, resolved_name)
    elif agent_name or agent_id:
        from local_operator.agents import AgentRegistry

        registry = AgentRegistry(root)
        row = None
        if agent_id:
            try:
                row = registry.get_agent(agent_id)
            except KeyError:
                row = None
        if row is None and agent_name:
            row = registry.get_agent_by_name(agent_name)
        if row is None:
            wanted = agent_id or agent_name
            return None, (
                f"no agent named {wanted!r} on this device, so the session was not created; "
                "nothing "
                "was run under that name. Send its definition first (from the device that has it: "
                "`lop network definitions push`)."
            )
        resolved_name = str(row.name)
        resolved_id = str(row.id)
        agent_kind = "agent"
        hosting = str(getattr(row, "hosting", "") or "")
        model = str(getattr(row, "model", "") or "")
        if hosting or model:
            birth = BirthModel(provider=hosting, model_id=model, reasoning_effort=effort or None)
        digest = _agent_digest(root, resolved_name)
        # Whether the row can carry its own instructions into a running session is
        # the product's question, not this module's: ``attach_agent_profile``
        # resolves roles and specialists only, and a legacy conversational row is
        # deliberately not attachable (an agent's private chat prompt must not be
        # pulled in by a coincidental name). Asking the ONE resolver rather than
        # re-deriving the rule here is what keeps the two answers equal.
        try:
            attach_kind, _row, prompt, _display = resolve_profile_or_specialist(
                agent_name or resolved_name, registry=registry
            )
            attachable = attach_kind is not None and (
                attach_kind == "specialist" or prompt is not None
            )
        except Exception:  # noqa: BLE001 — "cannot attach" is the safe answer
            attachable = False

    team_id = ""
    team_digest = ""
    resolved_team = ""
    if team_name:
        from local_operator.teams import TeamRegistry

        try:
            team = TeamRegistry(root).get_team_by_name(team_name)
        except Exception:  # noqa: BLE001 — a damaged registry is "cannot resolve"
            team = None
        if team is None:
            return None, (
                f"no team named {team_name!r} on this device, so the session was not created; "
                "nothing was run under that name. Send its definition first (from the device that "
                "has it: `lop network definitions push`)."
            )
        resolved_team = str(team.name)
        team_id = str(team.id)
        team_digest = digest_of(_team_row_from_bundle(_team_row_from_team(team)))

    if not (profile or agent_name or agent_id or team_name):
        return (
            CreateIdentity(
                birth=BirthModel(provider="", model_id="", reasoning_effort=effort or None)
            ),
            "",
        )
    if effort and birth is not None:
        birth = BirthModel(
            provider=birth.provider, model_id=birth.model_id, reasoning_effort=effort
        )
    return (
        CreateIdentity(
            agent_name=resolved_name,
            agent_kind=agent_kind,
            agent_id=resolved_id,
            agent_digest=digest,
            instructions_attachable=attachable,
            team_name=resolved_team,
            team_id=team_id,
            team_digest=team_digest,
            birth=birth,
        ),
        "",
    )


def _agent_digest(root: Path, name: str) -> str:
    """The digest of the agent row this device currently holds, or ``""``."""
    for row in _read_agents(root):
        if str(row.get("name") or "") == name:
            return digest_of(row)
    return ""


def check_expected(root: Path, expect: Any) -> str:
    """Compare a create frame's ``expect`` with what this device holds.

    ``expect`` is ``{"agents": {name: digest}, "teams": {...}}``: the revision the
    requester reconciled a moment ago. A mismatch is refused rather than run,
    because the whole point of carrying a definition is that the user chose it —
    a third device pushing a newer row in between must not silently change what
    their session runs.

    Returns ``""`` when it agrees, or the sentence naming the disagreement.
    """
    if not isinstance(expect, Mapping):
        return ""
    held_agents = {str(row["name"]): digest_of(row) for row in _read_agents(root)}
    held_teams = {str(row["name"]): digest_of(row) for row in _read_teams(root)}
    for key, held in (("agents", held_agents), ("teams", held_teams)):
        wanted = expect.get(key)
        if not isinstance(wanted, Mapping):
            continue
        for name, digest in wanted.items():
            mine = held.get(str(name))
            if mine is None:
                return (
                    f"this device holds no {key[:-1]} named {name!r} anymore, so the session was "
                    "not created — the definition it was asked to run is not here"
                )
            if str(digest) != mine:
                return (
                    f"this device's {key[:-1]} {name!r} is a different revision from the one that "
                    "was sent, so the session was not created: creating it would have run a "
                    "definition you did not choose"
                )
    return ""


# ---------------------------------------------------------------------------
# The op
# ---------------------------------------------------------------------------


def make_handler(server: "RelayServer") -> Callable[[Any, dict[str, Any]], dict[str, Any]]:
    """The ``net_definitions`` handler for one relay.

    It never issues a request over the link it is serving (``PeerLink.request``
    refuses that, build plan §0 finding 4): both phases read and write THIS
    device's own registries and the index, and answer. The outbound direction
    lives in :func:`push_to_peer`, which runs on the requester's own threads.
    """

    def _handle(link: Any, frame: dict[str, Any]) -> dict[str, Any]:
        phase = str(frame.get("phase") or "")
        names = frame.get("names") if isinstance(frame.get("names"), Mapping) else None
        if phase == "state":
            return {"phase": "state", **definition_state(server.root, names=names)}
        if phase == "apply":
            bundle = frame.get("bundle")
            if not isinstance(bundle, Mapping):
                raise DefinitionsRefused("bad_request", "an apply needs the bundle it applies")
            from local_operator.network.audit import AuditEvent

            summary = apply_bundle(server.root, bundle, origin_device=link.device_id)
            # AUDITED BY NAME, and never by content: an install that wrote rows on
            # this device is exactly the event an incident review asks about, and
            # the detail carries names and outcomes only.
            try:
                server.audit.record(
                    AuditEvent(
                        event="definitions_applied",
                        actor=link.device_id,
                        subject=str(bundle.get("origin_device") or link.device_id),
                        network_id=link.network_id,
                        epoch=link.epoch,
                        detail={
                            "installed": [
                                str(row.get("name") or "") for row in summary.get("installed") or []
                            ],
                            "updated": [
                                str(row.get("name") or "") for row in summary.get("updated") or []
                            ],
                            "conflicts": [
                                str(row.get("name") or "") for row in summary.get("conflicts") or []
                            ],
                            "refused": [
                                str(row.get("name") or "") for row in summary.get("refused") or []
                            ],
                        },
                    )
                )
            except Exception:  # noqa: BLE001 — a record is not a gate
                logger.debug("definitions: could not record the apply", exc_info=True)
            return {"phase": "apply", **summary}
        raise DefinitionsRefused("bad_request", f"unknown definitions phase {phase!r}")

    return _handle


def local_sync_handler(server: "RelayServer") -> Any:
    """The ``definitions_sync`` local verb: push this device's definitions out.

    One peer or every paired member, which is what
    ``lop network definitions push [--peer NAME] [--all]`` calls.
    """

    def _handle(frame: dict[str, Any]) -> dict[str, Any]:
        from local_operator.network import store

        peer = str(frame.get("peer") or "")
        results: list[dict[str, Any]] = []
        targets: list[str] = []
        if peer:
            targets = [server._resolve_peer(peer)]  # noqa: SLF001 — the one name resolver
        else:
            for record in store.list_networks(server.root):
                for member in record.active_members():
                    if member.device_id != server.identity.device_id:
                        targets.append(member.device_id)
        for device_id in targets:
            results.append(push_to_peer(server, device_id))
        ok = all(bool(item.get("ok")) for item in results) if results else True
        return {
            "ok": ok,
            "peers": results,
            "message": (
                "nothing to push to: this device is in no network with another member"
                if not results
                else "; ".join(
                    f"{item.get('device_id')}: {item.get('message')}" for item in results
                )
            ),
        }

    return _handle


# ---------------------------------------------------------------------------
# The cadence
# ---------------------------------------------------------------------------

#: One syncer per relay process, keyed like the sync slice's watcher and for the
#: same reason: ``install`` runs at relay CONSTRUCTION and the suite builds
#: hundreds of relays it never starts.
_syncer_lock = threading.Lock()
_syncers: dict[int, "DefinitionsSyncer"] = {}


class DefinitionsSyncer(threading.Thread):
    """The cadence half: keep every paired member's definitions current.

    WHY A TICK AND NOT A HOOK ON THE CREATE. The create path pushes on demand, so
    a NAMED create works against a bare peer. What the tick adds is everything a
    create does not cover: a definition edited on this device reaching a peer
    before anybody creates a session there (so a peer's own ``/agent`` or list
    shows it), a peer that was unreachable at the create's own moment, and a
    device that was announced after this one had already computed its set. None
    of those has a create to hang a push on, and a design whose sync only happens
    when the user types a command is one that does not support "pair a bare node
    and run workloads" — the operator's stated direction.

    CADENCE: the session-sync slice's own ``network.sync.tick_s``, read through
    the package's one config reader. The work per tick is a bundle build plus a
    per-link ``state`` round trip; the bundle is memoised on a digest so an
    unchanged install costs the local build once and nothing over the wire when
    every peer reports it in sync. ``STATE_MIN_INTERVAL_S`` bounds how often any
    one link is asked, so a mesh of many members cannot turn the tick into a
    probe storm.
    """

    def __init__(self, server: "RelayServer") -> None:
        super().__init__(name="mesh-definitions", daemon=True)
        self._server = server
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_state_at: dict[str, float] = {}
        self._last_pushed: dict[str, str] = {}
        self._tick_s = _tick_seconds(server.root)

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:  # pragma: no cover — the tick is what tests drive
        while not self._stop.wait(self._tick_s):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 — a syncer must never kill the relay
                logger.debug("definitions: sync tick failed", exc_info=True)

    def tick(self, *, now: float | None = None) -> list[tuple[str, str]]:
        """One pass over the live member links. Returns ``(device, outcome)``.

        Driven with an injected clock rather than only by the thread, for the same
        reason ``sync.SyncWatcher.tick`` is: a test that had to wait 15 real
        seconds to observe a tick is a test nobody runs.
        """
        moment = time.time() if now is None else now
        outcomes: list[tuple[str, str]] = []
        for link in list(self._server.links.values()):
            if not link.alive or link.phase != "member":
                continue
            with self._lock:
                last = self._last_state_at.get(link.device_id, 0.0)
                if moment - last < STATE_MIN_INTERVAL_S:
                    continue
                self._last_state_at[link.device_id] = moment
            result = push_to_peer(self._server, link.device_id)
            outcomes.append((link.device_id, str(result.get("code") or "")))
            with self._lock:
                if result.get("ok"):
                    self._last_pushed[link.device_id] = str(result.get("message") or "")
        return outcomes


#: The floor between two ``state`` probes on one link. The tick itself is the
#: session-sync cadence, so this is what bounds the traffic a relay generates per
#: peer: at most one two-envelope exchange per peer per this interval, and the
#: bundle is only SENT when the peer's manifest says it is behind.
STATE_MIN_INTERVAL_S = 60.0


def _tick_seconds(root: Path) -> float:
    from local_operator.network.sync import SYNC_TICK_S, SyncSettings

    try:
        return max(1.0, float(SyncSettings.from_config(root).tick_s))
    except Exception:  # noqa: BLE001 — an unreadable setting falls back to the shipped one
        return SYNC_TICK_S


def ensure_syncer(server: "RelayServer") -> "DefinitionsSyncer":
    """Start this relay's definitions syncer if it is not already running."""
    with _syncer_lock:
        existing = _syncers.get(id(server))
        if existing is not None and existing.is_alive():
            return existing
        syncer = DefinitionsSyncer(server)
        _syncers[id(server)] = syncer
        syncer.start()
        return syncer


def install(server: "RelayServer") -> None:
    """Register this slice's ops on ``server``.

    The cadence is registered as an ``on_start`` HOOK rather than started here:
    ``install`` runs at construction, and a thread per relay in a suite that
    builds hundreds of relays it never starts would be paid for by tests that
    never asked about definitions (the rule ``register_ops`` states).
    """
    server.register_ops(
        {"net_definitions": make_handler(server)},
        local_handlers={"definitions_sync": local_sync_handler(server)},
        slow={"net_definitions": DEFINITIONS_OP_DEADLINE_S},
        on_start={"mesh-definitions": lambda: ensure_syncer(server)},
    )
