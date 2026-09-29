"""MCP SERVER DEFINITIONS over the mesh: what makes a peer's MCP surface the same one.

WHY THIS EXISTS
===============

After pairing, a peer is reachable but NOT capable, and the first deficit the
offload run measured was this one: MCP server definitions had no sync path at
all. ``net_definitions`` carried agent and team bundles only, and the readiness
report said so out loud — "server config is per device and is not copied over
the mesh" — so a freshly paired pod had NO user-scope MCP servers, and every
workload that assumed one (a GitLab server, a Linear server) failed there for a
reason nobody could see from the origin device. This module is that sync path;
the readiness row's remedy now names it (``lop network mcp push``). The forward
direction this slice must not need a refactor for is the pool's: a pod boots a
bare ``lop``, pairs,
runs workloads, spins down (``docs/design/mesh-compute-pool.md``, R21), and
receives its servers from the cadence with nobody at its end.

WHY A NEW OP RATHER THAN A SECTION INSIDE ``net_definitions``
=============================================================

Recorded because the counter-position is real (the machinery — index, lock,
digests, syncer, refusal sentences — is shared) and was rejected for reasons
that still hold:

* **Payload readers differ.** A definitions bundle feeds the CREATE path:
  ``expect`` pins, per-name refusals, a receipt a create waits on. MCP rows
  have no create to pin to, and bolting one op's pins onto the other's payload
  is how a reader starts reasoning about rows it does not install.
* **Versioning must refuse whole.** A mismatch refuses the WHOLE payload
  (definitions: half-applying one is how a device ends up with a roster that
  resolves on one machine and not the other). One bundle over two subjects
  would either gate agents on the MCP schema or need an optional key old peers
  silently ignore — false success, the class this tree keeps refusing.
* **Apply semantics differ.** Definitions install into registries; MCP rows
  merge into the user's own ``mcp.json`` beside hand-written servers, with a
  different conflict surface. A config that can spawn commands deserves its own
  named op and its own audit event.
* **Degradation is independent.** Its own capability string, its own old-peer
  story, its own cadence step.

The shared machinery is REUSED BY IMPORT (``definitions.definition_lock``,
``read_index``/``write_index``, ``canonical``, ``digest_of``,
``credential_shape``, and the syncer's own per-member step seam,
``definitions.add_tick_step``) — never copied. The provenance index gains a
``servers`` section in the same ``<config>/network/definitions.json``: one
provenance home, one lock, sections disjoint.

WHAT TRAVELS, FIELD BY FIELD
============================

A row is ``{"kind": "server", "name": ..., "transport": ...,
"raw": {...}}``. The canonical ``raw`` carries:

* ``type``, ``command``, ``args``, ``url`` — carried and passed through the
  harness's ONE shape table (``definitions.credential_shape``). A row any of
  whose carried text trips it is WITHHELD and named at the sender, REFUSED by
  name at the receiver, never installed.
* ``env`` / ``headers`` — KEYS travel; each key's VALUE is replaced by a state:
  ``ref:<NAME>`` when the value is exactly one ``${NAME}`` reference (a ${REF}
  is a NAME, not a secret — classified by ``mcp.secret_refs.public_secret_refs``,
  the one parser, never a second regex), else ``literal-held``. A literal value
  NEVER travels: the file is the user's and there is no channel that can prove
  a literal is benign. ``TRANSPORT_OWNED_HEADERS`` are never carried (the
  transport overwrites them; a receiver refuses a row that carries one anyway).
* ``auth`` — ``{"type": ...}`` only, NEVER the ``oauth`` block or any
  ``client_secret`` (that reaches the token endpoint verbatim). The receiver
  re-runs dynamic client registration on connect, which is what the shipped
  writer's own shape does; a server needing a pre-shared client secret stays a
  local setup task.
* ``enabled``, ``enabledTools``/``disabledTools``/``preloadTools``,
  ``ownTurnOnly``, ``timeout`` — run-shaping configuration, not secrets.
* ``cwd`` is EXCLUDED (a machine-local path, the same exclusion definitions
  makes for ``current_working_directory``), and the install-wide
  ``enabledServers``/``disabledServers`` lists are never touched by a mirror.

THE PLACEHOLDER RULE (what a ``literal-held`` key becomes on the receiver)
=======================================================================

A withheld literal cannot be re-invented, but the server must still be fixable
on the receiving side: the pod's operator sets the value THERE (``lop secret
set``; the value store is deliberately not brokered — mesh-credentials.md
§4.8). The mirror therefore lands a withheld key as the reference ``${<key>}``:
the product's existing resolver then refuses the connect by NAME until the
value exists ("MCP server 'x' needs KEY …"), and provisioning the key makes the
mirror work. ``state`` lists those keys.

The consequence for digests, stated here because it is what ``landed`` in the
index exists for: the row ON DISK re-reads as ``ref:<key>`` while the row the
SOURCE sent says ``literal-held``, so ``digest(row on disk)`` deliberately
differs from the incoming digest for such a key. The idempotence short-circuit
therefore compares the incoming digest with the recorded SOURCE digest AND the
row on disk with the recorded ``landed`` digest — see ``_apply_bundle_locked``.
A key whose name is not a valid reference name cannot be store-provisioned:
its placeholder re-reads as literal text, and ``state`` lists no ref for it
(only valid reference names appear there), so a local edit on the pod is the
v1 remedy.

THE CONFLICT POLICY (definitions', adopted whole)
=================================================

Per name: the provenance index records origin/source digest/landed digest/at.
*Authored wins* — a name with no index entry is never overwritten. *A mirror
follows its own origin only* — an edited mirror (its disk digest no longer
matches ``landed``) or one whose name was mirrored from another device is a
conflict reported by name. *Idempotent* — a second apply of the same bundle is
``unchanged`` and writes nothing. *A deleted mirror is re-installed* — the
sync's job is that the server works there. *Deletions are not propagated* in
v1: a delete is indistinguishable from a partial view, so removals stay an
operator act on either side, visible in ``state``. Last-writer-wins is
deliberately NOT the policy, for the same reason definitions gives.

THE OPS
=======

``net_mcp_defs`` (peer, capability ``admin``): ``state`` = per-server digest
manifest; ``apply`` = a bundle, merged into the user-scope ``mcp.json`` through
the module's own atomic writer, audited by name (``mcp_defs_applied``).
``mcp_defs_sync`` (local): push one peer or every member. ``push_to_peer``
decides from the peer's manifest and sends only missing/differing rows. An
old peer that never advertised ``wire.MCP_DEFS_V1`` is refused with the remedy
before anything is sent. The cadence RIDES the existing ``mesh-definitions``
thread — no new thread, same floors — through ``definitions.add_tick_step``;
members whose capability table (``types.OP_CAPABILITY``, the one table) cannot
hold the op are skipped with a named outcome and no wire traffic.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from local_operator.network import definitions, wire
from local_operator.network.types import MeshRefusal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.network.relay import RelayServer

logger = logging.getLogger(__name__)

#: The bundle's own name and version, checked on arrival so a payload from a
#: future or foreign producer is refused BY NAME rather than partially applied
#: (definitions' rule, same reasoning: half-applying one is how two devices end
#: up with different servers under the same name).
BUNDLE_KIND = "lop.mesh.mcp.v1"
BUNDLE_VERSION = 1

#: The provenance section inside ``<config>/network/definitions.json``. One
#: index file, one lock, disjoint sections: definitions owns agents/teams, this
#: module owns servers, and neither reads the other's rows.
INDEX_SECTION = "servers"

#: Caps in the definitions style, smaller numbers: a server row is heavier than
#: an agent row (args/env/headers), and a peer's whole MCP surface is smaller
#: than its roster. The row cap is applied on both ends; the byte cap refuses
#: the whole payload rather than sending a truncated one.
MAX_MCP_ROWS = 200
MAX_MCP_BUNDLE_BYTES = 1024 * 1024

#: Owner-side deadline for ``net_mcp_defs``: an apply is disk work over up to
#: :data:`MAX_MCP_ROWS` rows, so it runs OFF the link's reader with a budget
#: that outlasts a slow disk — the same reasoning ``net_definitions`` gives.
MCP_DEFS_OP_DEADLINE_S = 60.0

#: How long one push may take from the requester's side, the peer's own
#: deadline plus margin (definitions' PUSH_TIMEOUT_S reasoning).
PUSH_TIMEOUT_S = 30.0


# ---------------------------------------------------------------------------
# Paths and the user-scope document
# ---------------------------------------------------------------------------


def _global_path(root: Path) -> Path:
    """The mirror target for ONE root: ``<config root>/mcp.json``.

    The root PARAMETER, never the ambient config dir: every function here acts
    for a root it is given (the relay's ``server.root``, or a two-install rig's
    synthetic root), and resolving this ambiently made an isolated rig read and
    write the PROCESS's own install — the exact escape ``_scope_path``'s
    docstring records for writers, reappearing through a reader. ``<root>/mcp.json``
    IS the user scope (``_scope_path(None, "global")`` when the root is the
    config dir), and ``readiness.mcp_servers_fact`` reads the same file the same
    way for the same reason.
    """
    return root / "mcp.json"


def _assert_inside_root(path: Path, root: Path) -> None:
    """Refuse to write anything that resolves outside the root being served.

    THE FAILURE THIS PREVENTS IS UNRECOVERABLE, and it happened: a path
    resolver that ignored its root argument aimed a scratch run's apply at the
    live install's ``mcp.json``. Every write this module makes therefore
    asserts its target against the root it was given FIRST, and fails hard —
    "write nothing" is the correct outcome on the one path that is about to
    write. Resolved on both sides so a symlinked root or ``/private/var``
    spelling cannot fake containment.
    """
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        raise RuntimeError(
            f"refusing to write {path}: it is outside the root this device serves ({root})"
        ) from None


def _read_document(root: Path) -> tuple[dict[str, Any] | None, str]:
    """The user-scope document and a problem sentence, or ``({}, "")`` when absent.

    A MISSING file is a definite empty document (the writers create it on first
    write). A file that EXISTS but cannot be read is NOT treated as empty: an
    apply that clobbered an unreadable-but-present config would destroy a
    hand-edited file's contents on the strength of a parse error, which is the
    one destructive shape this module must never take.
    """
    path = _global_path(root)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}, ""
    except OSError as exc:
        return None, f"it could not be read ({exc.__class__.__name__})"
    try:
        document = json.loads(text)
    except ValueError:
        return None, "it is not valid JSON"
    if not isinstance(document, dict):
        return None, "it is not a JSON object"
    return document, ""


def _entries_of(document: Mapping[str, Any] | None) -> dict[str, Any]:
    """The ``mcpServers`` map of a document, or ``{}`` for anything unusable."""
    entries = document.get("mcpServers") if isinstance(document, Mapping) else None
    return dict(entries) if isinstance(entries, Mapping) else {}


# ---------------------------------------------------------------------------
# Reading one entry into the canonical row
# ---------------------------------------------------------------------------


def _refs_by_binding(
    env: Mapping[str, Any], headers: Mapping[str, Any]
) -> dict[tuple[str, str], list[str]]:
    """``(field, key) -> [reference names]``, from the ONE secret-refs parser.

    ``public_secret_refs`` reads only ``env``/``headers`` off its argument (its
    documented object contract), and it reports exactly the fragments that are
    well-formed references — ``$${X}`` escapes and non-name inners are not
    references and never appear here. A namespace is passed rather than a typed
    config because this must also work for hand-written entries the models
    would reject elsewhere; the parse that matters is the reference parse.
    """
    from local_operator.mcp.secret_refs import public_secret_refs

    refs = public_secret_refs(SimpleNamespace(env=dict(env), headers=dict(headers)))
    binds: dict[tuple[str, str], list[str]] = {}
    for ref in refs:
        name = str(ref.get("id") or "")
        for binding in ref.get("bindings") or []:
            key = (str(binding.get("field") or ""), str(binding.get("key") or ""))
            if name not in binds.setdefault(key, []):
                binds[key].append(name)
    return binds


def environment_state(
    field: str, key: str, value: Any, refs: Mapping[tuple[str, str], Iterable[str]]
) -> str:
    """One env/header value's state: ``ref:<NAME>`` or ``literal-held``.

    ``ref:`` requires the value to be EXACTLY one reference — nothing before or
    after it. ``Bearer ${TOKEN}`` is deliberately NOT carried: the text around
    a reference is text the parser cannot vouch for, and a value like
    ``sk-live-…${SUFFIX}`` must not smuggle a literal through the one channel
    that exists because literals cannot be proven benign. Partial-reference
    values are ``literal-held`` (and land as the key's placeholder, see the
    module docstring).
    """
    text = "" if value is None else str(value)
    names = list(refs.get((field, key), ()))
    if len(names) == 1 and text == "${" + names[0] + "}":
        return "ref:" + names[0]
    return "literal-held"


def _bounded_text(value: Any, limit: int) -> str:
    return str(value if value is not None else "")[:limit]


def _string_list(value: Any) -> list[str] | None:
    """A carried string list, normalised: ``None`` when empty or unusable.

    Empty and absent must have ONE canonical spelling: two ends that disagreed
    (one writing ``[]``, one omitting the key) would make every row look edited
    locally on one of the devices.
    """
    if not isinstance(value, list):
        return None
    out = [str(item) for item in value if item is not None]
    return out or None


def _carried_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _carried_bool(value: Any) -> bool | None:
    return bool(value) if isinstance(value, bool) else None


def _fields_from_file(raw: Mapping[str, Any], transport: str) -> dict[str, Any]:
    """The carried fields of one FILE entry, normalised and classified."""
    if transport == "stdio":
        env_in = raw.get("env")
        env_in = env_in if isinstance(env_in, Mapping) else {}
        env = {str(key): str(value) for key, value in env_in.items()}
        args_raw = raw.get("args")
        args_in = args_raw if isinstance(args_raw, list) else []
        refs = _refs_by_binding(env, {})
        fields: dict[str, Any] = {
            # Bounded the SAME way the wire-side rebuild bounds them: an
            # asymmetric limit would make a long row's two digests disagree and
            # turn every push into a conflict.
            "command": _bounded_text(raw.get("command"), 4096),
            "args": [_bounded_text(item, 4096) for item in args_in],
            "env": {key: environment_state("env", key, value, refs) for key, value in env.items()},
        }
    else:
        headers_in = raw.get("headers")
        headers_in = headers_in if isinstance(headers_in, Mapping) else {}
        headers = {
            str(key): str(value)
            for key, value in headers_in.items()
            if str(key).lower() not in _transport_owned_headers()
        }
        refs = _refs_by_binding({}, headers)
        fields = {
            "url": _bounded_text(raw.get("url"), 4096),
            "headers": {
                key: environment_state("headers", key, value, refs)
                for key, value in headers.items()
            },
        }
    _carry_run_shaping(fields, raw)
    return fields


def _carry_run_shaping(fields: dict[str, Any], raw: Mapping[str, Any]) -> None:
    """The run-shaping fields, from either spelling a hand-written file may use."""
    fields["enabled"] = _carried_bool(raw.get("enabled"))
    fields["timeout"] = _carried_number(raw.get("timeout"))
    for key in ("enabledTools", "disabledTools", "preloadTools"):
        value = raw.get(key, raw.get(_snake(key)))
        fields[key] = _string_list(value)
    own_turn = raw.get("ownTurnOnly", raw.get("own_turn_only"))
    fields["ownTurnOnly"] = _carried_bool(own_turn)
    fields["auth_type"] = _auth_type(raw)


def _snake(name: str) -> str:
    return "".join(f"_{char.lower()}" if char.isupper() else char for char in name)


def _auth_type(raw: Mapping[str, Any]) -> str | None:
    """``auth.type`` only, from either the ``auth`` or the ``oauth`` block.

    NEVER the block itself: ``oauth.client_secret`` reaches the token endpoint
    verbatim and must not travel, and the receiver re-runs dynamic client
    registration anyway. An unknown declared type is dropped rather than
    carried: the receiver validates what it is asked to write, and carrying a
    type this build's schema does not know would make every mirror a refusal.
    """
    auth = raw.get("auth")
    if isinstance(auth, Mapping):
        declared = auth.get("type")
        if declared is None:
            return "oauth"
        return str(declared) if str(declared) in ("oauth", "apikey") else None
    if isinstance(raw.get("oauth"), Mapping):
        return "oauth"
    return None


def _transport_owned_headers() -> frozenset[str]:
    from local_operator.mcp import config as mcp_config

    return mcp_config.TRANSPORT_OWNED_HEADERS


def _assemble_row(name: str, transport: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    """The ONE canonical row shape, built identically on both ends.

    Both the file-side builder and the wire-side rebuilder go through here, so
    the digest the sender records and the digest the receiver computes are over
    the same key set even when one side's input omitted an optional key.
    """
    raw: dict[str, Any] = {"type": transport}
    if transport == "stdio":
        raw["command"] = str(fields.get("command") or "")
        raw["args"] = list(fields.get("args") or [])
        raw["env"] = {str(key): str(value) for key, value in (fields.get("env") or {}).items()}
    else:
        raw["url"] = str(fields.get("url") or "")
        raw["headers"] = {
            str(key): str(value) for key, value in (fields.get("headers") or {}).items()
        }
    for key in (
        "enabled",
        "timeout",
        "enabledTools",
        "disabledTools",
        "preloadTools",
        "ownTurnOnly",
    ):
        if fields.get(key) is not None:
            raw[key] = fields[key]
    auth_type = fields.get("auth_type")
    if auth_type:
        raw["auth"] = {"type": str(auth_type)}
    return {"kind": "server", "name": name, "transport": transport, "raw": raw}


def server_row(name: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    """One FILE entry as the canonical row both ends hash and install."""
    from local_operator.mcp import config as mcp_config

    transport = mcp_config.infer_transport(raw)
    return _assemble_row(str(name), transport, _fields_from_file(raw, transport))


def server_row_from_bundle(payload: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """The row a bundle will install/report, or ``(None, reason)``.

    The RECEIVER's gate. Everything is re-derived rather than trusted: state
    strings are re-validated with the same reference parser the sender used
    (``ref:<NAME>`` must re-classify as exactly that reference; anything else
    is refused, because it could not have been produced by a compliant sender
    and would not round-trip through the on-disk digest), and the transport
    must be one this build knows. A row this build cannot rebuild is refused
    BY NAME rather than installed half-formed.
    """
    transport = str(payload.get("transport") or "")
    if transport not in ("stdio", "http", "sse"):
        return None, f"its transport {transport!r} is not one this build knows"
    raw_in = payload.get("raw")
    raw_in = raw_in if isinstance(raw_in, Mapping) else {}
    fields: dict[str, Any] = {}
    if transport == "stdio":
        env_in = raw_in.get("env")
        env_in = env_in if isinstance(env_in, Mapping) else {}
        fields["command"] = _bounded_text(raw_in.get("command"), 4096)
        args_in = raw_in.get("args")
        fields["args"] = (
            [_bounded_text(item, 4096) for item in args_in] if isinstance(args_in, list) else []
        )
        env: dict[str, str] = {}
        for key, value in env_in.items():
            state = _wire_state("env", str(key), value)
            if state is None:
                return None, (
                    f"its env entry {str(key)!r} is not a reference state this build writes"
                )
            env[str(key)] = state
        fields["env"] = env
    else:
        fields["url"] = _bounded_text(raw_in.get("url"), 4096)
        headers_in = raw_in.get("headers")
        headers_in = headers_in if isinstance(headers_in, Mapping) else {}
        headers: dict[str, str] = {}
        for key, value in headers_in.items():
            if str(key).lower() in _transport_owned_headers():
                # The transport overwrites or breaks on these; a compliant
                # sender never carries one, so a row that does is foreign or
                # older and is refused rather than landing a broken header.
                return None, f"its header {str(key)!r} is one the transport owns"
            state = _wire_state("headers", str(key), value)
            if state is None:
                return None, (f"its header {str(key)!r} is not a reference state this build writes")
            headers[str(key)] = state
        fields["headers"] = headers
    fields["enabled"] = _carried_bool(raw_in.get("enabled")) if "enabled" in raw_in else None
    fields["timeout"] = _carried_number(raw_in.get("timeout")) if "timeout" in raw_in else None
    for key in ("enabledTools", "disabledTools", "preloadTools"):
        fields[key] = _string_list(raw_in.get(key)) if key in raw_in else None
    fields["ownTurnOnly"] = (
        _carried_bool(raw_in.get("ownTurnOnly")) if "ownTurnOnly" in raw_in else None
    )
    auth = raw_in.get("auth")
    auth_type = None
    if isinstance(auth, Mapping):
        auth_type = str(auth.get("type") or "")
        if auth_type not in ("oauth", "apikey"):
            return None, f"its auth type {auth_type!r} is not one this build writes"
    fields["auth_type"] = auth_type
    return _assemble_row(str(payload.get("name") or ""), transport, fields), ""


def _wire_state(field: str, key: str, value: Any) -> str | None:
    """A wire value's state, or ``None`` when the parser will not back it.

    ``literal-held`` is a literal token. ``ref:<NAME>`` is validated by
    RE-CLASSIFYING the reference it would materialise (``${NAME}``) with the
    same one parser — a name the parser would not publish (wrong characters,
    surrounding text, an escape) is refused here rather than written into a
    file where it would not round-trip.
    """
    if not isinstance(value, str):
        return None
    if value == "literal-held":
        return value
    if value.startswith("ref:"):
        name = value[4:]
        refs = _refs_by_binding(
            {"probe": "${" + name + "}"} if field == "env" else {},
            {"probe": "${" + name + "}"} if field == "headers" else {},
        )
        state = environment_state(field, "probe", "${" + name + "}", refs)
        if state == "ref:" + name:
            return value
    return None


def _materialize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """The canonical row as the FILE entry a mirror lands (one write shape)."""
    raw_in = row.get("raw")
    raw = raw_in if isinstance(raw_in, Mapping) else {}
    transport = str(row.get("transport") or "stdio")
    out: dict[str, Any] = {"type": transport}
    if transport == "stdio":
        out["command"] = str(raw.get("command") or "")
        args_in = raw.get("args")
        args = [str(item) for item in args_in] if isinstance(args_in, list) else []
        if args:
            out["args"] = args
        env_in = raw.get("env")
        env = env_in if isinstance(env_in, Mapping) else {}
        if env:
            out["env"] = {
                str(key): _materialize_state(str(key), str(state)) for key, state in env.items()
            }
    else:
        out["url"] = str(raw.get("url") or "")
        headers_in = raw.get("headers")
        headers = headers_in if isinstance(headers_in, Mapping) else {}
        if headers:
            out["headers"] = {
                str(key): _materialize_state(str(key), str(state)) for key, state in headers.items()
            }
    for key in (
        "enabled",
        "timeout",
        "enabledTools",
        "disabledTools",
        "preloadTools",
        "ownTurnOnly",
    ):
        if key in raw:
            out[key] = raw[key]
    auth = raw.get("auth")
    if isinstance(auth, Mapping) and auth.get("type"):
        out["auth"] = {"type": str(auth["type"])}
    return out


def _materialize_state(key: str, state: str) -> str:
    """One key's state as the value the mirror holds.

    ``ref:<NAME>`` lands the reference itself. ``literal-held`` lands the key's
    own name as a reference placeholder — the pod's ``lop secret set <key>`` is
    then the provisioning act, and until it runs the resolver refuses the
    connect BY NAME (the module docstring states the full rule).
    """
    if state.startswith("ref:"):
        return "${" + state[4:] + "}"
    return "${" + key + "}"


# ---------------------------------------------------------------------------
# The credential scan and the bundle
# ---------------------------------------------------------------------------


def _flatten_text(value: Any) -> list[str]:
    """Every string inside ``value``, whatever its nesting (args are lists)."""
    if isinstance(value, Mapping):
        out: list[str] = []
        for item in value.values():
            out.extend(_flatten_text(item))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_flatten_text(item))
        return out
    if value is None:
        return []
    return [str(value)]


def row_texts(row: Mapping[str, Any]) -> list[str]:
    """Everything one row CARRIES, as strings: the credential scan's whole input.

    Derived from the canonical row (not from a hand-written field list), so a
    field added to the payload is scanned the moment it travels — the lesson
    ``definitions._row_texts`` records. The state strings are scanned too: a
    reference NAME that is itself credential-shaped is as much a leak as a
    literal, and the withholding rule is "the row does not travel".
    """
    out = [str(row.get("name") or ""), str(row.get("transport") or "")]
    raw_in = row.get("raw")
    raw = raw_in if isinstance(raw_in, Mapping) else {}
    for key in ("command", "args", "url"):
        out.extend(_flatten_text(raw.get(key)))
    for field in ("env", "headers"):
        value = raw.get(field)
        if isinstance(value, Mapping):
            for key, state in value.items():
                out.extend(_flatten_text(key))
                out.extend(_flatten_text(state))
    auth = raw.get("auth")
    if isinstance(auth, Mapping):
        out.extend(_flatten_text(auth.get("type")))
    return out


def withheld_label(row: Mapping[str, Any]) -> str:
    """The shape-table label that keeps this row off the wire, or ``""``."""
    for text in row_texts(row):
        label = definitions.credential_shape(text)
        if label:
            return label
    return ""


def shape_likeness(label: str) -> str:
    """The article-correct "looks like …" phrase for a shape-table label.

    Several of the table's labels are vowel-initial ("authorization-bearer",
    "openssl-pass-phrase", …), and "looks like a authorization-bearer" was the
    register defect design round 1 named (D5). ONE spelling for the two
    surfaces that name a withheld row: the push receipt and the state marker.
    """
    text = label or "credential"
    article = "an" if text[:1].lower() in "aeiou" else "a"
    return f"looks like {article} {text}"


def _recorded(index: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    rows = index.get(INDEX_SECTION)
    entry = rows.get(name) if isinstance(rows, Mapping) else None
    return entry if isinstance(entry, Mapping) else None


def _authored_locally(root: Path, name: str, row: Mapping[str, Any]) -> bool:
    """Is this row the OPERATOR's, rather than a mirror of somebody else's?

    The question the conflict policy turns on, and the reason the bundle
    EXCLUDES mirrors: a device that re-sent rows it had mirrored would hand
    them to a third device as if it had authored them, which loses the
    authorship the policy rests on and lets two peers echo one row forever.
    """
    recorded = _recorded(definitions.read_index(root), name)
    if recorded is None:
        return True
    return str(recorded.get("digest") or "") != definitions.digest_of(row)


def local_rows(root: Path) -> list[dict[str, Any]]:
    """This device's AUTHORED user-scope servers, as canonical rows."""
    document, _problem = _read_document(root)
    out: list[dict[str, Any]] = []
    for name, raw in sorted(_entries_of(document).items()):
        if not isinstance(raw, Mapping):
            continue
        row = server_row(str(name), raw)
        if _authored_locally(root, str(name), row):
            out.append(row)
    return out


def _size_refusal(size: int) -> str:
    return (
        f"the rows this device would send are {size} bytes, over the "
        f"{MAX_MCP_BUNDLE_BYTES}-byte limit, so nothing was sent; remove or shorten "
        "server rows on this device and push again"
    )


def _checked_size(rows: Iterable[Mapping[str, Any]]) -> int:
    return len(definitions.canonical(list(rows)).encode("utf-8"))


def local_bundle(root: Path) -> dict[str, Any]:
    """This device's authored servers as one versioned, non-credential payload."""
    from local_operator.network import identity as identity_mod

    device_id = ""
    try:
        device_id = identity_mod.load_or_mint(root).device_id
    except Exception:  # noqa: BLE001 — an unnamed origin is still a usable bundle
        logger.debug("mcpdefs: no identity for the bundle origin", exc_info=True)
    bundle: dict[str, Any] = {
        "kind": BUNDLE_KIND,
        "version": BUNDLE_VERSION,
        "origin_device": device_id,
        "servers": [],
        "withheld": [],
    }
    for row in local_rows(root)[:MAX_MCP_ROWS]:
        label = withheld_label(row)
        if label:
            bundle["withheld"].append(
                {"kind": "server", "name": str(row.get("name") or ""), "shape": label}
            )
            continue
        bundle["servers"].append(row)
    size = _checked_size(bundle["servers"])
    if size > MAX_MCP_BUNDLE_BYTES:
        raise MeshRefusal("bundle_too_large", _size_refusal(size))
    bundle["digest"] = hashlib.sha256(
        definitions.canonical(
            {key: value for key, value in bundle.items() if key != "digest"}
        ).encode("utf-8")
    ).hexdigest()
    return bundle


# ---------------------------------------------------------------------------
# The receiver's manifest
# ---------------------------------------------------------------------------


def server_state(root: Path) -> dict[str, Any]:
    """A digest manifest of the user-scope servers THIS device holds.

    The sender's input: it sees that a peer already holds an equal revision
    (idempotence without a write) or a different one (a conflict to report
    rather than a row to clobber). Both ends canonicalise with
    :func:`server_row`, so the digests are comparable.

    ONE DIVERGENCE FROM ``definitions.definition_state``, and it is load-bearing:
    an UNEDITED MIRROR reports the digest of the row ITS ORIGIN sent, not of the
    bytes on disk. The two are usually the same; they differ exactly where a held
    value is involved, because the mirror materialises ``literal-held`` as a
    placeholder ref and therefore cannot hash back to its source form. Without
    this, a server with one held key would re-offer itself on EVERY push — the
    sender would never see its own revision arrive, and risk 3's "one state round
    trip when in sync" would be false forever for such a row. The index's
    ``landed`` digest is what proves the mirror is unedited; a hand-edited mirror
    (or one whose index record is gone) falls back to the disk digest and the
    conflict path reports it by name. An unreadable file reads as no servers.
    """
    document, _problem = _read_document(root)
    recorded = definitions.read_index(root).get(INDEX_SECTION)
    recorded = recorded if isinstance(recorded, Mapping) else {}
    servers: dict[str, str] = {}
    for name, raw in sorted(_entries_of(document).items()):
        if not isinstance(raw, Mapping):
            continue
        row_digest = definitions.digest_of(server_row(str(name), raw))
        record = recorded.get(str(name))
        if isinstance(record, Mapping):
            landed = str(record.get("landed") or "")
            source = str(record.get("digest") or "")
            if landed and source and row_digest == landed:
                servers[str(name)] = source
                continue
        servers[str(name)] = row_digest
    return {"servers": servers}


def state_rows(root: Path) -> list[dict[str, Any]]:
    """This device's servers, for ``lop network mcp state`` and the tool.

    One row per user-scope entry: name, transport, where it came from (the
    provenance index's origin, ``""`` for authored), its digest, the
    references it declares with whether the local store can resolve each —
    "the keys to set" (design §1.3) — and ``withheld``, the shape-table label
    that will keep the row off the wire at push time (``""`` when it travels).
    The withheld scan is the SAME one the bundle runs over the same canonical
    row, so the local inventory names a row that will never reach a peer
    BEFORE a push discovers it (design round 1, D4). A reference whose presence
    cannot be decided (no store, unreadable store) reports ``set: None``, never
    a false "does not exist": an absent answer is not a negative one.
    """
    document, _problem = _read_document(root)
    index = definitions.read_index(root)
    rows: list[dict[str, Any]] = []
    presence: dict[str, bool | None] = {}
    for name, raw in sorted(_entries_of(document).items()):
        if not isinstance(raw, Mapping):
            continue
        row = server_row(str(name), raw)
        recorded = _recorded(index, str(name))
        refs: list[dict[str, Any]] = []
        env_in = raw.get("env")
        env = env_in if isinstance(env_in, Mapping) else {}
        headers_in = raw.get("headers")
        headers = headers_in if isinstance(headers_in, Mapping) else {}
        for ref in _public_refs(env, headers):
            ref_id = str(ref.get("id") or "")
            if ref_id not in presence:
                presence[ref_id] = _reference_present(root, ref_id)
            refs.append({"id": ref_id, "set": presence[ref_id]})
        rows.append(
            {
                "name": str(name),
                "transport": str(row.get("transport") or ""),
                "origin": str(recorded.get("origin") or "") if recorded is not None else "",
                "digest": definitions.digest_of(row),
                "refs": refs,
                "withheld": withheld_label(row),
            }
        )
    return rows


def _public_refs(env: Mapping[str, Any], headers: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The reference names a FILE entry declares, via the one parser.

    Values are string-coerced first: a hand-written entry may hold a number
    where a reference is expected, and the parser's scan is over text.
    """
    from local_operator.mcp.secret_refs import public_secret_refs

    return public_secret_refs(
        SimpleNamespace(
            env={str(key): str(value) for key, value in env.items()},
            headers={str(key): str(value) for key, value in headers.items()},
        )
    )


def _reference_present(root: Path, name: str) -> bool | None:
    """Is ``name`` resolvable from this root's encrypted secret store?"""
    try:
        from local_operator.secrets import access
        from local_operator.secrets.errors import SecretNotFound
        from local_operator.secrets.keys import store_path

        if not store_path(root).exists():
            # A store that was never created is a definite "not there" — the
            # resolver's own rule; opening one with ``create=True`` here would
            # make a read verb a writer.
            return False
        store = access.open_store(root)
        try:
            store.describe(name)
            return True
        except SecretNotFound:
            return False
    except Exception:  # noqa: BLE001 — an unreadable store is "not known"
        logger.debug("mcpdefs: could not read the secret store", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Applying a bundle
# ---------------------------------------------------------------------------


def bundle_rows(bundle: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(rows, refused)`` from a bundle, each row rebuilt and re-checked.

    Raises for a bundle-level mismatch (kind or version), which refuses the
    WHOLE payload rather than applying the half this build understands. Rows
    this build cannot rebuild are returned as refused rows, named, so one bad
    row cannot hide the rest.
    """
    if str(bundle.get("kind") or "") != BUNDLE_KIND:
        raise MeshRefusal(
            "unknown_bundle",
            f"that payload is {bundle.get('kind')!r}, not {BUNDLE_KIND}, so nothing was installed",
        )
    try:
        version = int(bundle.get("version") or 0)
    except (TypeError, ValueError):
        version = 0
    if version != BUNDLE_VERSION:
        raise MeshRefusal(
            "unknown_bundle_version",
            f"that payload is version {version} of {BUNDLE_KIND} and this build understands "
            f"version {BUNDLE_VERSION}, so nothing was installed",
        )
    servers_in = bundle.get("servers")
    rows: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for payload in (servers_in if isinstance(servers_in, list) else [])[: MAX_MCP_ROWS + 1]:
        if not isinstance(payload, Mapping):
            continue
        name = str(payload.get("name") or "")
        row, problem = server_row_from_bundle(payload)
        if row is None:
            refused.append({"kind": "server", "name": name, "reason": problem})
            continue
        rows.append(row)
    return rows[:MAX_MCP_ROWS], refused


def apply_bundle(root: Path, bundle: Mapping[str, Any], *, origin_device: str) -> dict[str, Any]:
    """Install what this device does not have; report what it will not touch.

    ONE APPLY HOLDS THIS ROOT'S DEFINITION LOCK for its whole span — the index
    read, every row decision, the document write and the index write-back —
    through the definitions module's own re-entrant lock, because two peers'
    applies interleave otherwise and both ends of a read-modify-write can lose
    (the measured race ``definitions.definition_lock`` records).
    """
    with definitions.definition_lock(root):
        return _apply_bundle_locked(root, bundle, origin_device=origin_device)


def _apply_bundle_locked(
    root: Path, bundle: Mapping[str, Any], *, origin_device: str
) -> dict[str, Any]:
    """``apply_bundle``'s body. The caller holds this root's definition lock."""
    from local_operator.mcp import config as mcp_config  # SERVER_NAME_RE: the name rule

    rows, refused = bundle_rows(bundle)
    summary: dict[str, Any] = {
        "installed": [],
        "updated": [],
        "unchanged": [],
        "conflicts": [],
        "refused": list(refused),
        "digests": {},
    }
    if not rows and not refused:
        return summary
    document, problem = _read_document(root)
    if document is None:
        raise MeshRefusal(
            "unreadable_config",
            f"this device's mcp.json refused the sync because {problem}; nothing was "
            "installed — fix or move that file and push again",
        )
    origin = str(origin_device or bundle.get("origin_device") or "")
    index = definitions.read_index(root)
    servers = document.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise MeshRefusal(
            "unreadable_config",
            "this device's mcp.json has an mcpServers entry that is not an object; nothing "
            "was installed",
        )
    changed = False
    for row in rows:
        name = str(row.get("name") or "")
        if not name:
            summary["refused"].append({"kind": "server", "name": "", "reason": "unnamed"})
            continue
        if not mcp_config.SERVER_NAME_RE.match(name):
            summary["refused"].append(
                {
                    "kind": "server",
                    "name": name,
                    "reason": "its name is not one this device's writers accept",
                }
            )
            continue
        label = withheld_label(row)
        if label:
            # The receiver's half of the credential assertion: a compliant
            # sender withholds these, so one arriving is a foreign or older
            # producer — refused by name, never installed.
            summary["refused"].append(
                {
                    "kind": "server",
                    "name": name,
                    "reason": (
                        f"its text is shaped like a credential ({label}), which is never "
                        "installed"
                    ),
                }
            )
            continue
        try:
            outcome, extra = _apply_row(root, row, origin, index, servers)
        except Exception as exc:  # noqa: BLE001 — one bad row must not stop the rest
            logger.debug("mcpdefs: could not apply server %s", name, exc_info=True)
            summary["refused"].append({"kind": "server", "name": name, "reason": str(exc)})
            continue
        summary[outcome].append({"kind": "server", "name": name, **extra})
        if outcome in ("installed", "updated", "unchanged"):
            summary["digests"][name] = definitions.digest_of(row)
            if outcome != "unchanged":
                changed = True
    if changed:
        _write_document(root, document)
    if any(summary[key] for key in ("installed", "updated")):
        _assert_inside_root(definitions.index_path(root), root)
        definitions.write_index(root, index)
    return summary


def _write_document(root: Path, document: dict[str, Any]) -> None:
    """The module's ONE ``mcp.json`` write: containment first, then the product writer.

    A single seam so the containment assertion cannot be forgotten at a call site
    added later, and so "a second apply wrote nothing" is pinned by watching one
    function rather than by trusting every branch.
    """
    from local_operator.mcp import config as mcp_config

    path = _global_path(root)
    _assert_inside_root(path, root)
    mcp_config.write_scope_document(path, document)


def _apply_row(
    root: Path,
    row: Mapping[str, Any],
    origin: str,
    index: dict[str, Any],
    servers: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """One row's conflict-policy decision and, when it lands, its write.

    Returns ``(bucket, extra)`` where bucket is one of ``installed``,
    ``updated``, ``unchanged`` or ``conflicts``. The caller holds the lock and
    the document; this function only decides and mutates both.
    """
    name = str(row.get("name") or "")
    incoming = definitions.digest_of(row)
    recorded = _recorded(index, name)
    current_raw = servers.get(name)
    current = server_row(name, current_raw) if isinstance(current_raw, Mapping) else None
    current_digest = definitions.digest_of(current) if current is not None else ""

    # THE IDEMPOTENCE SHORT-CIRCUIT, RE-HASHING THE ROW ON DISK. Comparing the
    # incoming digest with what this device RECORDED is not enough: an operator
    # may have edited the mirrored row since it was installed (the recorded
    # digest still equals the sender's), and returning "unchanged" for that
    # would report a sync that is not one. ``landed`` is the digest of the row
    # THIS DEVICE wrote last — different from the source digest whenever a
    # ``literal-held`` key was materialised as its placeholder (the module
    # docstring states why), which is exactly why both are recorded.
    if (
        recorded is not None
        and str(recorded.get("digest") or "") == incoming
        and current is not None
        and current_digest == str(recorded.get("landed") or "")
    ):
        return "unchanged", {}

    if current is not None:
        if recorded is None:
            return "conflicts", {
                "reason": (
                    "this device authored an MCP server by that name, so it was not " "overwritten"
                ),
                "origin": origin,
            }
        if str(recorded.get("origin") or "") != origin:
            return "conflicts", {
                "reason": (
                    f"that name here is a copy of {str(recorded.get('origin') or '')}'s, so "
                    f"{origin or 'the sender'}'s row was not installed"
                ),
                "origin": str(recorded.get("origin") or ""),
            }
        if current_digest != str(recorded.get("landed") or ""):
            return "conflicts", {
                "reason": "the copy of that name here has local edits, so it was not overwritten"
            }

    # A DELETED MIRROR IS RE-INSTALLED (definitions' asymmetry, same reason):
    # the deletion removed this device's copy of a row whose owner is the
    # sending device, and the sync's whole job is that the server works there.
    materialized = _materialize_row(row)
    servers[name] = materialized
    landed_row = server_row(name, materialized)
    index.setdefault(INDEX_SECTION, {})[name] = {
        "origin": origin,
        "digest": incoming,
        "landed": definitions.digest_of(landed_row),
        "name": name,
        "at": _now(),
    }
    return ("updated" if current is not None else "installed"), {}


def _now() -> float:
    import time

    return time.time()


# ---------------------------------------------------------------------------
# Pushing
# ---------------------------------------------------------------------------


def push_to_peer(server: "RelayServer", device_id: str) -> dict[str, Any]:
    """Reconcile this device's MCP servers onto ``device_id``.

    Idempotent by construction: the peer's ``state`` decides what is sent, so a
    peer that already holds every row costs one round trip and writes nothing.
    Called from the local ``mcp_defs_sync`` verb and the cadence.
    """
    link = server._ensure_link(device_id)  # noqa: SLF001 — the one dial seam
    label = server._member_name(device_id) or device_id  # noqa: SLF001 — the mesh's own name
    if link is None:
        return {
            "ok": False,
            "code": "unreachable",
            "message": f"{label} is not answering right now, so its MCP servers were not synced",
            "device_id": device_id,
        }
    if wire.MCP_DEFS_V1 not in link.capabilities:
        # BOTH SIDES MUST ADVERTISE IT. An old peer never did, and asking anyway
        # would spend a refusal sentence it had to compose for an op it does not
        # have — the peer gets its remedy instead.
        return {
            "ok": False,
            "code": "peer_too_old",
            "message": (f"{label} predates MCP server sync; run `lop-update` there, then re-check"),
            "device_id": device_id,
        }
    try:
        state = link.request(
            {
                "op": "net_mcp_defs",
                "req": server._next_relay_req(),  # noqa: SLF001 — the one req counter
                "locality": "remote",
                "phase": "state",
            },
            timeout=PUSH_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 — a probe failure is reported, not raised
        return {"ok": False, "code": "state_failed", "message": str(exc), "device_id": device_id}
    if not isinstance(state, dict) or state.get("op") == "error":
        detail = state if isinstance(state, dict) else {}
        # A PEER THAT ANSWERED IS NOT A PEER THAT DID NOT (definitions' Q-R1-3):
        # the refusal's own code crosses when it carried one, so a policy "no"
        # is not reported as a transport failure.
        return {
            "ok": False,
            "code": str(detail.get("code") or "refused"),
            "message": str(
                detail.get("message") or "that device did not answer with its MCP servers"
            ),
            "device_id": device_id,
        }
    detail_in = state.get("detail")
    remote: dict[str, Any] = detail_in if isinstance(detail_in, dict) else {}
    servers_in = remote.get("servers")
    remote_servers: dict[str, Any] = servers_in if isinstance(servers_in, dict) else {}

    bundle = local_bundle(server.root)
    missing = [
        row
        for row in bundle["servers"]
        if remote_servers.get(str(row["name"])) != definitions.digest_of(row)
    ]
    if not missing:
        return {
            "ok": True,
            "code": "in_sync",
            "message": f"{label} and this device hold the same MCP servers",
            "device_id": device_id,
            "pushed": False,
            "servers": {str(row["name"]): definitions.digest_of(row) for row in bundle["servers"]},
            "withheld": bundle.get("withheld") or [],
        }
    # No size check on ``missing``: it is a subset of ``local_bundle()``'s rows,
    # which already raised at the same cap — the refusal is spelled once
    # (agent review round 1, MINOR).
    outbound = {
        "kind": BUNDLE_KIND,
        "version": BUNDLE_VERSION,
        "origin_device": bundle.get("origin_device") or "",
        "servers": missing,
    }
    try:
        reply = link.request(
            {
                "op": "net_mcp_defs",
                "req": server._next_relay_req(),  # noqa: SLF001
                "locality": "remote",
                "phase": "apply",
                "bundle": outbound,
            },
            timeout=PUSH_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 — see above
        return {"ok": False, "code": "apply_failed", "message": str(exc), "device_id": device_id}
    if not isinstance(reply, dict) or reply.get("op") == "error":
        detail = reply if isinstance(reply, dict) else {}
        return {
            "ok": False,
            "code": str(detail.get("code") or "apply_failed"),
            "message": str(detail.get("message") or "that device refused the MCP servers"),
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
            f"sent {len(missing)} MCP server "
            f"{'definition' if len(missing) == 1 else 'definitions'}"
            if ok
            else "that device would not take every MCP server: "
            + _describe_rows([*conflicts, *refused])
        ),
        "device_id": device_id,
        "pushed": True,
        "installed": summary.get("installed") or [],
        "updated": summary.get("updated") or [],
        "unchanged": summary.get("unchanged") or [],
        "conflicts": conflicts,
        "refused": refused,
        "withheld": bundle.get("withheld") or [],
    }


def _describe_rows(rows: Iterable[Mapping[str, Any]]) -> str:
    """``kind 'name' (reason)`` for a person; never the payload, never a value."""
    parts = []
    for row in rows:
        name = str(row.get("name") or "?")
        # The read stays in the rendered expression on purpose: the reason
        # surfaces guard finds a raw read by walking the nodes that build a
        # line, so routing it through a local would slip a peer's sentence past
        # that guard (definitions._describe_rows makes the same point).
        parts.append(
            f"{row.get('kind') or 'row'} {name!r} "
            f"({row.get('reason') or row.get('shape') or 'refused'})"
        )
    return "; ".join(parts) or "no reason given"


# ---------------------------------------------------------------------------
# The ops
# ---------------------------------------------------------------------------


def make_handler(server: "RelayServer") -> Callable[[Any, dict[str, Any]], dict[str, Any]]:
    """The ``net_mcp_defs`` handler for one relay.

    It never issues a request over the link it is serving (``PeerLink.request``
    refuses that): both phases read and write THIS device's own files and
    answer. The outbound direction lives in :func:`push_to_peer`.
    """

    def _handle(link: Any, frame: dict[str, Any]) -> dict[str, Any]:
        phase = str(frame.get("phase") or "")
        if phase == "state":
            return {"phase": "state", **server_state(server.root)}
        if phase == "apply":
            bundle = frame.get("bundle")
            if not isinstance(bundle, Mapping):
                raise MeshRefusal("bad_request", "an apply needs the bundle it applies")
            from local_operator.network.audit import AuditEvent

            summary = apply_bundle(server.root, bundle, origin_device=link.device_id)
            # AUDITED BY NAME, never by content: an install that wrote executable
            # server rows on this device is exactly the event an incident review
            # asks about, and the detail carries names and outcomes only.
            try:
                server.audit.record(
                    AuditEvent(
                        event="mcp_defs_applied",
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
                logger.debug("mcpdefs: could not record the apply", exc_info=True)
            return {"phase": "apply", **summary}
        raise MeshRefusal("bad_request", f"unknown MCP defs phase {phase!r}")

    return _handle


def local_sync_handler(server: "RelayServer") -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The ``mcp_defs_sync`` local verb: push this device's servers out.

    One peer or every paired member, which is what
    ``lop network mcp push [--peer NAME] [--all-peers]`` calls.
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
            # An AGGREGATE, not the per-device sentences joined (definitions'
            # QA round 1, Q2): the detail is the per-device rows, this line
            # answers "how did it go" once.
            "message": (
                "nothing to push to: this device is in no network with another member"
                if not results
                else (
                    f"{sum(1 for item in results if item.get('ok'))} of {len(results)} "
                    f"{'device holds' if len(results) == 1 else 'devices hold'} "
                    "this device's MCP servers"
                )
            ),
        }

    return _handle


def mesh_tick_step(step_server: "RelayServer", device_id: str) -> str:
    """One member's MCP push, run INSIDE the mesh-definitions thread's tick.

    Registered with ``definitions.add_tick_step`` so this cadence rides the
    existing thread and its floors (15 s / 60 s / refused) instead of starting a
    second one. A member whose capability table (``types.OP_CAPABILITY``, read
    through definitions' one checker) cannot hold ``net_mcp_defs`` is skipped
    with a named outcome and NO wire traffic — the same no-retry-storm rule the
    definitions tick applies, for the same measured reason.
    """
    blocked = definitions.unholdable_capability(step_server, device_id, "net_mcp_defs")
    if blocked:
        return f"skipped:no_{blocked}"
    if not local_bundle(step_server.root)["servers"]:
        # NOTHING TO SEND, NOTHING TO ASK: v1 is push-only (no delete propagation),
        # so a device with no user-scope servers can never change what the peer
        # holds, and a dial per member per minute to discover that would bill for
        # a question already known. The on-demand `push` still asks — its per-peer
        # report is the evidence an operator reads.
        return "in_sync"
    return str(push_to_peer(step_server, device_id).get("code") or "")


def install(server: "RelayServer") -> None:
    """Register this slice's ops and its cadence step on ``server``.

    The cadence is a STEP on the definitions syncer, not a start hook of its
    own: the step seam runs inside that thread's per-member loop, so this module
    starts no thread and inherits that thread's floors verbatim.
    """
    definitions.add_tick_step(mesh_tick_step)
    server.register_ops(
        {"net_mcp_defs": make_handler(server)},
        local_handlers={"mcp_defs_sync": local_sync_handler(server)},
        slow={"net_mcp_defs": MCP_DEFS_OP_DEADLINE_S},
    )
