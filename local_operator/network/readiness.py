"""Peer readiness: can this peer COMPLETE work this device offloads to it?

Slice 1 of the mesh remote-offload workstream. The question this answers is NOT
doctor's: ``lop network doctor`` answers "is the link healthy", and it answered
``ok: true`` on 2026-09-28 while every offloaded write/exec PARKED on the far
side — a peer with no operator authority, no git identity, no user-scope MCP
servers, a stale build and no local model credential. Readiness asks that
second question, per peer, and every finding carries the command that fixes it.

WIRE. A viewer's relay runs the LOCAL op ``peer_readiness`` (the ``peer_*``
boundary rule, ``types.LOCAL_OPS``): it probes the peer's endpoints, builds a
link, and — only when the peer advertises :data:`wire.PEER_READINESS_V1` —
asks the peer-scope op ``net_readiness`` (capability ``list``) for the peer's
own facts. The viewer composes rows from both halves and derives each row's
verdict. An old peer degrades to ``peer_too_old`` rows; a peer that does not
answer degrades to ``not_asked`` rows. An absent answer is never a pass.

READ-ONLY, AND PROVEN SO. Nothing in this module writes. The peer handler only
reads files (config.yml, mcp.json, the git config, the operator anchor, the
credential tables), and the one store that would CREATE a database in its
constructor (``AuthStore``) is only reached after its file is confirmed to
exist — see :func:`_open_store`. The unit suite pins that a report over a
fresh root leaves the tree byte-identical (the ``test_reads_create_nothing``
discipline, applied to this verb).

NAMES ONLY, NO MATERIAL. The peer's payload carries credential NAMES and
placement facts, never values: ``has_local``/``has_row`` are booleans, the
placement entry is owner/holders (the placement document has no material by
its own writer's guard), and any text that merely LOOKS like a credential is
withheld and named rather than sent (the definitions rule, through the one
shape table in ``redaction_shapes``).
"""

from __future__ import annotations

import configparser
import json
import os
import re
import shutil
import socket
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, NamedTuple, Protocol, Sequence

from local_operator.network import wire

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.network.relay import PeerLink, RelayServer

#: The peer payload's integer schema, matching the transport's whole-file
#: convention. Purely informational today (the op is additive and per-field
#: absence is handled), but a future breaking change reads it.
READINESS_SCHEMA = 1

#: How long a viewer waits for the peer's answer. The peer handler is local
#: reads only (no dials, no network), so 5 s is generous; a timeout renders
#: unknown rows rather than hanging the report (design §8.8).
READINESS_OP_TIMEOUT_S = 5.0

#: The capability ids a readiness row can carry — the design's checks (a)-(f),
#: plus (g) tooling: the lane's tool inventory, added by the 2026-10-06 request
#: (a lane on cloud-node-1 found node/npm/make/docker absent mid-task and the
#: report had no rows for any of it).
CAPABILITY_OPERATOR_AUTHORITY = "operator_authority"
CAPABILITY_BUILD = "build"
CAPABILITY_GIT = "git_identity"
CAPABILITY_MCP_SERVERS = "mcp_servers"
CAPABILITY_MODEL_CREDENTIAL = "model_credential"
CAPABILITY_MCP_CREDENTIAL = "mcp_credential"
CAPABILITY_TOOLING = "tooling"

#: The class a readiness row carries (F8 ruling, 2026-10-04). "The device is
#: onboarded" and "the equipment on it is set up" are DIFFERENT questions:
#: ADMISSION rows are the mesh facts (identity, network/membership,
#: reachability, build) and they hold the onboarding verdict
#: (:func:`onboarding_failures`); EQUIPMENT rows describe what offloaded work
#: would find there (operator authority, git identity, MCP surfaces, a served
#: model), and a failed one belongs at its POINT OF USE — where the work that
#: needs it actually runs.
CLASS_ADMISSION = "admission"
CLASS_EQUIPMENT = "equipment"

#: The equipment rows that do NOT hold the onboarding verdict (F8 ruling):
#: "every declared MCP server is signed in" is not "the device is onboarded".
#: The sites that genuinely need an MCP login refuse at their own point of use
#: (the share verb's ``no_local_credential`` refusal, the placement/borrow
#: path, the interactive sign-in gate). TOOLING joined this set with the same
#: shape of reason (2026-10-06): a device missing docker/node is not LESS
#: onboarded — which tools a lane will find is a finer-grained fit question it
#: reads before it starts, and a missing tool is repaired at its own point of
#: use (the lane acquires tooling through the product's console flow, which
#: asks before installing). The rest of the equipment set keeps gating by
#: default until the operator decides — this set is a policy, never a
#: per-call carve-out.
NON_GATING_EQUIPMENT: frozenset[str] = frozenset(
    {CAPABILITY_MCP_CREDENTIAL, CAPABILITY_MCP_SERVERS, CAPABILITY_TOOLING}
)

#: The fixed set of peer-side checks that exist even when no answer arrived —
#: the rows a ``peer_too_old`` / ``not_asked`` peer still gets, one per check,
#: so a consumer's checklist does not lose its shape when a peer goes dark.
#: The bare ``build`` row is NOT in this set: the build stamp rides the
#: handshake, so it is composed per link (and only joins this set when there is
#: no link at all).
PEER_SIDE_CHECKS: tuple[str, ...] = (
    CAPABILITY_OPERATOR_AUTHORITY,
    CAPABILITY_GIT,
    CAPABILITY_MCP_SERVERS,
    CAPABILITY_MODEL_CREDENTIAL,
    CAPABILITY_TOOLING,
)

#: Failure codes. OK rows carry no code; a code is a machine token for a
#: failure a consumer can branch on without reading the sentence.
CODE_NOT_INSTALLED = "not_installed"
CODE_ANCHOR_UNPINNED = "anchor_unpinned"
CODE_UNUSABLE = "unusable"
CODE_NO_GIT_IDENTITY = "no_git_identity"
CODE_NO_MCP_SERVERS = "no_mcp_servers"
CODE_NOT_CONFIGURED = "not_configured"
CODE_NOT_ASKED = "not_asked"
CODE_PEER_TOO_OLD = "peer_too_old"
CODE_UNKNOWN = "unknown"
CODE_BEHIND = "behind"
CODE_AHEAD = "ahead"
CODE_NO_CREDENTIAL = "no_credential"
CODE_NOT_SHARED = "not_shared"
CODE_OBSERVED_FAILURE = "observed_failure"
#: Tooling (2026-10-06): the distinct remedy-bearing states get distinct
#: machine tokens — an install, a one-line PATH fix and a sign-in are
#: different acts, and a consumer branches on which one it must arrange.
CODE_NOT_ON_PATH = "not_on_path"
CODE_NOT_AUTHENTICATED = "not_authenticated"

#: Rows read from the peer's own state carry ``source: "peer"``; rows about a
#: fact that could not be established carry ``"unknown"``. ``"local"`` is kept
#: for this side's own facts (the build comparison is composed here).
SOURCE_PEER = "peer"
SOURCE_LOCAL = "local"
SOURCE_UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _bounded(value: Any, limit: int) -> str:
    return str(value or "")[:limit]


def _credential_shape(text: Any) -> str:
    """The credential shape ``text`` is spelled like, through the ONE table.

    Reuses the definitions module's guard rather than a second pattern: the
    shape table in ``redaction_shapes`` is what every export, tool result and
    transcript is scanned with, and a payload guarded by a different copy of it
    would eventually disagree with the rest of the product about what a
    credential looks like.
    """
    from local_operator.network.definitions import credential_shape

    return credential_shape(text)


def _safe_text(value: Any, *, limit: int = 300) -> str:
    """A bounded string — or ``""`` when it LOOKS like a credential."""
    text = _bounded(value, limit)
    if text and _credential_shape(text):
        return ""
    return text


# ---------------------------------------------------------------------------
# This device's own facts (the peer handler's half)
# ---------------------------------------------------------------------------


def _open_store(root: Path) -> Any | None:
    """An ``AuthStore`` for ``root``, or ``None`` when no database exists yet.

    THE GUARD IS THE READ-ONLY PROMISE: ``AuthStore.__init__`` CREATES its
    database (mkdir, a 0600 file, then the schema), so constructing one on a
    device that has never signed in would make this report a WRITER — exactly
    the class ``test_reads_create_nothing`` exists to prevent. The database's
    absence is also the definite answer ("no store at all is a definite
    False", ``has_stored_row``), so nothing is lost by checking first.

    ``db_path`` is passed explicitly: the store derives its default database
    from ``config_dir()`` (ambient), and a report about ROOT A must read A's
    database even when the process's ambient root is B's.
    """
    database = root / "auth.db"
    if not database.exists():
        return None
    from local_operator.providers.auth_store import AuthStore

    return AuthStore(db_path=database, config_dir=root)


def operator_fact() -> dict[str, Any]:
    """The operator-authority level and what it rests on. Never raises.

    ``verify_only`` (remote-onboarding §2.5, OQ6): an anchor this host can
    VERIFY with but cannot SIGN with — the node-side case, where authority was
    installed as public data and the private half lives on the operator's own
    devices. The level is what the anchor CLAIMS; ``verify_only`` is the extra
    fact that stops the claim from being read as "signs here": a host can be
    ``operator-presence`` and still hold no key, because the anchor says which
    backend the OPERATOR's machine uses, not what this machine has.
    """
    try:
        from local_operator.operator import operator_authority_report

        report = operator_authority_report()
    except Exception:  # noqa: BLE001 — an unreadable anchor is a fact, not a crash
        return {
            "level": "unreported",
            "reason": "the authority report could not be built",
            "anchor_installed": False,
            "anchor_root_owned": False,
            "presence": False,
            "verify_only": False,
        }
    level = _bounded(report.get("level") or "unreported", 60)
    installed_levels = ("operator-presence", "operator-file-only")
    verify_only = level in installed_levels and not _can_sign_here()
    return {
        "level": level,
        # EVERY carried string goes through the shape guard, reasons included:
        # the module's rule is "any text that merely LOOKS like a credential is
        # withheld rather than sent", and a reason is text this device did not
        # write (round 1, MINOR-2 — the sweep was narrower than the claim).
        "reason": _safe_text(report.get("reason") or "", limit=200),
        "anchor_installed": bool(report.get("anchor_installed")),
        "anchor_root_owned": bool(report.get("anchor_root_owned")),
        "presence": bool(report.get("presence")),
        "verify_only": verify_only,
    }


def _can_sign_here() -> bool:
    """Whether THIS host holds the private half, probed without prompting.

    The probe is the operator module's own signer lookup (the anchor names the
    backend), and it is the ONLY sound question to ask: a keychain query from
    an unsigned process cannot see a ``secure-enclave`` item (the -25300 trap),
    so anything cheaper than the real loader would answer "no key" for a host
    that has one. ``False`` on every failure — a broken agent is "cannot sign
    NOW", which is the fact this answer is for — and the signer is closed
    immediately, because a probe must leave nothing running.
    """
    try:
        from local_operator.operator.keychain import choose_backend
        from local_operator.operator.trust import load_anchor
        from local_operator.paths import config_dir

        loaded = load_anchor()
        backend_name = loaded.anchor.backend if loaded.usable and loaded.anchor else "auto"
        signer: Any = choose_backend(backend_name, config_root=config_dir()).load()
        if signer is None:
            return False
        try:
            signer.close()
        except Exception:  # noqa: BLE001 — closing a probe is best-effort
            pass
        return True
    except Exception:  # noqa: BLE001 — see above: "cannot sign", not a crash
        return False


def git_identity_fact(home: Path | None = None) -> dict[str, Any]:
    """``user.name``/``user.email`` from the GLOBAL git config files.

    Read as FILES (configparser), never a ``git config`` subprocess: this runs
    inside a peer's relay on a request path, where a process spawn per check
    would be paid on every report, and the question v1 asks — are the values
    SET at global scope — needs no include/env resolution (design §8.5 defers
    those, with the credential-helper and ssh posture).

    Precedence matches git's own: ``~/.config/git/config`` is read first and
    ``~/.gitconfig`` overrides it (git's documented order — a single-valued
    variable in the XDG file is overwritten by whatever is in ``~/.gitconfig``).

    A missing file is the ordinary state; an unparsable one is skipped rather
    than fatal — this is a report, and the next file may still carry the
    answer.
    """
    root = home if home is not None else Path.home()
    parser = configparser.ConfigParser(interpolation=None)
    # ``interpolation=None`` is load-bearing: git config values are literal
    # text, and the default BasicInterpolation raises on a bare '%'.
    for path in (root / ".config" / "git" / "config", root / ".gitconfig"):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                parser.read_file(handle)
        except (OSError, UnicodeDecodeError, configparser.Error):
            continue
    # Section names are case-sensitive in configparser and case-INsensitive in
    # git, so the section is found case-insensitively ('[User]' counts).
    section = next((name for name in parser.sections() if name.strip().lower() == "user"), "")

    def value(option: str) -> str:
        if not section:
            return ""
        try:
            return _safe_text(parser.get(section, option, raw=True).strip(), limit=200)
        except (configparser.Error, ValueError):
            return ""

    return {"user_name": value("name"), "user_email": value("email")}


#: The command-line tools a lane is expected to find on a device it runs on.
#: Ordered, and one fixed set: the row reads it in this order, so a consumer's
#: checklist cannot drift from the run-time probe. The 2026-10-06 request: a
#: lane on cloud-node-1 discovered node/npm/make/docker were absent — and gh
#: was present but not on PATH and not signed in — only mid-task; this
#: inventory lets a planner read all of it BEFORE it starts.
TOOLING_TOOLS: tuple[str, ...] = ("gh", "glab", "node", "npm", "make", "docker")

#: How much of ``hosts.yml`` the structural scan may read. gh's file is a few
#: hundred bytes; the bound exists so a pathological file cannot make a report
#: read it wholesale — the scan needs only the top-level keys.
_GH_HOSTS_SCAN_LIMIT = 64 * 1024

#: A TOP-LEVEL YAML key line in ``hosts.yml`` — ``github.com:`` — with no
#: leading space and no inline value. The scan tests SHAPES ONLY: an indented
#: value line (where the token sits, when the token is in this file at all)
#: can never match, so no value is captured, and the match is folded to a
#: boolean before anything can leave this module.
_GH_HOSTS_ENTRY_RE = re.compile(rb"^[A-Za-z0-9][\w.:-]*:[ \t]*$", re.MULTILINE)


def _probe_reason(exc: BaseException) -> str:
    """The NAME of what went wrong, bounded — never the message it carries."""
    return _safe_text(type(exc).__name__, limit=80)


def _resolve_program(name: str, *, path: str | None = None) -> str | None:
    """``shutil.which`` behind ONE name, so the probe has one seam.

    Answers with the ABSOLUTE path (which is what the remedy's fact needs); the
    wrapper exists so a test can pin the failure path without patching
    ``shutil`` for the whole process, and so the PATH question is asked in
    exactly one place.
    """
    return shutil.which(name, path=path)


def _tool_location(name: str, local_bin: Path) -> dict[str, Any]:
    """Where ``name`` resolves for this device, as a four-state cell.

    ``on_path`` — ``shutil.which`` found it on THIS process's PATH, the same
    environment an offloaded request executes in. ``off_path`` — the PATH probe
    missed but the user-local bin directory has it. ``absent`` — neither probe
    found it. ``unknown`` — a probe RAISED, so neither "present" nor "absent"
    may be claimed (the never-a-false-ok discipline; the reason rides the cell).
    """
    try:
        found = _resolve_program(name)
    except Exception as exc:  # noqa: BLE001 — a broken probe is a fact, not a crash
        return {"state": "unknown", "path": "", "reason": _probe_reason(exc)}
    if found:
        return {"state": "on_path", "path": _safe_text(found, limit=300)}
    try:
        local = _resolve_program(name, path=str(local_bin))
    except Exception as exc:  # noqa: BLE001 — see above
        return {"state": "unknown", "path": "", "reason": _probe_reason(exc)}
    if local:
        return {"state": "off_path", "path": _safe_text(local, limit=300)}
    return {"state": "absent", "path": ""}


def _gh_auth_fact(home: Path) -> dict[str, Any]:
    """Whether a GitHub CLI login is STORED here — structure, never material.

    The honest signals the 2026-10-06 request names: ``~/.config/gh/hosts.yml``
    existence AND whether an entry exists — existence alone is not the second
    (a file left behind by ``gh auth logout`` carries no entry), so the file is
    opened READ-ONLY and scanned STRUCTURALLY: the scan tests top-level key
    shapes and keeps ONE boolean. No value is parsed, retained, or carried, so
    the token the file may hold cannot travel (the module's names-only rule);
    that it is opened at all is deliberate, documented, and the reason the
    signal can be called honest. An unreadable file answers ``None`` with the
    reason — never ``False``, which would be a false "not signed in".

    Deferred, like git's include/env resolution (§8.5): the DEFAULT path only,
    so a ``GH_CONFIG_DIR``/``XDG_CONFIG_HOME`` relocation is not chased in v1.
    That the token itself may live in the OS keychain (gh's secure-storage
    default) does not move this signal: the host entry is what ``gh auth
    login`` writes on every platform, and the keychain is not a fact this
    report may read.
    """
    path = home / ".config" / "gh" / "hosts.yml"
    fact: dict[str, Any] = {
        "hosts_file": False,
        "has_entry": False,
        "config_path": _safe_text(str(path), limit=300),
    }
    try:
        with open(path, "rb") as handle:
            raw = handle.read(_GH_HOSTS_SCAN_LIMIT)
    except FileNotFoundError:
        return fact
    except OSError as exc:  # the file is there but cannot be read: say why, never guess
        fact["hosts_file"] = True
        fact["has_entry"] = None
        fact["reason"] = _probe_reason(exc)
        return fact
    fact["hosts_file"] = True
    fact["has_entry"] = bool(_GH_HOSTS_ENTRY_RE.search(raw))
    return fact


def tooling_fact(home: Path | None = None) -> dict[str, Any]:
    """What command-line tooling a lane would find here, as facts. Names only.

    The probe reads THIS process's environment — its PATH and the home it is
    given — because that is what the relay answering the ask can honestly
    attest to; a tool present only in some other shell's PATH is not claimed
    here (the system-tools guide's "a shell that answered command not found is
    only evidence about THAT shell", applied to the reporter). Read-only: the
    module's ``test_reads_create_nothing`` discipline applies to every fact
    collected on a peer's request path, this one included.
    """
    root = Path.home() if home is None else home
    local_bin = root / ".local" / "bin"
    return {
        "tools": {name: _tool_location(name, local_bin) for name in TOOLING_TOOLS},
        "gh_auth": _gh_auth_fact(root),
        "local_bin_dir": _safe_text(str(local_bin), limit=300),
    }


def _transport_of(raw: Mapping[str, Any]) -> str:
    """The transport the entry spells, with mcp/config.py's own inference.

    Mirrors ``_coerce_server_config``: an explicit ``type`` wins; otherwise a
    ``command`` implies stdio and a ``url`` implies http. Kept as a mirror
    rather than an import because that helper is private to the config module
    and this is a read of ONE file, not the merged discovery list.
    """
    declared = raw.get("type")
    if declared in ("stdio", "http", "sse"):
        return str(declared)
    if raw.get("command"):
        return "stdio"
    if raw.get("url"):
        return "http"
    return "stdio"


def _mcp_row_exists(store: Any | None, url: str) -> bool | None:
    """Whether a credential row exists for ``url``; ``None`` = unreadable store.

    ``store is None`` — this root has no database (``_open_store`` refuses to
    create one) — is answered HERE as a definite ``False``, never handed to
    ``McpTokenStorage``: its ``store=None`` means the OPPOSITE thing ("build the
    ambient store"), so the construction resolved to ``AuthStore()`` for the
    process's ambient root — which CREATES the database. A read-only report would
    write on the device it is describing as having no store, and would read the
    wrong root's rows whenever the two roots differ (found by the shareability
    ledger's creates-nothing cell, design §2; the rule is the one
    ``ViewerFacts.holds_mcp_login`` / ``has_local_provider_credential`` already
    apply at their own call sites).
    """
    if store is None:
        return False
    try:
        from local_operator.mcp.auth import McpTokenStorage

        return McpTokenStorage(url, store=store).has_stored_row()
    except Exception:  # noqa: BLE001 — an unreadable store is "not known"
        return None


def mcp_login_remedy(url: str) -> str:
    """The ONE spelling of the "sign in here first" clause for an MCP server URL.

    Three surfaces compose it — the share verb's refusal (``cli``'s
    ``_require_local_credential``), the readiness row's detail (this module), and
    the shareability ledger's ``no login here yet`` row — and a second copy is
    exactly how one of them starts sending the operator to sign in somewhere other
    than "here".

    THE SENTENCE NAMES THE ACTION, NOT A COMMAND (§2.9's repave): the retired
    shape instructed its reader to run ``'/mcp login …'`` — a terminal command
    in a remedy — and this clause is now the product action alone. ``url`` stays
    in the signature because the three callers name it in the sentence this
    clause rides. The guides' canonical ``/mcp login`` route stays as
    documentation (§2.9's own carve-out); the MCP surfaces' own FAILURE copy
    (the login toasts, the broker's refusals in ``credentials/messages.py``) is
    a SEPARATE SWEEP owed against §2.9 — that rule names failure and refusal
    sentences outright, so those sites are debt, never an exemption.
    """
    return "sign in here first"


def mcp_servers_fact(root: Path) -> dict[str, Any]:
    """The USER-SCOPE MCP servers, and per HTTP/SSE server whether a login row exists.

    Reads the file the WRITERS target (``<root>/mcp.json``), not the merged
    discovery list: a project-scope server is cwd-dependent and says nothing
    about what an offload — whose cwd the sender picks — will see (design §8.6).
    A missing or unreadable file reads as "no servers"; the sentence for that
    state already says "missing or empty".
    """
    path = root / "mcp.json"
    withheld: list[str] = []
    servers: list[dict[str, Any]] = []
    document: Any = None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        document = None
    except (OSError, UnicodeDecodeError, ValueError):
        document = None
    entries = document.get("mcpServers") if isinstance(document, dict) else None
    if not isinstance(entries, Mapping):
        entries = {}
    store = None
    store_unreadable = False
    try:
        store = _open_store(root)
    except Exception:  # noqa: BLE001 — an unopenable store is "could not be read"
        store_unreadable = True
    try:
        for name, raw in sorted(entries.items(), key=lambda pair: str(pair[0])):
            if not isinstance(raw, Mapping):
                continue
            transport = _transport_of(raw)
            url = _safe_text(raw.get("url"), limit=300)
            shape = _credential_shape(str(name)) or _credential_shape(raw.get("url"))
            if shape:
                # THE DEFINITIONS RULE: a row whose text looks like a credential
                # is withheld and NAMED rather than sent. The name goes in the
                # withheld list (or the shape label when even the name is the
                # credential-shaped thing); the value never travels.
                held_name = str(name)
                withheld.append(
                    held_name
                    if not _credential_shape(held_name)
                    else f"a value shaped like {shape}"
                )
                continue
            has_row: bool | None = None
            has_placement = False
            if transport in ("http", "sse") and url:
                has_row = None if store_unreadable else _mcp_row_exists(store, url)
                # THIS device's own placement document: does an entry for the
                # server exist HERE? It is what tells the viewer whether a share
                # on the owner's books has REACHED this device — the pull is paid
                # on an explicit act (``lop network credentials``), never on the
                # provider path, so a peer that has not pulled cannot borrow and
                # must not be reported as ready to.
                from local_operator.network.credentials.placement import (
                    placement_entries_for,
                )

                has_placement = placement_entries_for(mcp_url=url, root=root) is not None
            servers.append(
                {
                    "name": _safe_text(name, limit=120),
                    "url": url,
                    "transport": transport,
                    # The config declared an auth block of its own: a hint for the
                    # credential verdict's copy, never a claim about the server.
                    "auth_declared": bool(raw.get("oauth") or raw.get("auth")),
                    "has_row": has_row,
                    "has_placement": has_placement,
                }
            )
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001 — a close failure must not fail the report
                pass
    return {"servers": servers, "withheld": withheld, "config_path": str(path)}


def shareable_lines(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """The ``shareable here`` block — one spelling for the CLI and the agent digest.

    Rendered from ``lop network credentials``'s own ``--json`` rows (the
    device-level ledger, design §2), so the two surfaces cannot drift the way the
    readiness rows once did: ``login held`` names the share command, ``no login here
    yet`` reuses :func:`mcp_login_remedy` verbatim, and an unreadable store says so
    rather than reading as "no login" (the three-valued rule the MCP credential rows
    already follow). An already-shared server nests its holders the way the network
    blocks do. Empty input renders NOTHING — the block is additive, and a device that
    declares no HTTP/SSE servers has no ledger to print.

    TWO ROW KINDS share the block: an MCP-server row (``server``/``transport``/
    ``login_here``, the original shape) and a PROVIDER-LOGIN row (``provider``/
    ``kind``/``identity_label``; Radient org projection, 2026-09-29). The provider
    row exists only when this device HOLDS the login — there is no server list to
    say "no login here yet" about — so it renders the held sentence, and, when the
    classifier read one, the identity the operator signed in as. The Radient row
    adds one caution line naming its person scope (design review round 1, D1).
    """
    materialized = [row for row in rows if isinstance(row, Mapping)]
    if not materialized:
        return []
    lines = ["shareable here:"]
    for row in materialized:
        if "provider" in row:
            lines.append(
                f"  {row.get('provider')}  {row.get('kind')}  "
                f"login held — share: {row.get('remedy') or ''}"
            )
            # ONE CAUTION LINE, RADIENT ONLY (design review round 1, D1): the org
            # login is the family's person-scoped account — it can publish and
            # delete teams and agents — and its row was otherwise word-for-word the
            # openai row, so the caution lived only in the GUIDE. Other provider
            # logins are lent under the same one-device rule; only this one reads
            # as an org-scoped grant without the cue.
            if row.get("provider") == "radient":
                lines.append("      organization account — share only to your own devices")
            label = str(row.get("identity_label") or "")
            if label:
                lines.append(f"      signed in as {label}")
        else:
            login = row.get("login_here")
            if login is True:
                summary = f"login held — share: {row.get('remedy') or ''}"
            elif login is False:
                summary = f"no login here yet — {row.get('remedy') or ''}"
            else:
                summary = "login state not known — this device's credential store could not be read"
            lines.append(f"  {row.get('server')}  {row.get('transport')}  {summary}")
        for holder in row.get("shared_with") or []:
            if not isinstance(holder, Mapping):
                continue
            lines.append(
                f"      shared with {holder.get('name') or holder.get('device')} "
                f"({holder.get('scope')})"
            )
    return lines


def default_model_fact(root: Path) -> dict[str, Any]:
    """The configured default model pair, resolved through the product's own resolver.

    ``bootstrap.resolve_hosting_model`` is the exact resolver
    ``resolve_model_configuration`` delegates the pair to (bootstrap.py:141);
    the latter additionally BUILDS a ``ModelConfiguration`` — provider client
    metadata and a best-effort static key read out of the credential store —
    that this fact never reads, and building it on a report path would drag
    the credential store into a question that is only about the pair. Same
    resolution, none of the surface.

    ``config.yml`` is checked for existence BEFORE ``ConfigManager`` is
    constructed: that constructor creates the config root on a device that has
    none, and a report must not be the reason a directory exists.
    """
    config_file = root / "config.yml"
    if not config_file.exists():
        return {"hosting": "", "model_name": "", "resolved": False, "reason": "no config.yml"}
    try:
        from local_operator.bootstrap import resolve_hosting_model
        from local_operator.config import ConfigManager

        hosting, model_name = resolve_hosting_model(ConfigManager(root), None, None, None)
    except ValueError as exc:
        # The product's own not-configured messages (hosting unset, no default for
        # the provider): a config that PARSES and names no usable pair.
        return {
            "hosting": "",
            "model_name": "",
            "resolved": False,
            "reason": _safe_text(str(exc), limit=200),
        }
    except Exception as exc:  # noqa: BLE001 — a config that cannot be read is a fact
        # A DIFFERENT sentence, because it is a different problem: a corrupt or
        # partial file is not "no default configured", and a row that said so
        # would send someone to set a model that is already set.
        return {
            "hosting": "",
            "model_name": "",
            "resolved": False,
            "reason": _safe_text(f"its config could not be read ({exc.__class__.__name__})"),
        }
    return {
        "hosting": _safe_text(hosting, limit=120),
        "model_name": _safe_text(model_name, limit=200),
        "resolved": True,
        "reason": "",
    }


def has_local_provider_credential(root: Path, provider: str) -> bool | None:
    """Whether this device holds a credential row for ``provider``.

    Three-valued like ``has_stored_row``: ``None`` is "the store could not be
    read", which must never read as "no login here" (that answer sends the
    operator to a login they may already have). An absent store file is a
    definite ``False`` — see :func:`_open_store`.
    """
    if not provider:
        return False
    try:
        store = _open_store(root)
    except Exception:  # noqa: BLE001 — an unopenable store is "could not be read"
        return None
    if store is None:
        return False
    try:
        return bool(list(store.list_credentials(provider)))
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            store.close()
        except Exception:  # noqa: BLE001
            pass


def placement_fact(*, provider: str = "", mcp_url: str = "", root: Path) -> dict[str, Any]:
    """The placement entry for a key (owner + holders) and this device's observation.

    ``{}`` when there is no entry — absence is a refusal (nobody may borrow)
    and the viewer reads it as such, never as "unknown". The observation rides
    the same lookup: it is the borrower-side memory of the last borrow attempt,
    which is what lets the verdict say ``owner_offline`` with a last-seen stamp
    instead of guessing from reachability.
    """
    from local_operator.network.credentials import placement as placement_mod
    from local_operator.network.credentials.state import PlacementState

    found = placement_mod.placement_entries_for(provider=provider, mcp_url=mcp_url, root=root)
    if found is None:
        return {}
    network_id, entry = found
    # ``PlacementState`` lives in its own module (the state file is the client's
    # durable memory); reached through ``credentials.placement`` it resolves at
    # runtime but not for the type checker, which is why every other caller
    # imports it from ``credentials.state`` directly (cli.py's credentials arm
    # is the precedent).
    observation = PlacementState.load(network_id, root=root).observation(entry.key)
    return {
        "owner_device": entry.owner_device,
        # A name another device chose: through the shape guard like every other
        # carried string (round 1, MINOR-2).
        "owner_device_name": _safe_text(entry.owner_device_name, limit=120),
        "holders": [{"device": holder.device, "scope": holder.scope} for holder in entry.holders],
        "observation": _bounded_observation(observation),
    }


def _bounded_observation(row: Mapping[str, Any] | None) -> dict[str, Any]:
    """The observation fields a verdict reads, and nothing else.

    The stored row also carries ``key`` (the credential's NAME) and
    ``last_grant_id``; neither is needed to word the sentence, and "names
    only" is cheaper to keep true by construction than to audit later.
    """
    if not isinstance(row, Mapping):
        return {}
    return {
        "status": _bounded(row.get("status") or "", 40),
        "reason": _safe_text(row.get("reason") or "", limit=200),
        "owner_device": _bounded(row.get("owner_device") or "", 80),
        "observed_at": row.get("observed_at"),
        "retry_after_ms": row.get("retry_after_ms"),
        "last_grant_at": row.get("last_grant_at"),
    }


def collect_peer_facts(root: Path, *, home: Path | None = None) -> dict[str, Any]:
    """Everything the viewer needs from THIS device, as facts. Names only."""
    model = default_model_fact(root)
    provider = str(model.get("hosting") or "")
    return {
        "schema": READINESS_SCHEMA,
        "default_model": {
            "hosting": model["hosting"],
            "model_name": model["model_name"],
            "resolved": model["resolved"],
            "reason": model["reason"],
        },
        "provider": provider,
        # The provider is the ``provider`` key above; the fact is three-valued
        # (True / False / None = the store could not be read).
        "has_local": has_local_provider_credential(root, provider),
        "credential_placement": placement_fact(provider=provider, root=root),
        "mcp": mcp_servers_fact(root),
        "git": git_identity_fact(home),
        "tooling": tooling_fact(home),
        "operator": operator_fact(),
    }


# ---------------------------------------------------------------------------
# This side's own credential facts (the composer's half)
# ---------------------------------------------------------------------------


class Viewer(Protocol):
    """What a verdict builder needs of THIS device's facts.

    A Protocol rather than the concrete :class:`ViewerFacts`, because the unit
    matrix drives every builder with small handwritten doubles (they pin which
    QUESTION a verdict asks, not how the answer is stored) — and a nominal
    parameter type turned each of those doubles into a type-check error instead
    of a test. ``ViewerFacts`` is the production implementation; nothing else
    should implement this accidentally. ``close()`` is deliberately absent: the
    composer owns lifetime and calls it on the concrete class.
    """

    device_id: str

    def provider_rows(self, provider: str) -> list[Any] | None: ...

    def holds_mcp_login(self, url: str) -> bool | None: ...

    def placement(self, *, provider: str = "", mcp_url: str = "") -> dict[str, Any]: ...


class ViewerFacts:
    """This device's credential facts, resolved lazily and closed in one place.

    One SQLite handle is reused across the provider row and every MCP server in
    one report (the ``_provider_rows``/``McpTokenStorage`` pattern), and closed
    by the composer's ``finally`` — a store left open per request is a handle
    leak on a path a listing can hit several times a minute.
    """

    def __init__(self, root: Path, *, device_id: str = "") -> None:
        self.root = root
        self.device_id = device_id
        self._store: Any = None
        self._opened = False
        self._store_unreadable = False

    def _open(self) -> Any:
        if not self._opened:
            try:
                self._store = _open_store(self.root)
            except Exception:  # noqa: BLE001 — unopenable is "could not be read"
                self._store = None
                self._store_unreadable = True
            self._opened = True
        return self._store

    def provider_rows(self, provider: str) -> list[Any] | None:
        """This device's rows for ``provider``: ``[]`` none, ``None`` unreadable."""
        if not provider:
            return []
        store = self._open()
        if store is None:
            return None if self._store_unreadable else []
        try:
            return list(store.list_credentials(provider))
        except Exception:  # noqa: BLE001
            return None

    def holds_mcp_login(self, url: str) -> bool | None:
        store = self._open()
        if store is None:
            return None if self._store_unreadable else False
        return _mcp_row_exists(store, url)

    def placement(self, *, provider: str = "", mcp_url: str = "") -> dict[str, Any]:
        return placement_fact(provider=provider, mcp_url=mcp_url, root=self.root)

    def close(self) -> None:
        if self._store is not None:
            try:
                self._store.close()
            except Exception:  # noqa: BLE001
                pass
            self._store = None


# ---------------------------------------------------------------------------
# Row builders (verdicts composed on the viewing side)
# ---------------------------------------------------------------------------


def _row_class(capability: str) -> str:
    """The class a capability row carries (F8): build parity is an ADMISSION
    fact — a peer on another build is not admitted to run this device's work —
    and every other capability row is equipment about that peer."""
    return CLASS_ADMISSION if capability == CAPABILITY_BUILD else CLASS_EQUIPMENT


def _capability_row(
    *,
    device_id: str,
    device_name: str,
    capability: str,
    ok: bool,
    detail: str,
    code: str = "",
    remedies: Iterable[str] = (),
    source: str,
    observed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "check": "readiness",
        "capability": capability,
        "class": _row_class(capability),
        "device_id": device_id,
        "device_name": device_name,
        "ok": bool(ok),
        "detail": detail,
        "remedies": [str(item) for item in remedies],
        "source": source,
    }
    if code:
        row["code"] = code
    if observed:
        row["observed"] = dict(observed)
    return row


def _label(member: Any) -> str:
    return str(getattr(member, "name", "") or "") or str(getattr(member, "device_id", ""))


def _peer_too_old_rows(member: Any) -> list[dict[str, Any]]:
    """One row per peer-side check for a peer that predates this report."""
    detail = "the peer predates readiness reporting, so its answer is not available"
    # §2.9 (QA round 1, Q3): a remedy names a product action, and the action here
    # is an update ON THE PEER — the old sentence printed two ``lop-`` commands.
    remedy = (
        f"if {_label(member)} is running an older build, ask Local Operator to update "
        "it, then re-check"
    )
    return [
        _capability_row(
            device_id=member.device_id,
            device_name=_label(member),
            capability=capability,
            ok=False,
            code=CODE_PEER_TOO_OLD,
            detail=detail,
            remedies=[remedy],
            source=SOURCE_UNKNOWN,
        )
        for capability in PEER_SIDE_CHECKS
    ]


def not_asked_rows(
    member: Any,
    *,
    detail: str,
    remedies: Iterable[str] = (),
    capabilities: Sequence[str] = PEER_SIDE_CHECKS,
) -> list[dict[str, Any]]:
    """One ``not_asked`` row per capability for a peer nothing could be asked of.

    TWO CALLERS, one shape: the composer uses it for a peer that did not answer
    (or answered too late), and ``cli._ready_locally`` uses it for the no-relay
    fallback — a row that was never dialled says so and is ``ok: false``, the
    same discipline the doctor's endpoint rows follow.
    """
    return [
        _capability_row(
            device_id=member.device_id,
            device_name=_label(member),
            capability=capability,
            ok=False,
            code=CODE_NOT_ASKED,
            detail=detail,
            remedies=remedies,
            source=SOURCE_UNKNOWN,
        )
        for capability in capabilities
    ]


def operator_row(member: Any, facts: Mapping[str, Any], *, peer_label: str) -> dict[str, Any]:
    """(a) Can an approval that needs the operator be ALLOWED there?"""
    fact = facts.get("operator")
    if not isinstance(fact, Mapping):
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_OPERATOR_AUTHORITY,
            ok=False,
            code=CODE_UNKNOWN,
            detail="the peer's answer did not carry this check",
            source=SOURCE_UNKNOWN,
        )
    level = str(fact.get("level") or "")
    reason = _bounded(fact.get("reason") or "", 200)
    # THE VERIFY-ONLY HOST (OQ6): an anchor is installed — public data — but the
    # private half is not on this host, so nothing can be SIGNED there. The row
    # stays ok (the authority exists and approvals can be answered) while saying
    # where the signing actually happens; the level values are unchanged.
    if fact.get("verify_only") and level in ("operator-presence", "operator-file-only"):
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_OPERATOR_AUTHORITY,
            ok=True,
            detail=(
                f"operator authority is installed on {peer_label}: approvals for "
                "offloaded work can be signed from your devices"
            ),
            source=SOURCE_PEER,
        )
    if level == "operator-presence":
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_OPERATOR_AUTHORITY,
            ok=True,
            detail=f"operator authority is installed on {peer_label}, with a presence check",
            source=SOURCE_PEER,
        )
    if level == "operator-file-only":
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_OPERATOR_AUTHORITY,
            ok=True,
            detail=(
                f"operator authority is installed on {peer_label}, but its key is a 0600 "
                "file: any process running as this user can sign for the operator — "
                "presence is not enforced there"
            ),
            source=SOURCE_PEER,
        )
    if level == "anchor-unpinned":
        code = CODE_ANCHOR_UNPINNED
        staged = ""
    elif level == "unreported":
        code = CODE_UNUSABLE
        staged = ""
    else:
        code = CODE_NOT_INSTALLED
        staged = " (an anchor is staged but not installed)" if fact.get("anchor_installed") else ""
    sentence = (
        f"operator authority is not installed on {peer_label}{staged}"
        + (f" ({reason})" if reason and reason != "ok" else "")
        + ": an approval that needs the operator — a write or command offloaded there — "
        "parks until someone installs it"
    )
    return _capability_row(
        device_id=member.device_id,
        device_name=peer_label,
        capability=CAPABILITY_OPERATOR_AUTHORITY,
        ok=False,
        code=code,
        detail=sentence,
        remedies=[
            # §2.9: a remedy names a PRODUCT action, never a terminal command — and
            # never a surface the reader may not have (design round 1, D1/D2: no
            # ship of a Mesh tab exists until the setup card does, so the interim
            # action is the agent path).
            f"ask Local Operator to set up operator authority on {peer_label} (one "
            "approval and one admin password prompt), then approvals for offloaded work "
            "can be answered from your devices"
        ],
        source=SOURCE_PEER,
    )


class BuildComparison(NamedTuple):
    """``compare_builds``'s answer: the verdict, plus the versions it compared.

    ``state`` is ``equal``/``behind``/``ahead``/``unknown`` — and ``unknown`` covers
    an absent OR unparsable version on either side, never a default: the
    ``build_row`` contract since slice 1 is that an unknown build must not read as
    parity.
    """

    state: str
    peer_version: str
    own_version: str


def compare_builds(peer_build: Any, own_build: Any) -> BuildComparison:
    """Order two build stamps — the ONE version comparison every surface reads.

    Extracted from :func:`build_row` so the ``peers`` row (``cli._peer_line`` and
    the agent digest) can state the same verdict in one segment without a second
    comparison drifting from this one: the same ``update.parse_version`` ordering
    and the same three-valued degradation — a stamp that is absent, or present but
    unparsable, on EITHER side comes back ``unknown``.
    """
    peer_version = (
        _bounded(peer_build.get("version") or "", 40) if isinstance(peer_build, Mapping) else ""
    )
    own_version = (
        _bounded(own_build.get("version") or "", 40) if isinstance(own_build, Mapping) else ""
    )
    if not peer_version:
        return BuildComparison("unknown", peer_version, own_version)
    from local_operator.update import parse_version

    mine = parse_version(own_version)
    theirs = parse_version(peer_version)
    if mine is None or theirs is None:
        return BuildComparison("unknown", peer_version, own_version)
    if theirs == mine:
        return BuildComparison("equal", peer_version, own_version)
    if theirs < mine:
        return BuildComparison("behind", peer_version, own_version)
    return BuildComparison("ahead", peer_version, own_version)


def build_suffix(comparison: BuildComparison) -> str:
    """The reachable row's build segment — ONE spelling for the CLI and the digest.

    ``""`` when nothing is known (the row must render exactly as it always has,
    never a default), the bare version when the parity is not actionable from here,
    and the ``lop-update`` hint ONLY when the peer is behind — that is the case
    whose remedy runs on the peer's side; an ahead peer's remedy would be on THIS
    device and stays ``ready``'s business.
    """
    if comparison.state == "unknown":
        return ""
    if comparison.state == "behind":
        # §2.9 (QA round 1, Q3's disposition): the suffix names the ACTION, not
        # the command — "run `lop-update` there" was the last command-shaped
        # remedy in this report.
        return (
            f"  build {comparison.peer_version} — behind this device "
            f"({comparison.own_version}); ask Local Operator to update it there"
        )
    return f"  build {comparison.peer_version}"


def build_row(
    member: Any,
    *,
    peer_build: Mapping[str, Any],
    own_build: Mapping[str, Any],
    peer_label: str,
) -> dict[str, Any]:
    """(b) Is the peer on the same build? Composed here, works for old peers."""
    comparison = compare_builds(peer_build, own_build)
    peer_version = comparison.peer_version
    own_version = comparison.own_version
    observed = {
        "this": own_version,
        "peer": peer_version,
        "peer_source_ref": _bounded(
            (peer_build.get("source_ref") if isinstance(peer_build, Mapping) else "") or "", 80
        ),
    }
    if not peer_version:
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_BUILD,
            ok=False,
            code=CODE_UNKNOWN,
            detail=(
                f"the build stamp {peer_label} runs did not arrive (it predates build "
                "reporting, or sent none), so build parity is not known"
            ),
            remedies=[f"ask Local Operator to update the build on {peer_label}, then re-check"],
            source=SOURCE_PEER,
            observed=observed,
        )
    if comparison.state == "unknown":
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_BUILD,
            ok=False,
            code=CODE_UNKNOWN,
            detail=(
                f"the build versions are not comparable (this device {own_version or 'unknown'}, "
                f"{peer_label} {peer_version}), so build parity is not known"
            ),
            remedies=[
                "ask Local Operator to update both devices to a comparable build, then " "re-check"
            ],
            source=SOURCE_PEER,
            observed=observed,
        )
    if comparison.state == "equal":
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_BUILD,
            ok=True,
            detail=f"{peer_label} runs the same build as this device ({peer_version})",
            source=SOURCE_PEER,
            observed=observed,
        )
    if comparison.state == "behind":
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_BUILD,
            ok=False,
            code=CODE_BEHIND,
            detail=(
                f"{peer_label} is behind this device: it runs {peer_version} and this device "
                f"runs {own_version} — work offloaded there runs its older build"
            ),
            remedies=[
                f"ask Local Operator to update it there ({peer_label} runs an older "
                "build), then re-check"
            ],
            source=SOURCE_PEER,
            observed=observed,
        )
    return _capability_row(
        device_id=member.device_id,
        device_name=peer_label,
        capability=CAPABILITY_BUILD,
        ok=False,
        code=CODE_AHEAD,
        detail=(
            f"{peer_label} is ahead ({peer_version} > {own_version}) — this side may lack "
            "capabilities the peer expects"
        ),
        remedies=["ask Local Operator to update this device, then re-check"],
        source=SOURCE_PEER,
        observed=observed,
    )


def git_row(member: Any, facts: Mapping[str, Any], *, peer_label: str) -> dict[str, Any]:
    """(c) Will offloaded work that commits have an author?"""
    fact = facts.get("git")
    if not isinstance(fact, Mapping):
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_GIT,
            ok=False,
            code=CODE_UNKNOWN,
            detail="the peer's answer did not carry this check",
            source=SOURCE_UNKNOWN,
        )
    name = _bounded(fact.get("user_name") or "", 200).strip()
    email = _bounded(fact.get("user_email") or "", 200).strip()
    if name and email:
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_GIT,
            ok=True,
            detail=(
                f"{peer_label} commits as {name} <{email}> (push credentials are a "
                "separate question this report does not cover yet; GitHub push "
                "through the mesh waits on a configured GitHub App — a short "
                "one-time setup, see the network guide)"
            ),
            source=SOURCE_PEER,
        )
    missing = " and ".join(
        label for label, value in (("user.name", name), ("user.email", email)) if not value
    )
    # THE REMEDY IS FILLED WITH THIS DEVICE'S OWN VALUES (design §3): an identity is
    # non-secret, one-time, per-device config, so the suggestion is made
    # copy-pasteable when this device has a value to suggest — verified against the
    # design note, which decided documented one-time setup rather than brokering
    # (brokering would cost a grants store, TTLs and a repair path for something
    # that is not a secret). A value this device lacks keeps the shipped ``"…"``
    # placeholder: inventing one for the peer would be a suggestion that cannot run,
    # and this row stays read-only — it is a suggestion the operator may replace.
    local = git_identity_fact()
    suggested_name = local.get("user_name") or "…"
    suggested_email = local.get("user_email") or "…"
    return _capability_row(
        device_id=member.device_id,
        device_name=peer_label,
        capability=CAPABILITY_GIT,
        ok=False,
        code=CODE_NO_GIT_IDENTITY,
        detail=(
            f"{peer_label} has no git identity ({missing} unresolved in its global "
            "config): offloaded work that commits will fail or commit under the wrong "
            "author"
        ),
        remedies=[
            f"on {peer_label} set its git author name and email (this device suggests "
            f"{suggested_name} <{suggested_email}>)"
        ],
        source=SOURCE_PEER,
    )


def mcp_servers_row(member: Any, facts: Mapping[str, Any], *, peer_label: str) -> dict[str, Any]:
    """(d) Does the peer have any user-scope MCP surface at all?"""
    fact = facts.get("mcp")
    if not isinstance(fact, Mapping):
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_MCP_SERVERS,
            ok=False,
            code=CODE_UNKNOWN,
            detail="the peer's answer did not carry this check",
            source=SOURCE_UNKNOWN,
        )
    servers = [row for row in fact.get("servers") or [] if isinstance(row, Mapping)]
    withheld = [str(item) for item in fact.get("withheld") or []]
    config_path = _bounded(fact.get("config_path") or "", 300) or "its mcp.json"
    if servers:
        count = len(servers)
        detail = (
            f"{peer_label} has {count} user-scope MCP "
            f"{'server' if count == 1 else 'servers'} declared in {config_path}"
        )
        if withheld:
            shown = ", ".join(withheld[:5])
            held = len(withheld)
            detail += (
                f" ({held} {'row' if held == 1 else 'rows'} withheld from this report: {shown})"
            )
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_MCP_SERVERS,
            ok=True,
            detail=detail,
            source=SOURCE_PEER,
        )
    detail = (
        f"{peer_label} has no user-scope MCP servers ({config_path} is missing or empty): "
        "work that needs MCP tooling cannot run there"
    )
    if withheld:
        held = len(withheld)
        # Pluralisation, not "row(s)": this is the register a person reads (round
        # 2, D5 — the same class as D4, on the rare branch below it).
        detail += (
            f" ({held} {'row' if held == 1 else 'rows'} withheld from this report "
            "as credential-shaped)"
        )
    return _capability_row(
        device_id=member.device_id,
        device_name=peer_label,
        capability=CAPABILITY_MCP_SERVERS,
        ok=False,
        code=CODE_NO_MCP_SERVERS,
        detail=detail,
        remedies=[
            f"push the MCP server definitions from the device that has them to "
            f"{peer_label} (definitions travel; secret values never do), or add the "
            f"servers on {peer_label} itself"
        ],
        source=SOURCE_PEER,
    )


def _name_list(names: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c`` — the register the tooling copy reads in."""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _tool_cell(entry: Any) -> tuple[str, str]:
    """``(state, path)`` from one tool's fact cell; anything else is ``unknown``.

    A missing or malformed cell is read as NEITHER ``absent`` (that would send
    the operator to install what may exist) NOR present: ``unknown`` is the
    only claim left standing when the fact cannot be read — the same
    discipline as the tri-state probes it carries.
    """
    if not isinstance(entry, Mapping):
        return "unknown", ""
    state = str(entry.get("state") or "")
    if state not in ("on_path", "off_path", "absent", "unknown"):
        return "unknown", ""
    return state, _bounded(entry.get("path") or "", 300)


def tooling_row(member: Any, facts: Mapping[str, Any], *, peer_label: str) -> dict[str, Any]:
    """(g) What command-line tooling would a lane find on this peer?

    The 2026-10-06 request: a lane on cloud-node-1 discovered mid-task that
    node/npm/make/docker were absent and ``gh`` was present but not on PATH and
    not signed in — and the report said nothing about any of it. Three DISTINCT
    remedy-bearing states, kept distinct here because they are different acts:
    an install, a one-line PATH fix, and a sign-in. The row is non-gating
    equipment (``NON_GATING_EQUIPMENT``): a device missing docker is not less
    onboarded, and the missing tool is repaired at its own point of use — the
    lane acquires tooling through the product's console flow, which asks first.
    """
    fact = facts.get("tooling")
    if not isinstance(fact, Mapping):
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_TOOLING,
            ok=False,
            code=CODE_UNKNOWN,
            detail="the peer's answer did not carry this check",
            source=SOURCE_UNKNOWN,
        )
    tools = fact.get("tools")
    tools = tools if isinstance(tools, Mapping) else {}
    cells = {name: _tool_cell(tools.get(name)) for name in TOOLING_TOOLS}
    absent = [name for name in TOOLING_TOOLS if cells[name][0] == "absent"]
    off_path = [name for name in TOOLING_TOOLS if cells[name][0] == "off_path"]
    unknown = [name for name in TOOLING_TOOLS if cells[name][0] == "unknown"]
    # The login question is asked only where a login could be USED: a gh that
    # is absent or unreadable makes "signed in?" moot, and the row already
    # names gh's own state. ``unknown`` covers both a login fact the answer
    # did not carry and an unreadable hosts file — neither may pass as signed in.
    auth = fact.get("gh_auth")
    auth = auth if isinstance(auth, Mapping) else {}
    gh_state = cells["gh"][0]
    login = "moot"
    login_reason = ""
    if gh_state in ("on_path", "off_path"):
        # NIT-1 (review round 1): only a real ``False`` reads "no stored login";
        # None, a missing key, or a malformed value (a skewed peer's shape) is
        # UNKNOWN WITH ITS REASON — a fact that cannot be read is never a claim.
        has_entry = auth.get("has_entry") if auth else None
        if has_entry is True:
            login = "ok"
        elif has_entry is False:
            login = "absent"
        else:
            login = "unknown"
            carried = auth.get("reason") if auth else None
            if isinstance(carried, str) and carried:
                login_reason = _bounded(carried, 80)
            elif auth and "has_entry" in auth:
                login_reason = "the reported value was not a boolean"
            else:
                login_reason = "the answer did not carry the login fact"
    if not absent and not off_path and not unknown and login == "ok":
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_TOOLING,
            ok=True,
            detail=(
                f"{peer_label} has the lane toolchain ({', '.join(TOOLING_TOOLS)}) and a "
                "stored GitHub CLI login"
            ),
            source=SOURCE_PEER,
        )
    clauses: list[str] = []
    if absent:
        # The clause carries the PROBE'S SCOPE, never "not installed" (review
        # round 1, MINOR-1): neither probe sees everywhere (an nvm-managed node
        # off PATH reads absent), so the sentence says where the check looked,
        # and the remedy below covers the installed-but-invisible possibility.
        clauses.append(f"has no {_name_list(absent)} on its PATH or in ~/.local/bin")
    if off_path:
        clauses.append(f"has {_name_list(off_path)} installed but not on its PATH")
    if unknown:
        clauses.append(f"could not complete its tooling check for {_name_list(unknown)}")
    if login == "absent":
        clauses.append("has no stored GitHub CLI login")
    elif login == "unknown":
        clause = "could not read its stored GitHub CLI login"
        if login_reason:
            clause += f" ({login_reason})"
        clauses.append(clause)
    if absent or off_path or login == "absent":
        tail = ": offloaded work that needs them will fail there"
    else:
        tail = ": whether that would block offloaded work is not known"
    # The code cell picks the FIRST fix in this order — an install, then a PATH
    # fix, then the login, then "could not read" last: a concrete repair a
    # reader can start on outranks a probe that could not answer.
    if absent:
        code = CODE_NOT_INSTALLED
    elif off_path:
        code = CODE_NOT_ON_PATH
    elif login == "absent":
        code = CODE_NOT_AUTHENTICATED
    else:
        code = CODE_UNKNOWN
    local_bin = _bounded(fact.get("local_bin_dir") or "", 300)
    remedies: list[str] = []
    if absent:
        remedies.append(
            f"ask Local Operator to install {_name_list(absent)} on {peer_label} — or, if "
            "one is already installed off its PATH there, to put it on the PATH (it asks "
            "before changing anything there)"
        )
    for name in off_path:
        remedy = f"on {peer_label}, {name} is installed at {cells[name][1]} but not on its PATH"
        if local_bin:
            remedy += f" — ask Local Operator to add {local_bin} to the PATH there"
        else:
            remedy += " — ask Local Operator to put its folder on the PATH there"
        remedy += " (work can use that path directly meanwhile)"
        remedies.append(remedy)
    if login == "absent":
        remedies.append(
            f"gh on {peer_label} is not signed in — ask Local Operator to sign it in there"
        )
    if unknown or login == "unknown":
        remedies.append(f"check the tooling on {peer_label} and re-check")
    return _capability_row(
        device_id=member.device_id,
        device_name=peer_label,
        capability=CAPABILITY_TOOLING,
        ok=False,
        code=code,
        detail=f"{peer_label} " + "; ".join(clauses) + tail,
        remedies=remedies,
        source=SOURCE_PEER,
        observed={"absent": absent, "off_path": off_path, "unknown": unknown, "gh_login": login},
    )


def _holder_names(placement: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for row in placement.get("holders") or []:
        if isinstance(row, Mapping) and row.get("device"):
            names.append(str(row["device"]))
    return names


def _owner_label(placement: Mapping[str, Any], *, fallback: str = "") -> str:
    # A name another device chose (it rides the placement document): through the
    # shape guard like every other carried string (round 1, MINOR-2).
    return (
        _safe_text(
            str(placement.get("owner_device_name") or "")
            or str(placement.get("owner_device") or ""),
            limit=120,
        )
        or fallback
    )


def _observation_failure(observation: Mapping[str, Any]) -> str:
    """The rendered sentence for a refused borrow memory, or ``""``.

    ``render_broker_error`` is the credentials slice's ONE sentence builder for
    a refused grant, so the observation's wording cannot drift from what the
    operator reads when the same refusal happens for real.
    """
    status = str(observation.get("status") or "")
    if status not in ("owner_offline", "grant_invalid"):
        return ""
    from local_operator.network.credentials.messages import render_broker_error
    from local_operator.network.credentials.types import BrokerError

    last_seen_s: float | None = None
    stamp = observation.get("last_grant_at")
    try:
        if stamp:
            last_seen_s = max(0.0, time.time() - float(stamp))
    except (TypeError, ValueError):
        last_seen_s = None
    error = BrokerError(
        code=status,
        message=_bounded(observation.get("reason") or "", 200),
        owner_device=_bounded(observation.get("owner_device") or "", 80),
        owner_device_name="",
    )
    return render_broker_error(
        error,
        owner_name=str(observation.get("owner_device") or "the owner device"),
        last_seen_s=last_seen_s,
    )


def model_credential_row(
    member: Any,
    facts: Mapping[str, Any],
    *,
    viewer: Viewer,
    peer_label: str,
) -> dict[str, Any]:
    """(e) Can the peer's default model be served — own login or a borrow?"""
    model = facts.get("default_model")
    if not isinstance(model, Mapping):
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_MODEL_CREDENTIAL,
            ok=False,
            code=CODE_UNKNOWN,
            detail="the peer's answer did not carry this check",
            source=SOURCE_UNKNOWN,
        )
    hosting = _bounded(model.get("hosting") or "", 120)
    resolved = bool(model.get("resolved"))
    if not hosting or not resolved:
        reason = _bounded(model.get("reason") or "", 200)
        if reason.startswith("its config could not be read"):
            sentence = (
                f"{peer_label}'s config could not be read, so its default model is "
                "unknown — work offloaded there cannot start"
            )
        else:
            sentence = (
                f"{peer_label} has no default model configured"
                + (f" ({reason})" if reason else "")
                + ": work offloaded there cannot start"
            )
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_MODEL_CREDENTIAL,
            ok=False,
            code=CODE_NOT_CONFIGURED,
            detail=sentence,
            remedies=[
                "set the peer's default (`/model default` in its window, or "
                "`local-operator config edit model_name <model>` there), then its "
                "provider needs a login or a share"
            ],
            source=SOURCE_PEER,
        )
    provider = _bounded(facts.get("provider") or hosting, 120)
    held = facts.get("has_local")
    placement = facts.get("credential_placement")
    placement = placement if isinstance(placement, Mapping) else {}
    observation = placement.get("observation")
    observation = observation if isinstance(observation, Mapping) else {}
    if held is True:
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_MODEL_CREDENTIAL,
            ok=True,
            detail=f"{peer_label} serves {provider} from its own login",
            source=SOURCE_PEER,
        )
    owner = str(placement.get("owner_device") or "")
    peer_is_holder = member.device_id in _holder_names(placement)
    if owner and peer_is_holder:
        observed_failure = _observation_failure(observation)
        if observed_failure:
            return _capability_row(
                device_id=member.device_id,
                device_name=peer_label,
                capability=CAPABILITY_MODEL_CREDENTIAL,
                ok=False,
                code=CODE_OBSERVED_FAILURE,
                detail=(
                    f"the credential {peer_label} would borrow for {provider} is not "
                    f"reachable. {observed_failure}"
                ),
                remedies=[
                    f"re-check after the owner is reachable, or sign in to {provider} on "
                    f"{peer_label}"
                ],
                source=SOURCE_PEER,
            )
        if owner == viewer.device_id:
            return _capability_row(
                device_id=member.device_id,
                device_name=peer_label,
                capability=CAPABILITY_MODEL_CREDENTIAL,
                ok=True,
                detail=(
                    f"{peer_label} borrows {provider} from this device (a grant is issued "
                    "per request; nothing to copy)"
                ),
                source=SOURCE_PEER,
            )
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_MODEL_CREDENTIAL,
            ok=True,
            detail=(
                f"{peer_label} borrows {provider} from "
                f"{_owner_label(placement, fallback='another device')} (not verified from here)"
            ),
            source=SOURCE_PEER,
        )
    if owner and owner == member.device_id:
        # The placement says the PEER owns this key, yet no local row was found
        # on it (``held`` is not True, or this branch was reached first). The
        # two documents disagree; report the disagreement rather than pick one.
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_MODEL_CREDENTIAL,
            ok=False,
            code=CODE_UNKNOWN,
            detail=(
                f"the placement entry for {provider} names {peer_label} as its owner, but "
                "no credential row was found there — the two do not agree yet"
            ),
            remedies=[f"sign in to {provider} on {peer_label} and re-check"],
            source=SOURCE_PEER,
        )
    if owner and not peer_is_holder:
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_MODEL_CREDENTIAL,
            ok=False,
            code=CODE_NOT_SHARED,
            detail=(
                f"{provider} is shared in this network, but {peer_label} is not among its "
                "holders: a borrow there would be refused"
            ),
            remedies=[
                (
                    f"share it with {peer_label} (run here)"
                    if owner == viewer.device_id
                    else f"ask the device that owns {provider} to add {peer_label} as a holder"
                )
            ],
            source=SOURCE_PEER,
        )
    mine = viewer.provider_rows(provider)
    if mine:
        # A SHARE THIS DEVICE ALREADY MADE IS NOT "NOT SHARED". ``pull_placement``
        # is paid on an explicit act (``lop network credentials``, client.py's
        # own docstring), never on the provider path — so a peer that has not
        # pulled yet answers with an EMPTY placement, and a report that stopped
        # here would (a) deny a share that is on the books on THIS device and
        # (b) send the operator to repeat a command that already ran. Read this
        # device's own document: a peer that is a holder THERE is missing the
        # pull, not the share.
        own_entry = viewer.placement(provider=provider)
        if member.device_id in _holder_names(own_entry):
            return _capability_row(
                device_id=member.device_id,
                device_name=peer_label,
                capability=CAPABILITY_MODEL_CREDENTIAL,
                ok=False,
                code=CODE_NOT_SHARED,
                detail=(
                    f"the share for {provider} is on this network's books, but {peer_label} "
                    "has not pulled the placement yet: nothing would reach it until it does"
                ),
                remedies=[
                    f"pull the placement on {peer_label} (it pulls the share), then re-check"
                ],
                source=SOURCE_LOCAL,
            )
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_MODEL_CREDENTIAL,
            ok=False,
            code=CODE_NOT_SHARED,
            detail=(
                f"this device is signed in to {provider} but does not share it with "
                f"{peer_label}, so nothing would reach it"
            ),
            remedies=[f"share it with {peer_label} (run here)"],
            source=SOURCE_LOCAL,
        )
    if held is None or mine is None:
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_MODEL_CREDENTIAL,
            ok=False,
            code=CODE_UNKNOWN,
            detail=(
                "a credential store could not be read (this side or the peer's), so "
                "whether this provider can be served is not known"
            ),
            remedies=[f"check the credential store on {peer_label} and here, then re-check"],
            source=SOURCE_UNKNOWN,
        )
    return _capability_row(
        device_id=member.device_id,
        device_name=peer_label,
        capability=CAPABILITY_MODEL_CREDENTIAL,
        ok=False,
        code=CODE_NO_CREDENTIAL,
        detail=(
            f"neither {peer_label} nor this device holds a credential for {provider}: "
            "offloaded work on that model cannot start"
        ),
        remedies=[
            f"sign in to {provider} here and share it, or sign in on {peer_label} "
            "to use its own account"
        ],
        source=SOURCE_LOCAL,
    )


def mcp_credential_rows(
    member: Any,
    facts: Mapping[str, Any],
    *,
    viewer: Viewer,
    peer_label: str,
) -> list[dict[str, Any]]:
    """(f) Per user-scope HTTP/SSE server: does a login path exist for the peer?"""
    fact = facts.get("mcp")
    if not isinstance(fact, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for server in fact.get("servers") or []:
        if not isinstance(server, Mapping):
            continue
        if str(server.get("transport") or "") not in ("http", "sse"):
            # Stdio servers are listed, not credential-checked (deferral §7):
            # their secrets live in the spawn environment, a different question.
            continue
        url = _bounded(server.get("url") or "", 300)
        name = _bounded(server.get("name") or "", 120) or url
        if not url:
            continue
        rows.append(
            _mcp_credential_row(
                member,
                url=url,
                name=name,
                server=server,
                viewer=viewer,
                peer_label=peer_label,
            )
        )
    return rows


def _mcp_credential_row(
    member: Any,
    *,
    url: str,
    name: str,
    server: Mapping[str, Any],
    viewer: Viewer,
    peer_label: str,
) -> dict[str, Any]:
    has_row = server.get("has_row")
    if has_row is True:
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_MCP_CREDENTIAL,
            ok=True,
            detail=f"{peer_label} holds its own login for {name} ({url})",
            source=SOURCE_PEER,
            observed={"server": name, "url": url},
        )
    placement = viewer.placement(mcp_url=url)
    owner = str(placement.get("owner_device") or "")
    peer_is_holder = member.device_id in _holder_names(placement)
    if owner and peer_is_holder:
        # THE PULL GAP, per server: this device's own document says the peer is a
        # holder, but the peer's OWN answer says it has no placement entry for
        # the url — the share has not reached it yet, and a borrow there would
        # find nothing until it pulls. Both halves are observed facts; the row
        # reports the gap rather than either half's comfortable reading.
        if not server.get("has_placement"):
            return _capability_row(
                device_id=member.device_id,
                device_name=peer_label,
                capability=CAPABILITY_MCP_CREDENTIAL,
                ok=False,
                code=CODE_NOT_SHARED,
                detail=(
                    f"the login for {name} is shared with {peer_label} on this device's "
                    "books, but it has not pulled the placement yet: nothing would reach "
                    "it until it does"
                ),
                remedies=[
                    f"pull the placement on {peer_label} (it pulls the share), then re-check"
                ],
                source=SOURCE_LOCAL,
                observed={"server": name, "url": url},
            )
        if owner == viewer.device_id:
            return _capability_row(
                device_id=member.device_id,
                device_name=peer_label,
                capability=CAPABILITY_MCP_CREDENTIAL,
                ok=True,
                detail=(
                    f"{peer_label} borrows the login for {name} from this device (a grant "
                    "is issued per request; nothing to copy)"
                ),
                source=SOURCE_PEER,
                observed={"server": name, "url": url},
            )
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_MCP_CREDENTIAL,
            ok=True,
            detail=(
                f"{peer_label} borrows the login for {name} from "
                f"{_owner_label(placement, fallback='another device')} (not verified from here)"
            ),
            source=SOURCE_PEER,
            observed={"server": name, "url": url},
        )
    if owner and not peer_is_holder:
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_MCP_CREDENTIAL,
            ok=False,
            code=CODE_NOT_SHARED,
            detail=(
                f"a login for {name} is shared in this network, but {peer_label} is not "
                "among its holders: a borrow there would be refused"
            ),
            remedies=[
                (
                    f"share it with {peer_label} (run here)"
                    if owner == viewer.device_id
                    else (
                        f"ask the device that owns the login for {name} to add "
                        f"{peer_label} as a holder"
                    )
                )
            ],
            source=SOURCE_PEER,
            observed={"server": name, "url": url},
        )
    if viewer.holds_mcp_login(url):
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_MCP_CREDENTIAL,
            ok=False,
            code=CODE_NOT_SHARED,
            detail=(
                f"this device holds a login for {name} but does not share it with "
                f"{peer_label}, so nothing would reach it"
            ),
            remedies=[f"share it with {peer_label} (run here)"],
            source=SOURCE_LOCAL,
            observed={"server": name, "url": url},
        )
    if has_row is None:
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_MCP_CREDENTIAL,
            ok=False,
            code=CODE_UNKNOWN,
            detail=(
                f"{peer_label}'s credential store could not be read, so whether it holds "
                f"a login for {name} is not known"
            ),
            remedies=[f"check the credential store on {peer_label}, then re-check"],
            source=SOURCE_UNKNOWN,
            observed={"server": name, "url": url},
        )
    if server.get("auth_declared"):
        return _capability_row(
            device_id=member.device_id,
            device_name=peer_label,
            capability=CAPABILITY_MCP_CREDENTIAL,
            ok=False,
            code=CODE_NO_CREDENTIAL,
            detail=f"this device has no MCP login for {url}; {mcp_login_remedy(url)}",
            remedies=[f"{mcp_login_remedy(url)}, then share it with {peer_label}"],
            source=SOURCE_LOCAL,
            observed={"server": name, "url": url},
        )
    # NEITHER SIDE HOLDS A ROW AND THE CONFIG DECLARES NO AUTH BLOCK. The copy
    # must not claim certainty about the server's auth — plenty of servers need
    # no sign-in at all — so it opens with "if" and keeps the same command the
    # certain case names.
    return _capability_row(
        device_id=member.device_id,
        device_name=peer_label,
        capability=CAPABILITY_MCP_CREDENTIAL,
        ok=False,
        code=CODE_NO_CREDENTIAL,
        detail=(
            f"if the `{name}` server needs a sign-in, this device has no MCP login for {url} — "
            f"{mcp_login_remedy(url)}"
        ),
        remedies=[f"{mcp_login_remedy(url)}, then share it with {peer_label}"],
        source=SOURCE_LOCAL,
        observed={"server": name, "url": url},
    )


# ---------------------------------------------------------------------------
# Reachability (this side; observed facts, never inference)
# ---------------------------------------------------------------------------


def _route_observation(host: str, port: int) -> dict[str, Any]:
    from local_operator.network import addresses as addresses_mod

    try:
        return dict(addresses_mod.route_observation(host, port))
    except Exception:  # noqa: BLE001 — an unobservable route is `null`, not an error
        return {"source_address": None, "interface": None, "error": ""}


def _observed_route_text(observed: Mapping[str, Any]) -> str:
    source = observed.get("source_address")
    interface = observed.get("interface")
    if source and interface:
        return f" (this device routes to it from {source} via {interface})"
    if source:
        return f" (this device routes to it from {source})"
    return ""


#: The outcomes that mean "this declared address did not lead to the peer from
#: here": a refused connection, silence, no route, an unresolvable or
#: undiallable address. When the member answers at ANOTHER address this run
#: IDENTIFIED, one of these renders INFORMATIONAL (see ``_reachability_rows``):
#: the relay advertises the host's own interface addresses — on EC2 that
#: includes the VPC-private one — and an address no remote device can use is not
#: evidence about the peer's health. With no usable address anywhere the row
#: keeps failing, which is the honest negative this verb must never lose.
#: Deliberately NOT here: ``connected_unverified``/``connected_elsewhere`` (a
#: stranger answered the port — a security fact), ``handshake_failed`` (the
#: address answered and no link came up — a defect at that address),
#: ``not_attempted`` (unproven), and ``no_endpoint`` (nothing was dialled).
_INFORMATIONAL_OUTCOMES: frozenset[str] = frozenset(
    {"refused", "no_answer", "no_answer_elsewhere", "no_route", "resolve_failed", "bad_endpoint"}
)


def _private_address_note(endpoint: str) -> str:
    """The provenance clause for an address no remote device can use.

    ``relay.advertise_endpoints`` publishes the HOST'S OWN interface addresses,
    and on EC2 that set includes the VPC-private address — which is exactly the
    kind of address a remote viewer cannot dial. Naming that provenance is what
    turns the row from "another failure" into a fact a reader can act on
    (use the other address). Empty for anything not private/link-local, where
    the failure's own sentence is the honest explanation.

    THE NOTE NAMES THE KIND, NOT AN ABSOLUTE (design round 1, D8): "remote
    devices cannot use it" overclaimed — a device on the same network could —
    while the sentence this rides says "not remote-usable from this device",
    the scope that is actually true. The kind is what this clause contributes;
    the scope lives in the clause it rides.
    """
    host, _port = _split_endpoint(endpoint)
    try:
        octets = socket.inet_aton(host)
    except OSError:
        return ""
    first, second = octets[0], octets[1]
    if first == 10 or (first == 172 and 16 <= second <= 31) or (first == 192 and second == 168):
        return " (the machine's own private address)"
    if first == 169 and second == 254:
        return " (a link-local address)"
    return ""


def reachability_reading(row: Mapping[str, Any]) -> str:
    """The human sentence for a reachability row — this verb's own register.

    Kept HERE rather than routed through ``resume.doctor_detail_words``, and the
    reason is the whole point of this report: doctor's renderer maps every
    ``connect_failed:`` stage — including ``ConnectionRefusedError`` — to
    "nothing answered at that address", which is exactly the conflation this
    slice exists to end. A refused connection means the host is UP and nothing
    is listening on that port; an operator told "nothing answered" goes to check
    a machine that already answered. Doctor's own output and tables are
    deliberately untouched (their pins stay exactly as they were); this function
    reads doctor's CODES (``detail``) plus the observed facts and composes the
    sentence this report owes.

    PEER CLAIMS ARE VERIFIED CLAIMS (design round 1, D1). A TCP accept is not a
    peer answer — any listener on a declared address satisfies it — so "the peer
    answered" appears only for the socket this report DIALED and identified (or
    an address) a completed handshake lives at (``connected_link``: the live
    link's own address, identified when that link came up). An accept nobody
    identified renders the observed fact and nothing more — EXCEPT under a live
    link this report cannot PIN to a declared address (an inbound link's source
    socket, or an outbound address the peer no longer declares): there the peer
    claim rides the link fact (the peer is up; ``connected_unpinned``) with no
    address named at all, and the peer's own reachable address is not reddened
    because of the link's shape (round 2, R2-1/Q-3; round 3, R3-2 — the token
    names the condition, not a direction).
    """
    observed = row.get("observed")
    observed = observed if isinstance(observed, Mapping) else {}
    outcome = str(observed.get("outcome") or "")
    detail = str(row.get("detail") or "")
    winner = str(observed.get("winner") or "")
    winner_verified = bool(observed.get("winner_verified"))
    link_address = str(observed.get("link_address") or "")
    if observed.get("out_of_scope"):
        # EXCLUDED FROM THE DECISION, VISIBLE IN THE READING (F9 ruling,
        # 2026-10-04): the candidate was never askable from this device's
        # network position, so its row must not read as a failure — and its raw
        # failure cause must not ride along either. The shared clause IS the
        # reading, one spelling for every reader.
        return out_of_scope_clause(row)
    if observed.get("informational"):
        # REPORTED, NOT FAILED (drill finding, 2026-10-03; design round 1, D1):
        # this address did not lead to the peer from this device, but the peer
        # ANSWERS at an address this run identified — so the member is reachable
        # and the row must not read as a failure of the check. And its raw
        # failure cause must not ride along either: "nothing is listening on
        # that port" names a wrong action in the exact row this flip exists
        # for. The shared clause IS the reading (D3) — the same string doctor's
        # renderers hang, so the two surfaces cannot drift.
        return informational_clause(row)
    if row.get("probed") is False:
        # The no-relay fallback: nothing was dialled, and its ``detail`` is
        # already a sentence for a person ("not probed: no relay is running …").
        return detail or "not probed"
    if outcome == "connected":
        return "the peer answered at this address"
    if outcome == "connected_link":
        return "the peer's link runs at this address"
    if outcome == "connected_unpinned":
        # R2-1/Q-3: the accept is the observed fact; the peer claim rides the
        # live link this report cannot pin to an address — namable neither as a
        # dialled host nor as a source socket — so none is named. THE VERB IS
        # ``something``, like ``connected_unverified`` (design round 1, D5): the
        # accept at this address is un-identified, and "this address accepted"
        # read as a self-contradiction beside a claim that the address cannot be
        # attributed at all. THE WHY LIVES IN ``detail`` (D5): the link's
        # recorded address is not among the peer's declared endpoints — an
        # address fact, not a caveat on the mesh, and never a direction (the
        # token names the condition).
        return (
            "the peer is up (its link is live); something accepted a TCP connection "
            "at this address"
        )
    if outcome == "connected_unverified":
        # THE D1 SENTENCE: something accepted a TCP connection, and nothing
        # identified it as the peer. The link's address is still a true fact and
        # helps the reader tell "a stranger holds this port" from "my peer is
        # fine on another address".
        sentence = (
            "something accepted a TCP connection at this address; it was not "
            "identified as the peer"
        )
        if link_address:
            sentence += f" (the peer's link runs at {link_address})"
        return sentence
    if outcome == "refused":
        return (
            "something answered this address and refused the connection — the host is up; "
            "nothing is listening on that port"
        )
    if outcome == "no_answer_elsewhere":
        if winner and winner_verified:
            where = (
                f"its link runs at {winner}" if winner == link_address else f"it answered {winner}"
            )
            return f"the peer is up ({where}); this address did not answer"
        if link_address:
            return f"the peer is up (its link runs at {link_address}); this address did not answer"
        if observed.get("link_unpinned"):
            # The link fact without an address (R2-1/Q-3): the peer IS up; only
            # this address failed.
            return "the peer is up (its link is live); this address did not answer"
        return "this address did not answer"
    if outcome == "no_answer":
        return "nothing answered this address before the budget ran out" + _observed_route_text(
            observed
        )
    if outcome == "no_route":
        return "this device's own routing said the address is unreachable from here"
    if outcome == "resolve_failed":
        return "the address does not resolve from this device"
    if outcome == "handshake_failed":
        # The dial REACHED the address and the link did not come up; the tail is
        # a refusal code the peer's own build decides, so it reads in doctor's
        # refusal words rather than as a bare wire token (round 1 reconciliation;
        # "the link was refused" is doctor's own wording for exactly this family).
        # ONE family reads in THIS VERB's words instead: doctor's gloss for the
        # report-budget state names the doctor's own clock ("the doctor ran out
        # of time before the handshake"), and the clock that actually expired
        # here is this report's (round 2, R2-2).
        from local_operator.network import relay as relay_mod
        from local_operator.resume import doctor_detail_words

        if detail.partition(":")[0] == relay_mod.HANDSHAKE_NOT_ATTEMPTED:
            return "the address answered, but the report ran out of time before the handshake"
        words = doctor_detail_words(detail) or detail
        return f"the address answered, but no link came up ({words})"
    if outcome == "bad_endpoint":
        return "the address it publishes cannot be dialled"
    if outcome == "not_attempted":
        return "the report ran out of time before this address was tried"
    if outcome == "no_endpoint":
        # Doctor's own vocabulary for a member that published nothing (round 1,
        # MINOR-1): the fall-through below printed the bare token into the human
        # line and the agent digest.
        return "no address published for it — nothing was dialled"
    if outcome == "connected_elsewhere":
        sentence = (
            "something accepted a TCP connection at this address; it was not "
            "identified as the peer"
        )
        if winner and winner_verified:
            sentence += (
                f" (the peer's link runs at {winner})"
                if winner == link_address
                else f" (the peer answered on {winner})"
            )
        elif link_address:
            sentence += f" (the peer's link runs at {link_address})"
        return sentence
    return detail or "no reading for this row"


def _classify_outcome(detail: str, *, route_ok: bool, winner_elsewhere: bool) -> str:
    from local_operator.network import relay as relay_mod

    if detail == relay_mod.DETAIL_NO_ANSWER:
        return "no_answer_elsewhere" if winner_elsewhere else "no_answer"
    if detail == relay_mod.DETAIL_NOT_ATTEMPTED:
        return "not_attempted"
    if detail == relay_mod.DETAIL_BAD_ENDPOINT:
        return "bad_endpoint"
    if detail.startswith(relay_mod.CONNECT_FAILED_PREFIX):
        klass = detail.split(":", 1)[1]
        if klass == "ConnectionRefusedError":
            return "refused"
        if klass in ("TimeoutError", "socket.timeout"):
            return "no_answer_elsewhere" if winner_elsewhere else "no_answer"
        if klass == "gaierror":
            return "resolve_failed"
        if not route_ok:
            # The kernel could not resolve a route at all: that is this side's
            # own routing talking, and it is a different remedy from a black
            # hole on the path.
            return "no_route"
        return "no_answer_elsewhere" if winner_elsewhere else "no_answer"
    return "no_answer_elsewhere" if winner_elsewhere else "no_answer"


def _reachability_remedies(
    outcome: str, *, peer_label: str, observed: Mapping[str, Any]
) -> list[str]:
    # WHOSE CLAIM IS IT: "the peer answered elsewhere" is advice built on a PEER
    # fact, so it rides only on an identified answer — the socket this run
    # dialled and handshaked, or the live link's own address (identified when
    # that link came up). An accept nobody identified must not send anyone to
    # "the working address" (design round 1, D1: a non-peer listener holding a
    # candidate port produced exactly that advice).
    winner = str(observed.get("winner") or "")
    winner_verified = bool(observed.get("winner_verified"))
    link_address = str(observed.get("link_address") or "")
    where = winner if (winner and winner_verified) else link_address
    if outcome == "refused":
        if where:
            # The peer IS up — it answered at an identified address — so "start
            # the relay" would be the wrong advice about THIS address; the
            # address is what needs fixing.
            return [f"point {peer_label} at the working address: it answered on {where}"]
        if observed.get("link_unpinned"):
            # The peer is up (its link is live) but the link names no address
            # this report may cite (R2-1/Q-3), so the advice is about the
            # address alone — and never "start the relay" for a peer that is
            # demonstrably running.
            return [
                f"the peer is up (its link is live), so this address is stale — "
                f"fix or drop it on {peer_label}"
            ]
        return [f"start {peer_label}'s relay, then re-check"]
    if outcome == "no_answer":
        admits = ""
        source = observed.get("source_address")
        interface = observed.get("interface")
        if source and interface:
            admits = f"this device's address ({source}, via {interface})"
        elif source:
            admits = f"this device's address ({source})"
        else:
            admits = "this device's address"
        return [
            f"on {peer_label} check it is up, then check that any firewall or security "
            f"group on the path admits {admits}"
        ]
    if outcome == "no_answer_elsewhere":
        if where:
            return [f"point the peer at the working address: it answered on {where}"]
        if observed.get("link_unpinned"):
            return [
                f"the peer is up (its link is live); this address looks stale — "
                f"fix or drop it on {peer_label}"
            ]
        return [f"re-check once {peer_label} answers at this address"]
    if outcome == "connected_unverified":
        return [f"check that {peer_label}'s relay is what listens at this address, then re-check"]
    if outcome == "connected_elsewhere":
        return [f"if {peer_label} should serve this address, check nothing else has taken it"]
    if outcome == "no_route":
        return ["fix this device's routing or VPN for this address, then re-check"]
    if outcome == "resolve_failed":
        return [
            "have the peer publish an address that resolves from here (or fix DNS on this device)"
        ]
    if outcome == "handshake_failed":
        # Two clocks can produce this state and the remedy differs by WHICH one
        # expired: the peer's handshake never came up vs THIS report's budget
        # running out after the address answered (round 3, D6). The stage word
        # rides `observed` so the remedy can split on it — same distinction the
        # reading makes.
        from local_operator.network import relay as relay_mod

        if observed.get("dial_stage") == relay_mod.HANDSHAKE_NOT_ATTEMPTED:
            return [
                "re-run this report — the address answered; the budget that ran "
                "out was this report's"
            ]
        return [f"re-check once {peer_label}'s relay answers a handshake again"]
    return []


def _identified_endpoints(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """The addresses this report IDENTIFIED an answer at — both row dialects.

    ``connected`` handshaked THIS run; ``connected_link`` is the live link's own
    declared address. ``connected_unpinned`` is excluded on purpose — a live
    link with no pinnable address is a real answer that names nothing, so it
    cannot vouch for another row (round 4's whole distinction). DOCTOR'S
    DIALECT counts a ``handshake`` row that came back ok: that row IS the
    verification at its endpoint, and a connected accept whose handshake the
    budget never ran (the ``handshake_not_attempted`` row) is not one — the
    same "an accept nobody identified must not vouch" rule, one row shape over.
    """
    usable: list[str] = []
    for row in rows:
        outcome = str((row.get("observed") or {}).get("outcome") or "")
        if outcome in ("connected", "connected_link"):
            usable.append(str(row["endpoint"]))
        elif row.get("check") == "handshake" and row.get("ok") and row.get("endpoint"):
            usable.append(str(row["endpoint"]))
    return usable


def _did_not_lead(row: Mapping[str, Any]) -> bool:
    """Whether this reachability row is an address that did not lead to the peer.

    The two dialects carry their facts in different fields and both are read
    here rather than normalized: the readiness row's ``observed.outcome`` (the
    closed set above) and the doctor row's ``detail`` — the probe's own codes,
    read through the PRODUCER's constants so a rename cannot drift this
    classifier away from the strings the probe writes. ``not_attempted`` is in
    neither family: unproven is not a fact about the address.
    """
    outcome = str((row.get("observed") or {}).get("outcome") or "")
    if outcome:
        return outcome in _INFORMATIONAL_OUTCOMES
    detail = str(row.get("detail") or "")
    if not detail:
        return False
    from local_operator.network import relay as relay_mod

    return detail in (
        relay_mod.DETAIL_NO_ANSWER,
        relay_mod.DETAIL_BAD_ENDPOINT,
    ) or detail.startswith(relay_mod.CONNECT_FAILED_PREFIX)


def informational_clause(row: Mapping[str, Any]) -> str:
    """The sentence an informational row reads in place of its raw failure.

    ONE SPELLING FOR EVERY READER (design round 1, D1/D3): ``ready``'s reading
    returns this string verbatim, and doctor's two renderers
    (``cli._cmd_doctor`` and the agent digest in ``network/tool.py``) hang it
    after the row's detail words — so order and punctuation cannot drift
    between the surfaces, and the doctor reader keeps the provenance note the
    readiness reader gets.

    THE RAW FAILURE IS NOT PARROTED (D1): the failed reading's own cause
    ("nothing is listening on that port") points the reader at a wrong action
    in the exact row this flip exists for, so the clause REPLACES it — stating
    what is true of the address (not remote-usable from here; the kind of
    address it is, when that is known; and the address that works). The raw
    outcome stays a machine fact in ``observed``/``detail``.

    Empty for a row the flip did not mark.
    """
    observed = row.get("observed")
    if not isinstance(observed, Mapping) or not observed.get("informational"):
        return ""
    usable = ", ".join(str(item) for item in observed.get("usable_elsewhere") or ())
    note = _private_address_note(str(row.get("endpoint") or ""))
    return f"not remote-usable from this device{note}; the peer is reachable at {usable}"


def out_of_scope_clause(row: Mapping[str, Any]) -> str:
    """The sentence an out-of-scope row reads in place of its raw failure.

    ONE SPELLING FOR EVERY READER (F9 ruling, 2026-10-04; same pattern as
    :func:`informational_clause`): ``ready``'s reading returns this string
    verbatim. The raw failure is NOT parroted — "nothing answered this address"
    points at a wrong action in the exact rows this rule exists for; the fact
    that matters is that the question cannot be asked from here at all.
    """
    host, _port = _split_endpoint(str(row.get("endpoint") or ""))
    return (
        f"out of scope: {host} is a private (RFC1918) address of a network this "
        "device is not on; the peer cannot be asked at it from here"
    )


def mark_informational(rows: list[dict[str, Any]]) -> None:
    """Mark "did not lead to the peer from here" rows informational, in place.

    ONE SEMANTICS, TWO PRODUCERS (drill finding, 2026-10-03): ``ready``'s
    ``_reachability_rows`` and ``doctor``'s ``RelayServer._probe_member`` emit
    different row shapes for the same fact — a declared address that did not
    lead to the peer from here — and both must report it as INFORMATION when
    the member answers at an address this report identified: the relay
    advertises the host's own interface addresses (on EC2, the VPC-private
    one), and that kind of address is not evidence about the peer's health. THE
    FLIP REQUIRES A VERIFIED ANSWER: with none, every row keeps failing — the
    honest negative (a member nothing answers for must never read healthy).
    Capability rows and the doctor's credential-repair rows are different
    checks and are never touched. The flipped row keeps its observed facts and
    gains ``informational`` plus the addresses that ARE usable; readiness
    remedies are cleared because action items under an ok row misread — the
    reading names the address to use instead.
    """
    usable = _identified_endpoints(rows)
    if not usable:
        return
    for row in rows:
        if row["ok"]:
            continue
        if str(row.get("check") or "") != "reachability":
            continue
        if not _did_not_lead(row):
            continue
        row["ok"] = True
        observed = row.setdefault("observed", {})
        observed["informational"] = True
        observed["usable_elsewhere"] = list(usable)
        if "remedies" in row:
            row["remedies"] = []


#: The outcomes in which the candidate gave NO answer at all — the only ones a
#: scope exclusion may cover. A refusal or a started handshake PROVES something
#: at that address answered, which refutes "cannot route to it"; a connected row
#: is a positive and is never touched. ``not_attempted`` is in: for a candidate
#: off this device's networks the report's budget is not what made the question
#: unanswerable.
_SCOPE_OUTCOMES: frozenset[str] = frozenset(
    {
        "no_answer",
        "no_answer_elsewhere",
        "no_route",
        "resolve_failed",
        "bad_endpoint",
        "not_attempted",
    }
)


def _no_answer_row(row: Mapping[str, Any]) -> bool:
    """Whether this reachability row recorded NO answer at all, in either dialect.

    The readiness dialect rides ``observed.outcome`` (the ``_SCOPE_OUTCOMES``
    set); the doctor dialect rides ``detail`` — the probe's own codes, read
    through the PRODUCER's constants so a rename cannot drift this gate away
    from the strings the probe writes (``_did_not_lead``'s own discipline).
    Anything that DID answer — a refusal, a started handshake, a connect — is
    not a scope candidate, and neither is a row whose answer cannot be read.
    """
    observed = row.get("observed")
    outcome = str(observed.get("outcome") or "") if isinstance(observed, Mapping) else ""
    if outcome:
        return outcome in _SCOPE_OUTCOMES
    detail = str(row.get("detail") or "")
    if not detail:
        return False
    from local_operator.network import relay as relay_mod

    if detail in (
        relay_mod.DETAIL_NO_ANSWER,
        relay_mod.DETAIL_BAD_ENDPOINT,
        relay_mod.DETAIL_NOT_ATTEMPTED,
    ):
        return True
    if detail.startswith(relay_mod.CONNECT_FAILED_PREFIX):
        # A REFUSED connection is an answer — something is listening at that
        # address; every other connect failure (a timeout, a resolution
        # failure, a route the kernel could not find) is the no-answer family
        # the scope rule covers.
        return detail.split(":", 1)[1] != "ConnectionRefusedError"
    return False


def _private_block(host: str) -> tuple[int, int] | None:
    """The candidate's private /16, when it is an RFC1918 address.

    THE CLASS HALF of the scope rule (F9 ruling, 2026-10-04): 10/8, 172.16/12
    and 192.168/16 are meaningful only on the networks that own them — a device
    off those networks cannot reach them by construction. The /16 (the first two
    octets) is the finest network boundary address math alone gives (netmasks
    are not read here), and it is the CONSERVATIVE granularity: only a candidate
    that shares NO /16 with any of this device's addresses is excluded, so an
    address a routed private range could plausibly reach keeps its honest
    negative.
    """
    try:
        octets = socket.inet_aton(host)
    except OSError:
        return None
    first, second = octets[0], octets[1]
    if first == 10 or (first == 172 and 16 <= second <= 31) or (first == 192 and second == 168):
        return (first, second)
    return None


def _off_network(host: str, own_addresses: Sequence[str]) -> bool:
    """Whether an RFC1918 candidate is off-network FOR THIS DEVICE (F9).

    THE POSITION HALF: this device's own addresses are the observed answer to
    "what networks is this device on", and holding none on the candidate's
    private /16 means the candidate is unreachable from here by construction —
    the dial is not a question this device can ask the peer, so it must not
    count as a failure. A property of the address class AND the reading device's
    position, stated generally — never a carve-out for one host or one pair —
    and decided from the CANDIDATE set (which addresses are even askable from
    here), not by re-grading a failed row's verdict after the fact.
    """
    block = _private_block(host)
    if block is None:
        return False
    own_blocks = {item for item in (_private_block(addr) for addr in own_addresses) if item}
    return block not in own_blocks


def mark_out_of_scope(rows: list[dict[str, Any]], own_addresses: Sequence[str]) -> None:
    """Mark never-askable candidates out of scope, in place (F9 ruling).

    EXCLUDED FROM THE DECISION, VISIBLE IN THE READING: a candidate off this
    device's networks does not participate in any verdict, while still being
    REPORTED — a candidate list that silently omits an address is a different
    lie from one that miscounts it. The row keeps every observed fact, gains
    ``out_of_scope``, and its ``ok`` flips so no fold counts it; the reading
    says "out of scope" in so many words, and remedies are cleared because
    action items under a non-failing row misread (the informational flip's own
    discipline).

    ONE SEMANTICS, TWO PRODUCERS (design round 1, D3): ``ready``'s own rows and
    ``doctor``'s endpoint rows are the two dialects that read the same address —
    this flip, like :func:`mark_informational` beside it, reads both
    (``_no_answer_row``), so the two surfaces cannot drift about a question that
    cannot be asked from here.

    ONLY ROWS THAT GOT NO ANSWER ARE CANDIDATES: a refusal or a started
    handshake is an answer, and an answer refutes "cannot route to it". A
    connected row is a positive and is never touched; the unverified-accept
    rule is untouched either — an out-of-scope row carries no usable address
    and never vouches for another row.
    """
    for row in rows:
        if row["ok"]:
            continue
        if not _no_answer_row(row):
            continue
        host, _port = _split_endpoint(str(row.get("endpoint") or ""))
        if not _off_network(host, own_addresses):
            continue
        # THE DEFAULT ARRIVES ON THE FLIP, not before the gate (agent review
        # round 1, N1): a row this rule does not claim must not gain an empty
        # ``observed`` on the way past — the sibling's own discipline.
        row["ok"] = True
        observed = row.setdefault("observed", {})
        if isinstance(observed, dict):
            observed["out_of_scope"] = True
        if "remedies" in row:
            row["remedies"] = []


def _reachability_rows(
    member: Any,
    *,
    record: Any,
    server: "RelayServer",
    deadline: float | None,
    since: float,
) -> tuple[list[dict[str, Any]], Any]:
    """Probe every declared endpoint; return the rows and whatever link we hold.

    THE PROBE ALWAYS RUNS, even when a link exists: the observed per-endpoint
    facts are this report's product, and having the link is not evidence about
    which of the peer's addresses answers from HERE. When a link already
    exists the probed winner socket is closed rather than dialled again —
    dialling would EVICT the live link (newest wins), and a read-only report
    must not churn the operator's link to learn what it already knows.

    A TCP ACCEPT IS NOT A PEER ANSWER (design round 1, D1): any listener on a
    declared address satisfies a bare connect — a non-peer process holding a
    candidate port made this verb state the peer answered. So this loop tracks
    WHICH socket was identified — the one this run dialled and handshaked
    (``connected``), or an address a completed handshake already lives at, the
    existing link's own address (``connected_link``). Everything else that
    merely accepted renders as the observed fact (``connected_unverified`` /
    ``connected_elsewhere``) and never as the peer; ``winner_verified`` /
    ``link_address`` ride ``observed`` so the readings and remedies make the
    same distinction this loop does.

    AND A LINK'S ADDRESS MUST PIN TO A DECLARED ENDPOINT (round 2, R2-1/Q-3).
    Only an outbound link records the dialled host, and even that may have left
    the declared set; a link the PEER established records its source socket —
    an ephemeral port that is nobody's address and dies with the connection.
    Such a link is presented as the link fact alone (``connected_unpinned``:
    the peer is up; no address named), and the peer's own reachable address is
    not reddened merely because the link's address cannot be named.
    """
    from local_operator.network import addresses as addresses_mod
    from local_operator.network import relay as relay_mod

    # THE VIEWER'S OWN POSITION, read once for the scope rule below: this
    # device's answer to "what networks am I on" (local reads plus one route
    # lookup — no traffic).
    own_addresses = addresses_mod.local_ipv4_addresses()
    peer_label = _label(member)
    link = server._link_for(member.device_id)  # noqa: SLF001 — the one link seam
    endpoints = list(member.endpoints)
    # THE ONE ADDRESS THAT NEEDS NO NEW HANDSHAKE: a link's recorded address may
    # name the peer only when the link CAME UP at one of the member's DECLARED
    # endpoints — the outbound case (``relay.py`` records the dialled host). An
    # inbound link records the dialer's SOURCE socket (``register_link`` writes
    # ``f"{addr[0]}:{addr[1]}"`` on the accepting side) and an outbound link can
    # lose its recorded address from the declared set the same way — in both
    # cases the recorded address is one the peer cannot be pointed at, so it is
    # published only on a declared-endpoint match and an unpinnable live link is
    # presented as the LINK FACT alone — the peer is up — with no address
    # anywhere (round 2, R2-1/Q-3). The flag names the CONDITION, not a
    # direction: an outbound link whose address left the declared set is "
    # unpinned too, and must not carry a direction claim (round 3, R3-2).
    raw_link_addr = str(link.peer_addr or "") if link is not None else ""
    prior_addr = raw_link_addr if raw_link_addr in endpoints else ""
    link_unpinned = link is not None and not prior_addr
    if not endpoints:
        # NOTHING WAS DECLARED, so nothing was dialled — doctor's ``no_endpoint``
        # vocabulary (the renderer is this verb's, see ``reachability_reading``).
        row = {
            "check": "reachability",
            "class": CLASS_ADMISSION,
            "device_id": member.device_id,
            "device_name": peer_label,
            "endpoint": "",
            "ok": False,
            "detail": "no_endpoint",
            "observed": {
                "outcome": "no_endpoint",
                "source_address": None,
                "interface": None,
                "elapsed_ms": None,
                "budget_s": None,
                "attempted": False,
                "last_seen_at": member.last_seen_at,
            },
            "remedies": [f"have {peer_label} publish an address for the relay, then re-check"],
        }
        return [row], link
    probe = relay_mod.probe_candidates(
        endpoints,
        deadline=deadline,
        connect_cap=relay_mod.PROBE_CONNECT_TIMEOUT_S,
        wait_all=True,
    )
    elapsed_ms = round((time.monotonic() - since) * 1000, 1)
    winner = probe.winner if probe.sock is not None else ""
    winner_identified = False
    dial_failed = ""
    if probe.sock is not None:
        if link is not None:
            relay_mod._close_quietly(probe.sock)  # noqa: SLF001 — see the docstring
        else:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining is not None and remaining <= 0:
                # The probe ANSWERED inside its cap, but the report's own budget
                # ran out before the handshake could start. Mirror doctor: close
                # the socket and say which clock ran out rather than start a
                # dial with no budget (or, worse, with no bound at all).
                relay_mod._close_quietly(probe.sock)  # noqa: SLF001
                dial_failed = relay_mod.handshake_not_attempted_reason(winner, budget="report")
            else:
                try:
                    link, reason = server.dial(
                        record.network_id,
                        host=winner,
                        epoch=record.epoch,
                        timeout_s=remaining,
                        connected=probe.sock,
                        expected_device=member.device_id,
                    )
                except Exception as exc:  # noqa: BLE001 — a refused handshake is a row
                    from local_operator.network.types import MeshRefusal

                    link, reason = None, exc.code if isinstance(exc, MeshRefusal) else str(exc)
                if link is None:
                    dial_failed = reason or "unreachable"
            # IDENTIFIED means THIS run's handshake completed: the only socket
            # whose acceptance may be called the peer's answer outright.
            winner_identified = link is not None
    # Does any PEER claim about the winner hold? The dialled socket identified
    # it; the live link's own address was identified when that link came up.
    winner_claim_ok = bool(winner) and (
        winner_identified or (bool(prior_addr) and winner == prior_addr)
    )
    rows: list[dict[str, Any]] = []
    for attempt in probe.attempts:
        route = _route_observation(*_split_endpoint(attempt.endpoint))
        route_ok = not route.get("error")
        winner_elsewhere = bool(winner) and attempt.endpoint != winner
        is_winner = bool(winner) and attempt.endpoint == winner
        is_link_addr = bool(prior_addr) and attempt.endpoint == prior_addr
        if is_link_addr and attempt.connected:
            # The live link's own address: identified when that link came up.
            outcome = "connected_link"
            detail = relay_mod.DETAIL_OK
            ok = True
        elif is_winner and winner_identified:
            outcome = "connected"
            detail = relay_mod.DETAIL_OK
            ok = True
        elif attempt.connected and link_unpinned:
            # ACCEPTED while a live link cannot be pinned to a declared address
            # (R2-1/Q-3): the accept is still un-identified — nothing here says
            # the peer answered AT this address — but the live link proves the
            # PEER is up, and rendering the peer's own reachable address as
            # unverified merely because the link's address cannot be named would
            # be a false red. The link fact is presented with no address at all,
            # and THE WHY RIDES ``detail`` (design round 1, D5): the reading
            # stays in the family's register while the address fact that
            # explains it lives one field over.
            outcome = "connected_unpinned"
            detail = (
                "accepted_unpinned_link: the link's recorded address is not one the peer declares"
            )
            ok = True
        elif is_winner and link is not None:
            # ACCEPTED, NEVER IDENTIFIED — the D1 state: a link existed, so the
            # winner socket was closed un-dialled (no-churn), and nothing here
            # may say the peer answered at this address.
            outcome = "connected_unverified"
            detail = "accepted_unverified"
            ok = False
        elif is_winner:
            # The address answered but the dial did not come up: a refusal, or
            # the report's budget expired before the handshake started.
            outcome = "handshake_failed"
            detail = dial_failed or "unreachable"
            ok = False
        elif attempt.connected:
            outcome = "connected_elsewhere"
            detail = (
                relay_mod.doctor_link_elsewhere_detail(winner)
                if winner_claim_ok
                else "accepted_unverified"
            )
            ok = False
        else:
            outcome = _classify_outcome(
                attempt.detail, route_ok=route_ok, winner_elsewhere=winner_elsewhere
            )
            detail = attempt.detail
            ok = False
        observed = {
            "outcome": outcome,
            "source_address": route.get("source_address"),
            "interface": route.get("interface"),
            "elapsed_ms": attempt.latency_ms if attempt.latency_ms else elapsed_ms,
            "budget_s": relay_mod.PROBE_CONNECT_TIMEOUT_S,
            "attempted": attempt.detail != relay_mod.DETAIL_NOT_ATTEMPTED,
            "last_seen_at": member.last_seen_at,
        }
        if winner:
            observed["winner"] = winner
            observed["winner_verified"] = bool(winner_claim_ok)
        if link_unpinned:
            # A live link this report cannot pin to a declared address: only the
            # LINK FACT may travel — never its recorded address (round 2,
            # R2-1/Q-3). The flag states that condition and nothing else (round
            # 3, R3-2).
            observed["link_unpinned"] = True
        if outcome == "handshake_failed":
            # The dial-failure STAGE word, for the one family whose remedy
            # differs: this report's own budget vs the peer's handshake (round
            # 3, D6). The stage home is relay's; the sentence is ours.
            observed["dial_stage"] = detail.partition(":")[0]
        if prior_addr and attempt.endpoint != prior_addr:
            # The one OTHER address this report may name as the peer's: the live
            # link's own. Published so readings and remedies cite a verified
            # elsewhere-answer instead of an accept.
            observed["link_address"] = prior_addr
        rows.append(
            {
                "check": "reachability",
                "class": CLASS_ADMISSION,
                "device_id": member.device_id,
                "device_name": peer_label,
                "endpoint": attempt.endpoint,
                "ok": bool(ok),
                "detail": detail,
                "observed": observed,
                "remedies": _reachability_remedies(
                    outcome, peer_label=peer_label, observed=observed
                ),
            }
        )
    # Scope first: an out-of-scope candidate must not be re-dressed by the
    # informational flip on the way out (that flip serves addresses that WERE
    # askable and did not lead; the two rules must not both claim a row).
    mark_out_of_scope(rows, own_addresses)
    mark_informational(rows)
    return rows, link


def _split_endpoint(endpoint: str) -> tuple[str, int]:
    """``host:port`` from an endpoint string, with a dial-meaningful default."""
    address, _, port_text = endpoint.rpartition(":")
    try:
        return (address or endpoint), int(port_text)
    except ValueError:
        return (address or endpoint), 0


# ---------------------------------------------------------------------------
# The composer (runs on the viewing side) and the ops
# ---------------------------------------------------------------------------


def _ask_readiness(
    server: "RelayServer", link: Any, *, timeout: float = READINESS_OP_TIMEOUT_S
) -> tuple[dict[str, Any] | None, str]:
    """Ask the peer for its facts. Returns ``(facts, "")`` or ``(None, reason)``."""
    request = {
        "op": "net_readiness",
        "req": server._next_relay_req(),  # noqa: SLF001
        "locality": "remote",
    }
    reply = link.request(request, timeout=timeout)
    if reply is None:
        return None, "the readiness request did not arrive before the bound"
    if reply.get("op") != "ack":
        return None, _bounded(reply.get("message") or "the peer refused the readiness request", 200)
    detail = reply.get("detail")
    if not isinstance(detail, Mapping):
        return None, "the peer's answer carried nothing this build can read"
    return dict(detail), ""


def _peer_checks(
    member: Any,
    *,
    record: Any,
    server: "RelayServer",
    viewer: Viewer,
    deadline: float | None,
) -> list[dict[str, Any]]:
    """Every readiness row for ONE member: reachability, then capabilities."""
    peer_label = _label(member)
    since = time.monotonic()
    rows, link = _reachability_rows(
        member, record=record, server=server, deadline=deadline, since=since
    )
    if link is None:
        # NO LINK, SO NOTHING WAS ASKED — and the build row degrades with the
        # rest: its source is the handshake, and there was no handshake.
        rows.extend(
            not_asked_rows(
                member,
                detail="not asked: the peer did not answer",
                capabilities=(CAPABILITY_BUILD, *PEER_SIDE_CHECKS),
            )
        )
        return rows
    rows.append(
        build_row(
            member,
            peer_build=link.peer_build,
            own_build=getattr(server, "build", {}) or {},
            peer_label=peer_label,
        )
    )
    if wire.PEER_READINESS_V1 not in link.capabilities:
        rows.extend(_peer_too_old_rows(member))
        return rows
    # THE ASK SPENDS THE REPORT'S OWN BUDGET, never more: every wait inside one
    # run shares the deadline the composer opened, so the whole op stays inside
    # the CLI's client guard (the listing budget plus its margin) — a report
    # that outlives its reader is one nobody reads. A peer reached too late in
    # the run degrades like any other unanswered ask.
    remaining = None if deadline is None else deadline - time.monotonic()
    if remaining is not None and remaining <= 0.5:
        rows.extend(
            not_asked_rows(
                member, detail="not asked: the report's budget ran out before the request"
            )
        )
        return rows
    timeout = (
        READINESS_OP_TIMEOUT_S if remaining is None else min(READINESS_OP_TIMEOUT_S, remaining)
    )
    facts, problem = _ask_readiness(server, link, timeout=timeout)
    if facts is None:
        rows.extend(not_asked_rows(member, detail=f"not asked: {problem}", remedies=()))
        return rows
    rows.append(operator_row(member, facts, peer_label=peer_label))
    rows.append(git_row(member, facts, peer_label=peer_label))
    rows.append(mcp_servers_row(member, facts, peer_label=peer_label))
    rows.append(model_credential_row(member, facts, viewer=viewer, peer_label=peer_label))
    rows.append(tooling_row(member, facts, peer_label=peer_label))
    rows.extend(mcp_credential_rows(member, facts, viewer=viewer, peer_label=peer_label))
    return rows


def compose(server: "RelayServer", *, peer: str = "") -> dict[str, Any]:
    """The ``peer_readiness`` local op's body: the readiness report.

    Row order is doctor's (identity → network → membership → per member:
    reachability → capability rows). ``peer`` names one device by id OR name —
    resolved through the relay's one name resolver rather than string-compared,
    because the verb is typed by a person (``--peer cloud-node-1``).
    """
    from local_operator.network import relay as relay_mod
    from local_operator.network import store as store_mod

    deadline = time.monotonic() + relay_mod.LISTING_PROBE_BUDGET_S
    device_id = server._resolve_peer(peer) if peer else ""  # noqa: SLF001
    identity_missing = server.identity is None or not server.identity.device_id
    checks: list[dict[str, Any]] = []
    if identity_missing:
        checks.append(
            {
                "check": "identity",
                "class": CLASS_ADMISSION,
                "ok": False,
                "detail": "identity_missing",
            }
        )
    viewer = ViewerFacts(
        server.root, device_id=server.identity.device_id if server.identity else ""
    )
    try:
        for record in store_mod.list_networks(server.root):
            if record.stale:
                checks.append(
                    {
                        "check": "network",
                        "class": CLASS_ADMISSION,
                        "network_id": record.network_id,
                        "ok": False,
                        "detail": record.stale,
                    }
                )
            standing = relay_mod.membership_state(record)
            if standing["state"] != "active":
                checks.append(
                    {
                        "check": "membership",
                        "class": CLASS_ADMISSION,
                        "network_id": record.network_id,
                        "ok": False,
                        "code": standing["state"],
                        "detail": standing["sentence"],
                        "remedies": standing["remedies"],
                    }
                )
            for member in record.active_members():
                if member.device_id == record.self_device_id:
                    continue
                if device_id and member.device_id != device_id:
                    continue
                checks.extend(
                    _peer_checks(
                        member, record=record, server=server, viewer=viewer, deadline=deadline
                    )
                )
    finally:
        viewer.close()
    return {
        "checks": checks,
        "identity_present": not identity_missing,
        "identity_dir": str(store_mod.network_root(server.root)),
        # AND THE BUILD THIS PROCESS RUNS (design round 1, D5): the drill's own
        # shape is a relay that ANSWERS while a build beyond the running one is
        # installed, and a bare "running, pid N" gave it a clean bill. Same
        # reading and same words as `lop network status` — a relay answering
        # this call is the one process that can read its own image path.
        "relay": (
            f"running, pid {os.getpid()}"
            + relay_mod.generation_clause(relay_mod.generation_reading(os.getpid()))
        ),
    }


def _non_gating_equipment(row: Mapping[str, Any]) -> bool:
    """Whether this row is equipment that never holds the onboarding verdict."""
    return (
        str(row.get("class") or "") == CLASS_EQUIPMENT
        and str(row.get("capability") or "") in NON_GATING_EQUIPMENT
    )


#: The clause a failed non-gating equipment row reads in place of a bare FAIL —
#: ONE spelling (design round 1, D1): the verify receipt appends it and every
#: renderer appends it, so the same state cannot read fatal on one surface and
#: "not required" on another (the ``informational_clause`` pattern).
NON_GATING_NOTE = "not required for onboarding"


def non_gating_clause(row: Mapping[str, Any]) -> str:
    """The clause a failed non-gating equipment row carries, for any reader.

    Empty for rows that are ok or gating; the clause exists exactly so the
    non-gating state is legible wherever the row would otherwise read as a bare
    FAIL.
    """
    return "" if row.get("ok") or not _non_gating_equipment(row) else NON_GATING_NOTE


#: The row-state register (F8 residual, 2026-10-04): the ONE cell that answers
#: what a row's reading is, for machines and people alike. The human renderer
#: prints these four tokens (``ok``/``n/a`` padded to its column) and the row
#: facts the ``ready`` JSON carries hold them verbatim — ``ready --json`` is
#: the drill's acceptance surface, and a reader there scans ROWS, not the
#: aggregate, so a reported-but-non-gating row must say so ON itself
#: (``warn``) instead of reading failure-shaped (a bare ``ok: false``). One
#: register, so the printed line and the JSON cannot disagree.
ROW_STATE_OK = "ok"
ROW_STATE_WARN = "warn"
ROW_STATE_FAIL = "FAIL"
ROW_STATE_NOT_APPLICABLE = "n/a"


def row_state(row: Mapping[str, Any]) -> str:
    """This row's state cell — the classification, computed once for every reader.

    Precedence, the renderer's historic one:

    - ``n/a`` — excluded from the decision, visible in the reading (the F9
      out-of-scope flip): not a failure, and not a verification either;
    - ``ok`` — the row's own positive fact;
    - ``warn`` — failed equipment that cannot hold the onboarding verdict
      (exactly the rows :func:`onboarding_failures` excludes): named and NOT
      fatal, the same reading the verify receipt gives it, so the row the
      operator is sent to read cannot contradict the flow;
    - ``FAIL`` — everything else that failed: admission rows and still-gating
      equipment.

    ADDITIVE, never a rewrite: the row keeps every fact it had (``ok`` stays
    the raw reported result — the fail is REPORTED, not rewritten) so a
    row-scanner can branch on this cell without misreading the facts beneath
    it. An unknown shape falls through to ``FAIL``: absence of classification
    never excuses a failure.
    """
    observed = row.get("observed")
    observed = observed if isinstance(observed, Mapping) else {}
    if observed.get("out_of_scope"):
        return ROW_STATE_NOT_APPLICABLE
    if row.get("ok"):
        return ROW_STATE_OK
    if _non_gating_equipment(row):
        return ROW_STATE_WARN
    return ROW_STATE_FAIL


#: The renderer's column forms of the register: ``ok``/``n/a`` carry the pad,
#: ``warn``/``FAIL`` fill the four cells. PRESENTATION ONLY — keyed by the
#: register above so the column and the machine cell are one spelling.
_STATE_COLUMN: dict[str, str] = {
    ROW_STATE_OK: "ok ",
    ROW_STATE_WARN: "warn",
    ROW_STATE_FAIL: "FAIL",
    ROW_STATE_NOT_APPLICABLE: "n/a ",
}


def onboarding_failures(checks: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The rows that hold the ONBOARDING verdict (F8 ruling, 2026-10-04).

    The flow's requirement is "the device is onboarded": admission rows plus
    every equipment row EXCEPT the non-gating MCP pair. A failed MCP row is
    still REPORTED — its own ``warn`` row in every renderer and the receipt's
    named note (:func:`equipment_note`) — it just cannot fail an onboarding,
    because its real dependency refuses at its own point of use: the share
    verb's ``no_local_credential`` refusal, the placement/borrow path, and the
    interactive sign-in gate — where the work that needs it actually runs.

    Operator/git/model equipment rows keep gating by default (same-class
    candidates for the operator to re-grade; see ``NON_GATING_EQUIPMENT``), so
    this fold is NOT "admission only" in the strict sense — it is admission
    plus the equipment still marked gating.

    ITS SECOND READER IS ``ready``'S OWN VERDICT (design round 1, D1): the
    report the operator is told to read must not call fatal the same state the
    onboarding flow calls "not required for onboarding" — the row reads
    ``warn … — not required for onboarding`` there too, and the two surfaces
    cannot disagree about whether it gates.
    """
    return [row for row in checks if not row.get("ok") and not _non_gating_equipment(row)]


def equipment_note(checks: Iterable[Mapping[str, Any]]) -> str:
    """One line naming failed non-gating equipment, for the verify receipt.

    Shape (F8 ruling): ``mcp logins: slack — not required for onboarding``.
    Empty when no named non-gating row failed. TOOLING joined the non-gating
    set on 2026-10-06 and is deliberately NOT named here: this note exists for
    the VERIFY receipt, an onboarding surface, while a tooling gap is a fit
    question for a lane's work — it is reported on the ready rows and the
    agent digest themselves, and it is repaired at its own point of use. The
    note names WHAT is not set up and
    defers the state to the rows themselves (their details carry the branch —
    no login, not shared, an unreadable store), so it cannot misdescribe a
    branch it does not read — WHICH IS WHY THE SERVERS BRANCH READS THE CODE
    (design round 1, D4): only the row that actually read the peer's own
    declaration may say "none declared"; a check the peer's answer did not
    carry is "unknown", never a claim about it.
    """
    logins: list[str] = []
    no_servers = False
    servers_unknown = False
    for row in checks:
        if row.get("ok") or not _non_gating_equipment(row):
            continue
        capability = str(row.get("capability") or "")
        if capability == CAPABILITY_MCP_SERVERS:
            if str(row.get("code") or "") == CODE_NO_MCP_SERVERS:
                no_servers = True
            else:
                servers_unknown = True
            continue
        observed = row.get("observed")
        observed = observed if isinstance(observed, Mapping) else {}
        name = str(observed.get("server") or observed.get("url") or "").strip()
        if name and name not in logins:
            logins.append(name)
    parts: list[str] = []
    if logins:
        parts.append(f"mcp logins: {', '.join(logins)}")
    if no_servers:
        parts.append("mcp servers: none declared")
    if servers_unknown:
        parts.append("mcp servers: unknown")
    if not parts:
        return ""
    return "; ".join(parts) + f" — {NON_GATING_NOTE}"


#: Appended when a report ran without a device identity. ONE string for the
#: CLI register and the agent digest (agent review round 1, NIT-2: the two
#: surfaces had already drifted — "no device identity (identity_missing)" on
#: one, "no mesh identity" on the other).
NO_IDENTITY_LINE = (
    "this device has no device identity (identity_missing): create a network here, or "
    "re-pair with a new invite"
)

#: The empty-report line, shared for the same reason.
NOTHING_TO_CHECK_LINE = "nothing to check: no networks, or no other members yet"


def render_check_lines(checks: Iterable[Mapping[str, Any]]) -> list[str]:
    """The ``ready`` rows, one loop for BOTH surfaces (CLI register + digest).

    The two surfaces built these rows separately and had already drifted in
    spacing and in the no-identity sentence (agent review round 1, NIT-2) — a
    drift class where an agent and the person beside it read different words
    about one state. Reachability rows read through :func:`reachability_reading`
    (this verb's register; doctor's renderer conflates a refusal with silence),
    capability rows carry composed sentences, and every remedy indents under
    its row the way the membership lines do. A failed non-gating equipment row
    reads ``warn`` with its clause (design round 1, D1) instead of ``FAIL`` —
    the state that must not read as a blocker on the very surface the operator
    is sent to. The column reads :func:`row_state`, one register with the
    ``state`` cell the machine row facts carry (F8 residual): a reader and a
    row-scanner cannot disagree about whether a row gates.
    """
    from local_operator.resume import doctor_detail_words

    lines: list[str] = []
    for check in checks:
        # The state column comes from the ONE register (:func:`row_state`) —
        # the same computation stamps the ``state`` cell the machine row facts
        # carry (F8 residual), so the column a person reads and the cell a
        # scanner branches on cannot disagree about whether a row gates.
        state = _STATE_COLUMN[row_state(check)]
        kind = str(check.get("check") or "")
        label = str(check.get("device_name") or check.get("device_id") or "")
        if kind == "readiness":
            line = (
                f"{state} {kind} {check.get('capability', '')} {label}: "
                f"{check.get('detail', '')}".rstrip()
            )
            clause = non_gating_clause(check)
            if clause:
                line += f" — {clause}"
            lines.append(line)
        elif kind == "reachability":
            lines.append(
                f"{state} {kind} {label} {check.get('endpoint', '')}: "
                f"{reachability_reading(check)}".rstrip()
            )
        else:
            # identity / network / membership: doctor's grammar, doctor's words.
            lines.append(
                f"{state} {kind} {check.get('device_id', '')} "
                f"{doctor_detail_words(str(check.get('detail', '')))}".rstrip()
            )
        for remedy in check.get("remedies") or ():
            lines.append(f"    - {remedy}")
    return lines


def make_handler(server: "RelayServer") -> Any:
    """The ``net_readiness`` peer-op handler: THIS device's facts, read-only."""

    def _handle(link: "PeerLink", frame: dict[str, Any]) -> dict[str, Any]:
        # The link is unused: every fact is this device's own, and the handler
        # never sends a request over the link it is serving (the deadlock guard
        # ``PeerLink.request`` states).
        del link, frame
        return collect_peer_facts(server.root)

    return _handle


def local_handler(server: "RelayServer") -> Any:
    """The ``peer_readiness`` local op: compose the report for one or every peer."""

    def _handle(frame: dict[str, Any]) -> dict[str, Any]:
        return compose(server, peer=str(frame.get("peer") or ""))

    return _handle


def install(server: "RelayServer") -> None:
    """Register this slice's ops on ``server`` (relay construction calls this).

    NOT ``slow``: the peer handler is local file reads. The LOCAL op dials and
    asks, bounded internally by the listing budget the composer starts with,
    exactly like ``net_doctor`` — a control call that can outlast its caller is
    the defect both verbs avoid by holding one budget.
    """
    server.register_ops(
        {"net_readiness": make_handler(server)},
        local_handlers={"peer_readiness": local_handler(server)},
    )
