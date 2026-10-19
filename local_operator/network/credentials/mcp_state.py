"""Durable per-key auth evidence: WHY a shared credential is not working here.

SEPARATE FROM ``state.py`` ON PURPOSE. That file is the borrower's refusal cache:
a TTL'd observation consulted before a dial to stop a retry storm. This file is
the answer to the operator's question — "is this server signed in, and if not,
whose login is it and what happened?" — and it must survive every restart and
outlive every TTL. A ``revoked`` that expired after a minute would make a
removed device look merely unsynced, and the design's §4/§5 split (reconcile
ends access; status says which problem it is) depends on the evidence staying
put until something real replaces it.

WHAT IT HOLDS AND WHAT IT CANNOT. ``{schema, network_id, entries: {key: {state,
code, owner_device, owner_name, at}}}`` — names, a closed state word, a machine
code and a timestamp. There is no field for a token or a value, and
:func:`local_operator.network.credentials.placement._assert_no_material` is
applied at the writer exactly as the placement and observation documents apply
it, so a future field named ``access_token`` is refused here rather than noticed
at review.

NEVER RAISES FROM A WRITE. Every caller is on a borrow, a pull or a reconcile
path whose real job is to move a credential, not to keep a diary: a read-only
config dir (a container, a test root, a broken home) must cost the evidence and
never the borrow. :func:`record` swallows ``OSError`` and a malformed existing
file; :func:`load` answers ``{}`` for anything it cannot read.

LOCKING. Reads take no lock (they are one small JSON parse and must never block
a provider call). Writes are read-modify-write and take the placement package's
thread lock plus its bounded cross-process lock, because the writer and the
reader are usually DIFFERENT PROCESSES here: the relay records what happened
when a member borrowed, and the CLI/session that renders ``/mcp`` reads it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from local_operator.network.credentials.placement import (
    _assert_no_material,
    _config_root,
    _cross_process_lock,
    credentials_root,
    _ensure_private_dir,
)

#: The file the evidence lives in, beside ``placement.json`` and
#: ``placement.state.json``. Own schema string rather than the placement's: the
#: two documents evolve for different reasons and a reader must be able to tell
#: which one it opened.
MCP_AUTH_SCHEMA = "lop.mcp_auth.v1"

MCP_AUTH_FILENAME = "mcp_auth.json"

#: The lock file beside it. Dotted, like ``.placement.lock``, and never unlinked
#: while it may be contended — see ``placement._cross_process_lock``.
MCP_AUTH_LOCK_FILENAME = ".mcp_auth.lock"

FILE_MODE = 0o600

#: The CLOSED state vocabulary a surface may render. One word means one problem,
#: and each maps to a different remedy:
#:
#: ``expired``       a source exists but its token is past expiry and could not be
#:                   refreshed (the owner must sign in again there);
#: ``revoked``       this device WAS a holder and is not any more, or lost the
#:                   network; access ended at reconcile;
#: ``unauthorized``  the server refused the bearer we presented (a 401);
#: ``unavailable``   the owner could not be reached;
#: ``synced``        the last thing we know is that it worked.
STATES: frozenset[str] = frozenset(
    {"expired", "revoked", "unauthorized", "unavailable", "synced"}
)

#: Broker code -> evidence state. The mapping lives HERE rather than at each
#: caller so the client, the sync engine and the CLI cannot disagree about what
#: ``refresh_failed`` means. A code absent from this table records NOTHING: it is
#: either a refusal that is about behaviour rather than the credential
#: (``quota_blocked``, ``rate_limited``) or one this build does not classify, and
#: inventing a state for it would overwrite a precise older answer with a vague
#: new one.
STATE_FOR_CODE: dict[str, str] = {
    "interactive_required": "expired",
    "refresh_failed": "expired",
    "not_a_holder": "revoked",
    "revoked": "revoked",
    "member_not_active": "revoked",
    "owner_offline": "unavailable",
    "not_authorised": "unauthorized",
    "grant_invalid": "unauthorized",
}


def state_for_code(code: str) -> str:
    """The evidence state for a broker refusal ``code``, or ``""`` (record nothing)."""
    return STATE_FOR_CODE.get(str(code or ""), "")


def mcp_auth_path(network_id: str, root: Path | None = None) -> Path:
    """The evidence file's path. NEVER CREATES anything, like ``placement_path``."""
    return credentials_root(root) / network_id / MCP_AUTH_FILENAME


def load(network_id: str, root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Every entry, or ``{}``. An unreadable file reads as no evidence.

    NO EVIDENCE IS THE HONEST ANSWER for a corrupt or missing file: the status
    surface falls back to what the placement and the stores themselves say, which
    is the same answer a device that never borrowed has. Guessing a state here
    would turn a parse bug into "revoked" on somebody's screen.
    """
    try:
        raw = mcp_auth_path(network_id, root).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    try:
        payload = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(payload, dict):
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, row in entries.items():
        if isinstance(row, dict) and str(row.get("state") or "") in STATES:
            out[str(key)] = dict(row)
    return out


def record(
    network_id: str,
    key: str,
    *,
    state: str,
    code: str = "",
    owner_device: str = "",
    owner_name: str = "",
    root: Path | None = None,
) -> None:
    """Write (or replace) one key's evidence. NEVER raises.

    ``state`` outside :data:`STATES` is a programming error, and it records
    nothing rather than writing a word no surface can render. A blank network id
    or key is the zero-peer no-op: there is no document to belong to.
    """
    if not network_id or not key or state not in STATES:
        return
    try:
        _write(
            network_id,
            {
                **load(network_id, root),
                key: {
                    "state": state,
                    # Bounded here rather than at every reader: these strings ride
                    # audit rows and --json documents, and a broker code or device
                    # name is a name, never a sentence the owner composed.
                    "code": str(code or "")[:64],
                    "owner_device": str(owner_device or "")[:128],
                    "owner_name": str(owner_name or "")[:128],
                    "at": time.time(),
                },
            },
            root,
        )
    except Exception:  # noqa: BLE001 — bookkeeping must never fail a borrow
        return


def clear(network_id: str, key: str, *, root: Path | None = None) -> None:
    """Forget one key's evidence, after a success that makes it stale. Never raises."""
    if not network_id or not key:
        return
    try:
        entries = load(network_id, root)
        if key not in entries:
            return
        entries.pop(key, None)
        _write(network_id, entries, root)
    except Exception:  # noqa: BLE001 — see :func:`record`
        return


def forget_device(network_id: str, device: str, *, root: Path | None = None) -> None:
    """Drop every entry owned by ``device``. Never raises.

    Called when a device leaves: evidence about a login that no longer exists
    anywhere is worse than none, because a status surface would keep naming an
    owner nobody can reach.
    """
    if not network_id or not device:
        return
    try:
        entries = load(network_id, root)
        kept = {
            key: row
            for key, row in entries.items()
            if str(row.get("owner_device") or "") != device
        }
        if len(kept) == len(entries):
            return
        _write(network_id, kept, root)
    except Exception:  # noqa: BLE001 — see :func:`record`
        return


def _write(network_id: str, entries: dict[str, dict[str, Any]], root: Path | None) -> None:
    """One atomic, 0600, locked write. The caller has already swallowed nothing."""
    from local_operator.network import store

    target = mcp_auth_path(network_id, root)
    payload = {
        "schema": MCP_AUTH_SCHEMA,
        "network_id": network_id,
        "entries": entries,
    }
    # The same structural refusal the placement writer makes: this document may
    # never grow a field that could hold material.
    _assert_no_material(payload, where="mcp auth")
    _ensure_private_dir(target.parent, root=_config_root(root))
    with _cross_process_lock(target.parent / MCP_AUTH_LOCK_FILENAME, root=_config_root(root)):
        store._write_private_json(target, payload)


def file_is_private(path: Path) -> bool:
    """Whether an existing evidence file carries the mode this module promises."""
    try:
        return (path.stat().st_mode & 0o777) == FILE_MODE
    except OSError:
        return False
