"""One quick-tunnel host per durable session family, not a site permission.

The pin precedes the approval RPC and survives cancellation, transport failure,
process death and resume. Otherwise a failed/denied prompt could be worked around
by minting another host. It lasts as long as the root session directory; ordinary
session retention removes it with that conversation. No grants are created here.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

from local_operator.divergence import ORIGIN_NAME


@dataclass(frozen=True)
class PreviewPin:
    host: str
    origin: str
    allowed: bool


def quick_tunnel_origin(url: str) -> tuple[str, str] | None:
    """Parse real HTTP(S) hosts, not suffixes in paths, credentials or lookalikes."""
    try:
        # Browser HTTP URLs treat backslashes as separators and decode host
        # escapes. Account for those aliases before comparing the parsed host,
        # or a second tunnel could evade the family pin with URL spelling alone.
        parsed = urlsplit(url.replace("\\", "/"))
        host = unquote(parsed.hostname or "").lower().removesuffix(".")
        port = parsed.port  # Validate the port before claiming a usable origin.
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not host.endswith(".trycloudflare.com"):
        return None
    if not all(
        re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label) for label in host.split(".")
    ):
        return None
    return host, f"{parsed.scheme}://{host}" + (f":{port}" if port is not None else "")


def _family_root(session_dir: Path) -> Path:
    """Follow ONLY creator-stamped origin.json parents in this sessions store.

    This is the durable child->parent contract used by divergence and resume,
    including nested children after a restart. Unlike a display/backfill reader,
    an approval guard must fail closed on damaged ancestry, not reinterpret a
    child as an unrelated family. No transcript scans or model-provided IDs.
    """
    store = session_dir.parent
    current = session_dir
    seen: set[str] = set()
    while True:
        if current.name in seen or not current.is_dir():
            raise ValueError("unavailable or cyclic session ancestry")
        seen.add(current.name)
        try:
            marker = json.loads((current / ORIGIN_NAME).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return current  # Original user conversations need not have a marker.
        if not isinstance(marker, dict):
            raise ValueError("invalid session ancestry")
        parent = marker.get("parent", "")
        if parent == "":
            if marker.get("origin") == "subagent":
                raise ValueError("missing delegated session ancestry")
            return current
        if not isinstance(parent, str) or parent in {".", ".."} or Path(parent).name != parent:
            raise ValueError("invalid session parent")
        current = store / parent


def pin_quick_tunnel(session_dir: Path, host: str, origin: str) -> PreviewPin:
    """First committed INSERT wins across threads/processes; never replace it.

    SQLite's existing transaction/locking primitives make comparison and pinning
    atomic without an in-memory race or a partially published JSON marker. A
    crash before commit cannot send a prompt (the caller awaits us); after commit
    the pin persists even if the RPC never arrives. Storage errors fail closed.
    """
    root = _family_root(session_dir)
    path = root / "browser-preview-origin.sqlite3"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.close(fd)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS pin "
            "(id INTEGER PRIMARY KEY CHECK (id = 1), host TEXT NOT NULL, origin TEXT NOT NULL)"
        )
        db.execute("INSERT OR IGNORE INTO pin VALUES (1, ?, ?)", (host, origin))
        row = db.execute("SELECT host, origin FROM pin WHERE id = 1").fetchone()
    assert row is not None
    return PreviewPin(host=row[0], origin=row[1], allowed=row[0] == host)


def fixed_origin_instruction(origin: str) -> str:
    # i18n: ignore model-facing browser tool policy, not operator UI copy
    return (
        "Fixed-origin rule: serve test builds on ONE fixed origin per task "
        "(localhost on one fixed port, or a stable preview URL), and reuse it "
        "across rounds and manager/child/sibling agents. This session family "
        f"is pinned to quick-tunnel host {origin}. Do not create or request "
        "another trycloudflare.com host; coordinate reuse of the original "
        "origin or use a fixed localhost/stable preview origin. Existing site "
        "approvals still apply; this rule grants no access. Cancellation, "
        "approval failure and runtime restart do not reset the pin."
    )
