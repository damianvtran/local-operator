"""Storage for the QwenCloud console session cookie.

The personal Token Plan window is invisible to the official CLI's BSS
gateway -- for a live account it answers ``IsGray: true`` with an empty seat
summary and zero instances on every commodity. The console gateway the web
UI itself calls does report it, and authenticates on exactly one thing: the
browser session cookie ``login_qwencloud_ticket``.

That makes this the broadest credential in the store. It is a FULL-ACCOUNT
console session, not a scoped API key, and ``~/.local-operator/auth.db`` is
plaintext SQLite protected only by its 0600 mode (no OS keychain is used).
So this module does three things beyond writing a row: it refuses to write
into a store whose file or directory modes are wider than 0600/0700, it
stamps a capture time so ``status`` can flag a cookie old enough to be dead,
and it never returns or logs the value.

The row lives under its own provider id, ``qwencloud-console``, which is
deliberately NOT in ``PROVIDER_REGISTRY`` -- the same row-namespace trick
``mcp-oauth`` uses. A row under ``alibaba-token-plan`` would satisfy
``ProviderController.has_any_credential`` (which matches the provider column
with no type or field filter) and local-operator would conclude it can run
CHAT traffic on a read-only console cookie.
"""

from __future__ import annotations

import stat
import time
from pathlib import Path
from typing import Any

#: Row namespace. Not a registry provider; see the module docstring.
QWENCLOUD_CONSOLE_PROVIDER = "qwencloud-console"

#: Pinned so `_identity_key_for`'s field loop (org_id, account_id, email,
#: project_id) yields a stable identity and re-entry UPSERTS IN PLACE rather
#: than inserting a second row on every paste.
QWENCLOUD_CONSOLE_PROJECT_ID = "qwencloud-console:personal"

#: A console session cookie observed to last about a week. Past this, the
#: likeliest explanation for a missing window is a dead cookie, and `status`
#: says so rather than leaving the user to guess.
QWENCLOUD_TICKET_STALE_MS = 7 * 24 * 60 * 60 * 1000


class TicketStoreError(RuntimeError):
    """The ticket cannot be stored safely. The message never carries a value."""


def _resolve_db_path(store: Any) -> Path:
    """Where ``store`` keeps its SQLite file, or raise.

    The store reports its OWN path (``AuthStore.db_path``) rather than this
    module deriving one from ``default_db_path()``: a store built on an
    explicit path would otherwise have a different file checked than the one
    written, so the check could pass while the real store was world-readable.

    Unresolvable is an ERROR, not a skip, and that branch is the whole point.
    The obvious spelling -- ``getattr(store, "db_path", None)`` guarding an
    ``if`` -- degrades to None for any object without the property, skipping
    the mode check entirely and writing a full-account cookie into a
    world-readable store while reporting success. A security precondition
    that cannot be EVALUATED must never be treated as SATISFIED.
    """
    value = getattr(store, "db_path", None)
    if value is None:
        raise TicketStoreError(
            "cannot determine the credential store's path, so its permissions "
            "cannot be checked; refusing to store a full-account session cookie"
        )
    return Path(value)


def _require_private_modes(db_path: Path) -> None:
    """Refuse to write a full-account cookie into a world- or group-readable store.

    Checked at WRITE time rather than trusted from creation: the store
    creates its file 0600 before sqlite opens it, but a file that has since
    been copied, restored from a backup, or chmod'd is exactly the case
    worth catching -- and this credential is the one with the most to lose.
    """
    directory = db_path.parent
    if directory.exists():
        mode = stat.S_IMODE(directory.stat().st_mode)
        if mode & 0o077:
            raise TicketStoreError(
                f"{directory} is mode {mode:04o}; refusing to store a full-account "
                f"session cookie outside a private directory. Fix with: "
                f"chmod 700 {directory}"
            )
    if db_path.exists():
        mode = stat.S_IMODE(db_path.stat().st_mode)
        if mode & 0o077:
            raise TicketStoreError(
                f"{db_path} is mode {mode:04o}; refusing to add a full-account "
                f"session cookie to a store others can read. Fix with: "
                f"chmod 600 {db_path}"
            )


def store_ticket(store: Any, ticket: str, *, now_ms: int | None = None) -> None:
    """Upsert the console cookie. Never returns or logs the value."""
    ticket = ticket.strip()
    if not ticket:
        raise TicketStoreError("empty ticket value")
    _require_private_modes(_resolve_db_path(store))
    payload = {
        # `ticket`, never `key`: the API-key cascade reads `data["key"]`, and
        # a console cookie resolved as an inference key would be sent to
        # DashScope as a bearer.
        "ticket": ticket,
        "project_id": QWENCLOUD_CONSOLE_PROJECT_ID,
        "captured_at": int(time.time() * 1000) if now_ms is None else now_ms,
    }
    # No `type` key: the store STAMPS `type` into the data it persists, and
    # carrying it back on the next write makes `_identity_key_for`
    # short-circuit to None (api_key rows get no identity key) and INSERT a
    # duplicate instead of updating in place. No `source="login"` either,
    # for the same reason -- it is the other short-circuit in that function.
    store.upsert_credential(QWENCLOUD_CONSOLE_PROVIDER, payload)


def read_ticket_record(store: Any) -> dict[str, Any] | None:
    """The stored row's metadata, or None. The VALUE is never included."""
    try:
        rows = store.list_credentials(QWENCLOUD_CONSOLE_PROVIDER)
    except Exception:  # noqa: BLE001 — an unreadable store has no ticket
        return None
    for row in rows:
        data = getattr(row, "data", None)
        if isinstance(data, dict) and data.get("ticket"):
            return {
                "credential_id": getattr(row, "id", 0),
                "captured_at": data.get("captured_at"),
                "length": len(str(data["ticket"])),
            }
    return None


def delete_ticket(store: Any) -> bool:
    """Remove the stored cookie. True when a row was removed."""
    record = read_ticket_record(store)
    if record is None:
        return False
    store.delete_credential(record["credential_id"])
    return True
