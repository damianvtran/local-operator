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

import json
import sqlite3
import stat
import time
from pathlib import Path
from typing import Any

#: Row namespace, and the SINGLE SOURCE OF TRUTH for this string. Not a
#: registry provider; see the module docstring. Anything else needing it --
#: the controller's console route, a fetcher -- imports it from here rather
#: than repeating the literal, so the row a writer creates and the row a
#: reader looks for cannot drift apart. Safe to import from anywhere: this
#: module pulls in only `stat`, `time`, `pathlib` and `typing`.
QWENCLOUD_CONSOLE_PROVIDER = "qwencloud-console"

#: Pinned so `_identity_key_for`'s field loop (org_id, account_id, email,
#: project_id) yields a stable identity and re-entry UPSERTS IN PLACE rather
#: than inserting a second row on every paste.
QWENCLOUD_CONSOLE_PROJECT_ID = "qwencloud-console:personal"

#: A console session cookie observed to last about a week. Past this, the
#: likeliest explanation for a missing window is a dead cookie, and `status`
#: says so rather than leaving the user to guess.
QWENCLOUD_TICKET_STALE_MS = 7 * 24 * 60 * 60 * 1000

#: Upper bound on a stored ticket, chosen from evidence rather than taste.
#: The live console cookie this feature was verified against is 172
#: characters, so 4096 leaves ~23x headroom for a longer session token while
#: still making the 10 MB paste QA landed here structurally impossible. 4096
#: is also the conventional single-header budget at the other end (nginx's
#: `large_client_header_buffers` is 8k for the whole block), so a value above
#: it would be refused by an intermediary even if it reached one.
QWENCLOUD_TICKET_MAX_LENGTH = 4096


class TicketStoreError(RuntimeError):
    """The ticket cannot be stored safely. The message never carries a value."""


class TicketStoreUnreadable(TicketStoreError):
    """The store could not be read, so whether a ticket exists is UNKNOWN.

    Distinct from "no ticket is stored" on purpose, and the distinction is the
    whole point: collapsing the two made ``rm`` report "No QwenCloud console
    ticket stored." with exit 0 while the plaintext full-account cookie was
    still on disk. The user accepted the plaintext risk on the understanding
    they could revoke it, so a revoke that cannot prove it worked must say so
    rather than claim success.
    """


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
    try:
        value = store.db_path
    except AttributeError as exc:
        raise TicketStoreError(
            "cannot determine the credential store's path, so its permissions "
            "cannot be checked; refusing to store a full-account session cookie"
        ) from exc
    if value is None:
        raise TicketStoreError(
            "the credential store reports no path, so its permissions cannot "
            "be checked; refusing to store a full-account session cookie"
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


def _reject_unsendable(ticket: str) -> None:
    """Refuse a ticket that could never reach the gateway, before storing it.

    The value's only use is interpolation into a ``cookie:`` request header
    (usage.py's console fetcher), and httpx validates header values locally:
    an embedded newline, CR or NUL raises ``LocalProtocolError`` and a
    non-latin-1 character raises ``UnicodeEncodeError``. The fetcher catches
    ``httpx.HTTPError`` and returns None, so the panel renders "no windows
    reported" with nothing linking it back to the paste. Storing such a value
    and reporting success is the failure controller.py:268-278 names: a bug
    dressed as a plausible degraded state.

    Rejecting at the boundary rather than sanitising follows
    ``credentials._reject_control_chars`` (credentials.py:29-40), which
    refuses the same byte classes for the same reason -- a legitimate token
    never contains one. The message never echoes the value: this is a
    full-account session cookie.

    CR and LF are NOT special-cased apart from the other control bytes. The
    interesting one is CRLF, which would be header injection if httpx did not
    stop it, but a value carrying any of them is equally incapable of being
    sent, so one rule covers both.
    """
    if len(ticket) > QWENCLOUD_TICKET_MAX_LENGTH:
        raise TicketStoreError(
            f"the ticket is {len(ticket)} characters, over the "
            f"{QWENCLOUD_TICKET_MAX_LENGTH}-character limit; this is not a "
            f"console session cookie. Copy only the login_qwencloud_ticket "
            f"value, not the surrounding request or document"
        )
    for char in ticket:
        if ord(char) < 0x20 or ord(char) == 0x7F:
            name = "a newline" if char in "\r\n" else f"a control character (0x{ord(char):02x})"
            raise TicketStoreError(
                f"the ticket contains {name}, so it could never be sent as a "
                f"cookie header and would fail silently. Paste the "
                f"login_qwencloud_ticket value as a SINGLE line, with no line "
                f"breaks"
            )
    try:
        # The exact encode httpx performs on a header value; doing it here
        # turns a `UnicodeEncodeError` raised from INSIDE the fetcher -- which
        # the fetcher's `except httpx.HTTPError` does not catch, breaching
        # usage.py:51-53's "a fetcher never raises" -- into a clear rejection
        # at the one place the user can act on it.
        ticket.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise TicketStoreError(
            "the ticket contains a non-ASCII character, so it could never be "
            "sent as a cookie header. Copy the login_qwencloud_ticket value "
            "exactly, with no surrounding text"
        ) from exc


def store_ticket(store: Any, ticket: str, *, now_ms: int | None = None) -> None:
    """Upsert the console cookie. Never returns or logs the value."""
    ticket = ticket.strip()
    if not ticket:
        raise TicketStoreError("empty ticket value")
    # Validated BEFORE the mode check and the write: a value that can never
    # reach the wire must not be stored at all, and must not be reported as
    # stored.
    _reject_unsendable(ticket)
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
    """The stored row's metadata, or None when no row exists.

    Raises :class:`TicketStoreUnreadable` when the store cannot be read, which
    is NOT the same answer as "nothing is stored" — see that class.

    The ``except`` is deliberately NARROW, following the rule
    ``ProviderController.has_any_credential`` records at controller.py:268-278:
    a locked, busy or corrupt store is an ENVIRONMENT fact to be reported,
    while ``sqlite3.ProgrammingError`` (a connection used across threads, a
    closed handle) is a BUG in the caller and must keep propagating. It is
    re-raised first because it subclasses ``DatabaseError``, so any clause
    broad enough to cover a corrupt store would otherwise swallow it.
    """
    try:
        # `include_disabled=True`: a soft-deleted row (`disabled_cause` set) is
        # filtered out of the default view (auth_store.py:578), which would
        # make a cookie that is still ON DISK invisible here -- and `rm` would
        # then report "No ... ticket stored." with exit 0 over a live
        # full-account credential, the same false success this module exists
        # to prevent. Precedent: `active_local_credential` (auth_store.py:585).
        rows = store.list_credentials(QWENCLOUD_CONSOLE_PROVIDER, include_disabled=True)
    except sqlite3.ProgrammingError:
        raise
    except (sqlite3.Error, OSError, json.JSONDecodeError) as exc:
        raise TicketStoreUnreadable(
            f"the credential store could not be read ({type(exc).__name__})"
        ) from exc
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
    """Remove the stored cookie. True when a row was removed, False when none was.

    Raises :class:`TicketStoreUnreadable` when the store cannot be read, and
    :class:`TicketStoreError` when a row was found but is still present
    afterwards.

    The deletion is CONFIRMED by re-reading rather than inferred from the call
    returning: ``AuthStore.delete_credential`` returns ``None`` whether or not
    it matched anything (auth_store.py:682-688), so "it did not raise" is not
    evidence the cookie is gone. For a revocation command on a full-account
    plaintext credential, the difference between those two is the entire value
    of the command.
    """
    record = read_ticket_record(store)
    if record is None:
        return False
    try:
        store.delete_credential(record["credential_id"])
    except sqlite3.ProgrammingError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise TicketStoreUnreadable(
            f"the ticket could not be deleted ({type(exc).__name__}); " "it may still be stored"
        ) from exc
    if read_ticket_record(store) is not None:
        raise TicketStoreError(
            "the ticket is still present after deleting it; it may still be stored"
        )
    return True
