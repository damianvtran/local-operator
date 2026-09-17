"""Storage for the QwenCloud console session cookie.

The personal Token Plan window is invisible to the official CLI's BSS
gateway -- for a live account it answers ``IsGray: true`` with an empty seat
summary and zero instances on every commodity. The console gateway the web
UI itself calls does report it, and authenticates on exactly one thing: the
browser session cookie ``login_qwencloud_ticket``.

That makes this the broadest credential in the store. It is a FULL-ACCOUNT
console session, not a scoped API key, so the two halves are stored apart:
the VALUE goes to the encrypted ``lop secret`` store under
:data:`QWENCLOUD_TICKET_SECRET_NAME`, while ``~/.local-operator/auth.db``
keeps only metadata -- capture time, length, and the secret's name. That
split is what lets ``status`` report presence, length and age without ever
retrieving the value.

Per ``guide://credentials`` this is a large and worthwhile increase in the
cost of stealing this credential, NOT a vault: anything running as the user
and willing to run ``lop`` can read the store. So the module still does the
things that do not depend on encryption: it refuses to write metadata into a
store whose file or directory modes are wider than 0600/0700, it stamps a
capture time so ``status`` can flag a cookie old enough to be dead, and it
never returns or logs the value.

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

#: The secret store name holding the ticket VALUE. Single source of truth,
#: for the same reason QWENCLOUD_CONSOLE_PROVIDER is: the name a writer
#: creates and the name a reader looks for cannot drift apart.
QWENCLOUD_TICKET_SECRET_NAME = "QWENCLOUD_CONSOLE_TICKET"

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


class TicketStoreLocked(TicketStoreUnreadable):
    """The secret store is hardened and locked, so the value cannot be read.

    A SUBCLASS of TicketStoreUnreadable rather than a sibling: every existing
    caller that treats "unreadable" as "not the same as absent" is already
    correct for this case, so no call site becomes wrong by omission. It is
    its own class because this state has a REMEDY the user can act on
    (``lop secret unlock``), and a message naming that remedy is the entire
    reason it is distinguishable.

    The message is OURS, not the store's. ``access.retrieve_secret`` re-raises
    the broker's raw wire text ("no lop session is registered with the
    broker"), which is about ancestry and useless to someone whose real
    problem is a locked store -- the re-wording that says "run ``lop secret
    unlock``" lives in ``master_key_for`` and does not fire on the retrieval
    path. Verified by probe against a real hardened, locked store.

    Because it subclasses ``TicketStoreUnreadable``, which subclasses
    ``TicketStoreError``, every ``except`` clause naming it MUST come first.
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
            "cannot be checked; refusing to write the ticket's metadata"
        ) from exc
    if value is None:
        raise TicketStoreError(
            "the credential store reports no path, so its permissions cannot "
            "be checked; refusing to write the ticket's metadata"
        )
    return Path(value)


def _require_private_modes(db_path: Path) -> None:
    """Refuse to write the ticket's METADATA into a world- or group-readable store.

    Scoped to metadata since the VALUE moved to the encrypted store: what
    this guards is now capture time, length and the secret's name, which
    still say that an account HAS a live console session and when it was
    taken. The value store enforces the same ``0o077``-clear rule of its own
    accord -- ``SecretStore._open`` calls ``check_mode(self._path)`` on every
    open, against ``_FORBIDDEN_MODE_BITS`` -- so both halves are covered, by
    their own owners rather than by this function reaching across.

    Checked at WRITE time rather than trusted from creation: the store
    creates its file 0600 before sqlite opens it, but a file that has since
    been copied, restored from a backup, or chmod'd is exactly the case
    worth catching.
    """
    directory = db_path.parent
    if directory.exists():
        mode = stat.S_IMODE(directory.stat().st_mode)
        if mode & 0o077:
            raise TicketStoreError(
                f"{directory} is mode {mode:04o}; refusing to write the ticket's "
                f"metadata outside a private directory. Fix with: "
                f"chmod 700 {directory}"
            )
    if db_path.exists():
        mode = stat.S_IMODE(db_path.stat().st_mode)
        if mode & 0o077:
            raise TicketStoreError(
                f"{db_path} is mode {mode:04o}; refusing to add the ticket's "
                f"metadata to a store others can read. Fix with: "
                f"chmod 600 {db_path}"
            )


def _reject_unsendable(ticket: str) -> None:
    """Refuse a ticket that could never reach the gateway, before storing it.

    The value's only use is interpolation into a ``cookie:`` request header
    (usage.py's console fetcher), and httpx validates header values locally:
    an embedded newline, CR or NUL raises ``LocalProtocolError`` and a
    non-latin-1 character raises ``UnicodeEncodeError``. The fetcher catches
    ``httpx.HTTPError`` and returns None, so with no report object at all the
    panel falls back to its generic empty-result row -- "no usage — no quota
    endpoint, or no credential for one", quoted verbatim -- naming a MISSING
    credential, with nothing linking it back to the paste. Storing such a value
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


def _store_secret_value(ticket: str, base: Path | None) -> None:
    """Write the ticket VALUE to the encrypted store, or raise.

    Imported INSIDE the function, not at module scope, following the idiom
    ``access.py`` itself uses: this module is imported by ``controller.py``
    and advertises in its header that it pulls in only stdlib, so the client's
    fcntl/socket machinery must not become a module-level dependency.
    """
    from local_operator.secrets import access
    from local_operator.secrets.client import BrokerDenied, BrokerLocked
    from local_operator.secrets.errors import SecretExists, SecretStoreError

    try:
        # `create=True` is correct HERE and only here: `open_store`'s docstring
        # reserves it for the write verbs, because a read against a store that
        # does not exist must say so rather than initialise an empty one.
        secret_store = access.open_store(base, create=True)
        secret_store.initialize()
        try:
            secret_store.set(QWENCLOUD_TICKET_SECRET_NAME, ticket.encode())
        except SecretExists:
            # The single-name spelling of `mcp/credentials.py`'s
            # `write = store.update if key in existing else store.set`. A
            # `describe` probe just to pick the verb would be a second round
            # trip and a second way of doing the same thing.
            secret_store.update(QWENCLOUD_TICKET_SECRET_NAME, ticket.encode())
    # BEFORE the SecretStoreError clause, because both subclass it. A broad
    # clause above these swallows the locked state and the user loses the one
    # remedy they can act on.
    except (BrokerDenied, BrokerLocked) as exc:
        raise TicketStoreLocked(
            "the secret store is hardened and locked, so the ticket cannot be "
            "stored. Run `lop secret unlock`, then retry"
        ) from exc
    except SecretStoreError as exc:
        raise TicketStoreError(
            f"the ticket could not be encrypted ({type(exc).__name__}); nothing was written"
        ) from exc


def store_ticket(
    store: Any, ticket: str, *, now_ms: int | None = None, base: Path | None = None
) -> None:
    """Upsert the console cookie. Never returns or logs the value.

    The VALUE goes to the encrypted store and only METADATA reaches ``store``.
    ``base`` overrides the secret store's location for tests; production
    callers leave it None so it resolves from ``LOCAL_OPERATOR_CONFIG_DIR``.

    **The secret is written FIRST, the metadata row second, and the order is
    deliberate.** A crash between the two then leaves a SECRET orphan -- a
    value nothing points at, invisible to this feature and overwritten by the
    next `set`. The reverse order leaves a METADATA orphan, which claims a
    credential that cannot be read. Prefer the failure that looks like
    "nothing happened" over the one that lies about what is stored.
    """
    ticket = ticket.strip()
    if not ticket:
        raise TicketStoreError("empty ticket value")
    # Validated BEFORE the mode check and the write: a value that can never
    # reach the wire must not be stored at all, and must not be reported as
    # stored.
    _reject_unsendable(ticket)
    _require_private_modes(_resolve_db_path(store))
    _store_secret_value(ticket, base)
    payload = {
        # No `ticket` key: the VALUE now lives in the encrypted store and
        # `auth.db` carries only what `status` needs to report presence,
        # length and age WITHOUT a retrieval. `key` would be worse still --
        # the API-key cascade reads `data["key"]`, and a console cookie
        # resolved as an inference key would be sent to DashScope as a bearer.
        "project_id": QWENCLOUD_CONSOLE_PROJECT_ID,
        "captured_at": int(time.time() * 1000) if now_ms is None else now_ms,
        "secret_name": QWENCLOUD_TICKET_SECRET_NAME,
        # Recorded rather than derived: `SecretRecord` has no `length` field,
        # so the only other way to report it would be to RETRIEVE the value,
        # which is exactly what this design exists to avoid.
        "length": len(ticket),
    }
    # No `type` key: the store STAMPS `type` into the data it persists, and
    # carrying it back on the next write makes `_identity_key_for`
    # short-circuit to None (api_key rows get no identity key) and INSERT a
    # duplicate instead of updating in place. No `source="login"` either,
    # for the same reason -- it is the other short-circuit in that function.
    #
    # Caught the same NARROW way as `read_ticket_record` and `delete_ticket`
    # below, and for the reason this module already states: a locked, busy or
    # corrupt store is an ENVIRONMENT fact to be reported, not a crash. Left
    # uncaught, a store held under BEGIN EXCLUSIVE printed a raw traceback
    # carrying absolute local paths, while `status` and `rm` on the same
    # locked store reported it cleanly -- inconsistent within one command.
    #
    # The clause ORDER is load-bearing: `sqlite3.ProgrammingError` subclasses
    # `DatabaseError` subclasses `Error`, so a bare `except sqlite3.Error`
    # would swallow a caller bug (a connection used across threads, a closed
    # handle) and dress it as a plausible degraded state.
    try:
        store.upsert_credential(QWENCLOUD_CONSOLE_PROVIDER, payload)
    except sqlite3.ProgrammingError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise TicketStoreError(
            f"the ticket could not be stored ({type(exc).__name__}); " "nothing was written"
        ) from exc


def _secret_is_present(base: Path | None) -> bool:
    """Whether the VALUE exists, WITHOUT retrieving it.

    ``describe``, never ``get``. ``SecretStore.describe`` opens
    ``for_write=False`` and appends no audit row, while ``SecretStore.get``
    does ``audit.append(connection, event="get", ...)`` and stamps
    ``last_used_at``. Measured on a throwaway store: five ``describe`` calls
    left the ``get`` event count at 0 and ``last_used_at`` at None; one ``get``
    took it to a float. That is the property this whole design is sold on --
    ``status`` reports presence, length and age having never read the value.

    (``open_store()`` itself DOES append a broker ``key``/``deny:key`` row per
    call, so "a status call writes no audit row at all" is false and must not
    be asserted. The property is the absence of a ``get``.)
    """
    from local_operator.secrets import access
    from local_operator.secrets.client import BrokerDenied, BrokerLocked
    from local_operator.secrets.errors import SecretNotFound, SecretStoreError
    from local_operator.secrets.keys import store_path

    # The existence check comes FIRST and is not an optimisation. Against a
    # config dir with no store, `open_store` SPAWNS A BROKER DAEMON before
    # failing, leaving `broker.lock` and `broker.sock` behind on the machine of
    # a user who has never run `lop secret set`. Measured: the guard costs
    # ~41-51 us, the unguarded call costs a stray daemon. Precedent for the
    # guard and for this whole tri-state is `mcp/credentials.py`'s
    # `credential_source`.
    if not store_path(base).exists():
        return False
    try:
        access.open_store(base).describe(QWENCLOUD_TICKET_SECRET_NAME)
        return True
    except SecretNotFound:
        return False
    # Before the SecretStoreError clause: both subclass it.
    except (BrokerDenied, BrokerLocked) as exc:
        raise TicketStoreLocked(
            "the secret store is hardened and locked, so whether the ticket's "
            "value is present is UNKNOWN. Run `lop secret unlock`, then retry"
        ) from exc
    except SecretStoreError as exc:
        raise TicketStoreUnreadable(
            f"the encrypted ticket store could not be read ({type(exc).__name__})"
        ) from exc


def read_ticket_record(store: Any, *, base: Path | None = None) -> dict[str, Any] | None:
    """The stored row's metadata, or None when no row exists.

    Returns ``credential_id``, ``captured_at``, ``length`` and
    ``secret_present``. **Never the value** -- see :func:`_secret_is_present`
    for why this stays retrieval-free.

    ``secret_present=False`` with a row present is a METADATA ORPHAN: something
    IS stored and the user must be told, so it is a record and not a None.
    Collapsing it would be the same false success :class:`TicketStoreUnreadable`
    exists to prevent.

    Raises :class:`TicketStoreUnreadable` when the store cannot be read, which
    is NOT the same answer as "nothing is stored" — see that class, and
    :class:`TicketStoreLocked` for the locked case, which has a remedy.

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
        if not isinstance(data, dict):
            continue
        if data.get("secret_name"):
            length = data.get("length")
            return {
                "credential_id": getattr(row, "id", 0),
                "captured_at": data.get("captured_at"),
                "length": int(length) if isinstance(length, int) else 0,
                "secret_present": _secret_is_present(base),
            }
        if data.get("ticket"):
            # A PRE-MIGRATION row: value still in `auth.db`, no secret written
            # yet. `status` on an un-migrated install must keep working, so
            # report length and age from the plaintext -- and, as everywhere
            # else here, never return the value itself.
            return {
                "credential_id": getattr(row, "id", 0),
                "captured_at": data.get("captured_at"),
                "length": len(str(data["ticket"])),
                "secret_present": True,
            }
    return None


def _delete_secret_value(base: Path | None) -> bool:
    """Remove the ticket VALUE from the encrypted store, confirming it is gone.

    Returns True when a value was actually removed and False when there was
    none to remove. The RETURN VALUE is what lets :func:`delete_ticket` tell a
    SECRET ORPHAN -- a value with no row pointing at it -- apart from an empty
    store, and those two are the difference between "Removed the stored
    QwenCloud console ticket." and "No QwenCloud console ticket stored." over a
    full-account credential that is still on disk.

    "It did not raise" is still not evidence: removal is CONFIRMED by
    re-reading below, and only that path returns True.
    """
    from local_operator.secrets import access
    from local_operator.secrets.client import BrokerDenied, BrokerLocked
    from local_operator.secrets.errors import SecretNotFound, SecretStoreError
    from local_operator.secrets.keys import store_path

    if not store_path(base).exists():
        return False
    try:
        access.open_store(base).delete(QWENCLOUD_TICKET_SECRET_NAME)
    except SecretNotFound:
        # A metadata orphan: nothing to remove on this side. NOT an error --
        # the row still has to go, and refusing here would strand it.
        return False
    except (BrokerDenied, BrokerLocked) as exc:
        raise TicketStoreLocked(
            "the secret store is hardened and locked, so the ticket's value "
            "could not be removed and IT MAY STILL BE STORED. Run "
            "`lop secret unlock`, then retry"
        ) from exc
    except SecretStoreError as exc:
        raise TicketStoreError(
            f"the ticket's encrypted value could not be removed "
            f"({type(exc).__name__}); it may still be stored in the secret store"
        ) from exc
    # CONFIRMED by re-reading, never inferred from the call returning -- the
    # same rule the metadata half already follows below.
    try:
        access.open_store(base).describe(QWENCLOUD_TICKET_SECRET_NAME)
    except SecretNotFound:
        return True
    raise TicketStoreError(
        "the ticket's encrypted value is still present after deleting it; "
        "it may still be stored in the secret store"
    )


def delete_ticket(store: Any, *, base: Path | None = None) -> bool:
    """Remove the stored cookie. True when anything was removed, False when nothing was.

    Raises :class:`TicketStoreUnreadable` when the store cannot be read,
    :class:`TicketStoreLocked` when a hardened store is locked (the value may
    survive, and the message says so), and :class:`TicketStoreError` when
    something was found but is still present afterwards.

    **The VALUE is deleted first, the metadata row second.** If the value goes
    and the row does not, what is left is an inert metadata orphan pointing at
    nothing. The reverse leaves a live encrypted credential with nothing
    pointing at it -- unreachable by this feature and therefore unrevocable
    through it.

    The deletion is CONFIRMED by re-reading rather than inferred from the call
    returning: ``AuthStore.delete_credential`` returns ``None`` whether or not
    it matched anything (auth_store.py:682-688), so "it did not raise" is not
    evidence the cookie is gone. That confirmation now covers BOTH stores, each
    re-read through its own reader. For a revocation command on a full-account
    credential, the difference between those two is the entire value of the
    command.

    **The VALUE is attempted whether or not a row exists**, which is why the
    secret delete is not behind the ``record is None`` return. A SECRET ORPHAN
    -- value present, row absent -- is a state :func:`store_ticket` can
    actually leave behind, because it writes the secret first and the row
    second and prefers that ordering on purpose; restoring an older ``auth.db``
    from a backup produces it too. Returning False before touching the secret
    made ``rm`` print "No QwenCloud console ticket stored." with exit 0 while a
    full-account console session sat in the encrypted store -- a revoke command
    that lies, the exact failure :class:`TicketStoreUnreadable` exists to
    prevent, reintroduced through the other half of the split.

    So the result is the OR of the two halves: True when either store gave
    something up, False only when both were genuinely empty.
    """
    record = read_ticket_record(store, base=base)
    # Not short-circuited on `record is None`: see the docstring. The value has
    # to be reachable without a row, or the orphan is unrevocable through this
    # command -- and this command is the whole mitigation.
    secret_removed = _delete_secret_value(base)
    if record is None:
        return secret_removed
    try:
        store.delete_credential(record["credential_id"])
    except sqlite3.ProgrammingError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise TicketStoreUnreadable(
            f"the ticket could not be deleted ({type(exc).__name__}); " "it may still be stored"
        ) from exc
    if read_ticket_record(store, base=base) is not None:
        raise TicketStoreError(
            "the ticket is still present after deleting it; it may still be "
            "stored in the credential store"
        )
    return True
