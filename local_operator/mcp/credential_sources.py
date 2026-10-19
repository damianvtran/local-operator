"""Adopt literal MCP credentials into the encrypted store, once, at the source.

WHY THIS EXISTS
===============

A user-scope ``mcp.json`` may hold a credential as a LITERAL in ``env`` or
``headers`` — that is how many hand-written entries and imported foreign configs
were authored. The mesh definitions channel (``network.mcpdefs``) deliberately
carries only *references*: a value is replaced by ``ref:<NAME>`` when it is
exactly one ``${NAME}``, and a literal never travels. So a head node with
literal credentials produced a mirror on every peer that declares

    GOOGLE_OAUTH_CLIENT_ID: "${GOOGLE_OAUTH_CLIENT_ID}"
    Authorization: "${Authorization}"

and no store anywhere holds those names. The mirror says "needs this key" but
there is no key to copy, because the needs-list (``public_secret_refs``) only
ever saw references. This module closes that loop at the owner: turn each
literal into a store row plus an exact ``${NAME}`` reference, and the ordinary
definitions/placement/copy path does the rest.

WHAT IS IN SCOPE, AND WHAT IS NOT
=================================

Only ``<root>/mcp.json`` — the USER-scope document every writer in
``mcp.config`` targets. A project ``.mcp.json`` is untrusted input (see the
trust model in ``docs/mcp.md``) and is never read here; a foreign config is
never a source for this operation either. OAuth material is never touched: MCP
OAuth tokens live in ``auth.db`` and are brokered, never copied, and this module
only ever reads the two plain maps above.

THE REWRITE IS THE WHOLE VALUE, INCLUDING ANY SCHEME
====================================================

``mcpdefs.environment_state`` only classifies a value as a reference when it is
EXACTLY ``${NAME}``. ``Authorization: Bearer abc`` must therefore become exactly
``${MCP_SLACK_AUTHORIZATION}`` with the WHOLE string — scheme included — in the
store row. Leaving ``Bearer `` outside the reference would keep the literal
off the wire while never resolving the header, which is the silent half-fix
this module exists to prevent.

NAMING, AND WHY A COLLISION IS A REFUSAL
========================================

* ``env``: the environment key itself when the store is free of it or already
  holds the identical bytes; otherwise ``MCP_<SERVER>_<KEY>``.
* ``headers``: always ``MCP_<SERVER>_<HEADER>`` (an ``Authorization`` key is
  far too generic to claim store-wide).

Names are uppercased with non-alphanumerics collapsed to ``_`` and capped at 96
characters. A name that already holds a DIFFERENT value is reported as a
conflict and never overwritten: the operator's existing secret is not this
module's to destroy, and two different credentials collapsing onto one name is
exactly the case a silent ``update`` would make unrecoverable.

ORDER OF WRITES, AND FAILURE
============================

Store rows are written first; the config is rewritten once, at the end, through
``mcp.config.write_scope_document`` (atomic, mode 0600). A store failure
therefore leaves the document untouched — the reference that would resolve is
never published ahead of the value it names. Nothing here logs, prints, returns
or raises with a value: the receipt carries names only, and exception messages
name the operation, never the input.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from local_operator.mcp.config import write_scope_document

#: The store's own names may be 256 characters; this operation's generated names
#: are capped well below that so a pathological server/key pair cannot produce a
#: name an operator cannot type or a display that overflows a row.
SECRET_NAME_LIMIT = 96

#: The audit attribution for adoption writes. A fixed, non-secret label: the
#: operator later inspecting ``lop secret audit`` can tell these rows apart from
#: live session reads.
ADOPTION_SESSION = "mcp-adopt"

_COLLAPSE = re.compile(r"[^A-Za-z0-9]+")

#: Fields this module reads. Everything else in an entry is preserved verbatim.
_FIELDS = ("env", "headers")


class CredentialAdoptionError(RuntimeError):
    """Adoption could not complete. The message never carries a value."""


def _collapsed(text: str) -> str:
    """Uppercase, with runs of non-alphanumerics collapsed to one underscore."""
    return "_".join(part for part in _COLLAPSE.split(text) if part).upper()


def _bounded(name: str) -> str:
    """Cap a generated secret name at :data:`SECRET_NAME_LIMIT` characters."""
    bounded = name[:SECRET_NAME_LIMIT].rstrip("_")
    return bounded or "MCP_CREDENTIAL"


def _generated_name(server: str, key: str) -> str:
    return _bounded(f"MCP_{_collapsed(server) or 'SERVER'}_{_collapsed(key) or 'VALUE'}")


def _usable_store_name(name: str) -> bool:
    """Whether ``name`` may be used as-is: an env key is not always a NAME.

    THE REFERENCE SPELLING IS PART OF THE CONTRACT, not cosmetics: the rewrite is
    ``${NAME}``, and a key like ``MY-TOKEN`` is a legal environment variable but
    NOT a legal reference name — ``public_secret_refs`` would never publish it,
    so adoption would reproduce the exact invisible-literal bug it fixes. The
    check therefore asks the ONE parser (with the key plugged in as its own
    reference) rather than keeping a second regex here.

    Also refused: a control character (the store's rule, which would abort
    adoption for every other server in the file) and the provider-owned
    namespace, which an ordinary ``mcp.json`` env key could otherwise shadow.
    """
    from types import SimpleNamespace

    from local_operator.mcp.secret_refs import public_secret_refs
    from local_operator.secrets.store import is_provider_secret_name

    if not name or any(ord(character) < 0x20 or ord(character) == 0x7F for character in name):
        return False
    if is_provider_secret_name(name):
        return False
    published = public_secret_refs(SimpleNamespace(env={name: "${" + name + "}"}, headers={}))
    # EXACTLY this name, not merely "something": ``${A}}`` publishes ``A``, so a
    # key containing ``}`` would be rewritten to a reference that resolves to a
    # DIFFERENT (absent) name and its literal would stop working.
    return [str(ref.get("id") or "") for ref in published] == [name]


def _read_document(path: Path) -> dict[str, Any]:
    """The user-scope document; ``{}`` when the file does not exist."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise CredentialAdoptionError(
            f"the user-scope MCP config could not be read ({exc.__class__.__name__})"
        ) from None
    try:
        document = json.loads(text)
    except ValueError:
        raise CredentialAdoptionError("the user-scope MCP config is not valid JSON") from None
    if not isinstance(document, dict):
        raise CredentialAdoptionError("the user-scope MCP config is not a JSON object")
    return document


def _entries(document: Mapping[str, Any]) -> dict[str, Any]:
    entries = document.get("mcpServers")
    return dict(entries) if isinstance(entries, Mapping) else {}


def _receipt() -> dict[str, list[dict[str, Any]]]:
    return {"adopted": [], "reused": [], "conflicts": [], "skipped": []}


def _row(server: str, field: str, key: str, **extra: Any) -> dict[str, Any]:
    return {"server": server, "field": field, "key": key, **extra}


def literal_credential_keys(root: Path) -> list[dict[str, Any]]:
    """Every ``env``/``headers`` value that is a literal, for status surfaces.

    One row per binding: ``server``, ``field`` (``env``/``headers``), ``key`` and
    ``has_value`` (False for an empty literal). No value is read out, copied or
    returned — this exists so readiness can say "these still need adoption"
    without a store read or a definitions push.
    """
    document = _read_document(root / "mcp.json")
    rows: list[dict[str, Any]] = []
    for server, raw in sorted(_entries(document).items(), key=lambda pair: str(pair[0])):
        if not isinstance(raw, dict):
            continue
        for field in _FIELDS:
            values = raw.get(field)
            if not isinstance(values, Mapping):
                continue
            for key, value in sorted(values.items(), key=lambda pair: str(pair[0])):
                if not isinstance(value, str) or "${" in value:
                    # A reference (or an escape) is not a literal; nothing to
                    # adopt and nothing for readiness to warn about.
                    continue
                rows.append(
                    {
                        "server": str(server),
                        "field": field,
                        "key": str(key),
                        "has_value": bool(value),
                    }
                )
    return rows


class _StoreIndex:
    """Existing store names, read lazily and compared byte-for-byte.

    ``open_store(create=True)`` is deliberately deferred until the first literal
    that would actually be written: a device with no literals must not gain a
    store (or a store database) from a status read.
    """

    def __init__(self, root: Path, *, dry_run: bool) -> None:
        self.root = root
        self.dry_run = dry_run
        self._store: Any = None
        self._names: set[str] = set()
        self._opened = False
        # Values written earlier in THIS run, so a second occurrence is compared
        # against the pending write rather than a not-yet-visible store row.
        self.pending: dict[str, bytes] = {}

    def _open(self) -> Any:
        """An existing store, or ``None`` when the database was never created.

        THE DATABASE FILE, NOT THE KEY, IS THE EXISTENCE TEST: ``open_store``
        resolves (and with ``create=True`` may install) a master key, but a key
        is not a database. ``SecretStore.list`` on a MISSING database file
        raises ``SecretStoreError`` from its read-only ``_open``; once the file
        exists, ``list`` returns ``[]`` (and reports damaged rows separately).
        A status read must not create either, so absence is answered here and
        ``write`` is the only path that lets ``set`` initialise one.
        """
        if not self._opened:
            from local_operator.secrets import access
            from local_operator.secrets.keys import store_path

            if store_path(self.root).exists():
                try:
                    self._store = access.open_store(self.root, create=True)
                    self._names = {record.name for record in self._store.list()}
                except Exception as exc:  # noqa: BLE001 — never quote a store exception here
                    raise CredentialAdoptionError(
                        f"the encrypted store could not be opened ({exc.__class__.__name__})"
                    ) from None
            self._opened = True
        return self._store

    def classify(self, name: str, value: bytes) -> str:
        """``free``, ``identical`` or ``conflict`` for one candidate name."""
        if name in self.pending:
            return "identical" if self.pending[name] == value else "conflict"
        store = self._open()
        if store is None or name not in self._names:
            return "free"
        from local_operator.secrets.errors import SecretNotFound

        try:
            _record, current = store.read_for_copy(name)
        except SecretNotFound:
            # A name the listing reported but a read cannot find is a race this
            # operation loses safely: treat it as free and let ``set`` say so.
            return "free"
        except Exception as exc:  # noqa: BLE001 — never quote a store exception here
            raise CredentialAdoptionError(
                f"the encrypted store could not be read ({exc.__class__.__name__})"
            ) from None
        return "identical" if current == value else "conflict"

    def write(self, name: str, value: bytes) -> None:
        if self.dry_run:
            self.pending[name] = value
            return
        store = self._open()
        try:
            if store is None:
                from local_operator.secrets import access

                # The first write is what initialises the database; every later
                # write reuses this handle.
                store = access.open_store(self.root, create=True)
                self._store = store
            store.set(
                name,
                value,
                description="Adopted from the user-scope MCP config",
                session_id=ADOPTION_SESSION,
            )
        except Exception as exc:  # noqa: BLE001 — never quote a store exception here
            raise CredentialAdoptionError(
                f"the encrypted store refused the write ({exc.__class__.__name__})"
            ) from None
        self.pending[name] = value
        self._names.add(name)


def adopt_literal_credentials(root: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Adopt every literal ``env``/``headers`` value from ``<root>/mcp.json``.

    Returns a receipt with four lists — ``adopted``, ``reused``, ``conflicts`` and
    ``skipped`` — each row naming ``server``, ``field``, ``key`` and (where one
    was chosen) ``secret``. No row, message or exception carries a value.

    ``reused`` means a store row already held the identical bytes; the config is
    still rewritten to its reference, which is the point of the operation. A
    conflict is a name occupied by DIFFERENT bytes: the entry is left exactly as
    it was and the row says which name collided.

    ``dry_run`` reports what would change and writes neither the store nor the
    document. Idempotent: after a successful run every adopted value is a
    reference, so a second run adopts nothing.
    """
    path = root / "mcp.json"
    document = _read_document(path)
    receipt = _receipt()

    candidates: list[tuple[str, str, str, str, dict[str, Any]]] = []
    for server, raw in sorted(_entries(document).items(), key=lambda pair: str(pair[0])):
        if not isinstance(raw, dict):
            continue
        for field in _FIELDS:
            values = raw.get(field)
            if not isinstance(values, dict):
                continue
            for key, value in sorted(values.items(), key=lambda pair: str(pair[0])):
                server_text = str(server)
                key_text = str(key)
                if not isinstance(value, str):
                    receipt["skipped"].append(_row(server_text, field, key_text, reason="not_text"))
                    continue
                if "${" in value:
                    # References and ``$${`` escapes are not literals: adopting
                    # either would rewrite a value that is already doing a job.
                    receipt["skipped"].append(
                        _row(server_text, field, key_text, reason="reference")
                    )
                    continue
                if not value:
                    receipt["skipped"].append(_row(server_text, field, key_text, reason="empty"))
                    continue
                candidates.append((server_text, field, key_text, value, raw))

    if not candidates:
        return receipt

    index = _StoreIndex(root, dry_run=dry_run)
    changed = False
    for server, field, key, value, raw in candidates:
        payload = value.encode("utf-8")
        if field == "env" and _usable_store_name(key):
            names = [key]
            fallback = _generated_name(server, key)
            if fallback != key:
                names.append(fallback)
        else:
            names = [_generated_name(server, key)]
        chosen = ""
        identical = False
        for name in names:
            state = index.classify(name, payload)
            if state in ("free", "identical"):
                chosen = name
                identical = state == "identical"
                break
        if not chosen:
            receipt["conflicts"].append(
                _row(
                    server,
                    field,
                    key,
                    secret=names[-1],
                    reason="name_holds_a_different_value",
                )
            )
            continue
        if not identical:
            index.write(chosen, payload)
        values = raw.get(field)
        assert isinstance(values, dict)  # the candidate scan required it
        values[key] = "${" + chosen + "}"
        changed = True
        receipt["reused" if identical else "adopted"].append(
            _row(server, field, key, secret=chosen)
        )

    if changed and not dry_run:
        try:
            write_scope_document(path, document)
        except OSError as exc:
            raise CredentialAdoptionError(
                f"the user-scope MCP config could not be written ({exc.__class__.__name__})"
            ) from None

    for name in receipt:
        receipt[name].sort(
            key=lambda row: (row.get("server", ""), row.get("field", ""), row.get("key", ""))
        )
    return receipt


__all__ = [
    "ADOPTION_SESSION",
    "CredentialAdoptionError",
    "SECRET_NAME_LIMIT",
    "adopt_literal_credentials",
    "literal_credential_keys",
]
