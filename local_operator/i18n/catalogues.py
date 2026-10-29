"""Catalogue access: ``catalogues/<locale>/<namespace>.json`` message maps.

A catalogue is one flat JSON object of message-id -> ICU message string (§2.2).
This module is the only reader of those files: the loader, the raw bytes behind
``content_sha256``/``ETag``, and the locale/namespace listings the pickers and
the route use.

SECURITY, because a public route takes these two path components: every
component is validated against a conservative grammar and the resolved path is
checked to stay inside the catalogue root. ``..`` and separators are refused at
the component level, so a traversal attempt raises :class:`CatalogueNotFound`
before any filesystem call — the route then answers 404 like any other unknown
name, and a malicious input never gets to describe a path back to the caller.

The hash rule is part of the contract, not an implementation detail: the
``content_sha256`` of a catalogue is the SHA-256 of the COMMITTED FILE BYTES
(not of a re-serialised map), because the translation ledger's
``source_sha256`` (§6) is taken the same way. Two spellings of the same map
would hash differently and staleness would fire; one spelling cannot.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

#: The shipped root. Tests monkeypatch it to point at fixture trees; the route
#: and the runtime always go through the module attribute so that stays true.
_CATALOGUES = Path(__file__).with_name("catalogues")

#: Conservative component grammar for a locale or a namespace: letters, digits,
#: dot, dash, underscore — and it must START with a letter or digit, so `.`,
#: `..` and dot-files are structurally impossible.
_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: Suffix distinguishing the en-only context sidecar from a namespace file.
CONTEXT_SUFFIX = ".context.json"

#: Backend-owned namespaces a client is allowed to fetch. RFC §2.1: `wire.*`
#: is core-owned and served; every client-owned namespace (`ui.*`, `tui.*`,
#: `relay.*`, ...) bundles with its own client, so serving it here would create
#: a second source of truth for strings that client already ships.
WIRE_NAMESPACE_PREFIX = "wire."

#: Wire namespaces excluded from the EAGER per-locale bundle and served on
#: demand through the existing per-namespace route instead.
#:
#: The set is a deliberate, reviewed policy rather than a size heuristic for
#: two reasons. (1) `wire.settings` is the settings registry's label/help copy
#: (804 keys, 80,658 B in `en`, 133,641 B in `hi`); the desktop settings surface
#: receives those as plain `label`/`help` strings from `/v1/settings`, not as
#: `{code, params, text}` envelope identities, so no envelope renderer needs
#: them at startup. (2) Re-splitting the file into child namespaces would move
#: already-audited translated content into new namespaces and invalidate the
#: translation ledger's per-namespace `source_sha256` records; deferral keeps
#: that audit trail intact while removing the same bytes from the eager load.
#: A per-namespace ceiling test (`tests/unit/i18n/test_catalogue_sizes.py`)
#: asserts this set exactly, so a new large namespace cannot join it silently.
DEFERRED_NAMESPACES: dict[str, str] = {
    "wire.settings": (
        "settings registry labels/help, consumed as plain strings by the "
        "settings surface; no desktop envelope identity uses this namespace"
    ),
}


class CatalogueNotFound(KeyError):
    """No catalogue file exists for the requested (locale, namespace)."""


class CatalogueInvalid(ValueError):
    """A catalogue file exists but is not a flat string->string JSON object."""


def catalogue_root() -> Path:
    """The directory catalogues are read from (monkeypatch point for tests)."""
    return _CATALOGUES


def _component(value: str, kind: str) -> str:
    if not _COMPONENT_RE.match(value):
        raise CatalogueNotFound(f"{kind} {value!r} is not a valid catalogue name")
    return value


def _file(locale: str, namespace: str, suffix: str = ".json") -> Path:
    safe_locale = _component(locale, "locale")
    safe_namespace = _component(namespace, "namespace")
    root = catalogue_root()
    path = (root / safe_locale / f"{safe_namespace}{suffix}").resolve()
    # Belt and braces after the grammar check: a symlinked locale directory
    # must not be able to escape the root.
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise CatalogueNotFound(f"{locale}/{namespace} escapes the catalogue root") from None
    return path


def _path(locale: str, namespace: str) -> Path:
    return _file(locale, namespace)


def locales() -> tuple[str, ...]:
    """Locales that have at least one namespace file, sorted."""
    root = catalogue_root()
    if not root.is_dir():
        return ()
    found = {
        child.name
        for child in root.iterdir()
        if child.is_dir() and any((child / f).name.endswith(".json") for f in child.iterdir())
    }
    return tuple(sorted(found))


def namespaces(locale: str) -> tuple[str, ...]:
    """Namespaces of ``locale``: its namespace files, sidecars excluded, sorted."""
    safe_locale = _component(locale, "locale")
    directory = catalogue_root() / safe_locale
    if not directory.is_dir():
        raise CatalogueNotFound(f"locale {locale!r} is not shipped")
    found = {
        entry.name[: -len(".json")]
        for entry in directory.iterdir()
        if entry.name.endswith(".json") and not entry.name.endswith(CONTEXT_SUFFIX)
    }
    return tuple(sorted(found))


def catalogue_bytes(locale: str, namespace: str) -> bytes:
    """The raw committed bytes of one catalogue file (hash input, ETag source)."""
    path = _path(locale, namespace)
    if not path.is_file():
        raise CatalogueNotFound(f"no catalogue for {locale}/{namespace}")
    return path.read_bytes()


def catalogue_sha256(locale: str, namespace: str) -> str:
    """``sha256`` hex digest of the file bytes — the ``content_sha256`` contract."""
    return hashlib.sha256(catalogue_bytes(locale, namespace)).hexdigest()


def load_catalogue(locale: str, namespace: str) -> dict[str, str]:
    """Parse one catalogue into a message map, validating its shape.

    A catalogue must be valid UTF-8 JSON forming an object whose values are
    all strings; anything else — a truncated or garbled file included — raises
    :class:`CatalogueInvalid` rather than handing a half-typed map to a
    renderer. The checker validates the same shape at build time, so this
    is the runtime twin of that guarantee, not the only line of defence.
    """
    raw = catalogue_bytes(locale, namespace)
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        # Decode/parse failures surface as THE catalogue exception the render
        # path handles. A bare ValueError here would escape `messages.render`
        # (and the resolver's probe) and break the never-raise contract a wire
        # client reads through — the runtime twin of the checker's parse gate.
        raise CatalogueInvalid(f"{locale}/{namespace}: file is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CatalogueInvalid(f"{locale}/{namespace}: catalogue root must be an object")
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise CatalogueInvalid(
                f"{locale}/{namespace}: message {key!r} must map a string id to a string"
            )
    return data


def wire_namespaces(locale: str) -> tuple[str, ...]:
    """The ``wire.*`` namespaces a locale carries, sorted.

    Deliberately scoped to :data:`WIRE_NAMESPACE_PREFIX`: the per-locale bundle
    serves the backend-owned set only, and the client-owned namespaces that ship
    alongside it in the same package are not this route's to serve.
    """
    return tuple(
        namespace for namespace in namespaces(locale) if namespace.startswith(WIRE_NAMESPACE_PREFIX)
    )


def wire_bundle(locale: str) -> dict[str, Any]:
    """Every ``wire.*`` namespace of one locale, ready for one client fetch.

    The eager part is the flattened message map: keys already repeat their
    namespace, so merging the files cannot collide (the checker guarantees the
    prefixes). Namespaces in :data:`DEFERRED_NAMESPACES` are described in
    ``deferred`` but their messages are omitted; a client fetches one later with
    the per-namespace route when a surface actually needs it.

    ``content_sha256`` is the bundle's cache identity: it covers the locale,
    every wire namespace's committed-file hash, and each namespace's eager or
    deferred membership. The membership term is load-bearing, not decoration:
    a client revalidates with the ETag and re-registers only on a 200 (see the
    desktop catalogue loader), so a namespace moving between the two sets must
    change the validator even though its own bytes did not. Without it, a
    namespace that became newly eager kept its old ETag, the client got a 304,
    and those messages stayed English indefinitely. It is a digest of the
    hashes and membership markers, not of a re-serialised map, keeping the same
    raw-bytes rule as the file hashes.
    """
    digest = hashlib.sha256()
    digest.update(locale.encode("utf-8") + b"\n")
    eager: dict[str, str] = {}
    eager_hashes: dict[str, str] = {}
    deferred: list[dict[str, Any]] = []
    for namespace in wire_namespaces(locale):
        sha = catalogue_sha256(locale, namespace)
        deferred_flag = namespace in DEFERRED_NAMESPACES
        digest.update(
            f"{namespace}\t{sha}\t{'deferred' if deferred_flag else 'eager'}\n".encode("utf-8")
        )
        if deferred_flag:
            deferred.append(
                {
                    "namespace": namespace,
                    "content_sha256": sha,
                    "bytes": len(catalogue_bytes(locale, namespace)),
                    "keys": len(load_catalogue(locale, namespace)),
                }
            )
            continue
        eager_hashes[namespace] = sha
        eager.update(load_catalogue(locale, namespace))
    return {
        "locale": locale,
        "messages": eager,
        "namespaces": eager_hashes,
        "deferred": deferred,
        "content_sha256": digest.hexdigest(),
    }


def context(locale: str, namespace: str) -> dict[str, dict[str, Any]]:
    """The en-only context sidecar for a namespace, or ``{}`` when absent.

    The sidecar (role/tone/maxCells/notes per key) is part of the contract for
    width-sensitive TUI strings; absent is a valid state until a slice adds it.
    """
    path = _file(locale, namespace, suffix=CONTEXT_SUFFIX)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CatalogueInvalid(f"{locale}/{namespace} context sidecar must be an object")
    return data
