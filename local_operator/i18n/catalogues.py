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
