"""Resolve ``${NAME}`` secret references in a server's ``env`` and ``headers``.

The desktop writer refuses to store a literal secret: ``MCPControl.validate_control``
in :mod:`local_operator.mcp.desktop` accepts only a secret reference such as
``${TOKEN}`` in ``env``/``headers`` and errors with "Environment and header values
must be secret references such as ${TOKEN}". Nothing used to read that reference
back, so a server added through the UI was handed the literal text
``${HUBSPOT_TOKEN}`` as its environment variable or sent it as an HTTP header,
and could never authenticate. This module is the reader half of that contract,
applied where the transport is built (:meth:`McpManager._connect_server`), so the
child process and the HTTP client are given the secret and never the reference.

**The store is the one the desktop Settings surface writes.**
``<config_dir>/credentials.env`` through
:class:`~local_operator.credentials.CredentialManager`, because the UI's
``credentials.update``/``credentials.list`` ops are ``PATCH``/``GET
/v1/credentials`` (``src/shared/desktop-contract.ts`` in local-operator-ui) and
that route (``local_operator/server/routes/credentials.py``) reads and writes
``CredentialManager``. The memory-only session credentials that ``/credential``
collects (``VariableStore._credentials``) are deliberately NOT consulted: they
never reach disk, so a config file referring to one could not resolve in the
next process, and a reference that works once and then silently does not is the
same class of failure this module removes.

``CredentialManager.get_credential`` falls back to ``os.environ`` for a key the
file does not hold, and that fallback is deliberately not used here. Two
reasons, and the second is the load-bearing one: the file's key set is exactly
what the credentials surface LISTS, so "the reference names something the user
can see and edit" stays true; and an environment fallback would let a
project-scoped ``.mcp.json`` — untrusted input, see the trust model in
``docs/mcp.md`` — copy any variable out of the daemon's own environment into a
remote server's headers, which the allowlisted stdio child environment
(``get_default_environment``) exists to prevent.

**The reference rule**, deliberately the same rule the writer enforces so that
writer and reader cannot drift apart again. A reference is ``${NAME}`` with NAME
``[A-Za-z_][A-Za-z0-9_]*``, and a value is read one of four ways:

* No ``${`` anywhere: passed through untouched, and the store is never read.
* No WELL-FORMED reference anywhere — ``${1BAD}``, ``${{x}}``, ``${a b}``, an
  unterminated ``${NAME``: also passed through untouched. It is not a reference,
  the desktop writer cannot produce one (its pattern is anchored to a whole
  value of ``${NAME}``), and refusing it would make a literal ``${`` impossible
  to express in a config that was written by hand.
* ``"${NAME}"`` as the whole value: the value is the secret. This is the only
  shape the desktop writer accepts.
* ``"Bearer ${NAME}"``: one or more well-formed references with literal text
  around them, each substituted. The writer never produces this shape, but a
  hand-written ``mcp.json`` needs it and a reader that refused to substitute it
  would hand the server a literal reference again.
* A value that holds a well-formed reference AND a malformed ``${`` fragment is
  REFUSED rather than substituted in part: ``"${TOKEN}${1BAD}"`` half-applied
  still leaves the server unable to authenticate, which is the failure this
  module exists to make loud.

A well-formed reference whose NAME is absent from the store, or present with an
empty value, is REFUSED with :class:`McpSecretRefError` naming the server, the
entry and the key — never resolved to the literal, and never downgraded to a
warning. A bare ``$NAME`` without braces is not a reference and is untouched.

Server ``args`` are not resolved at all: a secret in argv is world-readable
through ``ps``, which is why the writer's reference requirement covers exactly
``env`` and ``headers``.
"""

from __future__ import annotations

import re
from typing import TypeVar

from local_operator.mcp.config import MCPServerConfig
from local_operator.paths import config_dir

#: ``${NAME}`` — group 1 is whatever sits between the braces, valid or not, so a
#: malformed reference is RECOGNISED and can be refused rather than half-applied.
_REFERENCE_RE = re.compile(r"\$\{([^}]*)\}")

#: The name form both sides accept — the writer's own character class.
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: The concrete config subtype, preserved through resolution so a caller that
#: knows it holds a stdio config keeps the ``env`` attribute in view.
_ConfigT = TypeVar("_ConfigT", bound=MCPServerConfig)


class McpSecretRefError(RuntimeError):
    """A ``${NAME}`` reference in a server's ``env``/``headers`` did not resolve.

    Raised instead of handing the transport the reference text, and instead of
    starting the server with no value at all: both make a misconfigured server
    look like an authorization failure from the far end. The message names the
    server, the entry and the missing key, and never a resolved value — it
    reaches the session log through ``McpServerStderr.report_failure`` and the
    MCP status surfaces through the connect error path.
    """


def _malformed(server: str, field: str, entry: str) -> McpSecretRefError:
    """The refusal for a value that holds a reference AND something that is not one.

    Names the field and the entry but never quotes the value: a value that got
    this far may contain literal credential text, and every message here reaches
    the log.
    """
    return McpSecretRefError(
        f"MCP server {server!r} has a malformed secret reference in {field} "
        f"{entry} — a reference is written ${{NAME}}"
    )


def _store_values() -> dict[str, str]:
    """Every value in the credential store the Settings surface writes.

    Read fresh on each call rather than cached: the store is written by the API
    server process (``PATCH /v1/credentials``) while a session that already
    holds MCP connections reads it, so a cached snapshot would keep reporting a
    credential the user has just added as missing until something restarted.
    The file is a handful of lines.
    """
    from local_operator.credentials import CREDENTIALS_FILE_NAME, CredentialManager

    base = config_dir()
    if not (base / CREDENTIALS_FILE_NAME).exists():
        # No store exists, so every reference is unresolvable. Read directly
        # rather than constructing CredentialManager, whose constructor CREATES
        # an empty credentials file: a connect must not write to the config dir
        # as a side effect of reading a reference.
        return {}
    return {
        key: value.get_secret_value()
        for key, value in CredentialManager(base).get_credentials().items()
    }


def _resolve_value(
    value: str, *, server: str, field: str, entry: str, store: dict[str, str]
) -> str | None:
    """``value`` with its references substituted, or ``None`` when it has none."""
    if "${" not in value:
        return None
    complete = list(_REFERENCE_RE.finditer(value))
    well_formed = [match for match in complete if _NAME_RE.fullmatch(match.group(1))]
    # ``sub`` removes every COMPLETE ``${...}``, so anything left holding ``${``
    # is an unterminated fragment (``${NAME`` and friends).
    unterminated = "${" in _REFERENCE_RE.sub("", value)
    if not well_formed:
        # Nothing here IS a reference — a literal ``${`` in a hand-written config,
        # unterminated or not. Refusing it would make one inexpressible.
        return None
    if len(well_formed) != len(complete) or unterminated:
        raise _malformed(server, field, entry)
    pieces: list[str] = []
    cursor = 0
    for match in well_formed:
        secret = store.get(match.group(1), "")
        if not secret:
            # An empty value counts as missing, the way the credentials surface
            # lists only non-empty keys: substituting "" would send an empty
            # credential and re-create the failure this module removes.
            raise McpSecretRefError(
                f"MCP server {server!r} needs {match.group(1)} from the credential store for "
                f"{field} {entry} — add it in Settings > API credentials, then reconnect"
            )
        # The literal text BETWEEN references is kept verbatim, which is what
        # makes ``Bearer ${NAME}`` a usable header value.
        pieces.append(value[cursor : match.start()])
        pieces.append(secret)
        cursor = match.end()
    pieces.append(value[cursor:])
    return "".join(pieces)


def resolve_config_secrets(name: str, cfg: _ConfigT) -> _ConfigT:
    """``cfg`` with the ``${NAME}`` references in its ``env``/``headers`` resolved.

    Returns ``cfg`` ITSELF when no value carries a reference, so a config that
    uses no references behaves byte-identically to before this module existed:
    no store read, no copy, nothing new for the transports to see.

    Raises :class:`McpSecretRefError` for a reference that cannot be resolved,
    including a value that mixes one with a malformed fragment. Never a warning,
    and never the literal.
    """
    fields = (
        ("env", getattr(cfg, "env", None) or {}),
        ("headers", getattr(cfg, "headers", None) or {}),
    )
    if not any("${" in value for _, values in fields for value in values.values()):
        return cfg
    store = _store_values()
    updates: dict[str, dict[str, str]] = {}
    for field, values in fields:
        if not values:
            continue
        resolved = {
            entry: _resolve_value(value, server=name, field=field, entry=entry, store=store)
            for entry, value in values.items()
        }
        updates[field] = {
            entry: value if value is not None else values[entry]
            for entry, value in resolved.items()
        }
    # The PRISTINE config stays where the caller put it: ``config_digest`` keys
    # the tool cache on the stored config, and a digest of resolved secrets
    # would both persist evidence of them and change on every rotation.
    return cfg.model_copy(update=updates)


__all__ = ["McpSecretRefError", "resolve_config_secrets"]
