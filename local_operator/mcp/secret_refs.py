"""Resolve ``${NAME}`` secret references in a server's ``env`` and ``headers``.

The desktop writer refuses to store a literal secret: ``MCPControl.validate_control``
in :mod:`local_operator.mcp.desktop` accepts only a secret reference such as
``${TOKEN}`` in ``env``/``headers`` and errors with "Environment and header values
must be secret references such as ${TOKEN}". Nothing used to read that reference
back, so a server added through the UI was handed the literal text
``${HUBSPOT_TOKEN}`` as its environment variable or sent it as an HTTP header,
and could never authenticate. This module is the reader half of that contract,
applied at the top of :meth:`McpManager._connect_server` — before anything is
spawned or sent, so a refusal leaves nothing to unwind (an OAuth server's
proactive refresh, :func:`~local_operator.mcp.auth.ensure_mcp_oauth_fresh`, runs
after this and is therefore skipped entirely for a server that cannot start).

**The store is the one the desktop Settings surface writes.**
``<config dir>/credentials.env`` through
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

**The reference rule.** A reference is ``${NAME}`` with NAME
``[A-Za-z_][A-Za-z0-9_]*``, and a value is read one of five ways:

* No ``${`` anywhere: passed through untouched, and the store is never read.
* A ``${`` that is neither a well-formed reference nor a fragment naming a key
  the store holds — ``${1BAD}``, ``${{x}}``, ``${a b}``, an unterminated
  ``${NAME``, ``${HOME}``: passed through untouched. It is not a reference, the
  desktop writer cannot produce one (its pattern is anchored to a whole value of
  ``${NAME}``), and refusing it would break a project ``.mcp.json`` or an
  imported foreign config whose child expands ``${HOME}`` itself. **The escape
  is ``$${``**: ``$${HOME}`` passes the literal text ``${HOME}`` through, which
  is what a value meaning a reference literally should be written as.
* A fragment whose inner text — before shell/compose decorations (see
  :func:`_candidate_keys`) — IS a key the store holds: REFUSED, never handed
  over. The store accepts any key (``CredentialManager.set_credential`` checks
  only control characters), so ``${hubspot-token}`` and ``${TOKEN:-}` can name
  real credentials while sitting outside the reference's name class; a fragment
  naming a stored key cannot be a legitimate literal, and passing it through is
  the silent-unauthenticated-server failure this module exists to remove.
* ``"${NAME}"`` as the whole value: the value is the secret. This is the only
  shape the desktop writer accepts.
* ``"Bearer ${NAME}"``: one or more well-formed references with literal text
  around them, each substituted. The writer never produces this shape, but a
  hand-written ``mcp.json`` needs it.
* A value that holds a well-formed reference AND a ``${`` that is a fragment
  (unless escaped) is REFUSED rather than substituted in part: ``"${TOKEN}${1BAD}"``
  half-applied still leaves the server unable to authenticate.

A well-formed reference whose NAME is absent from the store, or present with an
empty value, is REFUSED with :class:`McpSecretRefError` naming the server, the
entry and the key — never resolved to the literal, and never downgraded to a
warning. A bare ``$NAME`` without braces is not a reference and is untouched.

**Not resolved, deliberately:** server ``args`` (a secret in argv is
world-readable through ``ps``, which is why the writer's reference requirement
covers exactly ``env`` and ``headers``) and the OAuth block's ``client_secret``
(``auth.auth``/``oauth.client_secret``), which reaches the token endpoint
verbatim.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Mapping, TypeVar

from local_operator.mcp.config import MCPServerConfig
from local_operator.paths import config_dir

if TYPE_CHECKING:
    from pydantic import SecretStr

#: The name form both sides accept — the writer's own character class.
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: Decorations a fragment may wrap a key name in. ``env:NAME`` is the compose
#: "from the environment" form; a shell parameter operator (``${NAME:-default}``,
#: ``${NAME-default}``, ``${NAME:?err}``, …) puts the key first and the operator
#: and its operand after it. Both are stripped to find the key the author meant.
_DECORATION_PREFIXES = ("env:", "dotenv:")
_OPERATOR_RE = re.compile(r":-|:=|:\?|:\+|-|\?|\+|:")

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


def _missing(server: str, field: str, entry: str, key: str) -> McpSecretRefError:
    """The refusal for a reference whose key the store does not hold.

    The escape is offered here because this message is what the user reads: a
    well-formed reference to a key that is not in the store is also how a config
    written for another tool (``${HOME}``, a shell ``${VAR}``) looks, and the
    remedy for that case is to keep the text literal.
    """
    return McpSecretRefError(
        f"MCP server {server!r} needs {key} from the credential store for {field} "
        f"{entry} — add it in Settings > API credentials, then reconnect "
        f"(or double the $ to pass it through literally)"
    )


def _malformed(server: str, field: str, entry: str) -> McpSecretRefError:
    """The refusal for a value that holds a reference AND something that is not one.

    Names the field and the entry but never quotes the value: a value that got
    this far may contain literal credential text, and every message here reaches
    the log. The escape is described rather than shown, for the same reason.
    """
    return McpSecretRefError(
        f"MCP server {server!r} has a malformed secret reference in {field} "
        f"{entry} — a reference is written ${{NAME}} (double the $ to keep a literal)"
    )


def _unusable(server: str, field: str, entry: str, key: str) -> McpSecretRefError:
    """The refusal for a fragment that names a key the store DOES hold.

    The distinction from :func:`_missing` is the whole point of this message: the
    key exists, so telling the user to add it would send them to look at a store
    entry they already have. What is wrong is the fragment's form, and the two
    remedies are a reference-shaped key or the escape.
    """
    return McpSecretRefError(
        f"MCP server {server!r} has an unusable secret reference in {field} {entry} — "
        f"{key} is in the credential store, so write ${{NAME}} or double the $ to keep "
        f"the literal"
    )


def _candidate_keys(inner: str) -> list[str]:
    """The key names ``inner`` may have meant, most specific first.

    Only used to decide whether a fragment is intent-to-reference at all, so a
    merely plausible candidate is enough to refuse: a value that names a stored
    credential cannot have meant it as literal text, and the escape is always
    available for a value that genuinely did.
    """
    candidates = [inner]
    body = inner
    for prefix in _DECORATION_PREFIXES:
        if body.startswith(prefix):
            body = body[len(prefix) :]
            candidates.append(body)
    head = _OPERATOR_RE.split(body, maxsplit=1)[0]
    if head != body:
        candidates.append(head)
    stripped = body.strip()
    if stripped != body:
        candidates.append(stripped)
    return [candidate for candidate in candidates if candidate]


def _store_values() -> dict[str, SecretStr]:
    """The credential store's values, as ``SecretStr``, keyed by name.

    Values stay WRAPPED: the mapping is built on every connect that has a
    reference, and unwrapping here would copy every credential in the file —
    unrelated providers' keys included — into plaintext ``str`` for no benefit.
    :func:`_resolve_value` unwraps the one key it actually substitutes.

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
    # ``dict(...)`` is a shallow copy: the manager hands back its own live dict.
    return dict(CredentialManager(base).get_credentials())


def _resolve_value(
    value: str, *, server: str, field: str, entry: str, store: Mapping[str, SecretStr]
) -> str | None:
    """``value`` with its references substituted, or ``None`` when nothing changes.

    One left-to-right scan, because the escape (``$${``) and the reference
    opener are the same two characters: a pass that looked for ``${`` first
    would see the escaped form as a reference and could not tell them apart
    afterwards.
    """
    if "${" not in value:
        return None
    pieces: list[str] = []
    cursor = 0
    index = 0
    resolved = 0
    fragments = 0
    escapes = 0
    unusable_key: str | None = None
    while index < len(value):
        if value[index] != "$":
            index += 1
            continue
        if value.startswith("$${", index):
            # The escape: a literal ``${``, never a reference. Consumed as one
            # unit, so the ``{…}`` behind it stays ordinary text.
            pieces.append(value[cursor:index])
            pieces.append("${")
            index += 3
            cursor = index
            escapes += 1
            continue
        if not value.startswith("${", index):
            index += 1
            continue
        close = value.find("}", index + 2)
        inner = None if close == -1 else value[index + 2 : close]
        if inner is not None and _NAME_RE.fullmatch(inner):
            secret = _substitute_value(store, inner)
            if secret is None:
                raise _missing(server, field, entry, inner)
            pieces.append(value[cursor:index])
            pieces.append(secret)
            index = close + 1
            cursor = index
            resolved += 1
            continue
        fragments += 1
        if inner is not None and unusable_key is None:
            unusable_key = next((key for key in _candidate_keys(inner) if key in store), None)
        # An unterminated fragment consumes the rest of the value: there is no
        # closing brace to resume from, and ``find`` found none.
        index = len(value) if inner is None else close + 1
    if unusable_key is not None:
        raise _unusable(server, field, entry, unusable_key)
    if fragments and resolved:
        raise _malformed(server, field, entry)
    if not resolved and not escapes:
        # Nothing here is a reference and there is nothing to unescape, so the
        # value is literal text and the caller must keep it byte-identical.
        return None
    pieces.append(value[cursor:])
    return "".join(pieces)


def _substitute_value(store: Mapping[str, SecretStr], name: str) -> str | None:
    """``name``'s value, unwrapped, or ``None`` when it is absent or empty.

    An empty value counts as missing, the way the credentials surface lists only
    non-empty keys: substituting ``""`` would send an empty credential and
    re-create the failure this module removes.
    """
    from pydantic import SecretStr

    entry = store.get(name)
    if not isinstance(entry, SecretStr):
        return None
    return entry.get_secret_value() or None


def resolve_config_secrets(name: str, cfg: _ConfigT) -> _ConfigT:
    """``cfg`` with the ``${NAME}`` references in its ``env``/``headers`` resolved.

    Returns ``cfg`` ITSELF when no value carries a reference, so a config that
    uses no references behaves byte-identically to before this module existed:
    no store read, no copy, nothing new for the transports to see.

    Raises :class:`McpSecretRefError` for a reference that cannot be resolved,
    for a fragment that names a stored key, and for a value that mixes one with
    the other. Never a warning, and never the literal.
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
