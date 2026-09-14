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
* A fragment whose inner text contains a name the store holds — whatever
decorates it (see :func:`_candidate_keys`) — is REFUSED, never handed over. The
store accepts any key (``CredentialManager.set_credential`` checks only control
characters), and a shell or compose file wraps one in anything at all —
``${hubspot-token}``, ``${TOKEN:-}``, ``${TOKEN#suffix}``, ``${!TOKEN}``,
``${env:TOKEN}`` — so a fragment naming a stored key cannot be a legitimate
literal, and passing it through is the silent-unauthenticated-server failure
this module exists to remove.
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
(``auth.client_secret``/``oauth.client_secret``), which reaches the token
endpoint verbatim.
"""

from __future__ import annotations

import re
from typing import Mapping, TypeVar

from pydantic import SecretStr

from local_operator.mcp.config import MCPServerConfig
from local_operator.paths import config_dir

#: The name form both sides accept — the writer's own character class.
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: Every run of name characters inside a ``${…}`` fragment is a candidate key —
#: see :func:`_candidate_keys`. ``.`` and ``-`` are in the class because the
#: store is free-form: ``hubspot-token`` and ``my.key`` are keys a user can hold.
_NAME_RUN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*")

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


def _unreadable(server: str, field: str, entry: str, key: str) -> McpSecretRefError:
    """The refusal for a store entry this build cannot read as a value.

    Distinct from :func:`_missing` on purpose: the key IS in the store, so
    "add it in Settings > API credentials" would send the user to an entry they
    already have. A value of an unexpected type used to report exactly that,
    because "not a ``SecretStr``" and "absent" were the same branch.
    """
    return McpSecretRefError(
        f"MCP server {server!r} cannot read {key} from the credential store for {field} "
        f"{entry} — the stored value is not a string; re-save it in Settings > "
        f"API credentials"
    )


def _candidate_keys(inner: str) -> list[str]:
    """The key names ``inner`` may have meant: every name-like run it contains.

    Deliberately NOT a list of decorations. Compose wraps a key in ``env:``, a
    shell puts an operator after it (``:-``, ``:?``, ``#``, ``%``, ``/``,
    ``^``, ``@``, ``#NAME``, ``!NAME``), and such a list is always one operator
    short — each miss hands the server the reference as its credential, which is
    the silent failure this module exists to remove. A name run standing inside
    a ``${…}`` fragment is a candidate whatever surrounds it, and the store is
    the oracle: only a name the store actually holds refuses, so a fragment
    naming nothing stored keeps its pass-through and the escape stays available
    for a value that genuinely means the text.

    TWO run shapes, because each closes a hole the other has: the maximal run
    (``.``/``-`` included) matches a store key that CONTAINS punctuation —
    ``hubspot-token``, ``my.key``, which are the keys a user actually holds —
    while the plain-name segments catch a key the shell glued an operator or a
    suffix onto, as in ``${TOKEN-SUB}`` or ``${TOKEN#x}``, where the maximal run
    would be ``TOKEN-SUB`` and would match nothing.

    The raw ``inner`` and its stripped form are candidates too, and that is not
    redundant: the store accepts ANY key, so ``MY KEY``, ``API:KEY`` and
    ``KEY#2`` are names Settings can hold, and no run shape can spell them —
    punctuation outside the run classes splits the text, so the fragment would
    be handed over as a literal. Adding the whole fragment back makes the
    predicate a strict superset of a plain ``inner`` test.
    """
    keys: list[str] = []
    for key in (
        *_NAME_RUN_RE.findall(inner),
        *_NAME_RE.findall(inner),
        inner,
        inner.strip(),
    ):
        if key and key not in keys:
            keys.append(key)
    return keys


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
            if inner not in store:
                raise _missing(server, field, entry, inner)
            secret = _substitute_value(
                store[inner], server=server, field=field, entry_name=entry, key=inner
            )
            if secret is None:
                # Present but empty. The credentials surface lists non-empty keys
                # only, so "needs it" is the same instruction here.
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


def _substitute_value(
    entry: object, *, server: str, field: str, entry_name: str, key: str
) -> str | None:
    """``entry``'s plaintext, or ``None`` when the stored value is empty.

    An empty value counts as missing, the way the credentials surface lists only
    non-empty keys: substituting ``""`` would send an empty credential and
    re-create the failure this module removes.

    A value of any OTHER type raises rather than returning ``None``: the key is
    present, so folding it into the missing-key path would send the user to
    Settings to add something the store already holds — which is exactly what
    the round-1 ``isinstance(entry, SecretStr)`` check did.
    """
    if isinstance(entry, SecretStr):
        return entry.get_secret_value() or None
    if isinstance(entry, str):
        # Not what ``CredentialManager`` returns, but a plain mapping is a shape
        # a test or a future store may hand us; accepting it keeps that from
        # masquerading as an unreadable value.
        return entry or None
    raise _unreadable(server, field, entry_name, key)


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
