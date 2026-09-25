"""The MCP catalog: the ONE row builder behind ``/v1/desktop/mcp``.

Scope, because the claim is easy to overstate: these rows are built here and
nowhere else, so no client of the sessionless surface can be shown a second
opinion of the same server. The LEGACY session route
(``/v1/desktop/sessions/{id}/mcp``) is deliberately NOT built here — desktop
builds that predate ``features.mcp_catalog`` read its frozen shape (its own key
names, ``stdio``/``http``, the manager's status words), and
``MCPDesktop.snapshot`` in ``mcp/desktop.py`` still derives it, because two
compatibility requirements cannot both be met by one body. What crosses between
them is ONE seam: ``live_facts_from_snapshot`` translates the runtime's rows
into the ``live`` overlay below. So the duplication is a translation of a frozen
shape, not a competing definition of the catalog's vocabulary.

Settings > Integrations used to read MCP state THROUGH a conversation: every
read and write needed a session id, a fresh install with no model could not
start one, and the page borrowed "your most recent conversation" to show a
configuration file that belongs to no conversation at all. Almost everything
that page shows is a pure function of files on disk — the merged config set,
the shared OAuth grant store (``auth.db``), the encrypted secret store and the
tool cache — so this module computes it from exactly those, with no runtime.

What only a runtime can know ("connected right now") arrives as an OPTIONAL
overlay (``live``), and what only an explicit test can know arrives as a
``probe`` result. Precedence per row, first match wins:

1. the config does not validate → ``error``;
2. an operation (test / sign-in) running for the server → ``connecting``, basis
   ``operation``;
3. a live overlay for a server that runtime actually loaded;
4. a probe result within :data:`PROBE_TTL_S` for the CURRENT config digest;
5. durable facts: a missing grant or key → ``needs_sign_in``, else
   ``not_started``.

The status vocabulary is deliberately plain (``connected``, ``needs_sign_in``,
``not_started``, ``connecting``, ``error``) and ``status_basis`` says how we
know, because a stored fact is not a live one: a token revoked upstream still
reads as signed in, so ``stored`` is never allowed to claim ``connected``.

The four bases are ``live`` (a warm runtime reported it), ``probe`` (an explicit
Test or sign-in ANSWERED it), ``operation`` (one is running right now) and
``stored`` (config and the durable stores). ``operation`` is a basis of its own
rather than a second meaning of ``probe`` for a rendering reason: a client that
reads ``probe`` as "a Test answered this" would draw a settled result — with a
tool count — for an operation that has not finished. As a pair with a null
``status_observed_at`` it is also the only honest reading of "nothing has
observed this yet": an in-flight operation is knowledge that something is
happening, not a measurement.

The row also carries the ACTIONS it supports, computed here, so no client
draws a button that dead-ends (a "Sign in" on a local command with no
credentials was exactly that).

Pure apart from reads: the only write is :class:`McpToolCache`'s documented
drop of a row whose digest no longer matches the config.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.parse import urlsplit

from local_operator.mcp.config import (
    MCPServerConfig,
    _scope_path,
    load_all_mcp_configs,
    owned_scope_for_source,
    project_scope_available,
    tool_enabled_by_config,
    validate_server_config,
)
from local_operator.paths import config_dir

#: How long a Test / sign-in outcome stays the row's answer. Long enough that
#: the list a user reads right after pressing Test shows that answer; short
#: enough that a server which died since is not reported healthy for long.
PROBE_TTL_S = 300.0

#: The bound on a public reason line. A reason is one sentence for a row, not a
#: log; the full text stays in the server log.
REASON_LIMIT = 300

CatalogStatus = Literal["connected", "needs_sign_in", "not_started", "connecting", "error"]

#: Which TOOL owns each foreign MCP config file, keyed by the trailing path
#: fragment ``load_all_mcp_configs`` reads it from. A refusal that names only
#: the file leaves the user hunting for what writes it; naming the tool makes
#: "remove it there" an instruction rather than a dead end. The Codex entry is
#: read-only for a reason that will not change soon: ``tomllib`` parses TOML
#: (``load``/``loads``) and cannot emit it, and ``tomli_w`` is not a
#: dependency, so refusing is the only correct answer for a Codex-imported
#: server rather than a policy we could relax (issue #367).
#:
#: ONE authority:
#: the ``/mcp remove`` refusal (``verbs._foreign_config_origin``) and the
#: catalog's ``source.kind`` both read this table, so "imported from Cursor"
#: and ``kind: "cursor"`` cannot drift apart. Order matters: the two-part
#: ``.claude/.mcp.json`` must match before the bare ``.mcp.json``.
SOURCE_KINDS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    ((".claude.json",), "claude-code", "imported from Claude Code"),
    ((".claude", ".mcp.json"), "claude-code", "imported from Claude Code"),
    ((".cursor", "mcp.json"), "cursor", "imported from Cursor"),
    ((".vscode", "mcp.json"), "vscode", "imported from VS Code"),
    ((".codex", "config.toml"), "codex", "imported from Codex CLI"),
    # Read by the loader, never written by ``_scope_path`` — foreign to the
    # writer despite living in the project the user is sitting in.
    ((".mcp.json",), "project-mcp-json", "a project .mcp.json local-operator does not write"),
)


def source_kind(source: str | os.PathLike[str] | None) -> tuple[str, str] | None:
    """``(kind, origin phrase)`` for a FOREIGN source file, or ``None``."""
    if not source:
        return None
    parts = Path(source).parts
    for fragment, kind, origin in SOURCE_KINDS:
        if len(parts) >= len(fragment) and tuple(parts[-len(fragment) :]) == fragment:
            return kind, origin
    return None


@dataclass(frozen=True)
class ProbeResult:
    """The outcome of one explicit Test / sign-in connect.

    ``digest`` pins the result to the config it was measured against: an edit
    to the server's command or URL makes the old answer about a different
    server, so a mismatched digest is ignored rather than trusted.
    """

    status: Literal["connected", "needs_sign_in", "error"]
    reason: str | None
    tool_count: int | None
    observed_at: float
    digest: str


@dataclass(frozen=True)
class LiveFacts:
    """What one warm runtime reports about one server it loaded.

    ``status`` is the manager's own wire word (``connected | connecting |
    auth-required | disconnected``) and ``failure`` its sanitized startup
    failure; the translation into the catalog vocabulary happens here, once.
    """

    status: str
    tool_count: int
    failure: str | None = None


_URL_USERINFO = re.compile(r"(?i)\b(https?://)[^/\s@]+@")
_URL_TAIL = re.compile(r"(?i)\b(https?://[^\s?#]+)[?#][^\s]*")
_WHITESPACE = re.compile(r"\s+")


def public_reason(text: object) -> str | None:
    """One sanitized, bounded line fit to cross HTTP, or ``None``.

    Failure text is composed from exceptions, and an MCP server can echo a
    rejected header back in its error body; a config error can quote a URL
    carrying a token in its query. So: registered credentials are scrubbed
    (the MCP redaction registry, the same one every MCP log sink reads), URL
    user-info and query/fragment are dropped, terminal control sequences are
    stripped, and the line is collapsed and bounded.
    """
    if text is None:
        return None
    from local_operator.ansi import strip_control_sequences
    from local_operator.mcp.redaction import scrub

    line = scrub(strip_control_sequences(str(text)))
    line = _URL_USERINFO.sub(r"\1[redacted]@", line)
    line = _URL_TAIL.sub(r"\1", line)
    line = _WHITESPACE.sub(" ", line).strip()
    if not line:
        return None
    if len(line) > REASON_LIMIT:
        line = line[: REASON_LIMIT - 1].rstrip() + "…"
    return line


def public_server_config(cfg: Any) -> dict[str, Any]:
    """Expose destinations, never legacy inline headers/env/argument secrets.

    The legacy session-route row shape (``stdio``/``http`` transport words),
    kept byte-compatible for desktop builds that predate the catalog.

    ``url`` is the RAW configured value unless redacted — NOT
    :func:`_public_url`'s answer, which maps an empty URL to ``None``. The
    catalog row wants that (``null`` = "no endpoint"), but this shape answered
    ``""`` for an http config with no URL on every build before the catalog,
    and an older renderer is exactly what this shape exists for (QA round 3,
    Q-1). Only the redaction FACT is shared.
    """
    from local_operator.mcp.auth import server_rejects_oauth
    from local_operator.mcp.secret_refs import public_secret_refs

    _, redacted = _public_url(cfg)
    command = getattr(cfg, "command", None)
    return {
        "transport": "stdio" if command else "http",
        "command": command,
        "argument_count": len(getattr(cfg, "args", [])),
        "url": None if redacted else getattr(cfg, "url", None),
        "endpoint_redacted": redacted,
        "environment_keys": sorted(getattr(cfg, "env", {})),
        "header_keys": sorted(getattr(cfg, "headers", {})),
        "secret_refs": public_secret_refs(cfg),
        "transport_oauth_supported": False if server_rejects_oauth(cfg) else None,
        "downstream_authorization": "unknown",
    }


def _public_url(cfg: Any) -> tuple[str | None, bool]:
    """The configured URL when it carries nothing secret-shaped, else redacted."""
    url = getattr(cfg, "url", None)
    if not url:
        return None, False
    try:
        parsed = urlsplit(url)
        redacted = bool(parsed.username or parsed.password or parsed.query or parsed.fragment)
    except ValueError:
        redacted = True
    return (None if redacted else url), redacted


def _scope_of(source: str | None, cwd: str, scope_available: bool) -> str:
    """Where a server APPLIES: a file under ``cwd`` is project, else global.

    Distinct from ``owned_scope`` (which file we may WRITE): ``<cwd>/.mcp.json``
    applies to this project and is not ours. With no separate project scope
    (cwd is home) everything is global — "This project" would be the global
    file under another name.
    """
    if not source or not scope_available:
        return "global"
    try:
        parent = Path(source).expanduser().resolve()
        root = Path(cwd).expanduser().resolve()
    except OSError:
        return "global"
    project_files = (
        root / ".local-operator" / "mcp.json",
        root / ".mcp.json",
        root / ".claude" / ".mcp.json",
        root / ".vscode" / "mcp.json",
    )
    return "project" if parent in project_files else "global"


def _secret_ref_states(cfg: Any, base: Path) -> list[dict[str, str]]:
    from local_operator.mcp.credentials import credential_source
    from local_operator.mcp.secret_refs import public_secret_refs

    return [
        {"id": ref["id"], "state": credential_source(ref["id"], base)}
        for ref in public_secret_refs(cfg)
    ]


#: Name fragments that mark a header as ONE a credential could travel in. The
#: value is a literal the loader hands to the transport untouched, and nothing on
#: disk records which header a hand-written or foreign-imported config MEANT as
#: its key, so the name is the only evidence there is. Erring wide here keeps a
#: real key header out of ``add_key`` (review round 3, R3-M1); erring narrow is
#: what offers a second key header, so a name that merely mentions a key is
#: counted as one.
#:
#: ``cookie``, ``bearer``, ``session`` and ``signature`` name a credential
#: carrier without containing any of the first six: a literal ``Cookie`` session
#: credential read as "sends no key", so the row offered ``add_key`` and said
#: its headers could not carry one (review round 5, m2). ``session`` also
#: matches the transport's own ``Mcp-Session-Id``, which is why
#: :func:`_headers_carry_credential` excludes transport-owned names.
#:
#: ``auth`` is NOT in this tuple: it is matched by :data:`_AUTH_WORD` instead.
_CREDENTIAL_HEADER_TOKENS = (
    "key",
    "token",
    "secret",
    "credential",
    "password",
    "cookie",
    "bearer",
    "session",
    "signature",
)

#: ``auth`` as a credential word, not as a substring (review round 5, m3). A
#: plain substring caught ``X-Author``/``X-Authority``, which name a person or an
#: issuer, and a server with ``auth.type: apikey`` sending one reached the dead
#: end R4-M1 closed: ``needs_sign_in`` after a 401 beside ``signed_in: true`` and
#: no key action. A word boundary cannot express the fix — ``\bauth`` still
#: matches ``Author`` and ``auth\b`` loses ``Authorization`` — so the rule is
#: "``auth`` not followed by ``or``, unless it continues as ``oriz``/``oris``":
#: ``Authorization``/``Proxy-Authorization``/``X-Authorisation``, ``X-Auth-Token``,
#: ``X-OAuth``, ``Authentication`` and camel-case ``XAuthToken`` all still count.
_AUTH_WORD = re.compile(r"auth(?!or(?!i[sz]))")


def _headers_carry_credential(headers: Any) -> bool:
    """Whether any header the config SENDS could be the server's key (R4-M1).

    The header arm of :func:`_needs_unbound_key` used to be "has a header",
    which is not the same question. A server that declares ``auth.type: apikey``
    and sends ``X-Tenant-Id`` — or the streamable-HTTP spec's ``Accept``, which
    every client sends — has NOT got somewhere its key travels, but the wide arm
    read it as one: the row claimed ``signed_in: true`` with no reference, no
    grant and nothing that could authenticate it, and after a Test's 401 it read
    ``needs_sign_in`` while ``offers_add_key`` refused the write — a dead end
    with no action that could produce a credential.

    Two things are not a credential carrier: the TRANSPORT's own headers
    (``config.TRANSPORT_OWNED_HEADERS`` — invented by the protocol, and the same
    list the header write refuses to bind), and a name that says nothing about a
    credential (``X-Tenant-Id``, ``X-Request-Id``). Nothing else is narrowed:
    the URL arm stays as wide as :func:`_url_carries_credential`.

    The transport exclusion is load-bearing for exactly one name today:
    ``Mcp-Session-Id`` contains the ``session`` token, but the protocol assigns
    it per connection and no key travels in it. Without the exclusion a keyless
    ``apikey`` server that echoes it would read as already sending a key and
    lose ``add_key`` — the R4-M1 dead end again (review round 5, n1: before
    ``session`` was a token, no transport-owned name matched and the exclusion
    changed no result).

    The name is the only evidence, so two residuals remain, both deliberate.
    A name that mentions a key without being one (``X-Idempotency-Key``) counts
    as a carrier; a key sent under a name with no credential word
    (``X-Custom``) does not, and that row is offered ``add_key``.
    """
    from local_operator.mcp.config import TRANSPORT_OWNED_HEADERS

    if not isinstance(headers, dict):
        return False
    return any(
        lowered not in TRANSPORT_OWNED_HEADERS
        and (
            _AUTH_WORD.search(lowered) is not None
            or any(token in lowered for token in _CREDENTIAL_HEADER_TOKENS)
        )
        for lowered in (str(name).lower() for name in headers)
    )


def _needs_unbound_key(cfg: Any, refs: list[dict[str, str]], granted: bool) -> bool:
    """A remote server that needs a key and declares no ``${ID}`` to hold one.

    Two ways to know, both without guessing: the config says ``auth.type:
    apikey``, or this process watched the server answer a 401/403 challenge and
    discovery found NO authorization server (``OAUTH_CHALLENGES[url] is
    False``, recorded by a Test's own connect). Before this such a row offered
    only Sign in, which fails with "No OAuth authorization server was
    discovered" — a user with a key had no way to enter it (UX round 1 of the
    Integrations redesign, U3).

    The ledger is per-process, so after a daemon restart the answer falls back
    to ``unknown`` + Sign in until the next Test re-observes the challenge.
    That is the ledger's documented scope, and it errs toward the old answer
    rather than toward claiming a fact nobody measured.

    A server that already SENDS something that can be its key is never
    unbound, whatever the ledger says: a header whose NAME can carry one (a
    literal one included — the loader passes it to the transport untouched), or
    credentials or a query in the URL. "Needs a key it has nowhere to put" is
    false for it, and calling it unbound flipped a working server to
    ``signed_in: false`` + ``needs_sign_in`` and offered ``add_key`` — which can
    only bind a SECOND header, since the write refuses the one it already sets.
    A 401 with such a config means the key it sends is wrong, and the fix is in
    that config (review round 3, R3-M1).

    A header that can carry NO credential is not that: ``X-Tenant-Id`` is a
    header, and no key could ever be in it, so the row stays unbound and keeps
    ``add_key`` (review round 4, R4-M1).
    """
    from local_operator.mcp.auth import OAUTH_CHALLENGES

    url = getattr(cfg, "url", None)
    if not url or refs or granted:
        return False
    if _headers_carry_credential(getattr(cfg, "headers", None)) or _url_carries_credential(url):
        return False
    auth_type = getattr(getattr(cfg, "auth", None), "type", None)
    if auth_type == "apikey":
        return True
    return auth_type is None and OAUTH_CHALLENGES.get(url) is False


def _url_carries_credential(url: str) -> bool:
    """User-info or a query: the two places a URL can hold a key.

    The fragment is not one — it never leaves the client — which is why this is
    not simply :func:`_public_url`'s redaction fact.

    A URL ``urlsplit`` refuses to parse answers ``True``: the answer decides
    whether a server is offered a header bind, and a URL nobody can read might
    be holding a key in a shape this parser does not know. That is the direction
    the sibling :func:`_public_url` already takes — it redacts the same input —
    and for a write gate the conservative direction is the one that does not
    attract a second credential (review round 4, R4-n2).
    """
    try:
        parsed = urlsplit(url)
    except ValueError:
        return True
    return bool(parsed.username or parsed.password or parsed.query)


def offers_add_key(cfg: Any) -> bool:
    """Whether ``cfg`` is a row the ``add_key`` action applies to (config half).

    The WRITE path's gate, so a header write lands only where the catalog
    offers ``add_key``: without it a client could post a header for an OAuth
    server and get a second ``Authorization`` bound beside the provider's own
    (review round 3, R3-m1). Ownership is the other half of the offer, and
    :func:`~local_operator.mcp.config.bind_header_secret` enforces it. Blocking:
    it reads the grant store.
    """
    from local_operator.mcp.auth import server_has_stored_grant
    from local_operator.mcp.secret_refs import public_secret_refs

    url = getattr(cfg, "url", None)
    granted = bool(url) and server_has_stored_grant(url)
    refs = [{"id": ref["id"]} for ref in public_secret_refs(cfg)]
    return _needs_unbound_key(cfg, refs, granted)


def _auth_facts(cfg: Any, refs: list[dict[str, str]]) -> dict[str, Any]:
    """``auth`` for one row: kind, signed-in state and reference states.

    ``unknown`` is the honest answer for a bare URL (typically a Codex or
    Claude import): whether it takes OAuth is only knowable from the network,
    so it is offered a sign-in and never claimed as "no auth needed".

    A server that needs a key it has nowhere to put (:func:`_needs_unbound_key`)
    is ``api_key`` with ``signed_in: false`` — ``all()`` over no references is
    ``True``, which would call a server we just watched refuse us signed in.
    """
    from local_operator.mcp.auth import server_has_stored_grant

    url = getattr(cfg, "url", None)
    auth_type = getattr(getattr(cfg, "auth", None), "type", None)
    granted = bool(url) and server_has_stored_grant(url)
    unbound = _needs_unbound_key(cfg, refs, granted)
    if url and (auth_type == "oauth" or granted):
        kind = "oauth"
    elif auth_type == "apikey" or refs or unbound:
        kind = "api_key"
    elif url:
        kind = "unknown"
    else:
        kind = "none"
    signed_in: bool | None
    if kind == "oauth":
        signed_in = granted
    elif unbound:
        signed_in = False
    elif kind == "api_key":
        signed_in = all(ref["state"] == "encrypted" for ref in refs)
    else:
        signed_in = None
    return {"kind": kind, "signed_in": signed_in, "secret_refs": refs, "_unbound": unbound}


def _last_seen_tools(name: str, cfg: Any, digest: str, tool_cache: Any) -> tuple[int, float] | None:
    """The cached tool count for this exact config, and WHEN it was listed.

    The time is the cache row's ``saved_at`` (epoch seconds), written only after
    a successful ``tools/list``. It travels with the count because a count with
    no age is what a client cannot render honestly after a reload: the row's
    ``status_observed_at`` is null on the ``stored`` basis by design, so without
    this the desktop had a number and nothing to say how old it was.
    """
    if tool_cache is None:
        return None
    entry = tool_cache.get_entry(name, digest)
    if entry is None or not entry[0]:
        return None
    cached, saved_at = entry
    count = sum(
        1
        for item in cached
        if isinstance(item, dict) and tool_enabled_by_config(cfg, str(item.get("name", "")))
    )
    return count, saved_at


def _row(
    name: str,
    cfg: MCPServerConfig,
    source: str | None,
    *,
    cwd: str,
    scope_available: bool,
    base: Path,
    tool_cache: Any,
    live: Mapping[str, LiveFacts] | None,
    probe: ProbeResult | None,
    running: bool,
    now: float,
) -> dict[str, Any]:
    from local_operator.mcp.tool_cache import config_digest

    digest = config_digest(cfg)
    owned = owned_scope_for_source(source, cwd)
    scope = _scope_of(source, cwd, scope_available)
    foreign = source_kind(source) if owned is None else None
    refs = _secret_ref_states(cfg, base)
    auth = _auth_facts(cfg, refs)
    # Internal only: it decides a status and an action below, and never crosses
    # HTTP (the published ``auth`` keys are pinned by the fixture drift test).
    unbound_key = bool(auth.pop("_unbound"))
    url, redacted = _public_url(cfg)
    remote = bool(getattr(cfg, "url", None))

    errors = validate_server_config(name, cfg)
    facts = live.get(name) if live is not None else None
    status: CatalogStatus
    reason: str | None = None
    observed: float | None = None
    basis: str
    tool_count: int | None = None
    tool_basis: str | None = None

    if errors:
        status, reason, basis = "error", public_reason("; ".join(errors)), "stored"
    elif running:
        # BEFORE the live overlay, because the two can disagree and the
        # operation is the fresher fact: a Test the user pressed a moment ago
        # runs on its OWN short-lived manager, while a live overlay is another
        # process's view of a different connection. The published contract says
        # a running test reads ``connecting``, and it must say so whether or
        # not the caller also passed a ``session_id`` — otherwise the same
        # press paints "Connecting" on a sessionless page and the runtime's
        # older status on one with an active conversation.
        #
        # ``operation``, not ``probe``: no probe result exists yet (the
        # operation is what will produce one), and reusing ``probe`` here made
        # one word mean both "a Test answered this" and "a Test is running",
        # with nothing in the payload to tell them apart.
        status, basis = "connecting", "operation"
    elif facts is not None:
        basis, observed = "live", now
        if facts.status == "connected":
            status, tool_count, tool_basis = "connected", facts.tool_count, "live"
        elif facts.status == "connecting":
            status = "connecting"
        elif facts.status == "auth-required":
            status = "needs_sign_in"
            reason = public_reason(facts.failure)
        elif facts.failure:
            status, reason = "error", public_reason(facts.failure)
        else:
            status = "not_started"
    elif probe is not None and probe.digest == digest and now - probe.observed_at <= PROBE_TTL_S:
        status, reason, basis, observed = probe.status, probe.reason, "probe", probe.observed_at
        if probe.tool_count is not None and probe.status == "connected":
            tool_count, tool_basis = probe.tool_count, "probe"
    else:
        basis = "stored"
        if (auth["kind"] == "oauth" and not auth["signed_in"]) or unbound_key:
            status = "needs_sign_in"
            headers = getattr(cfg, "headers", None)
            if unbound_key and isinstance(headers, dict) and headers:
                # A header the server sends, none of which can be its key: the
                # row has to SAY that, or the user reads "sends headers" beside
                # "needs a sign-in" and works out the rest themselves (review
                # round 4, R4-M1).
                reason = "The headers this server sends cannot carry a key."
        elif any(ref["state"] == "unavailable" for ref in refs):
            # The encrypted store could not be read, so the connect would fail
            # to resolve its references: "Ready" would be a promise we know is
            # false, and "add its key" would send the user to re-enter a key
            # that may well be stored.
            status, reason = "error", "The encrypted secret store could not be read."
        elif any(ref["state"] == "missing" for ref in refs):
            status = "needs_sign_in"
        else:
            status = "not_started"

    last_seen_at: float | None = None
    if tool_count is None:
        seen = _last_seen_tools(name, cfg, digest, tool_cache)
        if seen is not None:
            tool_count, last_seen_at = seen
            tool_basis = "last_seen"

    actions: list[str] = []
    if not errors:
        actions.append("test")
    if remote and auth["kind"] in ("oauth", "unknown") and auth["signed_in"] is not True:
        actions.append("sign_in")
    if refs:
        actions.append("set_key")
    elif unbound_key and owned is not None:
        # ``add_key``, not ``set_key``: there is no ``${ID}`` to fill, so the
        # client must also NAME the header the key travels in, and the write
        # binds it into this server's config — which is why it needs the row to
        # be ours (a foreign import is fixed in the tool that owns it). Never
        # beside ``sign_in``: ``auth.kind`` is ``api_key`` here, which the
        # sign-in rule above already excludes.
        actions.append("add_key")
    if auth["kind"] == "oauth" and auth["signed_in"]:
        actions.extend(["reauth", "sign_out"])
    if owned is not None:
        actions.append("remove")
    if facts is not None:
        actions.append("disconnect" if facts.status in ("connected", "connecting") else "connect")

    return {
        "id": name,
        "name": name,
        "scope": scope,
        # The row's field is the DIRECTORY the row's project scope applies to —
        # NOT the catalog-level ``project_path``, which is the mcp.json FILE a
        # user can open. Two kinds of value behind one name is what a client
        # gets wrong, so they carry two names.
        "project_cwd": cwd if scope == "project" else None,
        "source": {
            "kind": foreign[0] if foreign is not None else "local-operator",
            "path": str(source) if source else "",
            "editable": owned is not None,
            "owned_scope": owned,
        },
        "transport": "remote_url" if remote else "local_command",
        "endpoint": {
            "command": getattr(cfg, "command", None),
            "url": url,
            "endpoint_redacted": redacted,
        },
        "status": status,
        "status_reason": reason,
        "status_observed_at": observed,
        "status_basis": basis,
        "auth": auth,
        "tool_count": tool_count,
        "tool_count_basis": tool_basis,
        # Epoch SECONDS, set iff ``tool_count_basis == "last_seen"``: when that
        # count was listed. Deliberately NOT ``status_observed_at`` — a stored
        # status is not an observation, and that field keeps its live/probe
        # meaning. A client that reads it as milliseconds renders a date in 1970.
        "last_seen_at": last_seen_at,
        "actions": actions,
    }


def describe_servers(
    cwd: str,
    *,
    live: Mapping[str, LiveFacts] | None = None,
    session_id: str | None = None,
    probes: Mapping[str, ProbeResult] | None = None,
    running: frozenset[str] = frozenset(),
    operations: list[dict[str, Any]] | None = None,
    tool_cache: Any = None,
    secret_base: Path | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """The whole catalog document for ``cwd`` (see the module docstring).

    Blocking: it reads config files, ``auth.db``, the secret store and the tool
    cache, so async callers run it in a thread.

    ``live`` maps server name to what one warm runtime (``session_id``) reports;
    only servers that runtime actually loaded appear in it, so a server added
    after that conversation started honestly reads from config instead of as
    "not started in this conversation". ``running`` names servers with an
    operation in flight.

    ``status_source`` and ``session_id`` report the overlay ONLY when it
    applied to at least one row. An overlay that was consulted and used by
    nothing is not a different source of truth: an empty one (a warm runtime
    that loaded none of these servers) or one keyed entirely by servers this
    folder does not configure would otherwise stamp ``live`` on a document in
    which every status came from config, and echo an id that explains no field
    in it. A client reads ``session_id`` as "the session whose live facts were
    applied", so ``null`` means exactly "none were" — never "the backend lost
    the id you sent".
    """
    if tool_cache is None:
        from local_operator.mcp.tool_cache import McpToolCache

        tool_cache = McpToolCache()
    base = secret_base if secret_base is not None else config_dir()
    moment = time.time() if now is None else now
    available = project_scope_available(cwd)
    configs, sources = load_all_mcp_configs(cwd)
    servers = [
        _row(
            name,
            cfg,
            sources.get(name),
            cwd=cwd,
            scope_available=available,
            base=base,
            tool_cache=tool_cache,
            live=live,
            probe=(probes or {}).get(name),
            running=name in running,
            now=moment,
        )
        for name, cfg in configs.items()
    ]
    # Whether the overlay reached a row — read from the ROWS, not from the
    # overlay's keys: a running operation and a config error both take
    # precedence over the overlay, so an overlay keyed by a server being tested
    # right now names a configured server and still reaches no row (review
    # round 2, R2-m2). An empty overlay, or one keyed by names this folder does
    # not configure, is the same case.
    applied = any(row["status_basis"] == "live" for row in servers)
    return {
        "cwd": cwd,
        "project_scope_available": available,
        "global_path": str(_scope_path(cwd, "global")),
        "project_path": str(_scope_path(cwd, "project")) if available else None,
        "status_source": "live" if applied else "config",
        "session_id": session_id if applied else None,
        "servers": servers,
        "operations": list(operations or []),
    }


def live_facts_from_snapshot(rows: object) -> dict[str, LiveFacts]:
    """Rebuild :class:`LiveFacts` from a session route's legacy snapshot rows.

    The runtime's manager lives in ANOTHER process, so the overlay arrives as
    the owner's ``desktop_mcp`` list answer. Rows that are not well-formed are
    skipped: a malformed overlay must degrade to the config answer, never
    fail the read.
    """
    facts: dict[str, LiveFacts] = {}
    if not isinstance(rows, list):
        return facts
    for row in rows:
        if not isinstance(row, dict):
            continue
        name, status = row.get("name"), row.get("status")
        if not isinstance(name, str) or not isinstance(status, str) or status == "cold":
            continue
        # A server the runtime never loaded (added after the conversation
        # started) is not "not started in this conversation" — it is a config
        # fact, and the row must say what the config says.
        if row.get("loaded") is False:
            continue
        count = row.get("tool_count")
        failure = row.get("startup_failure")
        facts[name] = LiveFacts(
            status=status,
            tool_count=count if isinstance(count, int) else 0,
            failure=failure if isinstance(failure, str) else None,
        )
    return facts


__all__ = [
    "PROBE_TTL_S",
    "SOURCE_KINDS",
    "LiveFacts",
    "ProbeResult",
    "describe_servers",
    "live_facts_from_snapshot",
    "offers_add_key",
    "public_reason",
    "public_server_config",
    "source_kind",
]
