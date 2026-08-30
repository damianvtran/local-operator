"""MCP server configuration discovery and validation.

Ports the canonical MCP client config semantics
onto local-operator paths:

- Project config: ``<cwd>/.local-operator/mcp.json`` and ``<cwd>/.mcp.json``.
- User config: ``~/.local-operator/mcp.json``.
- Best-effort imports of foreign tool configs: ``~/.claude.json``
  (``mcpServers`` key), ``<cwd>/.claude/.mcp.json``, ``~/.cursor/mcp.json``,
  ``<cwd>/.vscode/mcp.json`` (``mcp.servers`` key), ``~/.codex/config.toml``
  (TOML, ``[mcp_servers.<name>]`` tables; issue #367).

Priority: project > user > imports; later sources NEVER override an earlier
one (first-seen wins at the name level). ``disabledServers`` lists collected
from the local-operator files win over ``enabledServers`` lists, which win
over a per-server ``enabled: false``.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

# Server names allow letters, digits, underscore, dash, dot, colon; max 100
# chars (config-writer rule; the colon covers namespaced plugin entries).
SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,100}$")


class MCPAuthConfig(BaseModel):
    """Auth block mirroring the canonical MCP auth config.

    ``apikey`` is accepted for shape parity but not implemented — put API keys
    in stdio ``env`` or remote ``headers`` instead.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["oauth", "apikey"] = "oauth"
    credential_id: str | None = None
    token_url: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    resource: str | None = None


class MCPOAuthConfig(BaseModel):
    """OAuth client knobs mirroring the canonical per-server ``oauth`` block."""

    model_config = ConfigDict(extra="allow")

    client_id: str | None = None
    client_secret: str | None = None
    redirect_uri: str | None = None
    callback_port: int | None = None
    callback_path: str | None = None
    prompt: str | None = None


class MCPStdioServerConfig(BaseModel):
    """A stdio MCP server: spawn ``command`` with ``args``."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: Literal["stdio"] = "stdio"
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    enabled: bool | None = None
    enabled_tools: list[str] = Field(
        default_factory=list,
        alias="enabledTools",
        description=(
            "Optional allowlist of MCP tool names (exact or glob patterns); "
            "empty means all tools."
        ),
    )
    disabled_tools: list[str] = Field(
        default_factory=list,
        alias="disabledTools",
        description=("MCP tool names/patterns to deny; wins over enabledTools."),
    )
    timeout: float | None = None  # milliseconds; 0 disables client-side timeout
    auth: MCPAuthConfig | None = None
    oauth: MCPOAuthConfig | None = None


class MCPHttpServerConfig(BaseModel):
    """A Streamable HTTP MCP server."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: Literal["http"] = "http"
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    enabled: bool | None = None
    enabled_tools: list[str] = Field(
        default_factory=list,
        alias="enabledTools",
        description=(
            "Optional allowlist of MCP tool names (exact or glob patterns); "
            "empty means all tools."
        ),
    )
    disabled_tools: list[str] = Field(
        default_factory=list,
        alias="disabledTools",
        description=("MCP tool names/patterns to deny; wins over enabledTools."),
    )
    timeout: float | None = None
    auth: MCPAuthConfig | None = None
    oauth: MCPOAuthConfig | None = None


class MCPSseServerConfig(BaseModel):
    """A legacy dual-endpoint SSE MCP server (deprecated by the spec, kept for
    compatibility)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: Literal["sse"] = "sse"
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    enabled: bool | None = None
    enabled_tools: list[str] = Field(
        default_factory=list,
        alias="enabledTools",
        description=(
            "Optional allowlist of MCP tool names (exact or glob patterns); "
            "empty means all tools."
        ),
    )
    disabled_tools: list[str] = Field(
        default_factory=list,
        alias="disabledTools",
        description=("MCP tool names/patterns to deny; wins over enabledTools."),
    )
    timeout: float | None = None
    auth: MCPAuthConfig | None = None
    oauth: MCPOAuthConfig | None = None


MCPServerConfig = MCPStdioServerConfig | MCPHttpServerConfig | MCPSseServerConfig

# Transport type keyword -> model, for parsing raw dicts.
_TYPE_MODELS: dict[str, type[MCPServerConfig]] = {
    "stdio": MCPStdioServerConfig,
    "http": MCPHttpServerConfig,
    "sse": MCPSseServerConfig,
}


def _coerce_server_config(raw: JsonValue) -> MCPServerConfig | None:
    """Parse one raw server entry into a typed config.

    Transport inference: explicit ``type`` wins; otherwise
    ``command`` implies stdio, ``url`` implies http. Malformed entries return
    ``None`` (validation reports them separately via
    :func:`validate_server_config`).
    """
    if not isinstance(raw, dict):
        return None
    data = dict(raw)
    transport = data.get("type")
    if transport not in ("stdio", "http", "sse"):
        if data.get("command"):
            transport = "stdio"
        elif data.get("url"):
            transport = "http"
        else:
            transport = "stdio"
    model = _TYPE_MODELS[transport]
    try:
        return model.model_validate(data)
    except Exception:
        # extra="allow" + defaults make this nearly total; guard anyway.
        return None


#: A candidate-file reader: path -> parsed document, or ``None`` when the file
#: is missing or unparsable. Every reader is best-effort by contract, so a
#: foreign config we do not own can never break discovery of the ones we do.
_ConfigReader = Callable[[Path], "dict[str, Any] | None"]


def _read_json(path: Path) -> dict[str, Any] | None:
    """Best-effort JSON read: missing file or bad JSON yields ``None``."""
    try:
        if not path.is_file():
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _read_toml(path: Path) -> dict[str, Any] | None:
    """Best-effort TOML read: missing file or bad TOML yields ``None``.

    Codex CLI (``~/.codex/config.toml``) is the one import source that is not
    JSON, so it needs its own reader rather than another ``_read_json`` entry
    (issue #367). The contract is deliberately identical to ``_read_json``'s:
    a config we do not own must never be able to break discovery of the ones
    we do, so every failure mode degrades to "this file contributes nothing".

    This import is READ-ONLY, permanently and by construction: ``tomllib`` is
    stdlib on the 3.12 floor but exposes ``load``/``loads`` only — it cannot
    emit TOML — and ``tomli_w`` is not a dependency. Writing the file back
    would also lose the comments and formatting of a file another tool owns.
    ``/mcp remove`` therefore refuses a Codex-imported server by name instead
    of pretending it can delete it (see ``_mcp_remove_result`` in the TUI).
    """
    try:
        if not path.is_file():
            return None
        with path.open("rb") as handle:  # tomllib requires binary + UTF-8
            loaded = tomllib.load(handle)
        return loaded if isinstance(loaded, dict) else None
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None


def _string_list(value: JsonValue) -> list[str]:
    """Coerce a JSON value to a list of strings, dropping non-strings."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _local_operator_servers(doc: dict[str, Any]) -> dict[str, Any]:
    """Server entries from a local-operator-shaped file (``mcpServers`` key)."""
    servers = doc.get("mcpServers")
    return servers if isinstance(servers, dict) else {}


def _claude_json_servers(doc: dict[str, Any], root: Path) -> dict[str, Any]:
    """Server entries from ``~/.claude.json`` (global + project scope).

    Claude Code keeps global servers under the top-level ``mcpServers`` key
    and project-scoped servers under ``projects.<absolute-path>.mcpServers``;
    project scope wins within the file. Best-effort: anything not shaped like
    that degrades to just the global key (MCP-18).
    """
    servers = _local_operator_servers(doc)
    projects = doc.get("projects")
    if isinstance(projects, dict):
        node = projects.get(str(root))
        if node is None:
            node = projects.get(str(root.resolve()))
        if isinstance(node, dict):
            scoped = node.get("mcpServers")
            if isinstance(scoped, dict):
                servers = {**servers, **scoped}
    return servers


def _imported_servers(doc: dict[str, Any], key_path: tuple[str, ...] | None) -> dict[str, Any]:
    """Server entries from a foreign tool config.

    ``key_path`` walks nested dicts (``("mcp", "servers")`` for VS Code,
    ``("mcp_servers",)`` for Codex's ``[mcp_servers.<name>]`` tables, which
    ``tomllib`` parses into exactly that nested-dict shape); ``None`` means
    the ``mcpServers`` key at the top level.
    """
    if key_path is None:
        return _local_operator_servers(doc)
    node: JsonValue = doc
    for key in key_path:
        if not isinstance(node, dict):
            return {}
        node = node.get(key)
    return node if isinstance(node, dict) else {}


def validate_server_config(name: str, cfg: MCPServerConfig | dict[str, Any] | None) -> list[str]:
    """Validate one server config, returning human-readable error strings.

    An empty list means the config is usable. Shape problems that prevented
    parsing surface as a single "invalid" error when ``cfg`` is not a model
    instance.
    """
    errors: list[str] = []
    if not SERVER_NAME_RE.match(name or ""):
        errors.append(f"invalid server name {name!r}: must match [A-Za-z0-9_.:-]{{1,100}}")
    if not isinstance(cfg, (MCPStdioServerConfig, MCPHttpServerConfig, MCPSseServerConfig)):
        errors.append(f"server {name!r}: invalid config (expected an object)")
        return errors
    if isinstance(cfg, MCPStdioServerConfig):
        if not cfg.command:
            errors.append(f"server {name!r}: stdio config requires 'command'")
    else:
        if not cfg.url:
            errors.append(f"server {name!r}: {cfg.type} config requires 'url'")
        elif not cfg.url.startswith(("http://", "https://")):
            errors.append(f"server {name!r}: url must be http(s), got {cfg.url!r}")
    if cfg.timeout is not None and cfg.timeout < 0:
        errors.append(f"server {name!r}: timeout must be >= 0")
    if cfg.auth is not None and cfg.auth.type not in ("oauth", "apikey"):
        errors.append(f"server {name!r}: auth.type must be 'oauth' or 'apikey'")
    if cfg.oauth is not None and cfg.oauth.callback_port is not None:
        port = cfg.oauth.callback_port
        if not 1 <= port <= 65535:
            errors.append(f"server {name!r}: oauth.callback_port out of range ({port})")
    return errors


def load_all_mcp_configs(
    cwd: str | os.PathLike[str],
) -> tuple[dict[str, MCPServerConfig], dict[str, str]]:
    """Load every MCP server config visible from ``cwd``.

    Returns ``(configs, sources)`` where ``sources`` maps server name to the
    path of the file that contributed it. Priority order (first source to
    define a name wins; later sources never override):

    1. ``<cwd>/.local-operator/mcp.json``
    2. ``<cwd>/.mcp.json``
    3. ``~/.local-operator/mcp.json``
    4. ``~/.claude.json`` (top-level ``mcpServers`` plus project-scoped
       ``projects.<cwd>.mcpServers``, project scope winning within the file)
    5. ``<cwd>/.claude/.mcp.json``
    6. ``~/.cursor/mcp.json``
    7. ``<cwd>/.vscode/mcp.json`` (``mcp.servers`` key)
    8. ``~/.codex/config.toml`` (TOML, ``[mcp_servers.<name>]`` tables;
       user scope only — Codex project-scoped config is not imported)

    Enable/disable resolution uses lists from the local-operator files
    (project first, then user): ``disabledServers`` wins over
    ``enabledServers``, which wins over a config's own ``enabled: false``.
    Suppressed (disabled) servers are dropped from the result entirely — they
    must not shadow lower-priority sources.
    """
    root = Path(cwd).expanduser()
    home = Path.home()

    project_path = root / ".local-operator" / "mcp.json"
    project_dot_path = root / ".mcp.json"
    user_path = home / ".local-operator" / "mcp.json"

    disabled: set[str] = set()
    enabled: set[str] = set()

    # (path, reader, key_path into doc for the servers dict) in priority
    # order. The reader travels WITH the candidate so a non-JSON source is
    # just another row here: the enable/disable lists, the two-pass
    # suppression resolution and the ``sources`` provenance map below all stay
    # format-agnostic instead of growing a parallel loop per file format.
    candidates: list[tuple[Path, _ConfigReader, tuple[str, ...] | None]] = [
        (project_path, _read_json, None),
        (project_dot_path, _read_json, None),
        (user_path, _read_json, None),
        (home / ".claude.json", _read_json, None),
        (root / ".claude" / ".mcp.json", _read_json, None),
        (home / ".cursor" / "mcp.json", _read_json, None),
        (root / ".vscode" / "mcp.json", _read_json, ("mcp", "servers")),
        # Codex CLI is APPENDED LAST on purpose (issue #367): first-seen-wins
        # makes position observable whenever two tools define the same server
        # name, and last is the only position that guarantees Codex can never
        # override local-operator's own files OR any of the other imports.
        # User scope only; Codex project-scoped config is out of scope here.
        (home / ".codex" / "config.toml", _read_toml, ("mcp_servers",)),
    ]

    claude_path = home / ".claude.json"
    # First pass: collect every candidate per name (priority order) and the
    # enable/disable lists. Suppression is evaluated in a SECOND pass over the
    # full candidate lists: collapsing first-seen-wins before suppression
    # (the old shape) let a project entry with ``enabled: false`` claim the
    # name slot, block the user-level entry of the same name, and then get
    # deleted — leaving the server absent instead of falling back.
    per_name: dict[str, list[tuple[MCPServerConfig, str]]] = {}
    for path, reader, key_path in candidates:
        doc = reader(path)
        if doc is None:
            continue
        # Local-operator files contribute enable/disable lists.
        if key_path is None and (path in (project_path, project_dot_path, user_path)):
            disabled.update(_string_list(doc.get("disabledServers")))
            enabled.update(_string_list(doc.get("enabledServers")))
        if path == claude_path:
            server_entries = _claude_json_servers(doc, root)
        else:
            server_entries = _imported_servers(doc, key_path)
        for name, raw in server_entries.items():
            cfg = _coerce_server_config(raw)
            if cfg is None:
                continue  # unparsable entry; validation flags it if referenced
            per_name.setdefault(name, []).append((cfg, str(path)))

    # Resolution: denylist beats allowlist beats ``enabled: false``; the first
    # NON-suppressed candidate in priority order wins the name.
    configs: dict[str, MCPServerConfig] = {}
    sources: dict[str, str] = {}
    for name, entries in per_name.items():
        for cfg, source in entries:
            suppressed = name in disabled or (cfg.enabled is False and name not in enabled)
            if not suppressed:
                configs[name] = cfg
                sources[name] = source
                break

    return configs, sources


# ---------------------------------------------------------------------------
# CLI helpers (`mcp list|add|remove`)
# ---------------------------------------------------------------------------


def _write_json_atomic(path: Path, doc: dict[str, Any]) -> None:
    """Write JSON with a stable two-space indent, atomically (MCP-15).

    Temp file in the target directory + ``os.replace``: readers never see a
    half-written mcp.json (a crash mid-write used to truncate the config).
    """
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_name, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_name)
        raise


def _local_operator_file_paths(cwd: str | os.PathLike[str]) -> list[Path]:
    """Project then user local-operator mcp.json paths (for list reads)."""
    root = Path(cwd).expanduser()
    return [
        root / ".local-operator" / "mcp.json",
        root / ".mcp.json",
        Path.home() / ".local-operator" / "mcp.json",
    ]


def read_disabled_servers(cwd: str | os.PathLike[str]) -> list[str]:
    """``disabledServers`` lists from project then user local-operator files."""
    names: list[str] = []
    for path in _local_operator_file_paths(cwd):
        doc = _read_json(path)
        if doc is not None:
            names.extend(_string_list(doc.get("disabledServers")))
    return names


def read_enabled_servers(cwd: str | os.PathLike[str]) -> list[str]:
    """``enabledServers`` lists from project then user local-operator files."""
    names: list[str] = []
    for path in _local_operator_file_paths(cwd):
        doc = _read_json(path)
        if doc is not None:
            names.extend(_string_list(doc.get("enabledServers")))
    return names


def list_effective_servers(cwd: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """Merged, enable/disable-resolved configs as plain dicts (for ``mcp list``)."""
    configs, _sources = load_all_mcp_configs(cwd)
    return {name: cfg.model_dump(exclude_none=True) for name, cfg in configs.items()}


def _scope_path(cwd: str | os.PathLike[str] | None, scope: str) -> Path:
    """Resolve the mcp.json path a CLI write targets."""
    if scope == "project":
        root = Path(cwd if cwd is not None else ".").expanduser()
        return root / ".local-operator" / "mcp.json"
    return Path.home() / ".local-operator" / "mcp.json"


class MCPConfigWriteError(Exception):
    """One refused config write, carrying every reason it was refused.

    The writers below have TWO callers with incompatible output channels: the
    CLI prints to stderr and answers in exit codes, while the TUI runs inside a
    Textual screen where any ``print`` lands on the terminal *underneath* the
    frame and corrupts the display. Raising is what lets one implementation
    serve both — each caller phrases the failure in its own vocabulary at its
    own boundary.

    ``errors`` is a LIST because validation reports every problem at once and
    the CLI has always printed one ``error:`` line per problem; collapsing them
    into a single string here would change CLI output that users and scripts
    read. ``str(exc)`` joins them for callers (the TUI) that render one line.
    """

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = list(errors)


def owned_scope_for_source(
    source: str | os.PathLike[str] | None,
    cwd: str | os.PathLike[str] | None = None,
) -> str | None:
    """The write scope whose file is ``source``, or ``None`` when unowned.

    ``load_all_mcp_configs`` merges eight sources but :func:`_scope_path` only
    ever writes two of them, so "where did this server come from" and "can we
    edit it here" are different questions. Callers that MUTATE a server (the
    TUI's ``/mcp remove``) must ask this one: removing an entry defined by
    ``~/.claude.json`` or ``<cwd>/.mcp.json`` would either fail or, worse,
    write a local-operator file that silently shadows a foreign config the user
    still maintains elsewhere.

    Comparison is on RESOLVED paths, never on strings: ``sources`` records the
    path as constructed, so a symlinked home, a ``/private/var`` vs ``/var``
    prefix on macOS, or a relative ``cwd`` would each defeat a string compare
    and turn an owned file into a refusal.
    """
    if source is None:
        return None
    try:
        resolved = Path(source).expanduser().resolve()
        for scope in ("project", "global"):
            if _scope_path(cwd, scope).expanduser().resolve() == resolved:
                return scope
    except OSError:
        # An unresolvable path cannot be proven ours, and "not ours" is the
        # safe answer for a question that gates a delete.
        return None
    return None


def add_server(
    name: str,
    *,
    command: str | None = None,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    url: str | None = None,
    headers: dict[str, str] | None = None,
    oauth: bool = False,
    scope: str = "global",
    cwd: str | os.PathLike[str] | None = None,
) -> Path:
    """Add one server to the scoped mcp.json (``mcp add``).

    ``oauth=True`` writes the canonical HTTP auth block used by the SDK's
    dynamic client-registration flow. It is rejected for stdio because that
    transport owns authentication inside its child process.

    Returns the path written, so a caller can NAME the file in its receipt —
    scope is otherwise an invisible default and "added" gives the user no way
    to tell which of the two config files just changed. Raises
    :class:`MCPConfigWriteError` on any refusal.
    """
    if command and url:
        raise MCPConfigWriteError(["pass either command (stdio) or url (http), not both"])
    if not command and not url:
        raise MCPConfigWriteError(["a server needs a command (stdio) or a url (http)"])
    if oauth and not url:
        raise MCPConfigWriteError(["OAuth is supported only for remote HTTP servers"])

    raw: dict[str, Any]
    if command:
        raw = {"type": "stdio", "command": command}
        if args:
            raw["args"] = args
        if env:
            raw["env"] = env
    else:
        raw = {"type": "http", "url": url}
        if headers:
            raw["headers"] = headers
        if oauth:
            raw["auth"] = {"type": "oauth"}

    cfg = _coerce_server_config(raw)
    errors = validate_server_config(name, cfg)
    if errors:
        raise MCPConfigWriteError(errors)

    path = _scope_path(cwd, scope)
    doc = _read_json(path) or {}
    servers = doc.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        servers = {}
        doc["mcpServers"] = servers
    if name in servers:
        raise MCPConfigWriteError([f"server {name!r} already exists in {path}"])
    servers[name] = raw
    try:
        _write_json_atomic(path, doc)
    except OSError as exc:
        raise MCPConfigWriteError([f"could not write {path}: {exc}"]) from exc
    return path


def set_http_oauth_server(
    name: str,
    url: str,
    *,
    scope: str = "global",
    cwd: str | os.PathLike[str] | None = None,
) -> int:
    """Atomically create or replace one scoped OAuth HTTP server.

    Provider setup is intentionally idempotent: a same-name non-OAuth entry
    must be repaired rather than reported as configured or rejected as a
    duplicate. Imported and higher-priority configs remain outside this
    mutation boundary; callers must first ensure this scope is effective.
    """
    import sys

    raw: dict[str, Any] = {
        "type": "http",
        "url": url,
        "auth": {"type": "oauth"},
    }
    errors = validate_server_config(name, _coerce_server_config(raw))
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    path = _scope_path(cwd, scope)
    doc = _read_json(path) or {}
    servers = doc.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        servers = {}
        doc["mcpServers"] = servers
    servers[name] = raw
    try:
        _write_json_atomic(path, doc)
    except OSError as exc:
        print(f"error: could not write {path}: {exc}", file=sys.stderr)
        return 1
    return 0


def remove_server(
    name: str,
    *,
    scope: str = "global",
    cwd: str | os.PathLike[str] | None = None,
) -> Path:
    """Remove one server from the scoped mcp.json (``mcp remove``).

    Returns the path written, for the same receipt reason as :func:`add_server`
    ("removed from WHICH file" is the fact a user needs when two scopes can
    define the same name). Raises :class:`MCPConfigWriteError` when the name is
    not present in that scope, or when the write fails.
    """
    path = _scope_path(cwd, scope)
    doc = _read_json(path)
    servers = doc.get("mcpServers") if doc is not None else None
    if doc is None or not isinstance(servers, dict) or name not in servers:
        raise MCPConfigWriteError([f"server {name!r} not found in {path}"])
    del servers[name]
    try:
        _write_json_atomic(path, doc)
    except OSError as exc:
        raise MCPConfigWriteError([f"could not write {path}: {exc}"]) from exc
    return path
