"""The sessionless MCP host behind ``/v1/desktop/mcp``.

Settings > Integrations manages MCP CONFIGURATION, and configuration belongs to
no conversation. The page used to reach it through a session: every read and
write went to ``/v1/desktop/sessions/{id}/mcp``, which on a fresh install (no
model configured) could not start a runtime at all, and otherwise started a
whole runtime — spawning every configured MCP server — to edit one JSON file.

This host answers without any runtime:

* **list / add / remove** are file operations plus the durable facts in
  :mod:`local_operator.mcp.catalog`; nothing is spawned.
* **test / login / reauth / logout** need a connection, so each runs as an
  OPERATION on a short-lived :class:`McpManager` that is created, used and torn
  down inside ONE task — anyio refuses to exit a transport's cancel scope from
  any other task, which the architect's probe tripped over. That is the pattern
  ``lop mcp login`` already uses from the CLI. Only one operation runs at a
  time (a second loopback OAuth listener would fight for the callback port).
* A grant lands in the shared ``auth.db``, so a running conversation whose
  server was auth-blocked heals on its own next revalidation tick
  (``AUTH_REVALIDATE_INTERVAL_S``); nothing here reaches into live runtimes.

Lifetime: the app's lifespan calls :meth:`McpHost.close` on shutdown, which
cancels and JOINS any running operation — the cancellation runs that
operation's ``disconnect_all`` in its own task, so no server child outlives the
process that spawned it.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from local_operator.mcp.catalog import (
    LiveFacts,
    ProbeResult,
    describe_servers,
    public_reason,
)
from local_operator.mcp.desktop import (
    MCPControl,
    McpOperations,
    MCPRefusal,
    grant_operation,
)
from local_operator.mcp.tool_cache import config_digest

#: The session id a sessionless write is attributed to in the secret store's
#: audit column. Not a real session: a stable label saying "Settings wrote this".
SETTINGS_SESSION_ID = "desktop-settings"

#: The actions this host runs. connect / disconnect / reload are about ONE
#: runtime's live connections and stay on the session route.
SESSIONLESS_ACTIONS = frozenset(
    {"list", "add", "remove", "test", "login", "reauth", "logout", "status", "cancel"}
)


def resolve_cwd(raw: str | None) -> str:
    """The folder a catalog is computed for; the user's home when omitted.

    Home is the desktop's default conversation folder, so "no folder" and "the
    default conversation" answer identically. A supplied folder must be an
    absolute path to an existing directory: it decides which project file a
    write lands in, and the session route's equivalent came from a session
    record rather than a request, so this check is the replacement guard.
    Raises ``ValueError`` (the route answers 422).
    """
    if raw is None or raw == "":
        return str(Path.home())
    if "\x00" in raw or not os.path.isabs(raw):
        raise ValueError("cwd must be an absolute path")
    path = Path(raw)
    if not path.is_dir():
        raise ValueError("cwd must be an existing directory")
    return str(path)


class _SettingsCredentialsHost:
    """The three facts ``store_credentials`` reads from a session, and no more.

    ``mcp_manager`` only needs ``cwd`` (which config set names the server) and
    the redaction sink, so a real manager — and the servers it would spawn —
    is not built for a key write.
    """

    def __init__(self, cwd: str, config_dir: Path) -> None:
        from local_operator.mcp import redaction

        self.session_id = SETTINGS_SESSION_ID
        self.config_dir = config_dir
        self.mcp_manager = _CwdOnly(cwd, redaction.register)
        self.variables = None


class _CwdOnly:
    def __init__(self, cwd: str, sink: Any) -> None:
        self.cwd = cwd
        self.register_secret_redaction = sink


class McpHost:
    """Process-wide sessionless MCP operations; lives on ``app.state``."""

    def __init__(self, config_dir: Path | None = None) -> None:
        self.config_dir = config_dir
        self.ops = McpOperations()
        #: Last probe outcome per ``(cwd, name)``; the catalog ignores one whose
        #: config digest no longer matches or that is older than its TTL.
        self.probes: dict[tuple[str, str], ProbeResult] = {}
        #: Which cwd each operation ran against, so a list for one folder does
        #: not paint another folder's same-named server as "connecting".
        self._op_cwd: dict[str, str] = {}
        self._closed = False

    # -- reads -------------------------------------------------------------

    def _base(self) -> Path:
        from local_operator.paths import config_dir

        return Path(self.config_dir) if self.config_dir is not None else config_dir()

    def _running_for(self, cwd: str) -> frozenset[str]:
        return frozenset(
            op["name"]
            for key, op in self.ops.operations.items()
            if key in self.ops.running and self._op_cwd.get(key) == cwd
        )

    async def catalog(
        self,
        cwd: str,
        *,
        live: dict[str, LiveFacts] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        probes = {name: result for (where, name), result in self.probes.items() if where == cwd}
        running = self._running_for(cwd)
        operations = [op for op in self.ops.records() if self._op_cwd.get(op["id"], cwd) == cwd]
        base = self._base()
        return await asyncio.to_thread(
            lambda: describe_servers(
                cwd,
                live=live,
                session_id=session_id,
                probes=probes,
                running=running,
                operations=operations,
                secret_base=base,
            )
        )

    # -- controls ----------------------------------------------------------

    async def execute(self, body: MCPControl, cwd: str) -> dict[str, Any] | None:
        """Run one control; return the operation it started or named, if any.

        The caller answers with the catalog document either way, so a client
        always gets the rows it must repaint and never a bare op record.
        Raises :class:`MCPRefusal` (or ``MCPConfigWriteError``) on refusal.
        """
        if self._closed:
            # ``grant_running`` rather than a code of its own: the bounded set is
            # the published contract, and the only thing a client could do about
            # a host that is shutting down is nothing — the daemon is leaving and
            # every reading client is about to re-home. The MESSAGE below is not
            # what crosses HTTP either (``refusal_detail`` owns the copy per
            # code), which is why it can be the honest sentence here.
            raise MCPRefusal("grant_running", "The MCP host is shutting down")
        if body.action not in SESSIONLESS_ACTIONS:
            raise MCPRefusal(
                "operation_unavailable", "connect/disconnect/reload are session controls"
            )
        if body.action == "list":
            return None
        if body.action == "status":
            return dict(self.ops.record(body.operation_id))
        if body.action == "cancel":
            return dict(await self.ops.cancel(body.operation_id))
        async with self.ops.lock:
            self.ops.ensure_idle()
            if body.action in ("add", "remove"):
                await asyncio.to_thread(self._write, body, cwd)
                # A config edit makes the old probe answer about a different
                # server; the digest check would ignore it anyway, and dropping
                # it keeps a re-added name from inheriting a stale result.
                self.probes.pop((cwd, body.name), None)
                return None
            return self._start(body, cwd)

    def _write(self, body: MCPControl, cwd: str) -> None:
        from local_operator.mcp.config import (
            add_server,
            load_all_mcp_configs,
            owned_scope_for_source,
            remove_server,
        )

        configs, sources = load_all_mcp_configs(cwd)
        if body.action == "add":
            if body.name in configs:
                raise MCPRefusal("exists", "That server already exists")
            add_server(
                body.name,
                command=body.command,
                args=body.args,
                env=body.env,
                url=body.url,
                headers=body.headers,
                oauth=body.oauth,
                scope=body.scope,
                cwd=cwd,
            )
            return
        if body.name not in configs:
            raise MCPRefusal("unknown_server", "Unknown MCP server")
        if owned_scope_for_source(sources.get(body.name), cwd) != body.scope:
            raise MCPRefusal("not_owned", "Not owned by the selected scope")
        remove_server(body.name, scope=body.scope, cwd=cwd)

    def _start(self, body: MCPControl, cwd: str) -> dict[str, Any]:
        from local_operator.mcp.auth import server_rejects_oauth
        from local_operator.mcp.config import load_all_mcp_configs

        configs, _ = load_all_mcp_configs(cwd)
        cfg = configs.get(body.name)
        if cfg is None:
            raise MCPRefusal("unknown_server", "Unknown MCP server")
        action = body.action
        if action in ("login", "reauth") and server_rejects_oauth(cfg):
            # Static and free: a local command or an explicit non-OAuth auth
            # type can never take a grant. The network capability probe for a
            # bare URL runs INSIDE the operation (it can take 30 s).
            raise MCPRefusal("oauth_unsupported", "This server does not use OAuth")
        name = body.name
        digest = config_digest(cfg)
        host = self

        async def work(op: dict[str, Any]) -> None:
            from local_operator.mcp.manager import McpManager
            from local_operator.mcp.redaction import register
            from local_operator.mcp.tool_cache import McpToolCache

            # Built HERE, inside the operation's task, and torn down in the
            # ``finally`` below in this same task (see the module docstring).
            manager = McpManager(
                cwd,
                McpToolCache(),
                secret_base=host._base(),
                register_secret=register,
            )
            try:
                if action == "test":
                    await host._test(manager, name, digest, cwd, op)
                else:
                    count = await grant_operation(manager, action, name, op, cfg)
                    if action != "logout":
                        host._record_grant(cwd, name, digest, op, count)
            finally:
                await manager.disconnect_all(in_task=True)

        op = self.ops.start(name, action, work)
        self._op_cwd[op["id"]] = cwd
        for key in [key for key in self._op_cwd if key not in self.ops.operations]:
            del self._op_cwd[key]
        return dict(op)

    async def _test(
        self, manager: Any, name: str, digest: str, cwd: str, op: dict[str, Any]
    ) -> None:
        """Connect once, NON-interactively, and record what happened.

        Non-interactive on purpose: Test must never open a browser. A server
        that needs a grant answers ``needs_sign_in``; the Sign-in action is the
        gesture that may open one.
        """
        from local_operator.mcp.auth import McpAuthChallengeError, McpAuthRequiredError
        from local_operator.mcp.manager import _unwrap_auth_required
        from local_operator.mcp.redaction import sanitize_exception

        status: str
        reason: str | None = None
        count: int | None = None
        try:
            await manager.connect_configured_server(name, interactive=False)
            count = len(manager.get_server_tools(name))
            status = "connected"
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — a failed test is a result
            sanitize_exception(exc)
            unwrapped = _unwrap_auth_required(exc)
            text = manager._auth_failure_text(name, unwrapped, manager._server_url(name))
            if isinstance(unwrapped, (McpAuthRequiredError, McpAuthChallengeError)):
                status = "needs_sign_in"
            else:
                status = "error"
            reason = public_reason(text)
        self.probes[(cwd, name)] = ProbeResult(
            status=status,  # type: ignore[arg-type]
            reason=reason,
            tool_count=count,
            observed_at=time.time(),
            digest=digest,
        )
        op["status"] = "complete" if status == "connected" else "failed"
        op["message"] = reason

    def _record_grant(
        self, cwd: str, name: str, digest: str, op: dict[str, Any], count: int | None
    ) -> None:
        """A completed sign-in IS a successful connect: remember it as a probe."""
        if op["status"] == "complete":
            self.probes[(cwd, name)] = ProbeResult(
                status="connected",
                reason=None,
                tool_count=count,
                observed_at=time.time(),
                digest=digest,
            )
        else:
            self.probes.pop((cwd, name), None)

    async def store_credentials(self, body: Any, cwd: str) -> dict[str, Any]:
        from local_operator.mcp.credentials import store_credentials

        return await store_credentials(_SettingsCredentialsHost(cwd, self._base()), body)

    async def close(self) -> None:
        """Refuse new work, then cancel and join every running operation."""
        self._closed = True
        await self.ops.close()


def mcp_host(app_state: Any) -> McpHost:
    """The app's one host, created on first use (the pool pattern)."""
    host = getattr(app_state, "desktop_mcp_host", None)
    if host is None:
        config_manager = getattr(app_state, "config_manager", None)
        host = McpHost(getattr(config_manager, "config_dir", None))
        app_state.desktop_mcp_host = host
    return host
