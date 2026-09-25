"""Typed desktop MCP controls executed on the existing session manager.

Configuration ownership, OAuth persistence and server lifetimes remain in the
MCP core. Only secret references enter configuration; grants never cross HTTP.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from typing import Any, Awaitable, Callable, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from local_operator.mcp.config import (
    add_server,
    load_all_mcp_configs,
    owned_scope_for_source,
    remove_server,
)
from local_operator.mcp.grants import login_allowed, resolve_server, run_grant


class MCPControl(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal[
        "list",
        "add",
        "remove",
        "reload",
        "connect",
        "probe",
        # Sessionless only (``/v1/desktop/mcp``): one non-interactive connect on
        # a short-lived manager, reported as an operation. The session route
        # answers it ``operation_unavailable`` — its live manager has ``connect``.
        "test",
        "disconnect",
        "login",
        "logout",
        "reauth",
        "status",
        "cancel",
    ]
    name: str = Field(default="", pattern=r"^[A-Za-z0-9_.:-]*$", max_length=100)
    scope: Literal["global", "project"] = "global"
    command: str | None = Field(default=None, min_length=1, max_length=4096)
    args: list[str] = Field(default_factory=list, max_length=128)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = Field(default=None, max_length=4096)
    headers: dict[str, str] = Field(default_factory=dict)
    oauth: StrictBool = False
    confirmed: StrictBool = False
    operation_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")

    @model_validator(mode="after")
    def validate_control(self):
        if self.action not in {"list", "reload", "status", "cancel"} and not self.name:
            raise ValueError("Choose an MCP server")
        if self.action in {"remove", "disconnect", "logout", "reauth"} and not self.confirmed:
            raise ValueError("Confirm this MCP change")
        if self.action in {"status", "cancel"} and not self.operation_id:
            raise ValueError("Choose an operation")
        if self.action == "add":
            if bool(self.command) == bool(self.url):
                raise ValueError("Supply either a command or a URL")
            if self.url:
                parsed = urlsplit(self.url)
                if (
                    parsed.scheme not in {"http", "https"}
                    or not parsed.hostname
                    or parsed.username
                    or parsed.password
                    or parsed.query
                    or parsed.fragment
                ):
                    raise ValueError(
                        "Use an HTTP URL without inline credentials, query or fragment"
                    )
            if any(len(arg) > 8192 for arg in self.args):
                raise ValueError("An argument exceeds the size limit")
            if any(
                not re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", value)
                for value in [*self.env.values(), *self.headers.values()]
            ):
                raise ValueError(
                    "Environment and header values must be secret references such as ${TOKEN}"
                )
        elif self.command or self.url or self.args or self.env or self.headers or self.oauth:
            raise ValueError("Configuration fields are only accepted by add")
        return self


class MCPRefusal(ValueError):
    """A refused desktop MCP control, carrying a BOUNDED reason code.

    Every refusal used to be one ``ValueError`` that the owner collapsed to
    ``mcp_control_refused`` and the route to one fixed sentence, so the desktop
    could only say "the server refused it without giving a reason" — for a
    duplicate name, an unowned source, a busy grant and an MCP manager that had
    simply not finished wiring alike. The code crosses process and HTTP
    boundaries; the message never does (config errors can quote credentials),
    so a client renders its own copy per code. Subclasses ``ValueError`` so every
    existing ``except ValueError`` keeps catching it.
    """

    def __init__(self, code: RefusalCode, message: str = "") -> None:
        super().__init__(message or code)
        self.code: RefusalCode = code


RefusalCode = Literal[
    "exists",
    "not_owned",
    "project_scope_unavailable",
    "unknown_server",
    "oauth_unsupported",
    "grant_running",
    "too_many_operations",
    "write_failed",
    "invalid_config",
    "mcp_starting",
    "operation_unavailable",
]

#: The fixed, credential-free sentence each code crosses HTTP with. Clients key
#: on ``code``; this is the fallback copy for one that does not know it yet.
REFUSAL_MESSAGES: dict[str, str] = {
    "exists": "An MCP server with that name already exists.",
    "not_owned": "That server is defined in a file Local Operator does not manage.",
    "project_scope_unavailable": (
        "This folder has no separate project scope; its project file is the global one."
    ),
    "unknown_server": "That MCP server is not configured.",
    "oauth_unsupported": "That MCP server does not use OAuth sign-in.",
    "grant_running": "Another MCP operation is running; wait for it or cancel it first.",
    "too_many_operations": "Too many MCP operations are pending.",
    "write_failed": "The MCP configuration could not be written.",
    "invalid_config": "That MCP server configuration is not valid.",
    "mcp_starting": "MCP is still starting in this conversation; try again in a moment.",
    # One code, two situations, so the sentence has to read for BOTH: a stale
    # ``operation_id`` (evicted from the bounded registry, or the daemon
    # restarted under the client) and a live-only action (connect / disconnect /
    # reload / probe) sent to the sessionless route, which has no runtime's
    # connection to control. "no longer available" was true only of the first,
    # and told the second user their control had expired when it was never
    # offered here at all. Clients key on ``code``; this is the fallback copy.
    "operation_unavailable": "That MCP operation is not available.",
    "mcp_control_refused": "The MCP control was refused.",
}


def refusal_code(exc: BaseException) -> str:
    """The bounded code for a refused control, whatever raised it."""
    from local_operator.mcp.config import MCPConfigWriteError

    if isinstance(exc, MCPRefusal):
        return exc.code
    if isinstance(exc, MCPConfigWriteError):
        return exc.code if exc.code in REFUSAL_MESSAGES else "write_failed"
    return "mcp_control_refused"


def refusal_detail(code: object) -> dict[str, str]:
    """The 409 ``detail`` object: ``{code, message}``, never exception text."""
    key = code if isinstance(code, str) and code in REFUSAL_MESSAGES else "mcp_control_refused"
    return {"code": key, "message": REFUSAL_MESSAGES[key]}


#: How long a session-route ``reload``/``add``/``remove`` waits for servers
#: deferred past the 250 ms startup gate before answering. The answer used to
#: be taken AT the gate, so every server slower than 250 ms (any npx/uvx
#: spawn) read "connecting / 0 tools" as Reload's result. Bounded well inside
#: the owner RPC's 15 s ack budget; a slower server keeps connecting and
#: honestly reads ``connecting``.
RELOAD_SETTLE_S = 8.0

#: Bound on one operation (test / sign-in). A human completes the browser step.
OPERATION_TIMEOUT_S = 300.0

#: Operations kept for ``status`` reads; settled ones are evicted first.
OPERATION_LIMIT = 64


class McpOperations:
    """The operation registry shared by the session route and the probe host.

    One operation at a time per registry (a second loopback OAuth listener
    would contend for the callback port), bounded history, cancel by id, and a
    record shape both routes publish: ``{id, name, action, status, created_at,
    credential_removed, browser_opened, authorization_url, message}``.
    """

    def __init__(self, tasks: set[asyncio.Task[None]] | None = None) -> None:
        self.tasks: set[asyncio.Task[None]] = tasks if tasks is not None else set()
        self.operations: dict[str, dict[str, Any]] = {}
        self.running: dict[str, asyncio.Task[None]] = {}
        self.lock = asyncio.Lock()

    def records(self) -> list[dict[str, Any]]:
        return [dict(op) for op in self.operations.values()]

    def running_names(self) -> frozenset[str]:
        return frozenset(op["name"] for key, op in self.operations.items() if key in self.running)

    def ensure_idle(self) -> None:
        if self.running:
            raise MCPRefusal("grant_running", "Wait for the active MCP operation")

    def record(self, operation_id: str | None) -> dict[str, Any]:
        op = self.operations.get(operation_id or "")
        if op is None:
            raise MCPRefusal("operation_unavailable", "This MCP operation is no longer available")
        return op

    async def cancel(self, operation_id: str | None) -> dict[str, Any]:
        op = self.record(operation_id)
        task = self.running.get(operation_id or "")
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return op

    def start(
        self,
        name: str,
        action: str,
        work: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> dict[str, Any]:
        """Register an operation and run ``work(op)`` as its task.

        ``work`` mutates ``op`` (status, message, browser facts) and must
        create, use AND tear down any manager inside itself: anyio refuses to
        exit a transport's cancel scope from a task other than the one that
        entered it, so teardown cannot be delegated to a sibling.
        """
        self.ensure_idle()
        if len(self.operations) >= OPERATION_LIMIT:
            settled = next((key for key in self.operations if key not in self.running), None)
            if settled is None:
                raise MCPRefusal("too_many_operations", "Too many MCP operations")
            del self.operations[settled]
        operation_id = uuid.uuid4().hex
        op: dict[str, Any] = {
            "id": operation_id,
            "name": name,
            "action": action,
            "status": "running",
            "created_at": time.time(),
            "credential_removed": False,
            "browser_opened": None,
            "authorization_url": None,
            "message": None,
        }
        self.operations[operation_id] = op

        async def run() -> None:
            from local_operator.mcp.catalog import public_reason

            try:
                async with asyncio.timeout(OPERATION_TIMEOUT_S):
                    await work(op)
            except asyncio.CancelledError:
                op["status"] = "cancelled"
                raise
            except TimeoutError:
                op["status"] = "failed"
                op["message"] = "The operation did not finish in time."
            except Exception as exc:  # noqa: BLE001 — an operation failure is a record
                from local_operator.mcp.redaction import sanitize_exception

                sanitize_exception(exc)
                op["status"] = "failed"
                op["message"] = public_reason(exc)
            finally:
                self.running.pop(operation_id, None)

        task = asyncio.create_task(run())
        self.running[operation_id] = task
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return op

    async def close(self) -> None:
        """Cancel and JOIN every running operation (its teardown runs inside).

        ``running`` is cleared afterwards rather than left to each task's own
        ``finally``: a task cancelled before its first step never executes, so its
        ``finally`` never runs and the entry would survive as a phantom
        "operation in flight". A read after that (a catalog render that happens
        between the host's close and the process actually exiting) would paint the
        row ``connecting`` with nothing behind it.
        """
        tasks = list(self.running.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.running.clear()


def observe_authorization(op: dict[str, Any]) -> Callable[[str, bool], None]:
    """An :data:`~local_operator.mcp.auth.AUTHORIZATION_OBSERVER` writing ``op``.

    Publishes whether a browser opened and, when none did, the URL a client
    can offer as a link. The URL is a one-time PKCE authorization request to
    the provider, not a credential, and the desktop plane is already behind the
    bearer token.
    """

    def observe(url: str, opened: bool) -> None:
        op["browser_opened"] = bool(opened)
        op["authorization_url"] = url

    return observe


async def grant_operation(
    manager: Any, action: str, name: str, op: dict[str, Any], cfg: Any = None
) -> int | None:
    """Run one grant verb into ``op``; the tool count on a successful sign-in.

    The capability probe runs HERE, inside the operation, rather than in the
    request: it is up to three sequential 10 s discovery GETs (measured 30.7 s
    against an unroutable host), past any client's control deadline.
    """
    from local_operator.mcp.auth import AUTHORIZATION_OBSERVER
    from local_operator.mcp.catalog import public_reason

    if action != "logout" and cfg is not None and not await login_allowed(manager, cfg):
        op["status"] = "failed"
        op["message"] = (
            "No OAuth authorization server was discovered for this server; "
            "check its URL and your network, or add its key instead."
        )
        return None
    forgotten: list[str] = []
    token = AUTHORIZATION_OBSERVER.set(observe_authorization(op))
    try:
        text, style = await run_grant(manager, action, name, forgotten)
    finally:
        AUTHORIZATION_OBSERVER.reset(token)
        op["credential_removed"] = bool(forgotten)
    op["status"] = "complete" if style == "success" else "failed"
    if style != "success":
        op["message"] = public_reason(text)
    if action == "logout" and style == "success":
        op["credential_removed"] = True
    if style == "success" and action != "logout":
        return len(manager.get_server_tools(name))
    return None


class MCPDesktop:
    """The SESSION route's MCP controls, run on the owner's live manager.

    Kept for desktop builds that predate the sessionless catalog
    (``/v1/desktop/mcp``, ``server/mcp_host.py``): its row shape and status
    words are unchanged. Refusals now carry a bounded :class:`MCPRefusal`
    code, and a reload answers after the startup gate's stragglers settle.
    """

    def __init__(self, session: Any, tasks: set[asyncio.Task[None]], cwd: str):
        self.session = session
        self.cwd = cwd
        self.ops = McpOperations(tasks)

    @property
    def operations(self) -> dict[str, dict[str, Any]]:
        return self.ops.operations

    @property
    def running(self) -> dict[str, asyncio.Task[None]]:
        return self.ops.running

    def snapshot(self) -> dict[str, Any]:
        """The LEGACY session-route row shape, derived here on purpose.

        Desktop builds that predate ``features.mcp_catalog`` read this shape —
        its own key names, ``stdio``/``http``, the manager's status words — so it
        cannot be replaced by the catalog's rows in ``mcp/catalog.py``, which
        carry the published catalog vocabulary. Two compatibility requirements
        cannot be met by one body, so this is a translation of a frozen shape
        rather than a second definition of the catalog; the catalog consumes it
        through ``live_facts_from_snapshot``, and nothing else crosses between
        them.
        """
        from local_operator.mcp.catalog import public_reason, public_server_config

        configs, sources = load_all_mcp_configs(self.cwd)
        manager = self.session.mcp_manager
        loaded = set(manager.get_all_server_names())
        # Annotated ``Any`` on purpose: it is OPTIONAL manager surface read with
        # ``getattr`` (a test double or a reduced host may not have it), and an
        # unannotated ``getattr`` narrows to ``object``, which pyright then refuses
        # to call or subscript.
        failures_of: Any = getattr(manager, "startup_failures", None)
        # ``isinstance(..., Callable)`` rather than ``callable(...)``: pyright
        # narrows ``callable()`` on an ``Any`` to ``object`` and then refuses both
        # the call and the result, while ``isinstance`` narrows it to a callable.
        failures: dict[str, str] = (
            {str(key): str(value) for key, value in failures_of().items()}
            if isinstance(failures_of, Callable)
            else {}
        )
        rows = []
        for name, cfg in configs.items():
            source = sources.get(name)
            scope = owned_scope_for_source(source, self.cwd)
            status = manager.get_connection_status(name)
            rows.append(
                {
                    "name": name,
                    "source": str(source) if source else None,
                    "owned_scope": scope,
                    "removable": scope is not None,
                    **public_server_config(cfg),
                    "setup": {
                        "kind": "session_prompt",
                        "text": (
                            f"Help me set up access for MCP server {name}; inspect its documented "
                            f"tools and request any needed user consent."
                        ),
                    },
                    "status": status,
                    "tool_count": len(manager.get_server_tools(name)),
                    # Additive, for the catalog's live overlay: whether THIS
                    # runtime loaded the server at all (one added after the
                    # conversation started is config-only until a reload), and
                    # why it failed, sanitized. Older readers ignore both.
                    "loaded": name in loaded,
                    "startup_failure": (
                        public_reason(failures.get(name))
                        if status not in ("connected", "connecting")
                        else None
                    ),
                }
            )
        return {"servers": rows, "operations": self.ops.records()}

    async def _reload(self, manager: Any) -> None:
        await manager.reload()
        # See ``snapshot`` on why this is ``isinstance`` rather than ``callable``.
        settle: Any = getattr(manager, "wait_settled", None)
        if isinstance(settle, Callable):
            await settle(RELOAD_SETTLE_S)

    async def execute(self, body: MCPControl) -> dict[str, Any]:
        manager = getattr(self.session, "mcp_manager", None)
        if manager is None:
            # MCP wiring is deferred past runtime start (publication-gated, with
            # imports warmed in a thread), so a control in that window is
            # "not yet", not "never" — a client can retry this one.
            raise MCPRefusal("mcp_starting", "MCP is not available in this session yet")
        if body.action == "list":
            return self.snapshot()
        if body.action == "status":
            return dict(self.ops.record(body.operation_id))
        if body.action == "cancel":
            return dict(await self.ops.cancel(body.operation_id))
        async with self.ops.lock:
            self.ops.ensure_idle()
            if body.action in {"add", "remove"}:
                configs, sources = load_all_mcp_configs(self.cwd)
                if body.action == "add":
                    if body.name in configs:
                        raise MCPRefusal(
                            "exists", "That server already exists; remove its owned definition"
                        )
                    add_server(
                        body.name,
                        command=body.command,
                        args=body.args,
                        env=body.env,
                        url=body.url,
                        headers=body.headers,
                        oauth=body.oauth,
                        scope=body.scope,
                        cwd=self.cwd,
                    )
                else:
                    if body.name not in configs:
                        raise MCPRefusal("unknown_server", "Unknown MCP server")
                    if owned_scope_for_source(sources.get(body.name), self.cwd) != body.scope:
                        raise MCPRefusal("not_owned", "Not owned by the selected scope")
                    remove_server(body.name, scope=body.scope, cwd=self.cwd)
                await self._reload(manager)
                return self.snapshot()
            if body.action == "reload":
                await self._reload(manager)
                return self.snapshot()
            if body.name not in manager.get_all_server_names():
                raise MCPRefusal("unknown_server", "Unknown MCP server")
            if body.action == "disconnect":
                await manager.disconnect_server(body.name)
                return self.snapshot()
            if body.action == "connect":
                await manager.reconnect_server(body.name)
                return self.snapshot()
            if body.action == "test":
                raise MCPRefusal("operation_unavailable", "Use connect on a session")
            resolved = resolve_server(self.session, body.name)
            if body.action == "probe":
                return await self._probe(body.name, resolved)
            if isinstance(resolved, str):
                raise MCPRefusal("oauth_unsupported", "This server does not support OAuth")
            if body.action != "logout" and not await login_allowed(*resolved):
                raise MCPRefusal("oauth_unsupported", "This server does not support OAuth")
            action = body.action
            name = body.name

            async def work(op: dict[str, Any]) -> None:
                # The capability probe already ran above (this route's
                # established contract answers an ineligible server with a
                # refusal, not a failed operation), so it is not repeated.
                await grant_operation(manager, action, name, op)

            return dict(self.ops.start(name, action, work))

    async def _probe(self, name: str, resolved: Any) -> dict[str, Any]:
        from pathlib import Path

        from local_operator.mcp.auth import server_rejects_oauth
        from local_operator.mcp.credentials import credential_source
        from local_operator.mcp.secret_refs import public_secret_refs
        from local_operator.paths import config_dir

        configs, _ = load_all_mcp_configs(self.cwd)
        cfg = configs.get(name)
        if cfg is None:
            raise MCPRefusal("unknown_server", "Unknown MCP server")
        # Best-effort discovery's False also means unreachable/unknown.
        # Only explicit transport/config refusal may claim non-OAuth.
        supported = (
            False
            if server_rejects_oauth(cfg)
            else (
                True if not isinstance(resolved, str) and await login_allowed(*resolved) else None
            )
        )
        refs = public_secret_refs(cfg)
        base = Path(getattr(self.session, "config_dir", None) or config_dir())
        states = await asyncio.to_thread(
            lambda: [
                {"id": ref["id"], "source": credential_source(ref["id"], base)} for ref in refs
            ]
        )
        return {
            "name": name,
            "transport_oauth_supported": supported,
            "downstream_authorization": "unknown",
            "secret_refs": refs,
            "credential_state": states,
            "key_submission_supported": True,
        }


def public_server_config(cfg: Any) -> dict[str, Any]:
    """Re-exported from :mod:`local_operator.mcp.catalog` for existing importers."""
    from local_operator.mcp.catalog import public_server_config as build

    return build(cfg)
