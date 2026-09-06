"""Typed desktop MCP controls executed on the existing session manager.

Configuration ownership, OAuth persistence and server lifetimes remain in the
MCP core. Only secret references enter configuration; grants never cross HTTP.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from typing import Any, Literal
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


def public_server_config(cfg: Any) -> dict[str, Any]:
    """Expose destinations, never legacy inline headers/env/argument secrets."""
    from local_operator.mcp.auth import server_rejects_oauth

    url = getattr(cfg, "url", None)
    redacted = False
    if url:
        try:
            parsed = urlsplit(url)
            redacted = bool(parsed.username or parsed.password or parsed.query or parsed.fragment)
        except ValueError:
            redacted = True
    command = getattr(cfg, "command", None)
    return {
        "transport": "stdio" if command else "http",
        "command": command,
        "argument_count": len(getattr(cfg, "args", [])),
        "url": None if redacted else url,
        "endpoint_redacted": redacted,
        "environment_keys": sorted(getattr(cfg, "env", {})),
        "header_keys": sorted(getattr(cfg, "headers", {})),
        "transport_oauth_supported": False if server_rejects_oauth(cfg) else None,
        "downstream_authorization": "unknown",
    }


class MCPDesktop:
    def __init__(self, session: Any, tasks: set[asyncio.Task[None]], cwd: str):
        self.session = session
        self.cwd = cwd
        self.tasks = tasks
        self.operations: dict[str, dict[str, Any]] = {}
        self.running: dict[str, asyncio.Task[None]] = {}
        self.lock = asyncio.Lock()

    def snapshot(self) -> dict[str, Any]:
        configs, sources = load_all_mcp_configs(self.cwd)
        manager = self.session.mcp_manager
        rows = []
        for name, cfg in configs.items():
            source = sources.get(name)
            scope = owned_scope_for_source(source, self.cwd)
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
                    "status": manager.get_connection_status(name),
                    "tool_count": len(manager.get_server_tools(name)),
                }
            )
        return {"servers": rows, "operations": list(self.operations.values())}

    async def execute(self, body: MCPControl) -> dict[str, Any]:
        manager = getattr(self.session, "mcp_manager", None)
        if manager is None:
            raise ValueError("MCP is not available in this session")
        if body.action == "list":
            return self.snapshot()
        if body.action in {"status", "cancel"}:
            op = self.operations.get(body.operation_id or "")
            if op is None:
                raise ValueError("This MCP operation is no longer available")
            if body.action == "cancel":
                task = self.running.get(body.operation_id or "")
                if task is not None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            return dict(op)
        async with self.lock:
            if self.running:
                raise ValueError("Wait for the active MCP grant or cancel it first")
            if body.action in {"add", "remove"}:
                configs, sources = load_all_mcp_configs(self.cwd)
                if body.action == "add":
                    if body.name in configs:
                        raise ValueError(
                            "That server already exists; remove its owned definition first"
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
                    if owned_scope_for_source(sources.get(body.name), self.cwd) != body.scope:
                        raise ValueError("This source is not owned by the selected scope")
                    remove_server(body.name, scope=body.scope, cwd=self.cwd)
                await manager.reload()
                return self.snapshot()
            if body.action == "reload":
                await manager.reload()
                return self.snapshot()
            if body.name not in manager.get_all_server_names():
                raise ValueError("Unknown MCP server")
            if body.action == "disconnect":
                await manager.disconnect_server(body.name)
                return self.snapshot()
            if body.action == "connect":
                await manager.reconnect_server(body.name)
                return self.snapshot()
            resolved = resolve_server(self.session, body.name)
            if body.action == "probe":
                supported = False if isinstance(resolved, str) else await login_allowed(*resolved)
                return {
                    "name": body.name,
                    "transport_oauth_supported": supported,
                    "downstream_authorization": "unknown",
                }
            if isinstance(resolved, str):
                raise ValueError("This server does not support OAuth")
            if body.action != "logout" and not await login_allowed(*resolved):
                raise ValueError("This server does not support OAuth")
            if any(
                op["name"] == body.name and op["status"] == "running"
                for op in self.operations.values()
            ):
                raise ValueError("This server already has a grant operation")
            if len(self.operations) >= 64:
                settled = next((key for key in self.operations if key not in self.running), None)
                if settled is None:
                    raise ValueError("Too many MCP operations")
                del self.operations[settled]
            operation_id = uuid.uuid4().hex
            op = {
                "id": operation_id,
                "name": body.name,
                "action": body.action,
                "status": "running",
                "created_at": time.time(),
                "credential_removed": False,
            }
            self.operations[operation_id] = op

            async def run():
                forgotten: list[str] = []
                try:
                    async with asyncio.timeout(300):
                        _, style = await run_grant(manager, body.action, body.name, forgotten)
                    op["status"] = "complete" if style == "success" else "failed"
                except asyncio.CancelledError:
                    op["status"] = "cancelled"
                    raise
                except Exception:
                    op["status"] = "failed"
                finally:
                    op["credential_removed"] = bool(forgotten) or (
                        body.action == "logout" and op["status"] == "complete"
                    )
                    self.running.pop(operation_id, None)

            task = asyncio.create_task(run())
            self.running[operation_id] = task
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
            return dict(op)
