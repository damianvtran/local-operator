"""First-turn readiness for the MCP servers a run NAMED.

WHY THIS EXISTS. ``McpManager`` returns at its 250 ms startup gate and finishes
slower servers on background continuations, so a session's first provider
request goes out with whatever settled in that window. That is the right trade
for a lazy server nobody named (interactive startup never waits on it), and the
wrong one for a server whose tools ARE the run: a cold pipeline worker
(``lop exec --tools mcp__bd_search,mcp__bd_fetch`` in a fresh config dir, so no
cached tool list either) sent its first requests with NO tools at all, the
model wrote its tool calls as prose and the run returned nulls (head-to-head
benchmark, 2026-11-20, caveat 1; reproduced by
``tests/unit/session/test_declared_mcp_ready.py``). Every fresh container
worker hits it, because a cold Python stdio server never makes 250 ms.

THE RULE. Before the session's FIRST turn, wait — bounded — for the servers that
own a declared ``mcp__`` name the inventory cannot reach yet (``lop exec
--tools``, the SDK's ``ApprovalPolicy.declared``, a child's inherited
declaration) and for ``preload_tools`` servers that have contributed nothing
yet. Only servers still ``connecting`` are awaited, and only the named ones
(``McpManager.wait_settled(servers=...)``), so an unrelated slow server never
holds the turn. Undeclared lazy servers keep the 250 ms gate untouched, and a
warm tool cache (deferred tools already in the inventory) waits on nothing.

FAILING LOUDLY. A declared server still not connected after the bound raises
:class:`DeclaredMcpToolsUnavailable` before anything is sent: exec prints it as
``exec failed: …`` and exits non-zero, an SDK caller gets the exception. Sending
the request without the run's declared reach is the silent failure this module
removes. A ``preload_tools``-only server is a config preference rather than a
declaration, so missing it logs a warning and the turn proceeds — an interactive
session must not be refused because an opted-in server is broken.

Servers that are not ``connecting`` (connected, failed at startup, auth-blocked)
are not waited on and never raise here: their failure is already reported by
the startup path, and the end-of-run unreachable report still names a declared
tool that never arrived.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection
from typing import Any

from local_operator.mcp.tool_bridge import mcp_tool_name_prefix

logger = logging.getLogger(__name__)

#: Upper bound on the first-turn wait for declared/preloaded MCP servers.
#:
#: MEASURED, NOT GUESSED (2026-11-20, this machine, load average ~60): spawn ->
#: ``initialize`` -> ``tools/list`` through ``McpManager`` in a fresh HOME and
#: config dir, n=12 each — a minimal mcp 2.x ``MCPServer`` stdio server: median
#: 0.68 s, max 3.48 s; the benchmark's ``bd_tools_mcp.py`` (httpx + MCPServer):
#: median 0.85 s, max 2.97 s. Every sample missed the 250 ms gate (min 0.50 s).
#: 30 s is ~8.6x the observed max, leaves room for ``uvx``/``npx`` cold spawns
#: that resolve a package first, and equals Codex's own MCP
#: ``DEFAULT_STARTUP_TIMEOUT`` (``codex-rs/codex-mcp/src/rmcp_client.rs``, cited
#: beside ``DEFAULT_MCP_TIMEOUT_MS`` in ``mcp/manager.py``), the parity figure
#: for "how long a client waits for a server to come up". It bounds a WEDGED
#: server; a healthy one returns as soon as it lists its tools.
DECLARED_MCP_READY_TIMEOUT_S = 30.0


class DeclaredMcpToolsUnavailable(RuntimeError):
    """A declared MCP server was not ready when the first request was due.

    A ``RuntimeError`` so the headless prompt queue records it as the turn's
    failure reason (``serving.run_headless_prompt``) exactly like any other
    raising turn.
    """

    def __init__(self, servers: dict[str, str], tools: Collection[str], timeout_s: float) -> None:
        self.servers = dict(servers)
        self.tools = tuple(sorted(tools))
        self.timeout_s = timeout_s
        detail = ", ".join(f"{name} ({status})" for name, status in sorted(servers.items()))
        super().__init__(
            # Technical diagnostic, carried as the dynamic payload of exec's
            # i18n-ignored ``exec failed:`` prefix: server names, statuses and
            # tool identifiers, never prose to translate.
            # i18n: ignore technical diagnostic (identifiers in exec's ignored prefix)
            f"declared MCP server(s) not ready after {timeout_s:g} s: {detail}; "
            f"declared tools unreachable: {', '.join(self.tools)}"
        )


def _owns(server: str, names: Collection[str]) -> bool:
    prefix = mcp_tool_name_prefix(server)
    return any(name.startswith(prefix) for name in names)


def servers_to_await(
    manager: Any, declared: Collection[str] | None, reachable: Collection[str]
) -> tuple[frozenset[str], frozenset[str]]:
    """``(declared owners, preload-only)`` servers worth waiting for right now.

    A declared owner is a ``connecting`` server whose name prefix matches a
    declared ``mcp__`` name the inventory does not reach yet; a preload server
    is a ``connecting`` ``preload_tools`` server with no tools registered (a
    warm cache gives it deferred tools, which already reach the request).
    """
    missing = [
        name for name in (declared or ()) if name.startswith("mcp__") and name not in reachable
    ]
    owners: set[str] = set()
    preload: set[str] = set()
    for server in manager.get_all_server_names():
        if manager.get_connection_status(server) != "connecting":
            continue
        if missing and _owns(server, missing):
            owners.add(server)
            continue
        config = manager.get_server_config(server)
        if bool(getattr(config, "preload_tools", False)) and not manager.get_server_tools(server):
            preload.add(server)
    return frozenset(owners), frozenset(preload)


async def await_named_mcp_servers(
    manager: Any,
    *,
    declared: Collection[str] | None,
    reachable: Callable[[], Collection[str]],
    materialize: Callable[[], object],
    timeout_s: float = DECLARED_MCP_READY_TIMEOUT_S,
) -> None:
    """Wait for the run's named MCP servers, then grant their declared tools.

    ``reachable`` reads the session's LIVE inventory (it changes during the
    wait); ``materialize`` is ``Session.materialize_declared_tools``, called
    after the wait because the manager's round-wide settle callback — the only
    other caller — fires only once EVERY deferred server is done, which an
    unrelated slow server can postpone past this turn.
    """
    if manager is None:
        return
    try:
        owners, preload = servers_to_await(manager, declared, reachable())
    except Exception:  # noqa: BLE001 — a reduced manager (a host double) has no rounds to wait on
        logger.debug("MCP first-turn readiness unavailable on this manager", exc_info=True)
        return
    awaited = owners | preload
    if not awaited:
        return
    logger.debug("first turn waits for MCP server(s): %s", ", ".join(sorted(awaited)))
    try:
        await manager.wait_settled(timeout_s, servers=awaited)
    except Exception:  # noqa: BLE001 — the status read below decides; CancelledError propagates
        logger.debug("MCP first-turn wait failed", exc_info=True)
    materialize()
    statuses = {server: manager.get_connection_status(server) for server in awaited}
    not_ready = {server: status for server, status in statuses.items() if status != "connected"}
    for server in sorted(set(not_ready) & preload):
        logger.warning(
            "preload_tools MCP server %r not ready after %g s (%s); first turn proceeds without it",
            server,
            timeout_s,
            not_ready[server],
        )
    declared_not_ready = {server: not_ready[server] for server in owners if server in not_ready}
    if not declared_not_ready:
        return
    now_reachable = set(reachable())
    unreachable = {
        name
        for name in (declared or ())
        if name not in now_reachable and any(_owns(server, [name]) for server in declared_not_ready)
    }
    raise DeclaredMcpToolsUnavailable(declared_not_ready, unreachable, timeout_s)
