"""MCP support for the local-operator harness rewrite.

Official ``mcp`` Python SDK for transports + OAuth machinery; established
harness semantics for everything else: fast-startup gate with deferred tools, reconnect
circuit breaker, multi-source config discovery, tool-name mangling, and
outbound argument hygiene. See ``docs/REWRITE.md`` section E.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from local_operator.harness.types import AgentTool
from local_operator.mcp.manager import McpLoadResult, McpManager
from local_operator.mcp.tool_cache import McpToolCache

if TYPE_CHECKING:
    from local_operator.mcp.auth import ManagedAuthStore

logger = logging.getLogger(__name__)

__all__ = [
    "McpLoadResult",
    "McpManager",
    "McpToolCache",
    "discover_and_load_mcp_tools",
]


async def discover_and_load_mcp_tools(
    cwd: str,
    tool_cache: McpToolCache | None = None,
    auth_store: ManagedAuthStore | None = None,
) -> tuple[McpManager, list[AgentTool], list[dict[str, str]]]:
    """Convenience loader that discovers and loads MCP tools in one pass.

    Returns ``(manager, tools, errors)``:

    - ``manager`` — the :class:`McpManager` (caller owns its lifecycle);
    - ``tools`` — the harness ``AgentTool`` list (live + deferred) sorted by
      name;
    - ``errors`` — ``[{"path": "mcp:<server>", "error": ...}]`` entries, one
      per failed or invalid server.

    A hard discovery failure never raises: it yields the manager, an empty
    tool list, and one synthetic error entry (established behavior).

    ``tool_cache`` defaults to :class:`McpToolCache` under :func:`config_dir`
    so a deferred server can advertise last-good schemas at the 250 ms gate
    without every runtime waiting on spawn+handshake. The owner still spawns;
    live ``tools/list`` after connect overwrites the row. Passing ``None``
    used to mean "no cache", which made every runtime pay the handshake even
    when a sibling had just listed the same server.
    """
    manager = McpManager(cwd, tool_cache or McpToolCache(), auth_store=auth_store)
    try:
        result = await manager.discover_and_connect()
    except Exception as exc:
        logger.warning("MCP discovery failed: %s", exc, exc_info=True)
        return manager, [], [{"path": ".mcp.json", "error": str(exc)}]

    errors = [{"path": f"mcp:{name}", "error": message} for name, message in result.errors.items()]
    return manager, result.tools, errors
