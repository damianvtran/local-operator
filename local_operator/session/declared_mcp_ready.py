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

THE RULE: DECIDED BY TOOL REACHABILITY, NOT SERVER STATUS. Before the
session's first turn (and again on any later turn until it succeeds), wait,
bounded, only while some declared ``mcp__`` name (``lop exec --tools``, the
SDK's ``ApprovalPolicy.declared``) is not reachable yet AND a server that could
provide it is still ``connecting``. The wait re-reads the inventory as each
awaited server settles (``McpManager.wait_settled(first=True)``) and ends the
moment every declared name is reachable. A tool name cannot be split back into
(server, tool):
``mcp__bd_x_search`` is server ``bd`` + ``x_search`` or server ``bd_x`` +
``search``, so every configured server whose prefix matches is a candidate.
Deciding on reachability is what keeps a wedged ``bd`` from holding, or
failing, a run whose ``mcp__bd_x_search`` already arrived from ``bd_x``
(agent review round 1, R1; QA Q-F1). An unrelated slow server is never a
candidate, undeclared lazy servers keep the 250 ms gate, and a warm tool cache
(deferred tools already in the inventory) waits on nothing.

NEVER IN A DELEGATED CHILD. A child's declaration is its parent's bound
carried down and its MCP tools are fixed by the parent's surface on a
borrowed manager (``Session._mcp_reach_inherited``), so the session skips this
gate there: the grant step would hand the child tools its parent withheld (an
``ownTurnOnly`` episode server's, merge-gate run 44707733427), and no wait can
deliver anything into that fixed surface.

PRELOAD-ONLY SERVERS. A ``preload_tools`` server with no tools yet is awaited
the same way, but ONLY on a headless host (exec, SDK, scheduler: no front end
attached). With a front end attached its tools land through the mid-session
merge, the documented TUI contract in ``session_factory.create_session``
(``defer_mcp_wiring``: "a server that opted into ``preload_tools`` is no
exception"), so the turn keeps the fast gate there (review round 1, R4).
Missing a preload-only server only logs a warning: it is a config preference,
not a declaration, and must never refuse a turn.

FAILING LOUDLY. When the wait ends (every candidate settled, or the bound
reached) and a declared name is still unreachable while one of its candidate
servers is not ``connected`` (still connecting, failed to start, crashed, or
auth-blocked), :class:`DeclaredMcpToolsUnavailable` is raised before anything
is sent: exec prints it as ``exec failed: …`` and exits non-zero, an SDK caller
gets the exception. That includes a server that had ALREADY failed when the
check ran (a bad command, an instant crash: QA Q-F2), which the status-keyed
version never raised for. A name whose candidates all connected without
offering it, or that no configured server could own, is a typo or a
role-derived name for a tool that exists on another machine; it is left to the
end-of-run report (``exec_startup.report_unresolved_declared_tools``), whose
contract is exactly that distinction.

CANCELLABLE. The wait runs before the turn's ``AbortSignal`` exists, so it
polls the caller's ``cancelled`` predicate (``Session._abort_requested``)
between short slices and returns early on an abort; the turn admission then
interrupts the aborted turn exactly as it would have after the wait.
"""

from __future__ import annotations

import logging
import time
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


#: How long one wait slice lasts before the gate re-checks ``cancelled``. The
#: wait also wakes the moment any awaited server settles, so this bounds only
#: how late an Esc is noticed, not how late a tool is seen.
_CANCEL_POLL_S = 0.25


class DeclaredMcpToolsUnavailable(RuntimeError):
    """Declared MCP tools were unreachable when the first request was due.

    A ``RuntimeError`` so the headless prompt queue records it as the turn's
    failure reason (``serving.run_headless_prompt``) exactly like any other
    raising turn. ``servers`` maps each not-``connected`` candidate server to
    its status at the end of the wait; ``elapsed_s`` is how long the gate
    actually waited, so the message says "failed after 0.3 s" for a bad
    command rather than claiming a 30 s timeout that never happened (QA O2).
    """

    def __init__(
        self,
        servers: dict[str, str],
        tools: Collection[str],
        timeout_s: float,
        elapsed_s: float,
    ) -> None:
        self.servers = dict(servers)
        self.tools = tuple(sorted(tools))
        self.timeout_s = timeout_s
        self.elapsed_s = elapsed_s
        bound_reached = elapsed_s >= timeout_s
        detail = "; ".join(
            f"server {name} {_failure_reason(status, timeout_s, bound_reached)}"
            for name, status in sorted(servers.items())
        )
        super().__init__(
            # Technical diagnostic, carried as the dynamic payload of exec's
            # i18n-ignored ``exec failed:`` prefix: server names, statuses and
            # tool identifiers, never prose to translate.
            # i18n: ignore technical diagnostic (identifiers in exec's ignored prefix)
            f"declared MCP tools unreachable after {elapsed_s:.1f} s: {', '.join(self.tools)} "
            f"({detail})"
        )


def _failure_reason(status: str, timeout_s: float, bound_reached: bool) -> str:
    """Why a candidate server did not deliver, in the manager's own vocabulary.

    "Timed out" only when the bound really ran out: a server parked in
    reconnect backoff reads ``connecting`` with no connect attempt in flight,
    so the wait ends early and must not claim a timeout that never happened.
    """
    if status == "connecting":
        if bound_reached:
            return f"timed out: still connecting at the {timeout_s:g} s bound"
        return "still connecting, with no connect attempt in flight to wait on"
    return f"failed: {status}"


def _owns(server: str, names: Collection[str]) -> bool:
    prefix = mcp_tool_name_prefix(server)
    return any(name.startswith(prefix) for name in names)


def _missing(declared: Collection[str] | None, reachable: Collection[str]) -> list[str]:
    """Declared ``mcp__`` names the inventory does not reach yet."""
    present = set(reachable)
    return sorted(
        name for name in (declared or ()) if name.startswith("mcp__") and name not in present
    )


def servers_to_await(
    manager: Any,
    declared: Collection[str] | None,
    reachable: Collection[str],
    *,
    wait_for_preload: bool = True,
) -> tuple[frozenset[str], frozenset[str]]:
    """``(declared candidates, preload-only)`` servers worth waiting for RIGHT NOW.

    A declared candidate is a ``connecting`` server whose name prefix matches a
    declared ``mcp__`` name the inventory does not reach yet; once every
    declared name is reachable there are none, whatever a prefix sibling is
    doing. A preload server is a ``connecting`` ``preload_tools`` server with
    no tools registered (a warm cache gives it deferred tools, which already
    reach the request), and only when ``wait_for_preload`` (a headless host).
    """
    missing = _missing(declared, reachable)
    owners: set[str] = set()
    preload: set[str] = set()
    for server in manager.get_all_server_names():
        if manager.get_connection_status(server) != "connecting":
            continue
        if missing and _owns(server, missing):
            owners.add(server)
            continue
        if not wait_for_preload:
            continue
        config = manager.get_server_config(server)
        if bool(getattr(config, "preload_tools", False)) and not manager.get_server_tools(server):
            preload.add(server)
    return frozenset(owners), frozenset(preload)


def _not_delivered(
    manager: Any, declared: Collection[str] | None, reachable: Collection[str]
) -> tuple[dict[str, str], list[str]]:
    """``({server: status}, names)`` for declared names a FAILED server owes.

    A name counts only while one of its candidate servers is not ``connected``:
    that server is the reason the name is missing. Names whose candidates all
    connected (the server simply has no such tool) or that no configured
    server could own are the end-of-run report's to name, see the module
    docstring.
    """
    statuses = {
        server: manager.get_connection_status(server) for server in manager.get_all_server_names()
    }
    servers: dict[str, str] = {}
    names: list[str] = []
    for name in _missing(declared, reachable):
        culprits = {
            server: status
            for server, status in statuses.items()
            if status != "connected" and _owns(server, [name])
        }
        if culprits:
            names.append(name)
            servers.update(culprits)
    return servers, names


async def await_named_mcp_servers(
    manager: Any,
    *,
    declared: Collection[str] | None,
    reachable: Callable[[], Collection[str]],
    materialize: Callable[[], object],
    wait_for_preload: bool = True,
    cancelled: Callable[[], bool] = lambda: False,
    timeout_s: float = DECLARED_MCP_READY_TIMEOUT_S,
) -> bool:
    """Wait for the run's named MCP servers, grant their declared tools, decide.

    Returns ``True`` once the gate is SETTLED (nothing declared is missing for
    a reason a wait could fix, preload servers settled or given up on), and
    ``False`` when ``cancelled`` cut the wait short. The caller records "done"
    only on ``True``, so a turn after an abort or after the raise below runs
    the gate again instead of sending without the declared reach (R3).

    ``reachable`` reads the session's LIVE inventory (it changes during the
    wait); ``materialize`` is ``Session.materialize_declared_tools``, called
    before the first read and after every wake, because the manager's
    round-wide settle callback — the only other caller — fires only once EVERY
    deferred server is done, which an unrelated slow server can postpone past
    this turn.
    """
    if manager is None:
        return True
    started = time.monotonic()
    deadline = started + max(0.0, timeout_s)
    preload_seen: set[str] = set()
    try:
        materialize()
        while True:
            owners, preload = servers_to_await(
                manager, declared, reachable(), wait_for_preload=wait_for_preload
            )
            preload_seen |= preload
            awaited = owners | preload
            remaining = deadline - time.monotonic()
            if not awaited or remaining <= 0:
                break
            if cancelled():
                return False
            logger.debug("first turn waits for MCP server(s): %s", ", ".join(sorted(awaited)))
            try:
                all_settled = await manager.wait_settled(
                    min(remaining, _CANCEL_POLL_S), servers=awaited, first=True
                )
            except Exception:  # noqa: BLE001 — the reachability read below decides
                logger.debug("MCP first-turn wait failed", exc_info=True)
                all_settled = True
            materialize()
            if all_settled and servers_to_await(
                manager, declared, reachable(), wait_for_preload=wait_for_preload
            ) == (owners, preload):
                # Nothing left to wait ON (no continuation for these servers,
                # e.g. one parked in reconnect backoff) and nothing moved: a
                # further pass would spin. Decide on what is reachable now.
                break
        servers, names = _not_delivered(manager, declared, reachable())
    except Exception:  # noqa: BLE001 — a reduced manager (a host double) has no rounds to wait on
        logger.debug("MCP first-turn readiness unavailable on this manager", exc_info=True)
        return True
    for server in sorted(preload_seen):
        status = manager.get_connection_status(server)
        if status != "connected":
            logger.warning(
                "preload_tools MCP server %r not ready after %.1f s (%s); first turn proceeds "
                "without it",
                server,
                time.monotonic() - started,
                status,
            )
    if names:
        raise DeclaredMcpToolsUnavailable(servers, names, timeout_s, time.monotonic() - started)
    return True
