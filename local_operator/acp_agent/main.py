"""``lop acp``'s entry: claim stdio, build the agent, serve until stdin EOF."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, cast

logger = logging.getLogger(__name__)


def _missing_extra_message() -> str:
    from local_operator.optional import missing_extra_error

    return missing_extra_error("acp", "`lop acp` (the Agent Client Protocol agent)")


def _session_roots(cwd: str) -> Any:
    """D10: the operator's ordinary roots, honouring the isolation overrides.

    ``allow_ambient`` is set only when the resolved roots ARE the uid defaults —
    an ACP client launching ``lop acp`` means "my Local Operator", so the
    operator's store is the intended one — while ``LOCAL_OPERATOR_CONFIG_DIR``/
    ``LOCAL_OPERATOR_HOME`` (and a redirected ``HOME``) still isolate a test.
    """
    from local_operator import paths
    from local_operator.session.spec import SessionRoots

    config = paths.config_dir().expanduser().resolve()
    home = paths.agent_home_dir().expanduser().resolve()
    probe = SessionRoots(config_dir=config, agent_home=home, cwd=cwd)
    ambient = config == probe.default_config_path or home == probe.default_agent_home_path
    return SessionRoots(
        config_dir=config,
        agent_home=home,
        cwd=cwd,
        allow_ambient=ambient,
        # The durable-root rule protects the operator's long-lived store. An
        # explicit LOCAL_OPERATOR_CONFIG_DIR is a caller's deliberate choice
        # (test rigs put it under $TMPDIR), so it is taken as said.
        allow_volatile=bool(os.environ.get("LOCAL_OPERATOR_CONFIG_DIR")),
    )


def make_session_opener(*, hosting: str | None, model: str | None) -> Callable[..., Any]:
    """The production opener: one SDK session per ACP session (D1)."""
    from local_operator import sdk
    from local_operator.session.spec import ApprovalPolicy, SessionSpec

    @asynccontextmanager
    async def opener(
        cwd: str, gate: Callable[..., Any], mcp_servers: dict[str, dict[str, Any]]
    ) -> AsyncIterator[Any]:
        spec = SessionSpec(
            hosting=hosting,
            model=model,
            approvals=ApprovalPolicy.callback(gate),
            mcp_servers=mcp_servers or None,
        )
        # Several ACP sessions in one process share one root, so the SDK's
        # single-root rule is satisfied; nothing here needs allow_multi_root.
        async with sdk.open_session(spec, roots=_session_roots(cwd)) as session:
            yield session

    return opener


async def _serve(reader_fd: int, writer_fd: int, *, hosting: str | None, model: str | None) -> int:
    import acp
    from acp.core import DEFAULT_STDIO_BUFFER_LIMIT_BYTES
    from acp.stdio import _WritePipeProtocol

    from local_operator.acp_agent.agent import LopAcpAgent

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=DEFAULT_STDIO_BUFFER_LIMIT_BYTES)
    await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(reader_fd, "rb", buffering=0)
    )
    # The SDK's own write-pipe protocol, so ``drain`` backpressure behaves
    # exactly as its ``stdio_streams`` would on fd 1.
    write_protocol = _WritePipeProtocol()
    transport, _ = await loop.connect_write_pipe(
        lambda: write_protocol, os.fdopen(writer_fd, "wb", buffering=0)
    )
    writer = asyncio.StreamWriter(transport, write_protocol, None, loop)

    agent = LopAcpAgent(make_session_opener(hosting=hosting, model=model))
    try:
        # Returns when the client closes our stdin (EOF): that is the
        # protocol's "goodbye", and the process must then exit cleanly.
        # ``cast``: ``acp.Agent`` is a Protocol listing every method, while the
        # SDK's router dispatches by name and answers "method not found" for
        # the ones this POC leaves out (session/load, modes, config options).
        await acp.run_agent(cast("acp.Agent", agent), writer, reader)
    finally:
        await agent.shutdown()
        transport.close()
    return 0


def run(
    *,
    hosting: str | None = None,
    model: str | None = None,
    stdio: tuple[int, int] | None = None,
) -> int:
    """``lop acp``: serve ACP on stdio. Returns the process exit status.

    ``stdio`` is the ``(reader, writer)`` pair :func:`claim_stdio` returned,
    when the caller already claimed it — ``cli.main`` does, straight after the
    parse, so its own startup work (the login-shell PATH probe, migrations)
    already runs with the protocol pipes private. Otherwise it is claimed here,
    before anything else (D4).
    """
    if stdio is None:
        from local_operator.acp_agent.stdio import claim_stdio

        stdio = claim_stdio()
    reader_fd, writer_fd = stdio
    try:
        import acp  # noqa: F401 — the presence check for the extra
    except ImportError:
        print(_missing_extra_message(), file=sys.stderr)
        return 1
    # The client is the user interface (D10): no desktop banners from here.
    os.environ.setdefault("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    try:
        return asyncio.run(_serve(reader_fd, writer_fd, hosting=hosting, model=model))
    except KeyboardInterrupt:
        return 130
