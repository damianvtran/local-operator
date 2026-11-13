"""The local server pins the wire clients below INFO, like every other entry point.

WHY (security round 2 on #3587, S-r2-1). httpx logs ``HTTP Request: <full
url>`` at INFO, and ``configure_console_logging`` sets the wire clients to the
ROOT level, so a server started with ``LOG_LEVEL=INFO`` printed one line per
outbound request, URL and all. The CLI, session host, file logging, relay and
mobile paths already call ``quiet_wire_clients()``; the server's lifespan did
not, though it builds ``TeamBridgeClient`` and the provider clients.

THE RIG enters the real ``lifespan`` and stops it right after the pin: the rest
of startup (credential, config and registry managers) is not what this pins,
and running it would make the test about everything else.
"""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI

from local_operator import logger as lo_logger
from local_operator.serve_logging import ServeLogHandler
from local_operator.server import app as app_module


class _StopAfterLoggingSetup(Exception):
    """Raised from the pin so the lifespan goes no further than logging setup."""


@pytest.mark.asyncio
async def test_the_server_lifespan_quiets_httpx_at_log_level_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    saved_clients = {name: logging.getLogger(name).level for name in lo_logger._CHATTY_WIRE_CLIENTS}
    saved_uvicorn = {
        name: (
            list(logging.getLogger(name).handlers),
            logging.getLogger(name).level,
            logging.getLogger(name).propagate,
        )
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access")
    }
    calls: list[str] = []
    configure = app_module.configure_serve_logging
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    # Begin permissive so the assertion proves the lifespan pins the clients,
    # rather than inheriting a WARNING filter installed by an earlier test.
    for name in saved_clients:
        logging.getLogger(name).setLevel(logging.INFO)

    def serve_logging() -> ServeLogHandler:
        calls.append("serve")
        return configure()

    def quiet(*args: object, **kwargs: object) -> None:
        calls.append("quiet")
        lo_logger.quiet_wire_clients()
        raise _StopAfterLoggingSetup

    monkeypatch.setattr(app_module, "configure_serve_logging", serve_logging)
    monkeypatch.setattr(app_module, "quiet_wire_clients", quiet)
    try:
        with pytest.raises(_StopAfterLoggingSetup):
            async with app_module.lifespan(FastAPI()):
                pass  # pragma: no cover - the pin stops startup first
        # Exercise the real nonblocking sink setup before the policy pin, through
        # the public lifespan wrapper that delegates startup to _lifespan.
        assert calls == ["serve", "quiet"]
        assert root.level == logging.INFO
        assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
        assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)
    finally:
        for name, level in saved_clients.items():
            logging.getLogger(name).setLevel(level)
        for open_handler in list(root.handlers):
            if open_handler not in saved_handlers:
                assert isinstance(open_handler, ServeLogHandler)
                open_handler.close()
                open_handler._worker.join(2)
                assert not open_handler._worker.is_alive()
        root.handlers = saved_handlers
        root.setLevel(saved_level)
        for name, (handlers, level, propagate) in saved_uvicorn.items():
            named = logging.getLogger(name)
            named.handlers, named.level, named.propagate = handlers, level, propagate
