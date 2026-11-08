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
    calls: list[str] = []

    def console(*args: object, **kwargs: object) -> None:
        calls.append("console")
        # The operator's LOG_LEVEL=INFO: the level that let httpx's line through.
        lo_logger.configure_console_logging(level=logging.INFO)

    def quiet(*args: object, **kwargs: object) -> None:
        calls.append("quiet")
        lo_logger.quiet_wire_clients()
        raise _StopAfterLoggingSetup

    monkeypatch.setattr(app_module, "configure_console_logging", console)
    monkeypatch.setattr(app_module, "quiet_wire_clients", quiet)
    try:
        with pytest.raises(_StopAfterLoggingSetup):
            async with app_module.lifespan(FastAPI()):
                pass  # pragma: no cover - the pin stops startup first
        # ORDER matters: the console configurator resets the wire clients to the
        # root level, so a pin that ran before it would be undone.
        assert calls == ["console", "quiet"]
        assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
        assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)
    finally:
        for name, level in saved_clients.items():
            logging.getLogger(name).setLevel(level)
        for open_handler in list(root.handlers):
            open_handler.close()
            root.removeHandler(open_handler)
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)
