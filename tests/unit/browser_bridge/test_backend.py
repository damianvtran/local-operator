from __future__ import annotations

import os
from pathlib import Path

import pytest

from local_operator.browser_bridge import state
from local_operator.browser_bridge.backend import (
    BridgeClient,
    BridgeError,
    BridgeUnreachable,
    format_error,
)
from local_operator.browser_bridge.protocol import PROTO_VERSION, ErrorCode


def publish(tmp_path: Path) -> state.BridgeState:
    current = state.BridgeState(
        pid=os.getpid(),
        port=4099,
        session_key="s" * 32,
        proto=PROTO_VERSION,
        extension_connected=True,
        paired=True,
    )
    state.publish(current, tmp_path)
    return current


@pytest.mark.parametrize(
    ("code", "fragment"),
    [
        (ErrorCode.EXTENSION_DISCONNECTED, "extension not connected"),
        (ErrorCode.NOT_PAIRED, "lop browser pair"),
        (ErrorCode.TAB_CLOSED, "Use 'open'"),
        (ErrorCode.NAV_FAILED, "navigation failed"),
        (ErrorCode.NAV_TIMEOUT, "navigation did not complete"),
        (ErrorCode.ELEMENT_NOT_FOUND, "new snapshot"),
        (ErrorCode.ORIGIN_DENIED, "Do not retry"),
        (ErrorCode.ORIGIN_PROMPT_PENDING, "waiting for the user"),
        (ErrorCode.DEBUGGER_CONFLICT, "close DevTools"),
        (ErrorCode.BUSY, "busy"),
        (ErrorCode.PROTO_MISMATCH, "protocol mismatch"),
        (ErrorCode.INTERNAL, "browser bridge error"),
    ],
)
def test_error_taxonomy(code: ErrorCode, fragment: str) -> None:
    error = BridgeError(code, "detail", {"origin": "https://example.com"})
    assert fragment in format_error(error, action="goto", surface="bridge:1:n")


@pytest.mark.asyncio
async def test_missing_state_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(BridgeUnreachable, match="lop browser status"):
        await BridgeClient(tmp_path).call("status", {})
