"""The capability gate: who may be SENT `download`/`upload`, and what a refusal says.

The advertisement is what keeps `PROTO_VERSION` at 1: capability travels as an
additive event and an additive record field, so an old daemon drops the event
harmlessly and an old harness ignores the key. These tests pin both directions —
including the DROP, because that is the property the compatibility claim rests
on — and the daemon's refusal to send a method the peer did not advertise, which
exists so a pre-feature host produces a typed, actionable answer instead of a
120-second wait on a bare `internal`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from local_operator.browser_bridge import state as state_store
from local_operator.browser_bridge.daemon import BridgeService
from local_operator.browser_bridge.protocol import (
    CAPABILITY_GATED_METHODS,
    COMMAND_TIMEOUTS,
    EXTENSION_CANNOT_SERVE,
    PROTO_VERSION,
    Capabilities,
    ErrorCode,
    Request,
    Response,
)


class _HttpRequest:
    """The minimum ``rpc()`` reads off a Starlette request: a key and a body.

    The same duck-typed double `test_bridge_wedge.py` uses, copied rather than
    imported so this module owns its fixture: the handler touches only
    ``headers["x-bridge-key"]`` and ``await json()`` before dispatch, and the
    ADMISSION checks this module tests live inside `rpc`.
    """

    def __init__(self, service: BridgeService, payload: dict[str, Any]) -> None:
        self.headers = {"x-bridge-key": service.state.session_key}
        self._payload = payload

    async def json(self) -> dict[str, Any]:
        return self._payload


async def _rpc(service: BridgeService, payload: dict[str, Any]) -> dict[str, Any]:
    response = await service.rpc(cast(Any, _HttpRequest(service, payload)))
    return _response_body(response)


async def _live(service: BridgeService) -> BridgeService:
    """A PAIRED, PROVEN link, so `rpc` reaches its admission checks.

    Paired through the daemon's own flow — mint a code, submit it — rather than by
    writing the allow-list by hand, because `_live_pairing_matches` re-reads the
    ON-DISK record and a hand-made link would be severed as `not_paired` before
    the capability gate ran. `proven` additionally needs a recent frame stamp: a
    peer that completed `hello` and then went quiet is deliberately not treated
    as a healthy idle one.
    """
    import time as _time

    from local_operator.browser_bridge.daemon import _pending_entries
    from local_operator.browser_bridge.protocol import PairRequest

    link = service.link
    link.extension_id = "e" * 32
    link.paired = True
    link.websocket = cast(Any, object())
    link.last_frame_at = _time.monotonic()
    service._ensure_pending(link.extension_id, "capability-gate test")
    code = str(_pending_entries(service.root)[link.extension_id]["code"])
    result = await service._try_pair(PairRequest(code=code), link)
    assert result.ok, result.message
    return service


def _response_body(response: Any) -> dict[str, Any]:
    return json.loads(bytes(response.body))


@pytest.mark.asyncio
async def test_a_method_the_peer_did_not_advertise_is_never_sent(tmp_path: Path) -> None:
    """The load-bearing half: the peer must not be asked and then time out."""
    service = await _live(BridgeService(root=tmp_path))
    sent: list[dict[str, Any]] = []

    async def send(payload: dict[str, Any], wire: Any = None) -> None:
        sent.append(payload)

    service.link.send = send  # type: ignore[method-assign]
    service.link.capabilities = ["upload"]

    body = await _rpc(service, {"id": "r-1", "method": "download", "params": {"tab": "bridge:1:n"}})
    assert body["ok"] is False
    assert body["error"]["code"] == ErrorCode.CAPABILITY_UNSUPPORTED.value
    assert body["error"]["data"]["method"] == "download"
    assert body["error"]["data"]["advertised"] == ["upload"]
    assert sent == [], "an unadvertised method must not reach the peer"


@pytest.mark.asyncio
async def test_an_advertised_method_is_forwarded(tmp_path: Path) -> None:
    service = await _live(BridgeService(root=tmp_path))
    sent: list[dict[str, Any]] = []

    async def send(payload: dict[str, Any], wire: Any = None) -> None:
        sent.append(payload)
        request = Request.model_validate(payload)
        future = service.link.pending.get(request.id)
        if future and not future.done():
            future.set_result(
                Response(id=request.id, ok=True, result={"inputs": ["#f"], "accepted": []})
            )

    service.link.send = send  # type: ignore[method-assign]
    service.link.capabilities = ["upload", "download"]

    body = await _rpc(
        service,
        {
            "id": "r-2",
            "method": "upload",
            "params": {"tab": "bridge:1:n", "selector": "#f", "paths": ["/tmp/a.pdf"]},
        },
    )
    assert body["ok"] is True
    assert body["result"] == {"inputs": ["#f"], "accepted": []}
    assert sent and sent[0]["params"]["selector"] == "#f"


@pytest.mark.asyncio
async def test_methods_that_predate_the_advertisement_are_not_gated(tmp_path: Path) -> None:
    """A pre-feature peer still serves `read`, and refusing it would break a host
    that works today."""
    assert "read" not in CAPABILITY_GATED_METHODS
    assert "open" not in CAPABILITY_GATED_METHODS


def test_the_capabilities_event_is_dropped_harmlessly_by_an_old_peer() -> None:
    """C2's whole claim: an old daemon must not choke on the new frame.

    An old daemon's receive loop validates every non-event frame as a `Response`;
    this asserts the event frame fails that validation, i.e. is skipped, rather
    than being mistaken for an answer to some pending request.
    """
    frame = {"event": "capabilities", "methods": ["upload"], "version": "0.1.18"}
    with pytest.raises(ValidationError):
        Response.model_validate(frame)
    parsed = Capabilities.model_validate(frame)
    assert parsed.methods == ["upload"] and parsed.version == "0.1.18"


def test_the_record_field_is_additive_in_both_directions() -> None:
    """`extra="ignore"` is what makes the field safe on both sides of a skew."""
    record = {
        "pid": 1,
        "port": 4099,
        "session_key": "k" * 32,
        "proto": PROTO_VERSION,
        "extension_connected": True,
        "something_an_older_daemon_wrote": True,
    }
    parsed = state_store.BridgeState.model_validate(record)
    assert parsed.capabilities == []
    # And a record carrying it round-trips.
    record["capabilities"] = ["upload"]
    assert state_store.BridgeState.model_validate(record).capabilities == ["upload"]


def test_the_daemon_stamps_the_record_with_the_advertisement_it_knows(tmp_path: Path) -> None:
    """R4: an empty `capabilities` list has TWO causes with opposite remedies.

    A record with no stamp was written by a bridge that predates the
    advertisement, so its empty list says nothing about the extension and the
    remedy is a restart, not an extension toggle (design §6.4). A daemon at or
    after the advertisement always stamps itself, which is what lets the refusal
    tell the two apart from the file alone.
    """
    old = {
        "pid": 1,
        "port": 4099,
        "session_key": "k" * 32,
        "proto": PROTO_VERSION,
        "extension_connected": True,
        "extension_version": "0.1.18",
    }
    assert state_store.BridgeState.model_validate(old).capabilities_known is False
    assert BridgeService(root=tmp_path).state.capabilities_known is True


def test_no_extension_build_can_serve_download() -> None:
    """The measured fact the refusal copy depends on (E1x, Chrome 153.0.8010.53)."""
    assert EXTENSION_CANNOT_SERVE == frozenset({"download"})
    assert "download" in COMMAND_TIMEOUTS, "it is still protocol vocabulary"
    assert "download" in CAPABILITY_GATED_METHODS


def test_a_download_may_extend_its_own_budget_and_nothing_else_may() -> None:
    service = BridgeService(root=Path("/tmp"))
    download = Request(id="r-1", method="download", params={"tab": "bridge:1:n", "timeout_s": 600})
    assert service._command_budget(download) == 600
    # Clamped to the shared ceiling, and a bogus value is ignored rather than
    # trusted: a caller must not be able to extend its own budget by inventing one.
    over = Request(id="r-2", method="download", params={"timeout_s": 10_000})
    assert service._command_budget(over) == pytest.approx(600)
    for bad in ("600", True, -5, None):
        request = Request(id="r-3", method="download", params={"timeout_s": bad})
        assert service._command_budget(request) == COMMAND_TIMEOUTS["download"]
    # No other method reads the key at all.
    upload = Request(id="r-4", method="upload", params={"timeout_s": 600})
    assert service._command_budget(upload) == COMMAND_TIMEOUTS["upload"]
