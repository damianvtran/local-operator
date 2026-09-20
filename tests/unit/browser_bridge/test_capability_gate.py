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

from local_operator.browser_bridge import protocol
from local_operator.browser_bridge import state as state_store
from local_operator.browser_bridge.daemon import BridgeService
from local_operator.browser_bridge.protocol import (
    CAPABILITY_GATED_METHODS,
    CAPABILITY_MIN_EXTENSION_VERSION,
    CAPABILITY_SWITCH_LABEL,
    CAPABILITY_SWITCH_PERMISSION,
    COMMAND_TIMEOUTS,
    PROTO_VERSION,
    Capabilities,
    CapabilitySwitches,
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


@pytest.mark.asyncio
async def test_a_switched_off_method_is_refused_before_it_is_sent(tmp_path: Path) -> None:
    """The operator's consent, enforced at the same place the advertisement is.

    A session that got past the file check — a record read a moment before the
    switch was flipped — must still not reach a capability the operator has turned
    off, and the refusal has to carry the three-state payload so the copy can name
    the switch rather than a version.
    """
    service = await _live(BridgeService(root=tmp_path))
    sent: list[dict[str, Any]] = []

    async def send(payload: dict[str, Any], wire: Any = None) -> None:
        sent.append(payload)

    service.link.send = send  # type: ignore[method-assign]
    service.link.capabilities = ["upload"]
    service.link.disabled_capabilities = ["download"]

    body = await _rpc(service, {"id": "r-1", "method": "download", "params": {"tab": "bridge:1:n"}})
    assert body["ok"] is False
    assert body["error"]["code"] == ErrorCode.CAPABILITY_UNSUPPORTED.value
    assert body["error"]["data"]["method"] == "download"
    assert body["error"]["data"]["disabled"] == ["download"]
    assert body["error"]["data"]["switches_known"] is True
    assert sent == [], "a switched-off capability must not reach the peer either"


@pytest.mark.asyncio
async def test_the_switch_answer_is_published_with_the_advertisement(tmp_path: Path) -> None:
    """The harness decides from the FILE, so both answers must land in it together."""
    service = await _live(BridgeService(root=tmp_path))
    service.link.capabilities = ["download", "upload"]
    service.link.disabled_capabilities = ["upload"]
    service.publish()

    assert service.state.capabilities == ["download", "upload"]
    assert service.state.disabled_capabilities == ["upload"]
    assert service.state.switches_known is True, "the daemon's own stamp, not the peer's"


def test_a_record_from_a_daemon_without_switches_reads_as_no_answer() -> None:
    """Absent keys must not read as "the operator enabled everything".

    Two causes share the empty list — a daemon that predates the switches and one
    whose extension reported none — and only the stamp separates them. The
    conservative reading is "nobody told us", which is why the fields default to
    empty/false rather than to an empty-allowed state.
    """
    old = {
        "pid": 1,
        "port": 4099,
        "session_key": "k" * 32,
        "proto": PROTO_VERSION,
        "extension_connected": True,
        "extension_version": "0.1.18",
        "capabilities": ["upload"],
        "capabilities_known": True,
    }
    record = state_store.BridgeState.model_validate(old)
    assert record.disabled_capabilities == []
    assert record.switches_known is False


def test_a_lockdown_field_on_the_capabilities_event_would_close_an_old_daemon() -> None:
    """Why the switch answer is a SECOND event and not a field on the first.

    Every envelope is ``extra="forbid"``, so adding `disabled` to `capabilities`
    would be a 4001 close (or a dropped frame, depending on the peer) on every
    already-released daemon — the failure a new field on `Hello` would cause, and
    the reason capability travels as an event at all. This test fails the day
    someone "simplifies" the two frames back into one.
    """
    with pytest.raises(ValidationError):
        Capabilities.model_validate(
            {"event": "capabilities", "methods": [], "version": "0.1.19", "disabled": ["upload"]}
        )
    # The sibling event accepts exactly the shape the extension sends, and rejects
    # anything else — so a malformed frame is DROPPED by the daemon rather than
    # allowed to blank a working answer (daemon.py's frame handler).
    switches = CapabilitySwitches.model_validate(
        {"event": "capability_switches", "disabled": ["download"], "version": "0.1.19"}
    )
    assert switches.disabled == ["download"]
    with pytest.raises(ValidationError):
        CapabilitySwitches.model_validate(
            {"event": "capability_switches", "disabled": ["download"], "methods": []}
        )


@pytest.mark.asyncio
async def test_a_switched_on_method_still_reaches_the_peer(tmp_path: Path) -> None:
    """The inverse, so a gate that refuses everything cannot pass this file."""
    service = await _live(BridgeService(root=tmp_path))
    sent: list[dict[str, Any]] = []

    async def send(payload: dict[str, Any], wire: Any = None) -> None:
        sent.append(payload)
        request = Request.model_validate(payload)
        future = service.link.pending.get(request.id)
        if future and not future.done():
            future.set_result(Response(id=request.id, ok=True, result={"armed": True, "files": []}))

    service.link.send = send  # type: ignore[method-assign]
    service.link.capabilities = ["download", "upload"]
    service.link.disabled_capabilities = ["upload"]

    body = await _rpc(service, {"id": "r-1", "method": "download", "params": {"tab": "bridge:1:n"}})
    assert body["ok"] is True
    assert [item["method"] for item in sent] == ["download"]


@pytest.mark.asyncio
async def test_download_with_the_switch_off_is_refused_before_any_socket_call(
    tmp_path: Path,
) -> None:
    """The tool's own gate, for the state the daemon will not be asked about.

    This is the path the model actually sees: the record is read from the FILE, so
    the refusal costs nothing and cannot race the daemon's own check. The copy has
    to name the switch and its location — sending this user to an update would be a
    remedy that cannot work, since their build already serves the method.
    """
    from local_operator.browser_bridge.backend import (
        HostCapabilities,
        capability_refusal,
        format_error,
    )

    caps = HostCapabilities(
        methods=("upload",),
        version="0.1.19",
        capabilities_known=True,
        disabled=("download",),
        switches_known=True,
    )
    copy = format_error(
        capability_refusal("download", host="extension", capabilities=caps),
        action="download",
        host="extension",
    )
    assert "switched off" in copy
    assert "Allow downloads" in copy
    assert "options page" in copy
    assert "No update is involved" in copy
    # The permission mechanics travel with the refusal (round-1 D5): turning the
    # switch on asks Chrome for a permission, and an agent that tells the user to
    # turn it on without saying a browser prompt will appear has sent them into a
    # dialog nothing warned them about.
    assert "'downloads' permission" in copy
    assert "Toolbar icon" in copy or "toolbar icon" in copy


def test_the_extension_can_serve_download_and_only_when_switched_on() -> None:
    """The amendment to E1x: `download` IS servable now, behind the operator's switch.

    E1x measured that a tab-scoped `chrome.debugger` session cannot put a download
    where the harness chooses (Chrome 153.0.8010.53), which is why the extension
    takes the optional `downloads` permission and lets Chrome write into the
    user's own download directory first. The retired ``EXTENSION_CANNOT_SERVE``
    set is asserted ABSENT rather than left implied: its only member is servable
    now, and a constant claiming "no build can" is a lie the refusal copy would
    act on (it would send the user away from an update that fixes it).
    """
    assert not hasattr(protocol, "EXTENSION_CANNOT_SERVE")
    assert CAPABILITY_MIN_EXTENSION_VERSION["download"] == "0.1.19"
    assert "download" in COMMAND_TIMEOUTS, "it is still protocol vocabulary"
    assert "download" in CAPABILITY_GATED_METHODS
    # The switch, its label and the permission it needs are the three facts the
    # refusal copy and the options page both read; the labels are generated into
    # the extension, so a rename here changes the words the user sees there.
    assert CAPABILITY_SWITCH_LABEL["download"] == "Allow downloads"
    assert CAPABILITY_SWITCH_LABEL["upload"] == "Allow uploads"
    assert CAPABILITY_SWITCH_PERMISSION["download"] == "downloads"
    assert "upload" not in CAPABILITY_SWITCH_PERMISSION, "uploads need no permission"


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


def test_a_consent_refusal_from_the_peer_names_the_switch_and_not_the_browser() -> None:
    """The extension's OWN last gate, rendered where it belongs (round-1 R4).

    `requireConsent` throws `capability_unsupported` at execution time, so the
    payload carries no advertisement — no `extension_version`, no `capabilities`,
    no `disabled` list, because the refusal never reached the record. An empty
    payload used to fall through to the "no browser is attached" branch: the model
    was told the browser was missing and no method was named at all, so the user was
    sent to look at their connection instead of at the switch that was off.
    """
    from local_operator.browser_bridge.backend import BridgeError, format_error

    copy = format_error(
        BridgeError(
            code=ErrorCode.CAPABILITY_UNSUPPORTED,
            message="'download' is switched off in the browser extension",
            data={"method": "download", "disabled_by_operator": True},
        ),
        action="download",
        host="extension",
    )
    assert "no browser is attached" not in copy
    assert "'download' is switched off" in copy
    assert "Allow downloads" in copy
    assert "options page" in copy
    assert "'downloads' permission" in copy
