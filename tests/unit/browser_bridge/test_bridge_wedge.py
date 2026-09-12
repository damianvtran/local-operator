"""Daemon-side guards for the browser-bridge wedge (design record §8.1).

The incident these pin: an extension worker whose module-global serialized
chains parked every later command forever, while `/health` and `lop browser
status` kept answering "extension connected: yes" and `owner_recover` — the one
command whose job is recovery — reported a version mismatch for a timeout.

The tests are deliberately STRUCTURAL (a request that never reaches the wire, a
future that is answered, a lock that is free, a close code that was sent) rather
than timing assertions: see AGENTS.md, "Timing, flakes, and how to assert that
something is fast". Where a row cannot fail on the pre-fix tree the docstring
says so instead of implying a guard that does not exist.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, cast

import pytest
from starlette.testclient import TestClient

from local_operator.browser_bridge import daemon as daemon_module
from local_operator.browser_bridge import resources
from local_operator.browser_bridge.backend import BridgeError, format_error
from local_operator.browser_bridge.daemon import (
    LINK_SILENCE_TIMEOUT_S,
    BridgeService,
    create_app,
)
from local_operator.browser_bridge.protocol import (
    PROTO_VERSION,
    ErrorCode,
    ErrorDetail,
    Request,
    Response,
)

EXTENSION_ID = "a" * 32
ORIGIN = f"chrome-extension://{EXTENSION_ID}"


class _RecordingSocket:
    """Stand-in for the extension leg's websocket: records close codes."""

    def __init__(self) -> None:
        self.closed: list[int | None] = []

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.closed.append(code)


def _connected(service: BridgeService, *, silent_for: float = 0.0) -> _RecordingSocket:
    """Give ``service`` an extension socket that spoke ``silent_for`` seconds ago."""
    socket = _RecordingSocket()
    service.link.websocket = socket  # type: ignore[assignment]
    service.link.last_frame_at = time.monotonic() - silent_for
    return socket


def _health_payload(tmp_path: Path) -> dict[str, Any]:
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        return client.get("/health").json()


# --- D1 -------------------------------------------------------------------


def test_mute_extension_is_not_advertised_as_connected(tmp_path: Path) -> None:
    """A socket that has said nothing for LINK_SILENCE_TIMEOUT_S is not
    "connected".

    Pre-fix `/health` derived `extension_connected` from
    `link.websocket is not None`, which is true whenever TCP is up and the peer
    has not closed — measured on the real incident at 42/42 samples "connected,
    paired" across an 84 s window in which every command hung.
    """
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        service = app.state.bridge
        with client.websocket_connect("/extension", headers={"origin": ORIGIN}) as socket:
            socket.send_json(
                {
                    "event": "hello",
                    "proto": PROTO_VERSION,
                    "token": "",
                    "extension_version": "0.1.0",
                    "browser": "Chrome/153",
                }
            )
            socket.receive_json()
            assert client.get("/health").json()["extension_connected"] is True
            service.link.last_frame_at = time.monotonic() - (LINK_SILENCE_TIMEOUT_S + 1.0)
            health = client.get("/health").json()
            assert health["extension_connected"] is False
            assert health["extension_unresponsive"] is True
            assert health["link_silent_s"] > LINK_SILENCE_TIMEOUT_S


# --- D2 -------------------------------------------------------------------


def test_command_against_a_silent_link_fails_fast_with_a_typed_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command against a known-mute link is refused BEFORE dispatch.

    The structural half is `sent == []`: the request never reaches the
    extension, so "it failed fast" is a fact about where the code returned
    rather than a measurement of how long it took.
    """
    monkeypatch.setitem(daemon_module.COMMAND_TIMEOUTS, "read", 0.05)
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        service = app.state.bridge
        with client.websocket_connect("/extension", headers={"origin": ORIGIN}) as socket:
            socket.send_json(
                {
                    "event": "hello",
                    "proto": PROTO_VERSION,
                    "token": "",
                    "extension_version": "0.1.0",
                    "browser": "Chrome/153",
                }
            )
            socket.receive_json()
            sent: list[dict[str, Any]] = []

            async def record(payload: dict[str, Any]) -> None:
                sent.append(payload)

            service.link.send = record  # type: ignore[method-assign]
            service.link.last_frame_at = time.monotonic() - (LINK_SILENCE_TIMEOUT_S + 1.0)
            response = client.post(
                "/rpc",
                headers={"X-Bridge-Key": app.state.bridge.state.session_key},
                json={"id": "r-mute", "method": "read", "params": {"tab": "bridge:1:n"}},
            )
            body = response.json()
            assert body["ok"] is False
            assert body["error"]["code"] == ErrorCode.EXTENSION_UNRESPONSIVE.value
            assert sent == [], "the request reached the extension"
            assert "r-mute" not in service.link.pending


# --- D3 -------------------------------------------------------------------


def test_silence_detector_does_not_fire_on_a_pinging_idle_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both directions in ONE test, so a detector that never fires cannot pass.

    Negative half: a peer that keeps answering the daemon's pings is never torn
    down, even though it sends nothing else. Positive half: stop answering and
    the detector fires. `LINK_SILENCE_TIMEOUT_S` is shrunk so the positive half
    does not cost a real 50 s.
    """
    monkeypatch.setattr(daemon_module, "PING_INTERVAL_S", 0.05)
    monkeypatch.setattr(daemon_module, "LINK_SILENCE_TIMEOUT_S", 0.25)
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        service = app.state.bridge
        with client.websocket_connect("/extension", headers={"origin": ORIGIN}) as socket:
            socket.send_json(
                {
                    "event": "hello",
                    "proto": PROTO_VERSION,
                    "token": "",
                    "extension_version": "0.1.0",
                    "browser": "Chrome/153",
                }
            )
            socket.receive_json()
            first_socket = service.link.websocket
            # Negative: pong for several silence-windows' worth of ping ticks.
            for _ in range(4):
                socket.send_json({"event": "pong"})
                time.sleep(0.1)
            assert service.link.proven is True
            assert service.link.websocket is first_socket, "a pinging peer was torn down"

            # Positive: go silent. The supervised ping loop tears the link down.
            deadline = time.monotonic() + 5.0
            while service.link.websocket is not None and time.monotonic() < deadline:
                time.sleep(0.05)
            assert service.link.websocket is None, "a silent peer was never dropped"


# --- D4 -------------------------------------------------------------------


def test_awaiting_origin_wait_is_not_torn_down_by_the_detector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The false-positive guard the design's riskiest claim rests on.

    A command blocked on a HUMAN for longer than the silence deadline must not
    lose its link: the extension keeps ponging while the popup waits, so
    `silent_for()` stays near zero. Rejected design alternative — tying the
    threshold to COMMAND_TIMEOUTS — would have to be ~95 s here, which is why
    the detector keys on frames rather than on command duration.

    CHARACTERISATION, not a pre-fix guard: the pre-fix tree has no detector at
    all, so this passes there. Its proof-of-failure is to disable the frame
    timestamp (the `last_frame_at` write in the receive loop) on the FIXED tree
    and watch it tear the link down mid-prompt.
    """
    monkeypatch.setattr(daemon_module, "PING_INTERVAL_S", 0.05)
    monkeypatch.setattr(daemon_module, "LINK_SILENCE_TIMEOUT_S", 0.2)
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        service = app.state.bridge
        with client.websocket_connect("/extension", headers={"origin": ORIGIN}) as socket:
            socket.send_json(
                {
                    "event": "hello",
                    "proto": PROTO_VERSION,
                    "token": "",
                    "extension_version": "0.1.0",
                    "browser": "Chrome/153",
                }
            )
            socket.receive_json()
            socket.send_json(
                {"event": "awaiting_origin", "id": "r-prompt", "origin": "https://slow.example"}
            )
            first_socket = service.link.websocket
            # Keep ponging for several times the silence deadline: the command is
            # "waiting on a human" for far longer than the detector's window. The
            # cadence must be comfortably INSIDE the deadline (0.04 s against
            # 0.2 s) or the test measures its own scheduling jitter rather than
            # the detector — a pong that lands late tears the link down and the
            # failure reads as a false positive in the code under test.
            for _ in range(15):
                socket.send_json({"event": "pong"})
                time.sleep(0.04)
            assert service.link.websocket is first_socket
            assert service.link.awaiting_origin == {"r-prompt": "https://slow.example"}


# --- D5 / D6 --------------------------------------------------------------


@pytest.mark.asyncio
async def test_command_timeout_against_a_silent_link_is_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Promotion rule: a dead command PLUS a silent link corroborates a dead
    peer, and the session hears `extension_unresponsive` rather than a generic
    `internal` it would read as a page-level problem.

    Pre-fix the timeout branch always returned NAV_TIMEOUT/INTERNAL.
    """
    monkeypatch.setitem(daemon_module.COMMAND_TIMEOUTS, "read", 0.05)
    monkeypatch.setattr(daemon_module, "PING_INTERVAL_S", 0.01)
    service = BridgeService(root=tmp_path)
    # Silent for 1 s: proven (well inside LINK_SILENCE_TIMEOUT_S, so the rpc gate
    # lets the command through) but already past one ping interval.
    _connected(service, silent_for=1.0)

    async def silent(payload: dict[str, Any]) -> None:
        # Delivered and accepted — the send RETURNS, it does not park — and then
        # no response ever arrives. Parking here instead would exercise the send
        # deadline (D7), not the promotion rule.
        return None

    service.link.send = silent  # type: ignore[method-assign]
    response = await service._dispatch_serialized(
        Request(id="r-dead", method="read", params={"tab": "bridge:1:n"})
    )
    body = bytes(response.body).decode()
    assert ErrorCode.EXTENSION_UNRESPONSIVE.value in body
    assert '"phase":"response"' in body.replace(" ", "")


@pytest.mark.asyncio
async def test_command_timeout_against_a_live_link_is_still_nav_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The "verify it still refuses where it should" half: proves D5 narrowed
    the timeout branch rather than replacing it.

    A slow page is slow INSIDE the extension, which is ponging normally, so the
    promotion rule must not fire and `open` must still report nav_timeout with
    the budget it exhausted. Fails if the promotion is unconditional.
    """
    monkeypatch.setitem(daemon_module.COMMAND_TIMEOUTS, "open", 0.05)
    service = BridgeService(root=tmp_path)
    socket = _connected(service, silent_for=0.0)

    async def ponging(payload: dict[str, Any]) -> None:
        # The extension received the command and is working on it. Its pongs
        # keep arriving for the whole budget — that is what a slow page looks
        # like on the wire. The send RETURNS; only the answer is missing.
        service.link.last_frame_at = time.monotonic()
        return None

    service.link.send = ponging  # type: ignore[method-assign]
    response = await service._dispatch_serialized(
        Request(id="r-slow", method="open", params={"url": "https://slow.example"})
    )
    body = bytes(response.body).decode().replace(" ", "")
    assert ErrorCode.NAV_TIMEOUT.value in body
    assert '"timeout_s":0.05' in body
    assert ErrorCode.EXTENSION_UNRESPONSIVE.value not in body
    assert service.link.websocket is socket, "a healthy-but-slow link was torn down"


# --- D7 -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blocked_send_does_not_outlive_its_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An un-writable socket must not hold the per-tab lock past its deadline.

    Pre-fix `_dispatch_locked` awaited `link.send()` unbounded while holding the
    lock key taken by `_dispatch_serialized`, so a stuck send (a peer that
    stopped draining its TCP receive buffer, or a stuck `send_lock` holder)
    queued every other command for that key behind it forever.

    The record phrased the structural half as "`service._tab_locks` is empty
    afterwards"; the per-TAB key is deliberately RETAINED until `close`
    (evicting it between commands would let two sessions interleave on one tab),
    so the invariant asserted here is the one that actually matters: the lock is
    FREE and the NEXT command on that key answers.
    """
    monkeypatch.setattr(daemon_module, "LINK_SEND_TIMEOUT_S", 0.05)
    monkeypatch.setattr(daemon_module, "PING_INTERVAL_S", 0.01)
    monkeypatch.setitem(daemon_module.COMMAND_TIMEOUTS, "read", 0.05)
    service = BridgeService(root=tmp_path)
    _connected(service, silent_for=0.0)

    async def blocked(payload: dict[str, Any]) -> None:
        await asyncio.sleep(3600)

    service.link.send = blocked  # type: ignore[method-assign]
    request = Request(id="r-send", method="read", params={"tab": "bridge:7:abc"})
    # BOUNDED, and that bound is part of the guard rather than test hygiene: the
    # stub under test parks for an hour, so a regression that drops the send
    # deadline presents as a HUNG test (RC=124 under `timeout`, and under
    # `-n auto --dist worksteal` a killed worker carrying unrelated tests, the
    # failure mode tests/e2e/watchdog.py exists to avoid) instead of a red one
    # (review R1-6, reproduced). 2 s is well above LINK_SEND_TIMEOUT_S (0.05)
    # and the command budget, so a correct implementation can never reach it,
    # while a regression fails in seconds with the defect named.
    try:
        response = await asyncio.wait_for(service._dispatch_serialized(request), timeout=2.0)
    except asyncio.TimeoutError:  # pragma: no cover - only on a regression
        raise AssertionError(
            "a blocked link.send() outlived its deadline: the per-tab lock is held forever"
        ) from None
    body = bytes(response.body).decode().replace(" ", "")
    assert ErrorCode.EXTENSION_UNRESPONSIVE.value in body
    assert '"phase":"send"' in body
    assert not service._tab_locks["bridge:7:abc"].locked()
    # Latched even though the peer's silence was ~0 when the send deadline
    # fired: "we severed an attached link" is a fact of its own, not a
    # function of how long the peer had been quiet.
    assert service.link.dropped_unproven() is True

    # The next command on the same key is served, not queued behind a corpse.
    async def answer(payload: dict[str, Any]) -> None:
        request_model = Request.model_validate(payload)
        future = service.link.pending.get(request_model.id)
        if future and not future.done():
            future.set_result(Response(id=request_model.id, ok=True, result={"text": "ok"}))

    _connected(service, silent_for=0.0)
    service.link.send = answer  # type: ignore[method-assign]
    try:
        second = await asyncio.wait_for(
            service._dispatch_serialized(
                Request(id="r-next", method="read", params={"tab": "bridge:7:abc"})
            ),
            timeout=2.0,
        )
    except asyncio.TimeoutError:  # pragma: no cover - only on a regression
        raise AssertionError("the lock outlived its holder: the next command never ran") from None
    assert '"ok":true' in bytes(second.body).decode().replace(" ", "")


# --- D8 -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_teardown_closes_with_4000_and_clears_pending(
    tmp_path: Path,
) -> None:
    """The teardown's whole wire-compatibility story is the close code.

    `worker.ts` handles 4000 explicitly (suppress the connState write, reset
    state, schedule the ~1 s fast-path reconnect), so an ALREADY-INSTALLED
    extension recovers from a daemon-initiated drop without a protocol bump. A
    code it did not know would be treated as an ordinary close; a code the peer
    must INTERPRET (the rejected 4005 "recycle") is not a free change.
    """
    service = BridgeService(root=tmp_path)
    socket = _connected(service, silent_for=LINK_SILENCE_TIMEOUT_S + 1.0)
    loop = asyncio.get_running_loop()
    pending: asyncio.Future[Response] = loop.create_future()
    service.link.pending["r-pending"] = pending
    service.link.awaiting_origin["r-pending"] = "https://x.example"
    service.link.note_driven("bridge:1:n", "https://x.example", "X")

    await service._drop_unproven_link("test")

    assert socket.closed == [4000]
    assert pending.done() and isinstance(pending.exception(), RuntimeError)
    assert service.link.pending == {}
    assert service.link.awaiting_origin == {}
    assert service.link.driven == {}
    assert service.link.websocket is None
    assert service.link.proven is False


# --- D9 -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recover_timeout_is_not_reported_as_a_version_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P3, in BOTH directions.

    The daemon attaches `timeout_s` to its own timeout errors and to nothing
    else, so that key is the discriminator between "the extension does not
    implement ownership" (still true of a genuinely old build, whose HANDLERS
    fallthrough returns `internal` with no data) and "the extension did not
    answer in time" — the one that was true. Pre-fix `resources.py` mapped both
    to "requires an updated Local Operator extension", which is a wrong
    diagnosis that costs real time: the reporting session was told to replace a
    current extension while a wedged worker went unmentioned.
    """

    class _FailingBridge:
        def __init__(self, error: BridgeError) -> None:
            self.error = error

        async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            raise self.error

    resource = resources.BrowserResource(tmp_path, tmp_path.name)
    resource.initialize()

    timeout_error = BridgeError(ErrorCode.INTERNAL, "owner_recover timed out", {"timeout_s": 20.0})
    monkeypatch.setattr(resources, "BridgeClient", lambda: _FailingBridge(timeout_error))
    with pytest.raises(BridgeError) as raised:
        await resource.recover()
    assert "updated Local Operator extension" not in str(raised.value)
    assert raised.value.data["timeout_s"] == 20.0

    legacy_error = BridgeError(ErrorCode.INTERNAL, "unknown method owner_recover")
    monkeypatch.setattr(resources, "BridgeClient", lambda: _FailingBridge(legacy_error))
    with pytest.raises(resources.BrowserOwnershipError) as legacy:
        await resource.recover()
    assert "requires an updated Local Operator extension" in str(legacy.value)


# --- D10 ------------------------------------------------------------------


def test_unresponsive_code_round_trips_and_has_client_copy() -> None:
    """The new code must survive the wire model AND the generated TS.

    `gen_ts --check` is the guard that stops Python-only protocol drift: the
    checked-in `protocol.gen.ts` is what the extension compiles against.
    """
    from local_operator.browser_bridge import gen_ts

    assert gen_ts.main(["--check"]) == 0, "protocol.gen.ts is stale"
    frame = Response(
        id="r-1",
        ok=False,
        error=ErrorDetail(code=ErrorCode.EXTENSION_UNRESPONSIVE, message="mute", data={}),
    )
    assert Response.model_validate_json(frame.model_dump_json()) == frame
    # The client copy is the dedicated one, not the generic fallback: it names
    # retry and the manual escape hatch. Reusing EXTENSION_DISCONNECTED's copy
    # would tell the user to open a browser that is already open.
    rendered = format_error(BridgeError(ErrorCode.EXTENSION_UNRESPONSIVE, "mute"), action="read")
    assert "stopped answering" in rendered
    assert "chrome://extensions" in rendered
    assert "no browser is attached" not in rendered


# --- D11 ------------------------------------------------------------------


def test_status_output_distinguishes_absent_from_unresponsive(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`lop browser status` must not say "not currently attached" about a
    browser that IS attached.

    Both states share `extension_connected: false`, so only the new
    `extension_unresponsive` discriminator separates them — and the two lines
    are mutually exclusive, which is what makes a wrong branch visible instead
    of merely incomplete.
    """
    from local_operator import cli
    from local_operator.browser_bridge import install

    def _run(health_extra: dict[str, Any]) -> str:
        monkeypatch.setattr(
            install,
            "status",
            lambda: {
                "installed": True,
                "healthy": True,
                "paired": True,
                "port": 4099,
                "log": "/tmp/synthetic-bridge.log",
                # Hermetic: `stale_heartbeat_age()` would read the discovery file
                # at the DEFAULT root, i.e. the operator's real run dir.
                "health": {
                    "extension_connected": False,
                    "driven_tabs": [],
                    **health_extra,
                },
            },
        )
        monkeypatch.setattr(install, "stale_heartbeat_age", lambda: None)
        assert cli.browser_command(_namespace("status")) == 0
        return capsys.readouterr().out

    unresponsive = _run(
        {"extension_unresponsive": True, "link_attached": True, "link_silent_s": 51.0}
    )
    assert "attached but not answering" in unresponsive
    assert "will drop and re-dial" in unresponsive, "still attached: the drop is ahead, not done"
    assert "not currently attached" not in unresponsive

    # The SAME state one teardown later. This is the case the guide sends the
    # user to `status` for, and it used to print the absent line (design D2 /
    # review R1-5 / QA Q1).
    dropped = _run({"extension_unresponsive": True, "link_attached": False, "link_silent_s": 51.0})
    assert "attached but not answering" in dropped
    assert "dropped the link" in dropped
    assert "will drop" not in dropped, "the past tense must not promise a severing that happened"
    assert "not currently attached" not in dropped

    # A daemon from an earlier head of this change reports the state but not the
    # side of the drop (`link_attached` is additive). Then the tense is
    # unknowable, so no mechanism may be asserted in either direction.
    undecidable = _run({"extension_unresponsive": True, "link_silent_s": 51.0})
    assert "attached but not answering" in undecidable
    assert "will drop" not in undecidable and "dropped the link" not in undecidable
    assert "not currently attached" not in undecidable
    assert "retry once in a few seconds" in undecidable

    absent = _run({})
    assert "not currently attached" in absent
    assert "attached but not answering" not in absent


def _namespace(command: str) -> Any:
    import argparse

    return argparse.Namespace(browser_command=command, repair=False)


# --- Round 1 remediation guards -------------------------------------------
#
# Every test below pins a defect that the round-1 review reproduced against the
# previous head. They are structural (a discriminator that is present, a link
# that is still the same object, a latch that survives a teardown) rather than
# timing assertions, per AGENTS.md.


class _HttpRequest:
    """The minimum ``rpc()`` reads off a Starlette request: a key and a body.

    Built by hand so a test can exercise the HTTP gate's own branches without a
    live port; the headers dict and ``json()`` are exactly the two surfaces
    ``rpc()`` touches before dispatch.
    """

    def __init__(self, service: BridgeService, payload: dict[str, Any]) -> None:
        self.headers = {"x-bridge-key": service.state.session_key}
        self._payload = payload

    async def json(self) -> dict[str, Any]:
        return self._payload


async def _rpc(service: BridgeService, payload: dict[str, Any]) -> Any:
    """Call the HTTP gate with the two attributes it reads off a request.

    The handler touches only ``headers["x-bridge-key"]`` and ``await json()``,
    so a duck-typed double is the honest fixture; the cast is only for pyright,
    which cannot know that a narrower object is sufficient here.
    """
    return await service.rpc(cast(Any, _HttpRequest(service, payload)))


async def _health(service: BridgeService) -> dict[str, Any]:
    """The daemon's own /health payload, without standing up a port."""
    import json as _json

    response = await service.health(None)  # type: ignore[arg-type]
    return _json.loads(bytes(response.body).decode("utf-8"))


# --- R1-1 ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stalled_owner_recover_is_not_reported_as_a_version_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stalled extension must never render as a version mismatch, either.

    `settle.ts`'s `deadline()` is a SECOND producer of a data-bearing INTERNAL:
    it rejects `internal` + `data.stalled` with NO `timeout_s`, and the worker
    passes `error.data` through verbatim. `owner_recover` reaches it through
    `withOwnership` → `withSessionMutation` → `scopes()`, so keying the
    "update your extension" mapping on the ABSENCE of `timeout_s` alone sent the
    recovery command back to the exact P3 misdiagnosis, from inside the incident
    that command exists to survive (review R1-1, reproduced against the previous
    head). The predicate must therefore key on both discriminators, and a new
    `internal` discriminator belongs in it rather than beside it.
    """

    class _FailingBridge:
        def __init__(self, error: BridgeError) -> None:
            self.error = error

        async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            raise self.error

    resource = resources.BrowserResource(tmp_path, tmp_path.name)
    resource.initialize()

    stalled = BridgeError(
        ErrorCode.INTERNAL,
        "chrome.storage.session.set(ownerScopes) did not respond within 5000ms",
        {"stalled": "chrome.storage.session.set(ownerScopes)"},
    )
    monkeypatch.setattr(resources, "BridgeClient", lambda: _FailingBridge(stalled))
    with pytest.raises(BridgeError) as raised:
        await resource.recover()
    assert "updated Local Operator extension" not in str(raised.value)
    assert raised.value.data["stalled"] == "chrome.storage.session.set(ownerScopes)"

    # And the genuine legacy case is still mapped: no discriminator at all.
    legacy = BridgeError(ErrorCode.INTERNAL, "unknown method owner_recover")
    monkeypatch.setattr(resources, "BridgeClient", lambda: _FailingBridge(legacy))
    with pytest.raises(resources.BrowserOwnershipError) as mapped:
        await resource.recover()
    assert "requires an updated Local Operator extension" in str(mapped.value)


# --- R1-2 ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_drifted_ping_tick_does_not_promote_a_healthy_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The promotion rule must carry slack for a dropped or delayed tick.

    A healthy peer speaks only when spoken to, so its silence sawtooths from 0
    to one ping interval plus whatever drift the DAEMON's own loop adds: a tick
    can be dropped, and `_supervise` inserts up to `SUPERVISOR_BACKOFF_CAP_S`
    after a failed one. With zero slack (`silent_for() > PING_INTERVAL_S`) a
    command timing out in that sliver tore down a healthy link — close 4000 and
    every session's pending futures failed, which is the record's own hazard at
    the sibling threshold (review R1-2, reproduced). This is the boundary BELOW
    the slack: 1.2 intervals of silence must NOT corroborate a dead peer.
    """

    monkeypatch.setitem(daemon_module.COMMAND_TIMEOUTS, "read", 0.05)
    monkeypatch.setattr(daemon_module, "PING_INTERVAL_S", 1.0)
    service = BridgeService(root=tmp_path)
    # 1.2 intervals: past a bare interval, inside the 1.5x slack. Proven (far
    # inside LINK_SILENCE_TIMEOUT_S), so the rpc gate lets the command through.
    socket = _connected(service, silent_for=1.2)

    async def silent(payload: dict[str, Any]) -> None:
        return None

    service.link.send = silent  # type: ignore[method-assign]
    response = await service._dispatch_serialized(
        Request(id="r-drift", method="read", params={"tab": "bridge:9:n"})
    )
    body = bytes(response.body).decode().replace(" ", "")
    assert ErrorCode.INTERNAL.value in body
    assert '"timeout_s":0.05' in body
    assert ErrorCode.EXTENSION_UNRESPONSIVE.value not in body
    assert service.link.websocket is socket, "a healthy link with a drifted tick was torn down"


@pytest.mark.asyncio
async def test_a_corroborated_dead_peer_still_promotes_past_the_slack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The boundary ABOVE the slack: the rule must still fire.

    Without this half, clamping the promotion rule to silence only would pass the
    test above and lose the corroboration the record added it for.
    """

    monkeypatch.setitem(daemon_module.COMMAND_TIMEOUTS, "read", 0.05)
    monkeypatch.setattr(daemon_module, "PING_INTERVAL_S", 1.0)
    service = BridgeService(root=tmp_path)
    # 2 intervals of silence: past the 1.5x slack, still proven.
    socket = _connected(service, silent_for=2.0)

    async def silent(payload: dict[str, Any]) -> None:
        return None

    service.link.send = silent  # type: ignore[method-assign]
    response = await service._dispatch_serialized(
        Request(id="r-dead", method="read", params={"tab": "bridge:9:n"})
    )
    body = bytes(response.body).decode().replace(" ", "")
    assert ErrorCode.EXTENSION_UNRESPONSIVE.value in body
    assert '"phase":"response"' in body
    assert socket.closed == [4000], "a corroborated dead peer must be severed"


# --- D2 / R1-5 / Q1 -------------------------------------------------------


@pytest.mark.asyncio
async def test_a_severed_silent_link_stays_honest_until_the_browser_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The honest state must outlive the teardown that answers with it.

    The single upstream defect here: the teardown nulls the socket as part of
    ANSWERING, and `/health` derived the discriminator from the live socket, so
    the "attached but not answering" line was observable for at most one command
    and then fell back to "browser not currently attached; it reconnects when
    opened" — advice to open a browser that is open, told to a user who had just
    been sent to `lop browser status` by the error copy. Both the health payload
    and the NEXT rpc must carry the reason (design D2 / review R1-5 / QA Q1).
    """
    service = BridgeService(root=tmp_path)
    socket = _connected(service, silent_for=LINK_SILENCE_TIMEOUT_S + 3.0)

    await service._drop_unproven_link("test guard")

    assert socket.closed == [4000]
    # The latch survives the teardown, with the silence measured AT the drop
    # rather than collapsing to 0 with the socket.
    assert service.link.recent_drop_silence() > LINK_SILENCE_TIMEOUT_S

    health = await _health(service)
    assert health["extension_connected"] is False
    assert health["extension_unresponsive"] is True
    assert health["link_attached"] is False, "the socket really is gone"
    assert health["link_silent_s"] > LINK_SILENCE_TIMEOUT_S

    # The retry the error copy advises must NOT land on "no browser is attached".
    response = await _rpc(service, {"id": "r-retry", "method": "tabs", "params": {}})
    body = bytes(response.body).decode().replace(" ", "")
    assert ErrorCode.EXTENSION_UNRESPONSIVE.value in body
    assert ErrorCode.EXTENSION_DISCONNECTED.value not in body
    assert '"phase":"dropped"' in body

    # And it expires: a browser that is genuinely gone must not read as
    # "attached but mute" forever.
    monkeypatch.setattr(daemon_module, "LINK_DROP_TTL_S", 0.01)
    time.sleep(0.02)
    assert service.link.recent_drop_silence() == 0.0
    health = await _health(service)
    assert health["extension_unresponsive"] is False
    response = await _rpc(service, {"id": "r-later", "method": "tabs", "params": {}})
    assert ErrorCode.EXTENSION_DISCONNECTED.value in bytes(response.body).decode()


def test_a_peer_that_closes_its_own_socket_is_absent_not_unresponsive(tmp_path: Path) -> None:
    """A link the daemon did NOT sever must never inherit the unresponsive label.

    The latch is cleared on every ordinary socket end and by a new authoritative
    socket, so a browser the user simply closed reads as absent — the honest
    line for that state — rather than as a mute peer.
    """
    app = create_app(root=tmp_path)
    with TestClient(app) as client:
        service = app.state.bridge
        with client.websocket_connect("/extension", headers={"origin": ORIGIN}) as socket:
            socket.send_json(
                {
                    "event": "hello",
                    "proto": PROTO_VERSION,
                    "token": "",
                    "extension_version": "0.1.0",
                    "browser": "Chrome/153",
                }
            )
            socket.receive_json()
            # Pretend an earlier silence drop latched the honest state...
            service.link.note_unproven_drop(LINK_SILENCE_TIMEOUT_S + 1.0)
            assert client.get("/health").json()["extension_unresponsive"] is True
        # ...and then the peer itself goes away.
        health = client.get("/health").json()
        assert health["extension_unresponsive"] is False
        assert health["link_attached"] is False
        assert service.link.dropped_unproven() is False


# --- D5 / D6 --------------------------------------------------------------


def test_a_daemon_side_timeout_names_a_remedy_not_a_raw_code() -> None:
    """`internal` + `timeout_s` must render as an action, not as a bare code.

    P3 correctly stopped a daemon-side timeout reading as a version mismatch,
    but left it on the generic fallback: an obscure code, the daemon's internal
    verb name, and no remedy at all — on `owner_recover`, whose entire job is
    recovery (design D5). The budget stays in details for diagnostics.
    """
    rendered = format_error(
        BridgeError(ErrorCode.INTERNAL, "owner_recover timed out", {"timeout_s": 20.0})
    )
    assert "did not answer within 20s" in rendered
    assert "chrome://extensions" in rendered
    assert "internal" not in rendered
    assert "owner_recover" not in rendered, "the daemon's internal verb does not belong in prose"
    assert "browser bridge error" not in rendered


def test_the_undrivable_tab_copy_names_no_opaque_handle() -> None:
    """No `(unknown)` and no `bridge:<tab>:<nonce>` capability in the sentence.

    `format_error`'s only handle here is the session's own opaque capability
    string, so the sentence used to read either `(unknown)` or a token in the
    middle of prose — neither human-meaningful (design D6).
    """
    rendered = format_error(
        BridgeError(
            ErrorCode.INTERNAL, "Chrome refused to debug this tab", {"undrivable_tab": "x"}
        ),
        action="read",
        surface="bridge:12:deadbeef",
    )
    assert "cannot be driven" in rendered
    assert "(unknown)" not in rendered
    assert "bridge:12:deadbeef" not in rendered
    assert "'open' with a URL" in rendered
