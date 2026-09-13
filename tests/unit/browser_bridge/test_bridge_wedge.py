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
import hashlib
import json
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, cast

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
    """Stand-in for the extension leg's websocket: records close codes and frames."""

    def __init__(self) -> None:
        self.closed: list[int | None] = []
        self.sent: list[dict[str, Any]] = []

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.closed.append(code)

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)


class _StallingSendSocket(_RecordingSocket):
    """A socket that ACCEPTS the write and never finishes it.

    The distinction matters: parking the coroutine before `send_json` is entered
    would never set ``entered``, and the test could not tell a write that was
    suspended from one that was never attempted.
    """

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.entered.set()
        await asyncio.Future()


class _StallingCloseSocket(_RecordingSocket):
    """A socket whose close() never resolves (audit A2's fake close)."""

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.closed.append(code)
        await asyncio.Future()


class _FakePeer:
    """A scripted /extension connection: feeds frames, records what it is sent.

    The receive path is a QUEUE rather than a list so a test can decide when a
    frame becomes available — which is the whole point of the stale-frame row:
    the frame must arrive strictly after the socket stops being authoritative.
    """

    def __init__(self, extension_id: str = EXTENSION_ID) -> None:
        self.headers = {"origin": f"chrome-extension://{extension_id}"}
        self.accepted = False
        self.closed: list[int | None] = []
        self.sent: list[dict[str, Any]] = []
        self._frames: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def accept(self) -> None:
        self.accepted = True

    async def receive_json(self) -> dict[str, Any]:
        return await self._frames.get()

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.closed.append(code)

    def push(self, frame: dict[str, Any]) -> None:
        self._frames.put_nowait(frame)


class _StallingClosePeer(_FakePeer):
    """A superseded connection that is asked to close and never finishes it."""

    def __init__(self, extension_id: str = EXTENSION_ID) -> None:
        super().__init__(extension_id)
        self.close_entered = asyncio.Event()

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.closed.append(code)
        self.close_entered.set()
        await asyncio.Future()


def _hello(**overrides: Any) -> dict[str, Any]:
    frame = {
        "event": "hello",
        "proto": PROTO_VERSION,
        "token": "",
        "extension_version": "0.1.10",
        "browser": "Chrome/153",
    }
    frame.update(overrides)
    return frame


async def _settles(predicate: Callable[[], bool], seconds: float = 2.0) -> bool:
    """Await ``predicate``, giving REAL timers room to fire.

    A loop of ``await asyncio.sleep(0)`` advances no clock, so it can never let a
    deadline expire — which is exactly the behaviour these callers assert on.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return predicate()


def _connected(service: BridgeService, *, silent_for: float = 0.0) -> _RecordingSocket:
    """Give ``service`` an extension socket that spoke ``silent_for`` seconds ago."""
    socket = _RecordingSocket()
    service.link.websocket = socket  # type: ignore[assignment]
    service.link.last_frame_at = time.monotonic() - silent_for
    return socket


def _peer_spoke(service: BridgeService) -> None:
    """Record that the extension just said something.

    The receive loop does TWO things with an arriving frame — stamps liveness
    and sets the probe's event — so a test stub that stands in for a live peer
    must do both, or it models a peer that is talking while refusing to answer
    (which is not a state the protocol can produce). The `wire` keyword on every
    stub below is the other half of the same change: a send is scoped to the
    socket the caller captured, so a stub that models the extension leg has to
    accept it.
    """
    service.link.last_frame_at = time.monotonic()
    service.link.frame_event.set()


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

            async def record(payload: dict[str, Any], *, wire: Any = None) -> None:
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

    async def silent(payload: dict[str, Any], *, wire: Any = None) -> None:
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

    async def ponging(payload: dict[str, Any], *, wire: Any = None) -> None:
        # The extension received the command and is working on it. Its pongs
        # keep arriving for the whole budget — that is what a slow page looks
        # like on the wire. The send RETURNS; only the answer is missing.
        _peer_spoke(service)
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

    async def blocked(payload: dict[str, Any], *, wire: Any = None) -> None:
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
    async def answer(payload: dict[str, Any], *, wire: Any = None) -> None:
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

    async def answering(payload: dict[str, Any], *, wire: Any = None) -> None:
        # A HEALTHY peer's job in this row is not to be asleep: it answers a
        # solicited ping within an event-loop turn whatever else it is doing
        # (record §2.2). The fixture therefore has to answer, because the rule
        # under test now ASKS the peer instead of inferring from a clock — and a
        # stub that stayed mute would be modelling the defect, not the healthy
        # link this row exists to protect.
        _peer_spoke(service)
        return None

    service.link.send = answering  # type: ignore[method-assign]
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

    async def silent(payload: dict[str, Any], *, wire: Any = None) -> None:
        return None

    service.link.send = silent  # type: ignore[method-assign]
    response = await service._dispatch_serialized(
        Request(id="r-dead", method="read", params={"tab": "bridge:9:n"})
    )
    body = bytes(response.body).decode().replace(" ", "")
    assert ErrorCode.EXTENSION_UNRESPONSIVE.value in body
    assert '"phase":"response"' in body
    assert socket.closed == [4000], "a corroborated dead peer must be severed"


@pytest.mark.asyncio
async def test_the_tight_budget_methods_can_still_corroborate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rule must fire for a 20 s method, not only past the 30 s slack.

    Review R2-3 / QA Q2-4 measured the regression this pins: `PING_INTERVAL_S *
    1.5` is 30 s, above the 20 s budget of `read`/`snapshot`/`screenshot`, so a
    frozen worker was answered `internal {timeout_s: 20}` where round 1 answered
    `extension_unresponsive` — the rule had gone inert for exactly the methods
    the record introduced it for ("requiring 50 s would make a `read` unable to
    ever corroborate, since its own timeout fires first", §2.4).

    Silence here is 1.2 intervals — deliberately INSIDE the slack, below the
    threshold that protects a healthy link — so this row can only pass because
    the daemon ASKS the peer and gets nothing back. The companion row above
    (1.2 intervals, peer answering) is what stops that from becoming a licence
    to tear down healthy links: together they pin "the clock alone cannot
    decide; the answer can".
    """

    monkeypatch.setitem(daemon_module.COMMAND_TIMEOUTS, "read", 0.05)
    monkeypatch.setattr(daemon_module, "PING_INTERVAL_S", 1.0)
    # The probe window is shortened so the row is deterministic and fast; the
    # REAL number is a magnitude below every method's budget, which is the whole
    # property under test (see daemon.PING_PROBE_TIMEOUT_S).
    monkeypatch.setattr(daemon_module, "PING_PROBE_TIMEOUT_S", 0.05)
    service = BridgeService(root=tmp_path)
    socket = _connected(service, silent_for=1.2)

    async def mute(payload: dict[str, Any], *, wire: Any = None) -> None:
        # Accepts the frame and answers NOTHING — not the command, and not the
        # probe's ping. This is the wedged worker: its socket is open and it is
        # still draining, so only a direct question distinguishes it from a slow
        # page.
        return None

    service.link.send = mute  # type: ignore[method-assign]
    response = await service._dispatch_serialized(
        Request(id="r-frozen", method="read", params={"tab": "bridge:9:n"})
    )
    body = bytes(response.body).decode().replace(" ", "")
    assert ErrorCode.EXTENSION_UNRESPONSIVE.value in body
    assert '"phase":"response"' in body
    assert socket.closed == [4000], "a mute peer inside the slack must be severed"


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


class _RecoverFailure:
    """A browser resource whose recovery preamble raises the given error.

    Deliberately the REAL seams `execute_browser` drives (`lock`, `initialize`,
    `recover`, `recovered`) rather than a stubbed render helper: the defect this
    stands in for was a GREEN guard over a broken path, because the guard called
    `format_error` directly with the default empty `action` while the tool passed
    the daemon's verb name (design D2-1, review R2-2).
    """

    def __init__(self, exc: BaseException, *, recovered: bool) -> None:
        self._exc = exc
        self.generation = "g-test"
        self.record: dict[str, Any] = {"surface_id": "bridge:9:nonce"}
        self.recovered = recovered
        self.lock = asyncio.Lock()

    def initialize(self) -> None:
        return None

    async def recover(self) -> dict[str, Any]:
        raise self._exc


def _recover_context(exc: BaseException, *, recovered: bool) -> Any:
    from local_operator.harness.types import BrowserSurface, ToolContext

    surface = BrowserSurface()
    surface.surface_id = "bridge:9:nonce"
    surface.resource = _RecoverFailure(exc, recovered=recovered)  # type: ignore[assignment]
    return ToolContext(browser=surface)


@pytest.mark.asyncio
async def test_a_daemon_side_timeout_names_a_remedy_not_a_raw_code() -> None:
    """`internal` + `timeout_s` must render as an action, not as a bare code,
    THROUGH THE TOOL — both of its render sites.

    P3 correctly stopped a daemon-side timeout reading as a version mismatch,
    but left it on the generic fallback: an obscure code, the daemon's internal
    verb name, and no remedy at all — on the recovery path, whose entire job is
    recovery (design D5). The budget stays in details for diagnostics.

    The two sites are the recovery preamble and an explicit `recover` action.
    They used to disagree: the preamble rendered `format_error`, the action
    returned `str(exc)` — the raw `owner_recover timed out`, with no remedy and
    no typed code (design D2-1, review R2-2). The previous version of this test
    called `format_error` directly and passed vacuously; it now drives the tool
    with a resource whose `recover()` raises.
    """
    from local_operator.tools import builtin

    error = BridgeError(ErrorCode.INTERNAL, "owner_recover timed out", {"timeout_s": 20.0})

    # The preamble site: the session had not recovered yet, which is the flow the
    # reported incident took.
    preamble = await builtin.execute_browser(
        "t", {"action": "read"}, None, None, _recover_context(error, recovered=False)
    )
    # The explicit-action site. `recovered=True` so the preamble passes and the
    # action's own `recover()` call is what fails.
    explicit = await builtin.execute_browser(
        "t", {"action": "recover"}, None, None, _recover_context(error, recovered=True)
    )

    for result in (preamble, explicit):
        assert "did not answer within 20s" in result.text
        assert "chrome://extensions" in result.text, "the remedy must travel with the failure"
        assert "internal" not in result.text
        assert "owner_recover" not in result.text, "the internal verb does not belong in prose"
        assert "browser bridge error" not in result.text
        assert "the browser tab recovery" in result.text
        assert (result.details or {}).get("error_code") == "internal"


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


# --- Audit round 1: A1 — the generation fence -------------------------------


@pytest.mark.asyncio
async def test_an_old_send_deadline_cannot_sever_its_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A1, the daemon's command half. The most severe of the audit's findings.

    An OLD `send_json` is suspended when the daemon's link is replaced by a
    freshly healthy one. The old send's deadline then fires, and pre-fix
    `_drop_unproven_link` captured `self.link.websocket` — the CURRENT socket —
    closing the HEALTHY replacement with 4000 and failing every new session's
    futures. Reproduced by the auditor against the pinned head:

        {"new_closed":[4000],"new_still_attached":false,
         "response":{"error":{"code":"extension_unresponsive",
                              "data":{"phase":"send",...}}}}

    The fence is what makes those two sockets distinguishable: the teardown is
    scoped to the wire the caller decided about, and a caller whose wire is gone
    is told so rather than being handed a lie about a peer that is answering.
    """

    monkeypatch.setattr(daemon_module, "LINK_SEND_TIMEOUT_S", 0.05)
    service = BridgeService(root=tmp_path)
    old = _StallingSendSocket()
    service.link.attach(old)  # type: ignore[arg-type]
    service.link.paired = True
    service.link.last_frame_at = time.monotonic()

    task = asyncio.create_task(
        service._dispatch_serialized(
            Request(id="r-old", method="read", params={"tab": "bridge:1:n"})
        )
    )
    await asyncio.wait_for(old.entered.wait(), timeout=2.0)

    # A later, healthy connection arrives while the old write is still in flight.
    new = _RecordingSocket()
    service.link.forget_link_state()
    service.link.attach(new)  # type: ignore[arg-type]
    service.link.paired = True
    service.link.last_frame_at = time.monotonic()

    response = await asyncio.wait_for(task, timeout=2.0)
    body = bytes(response.body).decode().replace(" ", "")

    assert new.closed == [], "the healthy replacement was severed by a stale decision"
    assert service.link.websocket is new
    assert service.link.dropped_unproven() is False, "nothing was severed, so nothing is latched"
    assert ErrorCode.EXTENSION_DISCONNECTED.value in body
    assert ErrorCode.EXTENSION_UNRESPONSIVE.value not in body
    assert "notdelivered" in body.replace('"', ""), body
    # ...and it carries `phase`, so the reader is not told to open a browser
    # that is already open (design D3-3). Without the discriminator the client
    # renders the code's own copy: "no browser is attached. Ask the user to
    # open their browser".
    assert '"phase":"replaced"' in body, body


@pytest.mark.asyncio
async def test_an_old_ping_deadline_cannot_sever_its_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A1, the ping path — the same shape, and the audit names it explicitly.

    `_ping_tick`'s bounded send is the other place a deadline can fire long
    after the socket it was measuring stopped being the socket. A ping whose
    send parks must not take the replacement down with it, or a single stalled
    tick costs every session on a bridge that is now healthy.
    """

    monkeypatch.setattr(daemon_module, "LINK_SEND_TIMEOUT_S", 0.05)
    monkeypatch.setattr(daemon_module, "PING_INTERVAL_S", 0.01)
    service = BridgeService(root=tmp_path)
    old = _StallingSendSocket()
    service.link.attach(old)  # type: ignore[arg-type]
    service.link.paired = True
    service.link.last_frame_at = time.monotonic()

    tick = asyncio.create_task(service._ping_tick())
    await asyncio.wait_for(old.entered.wait(), timeout=2.0)

    new = _RecordingSocket()
    service.link.forget_link_state()
    service.link.attach(new)  # type: ignore[arg-type]
    service.link.paired = True
    service.link.last_frame_at = time.monotonic()

    await asyncio.wait_for(tick, timeout=2.0)

    assert new.closed == [], "a stale ping tick severed the live replacement"
    assert service.link.websocket is new
    assert service.link.dropped_unproven() is False


@pytest.mark.asyncio
async def test_a_superseded_frame_cannot_touch_the_live_link(tmp_path: Path) -> None:
    """A1, the receive half: an abandoned connection keeps draining frames.

    The audit inserted `bridge:1:old` into the NEW connection's driven records
    through a frame that completed on the old socket
    (`staleDrivenAccepted:true`). It is protocol fault injection rather than
    proof that real Chrome delivers a post-close frame — but the receive loop
    had no fence at all, so any buffered or delayed frame was processed as if it
    belonged to the live link. This pins the fence on the LOOP BODY, not just on
    the `finally` that already checked identity.
    """

    service = BridgeService(root=tmp_path)
    stale = _FakePeer()
    stale.push(_hello())
    stale_task = asyncio.create_task(service.extension(stale))  # type: ignore[arg-type]
    for _ in range(20):
        await asyncio.sleep(0)
        if stale.sent:
            break
    assert stale.sent, "the handshake answer never went out"
    first_generation = service.link.generation

    live = _FakePeer()
    live.push(_hello())
    live_task = asyncio.create_task(service.extension(live))  # type: ignore[arg-type]
    for _ in range(20):
        await asyncio.sleep(0)
        if live.sent:
            break
    assert service.link.generation == first_generation + 1
    assert service.link.websocket is live
    assert stale.closed == [4000], "the superseded socket was closed"

    # Freeze the live link's observable liveness, then let the stale socket
    # deliver the frame it was sitting on.
    service.link.last_frame_at = 1234.5
    stale.push(
        {"event": "tab_update", "tab": "bridge:1:old", "url": "https://old.test", "title": "OLD"}
    )
    await asyncio.sleep(0.05)

    assert service.link.driven == {}, "a superseded frame published into the live link"
    assert service.link.last_frame_at == 1234.5, "a superseded frame stamped liveness"
    assert service.link.websocket is live

    for task in (stale_task, live_task):
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


# --- Audit round 1: A2 — the teardown close is bounded ----------------------


@pytest.mark.asyncio
async def test_a_stalled_teardown_close_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A2: a close that never resolves must not park the recovery it IS.

    `_drop_unproven_link` clears the link and then awaits `websocket.close()`;
    pre-fix that await had no deadline, so a send-stalled peer left the
    initiating RPC — or the ping supervisor — pending forever with the link
    already cleared. The auditor's probe against the pinned head:

        {"case":"drop with blocked close","link_cleared":true,"drop_completed":false}

    Bounded here, and asserted through an outer `wait_for` so a regression fails
    as a red assertion in ~0.05 s rather than as a hung test (RC=124 under
    `-n auto --dist worksteal` is a killed worker carrying unrelated tests).
    """

    monkeypatch.setattr(daemon_module, "LINK_CLOSE_TIMEOUT_S", 0.05)
    service = BridgeService(root=tmp_path)
    socket = _StallingCloseSocket()
    service.link.attach(socket)  # type: ignore[arg-type]
    service.link.last_frame_at = time.monotonic()
    service.link.paired = True

    dropped = await asyncio.wait_for(
        service._drop_unproven_link("audit stalled close"), timeout=2.0
    )

    assert dropped is True
    assert socket.closed == [4000], "the close was never attempted"
    assert service.link.websocket is None
    assert service.link.dropped_unproven() is True


# --- QA Q2-1 / review R2-1: a revoke clears the latch -----------------------


@pytest.mark.asyncio
async def test_a_revoke_clears_a_latched_unproven_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The latch must not outlive the pairing it describes.

    `lop browser pair --reset` runs in ANOTHER process, so the only thing the
    daemon can notice is the pairing file changing. Pre-fix the clear lived in
    `revoke()`, whose every path requires a live socket (`_revocation_tick`
    guarded on `websocket is not None and paired`) — and the latch exists only
    AFTER `disconnect()` has nulled both. So for the rest of `LINK_DROP_TTL_S`
    the daemon told every caller "attached and paired … pairing is preserved"
    about a bridge the user had deliberately unpaired (QA Q2-1, mechanism added
    by review R2-1).

    The fix moves the clear to where the revocation is OBSERVED. No socket here
    on purpose: that is the state a drop leaves behind, and the state the old
    guard could not act in.
    """

    service = BridgeService(root=tmp_path)
    service.link.extension_id = EXTENSION_ID
    daemon_module._private_write(
        daemon_module._pairing_path(tmp_path), {"extension_id": EXTENSION_ID, "token_hash": "x"}
    )
    service.link.note_unproven_drop(52.0)
    assert service.link.dropped_unproven() is True
    assert service.link.websocket is None

    daemon_module.reset_pairing(tmp_path)
    monkeypatch.setattr(daemon_module, "REVOKE_WATCH_S", 0.0)
    await service._revocation_tick()

    assert service.link.dropped_unproven() is False, "an unpaired bridge still reads as wedged"


@pytest.mark.asyncio
async def test_a_revocation_tick_leaves_a_live_pairing_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The inverse guard: a tick that finds the pairing intact changes nothing.

    Without this half, clearing the latch unconditionally on every tick would
    pass the row above and quietly disable the honest "attached but not
    answering" answer for a bridge that is merely mute.
    """

    service = BridgeService(root=tmp_path)
    service.link.extension_id = EXTENSION_ID
    daemon_module._private_write(
        daemon_module._pairing_path(tmp_path), {"extension_id": EXTENSION_ID, "token_hash": "x"}
    )
    service.link.note_unproven_drop(52.0)

    monkeypatch.setattr(daemon_module, "REVOKE_WATCH_S", 0.0)
    await service._revocation_tick()

    assert service.link.dropped_unproven() is True


@pytest.mark.asyncio
async def test_a_stalled_revoke_close_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A2's sibling: `revoke()` closes with 4003 and must not park either.

    The same hole one function over, and this one is reached from the RPC gate
    and from the revocation watcher — so an unresponsive peer could park the
    revocation itself (audit A2 asks for the sibling closes explicitly, so that
    the fix is not merely relocated).
    """

    monkeypatch.setattr(daemon_module, "LINK_CLOSE_TIMEOUT_S", 0.05)
    service = BridgeService(root=tmp_path)
    socket = _StallingCloseSocket()
    service.link.attach(socket)  # type: ignore[arg-type]
    service.link.paired = True

    await asyncio.wait_for(service.revoke(), timeout=2.0)

    assert socket.closed == [4003], "the 4003 revoke close was never attempted"
    assert service.link.websocket is None


@pytest.mark.asyncio
async def test_a_superseded_socket_that_never_closes_does_not_park_the_accept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A2's other sibling: the later-connection-wins eviction close.

    A second browser profile arriving while the first is mid-close used to make
    the ACCEPT path await that close, so the healthy new connection could not
    finish its handshake behind a peer that never answers a close frame.
    """

    monkeypatch.setattr(daemon_module, "LINK_CLOSE_TIMEOUT_S", 0.05)
    service = BridgeService(root=tmp_path)
    first = _StallingClosePeer()
    first.push(_hello())
    first_task = asyncio.create_task(service.extension(first))  # type: ignore[arg-type]
    assert await _settles(lambda: bool(first.sent)), "precondition: the first connection handshook"

    second = _FakePeer()
    second.push(_hello())
    second_task = asyncio.create_task(service.extension(second))  # type: ignore[arg-type]
    assert await _settles(
        lambda: bool(second.sent)
    ), "the replacement's handshake completed behind a stalled close"

    assert first.close_entered.is_set(), "the superseded socket was asked to close"
    assert second.sent, "the replacement's handshake completed behind a stalled close"
    assert service.link.generation == 2
    assert service.link.websocket is second

    for task in (first_task, second_task):
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


# --- Audit round 2 re-audit: A1, the handshake's shared-state install -------
#
# The auditor drove three overlapping hellos at the real `extension()` and
# showed that a superseded handshake could still WRITE. Their verdict was that
# the replacement path retained the stale-handler defect with an authorization
# consequence (`rpc not_paired -> ok`), and that a revoke could clear a
# replacement that arrived while its close was in flight. Both rows below fail
# on `bb039c18e`.


class _GatedClosePeer(_FakePeer):
    """A peer whose ``close()`` parks until the test releases it.

    A1's repro needs the superseded socket's close held OPEN while a third
    handshake arrives — that await is the window the daemon has to survive.
    Distinct from ``_StallingClosePeer``, which models a close that never
    finishes at all (A2's bound), rather than one that resumes into a race.
    """

    def __init__(self, extension_id: str = EXTENSION_ID) -> None:
        super().__init__(extension_id)
        self.close_entered = asyncio.Event()
        self.release_close = asyncio.Event()

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.closed.append(code)
        self.close_entered.set()
        await self.release_close.wait()


class _AnsweringPeer(_FakePeer):
    """A scripted peer that ANSWERS every command frame it is sent.

    The authorization half of the A1 row is that an invalid-token peer could be
    DRIVEN, so the test has to show the drive either landing (pre-fix) or never
    leaving the daemon (post-fix). A peer that stayed mute could not tell those
    apart — it would look identical to a daemon that refused.
    """

    def __init__(self, extension_id: str = EXTENSION_ID) -> None:
        super().__init__(extension_id)
        self.commands: list[dict[str, Any]] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        await super().send_json(payload)
        if "method" in payload:
            self.commands.append(payload)
            self.push({"id": payload["id"], "ok": True, "result": {"driven": True}})


def _saved_pairing(root: Path, token: str) -> None:
    """Write the REAL pairing record the handshake validates against.

    The auditor's script stubbed ``_read_json``; this writes the file the daemon
    actually reads, through the same writer ``_try_pair`` uses, inside the
    test's own root. Token validation, the origin check and ``reset_pairing``
    therefore all run for real.
    """
    daemon_module._private_write(
        daemon_module._pairing_path(root),
        {
            "extension_id": EXTENSION_ID,
            "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "paired_at": time.time(),
        },
    )


async def _reply(service: BridgeService, request_id: str) -> dict[str, Any]:
    """The decoded body of one `tabs` RPC through the real HTTP gate."""
    response = await _rpc(service, {"id": request_id, "method": "tabs", "params": {}})
    return json.loads(bytes(response.body).decode("utf-8"))


@pytest.mark.asyncio
async def test_a_superseded_handshake_cannot_authorize_its_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A1 (blocker): the install must not straddle an await, and the ack must be fenced.

    The auditor's sequence against the pinned head, through the production
    `extension()`, token check and `rpc()`: W1's close is held, W2 (valid token)
    installs itself and parks on that close, W3 arrives with the permitted
    extension id and an INVALID token, and when W1's close is released the
    superseded W2 writes its own verdict — `paired: True` — onto W3's link. W3
    then drives RPCs through the real pairing gate:

        before: not_paired      after: ok -> {"accepted_by_unpaired_peer": true}

    That is authorization of a rejected token, not a stale frame. The guard is
    ordering: every shared-state write (identity, pairing, liveness, latch)
    completes before the first yield after `attach`, so a handshake that no
    longer owns the link has nothing left to write, and the ack it would have
    sent is fenced by identity too.
    """

    monkeypatch.setattr(daemon_module, "LINK_CLOSE_TIMEOUT_S", 5.0)
    _saved_pairing(tmp_path, "good-token")
    service = BridgeService(root=tmp_path)

    first = _GatedClosePeer()
    first.push(_hello(token="good-token"))
    first_task = asyncio.create_task(service.extension(first))  # type: ignore[arg-type]
    assert await _settles(lambda: service.link.websocket is first)
    assert service.link.paired, "precondition: the first peer holds the saved, valid token"

    # W2: a VALID token. It becomes the link, then parks closing W1 — the
    # window the auditor exploited. (`attach` precedes that close on every
    # head, which is why only the INSTALL ORDER is the fix, not the arrival.)
    second = _FakePeer()
    second.push(_hello(token="good-token"))
    second_task = asyncio.create_task(service.extension(second))  # type: ignore[arg-type]
    await asyncio.wait_for(first.close_entered.wait(), timeout=2.0)
    assert service.link.websocket is second, "W2 is the authoritative link"
    assert second.sent == [], "precondition: W2 is still inside its handshake"

    # W3: same permitted extension id, INVALID token. It supersedes W2 while W2
    # is still parked, and gets its own honest `paired: false` ack.
    third = _AnsweringPeer()
    third.push(_hello(token="bad-token"))
    third_task = asyncio.create_task(service.extension(third))  # type: ignore[arg-type]
    assert await _settles(
        lambda: any(frame.get("event") == "hello_ack" for frame in third.sent)
    ), "the newest peer's handshake answer never went out"
    assert service.link.websocket is third
    assert service.link.paired is False, "the invalid token did not pair"

    before = await _reply(service, "before")
    assert before["error"]["code"] == ErrorCode.NOT_PAIRED.value, before

    # Release W1's close: the superseded W2 resumes INSIDE its handshake. One
    # buffered frame lets a superseded receive loop exit on either head, so the
    # discriminator below is the pairing verdict rather than a parked task.
    first.release_close.set()
    second.push({"event": "pong"})
    await asyncio.wait_for(second_task, timeout=2.0)

    after = await _reply(service, "after")
    assert (
        after.get("error", {}).get("code") == ErrorCode.NOT_PAIRED.value
    ), "a superseded handshake authorized the newest, invalid-token peer: " + json.dumps(after)
    assert service.link.websocket is third
    assert service.link.paired is False, "the stale handshake rewrote pairing"
    assert second.sent == [], "a handshake that lost authority still spoke on its wire"
    assert third.commands == [], "an unpaired peer was driven with a command"

    for task in (third_task, first_task):
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_a_revoke_does_not_clear_a_replacement_that_arrived_mid_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A1's revoke half: a scoped revoke must not tear down a later handshake.

    `revoke()` captures the socket, awaits its bounded close, and then cleared
    the link UNCONDITIONALLY. Installing a replacement during that await
    therefore had its socket nulled and its state forgotten by a revoke that had
    already closed the connection it was aimed at — the auditor's
    `replacementStillAuthoritative: false`.

    The control that makes this a fix and not a hole: preserving the connection
    is NOT authorization. The replacement computed its own `paired` from the
    pairing file `reset_pairing` had already removed, so it is refused by the
    same `not_paired` gate as any other unpaired peer.
    """

    monkeypatch.setattr(daemon_module, "LINK_CLOSE_TIMEOUT_S", 5.0)
    _saved_pairing(tmp_path, "good-token")
    service = BridgeService(root=tmp_path)

    revoked = _GatedClosePeer()
    revoked.push(_hello(token="good-token"))
    revoked_task = asyncio.create_task(service.extension(revoked))  # type: ignore[arg-type]
    assert await _settles(lambda: service.link.paired)
    generation = service.link.generation

    revoke_task = asyncio.create_task(service.revoke())
    await asyncio.wait_for(revoked.close_entered.wait(), timeout=2.0)

    # The replacement dials while the revoked socket's close is still parked.
    replacement = _AnsweringPeer()
    replacement.push(_hello(token="good-token"))
    replacement_task = asyncio.create_task(service.extension(replacement))  # type: ignore[arg-type]
    assert await _settles(lambda: service.link.websocket is replacement)

    revoked.release_close.set()
    await asyncio.wait_for(revoke_task, timeout=2.0)

    assert service.link.websocket is replacement, "the revoke cleared the link it did not close"
    assert service.link.generation == generation + 1, "the replacement's link state was forgotten"
    assert service.link.paired is False, "a revoked token paired the replacement"
    assert 4003 in revoked.closed, "the revoked socket was not closed as unpaired"

    after = await _reply(service, "after")
    assert after["error"]["code"] == ErrorCode.NOT_PAIRED.value, after
    assert replacement.commands == [], "a revoked pairing drove the replacement"

    for task in (replacement_task, revoked_task):
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_a_lost_answer_on_a_replaced_wire_names_the_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Design D3-3, the response half: "replaced" is not "no browser is attached".

    The reachable interleaving is a command registered in the gap between
    `forget_link_state()` and `attach()` during a handshake: the replacement
    does not fail that future (it was not in the map when the superseded link's
    state was dropped), so this command's answer never arrives on either wire.
    By the time its budget expires the promotion rule finds, on a mute
    replacement, nothing of ours left to sever — and the honest answer is that
    the extension replaced its connection.

    That answer carries `extension_disconnected`, whose copy tells the reader no
    browser is attached and to ask the user to open one. `phase: replaced` is
    what makes the client render the truthful copy instead (design D3-3), which
    is why this row asserts BOTH the phase and that the absent-browser sentence
    is not what the model would read.
    """

    monkeypatch.setitem(daemon_module.COMMAND_TIMEOUTS, "read", 0.05)
    monkeypatch.setattr(daemon_module, "PING_INTERVAL_S", 1.0)
    monkeypatch.setattr(daemon_module, "PING_PROBE_TIMEOUT_S", 0.05)
    service = BridgeService(root=tmp_path)
    old = _connected(service, silent_for=2.0)
    service.link.paired = True
    old_wire = (service.link.websocket, service.link.generation)

    # The REAL `link.send`: the frame must genuinely land on the superseded wire
    # (that is what makes this command "delivered, never answered"), and nothing
    # ever pushes a response frame back — `_RecordingSocket` only records.
    task = asyncio.get_running_loop().create_task(
        service._dispatch_serialized(
            Request(id="r-lost", method="read", params={"tab": "bridge:9:n"})
        )
    )
    await asyncio.sleep(0)
    for _ in range(100):
        if old.sent:
            break
        await asyncio.sleep(0.002)
    assert old.sent == [
        {"id": "r-lost", "method": "read", "params": {"tab": "bridge:9:n"}}
    ], "precondition: the command was delivered on the superseded wire"
    assert service.link.pending.get("r-lost") is not None, "precondition: nothing failed its future"

    # The replacement: attached WITHOUT the superseded link's state being
    # dropped, which is what leaves this command's future pending across it.
    replacement = _RecordingSocket()
    service.link.attach(replacement)  # type: ignore[arg-type]
    service.link.paired = True
    service.link.last_frame_at = time.monotonic()

    response = await asyncio.wait_for(task, timeout=5.0)
    body = bytes(response.body).decode().replace(" ", "")

    assert service.link.websocket is replacement, "the replacement was severed"
    assert (service.link.websocket, service.link.generation) != old_wire
    assert ErrorCode.EXTENSION_DISCONNECTED.value in body, body
    assert '"phase":"replaced"' in body, body
    assert "hassincereplaced" in body, body
    # The rendered copy, not just the code: this is what the model reads.
    from local_operator.browser_bridge.backend import BridgeError, format_error

    rendered = format_error(
        BridgeError(
            ErrorCode.EXTENSION_DISCONNECTED,
            "read was delivered on a connection the extension has since replaced",
            {"phase": "replaced"},
        ),
        action="read",
    )
    assert "no browser is attached" not in rendered
    assert "open their browser" not in rendered
    assert "retry the action" in rendered


@pytest.mark.asyncio
async def test_a_sibling_teardown_is_not_reported_as_a_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3-4 (major): the fence refused for TWO reasons and only one was true.

    Two sessions timing out on ONE frozen worker is the multi-session case this
    PR exists for. The first timeout severs the link (`disconnect()` nulls the
    socket and leaves the generation alone); the second then asks the fence about
    the wire it captured, is refused, and used to take the "a replacement
    answered" arm — telling the reader no browser is attached and to ask the user
    to open one, for a browser that is open and wedged, with nothing having been
    replaced. At `a733b08a4` both commands answered `extension_unresponsive`.

    The assertion is deliberately race-tolerant about WHICH command severs the
    link (the two probes overlap): what must hold is that NEITHER answer claims a
    replacement, because no replacement happened on this wire.
    """

    monkeypatch.setitem(daemon_module.COMMAND_TIMEOUTS, "read", 0.05)
    monkeypatch.setitem(daemon_module.COMMAND_TIMEOUTS, "snapshot", 0.05)
    monkeypatch.setattr(daemon_module, "PING_INTERVAL_S", 1.0)
    # Long enough that both commands are inside the probe at the same time, which
    # is the interleaving that puts one of them on the wrong arm.
    monkeypatch.setattr(daemon_module, "PING_PROBE_TIMEOUT_S", 0.15)
    service = BridgeService(root=tmp_path)
    socket = _connected(service, silent_for=2.0)
    service.link.paired = True
    generation = service.link.generation

    bodies = await asyncio.gather(
        *(
            service._dispatch_serialized(
                Request(id=f"r-{index}", method=method, params={"tab": f"bridge:{index}:n"})
            )
            for index, method in ((1, "read"), (2, "snapshot"))
        )
    )
    decoded = [bytes(body.body).decode().replace(" ", "") for body in bodies]

    for body in decoded:
        assert ErrorCode.EXTENSION_DISCONNECTED.value not in body, (
            "a command on a torn-down wire was told the extension replaced its connection: " + body
        )
        assert ErrorCode.EXTENSION_UNRESPONSIVE.value in body, body
        assert '"link_silent_s"' in body, body

    assert socket.closed == [4000], "the wedge is severed exactly once"
    assert service.link.websocket is None, "the link stays down until the peer re-dials"
    assert (
        service.link.generation == generation
    ), "no replacement attached, which is the fact both answers had to respect"
    assert service.link.dropped_unproven(), "the latched reason is what the answer reports"


class _WedgedPeer(_GatedClosePeer):
    """A paired peer that accepts a command write and never finishes it.

    R4-2's repro needs the peer wedged on BOTH legs at once, because the two are
    what produce the window: the parked command write is what makes `_admit`'s
    send deadline fire and call `_drop_unproven_link`, and the parked *close* is
    what keeps that teardown suspended after it has already cleared both maps.
    Distinct from `_GatedClosePeer`, which refuses nothing until the handshake
    asks it to close.
    """

    def __init__(self, extension_id: str = EXTENSION_ID) -> None:
        super().__init__(extension_id)
        self.write_entered = asyncio.Event()

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)
        if "method" in payload:
            self.write_entered.set()
            await asyncio.Future()  # a write that never completes


@pytest.mark.asyncio
async def test_a_reused_id_does_not_cost_the_new_request_its_approval_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4-2 (major): the marker's removal needs the SAME identity proof the future got.

    `_forget_pending` deleted `link.pending` by identity — because a teardown
    clears that map, so a later request may re-use an id — and then popped
    `link.awaiting_origin` by id ALONE. Both describe one request's lifetime, but
    only the identity can prove WHICH request, so the unguarded half deleted a
    live request's marker. The window needs no hand-inserted state:

    1. a paired peer wedges, an old command's send deadline fires inside `_admit`
       (which forgets our future and then tears the link down), and the
       teardown's bounded `websocket.close()` parks on the wedged peer AFTER
       `forget_link_state()` has cleared both maps;
    2. the worker genuinely re-dials inside that window and re-pairs through the
       real `extension()`;
    3. a NEW request re-using the old id passes the busy guard — the guard reads
       the map the teardown emptied — reaches the wire, and the extension
       announces a human approval for it through the real receive loop;
    4. the old task's parked close finishes, and its `finally` unwinds through
       `_forget_pending`.

    Pre-fix that unwind took the new marker with it: `NEW_FUTURE_PRESERVED: true,
    NEW_APPROVAL_MARKER_PRESERVED: false`, so the new request lost the deadline
    extension `_await_response` exists to give it and failed at its BASE timeout
    while the user was still looking at the prompt.

    This row is also the only one that can see the FUTURE's identity guard
    (review R4-4's missing coverage): reverting it to a bare `pop(id)` leaves
    every other row green.
    """

    monkeypatch.setattr(daemon_module, "LINK_SEND_TIMEOUT_S", 0.05)
    _saved_pairing(tmp_path, "good-token")
    service = BridgeService(root=tmp_path)

    wedged = _WedgedPeer()
    wedged.push(_hello(token="good-token"))
    wedged_task = asyncio.create_task(service.extension(wedged))  # type: ignore[arg-type]
    assert await _settles(lambda: service.link.websocket is wedged and service.link.paired)

    old_id = "r-abc123abcdef"
    old_task = asyncio.create_task(
        service._dispatch_serialized(
            Request(id=old_id, method="read", params={"tab": "bridge:1:n"})
        )
    )
    await asyncio.wait_for(wedged.write_entered.wait(), timeout=2.0)
    # Step 1: the send deadline fired and the teardown is now parked in its
    # bounded close, with both maps already cleared.
    await asyncio.wait_for(wedged.close_entered.wait(), timeout=2.0)
    assert service.link.pending == {}, "precondition: the teardown cleared `pending`"
    assert service.link.awaiting_origin == {}, "precondition: the teardown cleared the markers"

    # Step 2: the worker re-dials for real and re-pairs. A peer that records
    # frames without answering them, so the new request's future stays registered.
    reconnected = _FakePeer()
    reconnected.push(_hello(token="good-token"))
    reconnected_task = asyncio.create_task(service.extension(reconnected))  # type: ignore[arg-type]
    assert await _settles(
        lambda: service.link.websocket is reconnected and service.link.paired
    ), "the re-dial did not re-pair"

    # Step 3: a NEW request re-using the old id — a different tab, so it takes a
    # different key and is not queued behind the old task's per-tab lock.
    new_task = asyncio.create_task(
        service._dispatch_serialized(
            Request(id=old_id, method="read", params={"tab": "bridge:2:m"})
        )
    )
    assert await _settles(
        lambda: any(frame.get("id") == old_id for frame in reconnected.sent)
    ), "the new request never reached the wire"
    new_future = service.link.pending.get(old_id)
    assert new_future is not None, "the new request registered no future to preserve"

    # ...and the extension announces a genuine human approval for it.
    reconnected.push({"event": "awaiting_origin", "id": old_id, "origin": "https://example.com"})
    assert await _settles(lambda: old_id in service.link.awaiting_origin), "no marker was set"

    # Step 4: the old task's parked close finishes and it unwinds.
    wedged.release_close.set()
    with suppress(Exception):
        await asyncio.wait_for(old_task, timeout=2.0)

    assert (
        service.link.pending.get(old_id) is new_future
    ), "the old unwind popped a LATER request's future, which its identity guard prevents"
    assert (
        old_id in service.link.awaiting_origin
    ), "the old unwind popped the NEW request's approval marker, so it expires at the base deadline"

    for task in (new_task, reconnected_task, wedged_task):
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_a_real_replacement_handshake_names_the_replacement(tmp_path: Path) -> None:
    """R4-3 (minor): `phase: replaced` must cover the arm a re-dial actually reaches.

    The delta gave the phase to the two TIMEOUT fences, but a genuine replacement
    handshake calls `forget_link_state()`, which fails every pending future with
    `RuntimeError("extension disconnected")` immediately — so an in-flight command
    does not reach a timeout at all. It lands in the generic `except Exception`
    arm, and without a `replaced` test there it fell through to a bare
    `extension_disconnected` whose rendered copy is "the bridge daemon is running
    but no browser is attached. Ask the user to open their browser" — for a
    browser that is demonstrably open, paired and reconnected. That is precisely
    the misdirection class this PR exists to remove (design D3-3), on the path a
    worker re-dial takes most often.

    The assertion is on the RENDERED sentence, not the code, because the code is
    shared and only `phase` selects the copy.
    """

    _saved_pairing(tmp_path, "good-token")
    service = BridgeService(root=tmp_path)

    first = _FakePeer()
    first.push(_hello(token="good-token"))
    first_task = asyncio.create_task(service.extension(first))  # type: ignore[arg-type]
    assert await _settles(lambda: service.link.websocket is first and service.link.paired)

    command_task = asyncio.create_task(
        service._dispatch_serialized(
            Request(id="r-inflight", method="read", params={"tab": "bridge:1:n"})
        )
    )
    assert await _settles(
        lambda: any(frame.get("id") == "r-inflight" for frame in first.sent)
    ), "the command never reached the wire"

    # A GENUINE replacement handshake, through the production accept path.
    second = _FakePeer()
    second.push(_hello(token="good-token"))
    second_task = asyncio.create_task(service.extension(second))  # type: ignore[arg-type]
    assert await _settles(
        lambda: service.link.websocket is second and service.link.paired
    ), "the replacement did not install"

    response = await asyncio.wait_for(command_task, timeout=2.0)
    error = json.loads(bytes(response.body).decode("utf-8"))["error"]

    assert error["code"] == ErrorCode.EXTENSION_DISCONNECTED.value, error
    assert error.get("data", {}).get("phase") == "replaced", (
        "a replacement handshake answered without naming the replacement, so the "
        "reader is told to open a browser that is already open: " + json.dumps(error)
    )

    rendered = format_error(
        BridgeError(ErrorCode(error["code"]), error["message"], error.get("data"))
    )
    assert "no browser is attached" not in rendered, rendered
    assert "open their browser" not in rendered, rendered
    assert "retry the action" in rendered, rendered

    for task in (first_task, second_task):
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
