"""P0 plumbing (mesh build plan §5): slow ops, the own-link guard, the slice hook.

Two real relays over real loopback TCP (the shared ``devices`` fixture), so what is
measured is the production reader/writer threads: "a ping answers while a slow op
runs" is a claim about ``PeerLink._handle``, and a stubbed link would only test the
stub. The slow handlers are stand-ins registered through the SAME hook a slice
module uses (``RelayServer.register_ops``), with ``replace=True`` because the
shipped slice modules have already claimed those names with their refusals.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import credentials, mobility, relay, sync, types
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

Devices = tuple[relay.RelayServer, relay.RelayServer, str, int]


@pytest.fixture()
def pair(request: pytest.FixtureRequest) -> Devices:
    """The shared two-relay fixture under a name that does not shadow the import."""
    value: Devices = request.getfixturevalue("devices")
    return value


def _unstarted(root: Path) -> relay.RelayServer:
    return relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )


def _admin_link(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, register: Any
) -> tuple[relay.PeerLink, relay.RelayServer]:
    """Pair B into A as admin, let ``register(server_a)`` add handlers, and dial A.

    ``register_ops`` refuses once a relay has started, and the shared fixture
    starts A; so the registration here goes through the same hook with the
    started-guard lifted for the stand-in, which is the one liberty this file
    takes (the guard itself has its own test below).
    """
    server_a, server_b, host, port = pair
    record, _host, _port = _pair(pair, monkeypatch, role="admin")
    threads, server_a._threads = server_a._threads, []  # noqa: SLF001
    try:
        register(server_a)
    finally:
        server_a._threads = threads  # noqa: SLF001
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is not None, reason
    return link, server_a


# ---------------------------------------------------------------------------
# A slow op does not stall the link
# ---------------------------------------------------------------------------


def test_a_ping_answers_while_a_slow_op_holds_its_handler(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The finding-4 regression: inline, the reader is the thread in the handler,
    so the ping below would wait for the release. Off the reader it answers now."""
    entered = threading.Event()
    release = threading.Event()
    seen: dict[str, Any] = {}

    def slow_sync(link: relay.PeerLink, frame: dict[str, Any]) -> dict[str, Any]:
        seen["thread"] = threading.current_thread().name
        seen["remaining"] = relay.slow_op_remaining_s()
        entered.set()
        release.wait(20)
        return {"phase": frame.get("phase"), "bytes": 3}

    link, _server_a = _admin_link(
        pair,
        monkeypatch,
        lambda server: server.register_ops(
            {"net_sync": slow_sync}, slow={"net_sync": 30.0}, replace=True
        ),
    )
    try:
        answer: dict[str, Any] = {}
        requester = threading.Thread(
            target=lambda: answer.update(
                link.request({"op": "net_sync", "req": 501, "phase": "plan"}) or {}
            ),
            daemon=True,
        )
        requester.start()
        assert entered.wait(10), "the slow handler never ran"

        started = time.monotonic()
        pong = link.request({"op": "ping", "req": 502}, timeout=5.0)
        elapsed = time.monotonic() - started
        assert pong is not None and pong["detail"] == "pong", pong
        assert not release.is_set(), "the ping must answer while the slow op is still held"
        assert elapsed < 5.0

        release.set()
        requester.join(10)
        assert answer.get("op") == "ack" and answer["detail"] == {"phase": "plan", "bytes": 3}
        assert seen["thread"].startswith("mesh-slow"), seen
        assert seen["remaining"] is not None and 0 < seen["remaining"] <= 30.0
    finally:
        release.set()
        link.close("test")


def test_an_inline_op_still_runs_on_the_reader(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control: a handler registered WITHOUT ``slow`` is on the reader thread,
    which is what makes the test above a measurement rather than a tautology."""
    seen: dict[str, str] = {}

    def fast(link: relay.PeerLink, frame: dict[str, Any]) -> dict[str, Any]:
        seen["thread"] = threading.current_thread().name
        return {"ok": True}

    link, _server_a = _admin_link(
        pair, monkeypatch, lambda server: server.register_ops({"net_sync": fast}, replace=True)
    )
    try:
        reply = link.request({"op": "net_sync", "req": 511, "phase": "plan"}, timeout=5.0)
        assert reply is not None and reply["op"] == "ack", reply
        assert seen["thread"].startswith("mesh-read"), seen
    finally:
        link.close("test")


def test_an_overrunning_slow_op_answers_with_its_deadline_once(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The requester hears ``did not finish within`` at the OWNER's deadline, and
    the handler's late answer is dropped rather than sent as a second reply."""
    release = threading.Event()
    finished = threading.Event()

    def stuck(link: relay.PeerLink, frame: dict[str, Any]) -> dict[str, Any]:
        release.wait(20)
        finished.set()
        return {"late": True}

    link, _server_a = _admin_link(
        pair,
        monkeypatch,
        lambda server: server.register_ops(
            {"net_sync": stuck}, slow={"net_sync": 0.5}, replace=True
        ),
    )
    try:
        reply = link.request({"op": "net_sync", "req": 521, "phase": "fetch"})
        assert reply is not None and reply["op"] == "error", reply
        assert "did not finish within 0.5 s" in reply["message"]
        assert "ask for its status" in reply["message"], "it started, so it may still land"
        strays_before = link.stray_replies
        release.set()
        assert finished.wait(10)
        time.sleep(0.3)
        assert link.stray_replies == strays_before, "a second reply reached the requester"
    finally:
        release.set()
        link.close("test")


def test_the_requester_waits_past_the_default_for_a_slow_op(root: Path) -> None:
    server = _unstarted(root)
    try:
        assert server.slow_op_deadline("ping") is None
        assert server.slow_request_timeout("ping") is None
        deadline = server.slow_op_deadline("net_session_move")
        assert deadline is not None and deadline == mobility.MOVE_OP_DEADLINE_S
        assert server.slow_request_timeout("net_session_move") == (
            deadline + relay.SLOW_REPLY_MARGIN_S
        )
        assert deadline > server.settings.op_wait_s
    finally:
        server.stop()


def test_slow_admission_is_bounded(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Past the bound a slow op is refused at once, with a sentence, not queued."""
    monkeypatch.setattr(relay, "SLOW_OP_MAX_PENDING", 1)
    server = _unstarted(root)
    release = threading.Event()
    sent: list[dict[str, Any]] = []

    class _Link:
        link_id = "l_fake"

        def send(self, frame: dict[str, Any]) -> bool:
            sent.append(frame)
            return True

    def held(link: Any, frame: dict[str, Any]) -> dict[str, Any]:
        release.wait(10)
        return {}

    granted = types.Granted(action="net_sync")
    try:
        first = server._dispatch_slow(  # noqa: SLF001
            _Link(), {"req": 1}, held, granted, 10.0  # type: ignore[arg-type]
        )
        second = server._dispatch_slow(  # noqa: SLF001
            _Link(), {"req": 2}, held, granted, 10.0  # type: ignore[arg-type]
        )
        assert first is None
        assert second is not None and second["op"] == "error"
        assert "already running 1 long operations" in second["message"]
    finally:
        release.set()
        server.stop()


# ---------------------------------------------------------------------------
# A handler may not ask the peer it is answering, over the same link
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slow", [False, True], ids=["inline", "slow"])
def test_a_handler_calling_back_over_its_own_link_is_refused(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, slow: bool
) -> None:
    """Unsafe item 6. Inline this is a guaranteed deadlock until the timeout; the
    guard turns it into an immediate refusal the requester reads as a sentence."""
    raised: list[BaseException] = []

    def calls_back(link: relay.PeerLink, frame: dict[str, Any]) -> dict[str, Any]:
        try:
            link.request({"op": "ping", "req": 9001})
        except relay.OwnLinkRequestError as exc:
            raised.append(exc)
            raise
        return {"unreachable": True}

    link, _server_a = _admin_link(
        pair,
        monkeypatch,
        lambda server: server.register_ops(
            {"net_sync": calls_back}, slow={"net_sync": 5.0} if slow else None, replace=True
        ),
    )
    try:
        started = time.monotonic()
        reply = link.request({"op": "net_sync", "req": 531, "phase": "plan"})
        assert time.monotonic() - started < 5.0, "the guard must answer, not time out"
        assert reply is not None and reply["op"] == "error", reply
        assert "may not ask the peer it is answering" in reply["message"]
        assert len(raised) == 1
        # And the link survived it: the next request is answered normally.
        pong = link.request({"op": "ping", "req": 532}, timeout=5.0)
        assert pong is not None and pong["detail"] == "pong"
    finally:
        link.close("test")


def test_the_guard_is_per_link_and_per_thread() -> None:
    assert relay.serving_link_id() is None
    with relay._serving_link("l_one"):  # noqa: SLF001
        assert relay.serving_link_id() == "l_one"
        other: list[str | None] = []
        worker = threading.Thread(target=lambda: other.append(relay.serving_link_id()))
        worker.start()
        worker.join()
        assert other == [None], "another thread is not serving this link"
    assert relay.serving_link_id() is None


# ---------------------------------------------------------------------------
# The slice hook
# ---------------------------------------------------------------------------


def test_the_shipped_slices_route_their_ops_through_the_hook_with_the_same_refusal(
    root: Path,
) -> None:
    server = _unstarted(root)
    try:
        assert set(server._slow_ops) == {  # noqa: SLF001
            "net_session_move",
            "net_sync",
            "net_broker",
        }
        assert server.slow_op_deadline("net_sync") == sync.SYNC_OP_DEADLINE_S
        assert server.slow_op_deadline("net_broker") == credentials.BROKER_OP_DEADLINE_S
        for op in ("net_session_move", "net_sync", "net_broker"):
            handler = server._handlers[op]  # noqa: SLF001
            with pytest.raises(types.MeshRefusal) as excinfo:
                handler(None, {"req": 1})  # type: ignore[arg-type]
            assert excinfo.value.code == "not_implemented"
            assert f"{op} is not implemented in this build yet (" in excinfo.value.sentence
            assert ".md" in excinfo.value.sentence
    finally:
        server.stop()


def test_the_broker_deadline_outlasts_the_provider_refresh_budget() -> None:
    """Restated rather than imported (the auth store is too heavy for the relay's
    construction path), so this pins the relation instead of the number."""
    from local_operator.providers.auth_store import PROVIDER_REFRESH_TOTAL_BUDGET_S

    assert credentials.BROKER_OP_DEADLINE_S > PROVIDER_REFRESH_TOTAL_BUDGET_S


def test_register_ops_refuses_what_a_slice_may_not_do(root: Path) -> None:
    server = _unstarted(root)

    def peer_handler(link: Any, frame: dict[str, Any]) -> None:
        return None

    def local_handler(frame: dict[str, Any]) -> dict[str, Any]:
        return {}

    try:
        with pytest.raises(ValueError, match="not a peer op a slice may register"):
            server.register_ops({"net_epoch": peer_handler})
        with pytest.raises(ValueError, match="not a local op a slice may register"):
            server.register_ops({}, {"net_member_rm": local_handler})
        with pytest.raises(ValueError, match="already registered by another slice"):
            server.register_ops({"net_sync": peer_handler})
        with pytest.raises(ValueError, match="declared slow but no handler"):
            server.register_ops({}, slow={"net_broker": 5.0}, replace=True)
        with pytest.raises(ValueError, match="positive deadline"):
            server.register_ops({"net_sync": peer_handler}, slow={"net_sync": 0}, replace=True)
        server.register_ops({}, {"session_move": local_handler}, replace=True)
        reply = server.control_dispatch("session_move", {"req": 4})
        assert reply == {"op": "ack", "req": 4, "detail": {}}
        server._threads.append(threading.current_thread())  # noqa: SLF001
        with pytest.raises(RuntimeError, match="before the relay starts"):
            server.register_ops({"net_sync": peer_handler}, replace=True)
    finally:
        server._threads.clear()  # noqa: SLF001
        server.stop()


def test_a_slice_that_fails_to_install_is_reported_not_fatal(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(server: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(sync, "install", broken)
    server = _unstarted(root)
    try:
        assert "net_sync" not in server._slow_ops  # noqa: SLF001
        assert "net_session_move" in server._slow_ops  # noqa: SLF001
        assert "slice local_operator.network.sync did not install (boom)" in (
            capsys.readouterr().err
        )
    finally:
        server.stop()


def test_an_unregistered_local_slice_verb_says_which_slice_owns_it(root: Path) -> None:
    """A local verb whose slice has not landed names the document that owns it.

    ``credential_grant`` used to be the second example here and is now SERVED (the
    credentials slice registers all three of its local verbs, and its own tests cover
    them) — so this list is the three mobility verbs, which still are not.
    """
    server = _unstarted(root)
    try:
        for op in ("session_move", "session_sync", "session_lifecycle"):
            reply = server.control_dispatch(op, {"req": 3})
            assert reply["code"] == "not_implemented", reply
            assert ".md" in reply["message"] and "owns that slice" in reply["message"]
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# The frozen seams
# ---------------------------------------------------------------------------


def test_the_seams_raise_not_implemented_with_the_owning_slice() -> None:
    with pytest.raises(NotImplementedError, match="mobility slice"):
        mobility.request_move("s1", to="build-box")
    with pytest.raises(NotImplementedError, match="mobility slice"):
        mobility.lifecycle("s1", action="delete", peer="build-box")
    with pytest.raises(NotImplementedError, match="mobility slice"):
        sync.request_sync("s1")


def test_the_session_move_contract_is_frozen() -> None:
    """V and DB build against these names; a change here is a contract change."""
    assert mobility.MOVE_RESULT_PHASES == ("prepared", "handing_off", "committed", "done")
    assert mobility.MOVE_OPENABLE_PHASES == {"committed", "done"}
    assert set(mobility.SessionMoveResult.__annotations__) == {
        "ok",
        "session_id",
        "new_session_id",
        "mode",
        "from_device",
        "to_device",
        "phase",
        "phases",
    }
    assert set(mobility.SessionMoveRefusal.__annotations__) == {
        "ok",
        "code",
        "message",
        "session_id",
        "phase_reached",
        "changed",
    }
    assert {"busy", "unreachable", "deadline_exceeded"} <= mobility.MOVE_REFUSAL_CODES
