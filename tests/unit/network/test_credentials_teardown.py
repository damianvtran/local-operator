"""The relay owns the credential slice's loops (relay-stops-broker-loops).

TWO OWNERS BUILT A ``_BrokerLoop`` — ``MeshCredentialBroker`` and
``SyncEngine._broker_loop`` — and NEITHER had a production teardown caller. The
relay's ``on_stop`` seam existed and worked (#2914's five loops use it), but the
credentials slice registered no hook, so every relay generation that served a
borrow or scheduled a sync left a ``mesh-broker-loop-*`` thread for the life of
the process. That is the thread leak this file pins shut: the fixture-level cell
at the bottom is the one that says the CI thread number should move, because it
observes the device AFTER the ``devices`` fixture's own teardown has run.

Cells, and the brief's letter:

* (a) a real relay that served a real borrow stops with the broker loop dead and
  its thread name gone from ``threading.enumerate()``;
* (b) a loop that ignores its stop produces ONE warning and keeps its handle,
  and a wedged broker does not strand the sync engine's teardown beside it;
* (c) a second ``close`` neither warns again nor drops a survivor;
* (d) ``SyncEngine.close`` on an engine that never started a loop is a no-op;
* (e) the autouse fixture below asserts no ``mesh-broker-loop-*`` thread was
  added by the test once every fixture — including ``devices`` — has torn down.
"""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network.credentials import owner as owner_mod
from local_operator.network.credentials import sync as sync_mod
from local_operator.network.credentials.types import Grant
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    idp,
    stub_provider,
)
from tests.unit.network.test_credentials_real_link import (  # noqa: F401
    STUB_PROVIDER,
    _borrower_client,
    _pull,
    _share,
    mesh,
)
from tests.unit.network.test_relay_e2e import devices  # noqa: F401 — fixture by import

LOGGER = "local_operator.network.credentials.owner"


def _is_broker_loop(name: str) -> bool:
    """True for the thread family this file is about."""
    return name.startswith("mesh-broker-loop-")


def _broker_loop_names() -> set[str]:
    return {thread.name for thread in threading.enumerate() if _is_broker_loop(thread.name)}


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == LOGGER and record.levelno == logging.WARNING
    ]


async def _never() -> None:
    """A coroutine that starts nothing and returns at once (the submit path)."""
    return None


class _WedgedLoop:
    """Accepts ``loop.stop`` and never acts on it: the thread stays alive."""

    def stop(self) -> None:
        """The callback ``close`` queues — a wedged loop never runs it."""

    def call_soon_threadsafe(self, callback: Any, *args: Any) -> None:
        """Queue nothing: the whole point is that the stop is never acted on."""

    def is_closed(self) -> bool:
        return False


class _WedgedThread:
    name = "mesh-broker-loop-wedged"

    def __init__(self) -> None:
        self.joins: list[float | None] = []

    def join(self, timeout: float | None = None) -> None:
        self.joins.append(timeout)

    def is_alive(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def _no_broker_loop_thread_survives_teardown() -> Any:
    """(e) THE CI-NUMBER CELL: observe the process AFTER the fixtures are gone.

    Autouse fixtures are set up before an explicitly requested one at the same
    scope and are finalised in reverse setup order, so this generator resumes
    AFTER ``devices`` has called ``server_a.stop()``/``server_b.stop()`` — the
    same teardown the shared fixture performs for its eleven consumers. A set
    difference rather than a count, so a loop leaked by an earlier test in this
    worker cannot mask a new one here.
    """
    before = _broker_loop_names()
    yield
    after = _broker_loop_names()
    left = sorted(after - before)
    assert after <= before, f"a mesh-broker-loop-* thread outlived its fixture's teardown: {left}"


def test_a_real_relay_stop_reaps_the_broker_loop_it_served_with(
    mesh: Any,  # noqa: F811 — the parameter shadows the fixture import (house pattern)
) -> None:
    """(a) A real borrow starts the owner's loop; ``stop()`` returns with it dead.

    The borrow is the production path: ``_share`` writes the placement document,
    ``_pull`` learns it, and ``request_grant_sync`` drives ``net_broker`` over a
    real link, which is what calls ``MeshCredentialBroker.submit`` and starts the
    loop. Asserting the thread is GONE after ``stop()`` — not merely that close
    was called — is the post-condition ``stop()`` promises.
    """
    _share(mesh)
    _pull(mesh)
    grant = _borrower_client(mesh).request_grant_sync(STUB_PROVIDER, session_id="sess-teardown")
    assert isinstance(grant, Grant), grant
    assert grant.access_token, grant

    suffix = mesh.a.identity.device_id[-8:]
    owner_thread = f"mesh-broker-loop-{suffix}"
    assert owner_thread in _broker_loop_names(), "the real borrow never started the broker loop"

    broker = getattr(mesh.a, owner_mod._BROKER_ATTR)
    assert broker._loop._thread is not None  # noqa: SLF001 — the handle under test

    mesh.a.stop()

    assert owner_thread not in _broker_loop_names()
    assert broker._loop._thread is None, "a clean stop must clear the handle"  # noqa: SLF001
    assert broker._loop._closed is True  # noqa: SLF001


def test_a_wedged_loop_is_reported_once_and_keeps_its_handle(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """(b)+(c) One warning per survivor episode; a second close does not re-warn.

    A thread that ignores its stop cannot be killed from here, so the contract is
    visibility: the handle is retained (the only thing that can observe or retry
    it), the warning names the loop and the timeout, and re-entry still joins and
    re-checks but stays silent. This is #2913's latch shape on the instance.
    """
    loop = owner_mod._BrokerLoop("wedged")  # noqa: SLF001 — the class under test
    loop._loop = _WedgedLoop()  # type: ignore[assignment]  # noqa: SLF001
    wedged = _WedgedThread()
    loop._thread = wedged  # type: ignore[assignment]  # noqa: SLF001

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        loop.close(timeout=0.05)
        assert loop._thread is wedged  # noqa: SLF001 — retained, not dropped
        loop.close(timeout=0.05)

    warnings = _warnings(caplog)
    assert len(warnings) == 1, warnings
    assert "mesh-broker-loop-wedged" in warnings[0]
    assert "did not stop within" in warnings[0]
    assert wedged.joins == [0.05, 0.05], "a second close must still join and re-check"
    assert loop._survivor is wedged  # noqa: SLF001


def test_a_wedged_broker_does_not_strand_the_sync_engine(
    devices: tuple[Any, Any, str, int],  # noqa: F811 — the parameter shadows the import
    caplog: pytest.LogCaptureFixture,
) -> None:
    """(b) One slice's bad close must not stop the other owner's teardown.

    The relay runs stop hooks inline and per-hook: the broker's survivor warning
    is the broker's, and the sync engine must still be closed and joined beside
    it. A fake broker stands in for the wedged one so the test does not park a
    real wedged thread in the process; the engine's loop is REAL, because joining
    it is the behaviour under test.
    """
    server_a, _server_b, _host, _port = devices
    engine = sync_mod.sync_for_relay(server_a)
    assert engine is not None
    engine_loop = engine._broker_loop()  # noqa: SLF001 — start the real loop
    engine_loop.loop()
    engine_thread = engine_loop._thread
    assert engine_thread is not None and engine_thread.is_alive()

    wedged_loop = owner_mod._BrokerLoop("wedged")  # noqa: SLF001
    wedged_loop._loop = _WedgedLoop()  # type: ignore[assignment]  # noqa: SLF001
    wedged_loop._thread = _WedgedThread()  # type: ignore[assignment]  # noqa: SLF001
    setattr(server_a, owner_mod._BROKER_ATTR, SimpleNamespace(close=wedged_loop.close))

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        server_a.stop()

    assert not engine_thread.is_alive(), "the wedged broker stranded the sync engine's loop"
    assert engine_loop._closed is True  # noqa: SLF001
    assert len(_warnings(caplog)) == 1


def test_the_devices_fixture_teardown_reaps_both_slice_loops(
    devices: tuple[Any, Any, str, int],  # noqa: F811 — the parameter shadows the import
) -> None:
    """(e) The shared fixture's OWN teardown must reap both loops, no help from us.

    This cell deliberately does NOT call ``stop()``: the ``devices`` fixture's
    finally does, exactly as it does for its eleven consumers. The autouse
    fixture is finalised after that finally, so a slice that forgot its
    ``on_stop`` leaves a live ``mesh-broker-loop-*`` name for the assertion to
    catch — which is the pre-fix state of this tree, and the reason the CI thread
    count should move rather than merely look tidier.
    """
    server_a, server_b, _host, _port = devices
    broker = owner_mod.MeshCredentialBroker(
        root=server_a.root,
        self_device=server_a.identity.device_id,
        server=server_a,
    )
    setattr(server_a, owner_mod._BROKER_ATTR, broker)
    broker._loop.submit(_never(), timeout=5.0)  # noqa: SLF001 — the submit path starts it

    engine = sync_mod.sync_for_relay(server_b)
    assert engine is not None
    engine._broker_loop().loop()  # noqa: SLF001 — the engine's own lazy starter

    assert f"mesh-broker-loop-{server_a.identity.device_id[-8:]}" in _broker_loop_names()
    assert f"mesh-broker-loop-{server_b.identity.device_id[-8:]}" in _broker_loop_names()


def test_neither_accessor_builds_an_owner_once_the_relay_is_stopping(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resurrection guard: a late request must refuse, not start a fresh loop.

    ``stop()`` sets ``_stop`` BEFORE it runs the slice hooks, and the accessors are
    reachable while the hooks run (links are not torn down yet). Without this
    guard, a broker or engine built in that window would start a loop behind the
    hook that exists to reap one — the leak, restored. The refusal is the same
    by-name ``None`` an owning-nothing device already gets.
    """
    stop = threading.Event()
    stop.set()
    server = SimpleNamespace(
        root=tmp_path,
        identity=SimpleNamespace(device_id="d_" + "f" * 32, name="stopping"),
        _stop=stop,
    )
    built: list[Any] = []

    def _would_build(cls: Any, _server: Any) -> Any:
        built.append(_server)
        raise AssertionError("the broker was built after _stop was set")

    monkeypatch.setattr(owner_mod.MeshCredentialBroker, "for_relay", classmethod(_would_build))

    assert owner_mod.broker_for_relay(server) is None
    assert built == []
    assert sync_mod.sync_for_relay(server) is None


def test_sync_engine_close_without_a_loop_is_a_noop(
    root: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """(d) An engine that never scheduled anything has no loop, so close is silent.

    This is the common case for ``server.stop()``: most relays never sync a
    credential, and teardown must not create the loop it was asked to reap.
    """
    server = SimpleNamespace(
        root=root,
        identity=SimpleNamespace(device_id="d_" + "e" * 32, name="idle-device"),
    )
    engine = sync_mod.sync_for_relay(server)
    assert engine is not None
    assert engine._loop is None  # noqa: SLF001

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        engine.close()
        engine.close()

    assert engine._loop is None  # noqa: SLF001
    assert engine._closed is True  # noqa: SLF001
    assert _warnings(caplog) == []
    assert not any(name.endswith("eeeeeeee") for name in _broker_loop_names())
