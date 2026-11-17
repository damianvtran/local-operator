"""
Tests for the health endpoint of the FastAPI server.

This module contains tests for the health check endpoint to ensure
the server is responding correctly, and for the identity it reports: a 200
from ``/health`` used to be the ONLY way to recognise "the backend", and three
daemons on this machine (three builds) each produced one.
"""

import os
import sys
import time

import pytest

from local_operator.server.app import app
from local_operator.server.models.schemas import HealthCheckResponse


@pytest.mark.asyncio
async def test_health_check(test_app_client):
    """Test the health check endpoint using the test_app_client fixture."""
    response = await test_app_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == 200
    assert data.get("message") == "ok"


@pytest.mark.asyncio
async def test_health_check_identifies_the_answering_instance(test_app_client, monkeypatch):
    """The fields that tell one daemon from another, without a lifespan.

    ``instance_id`` comes from app state (minted by the lifespan) and is the
    value the serve rendezvous record also carries, so a discoverer can confirm
    that the process answering ``/health`` is the process it found. ``pid``,
    ``prefix`` and ``install_kind`` are read fresh here precisely because they
    can change under a running daemon: ``lop-update`` replaces the installed
    tree in place, and the install that is SERVING is the one to update.
    """
    monkeypatch.setattr(app.state, "instance_id", "minted-at-startup", raising=False)

    response = await test_app_client.get("/health")
    result = response.json()["result"]

    assert response.status_code == 200
    assert result["instance_id"] == "minted-at-startup"
    assert result["pid"] == os.getpid()
    assert result["prefix"] == sys.prefix
    assert result["install_kind"], "this venv is an editable install, not 'unknown' if empty"
    # The original field keeps its spelling and its meaning: it is what the
    # update banner reads, and an older client must keep working untouched.
    assert result["version"] == HealthCheckResponse(version=result["version"]).version


@pytest.mark.asyncio
async def test_health_check_still_answers_without_a_lifespan(test_app_client, monkeypatch):
    """A liveness probe must never be the route that breaks.

    The server suite builds this client WITHOUT running the lifespan, so there
    is no ``instance_id`` to read. An empty string is the truthful answer there
    — unidentified — and it must not become a 500: the field is additive, so a
    harness (or an older in-process caller) that has none still gets a usable
    response.
    """
    monkeypatch.delattr(app.state, "instance_id", raising=False)

    response = await test_app_client.get("/health")

    assert response.status_code == 200
    assert response.json()["result"]["instance_id"] == ""


def test_the_identity_fields_are_additive_for_an_older_client() -> None:
    """An older payload still parses, and the new one is a superset.

    Both directions matter during a rollout: a client written before these
    fields ignores them, and a server that predates them (whose payload has
    only ``version``) still validates here.
    """
    old_payload = HealthCheckResponse(version="0.54.32")
    assert old_payload.instance_id == ""
    assert old_payload.pid == 0
    assert old_payload.prefix == ""
    assert old_payload.install_kind == ""
    assert set(HealthCheckResponse(version="x").model_dump()) == {
        "version",
        "instance_id",
        "pid",
        "prefix",
        "install_kind",
        # The rolling-continuity contract, additive like the four above: an older
        # client drops the key, and a server that predates it (the old payload
        # asserted below) still validates. Pinned so the key cannot silently
        # disappear from the fresh-serving-process health report.
        "rolling_reload_contract",
    }


@pytest.mark.asyncio
async def test_a_probe_burst_reads_the_install_identity_once(test_app_client, monkeypatch):
    """Serve stall, 2026-11-15: ``/health``'s own install reads held the loop.

    Under a busy snapshot worker each of the reading's filesystem calls waits
    out a GIL switch interval, so the probe reuses one reading for
    ``INSTALL_IDENTITY_TTL_S`` -- and still re-reads after it, so an in-place
    ``lop-update`` shows within one poll.
    """
    from local_operator import update
    from local_operator.server.routes import health

    calls = {"n": 0}

    def counted() -> str:
        calls["n"] += 1
        return f"9.9.{calls['n']}"

    clock = {"now": 1_000.0}
    monkeypatch.setattr(update, "installed_version", counted)
    monkeypatch.setattr(health.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(health, "_install_identity", None)

    first = (await test_app_client.get("/health")).json()["result"]
    clock["now"] += health.INSTALL_IDENTITY_TTL_S / 2
    second = (await test_app_client.get("/health")).json()["result"]
    assert calls["n"] == 1 and first["version"] == second["version"] == "9.9.1"

    clock["now"] += health.INSTALL_IDENTITY_TTL_S
    third = (await test_app_client.get("/health")).json()["result"]
    assert calls["n"] == 2 and third["version"] == "9.9.2"


@pytest.fixture
def fresh_identity_reader(monkeypatch):
    """An empty identity cache and a reader of this test's own.

    The reader is module state (one in-flight read per route), so a test that
    leaves a stuck read behind must not hand it to the next test. ``raising``
    is off only so the base-code proof (which has no reader) can still run.
    """
    from local_operator.buildwatch import OffLoopBuildRead
    from local_operator.server.routes import health

    monkeypatch.setattr(health, "_install_identity", None)
    reader = OffLoopBuildRead("test /health install identity", timeout_s=0.2)
    monkeypatch.setattr(health, "_identity_reader", reader, raising=False)
    return health


@pytest.mark.asyncio
async def test_a_stuck_identity_read_does_not_stall_the_loop(
    test_app_client, monkeypatch, fresh_identity_reader
):
    """QA Q1 on #4143: ``/health`` read the install identity ON the loop.

    A disk that does not answer (the 145 s ``importlib.metadata`` read of
    2026-11-17) froze every coroutine with it, and ``/health`` -- the route
    pollers use to decide the daemon is alive -- was the one doing it. Now the
    read runs off the loop: a concurrent coroutine keeps ticking, and the probe
    answers within its bound with the last known identity, or ``unknown`` when
    there has never been one. The release timer only keeps the pre-fix code
    from hanging the suite; it fires long after every bound asserted here.
    """
    import asyncio
    import threading

    from local_operator import update

    health = fresh_identity_reader
    release = threading.Event()
    safety = threading.Timer(3.0, release.set)
    safety.start()

    def stuck() -> str:
        release.wait()
        return "9.9.9"

    monkeypatch.setattr(update, "installed_version", stuck)

    gaps: list[float] = []

    async def ticker(stop: asyncio.Event) -> None:
        loop = asyncio.get_running_loop()
        last = loop.time()
        while not stop.is_set():
            await asyncio.sleep(0.01)
            now = loop.time()
            gaps.append(now - last)
            last = now

    try:
        stop = asyncio.Event()
        ticking = asyncio.create_task(ticker(stop))
        await asyncio.sleep(0.05)
        loop = asyncio.get_running_loop()

        started = loop.time()
        cold = await test_app_client.get("/health")
        cold_elapsed = loop.time() - started

        # A stale reading exists now: the stuck read must not withhold it.
        health._install_identity = (time.monotonic() - 100.0, "1.2.3", "uv-tool")
        started = loop.time()
        warm = await test_app_client.get("/health")
        warm_elapsed = loop.time() - started

        stop.set()
        await ticking
    finally:
        release.set()
        safety.cancel()

    assert cold_elapsed < 1.5 and warm_elapsed < 1.5, (cold_elapsed, warm_elapsed)
    assert max(gaps) < 0.5, f"the loop stalled for {max(gaps):.2f} s"
    assert cold.status_code == 200
    assert (cold.json()["result"]["version"], cold.json()["result"]["install_kind"]) == (
        "unknown",
        "unknown",
    )
    assert warm.status_code == 200
    assert (warm.json()["result"]["version"], warm.json()["result"]["install_kind"]) == (
        "1.2.3",
        "uv-tool",
    )


@pytest.mark.asyncio
async def test_a_fresh_identity_is_served_without_a_read(
    test_app_client, monkeypatch, fresh_identity_reader
):
    """The TTL still means what it said: inside it, no read at all."""
    from local_operator import update

    health = fresh_identity_reader

    def must_not_read() -> str:
        raise AssertionError("a TTL hit read the disk")

    monkeypatch.setattr(update, "installed_version", must_not_read)
    health._install_identity = (time.monotonic(), "4.5.6", "pipx")

    result = (await test_app_client.get("/health")).json()["result"]

    assert (result["version"], result["install_kind"]) == ("4.5.6", "pipx")
    assert not health._identity_reader.failing


@pytest.mark.asyncio
async def test_a_cold_probe_reports_the_real_installed_version(
    test_app_client, fresh_identity_reader
):
    """The normal path, through the off-loop read, still reports this install."""
    import importlib.metadata

    from local_operator.update import install_kind, installed_version

    expected = installed_version() or importlib.metadata.version("local-operator")

    result = (await test_app_client.get("/health")).json()["result"]

    assert result["version"] == expected != "unknown"
    assert result["install_kind"] == install_kind().value
    assert fresh_identity_reader._install_identity is not None
