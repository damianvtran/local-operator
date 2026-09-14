"""The daemon's lifecycle: the record goes up with the app, and comes down.

This drives the REAL ``lifespan`` (no uvicorn, but exactly the startup and
shutdown halves uvicorn runs), because that is where the record is published
and removed and the parts can only be checked together: what a reader sees is
a file that exists exactly while the daemon can answer, plus a ``/health`` that
names the same instance the file does. A unit test of either half alone would
pass while the pair disagreed, which is the failure this whole change exists to
end.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from local_operator.server import registry as serve_registry
from local_operator.server.app import app, lifespan


@pytest.fixture
def restore_app_state() -> Iterator[None]:
    """Give the process the app state it had before this test.

    ``app`` is a module-level singleton: the lifespan sets a dozen ``app.state``
    attributes to ``None`` on the way out, and the rest of the suite shares
    those attributes (that is why ``test_server_models`` documents the same
    hazard). Snapshotting the whole state is deliberately blunt — a
    hand-written list is exactly the thing that goes stale as the lifespan
    grows.

    Taken through Starlette's mapping interface (``__iter__`` +
    ``__getitem__``), NOT ``vars(app.state)``: that returns ``{'_state': …}``,
    whose value is the same dict the app keeps writing into, so a snapshot taken
    that way would restore the test's own changes. It matters here because
    ``advertised_address(app)`` reads the announced address off this state.
    """
    saved = {key: app.state[key] for key in app.state}
    state = app.state
    try:
        yield
    finally:
        for key in list(state):
            del state[key]
        for key, value in saved.items():
            state[key] = value


@pytest.mark.asyncio
async def test_lifespan_publishes_the_record_and_health_identifies_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_app_state: None
) -> None:
    """Published on startup, removed on shutdown, and the two halves agree.

    The address is announced the way ``serve_command`` announces it, including
    an ephemeral-looking port that is NOT the ``--port`` argument, which is the
    case the record exists for.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    serve_registry.announce_address(app, "127.0.0.1", 58474)
    path = serve_registry.record_path(os.getpid(), tmp_path)

    async with lifespan(app):
        assert path.exists(), "the record is published with the app"
        record = json.loads(path.read_text())
        assert (record["pid"], record["host"], record["port"]) == (os.getpid(), "127.0.0.1", 58474)
        assert record["install_kind"] and record["prefix"]
        assert record["instance_id"] == app.state.instance_id

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        result = response.json()["result"]
        assert response.status_code == 200
        assert result["instance_id"] == record["instance_id"], "same process, same identity"
        assert result["pid"] == record["pid"]

    assert not path.exists(), "a clean exit leaves no record claiming a live daemon"
    assert app.state.serve_record is None


@pytest.mark.asyncio
async def test_lifespan_removes_the_record_even_when_teardown_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_app_state: None
) -> None:
    """The removal is under ``finally:`` and is not skippable.

    A raise from any teardown step used to abort the rest of the shutdown half;
    for the record that would mean a file left behind claiming a daemon at a
    port nothing listens on — which is precisely the confusion a stale record
    must never cause, since a reader has no way to tell it from a live one until
    the heartbeat ages out.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    serve_registry.announce_address(app, "127.0.0.1", 58474)
    path = serve_registry.record_path(os.getpid(), tmp_path)

    with pytest.raises(RuntimeError, match="teardown exploded"):
        async with lifespan(app):
            assert path.exists()
            # The FIRST teardown step, so everything after it is skipped.
            #
            # Set directly rather than through ``monkeypatch.setattr`` on
            # purpose: the fixture's restore removes every key the test added,
            # and monkeypatch's undo (which does not run until AFTER that
            # restore) would then ``delattr`` a key that is already gone —
            # ``State.__delattr__`` raises ``KeyError`` for that, turning a
            # passing test into a fixture teardown error.
            app.state.desktop_auth = type("Boom", (), {"close": staticmethod(_raise_teardown)})()

    assert not path.exists()


async def _raise_teardown() -> None:
    raise RuntimeError("teardown exploded")


@pytest.mark.asyncio
async def test_a_boot_that_was_never_announced_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_app_state: None
) -> None:
    """A record is a claim that a daemon is listening at an address.

    An app started by something other than ``lop serve`` — a bare ``uvicorn
    local_operator.server.app:app``, a wrapper script — has no address to make
    that claim with, so it publishes NOTHING. A placeholder record (an empty
    host, port 0) would be the artefact this module exists to remove: a reader
    cannot dial it, and cannot tell it from a live daemon's until the heartbeat
    ages out.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv(serve_registry.SERVE_ANNOUNCE_ENV, raising=False)

    async with lifespan(app):
        assert serve_registry.scan(tmp_path) == [], "no announcer, no record"
        assert getattr(app.state, "serve_record", None) is None
        assert getattr(app.state, "serve_heartbeat", None) is None
        # Identity is not conditional on being discoverable: `/health` still
        # names this process.
        assert app.state.instance_id


@pytest.mark.asyncio
async def test_a_boot_that_only_INHERITED_an_announcement_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_app_state: None
) -> None:
    """The false-rendezvous leak: a nested boot must not advertise its parent.

    ``--reload`` is the only path that announces through the environment, and
    the value names the process that made it, so it is honoured only by a child
    that process spawned. A process that merely INHERITED the variable — an
    agent's shell tool running inside the daemon, a wrapper, a nested boot of the
    same app — reads it, takes nothing from it, and publishes no record. Before
    this rule it published one naming its PARENT's listener with its own pid:
    exactly the record that claims a daemon where none is listening.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # A real announcement, in the real shape, addressed to a process that is not
    # our spawner: this process's own pid (alive, but it spawned nothing).
    monkeypatch.setenv(serve_registry.SERVE_ANNOUNCE_ENV, f"{os.getpid()} 10.0.0.1 9000")

    async with lifespan(app):
        assert serve_registry.scan(tmp_path) == []
        assert getattr(app.state, "serve_record", None) is None
        assert serve_registry.SERVE_ANNOUNCE_ENV not in os.environ, (
            "the inherited announcement is consumed, so nothing this process "
            "spawns can re-publish it"
        )


@pytest.mark.asyncio
async def test_lifespan_runs_the_retirement_poll_beside_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_app_state: None
) -> None:
    """The poll is started with the publisher and stopped by the shutdown half.

    Driven against the REAL lifespan with the poll itself replaced, because what
    is under test here is the wiring: a poll that a shutdown forgets to stop
    would keep the event loop alive after the listener closed, and one started
    without a record has no announcement channel at all (the next test).
    """
    import asyncio

    from local_operator.server import retire as serve_retire

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    serve_registry.announce_address(app, "127.0.0.1", 58474)

    stops: list[asyncio.Event] = []

    async def poll(application, publisher, *, stop, exit_process=None):  # noqa: ANN001
        stops.append(stop)
        await stop.wait()

    monkeypatch.setattr(serve_retire, "retirement_poll", poll)

    async with lifespan(app):
        # One turn of the loop: ``create_task`` schedules the poll, and this is
        # what lets it reach its first line before the assertions read it.
        await asyncio.sleep(0)
        assert len(stops) == 1, "started once, beside the record publisher"
        assert app.state.serve_retire is not None
        assert not stops[0].is_set()

    assert stops[0].is_set(), "the shutdown half ends the poll"
    assert app.state.serve_retire is None
    assert app.state.serve_retire_stop is None
    assert (
        app.state.serve_retiring is False
    ), "the one-way latch is cleared for the next boot in this process"


@pytest.mark.asyncio
async def test_lifespan_starts_no_retirement_poll_without_a_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_app_state: None
) -> None:
    """No record, no poll: a daemon nobody can discover has nobody to tell."""
    import asyncio

    from local_operator.server import retire as serve_retire

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    started: list[asyncio.Event] = []

    async def poll(application, publisher, *, stop, exit_process=None):  # noqa: ANN001
        started.append(stop)

    monkeypatch.setattr(serve_retire, "retirement_poll", poll)

    # No announcement, so no record is published (see the inherited-announcement
    # test above for the whole rule).
    async with lifespan(app):
        await asyncio.sleep(0)
        assert started == []
        assert getattr(app.state, "serve_retire", None) is None
