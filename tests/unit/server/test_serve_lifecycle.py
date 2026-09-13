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
    hazard). Snapshotting the whole state dict is deliberately blunt — a
    hand-written list is exactly the thing that goes stale as the lifespan
    grows.
    """
    saved = dict(vars(app.state))
    try:
        yield
    finally:
        state = vars(app.state)
        state.clear()
        state.update(saved)


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
    serve_registry.announce_address("127.0.0.1", 58474)
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
    serve_registry.announce_address("127.0.0.1", 58474)
    path = serve_registry.record_path(os.getpid(), tmp_path)

    with pytest.raises(RuntimeError, match="teardown exploded"):
        async with lifespan(app):
            assert path.exists()
            # The FIRST teardown step, so everything after it is skipped.
            # ``raising=False``: the attribute does not exist unless something
            # else already built a desktop auth for this process.
            monkeypatch.setattr(
                app.state,
                "desktop_auth",
                type("Boom", (), {"close": staticmethod(_raise_teardown)})(),
                raising=False,
            )

    assert not path.exists()


async def _raise_teardown() -> None:
    raise RuntimeError("teardown exploded")


@pytest.mark.asyncio
async def test_an_unannounced_address_is_recorded_as_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_app_state: None
) -> None:
    """An app started by something other than ``lop serve`` says so.

    ``("", 0)`` is the truthful answer there, and the reader's identity check —
    not the record's port — is what admits a candidate, so an unusable address
    is a reportable state rather than a wrong port that dials a stranger.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv(serve_registry.SERVE_HOST_ENV, raising=False)
    monkeypatch.delenv(serve_registry.SERVE_PORT_ENV, raising=False)
    path = serve_registry.record_path(os.getpid(), tmp_path)

    async with lifespan(app):
        record = json.loads(path.read_text())
        assert (record["host"], record["port"]) == ("", 0)
        # Still a complete, classifiable record: a daemon whose address is
        # unknown is still identifiable, which is the other half of the point.
        assert record["instance_id"] == app.state.instance_id
        assert [(r.pid, state) for r, state in serve_registry.scan(tmp_path)] == [
            (os.getpid(), "live")
        ]
