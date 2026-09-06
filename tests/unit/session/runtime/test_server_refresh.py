"""``refresh_if_idle`` and ``announce_retiring`` on the RuntimeServer.

The viewer-side belt for the runtime's self-refresh (design-runtime-autorefresh
§3.3): a resume in the seconds after ``lop-update`` binds to a stale idle
owner before its reaper has noticed, and asks it to retire now. Every
uncertain answer is ``kept``: a wrong "retire" costs a cold start nobody
asked for, a wrong "keep" costs the reaper's next check.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from local_operator import update as update_mod
from local_operator.session.runtime.server import RuntimeServer, _ClientConn
from local_operator.update import BuildStamp
from tests.unit.session.runtime.test_server import FakeHandle

OLD = BuildStamp(version="0.49.8", source_ref="46a4e9b1234567")
NEW = BuildStamp(version="0.49.9", source_ref="f4a70b991234567")


class RefreshableHandle(FakeHandle):
    def __init__(self, *, reason: str = "") -> None:
        super().__init__()
        self.reason = reason
        self.stopped = False
        self.probes = 0

    def may_refresh(self) -> str:
        self.probes += 1
        return self.reason

    def request_stop(self) -> None:
        self.stopped = True


def _conn(kind: str) -> _ClientConn:
    return _ClientConn(writer=cast(Any, object()), kind=cast(Any, kind))


@pytest.fixture
def stale(monkeypatch):
    """The server booted on OLD; the disk now carries NEW, settled."""
    monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: NEW)
    monkeypatch.setattr(update_mod, "build_marker_age_s", lambda *_a, **_k: 999.0)
    monkeypatch.delenv("LOP_BUILD_PREFIX", raising=False)


def _rig(handle: Any, *, boot: BuildStamp = OLD) -> tuple[RuntimeServer, list[dict[str, Any]]]:
    """A server booted on ``boot`` whose socket writes are captured, tagged
    with the recipient's kind so the daemon/attach split is assertable."""
    server = RuntimeServer(handle, kind="tui")
    server._boot_build = boot
    sent: list[dict[str, Any]] = []

    async def capture(target, frame):  # noqa: ANN001
        sent.append({"_recipient": target.kind, **frame})

    server._send_to = capture  # type: ignore[assignment]
    return server, sent


async def _ask(server: RuntimeServer, sent: list[dict[str, Any]], conn: _ClientConn) -> str:
    await server._on_request({"op": "refresh_if_idle", "req": 1}, conn)
    acks = [f for f in sent if f.get("op") in ("ack", "error")]
    assert acks, "the op never replied"
    reply = acks[-1]
    assert reply.get("op") == "ack", f"unexpected reply: {reply}"
    return str(reply.get("detail", ""))


@pytest.mark.asyncio
async def test_an_idle_stale_runtime_announces_and_retires(stale) -> None:
    handle = RefreshableHandle(reason="")
    server, sent = _rig(handle)
    viewer = _conn("attach")
    daemon = _conn("daemon")
    server._clients[id(viewer.writer)] = viewer
    server._clients[id(daemon.writer)] = daemon

    detail = await _ask(server, sent, viewer)

    assert detail == "retiring"
    assert handle.stopped is True
    retiring = [f for f in sent if f.get("op") == "retiring"]
    assert [f["_recipient"] for f in retiring] == ["attach"], "the phone daemon never sees it"
    assert retiring[0]["reason"] == "stale-build"
    assert retiring[0]["from"] == OLD.label() and retiring[0]["to"] == NEW.label()
    assert handle.probes == 2, "re-asked after the announce (the one await before the stop)"


@pytest.mark.asyncio
async def test_a_busy_stale_runtime_is_kept(stale) -> None:
    handle = RefreshableHandle(reason="busy")
    server, sent = _rig(handle)
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer

    assert await _ask(server, sent, viewer) == "kept: busy"
    assert handle.stopped is False
    assert not [f for f in sent if f.get("op") == "retiring"]


@pytest.mark.asyncio
async def test_a_matching_build_is_kept(stale, monkeypatch) -> None:
    monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: OLD)
    handle = RefreshableHandle(reason="")
    server, sent = _rig(handle)
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer

    assert (await _ask(server, sent, viewer)).startswith("kept: build on disk matches")
    assert handle.stopped is False


@pytest.mark.asyncio
async def test_an_unsettled_install_is_kept(stale, monkeypatch) -> None:
    monkeypatch.setattr(update_mod, "build_marker_age_s", lambda *_a, **_k: 1.0)
    handle = RefreshableHandle(reason="")
    server, sent = _rig(handle)
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer

    assert (await _ask(server, sent, viewer)).startswith("kept:")
    assert handle.stopped is False


@pytest.mark.asyncio
async def test_work_arriving_after_the_announce_is_kept(stale) -> None:
    class Flips(RefreshableHandle):
        def may_refresh(self) -> str:
            self.probes += 1
            return "" if self.probes == 1 else "busy"

    handle = Flips()
    server, sent = _rig(handle)
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer

    detail = await _ask(server, sent, viewer)
    assert detail.startswith("kept: busy")
    assert handle.stopped is False
    assert [f for f in sent if f.get("op") == "retiring"], "announced, then refused: safe"


@pytest.mark.asyncio
async def test_a_runtime_that_cannot_judge_itself_is_kept(stale) -> None:
    server, sent = _rig(FakeHandle())  # no may_refresh
    viewer = _conn("attach")
    server._clients[id(viewer.writer)] = viewer
    assert (await _ask(server, sent, viewer)).startswith("kept: this runtime cannot judge")
