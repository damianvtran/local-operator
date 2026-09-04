"""Eager runtime start, and the retirement that pays for it.

A viewer used to hold no runtime until the user's first keystroke, so a freshly
opened TUI painted a band with no MCP roster, no context/token reading and no
effective model — indistinguishable from a status bar that had failed to load.
The viewer now engages at mount instead.

That trades a lazy start for a process per opened terminal, and these tests pin
the two halves of the bargain:

1. The engage happens with NO input at all (``test_the_tui_engages_a_runtime_…``).
2. A viewer that leaves without using the session offers the runtime back, and
   the RUNTIME decides — refusing whenever another viewer is attached or
   anything durable exists. The refusals are the load-bearing half: a wrong
   "retire" ends a session someone is using, which is why every uncertain
   answer here must be "keep".
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.session.runtime.server import RuntimeServer, _ClientConn
from tests.unit.session.runtime.test_server import FakeHandle


class PristineHandle(FakeHandle):
    """A handle that can answer the retirement probe, like a real runtime.

    ``FakeHandle`` deliberately stays as it is — it stands in for a reduced or
    older runtime, and the test below uses it unmodified to pin the
    "cannot judge itself" refusal.
    """

    def __init__(self, *, pristine: bool = True) -> None:
        super().__init__()
        self.pristine = pristine
        self.stopped = False

    def is_pristine(self) -> bool:
        return self.pristine

    def request_stop(self) -> None:
        self.stopped = True


class ExplodingHandle(PristineHandle):
    """A runtime whose pristine probe raises. Uncertainty must keep it alive."""

    def is_pristine(self) -> bool:
        raise RuntimeError("state is unreadable")


def _conn(kind: str) -> _ClientConn:
    """A registered connection of the given kind, with a stand-in writer.

    The real dataclass rather than a look-alike, so the fields ``_on_request``
    reads (``kind``, ``locality``, ``watched_jobs``, ``writer`` as the registry
    key) are the production ones. Nothing is ever written to the socket: the
    test swaps ``_send_to`` for a capture.
    """
    return _ClientConn(writer=cast(Any, object()), kind=cast(Any, kind))


async def _retire(server: RuntimeServer, conn: _ClientConn) -> str:
    """Drive the op the way a viewer does and return the ack detail."""
    sent: list[dict[str, Any]] = []

    async def capture(target, frame):  # noqa: ANN001
        sent.append(frame)

    async def noop_broadcast(frame):  # noqa: ANN001
        return None

    server._send_to = capture  # type: ignore[assignment]
    server._broadcast = noop_broadcast  # type: ignore[assignment]
    await server._on_request({"op": "retire_if_pristine", "req": 1}, conn)
    assert sent, "the op never replied"
    reply = sent[-1]
    assert reply.get("op") == "ack", f"unexpected reply: {reply}"
    return str(reply.get("detail", ""))


def _register(server: RuntimeServer, conn: _ClientConn) -> None:
    server._clients[id(conn.writer)] = conn


@pytest.mark.asyncio
async def test_a_pristine_unobserved_runtime_retires() -> None:
    """The case eager start creates: opened, never used, viewer leaving."""
    handle = PristineHandle(pristine=True)
    server = RuntimeServer(handle, kind="tui")
    leaving = _conn("attach")
    _register(server, leaving)

    detail = await _retire(server, leaving)

    assert detail == "retired"
    assert handle.stopped is True, "a pristine unobserved runtime must stop"


@pytest.mark.asyncio
async def test_a_runtime_with_history_is_kept() -> None:
    """Not-pristine outranks everything: a real conversation is never dropped."""
    handle = PristineHandle(pristine=False)
    server = RuntimeServer(handle, kind="tui")
    leaving = _conn("attach")
    _register(server, leaving)

    detail = await _retire(server, leaving)

    assert detail == "kept: session has work or history"
    assert handle.stopped is False


@pytest.mark.asyncio
async def test_a_runtime_another_viewer_is_watching_is_kept() -> None:
    """The forgotten-TUI case the retirement exists to NOT break.

    A session left open for hours stays pristine forever, so emptiness alone
    would happily retire the runtime behind a second terminal that is still
    attached — and that terminal is exactly where a later instruction arrives.
    """
    handle = PristineHandle(pristine=True)
    server = RuntimeServer(handle, kind="tui")
    leaving = _conn("attach")
    observer = _conn("attach")
    _register(server, leaving)
    _register(server, observer)

    detail = await _retire(server, leaving)

    assert detail == "kept: 1 viewer(s) still attached"
    assert handle.stopped is False, "a runtime under observation must survive"


@pytest.mark.asyncio
async def test_the_leaving_viewer_does_not_count_as_its_own_observer() -> None:
    """The off-by-one that would make every retirement refuse itself.

    The leaving viewer's connection is still registered while its op is
    dispatched, so counting it would make the observer term never reach zero.
    """
    handle = PristineHandle(pristine=True)
    server = RuntimeServer(handle, kind="tui")
    leaving = _conn("attach")
    _register(server, leaving)

    assert server.attach_clients() == 1, "the leaving viewer is still registered"
    detail = await _retire(server, leaving)

    assert detail == "retired"


@pytest.mark.asyncio
async def test_a_daemon_connection_does_not_hold_a_runtime_open() -> None:
    """``daemon`` clients are not attention, exactly as the reaper reads them.

    The mobile daemon adopts EVERY session on the machine, so counting its
    connection would mean nothing is ever retired on a machine running
    ``lop mobile``.
    """
    handle = PristineHandle(pristine=True)
    server = RuntimeServer(handle, kind="tui")
    leaving = _conn("attach")
    daemon = _conn("daemon")
    _register(server, leaving)
    _register(server, daemon)

    detail = await _retire(server, leaving)

    assert detail == "retired"


@pytest.mark.asyncio
async def test_an_unreadable_pristine_probe_keeps_the_runtime() -> None:
    """Uncertainty is never a licence to stop a session."""
    handle = ExplodingHandle()
    server = RuntimeServer(handle, kind="tui")
    leaving = _conn("attach")
    _register(server, leaving)

    detail = await _retire(server, leaving)

    assert detail.startswith("kept: pristine probe failed")
    assert handle.stopped is False


@pytest.mark.asyncio
async def test_a_runtime_that_cannot_judge_itself_is_kept() -> None:
    """An older runtime, or a reduced handle, is left to the residency drain."""
    handle = FakeHandle()  # no is_pristine
    server = RuntimeServer(handle, kind="tui")
    leaving = _conn("attach")
    _register(server, leaving)

    detail = await _retire(server, leaving)

    assert detail == "kept: this runtime cannot judge itself pristine"


@pytest.mark.asyncio
async def test_is_pristine_reads_durable_rows_not_the_model_window(
    tmp_path: Path,
) -> None:
    """A compacted conversation is idle, and emphatically not disposable.

    ``is_busy`` answers "may this exit later", which a finished conversation
    satisfies. ``is_pristine`` must answer "did this session ever exist", so it
    reads the durable transcript rather than the model-facing history that
    compaction shrinks.
    """
    from local_operator.harness.types import Message
    from local_operator.session.runtime.owned import OwnedSessionHandle
    from local_operator.session.transcript import Transcript

    directory = tmp_path / "sessions" / "s1"
    directory.mkdir(parents=True)
    transcript = Transcript(directory)
    await transcript.append_message(Message.user("we talked about something"))

    class _Session:
        def __init__(self) -> None:
            self._transcript = transcript
            self.wake_scheduler = None

        def history(self):  # noqa: ANN202 — compaction emptied the window
            return []

    handle = object.__new__(OwnedSessionHandle)
    handle._session = _Session()  # type: ignore[attr-defined]

    # is_busy is stubbed: the point here is the DURABLE probe, and the real
    # one reads a dozen session internals this reduced double does not have.
    object.__setattr__(handle, "is_busy", lambda: False)

    assert handle.is_pristine() is False, "a transcript row means the session is real"


@pytest.mark.asyncio
async def test_the_tui_engages_a_runtime_without_any_input(tmp_path: Path, monkeypatch) -> None:
    """The reported bug, end to end: no keystroke, and a runtime starts anyway.

    Driven through the real ``OperatorApp`` boot path rather than by calling
    the helper, because the regression being pinned is one of ORDERING — the
    engage has to be reached from boot, after adoption, with nothing typed.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = tmp_path / "sessions" / "s1"
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")

    from local_operator.session.remote import RemoteSession
    from local_operator.tui.app import OperatorApp

    engaged = asyncio.Event()

    async def fake_engage(session_id, cwd, work, *, config_dir, deadline_s=30.0):  # noqa: ANN001
        engaged.set()
        raise ConnectionError("no runtime in this test")

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", fake_engage)

    async def _never():
        raise AssertionError("takeover was not expected")

    viewer = await RemoteSession.cold(
        "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )

    async def factory():
        return viewer

    app = OperatorApp(factory)
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(150):
                await pilot.pause()
                if engaged.is_set():
                    break
            assert engaged.is_set(), "the TUI never engaged a runtime without input"
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_leaving_a_session_offers_its_runtime_back(tmp_path: Path) -> None:
    """``RemoteSession.retire_if_unused`` asks, and reports what it was told."""
    from local_operator.session.remote import RemoteSession

    viewer = object.__new__(RemoteSession)

    class _Client:
        connected = True

        def __init__(self) -> None:
            self.asked = 0

        async def retire_if_pristine(self) -> str:
            self.asked += 1
            return "retired"

    client = _Client()
    viewer._client = client  # type: ignore[attr-defined]

    assert await viewer.retire_if_unused() == "retired"
    assert client.asked == 1


@pytest.mark.asyncio
async def test_a_cold_viewer_has_no_runtime_to_offer_back() -> None:
    """Quitting a viewer that never engaged must not raise on the way out."""
    from local_operator.session.remote import RemoteSession

    viewer = object.__new__(RemoteSession)
    viewer._client = None  # type: ignore[attr-defined]

    assert await viewer.retire_if_unused() == "no runtime attached"


@pytest.mark.asyncio
async def test_a_failed_offer_is_swallowed_on_the_way_out() -> None:
    """Teardown must not fail over a courtesy the residency drain also covers."""
    from local_operator.session.remote import RemoteSession

    viewer = object.__new__(RemoteSession)

    class _Client:
        connected = True

        async def retire_if_pristine(self) -> str:
            raise ConnectionError("socket already gone")

    viewer._client = _Client()  # type: ignore[attr-defined]

    assert (await viewer.retire_if_unused()).startswith("request failed:")
