"""The cwd-move consumer: consume at an idle boundary, never mid-work.

Fake handle, no RuntimeServer: the risk this pins is the GATE, and the gate is
all it drives. The real server adds the watcher's cadence, not its predicate.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.session import cwd_move
from local_operator.session.runtime.server import RuntimeServer

OWN = "own0000000001"


class _Handle:
    """The production predicates, with the facts spelled out per case."""

    def __init__(
        self,
        *,
        turn: bool = False,
        child: bool = False,
        job: bool = False,
        gate: bool = False,
        wake: bool = False,
        latch: bool = True,
    ) -> None:
        self.turn, self.child, self.job, self.gate, self.wake = turn, child, job, gate, wake
        self.latch = latch
        self.calls: list[str] = []

    def may_refresh(self) -> str:
        if self.turn or self.child or self.job or self.gate:
            return "busy"
        return "wake due within the warm window" if self.wake else ""

    def begin_retire(self, cause: str, detail: str = "") -> bool:
        self.calls.append(f"begin_retire:{cause}")
        return self.latch

    def request_stop(self) -> None:
        self.calls.append("request_stop")


class _Server:
    """Only the attributes the consumer touches; the METHOD is the real one."""

    _apply_queued_cwd_move = RuntimeServer._apply_queued_cwd_move

    def __init__(self, handle: _Handle, root: Path) -> None:
        self._handle = handle
        self._config_root = root
        self._record = SimpleNamespace(session_id=OWN)
        self._queued_move: dict[str, Any] | None = None
        self._queued_move_stage = ""
        self._retirement_committed = False
        self.announced: list[tuple[str, dict[str, Any]]] = []

    async def _handle_call_on_session_loop(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    async def announce_retiring(self, reason: str, **kwargs: Any) -> None:
        self.announced.append((reason, kwargs))


def _queue(root: Path) -> cwd_move.MoveIntent:
    destination = root / "dest"
    destination.mkdir(exist_ok=True)
    return cwd_move.enqueue_self_move(
        root, OWN, requested_cwd=str(destination), effective_cwd=str(root), owner_pid=1
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "facts",
    [
        {"turn": True},
        {"child": True},
        {"job": True},
        {"gate": True},
        {"wake": True},
    ],
    ids=["active-turn", "live-child", "live-job", "parked-gate", "wake-window"],
)
async def test_busy_keeps_the_intent_queued_and_touches_nothing(
    tmp_path: Path, facts: dict[str, bool]
) -> None:
    intent = _queue(tmp_path)
    handle = _Handle(**facts)
    server = _Server(handle, tmp_path)
    assert await server._apply_queued_cwd_move() == "waiting"
    assert handle.calls == [] and server.announced == []
    stored = cwd_move.read_move_intent(tmp_path, OWN)
    assert stored is not None and stored.stage == cwd_move.STAGE_QUEUED
    assert server._queued_move_stage == ""
    assert intent.request_id == stored.request_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setup",
    [
        lambda s: setattr(s, "_queued_move_stage", "finishing"),
        lambda s: setattr(s, "_retirement_committed", True),
        lambda s: setattr(s, "_queued_move", {"to_device": "peer-1", "to_name": "peer-1"}),
    ],
    ids=["stage-held", "already-retiring", "mesh-move-latched"],
)
async def test_a_runtime_already_moving_does_not_start_a_second_move(tmp_path: Path, setup) -> None:
    _queue(tmp_path)
    handle = _Handle()
    server = _Server(handle, tmp_path)
    setup(server)
    assert await server._apply_queued_cwd_move() == ""
    assert handle.calls == [] and server.announced == []


@pytest.mark.asyncio
async def test_idle_cutover_latches_announces_and_stops_in_that_order(tmp_path: Path) -> None:
    intent = _queue(tmp_path)
    handle = _Handle()
    server = _Server(handle, tmp_path)
    assert await server._apply_queued_cwd_move() == "retiring"
    assert handle.calls == ["begin_retire:cwd-move", "request_stop"]
    assert server.announced == [("cwd-move", {"cwd": intent.destination_cwd})]
    assert server._queued_move_stage == "retiring"
    # no peer ever rides a cwd move
    assert all("to" not in kwargs for _reason, kwargs in server.announced)


@pytest.mark.asyncio
async def test_a_refused_latch_keeps_the_intent_queued(tmp_path: Path) -> None:
    _queue(tmp_path)
    handle = _Handle(latch=False)
    server = _Server(handle, tmp_path)
    assert await server._apply_queued_cwd_move() == "waiting"
    assert server.announced == []
    stored = cwd_move.read_move_intent(tmp_path, OWN)
    assert stored is not None and stored.stage == cwd_move.STAGE_QUEUED


def test_no_intent_means_nothing_to_do(tmp_path: Path) -> None:
    import asyncio

    handle = _Handle()
    server = _Server(handle, tmp_path)
    assert asyncio.run(server._apply_queued_cwd_move()) == ""
    assert handle.calls == []


@pytest.mark.asyncio
async def test_identical_repeat_is_the_same_request_and_a_new_destination_supersedes(
    tmp_path: Path,
) -> None:
    first = _queue(tmp_path)
    again = _queue(tmp_path)
    assert again.request_id == first.request_id, "a retried tool call is one intent"
    other = tmp_path / "other"
    other.mkdir()
    second = cwd_move.enqueue_self_move(
        tmp_path, OWN, requested_cwd=str(other), effective_cwd=str(tmp_path), owner_pid=1
    )
    assert second.request_id != first.request_id
    assert second.stage == cwd_move.STAGE_QUEUED
    with pytest.raises(cwd_move.CwdMoveRefused) as refused:
        cwd_move.commit_move(tmp_path, OWN, first.request_id, effective_cwd=str(tmp_path))
    assert refused.value.code == "sessions_move_request_unknown"


def test_the_retiring_frame_carries_cwd_additively() -> None:
    import inspect

    source = inspect.getsource(RuntimeServer._announce_retiring_on_loop)
    assert '"cwd": cwd,' in source
    assert 'cwd: str = "",' in inspect.getsource(RuntimeServer.announce_retiring)
