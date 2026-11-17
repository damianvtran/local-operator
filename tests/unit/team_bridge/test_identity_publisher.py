"""The connector's rename publisher: event-driven, retried, stoppable.

Report 2026-11-17: a renamed chief of staff kept her old name on the hub until
the next fresh status read. ``identity_publisher.run`` publishes at start and on
every ``aida.name`` / ``team_bridge.session`` change, retrying a publish that did
not settle.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from local_operator.team_bridge.connector import identity_publisher

pytestmark = pytest.mark.asyncio


@dataclass
class _Change:
    changed_keys: frozenset[str]


class _Watcher:
    def __init__(self) -> None:
        self.listeners: list[Callable[[Any], None]] = []
        self.started = False

    def subscribe(self, listener: Callable[[Any], None]) -> Callable[[], None]:
        self.listeners.append(listener)
        return lambda: self.listeners.remove(listener)

    def start(self) -> None:
        self.started = True

    def emit(self, *keys: str) -> None:
        for listener in list(self.listeners):
            listener(_Change(frozenset(keys)))


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


async def _run(tmp_path: Path, answers: list[bool], delays=(0.01, 0.01)):
    calls: list[int] = []

    async def publish() -> bool:
        calls.append(1)
        return answers.pop(0) if answers else True

    watcher = _Watcher()
    stop = asyncio.Event()
    task = asyncio.create_task(
        identity_publisher.run(
            tmp_path,
            base_url="https://hub.invalid",
            stop=stop,
            publish=publish,
            watcher=watcher,
            retry_delays=delays,
        )
    )
    await _settle()
    return calls, watcher, stop, task


async def test_publishes_at_start_and_on_an_identity_change_only(tmp_path: Path):
    calls, watcher, stop, task = await _run(tmp_path, [])
    assert watcher.started and len(calls) == 1

    watcher.emit("compaction.threshold_percent")
    await _settle()
    assert len(calls) == 1

    watcher.emit("aida.name")
    await _settle()
    assert len(calls) == 2

    watcher.emit("team_bridge.session", "display.theme")
    await _settle()
    assert len(calls) == 3

    stop.set()
    await asyncio.wait_for(task, 1)
    assert watcher.listeners == []


async def test_an_unsettled_publish_is_retried_then_left_to_the_status_read(tmp_path: Path):
    calls, _watcher, stop, task = await _run(tmp_path, [False, False, False, False])
    await asyncio.sleep(0.1)
    # One attempt plus one per retry delay, then it stops trying.
    assert len(calls) == 3
    stop.set()
    await asyncio.wait_for(task, 1)


async def test_a_retry_stops_once_settled(tmp_path: Path):
    calls, _watcher, stop, task = await _run(tmp_path, [False, True])
    await asyncio.sleep(0.1)
    assert len(calls) == 2
    stop.set()
    await asyncio.wait_for(task, 1)


async def test_a_raising_publish_never_kills_the_publisher(tmp_path: Path):
    calls: list[int] = []

    async def publish() -> bool:
        calls.append(1)
        raise RuntimeError("boom")

    watcher = _Watcher()
    stop = asyncio.Event()
    task = asyncio.create_task(
        identity_publisher.run(
            tmp_path, base_url="x", stop=stop, publish=publish, watcher=watcher, retry_delays=()
        )
    )
    await _settle()
    watcher.emit("aida.name")
    await _settle()
    assert len(calls) == 2 and not task.done()
    stop.set()
    await asyncio.wait_for(task, 1)
