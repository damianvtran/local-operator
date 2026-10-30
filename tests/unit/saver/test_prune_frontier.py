"""While Radient Saver is on, the session's prune sweeps act only past the frontier (contract §7).

Rows already frozen into epochs are not on the wire, so blanking one saves
nothing and costs a chain truncation plus a re-upload. The sweep runs on the
session's own render, which is not index-aligned with the request the chain
counted, so the restriction is by message identity (``SaverChain.frozen_ids``).
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from local_operator.harness.types import Message, TextContent
from local_operator.session.session import Session


def _history(n: int) -> list[Message]:
    return [
        Message(
            id=f"m{i}",
            role="user" if i % 2 == 0 else "assistant",
            content=[TextContent(text=f"t{i}")],
        )
        for i in range(n)
    ]


class _Host:
    """The attributes ``_sweep_stale_outputs`` and ``_saver_frontier`` read."""

    _last_provider_request_ms = 0

    def __init__(self, frozen: frozenset[str] | None) -> None:
        chain = types.SimpleNamespace(frozen_ids=frozen) if frozen is not None else None
        self._stream_fn = types.SimpleNamespace(_saver_chain=chain)
        self.journalled: list[Any] = []

    async def _journal_prunes(self, history, before) -> None:
        self.journalled.append(before)

    _saver_frontier = Session._saver_frontier
    _sweep_stale_outputs = Session._sweep_stale_outputs


def _api(seen: list[list[str]]) -> Any:
    def prune_tool_outputs(messages, now_ms, last_ms, **kwargs):
        seen.append([m.id for m in messages])

    return types.SimpleNamespace(prune_tool_outputs=prune_tool_outputs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frozen,expected_first",
    [
        (None, "m0"),  # Saver never ran: the whole history, as today
        (frozenset(), "m0"),  # Saver on, nothing frozen yet
        (frozenset({"m0", "m1", "m2", "m3"}), "m4"),  # sweep starts past the frontier
        (frozenset({"m0", "m1", "x-not-in-render"}), "m2"),  # by identity, not index
    ],
)
async def test_prune_starts_at_the_frontier(frozen, expected_first) -> None:
    host = _Host(frozen)
    seen: list[list[str]] = []
    history = _history(10)
    await host._sweep_stale_outputs(_api(seen), None, history)
    assert seen and seen[0][0] == expected_first
    assert seen[0][-1] == "m9"
