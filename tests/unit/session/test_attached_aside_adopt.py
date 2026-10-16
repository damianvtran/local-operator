"""The attached follower's half of the queued aside adopt (review R7, D4).

Two follower changes ride the queued adopt and each is pinned here on a real
``AttachedSession`` with the socket stubbed:

* ``_track`` claims the row ids of an owner-relayed ``history_delta`` as painted,
  so a reconnect's durable-gap replay does not paint the landed pair twice;
* ``adopt_aside`` returns the owner's ``adopt_state`` decoded off the ack, with
  an absent field (an owner from before the queue) read as ``applied``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import AgentMessage, HistoryDeltaEvent, Message
from local_operator.session.attached import AttachedSession


def _never_take_over(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("no takeover in these cells")


def _follower(tmp_path: Path) -> AttachedSession:
    remote = AttachedSession(
        config_dir=tmp_path, session_id="s1", takeover_factory=_never_take_over
    )
    remote._ready_for_events = True
    return remote


@pytest.mark.asyncio
async def test_a_relayed_history_delta_is_not_painted_again_by_the_gap_replay(
    tmp_path: Path,
) -> None:
    remote = _follower(tmp_path)
    painted: list[list[str]] = []
    remote.subscribe(
        lambda event: (
            painted.append([m.id for m in event.messages])
            if isinstance(event, HistoryDeltaEvent)
            else None
        )
    )
    pair: list[AgentMessage] = [Message.user("why?"), Message.assistant("because.")]

    # The owner lands the queued pair and relays its delta live.
    remote._on_wire_event(HistoryDeltaEvent(messages=pair).model_dump(mode="json"))
    assert painted == [[m.id for m in pair]]

    # A reconnect's durable gap then contains the same rows. The replay QUEUES
    # its delta at the head of the event buffer (it is drained on the next
    # sync), so that buffer is where a second paint would appear.
    remote._replay_durable_suffix(list(pair))
    replayed = [e for e in remote._buffered_events if isinstance(e, HistoryDeltaEvent)]
    await remote.dispose()
    assert replayed == [], "the gap replay would paint the landed pair a second time"
    assert painted == [[m.id for m in pair]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ack", "expected"),
    [("queued", "queued"), ("applied", "applied"), ("anything-else", "applied")],
)
async def test_the_follower_returns_the_owner_adopt_state(
    tmp_path: Path, ack: str, expected: str
) -> None:
    remote = _follower(tmp_path)
    sent: list[list[dict[str, Any]]] = []

    class _Client:
        async def adopt_aside(self, messages: list[dict[str, Any]]) -> str:
            sent.append(messages)
            return ack

    remote._client = _Client()  # type: ignore[assignment]
    pair = [Message.user("q"), Message.assistant("a")]
    assert await remote.adopt_aside(pair) == expected
    assert [m["id"] for m in sent[0]] == [m.id for m in pair]
    remote._client = None
    await remote.dispose()
