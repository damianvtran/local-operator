"""``fold``'s one-pass event index answers exactly what the linear helpers did.

WHY. ``fold`` used to call ``_first``/``_last`` (each a full pass over the log)
up to eight times per ask: O(asks x events), ~28 ms per fold on a real 522-event
log, on every queue read and refresh. ``_EventIndex`` builds the same answers in
one pass. This parity test runs ``fold`` twice over randomized logs — once with
the index, once with a reference index that delegates to the ORIGINAL helpers
(still in the module for their other callers) — and requires identical output,
so any drift in ordering, ``str(ask_id)`` coercion or copy semantics fails here.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from local_operator.asks import store

KINDS = [
    store.EVENT_QUEUED,
    store.EVENT_ANSWERED,
    store.EVENT_DECLINED,
    store.EVENT_DISMISSED,
    store.EVENT_REVISED,
    store.EVENT_WITHDRAWN,
    store.EVENT_RELAYED,
    store.EVENT_RELAY_SURFACED,
    store.EVENT_CONTEXT,
    store.EVENT_CONSENT_USED,
]


class _LinearIndex:
    """The pre-index behaviour: every lookup is a linear helper call."""

    def __init__(self, events: Sequence[Mapping[str, Any]]) -> None:
        self._events = events

    def first(self, ask_id: str, kind: str) -> dict[str, Any] | None:
        return store._first(self._events, ask_id, kind)

    def last(self, ask_id: str, kind: str) -> dict[str, Any] | None:
        return store._last(self._events, ask_id, kind)


def _random_log(rng: random.Random, now: int) -> list[dict[str, Any]]:
    # Ids include an int and a numeric string so the ``str(ask_id)`` coercion
    # is exercised in both directions.
    ids: list[Any] = [f"a{i}" for i in range(rng.randint(1, 8))] + [7, "7"]
    events: list[dict[str, Any]] = []
    for seq in range(rng.randint(0, 60)):
        ask_id = rng.choice(ids)
        kind = rng.choice(KINDS)
        at = now + rng.randint(-3_600_000, 3_600_000)
        event: dict[str, Any] = {"kind": kind, "ask_id": ask_id, "at": at, "seq": seq}
        if kind == store.EVENT_QUEUED:
            event.update(
                expires_at=at + rng.choice([-1, 1]) * rng.randint(0, 7_200_000),
                timeout_s=rng.randint(0, 600),
                urgent=rng.random() < 0.3,
                tool_call_id=f"tc{seq}",
                questions=[{"id": "q1", "question": "?"}]
                + ([{"id": "q2", "question_id": "bq"}] if rng.random() < 0.3 else []),
            )
            if rng.random() < 0.3:
                event["consent"] = {"scope": "x"}
        elif kind == store.EVENT_ANSWERED:
            event["answers"] = {"q1": f"pick-{seq}"}
        elif kind == store.EVENT_REVISED:
            event["answers"] = {"q1": f"rev-{seq}"}
        elif kind == store.EVENT_RELAYED:
            event["relay"] = {"session_id": "origin", "ask_id": f"o{seq}"}
        elif kind == store.EVENT_RELAY_SURFACED:
            event["by"] = {"session_id": "w", "ask_id": f"w{seq}"}
        elif kind == store.EVENT_CONTEXT:
            event.update(title=f"t{seq}", body="b")
        elif kind == store.EVENT_WITHDRAWN and rng.random() < 0.5:
            event["by"] = {"surface": "desktop"}
        events.append(event)
    return events


@pytest.mark.parametrize("seed", range(300))
def test_indexed_fold_matches_the_linear_helpers(seed: int, monkeypatch) -> None:
    rng = random.Random(seed)
    now = 1_800_000_000_000
    events = _random_log(rng, now)
    present = {f"ask-response-a{i}" for i in range(3)}

    indexed = store.fold(events, now, present_ids=present, session_id="s")
    monkeypatch.setattr(store, "_EventIndex", _LinearIndex)
    linear = store.fold(events, now, present_ids=present, session_id="s")
    assert indexed == linear


def test_the_index_answers_first_last_and_copies() -> None:
    events = [
        {"kind": store.EVENT_REVISED, "ask_id": "a", "n": 1},
        {"kind": store.EVENT_REVISED, "ask_id": "a", "n": 2},
        {"kind": store.EVENT_REVISED, "ask_id": 5, "n": 3},
        {"kind": ["unhashable"], "ask_id": "a", "n": 4},
    ]
    index = store._EventIndex(events)
    first = index.first("a", store.EVENT_REVISED)
    assert first == {"kind": store.EVENT_REVISED, "ask_id": "a", "n": 1}
    assert index.last("a", store.EVENT_REVISED)["n"] == 2  # type: ignore[index]
    assert index.first("5", store.EVENT_REVISED)["n"] == 3  # type: ignore[index]
    assert index.first("a", store.EVENT_ANSWERED) is None
    # A fresh copy each time, as the helpers return: mutation never leaks.
    assert first is not None
    first["n"] = 99
    assert index.first("a", store.EVENT_REVISED)["n"] == 1  # type: ignore[index]
    assert events[0]["n"] == 1
