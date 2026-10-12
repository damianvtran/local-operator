"""H3 A5: serialized RR selector — fairness, request priority, no amplifier."""

from __future__ import annotations

from typing import Any

from local_operator.team_bridge.scheduler import select_next_screening_target


def _item(sender: str, kind: str, ident: int) -> dict[str, Any]:
    return {"sender": {"account_id": sender}, "type": kind, "id": ident}


def test_rotation_requests_first_and_single_target() -> None:
    items = [_item("a", "note", 1), _item("a", "request", 2), _item("b", "note", 3)]
    first = select_next_screening_target(items)
    assert first is not None and first["id"] == 2  # requests first within a
    second = select_next_screening_target(items, last_sender="a")
    assert second is not None and second["id"] == 3
    third = select_next_screening_target(items, last_sender="b")
    assert third is not None and third["id"] == 2


def test_no_starvation_and_empty() -> None:
    items = [_item("a", "note", 1), _item("b", "note", 2)]
    seen: list[Any] = []
    for last in (None, "a", "b"):
        nxt = select_next_screening_target(items, last_sender=last)
        assert nxt is not None
        seen.append(nxt["id"])
    assert seen[0] != seen[1], "one sender can never drain the rotation"
    assert select_next_screening_target([]) is None


def test_malformed_sender_forms_own_slot() -> None:
    items: list[dict[str, Any]] = [
        {"sender": "not-a-mapping", "type": "note", "id": 9},
        _item("a", "note", 1),
    ]
    first = select_next_screening_target(items, last_sender="a")
    assert first is not None and first["id"] == 9  # unknown sender never merges
