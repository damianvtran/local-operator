"""H2 types: cursor separation, strict projections, no-content law."""

from __future__ import annotations

import pytest

from local_operator.team_bridge.types import (
    TeamBridgeTypeError,
    feed_seq,
    inbox_cursor,
    project_inbox_page,
)


def test_cursors_are_not_interchangeable() -> None:
    assert inbox_cursor("abc") == "abc"
    assert feed_seq(4) == 4
    with pytest.raises(TeamBridgeTypeError):
        feed_seq("abc")  # type: ignore[arg-type]
    with pytest.raises(TeamBridgeTypeError):
        inbox_cursor(4)  # type: ignore[arg-type]


def test_projection_refuses_missing_ids_and_keeps_no_body() -> None:
    with pytest.raises(TeamBridgeTypeError):
        project_inbox_page({})
    page = project_inbox_page(
        {
            "items": [
                {
                    "message": {
                        "id": "bmsg_1",
                        "thread_id": "bth_1",
                        "tenant_id": "tnt_1",
                        "body": {"text": "CANARY"},
                        "body_hash": "b" * 64,
                    },
                    "delivery": {"state": "unread"},
                }
            ],
            "next_cursor": "cursor-2",
            "counts": {"unread": 1},
        }
    )
    assert page.items[0].body_hash == "b" * 64
    assert "CANARY" not in repr(page)
    assert page.next_cursor == "cursor-2"
