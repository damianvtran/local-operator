"""``session.row_clip``: the per-row projection and the session-bound ref.

The route-level behaviour (C4/C4b, the bare page, resync, the rows route) is
pinned in ``tests/unit/server/test_desktop_oversized_rows.py``; this file pins
the helper's own rules where a route test would be too coarse to see them.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest

from local_operator.session.row_clip import (
    ROW_CLIP_MAX_BYTES,
    ROW_RANGE_MAX_BYTES,
    RowRangeUnsatisfiable,
    bound_history_entries,
    canonical_row_bytes,
    clip_row,
    resolve_ref,
    row_range,
    row_ref,
)


def _tool(id_: str, *texts: str) -> dict[str, Any]:
    return {
        "id": id_,
        "ts": 1.0,
        "type": "message",
        "payload": {"kind": "message", "role": "tool", "content": [{"text": t} for t in texts]},
        "ts_source": "entry",
    }


def test_a_row_that_fits_is_returned_unchanged() -> None:
    row = _tool("a", "small")
    assert clip_row(row, session_id="s", reason="history_cap") is row


def test_two_huge_parts_both_keep_their_start_and_small_strings_survive() -> None:
    row = _tool("a", "x" * 12_000_000, "y" * 3_000_000, "short")
    clipped = clip_row(row, session_id="s", reason="history_cap")
    texts = [part["text"] for part in clipped["payload"]["content"]]
    assert texts[2] == "short"
    assert texts[0].startswith("xxx") and texts[1].startswith("yyy")
    marker = clipped["truncated"]
    assert marker["bytes_total"] == len(canonical_row_bytes(row))
    assert ROW_CLIP_MAX_BYTES * 0.9 <= marker["shown_bytes"] <= ROW_CLIP_MAX_BYTES
    projection = {key: value for key, value in clipped.items() if key != "truncated"}
    assert marker["shown_bytes"] == len(canonical_row_bytes(projection))


def test_multibyte_text_is_budgeted_in_bytes_not_characters() -> None:
    """The first draft sized the ceiling in characters and withheld this row."""
    row = _tool("a", "é" * 6_000_000)
    clipped = clip_row(row, session_id="s", reason="history_cap")
    assert "omitted" not in clipped["truncated"]
    assert ROW_CLIP_MAX_BYTES * 0.9 <= clipped["truncated"]["shown_bytes"] <= ROW_CLIP_MAX_BYTES


def test_a_checkpoint_is_withheld_not_cut() -> None:
    row = {
        "id": "c",
        "ts": 2.0,
        "type": "custom",
        "payload": {
            "custom_type": "frontend_state_checkpoint_v1",
            "details": {"s": "z" * 2_000_000},
        },
        "ts_source": "entry",
    }
    clipped = clip_row(row, session_id="s", reason="history_cap")
    assert clipped["payload"] == {"custom_type": "frontend_state_checkpoint_v1"}
    assert clipped["truncated"]["omitted"] is True
    assert {key: clipped[key] for key in ("id", "ts", "type", "ts_source")} == {
        key: row[key] for key in ("id", "ts", "type", "ts_source")
    }


def test_refs_are_bound_to_their_session() -> None:
    ref = row_ref("session-a", "row-1")
    assert resolve_ref("session-a", ref) == "row-1"
    assert resolve_ref("session-b", ref) is None
    assert row_ref("session-b", "row-1") != ref
    # Separator ambiguity cannot make two pairs share a tag.
    assert resolve_ref("a", row_ref("a\x00b", "c")) is None
    for bad in ("row-1", "r1.x", "r1.!!.!!", "r2." + ref[3:], ref[:-2] + "AA"):
        assert resolve_ref("session-a", bad) is None


def test_the_page_bound_keeps_the_newest_rows_and_says_so() -> None:
    rows = [_tool(f"r{n}", "q" * 1_000_000) for n in range(6)]
    kept, capped = bound_history_entries(rows, session_id="s", max_bytes=3_100_000)
    assert capped is True
    assert [row["id"] for row in kept] == ["r3", "r4", "r5"]
    # The newest row is kept even when it alone exceeds the page budget.
    alone, capped_alone = bound_history_entries(rows[-1:], session_id="s", max_bytes=10)
    assert [row["id"] for row in alone] == ["r5"] and capped_alone is False


def test_ranges_reassemble_and_clamp() -> None:
    content = ("é" * 1_500_000).encode()
    parts, offset = [], 0
    while True:
        answer = row_range(content, offset=offset, limit=10**9)
        assert answer["limit"] == ROW_RANGE_MAX_BYTES
        parts.append(base64.b64decode(answer["data"]))
        offset += answer["length"]
        if answer["eof"]:
            break
    assert b"".join(parts) == content
    assert row_range(content, offset=len(content), limit=1)["length"] == 0
    with pytest.raises(RowRangeUnsatisfiable):
        row_range(content, offset=len(content) + 1, limit=1)
