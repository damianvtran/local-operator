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
    wire_json,
    wire_size,
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
    # The budget is spent in wire bytes; ``shown_bytes`` reports UTF-8 (R2-1).
    projection = {key: value for key, value in clipped.items() if key != "truncated"}
    assert ROW_CLIP_MAX_BYTES * 0.9 <= wire_size(projection) <= ROW_CLIP_MAX_BYTES
    assert clipped["truncated"]["shown_bytes"] == len(canonical_row_bytes(projection))


@pytest.mark.parametrize("char", ["x", "ж", "😀"])
def test_shown_bytes_is_utf8_like_bytes_total(char: str) -> None:
    """R2-1: ``shown_bytes`` counted wire bytes while ``bytes_total`` counted UTF-8.

    On 56b696259 a clipped 400k-character Cyrillic row reported
    ``shown_bytes`` 1,572,687 against ``bytes_total`` 800,123 — showing ~2x
    MORE than the whole row. Both now count UTF-8, so shown is a true
    fraction of total; ``x`` is the ASCII control, where the units coincide.
    """
    row = _tool("a", char * 400_000) if char != "x" else _tool("a", char * 3_000_000)
    clipped = clip_row(row, session_id="s", reason="history_cap")
    marker = clipped["truncated"]
    assert "omitted" not in marker
    projection = {key: value for key, value in clipped.items() if key != "truncated"}
    assert marker["shown_bytes"] == len(canonical_row_bytes(projection))
    assert marker["bytes_total"] == len(canonical_row_bytes(row))
    assert marker["shown_bytes"] < marker["bytes_total"]
    # The budget itself is still enforced on the wire spelling.
    assert wire_size(projection) <= ROW_CLIP_MAX_BYTES


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
    kept, capped, capped_newer = bound_history_entries(rows, session_id="s", max_bytes=3_100_000)
    assert capped is True and capped_newer is False
    assert [row["id"] for row in kept] == ["r3", "r4", "r5"]
    # The newest row is kept even when it alone exceeds the page budget.
    alone, capped_alone, _ = bound_history_entries(rows[-1:], session_id="s", max_bytes=10)
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


#: The desktop renderer's per-record IPC cap (local-operator-ui), restated so a
#: drifting core constant cannot make these pass.
UI_RECORD_CAP = 8_388_608


def _sse_record_bytes(frame: Any) -> int:
    # Measured THROUGH the real serializer, not a re-derivation of it (R1-1).
    from local_operator.server.routes.desktop_sessions import _sse_data

    return len(_sse_data(frame).encode("utf-8"))


@pytest.mark.parametrize("char", ["ж", "€", "😀"])
def test_budgets_count_sse_wire_bytes_not_utf8(char: str) -> None:
    """R1-1: a non-Latin page bounded in UTF-8 was ~3x its bound on the SSE wire.

    On 2a397de74 twelve 200k-character Cyrillic rows passed the 4 MiB page
    bound at ~4 MB UTF-8 and became a ~12 MB SSE record (reviewer measured
    4,000,401 B UTF-8 vs 12,000,401 B wire), over the UI's 8 MiB cap.
    """
    rows = [_tool(f"r{n}", char * 200_000) for n in range(12)]
    kept, capped, _ = bound_history_entries(rows, session_id="s")
    assert capped is True
    frame = {"type": "snapshot", "payload": {"history": {"entries": kept}}}
    assert _sse_record_bytes(frame) <= UI_RECORD_CAP
    assert sum(wire_size(row) for row in kept) <= 4 * 1024 * 1024
    # And one oversized row alone, clipped, fits the per-row budget on the wire.
    clipped = clip_row(_tool("big", char * 2_000_000), session_id="s", reason="history_cap")
    assert _sse_record_bytes(clipped) - len("data: \n\n") <= ROW_CLIP_MAX_BYTES
    assert "omitted" not in clipped["truncated"]
    assert clipped["payload"]["content"][0]["text"].startswith(char * 10)
    # ``bytes_total`` still counts what the ref reassembles to: UTF-8.
    assert clipped["truncated"]["bytes_total"] == len(
        canonical_row_bytes(_tool("big", char * 2_000_000))
    )


def test_wire_json_is_the_sse_spelling() -> None:
    from local_operator.server.routes.desktop_sessions import _sse_data

    frame = {"a": "ж😀", "b": [1, 2]}
    assert _sse_data(frame) == "data: " + wire_json(frame) + "\n\n"
    assert wire_json(frame).isascii()


@pytest.mark.parametrize(
    "row",
    [
        _tool("ascii", "x" * 3_000_000),
        _tool("cyrillic", "ж" * 1_000_000),
        _tool("emoji", "😀" * 500_000),
        {
            "id": "ckpt",
            "ts": 2.0,
            "type": "custom",
            "payload": {"custom_type": "frontend_state_checkpoint_v1", "d": "z" * 2_000_000},
            "ts_source": "entry",
        },
    ],
    ids=["ascii", "cyrillic", "emoji", "withheld"],
)
def test_the_marker_fits_inside_the_row_budget(row: dict[str, Any]) -> None:
    """Q1: a clipped row plus its marker was 1,572,990 B, 126 B over 1,572,864."""
    clipped = clip_row(row, session_id="session-with-a-long-id", reason="open_frame_cap")
    assert "truncated" in clipped
    assert wire_size(clipped) <= ROW_CLIP_MAX_BYTES
    if "omitted" not in clipped["truncated"]:
        # Still spends the budget: the reserve is a marker, not a margin.
        assert wire_size(clipped) >= ROW_CLIP_MAX_BYTES * 0.9


def test_a_range_boundary_inside_a_multibyte_character_rejoins_exactly() -> None:
    """R1-2: 3-byte and 4-byte characters each split across ranges."""
    content = canonical_row_bytes(_tool("m", "a€b😀c"))
    euro = content.index("€".encode())
    emoji = content.index("😀".encode())
    parts: list[bytes] = []
    # Cut once inside the 3-byte character and once inside the 4-byte one.
    cuts = [0, euro + 1, emoji + 2, len(content)]
    for start, stop in zip(cuts, cuts[1:]):
        answer = row_range(content, offset=start, limit=stop - start)
        parts.append(base64.b64decode(answer["data"]))
    assert answer["eof"] is True
    # The cuts really do split characters: a range alone is not valid UTF-8.
    for broken in parts[:2]:
        with pytest.raises(UnicodeDecodeError):
            broken.decode("utf-8")
    joined = b"".join(parts)
    assert joined == content
    assert joined.decode("utf-8") == content.decode("utf-8")


def test_an_anchored_page_keeps_its_anchor() -> None:
    """R1-5: newer rows after the anchor used to spend the budget and drop it."""
    rows = [_tool(f"r{n}", "q" * 1_000_000) for n in range(8)]
    kept, capped_older, capped_newer = bound_history_entries(
        rows, session_id="s", max_bytes=3_100_000, anchor_id="r1"
    )
    ids = [row["id"] for row in kept]
    assert "r1" in ids
    assert ids == ["r0", "r1", "r2"]
    assert capped_older is False and capped_newer is True
    # Without the anchor the same page is a tail read and loses r1.
    tail, _, _ = bound_history_entries(rows, session_id="s", max_bytes=3_100_000)
    assert "r1" not in [row["id"] for row in tail]
