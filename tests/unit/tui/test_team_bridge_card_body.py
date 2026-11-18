"""The TUI Team Bridge card paints the resolved body inline (bridge-body PR 2).

Widget-level: the content a card builds is a pure function of its state, so
these cells drive ``apply_card_body`` with resolver answers and read the built
text. Rendered-frame evidence lives on the PR (AGENTS.md visual validation).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.tui.bridge_card_body import card_body_loader
from local_operator.tui.widgets.transcript import TeamBridgeBlock


def _answer(*items: dict[str, Any], more: int = 0) -> dict[str, Any]:
    return {"v": 1, "items": list(items), "more": more}


def _item(state: str, **extra: Any) -> dict[str, Any]:
    return {"message_id": "bcmsg_1", "conversation_id": "bcv_12345678", "state": state, **extra}


def _open(answer: dict[str, Any] | None, *, loader: Any = lambda: None) -> TeamBridgeBlock:
    block = TeamBridgeBlock("(no preview available)", {"name": "Ola"}, body_loader=loader)
    block._expanded = True
    block.apply_card_body(answer)
    return block


def test_without_a_loader_the_card_is_byte_identical_to_before() -> None:
    plain = TeamBridgeBlock("(no preview available)", {"name": "Ola"})
    plain._expanded = True
    assert "untrusted content" not in plain._build_content(80).plain
    assert "(no preview available)" in plain._build_content(80).plain


def test_an_admitted_body_renders_as_markdown_under_the_untrusted_frame() -> None:
    block = _open(_answer(_item("admitted", body="**Bold** and `code`", source="local")))
    text = block._build_content(80).plain
    lines = text.splitlines()
    frame = next(
        i for i, line in enumerate(lines) if "External message · untrusted content" in line
    )
    assert "Bold and code" in lines[frame + 1], text
    assert "**" not in lines[frame + 1]
    # The collapsed preview now carries the admitted text, not the receipt line.
    assert "(no preview available)" not in lines[0]


def test_a_long_body_clamps_at_twelve_rows_with_show_more_and_m_toggles() -> None:
    body = "\n".join(f"line {i}" for i in range(30))
    block = _open(_answer(_item("admitted", body=body, source="hub")))
    text = block._build_content(80).plain
    assert "line 11" in text and "line 12" not in text
    assert "Show more · 12 of" in text
    assert block._show_more_row == len(text.splitlines()) - 1
    assert block.toggle_show_more() is True
    full = block._build_content(80).plain
    assert "line 29" in full and "Show less" in full
    block.toggle_show_more()
    assert "line 12" not in block._build_content(80).plain


def test_a_short_body_claims_no_show_more_and_m_stays_with_the_composer() -> None:
    block = _open(_answer(_item("admitted", body="short", source="local")))
    block._build_content(80)
    assert block._show_more_row == -1
    assert block.toggle_show_more() is False


@pytest.mark.parametrize(
    ("item", "copy"),
    [
        (_item("held", hold={"class": "awaiting_review", "held_for_s": 5}), "Held by moderation"),
        (_item("refused"), "Refused by moderation"),
        (_item("pending"), "Waiting to be delivered to your agent"),
        (_item("not_admitted"), "Not delivered to your agent"),
        (_item("unavailable", reason="signed_out"), "Sign in to Radient to see this message"),
        (_item("unavailable", reason="other_account"), "Received under another Radient account"),
        (_item("unavailable", reason="not_member"), "You are no longer in this conversation"),
        (_item("unavailable", reason="offline"), "Couldn't reach Radient"),
        (_item("unavailable", reason="hub_unsupported"), "Message body unavailable"),
    ],
)
def test_every_state_says_what_happened(item: dict[str, Any], copy: str) -> None:
    text = _open(_answer(item))._build_content(80).plain
    assert copy in text
    # Never a reason/category beyond the closed sentence.
    assert "awaiting_review" not in text


def test_a_coalesced_card_counts_the_messages_it_does_not_show() -> None:
    text = (
        _open(_answer(_item("admitted", body="first", source="local"), _item("pending"), more=4))
        ._build_content(80)
        .plain
    )
    assert "first" in text and "Waiting to be delivered" in text
    assert "4 more messages in this card" in text


def test_loading_then_an_empty_answer_falls_back_to_todays_copy() -> None:
    block = TeamBridgeBlock("(no preview available)", {"name": "Ola"}, body_loader=lambda: None)
    block._expanded = True
    assert "Loading message…" in block._build_content(80).plain
    block.apply_card_body(None)
    assert "(no preview available)" in block._build_content(80).plain


def test_a_hostile_body_cannot_drive_the_terminal() -> None:
    text = _open(
        _answer(
            _item("admitted", body="hi \x1b[2J\x1b]8;;https://x\x07click\x1b]8;;\x07", source="hub")
        )
    )._build_content(80)
    assert "\x1b" not in text.plain and "\x07" not in text.plain


def test_refetch_only_while_a_state_can_still_change() -> None:
    final = _open(_answer(_item("admitted", body="x", source="local")))
    assert final._needs_body() is False
    moving = _open(_answer(_item("pending")))
    assert moving._needs_body() is True


def test_the_loader_is_keyed_to_the_sessions_own_journal(tmp_path: Path) -> None:
    directory = tmp_path / "sessions" / "abcdef012345"
    session = SimpleNamespace(_transcript=SimpleNamespace(directory=directory))
    assert card_body_loader(session, "entry1") is not None
    assert card_body_loader(SimpleNamespace(), "entry1") is None
    assert card_body_loader(session, "") is None
