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


def _no_load(**_kwargs: Any) -> None:
    return None


def _open(answer: dict[str, Any] | None, *, loader: Any = _no_load) -> TeamBridgeBlock:
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
    # D4: the count is SOURCE lines, and no key hint without focus.
    assert "Show full message · 30 lines" in text
    assert "· m" not in text and "press m" not in text
    assert block._show_more_row == len(text.splitlines()) - 1
    assert block.toggle_show_more() is True
    full = block._build_content(80).plain
    assert "line 29" in full and "Show less" in full
    assert not any(line.endswith("…") for line in full.splitlines()[1:])
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
        (
            _item("unavailable", reason="offline"),
            "Couldn't reach Radient — close and reopen to retry",
        ),
        (_item("unavailable", reason="deleted"), "This message was deleted"),
        (
            _item("held", hold={"class": "transient", "held_for_s": 5}),
            "Waiting for screening — not shown to your agent yet",
        ),
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
    block = TeamBridgeBlock("(no preview available)", {"name": "Ola"}, body_loader=_no_load)
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
    # The local-only mount answer is never final: the first open asks the hub.
    mounted = TeamBridgeBlock("x", {"name": "Ola"}, body_loader=_no_load)
    mounted.apply_card_body(_answer(_item("admitted", body="x", source="local")), verified=False)
    assert mounted._needs_body() is True
    moving = _open(_answer(_item("pending")))
    assert moving._needs_body() is True


def test_the_loader_is_keyed_to_the_sessions_own_journal(tmp_path: Path) -> None:
    directory = tmp_path / "sessions" / "abcdef012345"
    session = SimpleNamespace(_transcript=SimpleNamespace(directory=directory))
    assert card_body_loader(session, "entry1") is not None
    assert card_body_loader(SimpleNamespace(), "entry1") is None
    assert card_body_loader(session, "") is None


# -- round 1 remediation (PR #4238) ------------------------------------------


def _collapsed(answer: dict[str, Any], sender: dict[str, Any] | None = None) -> str:
    block = TeamBridgeBlock(
        "an organization member — (no preview available)",
        {"name": "Ola"} if sender is None else sender,
        body_loader=_no_load,
    )
    block.apply_card_body(answer, verified=False)
    return block._build_content(110).plain.splitlines()[0]


@pytest.mark.parametrize(
    ("item", "header"),
    [
        (_item("admitted", body="**Hello** from Ola", source="local"), "Hello from Ola"),
        (_item("held", hold={"class": "awaiting_review"}), "Held by moderation"),
        (_item("held", hold={"class": "transient"}), "Waiting for screening"),
        (_item("refused"), "Refused by moderation"),
        (_item("pending"), "Waiting to be delivered to your agent"),
        (_item("unavailable", reason="not_loaded"), "Open to load message"),
    ],
)
def test_a_collapsed_card_header_says_what_the_resolver_knows(
    item: dict[str, Any], header: str
) -> None:
    """D1: never "(no preview available)" once anything is known."""
    row = _collapsed(_answer(item))
    assert header in row and "no preview" not in row
    if item.get("hold", {}).get("class") == "transient":
        assert "moderation" not in row  # D3: a delay is not a decision


def test_the_no_name_header_keeps_its_subject() -> None:
    """D2: untrusted text never takes the sender's place."""
    row = _collapsed(_answer(_item("admitted", body="Hello from Ola", source="local")), sender={})
    assert "an organization member — Hello from Ola" in row


def test_mount_runs_the_local_only_pass_and_open_runs_the_full_one() -> None:
    """D1: the mount pass costs no network; the first open consults the hub."""
    calls: list[bool] = []
    block = TeamBridgeBlock("x", {"name": "Ola"}, body_loader=_no_load)
    record = lambda *, local_only: calls.append(local_only)  # noqa: E731
    block._start_load = record  # type: ignore[method-assign]
    block.on_mount()
    block.apply_card_body(_answer(_item("unavailable", reason="not_loaded")), verified=False)
    block._expanded = True
    # An open card says it is loading, not "open to load", while the full pass runs.
    assert "Loading message…" in block._build_content(80).plain
    block._maybe_load_body()
    assert calls == [True, False]


def test_show_more_hint_appears_only_with_focus_and_clamped_row_ends_in_ellipsis() -> None:
    """D4 at a narrow width: stable source count, focus-only key hint, ellipsis."""
    body = "\n".join(f"Line {i}: the quick brown fox jumps over the lazy dog" for i in range(30))
    block = _open(_answer(_item("admitted", body=body, source="local")))
    narrow = block._build_content(60).plain.splitlines()
    assert narrow[block._show_more_row].strip() == "Show full message · 30 lines"
    assert narrow[block._show_more_row - 1].rstrip().endswith("…")
    block.has_focus = True
    focused = block._build_content(60).plain.splitlines()
    assert focused[block._show_more_row].strip() == "Show full message · 30 lines · m"


def test_state_lines_use_the_header_ink_not_dim() -> None:
    """D5: the answer the operator opened the card for is not the faintest text."""
    from local_operator.tui import theme as theme_mod

    content = _open(_answer(_item("refused")))._build_content(80)
    start = content.plain.index("Refused by moderation")
    colours = {
        str(span.style.color.triplet.hex) if span.style.color.triplet else ""
        for span in content.spans
        if span.start <= start < span.end and hasattr(span.style, "color") and span.style.color
    }
    assert theme_mod.semantic_color("muted").lower() in {c.lower() for c in colours}
    assert theme_mod.semantic_color("dim").lower() not in {c.lower() for c in colours}


def test_body_markdown_uses_the_brand_markdown_theme() -> None:
    """D6: inline code takes the theme's ``markdown.code`` ink, not Rich's default."""
    from local_operator.tui import bindings
    from local_operator.tui.widgets.transcript import _markdown_rows

    (row,) = _markdown_rows("see `code` here", 60)
    start = row.plain.index("code")
    expected = bindings.markdown_theme().styles["markdown.code"].color
    painted = [
        span.style.color
        for span in row.spans
        if span.start <= start < span.end and getattr(span.style, "color", None)
    ]
    assert expected is not None and expected in painted
