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
from rich.style import Style

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
        (_item("held", hold={"class": "awaiting_review", "held_for_s": 5}), "Held for review"),
        (_item("refused"), "Refused by moderation"),
        (_item("pending"), "Waiting to be delivered to your agent"),
        (_item("not_admitted"), "Not delivered to your agent"),
        (_item("unavailable", reason="signed_out"), "Sign in to Radient to see this message"),
        (_item("unavailable", reason="other_account"), "Received under another Radient account"),
        (_item("unavailable", reason="not_member"), "You are no longer in this conversation"),
        (
            _item("unavailable", reason="offline"),
            "Couldn't reach Radient — retrying",
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
        (_item("held", hold={"class": "awaiting_review"}), "Held for review"),
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
    # ``Span.style`` is ``str | Style``; only a resolved ``Style`` carries an ink,
    # so narrow on the real type instead of probing attributes.
    colours = {
        str(span.style.color.triplet.hex) if span.style.color.triplet else ""
        for span in content.spans
        if span.start <= start < span.end and isinstance(span.style, Style) and span.style.color
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
        if span.start <= start < span.end and isinstance(span.style, Style) and span.style.color
    ]
    assert expected is not None and expected in painted


# -- #4238 deferred follow-ups ------------------------------------------------


def _recording_block(calls: list[bool]) -> TeamBridgeBlock:
    """A card whose loads are recorded, with the real ``_body_loading`` latch.

    The latch is the point: the bug was the open being swallowed because the
    mount pass was still marked in flight.
    """
    block = TeamBridgeBlock("x", {"name": "Ola"}, body_loader=_no_load)

    def record(*, local_only: bool) -> None:
        block._body_loading = True
        calls.append(local_only)

    block._start_load = record  # type: ignore[method-assign]
    return block


def test_opening_while_the_mount_pass_is_in_flight_still_runs_the_full_pass() -> None:
    """Review m1: an open during the mount pass must not strand 'Loading message…'."""
    calls: list[bool] = []
    block = _recording_block(calls)
    block.on_mount()
    # The reader opens the card before the local-only answer lands: the open is
    # refused by the in-flight latch, so nothing else would ever start it.
    block._expanded = True
    block._maybe_load_body()
    assert calls == [True]
    block.apply_card_body(_answer(_item("unavailable", reason="not_loaded")), verified=False)
    assert calls == [True, False]


def test_a_closed_card_does_not_go_to_the_hub_when_the_mount_pass_lands() -> None:
    """The mount pass stays local-only for cards nobody has opened (D1)."""
    calls: list[bool] = []
    block = _recording_block(calls)
    block.on_mount()
    block.apply_card_body(_answer(_item("unavailable", reason="not_loaded")), verified=False)
    assert calls == [True]


def test_the_closed_offline_header_says_open_not_close_and_reopen() -> None:
    """Design D8: the collapsed card is already closed, so its retry is to open it."""
    row = _collapsed(_answer(_item("unavailable", reason="offline")))
    assert "Couldn't reach Radient — open to retry" in row
    assert "close and reopen" not in row
    # The open card keeps the body line: an open card retries by itself.
    body = _open(_answer(_item("unavailable", reason="offline")))._build_content(110).plain
    assert "Couldn't reach Radient — retrying" in body


def test_an_open_offline_card_header_does_not_contradict_its_body() -> None:
    """QA Q1 (#4277): "open to retry" is only for the CLOSED card.

    On an open card the header sits right above the body's "close and reopen
    to retry", so it keeps the string main showed there (the body line), and it
    follows the card through both directions of the toggle.
    """
    offline = _answer(_item("unavailable", reason="offline"))
    block = TeamBridgeBlock("(no preview available)", {"name": "Ola"}, body_loader=_no_load)
    block.apply_card_body(offline, verified=True)
    assert "open to retry" in block._build_content(110).plain.splitlines()[0]
    block.toggle_expanded()
    header = block._build_content(110).plain.splitlines()[0]
    assert "Couldn't reach Radient — retrying" in header
    assert "— open to retry" not in header
    block.toggle_expanded()
    closed = block._build_content(110).plain.splitlines()[0]
    assert "Couldn't reach Radient — open to retry" in closed
    # Opened before the answer lands: the header is built open from the start.
    opened = _open(offline)
    assert "— open to retry" not in opened._build_content(110).plain.splitlines()[0]


# -- re-read while open (bridge card never showing a delivered body) ----------
#
# The reported bug: a card opened while its message was ``pending`` kept
# "Waiting to be delivered" until it was closed and reopened, though the body
# was handed off minutes later (measured p50 45 s, p90 21 min, max 8 h). These
# mount the card in a real Textual app and replace ``set_timer`` with a
# recording fake, so the backoff is asserted exactly and fired by hand.

from local_operator.tui.bridge_card_body import (  # noqa: E402
    REREAD_BACKOFF_S,
    REREAD_STEADY_S,
    answer_is_final,
    card_conversation,
    reread_delay,
)


class _FakeTimer:
    def __init__(self, delay: float, callback: Any) -> None:
        self.delay = delay
        self.callback = callback
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def _scripted_loader(answers: list[Any], calls: list[bool]) -> Any:
    async def load(*, local_only: bool = False) -> Any:
        calls.append(local_only)
        answer = answers.pop(0) if len(answers) > 1 else answers[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    return load


async def _mounted(block: TeamBridgeBlock, timers: list[_FakeTimer]) -> Any:
    from textual.app import App

    def fake_set_timer(delay: float, callback: Any, **_kw: Any) -> _FakeTimer:
        timer = _FakeTimer(delay, callback)
        timers.append(timer)
        return timer

    block.set_timer = fake_set_timer  # type: ignore[method-assign]

    class _Host(App[None]):
        def compose(self):  # type: ignore[no-untyped-def]
            yield block

    return _Host()


def _fire(timers: list[_FakeTimer]) -> None:
    live = [timer for timer in timers if not timer.stopped]
    assert live, "no re-read scheduled"
    live[-1].stopped = True
    live[-1].callback()


def test_the_policy_classifies_final_answers_and_names_the_backoff() -> None:
    assert answer_is_final(None) is False
    assert answer_is_final({"v": 1}) is False
    assert answer_is_final(_answer(_item("pending"))) is False
    assert answer_is_final(_answer(_item("held", hold={"class": "transient"}))) is False
    assert answer_is_final(_answer(_item("held", hold={"class": "awaiting_review"}))) is False
    assert answer_is_final(_answer(_item("unavailable", reason="offline"))) is False
    assert answer_is_final(_answer(_item("unavailable", reason="not_loaded"))) is False
    assert answer_is_final(_answer(_item("admitted", body="x"))) is True
    assert answer_is_final(_answer(_item("refused"))) is True
    assert answer_is_final(_answer(_item("unavailable", reason="deleted"))) is True
    assert answer_is_final(_answer(_item("from_the_future"))) is True
    assert answer_is_final(_answer()) is True
    assert REREAD_BACKOFF_S == (5.0, 10.0, 20.0, 30.0)
    assert [reread_delay(n) for n in (0, 1, 2, 3, 4, 500)] == [5, 10, 20, 30, 60, 60]
    assert REREAD_STEADY_S == 60.0


@pytest.mark.asyncio
async def test_an_open_pending_card_shows_the_body_once_delivered_without_reopening() -> None:
    calls: list[bool] = []
    admitted = _answer(_item("admitted", body="the delivered body", source="local"))
    loader = _scripted_loader(
        [_answer(_item("pending")), _answer(_item("pending")), _answer(_item("pending")), admitted],
        calls,
    )
    block = TeamBridgeBlock("(no preview available)", {"name": "Ola"}, body_loader=loader)
    timers: list[_FakeTimer] = []
    app = await _mounted(block, timers)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert calls == [True]  # the local-only mount pass
        block.toggle_expanded()
        await pilot.pause()
        assert calls == [True, False]
        assert "Waiting to be delivered to your agent" in block._build_content(110).plain
        assert [t.delay for t in timers] == [5.0]
        _fire(timers)
        # While the re-read runs, the last answer stays: never back to Loading.
        assert block._body_loading is True
        assert "Loading message…" not in block._build_content(110).plain
        await pilot.pause()
        assert [t.delay for t in timers] == [5.0, 10.0]
        _fire(timers)
        await pilot.pause()
        body = block._build_content(110).plain
        assert "the delivered body" in body
        assert "External message · untrusted content" in body
        assert "Waiting to be delivered" not in body
        # Final: nothing further is scheduled.
        assert [t.delay for t in timers] == [5.0, 10.0]
        assert all(t.stopped for t in timers)


@pytest.mark.asyncio
async def test_collapsing_or_unmounting_stops_the_re_read() -> None:
    calls: list[bool] = []
    loader = _scripted_loader([_answer(_item("pending"))], calls)
    block = TeamBridgeBlock("x", {"name": "Ola"}, body_loader=loader)
    timers: list[_FakeTimer] = []
    app = await _mounted(block, timers)
    async with app.run_test() as pilot:
        await pilot.pause()
        block.toggle_expanded()
        await pilot.pause()
        assert len(timers) == 1 and not timers[0].stopped
        block.toggle_expanded()  # collapse
        assert timers[0].stopped
        # Reopening starts a fresh backoff from the first step.
        block.toggle_expanded()
        await pilot.pause()
        assert timers[-1].delay == 5.0 and not timers[-1].stopped
        await block.remove()
        await pilot.pause()
        assert timers[-1].stopped
        asked = len(calls)
        # A timer that fired late on a removed/collapsed card asks nothing.
        block._expanded = False
        block._reread()
        assert len(calls) == asked


@pytest.mark.asyncio
async def test_a_failed_re_read_keeps_the_last_answer_and_asks_again() -> None:
    calls: list[bool] = []
    loader = _scripted_loader(
        [
            _answer(_item("pending")),
            _answer(_item("pending")),
            RuntimeError("io"),
            _answer(_item("refused")),
        ],
        calls,
    )
    block = TeamBridgeBlock("x", {"name": "Ola"}, body_loader=loader)
    timers: list[_FakeTimer] = []
    app = await _mounted(block, timers)
    async with app.run_test() as pilot:
        await pilot.pause()
        block.toggle_expanded()
        await pilot.pause()
        _fire(timers)
        await pilot.pause()
        assert "Waiting to be delivered to your agent" in block._build_content(110).plain
        assert [t.delay for t in timers] == [5.0, 10.0]
        _fire(timers)
        await pilot.pause()
        assert "Refused by moderation" in block._build_content(110).plain
        assert len(timers) == 2


def test_held_for_review_and_refused_are_final_copy_not_still_delivering() -> None:
    held = (
        _open(_answer(_item("held", hold={"class": "awaiting_review"})))._build_content(110).plain
    )
    assert "Held for review" in held and "Waiting to be delivered" not in held
    refused = _open(_answer(_item("refused")))._build_content(110).plain
    assert "Refused by moderation" in refused and "Waiting to be delivered" not in refused


def _open_in(conversation: dict[str, Any], answer: dict[str, Any]) -> TeamBridgeBlock:
    block = TeamBridgeBlock(
        "(no preview available)", {"name": "Ola"}, body_loader=_no_load, conversation=conversation
    )
    block._expanded = True
    block.apply_card_body(answer)
    return block


def test_an_open_card_shows_conversation_thread_and_time_with_the_body() -> None:
    from datetime import UTC, datetime

    now = datetime.now(UTC).replace(microsecond=0)
    block = _open_in(
        {"kind": "channel", "label": "general"},
        {
            **_answer(_item("admitted", body="hello there")),
            "card": {"received_at": now.isoformat().replace("+00:00", "Z"), "thread_reply": True},
        },
    )
    lines = block._build_content(110).plain.splitlines()
    frame = next(
        i for i, line in enumerate(lines) if "External message · untrusted content" in line
    )
    expected = f"#general · Thread reply · {now.astimezone().strftime('%H:%M')}"
    assert lines[frame + 1].strip() == expected
    assert "hello there" in "\n".join(lines[frame + 2 :])
    # The row facts survive a later answer that lacks them (an older core).
    block.apply_card_body(_answer(_item("admitted", body="hello there")), verified=False)
    assert expected in block._build_content(110).plain
    epoch = datetime.fromtimestamp(0, UTC).astimezone().strftime("%Y-%m-%d %H:%M")
    dm = _open_in(
        {"kind": "dm"},
        {
            **_answer(_item("pending")),
            "card": {"received_at": "1970-01-01T00:00:00Z", "thread_reply": False},
        },
    )
    assert f"Direct message · {epoch}" in dm._build_content(110).plain
    # Nothing known: no context line at all, not an empty dim row.
    plain = _open(_answer(_item("pending")))._build_content(110).plain.splitlines()
    frame = next(i for i, line in enumerate(plain) if "untrusted content" in line)
    assert "Waiting to be delivered" in plain[frame + 1]


def test_card_conversation_reads_only_the_validated_receipt() -> None:
    details = {
        "body": "untrusted wire text naming #elsewhere",
        "bridge_receipt": {
            "v": 1,
            "sender": {"name": "Ada"},
            "conversation": {"kind": "channel", "label": "general"},
        },
        "bridge_receipt_source": {
            "account_id": "a",
            "tenant_id": "t",
            "endpoint_id": "e",
            "conversation_id": "bcv_1",
            "message_ids": ["bcmsg_1"],
            "conversation_seq": 3,
        },
    }
    assert card_conversation(details) == {"kind": "channel", "label": "general"}
    assert card_conversation({"bridge_receipt": {"conversation": {"kind": "dm"}}}) == {
        "kind": None,
        "label": None,
    }
    assert card_conversation(None) == {"kind": None, "label": None}
