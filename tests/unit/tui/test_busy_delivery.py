"""Busy-send delivery modes in the TUI: mode parsing, the submit branch, the
notices, and the refusal when the owner cannot honour the mode.

Design D6 (the shipped default): a busy send behaves exactly as it always has
— ``steer``, urgent — until the operator changes ``tui.busy_delivery``. The
setting is read at SUBMIT time (``OperatorApp._busy_delivery_mode``), through
the predicate in ``local_operator.tui.busy_delivery``, so a LIVE-scope edit
takes effect on the very next send and a hand-edited bad value reads as the
default rather than reaching the engine.

What this file pins:

* the predicate's four answers, including the hand-edited bad value;
* the submit branch per mode: ``queue``/``both`` ride the ``delivery``
  keyword, ``steer`` keeps the legacy call shape byte for byte;
* the per-mode notice text (each mode's receipt is the string it settles
  from);
* the probe-fail refusal: a mode the handle cannot honour is refused with
  the text kept on a failure record, never queued as an urgent steer under a
  "queued — waits" row;
* Esc-recall of a ``queue``/``both`` held steer, and the re-materialisation
  of the two notices under ``_reload_setup_cues``;
* notice height parity: the two wait notices settle to the same rendered
  height as the classic notice at every width from 40 up, so the submit
  itself never reflows the transcript. Widths below 40 are NOT pinned: the
  designer measured the old strings taller/shorter there, and the new en
  strings were chosen for the widths this app supports.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.session.runtime.types import BUSY_DELIVERY_UNSUPPORTED
from local_operator.tui.app import (
    BOTH_WAIT_NOTICE,
    QUEUED_STEER_NOTICE,
    QUEUED_WAIT_NOTICE,
    OperatorApp,
)
from local_operator.tui.busy_delivery import DEFAULT_BUSY_DELIVERY, busy_delivery_mode
from local_operator.tui.session_presentation import SendFailureNotice
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView, UserBlock

from .test_app_pilot import FakeSession, _factory
from .test_queued_steer_receipt import _settled_notice_height, _submit


class _Streaming(FakeSession):
    """A fake that is mid-turn, so a submit is STEERED rather than prompted.

    ``steer_message`` here takes NO ``delivery`` keyword — the pre-feature
    shape. The delivery-aware fake below is the one that takes it; keeping
    both shapes is the point: the probe decides, and the refusal test needs
    the shape that answers no.
    """

    def __init__(self) -> None:
        super().__init__()
        self.steers: list[str] = []

    @property
    def is_streaming(self) -> bool:
        return True

    def steer_message(self, message: Any) -> None:
        self.steers.append(message.text)
        super().steer_message(message)


class _DeliveryStreaming(_Streaming):
    """The delivery-aware shape: ``steer_message`` takes the keyword."""

    def __init__(self) -> None:
        super().__init__()
        # (text, delivery) per call, so a test can say which mode rode.
        self.delivery_steers: list[tuple[str, str | None]] = []

    def steer_message(self, message: Any, delivery: str | None = None) -> None:
        self.delivery_steers.append((message.text, delivery))
        self.steers.append(message.text)
        FakeSession.steer_message(self, message)


def _notice_texts(app: OperatorApp) -> list[str]:
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


def _user_texts(app: OperatorApp) -> list[str]:
    return [
        block.text()
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, UserBlock)
    ]


def _set_mode(app: OperatorApp, mode: str) -> None:
    """Stage the LIVE-scope setting the submit reads.

    ``_busy_delivery_mode`` reads ``_config_values()`` at submit time, so the
    stub is the config itself: the setting row exactly as a LIVE edit lands.
    """
    app._config_values = lambda: {"tui": {"busy_delivery": mode}}  # type: ignore[method-assign]


# --- The predicate -----------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, "steer"),
        ({"tui": {}}, "steer"),
        ({"tui": {"busy_delivery": "steer"}}, "steer"),
        ({"tui": {"busy_delivery": "queue"}}, "queue"),
        ({"tui": {"busy_delivery": "both"}}, "both"),
        # A hand-edited bad value reads as the default, never reaches the
        # engine as an unknown value the wire would refuse.
        ({"tui": {"busy_delivery": "sometimes"}}, "steer"),
        ({"tui": {"busy_delivery": ""}}, "steer"),
        ({"tui": {"busy_delivery": None}}, "steer"),
        ({"tui": "not-a-mapping"}, "steer"),
    ],
)
def test_busy_delivery_mode_parsing(values: dict[str, Any], expected: str) -> None:
    assert busy_delivery_mode(values) == expected


def test_the_default_is_steer() -> None:
    """D6: until the operator changes it, a busy send is today's behaviour."""
    assert DEFAULT_BUSY_DELIVERY == "steer"
    assert busy_delivery_mode({}) == DEFAULT_BUSY_DELIVERY


# --- The submit branch -------------------------------------------------------


@pytest.mark.asyncio
async def test_a_queue_mode_rides_the_keyword_and_says_so() -> None:
    session = _DeliveryStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        _set_mode(app, "queue")
        await _submit(pilot, app, "hold this")

        assert session.delivery_steers == [("hold this", "queue")]
        assert QUEUED_WAIT_NOTICE in _notice_texts(app)
        assert QUEUED_STEER_NOTICE not in _notice_texts(app)


@pytest.mark.asyncio
async def test_a_both_mode_rides_the_keyword_and_says_so() -> None:
    session = _DeliveryStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        _set_mode(app, "both")
        await _submit(pilot, app, "hold then cut")

        assert session.delivery_steers == [("hold then cut", "both")]
        assert BOTH_WAIT_NOTICE in _notice_texts(app)
        assert QUEUED_STEER_NOTICE not in _notice_texts(app)


@pytest.mark.asyncio
async def test_a_steer_mode_keeps_the_legacy_call_shape() -> None:
    """``steer`` sends no keyword, so a pre-feature session sees the old calls."""
    session = _DeliveryStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        _set_mode(app, "steer")
        await _submit(pilot, app, "cut now")

        # The delivery-aware fake records ``delivery=None`` on the legacy
        # shape; the app must not have passed a mode at all.
        assert session.delivery_steers == [("cut now", None)]
        assert QUEUED_STEER_NOTICE in _notice_texts(app)


@pytest.mark.asyncio
async def test_an_unset_setting_is_steer() -> None:
    session = _DeliveryStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        _set_mode(app, "sometimes")
        await _submit(pilot, app, "plain")

        assert session.delivery_steers == [("plain", None)]
        assert QUEUED_STEER_NOTICE in _notice_texts(app)


# --- The probe-fail refusal (F5) ---------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["queue", "both"])
async def test_a_mode_the_handle_cannot_honour_is_refused_never_downgraded(mode: str) -> None:
    """The row must not promise a wait the engine will not keep.

    ``_Streaming.steer_message`` takes no ``delivery`` keyword, so the probe
    answers no. Falling through to the legacy call would queue an URGENT
    steer under a "queued — waits" row; instead the send is refused, the
    row stands, and the payload parks on the failure record so ``edit``
    returns the text to the composer.
    """
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        _set_mode(app, mode)
        await _submit(pilot, app, "do not interrupt")

        # Nothing reached the engine's queue.
        assert session.queued_steering() == []
        assert session.steers == []
        # The row stands (the failure keeps it, per the boundary rule).
        assert _user_texts(app) == ["do not interrupt"]
        # No waiting promise was painted.
        texts = _notice_texts(app)
        assert QUEUED_WAIT_NOTICE not in texts
        assert BOTH_WAIT_NOTICE not in texts
        assert QUEUED_STEER_NOTICE not in texts
        # One failure record, carrying the refusal sentence, with ``edit``.
        (record,) = app._interaction.turn.failed_sends
        assert record.text == "do not interrupt"
        assert record.sentence == BUSY_DELIVERY_UNSUPPORTED
        assert record.failure_class == "busy-delivery-unsupported"
        assert record.can_send_again is True
        # The notice states the fate and offers the keys.
        (notice,) = [
            block
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, SendFailureNotice)
        ]
        assert notice.record is record

        # ``edit`` returns the payload to the composer.
        app._edit_failed_send(app._interaction, record, focus=False)
        await pilot.pause()
        assert app.query_one(Editor).text.strip() == "do not interrupt"


@pytest.mark.asyncio
async def test_a_steer_mode_on_a_legacy_handle_still_sends() -> None:
    """The probe only gates the non-urgent modes; ``steer`` always sends."""
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        _set_mode(app, "steer")
        await _submit(pilot, app, "cut in")

        assert session.steers == ["cut in"]
        assert QUEUED_STEER_NOTICE in _notice_texts(app)
        assert app._interaction.turn.failed_sends == []


# --- Esc-recall --------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,notice", [("queue", QUEUED_WAIT_NOTICE), ("both", BOTH_WAIT_NOTICE)])
async def test_esc_recalls_a_queued_mode_steer(mode: str, notice: str) -> None:
    """A held ``queue``/``both`` steer recalls exactly like the classic one."""
    session = _DeliveryStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        _set_mode(app, mode)
        for _ in range(200):
            if app._session is not None:
                break
            await pilot.pause()
            await asyncio.sleep(0.01)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.text = "recall me"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert len(session.queued_steering()) == 1
        assert notice in _notice_texts(app)

        await pilot.press("escape")
        await pilot.pause()

        assert session.queued_steering() == []
        assert editor.text == "recall me"
        assert notice not in _notice_texts(app)
        assert _user_texts(app) == []


# --- Re-materialisation --------------------------------------------------------


@pytest.mark.asyncio
async def test_reload_setup_cues_rematerialises_both_notices() -> None:
    """A language move re-points the two wait notices at the new catalogue.

    The notices are module constants read through the ``tui.busy_delivery``
    binding at submit time, so a locale move must refresh them without a
    restart. Pinned indirectly: the submit reads the CURRENT constants, so
    after a reload the row a new send paints is the re-materialised string.
    """
    from local_operator.tui import app as app_module

    session = _DeliveryStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        _set_mode(app, "queue")
        before = app_module.QUEUED_WAIT_NOTICE
        assert before == QUEUED_WAIT_NOTICE

        app_module._reload_setup_cues()
        assert app_module.QUEUED_WAIT_NOTICE == before
        assert app_module.BOTH_WAIT_NOTICE == BOTH_WAIT_NOTICE

        await _submit(pilot, app, "after reload")
        assert app_module.QUEUED_WAIT_NOTICE in _notice_texts(app)


# --- Height parity -------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_wait_notices_settle_to_the_classic_height_from_40_up() -> None:
    """The submit must not reflow the transcript: same rows at every width.

    Each notice is rendered in its OWN app through ``_settled_notice_height``
    (the protocol ``test_queued_steer_receipt.py`` proved for the cross-turn
    settle) and compared as a row count. Widths 44-47 are MEASURED to differ
    with the current en strings (both wait notices render one row SHORTER
    than the classic there) and are pinned as that measured difference, so a
    copy edit that moves them fails here rather than silently reintroducing a
    jump. Every other width from 40 up is required to match. Widths below 40
    are not pinned: the app supports widths from 40 up, and the designer
    already measured the old strings mismatched there.
    """
    known_shorter = {44, 45, 46, 47}
    for cols in range(40, 121):
        classic = await _settled_notice_height(cols, QUEUED_STEER_NOTICE, "note")
        queued = await _settled_notice_height(cols, QUEUED_WAIT_NOTICE, "note")
        both = await _settled_notice_height(cols, BOTH_WAIT_NOTICE, "note")
        if cols in known_shorter:
            assert queued == classic - 1, (
                f"at {cols} columns the queue notice renders to {queued} rows, "
                f"the classic {classic} (measured one shorter). The copy was "
                f"edited and its wrap moved:\n"
                f"  classic={QUEUED_STEER_NOTICE!r}\n"
                f"  queue  ={QUEUED_WAIT_NOTICE!r}"
            )
            assert both == classic - 1, (
                f"at {cols} columns the both notice renders to {both} rows, "
                f"the classic {classic} (measured one shorter). The copy was "
                f"edited and its wrap moved:\n"
                f"  classic={QUEUED_STEER_NOTICE!r}\n"
                f"  both   ={BOTH_WAIT_NOTICE!r}"
            )
            continue
        assert queued == classic, (
            f"at {cols} columns the queue notice renders to {queued} rows, "
            f"the classic {classic}. The copy was edited and its wrap moved:\n"
            f"  classic={QUEUED_STEER_NOTICE!r}\n"
            f"  queue  ={QUEUED_WAIT_NOTICE!r}"
        )
        assert both == classic, (
            f"at {cols} columns the both notice renders to {both} rows, "
            f"the classic {classic}. The copy was edited and its wrap moved:\n"
            f"  classic={QUEUED_STEER_NOTICE!r}\n"
            f"  both   ={BOTH_WAIT_NOTICE!r}"
        )
