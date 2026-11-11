"""Busy-send delivery modes in the TUI: mode parsing, the submit branch, the
notices, and the refusal when the owner cannot honour the mode.

Design D6 (the shipped default): a busy send behaves exactly as it always has
— ``steer``, urgent — until the operator changes ``delivery.busy_default``. The
setting is read at SUBMIT time (``OperatorApp._busy_delivery_mode``), through
the predicate in ``local_operator.tui.busy_delivery``, so a LIVE-scope edit
takes effect on the very next send and a hand-edited bad value reads as the
default rather than reaching the engine.

What this file pins:

* the predicate's four answers, including the hand-edited bad value;
* the submit branch per mode: ``queue``/``both`` ride the ``delivery``
  keyword, ``steer`` keeps the legacy call shape byte for byte;
* the per-mode notice text (each mode's receipt is the string it settles
  from), the joined state a waiting steer puts the older queue/both rows
  into, and the revert when that steer is recalled;
* the probe-fail refusal: a mode the handle cannot honour is refused with
  the text kept on a failure record, never queued as an urgent steer under a
  "queued — waits" row;
* Esc-recall of a ``queue``/``both`` held steer, and the re-materialisation
  of the four notices under ``_reload_setup_cues``;
* receipt height: no receipt renders TALLER than the shipped baseline row at
  any supported width from 40 up, so a submit or an in-place restate never
  rolls the transcript below it. Some render one row shorter, which is
  harmless in a gap-spaced transcript and pinned as measured. Widths below
  40 are NOT pinned: the app supports widths from 40 up.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.session.runtime.types import BUSY_DELIVERY_UNSUPPORTED
from local_operator.tui.app import (
    BOTH_WAIT_NOTICE,
    DEFERRED_STEER_NOTICE,
    QUEUED_WAIT_NOTICE,
    STEER_JOINED_NOTICE,
    STEER_NOTICE,
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
    app._config_values = lambda: {"delivery": {"busy_default": mode}}  # type: ignore[method-assign]


# --- The predicate -----------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({}, "steer"),
        ({"delivery": {}}, "steer"),
        ({"delivery": {"busy_default": "steer"}}, "steer"),
        ({"delivery": {"busy_default": "queue"}}, "queue"),
        ({"delivery": {"busy_default": "both"}}, "both"),
        # A hand-edited bad value reads as the default, never reaches the
        # engine as an unknown value the wire would refuse.
        ({"delivery": {"busy_default": "sometimes"}}, "steer"),
        ({"delivery": {"busy_default": ""}}, "steer"),
        ({"delivery": {"busy_default": None}}, "steer"),
        ({"delivery": "not-a-mapping"}, "steer"),
        # The old TUI-only path is inert: nothing shipped under it (architect
        # D2), so a stale ``tui.busy_delivery`` never changes the answer.
        ({"tui": {"busy_delivery": "queue"}}, "steer"),
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
        assert STEER_NOTICE not in _notice_texts(app)


@pytest.mark.asyncio
async def test_a_both_mode_rides_the_keyword_and_says_so() -> None:
    session = _DeliveryStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        _set_mode(app, "both")
        await _submit(pilot, app, "hold then cut")

        assert session.delivery_steers == [("hold then cut", "both")]
        assert BOTH_WAIT_NOTICE in _notice_texts(app)
        assert STEER_NOTICE not in _notice_texts(app)


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
        assert STEER_NOTICE in _notice_texts(app)


@pytest.mark.asyncio
async def test_an_unset_setting_is_steer() -> None:
    session = _DeliveryStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        _set_mode(app, "sometimes")
        await _submit(pilot, app, "plain")

        assert session.delivery_steers == [("plain", None)]
        assert STEER_NOTICE in _notice_texts(app)


# --- Mode-aware receipts and the joined state --------------------------------


def test_the_five_receipts_are_distinct_and_none_promises_the_old_wait() -> None:
    """The receipt names what the message will do, so no two states share text.

    One row for every mode was the defect this fixes: a steer does not wait for
    the step to finish, so the queue row's sentence was false for it. The swept
    baseline sentence is the one that shipped, and no state may bring it back
    through a copy edit.
    """
    receipts = {
        "steer": STEER_NOTICE,
        "queue": QUEUED_WAIT_NOTICE,
        "both": BOTH_WAIT_NOTICE,
        "joined": STEER_JOINED_NOTICE,
        "deferred": DEFERRED_STEER_NOTICE,
    }
    assert len(set(receipts.values())) == len(receipts), receipts
    for name, text in receipts.items():
        assert "sends when this step finishes" not in text, name


@pytest.mark.asyncio
async def test_a_steer_after_a_queue_row_restates_the_earlier_row_to_joined() -> None:
    session = _DeliveryStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        _set_mode(app, "queue")
        await _submit(pilot, app, "wait for me")
        assert _notice_texts(app) == [QUEUED_WAIT_NOTICE]

        _set_mode(app, "steer")
        await _submit(pilot, app, "cut in now")

        # ONE row per message, the earlier one RESTATED in place: the engine
        # drains the whole queue at the next boundary, so the waiting message
        # really does go out with the steer.
        assert _notice_texts(app) == [STEER_JOINED_NOTICE, STEER_NOTICE]


@pytest.mark.asyncio
async def test_recalling_the_steer_reverts_the_row_it_joined() -> None:
    session = _DeliveryStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        _set_mode(app, "queue")
        await _submit(pilot, app, "wait for me")
        _set_mode(app, "steer")
        await _submit(pilot, app, "cut in now")
        assert _notice_texts(app) == [STEER_JOINED_NOTICE, STEER_NOTICE]

        app.query_one(Editor).focus()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        # The steer left the queue, so the wait the queue row promised is true
        # again. Derived from the queue, never remembered per row.
        assert [message.text for message in session.queued_steering()] == ["wait for me"]
        assert _notice_texts(app) == [QUEUED_WAIT_NOTICE]


@pytest.mark.asyncio
async def test_a_queue_row_submitted_while_a_steer_waits_starts_joined() -> None:
    session = _DeliveryStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        _set_mode(app, "steer")
        await _submit(pilot, app, "cut in now")
        _set_mode(app, "queue")
        await _submit(pilot, app, "wait for me")

        # It starts in the state that is true when it is painted: the steer
        # ahead of it ships at the next boundary, and this message goes too.
        assert _notice_texts(app) == [STEER_NOTICE, STEER_JOINED_NOTICE]


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
        assert STEER_NOTICE not in texts
        # One failure record, carrying the refusal sentence, with ``edit``
        # only: a resend of the same mode against the same legacy handle
        # fails identically, so ``can_send_again=False`` (D9) — the text is
        # recoverable via ``edit``, where a plain Enter (steer) can send it.
        (record,) = app._interaction.turn.failed_sends
        assert record.text == "do not interrupt"
        assert record.sentence == BUSY_DELIVERY_UNSUPPORTED
        assert record.failure_class == "busy-delivery-unsupported"
        assert record.can_send_again is False
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
        assert STEER_NOTICE in _notice_texts(app)
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
        assert app_module.STEER_NOTICE == STEER_NOTICE
        assert app_module.STEER_JOINED_NOTICE == STEER_JOINED_NOTICE

        await _submit(pilot, app, "after reload")
        assert app_module.QUEUED_WAIT_NOTICE in _notice_texts(app)


# --- Height parity -------------------------------------------------------------

#: The sentence this change replaces, exactly as it shipped. It is the height
#: baseline the new receipts are measured against, not a live constant.
SHIPPED_BASELINE = "queued — sends when this step finishes"

#: Widths where a receipt renders one row SHORTER than the baseline; measured
#: on a mounted `NoticeBlock` through `App.run_test` at 40-120 columns. Shorter
#: is harmless — the transcript spaces blocks by a gap, not by fixed rows — so
#: these are pinned as measured rather than treated as failures.
SHORTER_THAN_BASELINE = {
    "steer": {44, 45, 46, 47},
    "queue": {44, 45, 46, 47},
    "both": {45, 46, 47},
    "joined": {45, 46, 47},
}

#: The one receipt that renders TALLER than the baseline, and the only one this
#: change does not touch: `DEFERRED_STEER_NOTICE` is 43 cells, so at 48-52
#: columns (where the 38-cell baseline already fits on one row) it wraps to two.
#: Pinned as measured so a copy edit that moves it fails here rather than
#: silently reflowing the transcript; changing it is outside this rename.
DEFERRED_TALLER_THAN_BASELINE = {48, 49, 50, 51, 52}


@pytest.mark.asyncio
async def test_no_receipt_is_taller_than_the_shipped_baseline_from_40_up() -> None:
    """A submit or a restate must not push the rows below it down.

    The width-parity pin, rewritten around the claim that actually holds. The
    old test required the two wait notices to match the classic row exactly and
    pinned the widths where they did not; with the copy now naming the mode,
    the honest invariant is "no taller than the shipped baseline", because a
    SHORTER row costs nothing in a gap-spaced transcript while a TALLER one
    moves everything under it — the visible jump, and the one a reader sees
    while watching a submit land or a row restate in place.

    Each string is rendered in its OWN app through ``_settled_notice_height``
    (the protocol ``test_queued_steer_receipt.py`` proved for the cross-turn
    settle) and compared as a row count. The widths are the step boundaries
    the wrap can move at (every width up to 60, where all five strings are
    shorter than a body line, plus the two extremes); the height is a step
    function of the wrap, so an interior width would only re-render the same
    step. Widths below 40 are not pinned: the app supports widths from 40 up.
    """
    waiting = {
        "steer": STEER_NOTICE,
        "queue": QUEUED_WAIT_NOTICE,
        "both": BOTH_WAIT_NOTICE,
        "joined": STEER_JOINED_NOTICE,
    }
    widths = [*range(40, 61), 80, 120]
    for cols in widths:
        baseline = await _settled_notice_height(cols, SHIPPED_BASELINE, "note")
        for name, text in waiting.items():
            rows = await _settled_notice_height(cols, text, "note")
            expected = baseline - 1 if cols in SHORTER_THAN_BASELINE[name] else baseline
            assert rows == expected, (
                f"at {cols} columns the {name} receipt renders to {rows} rows, "
                f"the shipped baseline {baseline} (measured expectation "
                f"{expected}). The copy was edited and its wrap moved:\n"
                f"  baseline={SHIPPED_BASELINE!r}\n"
                f"  {name}={text!r}"
            )
            assert rows <= baseline, (
                f"at {cols} columns the {name} receipt is TALLER than the "
                f"shipped row ({rows} vs {baseline}), so a submit or restate "
                f"moves the transcript below it:\n  {text!r}"
            )
        deferred = await _settled_notice_height(cols, DEFERRED_STEER_NOTICE, "note")
        expected = baseline + 1 if cols in DEFERRED_TALLER_THAN_BASELINE else baseline
        assert deferred == expected, (
            f"at {cols} columns the deferred receipt renders to {deferred} rows, "
            f"the shipped baseline {baseline} (measured expectation {expected}):\n"
            f"  {DEFERRED_STEER_NOTICE!r}"
        )
