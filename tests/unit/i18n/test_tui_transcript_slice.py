"""T3 (``tui.transcript``): the transcript's extracted copy is byte-identical to extraction time.

The fixture next to this file is a SNAPSHOT of the transcript widget's copy as
it was before the extraction (``origin/main`` @ the slice's base): the two
queued-prompt markers, the pasted-attachment receipt and the wake catch-up
headline/count as the ICU plurals that replace their hand-rolled branches, the
peer sender's identity bits, and the ask card's three Q&A state words.

The tests below re-derive the live values and require every rendered string to
match the golden EXACTLY — the committed form of the slice's byte-identical
proof, re-runnable by review and QA, and the trip-wire for any later edit that
changes transcript copy without regenerating the catalogue. The PR-time
before/after capture (constants + every composition) is recorded in the PR body;
this file is its permanent, in-suite half.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.text import Text

from local_operator.i18n import catalogues, runtime
from local_operator.i18n.messages import Msg
from local_operator.i18n.text import tr
from local_operator.tui.widgets import transcript as tx

FIXTURE = Path(__file__).parent / "fixtures" / "tui_transcript_en_golden.json"
GOLDEN: dict[str, str] = json.loads(FIXTURE.read_text(encoding="utf-8"))


def _en(key: str, **params: object) -> str:
    """The message rendered from the FIXTURE source, never from the live file."""
    return runtime.render_message(GOLDEN[key], params, "en")


def test_catalogue_matches_the_pre_slice_golden() -> None:
    assert catalogues.load_catalogue("en", "tui.transcript") == GOLDEN


def test_the_queued_markers_are_the_golden_lines() -> None:
    assert tx.QUEUED_ROW_TEXT == _en("tui.transcript.queued.newest")
    assert tx.QUEUED_ROW_TEXT_OLDER == _en("tui.transcript.queued.older")
    # The older marker is a strict PREFIX of the newest: the only difference is
    # the recall offer, and a second row advertising the same press promises a
    # recall the press declines (UX round 2, U4). A wave must keep that shape.
    newest, older = _en("tui.transcript.queued.newest"), _en("tui.transcript.queued.older")
    assert newest.startswith(older) and newest != older


def test_the_receipt_and_wake_lines_are_the_golden_plurals() -> None:
    # Plural selection, not a hand-rolled branch: en renders byte for byte.
    assert _en("tui.transcript.attach.receipt", count=1) == "↑ 1 image attached"
    assert _en("tui.transcript.attach.receipt", count=3) == "↑ 3 images attached"
    assert _en("tui.transcript.catchup.missed", count=1) == "1 missed wake"
    assert _en("tui.transcript.catchup.missed", count=3) == "3 missed wakes"
    # The headline joins the count with the em dash in ONE line by design.
    label = _en("tui.transcript.catchup.missed", count=3)
    assert _en("tui.transcript.catchup.headline", label=label) == f"catch-up — {label}"


def test_the_sender_bits_and_ask_values_are_the_golden_lines() -> None:
    assert _en("tui.transcript.sender.pid", pid="4242") == "pid 4242"
    assert _en("tui.transcript.sender.unknown") == "another session"
    assert _en("tui.transcript.ask.key_name", keys="GITHUB_TOKEN") == "GITHUB_TOKEN (key name)"
    assert _en("tui.transcript.ask.not_provided") == "not provided"
    assert _en("tui.transcript.ask.not_answered") == "not answered"


@pytest.mark.asyncio
async def test_the_real_rows_paint_the_golden_copy() -> None:
    """The extraction's whole point: what a user READS is the catalogue copy.

    Drives the REAL ``UserBlock`` through its build inside the real sheet (the
    harness the marker-chip suite uses), so a stale literal left in the row
    authors fails here even if the catalogue is right.
    """
    from tests.unit.tui.test_marker_chip import ChipHost

    app = ChipHost()
    async with app.run_test(size=(100, 30)) as pilot:
        receipt = tx.UserBlock("look at this", attachments=3)
        await app.mount(receipt)
        await pilot.pause()
        built = receipt._build()
        assert isinstance(built, Text)
        assert _en("tui.transcript.attach.receipt", count=3) in built.plain

        offered = tx.UserBlock("queued prompt", queued=True, queued_offer=True)
        await app.mount(offered)
        await pilot.pause()
        built = offered._build()
        assert isinstance(built, Text)
        assert _en("tui.transcript.queued.newest") in built.plain


def test_unknown_messages_degrade_to_their_code() -> None:
    resolve = tr("tui.transcript")
    assert resolve(Msg("tui.transcript.nope", {})) == "tui.transcript.nope"
