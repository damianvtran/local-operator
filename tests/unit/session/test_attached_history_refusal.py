"""Which verdict a viewer gives when its owner's display window is not usable.

A display page comes back non-``ok`` for three unrelated reasons, and this module
pins all THREE, because reading one as another sends the operator to a remedy
that cannot exist:

* ``window is None`` — the owner's runtime predates the field. It served no
  window at all, and updating the peer really is the fix.
* ``window.status == "full_required"`` — a SIZE verdict from a runtime that is
  already current. It is the owner's answer when one message group exceeds the
  page budget (``history_window.DISPLAY_HISTORY_BYTES`` /
  ``DISPLAY_HISTORY_MESSAGES``), and what both wire fallbacks substitute when the
  attach push frame or the ``frontend_sync`` reply exceeds
  ``server._MAX_LINE_BYTES``. The peer needs no update; the reader has to reach
  the device that holds the conversation.
* ``window.status == "reset"`` — a cursor that stopped resolving, or a
  generation that moved, between the owner's snapshot and its capture. The peer
  is on THIS build (the DTO that carried the reset is this release's), so the
  remedy is the re-sync its own capture paths expect, and for a viewer the
  re-sync is reopening the conversation. #3624 established that ``reset`` must
  not be told its history is too large; #3644 gave it this verdict of its own,
  and this module is its home.

Before the split, every non-ok page raised the "runtime is too old" refusal, so
an oversized or reset session on the same build reported itself as a version
problem — the one symptom that cannot be acted on. These cells drive the real
``_load_frontend_history`` on a remote placement (and prove the local
placement's honest full replay still runs unchanged).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pytest

from local_operator.harness.types import Message
from local_operator.session.attached import AttachedSession
from local_operator.session.frontend_state import (
    FrontendModelSpec,
    FrontendSessionState,
    FrontendSync,
)
from local_operator.session.history_window import DisplayHistoryWindow
from local_operator.session.placement import SessionPlacement


async def _no_takeover() -> None:
    raise AssertionError("a display facade never takes over")


class _RemoteOwner:
    """The mesh owner seam, reduced to what the refusal path reads.

    Only ``placement`` is consulted before the refusal is raised, so a stub is
    the honest double here: a full remote owner would add a socket and a bind
    around the one branch under test.
    """

    def __init__(self) -> None:
        self.placement = SessionPlacement(
            mode="peer", network_id="net-1", home_device="peer-device"
        )

    def locate(self) -> tuple[None, None]:
        return None, None


def _model() -> FrontendModelSpec:
    return FrontendModelSpec(provider="p", model_id="m")


def _frontend(window: DisplayHistoryWindow | None) -> FrontendSync:
    state = FrontendSessionState(
        session_id="s1",
        epoch="e1",
        cwd="/tmp",
        selected_model=_model(),
        effective_model=_model(),
        history_generation=1,
    )
    return FrontendSync(
        epoch="e1", sequence=1, snapshot=state, live_cursor=None, display_history=window
    )


def _page(
    status: Literal["ok", "reset", "full_required"] = "full_required",
    total: int = 4312,
    has_more: bool = False,
) -> DisplayHistoryWindow:
    # ONE ROW IS PRESENT ON PURPOSE. The page this refusal sees is blanked by the
    # wire fallbacks, but the log line's contract is that it carries no history
    # CONTENT — a cell that shipped an empty row list cannot prove that, so the
    # row below is planted and the rendered line asserted verbatim (agent review
    # round 1, NIT-2).
    row = Message.user("PLANTED-ROW-CONTENT-7a41")
    return DisplayHistoryWindow(
        status=status,
        conversation_id="s1",
        owner_epoch="e1",
        history_generation=1,
        through_id=None,
        messages=[row],
        start=total,
        total_message_count=total,
        before_token=None,
        snapshot_token=None,
        has_more=has_more,
    )


def _remote_viewer(tmp_path: Path) -> AttachedSession:
    viewer = AttachedSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=_no_takeover,
        owner=_RemoteOwner(),  # type: ignore[arg-type] — the seam's one call
    )
    # The constructor already derives this from placement; asserted so a later
    # change to that derivation cannot silently turn these cells into no-ops.
    assert viewer._display_window_requested is True  # noqa: SLF001
    return viewer


def _local_viewer(tmp_path: Path) -> AttachedSession:
    viewer = AttachedSession(config_dir=tmp_path, session_id="s1", takeover_factory=_no_takeover)
    viewer._display_window_requested = True  # noqa: SLF001 — the flag under test
    return viewer


@pytest.mark.asyncio
async def test_a_size_verdict_names_the_size_and_never_the_build(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The defect: a same-build size verdict reported as "the peer is too old"."""
    viewer = _remote_viewer(tmp_path)
    # NO ``caplog.set_level`` HERE, deliberately: forcing the level is what makes
    # a level regression invisible (agent review round 1, MINOR-2). At the
    # default capture level an INFO line either arrives as INFO — failing the
    # ``levelno`` assertion below — or is filtered, failing ``assert records``.

    with pytest.raises(ConnectionError) as raised:
        await viewer._load_frontend_history(_frontend(_page()))  # noqa: SLF001

    message = str(raised.value)
    assert "too old" not in message, message
    assert "update" not in message, message
    assert "too large" in message and "device that holds it" in message, message
    # THE LANE BUDGET (design review round 1, D1): 110 cells is the transcript
    # lane at 120 columns, and a sentence over it orphans its closing clause.
    assert len(message) <= 110, f"{len(message)} chars: {message}"

    # The log line is the after-the-fact record of WHY the page was refused:
    # nothing on the wire carries the reason, so this is where size is told
    # from version. It must survive the daemon's WARNING default, and it must
    # carry the status and nothing off the rows.
    records = [r for r in caplog.records if r.name == "local_operator.session.attached"]
    assert records, "a refused non-ok page must leave a log line"
    assert records[-1].levelno == logging.WARNING, records[-1].levelname
    assert records[-1].getMessage() == "display history page is full_required (has_more=False)"
    assert "PLANTED-ROW-CONTENT-7a41" not in records[-1].getMessage()


@pytest.mark.asyncio
async def test_a_missing_window_still_names_the_build(tmp_path: Path) -> None:
    """The genuine old peer keeps its own verdict — this is not a regression."""
    viewer = _remote_viewer(tmp_path)

    with pytest.raises(ConnectionError) as raised:
        await viewer._load_frontend_history(_frontend(None))  # noqa: SLF001

    message = str(raised.value)
    assert "too old" in message and "update it" in message, message
    assert "too large" not in message, message
    # The other two claims belong to their own arms; a reword that smuggled one
    # into the old-peer sentence would read as advice the peer cannot take.
    assert "reopen" not in message, message


@pytest.mark.asyncio
async def test_a_reset_page_names_the_resync_and_no_other_verdict(
    tmp_path: Path,
) -> None:
    """``reset`` is a cursor verdict: re-sync it, do not update the peer.

    #3624 pinned that a reset page must not get the SIZE sentence (review round
    1, MINOR-1). #3644 is the verdict that decision was waiting for: the page
    fell through to the same-build "too old … update it" refusal, and the
    remedy a reset actually has — reopen, which re-captures — had no words.
    Both halves are asserted, because a future reword could reintroduce either
    misdiagnosis while still differing from the old text.
    """
    viewer = _remote_viewer(tmp_path)

    with pytest.raises(ConnectionError) as raised:
        await viewer._load_frontend_history(_frontend(_page(status="reset")))  # noqa: SLF001

    message = str(raised.value)
    assert message == (
        "this conversation's history changed on the device that holds it; "
        "reopen the conversation to reload it"
    ), message
    assert "too old" not in message, message
    assert "update" not in message, message
    assert "too large" not in message, message
    # THE LANE BUDGET (design review round 1, D1): 110 cells is the transcript
    # lane at 120 columns, and a sentence over it orphans its closing clause.
    assert len(message) <= 110, f"{len(message)} chars: {message}"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["reset", "full_required"])
async def test_a_local_placement_keeps_the_honest_full_replay(
    tmp_path: Path, status: Literal["ok", "reset", "full_required"]
) -> None:
    """Unchanged behaviour: a local placement escalates, it never refuses.

    Parametrised over BOTH non-ok verdicts that reach this branch, and with the
    window flag forced on so the page really arrives: an as-shipped local viewer
    never requests a window, which would make every status here
    indistinguishable from ``window is None`` — a control that cannot fail.
    """
    viewer = _local_viewer(tmp_path)

    await viewer._load_frontend_history(_frontend(_page(status=status)))  # noqa: SLF001

    assert viewer._display_window_supported is True  # noqa: SLF001
    assert viewer._display_history is None  # noqa: SLF001
    assert viewer._history_hydrated is True  # noqa: SLF001
    assert viewer._history == []  # noqa: SLF001 — an absent journal replays empty
