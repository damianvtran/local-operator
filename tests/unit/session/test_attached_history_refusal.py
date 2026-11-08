"""Which verdict a viewer gives when its owner's display window is not usable.

A display page comes back non-``ok`` for two unrelated reasons, and this module
pins BOTH, because reading one as the other sends the operator to a remedy that
cannot exist:

* ``window is None`` — the owner's runtime predates the field. It served no
  window at all, and updating the peer really is the fix.
* ``window.status`` non-``ok`` — a SIZE verdict from a runtime that is already
  current. ``full_required`` is the owner's answer when one message group
  exceeds the page budget (``history_window.DISPLAY_HISTORY_BYTES`` /
  ``DISPLAY_HISTORY_MESSAGES``), and it is what both wire fallbacks substitute
  when the attach push frame or the ``frontend_sync`` reply exceeds
  ``server._MAX_LINE_BYTES``. The peer needs no update; the reader has to reach
  the device that holds the conversation.

Before the split, every non-ok page raised the "runtime is too old" refusal, so
an oversized session on the same build reported itself as a version problem —
the one symptom that cannot be acted on. These cells drive the real
``_load_frontend_history`` on a remote placement (and, third, prove the local
placement's honest full replay still runs unchanged).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pytest

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
    return DisplayHistoryWindow(
        status=status,
        conversation_id="s1",
        owner_epoch="e1",
        history_generation=1,
        through_id=None,
        messages=[],
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
    caplog.set_level(logging.INFO, logger="local_operator.session.attached")

    with pytest.raises(ConnectionError) as raised:
        await viewer._load_frontend_history(_frontend(_page()))  # noqa: SLF001

    message = str(raised.value)
    assert "too old" not in message, message
    assert "update" not in message, message
    assert "too large" in message and "device that holds it" in message, message

    # The log line is the after-the-fact record of WHY the page was refused:
    # nothing on the wire carries the reason, so this is where size is told
    # from version. It must carry the status and counts, and no rows.
    records = [r for r in caplog.records if r.name == "local_operator.session.attached"]
    assert records, "a refused non-ok page must leave a log line"
    rendered = records[-1].getMessage()
    assert "full_required" in rendered, rendered
    assert "total_message_count=4312" in rendered, rendered


@pytest.mark.asyncio
async def test_a_missing_window_still_names_the_build(tmp_path: Path) -> None:
    """The genuine old peer keeps its own verdict — this is not a regression."""
    viewer = _remote_viewer(tmp_path)

    with pytest.raises(ConnectionError) as raised:
        await viewer._load_frontend_history(_frontend(None))  # noqa: SLF001

    message = str(raised.value)
    assert "too old" in message and "update it" in message, message
    assert "too large" not in message, message


@pytest.mark.asyncio
async def test_a_local_placement_keeps_the_honest_full_replay(tmp_path: Path) -> None:
    """Unchanged behaviour: a local placement escalates, it never refuses."""
    viewer = _local_viewer(tmp_path)

    await viewer._load_frontend_history(_frontend(_page()))  # noqa: SLF001

    assert viewer._display_history is None  # noqa: SLF001
    assert viewer._history_hydrated is True  # noqa: SLF001
    assert viewer._history == []  # noqa: SLF001 — an absent journal replays empty
