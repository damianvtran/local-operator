"""The three non-ok display envelopes a REMOTE viewer can be handed, and why
each one needs its own sentence.

``AttachedSession._load_frontend_history`` refuses a remote placement when the
owner's packet carries no usable window, because the legacy local replay below
it reads ``sessions/<id>/transcript.jsonl`` — a file that, for a session on
another device, is absent or a DIFFERENT conversation's history wearing the same
id (``docs/design/mesh-session-mobility.md`` §3.4). One refusal, but three
causes, and until this test file existed only two of them were distinguished:

* ``full_required`` — the owner's first frame did not fit the socket
  (``runtime/server.py`` substitutes this status when the sync frame overruns its
  line limit). Real remedy: open the conversation where the history already is.
* ``reset`` — the capture's cursor stopped resolving, or its generation moved,
  mid-flight (``history_window.py``). The peer is on THIS build: it just spoke
  a DTO this release introduced. Real remedy: re-sync, which for a viewer means
  reopening the conversation.
* ``window is None`` — a peer that genuinely never negotiated the window. Real
  remedy: update it.

The defect this pins: ``reset`` fell through to the third sentence, so a user on
a current peer was told to update a peer that had nothing to update. A raise is
the whole observable behaviour here, so these assertions ARE the feature test —
there is no path downstream of the raise to exercise.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.session.attached import AttachedSession
from local_operator.session.frontend_state import (
    FrontendModelSpec,
    FrontendSessionState,
    FrontendSync,
)
from local_operator.session.history_window import DisplayHistoryWindow
from local_operator.session.placement import SessionPlacement

#: The exact sentences, spelled out rather than imported: the point of the test
#: is what a user READS, so a constant shared with the implementation would make
#: a reworded message pass silently.
SIZE_SENTENCE = (
    "this session's newest history is too large to send here; "
    "open the conversation on the device that holds it"
)
RESET_SENTENCE = (
    "this conversation's history changed on the device that holds it; "
    "reopen the conversation to reload it"
)
STALE_SENTENCE = "this peer's runtime is too old to serve history over the mesh; update it"


async def _no_takeover() -> None:
    raise AssertionError("a display facade never takes over")


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


def _window(status: str) -> DisplayHistoryWindow:
    return DisplayHistoryWindow(
        status=status,
        conversation_id="s1",
        owner_epoch="e1",
        history_generation=1,
        through_id=None,
    )


def _viewer(tmp_path: Path, *, remote: bool) -> AttachedSession:
    viewer = AttachedSession(config_dir=tmp_path, session_id="s1", takeover_factory=_no_takeover)
    if remote:
        # A real placement, not a bare bool: ``is_local`` is derived from
        # ``mode``, and a stub that sets the property directly would keep
        # passing if that derivation changed under it.
        cast(Any, viewer)._owner = SimpleNamespace(
            placement=SessionPlacement(mode="peer", network_id="n_1", home_device="dev_b")
        )
    # The window is only requested for a remote placement (the constructor
    # derives this flag from placement); set it explicitly so the remote cases
    # below are not silently exercising the ``window is None`` arm.
    viewer._display_window_requested = remote  # noqa: SLF001 — the flag under test
    return viewer


@pytest.mark.asyncio
async def test_reset_names_the_resync_remedy_and_never_blames_the_peer(tmp_path: Path) -> None:
    viewer = _viewer(tmp_path, remote=True)

    with pytest.raises(ConnectionError) as caught:
        await viewer._load_frontend_history(_frontend(_window("reset")))  # noqa: SLF001

    message = str(caught.value)
    assert message == RESET_SENTENCE, message
    # The two misdiagnoses this arm exists to rule out. Spelled separately from
    # the sentence comparison because a future reword could reintroduce either
    # word while still differing from the old text.
    assert "too old" not in message, message
    assert "update" not in message, message


@pytest.mark.asyncio
async def test_a_too_large_newest_history_still_names_the_owning_device(tmp_path: Path) -> None:
    viewer = _viewer(tmp_path, remote=True)

    with pytest.raises(ConnectionError) as caught:
        await viewer._load_frontend_history(_frontend(_window("full_required")))  # noqa: SLF001

    message = str(caught.value)
    assert message == SIZE_SENTENCE, message
    # A size escalation is the one case where the words "too large" are TRUE;
    # it must not be lost to the reset arm beside it.
    assert "too large" in message, message


@pytest.mark.asyncio
async def test_a_peer_that_sent_no_window_is_still_told_to_update(tmp_path: Path) -> None:
    viewer = _viewer(tmp_path, remote=True)

    with pytest.raises(ConnectionError) as caught:
        await viewer._load_frontend_history(_frontend(None))  # noqa: SLF001

    assert str(caught.value) == STALE_SENTENCE, str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["reset", "full_required"])
async def test_a_local_placement_never_gets_a_mesh_verdict(
    tmp_path: Path, status: str
) -> None:
    """The remote sentences are a REMOTE verdict; a local owner keeps its replay.

    Asserted as behaviour rather than by reading the branch: the local path is
    the legacy full replay, so whatever it does with an unusable window, it must
    not be refused with a sentence about a device that holds the history — here
    this device does.
    """
    viewer = _viewer(tmp_path, remote=False)

    try:
        await viewer._load_frontend_history(_frontend(_window(status)))  # noqa: SLF001
    except ConnectionError as error:  # the replay may legitimately fail to read
        for sentence in (SIZE_SENTENCE, RESET_SENTENCE, STALE_SENTENCE):
            assert str(error) != sentence, str(error)
