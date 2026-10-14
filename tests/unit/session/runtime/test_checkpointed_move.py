"""Coordinator-side checkpointed move: open the window, never kill the worker.

Driven over the production handle methods with a stub session, so the assertion
lands on the ADMISSION and the latch rather than on a runtime boot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.session.runtime.test_update_window import PAIR, WindowHost

pytestmark = pytest.mark.asyncio


class LatchSession:
    """The slice of ``Session`` the checkpoint latch touches."""

    def __init__(self, directory: Path) -> None:
        from types import SimpleNamespace

        self.transcript = SimpleNamespace(directory=directory)
        self.latched: list[dict] = []
        self.cancelled = 0

    def request_checkpoint(self, **kwargs) -> bool:
        if kwargs.get("phase") not in ("resume", "deferred"):
            return False
        self.latched.append(kwargs)
        return True

    def cancel_checkpoint(self) -> None:
        self.cancelled += 1


class MoveHost(WindowHost):
    """``WindowHost`` plus the one production method under test."""

    request_checkpointed_move = ServingSessionHandle.request_checkpointed_move
    ready_to_resume = ServingSessionHandle.ready_to_resume


def _host(tmp_path: Path) -> tuple[MoveHost, LatchSession]:
    session = LatchSession(tmp_path / "sessions" / "s1")
    session.transcript.directory.mkdir(parents=True, exist_ok=True)
    return MoveHost(session), session


async def test_request_checkpointed_move_latches_and_opens_the_window(tmp_path):
    host, session = _host(tmp_path)
    assert host.request_checkpointed_move(PAIR, turn_id="t1", boot_build="A") is True
    assert session.latched and session.latched[0]["target_build"] == PAIR
    assert host._updating == PAIR, "the admission window must be open"
    assert session.cancelled == 0


async def test_empty_pair_refuses_and_latches_nothing(tmp_path):
    host, session = _host(tmp_path)
    assert host.request_checkpointed_move("") is False
    assert session.latched == [] and host._updating == ""


async def test_refused_window_cancels_the_latch(tmp_path):
    """A request whose window cannot open must leave nothing half-promised."""
    host, session = _host(tmp_path)
    host._updating = "some-other-pair"  # a live window is already open
    assert host.request_checkpointed_move(PAIR) is False
    assert session.cancelled == 1, "the latch must be cleared, not left to fire"


async def test_ready_to_resume_is_empty_without_a_record(tmp_path):
    host, session = _host(tmp_path)
    assert host.ready_to_resume() == ""


async def test_ready_to_resume_reads_a_resumable_record(tmp_path):
    from local_operator.service_handoff import checkpoint_record
    from local_operator.session.transcript import Transcript

    host, session = _host(tmp_path)
    transcript = Transcript(session.transcript.directory)
    record = checkpoint_record(session_id="s", turn_id="turn-9", committed_message_ids=[])
    await transcript.append_messages([], checkpoint=record)
    assert host.ready_to_resume() == "turn-9"
    # A deferred record is NOT resumable: readiness reads empty.
    deferred = checkpoint_record(
        session_id="s", turn_id="turn-10", committed_message_ids=[], phase="deferred"
    )
    await transcript.append_messages([], checkpoint=deferred)
    assert host.ready_to_resume() == ""


async def test_no_new_admission_is_accepted_while_the_window_is_open(tmp_path):
    """The window refuses new work by spooling — the admission, not a refusal."""
    host, session = _host(tmp_path)
    assert host.request_checkpointed_move(PAIR, turn_id="t") is True
    # ``updating`` reports the open window the admission paths read.
    assert host.updating == PAIR
