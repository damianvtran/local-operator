"""Successor resume: a delivered-once spooled row is not delivered twice.

Exercises the real boot drain (``_drain_inbox_into``) with a checkpoint record
whose ``inbox_consumed_ids`` name rows already delivered on the predecessor.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.session.runtime import process as child_mod
from local_operator.session.runtime.inbox import InboxLine, append_inbox


def _engaged(directory: Path) -> None:
    """A durable history, so the boot drain treats this session as engaged."""
    (directory / "transcript.jsonl").write_text(
        '{"id":"h1","ts":1,"type":"message","payload":{"kind":"message",'
        '"role":"user","content":[]}}\n',
        encoding="utf-8",
    )


async def _write_checkpoint(directory: Path, *, consumed: list[str], phase: str = "resume") -> None:
    from local_operator.service_handoff import checkpoint_record
    from local_operator.session.transcript import Transcript

    record = checkpoint_record(
        session_id="s",
        turn_id="t",
        committed_message_ids=[],
        inbox_consumed_ids=consumed,
        phase=phase,
    )
    await Transcript(directory).append_messages([], checkpoint=record)


class Handle:
    def __init__(self, directory: Path, seen: list[str]) -> None:
        self._session = SimpleNamespace(transcript=SimpleNamespace(directory=directory))
        self._seen = seen

    async def receive_peer_message(
        self, text, *, mode="mailbox", wake=False, sender=None, message_id=None
    ):
        self._seen.append(text)
        return "recorded"


@pytest.mark.asyncio
async def test_a_delivered_once_row_is_not_delivered_again(tmp_path: Path) -> None:
    _engaged(tmp_path)
    append_inbox(tmp_path, InboxLine(text="already ran", sender={}, message_id="m1"))
    append_inbox(tmp_path, InboxLine(text="still owed", sender={}, message_id="m2"))
    await _write_checkpoint(tmp_path, consumed=["m1"])
    seen: list[str] = []
    assert await child_mod._drain_inbox_into(Handle(tmp_path, seen)) == 1
    assert seen == ["still owed"], "the consumed row must not be delivered again"


@pytest.mark.asyncio
async def test_no_record_delivers_everything(tmp_path: Path) -> None:
    _engaged(tmp_path)
    append_inbox(tmp_path, InboxLine(text="a", sender={}, message_id="m1"))
    append_inbox(tmp_path, InboxLine(text="b", sender={}, message_id="m2"))
    seen: list[str] = []
    assert await child_mod._drain_inbox_into(Handle(tmp_path, seen)) == 2
    assert seen == ["a", "b"]


@pytest.mark.asyncio
async def test_deferred_record_does_not_filter(tmp_path: Path) -> None:
    """A deferred record is not permission to suppress a delivery."""
    _engaged(tmp_path)
    append_inbox(tmp_path, InboxLine(text="a", sender={}, message_id="m1"))
    await _write_checkpoint(tmp_path, consumed=["m1"], phase="deferred")
    seen: list[str] = []
    assert await child_mod._drain_inbox_into(Handle(tmp_path, seen)) == 1
    assert seen == ["a"]


def test_resume_consumed_ids_is_empty_without_a_record(tmp_path: Path) -> None:
    assert child_mod._resume_consumed_ids(tmp_path) == set()
