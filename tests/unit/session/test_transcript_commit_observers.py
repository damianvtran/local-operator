"""All-row notifications follow durable publication, never streamed settlement."""

import json
import threading

import pytest

from local_operator.harness.types import Message
from local_operator.session.transcript import Transcript


@pytest.mark.asyncio
async def test_committed_rows_notify_after_fsync_on_the_owner_loop(tmp_path):
    transcript = Transcript(tmp_path / "sess")
    loop_thread = threading.get_ident()
    seen = []

    def observed(rows, source):
        assert threading.get_ident() == loop_thread
        assert all(transcript.has_entry(row.id) for row in rows)
        disk = [json.loads(line)["id"] for line in transcript.path.read_text().splitlines()]
        assert disk[-len(rows) :] == [row.id for row in rows]
        assert source[0] == transcript.path.stat().st_ino
        assert source[1] == transcript.path.stat().st_size
        seen.append([row.id for row in rows])

    unsubscribe = transcript.subscribe_commits(observed)
    a = await transcript.append_message(Message.user("synthetic send"))
    c = await transcript.append_custom("session_model_switch", {"text": "synthetic switch"})
    assert seen == [[a.id], [c.id]], "not only command-id user admissions are notified"
    unsubscribe()
    unsubscribe()
    await transcript.append_message(Message.assistant("after unsubscribe"))
    assert seen == [[a.id], [c.id]]


@pytest.mark.asyncio
async def test_failed_durable_write_publishes_no_commit(tmp_path, monkeypatch):
    transcript = Transcript(tmp_path / "sess")
    seen = []
    transcript.subscribe_commits(lambda rows, source: seen.append(rows))

    def fail(*args, **kwargs):
        raise OSError("synthetic failed fsync")

    monkeypatch.setattr(transcript, "_write_entries", fail)
    with pytest.raises(OSError, match="synthetic failed fsync"):
        await transcript.append_message(Message.user("never durable"))
    assert seen == []
    assert list(transcript.entries()) == []


@pytest.mark.asyncio
async def test_observer_failure_cannot_undo_commit(tmp_path):
    transcript = Transcript(tmp_path / "sess")
    seen = []

    def failed(rows, source):
        raise RuntimeError("synthetic observer failure")

    transcript.subscribe_commits(failed)
    transcript.subscribe_commits(lambda rows, source: seen.extend(row.id for row in rows))
    entry = await transcript.append_message(Message.user("still durable"))
    assert transcript.has_entry(entry.id)
    assert seen == [entry.id]
