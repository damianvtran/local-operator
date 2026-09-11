"""A bookkeeping append must not move the transcript's mtime.

The transcript's mtime IS the session's activity clock
(``retention.session_activity_path``, THE ONE RANKING CLOCK). Every append
moves it, which is right for a turn and wrong for the boot-time
``session_incident`` journal: that row is bookkeeping ABOUT a session, never
work done IN it. Measured on the real store before the fix, 19 of 509 rows
displayed an age wrong by more than a minute, worst case 13.0 h, because an
incident written hours after the last turn restamped the file
(``FINDING-resume-clock.md``).

These tests pin the write-side behaviour at the seam that implements it:
``Transcript._write_entries`` restores the pre-append mtime when, and only
when, its caller asked for it. The opt-in default is as load-bearing as the
restore — a flag that leaked into ordinary appends would freeze the clock
instead of ignoring one kind of write.
"""

from __future__ import annotations

import os

import pytest

from local_operator.harness.types import CustomMessage, Message
from local_operator.incidents import SESSION_INCIDENT_MESSAGE_TYPE
from local_operator.session.transcript import TranscriptEntry

#: A stamp far enough in the past that no filesystem timestamp granularity can
#: confuse "restored" with "just written". Three days matches the backdating in
#: the FINDING's own reproduction.
PAST = 1_700_000_000.0


def _incident() -> CustomMessage:
    return CustomMessage(
        custom_type=SESSION_INCIDENT_MESSAGE_TYPE,
        attribution="system",
        details={"text": "provider returned 401 unauthorized", "raw": "401"},
    )


async def _seeded(transcript) -> None:
    """One ordinary turn, then the clock wound back to :data:`PAST`.

    Seeding through the real writer rather than hand-writing a file: the
    restore reads the mtime of a file the writer created, so a fixture that
    invented one could pass while the product failed.
    """
    await transcript.append_message(Message.user("real work"))
    os.utime(transcript.path, (PAST, PAST))


@pytest.mark.asyncio
async def test_a_bookkeeping_append_leaves_the_transcript_mtime_where_it_was(tmp_path):
    """The defect this ticket closes, at its narrowest seam."""
    from local_operator.session.transcript import Transcript

    transcript = Transcript(tmp_path / "sess")
    await _seeded(transcript)
    before_lines = len(transcript.path.read_text(encoding="utf-8").splitlines())

    await transcript.append_message(_incident(), preserve_mtime=True)

    assert transcript.path.stat().st_mtime == pytest.approx(PAST, abs=1e-6)

    lines = transcript.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == before_lines + 1
    entry = TranscriptEntry.from_json(lines[-1])
    assert entry is not None
    assert entry.payload["custom_type"] == SESSION_INCIDENT_MESSAGE_TYPE


@pytest.mark.asyncio
async def test_an_ordinary_append_still_advances_the_transcript_mtime(tmp_path):
    """The flag is opt-in. Real work must still rank as real work."""
    from local_operator.session.transcript import Transcript

    transcript = Transcript(tmp_path / "sess")
    await _seeded(transcript)

    await transcript.append_message(Message.user("more real work"))

    assert transcript.path.stat().st_mtime > PAST


@pytest.mark.asyncio
async def test_the_size_still_changes_so_stat_based_caches_still_see_the_append(tmp_path):
    """Asserted directly, because every stat-keyed cache downstream depends on
    it: ``search_index.build_index`` keys on ``[st_size, st_mtime,
    title_mtime]`` and ``mobile.durable.DurableFoldCache`` on ``(inode, size,
    mtime)``. A restored mtime cannot hide an append while the size moves.
    """
    from local_operator.session.transcript import Transcript

    transcript = Transcript(tmp_path / "sess")
    await _seeded(transcript)
    before = transcript.path.stat().st_size

    await transcript.append_message(_incident(), preserve_mtime=True)

    assert transcript.path.stat().st_size > before


@pytest.mark.asyncio
async def test_a_failed_utime_does_not_lose_the_incident(tmp_path, monkeypatch):
    """Best-effort by contract, the same as ``resume.write_session_title``'s.

    A failed restore costs one wrong age on one picker row; a raised one costs
    the incident itself, and the incident is the only thing telling the model
    why its last run died.
    """
    import local_operator.session.transcript as transcript_mod
    from local_operator.session.transcript import Transcript

    transcript = Transcript(tmp_path / "sess")
    await _seeded(transcript)

    def refuse(*args, **kwargs):
        raise OSError("read-only volume")

    monkeypatch.setattr(transcript_mod.os, "utime", refuse)

    entry = await transcript.append_message(_incident(), preserve_mtime=True)

    assert entry.payload["custom_type"] == SESSION_INCIDENT_MESSAGE_TYPE
    lines = transcript.path.read_text(encoding="utf-8").splitlines()
    assert TranscriptEntry.from_json(lines[-1]) is not None


@pytest.mark.asyncio
async def test_a_rebuild_does_not_attempt_a_restore(tmp_path):
    """A transcript recreated from memory has no meaningful prior mtime.

    The rebuild branch is left alone entirely: it recreates the whole file from
    committed history, so there is no "before" to restore to, and the birth
    stamp it writes through ``ensure_session_created_at`` is the timestamp that
    matters there.
    """
    from local_operator.session.transcript import Transcript

    transcript = Transcript(tmp_path / "sess")
    await _seeded(transcript)
    committed = len(transcript.path.read_text(encoding="utf-8").splitlines())
    transcript.path.unlink()

    await transcript.append_message(_incident(), preserve_mtime=True)

    lines = transcript.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == committed + 1
    assert TranscriptEntry.from_json(lines[-1]) is not None


@pytest.mark.asyncio
async def test_the_flag_refuses_a_batch_carrying_real_work(tmp_path):
    """``preserve_mtime`` must not be able to freeze the clock over a turn.

    QA round 1. ``BOOKKEEPING_CUSTOM_TYPES`` is defined in ``transcript.py``
    and cited by both :meth:`Transcript.append_message` and
    :meth:`Transcript._write_entries` as the rule that governs this flag —
    "an append carrying only :data:`BOOKKEEPING_CUSTOM_TYPES`" — but no code
    path evaluates it: the restore keys on the boolean alone. A caller that
    passes ``preserve_mtime=True`` on a batch containing a real user message
    therefore erases that turn from ``retention.session_activity``, which is
    the ONE RANKING CLOCK shared with ``session.cleanup``: the session ranks
    as older than it is on the picker AND ages toward deletion.

    Only ``Session.journal_incident`` passes the flag today, so this is not
    reachable in production — it is one careless caller away, and nothing
    guards it. The docstring's claim is the contract; this pins it.
    """
    from local_operator.session.retention import session_activity
    from local_operator.session.transcript import Transcript

    transcript = Transcript(tmp_path / "sess")
    await _seeded(transcript)
    before = session_activity(transcript.directory)
    assert before == pytest.approx(PAST, abs=1e-6)

    await transcript.append_messages(
        [_incident(), Message.user("deploy the release")],
        preserve_mtime=True,
    )

    assert "deploy the release" in transcript.path.read_text(encoding="utf-8")
    after = session_activity(transcript.directory)
    assert after is not None and after > PAST, (
        "a batch carrying a real user message was clock-frozen: the append "
        "claims to honour BOOKKEEPING_CUSTOM_TYPES but never reads it"
    )


@pytest.mark.asyncio
async def test_a_failed_bookkeeping_append_does_not_move_the_clock_either(tmp_path, monkeypatch):
    """The rollback path restores too, because the bytes roll back and the
    clock does not roll back with them.

    ``open("a")`` stamps the file before the write fails and ``os.truncate``
    stamps it again, so a failed bookkeeping append used to advance the
    activity clock by the full age of the transcript — measured at +24763 h on
    a forced fsync failure (QA round 2). That is the very defect this module
    exists to close, reintroduced through the error path, so it is pinned here
    rather than left to the success path's coverage.
    """
    import local_operator.session.transcript as transcript_mod
    from local_operator.session.retention import session_activity
    from local_operator.session.transcript import Transcript

    transcript = Transcript(tmp_path / "sess")
    await _seeded(transcript)
    before = session_activity(transcript.directory)
    size_before = transcript.path.stat().st_size

    def failing_fsync(fd):
        raise OSError(5, "I/O error")

    monkeypatch.setattr(transcript_mod.os, "fsync", failing_fsync)

    with pytest.raises(OSError):
        await transcript.append_message(_incident(), preserve_mtime=True)

    assert transcript.path.stat().st_size == size_before
    assert session_activity(transcript.directory) == pytest.approx(before, abs=1e-6)
