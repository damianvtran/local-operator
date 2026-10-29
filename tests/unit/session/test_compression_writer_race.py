"""The two write-path races compression opened, pinned as behaviour.

L1 — THE OPEN HOLDS ITS LOCK, OR THE CONVERSATION IS LOST. ``Transcript``
restores a compressed journal and then reads it. A compressor that held the
session's lock across that window publishes its archive and unlinks the plain
file after the restore found the plain file present; the runtime's next append
then creates a fresh, empty plain journal that every reader prefers over the
archive holding the real history. The acquisition is therefore a BOUNDED WAIT:
long enough to outlast one session's compress, short enough not to stall a
launch, and still proceed-with-a-warning when a second real runtime holds it —
because that hazard is a different one, and refusing would turn contention into
a session that cannot open.

L2 — A VANISHED PLAIN FILE IS NOT A MISSING JOURNAL. The append path heals a
vanished journal by rewriting it from memory. That is right for a directory an
external sweep removed and catastrophic when a compression pass replaced the
file: the rewrite starts at whatever this runtime replayed, shadowing the
archive beside it. The archive is restored before that decision, and an archive
this build cannot decode refuses the append rather than shadowing it.
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from local_operator.session import compression


def _store(tmp_path: Path) -> Path:
    config = tmp_path / "config"
    (config / "sessions").mkdir(parents=True)
    return config


def _rows(count: int = 4) -> list[dict[str, object]]:
    return [
        {
            "id": f"row{index:03d}",
            "ts": float(index),
            "type": "message",
            "payload": {"role": "user" if index % 2 == 0 else "assistant", "content": f"t{index}"},
        }
        for index in range(count)
    ]


def _journal(count: int = 4) -> bytes:
    return b"".join(json.dumps(row).encode() + b"\n" for row in _rows(count))


def _session(config: Path, name: str = "race01") -> Path:
    directory = config / "sessions" / name
    directory.mkdir(parents=True)
    (directory / compression.PLAIN_FILENAME).write_bytes(_journal())
    return directory


# ---------------------------------------------------------------------------
# L1 — the writer lock is waited for, and the wait is bounded
# ---------------------------------------------------------------------------


_HOLDER = """
import fcntl, os, sys, time
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
print("ready", flush=True)
time.sleep(float(sys.argv[2]))
os.close(fd)
"""


def _hold_lock(directory: Path, hold_s: float) -> subprocess.Popen[str]:
    """A REAL second holder: the lock is kernel state, so the test needs a process."""
    lock = directory / compression.LOCK_NAME
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, os.fspath(lock), str(hold_s)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "ready"
    return holder


def test_acquire_writer_waits_out_a_short_holder(tmp_path: Path) -> None:
    directory = _session(_store(tmp_path))
    holder = _hold_lock(directory, 0.3)
    try:
        started = time.monotonic()
        assert compression.acquire_writer(directory, wait_s=5.0) is True
        elapsed = time.monotonic() - started
    finally:
        holder.wait(timeout=10)
        compression.release_writer(directory)
    # It waited for the holder rather than failing on the first attempt, and it
    # did not wait out the whole budget.
    assert 0.2 <= elapsed < 5.0, elapsed


def test_acquire_writer_without_a_wait_still_answers_busy_at_once(tmp_path: Path) -> None:
    directory = _session(_store(tmp_path), "race02")
    holder = _hold_lock(directory, 1.0)
    try:
        started = time.monotonic()
        assert compression.acquire_writer(directory) is False
        assert time.monotonic() - started < 0.5
    finally:
        holder.wait(timeout=10)


# ---------------------------------------------------------------------------
# L2 — an append restores the archive that replaced its plain file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_append_restores_an_archive_that_replaced_the_plain_file(
    tmp_path: Path,
) -> None:
    from local_operator.session.transcript import Transcript

    directory = _session(_store(tmp_path), "race03")
    original = (directory / compression.PLAIN_FILENAME).read_bytes()
    live = Transcript(directory)
    try:
        assert live._writer_lock_held is True
        # THE RACE, SIMULATED: a verified archive appears under the archive name
        # and the plain file goes away, exactly as the compressor's publish+unlink
        # leaves the directory — after this runtime looked and saw a plain file.
        archive = directory / (compression.PLAIN_FILENAME + ".gz")
        archive.write_bytes(gzip.compress(original))
        (directory / compression.PLAIN_FILENAME).unlink()

        await live.append_custom("note", {"text": "after the race"})

        plain = directory / compression.PLAIN_FILENAME
        assert plain.exists()
        written = plain.read_bytes()
        # The archived rows were restored BEFORE the append: the file starts with
        # the original journal, not with an empty file plus one new row.
        assert written.startswith(original)
        reread = Transcript(directory)
        try:
            assert [entry.id for entry in reread.entries()] == [
                "row000",
                "row001",
                "row002",
                "row003",
                live.entries()[-1].id,
            ]
            assert reread.latest_custom("note") == {"text": "after the race"}
        finally:
            reread.close()
    finally:
        live.close()


@pytest.mark.asyncio
async def test_an_undecodable_archive_refuses_the_append_instead_of_shadowing_it(
    tmp_path: Path,
) -> None:
    from local_operator.session.transcript import Transcript

    directory = _session(_store(tmp_path), "race04")
    live = Transcript(directory)
    try:
        archive = directory / (compression.PLAIN_FILENAME + ".gz")
        archive.write_bytes(b"not a gzip stream at all")
        (directory / compression.PLAIN_FILENAME).unlink()

        with pytest.raises(OSError, match="cannot restore"):
            await live.append_custom("note", {"text": "must not land"})

        # Nothing shadowed it: the archive is untouched and no plain file was
        # invented from this runtime's memory.
        assert archive.read_bytes() == b"not a gzip stream at all"
        assert not (directory / compression.PLAIN_FILENAME).exists()
    finally:
        live.close()


# ---------------------------------------------------------------------------
# L3 — the removal is atomic with the decision (review round 1, M2)
# ---------------------------------------------------------------------------


def _late_row() -> bytes:
    return (
        json.dumps(
            {
                "id": "rowlate",
                "ts": 99.0,
                "type": "message",
                "payload": {"role": "user", "content": "written in the window"},
            }
        ).encode()
        + b"\n"
    )


def test_a_late_append_in_the_retirement_window_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE REPRO: a writer appends after the publish, before the removal.

    The old order (re-stat, then unlink) deleted the plain file the row had just
    landed in, and the archive — a copy of the PRE-append bytes — became the only
    copy: a silent hole. The protocol now renames the plain file first, so the
    append lands in the renamed inode and the stat of THAT file sees it.
    """
    directory = _session(_store(tmp_path), "race05")
    real_publish = compression._publish

    def publishing_with_a_late_writer(tmp: Path, destination: Path) -> None:
        if Path(destination).name == compression.QUARANTINE_NAME:
            # Exactly where a writer that timed out on the lock appends: the
            # plain path, an instant before the compressor renames it.
            with (directory / compression.PLAIN_FILENAME).open("ab") as handle:
                handle.write(_late_row())
                handle.flush()
                os.fsync(handle.fileno())
        real_publish(tmp, destination)

    monkeypatch.setattr(compression, "_publish", publishing_with_a_late_writer)

    outcome = compression.compress_session(directory)

    assert outcome.status == "skipped", outcome
    plain = directory / compression.PLAIN_FILENAME
    assert plain.exists()
    ids = [json.loads(line)["id"] for line in plain.read_bytes().splitlines()]
    assert "rowlate" in ids, ids
    assert ids[:4] == ["row000", "row001", "row002", "row003"], ids
    # The archive published from the pre-append bytes is withdrawn: two forms
    # that disagree is the state this protocol exists to avoid.
    assert compression.archive_path(directory) is None
    assert not (directory / compression.QUARANTINE_NAME).exists()


def test_a_normal_pass_leaves_no_quarantine(tmp_path: Path) -> None:
    directory = _session(_store(tmp_path), "race06")
    assert compression.compress_session(directory).status == "compressed"
    assert not (directory / compression.QUARANTINE_NAME).exists()
    assert not (directory / compression.PLAIN_FILENAME).exists()


def test_a_crash_between_the_rename_and_the_unlink_is_recovered(tmp_path: Path) -> None:
    """quarantine + archive + no plain: the open puts the plain file back."""
    from local_operator.session.transcript import Transcript

    directory = _session(_store(tmp_path), "race07")
    original = (directory / compression.PLAIN_FILENAME).read_bytes()
    assert compression.compress_session(directory, codec="gzip").status == "compressed"
    archive = compression.archive_path(directory)
    assert archive is not None

    # The exact on-disk state a crash at the rename leaves: the verified archive
    # published, and the plain file parked under the quarantine name.
    quarantine = directory / compression.QUARANTINE_NAME
    quarantine.write_bytes(original)
    assert not (directory / compression.PLAIN_FILENAME).exists()

    assert compression.restore_for_write(directory) is True

    plain = directory / compression.PLAIN_FILENAME
    assert plain.read_bytes() == original
    assert compression.archive_path(directory) is None
    assert not quarantine.exists()
    # And the runtime opens it with every row, which is the point of recovering.
    transcript = Transcript(directory)
    try:
        assert [entry.id for entry in transcript.entries()] == [row["id"] for row in _rows()]
    finally:
        transcript.close()


def test_the_pass_recovers_a_quarantine_before_skipping_the_session(tmp_path: Path) -> None:
    """The scan's own recovery: otherwise the session reads as already compressed."""
    config = _store(tmp_path)
    directory = _session(config, "race08")
    original = (directory / compression.PLAIN_FILENAME).read_bytes()
    # Old enough to be a candidate, in the crash state, with no archive at all:
    # the plain file is the only copy and it is parked under the dot-name.
    stamp = time.time() - 90 * 86400.0
    quarantine = directory / compression.QUARANTINE_NAME
    quarantine.write_bytes(original)
    os.utime(quarantine, (stamp, stamp))
    os.utime(directory, (stamp, stamp))
    (directory / compression.PLAIN_FILENAME).unlink()

    result = compression.run_compression(
        config, compression.CompressionPolicy(enabled=True, after_days=30), budget_s=60.0
    )

    plain = directory / compression.PLAIN_FILENAME
    assert plain.exists(), result
    assert plain.read_bytes() == original
    assert not quarantine.exists()
    # The recovery happened BEFORE the eligibility decision, so this session went
    # on to be compressed normally rather than skipped as "already compressed".
    assert result.compressed or "already compressed" not in result.skipped_by_reason


# ---------------------------------------------------------------------------
# Round 2 m2 — a quarantine that vanishes under the pass
# ---------------------------------------------------------------------------


def test_a_restore_inside_the_quarantine_window_skips_the_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE REPRO (review m2): ``restore_for_write`` while the plain file is parked.

    A reader on a low-disk device restores the transcript from the archive
    inside the window. The quarantine is then gone, so the pass can no longer
    prove the archived bytes are the source's — and it used to report
    ``compressed`` with no archive on disk, after re-keying the search index for
    a file that was not there. "Cannot prove unchanged" is a skip.
    """
    directory = _session(_store(tmp_path), "race09")
    real_publish = compression._publish
    restores: list[bool] = []

    def publishing_with_a_restore(tmp: Path, destination: Path) -> None:
        real_publish(tmp, destination)
        if Path(destination).name == compression.QUARANTINE_NAME:
            restores.append(compression.restore_for_write(directory))

    indexed: list[str] = []
    monkeypatch.setattr(compression, "_publish", publishing_with_a_restore)
    monkeypatch.setattr(compression, "_after_archive", lambda *args: indexed.append(str(args[1])))

    outcome = compression.compress_session(directory)

    assert restores == [True], restores
    assert outcome.status == "skipped", outcome
    assert "restored" in outcome.reason, outcome
    # No archive, no listing sidecar, and no search-index entry for a file that
    # no longer exists.
    assert compression.archive_path(directory) is None
    assert not (directory / compression.PREVIEW_SIDECAR_NAME).exists()
    assert indexed == []
    # The conversation is intact: a restore is a rename, not a rewrite.
    plain = directory / compression.PLAIN_FILENAME
    ids = [json.loads(line)["id"] for line in plain.read_bytes().splitlines()]
    assert ids == [row["id"] for row in _rows()]


def test_a_deleted_quarantine_with_no_plain_file_keeps_the_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other vanish: nothing survives but the archive, so it must stay.

    The branch above withdraws the archive because the restore put the plain
    file back. A quarantine removed by something else leaves NO copy, and the
    archive is the verified one — withdrawing it would delete the conversation.
    """
    directory = _session(_store(tmp_path), "race10")
    real_publish = compression._publish

    def publishing_then_dropping_the_quarantine(tmp: Path, destination: Path) -> None:
        real_publish(tmp, destination)
        if Path(destination).name == compression.QUARANTINE_NAME:
            os.unlink(destination)

    monkeypatch.setattr(compression, "_publish", publishing_then_dropping_the_quarantine)

    outcome = compression.compress_session(directory)

    assert outcome.status == "compressed", outcome
    archive = compression.archive_path(directory)
    assert archive is not None and archive.is_file()
    rows = list(compression.iter_lines(directory))
    assert [json.loads(line)["id"] for line in rows] == [row["id"] for row in _rows()]
