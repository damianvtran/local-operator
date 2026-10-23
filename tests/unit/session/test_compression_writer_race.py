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
