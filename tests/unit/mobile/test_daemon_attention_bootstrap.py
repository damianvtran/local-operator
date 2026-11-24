"""The mobile daemon's boot-time attention sweep survives unreadable sessions.

Field report (macOS, lop 0.96.5; still present on 0.102.4): the CLI's daemon ran
on Python 3.13 while the desktop app on 3.14 had compressed transcripts to
``transcript.jsonl.zst``. ``_bootstrap_mobile_attention`` raised
``ArchiveUnavailable`` on the first such session, so the caller never set
``_attention_bootstrapped`` and re-ran the sweep on every 2 s scan (57k+
warnings) while the phone showed "Not answering". The contract pinned here: one
bad session is skipped with one log line naming the session and the exception
class — never transcript content — and the sweep always returns.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from local_operator.mobile import daemon as daemon_module
from local_operator.session import attention, compression

#: Must never appear in a log line: the skip notice names the session, not what
#: it contains.
_SECRET = "transcript-content-must-not-be-logged"


def _plain(sessions: Path, sid: str) -> Path:
    directory = sessions / sid
    directory.mkdir(parents=True)
    (directory / compression.PLAIN_FILENAME).write_text(
        '{"type": "custom", "text": "%s"}\n' % _SECRET, encoding="utf-8"
    )
    return directory


def test_one_unreadable_session_never_aborts_the_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    sessions = tmp_path / "sessions"

    _plain(sessions, "good-a")
    # (i) the field case: a .zst on a build with no zstd decoder at all.
    zst = _plain(sessions, "zst-b")
    (zst / compression.PLAIN_FILENAME).unlink()
    (zst / "transcript.jsonl.zst").write_bytes(b"\x28\xb5\x2f\xfd" + _SECRET.encode())
    monkeypatch.setattr(compression, "_zstd", None)
    monkeypatch.setattr(compression, "_zstd_decoder", None)
    # (ii) plain corruption: a gzip archive that is not gzip.
    gz = _plain(sessions, "gz-c")
    (gz / compression.PLAIN_FILENAME).unlink()
    (gz / "transcript.jsonl.gz").write_bytes(b"\x1f\x8b" + _SECRET.encode())
    # (iii) the migration itself failing on a readable transcript.
    _plain(sessions, "boom-d")
    _plain(sessions, "good-e")

    migrated: list[str] = []

    def _record(transcript, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        name = transcript.directory.name
        if name == "boom-d":
            raise ValueError(_SECRET)
        migrated.append(name)

    monkeypatch.setattr(attention, "bootstrap_transcript", _record)

    with caplog.at_level(logging.INFO, logger=daemon_module.logger.name):
        daemon_module._bootstrap_mobile_attention()  # must not raise

    assert sorted(migrated) == ["good-a", "good-e"]
    messages = [record.getMessage() for record in caplog.records]
    skips = [message for message in messages if "bootstrap skipped session" in message]
    assert sorted(skips) == [
        "mobile attention bootstrap skipped session boom-d: ValueError",
        "mobile attention bootstrap skipped session gz-c: ArchiveCorrupt",
        "mobile attention bootstrap skipped session zst-b: ArchiveUnavailable",
    ]
    assert "finished with 3 skipped session(s)" in caplog.text
    # Only the daemon's own lines are this change's contract (the transcript
    # reader's pre-existing "dropping malformed line" warning is out of scope).
    daemon_lines = [r.getMessage() for r in caplog.records if r.name == daemon_module.logger.name]
    assert daemon_lines and all(_SECRET not in line for line in daemon_lines)


def test_a_clean_store_logs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _plain(tmp_path / "sessions", "good-a")
    migrated: list[str] = []
    monkeypatch.setattr(
        attention,
        "bootstrap_transcript",
        lambda transcript, *a, **k: migrated.append(transcript.directory.name),
    )

    with caplog.at_level(logging.INFO, logger=daemon_module.logger.name):
        daemon_module._bootstrap_mobile_attention()

    assert migrated == ["good-a"]
    assert not [r for r in caplog.records if r.name == daemon_module.logger.name]


# -- the relay's transcript reads never wait on a writer lock (wake latency) --
#
# Both relay callers of ``bootstrap_transcript`` used to open a full
# ``Transcript``: a writer lock waited on for up to 5 s, then the whole journal
# decoded -- to answer one-row questions. The operator's mobile.log carried 165
# "stayed locked for 5.0s" lines from these opens, right after relay restarts,
# while booting runtimes waited on the same locks. ``acquire_writer`` is
# replaced by a stand-in for a lock somebody else holds; a reader that touches
# it is the regression.


def _held_lock(calls: list[Path]):  # noqa: ANN202
    import time

    def acquire(directory: Path, *, wait_s: float = 0.0) -> bool:
        calls.append(directory)
        time.sleep(min(wait_s, 5.0))
        return False

    return acquire


def _session_with_started_row(sessions: Path, sid: str, token: str) -> Path:
    import asyncio

    from local_operator.session.transcript import Transcript

    directory = sessions / sid
    directory.mkdir(parents=True)
    asyncio.run(
        Transcript(directory).append_custom(
            "attention_started", {"conversation_id": f"session/{sid}", "token": token}
        )
    )
    return directory


def test_discovered_death_classification_reads_without_the_writer_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _session_with_started_row(tmp_path / "sessions", "dead-one", "tok-1")
    lock_calls: list[Path] = []
    monkeypatch.setattr(compression, "acquire_writer", _held_lock(lock_calls))
    answered: list[Any] = []

    def _record(transcript, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        entry = transcript.latest_custom_entry("attention_started")
        answered.append((transcript.directory.name, entry.payload["details"]["token"]))
        answered.append(transcript.latest_custom("attention_started"))

    monkeypatch.setattr(attention, "bootstrap_transcript", _record)
    daemon_module._classify_discovered_death("dead-one", reaped_owner=None)

    assert lock_calls == [], "the classification waited on the session's writer lock"
    # The bounded reader answers what the resident Transcript answered.
    assert answered[0] == ("dead-one", "tok-1")
    assert answered[1] == {"conversation_id": "session/dead-one", "token": "tok-1"}


def test_the_restart_sweep_reads_without_the_writer_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _session_with_started_row(tmp_path / "sessions", "swept-a", "tok-a")
    lock_calls: list[Path] = []
    monkeypatch.setattr(compression, "acquire_writer", _held_lock(lock_calls))
    tokens: list[str] = []
    monkeypatch.setattr(
        attention,
        "bootstrap_transcript",
        lambda transcript, *a, **k: tokens.append(
            transcript.latest_custom("attention_started")["token"]
        ),
    )

    daemon_module._bootstrap_mobile_attention()

    assert lock_calls == []
    assert tokens == ["tok-a"]


@pytest.mark.asyncio
async def test_the_first_scan_after_a_restart_does_not_wait_for_the_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new runtime is adopted on the restart's first pass, sweep or no sweep."""
    import asyncio
    import threading

    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import SessionRecord

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions").mkdir()
    release = threading.Event()
    swept: list[bool] = []

    def slow_sweep() -> None:
        # Stands in for 100 transcripts read under load; bounded so a build
        # that awaits it fails on timing rather than hanging.
        release.wait(timeout=5.0)
        swept.append(True)

    monkeypatch.setattr(daemon_module, "_bootstrap_mobile_attention", slow_sweep)
    record = SessionRecord(
        pid=2**22 + 77,
        kind="daemon",
        session_id="freshly-woken",
        conversation_name="freshly-woken",
        cwd=str(tmp_path),
        model_label="m",
        control_port=1,
        control_key="k",
    )
    monkeypatch.setattr(registry, "scan", lambda *a, **k: [(record, "live")])
    daemon = daemon_module.MobileDaemon(port=0, password="pw", dial_registrants=False)

    await asyncio.wait_for(daemon._scan_once(), timeout=30.0)

    assert record.pid in daemon.table.entries, "the new record was not picked up this pass"
    # Structural: the pass finished while the sweep was still held open.
    assert swept == [], "the scan waited for the restart sweep"
    release.set()
    await asyncio.gather(*daemon._scan_side_tasks)
    assert swept == [True], "the sweep must still run, beside the scan"
