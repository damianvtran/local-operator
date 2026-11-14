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
