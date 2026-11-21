"""The durable ``cwd.json`` record: written at birth, rewritten by the runtime (test f).

WHY (coordinator retest on v0.111.4): an idle exec/TUI-created session's only cwd
records were the runtime's discovery record — gone once the runtime exits — so a move
could not find its worktree. ``cwd.json`` is written where every surface's session is
born (the transcript constructor, from ``session_factory``'s cwd) and rewritten where
the runtime publishes its record (``RuntimeServer``'s boot and heartbeat), so it
follows a session that starts in, or moves to, another directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.mobile.types import SessionProjection
from local_operator.paths import config_dir
from local_operator.session import creation
from local_operator.session.transcript import Transcript
from tests.unit.session.runtime.test_server import FakeHandle, _wait_record


def _record(directory: Path) -> dict[str, object]:
    return json.loads((directory / "cwd.json").read_text(encoding="utf-8"))


def test_birth_writes_cwd_json_once_and_never_overwrites_it(tmp_path: Path) -> None:
    session = tmp_path / "sessions" / "s-birth"
    first, second = tmp_path / "first", tmp_path / "second"

    Transcript(session, cwd=str(first)).close()
    Transcript(session, cwd=str(second)).close()  # a reopen elsewhere is not a birth

    record = _record(session)
    assert record["cwd"] == str(first) and record["source"] == "birth" and record["v"] == 1
    assert oct((session / "cwd.json").stat().st_mode & 0o777) == "0o600"


def test_a_deferred_transcript_records_its_cwd_only_when_it_materialises(
    tmp_path: Path,
) -> None:
    import asyncio

    session = tmp_path / "sessions" / "s-deferred"
    transcript = Transcript(session, defer_materialise=True, cwd=str(tmp_path / "w"))
    assert not session.exists()  # a speculative runtime leaves nothing behind

    asyncio.run(transcript.append_custom("probe", {"x": 1}))
    transcript.close()

    assert _record(session)["cwd"] == str(tmp_path / "w")


def test_a_relative_cwd_is_never_recorded(tmp_path: Path) -> None:
    session = tmp_path / "s"
    session.mkdir()
    assert creation.record_session_cwd(session, "relative/dir", source="runtime") is False
    assert not (session / "cwd.json").exists()


class _MovableHandle(FakeHandle):
    """FakeHandle whose projection's session id and cwd the test chooses."""

    def __init__(self, cwd: str) -> None:
        super().__init__()
        self.set_cwd(cwd)

    def set_cwd(self, cwd: str) -> None:
        self._projection = SessionProjection(
            session_id="s1",
            pid=0,
            kind="tui",
            conversation_name="fake",
            cwd=cwd,
            model_label="test/model",
        )


@pytest.mark.asyncio
async def test_f_the_runtime_start_rewrites_cwd_json_when_the_cwd_changes(tmp_path: Path) -> None:
    from local_operator.session.runtime.server import RuntimeServer

    session = Path(config_dir()) / "sessions" / "s1"
    session.mkdir(parents=True)
    creation.write_session_cwd(session, str(tmp_path / "born-here"), source="birth")
    started, moved = tmp_path / "started-here", tmp_path / "moved-here"

    handle = _MovableHandle(str(started))
    server = RuntimeServer(handle, kind="tui")
    await server.start_in_process()
    try:
        await _wait_record()
        record = _record(session)
        assert record["cwd"] == str(started) and record["source"] == "runtime", record

        # The heartbeat's seam: a cwd move changes the seed, and the beat records it.
        handle.set_cwd(str(moved))
        server._note_session_cwd("s1", handle.session_projection_seed.cwd)  # noqa: SLF001
        assert _record(session)["cwd"] == str(moved)
    finally:
        server.close()


# ---------------------------------------------------------------------------
# Review round 1 remediation (PR #4403): L3 (S3) writer identity, L12 heartbeat.
# ---------------------------------------------------------------------------


def test_l3_a_symlinked_or_oversized_record_is_never_read(tmp_path: Path) -> None:
    session = tmp_path / "s"
    session.mkdir()
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text(json.dumps({"cwd": "/planted"}), encoding="utf-8")
    (session / "cwd.json").symlink_to(elsewhere)
    assert creation.read_session_cwd(session) == ""

    (session / "cwd.json").unlink()
    (session / "cwd.json").write_text(
        json.dumps({"cwd": "/big", "pad": "x" * (128 * 1024)}), encoding="utf-8"
    )
    assert creation.read_session_cwd(session) == ""


def test_l3_staging_never_writes_through_a_planted_name(tmp_path: Path) -> None:
    """The old staging name was predictable (pid + thread id) and opened with
    truncation, so a symlink planted there redirected the write. Every staging name is
    random and exclusively created now; nothing outside the session is touched."""
    import os
    import threading

    session = tmp_path / "s"
    session.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me", encoding="utf-8")
    planted = session / f"cwd.json.{os.getpid()}.{threading.get_ident()}.tmp"
    planted.symlink_to(victim)

    assert creation.write_session_cwd(session, str(tmp_path / "w"), source="runtime") is True
    assert victim.read_text(encoding="utf-8") == "keep me"
    assert _record(session)["cwd"] == str(tmp_path / "w")
    assert oct((session / "cwd.json").stat().st_mode & 0o777) == "0o600"
    assert sorted(p.name for p in session.iterdir()) == sorted(["cwd.json", planted.name])


def test_l3_birth_publication_never_clobbers_a_winner(tmp_path: Path) -> None:
    """A birth publishes link-or-fail (like ``ensure_session_created_at``): a record
    another writer already published is never overwritten, whoever checked first."""
    session = tmp_path / "s"
    session.mkdir()
    assert creation.write_session_cwd(session, "/first", source="birth", exclusive=True) is True
    assert creation.write_session_cwd(session, "/second", source="birth", exclusive=True) is False
    assert _record(session)["cwd"] == "/first"
    assert [p.name for p in session.iterdir()] == ["cwd.json"]  # no staging left behind


def test_l3_a_symlinked_session_directory_is_never_written_through(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "s"
    link.symlink_to(real, target_is_directory=True)
    assert creation.write_session_cwd(link, "/x", source="runtime") is False
    assert list(real.iterdir()) == []


@pytest.mark.asyncio
async def test_l12_the_heartbeat_itself_records_a_cwd_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-n1: drive real beats (not the seam directly) and see the record follow."""
    import asyncio

    from local_operator.session.runtime import server as server_module
    from local_operator.session.runtime.server import RuntimeServer

    monkeypatch.setattr(server_module, "HEARTBEAT_INTERVAL_S", 0.05)
    session = Path(config_dir()) / "sessions" / "s1"
    session.mkdir(parents=True)
    started, moved = tmp_path / "started-here", tmp_path / "moved-here"
    handle = _MovableHandle(str(started))
    server = RuntimeServer(handle, kind="tui")
    await server.start_in_process()
    try:
        await _wait_record()
        handle.set_cwd(str(moved))
        for _ in range(200):  # the event is the file changing; bounded at ~10 s
            if (session / "cwd.json").exists() and _record(session)["cwd"] == str(moved):
                break
            await asyncio.sleep(0.05)
        assert _record(session)["cwd"] == str(moved)
    finally:
        server.close()
