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
