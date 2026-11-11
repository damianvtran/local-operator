"""The temporary-chat mark: exact-schema reader, admission, Keep, tombstones.

``session/temporary.py`` decides which conversations the cleanup pass may
delete, so every reading that is not EXACTLY the shape it writes must come out
"not temporary" (retained). These cells pin that direction.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from local_operator.session import temporary
from local_operator.session.temporary import (
    CODE_IN_PROJECT,
    CODE_MARK_FAILED,
    CODE_NOT_FOUND,
    CODE_NOT_USER,
    CODE_PINNED,
    MARK_NAME,
    STATE_NAME,
    TOMBSTONES_NAME,
    clamp_idle_minutes,
    deleted_at,
    is_marked,
    is_marked_path,
    pin_refusal,
    read_mark,
    read_state,
    read_tombstones,
    row_info,
    write_mark,
)

SID = "a" * 12


def _session(root: Path, session_id: str = SID, *, origin: str | None = None) -> Path:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text('{"type":"message"}\n', encoding="utf-8")
    if origin is not None:
        (directory / "origin.json").write_text(json.dumps({"origin": origin}), encoding="utf-8")
    return directory


def _raw_mark(directory: Path, payload: object) -> None:
    (directory / MARK_NAME).write_text(
        payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
    )


# -- the reader: exact schema or nothing -------------------------------------


def test_a_written_mark_reads_back(tmp_path: Path) -> None:
    directory = _session(tmp_path)
    assert write_mark(tmp_path, SID, True, by="tui").ok
    mark = read_mark(directory)
    assert mark is not None and mark.session_id == SID and mark.by == "tui"
    assert is_marked(tmp_path, SID) and is_marked_path(directory)


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "{",
        '{"v": 1, "temporary": true, "session_id": "aaaaaaaaaaaa", "marked_at": 1',  # torn
        "[]",
        "null",
        {"v": 2, "temporary": True, "session_id": SID, "marked_at": 1.0},
        {"v": 1, "temporary": "true", "session_id": SID, "marked_at": 1.0},
        {"v": 1, "temporary": 1, "session_id": SID, "marked_at": 1.0},
        {"v": 1, "temporary": True, "session_id": "b" * 12, "marked_at": 1.0},
        {"v": 1, "temporary": True, "marked_at": 1.0},
        {"v": 1, "temporary": True, "session_id": SID},
        {"v": 1, "temporary": True, "session_id": SID, "marked_at": "1"},
        {"v": 1, "temporary": True, "session_id": SID, "marked_at": True},
        {"v": 1, "temporary": True, "session_id": SID, "marked_at": time.time() + 3600},
        {"v": 1, "temporary": False, "session_id": SID, "kept_at": 1.0},
    ],
)
def test_every_other_shape_reads_as_not_temporary(tmp_path: Path, payload: object) -> None:
    directory = _session(tmp_path)
    _raw_mark(directory, payload)
    assert read_mark(directory) is None
    assert not is_marked_path(directory)


def test_a_non_finite_marked_at_is_not_temporary(tmp_path: Path) -> None:
    directory = _session(tmp_path)
    (directory / MARK_NAME).write_text(
        '{"v": 1, "temporary": true, "session_id": "%s", "marked_at": NaN}' % SID,
        encoding="utf-8",
    )
    assert read_mark(directory) is None


def test_a_cut_multibyte_character_does_not_raise(tmp_path: Path) -> None:
    directory = _session(tmp_path)
    (directory / MARK_NAME).write_bytes(b'{"v": 1, "by": "\xe2\x82')
    assert read_mark(directory) is None


def test_a_malformed_id_is_never_temporary(tmp_path: Path) -> None:
    assert not is_marked(tmp_path, "../x")
    assert not is_marked(tmp_path, "")


# -- writing: Keep is a write, the mtime is preserved -------------------------


def test_keep_is_a_write_not_an_unlink(tmp_path: Path) -> None:
    directory = _session(tmp_path)
    assert write_mark(tmp_path, SID, True, by="desktop").ok
    assert write_mark(tmp_path, SID, False, by="desktop").ok
    payload = json.loads((directory / MARK_NAME).read_text(encoding="utf-8"))
    assert payload["temporary"] is False and "kept_at" in payload
    assert read_mark(directory) is None


def test_marking_is_not_activity(tmp_path: Path) -> None:
    directory = _session(tmp_path)
    os.utime(directory, (1000.0, 1000.0))
    assert write_mark(tmp_path, SID, True, by="cli").ok
    assert directory.stat().st_mtime == 1000.0
    # No temp file is left behind.
    assert sorted(p.name for p in directory.iterdir()) == [MARK_NAME, "transcript.jsonl"]


def test_a_missing_session_is_not_found(tmp_path: Path) -> None:
    assert write_mark(tmp_path, SID, True, by="cli").code == CODE_NOT_FOUND
    assert write_mark(tmp_path, "../etc", True, by="cli").code == CODE_NOT_FOUND


def test_a_write_that_cannot_land_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _session(tmp_path)
    monkeypatch.setattr(temporary, "_write_mark_file", lambda *_a, **_k: False)
    assert write_mark(tmp_path, SID, True, by="cli").code == CODE_MARK_FAILED


# -- admission ----------------------------------------------------------------


@pytest.mark.parametrize("origin", ["subagent", "agent-shell", "agent-workstream", "agent-config"])
def test_only_the_users_own_or_a_fork_may_be_marked(tmp_path: Path, origin: str) -> None:
    directory = _session(tmp_path, origin=origin)
    assert write_mark(tmp_path, SID, True, by="cli").code == CODE_NOT_USER
    assert not (directory / MARK_NAME).exists()


def test_a_fork_may_be_marked(tmp_path: Path) -> None:
    _session(tmp_path, origin="fork")
    assert write_mark(tmp_path, SID, True, by="cli").ok


def test_a_pinned_session_is_refused(tmp_path: Path) -> None:
    _session(tmp_path)
    (tmp_path / "sidebar-pins.json").write_text(json.dumps([SID]), encoding="utf-8")
    assert write_mark(tmp_path, SID, True, by="cli").code == CODE_PINNED


def test_unreadable_pins_refuse_the_mark(tmp_path: Path) -> None:
    _session(tmp_path)
    (tmp_path / "sidebar-pins.json").write_text("{not json", encoding="utf-8")
    assert write_mark(tmp_path, SID, True, by="cli").code == CODE_MARK_FAILED


def test_a_project_linked_session_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session(tmp_path)
    monkeypatch.setattr(temporary, "_linked_to_project", lambda *_a: True)
    assert write_mark(tmp_path, SID, True, by="cli").code == CODE_IN_PROJECT


def test_keep_has_no_admission(tmp_path: Path) -> None:
    _session(tmp_path, origin="subagent")
    assert write_mark(tmp_path, SID, False, by="cli").ok


def test_pin_refusal_names_the_code(tmp_path: Path) -> None:
    _session(tmp_path)
    assert pin_refusal(tmp_path, SID) == ""
    write_mark(tmp_path, SID, True, by="cli")
    assert pin_refusal(tmp_path, SID) == "session_temporary"


# -- tombstones ----------------------------------------------------------------


def _tombstones(root: Path, entries: object) -> None:
    (root / "sessions").mkdir(parents=True, exist_ok=True)
    (root / "sessions" / TOMBSTONES_NAME).write_text(json.dumps(entries), encoding="utf-8")


def test_a_tombstone_answers_only_when_the_directory_is_gone(tmp_path: Path) -> None:
    now = time.time()
    _tombstones(tmp_path, [{"id": SID, "deleted_at": now - 10}])
    assert deleted_at(tmp_path, SID, now=now) == now - 10
    _session(tmp_path)  # a removal that was refused: the session is alive
    assert deleted_at(tmp_path, SID, now=now) is None


def test_tombstones_prune_at_read(tmp_path: Path) -> None:
    now = time.time()
    _tombstones(
        tmp_path,
        [
            {"id": "b" * 12, "deleted_at": now - 8 * 86400},
            {"id": "c" * 12, "deleted_at": now},
            {"id": "../x", "deleted_at": now},
            {"id": "d" * 12, "deleted_at": "soon"},
            "junk",
        ],
    )
    assert [entry["id"] for entry in read_tombstones(tmp_path, now=now)] == ["c" * 12]


def test_tombstones_are_capped(tmp_path: Path) -> None:
    now = time.time()
    _tombstones(tmp_path, [{"id": f"{i:012x}", "deleted_at": now} for i in range(1200)])
    assert len(read_tombstones(tmp_path, now=now)) == temporary.TOMBSTONE_CAP


def test_reads_create_nothing(tmp_path: Path) -> None:
    (tmp_path / "sessions").mkdir()
    assert read_tombstones(tmp_path) == []
    assert deleted_at(tmp_path, SID) is None
    assert read_state(tmp_path)["sessions"] == {}
    assert list((tmp_path / "sessions").iterdir()) == []


# -- the window and the row object ---------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(15, 15), (30, 30), (1, 15), (16, 30), (31, 120), (121, 1440), (99999, 1440),
     ("30", 30), ("junk", 30), (None, 30), (True, 30), (29.5, 30)],
)
def test_the_window_clamps_up(value: object, expected: int) -> None:
    assert clamp_idle_minutes(value) == expected


def test_row_info_without_a_hold(tmp_path: Path) -> None:
    directory = _session(tmp_path)
    write_mark(tmp_path, SID, True, by="cli", now=1000.0)
    info = row_info(directory, idle_minutes=30, state=None, now=1100.0, activity=900.0)
    assert info == {"marked_at": 1000.0, "idle_minutes": 30, "deletes_at": 2800.0, "hold": None}


def test_row_info_with_a_fresh_hold_and_a_stale_one(tmp_path: Path) -> None:
    directory = _session(tmp_path)
    write_mark(tmp_path, SID, True, by="cli", now=1000.0)
    state = {"v": 1, "evaluated_at": 1050.0, "sessions": {SID: {"held_at": 1050.0, "hold": "running"}}}
    held = row_info(directory, idle_minutes=30, state=state, now=1100.0, activity=900.0)
    assert held is not None and held["hold"] == "running" and held["deletes_at"] is None
    stale = row_info(directory, idle_minutes=30, state=state, now=1050.0 + 200, activity=900.0)
    assert stale is not None and stale["hold"] is None
    # The window restarts from the end of the hold.
    assert stale["deletes_at"] == 1050.0 + 1800


def test_row_info_is_none_for_a_kept_chat(tmp_path: Path) -> None:
    directory = _session(tmp_path)
    write_mark(tmp_path, SID, False, by="cli")
    assert row_info(directory, idle_minutes=30, state=None) is None


def test_state_file_name_is_store_level() -> None:
    assert STATE_NAME.startswith(".") and TOMBSTONES_NAME.startswith(".")
