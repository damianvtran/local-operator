"""``librarian.state`` — identity, tolerant reads, atomic writes, kill switch."""

from __future__ import annotations

import json
from pathlib import Path

from local_operator.librarian import state
from tests.unit.aida.conftest import tree


def test_env_switch_truthiness(monkeypatch) -> None:
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv(state.ENV_DISABLE, value)
        assert state.env_disabled() is True
    for value in ("0", "false", "no", "off", ""):
        monkeypatch.setenv(state.ENV_DISABLE, value)
        assert state.env_disabled() is False
    monkeypatch.delenv(state.ENV_DISABLE)
    assert state.env_disabled() is False


def test_identity_is_false_without_a_state_file_and_writes_nothing(isolated_root: Path) -> None:
    before = tree(isolated_root)
    assert state.is_librarian_session(isolated_root, "abc123") is False
    assert tree(isolated_root) == before


def test_identity_matches_only_the_recorded_session(isolated_root: Path) -> None:
    state.update_state(isolated_root, session_id="abc123")
    assert state.is_librarian_session(isolated_root, "abc123") is True
    assert state.is_librarian_session(isolated_root, "other") is False
    assert state.is_librarian_session(isolated_root, "") is False


def test_corrupt_state_reads_as_absent(isolated_root: Path) -> None:
    path = state.state_path(isolated_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert state.read_state(isolated_root) is None
    # A write over unusable state is allowed: the writers are authoritative
    # about the fields they set, and refusing would strand every later
    # mutation behind one bad byte.
    state.update_state(isolated_root, session_id="fresh")
    assert state.session_id_of(isolated_root) == "fresh"


def test_unknown_schema_reads_as_absent(isolated_root: Path) -> None:
    path = state.state_path(isolated_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 99, "session_id": "x"}), encoding="utf-8")
    assert state.read_state(isolated_root) is None


def test_write_is_atomic_and_leaves_no_temp_file(isolated_root: Path) -> None:
    state.write_state(isolated_root, {"session_id": "abc"})
    files = sorted(p.name for p in state.librarian_dir(isolated_root).iterdir())
    assert files == ["state.json"]
    assert json.loads(state.state_path(isolated_root).read_text())["schema_version"] == 1


def test_ensure_lock_is_the_documented_name(isolated_root: Path) -> None:
    lock = state.wake_lock(isolated_root)
    assert lock.path.name == "ensure.lock"
    assert lock.path.parent == state.librarian_dir(isolated_root)


def test_refusal_shape_names_verb_and_configured_name() -> None:
    sentence = state.op_refusal_sentence("pause", "Nova", RuntimeError("busy"))
    assert sentence == "could not pause Nova: busy"
