"""The sidebar's durable pinned-session list.

Two properties carry this module. The first is that a pin SURVIVES — it is the
only thing the sidebar ranks by that is not live state, so a torn or half-written
file costs the user a deliberate choice rather than a cache. The second is that
it never costs them anything else: a pin is recorded immediately after a keypress
the user has already been given feedback for, so every failure mode here —
unreadable file, unwritable directory, a pin to a session that has since been
deleted — degrades to "no pins" instead of raising into the TUI.

Everything runs against a real temporary filesystem rather than mocks, for the
reason ``test_move_targets.py`` does: these functions exist to answer questions
about files, and a mocked filesystem would assert that the code calls what it
calls rather than that it gives right answers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from local_operator.tui.sidebar_pins import PINS_FILE, PINS_LIMIT, read_pins, toggle_pin


def _session(config: Path, session_id: str) -> Path:
    """A session directory, so the read-path prune keeps this id."""
    directory = config / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def test_an_absent_file_reads_as_no_pins(tmp_path: Path) -> None:
    """Reading must not be a write: a sidebar that has never pinned anything
    should leave no trace in the config directory."""
    assert read_pins(tmp_path) == []
    assert not (tmp_path / PINS_FILE).exists()


def test_a_pin_round_trips(tmp_path: Path) -> None:
    _session(tmp_path, "aaaaaaaaaaaa")
    assert toggle_pin(tmp_path, "aaaaaaaaaaaa") is True
    assert read_pins(tmp_path) == ["aaaaaaaaaaaa"]


def test_a_second_toggle_unpins(tmp_path: Path) -> None:
    """One chord is both verbs, so the return value is the NEW state."""
    _session(tmp_path, "aaaaaaaaaaaa")
    toggle_pin(tmp_path, "aaaaaaaaaaaa")
    assert toggle_pin(tmp_path, "aaaaaaaaaaaa") is False
    assert read_pins(tmp_path) == []


def test_the_newest_pin_leads(tmp_path: Path) -> None:
    for session_id in ("a" * 12, "b" * 12, "c" * 12):
        _session(tmp_path, session_id)
        toggle_pin(tmp_path, session_id)
    assert read_pins(tmp_path) == ["c" * 12, "b" * 12, "a" * 12]


def test_repinning_moves_it_to_the_front(tmp_path: Path) -> None:
    """Unpin then repin is how a user promotes an old pin, and the list is
    newest-first, so the repinned id must lead rather than return to its slot."""
    for session_id in ("a" * 12, "b" * 12):
        _session(tmp_path, session_id)
        toggle_pin(tmp_path, session_id)
    toggle_pin(tmp_path, "a" * 12)
    toggle_pin(tmp_path, "a" * 12)
    assert read_pins(tmp_path) == ["a" * 12, "b" * 12]


def test_the_cap_holds(tmp_path: Path) -> None:
    """The OLDEST pins are what a full list drops, never the newest."""
    for index in range(PINS_LIMIT + 10):
        session_id = f"{index:012x}"
        _session(tmp_path, session_id)
        toggle_pin(tmp_path, session_id)
    pins = read_pins(tmp_path)
    assert len(pins) == PINS_LIMIT
    assert pins[0] == f"{PINS_LIMIT + 9:012x}"
    assert f"{0:012x}" not in pins


@pytest.mark.parametrize("body", ["not json", '{"pins": []}'])
def test_a_corrupt_file_degrades_to_no_pins(tmp_path: Path, body: str) -> None:
    """Best-effort by contract: the sidebar ranks perfectly well without pins,
    so an unreadable file must never be the thing that costs the user the list."""
    (tmp_path / PINS_FILE).write_text(body)
    assert read_pins(tmp_path) == []


def test_non_string_members_are_dropped(tmp_path: Path) -> None:
    _session(tmp_path, "abc123abc123")
    (tmp_path / PINS_FILE).write_text(json.dumps([1, None, "abc123abc123", ""]))
    assert read_pins(tmp_path) == ["abc123abc123"]


def test_a_pin_to_a_deleted_session_is_pruned_at_read(tmp_path: Path) -> None:
    """Pruning on the READ path is what keeps this module free of any
    coordination with ``session/cleanup.py``: deletion needs to know nothing
    about pins, and a pin to a cleaned-up session renders as nothing rather
    than as a broken row."""
    _session(tmp_path, "a" * 12)
    toggle_pin(tmp_path, "a" * 12)
    toggle_pin(tmp_path, "b" * 12)
    assert read_pins(tmp_path) == ["a" * 12]


def test_an_id_bearing_a_path_separator_is_rejected(tmp_path: Path) -> None:
    """A pin is a session id — ONE bare directory name — and the check runs
    BEFORE the store-prune, because the prune is what would follow the escape:
    it joins the id onto ``sessions/``, and ``sessions / "/tmp"`` IS ``/tmp``
    while ``../agents`` climbs out of the store entirely. ``..`` is rejected
    too, and would otherwise survive both halves — ``Path("..").name`` is
    ``".."`` and ``sessions/..`` is a real directory.

    Defence in depth rather than a live bug: nothing renders from a bogus
    entry today, since ``load_catalog`` hydrates none of them. This keeps the
    house rule — discovery metadata cannot redirect a read outside
    ``sessions/`` — true of this file as well.
    """
    _session(tmp_path, "real")
    (tmp_path / "agents").mkdir()

    (tmp_path / PINS_FILE).write_text(json.dumps(["../agents", "..", "/tmp"]))
    assert read_pins(tmp_path) == []

    (tmp_path / PINS_FILE).write_text(
        json.dumps(["real", "../agents", "..", ".", "/tmp", "sessions/../../etc", "real/../real"])
    )
    assert read_pins(tmp_path) == ["real"]


def test_the_write_is_atomic(tmp_path: Path) -> None:
    """Same-directory temp then replace. Asserted by proving no temp file
    survives a successful write — a torn read would silently empty the pins."""
    _session(tmp_path, "a" * 12)
    toggle_pin(tmp_path, "a" * 12)
    leftovers = [path.name for path in tmp_path.iterdir() if path.name.startswith(".sidebar-pins-")]
    assert leftovers == []
    assert json.loads((tmp_path / PINS_FILE).read_text()) == ["a" * 12]


def test_a_failing_write_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The keypress has already been acknowledged on screen when this runs."""

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("no space left on device")

    _session(tmp_path, "a" * 12)
    monkeypatch.setattr(os, "replace", boom)
    assert toggle_pin(tmp_path, "a" * 12) is True
    leftovers = [path.name for path in tmp_path.iterdir() if path.name.startswith(".sidebar-pins-")]
    assert leftovers == [], "the temp file must not survive a failed replace"


@pytest.mark.skipif(os.getuid() == 0, reason="root writes to unwritable directories anyway")
def test_an_unwritable_directory_never_raises(tmp_path: Path) -> None:
    config = tmp_path / "readonly"
    config.mkdir()
    config.chmod(0o500)
    try:
        assert toggle_pin(config, "a" * 12) is True
        assert read_pins(config) == []
    finally:
        config.chmod(0o700)


def test_last_writer_wins(tmp_path: Path) -> None:
    """Documents the accepted multi-process behaviour rather than guarding
    against it: two sessions pinning at once means the second write is what the
    file holds. No precedent here takes a cross-process lock for a small index,
    and the atomic replace means a reader sees an older list, never a torn one."""
    for session_id in ("a" * 12, "b" * 12):
        _session(tmp_path, session_id)
    # Both "processes" observed the same empty base state.
    assert read_pins(tmp_path) == []
    toggle_pin(tmp_path, "a" * 12)
    toggle_pin(tmp_path, "b" * 12)
    assert read_pins(tmp_path) == ["b" * 12, "a" * 12]
