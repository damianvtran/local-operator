"""The durable archive index: shape, cap, atomic replace, and prune-at-read.

The store is small, so the tests that matter are the ones about its FAILURE
modes rather than its happy path. Every one of them is a bug the sibling pin
store has already been pictured shipping, and the reasoning is in
``local_operator/session/archived.py`` beside the code:

* **Unreadable degrades to "nothing is archived".** The file only ever NARROWS
  a listing that ranks perfectly well without it, so a truncated or hand-edited
  file must cost the user the archive and not the picker, the sidebar or the
  search.
* **Prune-at-read against the store.** An id whose directory is gone (retention
  swept it, or an explicit delete removed it) is dropped on the way OUT, which
  is what lets ``cleanup`` know nothing about archives — asserted here and again
  in ``test_session_delete.py``, because "deletion needs no cooperation" is a
  claim about a second module.
* **A no-op writes nothing.** Two frontends pressing the same state must not
  write the whole index against each other, and a re-archive must not reorder
  the list.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from local_operator.session.archived import (
    ARCHIVED_FILE,
    ARCHIVED_LIMIT,
    archived_ids,
    read_archived,
    set_archived,
)

A = "a" * 12
B = "b" * 12
C = "c" * 12


def _session(config_dir: Path, session_id: str) -> Path:
    directory = config_dir / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    return directory


def _file(config_dir: Path) -> Path:
    return config_dir / ARCHIVED_FILE


def test_the_file_is_a_bare_json_array(tmp_path: Path) -> None:
    """Not an object, deliberately — see the module docstring.

    The shape is asserted rather than assumed because the sibling pin store's
    docstring REFUSES to become an object for the same reason, and a later
    "let us add a version field" edit would otherwise look like a local change
    to this file alone.
    """
    _session(tmp_path, A)
    set_archived(tmp_path, A, True)
    assert json.loads(_file(tmp_path).read_text()) == [A]


def test_an_absent_file_reads_as_nothing_archived(tmp_path: Path) -> None:
    assert read_archived(tmp_path) == []
    assert archived_ids(tmp_path) == frozenset()


@pytest.mark.parametrize(
    "raw",
    [
        "{",  # truncated
        '{"archived": ["a"]}',  # an object rather than an array
        '["a", 7, null]',  # non-strings
        "null",
    ],
)
def test_an_unreadable_file_degrades_to_no_archive(tmp_path: Path, raw: str) -> None:
    _session(tmp_path, A)
    _file(tmp_path).write_text(raw, encoding="utf-8")
    assert read_archived(tmp_path) == []


def test_an_id_that_is_gone_from_the_store_is_pruned_at_read(tmp_path: Path) -> None:
    """The whole reason deletion needs no cooperation from this module.

    The entry is left ON DISK — nothing here rewrites the file — and the reader
    is what refuses to report it. If this ever moves to a write-time prune, the
    claim in ``cleanup`` and in the desktop delete route becomes false and a
    deletion would have to learn about archives.
    """
    _session(tmp_path, A)
    _session(tmp_path, B)
    set_archived(tmp_path, A, True)
    set_archived(tmp_path, B, True)
    shutil.rmtree(tmp_path / "sessions" / A)
    assert read_archived(tmp_path) == [B]
    # The record is still there: the prune is a READ, not a write.
    assert A in json.loads(_file(tmp_path).read_text())


@pytest.mark.parametrize("entry", ["/tmp", "../agents", "a/b", "", ".", "..", "a\x00b"])
def test_a_stored_entry_cannot_escape_the_store(tmp_path: Path, entry: str) -> None:
    """A file this app wrote must not be able to redirect a read outside ``sessions/``.

    ``Path.__truediv__`` does not keep an id inside the store — ``sessions /
    "/tmp"`` IS ``/tmp`` — so the shape check has to run before the prune joins
    the id onto the store directory.
    """
    outside = tmp_path / "agents"
    outside.mkdir()
    _file(tmp_path).write_text(json.dumps([entry]), encoding="utf-8")
    assert read_archived(tmp_path) == []


def test_setting_the_same_state_twice_writes_nothing(tmp_path: Path) -> None:
    """Idempotence is the property the HTTP route's retry safety rests on.

    Asserted by the file's inode rather than by its contents: a rewrite that
    happens to produce the same bytes is exactly the rewrite this must not do,
    because it is the one that can land inside another writer's window and
    discard what they added.
    """
    _session(tmp_path, A)
    _session(tmp_path, B)
    set_archived(tmp_path, A, True)
    set_archived(tmp_path, B, True)
    before = _file(tmp_path).stat().st_ino
    assert set_archived(tmp_path, A, True) is True
    assert _file(tmp_path).stat().st_ino == before
    # And the ORDER is untouched, which is the same fact read the other way: a
    # newest-first store that re-applied the entry would move it to the front.
    assert read_archived(tmp_path) == [B, A]


def test_unarchiving_an_unarchived_session_writes_nothing(tmp_path: Path) -> None:
    _session(tmp_path, A)
    assert set_archived(tmp_path, A, False) is False
    assert not _file(tmp_path).exists()


def test_the_last_unarchive_leaves_an_empty_array(tmp_path: Path) -> None:
    """The resting state, not an absent file — one write path, not two."""
    directory = _session(tmp_path, A)
    set_archived(tmp_path, A, True)
    assert set_archived(tmp_path, A, False) is False
    assert json.loads(_file(tmp_path).read_text()) == []
    assert read_archived(tmp_path) == []
    assert directory.is_dir()


def test_the_cap_drops_the_oldest_and_keeps_the_newest(tmp_path: Path) -> None:
    """A bound on the FILE, not a policy: the newest archive record always survives."""
    ids = [f"{index:012x}" for index in range(ARCHIVED_LIMIT + 5)]
    for session_id in ids:
        _session(tmp_path, session_id)
    for session_id in ids:
        set_archived(tmp_path, session_id, True)
    stored = read_archived(tmp_path)
    assert len(stored) == ARCHIVED_LIMIT
    assert stored[0] == ids[-1], "the archive just made is the one position can never drop"
    assert ids[0] not in stored


def test_a_read_only_config_root_costs_the_archive_and_not_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never raises, on either verb: the caller has already told the user something.

    Simulated by making the replace fail rather than by chmod, which is not
    reliable as a non-root user across filesystems. The contract under test is
    the swallow, not the errno.

    The RETURN VALUE is the desired state even though nothing was written, and
    that is inherited from the pin store rather than invented here: the store's
    verb reports the state it was asked for, and a read-back is the race the pin
    route's docstring already refuses (two writers inside one window). The
    honest reading — recorded so nobody calls it a bug later — is that a client
    on a read-only root renders an archive the store does not hold until its
    next listing settles the row.
    """
    _session(tmp_path, A)

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("read-only")

    monkeypatch.setattr(os, "replace", explode)
    assert set_archived(tmp_path, A, True) is True
    assert read_archived(tmp_path) == []


def test_a_failed_write_reports_no_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 2, NIT-2: the clause describes what happened to the FILE.

    ``archive_change`` returns the ids the cap dropped so the receipt can name the
    conversation that came back into every list. On a config root this process
    cannot write, nothing came back — the file did not change — so reporting the
    ids the slice computed would promise a revocation of the archive that did not
    happen. The STATE echo stays the desired state (that is the pin route's
    contract, asserted in the test above); the eviction report does not.
    """
    from local_operator.session.archived import archive_change

    for index in range(ARCHIVED_LIMIT):
        _session(tmp_path, f"{index:012x}")

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("read-only")

    monkeypatch.setattr(os, "replace", explode)
    state, evicted = archive_change(tmp_path, A, True)

    assert state is True
    assert evicted == [], "nothing was dropped, because nothing was written"
