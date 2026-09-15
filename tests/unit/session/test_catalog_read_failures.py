"""A read that failed must not be published as a reading.

Three defects, one theme, and every test here is a shape of the same mistake:
an ``OSError``-shaped failure on the catalogue path answering a question the
code was never in a position to answer.

* **An unreadable store reported as an empty one.** ``_scan_sessions`` put only
  the ``os.scandir`` CONSTRUCTOR inside its ``try`` and answered ``[]`` for any
  ``OSError`` — ``EMFILE`` under descriptor exhaustion, ``EACCES``, ``EIO`` —
  so the desktop list route answered ``200 {"sessions": []}`` and the sidebar,
  which adopts that answer as MEMBERSHIP, wiped its visible catalogue. A store
  that is simply NOT THERE is the one case that really is empty, and the tests
  below pin that boundary from both sides.
* **A mid-scan failure escaping as a bare 500.** The iterator was unguarded, so
  a directory that grew or rotated under the poll produced a half-built listing
  (or a bare ``OSError`` no route handler maps) instead of the same surfaced
  unavailability as a failed open.
* **A swallowed degradation published as a confident negative.** ``decorate_rows``
  swallowed the live-record scan, the wake index and the attention read, leaving
  the rows' defaults in place — and a defaulted ``live_state=""`` is
  indistinguishable from a measured "this session is cold", so the wire said
  ``active: false`` about a store with a running turn in it and the sidebar
  rendered "Nothing running right now" over it.

The renderer half of the fix lives in the UI repository; what is pinned here is
what the WIRE says, which is all this repository owns.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from local_operator.resume import _scan_sessions
from local_operator.session.catalog import load_catalog
from local_operator.session.errors import SessionStoreUnavailable


def _session(root: Path, session_id: str, name: str = "") -> Path:
    """One user-visible session directory, with the activity the scan requires."""
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        json.dumps({"type": "message", "role": "user", "content": name or session_id}) + "\n",
        encoding="utf-8",
    )
    return directory


def _store(root: Path, *session_ids: str) -> Path:
    for session_id in session_ids:
        _session(root, session_id)
    return root / "sessions"


def _failing_open(
    monkeypatch: pytest.MonkeyPatch, store: Path, error: OSError, *, nth: int = 1
) -> None:
    """Make the ``nth`` ``scandir`` OF ``store`` raise, and leave the rest alone.

    Scoped to the store's own path rather than patched globally: the catalogue
    reads several directories on one call, and a test that broke all of them
    would pass for the wrong reason.
    """
    real = os.scandir
    seen = {"count": 0}

    def fake(path: Any = ".", *args: Any, **kwargs: Any) -> Any:
        if isinstance(path, (str, Path)) and Path(path) == store:
            seen["count"] += 1
            if seen["count"] == nth:
                raise error
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", fake)


class _ExplodingScan:
    """A scan that yields ``after`` entries and then dies, as a rotating dir does."""

    def __init__(self, real: Any, after: int) -> None:
        self._real = real
        self._after = after
        self._seen = 0

    def __enter__(self) -> "_ExplodingScan":
        self._real.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._real.__exit__(*exc)

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        self._seen += 1
        if self._seen > self._after:
            raise OSError(errno.EIO, "Input/output error")
        return next(self._real)


def _failing_iteration(
    monkeypatch: pytest.MonkeyPatch, store: Path, error: OSError, *, after: int = 1
) -> None:
    real = os.scandir

    def fake(path: Any = ".", *args: Any, **kwargs: Any) -> Any:
        scan = real(path, *args, **kwargs)
        if isinstance(path, (str, Path)) and Path(path) == store:
            return _ExplodingScan(scan, after)
        return scan

    monkeypatch.setattr(os, "scandir", fake)


# --- defect 1: an unreadable store is not an empty one --------------------------


def test_an_unreadable_store_raises_instead_of_listing_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store holds two conversations; the read cannot be served, so it is not."""
    store = _store(tmp_path, "aaaaaaaaaaaa", "bbbbbbbbbbbb")
    _failing_open(monkeypatch, store, OSError(errno.EMFILE, "Too many open files"))

    with pytest.raises(SessionStoreUnavailable) as caught:
        load_catalog(tmp_path)

    # The cause is kept, so the log says WHAT failed even though the message
    # that crosses the transport names no errno and no path.
    assert isinstance(caught.value.__cause__, OSError)
    assert caught.value.__cause__.errno == errno.EMFILE
    assert str(tmp_path) not in str(caught.value), "a transport-bound message must name no path"
    # The conversations were never actually gone.
    assert len(list((tmp_path / "sessions").iterdir())) == 2


def test_a_store_that_is_not_there_is_still_an_empty_listing(tmp_path: Path) -> None:
    """Cold start and a fresh install keep answering "no conversations".

    This is the half of the boundary that must NOT regress: the scan swallows a
    missing store on purpose, and turning that into an error would break every
    client's first run to fix a failure they were not having.
    """
    assert not (tmp_path / "sessions").exists()
    assert load_catalog(tmp_path) == []


def test_a_sessions_entry_that_is_a_file_is_not_an_empty_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Absent" and "present but unwalkable" are different answers.

    Pinned explicitly because the cheap version of this fix swallows every
    ``OSError`` including ``ENOTDIR`` — something IS at the store's path, it
    just is not a directory, and reporting that as "you have no conversations"
    is the same lie in a less common costume.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "sessions").write_text("not a directory", encoding="utf-8")

    with pytest.raises(SessionStoreUnavailable):
        load_catalog(tmp_path)


def test_a_tolerant_caller_still_gets_the_empty_answer_and_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The documented best-effort contract survives — but no longer silently.

    Search, the phone's listing and the ``/resume`` picker answer display-only
    questions and document that a listing which fails is worse than one that is
    empty. They keep that. What they do not keep is the invisibility: the
    incident is logged at WARNING, because ``debug`` is invisible at the default
    level and an operator could not reconstruct from the logs why a catalogue
    had gone empty for a while.
    """
    store = _store(tmp_path, "aaaaaaaaaaaa")
    _failing_open(monkeypatch, store, OSError(errno.EACCES, "Permission denied"))

    with caplog.at_level(logging.WARNING, logger="local_operator.resume"):
        rows, hidden = _scan_sessions(tmp_path)

    assert (rows, hidden) == ([], set())
    assert any(record.levelno == logging.WARNING for record in caplog.records)
    assert "session store could not be read" in caplog.records[-1].getMessage()


# --- defect 2: the iterator is guarded like its constructor ----------------------


def test_a_scan_that_dies_mid_iteration_raises_instead_of_truncating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Half a listing is not a listing: the rows never reached read as deleted."""
    store = _store(tmp_path, "aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc")
    _failing_iteration(monkeypatch, store, OSError(errno.EIO, "Input/output error"))

    with caplog.at_level(logging.WARNING, logger="local_operator.resume"):
        with pytest.raises(SessionStoreUnavailable) as caught:
            load_catalog(tmp_path)

    assert isinstance(caught.value.__cause__, OSError)
    assert caught.value.__cause__.errno == errno.EIO
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_the_last_entry_of_a_scan_does_not_trip_the_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard must fire on a FAILURE, not on the scan's own end.

    A generator-based guard is one ``StopIteration`` away from turning every
    successful scan into a refused one, and the whole store would be unreachable
    — so the boundary is pinned rather than assumed.
    """
    store = _store(tmp_path, "aaaaaaaaaaaa")
    _failing_iteration(monkeypatch, store, OSError(errno.EIO, "Input/output error"), after=99)

    entries = load_catalog(tmp_path)

    assert [entry.id for entry in entries] == ["aaaaaaaaaaaa"]


def test_a_probe_that_cannot_read_the_store_surfaces_instead_of_dropping_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The desktop-marker probe is a second read of the same store, same rule.

    The rows that loop contributes are the desktop-created sessions with no
    transcript YET — the newest thing in the store, and the row its owner is
    most likely to be looking for. Omitting them silently for a directory that
    could not be read is the defect one layer down, so it surfaces instead.
    """
    store = _store(tmp_path, "aaaaaaaaaaaa")
    # The scan's own open is the first; the probe's is the second.
    _failing_open(monkeypatch, store, OSError(errno.EMFILE, "Too many open files"), nth=2)

    with pytest.raises(SessionStoreUnavailable):
        load_catalog(tmp_path)


def test_a_store_without_a_desktop_marker_is_probed_without_error(tmp_path: Path) -> None:
    """The common answer to the probe's stat is ENOENT, and it stays a non-match.

    The ``except OSError: continue`` INSIDE the probe loop is a different
    question from the one above — "no ``desktop.json`` here" — and it must not
    have been swept up with it.
    """
    _store(tmp_path, "aaaaaaaaaaaa", "bbbbbbbbbbbb")

    entries = load_catalog(tmp_path)

    assert {entry.id for entry in entries} == {"aaaaaaaaaaaa", "bbbbbbbbbbbb"}


# --- defect 3: uncertainty is carried, not asserted away -------------------------


def _live_store(tmp_path: Path) -> None:
    """Two sessions and a REAL published record: one of them is running."""
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import SessionRecord

    _store(tmp_path, "aaaaaaaaaaaa", "bbbbbbbbbbbb")
    (tmp_path / "run" / "mobile").mkdir(parents=True, exist_ok=True)
    registry.publish(
        SessionRecord(
            pid=os.getpid(),
            kind="tui",
            session_id="aaaaaaaaaaaa",
            conversation_name="alpha",
            cwd=str(tmp_path),
            model_label="p/m",
            control_port=1234,
            control_key="k",
            detached=False,
            busy=True,
        ),
        tmp_path,
    )


def _degraded(entries: list[Any]) -> dict[str, tuple[str, ...]]:
    return {entry.id: entry.row.degraded for entry in entries}


def test_a_failed_live_read_is_named_on_every_row(tmp_path: Path, monkeypatch) -> None:
    """A swallowed scan used to read as "nothing is running" — say so instead.

    The record is real and published, so the same store answers ``active: true``
    when the read succeeds (asserted in the test below). Here the read raises,
    and the rows carry the fact that their ``live_state`` is a default rather
    than a verdict — which is what lets a renderer say "I could not tell"
    instead of asserting a negative it never established.
    """
    from local_operator.session.runtime import registry

    _live_store(tmp_path)

    def explode(_directory: Path) -> Any:
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(registry, "scan", explode)
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Capture()
    catalog_logger = logging.getLogger("local_operator.session.catalog")
    catalog_logger.addHandler(handler)
    try:
        entries = load_catalog(tmp_path)
    finally:
        catalog_logger.removeHandler(handler)

    assert _degraded(entries) == {
        "aaaaaaaaaaaa": ("liveness",),
        "bbbbbbbbbbbb": ("liveness",),
    }
    live = next(entry for entry in entries if entry.id == "aaaaaaaaaaaa")
    # The defaults are untouched — this is a labelling change, not a claim that
    # something is running — but the row no longer reads as a measured "cold".
    assert live.row.live_state == "" and live.active is False
    assert any(record.levelno == logging.WARNING for record in records)


def test_a_healthy_poll_names_no_source(tmp_path: Path) -> None:
    """The field is only ever a report of a failure."""
    _live_store(tmp_path)

    entries = load_catalog(tmp_path)
    live = next(entry for entry in entries if entry.id == "aaaaaaaaaaaa")

    assert live.active is True and live.row.live_state == "busy"
    assert all(entry.row.degraded == () for entry in entries)


def test_a_failed_wake_read_is_named_and_the_rows_survive(tmp_path: Path, monkeypatch) -> None:
    """The wake index is a decoration too: its failure costs the mark, not the row."""
    import local_operator.wakes.store as wake_store

    _store(tmp_path, "aaaaaaaaaaaa", "bbbbbbbbbbbb")

    def explode(_directory: Path) -> Any:
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(wake_store, "read_index", explode)

    entries = load_catalog(tmp_path)

    assert {entry.id for entry in entries} == {"aaaaaaaaaaaa", "bbbbbbbbbbbb"}
    assert set(_degraded(entries).values()) == {("wakes",)}


def test_a_failed_attention_read_is_named(tmp_path: Path, monkeypatch) -> None:
    """``unseen`` is part of ``active``, so its default is a claim about reading.

    This is the third swallowed read and the one furthest from where the wire
    is built: a failed attention read leaves ``unseen=False``, which both drops
    an unread completion's mark and moves the row out of the Active section.
    """
    from local_operator.session.attention import AttentionStore

    _store(tmp_path, "aaaaaaaaaaaa", "bbbbbbbbbbbb")

    def explode(_self: Any, _identities: Any) -> Any:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(AttentionStore, "state_many", explode)

    entries = load_catalog(tmp_path)

    assert set(_degraded(entries).values()) == {("attention",)}
