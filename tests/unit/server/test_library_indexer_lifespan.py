"""The server lifespan stops the library indexer it started.

WHY THIS FILE EXISTS. Every in-process boot (``with TestClient(app)``) runs the
lifespan, and the lifespan calls ``library.indexer.ensure_started`` — which
starts two DAEMON threads, ``library-indexer`` and ``library-thumbs``. Nothing
stopped them at shutdown, so they outlived the test. Pytest then deleted the
passing test's ``tmp_path`` (``tmp_path_retention_policy = "failed"``) under
them, ``metadata._connect`` mkdir'd the directory back and switched a FRESH
database file to WAL while the thumbs thread still held connections to the old
one, and the xdist worker died with SIGBUS (CI: main run 43765368385, job
132982207581; PR #4223 control job 135255830431, ``worker gw1 crashed`` in
``test_desktop_aida_boot_gate.py``).

The assertions are the two observable symptoms: a live indexer thread after the
client exits, and a deleted config dir that something recreates.
"""

from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from local_operator.library import indexer
from local_operator.server.app import app

_INDEXER_THREADS = frozenset({"library-indexer", "library-thumbs"})


def _cached(root: Path) -> indexer.LibraryIndexer | None:
    # Read the cache directly rather than through a helper this fix adds, so the
    # same file also runs (and fails) against the pre-fix lifespan.
    return indexer._INSTANCES.get(str(root.resolve()))


def _running(root: Path) -> bool:
    instance = _cached(root)
    return instance is not None and instance.running


def _indexer_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name in _INDEXER_THREADS and t.is_alive()]


@pytest.fixture
def scratch_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    # Same redirection as the boot-gate tests, with Aida's boot ensure switched
    # off: this test is about the indexer's lifetime, not her bootstrap.
    root = tmp_path / ".local-operator"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_AIDA", "1")
    # A neighbour's leaked indexer would make the "no thread alive" assertion
    # about someone else's boot; start and end with none.
    indexer.reset_for_tests()
    yield root
    indexer.reset_for_tests()


def test_lifespan_exit_stops_the_indexer_and_never_recreates_a_deleted_root(
    scratch_root: Path,
) -> None:
    with TestClient(app):
        # Positive control: the boot really started the threads we then expect
        # gone, so a passing assertion below is not a boot that never indexed.
        assert _running(scratch_root), "the lifespan must start the indexer"
        assert {t.name for t in _indexer_threads()} >= {"library-indexer"}
        booted = _cached(scratch_root)
        assert booted is not None

    # What pytest's tmp_path cleanup does to a passing test, then a reconcile
    # request: a leaked worker answers it by reopening the database, which
    # mkdirs the deleted root back (the SIGBUS precursor). On a stopped
    # instance ``wake`` only enqueues, and nothing is left to read the queue.
    shutil.rmtree(scratch_root)
    booted.wake()
    # A condition wait, not a sleep: it ends the moment the root reappears or
    # the moment no indexer thread is left that could recreate it, and the
    # bound only covers a leaked worker's coalesce window.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if scratch_root.exists() or not _indexer_threads():
            break
        threading.Event().wait(0.05)
    assert not scratch_root.exists(), "a deleted config root was recreated after shutdown"

    assert _indexer_threads() == [], "lifespan shutdown left indexer threads alive"
    assert not _running(scratch_root)


def test_lifespan_leaves_an_indexer_it_did_not_start_running(scratch_root: Path) -> None:
    # Ownership: a running indexer the boot found (a lazy ``list_assets`` start,
    # a caller's own) belongs to whoever started it, not to this lifespan.
    scratch_root.mkdir(parents=True)
    owned = indexer.ensure_started(scratch_root)
    assert owned.running
    with TestClient(app):
        pass
    assert owned.running, "the lifespan stopped an indexer it did not start"
