"""A catalogue fetch lease left by a DEAD process must not stall the next read.

QA round 2 (Q-2, PR #2992): a daemon SIGTERMed while its boot warm held
``radient.listing.json.fetching`` left the lease standing, and the next process's
first cold read waited ``_LISTING_LEASE_WAIT_S`` (measured 2209.5 ms) for a
document the dead holder would never write. A LIVE holder keeps its
cross-process meaning: losers still defer to it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from local_operator.model import catalogue


def _dead_pid() -> int:
    """A pid that WAS a process on this host and has exited and been reaped."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def _plant_lease(document, holder_pid: int) -> None:
    lease = document.with_name(document.name + ".fetching")
    lease.parent.mkdir(parents=True, exist_ok=True)
    lease.write_text(
        json.dumps(
            {
                "holder": f"{holder_pid}:deadbeef",
                "expires_at": time.time() + catalogue._LISTING_LEASE_S,
            }
        ),
        encoding="utf-8",
    )


def test_a_lease_held_by_a_dead_pid_is_taken_at_once(tmp_path) -> None:
    document = catalogue._cache_path("radient.listing", tmp_path)
    _plant_lease(document, _dead_pid())
    calls: list[int] = []

    def fetch():
        calls.append(1)
        return {"models": []}

    began = time.monotonic()
    listing = catalogue.read_listing("radient.listing", fetch, cache_dir=tmp_path)
    elapsed = time.monotonic() - began

    assert calls == [1], "the dead holder's lease must not stop this process fetching"
    assert listing.fetched is True
    assert elapsed < catalogue._LISTING_LEASE_WAIT_S / 2, f"waited {elapsed:.3f}s"
    assert not document.with_name(document.name + ".fetching").exists()


def test_a_lease_held_by_a_live_pid_still_defers(tmp_path) -> None:
    """Control: the cross-process lease is unchanged for a holder that is alive."""
    document = catalogue._cache_path("radient.listing", tmp_path)
    _plant_lease(document, os.getppid())
    lease = catalogue._ListingFetchLease(document)
    assert lease.acquire() is False


def test_release_held_leases_frees_only_this_processs_own(tmp_path) -> None:
    """Teardown drops what THIS process holds and leaves a peer's standing."""
    mine = catalogue._ListingFetchLease(catalogue._cache_path("radient.listing", tmp_path))
    assert mine.acquire() is True
    peer_document = catalogue._cache_path("openrouter.listing", tmp_path)
    _plant_lease(peer_document, os.getppid())

    assert catalogue.release_held_leases() == 1
    assert not mine._path.exists()
    assert peer_document.with_name(peer_document.name + ".fetching").exists()


class _Race:
    """Drive ONE holder thread and ONE loser thread through ``read_listing``.

    Deterministic, with no sleeps (review M2): the holder's fetch blocks until the
    test releases it, and the loser announces it has LOST the lease by tripping a
    wrapper around ``_held_by_this_process`` — the first thing a loser does. The
    holder is released only after that, so the loser is provably waiting on it.

    ``release_before_lookup`` forces the window review M1 measured (holder
    finishes between the loser's lost ``acquire`` and its sibling lookup): the
    wrapper itself lets the holder finish and joins it before looking up.
    """

    def __init__(self, tmp_path, monkeypatch, *, outcome, release_before_lookup=False):
        import threading

        self.tmp_path = tmp_path
        self.calls: list[str] = []
        self.results: dict[str, catalogue.Listing] = {}
        self.elapsed: list[float] = []
        self._outcome = outcome
        self._inside = threading.Event()
        self._finish = threading.Event()
        self._lost = threading.Event()
        lookup = catalogue._held_by_this_process

        def announced_lookup(path):
            self._lost.set()
            if release_before_lookup:
                self._finish.set()
                self.holder.join(5)
            return lookup(path)

        monkeypatch.setattr(catalogue, "_held_by_this_process", announced_lookup)
        self.holder = threading.Thread(target=self._read, args=("warm",), name="warm")
        self.loser = threading.Thread(target=self._timed_read, name="open")

    def _fetch(self):
        import threading

        self.calls.append(threading.current_thread().name)
        self._inside.set()
        self._finish.wait(5)
        return self._outcome()

    def _read(self, slot):
        self.results[slot] = catalogue.read_listing(
            "radient.listing", self._fetch, ttl_s=60, cache_dir=self.tmp_path
        )

    def _timed_read(self):
        began = time.monotonic()
        self._read("loser")
        self.elapsed.append(time.monotonic() - began)

    def run(self):
        self.holder.start()
        assert self._inside.wait(5)
        self.loser.start()
        assert self._lost.wait(5), "the loser never lost the lease"
        self._finish.set()
        self.holder.join(5)
        self.loser.join(5)
        assert not self.loser.is_alive()
        return self


def _fail():
    raise RuntimeError("401")


def test_a_sibling_threads_failed_fetch_ends_the_wait_without_a_second_request(
    tmp_path, monkeypatch
) -> None:
    """l12b: the boot model warm and the first open race on one document.

    The loser used to poll for a document the failing holder never writes, for the
    full ``_LISTING_LEASE_WAIT_S`` (measured 2177-2352 ms on a real ``lop serve``
    open right after ``/health``), then re-ask the endpoint that had just failed.
    """
    race = _Race(tmp_path, monkeypatch, outcome=_fail).run()

    assert race.calls == ["warm"], "the loser must not repeat the request that just failed"
    assert race.results["loser"].failed is True and race.results["loser"].payload is None
    assert race.elapsed[0] < catalogue._LISTING_LEASE_WAIT_S / 2, race.elapsed


def test_a_sibling_that_released_before_the_lookup_does_not_strand_the_loser(
    tmp_path, monkeypatch
) -> None:
    """Review M1: the holder fails and releases between the loser's lost acquire
    and its sibling lookup. No holder is found, but the failure is on record, so
    the loser must not fall back to the 2 s peer poll (forced: 2193-2230 ms)."""
    race = _Race(tmp_path, monkeypatch, outcome=_fail, release_before_lookup=True).run()

    assert race.calls == ["warm"]
    assert race.results["loser"].failed is True
    assert race.elapsed[0] < catalogue._LISTING_LEASE_WAIT_S / 2, race.elapsed


def test_a_stale_document_beside_a_siblings_failure_is_reported_failed(
    tmp_path, monkeypatch
) -> None:
    """Review M3: with an EXPIRED document on disk, a sibling's failed refetch
    leaves that document in place. The loser serves it, but as ``failed=True``
    like the holder's own answer — ``configure``'s degraded memo (#2992) keys its
    60 s-vs-day lifetime on that flag."""
    document = catalogue._cache_path("radient.listing", tmp_path)
    catalogue._write_cache(document, {"models": [{"id": "auto"}]})
    stale = json.loads(document.read_text(encoding="utf-8"))
    stale["fetched_at"] = time.time() - 3600
    document.write_text(json.dumps(stale), encoding="utf-8")

    race = _Race(tmp_path, monkeypatch, outcome=_fail).run()

    assert race.calls == ["warm"]
    assert race.results["warm"].failed is True
    assert race.results["loser"].payload == {"models": [{"id": "auto"}]}
    assert race.results["loser"].failed is True, "a just-failed refetch is not a clean answer"


def test_a_sibling_threads_successful_fetch_is_served_to_the_loser(tmp_path, monkeypatch) -> None:
    """Control: a holder that succeeds hands its document to the waiting thread."""
    race = _Race(tmp_path, monkeypatch, outcome=lambda: {"models": [{"id": "auto"}]}).run()

    assert race.calls == ["warm"]
    assert race.results["loser"].payload == {"models": [{"id": "auto"}]}
    assert race.results["loser"].failed is False
