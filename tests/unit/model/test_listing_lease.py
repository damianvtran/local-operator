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


def test_a_sibling_threads_failed_fetch_ends_the_wait_without_a_second_request(tmp_path) -> None:
    """l12b: the boot model warm and the first open race on one document.

    The loser used to poll for a document the failing holder never writes, for the
    full ``_LISTING_LEASE_WAIT_S`` (measured 2177-2352 ms on a real ``lop serve``
    open right after ``/health``), then re-ask the endpoint that had just failed.
    """
    import threading

    holder_inside = threading.Event()
    let_it_fail = threading.Event()
    calls: list[str] = []

    def failing_fetch():
        calls.append(threading.current_thread().name)
        holder_inside.set()
        let_it_fail.wait(5)
        raise RuntimeError("401")

    results: dict[str, catalogue.Listing] = {}
    elapsed: list[float] = []
    holder = threading.Thread(
        target=lambda: results.setdefault(
            "holder", catalogue.read_listing("radient.listing", failing_fetch, cache_dir=tmp_path)
        ),
        name="warm",
    )
    holder.start()
    assert holder_inside.wait(5)

    def loser_read():
        began = time.monotonic()
        results["loser"] = catalogue.read_listing(
            "radient.listing", failing_fetch, cache_dir=tmp_path
        )
        elapsed.append(time.monotonic() - began)

    loser = threading.Thread(target=loser_read, name="open")
    loser.start()
    time.sleep(0.05)  # let the loser reach the lease wait
    let_it_fail.set()
    holder.join(5)
    loser.join(5)

    assert calls == ["warm"], "the loser must not repeat the request that just failed"
    assert results["loser"].failed is True and results["loser"].payload is None
    assert elapsed[0] < catalogue._LISTING_LEASE_WAIT_S / 2, elapsed


def test_a_sibling_threads_successful_fetch_is_served_to_the_loser(tmp_path) -> None:
    """Control: a holder that succeeds hands its document to the waiting thread."""
    import threading

    holder_inside = threading.Event()
    let_it_finish = threading.Event()
    calls: list[str] = []

    def fetch():
        calls.append(threading.current_thread().name)
        holder_inside.set()
        let_it_finish.wait(5)
        return {"models": [{"id": "auto"}]}

    holder = threading.Thread(
        target=lambda: catalogue.read_listing("radient.listing", fetch, cache_dir=tmp_path),
        name="warm",
    )
    holder.start()
    assert holder_inside.wait(5)
    out: dict[str, catalogue.Listing] = {}
    loser = threading.Thread(
        target=lambda: out.setdefault(
            "listing", catalogue.read_listing("radient.listing", fetch, cache_dir=tmp_path)
        )
    )
    loser.start()
    time.sleep(0.05)
    let_it_finish.set()
    holder.join(5)
    loser.join(5)

    assert calls == ["warm"]
    assert out["listing"].payload == {"models": [{"id": "auto"}]}
