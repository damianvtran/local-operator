from __future__ import annotations

import json
import multiprocessing as mp
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from local_operator.hub_sync import store as st


def _ignore(_result: object) -> None:
    """``mutate`` wants a ``-> None`` callback; ``_apply`` returns the item for readers."""


def _apply(doc, verdict="available", remote="fp1", reason="", **kw):
    return st.apply_check(
        doc,
        kind="agent",
        local_id="a1",
        name="coder",
        hub_id="h",
        tenant_id=None,
        verdict=verdict,
        classification="remote-only",
        baseline="known",
        local_fp="l",
        remote_fp=remote,
        reason=reason,
        **kw,
    )


def test_round_trip_and_atomic_write_leaves_no_temp(tmp_path: Path) -> None:
    store = st.StatusStore(tmp_path)
    assert store.mutate(lambda d: _ignore(_apply(d)))
    assert store.load()["items"]["agent:a1"]["state"] == "available"
    assert (
        sorted(
            p.name
            for p in store.path.parent.iterdir()
            if p.name.startswith(".status.") and p.suffix == ".tmp"
        )
        == []
    )


def test_a_corrupt_file_is_quarantined_and_rebuilt_not_fatal(tmp_path: Path) -> None:
    store = st.StatusStore(tmp_path)
    store.path.parent.mkdir(parents=True)
    store.path.write_text("{oops")
    assert store.load()["items"] == {}
    assert list(store.path.parent.glob("status.json.corrupt-*"))
    assert store.mutate(lambda d: _ignore(_apply(d)))


def test_a_crashed_updating_writer_reads_as_failed_after_five_minutes() -> None:
    item = {"state": "updating", "updating_since": "2020-01-01T00:00:00Z"}
    assert st.effective_state(item) == "failed"
    fresh = {"state": "updating", "updating_since": st.now_iso()}
    assert st.effective_state(fresh) == "updating"


def test_backoff_numbers_and_the_six_attempt_stop() -> None:
    mid = lambda lo, hi: (lo + hi) / 2  # noqa: E731
    assert [round(st.backoff_delay_s(n, mid)) for n in (1, 2, 3, 6, 9)] == [
        900,
        1800,
        3600,
        21600,
        21600,
    ]
    doc: dict[str, Any] = {}
    item = _apply(doc)
    for _ in range(6):
        st.record_failure(item, "provider-error/quota", "slow down", rng=mid)
    assert item["attempts"] == 6 and item["auto_retry"] is False and item["next_retry_at"] is None
    assert item["error_class"] == "provider-error" and item["error_subclass"] == "quota"


def test_failure_classes_follow_the_table() -> None:
    doc: dict[str, Any] = {}
    item = _apply(doc)
    st.record_failure(item, "merge-refused", "both changed")
    assert item["state"] == "available" and item["auto_retry"] is False  # a human decides
    st.record_failure(item, "concurrent-edit", "raced")
    assert item["attempts"] == 1  # uncounted
    st.record_failure(item, "prompt-too-long", "too big")
    assert item["state"] == "failed" and item["auto_retry"] is False
    st.record_failure(item, "model-unavailable", "no key")
    assert item["state"] == "available"
    delay = datetime.strptime(item["next_retry_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    assert timedelta(minutes=50) < delay - datetime.now(timezone.utc) < timedelta(minutes=70)


def test_a_new_remote_fingerprint_rearms_everything_a_human_had_stopped() -> None:
    doc: dict[str, Any] = {}
    item = _apply(doc)
    st.record_failure(item, "merge-refused", "x")
    assert item["auto_retry"] is False
    item = _apply(doc, remote="fp2")
    assert item["auto_retry"] is True and item["attempts"] == 0 and item["next_retry_at"] is None


def test_no_credential_is_recorded_without_counting_as_a_failed_attempt() -> None:
    """U11: the fact is kept on the item (the UI needs it) but it is not a failure.

    It is not counted, no retry is scheduled and the state is left alone - so a
    ``no-credential`` read never invents a failure for an item that was fine.
    """

    doc: dict[str, Any] = {}
    item = _apply(doc)
    item = _apply(doc, verdict="unavailable", reason="no-credential")
    assert item["error_class"] == "no-credential" and item["last_error"] is None
    assert item["state"] == "available" and item["attempts"] == 0
    assert item["next_retry_at"] is None and item["auto_retry"] is True


def test_a_signed_out_item_keeps_the_fact_until_a_fetch_succeeds() -> None:
    """The class is a live reading, not a latch: a successful fetch retires it."""

    doc: dict[str, Any] = {}
    item = _apply(doc, verdict="unavailable", reason="no-credential")
    assert item["error_class"] == "no-credential"
    item = _apply(doc)
    assert item["error_class"] is None and item["state"] == "available"


def test_a_signed_out_read_retires_the_schedule_the_class_it_replaced_armed() -> None:
    """M2: replacing a failure class must not leave that failure's timer behind."""

    doc: dict[str, Any] = {}
    item = _apply(doc)
    st.record_failure(item, "hub-error", "could not reach the hub: 500")
    assert item["next_retry_at"] is not None
    item = _apply(doc, verdict="unavailable", reason="no-credential")
    assert item["error_class"] == "no-credential" and item["next_retry_at"] is None


def test_a_manual_retry_that_computed_retires_the_stale_failure() -> None:
    """U10: the retry proved the update computes, so the row must offer it, not another retry."""

    item = {
        "state": "available",
        "error_class": "hub-error",
        "error_subclass": None,
        "last_error": "could not reach the hub: 500",
        "next_retry_at": "2999-01-01T00:00:00Z",
    }
    st.clear_failure(item)
    assert item["error_class"] is None and item["last_error"] is None
    assert item["next_retry_at"] is None and item["state"] == "available"


def test_a_404_stays_visible_as_failed_and_is_throttled_for_a_day() -> None:
    doc: dict[str, Any] = {}
    item = _apply(doc, verdict="unavailable", reason="hub-item-missing", detail="gone")
    assert item["state"] == "failed" and not st.check_due(item)
    assert st.check_due(item, datetime.now(timezone.utc) + timedelta(hours=25))


def test_manual_retry_ignores_the_schedule_and_clears_the_counters() -> None:
    item = {"auto_retry": False, "next_retry_at": "2999-01-01T00:00:00Z", "attempts": 6}
    assert not st.auto_apply_due(item) and st.auto_apply_due(item, manual=True)
    st.clear_retry(item)
    assert item["attempts"] == 0 and st.auto_apply_due(item)


def test_items_whose_row_is_gone_are_pruned() -> None:
    doc: dict[str, Any] = {}
    _apply(doc)
    st.prune_items(doc, set())
    assert doc["items"] == {}


def _hold(root: str, q) -> None:  # runs in a second process
    lease = st.RunnerLease(Path(root))
    q.put(lease.acquire())
    time.sleep(0.6)
    lease.release()


def test_the_lease_is_exclusive_across_processes_and_expires_for_a_dead_holder(
    tmp_path: Path,
) -> None:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_hold, args=(str(tmp_path), q))
    proc.start()
    assert q.get(timeout=30) is True
    assert st.RunnerLease(tmp_path).acquire() is False  # the child holds it
    proc.join(30)
    assert st.RunnerLease(tmp_path).acquire() is True  # released -> free
    # A crashed holder cannot block forever: an expired lease is stolen.
    path = tmp_path / "hub" / ".runner.lease"
    path.write_text(json.dumps({"holder": "dead", "expires_at": time.time() - 1}))
    assert st.RunnerLease(tmp_path).acquire() is True


def test_a_heartbeat_racing_release_never_recreates_the_lease_file(tmp_path: Path) -> None:
    """R2-3(a): ``renew`` after ``release`` must not write the lease back.

    Reproduced by making the heartbeat's file read yield to ``release`` between its
    ``_held`` check and its ``os.replace`` -- the exact interleaving a bare check
    loses. With the guard, ``release`` waits for the in-flight renew and the file
    is gone afterwards.
    """

    import threading

    lease = st.RunnerLease(tmp_path, ttl_s=600.0)
    assert lease.acquire()
    path = tmp_path / "hub" / ".runner.lease"
    inside = threading.Event()
    real_read = Path.read_text

    def slow_read(self: Path, *a: Any, **k: Any) -> str:
        text = real_read(self, *a, **k)
        if self == path and threading.current_thread().name == "renew-under-test":
            inside.set()
            time.sleep(0.3)  # release() is called while we hold the stale read
        return text

    Path.read_text = slow_read  # type: ignore[method-assign]
    try:
        t = threading.Thread(target=lease.renew, name="renew-under-test")
        t.start()
        assert inside.wait(5)
        lease.release()
        t.join(5)
    finally:
        Path.read_text = real_read  # type: ignore[method-assign]
    assert not path.exists()
    assert list(path.parent.glob(".runner.lease*")) == []
    assert lease.renew() is False and not path.exists()


def test_nothing_is_written_under_the_real_home(tmp_path: Path, monkeypatch) -> None:
    """The store writes ONLY under the config dir it was given (AGENTS.md: isolating a run)."""

    real_home = tmp_path / "realhome"
    monkeypatch.setenv("HOME", str(real_home))
    st.StatusStore(tmp_path / "cfg").mutate(lambda d: _ignore(_apply(d)))
    assert not real_home.exists()


def test_scrub_redacts_real_token_shapes_not_just_the_length() -> None:
    """R5: the persisted ``last_error`` is built from provider/hub exception text."""

    # Assembled at runtime so no secret-shaped literal sits in the source.
    jwt = ".".join(["eyJ" + "a" * 20, "eyJ" + "b" * 20, "c" * 20])
    api_key = "sk-" + "proj-" + "Z9" * 20
    text = st.scrub(f"401 Authorization: Bearer {jwt} for key {api_key}\nretry")
    assert jwt not in text and api_key not in text
    assert "\n" not in text and "retry" in text
