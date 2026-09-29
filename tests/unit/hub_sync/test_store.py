from __future__ import annotations

import json
import multiprocessing as mp
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from local_operator.hub_sync import store as st


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
    assert store.mutate(lambda d: _apply(d))
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
    assert store.mutate(lambda d: _apply(d))


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
    doc: dict = {}
    item = _apply(doc)
    for _ in range(6):
        st.record_failure(item, "provider-error/quota", "slow down", rng=mid)
    assert item["attempts"] == 6 and item["auto_retry"] is False and item["next_retry_at"] is None
    assert item["error_class"] == "provider-error" and item["error_subclass"] == "quota"


def test_failure_classes_follow_the_table() -> None:
    doc: dict = {}
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
    doc: dict = {}
    item = _apply(doc)
    st.record_failure(item, "merge-refused", "x")
    assert item["auto_retry"] is False
    item = _apply(doc, remote="fp2")
    assert item["auto_retry"] is True and item["attempts"] == 0 and item["next_retry_at"] is None


def test_no_credential_is_informational_not_a_failure() -> None:
    doc: dict = {}
    item = _apply(doc)
    item = _apply(doc, verdict="unavailable", reason="no-credential")
    assert item["state"] == "available" and item["error_class"] is None and item["attempts"] == 0


def test_a_404_stays_visible_as_failed_and_is_throttled_for_a_day() -> None:
    doc: dict = {}
    item = _apply(doc, verdict="unavailable", reason="hub-item-missing", detail="gone")
    assert item["state"] == "failed" and not st.check_due(item)
    assert st.check_due(item, datetime.now(timezone.utc) + timedelta(hours=25))


def test_manual_retry_ignores_the_schedule_and_clears_the_counters() -> None:
    item = {"auto_retry": False, "next_retry_at": "2999-01-01T00:00:00Z", "attempts": 6}
    assert not st.auto_apply_due(item) and st.auto_apply_due(item, manual=True)
    st.clear_retry(item)
    assert item["attempts"] == 0 and st.auto_apply_due(item)


def test_items_whose_row_is_gone_are_pruned() -> None:
    doc: dict = {}
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


def test_nothing_is_written_under_the_real_home(tmp_path: Path, monkeypatch) -> None:
    """The store writes ONLY under the config dir it was given (AGENTS.md: isolating a run)."""

    real_home = tmp_path / "realhome"
    monkeypatch.setenv("HOME", str(real_home))
    st.StatusStore(tmp_path / "cfg").mutate(lambda d: _apply(d))
    assert not real_home.exists()
