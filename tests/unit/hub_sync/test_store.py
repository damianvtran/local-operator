from __future__ import annotations

import json
import multiprocessing as mp
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from local_operator.hub_sync import store as st

FIXED = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _ignore(_result: object) -> None:
    """``mutate`` wants a ``-> None`` callback; ``_apply`` returns the item for readers."""


def _apply(
    doc,
    verdict="available",
    remote="fp1",
    reason="",
    classification: str | None = "remote-only",
    **kw,
):
    return st.apply_check(
        doc,
        kind="agent",
        local_id="a1",
        name="coder",
        hub_id="h",
        tenant_id=None,
        verdict=verdict,
        classification=classification,
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
    assert item["state"] == "failed" and item["auto_retry"] is True  # scheduled, not retired
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


def test_a_rejected_sign_in_is_uncounted_and_keeps_the_hubs_reason() -> None:
    """A 401 is recorded as ``no-credential``/``rejected``: no attempt, no schedule,
    and the sentence survives so the tooltip can say what was refused."""

    doc: dict[str, Any] = {}
    item = _apply(doc)
    for _ in range(st.MAX_ATTEMPTS + 2):
        item = _apply(
            doc,
            verdict="unavailable",
            reason="no-credential",
            reason_subclass="rejected",
            detail="the hub did not accept the sign-in (HTTP 401: invalid API key format)",
        )
    assert item["error_class"] == "no-credential" and item["error_subclass"] == "rejected"
    assert item["last_error"] == (
        "the hub did not accept the sign-in (HTTP 401: invalid API key format)"
    )
    assert item["attempts"] == 0 and item["auto_retry"] is True
    item = _apply(doc)
    assert item["error_class"] is None and item["last_error"] is None
    assert item["error_subclass"] is None
    # The up-to-date arm retires the sub-class with its class too.
    _apply(doc, verdict="unavailable", reason="no-credential", reason_subclass="rejected")
    item = _apply(doc, verdict="up-to-date", classification=None)
    assert item["error_class"] is None and item["error_subclass"] is None


def test_a_successful_fetch_heals_a_retired_fetch_failure_on_an_available_row() -> None:
    """The radientdev latch: a both-changed row stays ``available``, so before this
    the fetch-time ``hub-error`` (and its retired ``auto_retry``) outlived every
    later 200 and the projections derived ``failed`` forever."""

    doc: dict[str, Any] = {}
    item = _apply(doc, classification="both-changed")
    for _ in range(st.MAX_ATTEMPTS + 1):
        item = _apply(
            doc,
            verdict="unavailable",
            reason="hub-error",
            detail="the hub returned an error (HTTP 502: bad gateway)",
        )
    assert item["auto_retry"] is False and item["error_origin"] == "check"
    item = _apply(doc, classification="both-changed")
    assert item["state"] == "available" and item["classification"] == "both-changed"
    assert item["error_class"] is None and item["last_error"] is None
    assert item["attempts"] == 0 and item["auto_retry"] is True
    assert item["next_retry_at"] is None and "error_origin" not in item


def test_a_legacy_fetch_failure_row_without_an_origin_is_healed_too() -> None:
    """Rows written before ``error_origin`` (the operator's radientdev: attempts 11,
    auto_retry False) are recognised by the fetch sentence, else they stay latched."""

    doc: dict[str, Any] = {"items": {}}
    item = _apply(doc, classification="both-changed")
    item.update(
        error_class="hub-error",
        last_error="the hub refused the request (HTTP 401: invalid API key format)",
        attempts=11,
        auto_retry=False,
    )
    item.pop("error_origin", None)
    item = _apply(doc, classification="both-changed")
    assert item["error_class"] is None and item["auto_retry"] is True and item["attempts"] == 0


def _legacy_retired_below_the_cap(doc: dict[str, Any], **error: Any) -> dict[str, Any]:
    """A pre-``retired_origin`` row the APPLY retired, whose class a later fetch
    overwrote: no ``error_origin``, no stamp, ``auto_retry`` False, attempts < cap."""

    item = _apply(doc, classification="both-changed")
    item.update(auto_retry=False, **error)
    item.pop("error_origin", None)
    item.pop(st.RETIRED_ORIGIN, None)
    return item


def test_a_legacy_hub_error_row_retired_below_the_cap_stays_manual_after_a_200() -> None:
    """Agent review round 2, R2-M1 (S13; the operator's qa-tester row). Only the
    ``MAX_ATTEMPTS`` cap and the apply-only ``NO_AUTO_RETRY`` ever set
    ``auto_retry`` False, so retired at attempts 2 means the apply retired it. The
    200 clears the shown fetch error but must not re-arm unattended auto-apply."""

    doc: dict[str, Any] = {"items": {}}
    _legacy_retired_below_the_cap(
        doc,
        error_class="hub-error",
        last_error="could not reach the hub: Could not pull the team",
        attempts=2,
    )
    item = _apply(doc, classification="both-changed")
    assert item["error_class"] is None and item["last_error"] is None
    assert "error_origin" not in item
    assert item["auto_retry"] is False and item[st.RETIRED_ORIGIN] == "apply"
    assert item["attempts"] == 2 and st.auto_apply_due(item, FIXED) is False
    # A new remote text still re-arms it (the documented way out besides Retry).
    item = _apply(doc, classification="both-changed", remote="fp2")
    assert item["auto_retry"] is True and st.RETIRED_ORIGIN not in item


def test_a_legacy_no_credential_row_retired_below_the_cap_stays_manual_after_a_200() -> None:
    """R2-M1 (S6): ``no-credential`` is uncounted and never retires, so a legacy
    retired row carrying it was retired by an apply -- with or without a startup
    401 in between, the 200 clears the sign-in error and keeps it manual."""

    for with_401 in (False, True):
        doc: dict[str, Any] = {"items": {}}
        _legacy_retired_below_the_cap(
            doc,
            error_class="no-credential",
            last_error="sign in to the Radient Agent Hub",
            attempts=1,
        )
        if with_401:
            _startup_401(doc)
        item = _apply(doc, classification="both-changed")
        assert item["error_class"] is None, with_401
        assert item["auto_retry"] is False and item[st.RETIRED_ORIGIN] == "apply", with_401
        assert st.auto_apply_due(item, FIXED) is False, with_401


def test_a_legacy_tenant_mismatch_row_heals_on_a_200() -> None:
    """R2-m1: the check's tenant-mismatch arm writes ``hub-error`` directly (not
    through ``_failure_reason``); its origin-less legacy row heals once the hub
    serves the team again, instead of staying latched behind a log line."""

    doc: dict[str, Any] = {"items": {}}
    item = _apply(doc, classification="both-changed")
    item.update(
        error_class="hub-error",
        last_error="the hub now reports this team under organization org-2",
    )
    item.pop("error_origin", None)
    assert st.fetch_recorded_failure(item) is True
    item = _apply(doc, classification="both-changed")
    assert item["error_class"] is None and item["last_error"] is None


def test_an_apply_failure_survives_a_successful_fetch() -> None:
    """The fetch says nothing about whether the MERGE works: an apply-recorded
    failure is not the check's to clear (the next apply or a Retry decides)."""

    doc: dict[str, Any] = {}
    item = _apply(doc, classification="both-changed")
    st.record_failure(item, "hub-error", "could not write the update: disk full")
    item = _apply(doc, classification="both-changed")
    assert item["error_class"] == "hub-error" and item["error_origin"] == "apply"
    legacy = _apply(doc, classification="both-changed")
    legacy.pop("error_origin")
    assert st.fetch_recorded_failure(legacy) is False  # not a fetch sentence


def _startup_401(doc: dict[str, Any]) -> dict[str, Any]:
    return _apply(
        doc,
        verdict="unavailable",
        reason="no-credential",
        reason_subclass="rejected",
        detail="the hub refused the request (HTTP 401: invalid API key format)",
    )


def test_an_apply_retirement_survives_a_fetch_failure_then_a_200() -> None:
    """Agent review round 1, M1: ``error_origin`` has one slot, so a startup 401
    overwrote the apply's origin and the next 200 re-armed a merge-refused row
    (``NO_AUTO_RETRY``: needs a human) into unattended auto-apply. The 200 may
    clear the shown error; it must not undo the apply's retirement."""

    doc: dict[str, Any] = {}
    item = _apply(doc, classification="both-changed")
    # Behaviour-only assertions (no new field names), so the same test is a
    # discriminating negative control against the pre-fix store.
    st.record_failure(item, "merge-refused", "the result was refused: conflict")
    assert item["auto_retry"] is False
    item = _startup_401(doc)
    assert item["error_origin"] == "check"
    item = _apply(doc, classification="both-changed")
    assert item["error_class"] is None and item["last_error"] is None
    assert item["auto_retry"] is False and item["attempts"] == 1
    assert st.auto_apply_due(item, FIXED) is False


def test_a_provider_error_retired_at_max_attempts_survives_a_fetch_failure_then_a_200() -> None:
    """M1, the counted variant: six apply ``provider-error``s retire the row; a
    401 then a 200 must not hand it six more unattended (paid) merges."""

    doc: dict[str, Any] = {}
    item = _apply(doc, classification="both-changed")
    for _ in range(st.MAX_ATTEMPTS):
        st.record_failure(item, "provider-error", "the model provider failed", now=FIXED)
    assert item["auto_retry"] is False and item["attempts"] == st.MAX_ATTEMPTS
    item = _startup_401(doc)
    item = _apply(doc, classification="both-changed")
    assert item["error_class"] is None
    assert item["auto_retry"] is False and item["attempts"] == st.MAX_ATTEMPTS
    assert st.auto_apply_due(item, FIXED) is False


def test_a_check_hub_error_cannot_launder_an_apply_retirement() -> None:
    """A counted fetch failure on an apply-retired row crosses MAX_ATTEMPTS again
    with ``origin="check"``; the apply's stamp must win, so the 200 still keeps it."""

    doc: dict[str, Any] = {}
    item = _apply(doc, classification="both-changed")
    for _ in range(st.MAX_ATTEMPTS):
        st.record_failure(item, "provider-error", "the model provider failed", now=FIXED)
    item = _apply(
        doc,
        verdict="unavailable",
        reason="hub-error",
        detail="the hub returned an error (HTTP 502)",
    )
    assert item[st.RETIRED_ORIGIN] == "apply"
    item = _apply(doc, classification="both-changed")
    assert item["auto_retry"] is False


def test_a_legacy_apply_retired_row_is_pinned_before_a_401_overwrites_it() -> None:
    """A row retired by a merge-refused BEFORE ``retired_origin`` existed has no
    stamp; the 401 that overwrites its origin must attribute the retirement to
    the apply first (the error it replaces is not a fetch's)."""

    doc: dict[str, Any] = {"items": {}}
    item = _apply(doc, classification="both-changed")
    item.update(
        error_class="merge-refused", last_error="the result was refused: x", auto_retry=False
    )
    item.pop("error_origin", None)
    item = _startup_401(doc)
    assert item[st.RETIRED_ORIGIN] == "apply"
    item = _apply(doc, classification="both-changed")
    assert item["auto_retry"] is False


def test_a_fetch_retirement_then_a_401_then_a_200_still_re_arms() -> None:
    """The operator's case must keep healing: a row retired by FETCH failures,
    then a startup 401, then a 200, comes back to automatic updates."""

    doc: dict[str, Any] = {}
    item = _apply(doc, classification="both-changed")
    for _ in range(st.MAX_ATTEMPTS):
        item = _apply(
            doc, verdict="unavailable", reason="hub-error", detail="could not reach the hub: x"
        )
    assert item["auto_retry"] is False and item[st.RETIRED_ORIGIN] == "check"
    item = _startup_401(doc)
    item = _apply(doc, classification="both-changed")
    assert item["auto_retry"] is True and item["attempts"] == 0
    assert st.RETIRED_ORIGIN not in item and st.auto_apply_due(item, FIXED) is True


def test_a_legacy_local_read_failure_heals_and_an_unknown_one_is_logged_once(caplog) -> None:
    """m2: origin-less rows from the check's local reads heal like their stamped
    twins; an origin-less ``hub-error`` nothing can attribute stays, and says so
    once per row instead of failing silently on every check."""

    import logging

    doc: dict[str, Any] = {"items": {}}
    item = _apply(doc, classification="both-changed")
    item.update(error_class="hub-error", last_error="could not read the local team: locked")
    assert st.fetch_recorded_failure(item) is True

    st._UNRECOGNISED_LOGGED.discard("agent:a1")
    item.update(last_error="Could not pull the team from the Radient Agent Hub")
    with caplog.at_level(logging.INFO, logger=st.__name__):
        assert st.fetch_recorded_failure(item) is False
        assert st.fetch_recorded_failure(item) is False
    lines = [r for r in caplog.records if "no recorded origin" in r.getMessage()]
    assert len(lines) == 1 and "agent:a1" in lines[0].getMessage()


def test_every_error_clearing_writer_drops_the_stale_origins() -> None:
    """n1: a row with no error carries no origin; a re-armed row carries no
    retirement stamp."""

    item: dict[str, Any] = {"state": "available"}
    st.record_failure(item, "merge-refused", "the result was refused: x")
    st.clear_failure(item)
    assert "error_origin" not in item and item[st.RETIRED_ORIGIN] == "apply"
    st.clear_retry(item)
    assert st.RETIRED_ORIGIN not in item and item["auto_retry"] is True
    st.record_failure(item, "merge-refused", "the result was refused: x")
    st.settle_applied(item, {}, None)
    assert "error_origin" not in item and st.RETIRED_ORIGIN not in item
    st.record_failure(item, "merge-refused", "the result was refused: x")
    st.settle_unchanged(item)
    assert "error_origin" not in item and st.RETIRED_ORIGIN not in item


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
    """Absence from ``live`` alone drops NOTHING; a confirmed-gone answer does.

    Q4: a row is durable bookkeeping, so "this pass did not list it" is not evidence
    that its item is gone -- the pass may have failed to read it.
    """

    doc: dict[str, Any] = {}
    _apply(doc)
    st.prune_items(doc, set())
    assert "agent:a1" in doc["items"], "no predicate means nothing is confirmed gone"

    st.prune_items(doc, set(), lambda _kind, _local_id: True)
    assert doc["items"] == {}


def test_a_row_missing_from_a_failed_pass_is_kept() -> None:
    """The predicate is the whole decision: False keeps the row and its bookkeeping."""

    doc: dict[str, Any] = {}
    item = _apply(doc)
    item["applied_backup"] = "hub/backups/agent-a1.json"

    st.prune_items(doc, set(), lambda _kind, _local_id: False)

    kept = doc["items"]["agent:a1"]
    assert kept["applied_backup"] == "hub/backups/agent-a1.json"


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


def test_prompt_too_long_is_scheduled_not_retired_and_gated_by_fingerprint() -> None:
    """A too-long request retries on a schedule, but only when its input changed."""
    from local_operator.hub_sync import store as st

    item: dict[str, Any] = {"attempts": 0, "auto_retry": True, "next_retry_at": None}
    st.record_failure(item, "prompt-too-long", "section too long", now=FIXED)
    assert item["auto_retry"] is True  # NOT a human-only dead end
    assert item["next_retry_at"] is not None  # a cheap scheduled retry

    # Six failures do NOT retire it (no-forever-retirement).
    for _ in range(6):
        st.record_failure(item, "prompt-too-long", "still too long", now=FIXED)
    assert item["auto_retry"] is True

    item["prompt_fingerprint"] = "fp1"
    assert st.prompt_too_long_eligible(item, "fp1") is False  # unchanged -> do not re-send
    assert st.prompt_too_long_eligible(item, "fp2") is True  # changed -> re-arm


def test_merge_refused_still_retires() -> None:
    from local_operator.hub_sync import store as st

    item: dict[str, Any] = {"attempts": 0, "auto_retry": True, "next_retry_at": None}
    st.record_failure(item, "merge-refused", "conflict", now=FIXED)
    assert item["auto_retry"] is False


def test_pull_and_push_due_times_are_independent() -> None:
    from datetime import datetime, timezone

    from local_operator.hub_sync import store as st

    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    # A fresh pull does not make a stale push look serviced, or the reverse.
    item = {"last_pulled_at": "2026-01-01T11:59:00Z", "last_pushed_at": "2026-01-01T08:00:00Z"}
    assert st.pull_due(item, 30, now) is False
    assert st.push_due(item, 30, now) is True
    item = {"last_pulled_at": "2026-01-01T08:00:00Z", "last_pushed_at": "2026-01-01T11:59:00Z"}
    assert st.pull_due(item, 30, now) is True
    assert st.push_due(item, 30, now) is False
    # Never serviced -> both due.
    assert st.pull_due({}, 30, now) and st.push_due({}, 30, now)


# -- the publish-side fact survives the pull-side verdict (Q14) ------------------------


def test_a_local_only_check_keeps_its_classification() -> None:
    """ONE VERDICT, TWO FACTS: up to date to PULL, unpublished on the PUSH side.

    ``local-only`` folds onto the ``up-to-date`` verdict because there is nothing
    to fetch; blanking the classification with it made a locally edited copy read
    clean-too after one check, so no client could show "Modified".
    """

    item = _apply({}, verdict="up-to-date", classification="local-only")

    assert item["state"] == "up-to-date"
    assert item["classification"] == "local-only"
    # It stays an up-to-date row, which is what keeps it out of every apply arm.
    assert st.effective_state(item) == "up-to-date"


def test_an_equal_pair_leaves_no_classification_behind() -> None:
    """Equality is the absence of a fact: nothing to report, nothing to keep."""

    item = _apply({}, verdict="up-to-date", classification="up-to-date")

    assert item["state"] == "up-to-date"
    assert item["classification"] is None


def test_a_check_that_stops_being_local_only_clears_the_stale_mark() -> None:
    """The classification is the CHECK's answer, never a sticky past one."""

    doc: dict[str, Any] = {}
    _apply(doc, verdict="up-to-date", classification="local-only")
    item = _apply(doc, verdict="up-to-date", classification="up-to-date")

    assert item["classification"] is None


def test_an_accepted_push_retires_the_local_only_mark() -> None:
    """Once the hub HAS seen the bytes the sentence is false, not merely stale."""

    doc: dict[str, Any] = {}
    item = _apply(doc, verdict="up-to-date", classification="local-only")

    st.clear_local_only(item)
    assert item["classification"] is None

    # It retires ONLY that classification: any other stays for the next check.
    other_doc: dict[str, Any] = {}
    other = _apply(other_doc)
    st.clear_local_only(other)
    assert other["classification"] == "remote-only"


def test_a_no_write_inbound_pass_keeps_the_local_only_mark() -> None:
    """ "Keep my side" publishes nothing, so the copy is still unpublished."""

    doc: dict[str, Any] = {}
    item = _apply(doc, verdict="up-to-date", classification="local-only")

    st.settle_unchanged(item)

    assert item["state"] == "up-to-date"
    assert item["classification"] == "local-only"

    # A classification the pass DOES settle is still cleared.
    other = _apply({})
    st.settle_unchanged(other)
    assert other["classification"] is None
