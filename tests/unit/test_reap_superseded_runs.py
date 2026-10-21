"""The reaper cancels QUEUED runs and nothing else.

Every test here is a safety rule rather than a happy path, because the failure
mode of this script is cancelling work somebody needs. The rules under test are
the ones in `scripts/reap_superseded_runs.py`'s docstring:

* only `queued` runs are touched -- a started run is doing work;
* protected branches (`main`, `release-next-*`) are never touched -- a gate
  that can be cancelled is not a gate;
* the pull request's head is authoritative for a PR run: a current head keeps
  the run at ANY age (the fork-PR case), and a moved head makes it
  superseded, with the branch tip only corroborating;
* an UNKNOWN read failure (5xx, network) keeps the run, while a CONFIRMED
  absence (404) is what may put it past the abandonment bound;
* an unresolvable branch or PR is a reason to KEEP, never to cancel;
* the grace period protects a run whose head is mid-move.
"""

from __future__ import annotations

import datetime
import importlib.util
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "reap_superseded_runs.py"


def _load():
    spec = importlib.util.spec_from_file_location("reap_superseded_runs", SCRIPT)
    # C-2: spec_from_file_location returns ModuleSpec | None, and its loader is
    # Optional too. Asserting BOTH narrows them for the type checker rather
    # than leaving pyright to flag an unchecked optional at three call sites.
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reaper = _load()


def _run(**over):
    base = {
        "id": 1,
        "status": "queued",
        "head_branch": "feat/x",
        "head_sha": "a" * 40,
        "event": "push",
        "created_at": (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=30)
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pull_requests": [],
    }
    base.update(over)
    return base


@pytest.fixture
def api(monkeypatch):
    """Replace the three API reads; each test sets only the ones it needs."""
    state = {"pr": {}, "tip": {}}
    monkeypatch.setattr(reaper, "pr_head_sha", lambda repo, n: state["pr"].get(n))
    monkeypatch.setattr(reaper, "branch_tip", lambda repo, b: state["tip"].get(b))
    return state


def test_a_running_job_is_never_touched(api):
    should, reason = reaper.decide("o/r", _run(status="in_progress"), 5.0)
    assert not should
    assert "not queued" in reason


@pytest.mark.parametrize("branch", ["main", "release-next", "release-next-0-83-0"])
def test_protected_branches_are_never_touched(api, branch):
    api["tip"][branch] = "b" * 40  # tip HAS moved
    should, reason = reaper.decide("o/r", _run(head_branch=branch), 5.0)
    assert not should
    assert "protected branch" in reason


def test_a_run_inside_the_grace_period_is_kept(api):
    api["tip"]["feat/x"] = "b" * 40
    fresh = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    should, reason = reaper.decide("o/r", _run(created_at=fresh), 5.0)
    assert not should
    assert "grace period" in reason


def test_a_push_run_is_cancelled_when_the_branch_tip_moved(api):
    api["tip"]["feat/x"] = "b" * 40
    should, reason = reaper.decide("o/r", _run(), 5.0)
    assert should
    assert "tip moved" in reason


def test_a_push_run_is_kept_when_the_tip_is_its_own_commit(api):
    api["tip"]["feat/x"] = "a" * 40
    should, reason = reaper.decide("o/r", _run(), 5.0)
    assert not should
    assert "current" in reason


def test_a_deleted_branch_is_kept_rather_than_guessed(api):
    # No tip registered -> branch_tip returns None.
    should, reason = reaper.decide("o/r", _run(), 5.0)
    assert not should
    assert "gone" in reason


def test_a_pr_run_is_cancelled_when_the_head_and_the_tip_both_moved(api):
    """Both readings moved: this is the ordinary superseded PR case."""
    api["pr"][7] = "b" * 40
    api["tip"]["feat/x"] = "b" * 40
    should, reason = reaper.decide(
        "o/r", _run(event="pull_request", pull_requests=[{"number": 7}]), 5.0
    )
    assert should
    assert "moved past" in reason


def test_a_pr_run_is_kept_when_the_branch_tip_still_points_at_it(api):
    """The tip disagreeing with the PR head KEEPS the run.

    The PR head is authoritative (see the fork-PR test below), but when the
    head says "moved" and the branch tip still points at the run's commit, the
    run may be the one about to be current and cancelling it would delete live
    CI on one signal. Corroboration only ever protects.
    """
    api["pr"][7] = "b" * 40  # PR says moved
    api["tip"]["feat/x"] = "a" * 40  # but the branch tip still IS this commit
    should, reason = reaper.decide(
        "o/r", _run(event="pull_request", pull_requests=[{"number": 7}]), 5.0
    )
    assert not should
    assert "current" in reason


def test_a_pr_run_with_no_pull_requests_entry_is_kept(api):
    should, reason = reaper.decide("o/r", _run(event="pull_request"), 5.0)
    assert not should
    assert "no pull_requests entry" in reason


def test_an_unreadable_pr_head_is_kept(api):
    should, reason = reaper.decide(
        "o/r", _run(event="pull_request", pull_requests=[{"number": 999}]), 5.0
    )
    assert not should
    assert "cannot read PR #999" in reason


def test_a_pr_run_whose_head_moved_is_cancelled_even_when_the_branch_is_gone(api):
    """The fork-PR case, and the correction it forced in the rule.

    A fork's branch 404s in this repository, so the branch tip can never be
    read for a fork PR. Treating that as "unresolvable" made the rule protect
    EVERY superseded fork PR forever -- and, worse, the abandonment bound could
    then cancel one whose PR head was readable and CURRENT. The PR head is the
    authoritative reading for a pull_request run: if it has moved past the
    run's commit, the run is superseded, and the tip only corroborates. If it
    has NOT moved, the run is kept at any age (the test above).
    """
    api["pr"][7] = "b" * 40  # PR head moved past the run's commit
    # no tip registered -> the branch does not resolve, as for a fork PR
    should, reason = reaper.decide(
        "o/r", _run(event="pull_request", pull_requests=[{"number": 7}]), 5.0
    )
    assert should
    assert "moved past" in reason


def test_a_pr_run_is_kept_at_any_age_when_its_head_is_still_current(api):
    """The fork-PR hole that the QA round found, closed.

    A readable PR head equal to the run's commit means the run is about to be
    current. That must beat the abandonment bound: an earlier version fell
    through to the bound whenever the branch tip was unreadable, which is
    exactly a fork PR, so a live fork PR older than the bound was cancelled.
    """
    ancient = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=37)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    api["pr"][7] = "a" * 40  # PR head IS the run's commit
    should, reason = reaper.decide(
        "o/r",
        _run(
            event="pull_request",
            created_at=ancient,
            updated_at=ancient,
            pull_requests=[{"number": 7}],
        ),
        5.0,
    )
    assert not should
    assert "is current" in reason


def test_an_event_that_cannot_be_superseded_is_kept(api):
    should, reason = reaper.decide("o/r", _run(event="schedule"), 5.0)
    assert not should
    assert "not superseded by definition" in reason


def test_a_run_queued_past_the_abandonment_bound_is_cancelled_even_if_unresolvable(api):
    """The zombie case a live dry run actually found.

    Two pull_request runs from 2026-09-13 were still `queued` 37 days later, on
    branches that no longer existed (404) and with no `pull_requests` entry in
    either API view. Neither supersession rule could resolve them, so without
    this bound they would sit in the queue forever. Nothing queues for a day
    and then runs, and the threshold is far above any observed queue time.
    """
    ancient = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=37)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    should, reason = reaper.decide("o/r", _run(event="pull_request", created_at=ancient), 5.0, 24.0)
    assert should
    assert "abandonment bound" in reason


def test_a_run_just_under_the_abandonment_bound_still_uses_the_normal_rules(api):
    nearly = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=23)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    api["tip"]["feat/x"] = "a" * 40  # tip is current
    should, reason = reaper.decide("o/r", _run(created_at=nearly), 5.0, 24.0)
    assert not should
    assert "current" in reason


def test_abandonment_never_overrides_the_protected_branch_rule(api):
    ancient = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=37)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    should, reason = reaper.decide("o/r", _run(head_branch="main", created_at=ancient), 5.0, 24.0)
    assert not should
    assert "protected branch" in reason


def test_abandonment_never_cancels_an_event_that_cannot_be_superseded(api):
    """C-5: `schedule` runs are promised untouched, at ANY age.

    The bound used to run before the event rules, so an ancient scheduled run
    — the queue in this repository has held exactly that — was cancelled
    despite the docstring saying schedules are never touched. Resolve first;
    only an unresolvable run is ever subject to the bound.
    """
    ancient = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=37)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    should, reason = reaper.decide("o/r", _run(event="schedule", created_at=ancient), 5.0, 24.0)
    assert not should
    assert "not superseded by definition" in reason


def test_abandonment_never_cancels_a_resolvable_run_that_is_still_current(api):
    """C-5: a 25-hour-old PR run whose head is current is KEPT.

    Age must not decide anything while the run's own commit is still the head
    that would run again. Only an UNRESOLVABLE run falls through to the bound.
    """
    old = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=25)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    api["pr"][7] = "a" * 40
    api["tip"]["feat/x"] = "a" * 40
    should, reason = reaper.decide(
        "o/r", _run(event="pull_request", created_at=old, pull_requests=[{"number": 7}]), 5.0, 24.0
    )
    assert not should
    assert "current" in reason


def test_an_unresolvable_run_inside_the_bound_is_kept(api):
    old = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=6)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    should, reason = reaper.decide("o/r", _run(event="pull_request", created_at=old), 5.0, 24.0)
    assert not should
    assert "inside the abandonment bound" in reason


def test_a_rerun_of_an_old_run_is_protected_by_the_grace_period(api):
    """Q3: age must be QUEUE RESIDENCE, not the run object's birth time.

    Measured on run 39498542825: the run object reports created_at 21:13:02Z
    while attempt 2 was created at 22:02:55Z. A re-run therefore inherits the
    old created_at, and an age read from it put the re-run past the 24-hour
    bound on its FIRST sweep -- bypassing the grace period and the current-tip
    keep, and again on every retry. `updated_at` tracks the re-queue.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    long_ago = (now - datetime.timedelta(hours=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    just_now = (now - datetime.timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    should, reason = reaper.decide("o/r", _run(created_at=long_ago, updated_at=just_now), 5.0, 24.0)
    assert not should
    assert "grace period" in reason


def _old(hours: int) -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cancel_stub(status: int):
    """A gh_api that answers the pre-read then fails the cancel with `status`."""

    def fake(path: str, *, method: str = "GET"):
        if method == "POST":
            raise reaper.ApiError(f"HTTP {status}", status)
        return {"status": "queued"}

    return fake


def test_a_transient_read_failure_keeps_the_run(api, monkeypatch):
    """Q2: an outage is not evidence that anything is gone.

    The bound may act on a CONFIRMED absence (404). A 5xx, a rate limit or a
    network failure says nothing about whether the branch still exists, so the
    run is kept rather than cancelled on an unknown.
    """
    ancient = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=37)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    def boom(repo, branch):
        raise reaper.ApiError("HTTP 502", 502)

    monkeypatch.setattr(reaper, "branch_tip", boom)
    should, reason = reaper.decide("o/r", _run(created_at=ancient, updated_at=ancient), 5.0, 24.0)
    assert not should
    assert "transient" in reason


def test_a_confirmed_absence_still_reaches_the_abandonment_bound(api, monkeypatch):
    """The other half: a 404 IS evidence, so the bound may act on it."""
    ancient = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=37)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    monkeypatch.setattr(reaper, "branch_tip", lambda repo, branch: None)  # 404 -> gone
    should, reason = reaper.decide("o/r", _run(created_at=ancient, updated_at=ancient), 5.0, 24.0)
    assert should
    assert "abandonment bound" in reason


def test_a_transient_failure_on_the_corroborating_tip_read_keeps_a_pr_run(api, monkeypatch):
    """A moved PR head plus a failed tip read is two signals short of a cancel."""
    api["pr"][7] = "b" * 40  # PR head says moved

    def boom(repo, branch):
        raise reaper.ApiError("HTTP 500", 500)

    monkeypatch.setattr(reaper, "branch_tip", boom)
    should, reason = reaper.decide(
        "o/r", _run(event="pull_request", pull_requests=[{"number": 7}]), 5.0
    )
    assert not should
    assert "transient" in reason


def test_a_transient_failure_on_the_pr_head_read_keeps_even_past_the_bound(api, monkeypatch):
    """The exact Q2 regression: a 502 on the PR read must not become a bound.

    Round 2 reproduced the opposite behaviour, and the surviving mutant from
    round 3 showed this path had no test of its own.
    """
    ancient = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=37)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    def boom(repo, number):
        raise reaper.ApiError("HTTP 502", 502)

    monkeypatch.setattr(reaper, "pr_head_sha", boom)
    should, reason = reaper.decide(
        "o/r",
        _run(
            event="pull_request",
            created_at=ancient,
            updated_at=ancient,
            pull_requests=[{"number": 7}],
        ),
        5.0,
        24.0,
    )
    assert not should
    assert "transient" in reason


def test_branch_tip_returns_none_only_for_a_confirmed_404(monkeypatch):
    """The classification itself, which the surviving mutant disabled.

    `branch_tip` maps 404 to None ("gone") and re-raises everything else, so
    the distinction the abandonment bound depends on cannot be silently lost.
    """
    for code, expected_none in ((404, True), (500, False), (403, False)):

        def fake(path, *, method="GET", _code=code):
            raise reaper.ApiError(f"HTTP {_code}", _code)

        monkeypatch.setattr(reaper, "gh_api", fake)
        if expected_none:
            assert reaper.branch_tip("o/r", "feat/x") is None
        else:
            with pytest.raises(reaper.ApiError):
                reaper.branch_tip("o/r", "feat/x")


def test_a_409_on_the_cancel_is_a_ghost_not_a_refusal(monkeypatch, capsys):
    """Measured on runs 34749070299 / 34747986252.

    GitHub returns 409 'Cannot cancel a workflow run that has not been queued
    yet' for a run it reports as `queued` but never scheduled. That is
    terminal, so it must not be counted as a refusal -- a refusal implies the
    next sweep might succeed, and this one never will.
    """
    run = _run(id=99, head_branch="feat/x", created_at=_old(48), updated_at=_old(48))
    monkeypatch.setattr(reaper, "queued_runs", lambda repo: [run])
    monkeypatch.setattr(reaper, "decide", lambda *a, **k: (True, "unresolvable"))
    monkeypatch.setattr(reaper, "gh_api", _cancel_stub(409))
    monkeypatch.setattr(reaper.sys, "argv", ["x", "--repo", "o/r"])

    rc = reaper.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "GHOST" in out
    assert "REFUSE" not in out
    assert "refused 0" in out
    assert "ghosts 1" in out


def test_a_403_on_the_cancel_is_still_a_refusal(monkeypatch, capsys):
    """The other side: a permission error may be fixed, so it stays a refusal."""
    run = _run(id=98, head_branch="feat/y", created_at=_old(48), updated_at=_old(48))
    monkeypatch.setattr(reaper, "queued_runs", lambda repo: [run])
    monkeypatch.setattr(reaper, "decide", lambda *a, **k: (True, "unresolvable"))
    monkeypatch.setattr(reaper, "gh_api", _cancel_stub(403))
    monkeypatch.setattr(reaper.sys, "argv", ["x", "--repo", "o/r"])

    rc = reaper.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "REFUSE" in out
    assert "ghosts 0" in out
