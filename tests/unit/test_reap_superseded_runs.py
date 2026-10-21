"""The reaper cancels QUEUED runs and nothing else.

Every test here is a safety rule rather than a happy path, because the failure
mode of this script is cancelling work somebody needs. The rules under test are
the ones in `scripts/reap_superseded_runs.py`'s docstring:

* only `queued` runs are touched -- a started run is doing work;
* protected branches (`main`, `release-next-*`) are never touched -- a gate
  that can be cancelled is not a gate;
* a PR run is cancelled only when BOTH the PR head and the branch tip agree
  that it has been superseded;
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
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
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
    api["tip"][branch] = "b" * 40          # tip HAS moved
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


def test_a_pr_run_is_cancelled_only_when_both_readings_agree(api):
    api["pr"][7] = "b" * 40
    api["tip"]["feat/x"] = "b" * 40
    should, reason = reaper.decide(
        "o/r", _run(event="pull_request", pull_requests=[{"number": 7}]), 5.0
    )
    assert should
    assert "moved past" in reason


def test_a_pr_run_is_kept_when_the_two_readings_disagree(api):
    """The guard that matters: one reading saying 'current' is enough to KEEP.

    A cancellation requires both the PR head and the branch tip to have moved
    past this run. If either still points at the run's commit, the run may be
    the one that is about to be current, and cancelling it would delete live
    CI on a guess.
    """
    api["pr"][7] = "b" * 40    # PR says moved
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


def test_an_unreadable_branch_tip_on_a_pr_run_is_kept(api):
    api["pr"][7] = "b" * 40
    should, reason = reaper.decide(
        "o/r", _run(event="pull_request", pull_requests=[{"number": 7}]), 5.0
    )
    assert not should
    assert "refusing to act on one reading" in reason


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
    should, reason = reaper.decide(
        "o/r", _run(event="pull_request", created_at=ancient), 5.0, 24.0
    )
    assert should
    assert "abandonment bound" in reason


def test_a_run_just_under_the_abandonment_bound_still_uses_the_normal_rules(api):
    nearly = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=23)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    api["tip"]["feat/x"] = "a" * 40          # tip is current
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
