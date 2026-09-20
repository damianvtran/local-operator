"""The TTFT harness's arithmetic, budget rule and detectors, pinned without running it.

WHY THESE TESTS AND NOT AN END-TO-END ONE
=========================================
A six-channel run needs a machine, a daemon, spawned runtimes and several minutes,
so it cannot be a CI gate — and the parts of this harness that could be WRONG
without the run failing are exactly the parts that are pure functions: which
percentile a p99 over seven samples actually is, whether a missing observation is
counted as a fast one, and whether the budget verdict fires when it should. Those
are tested here against literals.

The one claim that is deliberately proved in both directions is the budget: a cell
that misses 300 ms must FAIL and a cell inside it must PASS, so the gate can be
shown to be capable of failing (AGENTS.md, "Prove the test can still fail").
"""

from __future__ import annotations

import pytest

from scripts import bench_ttft
from scripts.ttft import metrics as M
from scripts.ttft.channels import (
    REASONING_NAMES,
    TurnMarks,
    desktop_paint,
    jobs_paint,
    names_reasoning,
    phone_reasoning,
    phone_text,
)
from scripts.ttft.loopback import ProviderStamps, request_token
from scripts.ttft.report import render_report

# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------


def test_percentile_is_nearest_rank_not_interpolated():
    """A p99 over seven samples IS the maximum, and the harness says so.

    Interpolating would print a number between two observations and present it as a
    measurement; nearest-rank prints the largest value actually seen.
    """
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 999.0]
    assert M.percentile(values, 50) == 40.0
    assert M.percentile(values, 95) == 999.0
    assert M.percentile(values, 99) == 999.0


def test_percentile_rejects_an_empty_sample_set():
    with pytest.raises(ValueError):
        M.percentile([], 50)


def test_percentile_rejects_an_impossible_percent():
    with pytest.raises(ValueError):
        M.percentile([1.0], 0)


# ---------------------------------------------------------------------------
# Reduction: unavailable is not zero
# ---------------------------------------------------------------------------


def test_reduce_samples_excludes_and_counts_unavailable():
    reduced = M.reduce_samples([100.0, M.UNAVAILABLE, 200.0, M.UNAVAILABLE])
    assert reduced["n"] == 2
    assert reduced["unavailable"] == 2
    assert reduced["p50"] == 100.0
    assert reduced["max"] == 200.0


def test_reduce_samples_all_unavailable_reports_nothing_observed():
    reduced = M.reduce_samples([M.UNAVAILABLE, M.UNAVAILABLE])
    assert reduced["n"] == 0
    assert reduced["p50"] == M.UNAVAILABLE
    assert reduced["min"] == M.UNAVAILABLE


def test_reduce_samples_of_nothing_is_not_a_zero_latency():
    reduced = M.reduce_samples([])
    assert reduced["n"] == 0
    assert reduced["p95"] == M.UNAVAILABLE


# ---------------------------------------------------------------------------
# The budget rule
# ---------------------------------------------------------------------------


def _stats(first_event: dict[str, object]) -> dict[str, dict[str, object]]:
    return {M.FIRST_EVENT: first_event}


def test_an_enforced_cell_inside_the_budget_passes():
    verdict = M.judge(("tui", "warm"), _stats({"n": 7, "p50": 40.0, "p95": 90.0}))
    assert verdict.status == "PASS"
    assert not verdict.failed


def test_an_enforced_cell_over_the_budget_FAILS():
    """The gate can fail — otherwise it would be a report wearing a verdict's name."""
    verdict = M.judge(("tui", "warm"), _stats({"n": 7, "p50": 400.0, "p95": 900.0}))
    assert verdict.status == "FAIL"
    assert verdict.failed


def test_a_p95_over_budget_warns_and_does_not_fail():
    """A p95 is host contention on a shared machine, not a code regression.

    With seven runs a p99 IS the max, so asserting either would be asserting the
    machine's queue. It is reported, and reported LOUDLY.
    """
    verdict = M.judge(("desktop", "warm"), _stats({"n": 7, "p50": 120.0, "p95": 380.0}))
    assert verdict.status == "WARN"
    assert not verdict.failed


def test_a_pending_cell_over_the_budget_is_unmet_not_failed():
    """The cold desktop arm is measured against the budget before it can hold it.

    It is reported as UNMET with the change that closes it, and never silently
    dropped: the number is that change's evidence.
    """
    verdict = M.judge(("desktop", "cold"), _stats({"n": 7, "p50": 2600.0}))
    assert verdict.status == "UNMET"
    assert not verdict.failed
    assert "ack-before-engage" in verdict.reason


def test_a_pending_cell_that_now_passes_says_to_promote_it():
    verdict = M.judge(("desktop", "cold"), _stats({"n": 7, "p50": 90.0}))
    assert verdict.status == "PASS"
    assert "ENFORCED_CELLS" in verdict.reason


def test_the_assertion_is_scoped_to_the_concurrency_the_code_owns():
    """At 8 concurrent turns on this host the number is the HOST's queue.

    Every channel measured far over budget there (exec 26.9 s, sse-jobs 22.4 s,
    desktop 19.8 s cold / 733 ms warm, tui 1.2 s), which is why the assertion covers
    concurrency 1 and the rest is reported against the same budget rather than
    deleted or silently passed.
    """
    fast = _stats({"n": 7, "p50": 40.0, "p95": 90.0})
    slow = _stats({"n": 7, "p50": 1200.0, "p95": 1900.0})
    assert M.judge(("tui", "warm"), fast, concurrency=1).status == "PASS"
    assert M.judge(("tui", "warm"), slow, concurrency=1).status == "FAIL"
    eight = M.judge(("tui", "warm"), slow, concurrency=8)
    assert eight.status == "REPORTED"
    assert "not asserted" in eight.reason
    assert not eight.failed


def test_a_cell_with_no_budget_is_only_reported():
    verdict = M.judge(("exec", "cold"), _stats({"n": 7, "p50": 12_000.0}))
    assert verdict.status == "REPORTED"
    assert not verdict.failed


def test_a_cell_that_observed_nothing_is_reported_not_failed():
    verdict = M.judge(("tui", "warm"), _stats({"n": 0, "p50": M.UNAVAILABLE}))
    assert verdict.status == "REPORTED"
    assert not verdict.failed


def test_the_provider_columns_carry_no_budget():
    """The provider floor is reported and budgeted, never asserted.

    Encoded as a test so a future reader who "fixes" the harness by budgeting the
    first provider token has to delete an assertion that says why that is a lie.
    """
    assert M.BUDGET_MS == 300.0
    assert M.PROVIDER_REASONING not in M.TABLE_METRICS
    assert M.PROVIDER_TEXT not in M.TABLE_METRICS
    assert M.FIRST_EVENT in M.TABLE_METRICS
    assert M.PROVIDER_REASONING not in {cell for cell in M.ENFORCED_CELLS}


# ---------------------------------------------------------------------------
# The detectors
# ---------------------------------------------------------------------------


def test_desktop_text_frame_paints():
    frame = {"type": "event", "payload": {"type": "message_update", "delta": "Hello"}}
    assert desktop_paint(frame) == (True, True, False)


def test_desktop_empty_delta_does_not_paint():
    frame = {"type": "event", "payload": {"type": "message_update", "delta": ""}}
    assert desktop_paint(frame) == (False, False, False)


def test_desktop_reasoning_frame_paints_as_reasoning():
    for name in sorted(REASONING_NAMES):
        assert desktop_paint({"type": "event", "payload": {"type": name}}) == (True, False, True)


def test_desktop_bookkeeping_frames_do_not_paint():
    """The frames a cold submit produces before any content must not count.

    ``admitted_ms`` is reported for exactly this arm, because the first CONTENT
    frame is not the first thing the client receives.
    """
    for kind in ("agent_start", "turn_start", "provider_turn_start", "message_start"):
        frame = {"type": "event", "payload": {"type": kind}}
        assert desktop_paint(frame) == (False, False, False)


def test_jobs_delta_frame_paints():
    assert jobs_paint({"type": "message.delta", "delta": "Hi"}) == (True, True, False)
    assert jobs_paint({"type": "message.delta", "delta": ""}) == (False, False, False)
    assert jobs_paint({"type": "reasoning.delta", "delta": "..."}) == (True, False, True)
    assert jobs_paint({"type": "turn.start"}) == (False, False, False)


def test_reasoning_names_are_matched_case_insensitively():
    assert names_reasoning("reasoning.delta")
    assert names_reasoning("Reasoning.Delta")
    assert not names_reasoning("message.delta")


def test_phone_text_reads_only_assistant_rows():
    frame = {
        "transcript": [
            {"kind": "user", "text": "hi"},
            {"kind": "assistant", "text": "there"},
            {"kind": "tool", "text": "noise"},
        ]
    }
    assert phone_text(frame) == "there"


def test_phone_reasoning_has_no_kind_to_read_today():
    """Reasoning rows have no kind on this tree, so the column is unavailable.

    When the streamed-reasoning change lands, this test is the one that says what
    the phone has to start emitting for the column to fill in.
    """
    frame = {"transcript": [{"kind": "assistant", "text": "there"}]}
    assert phone_reasoning(frame) == ""
    assert phone_reasoning({"transcript": [{"kind": "reasoning", "text": "thought"}]}) == "thought"


# ---------------------------------------------------------------------------
# Turn marks: what is recorded, and what a missing observation looks like
# ---------------------------------------------------------------------------


def test_turn_marks_state_every_metric_and_default_to_unavailable():
    marks = TurnMarks(token="t", submit_monotonic=0.0, submit_epoch=0.0)
    sample = marks.sample(arm="warm", channel="tui")
    for metric in M.METRICS:
        assert sample[metric] == M.UNAVAILABLE, metric
    assert sample["arm"] == "warm"


def test_turn_marks_record_first_paint_but_not_a_later_one():
    marks = TurnMarks(token="t", submit_monotonic=100.0, submit_epoch=0.0)
    marks.note_paint_at(100.5, paints=True, emit_monotonic=100.4)
    marks.note_paint_at(100.9, paints=True)
    marks.note_paint_at(101.2, paints=True, carries_text=True)
    assert marks.marks[M.FIRST_EVENT] == pytest.approx(500.0)
    assert marks.marks[M.RUNTIME_EMIT] == pytest.approx(400.0)
    assert marks.marks[M.FIRST_TEXT] == pytest.approx(1200.0)


def test_a_paint_that_is_not_text_does_not_become_first_text():
    """The phone's activity line is a paint; it is not assistant text.

    Folding the two would report the phone's text latency as the moment its working
    line appeared.
    """
    marks = TurnMarks(token="t", submit_monotonic=0.0, submit_epoch=0.0)
    marks.note_paint_at(0.2, paints=True)
    sample = marks.sample(arm="cold", channel="mobile")
    assert sample[M.FIRST_EVENT] == pytest.approx(200.0)
    assert sample[M.FIRST_TEXT] == M.UNAVAILABLE


def test_a_paint_with_no_emit_stamp_is_unavailable_not_zero():
    marks = TurnMarks(token="t", submit_monotonic=0.0, submit_epoch=0.0)
    marks.note_paint_at(0.2, paints=True)
    assert marks.sample(arm="cold", channel="exec")[M.RUNTIME_EMIT] == M.UNAVAILABLE


# ---------------------------------------------------------------------------
# Provider correlation
# ---------------------------------------------------------------------------


def test_request_token_reads_the_marker_from_the_last_user_message():
    body = {
        "messages": [
            {"role": "user", "content": "[bench:aaaaaaaa] old"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "[bench:bbbbbbbb] Reply with one short sentence."},
        ]
    }
    assert request_token(body) == "bbbbbbbb"


def test_request_token_is_empty_for_a_helper_call():
    """A helper call (auto-naming, effort classification) carries no marker."""
    body = {"messages": [{"role": "user", "content": "Name this conversation."}]}
    assert request_token(body) == ""


def test_request_token_reads_a_content_block_list():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "[bench:cccccccc] hi"}],
            }
        ]
    }
    assert request_token(body) == "cccccccc"


def test_provider_stamp_delta_is_measured_from_the_submit_epoch():
    stamps = ProviderStamps()
    stamps.stamp_reasoning("t")
    stamped = stamps.reasoning["t"]
    assert stamps.delta_ms("reasoning", "t", stamped - 0.4) == pytest.approx(400.0, abs=1.0)


def test_provider_stamp_for_an_unseen_token_is_unavailable():
    assert ProviderStamps().delta_ms("text", "nope", 0.0) == M.UNAVAILABLE


def test_provider_stamp_outside_any_real_turn_is_unavailable():
    """A clock step must not read as a measurement.

    The provider columns are compared on the wall clock because submit and provider
    are different processes for most channels; that is exactly why a nonsensical
    delta is dropped rather than reported.
    """
    stamps = ProviderStamps()
    stamps.text["t"] = 1000.0
    assert stamps.delta_ms("text", "t", 2000.0) == M.UNAVAILABLE
    assert stamps.delta_ms("text", "t", 1000.0 - 10_000.0) == M.UNAVAILABLE


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _cell(
    channel: str, arm: str, metrics: dict[str, float] | None = None, *, concurrency: int = 1
) -> dict[str, object]:
    reduced = {
        metric: {
            "n": 7,
            "unavailable": 0,
            "min": value,
            "max": value,
            "p50": value,
            "p95": value,
            "p99": value,
        }
        for metric, value in (metrics or {}).items()
    }
    for metric in M.METRICS:
        reduced.setdefault(metric, {"n": 0, "unavailable": 7, "p50": M.UNAVAILABLE})
    return {
        "channel": channel,
        "arm": arm,
        "concurrency": concurrency,
        "runs": 7,
        "samples": 7,
        "stats": reduced,
        "warnings": [],
        "verdict": {"status": "REPORTED", "reason": "no budget", "budget_ms": M.BUDGET_MS},
    }


def test_report_renders_a_dash_for_an_unobserved_metric():
    """Never a zero: an unobserved column and a fast one are opposite findings."""
    text = render_report({"cells": [_cell("tui", "warm", {M.FIRST_EVENT: 40.0})]})
    assert "40/40/40" in text
    assert "first_reasoning" in text
    assert "0/0/0" not in text


def test_report_carries_the_run_count_and_the_tree():
    text = render_report(
        {
            "cells": [_cell("tui", "warm", {M.FIRST_EVENT: 40.0})],
            "tree": {"rev": "a7e6b9bdxxxx", "worktree_head": "aaaa", "subtree": "local_operator"},
            "runs": 7,
            "concurrency": [1, 4, 8],
            "arms": ["cold", "warm"],
        }
    )
    assert "a7e6b9bd" in text
    assert "runs: 7" in text


def test_report_orders_channels_cheapest_first():
    cells = [
        _cell("exec", "warm", {M.FIRST_EVENT: 10.0}),
        _cell("tui", "warm", {M.FIRST_EVENT: 20.0}),
    ]
    text = render_report({"cells": cells})
    assert text.index("tui") < text.index("exec")


def test_report_shows_cell_warnings():
    cell = _cell("desktop", "cold", {M.FIRST_EVENT: 2600.0})
    cell["warnings"] = ["provider emitted reasoning at p50 900 ms and NO front end received it"]
    text = render_report({"cells": [cell]})
    assert "NO front end received it" in text


def test_report_marks_the_measured_tree_as_verified():
    text = render_report(
        {
            "cells": [],
            "tree": {
                "rev": "abc123def",
                "worktree_head": "abc123def",
                "subtree": "local_operator",
                "verified": True,
            },
        }
    )
    assert "verified=True" in text


# ---------------------------------------------------------------------------
# Cell reduction, including the instrument self-check
# ---------------------------------------------------------------------------


def test_reduce_cells_flags_a_mis_paired_provider_column():
    """A provider stamp AFTER the client received the frame is impossible.

    It means the provider column is paired to the wrong request, so the cell says so
    instead of publishing a number that cannot be true.
    """
    samples = [
        {
            **{metric: M.UNAVAILABLE for metric in M.METRICS},
            "token": "t",
            "submit_epoch": 0.0,
            M.FIRST_EVENT: 100.0,
            M.FIRST_TEXT: 100.0,
            M.PROVIDER_TEXT: 500.0,
        }
    ]
    cells = bench_ttft._reduce_cells({("tui", "warm", 1): samples}, runs=1)
    assert any("mis-paired" in warning for warning in cells[0]["warnings"])


def test_reduce_cells_flags_reasoning_no_front_end_received():
    """THE FINDING, asserted at the level the harness reports it."""
    samples = [
        {
            **{metric: M.UNAVAILABLE for metric in M.METRICS},
            "token": "t",
            "submit_epoch": 0.0,
            M.FIRST_EVENT: 300.0,
            M.FIRST_TEXT: 300.0,
            M.PROVIDER_TEXT: 200.0,
            M.PROVIDER_REASONING: 100.0,
        }
    ]
    cells = bench_ttft._reduce_cells({("tui", "warm", 1): samples}, runs=1)
    assert any("NO front end received it" in warning for warning in cells[0]["warnings"])


def test_reduce_cells_says_nothing_when_reasoning_did_reach_the_front_end():
    samples = [
        {
            **{metric: M.UNAVAILABLE for metric in M.METRICS},
            "token": "t",
            "submit_epoch": 0.0,
            M.FIRST_EVENT: 300.0,
            M.FIRST_TEXT: 300.0,
            M.FIRST_REASONING: 200.0,
            M.PROVIDER_TEXT: 250.0,
            M.PROVIDER_REASONING: 100.0,
        }
    ]
    cells = bench_ttft._reduce_cells({("tui", "warm", 1): samples}, runs=1)
    assert not any("NO front end received it" in warning for warning in cells[0]["warnings"])


def test_reduce_cells_verdict_matches_the_budget_rule():
    fast = [
        {
            **{metric: M.UNAVAILABLE for metric in M.METRICS},
            "token": "t",
            "submit_epoch": 0.0,
            M.FIRST_EVENT: 50.0,
        }
    ]
    slow = [
        {
            **{metric: M.UNAVAILABLE for metric in M.METRICS},
            "token": "t",
            "submit_epoch": 0.0,
            M.FIRST_EVENT: 5000.0,
        }
    ]
    fast_verdict = bench_ttft._reduce_cells({("tui", "warm", 1): fast}, runs=1)[0]
    slow_verdict = bench_ttft._reduce_cells({("tui", "warm", 1): slow}, runs=1)[0]
    assert fast_verdict["verdict"]["status"] == "PASS"
    assert slow_verdict["verdict"]["status"] == "FAIL"
