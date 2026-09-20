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
    classify_agent_event,
    desktop_paint,
    is_admission_ack,
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
    return {M.FIRST_PAINT: first_event}


def test_an_enforced_cell_inside_the_budget_passes():
    verdict = M.judge(
        ("tui", "warm"),
        _stats({"n": 7, "p50": 40.0, "p95": 90.0}),
        provider_kind=M.ENFORCED_PROVIDER,
    )
    assert verdict.status == "PASS"
    assert not verdict.failed


def test_an_enforced_cell_over_the_budget_FAILS():
    """The gate can fail — otherwise it would be a report wearing a verdict's name."""
    verdict = M.judge(
        ("tui", "warm"),
        _stats({"n": 7, "p50": 400.0, "p95": 900.0}),
        provider_kind=M.ENFORCED_PROVIDER,
    )
    assert verdict.status == "FAIL"
    assert verdict.failed


def test_a_p95_over_budget_warns_and_does_not_fail():
    """A p95 is host contention on a shared machine, not a code regression.

    With seven runs a p99 IS the max, so asserting either would be asserting the
    machine's queue. It is reported, and reported LOUDLY.
    """
    verdict = M.judge(
        ("desktop", "warm"),
        _stats({"n": 7, "p50": 120.0, "p95": 380.0}),
        provider_kind=M.ENFORCED_PROVIDER,
    )
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
    assert (
        M.judge(("tui", "warm"), fast, concurrency=1, provider_kind=M.ENFORCED_PROVIDER).status
        == "PASS"
    )
    assert (
        M.judge(("tui", "warm"), slow, concurrency=1, provider_kind=M.ENFORCED_PROVIDER).status
        == "FAIL"
    )
    eight = M.judge(("tui", "warm"), slow, concurrency=8, provider_kind=M.ENFORCED_PROVIDER)
    assert eight.status == "REPORTED"
    assert "not asserted" in eight.reason
    assert not eight.failed


def test_a_cell_with_no_budget_is_only_reported():
    verdict = M.judge(("exec", "cold"), _stats({"n": 7, "p50": 12_000.0}))
    assert verdict.status == "REPORTED"
    assert not verdict.failed


def test_a_cell_that_observed_nothing_is_reported_not_failed():
    """For a cell the gate does not enforce, "no samples" is a fact about the channel.

    The ENFORCED version of this case is ``NO-DATA`` and exits 3 — see
    ``test_an_enforced_cell_that_observed_nothing_is_no_data_not_a_pass``: a gate with
    nothing to judge must not read as a pass.
    """
    verdict = M.judge(("exec", "cold"), _stats({"n": 0, "p50": M.UNAVAILABLE}))
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
    assert M.FIRST_PAINT in M.TABLE_METRICS
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
    assert marks.marks[M.FIRST_PAINT] == pytest.approx(500.0)
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
    assert sample[M.FIRST_PAINT] == pytest.approx(200.0)
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
    text = render_report({"cells": [_cell("tui", "warm", {M.FIRST_PAINT: 40.0})]})
    assert "40/40/40" in text
    assert "first_reasoning" in text
    assert "0/0/0" not in text


def test_report_carries_the_run_count_and_the_tree():
    text = render_report(
        {
            "cells": [_cell("tui", "warm", {M.FIRST_PAINT: 40.0})],
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
        _cell("exec", "warm", {M.FIRST_PAINT: 10.0}),
        _cell("tui", "warm", {M.FIRST_PAINT: 20.0}),
    ]
    text = render_report({"cells": cells})
    assert text.index("tui") < text.index("exec")


def test_report_shows_cell_warnings():
    cell = _cell("desktop", "cold", {M.FIRST_PAINT: 2600.0})
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
            M.FIRST_PAINT: 100.0,
            M.FIRST_TEXT: 100.0,
            M.PROVIDER_TEXT: 500.0,
        }
    ]
    cells = bench_ttft._reduce_cells(
        {("tui", "warm", 1): samples}, runs=1, provider_kind=M.ENFORCED_PROVIDER
    )
    assert any("mis-paired" in warning for warning in cells[0]["warnings"])


def test_reduce_cells_flags_reasoning_no_front_end_received():
    """THE FINDING, asserted at the level the harness reports it."""
    samples = [
        {
            **{metric: M.UNAVAILABLE for metric in M.METRICS},
            "token": "t",
            "submit_epoch": 0.0,
            M.FIRST_PAINT: 300.0,
            M.FIRST_TEXT: 300.0,
            M.PROVIDER_TEXT: 200.0,
            M.PROVIDER_REASONING: 100.0,
        }
    ]
    cells = bench_ttft._reduce_cells(
        {("tui", "warm", 1): samples}, runs=1, provider_kind=M.ENFORCED_PROVIDER
    )
    assert any("NO front end received it" in warning for warning in cells[0]["warnings"])


def test_reduce_cells_says_nothing_when_reasoning_did_reach_the_front_end():
    samples = [
        {
            **{metric: M.UNAVAILABLE for metric in M.METRICS},
            "token": "t",
            "submit_epoch": 0.0,
            M.FIRST_PAINT: 300.0,
            M.FIRST_TEXT: 300.0,
            M.FIRST_REASONING: 200.0,
            M.PROVIDER_TEXT: 250.0,
            M.PROVIDER_REASONING: 100.0,
        }
    ]
    cells = bench_ttft._reduce_cells(
        {("tui", "warm", 1): samples}, runs=1, provider_kind=M.ENFORCED_PROVIDER
    )
    assert not any("NO front end received it" in warning for warning in cells[0]["warnings"])


def test_reduce_cells_verdict_matches_the_budget_rule():
    fast = [
        {
            **{metric: M.UNAVAILABLE for metric in M.METRICS},
            "token": "t",
            "submit_epoch": 0.0,
            M.FIRST_PAINT: 50.0,
        }
    ]
    slow = [
        {
            **{metric: M.UNAVAILABLE for metric in M.METRICS},
            "token": "t",
            "submit_epoch": 0.0,
            M.FIRST_PAINT: 5000.0,
        }
    ]
    fast_verdict = bench_ttft._reduce_cells(
        {("tui", "warm", 1): fast}, runs=1, provider_kind=M.ENFORCED_PROVIDER
    )[0]
    slow_verdict = bench_ttft._reduce_cells(
        {("tui", "warm", 1): slow}, runs=1, provider_kind=M.ENFORCED_PROVIDER
    )[0]
    assert fast_verdict["verdict"]["status"] == "PASS"
    assert slow_verdict["verdict"]["status"] == "FAIL"


# ---------------------------------------------------------------------------
# What the gate is allowed to judge (review round 1, Q2 and Q3)
# ---------------------------------------------------------------------------


def test_the_gate_never_enforces_the_real_wire_arm():
    """A verdict on a cell with a real network hop inside it is a verdict on the box.

    Measured evidence for the rule: the same cell, on the same host, PASSED at 143 ms
    (load 86) and FAILED at 624 ms (load 111-131) on the real-wire arm, while the
    stubbed arm sat at 26 ms. Scoping enforcement by provider is what stops the exit
    code being decided by the weather, and it is a SCOPING change rather than a
    loosened threshold: BUDGET_MS did not move, and the cell is still printed against
    it.
    """
    stats = _stats({"n": 7, "p50": 900.0, "p95": 1200.0})
    verdict = M.judge(("tui", "warm"), stats, provider_kind="loopback")
    assert verdict.status == "REPORTED"
    assert not verdict.failed
    assert "NOT asserted" in verdict.reason
    assert M.BUDGET_MS == 300.0


def test_an_unknown_provider_arm_is_not_enforced_either():
    """Fail closed: an un-named arm is never asserted, so a caller that forgets the
    argument cannot claim a verdict it did not measure."""
    assert M.judge(("tui", "warm"), _stats({"n": 7, "p50": 900.0})).status == "REPORTED"


def test_an_out_of_band_enforced_cell_is_not_judged():
    """Neither pass nor fail, and exit 3 — the box was not a fair reading."""
    stats = _stats({"n": 7, "p50": 90.0, "p95": 120.0})
    verdict = M.judge(
        ("tui", "warm"),
        stats,
        provider_kind=M.ENFORCED_PROVIDER,
        load_per_cpu=M.BUDGET_LOAD_PER_CPU_MAX + 1,
    )
    assert verdict.status == "OUT-OF-BAND"
    assert verdict.indeterminate
    assert not verdict.failed
    assert M.INDETERMINATE_EXIT_CODE == 3


def test_an_enforced_cell_that_observed_nothing_is_no_data_not_a_pass():
    """A gate that passes on no data is worse than no gate (round 1, minor)."""
    verdict = M.judge(
        ("tui", "warm"),
        _stats({"n": 0, "p50": M.UNAVAILABLE}),
        provider_kind=M.ENFORCED_PROVIDER,
    )
    assert verdict.status == "NO-DATA"
    assert verdict.indeterminate
    assert not verdict.failed


def test_a_cell_whose_every_sample_died_is_failed_not_absent():
    """The sweep continues past a dead cell, but the run cannot come out green."""
    verdict = M.judge(
        ("tui", "warm"),
        _stats({"n": 0, "p50": M.UNAVAILABLE}),
        provider_kind=M.ENFORCED_PROVIDER,
        errors=7,
    )
    assert verdict.status == "FAILED"
    assert verdict.failed


# ---------------------------------------------------------------------------
# The exit code, pinned by the observable (review round 1, F1)
# ---------------------------------------------------------------------------


def _drive_amain(
    monkeypatch,
    tmp_path,
    *,
    argv,
    paints,
    dead_channels=(),
    load=1.0,
):
    """Run ``_amain`` with the drivers stubbed, and return its exit code.

    Drives the REAL parser so the flags the finding is about (``--no-assert-budget``)
    are the ones a user types, and stubs only the three things that need a box: the
    per-cell driver, the provenance check, and the ambient-env strip.
    """
    import asyncio

    async def fake_one_run(**kwargs):
        if kwargs["channel"] in dead_channels:
            raise RuntimeError("sqlite3.OperationalError: database is locked")
        paint = paints.get(kwargs["channel"], 40.0)
        return [
            {"arm": arm, M.FIRST_PAINT: paint, M.FIRST_TEXT: paint + 10.0}
            for arm in kwargs["arms"]
            for _ in range(kwargs["concurrency"])
        ]

    monkeypatch.setattr(bench_ttft, "_one_run", fake_one_run)
    monkeypatch.setattr(
        "scripts.bench_tree.describe",
        lambda rev: {
            "rev": "deadbeef",
            "worktree_head": "deadbeef",
            "subtree": "local_operator",
            "verified": True,
        },
    )
    monkeypatch.setattr(bench_ttft, "strip_inherited_runtime_env", lambda: {})
    monkeypatch.setattr(bench_ttft, "pin_shared_caches", lambda *a, **k: None)
    # THE BAND IS PINNED, because an exit-code assertion that reads the box's load is
    # the defect QA Q2 found wearing a test's clothes: on a busy evening these cells
    # would be OUT-OF-BAND and the assertion would fail for a reason that has nothing
    # to do with the code. The band's own behaviour is pinned separately.
    monkeypatch.setattr(bench_ttft, "load_per_cpu", lambda: load)
    args = bench_ttft.build_parser().parse_args(
        [*argv, "--pycache-prefix", str(tmp_path / "pc"), "--tiktoken-cache", str(tmp_path / "tk")]
    )
    return asyncio.run(bench_ttft._amain(args))


def test_the_gate_exit_code_is_pinned_by_the_observable(monkeypatch, tmp_path):
    """The budget's OBSERVABLE, which nothing pinned before (round 1, F1).

    The reviewer mutated a scratch copy — the gate's ``return 2`` changed to
    ``return 0``, the call deleted — and every test still passed, because the rule
    was pinned and the exit code was not. These three assertions are the missing
    half: over budget exits 2, inside exits 0, and ``--no-assert-budget`` exits 0
    over budget.
    """
    over = _drive_amain(
        monkeypatch,
        tmp_path,
        argv=["--channels", "tui", "--arms", "warm", "--provider", M.ENFORCED_PROVIDER],
        paints={"tui": 400.0},
    )
    assert over == 2, "an enforced cell over budget must exit 2"

    under = _drive_amain(
        monkeypatch,
        tmp_path,
        argv=["--channels", "tui", "--arms", "warm", "--provider", M.ENFORCED_PROVIDER],
        paints={"tui": 40.0},
    )
    assert under == 0, "an enforced cell inside budget must exit 0"

    suppressed = _drive_amain(
        monkeypatch,
        tmp_path,
        argv=[
            "--channels",
            "tui",
            "--arms",
            "warm",
            "--provider",
            M.ENFORCED_PROVIDER,
            "--no-assert-budget",
        ],
        paints={"tui": 400.0},
    )
    assert suppressed == 0, "--no-assert-budget must print the verdicts and exit 0"


def test_the_gate_exit_code_is_three_when_it_cannot_judge(monkeypatch, tmp_path, capsys):
    """Out of band is not a pass: the pipeline must not read green out of it."""
    code = _drive_amain(
        monkeypatch,
        tmp_path,
        argv=["--channels", "tui", "--arms", "warm", "--provider", M.ENFORCED_PROVIDER],
        paints={"tui": 40.0},
        load=M.BUDGET_LOAD_PER_CPU_MAX + 5,
    )
    assert code == M.INDETERMINATE_EXIT_CODE
    assert "NOT JUDGED" in capsys.readouterr().err


def test_a_real_wire_run_never_exits_non_zero_on_the_budget(monkeypatch, tmp_path):
    """The scoping is visible in the observable too, not just in the verdict text."""
    code = _drive_amain(
        monkeypatch,
        tmp_path,
        argv=["--channels", "tui", "--arms", "warm", "--provider", "loopback"],
        paints={"tui": 4000.0},
    )
    assert code == 0


def test_the_sweep_records_a_dead_cell_and_keeps_going(monkeypatch, tmp_path):
    """A cell that dies is a FAILED cell, not the end of the sweep (round 1, Q1).

    The reviewer's own repro: a TUI child aborting on the auth.db lock propagated out
    of the driver, and every remaining cell was lost — 2 of ~29 invocations, rc=1, no
    table. Here the first channel dies and the second must still be measured, with the
    dead one carried into the verdicts.
    """
    import json

    report = tmp_path / "report.json"
    code = _drive_amain(
        monkeypatch,
        tmp_path,
        argv=[
            "--channels",
            "tui,desktop",
            "--arms",
            "warm",
            "--provider",
            M.ENFORCED_PROVIDER,
            "--json",
            str(report),
        ],
        paints={"tui": 40.0, "desktop": 50.0},
        dead_channels=("tui",),
    )
    assert code == 2, "the dead cell is a failure, not a silent omission"
    written = json.loads(report.read_text())
    # Keyed by concurrency as well: the default sweep runs 1/4/8 and only the single
    # turn is asserted, so collapsing the key would read the 8-concurrent verdict.
    cells = {(c["channel"], c["arm"], c["concurrency"]): c for c in written["cells"]}
    assert ("desktop", "warm", 1) in cells, "the sweep must continue past a dead cell"
    assert cells[("tui", "warm", 1)]["verdict"]["status"] == "FAILED"
    assert cells[("desktop", "warm", 1)]["verdict"]["status"] == "PASS"
    assert written["errors"]


# ---------------------------------------------------------------------------
# The echo exclusion and the reverse self-check (review round 1, Q5)
# ---------------------------------------------------------------------------


def test_the_echo_of_the_submitted_message_is_never_a_paint():
    """The seam must not count the user's own row (round 1, Q5).

    The mutation this test exists for: a seam that stamps every event reads ~3 ms —
    the submit round trip — and would pass every gate in this harness for the wrong
    reason. Both wire readers that can see a user row are covered.
    """
    from local_operator.harness.types import Message, MessageUpdateEvent, TextContent

    echo = MessageUpdateEvent(
        message=Message(role="user", content=[TextContent(text="what the operator typed")]),
        delta="what the operator typed",
    )
    assert classify_agent_event(echo) == (False, False)

    answer = MessageUpdateEvent(
        message=Message(role="assistant", content=[TextContent(text="the model's answer")]),
        delta="the model's answer",
    )
    assert classify_agent_event(answer) == (True, False)

    desktop_echo = {
        "type": "message_update",
        "payload": {
            "type": "message_update",
            "delta": "what the operator typed",
            "message": {"role": "user", "content": [{"text": "what the operator typed"}]},
        },
    }
    assert desktop_paint(desktop_echo) == (False, False, False)
    desktop_answer = {
        "type": "message_update",
        "payload": {
            "type": "message_update",
            "delta": "the model's answer",
            "message": {"role": "assistant", "content": [{"text": "the model's answer"}]},
        },
    }
    assert desktop_paint(desktop_answer) == (True, True, False)


def test_a_frame_that_precedes_its_own_stream_is_flagged():
    """The other direction of the physical rule, which round 1 found missing (Q5).

    A front end cannot receive a frame before the runtime entered the stream it
    comes from. A seam that counts the submit echo reads exactly this shape — 3 ms
    against a stream entered at 11 ms — and nothing checked it.
    """
    cells = bench_ttft._reduce_cells(
        {
            ("tui", "warm", 1): [
                {M.FIRST_PAINT: 3.0, M.STREAM_ENTERED: 11.0, M.FIRST_TEXT: 40.0},
                {M.FIRST_PAINT: 4.0, M.STREAM_ENTERED: 12.0, M.FIRST_TEXT: 41.0},
            ]
        },
        2,
        provider_kind=M.ENFORCED_PROVIDER,
    )
    warnings = " ".join(cells[0]["warnings"])
    assert "cannot precede the stream" in warnings


def test_a_consistent_sample_is_not_flagged():
    """Prove the check can tell the two apart, or it would just be noise."""
    cells = bench_ttft._reduce_cells(
        {
            ("tui", "warm", 1): [
                {M.FIRST_PAINT: 90.0, M.STREAM_ENTERED: 11.0, M.FIRST_TEXT: 120.0},
            ]
        },
        1,
        provider_kind=M.ENFORCED_PROVIDER,
    )
    assert cells[0]["warnings"] == []


# ---------------------------------------------------------------------------
# The acknowledgement, carried from the ack-before-engage branch (#1343)
# ---------------------------------------------------------------------------


def test_the_acknowledgement_is_matched_on_the_submits_own_request_id():
    """The stream carries every turn's frames, so the name alone is not enough.

    A previous turn's acknowledgement arriving late would otherwise be read as this
    one's — which on the cold arm would report the new change's best case as an old
    turn's leftover.
    """
    mine = {"type": "admission.accepted", "payload": {"request_id": "abc", "mode": "turn"}}
    theirs = {"type": "admission.accepted", "payload": {"request_id": "zzz", "mode": "turn"}}
    nested = {"type": "event", "payload": {"type": "admission.accepted", "request_id": "abc"}}
    assert is_admission_ack(mine, "abc")
    assert not is_admission_ack(theirs, "abc")
    assert is_admission_ack(nested, "abc")
    assert not is_admission_ack({"type": "message_update", "payload": {}}, "abc")
    assert not is_admission_ack(mine, "")


def test_the_acknowledgement_gate_holds_the_median_and_the_tail():
    """Two ceilings, and the tail is the second (see ADMISSION_CEILING_MS)."""
    fast = {M.ADMITTED: {"n": 7, "p50": 62.0, "p95": 148.0}}
    slow_median = {M.ADMITTED: {"n": 7, "p50": 340.0, "p95": 380.0}}
    slow_tail = {M.ADMITTED: {"n": 7, "p50": 190.0, "p95": 620.0}}
    # Bound to a name and narrowed before the attribute read: `judge_admission`
    # returns None where the channel does not acknowledge, and the tests below are
    # about the verdicts rather than that case.
    fast_verdict = M.judge_admission(("desktop", "cold"), fast, provider_kind=M.ENFORCED_PROVIDER)
    assert fast_verdict is not None
    assert fast_verdict.status == "PASS"
    slow_verdict = M.judge_admission(
        ("desktop", "cold"), slow_median, provider_kind=M.ENFORCED_PROVIDER
    )
    assert slow_verdict is not None
    assert slow_verdict.status == "FAIL"
    tail = M.judge_admission(("desktop", "cold"), slow_tail, provider_kind=M.ENFORCED_PROVIDER)
    assert tail is not None and tail.status == "FAIL"
    assert "tail bound" in tail.reason


def test_the_acknowledgement_is_only_scored_where_it_exists_and_in_scope():
    """No frame on the other channels, no verdict; and the same scope rules apply."""
    stats = {M.ADMITTED: {"n": 7, "p50": 62.0, "p95": 148.0}}
    assert M.judge_admission(("tui", "warm"), stats, provider_kind=M.ENFORCED_PROVIDER) is None
    assert M.judge_admission(("exec", "cold"), stats, provider_kind=M.ENFORCED_PROVIDER) is None
    assert (
        M.judge_admission(("desktop", "cold"), stats, provider_kind="loopback") is None
    ), "the real-wire arm is never asserted, the ack included"
    assert (
        M.judge_admission(
            ("desktop", "cold"), stats, concurrency=4, provider_kind=M.ENFORCED_PROVIDER
        )
        is None
    )
    assert (
        M.judge_admission(("desktop", "cold"), stats, provider_kind=M.ENFORCED_PROVIDER) is not None
    )


def test_an_acknowledgement_that_never_arrived_is_no_data_not_a_pass():
    verdict = M.judge_admission(
        ("desktop", "cold"),
        {M.ADMITTED: {"n": 0, "p50": M.UNAVAILABLE}},
        provider_kind=M.ENFORCED_PROVIDER,
    )
    assert verdict is not None and verdict.status == "NO-DATA"
    assert verdict.indeterminate


def test_the_acknowledgement_gate_is_in_the_exit_code(monkeypatch, tmp_path):
    """A missed acknowledgement must move the exit code, not just the table."""

    async def fake_one_run(**kwargs):
        return [
            {"arm": arm, M.FIRST_PAINT: 40.0, M.FIRST_TEXT: 50.0, M.ADMITTED: 900.0}
            for arm in kwargs["arms"]
            for _ in range(kwargs["concurrency"])
        ]

    monkeypatch.setattr(bench_ttft, "_one_run", fake_one_run)
    monkeypatch.setattr(
        "scripts.bench_tree.describe",
        lambda rev: {
            "rev": "deadbeef",
            "worktree_head": "deadbeef",
            "subtree": "local_operator",
            "verified": True,
        },
    )
    monkeypatch.setattr(bench_ttft, "strip_inherited_runtime_env", lambda: {})
    monkeypatch.setattr(bench_ttft, "pin_shared_caches", lambda *a, **k: None)
    # Pinned: an exit-code assertion that reads the box's load would fail on a busy
    # evening for a reason that has nothing to do with the code (QA Q2's lesson).
    monkeypatch.setattr(bench_ttft, "load_per_cpu", lambda: 1.0)
    args = bench_ttft.build_parser().parse_args(
        [
            "--channels",
            "desktop",
            "--arms",
            "cold",
            "--concurrency",
            "1",
            "--provider",
            M.ENFORCED_PROVIDER,
            "--pycache-prefix",
            str(tmp_path / "pc"),
            "--tiktoken-cache",
            str(tmp_path / "tk"),
        ]
    )
    import asyncio

    code = asyncio.run(bench_ttft._amain(args))
    assert code == 2, "the paint is inside budget and the acknowledgement is not"
