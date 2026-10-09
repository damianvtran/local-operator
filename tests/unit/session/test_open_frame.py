"""The open frame's pure half: the strip, the cut, and the run facts on the wire.

Every case here is a rule a CLIENT depends on, so the tests state the rule and
the reason rather than the mechanism: what a surface paints, what a page does
when a cap binds, and what the facts say about a run that is still live.

The journal shapes are the same synthetic rows ``test_transcript_index`` uses
(the wire envelope: ``{id, ts, ts_source, type, payload}``), because a strip rule
pinned against a hand-drawn approximation of a row is a rule about the test.
"""

from __future__ import annotations

import json
from typing import Any

from local_operator.session import open_frame as of
from local_operator.session.transcript_index import (
    Checkpoint,
    RunRecord,
    ScanState,
    TranscriptIndex,
)


def row(
    id_: str,
    type_: str,
    payload: dict[str, Any],
    *,
    ts: float = 1.0,
) -> dict[str, Any]:
    return {"id": id_, "ts": ts, "ts_source": "entry", "type": type_, "payload": payload}


def message(id_: str, role: str, **payload: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"kind": "message", "role": role, "content": [{"text": "x"}]}
    body.update(payload)
    return row(id_, "message", body)


def custom(id_: str, custom_type: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return row(
        id_,
        "custom",
        {"custom_type": custom_type, "details": details or {}},
    )


def index_with(
    runs: list[RunRecord], *, checkpoints: list[Checkpoint] | None = None
) -> TranscriptIndex:
    return TranscriptIndex(
        checkpoints=checkpoints or [],
        messages=[],
        sig={},
        coverage={},
        naming={},
        scan=ScanState(rows=100, offset=0, window_offset=0, window_rows=0),
        runs=runs,
    )


def run(
    *,
    opening: str,
    closing: str,
    first_seq: int,
    last_seq: int,
    settled: bool = True,
    outcome: str | None = "complete",
    actions: int = 3,
    failed: int = 1,
    worked: float = 12.5,
    complete: bool = True,
) -> RunRecord:
    return RunRecord(
        opening_user_id=opening,
        closing_answer_id=closing,
        last_id=closing or opening,
        first_seq=first_seq,
        last_seq=last_seq,
        start_ts=float(first_seq),
        end_ts=float(last_seq),
        action_count=actions,
        failed_count=failed,
        worked_seconds=worked,
        settled=settled,
        outcome=outcome,
        complete=complete,
    )


# ---------------------------------------------------------------------------
# The strip: what a surface paints
# ---------------------------------------------------------------------------


def test_the_strip_drops_only_rows_the_desktop_cannot_paint() -> None:
    """The bookkeeping envelopes go; the one custom it paints stays.

    ``system_prefix`` is here ON PURPOSE. A module comment in the renderer claims
    the prefix rows paint, and the audit's strip list followed it — but the
    desktop's own projection refuses every ``type: "custom"`` row except
    ``completion_attention``, and this test is what keeps the list tied to that
    gate rather than to the comment.
    """
    for custom_type in (
        "frontend_state_checkpoint_v1",
        "session_spend.v1",
        "session_state",
        "system_prefix",
        "selected_model",
        "attention_started",
        "hub_communication",
        "wake_schedule",
        "prune",
    ):
        assert of.strip_entry(custom("c1", custom_type)) is None, custom_type
    kept = of.strip_entry(custom("c1", "completion_attention", {"anchor": "a", "kind": "error"}))
    assert kept is not None and kept["payload"]["custom_type"] == "completion_attention"


def test_the_strip_keeps_every_field_a_client_reads() -> None:
    """The kept half, named field by field: this is the contract, not a diet."""
    stripped = of.strip_entry(
        message(
            "m1",
            "tool",
            tool_call_id="call-1",
            tool_name="bash",
            is_error=True,
            provider_payload={
                "details": {"path": "x.py", "__fault": "skipped", "delivery": "mailbox"},
                "useless": False,
                "duration_s": 3.5,
                "harness_injected": True,
                "native_replay": {"items": ["large"]},
                "system_fingerprint": "fp_1",
                "id": "resp_1",
            },
        )
    )
    assert stripped is not None
    payload = stripped["payload"]
    assert payload["tool_call_id"] == "call-1" and payload["tool_name"] == "bash"
    assert payload["is_error"] is True
    assert payload["provider_payload"] == {
        "details": {"path": "x.py", "__fault": "skipped", "delivery": "mailbox"},
        "useless": False,
        "duration_s": 3.5,
        "harness_injected": True,
    }


def test_the_strip_drops_an_empty_provider_envelope_entirely() -> None:
    """A row whose envelope held only dropped keys carries no empty object.

    Cheap in bytes and honest in shape: ``{}`` is not a value any client reads,
    and leaving it would make "the producer sent nothing" and "the producer sent
    only replay material" indistinguishable on the wire.
    """
    stripped = of.strip_entry(
        message("m1", "assistant", provider_payload={"id": "r", "system_fingerprint": "f"})
    )
    assert stripped is not None
    assert "provider_payload" not in stripped["payload"]


def test_a_compaction_keeps_its_fingerprint_and_loses_its_summary() -> None:
    """The reducer pairs a live pass by ``tokens_before``; nothing reads the rest."""
    stripped = of.strip_entry(
        row(
            "k1",
            "compaction",
            {
                "summary": "s" * 5000,
                "first_kept_entry_id": "abc",
                "tokens_before": 180000,
                "preserved_user_turns": ["y" * 100],
                "preserve_data": {"snapcompact": {"blob": "z" * 100}},
            },
        )
    )
    assert stripped is not None
    assert stripped["payload"]["tokens_before"] == 180000
    assert len(stripped["payload"]["preview_text"]) <= of._COMPACTION_PREVIEW_CHARS
    assert set(stripped["payload"]) == {"tokens_before", "preview_text"}


def test_an_unknown_row_type_is_served_verbatim() -> None:
    """A row this build does not know is not this module's to judge.

    The fail-safe direction matters: a future row type is served rather than
    stripped, so a client that DOES learn to paint it finds it present. The list
    above is an explicit one for exactly this reason.
    """
    unknown = row("p1", "prune", {"note": "keep me"})
    assert of.strip_entry(unknown) == unknown


# ---------------------------------------------------------------------------
# The cut: what a page does when a cap binds
# ---------------------------------------------------------------------------


def test_align_page_starts_the_page_at_a_user_row_when_one_is_in_reach() -> None:
    """The cut's job, stated on rows alone.

    At least ``limit`` paintable rows, and a page that BEGINS at a user row when
    the hunt can find one — that user row is the client's own run opener, so the
    page begins a run instead of cutting one. The rows the hunt added are the
    extension, and they are bounded.
    """
    rows = [
        message("u1", "user"),
        message("t1", "tool"),
        message("t2", "tool"),
        message("u2", "user"),
        message("t3", "tool"),
        message("t4", "tool"),
        custom("c1", "session_spend.v1"),
        message("t5", "tool"),
    ]
    kept, reached = of.align_page(rows, 2)
    assert reached is True
    # The cut is a cut over the rows it was given, so a row the strip drops is
    # still IN the slice — what changes is what a client paints:
    assert [row["id"] for row in kept] == ["u2", "t3", "t4", "c1", "t5"]
    assert [row["id"] for row in of.paintable(kept)] == ["u2", "t3", "t4", "t5"]
    # The newest two PAINTABLE rows are t4 and t5, and the hunt found u2 above
    # them — so the page opens on a run's own user row, not mid-run.
    assert of.is_user_row(kept[0]) is True


def test_align_page_refuses_to_pay_for_a_head_it_cannot_reach() -> None:
    """A partial run either way is not made better by rows the hunt overshot to.

    The measured shape: a run of hundreds of rows ends above the window, so no
    budget reaches its head. The page is then exactly the rows asked for, and
    ``reached`` says the head is cut so the caller can state it on the wire.
    """
    rows = [message("u1", "user")] + [message(f"tool{i}", "tool") for i in range(60)]
    kept, reached = of.align_page(rows, 3, extra_rows=10)
    assert reached is False
    assert [row["id"] for row in kept] == ["tool57", "tool58", "tool59"]
    # The same rows, with the head inside the budget: reached, and the page is
    # bigger by exactly the rows between the limit and that head.
    kept_near, reached_near = of.align_page(rows, 3, extra_rows=60)
    assert reached_near is True
    assert kept_near[0]["id"] == "u1"
    assert len(kept_near) == 61


def test_align_page_leaves_a_short_journal_whole() -> None:
    """Fewer rows than asked for is the journal's answer, not a cut."""
    rows = [message("u1", "user"), message("t1", "tool")]
    kept, reached = of.align_page(rows, 50)
    assert [row["id"] for row in kept] == ["u1", "t1"]
    assert reached is True


def test_a_cap_keeps_the_NEWEST_rows_and_says_it_dropped_the_oldest() -> None:
    """A page truncated at its newest end would be the defect, not the diet.

    Today's failure is a reader sitting at the tail of a conversation; a page
    that lost its newest rows to satisfy a cap would blank the bottom of the
    screen while rows scrolled off above it. So the caps cut the OLD end, and
    ``capped`` is how the caller knows the ``has_more`` it must report.
    """
    rows = [message(f"m{i}", "tool", content=[{"text": "x" * 700}]) for i in range(450)]
    frame = of.build_frame(rows, index=None, runs_state="building")
    assert frame.capped is True
    assert len(frame.entries) < 450
    assert frame.entries[-1]["id"] == "m449"
    assert frame.entries[0]["id"] == f"m{449 - len(frame.entries) + 1}"
    assert of.served_bytes(frame.entries) <= of.OPEN_FRAME_MAX_BYTES


def test_a_page_that_fits_reports_no_cap() -> None:
    rows = [message(f"m{i}", "user") for i in range(10)]
    frame = of.build_frame(rows, index=None, runs_state="building")
    assert frame.capped is False
    assert len(frame.entries) == 10
    assert frame.dropped_rows == 0


def test_stripped_rows_do_not_count_toward_the_page() -> None:
    """The whole point of the unit change, stated as a number."""
    rows = [custom("c1", "frontend_state_checkpoint_v1"), message("u1", "user")]
    frame = of.build_frame(rows, index=None, runs_state="building")
    assert [entry["id"] for entry in frame.entries] == ["u1"]
    assert frame.dropped_rows == 1


# ---------------------------------------------------------------------------
# The run facts
# ---------------------------------------------------------------------------


def test_publish_runs_states_counts_for_a_settled_run_only() -> None:
    """A live tail's numbers would be corrected by the next row.

    The run is still LISTED — a client needs to know which run it is, and that it
    is live — with its counts absent rather than fabricated, which is the same
    honesty the page's ``head_cut`` states for the page.
    """
    index = index_with(
        [
            run(opening="u1", closing="a1", first_seq=1, last_seq=4),
            run(
                opening="u2",
                closing="",
                first_seq=5,
                last_seq=8,
                settled=False,
                outcome="open",
                actions=2,
                failed=0,
                worked=6.0,
            ),
        ]
    )
    facts = of.publish_runs(index, first_seq=1, last_seq=None)
    settled, live = facts[0], facts[1]
    assert settled["run_key"] == "a1" and settled["opening_user_id"] == "u1"
    assert settled["action_count"] == 3 and settled["failed_count"] == 1
    assert settled["worked_seconds"] == 12.5 and settled["settled"] is True
    assert live["settled"] is False and live["outcome"] == "open"
    assert live["action_count"] is None and live["worked_seconds"] is None
    # The identity a client can match on is still stated for the live run: the
    # run's LAST row id, which is the client's own fallback key when a run has no
    # elected answer yet.
    assert live["run_key"] == "u2"  # closing_answer_id is absent, so last_id wins
    assert live["closing_answer_id"] is None


def test_publish_runs_starts_one_run_early_so_a_straddle_is_covered() -> None:
    """The window is a superset on purpose: an extra run costs bytes, a missing
    one costs the feature, and a client looks facts up by an id it already holds."""
    index = index_with(
        [
            run(opening=f"u{i}", closing=f"a{i}", first_seq=i * 10, last_seq=i * 10 + 5)
            for i in range(4)
        ]
    )
    facts = of.publish_runs(index, first_seq=20, last_seq=None)
    assert [fact["opening_user_id"] for fact in facts] == ["u1", "u2", "u3"]


def test_page_seq_bounds_is_none_when_the_page_holds_no_run_boundary_row() -> None:
    """No checkpoint row on the page means no run a client could match.

    Emitting some other region's runs would be noise wearing the name of data,
    and the client keeps its own condensation either way.
    """
    index = index_with(
        [run(opening="u1", closing="a1", first_seq=1, last_seq=4)],
        checkpoints=[
            Checkpoint(id="u1", kind="user", turn=1, ts=1.0, seq=1, text="", outcome=None)
        ],
    )
    assert of.page_seq_bounds(index, [message("t9", "tool")], reaches_eof=True) is None
    assert of.page_seq_bounds(index, [message("u1", "user")], reaches_eof=True) == (1, None)


def test_page_seq_bounds_closes_the_window_above_a_page_that_is_not_the_tail() -> None:
    """A mid-journal page must not drag every later run into its answer."""
    runs = [
        run(opening=f"u{i}", closing=f"a{i}", first_seq=i * 10, last_seq=i * 10 + 5)
        for i in range(4)
    ]
    checkpoints = [
        Checkpoint(id=f"u{i}", kind="user", turn=i + 1, ts=1.0, seq=i * 10, text="", outcome=None)
        for i in range(4)
    ]
    index = index_with(runs, checkpoints=checkpoints)
    # The page's newest checkpoint row is u1 (seq 10), so the window runs to the
    # START of the next run — the one the page's last rows may belong to.
    assert of.page_seq_bounds(index, [message("u1", "user")], reaches_eof=False) == (10, 20)


def test_a_run_that_could_not_be_read_whole_says_so() -> None:
    """``complete`` is the honesty twin of ``head_cut``: a lower bound stated."""
    index = index_with([run(opening="u1", closing="a1", first_seq=1, last_seq=4, complete=False)])
    facts = of.publish_runs(index, first_seq=1, last_seq=None)
    assert facts[0]["complete"] is False


def test_the_frame_result_serialises_compactly_enough_to_be_measured() -> None:
    """``served_bytes`` measures the wire's own encoding, not a pretty one."""
    entries = [message("m1", "user")]
    assert of.served_bytes(entries) == len(json.dumps(entries[0], separators=(",", ":")))
