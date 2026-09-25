"""The projection fold is the phone's single source of render semantics —
these tests pin the TUI-parity contract: one line per tool call, diff counts,
streaming rows that update in place, subagent roster aggregation."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.harness.comms import SubagentComms
from local_operator.harness.jobs import AsyncJob, AsyncJobManager
from local_operator.harness.types import (
    AgentEndEvent,
    AgentMessage,
    AgentStartEvent,
    CustomMessage,
    ImageContent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ModelChangeEvent,
    NoticeEvent,
    ReasoningDeltaEvent,
    SubagentEndEvent,
    SubagentProgressEvent,
    SubagentStartEvent,
    TextContent,
    ToolCall,
    ToolCallComposeEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
    TurnEndEvent,
)
from local_operator.mobile.projection import (
    ProjectionFold,
    _diff_counts,
    _image_refs,
    _summarize_args,
    monotonic_from_epoch,
)
from local_operator.mobile.types import (
    PROJECTION_TRANSCRIPT_LIMIT,
    PendingRequest,
    SessionProjection,
    SubagentRow,
    _projection_from_json,
)
from local_operator.session.runtime.registry import SessionRecord
from local_operator.session.session import Session


def make_fold() -> ProjectionFold:
    return ProjectionFold(SessionProjection(session_id="s1", pid=1))


def test_model_change_repaints_the_composer_chip() -> None:
    """A fallback must rename the chip the phone shows, not just the notice."""
    fold = make_fold()
    fold.projection.model_label = "anthropic/claude-opus-4-8"
    fold.fold_event(
        ModelChangeEvent(
            provider="xai",
            model_id="grok-4.6",
            effort="high",
            is_fallback=True,
            reason="quota exhausted",
        )
    )
    assert fold.projection.model_label == "xai/grok-4.6"
    assert fold.projection.effort == "high"


def test_streaming_assistant_row_updates_in_place() -> None:
    fold = make_fold()
    fold.fold_event(AgentStartEvent(generation=1))
    assert fold.projection.streaming is True

    message = Message.assistant()
    fold.fold_event(MessageStartEvent(message=message))
    fold.fold_event(MessageUpdateEvent(message=message, delta="Hel"))
    fold.fold_event(MessageUpdateEvent(message=message, delta="lo"))
    fold.fold_event(
        MessageEndEvent(message=message.model_copy(update={"content": [TextContent(text="Hello")]}))
    )
    fold.fold_event(AgentEndEvent(generation=1))

    rows = [e for e in fold.projection.transcript if e.kind == "assistant"]
    assert len(rows) == 1
    assert rows[0].text == "Hello"
    assert rows[0].final is True
    assert fold.projection.streaming is False


def test_reasoning_streams_onto_one_row_above_the_answer() -> None:
    """Reasoning gets ONE row per model call, ordered above the answer.

    Three properties, each of which a simpler fold would break: the fragments
    accumulate onto a single row rather than one row per token (the phone
    re-renders the whole projection on every repaint); the row sits ABOVE the
    assistant row the same call opened at ``message_start`` (so the answer does
    not materialise above the thinking that produced it); and the row is sealed
    at message end so the NEXT call's phase opens its own.
    """
    fold = make_fold()
    fold.fold_event(AgentStartEvent(generation=1))
    message = Message.assistant()
    fold.fold_event(MessageStartEvent(message=message))
    fold.fold_event(ReasoningDeltaEvent(message_id=message.id, delta="weigh"))
    fold.fold_event(ReasoningDeltaEvent(message_id=message.id, delta="ing"))
    fold.fold_event(MessageUpdateEvent(message=message, delta="the answer"))

    reasoning_rows = [e for e in fold.projection.transcript if e.kind == "reasoning"]
    assert len(reasoning_rows) == 1
    assert reasoning_rows[0].text == "weighing"
    assert reasoning_rows[0].final is False
    kinds = [e.kind for e in fold.projection.transcript]
    assert kinds.index("reasoning") < kinds.index("assistant")

    fold.fold_event(
        MessageEndEvent(
            message=message.model_copy(update={"content": [TextContent(text="the answer")]})
        )
    )
    assert reasoning_rows[0].final is True

    # A second model call in the same turn reasons on a row of its own.
    second = Message.assistant()
    fold.fold_event(MessageStartEvent(message=second))
    fold.fold_event(ReasoningDeltaEvent(message_id=second.id, delta="again"))
    assert [e.text for e in fold.projection.transcript if e.kind == "reasoning"] == [
        "weighing",
        "again",
    ]


def test_reasoning_row_keeps_the_newest_words_and_never_becomes_the_answer() -> None:
    """Bounded to the TAIL, and it never touches the assistant row's text.

    The bound is a wire cost, not taste: this row rides the whole projection on
    every repaint, so an unbounded thinking phase would re-send its entire
    thought per frame. The tail rather than the head because reasoning streams --
    what a reader wants is what the model is thinking NOW. And the assistant row
    must stay exactly the answer: folding the private reasoning into it is the
    transcript corruption ``ReasoningDeltaEvent`` forbids.
    """
    from local_operator.mobile.projection import REASONING_PREVIEW_CHARS

    fold = make_fold()
    fold.fold_event(AgentStartEvent(generation=1))
    message = Message.assistant()
    fold.fold_event(MessageStartEvent(message=message))
    fold.fold_event(ReasoningDeltaEvent(message_id=message.id, delta="HEAD" + "x" * 2000))
    fold.fold_event(MessageUpdateEvent(message=message, delta="the answer"))
    fold.fold_event(
        MessageEndEvent(
            message=message.model_copy(update={"content": [TextContent(text="the answer")]})
        )
    )

    reasoning = next(e for e in fold.projection.transcript if e.kind == "reasoning")
    assert len(reasoning.text) == REASONING_PREVIEW_CHARS
    assert reasoning.text.startswith("…")
    assert "HEAD" not in reasoning.text
    # Not a transport truncation: the bound is this row's own, and there is no
    # fuller row to page. ``text_complete`` keeps its documented meaning (a
    # pageable PREFIX), so it stays true.
    assert reasoning.text_complete is True
    assistant = next(e for e in fold.projection.transcript if e.kind == "assistant")
    assert assistant.text == "the answer"


def test_tool_row_lifecycle_one_line_with_diff_counts() -> None:
    fold = make_fold()
    fold.fold_event(
        ToolExecutionStartEvent(
            tool_call_id="t1", tool_name="write", args={"path": "/a/b/c.py", "content": "x"}
        )
    )
    row = fold.projection.transcript[-1]
    assert row.kind == "tool"
    assert row.tool_state == "running"
    assert row.summary == "/a/b/c.py"

    fold.fold_event(
        ToolExecutionEndEvent(
            tool_call_id="t1",
            tool_name="write",
            duration_s=1.25,
            result=ToolResult(
                tool_call_id="t1",
                content=[TextContent(text="wrote 10 lines")],
                details={"added": 8, "removed": 2, "diff": "@@"},
                duration_s=1.25,
            ),
        )
    )
    assert row.tool_state == "done"
    assert (row.diff_added, row.diff_removed) == (8, 2)
    assert row.details["output"] == "wrote 10 lines"
    assert row.details["diff"] == "@@"
    assert row.details["args"]["path"] == "/a/b/c.py"
    assert row.elapsed_s == 1.2


def test_failed_tool_row_carries_the_error() -> None:
    fold = make_fold()
    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="t2", tool_name="bash", args={"command": "false"})
    )
    fold.fold_event(
        ToolExecutionEndEvent(
            tool_call_id="t2",
            tool_name="bash",
            result=ToolResult(
                tool_call_id="t2", content=[TextContent(text="exit 1: boom")], is_error=True
            ),
        )
    )
    row = fold.projection.transcript[-1]
    assert row.tool_state == "failed"
    assert "boom" in row.error


def test_subagent_roster_running_first_then_settled() -> None:
    fold = make_fold()
    fold.fold_event(SubagentStartEvent(job_id="j1", label="first"))
    fold.fold_event(SubagentStartEvent(job_id="j2", label="second"))
    fold.fold_event(SubagentProgressEvent(job_id="j1", label="first", progress="reading files"))
    fold.fold_event(
        SubagentEndEvent(job_id="j2", label="second", status="completed", result_text="done")
    )

    rows = fold.projection.subagents
    assert rows[0].job_id == "j1"  # running sorts first
    assert rows[0].progress == "reading files"
    assert rows[1].status == "completed"
    assert rows[1].result_text == "done"


def test_legacy_subagent_projection_rebuilds_new_detail_collections() -> None:
    """A rolling upgrade can pair an older registrant with the current daemon.

    The older row lacks every recursive-detail collection added by this PR;
    rebuilding at the shared socket seam must produce the complete browser
    contract rather than forwarding undefined values into React.
    """
    record = SessionRecord(
        pid=42,
        kind="tui",
        session_id="s1",
        conversation_name="legacy",
        cwd="/tmp",
        model_label="test/model",
        control_port=1,
        control_key="secret",
    )
    projection = _projection_from_json(
        {
            "session_id": "s1",
            "pid": 0,
            "subagents": [{"job_id": "legacy", "label": "legacy child"}],
        },
        record,
    )

    row = projection.subagents[0]
    assert projection.pid == 42
    assert row.ancestors == []
    assert row.ancestor_ids == []
    assert row.child_ids == []
    assert row.peer_ids == []
    assert row.transcript == []
    assert row.todos == []
    assert projection.to_json()["subagents"][0]["child_ids"] == []


def test_subagent_details_seed_nested_descendants_for_recursive_navigation() -> None:
    """Nested jobs never emit lifecycle events through the root fold.

    The shared registry must still project a complete root -> child ->
    grandchild graph so every advertised child id resolves on the phone, and a
    later refresh must keep enriching that same nested record.
    """

    class Jobs:
        def __init__(self) -> None:
            self.rows = {
                "parent": SimpleNamespace(status="running", agent_role="coder", latest_details={}),
                "child": SimpleNamespace(
                    status="completed", agent_role="reviewer", latest_details={}
                ),
                "grandchild": SimpleNamespace(
                    status="running", agent_role="scout", latest_details={"progress": "reading"}
                ),
            }

        def get(self, job_id: str):
            return self.rows.get(job_id)

    session = SimpleNamespace(jobs=Jobs())
    comms = SubagentComms(cast(Session, cast(Any, session)))
    comms.record_launch("parent", "parent", prompt="plan")
    comms.record_launch("child", "child", parent_job_id="parent", prompt="build")
    comms.record_launch("grandchild", "grandchild", parent_job_id="child", prompt="inspect")
    fold = make_fold()
    # Only the direct child reaches the root event stream.
    fold.fold_event(SubagentStartEvent(job_id="parent", label="parent"))

    fold.set_subagent_details(comms)
    by_id = {row.job_id: row for row in fold.projection.subagents}
    assert set(by_id) == {"parent", "child", "grandchild"}
    assert by_id["parent"].child_ids == ["child"]
    assert by_id["child"].child_ids == ["grandchild"]
    assert by_id["grandchild"].parent_job_id == "child"
    assert by_id["grandchild"].ancestors == ["parent", "child"]
    assert by_id["grandchild"].ancestor_ids == ["parent", "child"]
    assert by_id["grandchild"].activity == "reading"

    session.jobs.rows["grandchild"].latest_details = {"progress": "summarizing"}
    fold.set_subagent_details(comms)
    refreshed = {row.job_id: row for row in fold.projection.subagents}
    assert refreshed["grandchild"].activity == "summarizing"


def test_subagent_compaction_reuses_only_identical_sources_and_prunes_removed_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Memoize the O(L) normalization without changing any projection values."""
    import local_operator.mobile.projection as projection_module

    job = SimpleNamespace(
        status="running",
        agent_role="coder",
        model_label="test/model",
        latest_details={},
        result_text=None,
        error_text=None,
    )
    session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: job))
    comms = SubagentComms(cast(Session, cast(Any, session)))
    prompt = "  first line  \n\n second line\t"
    comms.record_launch("child", "child", prompt=prompt)
    fold = make_fold()

    calls = {"flat": 0, "multiline": 0}
    compact = projection_module._compact
    compact_multiline = projection_module._compact_multiline

    def count_flat(text: str, limit: int) -> str:
        calls["flat"] += 1
        return compact(text, limit)

    def count_multiline(text: str, limit: int) -> str:
        calls["multiline"] += 1
        return compact_multiline(text, limit)

    monkeypatch.setattr(projection_module, "_compact", count_flat)
    monkeypatch.setattr(projection_module, "_compact_multiline", count_multiline)

    fold.set_subagent_details(comms)
    first = fold.projection.subagents[0]
    prompt_value = first.prompt
    assert first.result_text == first.error_text == ""
    assert calls == {"flat": 1, "multiline": 2}

    fold.set_subagent_details(comms)
    assert calls == {"flat": 1, "multiline": 2}
    assert fold.projection.subagents[0].prompt == prompt_value

    # A distinct but equal string is a cache miss: equality/hash work must not be
    # substituted for the intended identity check on potentially huge sources.
    changed_prompt = ("!" + prompt)[1:]
    assert changed_prompt == prompt and changed_prompt is not prompt
    comms._records["child"].prompt = changed_prompt
    # The prompt and result deliberately hold the same source object: their
    # different normalizers must still produce different, uncached semantics.
    result = prompt
    job.status = "completed"
    job.result_text = result
    job.error_text = "provider failed\n  at call site"
    comms.record_outcome("child", "completed", result_text=result, error_text=job.error_text)
    fold.set_subagent_details(comms)
    settled = fold.projection.subagents[0]
    assert prompt_value == "first line second line"
    assert settled.prompt == prompt_value
    assert settled.result_text == "first line\n\nsecond line"
    assert settled.error_text == "provider failed\nat call site"
    assert calls == {"flat": 2, "multiline": 4}

    fold.set_subagent_details(comms)
    assert calls == {"flat": 2, "multiline": 4}

    # Whitespace/empty normalizer inputs retain their exact source distinctions;
    # a later source object, even if the compacted output is the same, is new work.
    whitespace = " \n  "
    comms._records["child"].prompt = whitespace
    job.result_text = whitespace
    job.error_text = ""
    comms.record_outcome("child", "completed", result_text=whitespace, error_text="")
    fold.set_subagent_details(comms)
    whitespace_row = fold.projection.subagents[0]
    assert whitespace_row.prompt == ""
    assert whitespace_row.result_text == ""
    assert whitespace_row.error_text == ""
    assert calls == {"flat": 3, "multiline": 6}

    # A node can outlive its lifecycle reader in compatibility facades. Clear
    # only terminal payload slots (the row's existing values remain untouched).
    real_roster_pass = comms.roster_pass
    current_pass = real_roster_pass()

    class MissingLifecyclePass:
        def roster(self) -> list[Any]:
            return []

        def lifecycles(self) -> dict[str, Any]:
            return {}

        def nodes(self) -> list[Any]:
            return current_pass.nodes()

        def job(self, job_id: str) -> Any:
            return current_pass.job(job_id)

    monkeypatch.setattr(comms, "roster_pass", lambda: MissingLifecyclePass())
    fold.set_subagent_details(comms)
    assert ("child", "prompt") in fold._subagent_compact_cache
    assert ("child", "result_text") not in fold._subagent_compact_cache
    assert ("child", "error_text") not in fold._subagent_compact_cache

    # When lifecycle data becomes available again, terminal outputs equal the
    # uncached normalizers and repopulate their slots.
    monkeypatch.setattr(comms, "roster_pass", real_roster_pass)
    fold.set_subagent_details(comms)
    assert fold.projection.subagents[0].result_text == ""
    assert fold.projection.subagents[0].error_text == ""
    assert calls == {"flat": 3, "multiline": 8}

    # Removing the registry record releases all field refs on the next pass.
    comms._records.pop("child")
    fold.set_subagent_details(comms)
    assert fold._subagent_compact_cache == {}
    assert len(fold.projection.subagents) == 1  # the row's legacy removal is unchanged


def _roster_with_todos(count: int) -> tuple[ProjectionFold, SubagentComms]:
    """A fold over ``count`` children: long prompts, one settled child, todos.

    The shape the frame cap actually meets on a deep roster — previews worth
    re-capping, and roster todos, which are the tier's real per-repaint work.
    """
    jobs = SimpleNamespace(rows={})

    class Jobs:
        def get(self, job_id: str) -> Any:
            return jobs.rows.get(job_id)

    comms = SubagentComms(cast(Session, cast(Any, SimpleNamespace(jobs=Jobs()))))
    for index in range(count):
        job_id = f"child-{index}"
        comms.record_launch(job_id, job_id, prompt=("preview line 工作项\n" * 20) + f"#{index}")
        jobs.rows[job_id] = SimpleNamespace(status="running", agent_role="coder", latest_details={})
    jobs.rows["child-1"].status = "completed"
    jobs.rows["child-1"].result_text = "settled result\nsecond line"
    comms.record_outcome("child-1", "completed", result_text=jobs.rows["child-1"].result_text)
    fold = make_fold()
    fold.set_subagent_details(comms)
    for index in range(count):
        fold.set_subagent_hydrated_details(
            f"child-{index}",
            [],
            [
                {
                    "name": "Verification",
                    "items": [
                        {
                            "text": f"todo {item} of child {index} " + "detail " * 90,
                            "status": "pending",
                            "reason": "reason " + "x" * 200,
                        }
                        for item in range(6)
                    ],
                }
            ],
        )
    return fold, comms


def _counted_compaction(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    """Count both normalizers BEFORE any memo is filled — see the test below."""
    import local_operator.mobile.projection as projection_module

    calls = {"flat": 0, "multiline": 0}
    compact = projection_module._compact
    compact_multiline = projection_module._compact_multiline

    def count_flat(text: str, limit: int) -> str:
        calls["flat"] += 1
        return compact(text, limit)

    def count_multiline(text: str, limit: int) -> str:
        calls["multiline"] += 1
        return compact_multiline(text, limit)

    monkeypatch.setattr(projection_module, "_compact", count_flat)
    monkeypatch.setattr(projection_module, "_compact_multiline", count_multiline)
    return calls


def _frame_row(data: dict[str, Any], job_id: str) -> dict[str, Any]:
    return next(row for row in data["subagents"] if row["job_id"] == job_id)


def _over_cap(projection: SessionProjection) -> int:
    """A cap the frame exceeds by roughly half, so the text tiers are reached.

    Self-scaling rather than a literal: where the tiers land is a property of
    the fixture's own size, and a literal would quietly stop exercising them the
    day that fixture changes.
    """
    from local_operator.mobile.projection import cap_projection_frame

    uncapped, degraded = cap_projection_frame(projection, cap_bytes=1_000_000_000)
    assert degraded is False
    return max(1, len(json.dumps(uncapped, sort_keys=True, ensure_ascii=False)) // 2)


def test_frame_cap_recaps_only_changed_sources_and_repeats_byte_for_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An over-cap repaint re-caps what CHANGED, not what the roster holds.

    Tiers 1 and 1c run on every repaint that is over the soft cap (~30x/s while
    a turn streams), and ``_compact`` walks the whole source even when the cap
    keeps a fraction of it, so an unchanged roster used to re-derive every
    preview and every roster todo on every frame — measured at N=256 with a
    hydrated roster as 13,824 normalizer calls scanning 3.56 MB per frame.

    The memo may only skip work it would reproduce exactly, so the frames are
    compared as WHOLE dicts, not field by field.
    """
    from local_operator.mobile.projection import (
        FRAME_CAP_PROMPT_CHARS,
        _compact,
        cap_projection_frame,
    )

    # Counters go in FIRST: an entry keys on the normalizer OBJECT, so a swap
    # after a fold or a cap has run invalidates every entry it wrote by design.
    calls = _counted_compaction(monkeypatch)
    fold, comms = _roster_with_todos(8)
    projection = fold.projection

    # Cold, every source once. The fold compacts one prompt and two outcome
    # fields per child as the fixture builds; the cap then re-caps a prompt and
    # two outcome fields per row, plus three todo items x (text, reason).
    cap = _over_cap(projection)
    fold_flat, fold_multiline = 8, 8 * 2
    cap_flat, cap_multiline = 8 + 8 * 12, 8 * 2
    cold = {"flat": fold_flat + cap_flat, "multiline": fold_multiline + cap_multiline}

    first, degraded = cap_projection_frame(projection, cap_bytes=cap)
    assert degraded is True
    assert calls == cold

    second, degraded_again = cap_projection_frame(projection, cap_bytes=cap)
    assert calls == cold, "nothing changed, nothing re-capped"
    assert second == first
    assert degraded_again is degraded

    # ONE record's prompt replaced by a NEW object of a different length: two
    # sources to re-derive — one in the fold's own memo, one in the cap's — and
    # the published preview follows it.
    rewritten = ("rewritten prompt 工作项\n" * 12).strip()
    comms._records["child-2"].prompt = rewritten
    fold.set_subagent_details(comms)
    third, _ = cap_projection_frame(projection, cap_bytes=cap)
    assert calls == {"flat": cold["flat"] + 2, "multiline": cold["multiline"]}
    assert _frame_row(third, "child-2")["prompt"] == _compact(rewritten, FRAME_CAP_PROMPT_CHARS)
    assert _frame_row(third, "child-2")["prompt"] != _frame_row(second, "child-2")["prompt"]

    # Re-hydrating ONE row's todos replaces its item objects, so that row's
    # items are new work (6 items x text/reason) and the other seven rows are
    # not: the cost tracks the change, not the roster. The texts stay long
    # enough that the cap still needs tier 1c to fit, or the row's items would
    # simply not be reached.
    fold.set_subagent_hydrated_details(
        "child-3",
        [],
        [
            {
                "name": "Verification",
                "items": [
                    {
                        "text": f"rehydrated {item} " + "detail " * 90,
                        "status": "pending",
                        "reason": "r" * 200,
                    }
                    for item in range(6)
                ],
            }
        ],
    )
    before = dict(calls)
    fourth, _ = cap_projection_frame(projection, cap_bytes=cap)
    assert calls == {"flat": before["flat"] + 12, "multiline": before["multiline"]}
    assert _frame_row(fourth, "child-3")["todos"][0]["items"][0]["text"].startswith("rehydrated 0 ")

    # A gate that skipped the normalizer whenever the value was already SHORT
    # would republish this raw double space: the cap still normalises it, and
    # the reuse path must not turn that into a second frame's work either.
    projection.subagents[0].prompt = "double  space\ttext"
    fifth, _ = cap_projection_frame(projection, cap_bytes=cap)
    short_row = _frame_row(fifth, projection.subagents[0].job_id)
    assert short_row["prompt"] == "double space text"
    before = dict(calls)
    cap_projection_frame(projection, cap_bytes=cap)
    assert calls == before


def test_frame_cap_memo_is_slotted_per_row_and_releases_departed_rows() -> None:
    """Bounded by the roster the frame publishes: one slot per row, and freed."""
    from local_operator.mobile.projection import cap_projection_frame

    fold, _comms = _roster_with_todos(4)
    projection = fold.projection
    cap = _over_cap(projection)
    cap_projection_frame(projection, cap_bytes=cap)
    memo = projection._frame_cap_memo

    assert set(memo) == {row.job_id for row in projection.subagents}
    # Per row: three text fields plus text/reason for each of its six items.
    assert {len(slot) for slot in memo.values()} == {3 + 2 * 6}

    # A projection that publishes a SHORTER roster than the last capped frame
    # releases the departed row's slot, which is what keeps the cache bounded
    # by the roster rather than by every child it has ever carried.
    departed = projection.subagents.pop()
    cap_projection_frame(projection, cap_bytes=cap)
    assert departed.job_id not in memo
    assert set(memo) == {row.job_id for row in projection.subagents}


def _memo_source_bytes(memo: dict[str, dict[Any, Any]], live: set[str]) -> int:
    """Bytes of source text the memo pins for rows the roster no longer carries.

    Superseded SOURCES are the whole of what the memo holds — the entry keeps the
    row's own preview/result/reason objects alive, not copies of any frame — so
    this is the quantity review round 1 and QA round 1 both measured on their own
    fixtures, quoted here in the bytes they measured (1,694,671 B for a 32 -> 3
    row shrink, 18,409,240 B at 256 x 25).

    ENCODED rather than ``len(str)``, and that is not cosmetic on this fixture:
    ``len`` is CHARACTERS, every prompt here ends in CJK (``工作项``), and
    measured with the memo filled at 32 rows the same 480 pinned sources are
    **175,284 characters against 179,124 bytes** — 32 of the 480 carrying the
    non-ASCII preview. The name, this docstring and the figures cited above are
    all bytes, so the sum has to be too (review round 2 on this PR, R2-3).

    Both callers assert ``== 0``, so no caller depends on the unit — only on the
    release being total.
    """
    return sum(
        len(entry[0].encode("utf-8"))
        for row_id, row_memo in memo.items()
        if row_id not in live
        for entry in row_memo.values()
    )


def test_frame_cap_memo_is_released_by_a_frame_that_comes_in_under_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1-1: the release keys on the PUBLISHED ROSTER, never on a tier having run.

    The frame that most needs the release is the one that just shrank a roster —
    and that frame is the SMALLER one, so it can land under the cap and return
    before a tier runs. That is exactly where the release used to live, which is
    why the slots of every departed row stayed pinned: 1,694,671 B measured on a
    32 -> 3 row shrink with the frame under the cap, and 18,409,240 B at 256 rows
    x 25 todos (review round 1 R1-1; QA round 1 section 5, byte figures).
    """
    from local_operator.mobile.projection import (
        PROJECTION_FRAME_SOFT_CAP_BYTES,
        cap_projection_frame,
    )

    fold, _comms = _roster_with_todos(32)
    projection = fold.projection
    memo = projection._frame_cap_memo
    calls = _counted_compaction(monkeypatch)
    cap_projection_frame(projection, cap_bytes=_over_cap(projection))
    assert len(memo) == 32
    live = {row.job_id for row in projection.subagents}
    assert _memo_source_bytes(memo, live) == 0

    while len(projection.subagents) > 3:
        projection.subagents.pop()
    shrunk_live = {row.job_id for row in projection.subagents}
    before = dict(calls)

    frame, degraded = cap_projection_frame(projection, cap_bytes=PROJECTION_FRAME_SOFT_CAP_BYTES)

    # The cell has to be the UNDER-cap one or it would not reproduce R1-1 at all:
    # no tier ran on this push, which is precisely why the old release site —
    # inside tier 1 — was never reached for it.
    assert degraded is False
    assert calls == before, "no tier ran on the shrunken frame"
    assert len(frame["subagents"]) == 3
    assert set(memo) == shrunk_live
    assert _memo_source_bytes(memo, shrunk_live) == 0
    assert sum(len(row_memo) for row_memo in memo.values()) < 32 * (3 + 2 * 6)


def test_the_memo_source_figure_counts_bytes_and_not_characters() -> None:
    """R2-3: the name, the docstring and the figures cited from it are all bytes.

    ``len(str)`` is CHARACTERS, and this fixture's prompts carry CJK, so the two
    units genuinely diverge: measured with the memo filled at 32 rows, the same
    480 pinned sources are 175,284 characters against 179,124 bytes, 32 of the
    480 being the non-ASCII preview. A helper that summed characters while
    reporting a byte figure would be quoting a quantity it never measured.
    """
    source = "工作项" * 1000
    entry = (source, 10, lambda text, limit: text, source)
    memo: dict[str, dict[Any, Any]] = {"departed": {"prompt": entry}}

    assert len(source.encode("utf-8")) > len(source)
    assert _memo_source_bytes(memo, set()) == len(source.encode("utf-8"))


def test_frame_cap_memo_is_reduced_to_the_shape_a_row_still_publishes() -> None:
    """A LIVE row's superseded slots are released too, not only a departed row's.

    Review round 1 measured a row grown to 40 items and then shrunk to 1 keeping
    all 89 of its slots (``3 + 2 x 40``), because entries are replaced in place
    and nothing dropped the ones the row had shed. What the memo holds per row is
    now the shape that row publishes — its three previews plus two slots per todo
    item it still carries — and nothing else.
    """
    from local_operator.mobile.projection import cap_projection_frame

    fold, _comms = _roster_with_todos(2)
    projection = fold.projection
    memo = projection._frame_cap_memo
    cap = _over_cap(projection)
    cap_projection_frame(projection, cap_bytes=cap)
    assert len(memo["child-0"]) == 3 + 2 * 6

    def hydrate(items: int) -> None:
        fold.set_subagent_hydrated_details(
            "child-0",
            [],
            [
                {
                    "name": "Verification",
                    "items": [
                        {
                            "text": f"grown {item} " + "detail " * 90,
                            "status": "pending",
                            "reason": "reason " + "x" * 200,
                        }
                        for item in range(items)
                    ],
                }
            ],
        )

    hydrate(40)
    cap_projection_frame(projection, cap_bytes=cap)
    assert len(memo["child-0"]) == 3 + 2 * 40

    hydrate(1)
    cap_projection_frame(projection, cap_bytes=cap)
    assert len(memo["child-0"]) == 3 + 2 * 1
    # The other row is untouched: the release is per row, not a reset.
    assert len(memo["child-1"]) == 3 + 2 * 6


def test_a_phase_reshape_costs_the_frame_a_sweep_and_not_a_grown_memo() -> None:
    """R2-2: the per-row bound is ``previews + 4 x``, and the envelope is ATTAINED.

    The reconcile's sweep test is a ``len()`` comparison taken BEFORE the tiers,
    against the shape the row is publishing. A row whose item COUNT is unchanged
    and whose ``(phase, item)`` POSITIONS moved passes it and then adds the new
    positions' keys, so the honest bound after any frame is ``previews + 4 x``
    rather than the ``+ 2 x`` the docstring used to claim (review round 2 on this
    PR, R2-2). Pinned here on this file's fixture, one row of six items, where
    ``previews + 2 x`` is 15 and ``previews + 4 x`` is 27:

    * ``[6] -> [6]`` — nothing moved, so nothing new: ``15 -> 15``.
    * ``[6] -> [3, 3]`` — three items moved: ``15 -> 21``.
    * ``[6] -> [0, 6]`` — no old position survives, because a leading empty phase
      is published rather than merged away, so the bound is reached: ``15 -> 27``.

    Each peak is one frame wide — the NEXT frame sweeps back to ``15`` — which is
    why the wider bound is documented rather than enforced in the code.
    """
    from local_operator.mobile.projection import cap_projection_frame

    def phases(shape: list[int]) -> list[dict[str, Any]]:
        return [
            {
                "name": f"Phase {index}",
                "items": [
                    {
                        "text": f"todo {index}.{item} " + "detail " * 90,
                        "status": "pending",
                        "reason": "reason " + "x" * 200,
                    }
                    for item in range(count)
                ],
            }
            for index, count in enumerate(shape)
        ]

    def reshape_frame(before: list[int], after: list[int]) -> tuple[int, int, int]:
        """``(slots before the reshaping frame, its peak, the next frame)``.

        A FRESH fixture per pair, because the peak depends on the positions the
        row was holding when the frame began — the sweep the previous frame left
        behind is what decides how many of this frame's keys are new.
        """
        fold, _comms = _roster_with_todos(2)
        projection = fold.projection
        memo = projection._frame_cap_memo
        cap = _over_cap(projection)
        cap_projection_frame(projection, cap_bytes=cap)
        fold.set_subagent_hydrated_details("child-0", [], phases(before))
        cap_projection_frame(projection, cap_bytes=cap)
        before_slots = len(memo["child-0"])
        fold.set_subagent_hydrated_details("child-0", [], phases(after))
        cap_projection_frame(projection, cap_bytes=cap)
        peak = len(memo["child-0"])
        cap_projection_frame(projection, cap_bytes=cap)
        # The sibling is never reshaped, so it stays on the settled shape.
        assert len(memo["child-1"]) == 3 + 2 * 6
        return before_slots, peak, len(memo["child-0"])

    two_x = 3 + 2 * 6
    # Nothing moved, so no key is added: the frame is a pure hit.
    assert reshape_frame([6], [6]) == (two_x, two_x, two_x)
    # Three items move to positions the row had nothing at. The count is
    # unchanged, so the frame's own sweep test passes and the keys land on top.
    assert reshape_frame([6], [3, 3]) == (two_x, two_x + 2 * 3, two_x)
    # No old position survives — a leading empty phase is published rather than
    # merged away — so this is the bound itself, not a step toward it.
    assert reshape_frame([6], [0, 6]) == (two_x, 3 + 4 * 6, two_x)


def test_nested_subagent_completion_refreshes_selected_detail() -> None:
    """A nested row has no root lifecycle event to settle its phone detail."""

    job = SimpleNamespace(
        status="running",
        agent_role="coder",
        model_label="test/running",
        latest_details={"progress": "working"},
        result_text=None,
        error_text=None,
    )
    session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: job))
    comms = SubagentComms(cast(Session, cast(Any, session)))
    comms.record_launch("parent", "parent")
    comms.record_launch("nested", "nested", parent_job_id="parent")
    fold = make_fold()

    fold.set_subagent_details(comms)
    selected = {row.job_id: row for row in fold.projection.subagents}["nested"]
    assert (selected.status, selected.progress, selected.activity) == (
        "running",
        "working",
        "working",
    )

    job.status = "completed"
    job.model_label = "test/completed"
    job.latest_details = {}
    job.result_text = "finished result"
    fold.set_subagent_details(comms)

    selected = {row["job_id"]: row for row in fold.projection.to_json()["subagents"]}["nested"]
    assert selected["status"] == "completed"
    assert selected["model_label"] == "test/completed"
    assert selected["result_text"] == "finished result"
    assert selected["error_text"] == ""
    assert selected["progress"] == ""
    assert selected["activity"] == ""


@pytest.mark.parametrize(
    ("status", "result_text", "error_text"),
    [
        ("completed", "durable completed result", None),
        ("failed", None, "provider failed; retry from the parent"),
    ],
)
def test_swept_nested_outcome_survives_fresh_projection_and_reconnect(
    status: str, result_text: str | None, error_text: str | None
) -> None:
    """The comms record is the only lifecycle source after manager retention."""

    class Jobs:
        def __init__(self) -> None:
            self.rows = {
                "nested": SimpleNamespace(
                    status="running",
                    agent_role="coder",
                    model_label="test/model",
                    latest_details={"progress": "stale progress"},
                    result_text=None,
                    error_text=None,
                )
            }

        def get(self, job_id: str):
            return self.rows.get(job_id)

    session = SimpleNamespace(jobs=Jobs())
    comms = SubagentComms(cast(Session, cast(Any, session)))
    comms.record_launch("parent", "parent")
    comms.record_launch("nested", "nested", parent_job_id="parent")
    fold = make_fold()
    fold.set_subagent_details(comms)
    assert {row.job_id: row for row in fold.projection.subagents}["nested"].activity == (
        "stale progress"
    )

    comms.record_outcome("nested", status, error_text=error_text, result_text=result_text)
    del session.jobs.rows["nested"]

    # Refreshing the selected projection must clear stale live activity, and a
    # brand-new fold models the first SSE snapshot after a reconnect.
    fold.set_subagent_details(comms)
    reconnect = make_fold()
    reconnect.set_subagent_details(comms)
    for projection in (fold.projection, reconnect.projection):
        selected = {row.job_id: row for row in projection.subagents}["nested"]
        assert selected.status == status
        assert selected.result_text == (result_text or "")
        assert selected.error_text == (error_text or "")
        assert selected.progress == ""
        assert selected.activity == ""


@pytest.mark.parametrize(
    "parents",
    [
        {"a": "a"},
        {"a": "b", "b": "a"},
    ],
)
def test_subagent_metadata_projection_tolerates_legacy_parent_cycles(
    parents,
) -> None:  # noqa: ANN001
    session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: None))
    comms = SubagentComms(cast(Session, cast(Any, session)))
    for job_id, parent_id in parents.items():
        comms.record_launch(job_id, job_id, parent_job_id=parent_id)
    fold = make_fold()
    fold.set_subagent_details(comms)
    assert {row.job_id for row in fold.projection.subagents} == set(parents)


def test_subagent_metadata_projection_never_constructs_transcript(monkeypatch) -> None:
    """The ordinary event path must remain memory-only regardless of child count."""
    session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: None))
    comms = SubagentComms(cast(Session, cast(Any, session)))
    for index in range(75):
        comms.record_launch(f"child-{index}", f"child {index}")

    class ForbiddenTranscript:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            raise AssertionError("metadata projection opened a child transcript")

    monkeypatch.setattr("local_operator.session.transcript.Transcript", ForbiddenTranscript)
    fold = make_fold()
    for _ in range(100):
        fold.set_subagent_details(comms)
    assert len(fold.projection.subagents) == 75


def test_hydrated_subagent_details_never_place_transcript_on_the_wire() -> None:
    """A subagent's transcript is fetched lazily, never carried in the fold.

    Embedding even a tail-capped child transcript per subagent pushed the
    full-repaint projection past the daemon's 1 MB control-frame limit, so every
    push was dropped as oversized and the phone fell back to the stale durable
    fold. ``set_subagent_hydrated_details`` must therefore land todos (small,
    needed for the live working line) while leaving ``row.transcript`` empty; the
    transcript is served on demand from the child-history endpoint instead.
    """
    from local_operator.mobile.types import TranscriptEntry

    session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: None))
    comms = SubagentComms(cast(Session, cast(Any, session)))
    comms.record_launch("child", "child")
    fold = make_fold()
    fold.set_subagent_details(comms)

    heavy = [
        TranscriptEntry(id=f"row-{i}", kind="assistant", text="x" * 4096)
        for i in range(PROJECTION_TRANSCRIPT_LIMIT * 2)
    ]
    assert fold.set_subagent_hydrated_details(
        "child", heavy, [{"text": "verify", "status": "pending"}]
    )
    row = fold._subagents["child"]
    assert row.transcript == []
    assert row.todos and row.todos[0].items[0].text == "verify"
    # The serialized wire frame must contain no subagent transcript entries.
    wire = fold.projection.to_json()
    assert all(sub["transcript"] == [] for sub in wire["subagents"])


def test_live_fold_bounds_subagent_prompt_and_outcome_on_the_wire() -> None:
    """Uncapped prompt/result text re-wedges the frame the transcript cap saved.

    The list projection is a full repaint pushed ~30x/s and every subagent row
    rides in it, so uncapped prompt/result/error text scales the frame with
    roster depth: a power-user session at 80+ subagents put hundreds of KB of
    prompt text into one frame, back toward the 1 MB control-frame cap. The row
    only needs a preview; the full text is retained by the daemon and served
    through getSubagentDetail. This pins the wire bounds so the regression cannot
    silently return.
    """
    from local_operator.mobile.projection import (
        SUBAGENT_OUTCOME_CHARS,
        SUBAGENT_PROMPT_PREVIEW_CHARS,
    )

    session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: None))
    comms = SubagentComms(cast(Session, cast(Any, session)))
    comms.record_launch("child", "child", prompt="P" * 50_000)
    comms.record_outcome("child", "completed", result_text="R" * 50_000)
    fold = make_fold()
    fold.set_subagent_details(comms)

    wire_row = fold.projection.to_json()["subagents"][0]
    assert len(wire_row["prompt"]) <= SUBAGENT_PROMPT_PREVIEW_CHARS
    assert len(wire_row["result_text"]) <= SUBAGENT_OUTCOME_CHARS
    # An empty error field stays empty (a cap must not manufacture a placeholder).
    assert wire_row["error_text"] == ""


#: Production-shaped job ids: 12 hex chars, as ``uuid4().hex[:12]`` builds them.
#: The LENGTH is load-bearing — ``peer_ids`` costs ~16 bytes per sibling id on
#: the wire, so a 256-wide group of this shape crosses the 1 MiB line limit
#: while the same group with ``child-0``-style ids stays ~100 KB under it.
_SIBLING_ID_HEX = 12


def _flat_roster_fold(width: int) -> ProjectionFold:
    """A flat sibling group folded by the REAL roster path.

    Every child sits under one parent (``parent_job_id: None``), which is the
    production shape — 256 parallel eval children — and the one that makes
    ``peer_ids`` O(n^2): each row lists every sibling but itself.
    """
    session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: None))
    comms = SubagentComms(cast(Session, cast(Any, session)))
    for index in range(width):
        comms.record_launch(
            f"{index:0{_SIBLING_ID_HEX}x}",
            f"osworld-eval-{index}",
            prompt="Analyse the episode and click the correct element " * 4,
        )
    fold = make_fold()
    fold.set_subagent_details(comms)
    for row in fold.projection.subagents:
        row.result_text = "observed the agent fail to click the correct element " * 4
    return fold


def _projection_line(data: dict[str, Any]) -> int:
    """Exactly what the socket writes for a projection frame."""
    return len(json.dumps({"op": "projection", "data": data}).encode()) + 1


@pytest.mark.parametrize("width", [1, 50, 250, 1000])
def test_a_wide_sibling_group_cannot_make_the_projection_unreadable(width: int) -> None:
    """The cap's CONTRACT, at every roster width that matters.

    ``cap_projection_frame`` is a soft cap: it degrades optional tiers and may
    still return an over-limit payload, which the socket then writes raw. The
    welcome projection is therefore the one frame family that could make a
    session unopenable by ANY viewer — the client's ``readline`` raises over the
    same 1 MiB, its pump dies, and every retry dies the same way. This pins the
    contract structurally (the frame the send path would report as oversize is
    None) rather than with a byte ceiling, so it stays true whatever the row
    payload happens to weigh.
    """
    from local_operator.mobile.projection import (
        FRAME_CAP_DERIVED_ROSTER_FIELDS,
        cap_projection_frame,
    )
    from local_operator.session.frontend_state import oversized_frame_report
    from local_operator.session.runtime.server import _MAX_LINE_BYTES

    fold = _flat_roster_fold(width)
    data, _degraded = cap_projection_frame(fold.projection)

    assert oversized_frame_report({"op": "projection", "data": data}, _MAX_LINE_BYTES) is None

    rows = data["subagents"]
    assert rows
    # ``parent_job_id`` is what every shed field can be REBUILT from, so it must
    # survive on every row no matter which tier fired (the canonical side
    # reasons the same way, and the phone normalises an absent list as empty).
    assert all("parent_job_id" in row for row in rows)
    # The derived graph is shed ALL OR NOTHING per frame: a half-shed roster
    # would leave a reader unable to tell "no peers" from "not carried".
    for field in FRAME_CAP_DERIVED_ROSTER_FIELDS:
        carried = {bool(row.get(field)) for row in rows}
        assert len(carried) == 1, f"{field} was shed for only some rows"


def test_the_derived_roster_graph_is_what_fits_a_cliff_width_roster() -> None:
    """256 flat siblings: the frame production could not send, and what fixed it.

    The tiers that existed before this shed only TEXT, and none of them can
    touch the field that actually grows this frame: ``peer_ids`` is every
    sibling's job id, so a flat group of width n costs O(n^2). At production
    width the derived graph alone is ~1 MB against a 1 MiB line limit, which is
    why the cap's own "the control socket will drop it" warning (30,839 lines
    across five sessions) was followed by an unreadable write.
    """
    from local_operator.mobile.projection import cap_projection_frame
    from local_operator.session.runtime.server import _MAX_LINE_BYTES

    fold = _flat_roster_fold(256)
    naive = _projection_line(fold.projection.to_json())
    derived = sum(len(str(list(row.peer_ids)).encode()) for row in fold.projection.subagents)
    assert naive > _MAX_LINE_BYTES, (
        f"the fixture no longer reproduces the unreadable welcome: {naive:,} B is "
        f"inside the {_MAX_LINE_BYTES:,} B limit"
    )
    assert derived > 1_000_000, (
        "the fixture no longer reproduces WHY it was fatal: without an O(n^2) "
        f"derived graph ({derived:,} B here) the text tiers could have fitted it"
    )

    data, degraded = cap_projection_frame(fold.projection)
    rows = data["subagents"]
    assert degraded is True
    assert len(rows) == 256, "the shed tier must shed fields, not children"
    assert all(row.get("peer_ids") == [] for row in rows)
    assert all(row.get("child_ids") == [] for row in rows)
    assert all(row.get("ancestor_ids") == [] for row in rows)
    assert all(row.get("ancestors") == [] for row in rows)
    assert all("parent_job_id" in row for row in rows)
    assert _projection_line(data) <= _MAX_LINE_BYTES
    # Nothing else was spent to get there: the row TEXT is still the reader's.
    assert all(row["label"] for row in rows)


def test_the_roster_falls_back_to_identity_rows_when_shedding_is_not_enough() -> None:
    """The last tier before the honest warning: identity rows only.

    A roster wide enough that even identity rows cannot fit is genuinely
    unbounded, so the frame degrades to one row per child carrying only what a
    reader can neither derive nor fetch: job id, label, parent edge and
    lifecycle. The rows ARE the count (no ``subagent_count`` key: the client's
    rebuild filters to ``SessionProjection``'s fields and would drop an unknown
    key on the floor). Everything else is fetchable per child — except
    ``error_text``, which exists nowhere else and is the real price of this
    tier.
    """
    from local_operator.mobile.projection import (
        FRAME_CAP_ROSTER_IDENTITY_FIELDS,
        cap_projection_frame,
    )
    from local_operator.session.frontend_state import oversized_frame_report
    from local_operator.session.runtime.server import _MAX_LINE_BYTES

    rows = [
        SubagentRow(
            job_id=f"child-{index}",
            label=f"child {index}",
            parent_job_id=None,
            progress="working",
            prompt="P" * 120,
        )
        for index in range(4_000)
    ]
    projection = SessionProjection(session_id="s1", pid=1, subagents=rows)
    data, degraded = cap_projection_frame(projection)

    assert degraded is True
    assert len(data["subagents"]) == 4_000
    assert all(set(row) <= set(FRAME_CAP_ROSTER_IDENTITY_FIELDS) for row in data["subagents"])
    assert all("parent_job_id" in row for row in data["subagents"])
    assert "subagent_count" not in data
    assert oversized_frame_report({"op": "projection", "data": data}, _MAX_LINE_BYTES) is None
    # The frame must still rebuild into a projection on the client, or the
    # degradation would trade an unreadable frame for an unusable one.
    record = SessionRecord(
        pid=7,
        kind="tui",
        session_id="s1",
        conversation_name="c",
        cwd="/tmp",
        model_label="m",
        control_port=1,
        control_key="k",
        protocol=1,
    )
    rebuilt = _projection_from_json(data, record)
    assert rebuilt.session_id == "s1"
    assert len(rebuilt.subagents) == 4_000
    assert rebuilt.subagents[0].label == "child 0"


def test_live_fold_keeps_failed_child_error_text_generous() -> None:
    """A failed child's ``error_text`` must survive on the wire, unlike result.

    ``error_text`` is ``str(exc)`` from the parent runner and is never in the
    child transcript, so the lazy /history fetch cannot recover it — the wire
    value is the only copy the phone's Outcome panel renders. Capping it at the
    200-char ``result_text`` preview would truncate the failure tail everywhere
    with no recovery (F1), so the live lifecycle merge must carry it generously
    (``SUBAGENT_ERROR_CHARS``) while still bounding it. Pins that behaviour and
    that a multi-line trace keeps its line breaks.
    """
    from local_operator.mobile.projection import (
        SUBAGENT_ERROR_CHARS,
        SUBAGENT_OUTCOME_CHARS,
    )

    error = "Traceback (most recent call last):\n" + "\n".join(
        f"  frame {i}: boom in module_{i}" for i in range(200)
    )
    assert len(error) > SUBAGENT_ERROR_CHARS  # long enough to exercise the cap
    job = SimpleNamespace(
        status="running",
        agent_role="reviewer",
        model_label="test/model",
        latest_details={"progress": "checking"},
        result_text=None,
        error_text=None,
    )
    session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: job))
    comms = SubagentComms(cast(Session, cast(Any, session)))
    comms.record_launch("child", "child")
    comms.record_outcome("child", "failed", error_text=error)
    fold = make_fold()
    fold.set_subagent_details(comms)

    wire_row = fold.projection.to_json()["subagents"][0]
    # NOT clipped to the 200-char result preview; the failure tail rides.
    assert len(wire_row["error_text"]) > SUBAGENT_OUTCOME_CHARS
    assert len(wire_row["error_text"]) <= SUBAGENT_ERROR_CHARS
    assert "\n" in wire_row["error_text"]  # multi-line structure preserved


def test_a_capacity_parked_child_is_not_drawn_as_running() -> None:
    """UX round 3: the roster header COUNTS this field, so ``queued`` ≠ ``running``.

    ``mobile/projection.py``'s mapping is what the session view's roster header
    counts — it prints ``{running}/{direct.length} running`` over these rows — so
    folding a capacity-parked child into ``running`` made the view claim a child
    waiting for a slot was spending, one tap after a list chip that had just been
    taught to keep the two apart (UX round 3's contradiction). The runtime keeps
    them apart in ``RUNNING_SUBAGENT_STATUSES`` and the phone's summary does too;
    only the fold disagreed.

    ``starting`` is deliberately left in the running lane by the mapping (an
    admitted child spinning up IS spending); this route cannot produce that
    status from a job row, so it is stated in the mapping rather than asserted
    here.
    """

    def job(*, queued: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            status="running",
            queued=queued,
            agent_role="coder",
            model_label="test/model",
            latest_details={},
            result_text=None,
            error_text=None,
        )

    jobs = {
        "admitted": job(),
        "waiting": job(queued=True),
    }
    session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: jobs[job_id]))
    comms = SubagentComms(cast(Session, cast(Any, session)))
    for job_id in jobs:
        comms.record_launch(job_id, job_id)

    fold = make_fold()
    fold.set_subagent_details(comms)

    statuses = {row.job_id: row.status for row in fold.projection.subagents}
    assert statuses == {"admitted": "running", "waiting": "queued"}, statuses

    # The roster header's own arithmetic, over the same rows: one of the two
    # direct children is spending, so the view reads `1/2 running · 1 queued`
    # rather than `2/2 running`.
    direct = [row for row in fold.projection.subagents if row.parent_job_id is None]
    assert sum(1 for row in direct if row.status == "running") == 1, [
        (row.job_id, row.status) for row in direct
    ]
    assert sum(1 for row in direct if row.status == "queued") == 1, [
        (row.job_id, row.status) for row in direct
    ]


def test_recorded_terminal_outcome_never_regresses_to_running_job_row() -> None:
    """The runner records terminal state before the manager stamps its row."""

    job = SimpleNamespace(
        status="running",
        agent_role="coder",
        model_label="test/model",
        latest_details={"progress": "stale progress"},
        result_text=None,
        error_text=None,
    )
    session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: job))
    comms = SubagentComms(cast(Session, cast(Any, session)))
    comms.record_launch("nested", "nested")
    comms.record_outcome("nested", "completed", result_text="settled result")

    fold = make_fold()
    fold.set_subagent_details(comms)
    selected = fold.projection.subagents[0]
    assert (selected.status, selected.result_text) == ("completed", "settled result")
    assert (selected.progress, selected.activity) == ("", "")


def test_nested_subagent_failure_refreshes_error_and_clears_progress() -> None:
    job = SimpleNamespace(
        status="running",
        agent_role="reviewer",
        model_label="test/model",
        latest_details={"progress": "checking"},
        result_text=None,
        error_text=None,
    )
    session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: job))
    comms = SubagentComms(cast(Session, cast(Any, session)))
    comms.record_launch("parent", "parent")
    comms.record_launch("nested", "nested", parent_job_id="parent")
    fold = make_fold()

    fold.set_subagent_details(comms)
    job.status = "failed"
    job.latest_details = {"progress": "stale progress"}
    job.error_text = "provider failed"
    fold.set_subagent_details(comms)

    selected = {row.job_id: row for row in fold.projection.subagents}["nested"]
    assert selected.status == "failed"
    assert selected.result_text == ""
    assert selected.error_text == "provider failed"
    assert selected.progress == ""
    assert selected.activity == ""


def test_history_fold_pairs_tool_calls_with_results() -> None:
    fold = make_fold()
    call = ToolCall(id="c1", name="read", arguments={"path": "/x.py"})
    history: list[AgentMessage] = [
        Message.user("look at x"),
        Message.assistant("reading it", tool_calls=[call]),
        Message.tool_result(ToolResult(tool_call_id="c1", content=[TextContent(text="file body")])),
    ]
    fold.fold_history(history)
    kinds = [e.kind for e in fold.projection.transcript]
    assert kinds == ["user", "assistant", "tool"]
    tool_row = fold.projection.transcript[-1]
    assert tool_row.tool_state == "done"
    assert tool_row.summary == "/x.py"
    assert tool_row.details["output"] == "file body"


def test_history_fold_maps_peer_message_to_its_own_kind() -> None:
    from local_operator.harness.message_types import PEER_MESSAGE_MESSAGE_TYPE

    fold = make_fold()
    sender = {"pid": 42, "conversation_name": "peer", "model_label": "test/model"}
    peer = CustomMessage(
        custom_type=PEER_MESSAGE_MESSAGE_TYPE,
        attribution="user",
        details={"text": "<wrapped>hi</wrapped>", "body": "hi there", "sender": sender},
    )
    fold.fold_history([peer])
    rows = fold.projection.transcript
    assert len(rows) == 1
    # The phone renders the raw body, never the model-facing wrapped envelope,
    # and carries the sender for the card label.
    assert rows[0].kind == "peer_message"
    assert rows[0].text == "hi there"
    assert rows[0].details["sender"] == sender


def test_note_peer_message_appends_optimistic_row() -> None:
    fold = make_fold()
    fold.note_peer_message("live echo", sender={"pid": 7, "conversation_name": "peer"})
    row = fold.projection.transcript[-1]
    assert row.kind == "peer_message"
    assert row.text == "live echo"
    assert row.details["sender"]["pid"] == 7


def _steer_envelope(body: str) -> str:
    """The model-facing wrapper ``SubagentComms._format_to_child`` builds for a
    steer, restated here so the fold tests pin the render contract against the
    exact shape a persisted steer row carries."""
    return (
        "<parent-message>\n"
        "This changes your instructions. Apply it from now on, and drop work it "
        "makes pointless.\n\n"
        f"{body}\n"
        "</parent-message>"
    )


def test_history_fold_renders_a_persisted_steer_as_the_parents_words() -> None:
    """A hub steer persists as a plain user Message whose text is the
    model-facing envelope. The phone must show the parent's own words as a
    parent_message row, never the XML — the phone's history fold sees only
    LLM-visible messages (the journaled fact is a custom row that never
    replays), so body extraction is its only path."""
    from local_operator.mobile.projection import fold_messages_to_entries

    steer = Message.user(_steer_envelope("Focus on retries"), id="steer-1")
    entries = fold_messages_to_entries([Message.user("do the thing"), steer])

    assert [(entry.kind, entry.text) for entry in entries] == [
        ("user", "do the thing"),
        ("parent_message", "Focus on retries"),
    ]
    assert "<parent-message>" not in " ".join(entry.text for entry in entries)


def test_fold_history_renders_a_persisted_steer_as_the_parents_words() -> None:
    """The attach rebuild applies the same rule as the lazy-load fold, so an
    attaching phone and a paged history agree about the row."""
    fold = make_fold()
    fold.fold_history([Message.user(_steer_envelope("Focus on retries"), id="steer-1")])

    rows = fold.projection.transcript
    assert len(rows) == 1
    assert rows[0].kind == "parent_message"
    assert rows[0].text == "Focus on retries"
    assert "<parent-message>" not in rows[0].text


def test_a_live_delivered_steer_never_renders_the_envelope() -> None:
    """A hub steer delivered mid-turn announces itself as a user
    MessageStartEvent carrying the envelope text. The fold must paint the
    parent's words, not the XML, exactly like the durable folds."""
    fold = make_fold()
    message = Message.user(_steer_envelope("Focus on retries"), id="steer-1")

    added = fold.absorb_user_event(message)

    assert added is True
    row = fold.projection.transcript[-1]
    assert row.kind == "parent_message"
    assert row.text == "Focus on retries"
    assert "<parent-message>" not in row.text


def test_a_phone_typed_envelope_reconciles_its_echo_instead_of_doubling() -> None:
    """A message the phone sent that happens to carry the envelope shape must
    reconcile against its optimistic echo, not append beside it.

    The envelope branch used to return before the pending-echo pop, so the row
    rendered twice under ONE id — and the stranded echo still displayed the raw
    XML this path exists to suppress. Two rows sharing an id is also bad input
    for the web client's list reconciliation.
    """
    fold = make_fold()
    envelope = _steer_envelope("Focus on retries")
    fold.note_user_message(envelope, steer=True, message_id="cmd-1")

    added = fold.absorb_user_event(Message.user(envelope, id="cmd-1"))

    assert added is False
    rows = fold.projection.transcript
    assert len(rows) == 1
    assert (rows[0].id, rows[0].kind, rows[0].text) == (
        "cmd-1",
        "parent_message",
        "Focus on retries",
    )
    assert "<parent-message>" not in rows[0].text


def test_a_user_quoting_the_envelope_keeps_their_own_words() -> None:
    """Extraction requires the builder's exact instruction preamble, so a human
    quoting the wrapper keeps their words and their ``user`` row."""
    from local_operator.mobile.projection import fold_messages_to_entries

    quoted = "<parent-message>\nwhy does my log show this?\n\nsecret plan\n</parent-message>"
    entries = fold_messages_to_entries([Message.user(quoted, id="human-1")])

    assert [(entry.kind, entry.text) for entry in entries] == [("user", quoted)]


def test_transcript_is_capped_from_the_front() -> None:
    fold = make_fold()
    for i in range(PROJECTION_TRANSCRIPT_LIMIT + 25):
        fold.fold_event(NoticeEvent(text=f"note {i}"))
    assert len(fold.projection.transcript) == PROJECTION_TRANSCRIPT_LIMIT
    # The OLDEST rows dropped: the tail is what a phone renders.
    assert fold.projection.transcript[-1].text == f"note {PROJECTION_TRANSCRIPT_LIMIT + 24}"
    assert fold.projection.transcript[0].text == "note 25"


def test_mid_run_turn_end_does_not_settle_streaming() -> None:
    """Regression: a multi-batch turn must stay "in progress" across the
    per-model-turn boundaries inside it.

    ``TurnEndEvent`` fires after every assistant turn that made tool calls and
    the run will continue (harness.loop ~589), so a turn with several tool
    batches emits several TurnEndEvents before its single AgentEndEvent. The
    session keeps ``is_streaming`` True across them and the TUI working line
    stays up; the phone must match. Folding TurnEndEvent as a streaming
    terminal blanked the working line mid-run and (with the reconcile latch)
    pinned it off. Only AgentEndEvent settles the turn.
    """
    fold = make_fold()
    fold.fold_event(AgentStartEvent(generation=1))

    # Batch 1: a tool runs, then a mid-run TurnEndEvent (more work to come).
    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="c1", tool_name="bash", args={}, intent="step 1")
    )
    fold.fold_event(
        ToolExecutionEndEvent(
            tool_call_id="c1",
            tool_name="bash",
            result=ToolResult(tool_call_id="c1", content=[TextContent(text="ok")], is_error=False),
        )
    )
    fold.fold_event(TurnEndEvent(message=Message.assistant()))

    # Still streaming; the working line shows the model-wait, not blank.
    assert fold.projection.streaming is True
    assert fold.projection.activity == "thinking"

    # Batch 2 continues normally, then the run's single terminal settles it.
    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="c2", tool_name="bash", args={}, intent="step 2")
    )
    assert fold.projection.streaming is True
    assert fold.projection.activity == "step 2"
    fold.fold_event(AgentEndEvent(generation=1))
    assert fold.projection.streaming is False
    assert fold.projection.activity == ""
    assert fold.projection.stop_reason == "completed"


def test_working_line_says_thinking_until_text_actually_streams() -> None:
    """``message_start`` is a model call in flight, not prose.

    The loop yields ``MessageStartEvent`` from a placeholder at the top of every
    provider call, before the first token; a tool-only call never streams text
    after it. Folding that as "responding" told the phone the model was
    writing for the whole of every call. The TUI's WorkingBlock keys on the
    first non-empty delta, and the phone's working line follows the same rule.
    """
    fold = make_fold()
    fold.fold_event(AgentStartEvent(generation=1))

    # A tool-only call in the loop's real order: placeholder, the call being
    # dictated, message_end closing the model call, then (after the approval
    # gate's wait) the tool starting. The composed intent must hold across
    # message_end — that boundary ends prose, not a call still being composed.
    first = Message.assistant()
    fold.fold_event(MessageStartEvent(message=first))
    assert fold.projection.activity == "thinking"
    fold.fold_event(ToolCallComposeEvent(tool_call_id="c1", tool_name="bash", intent="probing"))
    assert fold.projection.activity == "probing"
    fold.fold_event(MessageEndEvent(message=first))
    assert fold.projection.activity == "probing"
    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="c1", tool_name="bash", args={}, intent="probing")
    )
    assert fold.projection.activity == "probing"
    fold.fold_event(
        ToolExecutionEndEvent(
            tool_call_id="c1",
            tool_name="bash",
            result=ToolResult(tool_call_id="c1", content=[TextContent(text="ok")], is_error=False),
        )
    )
    assert fold.projection.activity == "thinking"

    # A prose call: still the model-wait until the first non-empty delta.
    second = Message.assistant()
    fold.fold_event(MessageStartEvent(message=second))
    assert fold.projection.activity == "thinking"
    fold.fold_event(MessageUpdateEvent(message=second, delta=""))
    assert fold.projection.activity == "thinking"
    fold.fold_event(MessageUpdateEvent(message=second, delta="Here "))
    assert fold.projection.activity == "responding"
    fold.fold_event(MessageUpdateEvent(message=second, delta="it is."))
    assert fold.projection.activity == "responding"
    fold.fold_event(
        MessageEndEvent(
            message=second.model_copy(update={"content": [TextContent(text="Here it is.")]})
        )
    )
    assert fold.projection.activity == "thinking"
    fold.fold_event(AgentEndEvent(generation=1))
    assert fold.projection.activity == ""


def test_pinned_opener_never_costs_the_newest_row() -> None:
    """Regression: once the transcript passed the cap WITH the opening user
    message pinned at the head, the newest row stopped reaching the phone.

    ``_cap_tail`` pins the first user message and fills the rest from the tail.
    It used to make room by dropping ``tail[-1]`` — the row JUST appended — and
    because the cap runs on every append, the transcript froze: past the cap no
    new tool call or notice ever appeared (the field report's missing "last
    several tool calls"). The pin must cost the OLDEST tail row, never the
    newest, and the tail must keep advancing.
    """
    fold = make_fold()
    fold.note_user_message("the opening ask — this names the whole conversation")
    for i in range(PROJECTION_TRANSCRIPT_LIMIT + 40):
        fold.fold_event(NoticeEvent(text=f"note {i}"))

    transcript = fold.projection.transcript
    assert len(transcript) == PROJECTION_TRANSCRIPT_LIMIT
    # Opener stays pinned at the head so the phone always knows the topic.
    assert transcript[0].kind == "user"
    assert transcript[0].text.startswith("the opening ask")
    # The single newest row is present — the whole point of the fix.
    assert transcript[-1].text == f"note {PROJECTION_TRANSCRIPT_LIMIT + 39}"

    # The tail keeps advancing on further appends (it used to be frozen).
    fold.fold_event(NoticeEvent(text="the very latest"))
    transcript = fold.projection.transcript
    assert len(transcript) == PROJECTION_TRANSCRIPT_LIMIT
    assert transcript[0].kind == "user"
    assert transcript[-1].text == "the very latest"


def test_todo_refresh_replaces_wholesale() -> None:
    # The store is phased: one implicit "Todos" phase carries a flat list.
    fold = make_fold()
    fold.set_todos(
        [
            {
                "name": "Todos",
                "items": [
                    {"text": "a", "status": "pending"},
                    {"text": "b", "status": "done"},
                ],
            }
        ]
    )
    assert len(fold.projection.todos) == 1
    assert [t.status for t in fold.projection.todos[0].items] == ["pending", "done"]
    fold.set_todos([{"name": "Todos", "items": [{"text": "a", "status": "done"}]}])
    assert len(fold.projection.todos) == 1
    assert len(fold.projection.todos[0].items) == 1


def test_todo_multi_phase_structure_preserved() -> None:
    """A real multi-phase store maps to TodoPhase/TodoItem with every field —
    text/status/reason — intact and phase order preserved."""
    fold = make_fold()
    fold.set_todos(
        [
            {
                "name": "Design",
                "items": [
                    {"text": "sketch", "status": "done"},
                    {"text": "review", "status": "dropped"},
                ],
            },
            {
                "name": "Build",
                "items": [
                    {"text": "impl", "status": "pending"},
                    {"text": "deploy", "status": "blocked", "reason": "waiting on infra"},
                ],
            },
        ]
    )
    phases = fold.projection.todos
    assert [p.name for p in phases] == ["Design", "Build"]
    assert [t.text for t in phases[0].items] == ["sketch", "review"]
    blocked = phases[1].items[1]
    assert blocked.status == "blocked"
    assert blocked.reason == "waiting on infra"
    # to_json is the wire shape the front-end reads: nested phase → items.
    wire = phases[1].to_json()
    assert wire == {
        "name": "Build",
        "items": [
            {"text": "impl", "status": "pending", "reason": ""},
            {"text": "deploy", "status": "blocked", "reason": "waiting on infra"},
        ],
    }


def test_todo_legacy_flat_list_coerced_to_one_phase() -> None:
    """A hand-attached legacy flat list (item dicts at the top level, no phase
    wrapper) is coerced via ``_as_phases`` to a single implicit "Todos" phase
    rather than rendering empty-text rows — the bug the phased callers exposed."""
    fold = make_fold()
    fold.set_todos([{"text": "a", "status": "pending"}, {"text": "b", "status": "done"}])
    assert len(fold.projection.todos) == 1
    assert fold.projection.todos[0].name == "Todos"
    assert [t.text for t in fold.projection.todos[0].items] == ["a", "b"]


def test_todo_open_count_across_phases() -> None:
    """The session-list ``todos_open`` badge counts open (pending OR blocked)
    items across ALL phases; done/dropped never count."""
    fold = make_fold()
    fold.set_todos(
        [
            {
                "name": "A",
                "items": [
                    {"text": "1", "status": "pending"},
                    {"text": "2", "status": "done"},
                ],
            },
            {
                "name": "B",
                "items": [
                    {"text": "3", "status": "blocked", "reason": "x"},
                    {"text": "4", "status": "dropped"},
                    {"text": "5", "status": "pending"},
                ],
            },
        ]
    )
    # Mirror the daemon's list-summary arithmetic: pending + blocked across
    # phases = 3 (items 1, 3, 5), done/dropped excluded.
    p = fold.projection
    open_count = sum(
        1 for phase in p.todos for t in phase.items if t.status in ("pending", "blocked")
    )
    assert open_count == 3


def test_summarize_args_priority_and_compaction() -> None:
    assert _summarize_args("write", {"path": "/p/q.py", "content": "long"}) == "/p/q.py"
    assert _summarize_args("bash", {"command": "ls -la"}) == "ls -la"
    long_value = "x" * 200
    assert len(_summarize_args("read", {"path": long_value})) <= 80
    assert _summarize_args("noop", {}) == "noop"


def test_diff_counts_only_from_reported_details() -> None:
    assert _diff_counts(None) == (0, 0)
    assert _diff_counts({}) == (0, 0)
    assert _diff_counts({"added": 3, "removed": 1}) == (3, 1)
    assert _diff_counts({"lines_added": "5", "lines_removed": 2}) == (5, 2)
    assert _diff_counts({"added": "junk"}) == (0, 0)


def test_a_live_notice_carries_its_severity_to_the_phone() -> None:
    """Design round 1, D1: a LIVE notice must arrive with its tier.

    ``NoticeRow`` reads the glyph and the ink from ``details.severity`` alone,
    so a fold that drops it draws a ``warning`` truncation as the quiet ``·``
    in ``text-ink-dim`` -- and then flips the SAME event to amber ``!`` on the
    next refresh, when it arrives through the replay fold, which carries the
    field. The two produces must agree, so this asserts the live entry against
    the replayed one rather than against a literal.
    """
    from local_operator.mobile.projection import fold_messages_to_entries

    fold = make_fold()
    live = NoticeEvent(text="the model hit the output limit", kind="warning")
    fold.fold_event(live)
    live_entry = fold.projection.transcript[-1]
    assert live_entry.kind == "notice"
    assert live_entry.details["severity"] == "warning"

    # The same event as a REPLAYED row (the message the harness journals), read
    # through the fold that has always carried the tier.
    replayed = fold_messages_to_entries(
        [
            Message(
                role="assistant",
                content=[TextContent(text="partial answer")],
                id="a1",
                stop_reason="length",
            )
        ]
    )
    replayed_notices = [e for e in replayed if e.kind == "notice"]
    assert replayed_notices, "the replay fold must also emit the notice row"
    assert live_entry.details["severity"] == replayed_notices[-1].details["severity"]

    # All three kinds map across, not just the one this PR made visible.
    for kind in ("info", "warning", "error"):
        fold.fold_event(NoticeEvent(text=f"note {kind}", kind=kind))
        assert fold.projection.transcript[-1].details["severity"] == kind


def test_projection_version_bumps_on_every_fold() -> None:
    fold = make_fold()
    v0 = fold.projection.version
    fold.fold_event(NoticeEvent(text="hi"))
    assert fold.projection.version > v0


def test_pending_queue_shows_front_and_counts() -> None:
    """A parallel tool batch opens several approvals at once. The fold must
    show the FRONT one and report the total, and clearing one by id must
    surface the next — not dismiss every sibling (the mobile hang report)."""
    fold = make_fold()
    a = PendingRequest(request_id="a", kind="approval", title="bash")
    b = PendingRequest(request_id="b", kind="approval", title="write")
    fold.push_pending(a)
    fold.push_pending(b)
    assert fold.projection.pending is a
    assert fold.projection.pending_count == 2

    # Answering the FRONT reveals the next, count decrements.
    fold.pop_pending("a")
    assert fold.projection.pending is b
    assert fold.projection.pending_count == 1

    # Answering an out-of-order id (the second card) is honoured too.
    fold.pop_pending("b")
    assert fold.projection.pending is None
    assert fold.projection.pending_count == 0


def test_pop_pending_out_of_order_keeps_the_other_card() -> None:
    """Concurrent gates settle in whatever order the user answers, not the
    order enqueued: popping the second must leave the first still showing."""
    fold = make_fold()
    fold.push_pending(PendingRequest(request_id="a", kind="approval", title="bash"))
    fold.push_pending(PendingRequest(request_id="b", kind="ask", title="which?"))
    fold.pop_pending("b")
    assert fold.projection.pending is not None
    assert fold.projection.pending.request_id == "a"
    assert fold.projection.pending_count == 1


def test_set_pending_still_replaces_for_the_tui_mirror() -> None:
    """The TUI-mirror handle uses set_pending: the terminal serializes its own
    approvals, so the phone mirrors exactly one card or none."""
    fold = make_fold()
    fold.push_pending(PendingRequest(request_id="a", kind="approval", title="bash"))
    fold.set_pending(PendingRequest(request_id="z", kind="ask", title="q"))
    assert fold.projection.pending_count == 1
    assert fold.projection.pending is not None
    assert fold.projection.pending.request_id == "z"
    fold.set_pending(None)
    assert fold.projection.pending is None
    assert fold.projection.pending_count == 0


def test_image_refs_are_index_and_mime_only() -> None:
    """User-turn attachments project as lightweight references (image-only
    index + mime), never bytes — the pixels are fetched lazily so a per-token
    repaint stays small. A text caption does not shift the index."""
    message = Message.user(
        "look at these",
        [
            ImageContent(data="AAAA", mime_type="image/png"),
            ImageContent(data="BBBB", mime_type="image/jpeg"),
        ],
    )
    refs = _image_refs(message)
    assert refs == [
        {"index": 0, "mime_type": "image/png"},
        {"index": 1, "mime_type": "image/jpeg"},
    ]
    # No base64 leaks into the reference.
    assert all("data" not in r for r in refs)


def test_history_fold_carries_image_refs_on_user_rows() -> None:
    fold = make_fold()
    history: list[AgentMessage] = [
        Message.user("with a shot", [ImageContent(data="AAAA", mime_type="image/png")]),
    ]
    fold.fold_history(history)
    user_rows = [e for e in fold.projection.transcript if e.kind == "user"]
    assert len(user_rows) == 1
    assert user_rows[0].images == [{"index": 0, "mime_type": "image/png"}]


def test_absorb_user_event_upgrades_echoed_row_with_image_refs() -> None:
    """A phone-sent prompt is echoed WITHOUT refs (the handle has no persisted
    id yet); the real MessageStartEvent then carries the attachments, and the
    fold must upgrade the echoed row in place so the sender sees thumbnails."""
    fold = make_fold()
    fold.note_user_message("with a shot")  # optimistic echo, no images
    echoed = fold.projection.transcript[-1]
    assert echoed.images == []

    message = Message.user("with a shot", [ImageContent(data="AAAA", mime_type="image/png")])
    added = fold.absorb_user_event(message)
    assert added is False  # de-duped, not a second row
    assert len([e for e in fold.projection.transcript if e.kind == "user"]) == 1
    upgraded = fold.projection.transcript[-1]
    assert upgraded.id == message.id
    assert upgraded.images == [{"index": 0, "mime_type": "image/png"}]


# --- issue #231: user echoes de-duped by id, not by a 3-entry tail window ----


def test_a_phone_steer_is_not_repainted_once_assistant_rows_push_it_out() -> None:
    """Issue #231, reproduced: the defect the tail window guaranteed.

    A phone steer is echoed optimistically, then the turn keeps running — the
    engine only drains the steering queue at a LATER tool boundary, so by the
    time the message's own ``MessageStartEvent`` arrives the echo is several
    assistant rows back. The three-entry scan could no longer see it and
    painted the steer a second time. The registry keys on the message id, so
    distance from the tail is irrelevant.
    """
    fold = make_fold()
    command_id = "cmd-steer-1"
    fold.note_user_message("use the other endpoint", steer=True, message_id=command_id)
    assert len([e for e in fold.projection.transcript if e.kind == "steer"]) == 1

    # The turn goes on: four assistant rows push the echo well past the window
    # the old scan looked at.
    for index in range(4):
        fold.fold_event(
            MessageStartEvent(message=Message.assistant(f"still working {index}")),
        )
    assert len(fold.projection.transcript) == 5

    # Only NOW is the steer drained and announced, carrying the id the handle
    # supplied as the message id.
    fold.absorb_user_event(Message.user("use the other endpoint", id=command_id))

    rows = [e for e in fold.projection.transcript if e.text == "use the other endpoint"]
    assert len(rows) == 1, "the drain repainted a steer already on the phone"
    assert rows[0].kind == "steer", "the echo's own row survives, not a fresh user row"


def test_absorb_user_event_returns_false_for_a_registered_echo() -> None:
    """The de-dupe contract the callers read: a registered echo is folded into
    the existing row, so the fold reports it did NOT add one."""
    fold = make_fold()
    fold.note_user_message("already on screen", steer=True, message_id="cmd-2")
    added = fold.absorb_user_event(Message.user("already on screen", id="cmd-2"))
    assert added is False
    assert len([e for e in fold.projection.transcript if e.text == "already on screen"]) == 1


def test_a_distinct_message_with_colliding_words_still_paints() -> None:
    """The mirror image of the same defect: a genuinely NEW message whose text
    matched a recent row was swallowed by the window. Different id, different
    message, so it must appear."""
    fold = make_fold()
    fold.note_user_message("continue", steer=True, message_id="cmd-3")
    fold.absorb_user_event(Message.user("continue", id="a-different-message"))
    rows = [e for e in fold.projection.transcript if e.text == "continue"]
    assert len(rows) == 2, "a distinct message must not be eaten by a word collision"


def test_a_registered_echo_row_is_not_consumed_by_a_colliding_neighbour() -> None:
    """The tail fallback (for handles with no id) must not spend a row that has
    an EXACT event coming: that row's own announcement would then find its
    entry gone and paint the duplicate this issue exists to remove."""
    fold = make_fold()
    fold.note_user_message("same words", steer=True, message_id="cmd-4")

    # An id-less announcement carrying the same words: not the registered
    # steer's, so it paints rather than consuming that row.
    fold.absorb_user_event(Message.user("same words", id="foreign"))
    assert len([e for e in fold.projection.transcript if e.text == "same words"]) == 2

    # The registered steer's own event still upgrades its row, adding nothing.
    fold.absorb_user_event(Message.user("same words", id="cmd-4"))
    assert len([e for e in fold.projection.transcript if e.text == "same words"]) == 2


def test_the_echoed_row_adopts_the_persisted_message_id() -> None:
    """The row is keyed by the command id until the real message arrives; from
    then on it must carry the message id, which is what a later history fold
    and the web client's list reconciliation agree on."""
    fold = make_fold()
    fold.note_user_message("key me", steer=True, message_id="cmd-5")
    assert fold.projection.transcript[-1].id == "cmd-5"
    fold.absorb_user_event(Message.user("key me", id="cmd-5"))
    assert fold.projection.transcript[-1].id == "cmd-5"


def test_a_legacy_handle_without_an_id_keeps_the_tail_dedup() -> None:
    """A handle that supplies no id (older/third-party) still de-dupes its own
    echo through the historical tail scan — the compatibility path."""
    fold = make_fold()
    fold.note_user_message("no id supplied", steer=True)
    added = fold.absorb_user_event(Message.user("no id supplied"))
    assert added is False
    assert len([e for e in fold.projection.transcript if e.text == "no id supplied"]) == 1


def test_a_history_fold_clears_pending_echoes() -> None:
    """A wholesale rebuild replaces the rows the entries pointed at, so a
    surviving entry would reference a row no longer in the transcript."""
    fold = make_fold()
    fold.note_user_message("pending across the fold", steer=True, message_id="cmd-6")
    assert fold._pending_user_echoes
    fold.fold_history([Message.user("something else entirely")])
    assert fold._pending_user_echoes == {}


def test_a_promoted_compose_key_rekeys_the_phone_row_instead_of_appending() -> None:
    """The phone must show ONE row for a call whose id arrived late.

    ``_tool_row`` correlates by ``tool_call_id``. A provider that sends the
    name before the id makes the loop announce the row under ``compose:0`` and
    then promote it to the real id on one frame naming both. Without honouring
    that hand-off the lookup misses, a SECOND row is appended, and the
    placeholder row is left composing forever because every later start and end
    carries the real id.

    The entry's own ``id`` deliberately does not move: clients diff the
    transcript by row id, so re-identifying a row mid-turn would read as the
    row being replaced rather than updated.
    """
    fold = make_fold()
    fold.fold_event(AgentStartEvent(generation=1))
    fold.fold_event(
        ToolCallComposeEvent(tool_call_id="compose:0", tool_name="bash", argument_bytes=8)
    )
    tool_rows = [row for row in fold.projection.transcript if row.kind == "tool"]
    assert len(tool_rows) == 1
    original_row_id = tool_rows[0].id

    fold.fold_event(
        ToolCallComposeEvent(
            tool_call_id="real_0",
            tool_name="bash",
            argument_bytes=64,
            supersedes_tool_call_id="compose:0",
        )
    )
    tool_rows = [row for row in fold.projection.transcript if row.kind == "tool"]
    assert len(tool_rows) == 1
    assert tool_rows[0].id == original_row_id
    assert tool_rows[0].tool_call_id == "real_0"

    # The real start and end settle THAT row rather than opening another.
    fold.fold_event(ToolExecutionStartEvent(tool_call_id="real_0", tool_name="bash", args={}))
    fold.fold_event(
        ToolExecutionEndEvent(
            tool_call_id="real_0",
            tool_name="bash",
            result=ToolResult(
                tool_call_id="real_0", content=[TextContent(text="ok")], is_error=False
            ),
        )
    )
    tool_rows = [row for row in fold.projection.transcript if row.kind == "tool"]
    assert len(tool_rows) == 1
    assert tool_rows[0].tool_state == "done"


def test_a_compose_event_without_the_supersedes_field_is_unchanged() -> None:
    """An older runtime omits the field; the phone must behave as before."""
    fold = make_fold()
    fold.fold_event(AgentStartEvent(generation=1))
    fold.fold_event(ToolCallComposeEvent(tool_call_id="c1", tool_name="bash", argument_bytes=8))
    fold.fold_event(ToolCallComposeEvent(tool_call_id="c2", tool_name="bash", argument_bytes=8))
    tool_rows = [row for row in fold.projection.transcript if row.kind == "tool"]
    assert [row.tool_call_id for row in tool_rows] == ["c1", "c2"]


def test_a_repeated_supersession_does_not_disturb_an_already_rekeyed_row() -> None:
    """The hand-off is repeated on every later frame and must stay idempotent.

    The second and third promotions find no placeholder to move, so they take
    the ordinary path and update the row already keyed by the real id rather
    than appending another.
    """
    fold = make_fold()
    fold.fold_event(AgentStartEvent(generation=1))
    fold.fold_event(
        ToolCallComposeEvent(tool_call_id="compose:0", tool_name="bash", argument_bytes=5)
    )
    for size in (10, 20, 30):
        fold.fold_event(
            ToolCallComposeEvent(
                tool_call_id="real_0",
                tool_name="bash",
                argument_bytes=size,
                supersedes_tool_call_id="compose:0",
            )
        )
    tool_rows = [row for row in fold.projection.transcript if row.kind == "tool"]
    assert len(tool_rows) == 1
    assert tool_rows[0].tool_call_id == "real_0"
    assert tool_rows[0].details["argument_bytes"] == 30


def test_a_cut_off_turn_still_offers_the_phones_resume_affordance() -> None:
    """MAJOR-1: a cut-off must not read as "completed" on the phone.

    ``stop_reason`` is the wire fact ``composer.tsx`` gates
    ``interrupted — tap to resume`` on, and the taxonomy flip reports a cut-off
    as ``aborted=False, error=<notice>`` — so folding ``aborted`` alone told the
    phone the turn had FINISHED and silently removed the only recovery
    affordance for exactly the sessions the operator reports losing. A cut-off
    did not complete; it was cut off, which is what ``cut_off``/``cut_off_cause``
    states.
    """
    from local_operator.incidents import format_cut_off_notice, render_cut_off_reason

    fold = ProjectionFold(SessionProjection(session_id="cutoff-phone", pid=1))
    fold.fold_event(
        AgentEndEvent(
            generation=1,
            aborted=False,
            error=format_cut_off_notice("owner-lost"),
            cut_off=render_cut_off_reason("owner-lost"),
            cut_off_cause="owner-lost",
        )
    )
    assert fold.projection.streaming is False
    assert fold.projection.stop_reason == "aborted"

    # The control: a clean end still reads as a completion, or every finished
    # turn would offer a pointless resume.
    clean = ProjectionFold(SessionProjection(session_id="clean-phone", pid=1))
    clean.fold_event(AgentEndEvent(generation=1))
    assert clean.projection.stop_reason == "completed"


def test_a_terminal_dictation_frame_queues_the_phone_row_and_the_activity_line() -> None:
    """The phone is a consumer of the same frames, and it kept the same lie.

    A call queued behind a long sibling was rendered ``dictating <tool>`` on the
    phone for the sibling's whole run — the row AND the activity line above it —
    because the compose frame was the last thing that surface heard about the
    call. The producer now sends one more frame saying the dictation is over,
    and the phone has to read it as the TUI does: the row is ``queued`` (live,
    waiting, executing nothing) and the line says what the harness is waiting
    for rather than what the model finished writing.
    """
    fold = make_fold()
    fold.fold_event(AgentStartEvent(generation=1))
    fold.fold_event(
        ToolCallComposeEvent(tool_call_id="call_wake", tool_name="wake", argument_bytes=14)
    )
    row = [entry for entry in fold.projection.transcript if entry.kind == "tool"][0]
    assert row.tool_state == "composing"
    assert fold.projection.activity == "dictating wake"

    fold.fold_event(
        ToolCallComposeEvent(
            tool_call_id="call_wake",
            tool_name="wake",
            argument_bytes=14,
            dictation_complete=True,
        )
    )
    row = [entry for entry in fold.projection.transcript if entry.kind == "tool"][0]
    assert row.tool_state == "queued"
    assert len([entry for entry in fold.projection.transcript if entry.kind == "tool"]) == 1
    assert "waiting to run wake" in fold.projection.activity

    # ...and the call still becomes a running row, then a finished one. `queued`
    # is a state on the way, not a verdict.
    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="call_wake", tool_name="wake", args={"text": "30m"})
    )
    row = [entry for entry in fold.projection.transcript if entry.kind == "tool"][0]
    assert row.tool_state == "running"


def test_a_never_run_verdict_fails_the_phone_row_with_the_reason() -> None:
    """The phone has no retirement pass, so the verdict has to settle it there.

    A planning failure, a duplicate id or a steering skip leaves the row
    announcing the call with nothing else coming: no start, no end. The reason
    rides the terminal compose frame, and it is what the row must show — a row
    that merely stopped saying ``dictating`` would leave the phone unable to tell
    "queued" from "dead".
    """
    fold = make_fold()
    fold.fold_event(AgentStartEvent(generation=1))
    fold.fold_event(
        ToolCallComposeEvent(tool_call_id="call_x", tool_name="wake", argument_bytes=14)
    )
    fold.fold_event(
        ToolCallComposeEvent(
            tool_call_id="call_x",
            tool_name="wake",
            argument_bytes=14,
            dictation_complete=True,
            not_run_reason="Tool not found: wake",
        )
    )
    row = [entry for entry in fold.projection.transcript if entry.kind == "tool"][0]
    assert row.tool_state == "failed"
    assert row.error == "Tool not found: wake"
    assert row.summary == "Tool not found: wake"


def test_a_verdict_for_a_call_that_already_started_is_not_applied() -> None:
    """The phone's own guard, matching the TUI's running registry.

    A relayed or replayed terminal frame can reach a surface after its call's
    start — a seed folded out of order, an attach re-reading the relay — and the
    verdict describes a row that has outgrown it. Without the guard the row of a
    call the user is watching execute was relabelled ``failed`` on the phone,
    and because the terminal frame also carries ``dictation_complete``, the next
    arm would have walked it back to ``queued``: two lies instead of one.
    """
    fold = make_fold()
    fold.fold_event(AgentStartEvent(generation=1))
    fold.fold_event(
        ToolCallComposeEvent(tool_call_id="call_x", tool_name="wake", argument_bytes=14)
    )
    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="call_x", tool_name="wake", args={"text": "30m"})
    )
    running = [entry for entry in fold.projection.transcript if entry.kind == "tool"][0]
    assert running.tool_state == "running"
    was_summary = running.summary

    fold.fold_event(
        ToolCallComposeEvent(
            tool_call_id="call_x",
            tool_name="wake",
            argument_bytes=14,
            dictation_complete=True,
            not_run_reason="Duplicate call id 'call_x' skipped",
        )
    )

    row = [entry for entry in fold.projection.transcript if entry.kind == "tool"][0]
    assert row.tool_state == "running", "a started call is not relabelled never-run"
    assert row.summary == was_summary
    assert row.error == ""


def test_a_duplicate_id_winner_clears_the_losers_failure_text() -> None:
    """The revive path drops the failure TEXT with the failure STATE.

    Two calls can share an id: the loser settles the row with the harness's
    reason and the winner then executes it through this same row. The renderer
    draws ``error`` as a red danger line inside the expansion for ANY state, so
    the phone kept ``Duplicate call id … skipped.`` over a call that had just
    succeeded — and ``hasDetails`` true because of it.
    """
    fold = make_fold()
    fold.fold_event(AgentStartEvent(generation=1))
    fold.fold_event(
        ToolCallComposeEvent(
            tool_call_id="call_dup",
            tool_name="wait",
            argument_bytes=20,
            dictation_complete=True,
            not_run_reason="Duplicate call id 'call_dup' skipped",
        )
    )
    failed = [entry for entry in fold.projection.transcript if entry.kind == "tool"][0]
    assert failed.tool_state == "failed" and failed.error

    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="call_dup", tool_name="wait", args={"text": "1"})
    )
    row = [entry for entry in fold.projection.transcript if entry.kind == "tool"][0]
    assert row.tool_state == "running"
    assert row.error == "", "the reason goes with the state it described"
    assert len([entry for entry in fold.projection.transcript if entry.kind == "tool"]) == 1


def test_the_new_compose_fields_absent_on_the_wire_change_nothing() -> None:
    """BACKWARD COMPATIBILITY: an older runtime omits both new fields.

    The phone's fold reads them off the model, so the older frame arrives with
    today's defaults and must take exactly today's path: a composing row whose
    activity line says the model is dictating. This is the control that says the
    new states are additive rather than a change of meaning.
    """
    fold = make_fold()
    fold.fold_event(AgentStartEvent(generation=1))
    fold.fold_event(
        ToolCallComposeEvent(tool_call_id="call_old", tool_name="bash", argument_bytes=8)
    )
    row = [entry for entry in fold.projection.transcript if entry.kind == "tool"][0]
    assert row.tool_state == "composing"
    assert fold.projection.activity == "dictating bash"


# --- the attach clock: a producer's start instant dates work already in flight
#
# The operator's report: a phone that attaches (or a runtime resumed) onto a
# session with work in flight restarted the elapsed timer from the moment of
# attach, counting up from `0s` — a live tool row and the working band above it
# both reading their own arrival as the work's start. The fold cannot derive
# either instant from the events it sees, because it never saw the event that
# began the work, so both come from the PRODUCER's folded state through
# `reconcile_clocks` (or from the `started_at_epoch` a start event states).
#
# These tests drive the real arithmetic on real clocks — a `time.time()`-based
# epoch 180s in the past, asserted against the age the projection publishes —
# rather than pinning a mocked constant, because the defect lives in the
# plumbing between the two clocks and not in a formula.


class _ClockSession:
    """A session publishing the folded instants an attach reads.

    Both are real protocol members (`session/protocol.py`), answered the way
    `Session` answers them: the phase with its zero in one call, and the
    live-call map with `None` for a call whose producer stated no epoch.
    """

    def __init__(
        self,
        phase: tuple[str, float | None] = ("", None),
        epochs: dict[str, float | None] | None = None,
    ) -> None:
        self._phase = phase
        self._epochs = dict(epochs or {})

    def activity_phase_clock(self) -> tuple[str, float | None]:
        return self._phase

    def live_tool_start_epochs(self) -> dict[str, float | None]:
        return dict(self._epochs)


def _attached(session: Any = None) -> ProjectionFold:
    """The fold a handle builds at attach: history, streaming, then the clocks.

    Every step mirrors the real path in order — `fold_history`, then
    `reconcile_streaming` (the flag a subscribing phone never witnessed), then
    `reconcile_clocks` — because the defect is in what an ATTACH-time fold
    lacks, and a fold built any other way would not reproduce it.
    """
    fold = make_fold()
    fold.fold_history(
        [
            Message.user("what is left on the tenant rollup?"),
            Message.assistant("Four rows. Let me check the table itself."),
        ]
    )
    fold.reconcile_streaming(True)
    if session is not None:
        fold.reconcile_clocks(session)
    return fold


def test_a_late_tool_start_is_dated_from_the_producers_epoch() -> None:
    """The reported band: an attach must not date a running call from itself.

    A start event that reaches a fold built at attach states the call's own
    start epoch, and BOTH readings the phone shows have to come off it — the
    working band's elapsed clock and the row's duration at its end. Measured
    before the fix: both read ~0s and counted up from the attach.
    """
    fold = _attached()
    started = time.time() - 180.0
    fold.fold_event(
        ToolExecutionStartEvent(
            tool_call_id="c1",
            tool_name="bash",
            args={"command": "psql -c 'select count(*) from tenants'"},
            intent="counting tenants",
            started_at_epoch=started,
        )
    )
    assert fold.projection.activity == "counting tenants"
    assert fold.projection.activity_started_s == pytest.approx(180.0, abs=2.0)

    fold.fold_event(
        ToolExecutionEndEvent(
            tool_call_id="c1",
            tool_name="bash",
            result=ToolResult(tool_call_id="c1", content=[TextContent(text="4")], duration_s=200.0),
        )
    )
    row = [entry for entry in fold.projection.transcript if entry.kind == "tool"][0]
    assert row.tool_state == "done"


def test_a_call_that_started_before_the_fold_existed_measures_its_real_duration() -> None:
    """The row's duration, when the fold never saw the call's start at all.

    `live_tool_start_epochs` is the producer's own map for calls in flight, so
    the fold seeds from it at attach; a call that began 180s before the phone
    attached reports 180s when it ends, rather than the time since the attach.
    The end event states no `duration_s` here on purpose: that is the path the
    measurement is on, and it is the one the defect corrupted.
    """
    started = time.time() - 180.0
    fold = _attached(_ClockSession(epochs={"c1": started}))
    fold.fold_event(
        ToolExecutionEndEvent(
            tool_call_id="c1",
            tool_name="bash",
            result=ToolResult(tool_call_id="c1", content=[TextContent(text="4")]),
        )
    )
    row = [entry for entry in fold.projection.transcript if entry.kind == "tool"][0]
    assert row.tool_state == "done"
    assert row.elapsed_s == pytest.approx(180.0, abs=2.0)


def test_a_tool_start_the_fold_never_watched_withholds_the_band() -> None:
    """Rule: never fabricate a start, on the band as well as the row (D7).

    A producer that states no `started_at_epoch` (an older runtime) leaves this
    fold with nothing to date a call it never watched begin: the start event
    arrives late, so the fold's own arrival is not the call's start. Publishing
    it as a KNOWN zero put `0s` counting from the phone's mount on the band while
    the call's own row withheld its duration in the same frame — two answers to
    one state on one screen, which is design round 3's D7. The band now withholds
    too, and the ROW keeps the reading it always had: a call with no epoch and no
    seed measures from this fold's observation of its start, which is the
    behaviour round 1 accepted and pinned.
    """
    fold = _attached(_ClockSession(epochs={"c1": None}))
    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="c2", tool_name="bash", args={}, intent="probing")
    )
    assert fold.projection.activity == "probing"
    assert fold.projection.activity_started_s is None

    fold.fold_event(
        ToolExecutionEndEvent(
            tool_call_id="c1",
            tool_name="bash",
            result=ToolResult(tool_call_id="c1", content=[TextContent(text="")]),
        )
    )
    row = [entry for entry in fold.projection.transcript if entry.kind == "tool"][0]
    assert row.elapsed_s == pytest.approx(0.0, abs=1.0), "a start with no epoch seeds no instant"


def test_a_tool_start_without_an_epoch_the_fold_watched_still_zeroes_the_band() -> None:
    """The other half of D7: a call the fold DID watch begin keeps its clock.

    No `started_at_epoch` and a live edge are different states, and only the
    second may publish a zero. Here the fold sees the model call begin and the
    tool start on its own stream, so the call's start IS the event it is folding
    and the band reads `0s` from that frame — `None` here would be the same
    over-refusal review round 2 found on the phase arms.
    """
    fold = make_fold()
    fold.reconcile_streaming(True)
    fold.fold_event(AgentStartEvent(generation=1))
    fold.fold_event(MessageStartEvent(message=Message.assistant()))
    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="c1", tool_name="bash", args={}, intent="probing")
    )
    assert fold.projection.activity == "probing"
    assert fold.projection.activity_started_s == pytest.approx(0.0, abs=1.0)


def test_a_call_already_in_flight_at_attach_is_settled_exactly_once() -> None:
    """A seeded entry is consumed by its own end event, like an observed one.

    The seed must not leave a second reading behind: the call's end pops the
    instant it was seeded with, so a later end event for the same id measures
    from its own start rather than from the stale seed. Driven with a real
    second call to prove the pop, not with an internal assertion about the map.
    """
    first = time.time() - 180.0
    fold = _attached(_ClockSession(epochs={"c1": first}))
    fold.fold_event(
        ToolExecutionEndEvent(
            tool_call_id="c1",
            tool_name="bash",
            result=ToolResult(tool_call_id="c1", content=[TextContent(text="done")]),
        )
    )
    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="c1", tool_name="bash", args={}, intent="again")
    )
    fold.fold_event(
        ToolExecutionEndEvent(
            tool_call_id="c1",
            tool_name="bash",
            result=ToolResult(tool_call_id="c1", content=[TextContent(text="done")]),
        )
    )
    row = [entry for entry in fold.projection.transcript if entry.kind == "tool"][0]
    assert row.elapsed_s == pytest.approx(0.0, abs=1.0), "the second call is dated by its own start"


def test_the_attach_phase_anchor_dates_a_phase_the_producer_holds() -> None:
    """The half with no tool call behind it, on the arm a real stream reaches.

    A phone that attaches mid-prose has no call to key an instant by, so the
    working line would count from the attach. The producer's folded phase
    instant dates it — and `responding` is where that lands in practice: the
    prose edge is the first non-empty delta of a model call and the producer
    zeroes it once, so a fold reading it at attach is reading the producer's own
    number. (QA round 1's M4-B is this cell driven over a real socket.)
    """
    fold = _attached(_ClockSession(phase=("responding", time.time() - 180.0)))
    fold.fold_event(MessageUpdateEvent(message=Message.assistant(), delta="Here "))
    assert fold.projection.activity == "responding"
    assert fold.projection.activity_started_s == pytest.approx(180.0, abs=2.0)

    # The same for the dictation phase: one zero per batch, held until the batch
    # starts, so a label that revises the sentence still counts from the batch.
    dictating = _attached(_ClockSession(phase=("composing", time.time() - 120.0)))
    dictating.fold_event(
        ToolCallComposeEvent(
            tool_call_id="c1", tool_name="bash", argument_bytes=40, intent="counting tenants"
        )
    )
    assert dictating.projection.activity == "counting tenants"
    assert dictating.projection.activity_started_s == pytest.approx(120.0, abs=2.0)


def test_a_thinking_label_never_adopts_the_attach_anchor() -> None:
    """A phase the producer RE-ZEROES is never dated by a stale instant.

    Every route this fold can take to a `thinking` label is an event at which
    the producer restamps the phase (`message_start`, `turn_end`,
    `tool_execution_end`, `agent_start`, and `message_end` for prose), so an
    instant folded before the attach describes a thinking phase that has already
    ended. Adopting it would pair one phase's zero with another phase's label —
    the pairing `activity_phase_clock` exists to prevent — and the honest answer
    there is the arrival instant, which is what the producer itself holds.

    This is the reviewer's round-1 minor 2 answered in code rather than in
    prose: the arm is not merely defensive, it is refused by rule, and the
    refusal is pinned here on the event that would otherwise be a false 180s.
    """
    fold = _attached(_ClockSession(phase=("thinking", time.time() - 180.0)))
    fold.fold_event(MessageStartEvent(message=Message.assistant()))
    assert fold.projection.activity == "thinking"
    # The refusal is a stale instant REFUSED, not a clock withheld: this event is
    # the phase's own edge (the producer restamps `thinking` here), so the fold
    # publishes a known zero — 0.0, not None — and the phone paints `0s` and
    # counts up from it.
    assert fold.projection.activity_started_s == pytest.approx(0.0, abs=1.0)
    assert fold.projection.activity_started_s is not None


def test_a_phase_mismatch_seeds_nothing() -> None:
    """Rule: the phase must MATCH, or the anchor is not used at all.

    The producer is mid-`thinking`; the fold is about to display a running
    call. Pairing one phase's zero with another phase's label would print a
    confident wrong age, so the mismatch adopts nothing and the clock starts
    where today's code starts it.
    """
    fold = _attached(_ClockSession(phase=("thinking", time.time() - 180.0)))
    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="c1", tool_name="bash", args={}, intent="probing")
    )
    assert fold.projection.activity == "probing"
    assert fold.projection.activity_started_s == pytest.approx(0.0, abs=1.0)


def test_a_running_phase_edge_never_dates_a_call() -> None:
    """D9: the running phase is dated by the call, not by the batch's edge.

    `activity_phase_started_at` for `running` is the start of the batch's FIRST
    call, so a batch that sheds a sibling (the first call finishes, a second
    still runs) would report the shed call's age under the survivor's label.
    The exclusion is why a start with no stated epoch keeps the arrival instant
    even while the producer is folded into the running phase.
    """
    fold = _attached(_ClockSession(phase=("running", time.time() - 180.0)))
    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="c2", tool_name="bash", args={}, intent="still going")
    )
    # No clock rather than the batch's: `running` is outside the adoptable set, so
    # an attach with no stated epoch leaves this label undatable — which is also
    # what the call's own row does in the same frame (design round 3, D7).
    assert fold.projection.activity_started_s is None


def test_the_attach_anchor_never_dates_the_next_turn() -> None:
    """A retired anchor is how one turn's attach cannot date the next turn's.

    The turn that was in flight at attach settles; the next turn's phases are
    observed live, so its prose counts from its own delta. Without the
    retirement the second turn would inherit the first turn's age.
    """
    fold = _attached(_ClockSession(phase=("responding", time.time() - 180.0)))
    fold.fold_event(MessageUpdateEvent(message=Message.assistant(), delta="Here "))
    assert fold.projection.activity_started_s == pytest.approx(180.0, abs=2.0)

    fold.fold_event(AgentEndEvent(generation=1))
    assert fold.projection.activity == ""
    fold.fold_event(AgentStartEvent(generation=2))
    fold.fold_event(MessageUpdateEvent(message=Message.assistant(), delta="Again "))
    assert fold.projection.activity == "responding"
    assert fold.projection.activity_started_s == pytest.approx(0.0, abs=1.0)


def test_every_live_call_in_the_producers_map_is_seeded() -> None:
    """The batch seed: a resumed batch presents SEVERAL live ids at once.

    The single-id tests leave the loop to inspection; a batch is the shape a
    resumed session actually reports, and each call needs its OWN instant — the
    older sibling's age under the newer call's end event is exactly the wrong
    number this path exists to avoid.
    """
    fold = _attached(_ClockSession(epochs={"a": time.time() - 180.0, "b": time.time() - 60.0}))
    for call_id in ("a", "b"):
        fold.fold_event(
            ToolExecutionEndEvent(
                tool_call_id=call_id,
                tool_name="bash",
                result=ToolResult(tool_call_id=call_id, content=[TextContent(text="ok")]),
            )
        )
    rows = {row.tool_call_id: row for row in fold.projection.transcript if row.kind == "tool"}
    assert rows["a"].elapsed_s == pytest.approx(180.0, abs=2.0)
    assert rows["b"].elapsed_s == pytest.approx(60.0, abs=2.0)


def test_an_unusable_phase_answer_is_read_as_cannot_say() -> None:
    """Rule: a probed read must survive a host that answers the WRONG SHAPE.

    The accessor is probed, so what comes back is whatever the host returned —
    `None`, a one-element tuple, a stand-in's own object. Unpacking that raises,
    and the call sits on the unattended attach path (`RuntimeServer._serve`
    ends the runtime on a raise; the app's rebind swallows it and leaves the
    bridge unsubscribed), so a facade with a badly-shaped accessor would cost
    the phone the session — a failure mode the pre-fix code could not have.
    Each shape must be read as "cannot say": no raise, nothing seeded, today's
    behaviour intact.
    """
    for answer in (None, ("only-one",), "responding", SimpleNamespace(phase="responding")):
        session = _ClockSession(epochs={"c1": time.time() - 180.0})
        session.activity_phase_clock = lambda answer=answer: answer  # type: ignore[assignment]
        fold = _attached(session)
        fold.fold_event(
            ToolExecutionStartEvent(tool_call_id="c2", tool_name="bash", args={}, intent="probe")
        )
        assert fold.projection.activity == "probe", answer
        # No instant AND no watched edge: a start the fold never saw begin is
        # published without a clock rather than with one counted from the attach
        # (design round 3, D7) — the same refusal the band owes any work it
        # cannot date.
        assert fold.projection.activity_started_s is None, answer

    # ...and the map half still seeds, so a bad PHASE answer is not a dead
    # accessor for the whole attach.
    session = _ClockSession(epochs={"c1": time.time() - 180.0})
    session.activity_phase_clock = lambda: None  # type: ignore[assignment]
    fold = _attached(session)
    fold.fold_event(
        ToolExecutionEndEvent(
            tool_call_id="c1",
            tool_name="bash",
            result=ToolResult(tool_call_id="c1", content=[TextContent(text="ok")]),
        )
    )
    row = [entry for entry in fold.projection.transcript if entry.kind == "tool"][0]
    assert row.elapsed_s == pytest.approx(180.0, abs=2.0)


def test_a_session_that_cannot_answer_seeds_nothing() -> None:
    """The probed read: a reduced facade must not raise on attach.

    An embedder, a test double or a legacy producer need not implement either
    accessor. Attaching to one must leave today's behaviour in place rather
    than faulting the fold on a phone subscribing mid-turn.
    """
    fold = _attached(SimpleNamespace())
    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="c1", tool_name="bash", args={}, intent="probing")
    )
    assert fold.projection.activity == "probing"
    # Without an accessor the fold cannot know whether this call began before it
    # arrived, so no clock is published (D7). "Nothing seeded" is not "seeded
    # with this fold's arrival": the latter is the fabricated start.
    assert fold.projection.activity_started_s is None


def test_the_phone_epoch_conversion_matches_the_tui_widgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second copy is a deliberate boundary, so pin the two equal.

    `tui/widgets/tool_card.monotonic_from_epoch` and this module's copy must
    divide, negate and clamp identically — a divergence would make one surface
    report an age the other does not for the same producer instant, silently.
    The TUI's helper cannot be imported by the phone (it lives in a Textual
    widget module, and the phone daemon must not pull the widget tree in to
    divide one number), so the coupling is a test: this table is the contract,
    and the import is inside the test body for the same reason the module's own
    Textual imports are call-time.

    Both clocks are frozen — the wall clock too, not just the monotonic one —
    because the two functions read `time.time()` themselves and two invocations
    a microsecond apart differ in the last digits: the assertion is about the
    ARITHMETIC, and only a frozen instant makes an inequality meaningful.
    """
    from local_operator.tui.widgets.tool_card import (
        monotonic_from_epoch as tui_monotonic_from_epoch,
    )

    now_epoch = time.time()
    epochs = [
        now_epoch,
        now_epoch - 0.4,
        now_epoch - 45.0,
        now_epoch - 180.0,
        now_epoch - 3_602_400.0,
        # a producer whose clock is AHEAD of ours: the clamp is the interesting
        # half of the arithmetic and the half a future refactor would drop
        now_epoch + 60.0,
        # the epoch itself, for the guard against a value that never was one
        0.0,
    ]
    frozen_clock = 1_000_000.0
    monkeypatch.setattr(time, "time", lambda: now_epoch)
    for epoch in epochs:
        assert monotonic_from_epoch(epoch, clock=lambda: frozen_clock) == tui_monotonic_from_epoch(
            epoch, clock=lambda: frozen_clock
        ), epoch


class _StubClock:
    """A `time` stand-in the test advances by hand.

    The band's arithmetic has to be DRIVEN, not slept through: a test that waits
    a real second to watch a counter tick costs the suite that second on every
    run, and the repo's epoch tests inject a clock for the same reason. Both
    entries the fold reads are here — `monotonic` for the age it publishes and
    `time` for the epochs the producer stated, plus `time_ns` for the notice ids
    the fold mints — because a stub that moved one without the other would make a
    stamped start and the fold's own arrival disagree by the test's whole drift.
    """

    def __init__(self) -> None:
        self._wall = time.time()
        self._monotonic = 5_000.0

    def advance(self, seconds: float) -> None:
        self._monotonic += seconds
        self._wall += seconds

    def time(self) -> float:
        return self._wall

    def time_ns(self) -> int:
        return int(self._wall * 1_000_000_000)

    def monotonic(self) -> float:
        return self._monotonic


def _freeze(monkeypatch: pytest.MonkeyPatch) -> _StubClock:
    """Point the projection module's `time` at a clock this test owns."""
    import local_operator.mobile.projection as projection_module

    clock = _StubClock()
    monkeypatch.setattr(projection_module, "time", clock)
    return clock


def test_a_watched_phase_edge_publishes_a_known_zero_that_ticks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review round 2, MAJOR 1: a phase the phone WATCHED BEGIN keeps its clock.

    The band's gate is "does an instant exist", not "is the instant greater than
    zero". Every phase edge publishes 0.0, so a gate on the VALUE deleted the
    clock for the whole life of any phase this fold watched begin — the ordinary
    case, including the single running tool call the clock is most needed for.
    The two states are told apart here: a fold's OWN edges publish a known zero
    and count up from it, and
    `test_a_phase_joined_mid_flight_with_no_stated_instant_is_published_without_a_clock`
    below pins the refusal on the case it was meant for.
    """
    clock = _freeze(monkeypatch)
    fold = make_fold()
    fold.fold_event(AgentStartEvent(generation=1))
    assert fold.projection.activity == "thinking"
    assert fold.projection.activity_started_s == 0.0  # known, and not withheld

    # The producer restamps `thinking` at every message_start, so the label is
    # re-derived without moving its zero — and the number the phone reads now
    # describes the phase, not the event that repainted it.
    clock.advance(300)
    fold.fold_event(MessageStartEvent(message=Message.assistant()))
    assert fold.projection.activity_started_s == 300.0

    # The prose edge is this fold's own edge, so it zeroes there too — and a
    # later delta of the SAME phase neither re-dates it nor withholds it: the
    # wire carries the phase's own zero, and the phone's 1 Hz tick runs it
    # forward between repaints (the component test pins that half). This is the
    # state the previous head rendered with no digits for the phase's whole
    # life.
    fold.fold_event(MessageUpdateEvent(message=Message.assistant(), delta="Here "))
    assert fold.projection.activity == "responding"
    assert fold.projection.activity_started_s == 0.0

    clock.advance(45)
    fold.fold_event(MessageUpdateEvent(message=Message.assistant(), delta=" more "))
    assert fold.projection.activity_started_s == 0.0
    assert fold.projection.activity_started_s is not None


def test_a_phase_joined_mid_flight_with_no_stated_instant_is_published_without_a_clock() -> None:
    """The attach case the withhold was FOR: no instant, so no digits.

    A producer that names the phase it is in but states no instant for it leaves
    the fold with nothing it can date the label from — that phase was already
    running when this fold arrived, and the event the fold is folding is the
    middle of it. `None` on the wire is that answer, and the phone renders it by
    withholding the digits while keeping the reserved cells. The alternative, the
    fold's own arrival instant, is the fabricated zero this path exists to
    remove.
    """
    for phase, event in (
        (
            "responding",
            MessageUpdateEvent(message=Message.assistant(), delta="mid-prose"),
        ),
        (
            "composing",
            ToolCallComposeEvent(
                tool_call_id="compose:0", tool_name="bash", argument_bytes=8, intent="counting rows"
            ),
        ),
    ):
        fold = _attached(_ClockSession(phase=(phase, None)))
        fold.fold_event(event)
        assert fold.projection.activity_started_s is None, phase


def test_the_queued_labels_carry_no_clock() -> None:
    """TUI parity for the one arm that has no instant at all.

    `frontend_state`'s own comment beside the phase constants is the rule: the
    TUI's working line passes no clock for a call waiting to run, "because there
    is no instant a 'waiting to run' age could honestly count from". Both
    terminal dictation frames are dated `queued` — a phase the producer never
    folds — and carry no number even on a fold that watched the whole batch
    begin, because an announcement is not a start.
    """
    fold = make_fold()
    fold.fold_event(AgentStartEvent(generation=1))
    fold.fold_event(
        ToolCallComposeEvent(
            tool_call_id="compose:0", tool_name="bash", argument_bytes=8, intent="counting"
        )
    )
    assert fold.projection.activity == "counting"

    fold.fold_event(
        ToolCallComposeEvent(
            tool_call_id="compose:0", tool_name="bash", argument_bytes=8, dictation_complete=True
        )
    )
    assert fold.projection.activity == "waiting to run bash"
    assert fold.projection.activity_started_s is None

    fold.fold_event(
        ToolCallComposeEvent(
            tool_call_id="compose:1",
            tool_name="grep",
            argument_bytes=4,
            not_run_reason="Duplicate call id: skipped",
        )
    )
    assert fold.projection.activity == "Duplicate call id: skipped"
    assert fold.projection.activity_started_s is None


def test_a_second_call_in_a_batch_keeps_the_batchs_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clock belongs to the PHASE, which is the TUI's own rule.

    `WorkingBlock`'s clock moves when the phase moves, so a batch relabelling
    from its first call to its second keeps counting the batch rather than
    restarting — and the producer folds exactly ONE zero for a batch, so a fresh
    zero here would be a number that never existed anywhere. The fold tracks its
    own phase for this: only a phase change, a forced edge, or the first label
    may zero the clock.
    """
    clock = _freeze(monkeypatch)
    fold = make_fold()
    fold.fold_event(AgentStartEvent(generation=1))
    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="c1", tool_name="bash", args={}, intent="first")
    )
    assert fold.projection.activity_started_s == 0.0

    clock.advance(10)
    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="c2", tool_name="grep", args={}, intent="second")
    )
    assert fold.projection.activity == "second"
    assert fold.projection.activity_started_s == 10.0

    # A batch's later dictation announcement is the same phase and behaves the
    # same way: the producer does not restamp `composing` either.
    fold.fold_event(
        ToolExecutionEndEvent(
            tool_call_id="c1",
            tool_name="bash",
            result=ToolResult(tool_call_id="c1", content=[TextContent(text="ok")], is_error=False),
        )
    )
    assert fold.projection.activity == "thinking"
    assert fold.projection.activity_started_s == 0.0
    fold.fold_event(
        ToolCallComposeEvent(
            tool_call_id="compose:0", tool_name="read", argument_bytes=8, intent="reading"
        )
    )
    assert fold.projection.activity == "reading"
    fold.fold_event(
        ToolCallComposeEvent(
            tool_call_id="compose:1", tool_name="read", argument_bytes=8, intent="reading too"
        )
    )
    assert fold.projection.activity == "reading too"
    assert fold.projection.activity_started_s == 0.0

    # A PHASE change still zeroes it: the dictation gave way to the model call.
    clock.advance(7)
    fold.fold_event(MessageStartEvent(message=Message.assistant()))
    assert fold.projection.activity == "thinking"
    assert fold.projection.activity_started_s == 0.0


def test_a_late_live_call_map_answer_is_read_as_cannot_say() -> None:
    """Review round 2, MINOR 2: the map read is shape-checked, like the pair.

    Both attach-time reads are PROBED, so what comes back is whatever the host
    answered. The phase half was shape-checked in round 1; the live-call map was
    not, and `dict(answer or {})` raised on every shape that is not a mapping —
    on the same unattended path (`RuntimeServer._serve`), where a raise ends the
    runtime. A non-mapping answer means "cannot say", and the phase half of the
    same attach must still be read.
    """
    for answer in (None, 3.5, "ab", SimpleNamespace(tool_call_id="c1")):
        session = _ClockSession(phase=("responding", time.time() - 180.0))
        session.live_tool_start_epochs = lambda answer=answer: answer  # type: ignore[method-assign]
        fold = _attached(session)
        fold.fold_event(MessageUpdateEvent(message=Message.assistant(), delta="attaching "))
        assert fold.projection.activity == "responding", answer
        assert fold.projection.activity_started_s == pytest.approx(180.0, abs=2.0), answer

        # The map half seeds nothing, so a call this fold never saw start still
        # measures from the fold's own arrival at the end event (today's
        # behaviour) rather than raising before it gets there.
        fold.fold_event(
            ToolExecutionStartEvent(tool_call_id="c9", tool_name="bash", args={}, intent="probing")
        )
        fold.fold_event(
            ToolExecutionEndEvent(
                tool_call_id="c9",
                tool_name="bash",
                result=ToolResult(
                    tool_call_id="c9", content=[TextContent(text="ok")], is_error=False
                ),
            )
        )
        row = [entry for entry in fold.projection.transcript if entry.tool_call_id == "c9"][-1]
        assert row.elapsed_s == pytest.approx(0.0, abs=1.0), answer


def test_refreshing_the_band_age_redates_a_frozen_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review round 3, MAJOR 1: the age is re-dated when a frame is built.

    ``activity_started_s`` is written when the phase MOVES, and long phases are
    exactly the ones with no band event in them — prose streams deltas that do
    not re-enter the label's arm, and a running call has no arm at all. A frame
    built mid-phase therefore carried the age as of the last edge, so a viewer
    attaching then was served a stale number: at a known zero, `0s` counting from
    its own mount — the operator-reported defect, on the surface this PR exists
    for. Re-dating happens at the moment the number becomes an answer.
    """
    clock = _freeze(monkeypatch)
    fold = make_fold()
    fold.reconcile_streaming(True)
    fold.fold_event(AgentStartEvent(generation=1))
    fold.fold_event(MessageStartEvent(message=Message.assistant()))
    fold.fold_event(MessageUpdateEvent(message=Message.assistant(), delta="Here "))
    assert fold.projection.activity == "responding"
    assert fold.projection.activity_started_s == 0.0

    # Five minutes of prose with no band event: the stored reading is stale, and
    # the refresh is what a viewer attaching now must be served instead.
    clock.advance(300)
    assert fold.projection.activity_started_s == 0.0, "the stored snapshot does not move on its own"
    fold.redate_from_phase()
    assert fold.projection.activity_started_s == 300.0

    # An UNKNOWN instant stays unknown through the same call, and a fold that
    # owns no instant writes nothing at all: `_set_activity` published the
    # absence, and this may neither turn it into a zero nor clear a value that
    # belongs to another fold sharing the projection object (review round 4,
    # BLOCKER 1).
    fold.fold_event(AgentEndEvent(generation=1))
    assert fold.projection.activity_started_s is None
    clock.advance(45)
    fold.redate_from_phase()
    assert fold.projection.activity_started_s is None


def test_the_wire_runs_the_age_forward_from_its_arrival_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The COPY's half of the same fix: the daemon between the runtime's frames.

    The daemon holds a deserialized projection — no fold, no producer instant —
    so it cannot recompute the age. What it can do is remember WHEN the reading
    arrived and run it forward on its own monotonic clock, which is the same
    one-shot discipline ``monotonic_from_epoch`` applies to a producer's epoch:
    only the interval already elapsed comes from the other process. Without it a
    phone attaching to a session whose runtime has been quiet for minutes is
    served the age from the last frame it sent.
    """
    import local_operator.mobile.daemon as daemon_module
    import local_operator.mobile.types as types_module

    clock = _StubClock()
    monkeypatch.setattr(types_module, "time", clock)
    record = SessionRecord(
        pid=1,
        kind="tui",
        session_id="s1",
        conversation_name="",
        cwd="",
        model_label="",
        control_port=0,
        control_key="",
    )
    ingested = _projection_from_json(
        {"session_id": "s1", "pid": 1, "activity": "responding", "activity_started_s": 0.0},
        record,
    )
    assert ingested.activity_started_s == 0.0
    clock.advance(90)
    frame = daemon_module._projection_frame(ingested)
    assert frame["activity_started_s"] == pytest.approx(90.0, abs=0.1)

    # A projection whose instant is UNKNOWN crosses unchanged: `None` is not a
    # zero, and nothing on this path may invent one.
    unknown = _projection_from_json(
        {"session_id": "s1", "pid": 1, "activity": "responding", "activity_started_s": None},
        record,
    )
    clock.advance(90)
    assert daemon_module._projection_frame(unknown)["activity_started_s"] is None

    # And a projection that never crossed a wire (a durable rebuild) has no
    # reference to run from: its value is left exactly as it was found.
    rebuilt = SessionProjection(
        session_id="s1", pid=1, activity="responding", activity_started_s=12.0
    )
    clock.advance(90)
    daemon_module._projection_frame(rebuilt)
    assert rebuilt.activity_started_s == 12.0


def test_a_roster_row_with_no_age_publishes_no_age() -> None:
    """Design round 3, D8: a child the roster cannot date has NO age, not zero.

    The drill-in renders ``SubagentRow.elapsed_s`` through the same
    ``WorkingLine`` gate as the band, so a plain float made one value mean both
    "the child began this instant" and "this roster has no age for it" — and the
    phone painted `0s` counting from the viewer's own mount where the TUI
    withholds the number (``transcript.py``: "the number is withheld rather than
    invented — ``clock=False``"). The wire now carries the absence, and a child
    whose row DOES have an age keeps it.
    """
    node = SimpleNamespace(
        job_id="c1",
        label="child",
        parent_job_id=None,
        session_id="s-child",
        prompt="audit the rollout",
        launch_message_id="m1",
        effort="high",
        agent_role="reviewer",
    )

    def lifecycle(age_s: float | None) -> SimpleNamespace:
        return SimpleNamespace(
            job_id="c1",
            label="child",
            status="running",
            age_s=age_s,
            result_text=None,
            error_text=None,
        )

    class Registry:
        """The members the roster fold reads, answering like the real one.

        ``roster_pass`` returns ``self`` because the fold now reads one pass;
        this fake IS the pass, so the same three members answer (see
        ``SubagentComms.roster_pass``).
        """

        def __init__(self, age_s: float | None) -> None:
            self._age = age_s

        def roster_pass(self) -> "Registry":
            return self

        def roster(self) -> list[Any]:
            return [lifecycle(self._age)]

        def lifecycles(self) -> dict[str, Any]:
            # The read the fold actually makes (``RosterPass.lifecycles``).
            return {"c1": lifecycle(self._age)}

        def nodes(self) -> list[Any]:
            return [node]

        def job(self, job_id: str) -> SimpleNamespace:
            return SimpleNamespace(agent_role="reviewer", model_label="", latest_details={})

    undated = make_fold()
    undated.set_subagent_details(Registry(None))
    assert undated.projection.subagents[0].elapsed_s is None

    dated = make_fold()
    dated.set_subagent_details(Registry(12.0))
    assert dated.projection.subagents[0].elapsed_s == pytest.approx(12.0)


def test_set_subagent_details_reads_ONE_pass_and_publishes_the_same_fields() -> None:
    """The fold must not walk the registry once per collection.

    ``set_subagent_details`` runs on EVERY root event (``serving._refresh_state``),
    and each walk is a synchronous read of a registry capped at 256 records, so
    three walks per event — one of which was quadratic underneath, see
    ``SubagentComms.RosterPass`` — was event-loop work paid on the turn's critical
    path. The spy pins the shape: exactly one ``roster_pass()`` and no calls to
    the single-walk methods at all.
    """

    class Jobs:
        def __init__(self) -> None:
            self.rows = {
                "running": SimpleNamespace(
                    status="running",
                    start_time=1_000.0,
                    agent_role="coder",
                    model_label="test/child",
                    latest_details={"progress": "reading"},
                    result_text=None,
                    error_text=None,
                ),
                "settled": SimpleNamespace(
                    status="completed",
                    start_time=900.0,
                    agent_role="reviewer",
                    model_label="test/child",
                    latest_details={},
                    result_text="review posted",
                    error_text=None,
                ),
            }

        def get(self, job_id: str) -> Any:
            return self.rows.get(job_id)

    class Spy(SubagentComms):
        """Counts the walks, so "one pass" is asserted rather than assumed."""

        def __init__(self, session: Any) -> None:
            super().__init__(session)
            self.calls: dict[str, int] = {"roster_pass": 0, "roster": 0, "nodes": 0, "job": 0}

        def roster_pass(self, now: Any = None) -> Any:
            self.calls["roster_pass"] += 1
            return super().roster_pass(now)

        def roster(self) -> Any:
            self.calls["roster"] += 1
            return super().roster()

        def nodes(self) -> Any:
            self.calls["nodes"] += 1
            return super().nodes()

        def job(self, job_id: str) -> Any:
            self.calls["job"] += 1
            return super().job(job_id)

    session = SimpleNamespace(jobs=Jobs())
    comms = Spy(cast(Session, cast(Any, session)))
    comms.record_launch("running", "runner", prompt="go and read")
    comms.record_launch("settled", "settler", prompt="review the diff")
    # A live child, so one row lands as running and the other as completed. The
    # cast is for the ``ChildSession`` protocol's shape, not for the value: the
    # fold only reads ``session_id`` off it.
    comms._records["running"].child = cast(Any, SimpleNamespace(session_id="child-session"))
    comms._records["settled"].outcome = "completed"

    fold = make_fold()
    # The two ``record_launch`` calls above resolve a job row through the public
    # ``job()``; the count below is about the FOLD, so it starts from here.
    comms.calls = dict.fromkeys(comms.calls, 0)

    fold.set_subagent_details(comms)

    assert comms.calls == {"roster_pass": 1, "roster": 0, "nodes": 0, "job": 0}, (
        "the fold must read one pass: a second walk here is per-event event-loop "
        f"work, and this one runs on every root event. Got {comms.calls}"
    )

    rows = {row.job_id: row for row in fold.projection.subagents}
    assert set(rows) == {"running", "settled"}
    assert rows["running"].label == "runner"
    assert rows["running"].status == "running"
    assert rows["running"].activity == "reading"
    assert rows["running"].prompt == "go and read"
    assert rows["running"].agent == "coder"
    assert rows["running"].model_label == "test/child"
    # The child's age comes off the pass's ``ChildInfo``, not the row's default.
    assert rows["running"].elapsed_s is not None
    assert rows["settled"].label == "settler"
    assert rows["settled"].status == "completed"
    assert rows["settled"].result_text == "review posted"
    assert rows["settled"].activity == ""
    # A second refresh must publish the same rows rather than falling back to
    # ``SubagentRow``'s running default for a settled child.
    fold.set_subagent_details(comms)
    again = {row.job_id: row for row in fold.projection.subagents}
    assert again["settled"].status == "completed"
    assert again["settled"].result_text == "review posted"


@pytest.mark.parametrize("size", [1, 7, 20, 50, 100])
@pytest.mark.parametrize("shape", ["siblings", "chain"])
def test_roster_job_snapshot_and_graph_counts_are_deterministic(size: int, shape: str) -> None:
    """Count real fold work without a timing threshold at representative shapes.

    The indexed manager snapshot makes the per-node job lookup O(1) after one
    O(N) pass. Peer and ancestor vectors remain intentionally complete because
    the phone's Agents sheet consumes them; the counts make their unavoidable
    quadratic wire representation explicit rather than pretending to remove it.
    """

    class CountingJobs:
        def __init__(self) -> None:
            self.rows: dict[str, Any] = {}
            self.snapshots = 0
            self.snapshot_rows = 0
            self.gets = 0

        def lookup_snapshot(self) -> dict[str, Any]:
            self.snapshots += 1
            self.snapshot_rows += len(self.rows)
            return dict(self.rows)

        def get(self, job_id: str) -> Any:
            self.gets += 1
            return self.rows.get(job_id)

    manager = CountingJobs()
    session = SimpleNamespace(jobs=manager)
    comms = SubagentComms(cast(Session, cast(Any, session)))
    for index in range(size):
        job_id = f"job-{index:03d}"
        parent_id = None if shape == "siblings" or index == 0 else f"job-{index - 1:03d}"
        manager.rows[job_id] = SimpleNamespace(
            status="running",
            start_time=1_000.0,
            agent_role="task",
            model_label="",
            latest_details={},
            result_text=None,
            error_text=None,
        )
        comms.record_launch(job_id, job_id, parent_job_id=parent_id)

    # Ignore launch-time lookups: the measured operation is one projection fold.
    manager.gets = 0
    manager.snapshots = 0
    manager.snapshot_rows = 0
    fold = make_fold()
    fold.set_subagent_details(comms)

    assert manager.snapshots == 1
    assert manager.snapshot_rows == size
    # Existing lifecycle/node derivations use three direct gets per record;
    # the indexed per-node resolution adds no N-by-N manager traversal.
    assert manager.gets == 3 * size
    assert len(fold.projection.subagents) == size
    if shape == "siblings":
        assert sum(len(row.peer_ids) for row in fold.projection.subagents) == size * (size - 1)
        assert sum(len(row.ancestor_ids) for row in fold.projection.subagents) == 0
    else:
        assert sum(len(row.peer_ids) for row in fold.projection.subagents) == 0
        assert sum(len(row.ancestor_ids) for row in fold.projection.subagents) == (
            size * (size - 1) // 2
        )


def test_roster_index_canonicalizes_attempt_alias_before_manager_lookup() -> None:
    """Resolve a stale attempt through the current id before probing managers."""
    current = SimpleNamespace(
        status="running",
        start_time=1_000.0,
        agent_role="root",
        model_label="",
        latest_details={},
        result_text=None,
        error_text=None,
    )
    later = SimpleNamespace(
        status="running",
        start_time=1_000.0,
        agent_role="child",
        model_label="",
        latest_details={},
        result_text=None,
        error_text=None,
    )

    class DuckJobs:
        """A legacy manager with only point lookup, as extension hosts may use."""

        def __init__(self, rows: dict[str, Any]) -> None:
            self.rows = rows

        def get(self, job_id: str) -> Any:
            return self.rows.get(job_id)

    class IndexedJobs(DuckJobs):
        def lookup_snapshot(self) -> dict[str, Any]:
            return dict(self.rows)

    root = DuckJobs({"current": current})
    # A stale alias row in a later manager must not shadow the canonical root
    # row; the legacy resolver looked up the canonical record id in both.
    child = IndexedJobs({"previous": later})
    comms = SubagentComms(cast(Session, cast(Any, SimpleNamespace(jobs=root))))
    comms.record_launch("current", "current")
    comms._aliases["previous"] = "current"
    comms._records["current"].child = cast(
        Any, SimpleNamespace(jobs=child, session_id="child-session")
    )

    read = comms.roster_pass()
    # The root has only the canonical row. The later child has a stale row that
    # would incorrectly win if the index probed the original alias first.
    assert read.job("previous") is current


def test_roster_index_probes_duck_typed_manager_once_for_missing_id() -> None:
    """An unindexed miss visits each extension manager once, not twice."""

    class DuckJobs:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, job_id: str) -> Any:
            self.calls += 1
            return None

    class IndexedJobs:
        def __init__(self) -> None:
            self.rows: dict[str, Any] = {}

        def lookup_snapshot(self) -> dict[str, Any]:
            return dict(self.rows)

        def get(self, job_id: str) -> Any:
            # RosterPass separately derives each record's running bit from the
            # root manager; this lookup is not the fallback miss being measured.
            return self.rows.get(job_id)

    root = SimpleNamespace(jobs=IndexedJobs())
    fallback = DuckJobs()
    comms = SubagentComms(cast(Session, cast(Any, root)))
    comms.record_launch("known", "known")
    comms._records["known"].child = cast(
        Any, SimpleNamespace(jobs=fallback, session_id="fallback-session")
    )

    read = comms.roster_pass()
    fallback.calls = 0
    assert read.job("missing") is None
    assert fallback.calls == 1


def test_roster_index_falls_back_after_a_raising_optional_manager() -> None:
    """An extension manager failure remains isolated from later child ledgers."""
    child_job = SimpleNamespace(
        status="running",
        start_time=1_000.0,
        agent_role="child",
        model_label="",
        latest_details={},
        result_text=None,
        error_text=None,
    )

    class BrokenJobs:
        calls = 0

        def get(self, job_id: str) -> Any:
            self.calls += 1
            raise RuntimeError("optional host lookup failed")

    class IndexedJobs:
        def __init__(self, rows: dict[str, Any] | None = None) -> None:
            self.rows = rows or {}

        def lookup_snapshot(self) -> dict[str, Any]:
            return dict(self.rows)

        def get(self, job_id: str) -> Any:
            return self.rows.get(job_id)

    root = SimpleNamespace(jobs=IndexedJobs())
    broken = BrokenJobs()
    comms = SubagentComms(cast(Session, cast(Any, root)))
    comms.record_launch("first", "first")
    comms.record_launch("child", "child")
    comms._records["first"].child = cast(
        Any, SimpleNamespace(jobs=broken, session_id="broken-session")
    )
    comms._records["child"].child = cast(
        Any, SimpleNamespace(jobs=IndexedJobs({"child": child_job}), session_id="child-session")
    )
    broken.calls = 0

    assert comms.roster_pass().job("child") is child_job
    assert broken.calls == 1


def test_roster_index_agrees_with_get_and_the_ordered_resolver_on_a_swept_alias() -> None:
    """Snapshot, ``get()`` and the manager-ordered resolver must agree (R1-1).

    ``_sweep_due()`` drops an expired row but LEAVES the ``_aliases`` entry that
    pointed at it, and a later row may reuse the swept alias's key. ``get()``
    resolves aliases BEFORE direct row ids, so a direct row filed under that key
    is unreachable; the index the roster pass builds from ``lookup_snapshot()``
    must not resurrect it over the later manager's real row — the answer the
    historical manager-ordered resolver gives — or a node would be described
    from a row the sender's own ``get()`` cannot see.

    Three-way agreement is the assertion, because each leg can be "right" alone:
    the historical resolver, ``get()`` on the root manager, and the snapshot the
    index is merged from are read by three different callers.
    """

    def row(label: str) -> AsyncJob:
        # Same id as the key it is filed under, so the collision is between the
        # two MANAGERS (and the alias) rather than between a key and a row id.
        return AsyncJob(id="alias", type="task", status="running", start_time=1_000.0, label=label)

    root = AsyncJobManager()
    # The post-sweep state: the alias target is gone while the mapping and a
    # stale direct row under the same key both remain.
    root._aliases["alias"] = "already-swept-target"
    root._jobs["alias"] = row("stale-direct-alias-key")
    later = AsyncJobManager()
    later_row = row("later-manager-current-row")
    later._jobs["alias"] = later_row

    # Leg 1: the alias's absent target wins over the colliding direct row.
    assert root.get("alias") is None
    snapshot = root.lookup_snapshot()
    assert snapshot.get("alias") is root.get("alias")
    assert "alias" not in snapshot

    comms = SubagentComms(cast(Session, cast(Any, SimpleNamespace(jobs=root))))
    comms.record_launch("alias", "alias")
    comms._records["alias"].child = cast(
        Any, SimpleNamespace(jobs=later, session_id="later-session")
    )

    # Leg 2: ``comms.job`` IS the historical manager-ordered resolver (root
    # first, then each live child in insertion order) that the pass replaced.
    assert comms.job("alias") is later_row
    # Leg 3: the roster pass's indexed lookup must answer the same row.
    assert comms.roster_pass().job("alias") is later_row


def test_a_held_child_report_and_a_delivered_one_do_not_read_alike_on_the_phone() -> None:
    """Review round 2 MINOR-B / design D9 / UX U8: the shared home has two callers.

    ``harness/rows.py::held_delivery_notice`` states the rule this decision lives
    by — "the phone and the TUI must agree on the words and the tier" — and Round 1
    added only the TUI branch, so on the phone a held row and a delivered one
    painted as identical ``notice``s. That is the U6 defect unfixed on the surface
    where, by Round 1's own U2 reasoning, this row is load-bearing rather than a
    duplicate of the assistant's answer.

    Both rows are folded together on purpose: they differ by exactly one flag, so
    an assertion on either alone would pass for the wrong reason. The delivered
    row keeps the generic fallback (the phone's pre-existing behaviour); only the
    held row takes the shared decision, and it takes its tier from it.
    """
    from local_operator.harness.jobs import JOB_RESULT_MESSAGE_TYPE
    from local_operator.harness.rows import HELD_DELIVERY_NOTICE
    from local_operator.harness.types import CustomMessage
    from local_operator.mobile.projection import fold_messages_to_entries

    def row(job_id: str, *, held: bool) -> CustomMessage:
        details: dict[str, object] = {
            "job_id": job_id,
            "text": f"background job '{job_id}' completed:\nthe report",
        }
        if held:
            details["held"] = True
        return CustomMessage(
            custom_type=JOB_RESULT_MESSAGE_TYPE, attribution="user", details=details
        )

    entries = fold_messages_to_entries([row("qa-r2", held=True), row("rev-r6", held=False)])
    held, delivered = entries

    assert held.kind == "notice" and delivered.kind == "notice"
    assert held.text.endswith(HELD_DELIVERY_NOTICE.split("\n")[-1])
    assert (
        held.details.get("severity") == "warning"
    ), "the held row takes the tier the shared decision gives it"
    # The delivered row carries the child's text and NO held marker, so the two
    # are distinguishable — which is the whole finding.
    assert HELD_DELIVERY_NOTICE.split("\n")[-1] not in delivered.text
    assert "held when it arrived" in held.text
    assert "held when it arrived" not in delivered.text
