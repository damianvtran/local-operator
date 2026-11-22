"""The tool outline's shape, its keys and its bar summary -- pure, no TUI.

These cases pin the rules the widgets rely on: the run thresholds, every
splitter, failure accounting (``error`` only), the died/live flags that decide
whether a turn bar may exist, key stability under appends, and the summary
budget's degradation order.
"""

import pytest
from rich.cells import cell_len

from local_operator.tui.tool_outline import (
    LooseRow,
    OutlineRow,
    RowKind,
    Run,
    SubGroup,
    ToolFacts,
    ToolState,
    Turn,
    outline_of,
    summary_clauses,
    tool_facts,
)


def user(row_id: str = "u") -> OutlineRow:
    return OutlineRow(row_id, "user")


def tool(
    row_id: str,
    tool_name: str = "bash",
    state: ToolState = "success",
    duration_s: float | None = 0.5,
) -> OutlineRow:
    return OutlineRow(row_id, "tool", tool_name, state, duration_s)


def prose(row_id: str, kind: RowKind = "prose_narration") -> OutlineRow:
    return OutlineRow(row_id, kind)


def notes(*nodes: Run | LooseRow) -> list[tuple[str, ...]]:
    """A turn's nodes written out as what a reader would see, in order."""
    out = []
    for node in nodes:
        if isinstance(node, Run):
            out.append(("run", node.key, *node.tool_ids))
        else:
            out.append(("row", node.id, node.kind))
    return out


def dump(turn: Turn) -> list[tuple[str, ...]]:
    return notes(*turn.nodes)


def run_at(turn: Turn, index: int = 0) -> Run:
    """A turn node asserted to be a run -- the union narrowed once, here."""
    node = turn.nodes[index]
    assert isinstance(node, Run)
    return node


def subgroup_in(run: Run, index: int) -> SubGroup:
    child = run.children[index]
    assert isinstance(child, SubGroup)
    return child


def cells(facts: ToolFacts, budget: int, noanswer: bool = False, **extra) -> str:
    return "".join(clause.text for clause in summary_clauses(facts, budget, noanswer, **extra))


def test_two_back_to_back_calls_stay_loose_rows() -> None:
    turn = outline_of([user(), tool("t1"), tool("t2"), prose("a", "prose_final")]).turns[0]

    assert dump(turn) == [("row", "t1", "tool"), ("row", "t2", "tool")]


def test_three_back_to_back_calls_make_one_run_keyed_on_the_first() -> None:
    turn = outline_of(
        [user(), tool("t1"), tool("t2"), tool("t3"), prose("a", "prose_final")]
    ).turns[0]

    assert dump(turn) == [("run", "tf:run:t1", "t1", "t2", "t3")]
    assert turn.key == "tf:turn:u"


def test_three_calls_of_one_tool_stay_plain_children_of_the_run() -> None:
    turn = outline_of(
        [user(), tool("t1"), tool("t2"), tool("t3"), prose("a", "prose_final")]
    ).turns[0]
    run = turn.nodes[0]

    assert isinstance(run, Run)
    assert run.subgroups == ()
    assert run.children == ("t1", "t2", "t3")


def test_four_calls_of_one_tool_become_a_subgroup() -> None:
    turn = outline_of(
        [user(), *(tool(f"t{i}") for i in range(1, 5)), prose("a", "prose_final")]
    ).turns[0]
    run = turn.nodes[0]

    assert isinstance(run, Run)
    assert subgroup_in(run, 0).key == "tf:sub:t1"
    assert subgroup_in(run, 0).tool_ids == ("t1", "t2", "t3", "t4")
    assert run.tool_ids == ("t1", "t2", "t3", "t4")
    assert [group.key for group in run.subgroups] == ["tf:sub:t1"]


def test_subgroups_split_on_a_tool_change_and_need_four_in_a_row() -> None:
    turn = outline_of(
        [
            user(),
            tool("t1", "browser"),
            tool("t2", "browser"),
            tool("t3", "browser"),
            tool("t4", "bash"),
            tool("t5", "browser"),
            tool("t6", "browser"),
            tool("t7", "browser"),
            tool("t8", "browser"),
            tool("t9", "browser"),
            prose("a", "prose_final"),
        ]
    ).turns[0]
    run = turn.nodes[0]

    assert isinstance(run, Run)
    # Three browser calls are only a rhythm at four; the five after the bash
    # call are a sub-group of their own, keyed on the first of THAT span.
    assert run.children[:4] == ("t1", "t2", "t3", "t4")
    assert subgroup_in(run, 4).key == "tf:sub:t5"
    assert subgroup_in(run, 4).tool_ids == ("t5", "t6", "t7", "t8", "t9")


def test_narration_splits_a_run_in_two() -> None:
    turn = outline_of(
        [
            user(),
            tool("t1"),
            tool("t2"),
            tool("t3"),
            prose("n"),
            tool("t4"),
            tool("t5"),
            tool("t6"),
            prose("a", "prose_final"),
        ]
    ).turns[0]

    assert dump(turn) == [
        ("run", "tf:run:t1", "t1", "t2", "t3"),
        ("row", "n", "prose_narration"),
        ("run", "tf:run:t4", "t4", "t5", "t6"),
    ]


def test_an_ask_or_image_splits_a_run() -> None:
    turn = outline_of(
        [
            user(),
            tool("t1"),
            tool("t2"),
            tool("t3"),
            OutlineRow("ask", "other_visible"),
            tool("t4"),
            tool("t5"),
            tool("t6"),
            prose("a", "prose_final"),
        ]
    ).turns[0]

    assert dump(turn) == [
        ("run", "tf:run:t1", "t1", "t2", "t3"),
        ("row", "ask", "other_visible"),
        ("run", "tf:run:t4", "t4", "t5", "t6"),
    ]


def test_a_call_awaiting_approval_splits_the_run_and_stays_visible() -> None:
    turn = outline_of(
        [
            user(),
            tool("t1"),
            tool("t2"),
            tool("t3", "bash", "waiting"),
            tool("t4"),
            tool("t5"),
            tool("t6"),
            prose("a", "prose_final"),
        ]
    ).turns[0]

    assert dump(turn) == [
        ("row", "t1", "tool"),
        ("row", "t2", "tool"),
        ("row", "t3", "tool"),
        ("run", "tf:run:t4", "t4", "t5", "t6"),
    ]


def test_appending_rows_leaves_every_earlier_key_and_turn_unchanged() -> None:
    records = [
        user("u1"),
        tool("t1"),
        tool("t2"),
        tool("t3"),
        prose("a1", "prose_final"),
        user("u2"),
        tool("t4"),
        tool("t5"),
        tool("t6"),
    ]
    before = outline_of(records, live_from="u2")
    after = outline_of([*records, prose("n2"), tool("t7"), tool("t8"), tool("t9")], live_from="u2")
    grown = outline_of([*records, tool("t7")], live_from="u2")

    assert len(before.turns) == len(after.turns) == 2
    assert after.turns[0] == before.turns[0]
    assert after.turns[0].key == "tf:turn:u1"
    assert run_at(after.turns[0]).key == "tf:run:t1"
    assert after.turns[1].key == before.turns[1].key == "tf:turn:u2"
    # Nothing already on screen changed: narration split the appended calls
    # into a run of their own, which keys on the row that opened IT.
    assert after.turns[1].nodes[0] == before.turns[1].nodes[0]
    assert run_at(after.turns[1], 2).key == "tf:run:t7"
    # A call appended INTO the open run leaves its key and its first rows
    # where they were; only the facts it now has to state grow.
    assert run_at(grown.turns[1]).key == "tf:run:t4"
    assert run_at(grown.turns[1]).tool_ids[:3] == ("t4", "t5", "t6")
    assert run_at(grown.turns[1]).facts.count == 4


def test_facts_count_kinds_in_first_appearance_order_and_failures() -> None:
    rows = [
        tool("t1", "bash"),
        tool("t2", "read"),
        tool("t3", "bash", "error"),
        tool("t4", "hub"),
        tool("t5", "send"),
    ]

    facts = tool_facts(rows)

    assert facts.count == 5
    assert facts.kinds == (("bash", 2), ("read", 1), ("hub", 1), ("send", 1))
    assert facts.failed == 1
    assert facts.running_id is None


def test_duration_is_the_sum_of_known_calls_only_and_none_when_none_known() -> None:
    known = tool_facts([tool("t1", duration_s=0.5), tool("t2", duration_s=None)])
    unknown = tool_facts([tool("t1", duration_s=None), tool("t2", duration_s=None)])

    assert known.duration_s == pytest.approx(0.5)
    assert unknown.duration_s is None


def test_a_running_call_is_reported_and_not_counted_in_the_duration() -> None:
    facts = tool_facts([tool("t1", duration_s=0.5), tool("t2", "bash", "running", duration_s=11.0)])

    assert facts.running_id == "t2"
    assert facts.duration_s == pytest.approx(0.5)


def test_an_interrupted_call_is_not_a_failure_and_keeps_its_duration() -> None:
    facts = tool_facts([tool("t1", "bash", "interrupted", duration_s=2.0)])

    assert facts.failed == 0
    assert facts.duration_s == pytest.approx(2.0)


def test_a_failed_call_stays_inside_its_run_and_is_counted_on_it() -> None:
    turn = outline_of(
        [
            user(),
            tool("t1"),
            tool("t2", "bash", "error"),
            tool("t3"),
            prose("a", "prose_final"),
        ]
    ).turns[0]
    run = turn.nodes[0]

    assert isinstance(run, Run)
    assert run.tool_ids == ("t1", "t2", "t3")
    assert run.facts.failed == 1


def test_a_turn_dies_only_when_a_failure_notice_says_so() -> None:
    """No answer alone is not a failure: a steer, a stop and a cut-off stay neutral."""
    silent = outline_of([user(), tool("t1"), tool("t2"), tool("t3")]).turns[0]
    failed = outline_of(
        [
            user(),
            tool("t1"),
            tool("t2"),
            tool("t3"),
            OutlineRow("n", "notice_failed", notice_head="turn failed"),
        ]
    ).turns[0]
    cut_off = outline_of(
        [
            user(),
            tool("t1"),
            tool("t2"),
            OutlineRow("n", "notice_failed", notice_head="turn cut off — provider"),
        ]
    ).turns[0]

    assert not silent.died and silent.final_id is None
    assert failed.died and failed.final_id is None
    assert not cut_off.died, "a cut-off turn must stay neutral"
    assert not silent.has_turn_bar
    assert not failed.has_turn_bar


def test_an_answered_settled_turn_has_a_turn_bar_and_has_not_died() -> None:
    turn = outline_of([user(), tool("t1"), prose("a", "prose_final")]).turns[0]

    assert turn.final_id == "a"
    assert not turn.died
    assert turn.has_turn_bar


def test_a_live_turn_has_no_turn_bar_and_never_reads_as_dead() -> None:
    rows = [user(), tool("t1"), tool("t2")]
    closed = outline_of(rows).turns[0]
    live = outline_of(rows, live_from="u").turns[0]

    # A settled turn with no answer and no failure notice is not "dead": the
    # harness never said it failed (L6). The live one is settling still.
    assert not closed.died and closed.final_id is None
    assert live.live and not live.died
    assert not live.has_turn_bar
    assert live.key == closed.key


def test_a_live_row_below_a_closed_turn_leaves_that_turn_settled() -> None:
    turns = outline_of(
        [user("u1"), tool("t1"), prose("a1", "prose_final"), user("u2"), tool("t2")],
        live_from="u2",
    ).turns

    assert not turns[0].live and turns[0].has_turn_bar
    assert turns[1].live


def test_a_live_from_id_that_is_gone_marks_nothing_live() -> None:
    turn = outline_of([user(), tool("t1"), prose("a", "prose_final")], live_from="gone").turns[0]

    assert not turn.live and turn.has_turn_bar


def test_a_turn_with_exactly_one_run_is_flat() -> None:
    alone = outline_of(
        [user(), tool("t1"), tool("t2"), tool("t3"), prose("a", "prose_final")]
    ).turns[0]
    mixed = outline_of(
        [user(), prose("n"), tool("t1"), tool("t2"), tool("t3"), prose("a", "prose_final")]
    ).turns[0]
    two_runs = outline_of(
        [
            user(),
            tool("t1"),
            tool("t2"),
            tool("t3"),
            prose("n"),
            tool("t4"),
            tool("t5"),
            tool("t6"),
            prose("a", "prose_final"),
        ]
    ).turns[0]

    assert alone.flat
    # A loose narration line does not earn the run a second bar: the turn bar's
    # children are still that run's children.
    assert mixed.flat
    assert not two_runs.flat


def test_turn_facts_cover_loose_calls_not_only_runs() -> None:
    turn = outline_of(
        [
            user(),
            tool("t1"),
            tool("t2"),
            prose("n"),
            tool("t3"),
            tool("t4"),
            tool("t5"),
            prose("a", "prose_final"),
        ]
    ).turns[0]

    assert turn.facts.count == 5
    assert turn.facts.duration_s == pytest.approx(2.5)


def test_a_page_seam_turn_has_no_user_row_and_keys_on_its_first_row() -> None:
    turn = outline_of([tool("t1"), tool("t2"), tool("t3")]).turns[0]

    assert turn.user_id is None
    assert turn.key == "tf:turn:t1"
    assert dump(turn) == [("run", "tf:run:t1", "t1", "t2", "t3")]


def test_deriving_twice_gives_the_same_outline() -> None:
    rows = [
        user("u1"),
        tool("t1"),
        tool("t2"),
        tool("t3"),
        prose("a1", "prose_final"),
        user("u2"),
        tool("t4", "read"),
    ]

    assert outline_of(rows, live_from="u2") == outline_of(list(rows), live_from="u2")


def test_summary_names_actions_shortened_kinds_and_deduped_messages() -> None:
    facts = ToolFacts(
        count=5,
        kinds=(("bash", 2), ("hub", 1), ("send", 1), ("read", 1)),
        failed=0,
        duration_s=None,
        running_id=None,
    )

    assert cells(facts, 40) == "5 actions · shell, messages, read"


def test_summary_uses_the_plural_callback_for_the_count_clause() -> None:
    facts = ToolFacts(1, (("bash", 1),), 0, None, None)

    assert cells(facts, 40, plural=lambda n: f"{n} TAT") == "1 TAT · shell"
    assert cells(facts, 40) == "1 action · shell"


def test_summary_degrades_on_an_80_cell_budget_and_holds_the_failure_count() -> None:
    facts = ToolFacts(
        count=15,
        kinds=(
            ("browser_snapshot", 3),
            ("playwright_click", 3),
            ("filesystem_readdir", 2),
            ("terminal_write", 2),
            ("web_search", 2),
            ("imagegen", 3),
        ),
        failed=2,
        duration_s=None,
        running_id=None,
    )

    wide = cells(facts, 100)
    narrow = cells(facts, 80)

    assert "browser_snapshot, playwright_click, filesystem_readdir +3" in wide
    # The wide summary really does overrun the narrow budget, so the narrower
    # one cannot be a no-op: the third kind must be the thing that gives way.
    assert len(wide) > 80
    assert "browser_snapshot, playwright_click +4" in narrow
    assert len(narrow) <= 80
    assert "✗ 2 failed" in wide and "✗ 2 failed" in narrow


def test_summary_never_drops_the_failure_count() -> None:
    facts = ToolFacts(9, (("bash", 3), ("read", 6)), 2, None, None)

    for budget in (0, 10, 25, 40):
        assert cells(facts, budget).endswith("✗ 2 failed")
    assert cells(facts, 0) == "9 actions · ✗ 2 failed"


def test_summary_flags_failure_ink_only_when_the_turn_died() -> None:
    facts = ToolFacts(3, (("bash", 3),), 1, None, None)

    roles = {clause.text: clause.role for clause in summary_clauses(facts, 40, False)}
    dead = {clause.text: clause.role for clause in summary_clauses(facts, 40, True)}

    assert roles["✗ 1 failed"] == "failed_muted"
    assert dead["✗ 1 failed"] == "danger"
    assert all(
        clause.role in {"muted", "failed_muted", "dim", "danger"}
        for clause in summary_clauses(facts, 40, True)
    )


def test_a_wide_script_summary_drops_kinds_instead_of_ellipsizing() -> None:
    """The budget is in CELLS: a CJK count clause is twice its character count.

    ``_cells`` summed ``len``, so a zh-CN bar believed ``16项操作`` cost five
    cells and kept three kinds on a budget of 34; the row then fell through to
    the bar's ellipsis cut with the kinds still in front of the failure tally,
    which is the one clause the design never drops. Counted with ``cell_len``,
    the kinds give way first — the design's degradation order — and the tally
    survives at the same budget.
    """
    facts = ToolFacts(16, (("bash", 10), ("read", 3), ("grep", 3)), 3, None, None)
    plural = lambda n: f"{n}项操作"  # noqa: E731
    tally = lambda n: f"✗ {n}项失败"  # noqa: E731

    everything = cells(facts, 10_000, plural=plural, failed=tally)
    assert "read" in everything and "grep" in everything
    # The under-count, stated exactly: the all-kinds summary FITS a 34-cell
    # budget by character count and overruns it by six cells on screen.
    assert len(everything) <= 34 < cell_len(everything)

    rendered = cells(facts, 34, plural=plural, failed=tally)
    assert cell_len(rendered) <= 34, rendered
    assert "read" not in rendered and "grep" not in rendered, rendered
    assert rendered.endswith("✗ 3项失败"), rendered
