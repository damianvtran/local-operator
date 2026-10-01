"""The ``monitor`` tool: createIf gate, card, and the create/list/cancel ops.

The ops run against a REAL :class:`MonitorScheduler` (stub callbacks), because
the arm flow — shape validation, dedupe, caps, ids — lives there, and a fake
would test the fake. The sentence-shapes the tool returns are pinned here;
the scheduler's own behaviour is pinned in ``tests/unit/monitors``.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.monitors.scheduler import CheckOutcome, MonitorScheduler
from local_operator.monitors.settings import MonitorSettings
from local_operator.monitors.spec import MonitorSpec
from local_operator.tools import builtin
from local_operator.tools.registry import create_tools

NOW = 1_756_000_000_000


async def _run(monitor: MonitorSpec) -> CheckOutcome:
    return {"text": "x", "error": None}


def make_scheduler(tmp_path: Any) -> MonitorScheduler:
    return MonitorScheduler(
        now=lambda: NOW,
        config_dir=tmp_path / "cfg",
        session_id="s",
        settings=MonitorSettings(),
        validate=lambda tool, args: None,
        run_check=_run,
        deliver=lambda delivery: None,
        persist=lambda monitors: None,
        uniform=lambda low, high: low,
    )


def make_context(tmp_path: Any, scheduler: MonitorScheduler | None) -> ToolContext:
    return ToolContext(cwd=str(tmp_path), session_id="s", monitor_scheduler=scheduler)


def test_builder_returns_none_without_scheduler(tmp_path: Any) -> None:
    assert builtin.build_monitor_tool(ToolContext(cwd=str(tmp_path))) is None
    assert "monitor" not in {t.name for t in create_tools(ToolContext(cwd=str(tmp_path)))}


def test_the_card_matches_the_house_conventions(tmp_path: Any) -> None:
    scheduler = make_scheduler(tmp_path)
    try:
        tool = builtin.build_monitor_tool(make_context(tmp_path, scheduler))
        assert tool is not None
        assert tool.name == "monitor"
        # write tier: create persists and arms unattended future calls; list is
        # a read and must not prompt.
        assert tool.approval_tier == "write"
        assert tool.call_approval_tier is not None
        assert tool.call_approval_tier({"op": "list"}) == "read"
        assert tool.call_approval_tier({"op": "create"}) == "write"
        assert tool.call_approval_tier({"op": "cancel"}) == "write"
        assert tool.concurrency == "exclusive"
        assert tool.interruptible is False
    finally:
        scheduler.dispose()


def test_describe_approval_names_the_watched_call(tmp_path: Any) -> None:
    describe = builtin._describe_monitor_approval
    assert (
        describe(
            {
                "op": "create",
                "tool": "bash",
                "arguments": {"command": "gh pr view 1710 --json state"},
                "every": "60s",
            },
            "/work",
        )
        == "watch: bash `gh pr view 1710 --json state` every 60s"
    )
    assert (
        describe(
            {
                "op": "create",
                "tool": "web_fetch",
                "arguments": {"url": "https://x/y"},
                "until": "2030-01-01T00:00:00",
            },
            "/work",
        )
        == 'watch: web_fetch {"url":"https://x/y"} every 60s until 2030-01-01T00:00:00'
    )
    assert describe({"op": "cancel", "id": "m2"}, "/work") == "cancel monitor: m2"
    assert describe({"op": "list"}, "/work") == "monitor: list"


@pytest.mark.asyncio
async def test_create_list_cancel_round_trip(tmp_path: Any) -> None:
    scheduler = make_scheduler(tmp_path)
    try:
        context = make_context(tmp_path, scheduler)
        tools = {t.name: t for t in create_tools(context)}
        monitor = tools["monitor"]

        created = await monitor.execute(
            "c",
            {
                "op": "create",
                "name": "date-watch",
                "tool": "bash",
                "arguments": {"command": "date -u"},
                "every": "90s",
            },
            None,
            None,
            context,
        )
        assert created.is_error is False
        assert created.text is not None and created.text.startswith(
            "Armed monitor 'date-watch' (m1) — ticks run only while this session is open. "
            "bash `date -u` every 1m30s, durable."
        )
        assert "you'll be told only what changes" in created.text
        assert len(scheduler.monitors) == 1

        listed = await monitor.execute("c", {"op": "list"}, None, None, context)
        assert listed.text is not None and "date-watch" in listed.text
        assert "m1" in listed.text and "every 1m30s" in listed.text
        # QA round-1 observation 1: the row names the watched call, not only
        # the tool, so two same-tool monitors are told apart by what they
        # watch.
        assert "`date -u`" in listed.text

        cancelled = await monitor.execute("c", {"op": "cancel", "id": "m1"}, None, None, context)
        assert cancelled.is_error is False and cancelled.text == "Cancelled monitor 'm1'."
        assert scheduler.monitors == ()
    finally:
        scheduler.dispose()


@pytest.mark.asyncio
async def test_list_is_empty_and_marked_useless(tmp_path: Any) -> None:
    scheduler = make_scheduler(tmp_path)
    try:
        context = make_context(tmp_path, scheduler)
        monitor = builtin.build_monitor_tool(context)
        assert monitor is not None
        listed = await monitor.execute("c", {"op": "list"}, None, None, context)
        assert listed.text == "No monitors."
        assert listed.details is not None and listed.details.get("useless") is True
    finally:
        scheduler.dispose()


@pytest.mark.asyncio
async def test_create_requires_a_tool(tmp_path: Any) -> None:
    scheduler = make_scheduler(tmp_path)
    try:
        context = make_context(tmp_path, scheduler)
        monitor = builtin.build_monitor_tool(context)
        assert monitor is not None
        result = await monitor.execute("c", {"op": "create", "name": "x"}, None, None, context)
        assert result.is_error is True
        assert "'create' requires 'tool'" in (result.text or "")
    finally:
        scheduler.dispose()


@pytest.mark.asyncio
async def test_a_duplicate_says_already_watches(tmp_path: Any) -> None:
    scheduler = make_scheduler(tmp_path)
    try:
        context = make_context(tmp_path, scheduler)
        monitor = builtin.build_monitor_tool(context)
        assert monitor is not None
        args = {"op": "create", "tool": "bash", "arguments": {"command": "date -u"}}
        first = await monitor.execute("c", dict(args), None, None, context)
        assert first.is_error is False
        second = await monitor.execute("c", dict(args), None, None, context)
        assert second.is_error is False
        assert second.text == "Monitor 'bash' (m1) already watches that call every 1m."
    finally:
        scheduler.dispose()


@pytest.mark.asyncio
async def test_cancel_unknown_id_lists_the_known_ones(tmp_path: Any) -> None:
    scheduler = make_scheduler(tmp_path)
    try:
        context = make_context(tmp_path, scheduler)
        monitor = builtin.build_monitor_tool(context)
        assert monitor is not None
        missing = await monitor.execute("c", {"op": "cancel", "id": "m9"}, None, None, context)
        assert missing.is_error is True
        assert "No monitor with id 'm9' (known: none)" in (missing.text or "")

        await monitor.execute(
            "c",
            {"op": "create", "tool": "bash", "arguments": {"command": "date -u"}},
            None,
            None,
            context,
        )
        missing = await monitor.execute("c", {"op": "cancel", "id": "m9"}, None, None, context)
        assert missing.is_error is True
        assert "(known: m1)" in (missing.text or "")
    finally:
        scheduler.dispose()


@pytest.mark.asyncio
async def test_a_read_only_refusal_flows_through_as_an_error(tmp_path: Any) -> None:
    scheduler = MonitorScheduler(
        now=lambda: NOW,
        config_dir=tmp_path / "cfg",
        session_id="s",
        settings=MonitorSettings(),
        validate=lambda tool, args: 'monitor can\'t watch "eval": arbitrary code.',
        run_check=_run,
        deliver=lambda delivery: None,
        persist=lambda monitors: None,
        uniform=lambda low, high: low,
    )
    try:
        context = make_context(tmp_path, scheduler)
        monitor = builtin.build_monitor_tool(context)
        assert monitor is not None
        result = await monitor.execute(
            "c", {"op": "create", "tool": "eval", "arguments": {"code": "1"}}, None, None, context
        )
        assert result.is_error is True
        assert "can't watch" in (result.text or "")
    finally:
        scheduler.dispose()


# ---------------------------------------------------------------------------
# §D8: the arm receipt's hosting caveat, and §D6: the list row's health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_arm_receipt_states_the_hosting_caveat(tmp_path: Any) -> None:
    """The live store carried arms with ``checks=0``: the operator armed a
    watch, closed the conversation, and nothing told them it would not run. The
    caveat belongs on the receipt, not only in the guide.
    """
    scheduler = make_scheduler(tmp_path)
    try:
        context = make_context(tmp_path, scheduler)
        monitor = builtin.build_monitor_tool(context)
        assert monitor is not None
        created = await monitor.execute(
            "c",
            {
                "op": "create",
                "name": "date-watch",
                "tool": "bash",
                "arguments": {"command": "date -u"},
                "every": "60s",
            },
            None,
            None,
            context,
        )
        text = created.text or ""
        # Design review round 1, D9: the hosting caveat is a FIRST-CLAUSE fact
        # now — a receipt is one card whose collapsed row shows ~90 cells, and
        # at the end of the paragraph it never reached them.
        assert (
            "Armed monitor 'date-watch' (m1) — ticks run only while this session is open." in text
        )
        assert "with one consolidated delta, when it reopens" in text
        assert text.index("ticks run only") < text.index("bash `date -u`")
        # No MCP clause on a non-MCP tool: it would name a failure mode the
        # monitor cannot have.
        assert "server reconnects" not in text
    finally:
        scheduler.dispose()


@pytest.mark.asyncio
async def test_an_mcp_arm_receipt_adds_the_reconnect_clause(tmp_path: Any) -> None:
    scheduler = make_scheduler(tmp_path)
    try:
        context = make_context(tmp_path, scheduler)
        monitor = builtin.build_monitor_tool(context)
        assert monitor is not None
        created = await monitor.execute(
            "c",
            {
                "op": "create",
                "name": "dd-hosts",
                "tool": "mcp__datadog_search_datadog_hosts",
                "arguments": {"query": "up"},
                "every": "60s",
            },
            None,
            None,
            context,
        )
        text = created.text or ""
        assert "If its server reconnects the monitor waits (no failed checks)." in text
        assert "30 minutes" in text
    finally:
        scheduler.dispose()


@pytest.mark.asyncio
async def test_a_list_row_carries_the_health_hint(tmp_path: Any) -> None:
    """A monitor with 0 deliveries after several checks, and one that never
    checked, otherwise read exactly like a healthy row on every surface.
    """
    scheduler = make_scheduler(tmp_path)
    try:
        context = make_context(tmp_path, scheduler)
        monitor = builtin.build_monitor_tool(context)
        assert monitor is not None
        await monitor.execute(
            "c",
            {
                "op": "create",
                "name": "quiet-watch",
                "tool": "bash",
                "arguments": {"command": "date -u"},
                "every": "60s",
            },
            None,
            None,
            context,
        )
        row = scheduler.index_rows()[0]
        # Seven checks, no delivery, and a due instant hours in the past (a
        # session that is not hosting it): both hints are earned.
        rendered = builtin._monitor_row_text(
            {
                **row,
                "checks": 7,
                "deliveries": 0,
                "created_at": NOW - 7_200_000,
                "next_due_at": NOW - 3_600_000,
            },
            NOW,
        )
        assert "[7 checks, 0 deliveries" in rendered
        assert "session not open" in rendered

        # A healthy monitor says nothing extra.
        healthy = builtin._monitor_row_text(
            {**row, "checks": 7, "deliveries": 3, "next_due_at": NOW + 30_000}, NOW
        )
        assert "[" not in healthy
    finally:
        scheduler.dispose()


@pytest.mark.asyncio
async def test_a_glob_with_an_unknown_argument_is_refused_through_the_session(
    tmp_path: Any,
) -> None:
    """The in-session half of the §D2 report, through a REAL Session validator:
    the sentence the agent sees is the validator's own, so the arm path and the
    CLI/desktop path cannot describe one call two ways.
    """
    from local_operator.harness.types import StreamEndEvent
    from local_operator.monitors.spec import MonitorSpec
    from local_operator.tools.registry import create_tools
    from tests.unit.session.test_session import ScriptedStream, make_session

    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    # The tool must be in the session's set for the gate to reach its SHAPE
    # check: an absent tool is refused one sentence earlier (by design).
    glob_tool = [t for t in create_tools(ToolContext(cwd=str(tmp_path))) if t.name == "glob"]
    assert glob_tool, "the glob tool must exist to test its shape check"
    session = make_session(tmp_path, stream, tools=glob_tool)
    try:
        spec = MonitorSpec(
            id="m1",
            name="g",
            tool="glob",
            arguments={"pattern": "*.py", "path": "/tmp"},
            every_ms=60_000,
            created_at=NOW,
        )
        reason = session._validate_monitor_call("glob", spec.arguments)
        assert reason == (
            'monitor can\'t watch "glob": unknown argument(s) "path" — glob accepts: pattern.'
        )
        # A tick re-validates too, so a monitor armed before this build (or one
        # whose tool changed shape) fails with the same sentence rather than a
        # mystery from the tool's own model.
        outcome = await session._run_monitor_check(spec)
        assert "unknown argument" in str(outcome.get("error"))
    finally:
        await session.dispose()
