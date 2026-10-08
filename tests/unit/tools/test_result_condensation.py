"""Tool results that re-bill what the model already has, condensed at the source.

Each result here is appended to the transcript verbatim and re-sent on every
later turn, so its size is paid many times over. The fleet measurements behind
each bound (2026-10-08, 1,143 transcripts) are cited at the constant that sets
it; these tests pin the SHAPES, and that nothing actionable was hidden:

- ``hub op=list`` folds old completed children into a count plus a handle, and
  states the transcript how-to once instead of under every row;
- ``todo`` mutation receipts carry counts and clipped names, while ``view`` and
  the partial-match error keep the full texts the model must echo exactly;
- ``secret op=list`` clips descriptions, keeps every name, spills the full list;
- ``jobs op=peek`` returns only the latest frame of a redrawing watcher;
- ``read spill://…?q=`` no longer returns whole multi-hundred-KB lines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.jobs import AsyncJobManager
from local_operator.harness.types import AbortSignal, AgentTool, ToolContext, ToolResult
from local_operator.tools import builtin
from local_operator.tools.builtin import (
    HUB_LIST_COMPLETED_SHOWN,
    READ_OUTPUT_LIMIT_CHARS,
    _collapse_refreshing_frames,
    _hub_list,
)
from local_operator.tools.registry import create_tools
from local_operator.tools.spill import get_store


def _text(result: ToolResult) -> str:
    return "".join(getattr(block, "text", "") for block in result.content)


async def _call(
    tools: dict[str, AgentTool], name: str, args: dict[str, Any], context: ToolContext
) -> ToolResult:
    return await tools[name].execute("call-1", args, None, None, context)  # type: ignore[operator]


@pytest.fixture
def isolated_spill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The spill store resolves under the config dir per call; keep it here."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    return tmp_path


# --------------------------------------------------------------------------- hub


class _Row:
    """The slice of ``ChildInfo`` the roster renderer reads."""

    def __init__(self, job_id: str, status: str, *, session_id: str | None = None) -> None:
        self.job_id = job_id
        self.label = f"label-{job_id}"
        self.status = status
        self.resumable = status != "running"
        self.age_s = 100.0
        self.detail = None
        self.cut_off_cause = ""
        self.session_id = session_id
        self.last_progress_at = None
        self.ended_by = ""
        self.cancel_reason = ""


class _Comms:
    def __init__(self, rows: list[_Row]) -> None:
        self._rows = rows

    def roster(self) -> list[_Row]:
        return self._rows


def test_hub_list_folds_old_completed_children_and_keeps_every_live_one(isolated_spill):
    rows = [_Row(f"done{i:03d}", "completed", session_id=f"s{i:03d}") for i in range(40)]
    rows.insert(5, _Row("failedjob", "failed", session_id="sfail"))
    rows.append(_Row("livejob", "running"))
    result = _hub_list("c", _Comms(rows), None, ToolContext(cwd=str(isolated_spill)))
    text = _text(result)

    # Running and failed rows are always shown, however old.
    assert "(livejob): running" in text and "(failedjob): failed" in text
    # Only the newest N completed rows are listed in full.
    shown_completed = [f"done{i:03d}" for i in range(40 - HUB_LIST_COMPLETED_SHOWN, 40)]
    assert all(f"({job_id})" in text for job_id in shown_completed)
    assert "(done000)" not in text
    assert f"+ {40 - HUB_LIST_COMPLETED_SHOWN} older completed subagent(s) not shown" in text
    # The folded rows are one read away, rendered the same way.
    details = result.details or {}
    handle = details["spill"]["handle"]
    assert handle in text
    stored = get_store().read_lines(handle, 1, None)
    assert stored is not None and any("(done000)" in line for line in stored[0])
    # The machine payload still carries every child.
    assert details["count"] == 42 and len(details["children"]) == 42
    # The transcript how-to appears ONCE, not under every row.
    assert text.count("lop --resume") == 1
    assert "transcript s039" in text


def test_hub_list_with_few_children_is_not_folded(isolated_spill):
    rows = [_Row(f"j{i}", "completed", session_id=f"s{i}") for i in range(3)]
    result = _hub_list("c", _Comms(rows), None, ToolContext(cwd=str(isolated_spill)))
    assert "not shown" not in _text(result)
    assert "spill" not in (result.details or {})


# -------------------------------------------------------------------------- todo


@pytest.fixture
def todo_env(tmp_path: Path):
    context = ToolContext(cwd=str(tmp_path), session_id="condense-todo")
    tools = {tool.name: tool for tool in create_tools(context)}
    yield tools, context
    builtin.TODO_STORE.pop("condense-todo", None)


LONG = "Implement the very long item text that describes a whole slice of work in detail"


@pytest.mark.asyncio
async def test_todo_mutation_receipts_are_compact(todo_env):
    tools, context = todo_env
    items = [f"{LONG} #{i}" for i in range(12)]
    await _call(tools, "todo", {"op": "init", "items": items}, context)

    done = await _call(tools, "todo", {"op": "done", "items": items}, context)
    text = _text(done)
    assert text.startswith("Marked done 12 item(s): ")
    assert "+4 more" in text and text.endswith("(12/12 resolved).")
    # Names are clipped: the model sent the full texts and still holds them.
    assert LONG not in text
    assert len(text) < 600

    added = await _call(tools, "todo", {"op": "add", "items": [LONG]}, context)
    assert "Added 1 item(s): " in _text(added) and LONG not in _text(added)
    assert _text(added).endswith("(12/13 resolved).")

    dropped = await _call(tools, "todo", {"op": "drop", "items": [LONG]}, context)
    assert _text(dropped).startswith("Dropped 1 item(s): ")


@pytest.mark.asyncio
async def test_todo_miss_error_and_view_keep_full_open_texts(todo_env):
    """The texts the model must echo exactly are never clipped."""
    tools, context = todo_env
    await _call(tools, "todo", {"op": "init", "items": [LONG, "short"]}, context)
    miss = await _call(tools, "todo", {"op": "done", "items": ["short", "ghost"]}, context)
    assert miss.is_error and f"- [ ] {LONG}" in _text(miss)
    view = await _call(tools, "todo", {"op": "view"}, context)
    assert f"- [ ] {LONG}" in _text(view)


@pytest.mark.asyncio
async def test_todo_view_folds_fully_resolved_phases(todo_env):
    tools, context = todo_env
    await _call(
        tools,
        "todo",
        {
            "op": "init",
            "phases": [
                {"phase": "Done phase", "items": ["a", "b"]},
                {"phase": "Open phase", "items": ["c", "d"]},
            ],
        },
        context,
    )
    await _call(tools, "todo", {"op": "done", "phase": "Done phase"}, context)
    await _call(tools, "todo", {"op": "done", "items": ["c"]}, context)

    view = await _call(tools, "todo", {"op": "view"}, context)
    assert _text(view).splitlines() == [
        "Done phase · 2/2 — all resolved",
        "Open phase · 1/2",
        "- [x] c",
        "- [ ] d",
    ]


# ------------------------------------------------------------------------ secret


@pytest.fixture
def secret_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """HOME and the config dir both inside tmp_path, as ``tests/unit/secrets``
    does — the tool opens the default store, so the redirect is asserted."""
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.secrets.keys import secrets_dir

    root = tmp_path / "config"
    root.mkdir()
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv(CONFIG_DIR_ENV, str(root))
    assert secrets_dir().is_relative_to(root), "store escaped the sandbox"
    return root


async def _secret(op: str, **kwargs: object) -> ToolResult:
    from local_operator.tools.secret_tool import execute_secret

    return await execute_secret("c", {"op": op, **kwargs}, AbortSignal(), None, ToolContext())


@pytest.mark.asyncio
async def test_secret_list_clips_descriptions_and_spills_the_full_list(secret_store):
    long_description = "Production API token for the billing integration " * 4
    await _secret("store", name="BILLING_TOKEN", value="v1", description=long_description)
    await _secret("store", name="SHORT", value="v2", description="short one")

    listed = await _secret("list")
    text = _text(listed)
    assert "BILLING_TOKEN" in text and "SHORT" in text
    assert long_description.strip() not in text and "\u2026" in text
    handle = (listed.details or {})["spill"]["handle"]
    assert handle in text and "descriptions clipped to 80 chars" in text
    stored = get_store().read_lines(handle, 1, None)
    assert stored is not None and long_description.strip() in "\n".join(stored[0])


@pytest.mark.asyncio
async def test_a_long_secret_list_keeps_every_name_inline(secret_store):
    names = [f"SERVICE_{i:03d}_ACCESS_TOKEN" for i in range(60)]
    for name in names:
        await _secret("store", name=name, value="v", description="x" * 70)
    text = _text(await _secret("list"))
    assert all(name in text for name in names), "a name hidden behind a handle can be missed"
    assert "descriptions omitted" in text
    assert len(text) < 4_000


@pytest.mark.asyncio
async def test_a_short_secret_list_is_unchanged(secret_store):
    await _secret("store", name="A", value="v", description="first")
    listed = await _secret("list")
    assert _text(listed).splitlines() == [
        "1 stored secret(s) — names only, never values:",
        "  A  first",
    ]
    assert "spill" not in (listed.details or {})


# -------------------------------------------------------------------------- jobs

FRAME = (
    "Refreshing run status every 60 seconds. Press Ctrl+C to quit.\n\n"
    "* v1.0 Build · 123\nTriggered via release about {n} minute ago\n\nJOBS\n"
    "✓ Validate in 9s\n  ✓ Set up job\n  ✓ Complete job\n"
    "{mark} Publish (ID 7)\n  ✓ Set up job\n  {mark} Publish to npm\n\n"
)


def test_refreshing_frames_collapse_to_the_latest():
    text = "".join(FRAME.format(n=n, mark="*" if n < 3 else "✓") for n in range(1, 4))
    latest, elided = _collapse_refreshing_frames(text)
    assert elided == 2
    assert latest.startswith("Refreshing run status") and "about 3 minute" in latest
    assert "about 1 minute" not in latest and "about 2 minute" not in latest


def test_a_clear_screen_sequence_marks_a_redraw():
    latest, elided = _collapse_refreshing_frames("old\nframe\x1b[2Jnew frame\nline")
    assert (latest, elided) == ("new frame\nline", 1)


@pytest.mark.parametrize(
    "text",
    [
        # An ordinary log with a repeated line is not a redraw.
        "\n".join("PASS" if i % 5 == 0 else f"test {i} ok" for i in range(40)),
        "a\nb\na\nc\nd\ne",
        "single line",
    ],
)
def test_ordinary_output_is_left_alone(text):
    assert _collapse_refreshing_frames(text) == (text, 0)


@pytest.mark.asyncio
async def test_peek_returns_the_latest_frame_and_keeps_the_rest_reachable(isolated_spill):
    manager = AsyncJobManager()
    context = ToolContext(cwd=str(isolated_spill), session_id="s", jobs=manager)
    tools = {tool.name: tool for tool in create_tools(context)}

    async def runner(job_id: str, signal: Any, report_progress) -> str:
        import asyncio

        await asyncio.sleep(30)
        return "never"

    job_id = manager.register("bash", "gh run watch", runner)
    frames = "".join(FRAME.format(n=n, mark="*") for n in range(1, 6))
    manager.append_output(job_id, frames)
    peek = await _call(tools, "jobs", {"op": "peek", "job_id": job_id}, context)
    text = _text(peek)
    details = peek.details or {}
    assert "[4 earlier repeated frame(s) elided; showing the latest" in text
    assert "about 5 minute" in text and "about 1 minute" not in text
    assert details["frames_elided"] == 4 and details["new_chars"] == len(frames)
    stored = get_store().read_lines(details["spill"]["handle"], 1, None)
    assert stored is not None and any("about 1 minute" in line for line in stored[0])
    await manager.cancel(job_id)
    await manager.dispose()


# -------------------------------------------------------------------------- read


@pytest.mark.asyncio
async def test_spill_search_is_bounded_even_when_matched_lines_are_huge(isolated_spill):
    """Every ``read`` over 20k chars on the fleet since 09-29 came from here."""
    context = ToolContext(cwd=str(isolated_spill), session_id="s")
    tools = {tool.name: tool for tool in create_tools(context)}
    huge = "\n".join(f"export {{ Icon{i} }}; " + "x" * 50_000 + " needle" for i in range(8))
    meta = get_store().write(huge, tool_name="bash", session_id="s")
    assert meta is not None

    result = await _call(tools, "read", {"path": f"{meta.handle}?q=needle"}, context)
    text = _text(result)
    assert len(text) <= READ_OUTPUT_LIMIT_CHARS + 1_000
    assert "needle" in text and "chars\u2026]" not in text.split("\n", 1)[0]
    # Each hit is shown in a window that says how much it dropped.
    assert "[\u2026" in text
    # A clipped match list names its next page in match coordinates.
    assert "of 8 match(es)" in text
