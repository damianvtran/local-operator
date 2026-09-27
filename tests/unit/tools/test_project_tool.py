"""The ``project`` tool: create, progress, links, milestones, delete.

The tool is the model-facing surface over :mod:`local_operator.projects`, and
these tests pin what a model actually reads back: the receipts (which name what
happened, including the no-ops), the refusal sentences, and the createIf split
that keeps both tools out of a session with no store behind them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.projects import PROJECT_PROGRESS_STALE_S, ProjectRegistry
from local_operator.tools.project_tool import (
    build_project_delete_tool,
    build_project_tool,
    execute_project,
    execute_project_delete,
)

SESSION = "4e92693767fa"


@pytest.fixture()
def registry(tmp_path) -> ProjectRegistry:
    return ProjectRegistry(tmp_path)


@pytest.fixture()
def context(registry: ProjectRegistry) -> ToolContext:
    return ToolContext(cwd=".", session_id=SESSION, project_registry=registry)


async def call(context: ToolContext, **args: Any) -> str:
    result = await execute_project("tc", args, None, None, context)
    return result.text


async def delete(context: ToolContext, name: str) -> str:
    result = await execute_project_delete("tc", {"name": name}, None, None, context)
    return result.text


def test_the_tools_are_not_advertised_without_a_registry() -> None:
    assert build_project_tool(ToolContext(cwd=".")) is None
    assert build_project_delete_tool(ToolContext(cwd=".")) is None
    assert build_project_tool(ToolContext(cwd=".", project_registry=object())) is not None
    assert build_project_delete_tool(ToolContext(cwd=".", project_registry=object())) is not None


def test_only_irreversible_removal_requires_approval(context) -> None:
    authoring = build_project_tool(context)
    deleting = build_project_delete_tool(context)
    assert authoring is not None and authoring.approval_tier == "read"
    assert deleting is not None and deleting.approval_tier == "write"
    assert deleting.describe_approval is not None
    assert "Delete project 'alpha' permanently" == deleting.describe_approval(
        {"name": "alpha"}, "."
    )


@pytest.mark.asyncio
async def test_list_empty_points_at_create_and_the_guide(context) -> None:
    body = await call(context, op="list")
    assert "no projects yet" in body
    assert "guide://projects" in body


@pytest.mark.asyncio
async def test_create_auto_links_the_calling_session_and_says_so(context) -> None:
    body = await call(context, op="create", name="payments-migration", description="Payments")
    assert "created project 'payments-migration'" in body
    assert SESSION in body
    listed = await call(context, op="list")
    assert "payments-migration" in listed and "1 session" in listed


@pytest.mark.asyncio
async def test_create_without_a_session_identity_makes_an_unlinked_project(
    registry: ProjectRegistry,
) -> None:
    bare = ToolContext(cwd=".", project_registry=registry)
    body = await call(bare, op="create", name="loose-ends")
    assert "no session link" in body
    created = registry.get_project_by_name("loose-ends")
    assert created is not None and created.sessions == []


@pytest.mark.asyncio
async def test_create_refuses_a_taken_name_and_names_update(context) -> None:
    await call(context, op="create", name="alpha")
    body = await call(context, op="create", name="ALPHA")
    assert "already exists" in body and "op='update'" in body


@pytest.mark.asyncio
async def test_progress_writes_stamp_the_reporter_and_no_op_is_reported(context, registry) -> None:
    await call(context, op="create", name="alpha")
    body = await call(context, op="update", name="alpha", progress="2026-09-26 moved")
    assert "updated project 'alpha'" in body
    project = registry.get_project_by_name("alpha")
    assert project.progress == "2026-09-26 moved"
    assert project.progress_reported_by == SESSION

    again = await call(context, op="update", name="alpha", progress="2026-09-26 moved")
    assert "already held those values" in again
    assert "nothing written" in again


@pytest.mark.asyncio
async def test_identical_progress_on_a_stale_record_refreshes_it(
    context, registry, tmp_path
) -> None:
    await call(context, op="create", name="alpha")
    await call(context, op="update", name="alpha", progress="still true")
    project = registry.get_project_by_name("alpha")
    # Backdate the stamp ON DISK — far past the staleness window. (In-memory
    # surgery would be discarded: every mutation reloads under the lock.)
    path = tmp_path / "projects" / f"{project.id}.json"
    payload = json.loads(path.read_text())
    payload["progress_updated_at"] = time.time() - PROJECT_PROGRESS_STALE_S - 60
    path.write_text(json.dumps(payload))

    body = await call(context, op="update", name="alpha", progress="still true")
    assert "refreshed" in body
    refreshed = registry.get_project_by_name("alpha")
    assert refreshed.progress == "still true"
    assert refreshed.progress_updated_at > project.progress_updated_at


@pytest.mark.asyncio
async def test_status_done_stamps_completion_and_an_empty_string_clears_it(
    context, registry
) -> None:
    await call(context, op="create", name="alpha")
    await call(context, op="update", name="alpha", status="done")
    project = registry.get_project_by_name("alpha")
    assert project.completed_at is not None
    await call(context, op="update", name="alpha", completed_at="")
    assert registry.get_project_by_name("alpha").completed_at is None


@pytest.mark.asyncio
async def test_update_of_an_unknown_name_points_at_list(context) -> None:
    body = await call(context, op="update", name="nope", progress="x")
    assert "no project named 'nope'" in body and "op='list'" in body


@pytest.mark.asyncio
async def test_link_and_unlink_receipts_name_the_resulting_set(context, registry) -> None:
    await call(context, op="create", name="alpha")
    body = await call(context, op="link", name="alpha", session_id="abcdef012345")
    assert "linked session abcdef012345" in body and "2 linked" in body
    body = await call(context, op="unlink", name="alpha", session_id="abcdef012345")
    assert "unlinked session abcdef012345" in body and "1 linked" in body

    absent = await call(context, op="unlink", name="alpha", session_id="abcdef012345")
    assert "not linked" in absent
    assert registry.get_project_by_name("alpha").sessions == [SESSION]


@pytest.mark.asyncio
async def test_link_past_the_cap_names_unlink(context, registry) -> None:
    await call(context, op="create", name="alpha")
    project = registry.get_project_by_name("alpha")
    # Fill the cap through the public API: the calling session is already
    # linked, so 63 more reach 64.
    for index in range(1, 64):
        registry.link_session(project.id, f"{index:012x}")
    body = await call(context, op="link", name="alpha", session_id="abcdef012345")
    assert "64 linked sessions" in body and "unlink" in body


@pytest.mark.asyncio
async def test_milestone_op_adds_completes_removes_and_reports_no_change(context, registry) -> None:
    await call(context, op="create", name="alpha")
    body = await call(
        context,
        op="milestone",
        name="alpha",
        milestone="beta cut",
        milestone_target_date="2026-10-01",
    )
    assert "added milestone 'beta cut'" in body
    body = await call(
        context, op="milestone", name="alpha", milestone="Beta Cut", milestone_completed=True
    )
    assert "updated milestone 'Beta Cut'" in body
    assert registry.get_project_by_name("alpha").milestones[0].completed_at is not None

    body = await call(context, op="milestone", name="alpha", milestone="beta cut")
    assert "nothing written" in body
    body = await call(context, op="milestone", name="alpha", milestone="beta cut", remove=True)
    assert "removed milestone 'beta cut'" in body
    assert registry.get_project_by_name("alpha").milestones == []

    body = await call(context, op="milestone", name="alpha", milestone="ghost", remove=True)
    assert "no milestone named 'ghost'" in body


@pytest.mark.asyncio
async def test_show_reports_the_record_and_its_linked_sessions(context) -> None:
    await call(
        context,
        op="create",
        name="alpha",
        description="Alpha stream",
        estimate=13.0,
        milestones=[{"name": "beta cut", "target_date": "2026-10-01"}],
    )
    await call(context, op="update", name="alpha", progress="moved")
    body = await call(context, op="show", name="alpha")
    assert "alpha [active]" in body
    assert "est 13pt" in body
    assert "milestones (1/20):" in body and "- beta cut [upcoming]" in body
    assert "linked sessions (1/64):" in body
    assert f"- {SESSION} [missing] — no session directory" in body


@pytest.mark.asyncio
async def test_invalid_values_return_the_stores_sentence(context) -> None:
    await call(context, op="create", name="alpha")
    body = await call(context, op="update", name="alpha", tags=["Q4!"])
    assert "value error" in body.lower()
    body = await call(context, op="update", name="alpha", estimate=0)
    assert "estimate" in body
    result = await execute_project("tc", {"op": "teleport"}, None, None, context)
    assert "teleport" in result.text or "op" in result.text


@pytest.mark.asyncio
async def test_delete_removes_the_row_and_unknown_names_are_refused(context, registry) -> None:
    await call(context, op="create", name="alpha")
    assert "deleted project 'alpha'" in await delete(context, "alpha")
    assert registry.get_project_by_name("alpha") is None
    assert "no project named 'alpha'" in await delete(context, "alpha")


@pytest.mark.asyncio
async def test_owner_team_and_title_flow_through_create_update_and_reads(context, registry) -> None:
    await call(
        context,
        op="create",
        name="alpha",
        owner="Damian",
        team="Platform",
        title=" Alpha Stream ",
    )
    project = registry.get_project_by_name("alpha")
    assert (project.owner, project.team, project.title) == ("Damian", "Platform", "Alpha Stream")

    body = await call(context, op="show", name="alpha")
    assert "Alpha Stream [active]" in body
    assert "key: alpha" in body
    assert "owner: Damian" in body and "team: Platform" in body

    listed = await call(context, op="list")
    assert "- Alpha Stream (alpha) [active]" in listed

    await call(context, op="update", name="alpha", title="")
    project = registry.get_project_by_name("alpha")
    assert project.title is None
    # Untitled reads fall back to the key everywhere, with no key line (the
    # first line IS the key).
    body = await call(context, op="show", name="alpha")
    assert "alpha [active]" in body and "key:" not in body
    assert "- alpha [active]" in await call(context, op="list")


@pytest.mark.asyncio
async def test_invalid_attributions_return_the_stores_sentence(context) -> None:
    await call(context, op="create", name="alpha")
    body = await call(context, op="update", name="alpha", owner="x" * 81)
    assert "owner must be at most 80 characters" in body


@pytest.mark.asyncio
async def test_the_history_shows_a_tail_and_attach_stores_files(
    context, registry, tmp_path
) -> None:
    await call(context, op="create", name="alpha")
    await call(context, op="update", name="alpha", progress="first line")

    shot = tmp_path / "shot.png"
    shot.write_bytes(b"p" * 512)
    body = await call(
        context, op="update", name="alpha", progress="second line", attach=[str(shot)]
    )
    assert "1 attachment stored" in body

    await call(context, op="update", name="alpha", progress="third line")
    shown = await call(context, op="show", name="alpha")
    assert "history (3):" in shown
    assert "first line" in shown and "third line" in shown
    assert "attachment: shot.png [image, 512 B]" in shown
    stored = registry.get_project_by_name("alpha").updates[1].attachments[0].path
    assert Path(stored).exists()

    tailed = await call(context, op="show", name="alpha", history=1)
    assert "history (3, latest 1 shown):" in tailed
    assert "third line" in tailed and "first line" not in tailed
    assert "history" not in await call(context, op="show", name="alpha", history=0)


@pytest.mark.asyncio
async def test_attach_refusals_surface_the_stores_sentence(context, tmp_path) -> None:
    await call(context, op="create", name="alpha")
    shot = tmp_path / "s.png"
    shot.write_bytes(b"x")

    body = await call(context, op="update", name="alpha", attach=[str(shot)])
    assert "NEW progress line" in body

    body = await call(context, op="create", name="beta", progress="x", attach=[str(shot)])
    assert "attach works only with op='update'" in body

    body = await call(
        context, op="update", name="alpha", progress="x", attach=[str(tmp_path / "nope.png")]
    )
    assert "no file at" in body
