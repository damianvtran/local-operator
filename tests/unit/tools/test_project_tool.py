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
from local_operator.projects import (
    DESCRIPTION_MAX,
    PROGRESS_MAX,
    PROJECT_PROGRESS_STALE_S,
    ProjectRegistry,
)
from local_operator.tools.project_tool import (
    ProjectParams,
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
    assert "payments-migration" in listed and "1 working session" in listed


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
async def test_create_accepts_multi_paragraph_markdown_descriptions(context, registry) -> None:
    """Issue #1815: the schema's promise — paragraphs, headings, lists — is the
    contract; the old 240 cap refused exactly the prose it advertised."""
    description = (
        "# Scope\n\n"
        "Cutover plan for the payments migration.\n\n"
        "## Steps\n\n"
        "- parity on staging\n"
        "- window 2026-10-02\n\n"
        "## Risks\n\n" + "Legacy writes must be drained before the switch. " * 10
    )
    assert len(description) > 500
    body = await call(context, op="create", name="markdown-ok", description=description)
    assert "created project 'markdown-ok'" in body
    stored = registry.get_project_by_name("markdown-ok")
    assert stored is not None and stored.description == description.strip()
    # The boundary the schema text names: exactly at the cap is accepted.
    at_cap = await call(context, op="create", name="at-cap", description="e" * 2000)
    assert "created project 'at-cap'" in at_cap


@pytest.mark.asyncio
async def test_an_over_cap_description_is_refused_with_the_remedy(context) -> None:
    """Issue #1815: no bare pydantic sentence — field, submitted size, exact
    cap and the remedy, so the next call is not a blind retry. The sentence uses
    the sibling refusals' ``(submitted n)`` shape (design review round 1, D4)."""
    result = await execute_project(
        "tc", {"op": "create", "name": "too-long", "description": "D" * 4000}, None, None, context
    )
    assert result.is_error
    assert f"project 'description' is over the {DESCRIPTION_MAX}-character cap" in result.text
    assert "(submitted 4000)" in result.text
    assert "progress lines (op='update')" in result.text
    assert "String should have at most" not in result.text


@pytest.mark.asyncio
async def test_update_refuses_an_over_cap_description_with_the_remedy(context) -> None:
    await call(context, op="create", name="alpha")
    result = await execute_project(
        "tc", {"op": "update", "name": "alpha", "description": "D" * 2001}, None, None, context
    )
    assert result.is_error
    assert f"project 'description' is over the {DESCRIPTION_MAX}-character cap" in result.text
    assert "(submitted 2001)" in result.text


@pytest.mark.asyncio
async def test_an_over_cap_progress_is_refused_with_the_remedy(context) -> None:
    """Issue #1815 / design review round 1, D2: the description refusal tells
    the writer to keep the long detail in progress lines, so an over-cap
    progress line is the FIRST thing that remedy reaches. It used to answer
    with the store's bare pydantic sentence — the message this issue exists to
    kill — and it is refused for BOTH write verbs that carry a line."""
    created = await execute_project(
        "tc",
        {"op": "create", "name": "alpha", "progress": "y" * (PROGRESS_MAX + 200)},
        None,
        None,
        context,
    )
    assert created.is_error
    assert f"project 'progress' is over the {PROGRESS_MAX}-character cap" in created.text
    assert f"(submitted {PROGRESS_MAX + 200})" in created.text
    assert "earlier updates stay in the history" in created.text
    assert "String should have at most" not in created.text

    await call(context, op="create", name="beta")
    updated = await execute_project(
        "tc",
        {"op": "update", "name": "beta", "progress": "y" * (PROGRESS_MAX + 1)},
        None,
        None,
        context,
    )
    assert updated.is_error
    assert f"project 'progress' is over the {PROGRESS_MAX}-character cap" in updated.text
    assert f"(submitted {PROGRESS_MAX + 1})" in updated.text
    # The boundary the cap names: exactly at it is accepted.
    at_cap = await call(context, op="update", name="beta", progress="y" * PROGRESS_MAX)
    assert "progress" in at_cap and not at_cap.startswith("invalid")


def test_the_schema_text_names_the_cap_the_refusal_enforces() -> None:
    """Schema, guide and enforcement move together (issue #1815): the field
    text is built FROM the constant, so a future cap change cannot leave the
    advertised limit behind (the drift this issue was made of)."""
    text = str(ProjectParams.model_fields["description"].description)
    assert f"(<= {DESCRIPTION_MAX} chars)" in text


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
    backdated = time.time() - PROJECT_PROGRESS_STALE_S - 60
    payload["progress_updated_at"] = backdated
    path.write_text(json.dumps(payload))

    body = await call(context, op="update", name="alpha", progress="still true")
    assert "refreshed" in body and "progress unchanged" in body and "no new content" in body
    refreshed = registry.get_project_by_name("alpha")
    assert refreshed.progress == "still true"
    # Refresh ≠ update: the CONTENT clock did not move (the badge keeps its
    # truth), and the assertion pair is what the call wrote.
    assert refreshed.progress_updated_at == backdated
    assert refreshed.progress_refreshed_at is not None
    assert refreshed.progress_refreshed_by == SESSION


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
    assert "linked session abcdef012345" in body and "2 working" in body
    body = await call(context, op="unlink", name="alpha", session_id="abcdef012345")
    assert "unlinked session abcdef012345" in body and "1 working" in body

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
async def test_update_with_a_partial_milestones_list_is_refused_and_changes_nothing(
    context, registry, tmp_path
) -> None:
    """The incident shape: editing ONE milestone through op='update' with a
    one-entry list. It used to replace the whole list silently — the refusal
    must name both safe paths, and the stored row must not move a byte."""
    await call(
        context,
        op="create",
        name="alpha",
        milestones=[
            {"name": "beta cut", "target_date": "2026-10-01"},
            {"name": "gamma review"},
            {"name": "delta sign-off"},
        ],
    )
    project = registry.get_project_by_name("alpha")
    path = tmp_path / "projects" / f"{project.id}.json"
    before = path.read_bytes()

    body = await call(
        context,
        op="update",
        name="alpha",
        milestones=[{"name": "beta cut", "completed_at": "2026-09-27"}],
    )
    assert "update would REPLACE all milestones (3 currently stored)" in body
    assert "op='milestone'" in body
    assert "add/update/remove ONE milestone by name" in body
    assert "replace_milestones=true" in body
    # No write happened at all: byte-identical row on disk, siblings intact.
    assert path.read_bytes() == before
    assert [m.name for m in registry.get_project_by_name("alpha").milestones] == [
        "beta cut",
        "gamma review",
        "delta sign-off",
    ]


@pytest.mark.asyncio
async def test_replace_milestones_true_replaces_the_list_and_the_receipt_says_so(
    context, registry
) -> None:
    await call(
        context,
        op="create",
        name="alpha",
        milestones=[
            {"name": "beta cut"},
            {"name": "gamma review"},
            {"name": "delta sign-off"},
        ],
    )
    body = await call(
        context,
        op="update",
        name="alpha",
        milestones=[{"name": "beta cut", "target_date": "2026-10-02"}],
        replace_milestones=True,
    )
    assert "milestones replaced deliberately" in body
    assert "replace_milestones=true" in body
    stored = registry.get_project_by_name("alpha").milestones
    assert [m.name for m in stored] == ["beta cut"]
    assert stored[0].target_date == "2026-10-02"


@pytest.mark.asyncio
async def test_the_milestone_op_upsert_keeps_siblings(context, registry) -> None:
    await call(
        context,
        op="create",
        name="alpha",
        milestones=[
            {"name": "beta cut"},
            {"name": "gamma review"},
            {"name": "delta sign-off"},
        ],
    )
    body = await call(
        context, op="milestone", name="alpha", milestone="beta cut", milestone_completed=True
    )
    assert "updated milestone 'beta cut'" in body
    stored = registry.get_project_by_name("alpha").milestones
    assert [m.name for m in stored] == ["beta cut", "gamma review", "delta sign-off"]
    assert stored[0].completed_at is not None
    assert stored[1].completed_at is None and stored[2].completed_at is None

    body = await call(context, op="milestone", name="alpha", milestone="epsilon cut")
    assert "added milestone 'epsilon cut'" in body
    assert [m.name for m in registry.get_project_by_name("alpha").milestones] == [
        "beta cut",
        "gamma review",
        "delta sign-off",
        "epsilon cut",
    ]


@pytest.mark.asyncio
async def test_create_still_takes_its_milestone_list(context, registry) -> None:
    body = await call(
        context,
        op="create",
        name="alpha",
        milestones=[{"name": "beta cut"}, {"name": "gamma review", "target_date": "2026-11-01"}],
    )
    assert "created project 'alpha'" in body
    stored = registry.get_project_by_name("alpha").milestones
    assert [m.name for m in stored] == ["beta cut", "gamma review"]
    assert stored[1].target_date == "2026-11-01"


@pytest.mark.asyncio
async def test_a_refresh_and_a_deliberate_replace_both_say_so(context, registry, tmp_path) -> None:
    """One call, two verbs: the stale-identical progress line refreshes AND the
    milestone list is replaced — the refreshed receipt must still name the
    replace (agent review round 1, M1)."""
    await call(
        context,
        op="create",
        name="alpha",
        milestones=[{"name": "beta cut"}, {"name": "gamma review"}],
    )
    await call(context, op="update", name="alpha", progress="still true")
    project = registry.get_project_by_name("alpha")
    # Backdate the stamp ON DISK past the staleness window so the identical
    # line REFRESHES instead of no-oping — every mutation reloads, so in-memory
    # surgery would be lost. Constant-driven: the S6d slice moved the window to
    # four hours, and a hardcoded hour stopped being stale under it.
    path = tmp_path / "projects" / f"{project.id}.json"
    payload = json.loads(path.read_text())
    payload["progress_updated_at"] = time.time() - PROJECT_PROGRESS_STALE_S - 60
    path.write_text(json.dumps(payload))

    body = await call(
        context,
        op="update",
        name="alpha",
        progress="still true",
        milestones=[{"name": "beta cut"}],
        replace_milestones=True,
    )
    assert "refreshed project 'alpha'" in body
    assert "milestones replaced deliberately" in body
    assert "replace_milestones=true" in body
    stored = registry.get_project_by_name("alpha")
    assert [m.name for m in stored.milestones] == ["beta cut"]
    assert stored.progress == "still true"


@pytest.mark.asyncio
async def test_an_explicit_null_milestones_is_a_no_op_not_a_refusal(context, registry) -> None:
    """`milestones=null` is the store's "leave it": no refusal, no replace
    claim, and the other fields the call carries still apply (M2)."""
    await call(
        context,
        op="create",
        name="alpha",
        milestones=[{"name": "beta cut"}, {"name": "gamma review"}],
    )
    body = await call(
        context, op="update", name="alpha", progress="2026-09-27 moved", milestones=None
    )
    assert "REPLACE" not in body
    assert "updated project 'alpha'" in body
    stored = registry.get_project_by_name("alpha")
    assert stored.progress == "2026-09-27 moved"
    assert [m.name for m in stored.milestones] == ["beta cut", "gamma review"]


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
    assert "working sessions (1/64):" in body
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
async def test_history_collapses_adjacent_identical_normalized_entries(context, registry) -> None:
    """Display-only collapse (ruling §4.4, agent review r1 F2): a legacy run of
    adjacent identical-NORMALIZED entries renders as its NEWEST row plus
    "(re-sent N×)" — the stored log is never rewritten — while a near-identical
    neighbour stays its own row (the refresh rule's false-positive argument)."""
    await call(context, op="create", name="alpha")
    project = registry.get_project_by_name("alpha")
    assert project is not None
    path = registry.projects_dir / f"{project.id}.json"
    payload = json.loads(path.read_text())
    # A run an older build could have left: whitespace-only differences
    # normalize equal; the near-identical suffix variant must stay separate.
    payload["updates"] = [
        {"at": "2026-09-30T01:00:00Z", "text": "checked: still true", "by": SESSION},
        {"at": "2026-09-30T02:00:00Z", "text": "checked:  still   true", "by": SESSION},
        {"at": "2026-09-30T03:00:00Z", "text": "checked: still true", "by": SESSION},
        {"at": "2026-09-30T04:00:00Z", "text": "checked: still true too", "by": SESSION},
    ]
    path.write_text(json.dumps(payload))
    # The registry's snapshot reloads on the directory mtime or the interval;
    # this edit moves neither, so force the bounded re-read the flag surfaces
    # use (``refresh``) before the show reads the rewritten row.
    registry.refresh()

    shown = await call(context, op="show", name="alpha")
    assert "history (4, latest 2 shown):" in shown
    # The NEWEST copy of the run survives; the elided copies' stamps are gone.
    assert f"  - 2026-09-30T03:00:00Z by {SESSION}: checked: still true (re-sent 3×)" in shown
    assert "2026-09-30T01:00:00Z" not in shown and "2026-09-30T02:00:00Z" not in shown
    assert f"  - 2026-09-30T04:00:00Z by {SESSION}: checked: still true too" in shown
    # Display-only: the stored log keeps all four entries.
    kept = registry.get_project_by_name("alpha")
    assert kept is not None and len(kept.updates) == 4


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


@pytest.mark.asyncio
async def test_history_zero_omits_the_section_even_when_empty(context) -> None:
    """``history=0`` honours "0 omits the section" for an empty log too (review F2)."""
    await call(context, op="create", name="alpha")
    shown = await call(context, op="show", name="alpha")
    assert "history: none recorded" in shown
    assert "history" not in await call(context, op="show", name="alpha", history=0)


@pytest.mark.asyncio
async def test_attachment_sizes_print_one_decimal_across_the_unit(context, tmp_path) -> None:
    """The pin for the D2 style: one decimal place for KB and MB, bytes exact."""
    await call(context, op="create", name="alpha")
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"k" * 4104)
    await call(context, op="update", name="alpha", progress="kb line", attach=[str(frame)])
    shown = await call(context, op="show", name="alpha")
    assert "attachment: frame.png [image, 4.0 KB]" in shown


@pytest.mark.asyncio
async def test_lifecycle_statuses_are_accepted_and_shown(context) -> None:
    created = await call(context, op="create", name="alpha", status="planning")
    assert "[planning]" in created
    moved = await call(context, op="update", name="alpha", status="qa")
    assert "status qa" in moved


@pytest.mark.asyncio
async def test_done_refusal_teaches_and_force_done_closes(context) -> None:
    await call(context, op="create", name="alpha", milestones=[{"name": "beta cut"}])
    refused = await call(context, op="update", name="alpha", status="done")
    assert "cannot set status 'done'" in refused
    assert "'beta cut'" in refused and "force_done=true" in refused
    forced = await call(context, op="update", name="alpha", status="done", force_done=True)
    assert "[done]" in forced
    # The receipt names the deliberate close (the M1 rule), not just the state.
    assert "force_done=true" in forced


@pytest.mark.asyncio
async def test_an_unknown_status_refusal_lists_the_vocabulary(context) -> None:
    await call(context, op="create", name="alpha")
    refused = await call(context, op="update", name="alpha", status="shipped")
    assert "planning" in refused and "validation" in refused and "archived" in refused


@pytest.mark.asyncio
async def test_a_forced_create_names_the_deliberate_close(context) -> None:
    """N2: the update path's deliberate-act rule applies to create too."""
    forced = await call(
        context,
        op="create",
        name="alpha",
        status="done",
        milestones=[{"name": "open"}],
        force_done=True,
    )
    assert "[done]" in forced
    assert "status 'done' forced with milestones incomplete (force_done=true)" in forced
    # An ordinary create claims nothing: no force clause without the force.
    plain = await call(context, op="create", name="beta", status="done")
    assert "force_done" not in plain


# -- the role split and refresh ≠ update (schema 2, P1/P3) -------------------


def _backdate(registry: ProjectRegistry, name: str, age_s: float) -> float:
    """Age the stored content clock; returns the stamp the store now holds."""
    project = registry.get_project_by_name(name)
    assert project is not None
    path = registry.projects_dir / f"{project.id}.json"
    payload = json.loads(path.read_text())
    payload["progress_updated_at"] = time.time() - age_s
    path.write_text(json.dumps(payload))
    return float(payload["progress_updated_at"])


@pytest.mark.asyncio
async def test_create_files_a_chief_of_staff_session_instead_of_joining_it(
    registry: ProjectRegistry, context: ToolContext, tmp_path: Path
) -> None:
    """The CoS's create-time auto-link is ROLE-ASSIGNED at the write surface:
    her id lands in ``coordination_sessions`` (provenance — never a liveness
    fact, never a nudge), ``sessions`` stays empty, and the receipt says so.
    A worker's create keeps the working auto-link."""
    from local_operator.aida.state import write_state

    write_state(tmp_path, {"session_id": SESSION})

    body = await call(context, op="create", name="filed-for-a-worker")
    assert "filed by this session" in body and "not a working session" in body
    project = registry.get_project_by_name("filed-for-a-worker")
    assert project is not None
    assert project.sessions == []
    assert project.coordination_sessions == [SESSION]

    other = ToolContext(cwd=".", session_id="abcdef012345", project_registry=registry)
    worker_body = await call(other, op="create", name="worker-owned")
    assert "linked this session" in worker_body
    worker = registry.get_project_by_name("worker-owned")
    assert worker is not None
    assert worker.sessions == ["abcdef012345"] and worker.coordination_sessions == []

    # The listing counts the two kinds separately, and the show splits them.
    listed = await call(context, op="list")
    assert "0 working sessions · 1 filed" in listed
    shown = await call(context, op="show", name="filed-for-a-worker")
    assert "working sessions (0/64):" not in shown  # an empty section is not painted
    assert f"filed by (1):\n  - {SESSION} [filed]" in shown


@pytest.mark.asyncio
async def test_a_chief_of_staff_self_link_files_and_never_re_kinds_a_link(
    registry: ProjectRegistry, context: ToolContext, tmp_path: Path
) -> None:
    """The create-time role decision, applied at the LINK surface (agent review
    r1, F3): her self-link is provenance — it can neither silently flip her
    filed→working nor demote a work link she already holds. An explicit link
    naming her id from ANOTHER session still lands as work: the "a wrong
    demotion is one op='link' from restored" repair path (ruling §2.3)."""
    from local_operator.aida.state import write_state

    write_state(tmp_path, {"session_id": SESSION})
    worker = ToolContext(cwd=".", session_id="abcdef012345", project_registry=registry)

    await call(worker, op="create", name="plain")
    # A fresh self-link FILES: she filed it, she does not work it.
    body = await call(context, op="link", name="plain")
    assert "filed session" in body and "not a working session" in body
    project = registry.get_project_by_name("plain")
    assert project is not None
    assert project.sessions == ["abcdef012345"] and project.coordination_sessions == [SESSION]

    # A re-link is a no-op that still names the ROLE — never "linked ... as a
    # working session".
    again = await call(context, op="link", name="plain")
    assert "already filed" in again and "not a working session" in again

    # A work link she already holds survives a self-link untouched (the other
    # silent re-kind is refused too).
    registry.link_session(project.id, SESSION, role="work")
    untouched = await call(context, op="link", name="plain")
    assert "already linked" in untouched and "as a working session" in untouched
    settled = registry.get_project_by_name("plain")
    assert settled is not None and settled.sessions == ["abcdef012345", SESSION]

    # The repair path: another session explicitly naming her id lands as WORK —
    # and the receipt NAMES the re-kind it performed.
    await call(worker, op="create", name="repair")
    await call(context, op="link", name="repair")
    repaired = await call(worker, op="link", name="repair", session_id=SESSION)
    assert repaired == (
        f"moved session {SESSION} from filed to working links on 'repair' (2 working now)."
    )
    row = registry.get_project_by_name("repair")
    assert row is not None
    assert row.sessions == ["abcdef012345", SESSION] and row.coordination_sessions == []


@pytest.mark.asyncio
async def test_refresh_records_a_check_without_moving_the_content_clock(
    registry: ProjectRegistry, context: ToolContext
) -> None:
    await call(context, op="create", name="alpha")
    await call(context, op="update", name="alpha", progress="still true")
    backdated = _backdate(registry, "alpha", PROJECT_PROGRESS_STALE_S + 60)

    body = await call(context, op="refresh", name="alpha")
    assert "refreshed project 'alpha'" in body
    assert "progress unchanged" in body and "no new content" in body
    row = registry.get_project_by_name("alpha")
    assert row is not None
    assert row.progress == "still true"
    assert row.progress_updated_at == backdated  # the badge's clock never moved
    assert row.progress_refreshed_at is not None
    assert row.progress_refreshed_by == SESSION

    # On a content-FRESH record the same op writes nothing ("no reason to send
    # it every turn"), and the receipt says exactly that. A re-refresh of a
    # still-stale record is deliberately allowed (the gate is content-stale,
    # not assertion-age) and re-dates the assertion — the ruling's risk note
    # accepts that as visible and badge-neutral.
    await call(context, op="update", name="alpha", progress="moved on")
    fresh = await call(context, op="refresh", name="alpha")
    assert "not stale — nothing written" in fresh


@pytest.mark.asyncio
async def test_refresh_without_recorded_progress_names_the_right_act(
    registry: ProjectRegistry, context: ToolContext
) -> None:
    await call(context, op="create", name="alpha")
    body = await call(context, op="refresh", name="alpha")
    assert "no recorded progress to refresh" in body
    assert "op='update'" in body


@pytest.mark.asyncio
async def test_refresh_is_textless_and_progress_rides_update(context) -> None:
    await call(context, op="create", name="alpha")
    body = await call(context, op="refresh", name="alpha", progress="a line")
    assert "op='refresh' is textless" in body and "op='update'" in body


@pytest.mark.asyncio
async def test_the_update_receipt_for_a_refresh_never_claims_freshness(
    registry: ProjectRegistry, context: ToolContext
) -> None:
    """The old receipt said "re-stamped just now", which reads as "the record
    is fresh again" — precisely the belief P3 removes. The new one dates the
    CHECK and says the line did not move."""
    await call(context, op="create", name="alpha")
    await call(context, op="update", name="alpha", progress="still true")
    _backdate(registry, "alpha", PROJECT_PROGRESS_STALE_S + 60)
    body = await call(context, op="update", name="alpha", progress="still true")
    assert "refreshed project 'alpha' — progress unchanged" in body
    assert "no new content" in body
    assert "re-stamped" not in body


@pytest.mark.asyncio
async def test_linking_a_filed_session_moves_it_to_working(registry, context) -> None:
    """One op re-kinds: linking an id that was filed MOVES it between lists
    (the repair path a wrong migration demotion takes), and the receipt names
    the role change rather than hiding it behind a plain "linked"."""
    await call(context, op="create", name="alpha")
    registry.link_session(registry.get_project_by_name("alpha").id, SESSION, role="coordination")
    body = await call(context, op="link", name="alpha", session_id=SESSION)
    assert "moved session" in body and "from filed to working links" in body
    project = registry.get_project_by_name("alpha")
    assert project is not None
    assert project.sessions == [SESSION] and project.coordination_sessions == []

    # Unlink targets the id across either list: after re-filing, the same op
    # still removes it.
    registry.link_session(project.id, SESSION, role="coordination")
    body = await call(context, op="unlink", name="alpha", session_id=SESSION)
    assert "1 filed" in body or "0 working" in body
    cleaned = registry.get_project_by_name("alpha")
    assert cleaned is not None
    assert cleaned.sessions == [] and cleaned.coordination_sessions == []
