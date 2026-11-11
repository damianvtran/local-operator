"""Profile proposals store: filing, fold, accept/reject/edit/rebase, legacy import.

Isolated config roots only (``tmp_path``); the legacy-import tests stage a real
file-only operator key exactly like ``test_profile_approval`` so the mesh
records are byte-identical to what an older runtime files.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from local_operator import profile_proposals as PP
from local_operator.agents import AgentRegistry
from local_operator.network import approvals as A
from local_operator.teams import TeamRegistry
from tests.unit.network.test_approvals_store import _make_key
from tests.unit.network.test_approvals_store import root as approvals_root
from tests.unit.tools.test_agent_tool import _edit_fields

root = approvals_root


def _agent(root, name="reviewer", prompt="Before."):
    from local_operator.profile_approval import snapshot

    registry = AgentRegistry(root)
    row = registry.create_agent(_edit_fields(name=name, tags=["role"], categories=["role"]))
    registry.set_agent_system_prompt(row.id, prompt)
    before = snapshot(root, "agent", row.id)
    assert before is not None
    return row, before


def _file(root, before, session_id="sess-1", **after_fields):
    return PP.file(
        root,
        profile_kind="agent",
        operation="update",
        target_id=before["id"],
        name=before["name"],
        before=before,
        after=dict(before, **after_fields),
        session_id=session_id,
        tool_call_id="call-1",
        trust="agent",
    )


def _prompt(root, agent_id):
    return AgentRegistry(root, migrate=False).get_agent_system_prompt(agent_id)


def test_file_is_idempotent_and_changes_nothing(root):
    row, before = _agent(root)
    first = _file(root, before, instructions="After.")
    second = _file(root, before, instructions="After.")
    assert first["id"] == second["id"] and first["id"].startswith("pp_")
    assert first["fields"] == ["instructions"]
    assert _prompt(root, row.id) == "Before."
    assert [p["id"] for p in PP.list_proposals(root)] == [first["id"]]
    audit = (PP.proposals_dir(root) / PP.AUDIT_FILENAME).read_text().splitlines()
    assert [json.loads(line)["action"] for line in audit] == ["filed"]


def test_newer_proposal_supersedes_older_for_same_target(root):
    _row, before = _agent(root)
    old = _file(root, before, instructions="One.")
    new = _file(root, before, instructions="Two.")
    assert _must_get(root, old["id"])["state"] == PP.STATE_SUPERSEDED
    assert [p["id"] for p in PP.list_proposals(root)] == [new["id"]]


def test_accept_applies_backs_up_and_audits(root):
    row, before = _agent(root)
    record = _file(root, before, instructions="After.")
    (root / "sessions" / "sess-1").mkdir(parents=True)
    result = PP.apply(root, record["id"], actor="desktop")
    assert _prompt(root, row.id) == "After."
    assert result["proposal"]["state"] == PP.STATE_ACCEPTED
    backup = json.loads(open(result["backup_path"]).read())
    # The agent tool's revert reads exactly these keys (agent_tool._op_revert).
    assert backup["kind"] == "agent" and backup["id"] == row.id
    assert backup["instructions"] == "Before."
    assert backup["saved_revision"] and backup["definition_fields"]["name"] == "reviewer"
    assert result["backup_path"].startswith(str(root / "sessions" / "sess-1" / "profile-backups"))
    with pytest.raises(PP.ProposalError) as again:
        PP.apply(root, record["id"], actor="desktop")
    assert again.value.code == "profile_proposal_not_pending"
    actions = [
        json.loads(line)["action"]
        for line in (PP.proposals_dir(root) / PP.AUDIT_FILENAME).read_text().splitlines()
    ]
    assert actions == ["filed", "accepted"]


def test_backup_falls_back_to_proposals_dir_without_session_dir(root):
    _row, before = _agent(root)
    record = _file(root, before, session_id="gone", instructions="After.")
    result = PP.apply(root, record["id"], actor="desktop")
    assert result["backup_path"].startswith(str(PP.proposals_dir(root)))


def test_agent_revert_restores_from_proposal_backup(root):
    """The backup is readable by ``agent op=revert`` from the filing session."""
    import asyncio

    from local_operator.harness.types import ToolContext
    from local_operator.tools.agent_tool import _op_revert

    row, before = _agent(root)
    record = _file(root, before, instructions="After.")
    session_dir = root / "sessions" / "sess-1"
    session_dir.mkdir(parents=True)
    result = PP.apply(root, record["id"], actor="desktop")

    async def grant(_intent):
        return "owner-prompt:test"

    context = ToolContext(
        agent_registry=AgentRegistry(root),
        session_dir=str(session_dir),
        authorize_durable_write=grant,
    )
    reverted = asyncio.run(_op_revert(context, "call-r", result["backup_id"]))
    assert not reverted.is_error, reverted.content
    assert _prompt(root, row.id) == "Before."


def test_reject_leaves_profile(root):
    row, before = _agent(root)
    record = _file(root, before, instructions="After.")
    out = PP.reject(root, record["id"], actor="desktop", note="no")
    assert out["state"] == PP.STATE_REJECTED and out["note"] == "no"
    assert _prompt(root, row.id) == "Before."
    assert PP.list_proposals(root) == []


def test_edit_then_accept_only_proposed_fields(root):
    row, before = _agent(root)
    record = _file(root, before, instructions="After.")
    with pytest.raises(PP.ProposalError) as bad:
        PP.apply(root, record["id"], actor="desktop", edited_after={"label": "x"})
    assert bad.value.code == "profile_proposal_invalid"
    PP.apply(root, record["id"], actor="desktop", edited_after={"instructions": "Edited."})
    assert _prompt(root, row.id) == "Edited."


def test_stale_base_refuses_even_disjoint_changes(root):
    row, before = _agent(root)
    record = _file(root, before, instructions="After.")
    AgentRegistry(root).update_agent(row.id, _edit_fields(description="moved meanwhile"))
    assert _must_get(root, record["id"])["stale"] is True
    with pytest.raises(PP.ProposalError) as stale:
        PP.apply(root, record["id"], actor="desktop")
    assert stale.value.code == "profile_proposal_stale"
    assert _prompt(root, row.id) == "Before."
    # Refreshing status cannot reset the base or make Accept safe.
    assert _must_get(root, record["id"])["base_commitment"] == record["base_commitment"]
    assert PP.reject(root, record["id"], actor="desktop")["state"] == "rejected"


def test_team_proposal_accept(root):
    from local_operator.profile_approval import snapshot
    from local_operator.teams import TeamEditFields

    registry = TeamRegistry(root)
    team = registry.create_team(TeamEditFields(name="lopdev", instructions="Old team brief."))
    before = snapshot(root, "team", str(team.id))
    assert before is not None
    record = PP.file(
        root,
        profile_kind="team",
        operation="update",
        target_id=str(team.id),
        name="lopdev",
        before=before,
        after=dict(before, instructions="New team brief."),
        session_id="sess-t",
    )
    detail = PP.detail(root, _must_get(root, record["id"]))
    assert detail["changes"][0]["before_text"] == "Old team brief."
    assert detail["changes"][0]["after_text"] == "New team brief."
    PP.apply(root, record["id"], actor="desktop")
    assert TeamRegistry(root).get_team(str(team.id)).instructions == "New team brief."


def _agent_create_image(name="fresh-role", agent_id="11111111-2222-3333-4444-555555555555"):
    return {
        "id": agent_id,
        "name": name,
        # The shape agent_tool._agent_proposal files: label already derived,
        # model "" (the stored value) rather than None.
        "label": "Fresh Role",
        "description": "A role proposed from a conversation.",
        "tags": ["role"],
        "categories": ["role"],
        "model": "",
        "instructions": "Do the new thing.",
    }


def test_create_agent_proposal_accept(root):
    """A create has no before image, so apply must mint the row's own metadata.

    ``AgentImage`` carries only authored fields; ``created_date`` and ``version``
    are required on ``AgentData`` and belong to the registry, not to the proposal.
    """
    from local_operator.profile_approval import snapshot

    image = _agent_create_image()
    record = PP.file(
        root,
        profile_kind="agent",
        operation="create",
        target_id=image["id"],
        name=image["name"],
        before=None,
        after=image,
        session_id="sess-c",
    )
    assert snapshot(root, "agent", image["id"]) is None
    result = PP.apply(root, record["id"], actor="desktop")
    assert result["proposal"]["state"] == PP.STATE_ACCEPTED
    registry = AgentRegistry(root, migrate=False)
    row = registry.get_agent(image["id"])
    assert row.name == "fresh-role"
    assert row.created_date is not None and row.version
    assert registry.get_agent_system_prompt(image["id"]) == "Do the new thing."
    # The written row round-trips to exactly the proposed authored image.
    assert snapshot(root, "agent", image["id"]) == image
    # ...and the files on disk load back through the model that rejected them.
    import yaml

    from local_operator.agents import AgentData

    folder = root / "agents" / image["id"]
    loaded = AgentData.model_validate(yaml.safe_load((folder / "agent.yml").read_text()))
    assert loaded.id == image["id"] and loaded.created_date and loaded.version
    assert (folder / "system_prompt.md").read_text() == "Do the new thing."


def test_create_team_proposal_accept(root):
    from local_operator.profile_approval import snapshot

    image = {
        "id": "team-fresh",
        "name": "freshteam",
        "label": "Freshteam",
        "description": "",
        "manager": "manager",
        "members": [],
        "aliases": [],
        "instructions": "Brief.",
        "project": "",
    }
    record = PP.file(
        root,
        profile_kind="team",
        operation="create",
        target_id=image["id"],
        name=image["name"],
        before=None,
        after=image,
        session_id="sess-ct",
    )
    result = PP.apply(root, record["id"], actor="desktop")
    assert result["proposal"]["state"] == PP.STATE_ACCEPTED
    team = TeamRegistry(root).get_team(image["id"])
    assert team.created_date is not None
    assert snapshot(root, "team", image["id"]) == image
    import yaml

    from local_operator.teams import Team

    folder = root / "teams" / image["id"]
    loaded = Team.model_validate(yaml.safe_load((folder / "team.yml").read_text()))
    assert loaded.id == image["id"] and loaded.created_date
    assert (folder / "instructions.md").read_text() == "Brief."


# --- legacy mesh records ----------------------------------------------------


def _must_get(root, proposal_id: str) -> dict[str, Any]:
    """``PP.get`` for a record the test filed: absent is a test failure."""
    record = PP.get(root, proposal_id)
    assert record is not None, proposal_id
    return record


def _legacy(root, instructions, session_id="legacy-session"):
    from local_operator import profile_approval as P

    _make_key(root)
    registry = AgentRegistry(root)
    existing = registry.get_agent_by_name("legacy")
    if existing is None:
        existing = registry.create_agent(_edit_fields(name="legacy", tags=["role"]))
        registry.set_agent_system_prompt(existing.id, "Old.")
    before = P.snapshot(root, "agent", existing.id)
    assert before is not None
    data = P.build(
        root,
        profile_kind="agent",
        operation="update",
        target_id=existing.id,
        name="legacy",
        session_id=session_id,
        tool_call_id=f"call-{instructions}",
        before=before,
        after=dict(before, instructions=instructions),
    )
    return P.file_request(root, data)


@pytest.mark.parametrize("terminal", ["denied", "expired", "connected", "withdrawn"])
def test_legacy_import_requested_only_and_idempotent(root, terminal):
    pending = _legacy(root, "pending text", session_id="s-a")
    denied = _legacy(root, "denied text", session_id="s-b")
    path = A.approvals_dir(root) / f"{denied['approval_id']}.json"
    raw = json.loads(path.read_text())
    raw["state"] = terminal
    path.write_text(json.dumps(raw))
    broken = A.approvals_dir(root) / "ap_brokenbroken.json"
    broken.write_text(
        json.dumps(
            {
                "approval_id": "ap_brokenbroken",
                "kind": "profile_save",
                "state": "requested",
                "what": {},
            }
        )
    )

    assert PP.import_legacy(root) == 1
    assert PP.import_legacy(root) == 0  # idempotent
    rows = PP.list_proposals(root)
    assert len(rows) == 1
    assert rows[0]["origin"]["legacy_approval_id"] == pending["approval_id"]
    assert rows[0]["after"]["instructions"] == "pending text"
    # The mesh record is untouched on disk and hidden from mesh listings.
    assert (
        json.loads((A.approvals_dir(root) / f"{pending['approval_id']}.json").read_text())["state"]
        == "requested"
    )
    assert broken.exists()


def test_mesh_listing_hides_profile_kind(root):
    from local_operator.server.utils.desktop_approvals import approval_rows

    record = _legacy(root, "hidden")
    assert record["kind"] == A.KIND_PROFILE_SAVE
    assert A.list_records(root=root) == []
    assert [r["approval_id"] for r in A.list_records(root=root, exclude_kinds=frozenset())] == [
        record["approval_id"]
    ]
    assert approval_rows(root) == [] or all(
        row.get("kind") != A.KIND_PROFILE_SAVE for row in approval_rows(root)
    )


def test_legacy_import_skips_lapsed_request_stored_as_requested(root):
    """F1: mesh expiry is folded on READ, so a lapsed request's stored state is
    still ``requested``. Importing from the raw field resurrected it as a
    pending Agents proposal; it must stay an expired outcome."""
    lapsed = _legacy(root, "lapsed text", session_id="s-lapsed")
    path = A.approvals_dir(root) / f"{lapsed['approval_id']}.json"
    raw = json.loads(path.read_text())
    raw["expires_at"] = time.time() - 25 * 3600  # past the 24 h mesh window
    path.write_text(json.dumps(raw))
    assert json.loads(path.read_text())["state"] == "requested"  # stored, unfolded
    assert A.presented(raw)["state"] == "expired"

    assert PP.import_legacy(root) == 0
    assert PP.list_proposals(root) == []
    assert PP.list_proposals(root, state=None) == []
