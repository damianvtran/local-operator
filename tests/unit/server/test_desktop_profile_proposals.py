"""Desktop human decision endpoints over isolated real profile stores."""

import uuid

import pytest

from local_operator.server.routes import desktop_profile_proposals
from tests.unit.server.test_desktop_profiles import api as profile_api
from tests.unit.test_profile_proposals import _agent, _file, _prompt

api = profile_api

pytestmark = pytest.mark.asyncio


async def test_list_get_edit_accept_and_idempotency(api):
    client, root = api
    client._transport.app.include_router(desktop_profile_proposals.router)
    row, before = _agent(root)
    record = _file(root, before, instructions="After.")
    path = "/v1/desktop/profile-proposals"
    response = await client.get(path)
    assert response.status_code == 200, response.text
    assert response.json()["result"]["counts"]["agent"]["reviewer"] == 1
    detail = await client.get(path + "/" + record["id"])
    assert detail.json()["result"]["changes"][0]["after_text"] == "After."
    body = {"request_id": str(uuid.uuid4()), "after": {"instructions": "Edited."}}
    accepted = await client.post(path + "/" + record["id"] + "/accept", json=body)
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["result"]["proposal"]["state"] == "accepted"
    assert _prompt(root, row.id) == "Edited."
    assert accepted.json()["result"]["backup_id"]
    repeated = await client.post(path + "/" + record["id"] + "/accept", json=body)
    replay = repeated.json()["result"]
    assert replay.pop("replayed") is True
    assert replay == accepted.json()["result"]
    assert accepted.json()["result"]["proposal"]["changes"][0]["after_text"] == "Edited."


async def test_reject_and_stale_conflict(api):
    client, root = api
    client._transport.app.include_router(desktop_profile_proposals.router)
    row, before = _agent(root)
    record = _file(root, before, instructions="After.")
    from local_operator.agents import AgentRegistry

    AgentRegistry(root).set_agent_system_prompt(row.id, "Concurrent.")
    path = "/v1/desktop/profile-proposals/" + record["id"]
    accepted = await client.post(path + "/accept", json={"request_id": str(uuid.uuid4())})
    assert accepted.status_code == 409, accepted.text
    assert accepted.json()["detail"]["code"] == "profile_proposal_stale"
    assert accepted.json()["detail"]["current"]["instructions"] == "Concurrent."
    rejected = await client.post(
        path + "/reject", json={"request_id": str(uuid.uuid4()), "note": "no"}
    )
    assert rejected.status_code == 200, rejected.text
    assert rejected.json()["result"]["proposal"]["state"] == "rejected"
    assert _prompt(root, row.id) == "Concurrent."


async def test_invalid_edit_does_not_apply(api):
    client, root = api
    client._transport.app.include_router(desktop_profile_proposals.router)
    row, before = _agent(root)
    record = _file(root, before, instructions="After.")
    response = await client.post(
        "/v1/desktop/profile-proposals/" + record["id"] + "/accept",
        json={"request_id": str(uuid.uuid4()), "after": {"instructions": False}},
    )
    assert response.status_code == 422, response.text
    assert _prompt(root, row.id) == "Before."


# --- skill proposals ---------------------------------------------------------
#
# A skill proposal is filed by the ``skill`` tool when its run read untrusted
# input; the Agents review surface parses it with the SAME schema as agent/team
# proposals (local-operator-ui ``shared/api/local-operator/profile-proposals.ts``):
# one ``changes`` row per changed FILE, ``format: "text"``.

_SKILL_BEFORE = "---\nname: marketing\ndescription: Marketing checklist.\n---\n# Marketing\n"
_SKILL_AFTER = _SKILL_BEFORE + "Measure every launch.\n"


def _skill_proposal(root):
    from local_operator import profile_proposals as P
    from local_operator.server.utils import desktop_skill_writes as writes

    writes.create(root, "marketing", files=[{"path": "SKILL.md", "content": _SKILL_BEFORE}])
    before = P.skill_image(root, "marketing")
    assert before is not None
    after = {**before, "files": {**before["files"], "SKILL.md": _SKILL_AFTER}}
    return P.file(
        root,
        profile_kind="skill",
        operation="update",
        target_id="marketing",
        name="marketing",
        before=before,
        after=after,
        session_id="sess-1",
        trust="untrusted",
    )


def _skill_text(root) -> str:
    return (root / "skills" / "marketing" / "SKILL.md").read_text()


async def test_skill_proposal_lists_and_details_in_ui_shape_then_accepts(api):
    client, root = api
    client._transport.app.include_router(desktop_profile_proposals.router)
    record = _skill_proposal(root)
    path = "/v1/desktop/profile-proposals"
    listed = (await client.get(path)).json()["result"]
    (row,) = listed["proposals"]
    assert row["kind"] == "skill" and row["name"] == "marketing"
    assert row["fields"] == ["SKILL.md"] and row["state"] == "pending"
    assert row["stale"] is False and row["origin"]["trust"] == "untrusted"
    assert listed["counts"]["skill"] == {"marketing": 1}
    detail = (await client.get(path + "/" + record["id"])).json()["result"]
    (change,) = detail["changes"]
    assert change["field"] == "SKILL.md" and change["format"] == "text"
    assert change["before_text"] == _SKILL_BEFORE and change["after_text"] == _SKILL_AFTER
    body = {"request_id": str(uuid.uuid4())}
    accepted = await client.post(path + "/" + record["id"] + "/accept", json=body)
    assert accepted.status_code == 200, accepted.text
    result = accepted.json()["result"]
    assert result["proposal"]["state"] == "accepted"
    assert result["profile"]["name"] == "marketing"
    assert _skill_text(root) == _SKILL_AFTER
    assert any(p.name.startswith("skill-") for p in (root / "hub" / "backups").iterdir())


async def test_skill_proposal_edit_accept_reject_and_stale(api):
    client, root = api
    client._transport.app.include_router(desktop_profile_proposals.router)
    record = _skill_proposal(root)
    path = "/v1/desktop/profile-proposals/" + record["id"]
    rejected = await client.post(path + "/reject", json={"request_id": str(uuid.uuid4())})
    assert rejected.status_code == 200, rejected.text
    assert rejected.json()["result"]["proposal"]["state"] == "rejected"
    assert _skill_text(root) == _SKILL_BEFORE
    assert not (root / "hub" / "backups").exists() or not any((root / "hub" / "backups").iterdir())
    # A second proposal, then the skill moves under it: Accept is a 409 that
    # carries the current image, and nothing is written.
    from local_operator import profile_proposals as P
    from local_operator.server.utils import desktop_skill_writes as writes

    before = P.skill_image(root, "marketing")
    assert before is not None
    second = P.file(
        root,
        profile_kind="skill",
        operation="update",
        target_id="marketing",
        name="marketing",
        before=before,
        after={**before, "files": {"SKILL.md": _SKILL_BEFORE + "Second.\n"}},
        session_id="sess-2",
    )
    # Edit-then-accept may only touch proposed files.
    invalid = await client.post(
        "/v1/desktop/profile-proposals/" + second["id"] + "/accept",
        json={"request_id": str(uuid.uuid4()), "after": {"references/x.md": "x"}},
    )
    assert invalid.status_code == 422, invalid.text
    from local_operator.hub_sync.skill_document import SkillDocument

    fingerprint = SkillDocument.from_directory(root / "skills" / "marketing").fingerprint
    writes.update(
        root,
        "marketing",
        [{"path": "SKILL.md", "content": "---\nname: marketing\n" "description: Moved.\n---\n"}],
        fingerprint,
    )
    stale = await client.post(
        "/v1/desktop/profile-proposals/" + second["id"] + "/accept",
        json={"request_id": str(uuid.uuid4())},
    )
    assert stale.status_code == 409, stale.text
    assert stale.json()["detail"]["code"] == "profile_proposal_stale"
    assert "Moved." in stale.json()["detail"]["current"]["files"]["SKILL.md"]
    assert "Second." not in _skill_text(root)
