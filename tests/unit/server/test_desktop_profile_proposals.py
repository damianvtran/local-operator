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
