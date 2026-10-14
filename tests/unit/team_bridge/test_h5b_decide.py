"""H5b — the decide executor: only a human answer calls /decide, and the
no-call outcomes (decline/timeout/dismiss) never become a denial."""

from __future__ import annotations

import json

import httpx
import pytest

from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.decide import (
    NotAHumanDecision,
    decide_batch_from_human_selection,
    decide_from_human_answer,
    human_decision,
)

from tests.unit.team_bridge.test_client import FakeStore, _ok

pytestmark = pytest.mark.asyncio

TENANT = "ten_1"


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


def _make_client(handler) -> TeamBridgeClient:
    return TeamBridgeClient(
        None,
        base_url="https://hub.example/v1",
        store=FakeStore(),
        transport=httpx.MockTransport(handler),
    )


async def test_an_approve_drives_the_bound_receipt() -> None:
    requests: list[httpx.Request] = []
    client = _make_client(lambda request: (requests.append(request), _ok({"request": {}}))[1])
    answer = human_decision(message_id="bmsg_1", ask_id="a-3f9c", word="approve")
    await decide_from_human_answer(client, answer, tenant_id=TENANT)
    body = json.loads(requests[0].content)
    assert body["decision"] == "approved"
    assert body["ask_ref"] == {"ask_id": "a-3f9c", "question_id": "bmsg_1"}
    assert requests[0].url.path.endswith("/v1/messaging/messages/bmsg_1/decide")


async def test_the_no_call_outcomes_never_reach_the_wire() -> None:
    """Decline/timeout/dismiss are stated outcomes of the ask surface; mapping
    any of them onto a decide would fabricate the human's answer."""
    requests: list[httpx.Request] = []
    client = _make_client(lambda request: requests.append(request) and _ok({}))
    for word in ("decline", "timeout", "dismiss", "snooze"):
        with pytest.raises(NotAHumanDecision):
            human_decision(message_id="bmsg_1", ask_id="a-3f9c", word=word)
    with pytest.raises(NotAHumanDecision):
        human_decision(message_id="bmsg_1", ask_id="a-3f9c", word="approved")  # the SERVER word is not the CLIENTS word
    assert requests == []


async def test_a_batch_carries_only_the_explicit_selection() -> None:
    requests: list[httpx.Request] = []
    client = _make_client(lambda request: (requests.append(request), _ok({"results": []}))[1])
    selection = [
        human_decision(message_id="bmsg_1", ask_id="a-0001", word="approve"),
        human_decision(message_id="bmsg_2", ask_id="a-0002", word="deny"),
    ]
    await decide_batch_from_human_selection(client, selection, tenant_id=TENANT)
    body = json.loads(requests[0].content)
    assert [entry["message_id"] for entry in body["decisions"]] == ["bmsg_1", "bmsg_2"]
    assert [entry["decision"] for entry in body["decisions"]] == ["approved", "denied"]
    assert requests[0].url.path.endswith("/v1/messaging/requests/decide")

    with pytest.raises(NotAHumanDecision):
        await decide_batch_from_human_selection(client, [], tenant_id=TENANT)
    assert len(requests) == 1
