"""
Tests for the instruction-set publication routes.

These cover the two properties the publication path exists to establish: the hub
receives a DOCUMENT (an instruction set and nothing else about the machine it was
authored on), and every refusal reaches the desktop app as a structure with a code
it can switch on rather than as one prose sentence.
"""

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.clients._http import APIError
from local_operator.clients.radient import INSTRUCTION_SET_FIELDS
from local_operator.server.routes.agents import (
    AgentPublicationRequest,
    _instruction_set_fields,
)
from local_operator.types import (
    AgentState,
    CodeExecutionResult,
    ConversationRecord,
    ConversationRole,
)

#: Every private thing a published agent must not carry, written into the agent's
#: directory before a publish. The markers are unique strings so that a leak is
#: found by searching the wire body for the CONTENT, not only by checking keys: a
#: field could be renamed and still carry the bytes.
PRIVATE_MARKERS = (
    "SECRET-CONVERSATION",
    "SECRET-EXECUTION",
    "SECRET-LEARNING",
    "SECRET-PLAN",
    "SECRET-INSTRUCTION-DETAILS",
    "SECRET-CONTEXT",
    "SECRET-SECURITY-PROMPT",
    "SECRET-CWD",
    "SECRET-MODEL",
    "SECRET-HOSTING",
    "SECRET-LAST-MESSAGE",
)

#: Document keys that would mean something other than an instruction set is being
#: published. `_EXPORT_SKIP_NAMES` is where this guarantee lived for the archive
#: path; this is the same property, asserted on the wire.
FORBIDDEN_BODY_KEYS = (
    "conversation",
    "execution_history",
    "learnings",
    "schedules",
    "current_plan",
    "instruction_details",
    "context",
    "model",
    "hosting",
    "security_prompt",
    "current_working_directory",
    "last_message",
    "agent_system_prompt",
)


@pytest.fixture(autouse=True)
def fake_radient_credential():
    """Every publication route resolves a Radient credential first.

    Patched on the module the handler imports from at call time, so the route under
    test is the real one; only the credential lookup is replaced.
    """
    with patch(
        "local_operator.providers.radient_credentials.resolve_radient_credential",
        new=AsyncMock(return_value=SecretStr("test-key")),
    ) as resolver:
        yield resolver


def _new_agent(registry: AgentRegistry, **overrides) -> Any:
    """Register an agent, spelling out every ``AgentEditFields`` field.

    Spelled out rather than passed partially because pyright does not read
    pydantic's ``Field(None, ...)`` as a default here, so a short call is a type
    error; the existing server tests pass every field for the same reason.
    """
    fields: Dict[str, Any] = {
        "name": "coder",
        "security_prompt": None,
        "hosting": None,
        "model": None,
        "description": "Writes code.",
        "tags": None,
        "categories": None,
        "last_message": None,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "max_tokens": None,
        "stop": None,
        "frequency_penalty": None,
        "presence_penalty": None,
        "seed": None,
        "current_working_directory": None,
    }
    fields.update(overrides)
    return registry.create_agent(AgentEditFields(**fields))


def _agent_with_everything_private(registry: AgentRegistry) -> None:
    """Register an agent whose directory holds every kind of private state."""

    agent = _new_agent(
        registry,
        name="secretive-agent",
        security_prompt=PRIVATE_MARKERS[6],
        hosting=PRIVATE_MARKERS[9],
        model=PRIVATE_MARKERS[8],
        description="An agent with private state.",
        last_message=PRIVATE_MARKERS[10],
        tags=["role", "tools:read,grep", "delegate:yes", "seed:reviewer", "osint"],
        categories=["specialist"],
        current_working_directory=PRIVATE_MARKERS[7],
    )
    registry.set_agent_system_prompt(agent.id, "You have an instruction set.")
    registry.save_agent_state(
        agent.id,
        AgentState(
            version="v1",
            conversation=[
                ConversationRecord(
                    role=ConversationRole.USER,
                    content=PRIVATE_MARKERS[0],
                )
            ],
            execution_history=[
                CodeExecutionResult(
                    stdout=PRIVATE_MARKERS[1],
                    message=PRIVATE_MARKERS[1],
                    code=PRIVATE_MARKERS[1],
                )
            ],
            learnings=[PRIVATE_MARKERS[2]],
            current_plan=PRIVATE_MARKERS[3],
            instruction_details=PRIVATE_MARKERS[4],
            agent_system_prompt=None,
        ),
    )
    registry.save_agent_context(agent.id, {"note": PRIVATE_MARKERS[5]})


def _hub_envelope(result: Dict[str, Any]) -> MagicMock:
    response = MagicMock()
    response.status_code = 201
    response.json.return_value = {"msg": "Agent published successfully", "result": result}
    return response


@pytest.mark.asyncio
async def test_publish_body_carries_the_instruction_set_and_nothing_else(
    test_app_client, dummy_registry: AgentRegistry
) -> None:
    """THE property this program exists to restore, asserted on the wire bytes.

    The desktop app publishes by building a document here, so this exercises the
    real client against a patched transport and inspects the body that would leave
    the machine: its key set must be a subset of the document schema, and none of
    the agent's private content may appear anywhere in it.
    """
    _agent_with_everything_private(dummy_registry)
    agent = dummy_registry.get_agent_by_name("secretive-agent")
    assert agent is not None

    captured: Dict[str, Any] = {}

    def _post(url, headers=None, json=None, **kwargs):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json
        return _hub_envelope(
            {"agent_id": "hub-1", "name": "secretive-agent", "document_version": 1}
        )

    with patch("requests.post", side_effect=_post):
        response = await test_app_client.post(f"/v1/agents/{agent.id}/publish", json={})

    assert response.status_code == 200, response.text
    body = captured["body"]
    assert captured["url"].endswith("/agents/publish")

    # The key set is the document schema's, not a superset of it.
    assert set(body) <= set(INSTRUCTION_SET_FIELDS)
    for key in FORBIDDEN_BODY_KEYS:
        assert key not in body, f"{key} rode along in the publish body"

    # And the private CONTENT is absent, not merely its key: a renamed field that
    # still carried the bytes would pass the check above.
    serialized = json.dumps(body)
    for marker in PRIVATE_MARKERS:
        assert marker not in serialized, f"{marker} rode along in the publish body"

    # The instruction set itself is what the body is for.
    assert body["instructions"] == "You have an instruction set."
    assert body["name"] == "secretive-agent"
    assert body["document_type"] == "radient.agent-instruction-set"
    assert body["document_version"] == 1
    # `specialist` is a local category; the hub's `kind` carries it instead, and
    # the local profile-encoding tags stay local.
    assert body["kind"] == "specialist"
    assert body["tags"] == ["osint"]
    assert body["tools"] == ["read", "grep"]
    assert body["delegate"] is True


@pytest.mark.asyncio
async def test_publish_returns_the_hub_result(test_app_client, dummy_registry: AgentRegistry):
    """A published agent's identity comes from the hub, not from the request."""
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")
    result = {
        "agent_id": "hub-9",
        "name": "coder",
        "version": "1.0.0",
        "document_version": 1,
        "moderation": {"verdict": "allow", "model": "deepseek/deepseek-v4.1-flash"},
    }

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.publish_agent_instruction_set.return_value = result
        response = await test_app_client.post(f"/v1/agents/{agent.id}/publish", json={})

    assert response.status_code == 200
    assert response.json()["result"] == result
    document = mock_client.return_value.publish_agent_instruction_set.call_args.args[0]
    assert document["instructions"] == "You write code."


@pytest.mark.asyncio
async def test_publish_applies_caller_overrides(test_app_client, dummy_registry: AgentRegistry):
    """The renderer's edits reach the document; the rest is read from the row."""
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")
    result = {"agent_id": "hub-9", "name": "coder-2"}

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.publish_agent_instruction_set.return_value = result
        response = await test_app_client.post(
            f"/v1/agents/{agent.id}/publish",
            json={
                "document": {
                    "name": "coder-2",
                    "categories": ["software"],
                    "when_to_use": "Writing code.",
                    "version": "2.0.0",
                }
            },
        )

    assert response.status_code == 200, response.text
    document = mock_client.return_value.publish_agent_instruction_set.call_args.args[0]
    assert document["name"] == "coder-2"
    assert document["categories"] == ["software"]
    assert document["when_to_use"] == "Writing code."
    assert document["version"] == "2.0.0"
    assert document["instructions"] == "You write code."


@pytest.mark.asyncio
async def test_publish_refuses_an_unknown_override_field(
    test_app_client, dummy_registry: AgentRegistry
) -> None:
    """An undefined key is refused by name, as the hub refuses it."""
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        response = await test_app_client.post(
            f"/v1/agents/{agent.id}/publish",
            json={"document": {"instructions": "x", "conversation": []}},
        )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "invalid_instruction_set"
    assert detail["details"] == {
        "field": "conversation",
        "rule": "is not a recognised field",
    }
    mock_client.assert_not_called()


@pytest.mark.asyncio
async def test_publish_refuses_a_request_body_that_is_not_the_request(
    test_app_client, dummy_registry: AgentRegistry
) -> None:
    """A misspelled routing key is a validation error, not a silent no-op."""
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")

    response = await test_app_client.post(
        f"/v1/agents/{agent.id}/publish", json={"documnet": {"name": "coder"}}
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_publish_does_not_truncate_an_over_long_instruction_body(
    test_app_client, dummy_registry: AgentRegistry
) -> None:
    """An over-long body is refused, never silently cut to the cap.

    The profile's instructions are truncated for delegation (they are billed every
    turn); a publication truncated the same way would be a document the author
    never wrote, published under their name.
    """
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "x" * 9000)

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        response = await test_app_client.post(f"/v1/agents/{agent.id}/publish", json={})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "invalid_instruction_set"
    assert detail["details"]["field"] == "instructions"
    assert detail["details"]["rule"] == "must be at most 8000 characters"
    mock_client.assert_not_called()


@pytest.mark.asyncio
async def test_publish_refuses_an_agent_that_cannot_be_a_document(
    test_app_client, dummy_registry: AgentRegistry
) -> None:
    """A row the document schema cannot describe is refused with the field."""
    agent = _new_agent(dummy_registry, name="coder", description="")

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        response = await test_app_client.post(f"/v1/agents/{agent.id}/publish", json={})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["details"] == {"field": "description", "rule": "must not be empty"}
    mock_client.assert_not_called()


@pytest.mark.asyncio
async def test_publish_requires_a_radient_credential(
    test_app_client, dummy_registry: AgentRegistry, fake_radient_credential
) -> None:
    """Without a credential the request stops before the hub is called."""
    fake_radient_credential.return_value = None
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        response = await test_app_client.post(f"/v1/agents/{agent.id}/publish", json={})

    assert response.status_code == 401
    assert response.json()["detail"] == "RADIENT_API_KEY is required"
    mock_client.assert_not_called()


@pytest.mark.asyncio
async def test_publish_unknown_local_agent_keeps_the_local_404(
    test_app_client, dummy_registry: AgentRegistry
) -> None:
    """A missing LOCAL agent is not the hub's `agent_not_found`.

    The hub's code means "that listing no longer exists" and its copy offers
    "refresh the hub"; a typo in a local id is a different failure with a different
    fix, so it keeps this file's existing prose 404 rather than borrowing a code
    that would misdescribe it.
    """
    response = await test_app_client.post("/v1/agents/does-not-exist/publish", json={})

    assert response.status_code == 404
    assert response.json()["detail"] == "Agent with ID does-not-exist not found"


@pytest.mark.asyncio
async def test_publish_local_failure_is_a_local_500(
    test_app_client, dummy_registry: AgentRegistry
) -> None:
    """A failure on this side is not reported as the hub's."""
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")

    with patch.object(dummy_registry, "get_agent_system_prompt", side_effect=OSError("disk gone")):
        response = await test_app_client.post(f"/v1/agents/{agent.id}/publish", json={})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["code"] == "local_failure"
    assert "disk gone" not in json.dumps(detail)


@pytest.mark.parametrize(
    "code,status,details",
    [
        ("name_taken", 409, {"existing_agent_id": "hub-9", "owned_by_caller": False}),
        ("name_reserved_builtin", 409, {"builtin_name": "reviewer"}),
        ("moderation_rejected", 422, {"categories": ["fraud_or_deception"]}),
        ("invalid_instruction_set", 422, {"field": "kind"}),
        ("payload_too_large", 413, {"limit_bytes": 65536}),
        ("moderation_unavailable", 503, {"attempts": 2}),
        ("not_owner", 403, {}),
        ("agent_not_found", 404, {}),
    ],
)
@pytest.mark.asyncio
async def test_publish_maps_every_hub_code_onto_its_status_and_structure(
    test_app_client, dummy_registry: AgentRegistry, code, status, details
) -> None:
    """One test per code in the vocabulary: the renderer switches on `code` alone.

    The status is the hub's own, and the detail is a structure — code, message,
    details — because a duplicate name, a reserved built-in, a moderation refusal
    and an oversized document are indistinguishable behind one prose sentence.
    """
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.publish_agent_instruction_set.side_effect = APIError(
            "The hub refused this publication.", status_code=status, code=code, details=details
        )
        response = await test_app_client.post(f"/v1/agents/{agent.id}/publish", json={})

    assert response.status_code == status
    detail = response.json()["detail"]
    assert detail["code"] == code
    assert detail["message"] == "The hub refused this publication."
    assert detail["details"] == details


@pytest.mark.asyncio
async def test_publish_reports_a_hub_failure_it_did_not_describe(
    test_app_client, dummy_registry: AgentRegistry
) -> None:
    """A failure without a code is a 502, not a rejection of the document."""
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.publish_agent_instruction_set.side_effect = APIError(
            "Could not publish the agent to the Radient Agent Hub (HTTP 500)",
            status_code=500,
        )
        response = await test_app_client.post(f"/v1/agents/{agent.id}/publish", json={})

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["code"] == "hub_unavailable"
    assert detail["details"] == {}


@pytest.mark.asyncio
async def test_republish_requires_the_hub_listing_id(
    test_app_client, dummy_registry: AgentRegistry
) -> None:
    """A republish names the listing it updates, or it is refused."""
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        response = await test_app_client.put(f"/v1/agents/{agent.id}/publish", json={})

    assert response.status_code == 422
    assert response.json()["detail"]["details"]["field"] == "hub_agent_id"
    mock_client.assert_not_called()


@pytest.mark.asyncio
async def test_republish_puts_the_document_to_the_hub_listing(
    test_app_client, dummy_registry: AgentRegistry
) -> None:
    """The hub listing id and the local agent id are different ids, both named."""
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")
    result = {"agent_id": "hub-9", "name": "coder", "version": "1.0.0"}

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.republish_agent_instruction_set.return_value = result
        response = await test_app_client.put(
            f"/v1/agents/{agent.id}/publish", json={"hub_agent_id": "hub-9"}
        )

    assert response.status_code == 200
    assert response.json()["result"] == result
    call = mock_client.return_value.republish_agent_instruction_set.call_args
    assert call.args[0] == "hub-9"
    assert call.args[1]["instructions"] == "You write code."
    # The republish path must not create a second listing.
    mock_client.return_value.publish_agent_instruction_set.assert_not_called()


@pytest.mark.asyncio
async def test_availability_returns_the_hubs_answer(
    test_app_client, dummy_registry: AgentRegistry
) -> None:
    """An unavailable name is an ANSWER, not a failure: 200 with the code."""
    payload = {
        "name": "coder",
        "name_key": "coder",
        "available": False,
        "code": "name_taken",
        "details": {"owned_by_caller": False},
    }

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.check_agent_name_availability.return_value = payload
        response = await test_app_client.get("/v1/agent-name-availability?name=coder")

    assert response.status_code == 200
    assert response.json()["result"] == payload
    assert mock_client.return_value.check_agent_name_availability.call_args.args[0] == "coder"


@pytest.mark.asyncio
async def test_availability_maps_a_hub_refusal(test_app_client, dummy_registry: AgentRegistry):
    """An illegal name is refused by the hub, with the hub's own code."""
    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.check_agent_name_availability.side_effect = APIError(
            "The agent document is not valid: name must not contain whitespace.",
            status_code=422,
            code="invalid_instruction_set",
            details={"field": "name", "rule": "must not contain whitespace"},
        )
        response = await test_app_client.get("/v1/agent-name-availability?name=two%20words")

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "invalid_instruction_set"
    assert detail["details"]["field"] == "name"


@pytest.mark.asyncio
async def test_availability_reports_a_hub_it_could_not_reach(
    test_app_client, dummy_registry: AgentRegistry
) -> None:
    """A check that never reached the hub says so rather than blaming the name."""
    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.check_agent_name_availability.side_effect = APIError(
            "Could not check the agent name on the Radient Agent Hub"
        )
        response = await test_app_client.get("/v1/agent-name-availability?name=coder")

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "hub_unavailable"


def test_a_plain_conversational_row_publishes_as_a_role(
    dummy_registry: AgentRegistry,
) -> None:
    """A plain conversational row publishes as a role.

    Locally, role and specialist are a tag and a category; the hub needs one
    explicit `kind`. Refusing a row that is neither would leave the user unable to
    publish an agent for a reason the dialog cannot explain or offer a fix for.
    """
    agent = _new_agent(dummy_registry, name="plain", description="Just an agent.", tags=["misc"])
    dummy_registry.set_agent_system_prompt(agent.id, "You help.")

    fields = _instruction_set_fields(dummy_registry, agent, {})

    assert fields["kind"] == "role"
    assert fields["tags"] == ["misc"]
    assert "tools" not in fields
    assert "when_to_use" not in fields
    assert "categories" not in fields
    assert fields["version"] == "1.0.0"


def test_publication_request_requires_hub_id_within_the_name_bound() -> None:
    """The request model is strict about its own shape."""
    assert AgentPublicationRequest().document == {}
    assert AgentPublicationRequest(hub_agent_id="hub-1").hub_agent_id == "hub-1"
    with pytest.raises(Exception):
        AgentPublicationRequest(hub_agent_id="x" * 129)
