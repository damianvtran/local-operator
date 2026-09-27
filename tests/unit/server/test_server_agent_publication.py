"""
Tests for the instruction-set publication routes.

These cover the two properties the publication path exists to establish: the hub
receives a DOCUMENT (an instruction set and nothing else about the machine it was
authored on), and every refusal reaches the desktop app as a structure with a code
it can switch on rather than as one prose sentence.
"""

import json
from typing import Any, Dict, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from local_operator.agents import AgentEditFields, AgentRegistry, instruction_set_fields
from local_operator.clients._http import REDACTION_MARKER, APIError
from local_operator.clients.radient import INSTRUCTION_SET_FIELDS
from local_operator.env import (
    DEFAULT_RADIENT_API_BASE_URL,
    resolve_radient_api_base_url,
)
from local_operator.providers.radient_credentials import (
    ORG_LOGIN_REMEDY,
    org_destination_refused_sentence,
)
from local_operator.server.routes.agents import (
    PUBLICATION_STATUS_BY_CODE,
    AgentPublicationRequest,
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
        # The tags a person wrote ("osint"), the profile-encoding keys, and one
        # of EVERY provenance prefix: publishing must strip all of them, and a
        # leaked `hub:<id>` is the sharpest case — it would point a consumer's
        # future sync at another listing (agent review round 1, m2).
        tags=[
            "role",
            "tools:read,grep",
            "delegate:yes",
            "seed:reviewer",
            "seed_version:1.0.0",
            f"seed_sha256:{'c' * 64}",
            "hub:abc-123",
            f"hub_sha256:{'d' * 64}",
            "osint",
        ],
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
    # the local profile-encoding AND provenance tags stay local — the seeded
    # fixture above carries every prefix, so this one assertion covers them.
    assert body["kind"] == "specialist"
    assert body["tags"] == ["osint"]
    assert body["tools"] == ["read", "grep"]
    assert body["delegate"] is True


@pytest.mark.asyncio
async def test_publish_sends_a_spaced_name_rather_than_refusing_it(
    test_app_client, dummy_registry: AgentRegistry
) -> None:
    """A name with ordinary spaces is the hub's to accept or normalise.

    agents the user publishes can be named the way they read ("Product Manager"),
    and the hub is the authority on the stored spelling. Refusing it here would be
    a client bound stricter than the server — the one direction the contract calls
    a bug report — so the override reaches the document untouched.
    """
    agent = _new_agent(dummy_registry)
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.publish_agent_instruction_set.return_value = {
            "agent_id": "hub-9",
            "name": "Product Manager",
        }
        response = await test_app_client.post(
            f"/v1/agents/{agent.id}/publish",
            json={"document": {"name": "  Product  Manager  "}},
        )

    assert response.status_code == 200, response.text
    document = mock_client.return_value.publish_agent_instruction_set.call_args.args[0]
    assert document["name"] == "Product  Manager"


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
        # The ninth code, and the one whose next step differs from name_taken's:
        # no row holds the name, a concurrent write does, and the caller retries.
        # `retryable` is what says so mechanically.
        ("name_claim_in_flight", 409, {"owned_by_caller": False, "retryable": True}),
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

    `details` here is the hub's own, EXACTLY as it sent it, and no `hub_code` is added
    alongside it: on this arm the outward `code` IS the hub's code, so carrying it a
    second time would be an echoed duplicate a renderer could only read as a
    disagreement. The auth arm is the one where the two differ (see below).
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
async def test_publish_masks_a_credential_the_hub_put_in_the_details(
    test_app_client, dummy_registry: AgentRegistry
) -> None:
    """The known-code arm carries the hub's `details`, so it carries the masker with them.

    QA round 2's A4 cell and review round 2's MINOR 1, both on the wire: the identical
    hub text that the `message` half masks arrived VERBATIM in `details` -- the field
    this contract says the renderer reads (`rule`, `field`), so an unscrubbed one is a
    rendered sentence and not a hidden field. Asserted on the RESPONSE, not on the
    exception, because what a caller receives is a formatted body: an arm that builds
    one is where a guarantee can be lost, whatever produced the refusal underneath.
    """
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")
    canary = "sk-radient-PUBLICATION-CANARY-0000"

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.publish_agent_instruction_set.side_effect = APIError(
            f"the hub saw Authorization: Bearer {canary} and refused",
            status_code=422,
            code="moderation_rejected",
            details={
                "note": f"Authorization: Bearer {canary}",
                "rule": f"quoted back: {canary}",
                "existing_agent_id": "hub-7",
                "categories": ["fraud_or_deception"],
            },
        )
        response = await test_app_client.post(f"/v1/agents/{agent.id}/publish", json={})

    assert response.status_code == 422
    assert canary not in response.text
    detail = response.json()["detail"]
    assert detail["code"] == "moderation_rejected"
    # Masked, not dropped: the field and the classification a renderer switches on are
    # still there, and the nested list keeps its shape.
    assert detail["details"]["existing_agent_id"] == "hub-7"
    assert detail["details"]["categories"] == ["fraud_or_deception"]
    assert REDACTION_MARKER in json.dumps(detail)


@pytest.mark.parametrize(
    "hub_status,hub_code",
    [
        # The two shapes a hub-side credential rejection can take: with no code at
        # all, and with one this proxy's vocabulary does not define. Neither was
        # in the code table, so both fell through to `hub_unavailable`/502.
        (401, None),
        (401, "unauthorized"),
        (401, "invalid_api_key"),
        # A code that CARRIES information, and the one QA round 2's B3 cell uses: an
        # expired key and a revoked one answer with the same outward code, so if the
        # hub's own code is dropped the two are indistinguishable to a renderer.
        (401, "token_expired"),
        # A 403 with no recognised code is the same class of refusal -- the hub
        # answered and refused the caller -- where `not_owner` is a different one
        # and keeps its own code (the table test above pins that).
        (403, None),
    ],
    ids=[
        "401-no-code",
        "401-unknown-code",
        "401-other-spelling",
        "401-expired-key",
        "403-no-code",
    ],
)
@pytest.mark.asyncio
async def test_publish_reports_a_refused_credential_rather_than_a_hub_outage(
    test_app_client, dummy_registry: AgentRegistry, hub_status, hub_code
) -> None:
    """A refused credential asks the user to re-authenticate, not to retry.

    The whole point of the `code` field is the next step it selects, and
    `hub_unavailable` selects "retry" -- the one thing that cannot fix an expired
    key, and the exact instruction the wrong code sent. The hub's own sentence
    travels in `message` unchanged, so a caller that ignores `code` loses nothing.

    The hub's own code is CARRIED in `details.hub_code` rather than published as the
    outward `code`: the outward code is a closed set in the renderer, and a member it
    has no treatment for is the objection that keeps the 429 out of the vocabulary.
    Carried, it is additive -- a renderer that does not read the key answers exactly
    as it did -- and it is the difference between an expired key and a revoked one.
    The hub's own `details` still do NOT travel: an unrecognised refusal's details are
    the one part of that body this proxy has no shape for.
    """
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.publish_agent_instruction_set.side_effect = APIError(
            "Invalid API key provided.",
            status_code=hub_status,
            code=hub_code,
            # An unrecognised refusal's details have no shape this proxy knows, and
            # none travel: nothing here may reach the response.
            details={"echo": "whatever the body held"},
        )
        response = await test_app_client.post(f"/v1/agents/{agent.id}/publish", json={})

    assert response.status_code == hub_status
    detail = response.json()["detail"]
    assert detail["code"] == "hub_unauthorized"
    assert detail["message"] == "Invalid API key provided."
    # The hub's code is carried when it sent one, and nothing is invented when it did
    # not -- the live hub's prose-only 401 answers exactly as it did before.
    assert detail["details"] == ({"hub_code": hub_code} if hub_code else {})
    assert "echo" not in json.dumps(detail)


@pytest.mark.asyncio
async def test_a_hub_rate_limit_is_not_reported_as_a_refused_credential(
    test_app_client, dummy_registry: AgentRegistry
) -> None:
    """A throttle keeps the retry-shaped answer; only the credential arm changed.

    DELIBERATE, and named in the remediation comment rather than fixed here: a
    rate limit wants its own code (with the retry semantics that go with it), and
    inventing one in this PR would be a code no renderer knows. It stays on the
    retry arm, which is the correct ACTION even though `hub_unavailable` is not the
    correct word for it.
    """
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.publish_agent_instruction_set.side_effect = APIError(
            "rate limited", status_code=429, code="too_many_requests"
        )
        response = await test_app_client.post(f"/v1/agents/{agent.id}/publish", json={})

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "hub_unavailable"


#: The fallback status per hub code, written out rather than read from the table
#: under test. Reading them would make the test agree with ANY table, which is how
#: the table stayed mutation-blind through review round 1: sabotaging every value
#: left the suite green because every expectation was re-derived from the sabotage.
#: A code added to the table without a case here still fails -- see the test below.
STATUS_FALLBACK_CASES: Tuple[Tuple[str, int], ...] = (
    ("invalid_instruction_set", 422),
    ("invalid_team_document", 422),
    ("moderation_rejected", 422),
    ("payload_too_large", 413),
    ("name_taken", 409),
    ("name_claim_in_flight", 409),
    ("name_reserved_builtin", 409),
    ("not_owner", 403),
    ("agent_not_found", 404),
    ("team_not_found", 404),
    ("moderation_unavailable", 503),
    # Organization sharing's frozen membership refusals (§2.2): known codes,
    # so they keep their 403 instead of falling to the credential arm.
    ("not_a_member", 403),
    ("insufficient_role", 403),
    ("team_plan_required", 403),
)


def test_every_mapped_hub_code_has_a_pinned_fallback_status() -> None:
    """No code may sit in the table without a literal expectation of its status."""
    assert {code for code, _status in STATUS_FALLBACK_CASES} == set(PUBLICATION_STATUS_BY_CODE)


@pytest.mark.parametrize(
    "code,fallback_status",
    STATUS_FALLBACK_CASES,
    ids=[case[0] for case in STATUS_FALLBACK_CASES],
)
@pytest.mark.asyncio
async def test_publish_falls_back_to_the_mapped_status_when_the_hub_sent_none(
    test_app_client, dummy_registry: AgentRegistry, code, fallback_status
) -> None:
    """Every value in the table is reachable, and sabotaging one fails this test.

    ``api_error_from_response`` always sets the status from the response, so the
    table's fallback branch cannot be entered through the client -- the other test
    per code therefore passes its left-hand side every time and pinned nothing.
    Review round 1 proved that by setting every value to 599 and watching 102 tests
    stay green. This is the case that pins the values themselves.
    """
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.publish_agent_instruction_set.side_effect = APIError(
            "The hub refused this publication.", status_code=None, code=code
        )
        response = await test_app_client.post(f"/v1/agents/{agent.id}/publish", json={})

    assert response.status_code == fallback_status
    assert response.json()["detail"]["code"] == code


@pytest.mark.parametrize(
    "override,field",
    [
        # The coercion bugs review round 1 observed on head: bool("false") is True
        # and list("osint") is five one-character tags -- both published silently.
        ({"delegate": "false"}, "delegate"),
        ({"delegate": 1}, "delegate"),
        ({"tags": "osint"}, "tags"),
        ({"tools": "read"}, "tools"),
        ({"categories": "software"}, "categories"),
        # And the shapes that escaped as an AttributeError reported as a 500
        # blaming this machine for a request the hub would have refused.
        ({"instructions": ["a"]}, "instructions"),
        ({"name": 42}, "name"),
        ({"when_to_use": None}, "when_to_use"),
        ({"tags": ["osint", 7]}, "tags"),
    ],
)
@pytest.mark.asyncio
async def test_publish_refuses_an_override_of_the_wrong_shape(
    test_app_client, dummy_registry: AgentRegistry, override, field
) -> None:
    """A value of the wrong shape is refused with the hub's code, never coerced.

    The client bound must not be looser than the server's: a coercion publishes
    something the caller did not ask for, and an AttributeError reports the caller's
    malformed request as a failure of this machine.
    """
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        response = await test_app_client.post(
            f"/v1/agents/{agent.id}/publish", json={"document": override}
        )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "invalid_instruction_set"
    assert detail["details"]["field"] == field
    assert detail["details"]["rule"]
    mock_client.assert_not_called()


@pytest.mark.asyncio
async def test_publish_publishes_the_values_whose_shapes_are_right(
    test_app_client, dummy_registry: AgentRegistry
) -> None:
    """The positive control for the shape check: nothing valid is refused.

    ``delegate: false`` reaching the wire as ``false`` is the specific inversion the
    shape check exists to stop, so it is asserted on the document, not on the
    absence of an error.
    """
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.publish_agent_instruction_set.return_value = {"agent_id": "hub-9"}
        response = await test_app_client.post(
            f"/v1/agents/{agent.id}/publish",
            json={
                "document": {
                    "delegate": False,
                    "tags": ["osint"],
                    "tools": ["read"],
                    "instructions": "You help.",
                }
            },
        )

    assert response.status_code == 200, response.text
    document = mock_client.return_value.publish_agent_instruction_set.call_args.args[0]
    assert document["delegate"] is False
    assert document["tags"] == ["osint"]
    assert document["tools"] == ["read"]
    assert document["instructions"] == "You help."


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

    fields = instruction_set_fields(dummy_registry, agent, {})

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


# --- Organization targets (design §4.4/§4.7) ----------------------------------
#
# The org target is a SCOPE, not a document field: it rides on query params, the
# server resolves the signed-in account's OAuth token to spend (never the
# machine's API key), and the hub's frozen membership refusals reach the caller
# with their codes intact -- a member refusal is not a credential refusal.


@pytest.mark.asyncio
async def test_publish_org_target_sends_the_query_pair_and_spends_the_oauth_row(
    test_app_client,
    dummy_registry: AgentRegistry,
    fake_org_credential,
    fake_radient_credential,
) -> None:
    """`visibility=org` tags the target onto the query and rides the person's bearer.

    The client the route builds is the one whose ``api_key`` is the stored OAuth
    access token: the org path must never fall back to the machine's key (§2.2),
    so the key resolver is asserted untouched.
    """
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")
    result = {"agent_id": "hub-9", "name": "coder", "version": "1.0.0"}

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.publish_agent_instruction_set.return_value = result
        response = await test_app_client.post(
            f"/v1/agents/{agent.id}/publish?visibility=org&tenant_id=org-1", json={}
        )

    assert response.status_code == 200, response.text
    assert response.json()["result"] == result
    call = mock_client.return_value.publish_agent_instruction_set.call_args
    assert call.kwargs == {"visibility": "org", "tenant_id": "org-1"}
    spending = mock_client.call_args.kwargs
    assert spending["api_key"].get_secret_value() == "org-access-token"
    assert spending["base_url"] == resolve_radient_api_base_url()
    fake_org_credential.assert_awaited_once()
    fake_radient_credential.assert_not_called()


@pytest.mark.asyncio
async def test_publish_org_target_without_a_login_shows_the_re_login_remedy(
    test_app_client, dummy_registry: AgentRegistry, fake_org_credential
) -> None:
    """No OAuth row: 401 with the CLI's one-line remedy, before any hub call."""
    fake_org_credential.return_value = None
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        response = await test_app_client.post(
            f"/v1/agents/{agent.id}/publish?visibility=org&tenant_id=org-1", json={}
        )

    assert response.status_code == 401
    assert response.json()["detail"] == ORG_LOGIN_REMEDY
    mock_client.assert_not_called()


@pytest.mark.asyncio
async def test_publish_org_target_at_a_refused_hub_names_the_configuration_remedy(
    test_app_client, dummy_registry: AgentRegistry, fake_org_credential, monkeypatch
) -> None:
    """A non-canonical hub is refused with its own sentence, not the login one."""
    fake_org_credential.return_value = None
    monkeypatch.delenv("RADIENT_ORG_ALLOW_NONCANONICAL_BASE", raising=False)
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")

    with patch(
        "local_operator.providers.radient_credentials.configured_radient_base_url",
        return_value="http://127.0.0.1:4999/v1",
    ):
        with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
            response = await test_app_client.post(
                f"/v1/agents/{agent.id}/publish?visibility=org&tenant_id=org-1", json={}
            )

    assert response.status_code == 400
    # Byte-equal to the ONE shared sentence the CLI prints too (agent review
    # round 1, MINOR-2): the two surfaces render the same object.
    assert response.json()["detail"] == org_destination_refused_sentence(
        "http://127.0.0.1:4999/v1", DEFAULT_RADIENT_API_BASE_URL
    )
    mock_client.assert_not_called()


@pytest.mark.asyncio
async def test_publish_org_target_honours_the_noncanonical_opt_in(
    test_app_client, dummy_registry: AgentRegistry, fake_org_credential, monkeypatch
) -> None:
    """With the explicit opt-in the local hub is addressed, exactly as the CLI does."""
    monkeypatch.setenv("RADIENT_ORG_ALLOW_NONCANONICAL_BASE", "1")
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")

    with patch(
        "local_operator.providers.radient_credentials.configured_radient_base_url",
        return_value="http://127.0.0.1:4999/v1",
    ):
        with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
            mock_client.return_value.publish_agent_instruction_set.return_value = {
                "agent_id": "hub-9"
            }
            response = await test_app_client.post(
                f"/v1/agents/{agent.id}/publish?visibility=org&tenant_id=org-1", json={}
            )

    assert response.status_code == 200, response.text
    fake_org_credential.assert_awaited_once()
    assert mock_client.call_args.kwargs["base_url"] == "http://127.0.0.1:4999/v1"


@pytest.mark.parametrize(
    ("visibility", "tenant_id", "message"),
    [
        ("org", None, "tenant_id is required when visibility=org"),
        ("public", "org-1", "tenant_id is only valid together with visibility=org"),
        ("weird", None, 'visibility must be "public" or "org"'),
    ],
)
@pytest.mark.asyncio
async def test_publish_org_target_refuses_a_malformed_pair_before_any_credential(
    test_app_client,
    dummy_registry: AgentRegistry,
    fake_org_credential,
    visibility: Any,
    tenant_id: Any,
    message: str,
) -> None:
    """The hub's own 400 rule, refused locally -- and before a token is resolved."""
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    params: Dict[str, Any] = {"visibility": visibility}
    if tenant_id is not None:
        params["tenant_id"] = tenant_id

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        response = await test_app_client.post(
            f"/v1/agents/{agent.id}/publish", params=params, json={}
        )

    assert response.status_code == 400
    assert response.json()["detail"] == message
    mock_client.assert_not_called()
    fake_org_credential.assert_not_awaited()


@pytest.mark.asyncio
async def test_publish_without_a_target_stays_the_public_call(
    test_app_client, dummy_registry: AgentRegistry, fake_org_credential
) -> None:
    """No org params: no org kwargs, and only the API key is resolved."""
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.publish_agent_instruction_set.return_value = {"agent_id": "hub-9"}
        response = await test_app_client.post(f"/v1/agents/{agent.id}/publish", json={})

    assert response.status_code == 200, response.text
    call = mock_client.return_value.publish_agent_instruction_set.call_args
    assert call.kwargs == {}
    assert mock_client.call_args.kwargs["api_key"].get_secret_value() == "test-key"
    fake_org_credential.assert_not_awaited()


@pytest.mark.parametrize(
    ("code", "details"),
    [
        ("not_a_member", {}),
        ("insufficient_role", {"required": "admin"}),
        ("team_plan_required", {}),
    ],
)
@pytest.mark.asyncio
async def test_publish_org_refusals_keep_the_frozen_membership_codes(
    test_app_client,
    dummy_registry: AgentRegistry,
    fake_org_credential,
    code: str,
    details: Dict[str, Any],
) -> None:
    """A membership refusal is not a credential refusal: code and details survive."""
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.publish_agent_instruction_set.side_effect = APIError(
            "The hub refused this publication.", status_code=403, code=code, details=details
        )
        response = await test_app_client.post(
            f"/v1/agents/{agent.id}/publish?visibility=org&tenant_id=org-1", json={}
        )

    assert response.status_code == 403
    detail = response.json()["detail"]
    assert detail["code"] == code
    assert detail["details"] == details


@pytest.mark.asyncio
async def test_republish_org_target_sends_the_query_pair_to_the_listing(
    test_app_client, dummy_registry: AgentRegistry, fake_org_credential
) -> None:
    """A republish's org target goes out on the same query pair, against the listing."""
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")
    result = {"agent_id": "hub-9", "name": "coder", "version": "1.1.0"}

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.republish_agent_instruction_set.return_value = result
        response = await test_app_client.put(
            f"/v1/agents/{agent.id}/publish?visibility=org&tenant_id=org-1",
            json={"hub_agent_id": "hub-9"},
        )

    assert response.status_code == 200, response.text
    call = mock_client.return_value.republish_agent_instruction_set.call_args
    assert call.args[0] == "hub-9"
    assert call.kwargs == {"visibility": "org", "tenant_id": "org-1"}
    assert mock_client.call_args.kwargs["api_key"].get_secret_value() == "org-access-token"


@pytest.mark.asyncio
async def test_republish_org_target_without_a_login_shows_the_re_login_remedy(
    test_app_client, dummy_registry: AgentRegistry, fake_org_credential
) -> None:
    fake_org_credential.return_value = None
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        response = await test_app_client.put(
            f"/v1/agents/{agent.id}/publish?visibility=org&tenant_id=org-1",
            json={"hub_agent_id": "hub-9"},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == ORG_LOGIN_REMEDY
    mock_client.assert_not_called()


@pytest.mark.asyncio
async def test_publish_org_masks_a_reflected_credential(
    test_app_client, dummy_registry: AgentRegistry, fake_org_credential
) -> None:
    """The relayed result is scrubbed of this call's credential (S-1).

    An upstream is free to reflect the request it received -- the bearer
    included -- into its answer; the relay must not carry it back to the
    renderer.
    """
    agent = _new_agent(dummy_registry, name="coder", description="Writes code.")
    dummy_registry.set_agent_system_prompt(agent.id, "You write code.")
    result = {
        "agent_id": "hub-9",
        "moderation": {"echo": "Bearer org-access-token"},
    }

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.publish_agent_instruction_set.return_value = result
        response = await test_app_client.post(
            f"/v1/agents/{agent.id}/publish?visibility=org&tenant_id=org-1", json={}
        )

    assert response.status_code == 200, response.text
    assert "org-access-token" not in response.text
    assert "[redacted]" in response.text
