"""Organization sharing on the Radient client (design §8.3).

These methods speak the same ``{msg, result}`` envelope as the rest of the hub
API and carry the credential the client was constructed with; what the tests
pin is the SHAPE of each call (path, query params, the document untouched) and
that refusals keep their machine-readable half (``code``/``details``), because
the CLI renders a different next step for ``name_taken`` than for
``team_plan_required``.
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import SecretStr

from local_operator.clients._http import APIError
from local_operator.clients.radient import RadientClient, build_instruction_set_document


@pytest.fixture
def base_url() -> str:
    return "https://api.test.radient.com"


@pytest.fixture
def radient_client(base_url: str) -> RadientClient:
    return RadientClient(api_key=SecretStr("test_api_key"), base_url=base_url)


def _envelope(result: Any, *, status: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = {"msg": "ok", "result": result}
    return response


def _refusal(status: int, body: Dict[str, Any]) -> requests.exceptions.HTTPError:
    response = MagicMock()
    response.status_code = status
    response.content = json.dumps(body).encode()
    return requests.exceptions.HTTPError("refused", response=response)


def test_list_memberships_reads_the_person_scoped_endpoint(
    radient_client: RadientClient, base_url: str
) -> None:
    memberships = [
        {
            "tenant_id": "home-1",
            "tenant_name": "Home",
            "role": "owner",
            "status": "active",
            "is_home": True,
            "plan": {"status": "none", "seats": None},
        },
        {
            "tenant_id": "org-a",
            "tenant_name": "Org A",
            "role": "admin",
            "status": "active",
            "is_home": False,
            "plan": {"status": "active", "seats": 3},
        },
    ]

    with patch("requests.get", return_value=_envelope({"memberships": memberships})) as mock_get:
        result = radient_client.list_memberships()

    assert result == memberships
    args, kwargs = mock_get.call_args
    assert args[0] == f"{base_url}/me/memberships"
    assert kwargs["headers"]["Authorization"] == "Bearer test_api_key"
    assert kwargs["headers"]["Content-Type"] == "application/json"


def test_list_memberships_without_a_list_is_empty_not_an_error(
    radient_client: RadientClient,
) -> None:
    with patch("requests.get", return_value=_envelope({})):
        assert radient_client.list_memberships() == []


def test_list_org_agents_pins_the_org_namespace(
    radient_client: RadientClient, base_url: str
) -> None:
    """``visibility=org`` is sent explicitly: the route lists one namespace."""
    result = {
        "page": 1,
        "per_page": 20,
        "total_pages": 1,
        "total_records": 1,
        "records": [{"id": "org-agent-1", "name": "Org Screener", "visibility": "org"}],
    }

    with patch("requests.get", return_value=_envelope(result)) as mock_get:
        listed = radient_client.list_org_agents("org-a")

    assert listed == result
    args, kwargs = mock_get.call_args
    assert args[0] == f"{base_url}/tenants/org-a/agents"
    assert kwargs["params"] == {"visibility": "org", "page": 1, "per_page": 20}


def test_list_org_teams_returns_the_workspace_rows(
    radient_client: RadientClient, base_url: str
) -> None:
    teams = [{"id": "team-1", "name": "Release Crew"}]

    with patch("requests.get", return_value=_envelope({"teams": teams})) as mock_get:
        listed = radient_client.list_org_teams("org-a")

    assert listed == teams
    args, _kwargs = mock_get.call_args
    assert args[0] == f"{base_url}/tenants/org-a/teams"


def test_get_team_pulls_the_full_document_by_id(
    radient_client: RadientClient, base_url: str
) -> None:
    """The pull path carries the brief; the tenant stays out of the URL."""
    document = {
        "id": "team-1",
        "tenant_id": "org-a",
        "name": "Release Crew",
        "instructions": "You ship the release.",
        "members": [{"role": "coder", "kind": "agent", "count": 2}],
        "version": "1.0.0",
    }

    with patch("requests.get", return_value=_envelope(document)) as mock_get:
        pulled = radient_client.get_team("team-1")

    assert pulled == document
    args, _kwargs = mock_get.call_args
    assert args[0] == f"{base_url}/teams/team-1"


def test_org_calls_refuse_an_upstream_redirect(
    radient_client: RadientClient, base_url: str
) -> None:
    """A 3xx is refused, never followed (security review round 1, S-2).

    Followed, a same-host redirect re-sends the caller's bearer to wherever it
    points; the desktop transport already refuses these, and every org-capable
    call now does the same.
    """
    document = {"name": "Coder", "description": "Writes code.", "version": "1.0.0"}
    calls = (
        ("requests.get", lambda: radient_client.list_memberships()),
        ("requests.get", lambda: radient_client.get_team("team-1")),
        ("requests.post", lambda: radient_client.publish_agent_instruction_set(document)),
        (
            "requests.put",
            lambda: radient_client.republish_agent_instruction_set("hub-1", document),
        ),
        ("requests.post", lambda: radient_client.publish_team_document(document, "org-a")),
    )

    for target, call in calls:
        redirected = MagicMock()
        redirected.status_code = 302
        with patch(target, return_value=redirected) as mock_call:
            with pytest.raises(APIError) as exc_info:
                call()
        assert "unexpected redirect" in str(exc_info.value)
        assert mock_call.call_args.kwargs["allow_redirects"] is False


def test_publish_team_document_posts_with_the_tenant_query(
    radient_client: RadientClient, base_url: str
) -> None:
    document = {
        "name": "Release Crew",
        "description": "",
        "manager": "manager",
        "members": [{"role": "coder", "kind": "agent", "count": 2}],
        "instructions": "You ship the release.",
        "project": "",
        "version": "1.0.0",
    }
    result = {"team": {"id": "team-1", "name": "Release Crew", "version": "1.0.0"}}

    with patch("requests.post", return_value=_envelope(result, status=201)) as mock_post:
        published = radient_client.publish_team_document(document, "org-a")

    assert published == result
    args, kwargs = mock_post.call_args
    assert args[0] == f"{base_url}/teams/publish"
    assert kwargs["params"] == {"tenant_id": "org-a"}
    assert kwargs["json"] == document


def test_publish_team_document_refuses_an_empty_tenant(radient_client: RadientClient) -> None:
    with pytest.raises(ValueError):
        radient_client.publish_team_document({"name": "X"}, "  ")


def test_publish_agent_instruction_set_org_target_rides_on_query_params(
    radient_client: RadientClient, base_url: str
) -> None:
    """The org target is a query param; the document shape is unchanged (§4.4)."""
    document = build_instruction_set_document(
        name="Coder",
        description="Writes code.",
        instructions="You write code.",
        kind="role",
        version="1.0.0",
    )
    result = {"agent_id": "hub-1", "name": "Coder", "version": "1.0.0"}

    with patch("requests.post", return_value=_envelope(result, status=201)) as mock_post:
        published = radient_client.publish_agent_instruction_set(
            document, visibility="org", tenant_id="org-a"
        )

    assert published == result
    args, kwargs = mock_post.call_args
    assert args[0] == f"{base_url}/agents/publish"
    assert kwargs["params"] == {"visibility": "org", "tenant_id": "org-a"}
    assert kwargs["json"] == document
    assert "visibility" not in document and "tenant_id" not in document


def test_publish_agent_instruction_set_without_a_target_is_todays_call(
    radient_client: RadientClient,
) -> None:
    document = {"document_type": "agent-instruction-set", "document_version": 1}

    with patch("requests.post", return_value=_envelope({"agent_id": "hub-1"})) as mock_post:
        radient_client.publish_agent_instruction_set(document)

    _args, kwargs = mock_post.call_args
    assert kwargs["params"] == {}


@pytest.mark.parametrize(
    ("visibility", "tenant_id", "message"),
    [
        ("org", None, "tenant_id is required when visibility=org"),
        (None, "org-a", "tenant_id is only valid together with visibility=org"),
        ("weird", None, 'visibility must be "public" or "org"'),
    ],
)
def test_publish_agent_instruction_set_refuses_a_malformed_target(
    radient_client: RadientClient, visibility: Any, tenant_id: Any, message: str
) -> None:
    """The hub's own input rule, refused locally instead of answered 400."""
    with patch("requests.post") as mock_post:
        with pytest.raises(ValueError) as exc_info:
            radient_client.publish_agent_instruction_set(
                {"name": "Coder"}, visibility=visibility, tenant_id=tenant_id
            )
    assert str(exc_info.value) == message
    mock_post.assert_not_called()


def test_org_refusals_keep_their_machine_readable_half(radient_client: RadientClient) -> None:
    refusal = _refusal(
        403,
        {
            "error": "the organization's plan does not cover this",
            "code": "team_plan_required",
            "details": {},
        },
    )

    with patch("requests.get", side_effect=refusal):
        with pytest.raises(APIError) as exc_info:
            radient_client.list_org_agents("org-a")

    assert exc_info.value.status_code == 403
    assert exc_info.value.code == "team_plan_required"
    assert str(exc_info.value) == "the organization's plan does not cover this"


def test_org_transport_failure_reports_no_status(radient_client: RadientClient) -> None:
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
        with pytest.raises(APIError) as exc_info:
            radient_client.get_team("team-1")

    assert exc_info.value.status_code is None
    assert str(exc_info.value) == "Could not pull the team from the Radient Agent Hub"


def test_get_agent_carries_the_bearer_only_when_asked(
    radient_client: RadientClient, base_url: str
) -> None:
    """The org pull's pre-flight reads the row AS THE PERSON; others stay anonymous.

    §8.2: an organization row is answered 404 to anyone who cannot prove
    membership, so the pull's tenant check passes ``with_credential=True`` --
    and every existing caller of ``get_agent`` keeps its pre-change anonymous
    request (the public default is untouched).
    """
    with patch("requests.get", return_value=_envelope({"id": "agent-1"})) as mock_get:
        radient_client.get_agent("agent-1")
        args, kwargs = mock_get.call_args
        assert args[0] == f"{base_url}/agents/agent-1"
        assert "Authorization" not in kwargs["headers"]

        radient_client.get_agent("agent-1", with_credential=True)
        assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer test_api_key"


def test_republish_agent_instruction_set_org_target_rides_on_query_params(
    radient_client: RadientClient, base_url: str
) -> None:
    """The republish's org target is the publish's query pair, on the PUT."""
    document = build_instruction_set_document(
        name="Coder",
        description="Writes code.",
        instructions="You write code.",
        kind="role",
        version="1.1.0",
    )
    result = {"agent_id": "hub-1", "name": "Coder", "version": "1.1.0"}

    with patch("requests.put", return_value=_envelope(result)) as mock_put:
        republished = radient_client.republish_agent_instruction_set(
            "hub-1", document, visibility="org", tenant_id="org-a"
        )

    assert republished == result
    args, kwargs = mock_put.call_args
    assert args[0] == f"{base_url}/agents/hub-1/publish"
    assert kwargs["params"] == {"visibility": "org", "tenant_id": "org-a"}
    assert kwargs["json"] == document
    assert "visibility" not in document and "tenant_id" not in document


def test_republish_agent_instruction_set_without_a_target_is_todays_call(
    radient_client: RadientClient,
) -> None:
    document = {"document_type": "agent-instruction-set", "document_version": 1}

    with patch("requests.put", return_value=_envelope({"agent_id": "hub-1"})) as mock_put:
        radient_client.republish_agent_instruction_set("hub-1", document)

    _args, kwargs = mock_put.call_args
    assert kwargs["params"] == {}


def test_republish_agent_instruction_set_refuses_a_malformed_target(
    radient_client: RadientClient,
) -> None:
    with patch("requests.put") as mock_put:
        with pytest.raises(ValueError) as exc_info:
            radient_client.republish_agent_instruction_set(
                "hub-1", {"name": "Coder"}, visibility="org"
            )

    assert str(exc_info.value) == "tenant_id is required when visibility=org"
    mock_put.assert_not_called()
