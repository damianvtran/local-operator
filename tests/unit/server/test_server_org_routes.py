"""The organization surfaces on the local server (design §4.7/§8.3).

Memberships and team publish/pull all resolve the signed-in account's OAuth
token -- never the machine's tenant API key -- and read hub refusals through the
publication family's one mapping. What these tests pin: the credential the
routes spend, the team document a publish sends (built by the SAME
``hub_team_document`` mapping the CLI uses), the local reconstruction and its
rename report on pull, and the frozen membership codes on the wire.
"""

from typing import Any
from unittest.mock import patch

import pytest

from local_operator.clients._http import APIError
from local_operator.server.routes.agents import ORG_LOGIN_REMEDY
from local_operator.teams import (
    TeamEditFields,
    TeamRegistry,
    hub_team_document,
    parse_members,
)


def _new_team(config_dir: Any) -> Any:
    """One local team the publish route can push."""
    registry = TeamRegistry(config_dir)
    return registry.create_team(
        TeamEditFields(
            name="release-crew",
            description="Ships the release.",
            manager="manager",
            members=parse_members(["coder:2"]),
            instructions="You ship the release.",
            project="rad-1",
        )
    )


@pytest.mark.asyncio
async def test_memberships_returns_the_person_scoped_list(
    test_app_client, temp_dir, fake_org_credential
) -> None:
    """GET /v1/memberships reads as the person and relays the rows."""
    memberships = [
        {
            "tenant_id": "org-1",
            "tenant_name": "Org One",
            "role": "admin",
            "status": "active",
            "is_home": False,
            "plan": {"status": "active", "seats": 3},
        }
    ]

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.list_memberships.return_value = memberships
        response = await test_app_client.get("/v1/memberships")

    assert response.status_code == 200, response.text
    assert response.json()["result"] == {"memberships": memberships}
    assert mock_client.call_args.kwargs["api_key"].get_secret_value() == "org-access-token"
    fake_org_credential.assert_awaited_once()


@pytest.mark.asyncio
async def test_memberships_without_a_login_shows_the_re_login_remedy(
    test_app_client, fake_org_credential
) -> None:
    fake_org_credential.return_value = None

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        response = await test_app_client.get("/v1/memberships")

    assert response.status_code == 401
    assert response.json()["detail"] == ORG_LOGIN_REMEDY
    mock_client.assert_not_called()


@pytest.mark.asyncio
async def test_memberships_maps_a_hub_refusal_onto_its_class(
    test_app_client, fake_org_credential
) -> None:
    """The hub's 401 is the re-authenticate class, not a local failure."""
    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.list_memberships.side_effect = APIError(
            "The hub refused this request.", status_code=401, code="unauthorized"
        )
        response = await test_app_client.get("/v1/memberships")

    assert response.status_code == 401
    detail = response.json()["detail"]
    assert detail["code"] == "hub_unauthorized"
    assert detail["details"]["hub_code"] == "unauthorized"


@pytest.mark.asyncio
async def test_team_publish_builds_the_document_and_posts_the_tenant(
    test_app_client, temp_dir, fake_org_credential
) -> None:
    """The route sends the CLI's one-way mapping, as the person."""
    team = _new_team(temp_dir)
    result = {"team": {"id": "team-9", "name": team.name, "version": "1.0.0"}}

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.publish_team_document.return_value = result
        response = await test_app_client.post(f"/v1/teams/{team.id}/publish?tenant_id=org-1")

    assert response.status_code == 200, response.text
    assert response.json()["result"] == result
    call = mock_client.return_value.publish_team_document.call_args
    assert call.args[0] == hub_team_document(team)
    assert call.args[1] == "org-1"
    assert mock_client.call_args.kwargs["api_key"].get_secret_value() == "org-access-token"


@pytest.mark.asyncio
async def test_team_publish_requires_the_tenant(
    test_app_client, temp_dir, fake_org_credential
) -> None:
    """Teams are org-only in v1: no tenant, no request."""
    team = _new_team(temp_dir)

    response = await test_app_client.post(f"/v1/teams/{team.id}/publish")

    assert response.status_code == 422
    fake_org_credential.assert_not_awaited()


@pytest.mark.asyncio
async def test_team_publish_unknown_local_team_keeps_the_local_404(
    test_app_client, temp_dir, fake_org_credential
) -> None:
    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        response = await test_app_client.post("/v1/teams/no-such-team/publish?tenant_id=org-1")

    assert response.status_code == 404
    assert response.json()["detail"] == "Team with ID no-such-team not found"
    mock_client.return_value.publish_team_document.assert_not_called()
    # The local refusal happens before any hub client exists, matching the
    # agent routes' ordering.
    mock_client.assert_not_called()


@pytest.mark.asyncio
async def test_team_publish_without_a_login_shows_the_re_login_remedy(
    test_app_client, temp_dir, fake_org_credential
) -> None:
    fake_org_credential.return_value = None
    team = _new_team(temp_dir)

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        response = await test_app_client.post(f"/v1/teams/{team.id}/publish?tenant_id=org-1")

    assert response.status_code == 401
    assert response.json()["detail"] == ORG_LOGIN_REMEDY
    mock_client.assert_not_called()


@pytest.mark.asyncio
async def test_team_publish_keeps_the_frozen_refusals(
    test_app_client, temp_dir, fake_org_credential
) -> None:
    team = _new_team(temp_dir)

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.publish_team_document.side_effect = APIError(
            "The hub refused this publication.", status_code=403, code="team_plan_required"
        )
        response = await test_app_client.post(f"/v1/teams/{team.id}/publish?tenant_id=org-1")

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "team_plan_required"


@pytest.mark.asyncio
async def test_team_pull_reconstructs_the_local_row(
    test_app_client, temp_dir, fake_org_credential
) -> None:
    """Pull by id (the designed path): a fresh local row, no rename notes."""
    document = {
        "id": "team-9",
        "tenant_id": "org-1",
        "name": "release-crew",
        "description": "Ships the release.",
        "manager": "manager",
        "members": [{"role": "coder", "kind": "agent", "count": 2}],
        "instructions": "You ship the release.",
        "project": "rad-1",
        "version": "1.0.0",
    }

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.get_team.return_value = document
        response = await test_app_client.get("/v1/teams/pull/team-9")

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["name"] == "release-crew"
    assert result["manager"] == "manager"
    assert result["renamed_from"] is None
    assert result["invalid_name"] is False
    stored = TeamRegistry(temp_dir).get_team_by_name("release-crew")
    assert stored is not None
    assert stored.member_count() == 2
    mock_client.return_value.get_team.assert_called_once_with("team-9")


@pytest.mark.asyncio
async def test_team_pull_refuses_a_document_owned_by_another_tenant(
    test_app_client, temp_dir, fake_org_credential
) -> None:
    """A statement of ownership that does not match is refused, not stored."""
    document = {"id": "team-9", "tenant_id": "org-2", "name": "release-crew", "members": []}

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.get_team.return_value = document
        response = await test_app_client.get("/v1/teams/pull/team-9?tenant_id=org-1")

    assert response.status_code == 409
    assert "belongs to organization org-2" in response.json()["detail"]
    assert TeamRegistry(temp_dir).list_teams() == []


@pytest.mark.asyncio
async def test_team_pull_maps_the_hubs_team_not_found(
    test_app_client, temp_dir, fake_org_credential
) -> None:
    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.get_team.side_effect = APIError(
            "Team not found.", status_code=404, code="team_not_found"
        )
        response = await test_app_client.get("/v1/teams/pull/missing")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "team_not_found"


@pytest.mark.asyncio
async def test_team_pull_reports_the_name_it_actually_stored(
    test_app_client, temp_dir, fake_org_credential
) -> None:
    """A colliding published name takes the rename-with-suffix convention."""
    _new_team(temp_dir)  # local "release-crew" is taken
    document = {"id": "team-9", "tenant_id": "org-1", "name": "Release Crew", "members": []}

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.get_team.return_value = document
        response = await test_app_client.get("/v1/teams/pull/team-9")

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["name"] == "Release-Crew-2"
    assert result["renamed_from"] == "Release Crew"
    assert result["invalid_name"] is True
