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
from local_operator.providers.radient_credentials import ORG_LOGIN_REMEDY
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
async def test_team_publish_refuses_an_invalid_document_locally(
    test_app_client, temp_dir, fake_org_credential
) -> None:
    """A document the hub would refuse is refused HERE: 422, no hub client.

    The oversize brief is written straight to the row -- the registry re-reads
    briefs from disk without re-bounding them, so this is exactly the document
    a pull-then-push hands the builder, and exactly why the preflight exists.
    """
    team = _new_team(temp_dir)
    (temp_dir / "teams" / team.id / "instructions.md").write_text("x" * 41_200, encoding="utf-8")

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        response = await test_app_client.post(f"/v1/teams/{team.id}/publish?tenant_id=org-1")

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "invalid_team_document"
    assert detail["details"] == {
        "field": "instructions",
        "rule": "must be at most 32768 characters (submitted 41200)",
    }
    mock_client.assert_not_called()


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


#: The token the conftest's org-credential fixture seeds. A reflecting upstream
#: would echo exactly this value in any field of its answer; the assertions
#: below are about the scrub, not about this spelling being a real credential.
_REFLECTED = "org-access-token"


@pytest.mark.parametrize("encoded", ["a%3Ftenant_id=org-9", "a%23frag"])
@pytest.mark.asyncio
async def test_team_pull_refuses_an_id_that_could_escape_its_segment(
    test_app_client, fake_org_credential, encoded: str
) -> None:
    """A decoded `?`/`#` must not become query structure on the hub request.

    Review round 1's MAJOR: ``get_team`` interpolates the id into the path, so
    before validation ``a%3Ftenant_id=org-9`` reached the hub as
    ``/teams/a?tenant_id=org-9`` -- a caller-chosen query on a request this
    machine makes as the signed-in PERSON. The id is validated with the
    registry's own reader before any bearer-carrying URL exists.
    """
    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        response = await test_app_client.get(f"/v1/teams/pull/{encoded}")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
    mock_client.assert_not_called()


@pytest.mark.parametrize("encoded", ["a%2Fb", "..%2F..%2Fme%2Fmemberships", ".."])
@pytest.mark.asyncio
async def test_team_pull_refuses_multi_segment_escapes_before_the_handler(
    test_app_client, fake_org_credential, encoded: str
) -> None:
    """Encoded slashes and dot-segments never become path structure either."""
    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        response = await test_app_client.get(f"/v1/teams/pull/{encoded}")

    assert response.status_code == 404
    mock_client.assert_not_called()


@pytest.mark.asyncio
async def test_team_pull_refuses_a_tenantless_document_when_a_tenant_is_declared(
    test_app_client, temp_dir, fake_org_credential
) -> None:
    """A declared tenant must be checkable: no owner named -> 409, not stored."""
    document = {"id": "team-9", "name": "release-crew", "members": []}

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.get_team.return_value = document
        response = await test_app_client.get("/v1/teams/pull/team-9?tenant_id=org-1")

    assert response.status_code == 409
    assert "does not name an owning organization" in response.json()["detail"]
    assert TeamRegistry(temp_dir).list_teams() == []


@pytest.mark.asyncio
async def test_team_pull_without_a_declared_tenant_imports_a_tenantless_document(
    test_app_client, temp_dir, fake_org_credential
) -> None:
    """The by-id path stays usable when the caller asserts no tenant."""
    document = {"id": "team-9", "name": "release-crew", "members": []}

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.get_team.return_value = document
        response = await test_app_client.get("/v1/teams/pull/team-9")

    assert response.status_code == 200, response.text
    assert TeamRegistry(temp_dir).get_team_by_name("release-crew") is not None


@pytest.mark.asyncio
async def test_memberships_masks_a_reflected_credential(
    test_app_client, fake_org_credential
) -> None:
    """The relay scrubs this call's credential out of ANY field (S-1)."""
    memberships = [{"tenant_id": "org-1", "note": f"token={_REFLECTED}"}]

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.list_memberships.return_value = memberships
        response = await test_app_client.get("/v1/memberships")

    assert response.status_code == 200, response.text
    assert _REFLECTED not in response.text
    assert "[redacted]" in response.text


@pytest.mark.asyncio
async def test_team_publish_masks_a_reflected_credential(
    test_app_client, temp_dir, fake_org_credential
) -> None:
    team = _new_team(temp_dir)
    result = {"team": {"id": "team-9", "name": team.name, "echo": f"Bearer {_REFLECTED}"}}

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.publish_team_document.return_value = result
        response = await test_app_client.post(f"/v1/teams/{team.id}/publish?tenant_id=org-1")

    assert response.status_code == 200, response.text
    assert _REFLECTED not in response.text
    assert "[redacted]" in response.text


@pytest.mark.asyncio
async def test_team_pull_masks_reflected_credentials_before_storing(
    test_app_client, temp_dir, fake_org_credential
) -> None:
    """The scrub happens BEFORE import: a credential must not be STORED (S-1)."""
    document = {
        "id": "team-9",
        "tenant_id": "org-1",
        "name": "release-crew",
        "members": [],
        "instructions": f"You ship. Authorization: Bearer {_REFLECTED}",
    }

    with patch("local_operator.server.routes.agents.RadientClient") as mock_client:
        mock_client.return_value.get_team.return_value = document
        response = await test_app_client.get("/v1/teams/pull/team-9")

    assert response.status_code == 200, response.text
    assert _REFLECTED not in response.text
    assert "[redacted]" in response.json()["result"]["instructions"]
    stored = TeamRegistry(temp_dir).get_team_by_name("release-crew")
    assert stored is not None
    assert _REFLECTED not in stored.instructions
    assert "[redacted]" in stored.instructions
