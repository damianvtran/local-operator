"""Radient client: the publish-preview protocol and the public team reads (p2p3 §4/§7.4).

What these pin: the four preview calls' SHAPE (envelope with ``preview_version``,
the target query params, the 60 s timeout, no redirects), the commit kwargs that
ride on the three publish methods (``preview_token`` header,
``accept_unresolved`` ids, ``moderation_allowance`` param), that with all three
at their defaults the request is byte-identical to what the methods sent before
the parameters existed, and the two public reads the CLI's new arms use.

Refusals keep their machine-readable half (``code``/``details``), because the
CLI renders a different next step for ``preview_expired`` than for
``moderation_unavailable``.
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import SecretStr

from local_operator.clients._http import APIError
from local_operator.clients.radient import (
    PREVIEW_ACCEPT_HEADER,
    PREVIEW_ENVELOPE_VERSION,
    PREVIEW_TIMEOUT_SECONDS,
    PREVIEW_TOKEN_HEADER,
    RadientClient,
    build_instruction_set_document,
)


@pytest.fixture
def base_url() -> str:
    return "https://api.test.radient.com"


@pytest.fixture
def radient_client(base_url: str) -> RadientClient:
    return RadientClient(api_key=SecretStr("test_api_key"), base_url=base_url)


@pytest.fixture
def anonymous_client(base_url: str) -> RadientClient:
    return RadientClient(api_key=None, base_url=base_url)


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


_TEAM_DOCUMENT: Dict[str, Any] = {
    "name": "Release Crew",
    "description": "Ships it.",
    "manager": "manager",
    "members": [{"role": "coder", "kind": "agent", "count": 2}],
    "instructions": "You ship.",
    "project": "",
    "version": "1.0.0",
}


def test_preview_publish_agent_wraps_the_document_in_the_envelope(
    radient_client: RadientClient, base_url: str
) -> None:
    preview = {"status": "unchanged", "document": {"a": 1}}
    with patch("requests.request", return_value=_envelope(preview)) as mock_request:
        result = radient_client.preview_publish_agent_instruction_set({"a": 1})

    assert result == preview
    args, kwargs = mock_request.call_args
    assert args[0] == "POST"
    assert args[1] == f"{base_url}/agents/publish/preview"
    assert kwargs["json"] == {"preview_version": PREVIEW_ENVELOPE_VERSION, "document": {"a": 1}}
    assert kwargs["params"] == {}
    assert kwargs["timeout"] == PREVIEW_TIMEOUT_SECONDS
    assert kwargs["allow_redirects"] is False


def test_preview_org_target_rides_on_query_params(
    radient_client: RadientClient, base_url: str
) -> None:
    with patch("requests.request", return_value=_envelope({})) as mock_request:
        radient_client.preview_publish_agent_instruction_set(
            {"a": 1}, visibility="org", tenant_id="org-a"
        )

    _args, kwargs = mock_request.call_args
    assert kwargs["params"] == {"visibility": "org", "tenant_id": "org-a"}


def test_preview_republish_agent_uses_put(radient_client: RadientClient, base_url: str) -> None:
    with patch("requests.request", return_value=_envelope({})) as mock_request:
        radient_client.preview_republish_agent_instruction_set("hub-1", {"a": 1})

    args, _kwargs = mock_request.call_args
    assert args[0] == "PUT"
    assert args[1] == f"{base_url}/agents/hub-1/publish/preview"


def test_preview_team_calls_use_the_team_target_rules(
    radient_client: RadientClient, base_url: str
) -> None:
    with patch("requests.request", return_value=_envelope({})) as mock_request:
        radient_client.preview_publish_team_document(dict(_TEAM_DOCUMENT), tenant_id="org-a")
    args, kwargs = mock_request.call_args
    assert (args[0], args[1]) == ("POST", f"{base_url}/teams/publish/preview")
    assert kwargs["params"] == {"tenant_id": "org-a"}  # the legacy lone-tenant org spelling

    with patch("requests.request", return_value=_envelope({})) as mock_request:
        radient_client.preview_republish_team_document(
            "hub-team-1", dict(_TEAM_DOCUMENT), visibility="public"
        )
    args, kwargs = mock_request.call_args
    assert (args[0], args[1]) == ("PUT", f"{base_url}/teams/hub-team-1/publish/preview")
    assert kwargs["params"] == {}


def test_preview_refusal_keeps_its_code_and_details(radient_client: RadientClient) -> None:
    refusal = _refusal(
        503,
        {
            "error": "Publication review is temporarily unavailable. Try again shortly.",
            "code": "moderation_unavailable",
            "details": {"stage": "generalization", "attempts": 0},
        },
    )
    with patch("requests.request", side_effect=refusal):
        with pytest.raises(APIError) as exc_info:
            radient_client.preview_publish_agent_instruction_set({"a": 1})

    assert exc_info.value.code == "moderation_unavailable"
    assert exc_info.value.details["stage"] == "generalization"
    assert exc_info.value.status_code == 503


def test_publish_commit_kwargs_ride_on_headers_and_params(
    radient_client: RadientClient, base_url: str
) -> None:
    with patch(
        "requests.post", return_value=_envelope({"agent_id": "hub-1"}, status=201)
    ) as mock_post:
        radient_client.publish_agent_instruction_set(
            {"name": "Coder"},
            visibility="org",
            tenant_id="org-a",
            preview_token="pin-1",
            accept_unresolved=["u1", "u2"],
            moderation_allowance="org_internal_ops_v1",
        )

    _args, kwargs = mock_post.call_args
    assert kwargs["headers"][PREVIEW_TOKEN_HEADER] == "pin-1"
    assert kwargs["headers"][PREVIEW_ACCEPT_HEADER] == "u1,u2"
    assert kwargs["params"] == {
        "visibility": "org",
        "tenant_id": "org-a",
        "moderation_allowance": "org_internal_ops_v1",
    }


def test_publish_without_the_new_kwargs_is_byte_identical(
    radient_client: RadientClient, base_url: str
) -> None:
    """All three defaults None: no preview header, no accept header, no allowance param."""
    document = build_instruction_set_document(
        name="Coder",
        description="Writes code.",
        instructions="You write code.",
        kind="role",
        version="1.0.0",
    )
    with patch(
        "requests.post", return_value=_envelope({"agent_id": "hub-1"}, status=201)
    ) as mock_post:
        radient_client.publish_agent_instruction_set(document)

    _args, kwargs = mock_post.call_args
    assert PREVIEW_TOKEN_HEADER not in kwargs["headers"]
    assert PREVIEW_ACCEPT_HEADER not in kwargs["headers"]
    assert kwargs["params"] == {}
    assert "moderation_allowance" not in kwargs["params"]
    assert kwargs["json"] == document


def test_republish_commit_kwargs_ride_too(radient_client: RadientClient, base_url: str) -> None:
    with patch("requests.put", return_value=_envelope({"agent_id": "hub-1"})) as mock_put:
        radient_client.republish_agent_instruction_set(
            "hub-1", {"name": "Coder"}, preview_token="pin-2", accept_unresolved=["u9"]
        )

    _args, kwargs = mock_put.call_args
    assert kwargs["headers"][PREVIEW_TOKEN_HEADER] == "pin-2"
    assert kwargs["headers"][PREVIEW_ACCEPT_HEADER] == "u9"
    assert "timeout" not in kwargs  # commits keep their no-timeout stance


def test_republish_team_document_puts_with_the_token(
    radient_client: RadientClient, base_url: str
) -> None:
    result = {"team": {"id": "hub-team-1", "name": "Release Crew", "version": "1.0.0"}}
    with patch("requests.put", return_value=_envelope(result)) as mock_put:
        republished = radient_client.republish_team_document(
            "hub-team-1",
            dict(_TEAM_DOCUMENT),
            visibility="org",
            tenant_id="org-a",
            preview_token="pin-3",
        )

    assert republished == result
    args, kwargs = mock_put.call_args
    assert args[0] == f"{base_url}/teams/hub-team-1/publish"
    assert kwargs["params"] == {"visibility": "org", "tenant_id": "org-a"}
    assert kwargs["headers"][PREVIEW_TOKEN_HEADER] == "pin-3"
    assert kwargs["allow_redirects"] is False


def test_republish_team_document_refuses_an_empty_tenant(radient_client: RadientClient) -> None:
    with pytest.raises(ValueError):
        radient_client.republish_team_document("hub-team-1", {"name": "X"}, tenant_id="  ")


def test_list_public_teams_reads_the_listing_anonymously(
    anonymous_client: RadientClient, base_url: str
) -> None:
    envelope = {
        "page": 2,
        "per_page": 20,
        "total_pages": 3,
        "total_records": 43,
        "records": [{"id": "t1", "name": "crew"}],
    }
    with patch("requests.get", return_value=_envelope(envelope)) as mock_get:
        result = anonymous_client.list_public_teams(page=2, per_page=20)

    assert result == envelope
    args, kwargs = mock_get.call_args
    assert args[0] == f"{base_url}/teams"
    assert kwargs["params"] == {"page": 2, "per_page": 20}
    assert "Authorization" not in kwargs["headers"]


def test_get_team_anonymous_sends_no_authorization(
    anonymous_client: RadientClient, base_url: str
) -> None:
    """The public pull arm's switch: with_credential=False sends no bearer at all.

    The route is optional-auth; the CLI's no-`--org` pull is anonymous so it needs
    no login and cannot read an org row by accident. With the default (True) the
    same client REFUSES locally -- which is the org arm's contract.
    """
    document = {"id": "t1", "name": "crew", "instructions": "You ship."}
    with patch("requests.get", return_value=_envelope(document)) as mock_get:
        result = anonymous_client.get_team("t1", with_credential=False)

    assert result == document
    _args, kwargs = mock_get.call_args
    assert "Authorization" not in kwargs["headers"]

    with pytest.raises(RuntimeError):
        anonymous_client.get_team("t1")
