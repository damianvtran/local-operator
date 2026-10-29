"""Radient client: the publish-preview protocol and the public team reads (p2p3 §4/§7.4).

What these pin: the four preview calls' SHAPE (envelope with ``preview_version``,
the target query params, the resolved preview budget, no redirects), the budget's
resolution (default, override, floor, typo), the timeout error naming the setting and
the budget it exceeded, the commit kwargs that ride on the three publish methods
(``preview_token`` header, ``accept_unresolved`` ids, ``moderation_allowance`` param),
that with all three at their defaults the request is byte-identical to what the
methods sent before the parameters existed, and the two public reads the CLI's new
arms use.

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
    PREVIEW_TIMEOUT_DEFAULT_SECONDS,
    PREVIEW_TIMEOUT_ENV,
    PREVIEW_TIMEOUT_FLOOR_SECONDS,
    PREVIEW_TOKEN_HEADER,
    RadientClient,
    build_instruction_set_document,
    preview_timeout_error,
    resolve_preview_timeout_seconds,
)


@pytest.fixture(autouse=True)
def _isolate_preview_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own export must never decide what these tests measure."""
    monkeypatch.delenv(PREVIEW_TIMEOUT_ENV, raising=False)


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
    assert kwargs["timeout"] == PREVIEW_TIMEOUT_DEFAULT_SECONDS
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


def test_preview_timeout_defaults_to_a_multi_minute_budget() -> None:
    """The default must outlast the transform, not just the 25 s attempt it priced."""
    assert resolve_preview_timeout_seconds() == PREVIEW_TIMEOUT_DEFAULT_SECONDS
    assert PREVIEW_TIMEOUT_DEFAULT_SECONDS > PREVIEW_TIMEOUT_FLOOR_SECONDS


def test_preview_timeout_env_raises_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PREVIEW_TIMEOUT_ENV, "900")
    assert resolve_preview_timeout_seconds() == 900.0


def test_preview_timeout_env_below_the_floor_is_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PREVIEW_TIMEOUT_ENV, "5")
    assert resolve_preview_timeout_seconds() == PREVIEW_TIMEOUT_FLOOR_SECONDS


def test_preview_timeout_env_typo_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PREVIEW_TIMEOUT_ENV, "5m")
    assert resolve_preview_timeout_seconds() == PREVIEW_TIMEOUT_DEFAULT_SECONDS


def test_preview_timeout_env_zero_and_negative_fall_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deliberately unlike ``resolve_mcp_timeout_s``: 0 does NOT mean "no timeout".

    The MCP resolver reads 0 as "off" because an MCP server is a local transport the
    operator can see. A preview is a PAID call to a remote hub: unbounded, it would hold
    the caller's slot for as long as the transform runs, which is not a setting anyone
    chooses on purpose -- it is a typo that would never fail. The divergence is pinned so
    the next reader does not "fix" it back into the MCP shape.
    """
    for raw in ("0", "-5"):
        monkeypatch.setenv(PREVIEW_TIMEOUT_ENV, raw)
        assert resolve_preview_timeout_seconds() == PREVIEW_TIMEOUT_DEFAULT_SECONDS


def test_preview_request_carries_the_overridden_budget(
    radient_client: RadientClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PREVIEW_TIMEOUT_ENV, "1200")
    with patch("requests.request", return_value=_envelope({})) as mock_request:
        radient_client.preview_publish_team_document(dict(_TEAM_DOCUMENT), tenant_id="org-a")

    _args, kwargs = mock_request.call_args
    assert kwargs["timeout"] == 1200.0


def test_preview_timeout_error_names_the_setting_and_the_budget(
    radient_client: RadientClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure a timeout produces must be actionable, not the generic fallback."""
    monkeypatch.setenv(PREVIEW_TIMEOUT_ENV, "400")
    with patch("requests.request", side_effect=requests.exceptions.Timeout("timed out")):
        with pytest.raises(APIError) as exc_info:
            radient_client.preview_publish_team_document(dict(_TEAM_DOCUMENT), tenant_id="org-a")

    message = str(exc_info.value)
    assert PREVIEW_TIMEOUT_ENV in message
    assert "400 s" in message  # the budget actually used, not the fallback message
    assert "800" in message  # a concrete next value to raise it to
    assert "Could not preview" not in message
    assert exc_info.value.code == "preview_timeout"
    assert exc_info.value.status_code is None


def test_preview_connect_timeout_keeps_the_generic_message(
    radient_client: RadientClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ConnectTimeout`` subclasses ``Timeout`` but never reached the hub.

    The budget cannot be what ran out when the connection was never made, so telling the
    caller to raise it would send them to a knob that cannot help them.
    """
    monkeypatch.setenv(PREVIEW_TIMEOUT_ENV, "400")
    with patch("requests.request", side_effect=requests.exceptions.ConnectTimeout("offline")):
        with pytest.raises(APIError) as exc_info:
            radient_client.preview_publish_team_document(dict(_TEAM_DOCUMENT), tenant_id="org-a")

    assert PREVIEW_TIMEOUT_ENV not in str(exc_info.value)
    assert "Could not preview the team publication" in str(exc_info.value)
    assert exc_info.value.status_code is None


def test_preview_timeout_error_reports_the_floor_that_was_used() -> None:
    """A clamped budget is reported as the floor, never as the value ignored."""
    error = preview_timeout_error(PREVIEW_TIMEOUT_FLOOR_SECONDS)
    assert f"{PREVIEW_TIMEOUT_FLOOR_SECONDS:g} s" in str(error)
    assert PREVIEW_TIMEOUT_ENV in str(error)


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

    args, kwargs = mock_post.call_args
    # The full request shape, frozen: a stray default header or an accidentally
    # added kwarg would pass the absence checks but fail this (review round 1,
    # N1). "Byte-identical" means this dict and nothing else.
    assert args[0] == f"{base_url}/agents/publish"
    assert kwargs == {
        "headers": {
            "X-Title": "Local Operator",
            "HTTP-Referer": "https://local-operator.com",
            "Authorization": "Bearer test_api_key",
            "Content-Type": "application/json",
        },
        "json": document,
        "params": {},
        "allow_redirects": False,
    }


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
    # The listing refuses redirects like every other hub call (S-1).
    assert kwargs["allow_redirects"] is False


def test_list_public_teams_refuses_a_redirect(anonymous_client: RadientClient) -> None:
    """A 3xx listing is refused instead of followed (security round 1, S-1).

    Following it would render (or import) another origin's rows as "the public
    hub"; no credential can ride this call, so the refusal is provenance, not
    secrecy -- and it matches every other hub call this slice adds.
    """
    response = MagicMock()
    response.status_code = 302
    with patch("requests.get", return_value=response) as mock_get:
        with pytest.raises(APIError) as excinfo:
            anonymous_client.list_public_teams()

    assert getattr(excinfo.value, "code", None) == "unexpected_redirect"
    assert mock_get.call_args.kwargs["allow_redirects"] is False


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


_SKILL_DOCUMENT: Dict[str, Any] = {
    "document_version": 1,
    "name": "weather-buddy",
    "description": "Small skill.",
    "files": [
        {
            "path": "SKILL.md",
            "content": "---\nname: weather-buddy\ndescription: Small skill.\n---\n\nBody.\n",
        }
    ],
}


def test_publish_skill_document_posts_the_bare_manifest(
    radient_client: RadientClient, base_url: str
) -> None:
    result = {"skill": {"id": "sk-1", "name": "weather-buddy"}, "revision": '"r1"'}
    with patch("requests.post", return_value=_envelope(result)) as mock_post:
        published = radient_client.publish_skill_document(dict(_SKILL_DOCUMENT))

    assert published == result
    args, kwargs = mock_post.call_args
    assert args[0] == f"{base_url}/skills/publish"
    # The commit decoders read the BARE manifest; a preview envelope is a 422.
    assert kwargs["json"] == _SKILL_DOCUMENT
    assert kwargs["params"] == {}
    assert kwargs["allow_redirects"] is False
    assert PREVIEW_TOKEN_HEADER not in kwargs["headers"]
    assert PREVIEW_ACCEPT_HEADER not in kwargs["headers"]


def test_publish_skill_document_puts_the_org_target_on_query_params(
    radient_client: RadientClient, base_url: str
) -> None:
    with patch("requests.post", return_value=_envelope({})) as mock_post:
        radient_client.publish_skill_document(
            dict(_SKILL_DOCUMENT), visibility="org", tenant_id="org-a"
        )

    _args, kwargs = mock_post.call_args
    assert kwargs["params"] == {"visibility": "org", "tenant_id": "org-a"}

    with patch("requests.post", return_value=_envelope({})) as mock_post:
        radient_client.publish_skill_document(
            dict(_SKILL_DOCUMENT), visibility="public", tenant_id=None
        )
    _args, kwargs = mock_post.call_args
    assert kwargs["params"] == {}


def test_publish_skill_document_carries_the_commit_protocol(
    radient_client: RadientClient, base_url: str
) -> None:
    with patch("requests.post", return_value=_envelope({})) as mock_post:
        radient_client.publish_skill_document(
            dict(_SKILL_DOCUMENT),
            preview_token="pin-1",
            accept_unresolved=["u-1", "u-2"],
            moderation_allowance="allow-1",
        )

    _args, kwargs = mock_post.call_args
    assert kwargs["headers"][PREVIEW_TOKEN_HEADER] == "pin-1"
    assert kwargs["headers"][PREVIEW_ACCEPT_HEADER] == "u-1,u-2"
    assert kwargs["params"] == {"moderation_allowance": "allow-1"}


def test_publish_skill_document_keeps_the_hub_refusal_code(radient_client: RadientClient) -> None:
    with patch(
        "requests.post",
        return_value=MagicMock(
            status_code=422,
            raise_for_status=MagicMock(side_effect=_refusal(422, {"code": "name_taken"})),
        ),
    ):
        with pytest.raises(APIError) as caught:
            radient_client.publish_skill_document(dict(_SKILL_DOCUMENT))

    assert caught.value.code == "name_taken"
