"""Unit pins for the Radient proxy's closed-operation contract (design §4.7).

The e2e file drives the transport over real loopback HTTP; this one pins the
pure pieces at unit speed so the PR-H contract -- operation NAMES, the pinned
``visibility=org`` query, the endpoint mapping, and the org refusal-code reader
-- is regression-proof without standing a server up. The four org operations'
request/response shapes (including the refusal codes carried through) are the
interface local-operator-ui's ``RadientOperation`` union must match.
"""

import pytest

from local_operator.server.routes.desktop_radient import (
    _ORG_REFUSAL_MESSAGES,
    ORG_OPERATIONS,
    RESEND_NOTHING_TO_RESEND,
    RESEND_RATE_LIMITED,
    RadientRequest,
    _operation_params,
    _org_refusal_code,
    _upstream_refusal_for,
    endpoint,
)


def test_the_org_operations_are_exactly_the_documented_four() -> None:
    """The dispatch set and the Literal's org entries cannot silently diverge."""
    assert ORG_OPERATIONS == frozenset(
        {"memberships.list", "org_agents.list", "org_teams.list", "org_team.get"}
    )
    assert set(_ORG_REFUSAL_MESSAGES) == {
        "not_a_member",
        "insufficient_role",
        "team_plan_required",
    }


def test_operation_params_pins_visibility_org_for_the_workspace_list() -> None:
    body = RadientRequest.model_validate(
        {
            "operation": "org_agents.list",
            "tenant_id": "org-1",
            "query": {"page": 1, "name": "x"},
        }
    )
    assert _operation_params(body) == {"page": 1, "name": "x", "visibility": "org"}


def test_operation_params_leaves_every_other_operation_verbatim() -> None:
    body = RadientRequest.model_validate({"operation": "org_teams.list", "tenant_id": "org-1"})
    assert _operation_params(body) == {}


def test_the_org_query_vocabulary_refuses_a_caller_supplied_visibility() -> None:
    """`visibility` is pinned server-side and must not be a caller input."""
    with pytest.raises(ValueError, match="Unsupported query fields"):
        RadientRequest.model_validate(
            {
                "operation": "org_agents.list",
                "tenant_id": "org-1",
                "query": {"visibility": "public"},
            }
        )


def test_org_operations_need_their_own_identifier() -> None:
    """A missing tenant is refused at validation, not at request time."""
    with pytest.raises(ValueError, match="organization"):
        RadientRequest.model_validate({"operation": "org_agents.list"})
    with pytest.raises(ValueError, match="team"):
        RadientRequest.model_validate({"operation": "org_team.get"})


def test_endpoint_maps_the_org_family() -> None:
    """The four names H must match, each to its one pinned path."""
    assert endpoint(RadientRequest(operation="memberships.list")) == ("GET", "/me/memberships")
    assert endpoint(RadientRequest(operation="org_agents.list", tenant_id="org-1")) == (
        "GET",
        "/tenants/org-1/agents",
    )
    assert endpoint(RadientRequest(operation="org_teams.list", tenant_id="org-1")) == (
        "GET",
        "/tenants/org-1/teams",
    )
    assert endpoint(RadientRequest(operation="org_team.get", team_id="team-1")) == (
        "GET",
        "/teams/team-1",
    )


@pytest.mark.parametrize(
    ("operation", "envelope", "expected"),
    [
        ("org_agents.list", {"code": "team_plan_required"}, "team_plan_required"),
        ("org_agents.list", {"code": "insufficient_role", "details": {}}, "insufficient_role"),
        ("org_agents.list", {"code": "not_a_member"}, "not_a_member"),
        ("org_agents.list", {"code": "something_else"}, None),
        ("org_agents.list", ["not", "a", "dict"], None),
        ("agents.list", {"code": "team_plan_required"}, None),
    ],
)
def test_org_refusal_code_reads_only_frozen_codes_for_org_operations(
    operation: str, envelope: object, expected: str | None
) -> None:
    assert _org_refusal_code(operation, envelope) == expected


# --- signup.resend ------------------------------------------------------------


def _code(failure) -> str:
    """The ``code`` of a proxy refusal (``HTTPException.detail`` is typed ``str``)."""
    detail: dict[str, object] = failure.detail  # type: ignore[assignment]
    return str(detail["code"])


def test_signup_resend_maps_to_the_upstream_post_and_needs_a_request_id() -> None:
    body = RadientRequest(
        operation="signup.resend", request_id="12345678-1234-1234-1234-123456789abc"
    )
    assert endpoint(body) == ("POST", "/auth/signup/resend")
    # A mutation that sends mail: no request id, no request.
    with pytest.raises(ValueError):
        RadientRequest(operation="signup.resend")


def test_signup_resend_is_not_an_org_operation() -> None:
    """Its refusals have their own mapping; the org reader must not claim them."""
    assert "signup.resend" not in ORG_OPERATIONS


@pytest.mark.parametrize(
    "status,code", [(429, RESEND_RATE_LIMITED), (409, RESEND_NOTHING_TO_RESEND)]
)
@pytest.mark.asyncio
async def test_signup_resend_refusal_codes_are_not_credential_refusals(
    status: int, code: str
) -> None:
    class Response:
        status_code = status

    failure = await _upstream_refusal_for("signup.resend", Response(), "token")
    assert failure.status_code == status
    assert _code(failure) == code
    assert _code(failure) != "radient_credential_refused"


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.asyncio
async def test_signup_resend_credential_refusals_keep_their_meaning(status: int) -> None:
    class Response:
        status_code = status

    failure = await _upstream_refusal_for("signup.resend", Response(), "token")
    assert _code(failure) == "radient_credential_refused"


@pytest.mark.asyncio
async def test_other_operations_keep_the_generic_429_reading() -> None:
    class Response:
        status_code = 429

    failure = await _upstream_refusal_for("agents.list", Response(), "token")
    assert _code(failure) == "radient_credential_refused"
