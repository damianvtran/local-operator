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
    _operation_payload,
    _org_refusal_code,
    _upstream_refusal_for,
    endpoint,
)


def test_the_org_operations_are_exactly_the_documented_set() -> None:
    """The dispatch set and the Literal's org entries cannot silently diverge."""
    assert ORG_OPERATIONS == frozenset(
        {
            "memberships.list",
            "org_agents.list",
            "org_teams.list",
            "org_team.get",
            # The skills lane's two reads: an org catalogue refusal must carry
            # the same membership vocabulary as its siblings.
            "org_skills.list",
            "org_skill.get",
            # The Team trial ops (design §9 Lane E).
            "team_trial.eligibility",
            "team_trial.start",
            "team_trial.status",
        }
    )
    assert set(_ORG_REFUSAL_MESSAGES) >= {
        "not_a_member",
        "insufficient_role",
        "team_plan_required",
        "trial_not_eligible",
        "terms_outdated",
        "team_trial_ended",
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
    with pytest.raises(ValueError, match="organization"):
        RadientRequest.model_validate({"operation": "org_skills.list"})
    with pytest.raises(ValueError, match="organization"):
        RadientRequest.model_validate({"operation": "org_skill.get", "skill_id": "sk-1"})
    with pytest.raises(ValueError, match="skill"):
        RadientRequest.model_validate({"operation": "org_skill.get", "tenant_id": "org-1"})
    # The skill id becomes an upstream path segment: characters outside the
    # shared identifier grammar are refused before a request could be built.
    with pytest.raises(ValueError):
        RadientRequest.model_validate(
            {"operation": "org_skill.get", "tenant_id": "org-1", "skill_id": ".."}
        )
    with pytest.raises(ValueError, match="Unsupported query fields"):
        RadientRequest.model_validate(
            {"operation": "org_skills.list", "tenant_id": "org-1", "query": {"page": 1}}
        )


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
    assert endpoint(RadientRequest(operation="org_skills.list", tenant_id="org-1")) == (
        "GET",
        "/tenants/org-1/skills",
    )
    assert endpoint(
        RadientRequest(operation="org_skill.get", tenant_id="org-1", skill_id="sk-1")
    ) == ("GET", "/tenants/org-1/skills/sk-1")


@pytest.mark.parametrize(
    ("operation", "envelope", "expected"),
    [
        ("org_agents.list", {"code": "team_plan_required"}, "team_plan_required"),
        ("org_agents.list", {"code": "insufficient_role", "details": {}}, "insufficient_role"),
        ("org_agents.list", {"code": "not_a_member"}, "not_a_member"),
        ("org_agents.list", {"code": "something_else"}, None),
        ("org_agents.list", ["not", "a", "dict"], None),
        ("agents.list", {"code": "team_plan_required"}, None),
        # The skill reads joined the org family: their refusals take the same
        # code-preserving reading, and a non-org op still does not.
        ("org_skills.list", {"code": "not_a_member"}, "not_a_member"),
        ("org_skill.get", {"code": "insufficient_role"}, "insufficient_role"),
        ("org_skill.get", {"code": "skill_not_found"}, None),
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


# --- team_trial.* (teams-trial design §3, Lane E) ------------------------------

RID = "12345678-1234-1234-1234-123456789abc"
TERMS = {"terms_version": "2026-07-29", "terms_accepted": True}


def test_team_trial_ops_map_to_the_designed_routes() -> None:
    assert endpoint(RadientRequest(operation="team_trial.eligibility")) == (
        "GET",
        "/team-trial/eligibility",
    )
    assert endpoint(RadientRequest(operation="team_trial.status", tenant_id="ten-1")) == (
        "GET",
        "/tenants/ten-1/team-trial",
    )
    start = RadientRequest(
        operation="team_trial.start", tenant_id="ten-1", request_id=RID, payload=TERMS
    )
    assert endpoint(start) == ("POST", "/tenants/ten-1/team-trial")


def test_team_trial_start_pins_source_lop_and_refuses_a_caller_source() -> None:
    start = RadientRequest(
        operation="team_trial.start",
        tenant_id="ten-1",
        request_id=RID,
        payload={**TERMS, "org_name": "Acme"},
    )
    assert _operation_payload(start) == {**TERMS, "org_name": "Acme", "source": "lop"}
    with pytest.raises(ValueError, match="Unsupported payload fields"):
        RadientRequest(
            operation="team_trial.start",
            tenant_id="ten-1",
            request_id=RID,
            payload={**TERMS, "source": "site"},
        )
    # Every other op's payload travels verbatim.
    other = RadientRequest(
        operation="comments.create", agent_id="ag-1", request_id=RID, payload={"text": "x"}
    )
    assert _operation_payload(other) == {"text": "x"}


@pytest.mark.parametrize(
    "payload",
    [
        {"terms_version": "2026-07-29"},  # no acceptance
        {"terms_version": "2026-07-29", "terms_accepted": False},
        # A truthy non-boolean is NOT consent.
        {"terms_version": "2026-07-29", "terms_accepted": "true"},
        {"terms_version": "2026-07-29", "terms_accepted": 1},
        {"terms_accepted": True},  # no version
        {"terms_version": "  ", "terms_accepted": True},
        {**TERMS, "org_name": "   "},
        {**TERMS, "org_name": "x" * 81},
        {**TERMS, "org_name": 7},
    ],
)
def test_team_trial_start_refuses_a_start_the_hub_could_only_refuse(payload) -> None:
    with pytest.raises(ValueError):
        RadientRequest(
            operation="team_trial.start", tenant_id="ten-1", request_id=RID, payload=payload
        )


def test_team_trial_start_needs_a_request_id_and_a_tenant() -> None:
    with pytest.raises(ValueError, match="request identifier"):
        RadientRequest(operation="team_trial.start", tenant_id="ten-1", payload=TERMS)
    with pytest.raises(ValueError, match="organization"):
        RadientRequest(operation="team_trial.start", request_id=RID, payload=TERMS)
    with pytest.raises(ValueError, match="organization"):
        RadientRequest(operation="team_trial.status")
    # An 80-character name (after trim) is the hub's own upper bound.
    RadientRequest(
        operation="team_trial.start",
        tenant_id="ten-1",
        request_id=RID,
        payload={**TERMS, "org_name": " " + "x" * 80 + " "},
    )


@pytest.mark.parametrize(
    ("operation", "envelope", "expected"),
    [
        ("team_trial.start", {"code": "trial_not_eligible"}, "trial_not_eligible"),
        ("team_trial.start", {"code": "terms_outdated"}, "terms_outdated"),
        ("team_trial.start", {"code": "terms_required"}, "terms_required"),
        ("team_trial.start", {"code": "rate_limited"}, "rate_limited"),
        ("team_trial.start", {"code": "team_trial_unavailable"}, "team_trial_unavailable"),
        ("team_trial.status", {"code": "not_a_member"}, "not_a_member"),
        ("team_trial.status", {"code": "team_trial_ended"}, "team_trial_ended"),
        # An org op reads the paused code too (the hub raises it at org seams)...
        ("org_agents.list", {"code": "team_trial_ended"}, "team_trial_ended"),
        # ...but the trial-only codes stay scoped: an existing org op's 429 keeps
        # its old reading byte-for-byte.
        ("org_agents.list", {"code": "rate_limited"}, None),
        ("team_trial.start", {"code": "something_else"}, None),
    ],
)
def test_team_trial_refusal_codes_are_carried(
    operation: str, envelope: object, expected: str | None
) -> None:
    assert _org_refusal_code(operation, envelope) == expected


def test_every_carried_code_has_a_desktop_sentence() -> None:
    from local_operator.server.routes.desktop_radient import (
        _ORG_REFUSAL_CODES,
        _TEAM_TRIAL_REFUSAL_CODES,
    )

    assert set(_ORG_REFUSAL_MESSAGES) == _ORG_REFUSAL_CODES | _TEAM_TRIAL_REFUSAL_CODES
