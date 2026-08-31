"""Contract tests for pure dependency, preflight, and budget receipts."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from local_operator.evaluation.protocol import ArtifactRef
from local_operator.evaluation.receipts import (
    BUDGET_RESOURCES,
    AvailableUsage,
    BudgetAuthorization,
    BudgetReconciliation,
    BudgetReservation,
    CappedAllowance,
    ClockRequirement,
    ComputeRequirement,
    DependencyPlan,
    DisplayRequirement,
    ExternalServiceRequirement,
    ModelCapabilityRequirement,
    NetworkRequirement,
    PinnedInputRequirement,
    PreflightReceipt,
    RedactionSet,
    Requirement,
    ResourceAmount,
    SealedPreflight,
    UnavailableUsage,
    UncappedAllowance,
    Usage,
    reconcile_budget,
    record_preflight,
    reserve_budget,
    seal_preflight,
)

DIGEST = "0123456789abcdef" * 4


def _requirements() -> tuple[Requirement, ...]:
    common: dict[str, Any] = {"necessity": "required", "reportability": "required"}
    return (
        ComputeRequirement(
            requirement_id="compute",
            cpu_class="high-throughput",
            memory_class="large",
            disk_bytes=10_000,
            display=DisplayRequirement(
                native_width=1920,
                native_height=1080,
                model_width=1280,
                model_height=720,
                platform_capability="desktop.linux",
            ),
            **common,
        ),
        NetworkRequirement(
            requirement_id="network",
            endpoint_id="model-api",
            service_id="provider",
            protocol="https",
            ports=(443,),
            proxy_capability="allowed",
            geography="us",
            **common,
        ),
        ExternalServiceRequirement(
            requirement_id="service",
            service_id="workspace",
            capability="read",
            account_ref="WORKSPACE_ACCOUNT",
            **common,
        ),
        ModelCapabilityRequirement(
            requirement_id="model-agent",
            role="agent",
            modalities=("text", "image"),
            tools=("computer",),
            min_context_tokens=100_000,
            min_output_tokens=4_096,
            route_pin="provider.route-1",
            fallback_policy="forbid",
            **common,
        ),
        ClockRequirement(
            requirement_id="clock",
            timezone="UTC",
            fixed_clock="2026-08-30T12:00:00Z",
            **common,
        ),
        PinnedInputRequirement(
            requirement_id="input",
            release_id="benchmark-v1",
            artifact=ArtifactRef(sha256=DIGEST, media_type="application/json", byte_count=42),
            content_sha256=DIGEST,
            **common,
        ),
    )


def _plan(requirements: tuple[Requirement, ...] | None = None) -> DependencyPlan:
    return DependencyPlan.model_validate(
        {
            "release_id": "release-1",
            "task_id": "task-1",
            "attempt_id": "attempt-1",
            "requirements": requirements or _requirements(),
        }
    )


def _receipts(plan: DependencyPlan) -> tuple[PreflightReceipt, ...]:
    return tuple(
        record_preflight(
            plan,
            item.requirement_id,
            status="pass",
            evidence={"nested": ["safe", {"ok": True}]},
            duration_ms=1,
        )
        for item in plan.requirements
    )


def _authorization(*, cap: int = 100, uncapped: set[str] | None = None) -> BudgetAuthorization:
    uncapped = uncapped or set()
    allowances = []
    for resource in reversed(BUDGET_RESOURCES):
        if resource in uncapped:
            allowances.append(
                UncappedAllowance(
                    resource=resource,
                    reason="operator approved exploratory run",
                    authorized_by="operator",
                    authorized_at_ms=1,
                    reporting="required",
                )
            )
        else:
            allowances.append(CappedAllowance(resource=resource, value=cap, reporting="required"))
    return BudgetAuthorization(episode_id="episode-1", allowances=tuple(allowances))


def _usage(value: int = 50) -> tuple[AvailableUsage, ...]:
    return tuple(AvailableUsage(resource=resource, value=value) for resource in BUDGET_RESOURCES)


def test_every_requirement_kind_round_trips_canonically() -> None:
    for requirement in _requirements():
        parsed = type(requirement).from_canonical_json(requirement.to_canonical_json())
        assert parsed == requirement
        assert parsed.to_canonical_json() == requirement.to_canonical_json()


def test_dependency_identity_is_stable_under_declaration_shuffle() -> None:
    requirements = _requirements()
    expected = _plan(requirements)
    for offset in range(len(requirements)):
        shuffled = requirements[offset:] + requirements[:offset]
        candidate = _plan(shuffled)
        assert candidate.plan_id == expected.plan_id
        assert [item.requirement_id for item in candidate.requirements] == sorted(
            item.requirement_id for item in requirements
        )
    assert DependencyPlan.from_canonical_json(expected.to_canonical_json()) == expected


def test_dependency_plan_rejects_duplicate_ids_and_conflicting_targets() -> None:
    compute = _requirements()[0]
    with pytest.raises(ValidationError, match="duplicate requirement IDs"):
        _plan((compute, compute))
    conflicting = compute.model_copy(update={"requirement_id": "other", "cpu_class": "small"})
    with pytest.raises(ValidationError, match="conflicting requirements"):
        _plan((compute, conflicting))


def test_requirement_metadata_is_bounded_portable_and_frozen() -> None:
    requirement = _requirements()[0]
    assert "aws" not in requirement.to_canonical_json().decode().lower()
    with pytest.raises(ValidationError):
        requirement.model_copy(update={"metadata": {"bad": 1.5}})
    with pytest.raises(ValidationError):
        NetworkRequirement.model_validate({**_requirements()[1].model_dump(), "ports": [443, 443]})
    with pytest.raises(ValidationError):
        ExternalServiceRequirement.model_validate(
            {**_requirements()[2].model_dump(), "account_ref": "raw secret value"}
        )


def test_receipts_enforce_skip_and_safe_timing() -> None:
    required_plan = _plan((_requirements()[0],))
    with pytest.raises(ValidationError, match="only optional"):
        record_preflight(required_plan, "compute", status="skip", duration_ms=0)
    optional = _requirements()[0].model_copy(
        update={"requirement_id": "optional-compute", "necessity": "optional"}
    )
    optional_plan = _plan((optional,))
    receipt = record_preflight(
        optional_plan,
        "optional-compute",
        status="skip",
        started_at_ms=10,
        ended_at_ms=12,
        duration_ms=2,
    )
    assert PreflightReceipt.from_canonical_json(receipt.to_canonical_json()) == receipt
    with pytest.raises(ValidationError, match="disagrees"):
        receipt.model_copy(update={"duration_ms": 3})


def test_preflight_requires_exactly_one_receipt_and_blocks_required_failure() -> None:
    plan = _plan()
    receipts = _receipts(plan)
    redactions = RedactionSet.from_resolved_values(("definitely-not-present",))
    sealed = seal_preflight(plan, receipts, redactions)
    assert sealed.successful
    with pytest.raises(ValidationError, match="validated receipts"):
        SealedPreflight.from_canonical_json(sealed.to_canonical_json())
    with pytest.raises(ValueError, match="exactly one"):
        seal_preflight(plan, receipts[:-1], redactions)
    with pytest.raises(ValueError, match="duplicate"):
        seal_preflight(plan, receipts + (receipts[0],), redactions)
    failed = PreflightReceipt.model_validate(
        {**receipts[0].model_dump(exclude={"receipt_id"}), "status": "fail"}
    )
    with pytest.raises(ValueError, match="required dependency failure"):
        seal_preflight(plan, (failed,) + receipts[1:], redactions)


def test_optional_failure_or_skip_seals_but_is_not_successful() -> None:
    optional = _requirements()[0].model_copy(
        update={"requirement_id": "optional-compute", "necessity": "optional"}
    )
    plan = _plan((optional,))
    for status in ("fail", "skip"):
        receipt = record_preflight(
            plan,
            optional.requirement_id,
            status=status,
            duration_ms=0,
        )
        sealed = seal_preflight(plan, (receipt,), RedactionSet.from_resolved_values(()))
        assert sealed.successful
        assert (optional.requirement_id in sealed.failed_requirement_ids) is (status == "fail")


@pytest.mark.parametrize(
    "location",
    ["requirement_id", "metadata_value", "metadata_list", "metadata_map", "evidence_value"],
)
def test_secret_canaries_fail_at_every_nested_location_without_echo(
    location: str,
) -> None:
    secret = "CANARY-super-secret-42"
    requirement = _requirements()[0]
    if location == "requirement_id":
        requirement = requirement.model_copy(update={"requirement_id": f"id-{secret}"})
    elif location == "metadata_value":
        requirement = requirement.model_copy(update={"metadata": {"value": secret}})
    elif location == "metadata_list":
        requirement = requirement.model_copy(update={"metadata": {"values": ["safe", secret]}})
    elif location == "metadata_map":
        requirement = requirement.model_copy(update={"metadata": {"nested": {"value": secret}}})
    plan = _plan((requirement,))
    receipt = record_preflight(
        plan,
        requirement.requirement_id,
        status="pass",
        evidence={"value": secret} if location == "evidence_value" else {"value": "safe"},
        duration_ms=1,
    )
    with pytest.raises(ValueError) as caught:
        seal_preflight(plan, (receipt,), RedactionSet.from_resolved_values((secret,)))
    assert secret not in str(caught.value)


def test_secret_reference_names_serialize_but_resolved_values_never_seal() -> None:
    secret = "resolved-token-value"
    requirement = _requirements()[2]
    plan = _plan((requirement,))
    assert b'"account_ref":"WORKSPACE_ACCOUNT"' in plan.to_canonical_json()
    safe = record_preflight(
        plan,
        requirement.requirement_id,
        status="pass",
        evidence={"account_ref": "WORKSPACE_ACCOUNT"},
        duration_ms=1,
    )
    assert seal_preflight(plan, (safe,), RedactionSet.from_resolved_values((secret,))).successful
    unsafe = PreflightReceipt.model_validate(
        {
            **safe.model_dump(exclude={"receipt_id"}),
            "evidence": {"result": f"prefix-{secret}-suffix"},
        }
    )
    with pytest.raises(ValueError, match="secret canary survived"):
        seal_preflight(plan, (unsafe,), RedactionSet.from_resolved_values((secret,)))
    assert secret not in repr(RedactionSet.from_resolved_values((secret,)))


def test_capped_zero_and_explicit_uncapped_are_distinct_closed_union() -> None:
    authorization = _authorization(cap=0, uncapped={"wall_milliseconds"})
    capped = authorization.allowance_for("provider_usd_micros")
    uncapped = authorization.allowance_for("wall_milliseconds")
    assert capped.kind == "capped" and capped.value == 0
    assert uncapped.kind == "uncapped" and uncapped.authorized_by == "operator"
    payload = json.loads(authorization.to_canonical_json())
    assert all("value" in item or item["kind"] == "uncapped" for item in payload["allowances"])
    with pytest.raises(ValidationError):
        BudgetAuthorization.model_validate(
            {"episode_id": "e", "allowances": [{"kind": "capped", "resource": "x"}]}
        )


def test_reservation_prevents_overcommit_and_identities_round_trip() -> None:
    authorization = _authorization(cap=10)
    first = reserve_budget(
        authorization,
        "reservation-1",
        (ResourceAmount(resource="provider_usd_micros", value=6),),
    )
    second = reserve_budget(
        authorization,
        "reservation-2",
        (ResourceAmount(resource="provider_usd_micros", value=4),),
        (first,),
    )
    assert BudgetReservation.from_canonical_json(first.to_canonical_json()) == first
    assert first.reservation_id != second.reservation_id
    with pytest.raises(ValueError, match="exceeds"):
        reserve_budget(
            authorization,
            "reservation-over",
            (ResourceAmount(resource="provider_usd_micros", value=5),),
            (first,),
        )
    uncapped = _authorization(cap=0, uncapped={"provider_usd_micros"})
    reserve_budget(
        uncapped,
        "reservation-uncapped",
        (ResourceAmount(resource="provider_usd_micros", value=10**9),),
    )


@pytest.mark.parametrize(("actual", "overrun"), [(5, 0), (10, 0), (11, 1)])
def test_reconciliation_records_under_exact_and_over_cap(actual: int, overrun: int) -> None:
    authorization = _authorization(cap=10)
    reservation = reserve_budget(
        authorization,
        "reservation-all",
        tuple(ResourceAmount(resource=resource, value=10) for resource in BUDGET_RESOURCES),
    )
    reconciliation = reconcile_budget(authorization, (reservation,), _usage(actual))
    assert reconciliation.reportable
    assert all(entry.overrun == overrun for entry in reconciliation.entries)
    assert (
        type(reconciliation).from_canonical_json(reconciliation.to_canonical_json())
        == reconciliation
    )


def test_unknown_required_usage_is_explicit_and_not_reportable() -> None:
    authorization = _authorization()
    usage: list[Usage] = list(_usage())
    usage[0] = UnavailableUsage(resource=usage[0].resource, reason="provider omitted usage")
    reconciliation = reconcile_budget(authorization, (), usage)
    assert not reconciliation.reportable
    assert reconciliation.entries[0].usage.kind in ("available", "unavailable")
    assert b'"kind":"unavailable"' in reconciliation.to_canonical_json()
    with pytest.raises(ValueError, match="one usage"):
        reconcile_budget(authorization, (), usage[:-1])


def test_money_is_integer_micros_and_has_no_hidden_seventy_five_dollar_default() -> None:
    with pytest.raises(ValidationError):
        CappedAllowance.model_validate(
            {"resource": "provider_usd_micros", "value": 1.5, "reporting": "required"}
        )
    source = __import__("local_operator.evaluation.receipts", fromlist=["x"]).__file__
    assert source is not None
    text = open(source, encoding="utf-8").read()  # noqa: SIM115 - test fixture inspection
    assert "75000000" not in text
    assert "$75" not in text


def test_preflight_receipt_replay_rejects_changed_plan_or_declaration() -> None:
    plan = _plan((_requirements()[0],))
    receipt = record_preflight(plan, "compute", status="pass", duration_ms=1)
    changed_requirement = _requirements()[0].model_copy(update={"disk_bytes": 10_001})
    changed = _plan((changed_requirement,))
    with pytest.raises(ValueError, match="another dependency plan|another requirement"):
        seal_preflight(changed, (receipt,), RedactionSet.from_resolved_values(()))
    mutated = PreflightReceipt.model_validate(
        {**receipt.model_dump(exclude={"receipt_id"}), "requirement_digest": DIGEST}
    )
    with pytest.raises(ValueError, match="another requirement"):
        seal_preflight(plan, (mutated,), RedactionSet.from_resolved_values(()))


def test_reservation_keys_disambiguate_equal_amounts_and_reject_retry() -> None:
    authorization = _authorization(cap=10)
    amount = (ResourceAmount(resource="guest_actions", value=5),)
    first = reserve_budget(authorization, "operation-a", amount)
    second = reserve_budget(authorization, "operation-b", amount, (first,))
    assert first.reservation_id != second.reservation_id
    with pytest.raises(ValueError, match="duplicate reservation key"):
        reserve_budget(authorization, "operation-a", amount, (first,))


def test_reconciliation_rejects_allowance_overrun_and_commitment_mutation() -> None:
    authorization = _authorization(cap=10)
    reservation = reserve_budget(
        authorization,
        "operation",
        (ResourceAmount(resource="guest_actions", value=5),),
    )
    reconciliation = reconcile_budget(authorization, (reservation,), _usage(11))
    payload = reconciliation.model_dump()
    entries = [dict(entry) for entry in payload["entries"]]
    entries[0]["overrun"] = 0
    with pytest.raises(ValidationError, match="overrun"):
        BudgetReconciliation.model_validate({**payload, "entries": entries})
    entries = [dict(entry) for entry in payload["entries"]]
    entries[0]["allowance"] = CappedAllowance(
        resource=entries[0]["resource"], value=9, reporting="required"
    )
    with pytest.raises(ValidationError, match="allowance"):
        BudgetReconciliation.model_validate({**payload, "entries": entries})
    with pytest.raises(ValidationError, match="commitment"):
        BudgetReconciliation.model_validate({**payload, "commitment_id": DIGEST})


def test_clock_values_are_real_and_canonical() -> None:
    base = _requirements()[4].model_dump()
    for zone in ("UTC", "America/New_York"):
        assert ClockRequirement.model_validate({**base, "timezone": zone}).timezone == zone
    for field, value in (
        ("date", "2026-02-30"),
        ("fixed_clock", "2026-13-30T12:00:00Z"),
        ("timezone", "Mars/Olympus"),
    ):
        payload = {**base, "date": None, "fixed_clock": "2026-08-30T12:00:00Z", field: value}
        if field == "date":
            payload["fixed_clock"] = None
        with pytest.raises(ValidationError) as caught:
            ClockRequirement.model_validate(payload)
        if field == "timezone":
            assert "IANA zone" in str(caught.value)


@pytest.mark.parametrize("encoded", ["base64", "urlsafe", "hex", "percent"])
def test_redaction_rejects_common_encoded_canary_variants(encoded: str) -> None:
    import base64
    from urllib.parse import quote

    secret = "s3cret/value+42"
    variants = {
        "base64": base64.b64encode(secret.encode()).decode(),
        "urlsafe": base64.urlsafe_b64encode(secret.encode()).decode().rstrip("="),
        "hex": secret.encode().hex(),
        "percent": quote(secret, safe=""),
    }
    plan = _plan((_requirements()[0],))
    receipt = record_preflight(
        plan,
        "compute",
        status="pass",
        evidence={"encoded": variants[encoded]},
        duration_ms=1,
    )
    with pytest.raises(ValueError) as caught:
        seal_preflight(plan, (receipt,), RedactionSet.from_resolved_values((secret,)))
    assert secret not in str(caught.value)


@pytest.mark.parametrize(
    ("encoding", "candidate"),
    [
        ("percent", "Secret%2fValue%2b42"),
        ("percent", "Secret%2FValue%2B42"),
        ("percent", "Secret%2fValue%2B42"),
        ("hex", "Secret/Value+42".encode().hex().lower()),
        ("hex", "Secret/Value+42".encode().hex().upper()),
        (
            "hex",
            "".join(
                character.upper() if index % 2 else character.lower()
                for index, character in enumerate("Secret/Value+42".encode().hex())
            ),
        ),
    ],
)
def test_redaction_matches_encoded_case_semantics(encoding: str, candidate: str) -> None:
    secret = "Secret/Value+42"
    plan = _plan((_requirements()[0],))
    receipt = record_preflight(
        plan,
        "compute",
        status="pass",
        evidence={"nested": ["safe", {"encoded": candidate}]},
        duration_ms=1,
    )
    with pytest.raises(ValueError) as caught:
        seal_preflight(plan, (receipt,), RedactionSet.from_resolved_values((secret,)))
    assert secret not in str(caught.value)


def test_percent_redaction_preserves_unescaped_literal_case() -> None:
    secret = "Secret/Value+42"
    plan = _plan((_requirements()[0],))
    receipt = record_preflight(
        plan,
        "compute",
        status="pass",
        evidence={"nested": {"encoded": "secret%2fvalue%2b42"}},
        duration_ms=1,
    )
    assert seal_preflight(plan, (receipt,), RedactionSet.from_resolved_values((secret,))).successful
