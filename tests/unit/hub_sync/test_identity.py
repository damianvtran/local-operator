"""Identity/policy proofs do not dispatch network or read an operator profile."""

from dataclasses import replace
from typing import Any

import pytest

from local_operator.hub_sync.identity import (
    LinkError,
    LinkIdentity,
    LinkPolicy,
    PublicEgressGrant,
    approve_source,
    normalize_source,
)


def identity(**kwargs: Any) -> LinkIdentity:
    values: dict[str, Any] = {
        "kind": "agent",
        "local_id": "local-1",
        "hub_source": "https://hub.example/api",
        "visibility": "public",
        "tenant_id": None,
        "hub_id": "remote-1",
    }
    values.update(kwargs)
    return LinkIdentity(**values)


def test_full_origin_and_scope_identity_not_display_name() -> None:
    original = identity()
    assert len(original.link_id) == 64
    assert LinkIdentity.from_json(original.to_json()) == original
    assert identity(hub_source="HTTPS://HUB.EXAMPLE:443/api/").link_id == original.link_id
    for different in (
        identity(kind="team"),
        identity(kind="skill"),
        identity(local_id="another"),
        identity(hub_id="another"),
        identity(hub_source="https://hub.example/other-api"),
        identity(visibility="org", tenant_id="org-one"),
        identity(visibility="org", tenant_id="org-two"),
    ):
        assert different.link_id != original.link_id
        with pytest.raises(LinkError) as refusal:
            original.require_same_origin(different)
        assert refusal.value.code == "origin-conflict"


@pytest.mark.parametrize(
    "source",
    [
        "https://user:password@hub.example/api",
        "https://hub.example/api?token=secret",
        "https://hub.example/api?",
        "https://hub.example/api#",
        "file:///tmp/hub",
        "https://hub.example:invalid/api",
        "https://hub.example/../api",
        " https://hub.example",
        "https://hub.example/\\api",
        "https://hub.example/\x00api",
    ],
)
def test_source_refusals_do_not_echo_credentials(source: str) -> None:
    with pytest.raises(LinkError) as refusal:
        normalize_source(source)
    assert refusal.value.code == "origin-conflict"
    assert "secret" not in str(refusal.value) and "password" not in str(refusal.value)


def test_source_syntax_does_not_authorize_credential_egress() -> None:
    assert (
        approve_source("https://HUB.example/api/", ("https://hub.example/api",))
        == "https://hub.example/api"
    )
    with pytest.raises(LinkError):
        approve_source("https://attacker.example/api", ("https://hub.example/api",))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"visibility": "public", "tenant_id": "org"},
        {"visibility": "org", "tenant_id": None},
        {"visibility": "org", "tenant_id": "../org"},
        {"kind": "unknown"},
        {"local_id": "../local"},
    ],
)
def test_invalid_scope_and_id(kwargs) -> None:
    with pytest.raises(LinkError):
        identity(**kwargs)


@pytest.mark.parametrize(
    "subscribe,auto_push", [(False, False), (False, True), (True, False), (True, True)]
)
def test_independent_policy_roundtrip_and_direct_edit_revision(subscribe, auto_push) -> None:
    original = identity()
    policy = LinkPolicy(subscribe, auto_push)
    assert LinkPolicy.from_json(policy.to_json()) == policy
    revision = policy.revision(original)
    assert revision != replace(policy, subscribe=not subscribe).revision(original)
    assert revision != replace(policy, auto_push=not auto_push).revision(original)
    assert revision != policy.revision(identity(hub_source="https://hub.example/other-api"))
    grant = PublicEgressGrant(original.link_id)
    assert revision != policy.revision(original, grant)


@pytest.mark.parametrize("value", ["true", "false", 0, 1, None, [], {}])
def test_policies_never_coerce_truthy_values(value) -> None:
    with pytest.raises(LinkError) as refusal:
        LinkPolicy.from_json({"subscribe": value, "auto_push": False})
    assert refusal.value.code == "invalid-link-policy"


def test_public_grant_cannot_cross_scope_or_origin() -> None:
    grant = PublicEgressGrant(identity().link_id)
    for target in (identity(hub_id="other"), identity(visibility="org", tenant_id="org-one")):
        with pytest.raises(LinkError) as refusal:
            LinkPolicy().revision(target, grant)
        assert refusal.value.code == "invalid-link-policy"
    assert LinkPolicy().to_json() == {"subscribe": True, "auto_push": False}
