"""Independent per-link policy CAS through the existing coordinated config writer.

The legacy kind switches become creation defaults. Materializing an existing
link preserves a real legacy OFF exactly once; future changes to that default
cannot override explicit link values. Revisions are content digests, not counters
that a text editor would have to increment. This module performs local work
only: remote capability/lineage verification belongs to the shared service BEFORE
entering the bounded config transaction, then fresh eligibility at dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from local_operator.config_writer import config_write_lock
from local_operator.hub_sync.identity import (
    PUBLICATION_CONTRACT,
    LinkError,
    LinkIdentity,
    LinkPolicy,
    PublicEgressGrant,
)

if TYPE_CHECKING:
    from local_operator.config import ConfigManager


@dataclass(frozen=True)
class PolicySnapshot:
    policy: LinkPolicy
    revision: str
    public_egress_grant: PublicEgressGrant | None

    def to_json(self) -> dict[str, Any]:
        return {**self.policy.to_json(), "revision": self.revision}


def _hub(values: Mapping[str, Any]) -> Mapping[str, Any]:
    hub = values.get("hub", {})
    if not isinstance(hub, Mapping):
        raise LinkError("invalid-link-policy", "hub policy configuration must be a mapping")
    return hub


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LinkError("invalid-link-policy", "hub link policies must be a mapping")
    return value


def creation_policy(values: Mapping[str, Any], kind: str) -> LinkPolicy:
    defaults = _mapping(_hub(values).get("auto_update", {}))
    if kind not in ("agent", "team", "skill"):
        raise LinkError("invalid-link-policy", "hub link kind is unsupported")
    default = (
        True if kind == "skill" else defaults.get("agents" if kind == "agent" else "teams", True)
    )
    # Preserve an explicitly stored BOOL OFF, not truthiness of arbitrary YAML.
    # Skills have no new shared settings key; missing means the canonical ON.
    return LinkPolicy(subscribe=default if type(default) is bool else True, auto_push=False)


def read_policy(values: Mapping[str, Any], identity: LinkIdentity) -> PolicySnapshot:
    hub = _hub(values)
    policies = _mapping(hub.get("links", {}))
    policy = (
        LinkPolicy.from_json(policies[identity.link_id])
        if identity.link_id in policies
        else creation_policy(values, identity.kind)
    )
    grants = _mapping(hub.get("public_egress_grants", {}))
    raw = grants.get(identity.link_id)
    grant = None
    if raw is not None:
        if not isinstance(raw, Mapping) or set(raw) != {"link_id", "publication_contract"}:
            raise LinkError("invalid-link-policy", "public egress grant is invalid")
        grant = PublicEgressGrant(raw["link_id"], raw["publication_contract"])
    return PolicySnapshot(policy, policy.revision(identity, grant), grant)


def materialize_policy(
    manager: "ConfigManager", identity: LinkIdentity, *, legacy_subscribe_hint: bool | None = None
) -> PolicySnapshot:
    """Migrate one existing link once, without rewriting any explicit policy."""
    from local_operator import settings_io

    if legacy_subscribe_hint is not None and type(legacy_subscribe_hint) is not bool:
        raise LinkError("invalid-link-policy", "legacy policy hint must be a boolean")
    with config_write_lock(manager.config_dir):
        settings_io._reload_before_write(manager)
        values = manager.get_config().values
        current = read_policy(values, identity)
        if identity.link_id not in _mapping(_hub(values).get("links", {})):
            if legacy_subscribe_hint is not None:
                policy = LinkPolicy(subscribe=legacy_subscribe_hint, auto_push=False)
                grant = current.public_egress_grant
                current = PolicySnapshot(policy, policy.revision(identity, grant), grant)
            settings_io._store(
                manager, ("hub", "links", identity.link_id), current.policy.to_json()
            )
        return current


def link_policy_for(values: Mapping[str, Any], link_id: str, kind: str) -> LinkPolicy:
    """The stored per-link policy for a link id, else its kind's creation default.

    Used by the runner, which has a link id (from a baseline identity) but may not
    have a full :class:`LinkIdentity` at enumeration time.
    """
    policies = _mapping(_hub(values).get("links", {}))
    if link_id in policies:
        return LinkPolicy.from_json(policies[link_id])
    return creation_policy(values, kind)


def update_policy(
    manager: "ConfigManager",
    identity: LinkIdentity,
    *,
    expected_revision: str,
    subscribe: bool | None = None,
    auto_push: bool | None = None,
    confirm_public_egress: bool = False,
    can_enable_auto_push: bool = False,
    ineligible_reason: str = "unsupported-precondition",
) -> PolicySnapshot:
    """Compare validated fresh policy/grant identity, then update one config subtree.

    ``can_enable_auto_push`` is a service attestation, not a request field. It
    must never be derived from preview pin existence or user-supplied scope.
    Existing ON may remain paused when eligibility changes; turning it OFF or
    changing Subscribe does not require a working hub/model connection.
    """
    from local_operator import settings_io

    if not isinstance(expected_revision, str) or not expected_revision:
        raise LinkError("invalid-link-policy", "an expected policy revision is required")
    if subscribe is None and auto_push is None:
        raise LinkError("invalid-link-policy", "at least one policy value is required")
    for value in (subscribe, auto_push):
        if value is not None and type(value) is not bool:
            raise LinkError("invalid-link-policy", "link policy values must be booleans")
    if type(can_enable_auto_push) is not bool:
        raise LinkError("invalid-link-policy", "publication eligibility must be a boolean")
    if type(confirm_public_egress) is not bool:
        raise LinkError("invalid-link-policy", "public egress acknowledgement must be a boolean")
    with config_write_lock(manager.config_dir):
        settings_io._reload_before_write(manager)
        current = read_policy(manager.get_config().values, identity)
        if current.revision != expected_revision:
            raise LinkError("policy-revision-changed", "link policy changed; reload before saving")
        chosen = LinkPolicy(
            current.policy.subscribe if subscribe is None else subscribe,
            current.policy.auto_push if auto_push is None else auto_push,
        )
        grant = current.public_egress_grant
        if auto_push is True:
            if not can_enable_auto_push:
                raise LinkError(
                    ineligible_reason, "automatic publication is not eligible for this origin"
                )
            if identity.visibility == "public" and (
                grant is None or grant.publication_contract != PUBLICATION_CONTRACT
            ):
                if not confirm_public_egress:
                    raise LinkError(
                        "unsafe-publication",
                        "explicit future public publication consent is required",
                    )
                grant = PublicEgressGrant(identity.link_id)
        # ONE existing top-level write contains both policy and grant, so a
        # process death cannot persist ON without its matching explicit grant.
        hub = dict(_hub(manager.get_config().values))
        links = dict(_mapping(hub.get("links", {})))
        links[identity.link_id] = chosen.to_json()
        hub["links"] = links
        grants = dict(_mapping(hub.get("public_egress_grants", {})))
        if grant is not None:
            grants[identity.link_id] = grant.to_json()
        if grants:
            hub["public_egress_grants"] = grants
        manager.set_config_value("hub", hub)
        return PolicySnapshot(chosen, chosen.revision(identity, grant), grant)
